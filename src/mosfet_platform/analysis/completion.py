"""Optional model evidence and missing-only completion, separate from raw statistics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math

from mosfet_platform.analysis.diagnosis import VALID, diagnose_records, finite_number
from mosfet_platform.analysis.spec import SPEC_RULES
from mosfet_platform.artifacts.frozen_model import load_frozen_model
from mosfet_platform.extraction.formal import extract_bundle_metrics
from mosfet_platform.model.prediction import build_predicted_bundle
from mosfet_platform.provenance import file_sha256


def validate_completion(settings):
    if not isinstance(settings, dict) or set(settings) - {"model_manifest", "absolute_tolerances"}:
        raise ValueError("completion accepts model_manifest and optional absolute_tolerances.")
    if not isinstance(settings.get("model_manifest"), str) or not settings["model_manifest"].strip():
        raise ValueError("completion.model_manifest must be a non-empty path.")
    tolerances = settings.get("absolute_tolerances", {})
    if not isinstance(tolerances, dict):
        raise ValueError("absolute_tolerances must be a metric-to-number mapping.")
    for metric, value in tolerances.items():
        if metric not in {rule[1] for rule in SPEC_RULES} or not isinstance(value, (int, float)) or isinstance(value, bool) or finite_number(value) is None or value < 0:
            raise ValueError("absolute_tolerances require known metrics and finite nonnegative numbers in native units.")


def complete_diagnosis(records, raw_spec, contract, data, settings, root, hashes):
    """Attach model/combined evidence without modifying raw rows or their statistics."""
    validate_completion(settings)
    manifest = root / settings["model_manifest"]
    tolerances = settings.get("absolute_tolerances", {})
    frozen, load_error = None, None
    try:
        hashes[manifest.resolve()] = file_sha256(manifest)
        frozen = load_frozen_model(manifest, root=root)
        hashes[frozen.model_path] = frozen.model_sha256
    except (ValueError, OSError, KeyError, TypeError) as error:
        load_error = str(error)

    model = {"available": frozen is not None, "error": load_error,
             "manifest": str(manifest.resolve()), "absolute_tolerances": tolerances,
             "prediction_interval": None,
             "limitations": ["No calibrated prediction intervals; predicted decisions are provisional.",
                             "Envelope membership does not establish local joint training coverage.",
                             "Agreement on observed metrics does not validate missing metrics."]}
    if frozen:
        model.update({"model_id": frozen.run_id, "model_sha256": frozen.model_sha256,
                      "model_path": str(frozen.model_path), "family": frozen.family,
                      "status": frozen.status, "validation_status": frozen.validation_status,
                      "independent_validation_status": frozen.independent_validation_status,
                      "envelope": asdict(frozen.model.envelope),
                      "temperature_K": frozen.model.temperature_K, "width_m": frozen.model.width_m})

    combined_devices, combined_metrics, checks = [], [], []
    for row, original_device in zip(records, data["devices"]):
        identity = {key: original_device[key] for key in ("device_id", "condition_id", "source_type")}
        originals = [item for item in data["metric_details"] if all(item[key] == value for key, value in identity.items())]
        reasons, predictions = [], {}
        if frozen is None:
            reasons.append("MODEL_UNAVAILABLE")
        if row.get("context_errors") or str(row.get("formal_eligible", True)).lower() not in {"true", "1"} or str(row.get("data_quality_status", "ok")).lower() not in {"ok", "valid"}:
            reasons.append("INVALID_CONTEXT")
        if row.get("condition_id") != contract.condition_id or str(row.get("device_type", "")).lower() != contract.device_type.lower():
            reasons.append("CONDITION_MISMATCH")
        if frozen:
            for field, expected in (("temperature_K", frozen.model.temperature_K), ("width_m", frozen.model.width_m)):
                value = finite_number(row.get(field))
                if value is None or not math.isclose(value, expected, rel_tol=1e-9, abs_tol=0):
                    reasons.append(field.upper() + "_MISMATCH")
            if not math.isclose(contract.temperature_K, frozen.model.temperature_K, rel_tol=1e-9):
                reasons.append("MODEL_CONTRACT_TEMPERATURE_MISMATCH")
            if contract.device_type.lower() != "nmos":
                reasons.append("UNSUPPORTED_DEVICE_TYPE")
        if not reasons:
            try:
                built, _ = build_predicted_bundle(row, contract, frozen)
                if built.bundle is None:
                    reasons.extend(issue.code for issue in built.issues)
                else:
                    predicted = {**extract_bundle_metrics(built.bundle, contract),
                                 "device_type": contract.device_type, "source_type": "estimated"}
                    evaluated = diagnose_records([predicted], raw_spec, group_by=())
                    predictions = {item["metric"]: item for item in evaluated["metric_details"]}
            except (ValueError, RuntimeError, TypeError, KeyError, FloatingPointError, OverflowError) as error:
                reasons.append("PREDICTION_ERROR:" + str(error))

        comparisons = {}
        for item in originals:
            metric = item["metric"]
            pred = predictions.get(metric)
            delta = (finite_number(item["value"] - pred["value"])
                     if item["status"] in VALID and pred and pred["status"] in VALID else None)
            tolerance = tolerances.get(metric)
            check = ("EXCEEDS_TOLERANCE" if abs(delta) > tolerance else "WITHIN_TOLERANCE") if delta is not None and tolerance is not None else "NOT_ASSESSED"
            comparisons[metric] = {"prediction": pred["value"] if pred and pred["status"] in VALID else None,
                                   "delta": delta, "absolute_tolerance": tolerance, "comparison": check}
        consistency = ("MISMATCH" if any(c["comparison"] == "EXCEEDS_TOLERANCE" for c in comparisons.values()) else
                       "WITHIN_CONFIGURED_TOLERANCES" if any(c["comparison"] == "WITHIN_TOLERANCE" for c in comparisons.values()) else "NOT_ASSESSED")
        blockers = list(reasons)
        if consistency == "MISMATCH":
            blockers.append("OBSERVED_MODEL_MISMATCH")
        if frozen and frozen.independent_validation_status != "PASS":
            blockers.append("NO_INDEPENDENT_ENGINEERING_VALIDATION")
        used = []
        device_metrics = []
        for item in originals:
            metric = item["metric"]
            pred = predictions.get(metric)
            apply = item["status"] == "MISSING" and not blockers and pred and pred["status"] in VALID
            entry = deepcopy(item)
            entry.update({"original_value": item["value"], "original_status": item["status"],
                          "value_source": item["source_type"], "predicted": False,
                          "model_id": frozen.run_id if frozen else None, **comparisons[metric]})
            if apply:
                used.append(metric)
                entry.update({key: deepcopy(pred[key]) for key in ("value", "status", "reason", "failed_rules", "normalized_exceedance")})
                entry.update({"value_source": "estimated", "predicted": True, "method": "frozen_model_completion",
                              "evidence": [str(frozen.manifest_path), str(frozen.model_path)]})
            entry["completion_reason"] = ("PREDICTED_PROVISIONAL" if apply else
                                           ";".join(blockers) or "PREDICTION_UNAVAILABLE") if item["status"] == "MISSING" else "ORIGINAL_RETAINED"
            device_metrics.append(entry)
        complete = all(item["status"] in VALID for item in device_metrics)
        predicted_failures = [item["metric"] for item in device_metrics if item["predicted"] and item["status"] == "FAIL"]
        status = ("FAIL" if original_device["status"] == "FAIL" else
                  "PREDICTED_FAIL" if predicted_failures else
                  "INCOMPLETE" if not complete else
                  "PREDICTED_PASS" if used or original_device["estimated"] else "PASS")
        combined_devices.append({**deepcopy(original_device), "status": status, "original_status": original_device["status"],
                                 "complete": complete, "predicted_metrics": used, "provisional": bool(used) or original_device["estimated"],
                                 "conclusion_basis": "MIXED_WITH_PREDICTION" if used else original_device["conclusion_basis"],
                                 "failed_metrics": [item["metric"] for item in device_metrics if item["status"] == "FAIL"],
                                 "unavailable_metrics": [item["metric"] for item in device_metrics if item["status"] not in VALID]})
        combined_metrics.extend(device_metrics)
        checks.append({**identity, "applicability": "BLOCKED" if reasons else "IN_ENVELOPE",
                       "consistency": consistency, "blockers": blockers,
                       "inputs": {key: row.get(key) for key in ("width_m", "length_m", "oxide_thickness_m", "temperature_K")},
                       "predicted_metrics": used})
    data.update({"model_evidence": model, "model_checks": checks,
                 "combined_devices": combined_devices, "combined_metric_details": combined_metrics})
    return any(item["predicted"] for item in combined_metrics)
