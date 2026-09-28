"""Evaluate curves and imported metrics through the same Spec rules."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from mosfet_platform.analysis.results import (
    judge_results,
    order_results,
    summarize_results,
)
from mosfet_platform.analysis.spec import parse_spec_document
from mosfet_platform.case_manifest import load_comsol_case_manifest
from mosfet_platform.extraction.formal import curve_rows, extract_bundle_metrics
from mosfet_platform.measurement import load_measurement_contract
from mosfet_platform.io.metric_table import METRIC_UNITS, load_metric_table
from mosfet_platform.io.measured import (
    ManifestMeasuredSource,
    RawMeasuredDevice,
    build_measured_bundle,
)
from mosfet_platform.workflows.diagnostics import write_diagnostics


_RESULT_COLUMNS = (
    "status",
    "primary_failure_reason",
    "primary_metric",
    "primary_value",
    "primary_limit",
    "normalized_exceedance",
    "fail_reason",
    "device_id",
    "condition_id",
    "width_m",
    "length_m",
    "oxide_thickness_m",
    "result_origin",
    "source_id",
    "source_type",
    "ion",
    "ioff",
    "ion_ioff_cross_bias",
    "ion_ioff",
    "gm_max",
    "vth",
    "ss_mv_dec",
    "dibl_mV_per_V",
)
_METRIC_COLUMNS = (
    "device_id", "condition_id", "device_type", "width_m", "length_m",
    "oxide_thickness_m", "temperature_K", "ion", "ioff", "vth", "ss_mv_dec",
    "gm_max", "vth_status", "ss_status", "formal_eligible",
)


@dataclass(frozen=True)
class EvaluationResult:
    results: pd.DataFrame
    summary: pd.DataFrame
    counts: dict[str, int]
    results_path: Path | None = None
    summary_path: Path | None = None
    diagnostics_path: Path | None = None
    metrics_path: Path | None = None


def _resolve(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _load_yaml(path: Path) -> Mapping[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return raw


def _check_spec_contract(contract: Any, spec: Mapping[str, Any]) -> None:
    document = parse_spec_document(spec)
    if document.metadata.condition_id != contract.condition_id:
        raise ValueError("Spec condition_id does not match the MeasurementContract.")
    if document.metadata.device_type.lower() != contract.device_type.lower():
        raise ValueError("Spec device_type does not match the MeasurementContract.")


def _check_contract(manifest: Any, contract: Any, spec: Mapping[str, Any]) -> None:
    _check_spec_contract(contract, spec)
    common = manifest.raw["common_conditions"]
    if not np.isclose(float(common["width_m"]), contract.geometry.width_m, rtol=1e-9, atol=0.0):
        raise ValueError("Manifest width does not match the MeasurementContract.")
    if not np.isclose(float(common["temperature_K"]), contract.temperature_K, rtol=1e-9, atol=0.0):
        raise ValueError("Manifest temperature does not match the MeasurementContract.")
    if str(common.get("device_type", "")).lower() != contract.device_type.lower():
        raise ValueError("Manifest device_type does not match the MeasurementContract.")


def _issue_row(
    raw: RawMeasuredDevice,
    contract: Any,
    result: Any,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    geometry = raw.case["geometry"]
    recoverable = result.issues and all(issue.recoverable for issue in result.issues)
    status = "RETEST" if recoverable else "INVALID"
    return {
        "status": status,
        "primary_failure_reason": "",
        "primary_metric": "",
        "primary_value": np.nan,
        "primary_limit": np.nan,
        "normalized_exceedance": 0.0,
        "fail_reason": ";".join(issue.code for issue in result.issues),
        "device_id": raw.device_id,
        "condition_id": contract.condition_id,
        "width_m": raw.width_m,
        "length_m": float(geometry["length_m"]),
        "oxide_thickness_m": float(geometry["oxide_thickness_m"]),
        "formal_eligible": False,
        **source,
    }


def _select(
    source: ManifestMeasuredSource,
    device_id: str | None,
    batch_id: str | None,
) -> tuple[RawMeasuredDevice, ...]:
    if device_id and batch_id:
        raise ValueError("Choose either device_id or batch_id.")
    if device_id:
        return (source.load_device(device_id),)
    return tuple(source.load_batch(batch_id))


def run_evaluation(
    *,
    cases: str | Path,
    contract: str | Path,
    spec: str | Path,
    output: str | Path | None = None,
    root: str | Path | None = None,
    device_id: str | None = None,
    batch_id: str | None = None,
    diagnostics: bool = False,
) -> EvaluationResult:
    if diagnostics and (device_id is None or output is None):
        raise ValueError("Diagnostics require device_id and output.")
    project = Path(root or Path.cwd()).resolve()
    manifest = load_comsol_case_manifest(_resolve(cases, project), project_root=project)
    contract_doc = load_measurement_contract(_resolve(contract, project))
    spec_doc = _load_yaml(_resolve(spec, project))
    _check_contract(manifest, contract_doc, spec_doc)

    source = ManifestMeasuredSource(manifest)
    metric_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    diagnostic_curves = pd.DataFrame()
    diagnostic_source: Mapping[str, Any] = {}
    diagnostic_issues: tuple[Any, ...] = ()
    for raw in _select(source, device_id, batch_id):
        built, provenance = build_measured_bundle(
            raw,
            contract_doc,
            project,
            diagnostics=diagnostics,
        )
        if diagnostics:
            geometry = raw.case["geometry"]
            traces = built.bundle.curves if built.bundle else built.diagnostic_curves
            diagnostic_curves = pd.DataFrame(
                curve_rows(
                    device_id=raw.device_id,
                    condition_id=contract_doc.condition_id,
                    geometry=(
                        raw.width_m,
                        float(geometry["length_m"]),
                        float(geometry["oxide_thickness_m"]),
                    ),
                    curves=traces,
                    meta={
                        "evidence_status": "FORMAL_INPUT" if built.bundle else "PARTIAL_INPUT",
                        **provenance,
                    },
                )
            )
            diagnostic_source = provenance
            diagnostic_issues = built.issues
        if built.bundle is None:
            issue_rows.append(_issue_row(raw, contract_doc, built, provenance))
            continue
        metrics = extract_bundle_metrics(built.bundle, contract_doc)
        metrics.update(provenance)
        metrics["device_type"] = contract_doc.device_type
        metric_rows.append(metrics)

    formal = (
        judge_results(pd.DataFrame(metric_rows), spec_doc)
        if metric_rows
        else pd.DataFrame()
    )
    details = pd.concat([formal, pd.DataFrame(issue_rows)], ignore_index=True, sort=False)
    evaluated = _finish_evaluation(
        details, pd.DataFrame(metric_rows), output=output, project=project,
    )
    if diagnostics:
        evaluated = replace(evaluated, diagnostics_path=write_diagnostics(
            _resolve(output, project),
            row=details.iloc[0].to_dict(),
            curves=diagnostic_curves,
            source=diagnostic_source,
            issues=diagnostic_issues,
        ))
    return evaluated


def _finish_evaluation(
    results: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    output: str | Path | None,
    project: Path,
) -> EvaluationResult:
    summary = summarize_results(results)
    results = order_results(results).reindex(columns=_RESULT_COLUMNS)
    for name in METRIC_UNITS.keys() & set(_RESULT_COLUMNS):
        results[name] = pd.to_numeric(results[name], errors="coerce")

    statuses = results["status"].astype(str)
    counts = {"devices": len(results)}
    counts.update(
        {
            state: int((statuses == state).sum())
            for state in ("PASS", "FAIL", "INVALID", "RETEST")
        }
    )

    results_path = summary_path = metrics_path = None
    if output is not None:
        directory = _resolve(output, project)
        directory.mkdir(parents=True, exist_ok=True)
        results_path = directory / "measured_results.csv"
        summary_path = directory / "measured_summary.csv"
        metrics_path = directory / "measured_metrics.csv"
        results.to_csv(results_path, index=False)
        summary.to_csv(summary_path, index=False)
        metric_columns = list(dict.fromkeys([*_METRIC_COLUMNS, *metrics.columns]))
        metrics.reindex(columns=metric_columns).to_csv(metrics_path, index=False)

    return EvaluationResult(
        results=results,
        summary=summary,
        counts=counts,
        results_path=results_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
    )


def run_metric_evaluation(
    *,
    metrics: str | Path,
    contract: str | Path,
    spec: str | Path,
    output: str | Path | None = None,
    root: str | Path | None = None,
    device_id: str | None = None,
    batch_id: str | None = None,
) -> EvaluationResult:
    project = Path(root or Path.cwd()).resolve()
    contract_doc = load_measurement_contract(_resolve(contract, project))
    spec_doc = _load_yaml(_resolve(spec, project))
    _check_spec_contract(contract_doc, spec_doc)
    input_path = _resolve(metrics, project)
    if output is not None and input_path in {
        _resolve(output, project) / name
        for name in ("measured_metrics.csv", "measured_results.csv", "measured_summary.csv")
    }:
        raise ValueError("Evaluation output must not overwrite the input metric table.")
    frame = load_metric_table(input_path, contract_doc)
    if device_id and batch_id:
        raise ValueError("Choose either device_id or batch_id.")
    if batch_id not in {None, "all"}:
        raise KeyError(f"Metric table has no batch: {batch_id}")
    if device_id is not None:
        frame = frame.loc[frame["device_id"].eq(device_id)].copy()
        if frame.empty:
            raise KeyError(f"Metric device not found: {device_id}")
    return _finish_evaluation(
        judge_results(frame, spec_doc), frame, output=output, project=project,
    )
