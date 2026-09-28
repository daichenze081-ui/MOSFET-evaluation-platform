"""Predict devices with a qualified frozen model."""

from __future__ import annotations

from dataclasses import dataclass
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
from mosfet_platform.artifacts.frozen_model import load_frozen_model
from mosfet_platform.extraction.formal import extract_bundle_metrics
from mosfet_platform.measurement import load_measurement_contract
from mosfet_platform.model.prediction import (
    build_predicted_bundle,
    curve_rows,
    diagnostic_rows,
)
from mosfet_platform.workflows.diagnostics import write_diagnostics


_INPUT_COLUMNS = ("device_id", "width_m", "length_m", "oxide_thickness_m")
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
    "model_id",
    "model_family",
    "ion",
    "ioff",
    "ion_ioff_cross_bias",
    "ion_ioff",
    "gm_max",
    "vth",
    "ss_mv_dec",
    "dibl_mV_per_V",
)


@dataclass(frozen=True)
class PredictionResult:
    results: pd.DataFrame
    summary: pd.DataFrame
    curves: pd.DataFrame
    counts: dict[str, int]
    results_path: Path | None = None
    summary_path: Path | None = None
    curves_path: Path | None = None
    diagnostics_path: Path | None = None


def _resolve(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _mapping(path: Path) -> Mapping[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return raw


def _devices(path: Path, device_id: str | None) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"device_id": str}, keep_default_na=False)
    missing = [column for column in _INPUT_COLUMNS if column not in frame]
    if missing or frame.empty:
        detail = ", ".join(missing) if missing else "no rows"
        raise ValueError(f"Prediction input is invalid: {detail}")
    frame = frame.loc[:, _INPUT_COLUMNS].copy()
    if device_id is not None:
        frame = frame.loc[frame["device_id"] == device_id]
        if frame.empty:
            raise KeyError(f"Unknown device_id: {device_id}")
    return frame


def _issue_row(
    row: Mapping[str, Any],
    contract: Any,
    built: Any,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    codes = [issue.code for issue in built.issues]
    if "MODEL_ERROR" in codes:
        message = next(
            issue.message for issue in built.issues if issue.code == "MODEL_ERROR"
        )
        raise RuntimeError(message)
    status = "OUT_OF_ENVELOPE" if "OUT_OF_ENVELOPE" in codes else "INVALID_INPUT"
    return {
        "status": status,
        "primary_failure_reason": "",
        "primary_metric": "",
        "primary_value": np.nan,
        "primary_limit": np.nan,
        "normalized_exceedance": 0.0,
        "fail_reason": ";".join(codes),
        "device_id": str(row.get("device_id", "")),
        "condition_id": contract.condition_id,
        "width_m": row.get("width_m"),
        "length_m": row.get("length_m"),
        "oxide_thickness_m": row.get("oxide_thickness_m"),
        "formal_eligible": False,
        **source,
    }


def run_prediction(
    *,
    config: str | Path,
    root: str | Path | None = None,
    output: str | Path | None = None,
    device_id: str | None = None,
    include_curves: bool = False,
    diagnostics: bool = False,
) -> PredictionResult:
    if diagnostics and (device_id is None or output is None):
        raise ValueError("Diagnostics require device_id and output.")
    project = Path(root or Path.cwd()).resolve()
    raw = _mapping(_resolve(config, project))
    settings = raw.get("prediction")
    if not isinstance(settings, Mapping):
        raise ValueError("Prediction config must contain a prediction mapping.")

    contract = load_measurement_contract(
        _resolve(str(settings["measurement_contract"]), project)
    )
    spec = _mapping(_resolve(str(settings["spec"]), project))
    document = parse_spec_document(spec)
    if document.metadata.condition_id != contract.condition_id:
        raise ValueError("Spec condition_id does not match the MeasurementContract.")
    if document.metadata.device_type.lower() != contract.device_type.lower():
        raise ValueError("Spec device_type does not match the MeasurementContract.")
    frozen = load_frozen_model(str(settings["model_manifest"]), root=project)
    if not np.isclose(frozen.model.temperature_K, contract.temperature_K):
        raise ValueError("Frozen-model temperature does not match the MeasurementContract.")

    devices = _devices(_resolve(str(settings["input"]), project), device_id)
    metric_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    diagnostic_curves = pd.DataFrame()
    diagnostic_source: Mapping[str, Any] = {}
    diagnostic_issues: tuple[Any, ...] = ()
    for row in devices.to_dict(orient="records"):
        built, source = build_predicted_bundle(
            row, contract, frozen, diagnostics=diagnostics
        )
        if diagnostics:
            evidence = (
                curve_rows(built.bundle, source)
                if built.bundle
                else diagnostic_rows(row, contract, built.diagnostic_curves, source)
            )
            diagnostic_curves = pd.DataFrame(evidence)
            diagnostic_source = source
            diagnostic_issues = built.issues
        if built.bundle is None:
            issue_rows.append(_issue_row(row, contract, built, source))
            continue
        metrics = extract_bundle_metrics(built.bundle, contract)
        metrics.update(source)
        metric_rows.append(metrics)
        curves.extend(curve_rows(built.bundle, source))

    formal = (
        judge_results(pd.DataFrame(metric_rows), spec, predicted=True)
        if metric_rows
        else pd.DataFrame()
    )
    results = pd.concat([formal, pd.DataFrame(issue_rows)], ignore_index=True, sort=False)
    details = results.copy()
    for column in _RESULT_COLUMNS:
        if column not in results:
            results[column] = np.nan
    summary = summarize_results(results)
    results = order_results(results).loc[:, _RESULT_COLUMNS]
    curve_frame = pd.DataFrame(curves)

    status = results["status"].astype(str)
    states = ("PREDICTED_PASS", "PREDICTED_FAIL", "OUT_OF_ENVELOPE", "INVALID_INPUT")
    counts = {"devices": len(results)}
    counts.update({state: int((status == state).sum()) for state in states})

    results_path = summary_path = curves_path = diagnostics_path = None
    if output is not None:
        directory = _resolve(output, project)
        directory.mkdir(parents=True, exist_ok=True)
        results_path = directory / "predicted_results.csv"
        summary_path = directory / "predicted_summary.csv"
        results.to_csv(results_path, index=False)
        summary.to_csv(summary_path, index=False)
        if include_curves:
            curves_path = directory / "predicted_curves.csv"
            curve_frame.to_csv(curves_path, index=False)
        if diagnostics:
            diagnostics_path = write_diagnostics(
                directory,
                row=details.iloc[0].to_dict(),
                curves=diagnostic_curves,
                source=diagnostic_source,
                issues=diagnostic_issues,
            )

    return PredictionResult(
        results,
        summary,
        curve_frame,
        counts,
        results_path,
        summary_path,
        curves_path,
        diagnostics_path,
    )
