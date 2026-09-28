"""Prepare registered curves for model training."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from mosfet_platform.provenance import file_sha256, git_commit, project_relative_path, software_version
from mosfet_platform.case_manifest import COMSOLCaseManifest, load_comsol_case_manifest

from mosfet_platform.measurement import MeasurementContract, load_measurement_contract
from mosfet_platform.extraction.idvd import extract_idvd_metrics_from_dataframe
from mosfet_platform.extraction.metrics import (
    extract_ion_sample_at_bias,
    extract_ion_ioff_ratio,
    extract_transfer_metrics,
    extract_vth_constant_current_robust,
    require_ion_consistency,
)
from mosfet_platform.extraction.selection import select_exact_curve_row
from mosfet_platform.io.comsol_curve import load_comsol_curve
from mosfet_platform.training._inventory import inventory_counts


ERROR_COLUMNS = (
    "scope",
    "case_id",
    "curve_type",
    "fixed_bias_V",
    "input_path",
    "error_type",
    "error_message",
)
DIAGNOSTIC_COLUMNS = (
    "case_id", "is_nominal", "curve_type", "fixed_bias_name", "fixed_bias_V",
    "width_m", "length_m", "oxide_thickness_m", "temperature_K", "qc_status",
    "analysis_role", "qc_reason", "input_path", "input_sha256", "point_count",
    "sweep_min_V", "sweep_max_V", "current_min_A", "current_max_A", "current_span_A",
    "current_status", "current_sign_available", "numerical_zero_clipped_count",
)

@dataclass(frozen=True)
class CharacterizationPaths:
    root: Path
    table_dir: Path
    curve_inventory: Path
    idvg_metrics: Path
    idvd_metrics: Path
    isolated_diagnostics: Path
    case_summary: Path
    extraction_errors: Path
    workflow_manifest: Path


@dataclass(frozen=True)
class CharacterizationResult:
    run_id: str
    status: str
    paths: CharacterizationPaths
    counts: Mapping[str, int]
    errors: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return raw


def _section(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"'{key}' section must be a YAML mapping.")
    return dict(value)


def _project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _paths(output_root: Path) -> CharacterizationPaths:
    table_dir = output_root / "tables"
    return CharacterizationPaths(
        root=output_root,
        table_dir=table_dir,
        curve_inventory=table_dir / "curve_inventory.csv",
        idvg_metrics=table_dir / "idvg_metrics_by_curve.csv",
        idvd_metrics=table_dir / "idvd_metrics_by_curve.csv",
        isolated_diagnostics=table_dir / "isolated_curve_diagnostics.csv",
        case_summary=table_dir / "case_summary.csv",
        extraction_errors=table_dir / "extraction_errors.csv",
        workflow_manifest=output_root / "workflow_manifest.json",
    )


def _error(
    *,
    scope: str,
    error: Exception | str,
    case_id: str = "",
    curve_type: str = "",
    fixed_bias_V: float | None = None,
    input_path: str = "",
) -> dict[str, Any]:
    return {
        "scope": scope,
        "case_id": case_id,
        "curve_type": curve_type,
        "fixed_bias_V": fixed_bias_V,
        "input_path": input_path,
        "error_type": type(error).__name__ if isinstance(error, Exception) else "CharacterizationContractError",
        "error_message": str(error),
    }


def _curve_inventory(
    manifest: COMSOLCaseManifest,
) -> pd.DataFrame:
    common = manifest.raw["common_conditions"]
    rows: list[dict[str, Any]] = []
    for case in manifest.cases:
        geometry = case["geometry"]
        for curve_type in ("idvg", "idvd"):
            bias_key = "vds_V" if curve_type == "idvg" else "vgs_V"
            for curve in case[curve_type]:
                path = manifest.project_root / str(curve["path"])
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "is_nominal": case["case_id"] == manifest.nominal_case_id,
                        "curve_type": curve_type,
                        "fixed_bias_name": bias_key,
                        "fixed_bias_V": float(curve[bias_key]),
                        "width_m": float(common["width_m"]),
                        "length_m": float(geometry["length_m"]),
                        "length_nm": float(geometry["length_m"]) * 1.0e9,
                        "oxide_thickness_m": float(geometry["oxide_thickness_m"]),
                        "oxide_thickness_nm": float(geometry["oxide_thickness_m"]) * 1.0e9,
                        "temperature_K": float(common["temperature_K"]),
                        "qc_status": str(curve["qc_status"]),
                        "analysis_role": str(curve["analysis_role"]),
                        "qc_reason": str(curve.get("qc_reason", "")),
                        "input_path": project_relative_path(path, manifest.project_root),
                        "bytes": int(path.stat().st_size),
                        "sha256": file_sha256(path),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["case_id", "curve_type", "fixed_bias_V"]
    ).reset_index(drop=True)



def _base_metric_record(record: pd.Series) -> dict[str, Any]:
    return {
        "case_id": record["case_id"],
        "is_nominal": bool(record["is_nominal"]),
        "width_m": float(record["width_m"]),
        "length_m": float(record["length_m"]),
        "length_nm": float(record["length_nm"]),
        "oxide_thickness_m": float(record["oxide_thickness_m"]),
        "oxide_thickness_nm": float(record["oxide_thickness_nm"]),
        "temperature_K": float(record["temperature_K"]),
        "input_path": record["input_path"],
        "input_sha256": record["sha256"],
    }


def _extract_idvg_record(
    record: pd.Series,
    *,
    project_root: Path,
    condition: MeasurementContract,
    numerical_zero_current_A: float,
) -> dict[str, Any]:
    path = project_root / str(record["input_path"])
    vds = float(record["fixed_bias_V"])
    frame, provenance = load_comsol_curve(
        path,
        curve_type="idvg",
        fixed_bias_V=vds,
        numerical_zero_current_A=numerical_zero_current_A,
    )
    vgs = frame["vgs"].to_numpy(dtype=float)
    ids = frame["id_magnitude"].to_numpy(dtype=float)
    target_current = condition.target_current_A(
        width_m=float(record["width_m"]),
        length_m=float(record["length_m"]),
    )
    vth, vth_status = extract_vth_constant_current_robust(
        vgs=vgs,
        ids=ids,
        target_current=target_current,
    )
    result: dict[str, Any] = {
        **_base_metric_record(record),
        "condition_id": condition.condition_id,
        "condition_id_source": "manifest_validated",
        "curve_type": "idvg",
        "fixed_bias_V": vds,
        "vds_V": vds,
        "point_count": int(len(frame)),
        "vgs_min_V": float(frame["vgs"].min()),
        "vgs_max_V": float(frame["vgs"].max()),
        "ioff_A": np.nan,
        "same_vds_on_current_A": np.nan,
        "ion_at_formal_bias_A": np.nan,
        "gm_max_S": np.nan,
        "vth_V": float(vth),
        "vth_target_current_A": float(target_current),
        "vth_status": str(vth_status),
        "vth_method": str(condition.vth.method),
        "ss_V_per_dec": np.nan,
        "ss_mV_per_dec": np.nan,
        "ss_status": "not_applicable_high_vds",
        **provenance,
    }
    if np.isclose(vds, condition.transfer.vds_V, rtol=0.0, atol=1.0e-12):
        transfer = extract_transfer_metrics(
            vgs=vgs,
            ids=ids,
            vds=vds,
            measurement_contract=condition,
            width_m=float(record["width_m"]),
            length_m=float(record["length_m"]),
        )
        same_vds_sample = extract_ion_sample_at_bias(
            sweep_values=vgs,
            ids=ids,
            curve_type="idvg",
            fixed_bias_V=vds,
            target_vgs_V=condition.transfer.vgs_on_V,
            target_vds_V=condition.transfer.vds_V,
            measurement_contract=condition,
        )
        result.update(
            {
                "ioff_A": float(transfer["ioff"]),
                "same_vds_on_current_A": float(same_vds_sample.current_A),
                "gm_max_S": float(transfer["gm_max"]),
                "vth_V": float(transfer["vth"]),
                "vth_status": str(transfer["vth_status"]),
                "ss_V_per_dec": float(transfer["ss_v_dec"]),
                "ss_mV_per_dec": float(transfer["ss_mv_dec"]),
                "ss_status": str(transfer["ss_status"]),
                "formal_eligible": bool(transfer["formal_eligible"]),
                "formal_ineligibility_reason": str(
                    transfer["formal_ineligibility_reason"]
                ),
                "ioff_requested_vgs_V": float(transfer["ioff_requested_vgs_V"]),
                "ioff_actual_vgs_V": float(transfer["ioff_actual_vgs_V"]),
                "ioff_sampling_method": str(transfer["ioff_sampling_method"]),
                "ioff_interpolated": bool(transfer["ioff_interpolated"]),
                "same_vds_requested_vgs_V": float(
                    same_vds_sample.requested_voltage_V
                ),
                "same_vds_actual_vgs_V": float(same_vds_sample.actual_voltage_V),
                "same_vds_sampling_method": str(same_vds_sample.sampling_method),
                "same_vds_interpolated": bool(same_vds_sample.interpolated),
            }
        )
        required = (
            result["ioff_A"],
            result["same_vds_on_current_A"],
            result["gm_max_S"],
            result["vth_V"],
            result["ss_V_per_dec"],
        )
    elif np.isclose(vds, condition.ion.vds_V, rtol=0.0, atol=1.0e-12):
        ion_sample = extract_ion_sample_at_bias(
            sweep_values=vgs,
            ids=ids,
            curve_type="idvg",
            fixed_bias_V=vds,
            target_vgs_V=condition.ion.vgs_V,
            target_vds_V=condition.ion.vds_V,
            measurement_contract=condition,
        )
        result.update(
            {
                "ion_at_formal_bias_A": float(ion_sample.current_A),
                "ion_requested_vgs_V": float(ion_sample.requested_voltage_V),
                "ion_actual_vgs_V": float(ion_sample.actual_voltage_V),
                "ion_sampling_method": str(ion_sample.sampling_method),
                "ion_interpolated": bool(ion_sample.interpolated),
            }
        )
        required = (result["ion_at_formal_bias_A"], result["vth_V"])
    else:
        raise ValueError(
            f"Formal Id-Vg Vds={vds:g} V is not declared by MeasurementContract."
        )
    if not np.isfinite(np.asarray(required, dtype=float)).all():
        raise ValueError("Id-Vg produced a non-finite required metric.")
    return result


def _extract_idvd_record(
    record: pd.Series,
    *,
    project_root: Path,
    condition: MeasurementContract,
    numerical_zero_current_A: float,
    extraction: Mapping[str, Any],
) -> dict[str, Any]:
    path = project_root / str(record["input_path"])
    vgs = float(record["fixed_bias_V"])
    frame, provenance = load_comsol_curve(
        path,
        curve_type="idvd",
        fixed_bias_V=vgs,
        numerical_zero_current_A=numerical_zero_current_A,
    )
    metrics = extract_idvd_metrics_from_dataframe(
        frame[["vgs", "vds", "id"]],
        ron_vds=float(extraction.get("ron_vds", 0.05)),
        low_vds_max=float(extraction.get("low_vds_max", 0.05)),
        high_vds_min=float(extraction.get("high_vds_min", 0.2)),
    )
    if len(metrics) != 1:
        raise ValueError("One manifest Id-Vd file must produce exactly one fixed-Vgs row.")
    metric = metrics.iloc[0].to_dict()
    required = (
        metric["id_at_vd_0p05"], metric["id_at_vd_0p2"],
        metric["low_vd_slope_S"], metric["gds_S"],
    )
    status = "ok"
    if np.isclose(vgs, 0.0, rtol=0.0, atol=1.0e-12):
        status = "not_applicable_at_vgs_zero"
    elif not np.isfinite(np.asarray(required + (metric["ron_ohm"],), dtype=float)).all():
        raise ValueError("Nonzero-Vgs Id-Vd produced a non-finite required metric.")
    formal_ion = np.nan
    if np.isclose(vgs, condition.ion.vgs_V, rtol=0.0, atol=1.0e-12):
        formal_ion_sample = extract_ion_sample_at_bias(
            sweep_values=frame["vds"].to_numpy(dtype=float),
            ids=frame["id_magnitude"].to_numpy(dtype=float),
            curve_type="idvd",
            fixed_bias_V=vgs,
            target_vgs_V=condition.ion.vgs_V,
            target_vds_V=condition.ion.vds_V,
            measurement_contract=condition,
        )
        formal_ion = formal_ion_sample.current_A
    return {
        **_base_metric_record(record),
        "condition_id": condition.condition_id,
        "condition_id_source": "manifest_validated",
        "curve_type": "idvd",
        "fixed_bias_V": vgs,
        "vgs_V": vgs,
        "ion_at_formal_bias_A": float(formal_ion),
        "ion_requested_vds_V": (
            float(formal_ion_sample.requested_voltage_V)
            if np.isfinite(formal_ion)
            else np.nan
        ),
        "ion_actual_vds_V": (
            float(formal_ion_sample.actual_voltage_V)
            if np.isfinite(formal_ion)
            else np.nan
        ),
        "ion_sampling_method": (
            str(formal_ion_sample.sampling_method)
            if np.isfinite(formal_ion)
            else "not_applicable"
        ),
        "ion_interpolated": (
            bool(formal_ion_sample.interpolated)
            if np.isfinite(formal_ion)
            else False
        ),
        **{key: value for key, value in metric.items() if key != "vgs"},
        "metric_status": status,
        **provenance,
    }


def _extract_all(
    inventory: pd.DataFrame,
    *,
    project_root: Path,
    condition: MeasurementContract,
    extraction: Mapping[str, Any],
    numerical_zero_current_A: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[str]]:
    idvg_rows: list[dict[str, Any]] = []
    idvd_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    active = inventory[
        (inventory["qc_status"] == "active")
        & (inventory["analysis_role"] == "formal")
    ]
    for record in active.itertuples(index=False):
        series = pd.Series(record._asdict())
        try:
            if record.curve_type == "idvg":
                row = _extract_idvg_record(
                    series,
                    project_root=project_root,
                    condition=condition,
                    numerical_zero_current_A=numerical_zero_current_A,
                )
                idvg_rows.append(row)
            else:
                row = _extract_idvd_record(
                    series,
                    project_root=project_root,
                    condition=condition,
                    numerical_zero_current_A=numerical_zero_current_A,
                    extraction=extraction,
                )
                idvd_rows.append(row)
            if int(row["numerical_zero_clipped_count"]) > 0:
                warnings.append(
                    f"{record.case_id} {record.curve_type} {record.input_path}: "
                    f"clipped {row['numerical_zero_clipped_count']} signed value(s) "
                    f"within ±{numerical_zero_current_A:.3g} A numerical-zero tolerance."
                )
        except Exception as error:
            errors.append(
                _error(
                    scope="curve_extraction",
                    error=error,
                    case_id=str(record.case_id),
                    curve_type=str(record.curve_type),
                    fixed_bias_V=float(record.fixed_bias_V),
                    input_path=str(record.input_path),
                )
            )
    return pd.DataFrame(idvg_rows), pd.DataFrame(idvd_rows), errors, warnings


def _extract_isolated_diagnostics(
    inventory: pd.DataFrame,
    *,
    project_root: Path,
    numerical_zero_current_A: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    selected = inventory[
        (inventory["qc_status"] == "isolated")
        & (inventory["analysis_role"] == "diagnostic_only")
    ]
    for record in selected.itertuples(index=False):
        try:
            path = project_root / str(record.input_path)
            frame, provenance = load_comsol_curve(
                path,
                curve_type=str(record.curve_type),
                fixed_bias_V=float(record.fixed_bias_V),
                numerical_zero_current_A=numerical_zero_current_A,
            )
            sweep_column = "vgs" if record.curve_type == "idvg" else "vds"
            current = frame["id_magnitude"].to_numpy(dtype=float)
            rows.append(
                {
                    "case_id": str(record.case_id),
                    "is_nominal": bool(record.is_nominal),
                    "curve_type": str(record.curve_type),
                    "fixed_bias_name": str(record.fixed_bias_name),
                    "fixed_bias_V": float(record.fixed_bias_V),
                    "width_m": float(record.width_m),
                    "length_m": float(record.length_m),
                    "oxide_thickness_m": float(record.oxide_thickness_m),
                    "temperature_K": float(record.temperature_K),
                    "qc_status": str(record.qc_status),
                    "analysis_role": str(record.analysis_role),
                    "qc_reason": str(record.qc_reason),
                    "input_path": str(record.input_path),
                    "input_sha256": str(record.sha256),
                    "point_count": int(len(frame)),
                    "sweep_min_V": float(frame[sweep_column].min()),
                    "sweep_max_V": float(frame[sweep_column].max()),
                    "current_min_A": float(np.min(current)),
                    "current_max_A": float(np.max(current)),
                    "current_span_A": float(np.max(current) - np.min(current)),
                    **provenance,
                }
            )
            if int(provenance["numerical_zero_clipped_count"]) > 0:
                warnings.append(
                    f"{record.case_id} diagnostic {record.input_path}: clipped "
                    f"{provenance['numerical_zero_clipped_count']} signed value(s) "
                    f"within ±{numerical_zero_current_A:.3g} A tolerance."
                )
        except Exception as error:
            errors.append(
                _error(
                    scope="isolated_diagnostic",
                    error=error,
                    case_id=str(record.case_id),
                    curve_type=str(record.curve_type),
                    fixed_bias_V=float(record.fixed_bias_V),
                    input_path=str(record.input_path),
                )
            )
    return pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS), errors, warnings


def _case_summary(
    manifest: COMSOLCaseManifest,
    inventory: pd.DataFrame,
    idvg: pd.DataFrame,
    idvd: pd.DataFrame,
    *,
    condition: MeasurementContract,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for case in manifest.cases:
        case_id = str(case["case_id"])
        case_idvg = idvg[idvg.get("case_id", pd.Series(dtype=str)) == case_id]
        case_idvd = idvd[idvd.get("case_id", pd.Series(dtype=str)) == case_id]
        try:
            low_row = select_exact_curve_row(
                case_idvg,
                case_id=case_id,
                curve_type="idvg",
                fixed_bias_V=condition.transfer.vds_V,
                condition_id=condition.condition_id,
            )
            high_row = select_exact_curve_row(
                case_idvg,
                case_id=case_id,
                curve_type="idvg",
                fixed_bias_V=condition.ion.vds_V,
                condition_id=condition.condition_id,
            )
            output_row = select_exact_curve_row(
                case_idvd,
                case_id=case_id,
                curve_type="idvd",
                fixed_bias_V=condition.ion.vgs_V,
                condition_id=condition.condition_id,
            )
            if low_row["vth_status"] != "ok" or high_row["vth_status"] != "ok":
                raise ValueError("DIBL requires two Vth extractions with status 'ok'.")
            if low_row["ss_status"] != "ok" or not bool(low_row["formal_eligible"]):
                raise ValueError(
                    "Formal transfer metrics require Vth and SS extraction status 'ok'."
                )
            primary_ion = float(high_row["ion_at_formal_bias_A"])
            verification_ion = float(output_row["ion_at_formal_bias_A"])
            ion_consistency_error = require_ion_consistency(
                primary_ion,
                verification_ion,
                measurement_contract=condition,
            )
            ioff = float(low_row["ioff_A"])
            same_vds_on_current = float(low_row["same_vds_on_current_A"])
        except (KeyError, TypeError, ValueError) as error:
            errors.append(_error(scope="case_summary", error=error, case_id=case_id))
            continue
        delta_vds = condition.dibl.high_vds_V - condition.dibl.low_vds_V
        dibl = (float(low_row["vth_V"]) - float(high_row["vth_V"])) / delta_vds
        case_inventory = inventory[inventory["case_id"] == case_id]
        rows.append(
            {
                "case_id": case_id,
                "condition_id": condition.condition_id,
                "condition_id_source": "manifest_validated",
                "is_nominal": case_id == manifest.nominal_case_id,
                "width_m": float(low_row["width_m"]),
                "length_m": float(low_row["length_m"]),
                "length_nm": float(low_row["length_nm"]),
                "oxide_thickness_m": float(low_row["oxide_thickness_m"]),
                "oxide_thickness_nm": float(low_row["oxide_thickness_nm"]),
                "temperature_K": float(low_row["temperature_K"]),
                "ioff_A": ioff,
                "ion_A": primary_ion,
                "ion_idvg_A": primary_ion,
                "ion_idvd_A": verification_ion,
                "ion_consistency_symmetric_relative_error": ion_consistency_error,
                "same_vds_on_current_A": same_vds_on_current,
                "ion_ioff_cross_bias": extract_ion_ioff_ratio(primary_ion, ioff),
                "ion_ioff_same_vds_0p1": extract_ion_ioff_ratio(
                    same_vds_on_current,
                    ioff,
                ),
                "gm_max_S": float(low_row["gm_max_S"]),
                "vth_V": float(low_row["vth_V"]),
                "vth_target_current_A": float(low_row["vth_target_current_A"]),
                "vth_status": str(low_row["vth_status"]),
                "ss_mV_per_dec": float(low_row["ss_mV_per_dec"]),
                "ss_status": str(low_row["ss_status"]),
                "formal_eligible": True,
                "formal_ineligibility_reason": "",
                "dibl_V_per_V": dibl,
                "dibl_mV_per_V": dibl * 1000.0,
                "dibl_low_vds_V": condition.dibl.low_vds_V,
                "dibl_high_vds_V": condition.dibl.high_vds_V,
                "ioff_vgs_V": condition.transfer.vgs_off_V,
                "ioff_vds_V": condition.transfer.vds_V,
                "ion_vgs_V": condition.ion.vgs_V,
                "ion_vds_V": condition.ion.vds_V,
                "ioff_sampling_method": str(low_row["ioff_sampling_method"]),
                "ioff_interpolated": bool(low_row["ioff_interpolated"]),
                "same_vds_sampling_method": str(
                    low_row["same_vds_sampling_method"]
                ),
                "same_vds_interpolated": bool(low_row["same_vds_interpolated"]),
                "ion_idvg_sampling_method": str(high_row["ion_sampling_method"]),
                "ion_idvg_interpolated": bool(high_row["ion_interpolated"]),
                "ion_idvd_sampling_method": str(output_row["ion_sampling_method"]),
                "ion_idvd_interpolated": bool(output_row["ion_interpolated"]),
                "transfer_source_path": low_row["input_path"],
                "ion_idvg_source_path": high_row["input_path"],
                "ion_idvd_source_path": output_row["input_path"],
                "dibl_low_source_path": low_row["input_path"],
                "dibl_high_source_path": high_row["input_path"],
                "ron_ohm_at_vgs_0p85": float(output_row["ron_ohm"]),
                "gds_S_at_vgs_0p85": float(output_row["gds_S"]),
                "active_curve_count": int((case_inventory["qc_status"] == "active").sum()),
                "isolated_curve_count": int((case_inventory["qc_status"] == "isolated").sum()),
                "formal_curve_count": int((case_inventory["analysis_role"] == "formal").sum()),
                "diagnostic_only_curve_count": int((case_inventory["analysis_role"] == "diagnostic_only").sum()),
                "inventory_only_curve_count": int((case_inventory["analysis_role"] == "inventory_only").sum()),
            }
        )
    return pd.DataFrame(rows), errors


def _observed_counts(
    inventory: pd.DataFrame,
    idvg: pd.DataFrame,
    idvd: pd.DataFrame,
    diagnostics: pd.DataFrame,
    cases: pd.DataFrame,
) -> dict[str, int]:
    return {
        **inventory_counts(inventory),
        "idvg_metric_rows": int(len(idvg)),
        "idvd_metric_rows": int(len(idvd)),
        "isolated_diagnostic_rows": int(len(diagnostics)),
        "case_summary_rows": int(len(cases)),
    }


def _contract_errors(inventory: pd.DataFrame, counts: Mapping[str, int]) -> list[dict[str, Any]]:
    expected = inventory_counts(inventory)
    errors = []
    for key, expected_value in expected.items():
        if counts[key] != expected_value:
            errors.append(
                _error(
                    scope="output_contract",
                    error=f"{key} must be {expected_value}, got {counts[key]}.",
                )
            )
    return errors


def run_characterization(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> CharacterizationResult:
    config_path = Path(config_path)
    root = Path(project_root or Path.cwd()).resolve()
    config = _load_yaml(config_path)
    comsol = _section(config, "comsol")
    characterization = _section(config, "characterization")
    extraction = _section(config, "extraction")

    manifest_value = comsol.get("case_manifest")
    if not manifest_value:
        raise ValueError("comsol.case_manifest is required for Characterization.")
    manifest_path = _project_path(str(manifest_value), root)
    manifest = load_comsol_case_manifest(
        manifest_path, project_root=root, include_independent_validation=False,
    )
    if comsol.get("nominal_case_id") != manifest.nominal_case_id:
        raise ValueError("project nominal_case_id must match the COMSOL case manifest.")

    condition_value = config.get("measurement_contract")
    if not condition_value:
        raise ValueError("measurement_contract is required for Characterization.")
    condition_path = _project_path(str(condition_value), root)
    condition = load_measurement_contract(condition_path)
    common = manifest.raw["common_conditions"]
    expected_conditions = {
        "width_m": condition.geometry.width_m,
        "temperature_K": condition.temperature_K,
    }
    for name, expected in expected_conditions.items():
        if not np.isclose(float(common[name]), expected, rtol=1e-9, atol=0.0):
            raise ValueError(f"Manifest {name} does not match the MeasurementContract.")
    if str(common["device_type"]).lower() != condition.device_type.lower():
        raise ValueError("Manifest device_type does not match the MeasurementContract.")

    output_root = _project_path(characterization.get("output_dir", "outputs/characterization"), root)
    paths = _paths(output_root)
    paths.table_dir.mkdir(parents=True, exist_ok=True)
    for owned in (
        paths.curve_inventory, paths.idvg_metrics, paths.idvd_metrics,
        paths.isolated_diagnostics, paths.case_summary,
        paths.extraction_errors, paths.workflow_manifest,
    ):
        owned.unlink(missing_ok=True)

    run_id = f"characterization_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    numerical_zero_current_A = float(characterization.get("numerical_zero_current_A", 1.0e-14))
    if numerical_zero_current_A < 0.0:
        raise ValueError("characterization.numerical_zero_current_A must be non-negative.")

    inventory = _curve_inventory(manifest)
    original_hashes = dict(zip(inventory["input_path"], inventory["sha256"]))
    idvg, idvd, errors, warnings = _extract_all(
        inventory,
        project_root=root,
        condition=condition,
        extraction=extraction,
        numerical_zero_current_A=numerical_zero_current_A,
    )
    diagnostics, diagnostic_errors, diagnostic_warnings = (
        _extract_isolated_diagnostics(
            inventory,
            project_root=root,
            numerical_zero_current_A=numerical_zero_current_A,
        )
    )
    errors.extend(diagnostic_errors)
    warnings.extend(diagnostic_warnings)

    configured_biases = {
        "low_vds_V": condition.dibl.low_vds_V,
        "high_vds_V": condition.dibl.high_vds_V,
        "summary_vgs_V": condition.ion.vgs_V,
    }
    for key, expected in configured_biases.items():
        if key in characterization and not np.isclose(
            float(characterization[key]), expected, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                f"characterization.{key} conflicts with MeasurementContract: "
                f"expected {expected}, got {characterization[key]}."
            )
    cases, case_errors = _case_summary(
        manifest,
        inventory,
        idvg,
        idvd,
        condition=condition,
    )
    errors.extend(case_errors)

    counts = _observed_counts(inventory, idvg, idvd, diagnostics, cases)
    errors.extend(_contract_errors(inventory, counts))

    for input_path, before_hash in original_hashes.items():
        after_hash = file_sha256(root / input_path)
        if after_hash != before_hash:
            errors.append(
                _error(
                    scope="input_integrity",
                    error="Input file hash changed during Characterization.",
                    input_path=input_path,
                )
            )

    inventory.to_csv(paths.curve_inventory, index=False)
    idvg.to_csv(paths.idvg_metrics, index=False)
    idvd.to_csv(paths.idvd_metrics, index=False)
    diagnostics.to_csv(paths.isolated_diagnostics, index=False)
    cases.to_csv(paths.case_summary, index=False)
    errors_frame = pd.DataFrame(errors, columns=ERROR_COLUMNS)
    errors_frame.to_csv(paths.extraction_errors, index=False)
    status = "FAIL" if errors else "WARNING" if warnings else "PASS"
    output_files = [
        paths.curve_inventory, paths.idvg_metrics, paths.idvd_metrics,
        paths.isolated_diagnostics, paths.case_summary, paths.extraction_errors,
    ]
    workflow_manifest = {
        "workflow": "characterization",
        "formal": True,
        "workflow_class": "formal_measurement_analysis",
        "run_id": run_id,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "software_version": software_version(),
        "git_commit": git_commit(root),
        "nominal_case_id": manifest.nominal_case_id,
        "condition_id": condition.condition_id,
        "numerical_zero_current_A": numerical_zero_current_A,
        "counts": counts,
        "warnings": warnings,
        "errors": errors,
        "config_files": [
            {"path": project_relative_path(config_path, root), "sha256": file_sha256(config_path)},
            {"path": project_relative_path(condition_path, root), "sha256": file_sha256(condition_path)},
            {"path": project_relative_path(manifest_path, root), "sha256": file_sha256(manifest_path)},
        ],
        "input_files": [
            {"path": row.input_path, "sha256": row.sha256}
            for row in inventory.itertuples(index=False)
        ],
        "output_files": [
            {"path": project_relative_path(path, root), "sha256": file_sha256(path)}
            for path in output_files
            if path.is_file()
        ],
        "interpretation_limit": (
            "Preparation validates registered curves; model validation is separate."
        ),
    }
    paths.workflow_manifest.write_text(
        json.dumps(workflow_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return CharacterizationResult(
        run_id=run_id,
        status=status,
        paths=paths,
        counts=counts,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
