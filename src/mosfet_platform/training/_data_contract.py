from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from mosfet_platform.io.comsol_curve import load_comsol_curve
from mosfet_platform.provenance import file_sha256
from mosfet_platform.training._optimizer import CurveData
from mosfet_platform.training._inventory import inventory_counts
from mosfet_platform.measurement import MeasurementContract, load_measurement_contract


class CalibrationContractError(ValueError):
    """Raised when Characterization cannot be used as an auditable Geometry calibration input."""


@dataclass(frozen=True)
class CalibrationInputs:
    project_root: Path
    config_path: Path
    config: Mapping[str, Any]
    calibration_config: Mapping[str, Any]
    characterization_root: Path
    characterization_manifest_path: Path
    characterization_manifest: Mapping[str, Any]
    condition_path: Path
    condition: MeasurementContract
    base_config_path: Path
    inventory: pd.DataFrame
    idvg_metrics: pd.DataFrame
    idvd_metrics: pd.DataFrame
    isolated_diagnostics: pd.DataFrame
    case_summary: pd.DataFrame
    curves: tuple[CurveData, ...]
    diagnostic_curves: tuple[CurveData, ...]
    warnings: tuple[str, ...]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return dict(raw)


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    raw = config.get(name, {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"'{name}' section must be a YAML mapping.")
    return dict(raw)


def _project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _manifest_hash_for_output(
    manifest: Mapping[str, Any],
    path: Path,
) -> str:
    entries = manifest.get("output_files")
    if not isinstance(entries, list):
        raise CalibrationContractError("Characterization manifest output_files is invalid.")
    candidates = [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and Path(str(entry.get("path", ""))).name == path.name
    ]
    if len(candidates) != 1:
        raise CalibrationContractError(
            f"Characterization manifest must record exactly one hash for {path.name}."
        )
    value = str(candidates[0].get("sha256", ""))
    if not value:
        raise CalibrationContractError(
            f"Characterization manifest hash is missing for {path.name}."
        )
    return value


def _read_verified_table(
    path: Path,
    manifest: Mapping[str, Any],
) -> pd.DataFrame:
    if not path.is_file():
        raise CalibrationContractError(f"Characterization table is missing: {path}")
    expected_hash = _manifest_hash_for_output(manifest, path)
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise CalibrationContractError(
            f"Characterization table hash mismatch for {path.name}."
        )
    return pd.read_csv(path, dtype={"case_id": str})


def _require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    label: str,
) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise CalibrationContractError(
            f"{label} is missing columns: {', '.join(sorted(missing))}."
        )


def _validate_characterization_contract(
    manifest: Mapping[str, Any],
    inventory: pd.DataFrame,
    idvg: pd.DataFrame,
    idvd: pd.DataFrame,
    diagnostics: pd.DataFrame,
    cases: pd.DataFrame,
) -> None:
    if manifest.get("workflow") != "characterization":
        raise CalibrationContractError("Upstream manifest is not a Characterization manifest.")
    if manifest.get("formal") is not True:
        raise CalibrationContractError("Characterization manifest must declare formal=true.")
    if manifest.get("workflow_class") != "formal_measurement_analysis":
        raise CalibrationContractError(
            "Characterization workflow_class must be formal_measurement_analysis."
        )
    if manifest.get("status") not in {"PASS", "WARNING"}:
        raise CalibrationContractError(
            "Characterization status must be PASS or WARNING before Geometry calibration."
        )
    expected = inventory_counts(inventory)
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise CalibrationContractError("Characterization manifest counts are invalid.")
    for name, value in expected.items():
        if int(counts.get(name, -1)) != value:
            raise CalibrationContractError(
                f"Characterization count {name} must be {value}."
            )
    actual = {
        "idvg_metric_rows": len(idvg),
        "idvd_metric_rows": len(idvd),
        "isolated_diagnostic_rows": len(diagnostics),
        "case_summary_rows": len(cases),
    }
    for name, value in actual.items():
        if value != expected[name]:
            raise CalibrationContractError(
                f"Characterization table count {name} must be {expected[name]}, got {value}."
            )
    if (
        cases["case_id"].duplicated().any()
        or set(cases["case_id"]) != set(inventory["case_id"])
        or int(cases["is_nominal"].sum()) != 1
    ):
        raise CalibrationContractError(
            "Characterization must cover each registered case once, with one nominal case."
        )


def _characterization_input_hashes(
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    entries = manifest.get("input_files")
    if not isinstance(entries, list):
        raise CalibrationContractError("Characterization manifest input_files is invalid.")
    result: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise CalibrationContractError("Characterization input hash entry is invalid.")
        path = str(entry.get("path", "")).replace("\\", "/")
        digest = str(entry.get("sha256", ""))
        if not path or not digest or path in result:
            raise CalibrationContractError(
                "Characterization input hashes must use unique non-empty paths."
            )
        result[path] = digest
    return result


def _curve_metric_point_count(
    row: pd.Series,
    idvg: pd.DataFrame,
    idvd: pd.DataFrame,
) -> int:
    metric = idvg if row["curve_type"] == "idvg" else idvd
    bias_column = "vds_V" if row["curve_type"] == "idvg" else "vgs_V"
    selected = metric[
        (metric["case_id"] == row["case_id"])
        & np.isclose(
            metric[bias_column].astype(float),
            float(row["fixed_bias_V"]),
            rtol=0.0,
            atol=1.0e-12,
        )
    ]
    if len(selected) != 1:
        raise CalibrationContractError(
            "Every active inventory curve must have exactly one Characterization metric row."
        )
    return int(selected.iloc[0]["point_count"])


def _verify_inventory_hashes(
    *,
    project_root: Path,
    inventory: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> None:
    manifest_hashes = _characterization_input_hashes(manifest)
    for row in inventory.itertuples(index=False):
        relative_path = str(row.input_path).replace("\\", "/")
        path = project_root / relative_path
        if not path.is_file():
            raise CalibrationContractError(
                f"Characterization registered raw curve is missing: {relative_path}."
            )
        actual_hash = file_sha256(path)
        inventory_hash = str(row.sha256)
        if actual_hash != inventory_hash:
            raise CalibrationContractError(
                f"Raw curve hash mismatch for {relative_path}."
            )
        if manifest_hashes.get(relative_path) != inventory_hash:
            raise CalibrationContractError(
                f"Characterization manifest input hash mismatch for {relative_path}."
            )


def _load_active_curves(
    *,
    project_root: Path,
    inventory: pd.DataFrame,
    idvg: pd.DataFrame,
    idvd: pd.DataFrame,
    manifest: Mapping[str, Any],
    numerical_zero_current_A: float,
) -> tuple[CurveData, ...]:
    manifest_hashes = _characterization_input_hashes(manifest)
    curves: list[CurveData] = []
    formal = inventory[
        (inventory["qc_status"] == "active")
        & (inventory["analysis_role"] == "formal")
    ]
    for _, row in formal.iterrows():
        relative_path = str(row["input_path"]).replace("\\", "/")
        path = project_root / relative_path
        if not path.is_file():
            raise CalibrationContractError(
                f"Characterization active raw curve is missing: {relative_path}."
            )
        actual_hash = file_sha256(path)
        inventory_hash = str(row["sha256"])
        if actual_hash != inventory_hash:
            raise CalibrationContractError(
                f"Raw curve hash mismatch for {relative_path}."
            )
        if manifest_hashes.get(relative_path) != inventory_hash:
            raise CalibrationContractError(
                f"Characterization manifest input hash mismatch for {relative_path}."
            )
        curve_type = str(row["curve_type"])
        frame, provenance = load_comsol_curve(
            path,
            curve_type=curve_type,
            fixed_bias_V=float(row["fixed_bias_V"]),
            numerical_zero_current_A=numerical_zero_current_A,
        )
        expected_points = _curve_metric_point_count(row, idvg, idvd)
        if len(frame) != expected_points:
            raise CalibrationContractError(
                f"Raw curve point count changed for {relative_path}."
            )
        sweep_column = "vgs" if curve_type == "idvg" else "vds"
        curves.append(
            CurveData(
                case_id=str(row["case_id"]),
                curve_type=curve_type,
                fixed_bias_V=float(row["fixed_bias_V"]),
                width_m=float(row["width_m"]),
                length_m=float(row["length_m"]),
                tox_m=float(row["oxide_thickness_m"]),
                temperature_K=float(row["temperature_K"]),
                input_path=relative_path,
                input_sha256=inventory_hash,
                sweep_values=frame[sweep_column].to_numpy(dtype=float),
                reference_current_A=frame["id_magnitude"].to_numpy(dtype=float),
                current_status=str(provenance["current_status"]),
                numerical_zero_clipped_count=int(
                    provenance["numerical_zero_clipped_count"]
                ),
            )
        )
    return tuple(curves)


def _load_diagnostic_curves(
    *,
    project_root: Path,
    inventory: pd.DataFrame,
    diagnostics: pd.DataFrame,
    numerical_zero_current_A: float,
) -> tuple[CurveData, ...]:
    curves: list[CurveData] = []
    selected = inventory[
        (inventory["qc_status"] == "isolated")
        & (inventory["analysis_role"] == "diagnostic_only")
    ]
    for _, row in selected.iterrows():
        curve_type = str(row["curve_type"])
        fixed_bias = float(row["fixed_bias_V"])
        metric = diagnostics[
            (diagnostics["case_id"] == row["case_id"])
            & (diagnostics["curve_type"] == curve_type)
            & np.isclose(
                diagnostics["fixed_bias_V"].astype(float),
                fixed_bias,
                rtol=0.0,
                atol=1.0e-12,
            )
        ]
        if len(metric) != 1:
            raise CalibrationContractError(
                "Every diagnostic-only curve must have exactly one Characterization "
                "isolated diagnostic row."
            )
        relative_path = str(row["input_path"]).replace("\\", "/")
        path = project_root / relative_path
        frame, provenance = load_comsol_curve(
            path,
            curve_type=curve_type,
            fixed_bias_V=fixed_bias,
            numerical_zero_current_A=numerical_zero_current_A,
        )
        if len(frame) != int(metric.iloc[0]["point_count"]):
            raise CalibrationContractError(
                f"Diagnostic raw curve point count changed for {relative_path}."
            )
        sweep_column = "vgs" if curve_type == "idvg" else "vds"
        curves.append(
            CurveData(
                case_id=str(row["case_id"]),
                curve_type=curve_type,
                fixed_bias_V=fixed_bias,
                width_m=float(row["width_m"]),
                length_m=float(row["length_m"]),
                tox_m=float(row["oxide_thickness_m"]),
                temperature_K=float(row["temperature_K"]),
                input_path=relative_path,
                input_sha256=str(row["sha256"]),
                sweep_values=frame[sweep_column].to_numpy(dtype=float),
                reference_current_A=frame["id_magnitude"].to_numpy(dtype=float),
                current_status=str(provenance["current_status"]),
                numerical_zero_clipped_count=int(
                    provenance["numerical_zero_clipped_count"]
                ),
            )
        )
    return tuple(curves)


def load_calibration_inputs(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> CalibrationInputs:
    """Load verified Characterization formal curves plus linked QC diagnostics."""

    root = Path(project_root or Path.cwd()).resolve()
    config_path = Path(config_path)
    config = _load_yaml(config_path)
    calibration = _section(config, "geometry_calibration")
    characterization_root = _project_path(
        calibration.get("characterization_input_dir", "outputs/characterization"),
        root,
    )
    manifest_path = characterization_root / "workflow_manifest.json"
    if not manifest_path.is_file():
        raise CalibrationContractError(
            f"Characterization manifest is missing: {manifest_path}."
        )
    manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_raw, Mapping):
        raise CalibrationContractError("Characterization manifest must contain a mapping.")
    manifest = dict(manifest_raw)

    table_root = characterization_root / "tables"
    inventory = _read_verified_table(
        table_root / "curve_inventory.csv",
        manifest,
    )
    idvg = _read_verified_table(
        table_root / "idvg_metrics_by_curve.csv",
        manifest,
    )
    idvd = _read_verified_table(
        table_root / "idvd_metrics_by_curve.csv",
        manifest,
    )
    diagnostics = _read_verified_table(
        table_root / "isolated_curve_diagnostics.csv",
        manifest,
    )
    cases = _read_verified_table(table_root / "case_summary.csv", manifest)
    _require_columns(
        inventory,
        {
            "case_id",
            "curve_type",
            "fixed_bias_V",
            "width_m",
            "length_m",
            "oxide_thickness_m",
            "temperature_K",
            "qc_status",
            "analysis_role",
            "input_path",
            "sha256",
        },
        "curve_inventory.csv",
    )
    _require_columns(
        idvg,
        {"case_id", "vds_V", "point_count", "vth_V", "ss_mV_per_dec"},
        "idvg_metrics_by_curve.csv",
    )
    _require_columns(
        idvd,
        {"case_id", "vgs_V", "point_count", "lambda_estimate_1_V"},
        "idvd_metrics_by_curve.csv",
    )
    _require_columns(
        diagnostics,
        {
            "case_id",
            "curve_type",
            "fixed_bias_V",
            "analysis_role",
            "qc_reason",
            "point_count",
            "input_path",
            "input_sha256",
        },
        "isolated_curve_diagnostics.csv",
    )
    _require_columns(
        cases,
        {
            "case_id",
            "is_nominal",
            "width_m",
            "length_m",
            "oxide_thickness_m",
            "temperature_K",
            "ion_A",
            "vth_V",
            "ss_mV_per_dec",
            "dibl_V_per_V",
        },
        "case_summary.csv",
    )
    _validate_characterization_contract(
        manifest, inventory, idvg, idvd, diagnostics, cases
    )
    numerical_zero = float(
        calibration.get(
            "numerical_zero_current_A",
            manifest.get("numerical_zero_current_A", 1.0e-14),
        )
    )
    if numerical_zero < 0.0 or not np.isclose(
        numerical_zero,
        float(manifest.get("numerical_zero_current_A", numerical_zero)),
        rtol=0.0,
        atol=0.0,
    ):
        raise CalibrationContractError(
            "Geometry calibration numerical-zero tolerance must match Characterization."
        )
    _verify_inventory_hashes(
        project_root=root,
        inventory=inventory,
        manifest=manifest,
    )
    curves = _load_active_curves(
        project_root=root,
        inventory=inventory,
        idvg=idvg,
        idvd=idvd,
        manifest=manifest,
        numerical_zero_current_A=numerical_zero,
    )
    diagnostic_curves = _load_diagnostic_curves(
        project_root=root,
        inventory=inventory,
        diagnostics=diagnostics,
        numerical_zero_current_A=numerical_zero,
    )
    condition_value = config.get("measurement_contract")
    if not condition_value:
        raise ValueError("measurement_contract is required for Geometry calibration.")
    condition_path = _project_path(str(condition_value), root)
    condition = load_measurement_contract(condition_path)
    model = _section(config, "model")
    base_value = model.get("base_config")
    if not base_value:
        raise ValueError("model.base_config is required for Geometry calibration.")
    base_config_path = _project_path(str(base_value), root)
    if not base_config_path.is_file():
        raise FileNotFoundError(
            f"Geometry calibration base model config is missing: {base_config_path}"
        )

    warnings = tuple(str(value) for value in manifest.get("warnings", []))
    return CalibrationInputs(
        project_root=root,
        config_path=config_path,
        config=config,
        calibration_config=calibration,
        characterization_root=characterization_root,
        characterization_manifest_path=manifest_path,
        characterization_manifest=manifest,
        condition_path=condition_path,
        condition=condition,
        base_config_path=base_config_path,
        inventory=inventory,
        idvg_metrics=idvg,
        idvd_metrics=idvd,
        isolated_diagnostics=diagnostics,
        case_summary=cases,
        curves=curves,
        diagnostic_curves=diagnostic_curves,
        warnings=warnings,
    )
