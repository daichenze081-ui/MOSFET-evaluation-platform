from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from mosfet_platform.io.comsol_curve import load_comsol_curve

from mosfet_platform.training._data_contract import (
    CalibrationInputs,
    load_calibration_inputs,
)
from mosfet_platform.training._inventory import inventory_counts
from mosfet_platform.training.fit import (
    build_initial_model,
    optimization_settings_from_config,
    validation_limits_from_config,
)
from mosfet_platform.provenance import file_sha256, git_commit, project_relative_path, software_version
from mosfet_platform.case_manifest import load_comsol_case_manifest
from mosfet_platform.training._optimizer import (
    GeometryCalibrationResult,
    CurveData,
    OptimizationSettings,
    fit_geometry_aware_model,
)
from mosfet_platform.training._validation_metrics import (
    ValidationLimits,
    evaluate_model_curves,
    summarize_case_validation,
)
from mosfet_platform.model.geometry_aware import GeometryAwareModelParameters


@dataclass(frozen=True)
class GeneralizationPaths:
    root: Path
    table_dir: Path
    model_dir: Path
    validation_summary: Path
    selected_model: Path
    workflow_manifest: Path


@dataclass(frozen=True)
class GeneralizationResult:
    run_id: str
    status: str
    validation_status: str
    prediction_ready: bool
    qualification_ready: bool
    selected_family: str
    paths: GeneralizationPaths
    counts: Mapping[str, int]
    warnings: tuple[str, ...]
    errors: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class CandidateArtifacts:
    family: str
    model: GeometryAwareModelParameters
    fit_result: GeometryCalibrationResult
    loco_cases: pd.DataFrame
    lg_rows: pd.DataFrame
    tox_columns: pd.DataFrame
    envelope: pd.DataFrame
    training_cases: pd.DataFrame

    @property
    def validation_rows(self) -> pd.DataFrame:
        return pd.concat(
            [self.training_cases, self.loco_cases, self.lg_rows, self.tox_columns],
            ignore_index=True,
        )

    @property
    def validation_pass(self) -> bool:
        rows = self.validation_rows
        return not rows.empty and rows["validation_status"].eq("PASS").all()

    @property
    def envelope_pass(self) -> bool:
        return bool(
            not self.envelope.empty
            and self.envelope["physical_pass"].astype(bool).all()
        )

    @property
    def worst_ratio(self) -> float:
        return float(self.validation_rows["worst_normalized_gate_ratio"].max())


def _paths(root: Path) -> GeneralizationPaths:
    table_dir = root / "tables"
    model_dir = root / "model"
    return GeneralizationPaths(
        root=root,
        table_dir=table_dir,
        model_dir=model_dir,
        validation_summary=table_dir / "validation_summary.csv",
        selected_model=model_dir / "selected_geometry_aware_model.yaml",
        workflow_manifest=root / "workflow_manifest.json",
    )


def _project_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    raw = config.get(name, {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} config must be a YAML mapping.")
    return dict(raw)


def _load_independent_validation_curves(
    config: Mapping[str, Any],
    *,
    root: Path,
    numerical_zero_current_A: float,
) -> tuple[
    tuple[CurveData, ...],
    pd.DataFrame,
    list[dict[str, Any]],
    int,
    Path,
]:
    comsol = _section(config, "comsol")
    manifest_value = comsol.get("case_manifest")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise ValueError("comsol.case_manifest is required for independent validation.")
    manifest_path = _project_path(manifest_value, root)
    manifest = load_comsol_case_manifest(manifest_path, project_root=root)
    if not manifest.independent_validation_cases:
        raise ValueError("Required independent validation has no registered cases.")
    common = manifest.raw["common_conditions"]
    width_m = float(common["width_m"])
    temperature_K = float(common["temperature_K"])
    curves: list[CurveData] = []
    geometries: list[dict[str, Any]] = []
    input_files: list[dict[str, Any]] = []
    diagnostic_count = 0
    for case in manifest.independent_validation_cases:
        geometry = case["geometry"]
        length_m = float(geometry["length_m"])
        tox_m = float(geometry["oxide_thickness_m"])
        geometries.append(
            {
                "case_id": str(case["case_id"]),
                "length_m": length_m,
                "oxide_thickness_m": tox_m,
            }
        )
        for curve_type in ("idvg", "idvd"):
            bias_key = "vds_V" if curve_type == "idvg" else "vgs_V"
            for curve in case[curve_type]:
                relative_path = str(curve["path"]).replace("\\", "/")
                path = root / relative_path
                digest = file_sha256(path)
                input_files.append(
                    {
                        "path": relative_path,
                        "sha256": digest,
                        "case_id": str(case["case_id"]),
                        "curve_type": curve_type,
                        "qc_status": str(curve["qc_status"]),
                    }
                )
                if curve["analysis_role"] != "formal":
                    diagnostic_count += int(curve["analysis_role"] == "diagnostic_only")
                    continue
                frame, provenance = load_comsol_curve(
                    path,
                    curve_type=curve_type,
                    fixed_bias_V=float(curve[bias_key]),
                    numerical_zero_current_A=numerical_zero_current_A,
                )
                sweep_column = "vgs" if curve_type == "idvg" else "vds"
                curves.append(
                    CurveData(
                        case_id=str(case["case_id"]),
                        curve_type=curve_type,
                        fixed_bias_V=float(curve[bias_key]),
                        width_m=width_m,
                        length_m=length_m,
                        tox_m=tox_m,
                        temperature_K=temperature_K,
                        input_path=relative_path,
                        input_sha256=digest,
                        sweep_values=frame[sweep_column].to_numpy(dtype=float),
                        reference_current_A=frame["id_magnitude"].to_numpy(dtype=float),
                        current_status=str(provenance["current_status"]),
                        numerical_zero_clipped_count=int(
                            provenance["numerical_zero_clipped_count"]
                        ),
                    )
                )

    expected_cases = {str(case["case_id"]) for case in manifest.independent_validation_cases}
    if {curve.case_id for curve in curves} != expected_cases:
        raise ValueError("Every independent validation case needs formal curves.")
    return (
        tuple(curves),
        pd.DataFrame(geometries),
        input_files,
        diagnostic_count,
        manifest_path,
    )

def _gate_ratios(row: Mapping[str, Any], limits: ValidationLimits) -> dict[str, float]:
    pairs = {
        "idvg_log_rmse_ratio": ("idvg_log_rmse_dec", limits.idvg_log_rmse_dec_max),
        "ion_error_ratio": ("ion_error_percent", limits.ion_error_percent_max),
        "vth_error_ratio": ("vth_error_mV", limits.vth_error_mV_max),
        "ss_error_ratio": ("ss_error_mV_per_dec", limits.ss_error_mV_per_dec_max),
        "dibl_error_ratio": ("dibl_error_mV_per_V", limits.dibl_error_mV_per_V_max),
        "idvd_nrmse_ratio": ("idvd_nrmse", limits.idvd_nrmse_max),
    }
    result: dict[str, float] = {}
    for output, (source, limit) in pairs.items():
        value = float(row[source])
        result[output] = value / limit if limit > 0.0 else (0.0 if value == 0.0 else np.inf)
    result["worst_normalized_gate_ratio"] = max(result.values())
    return result


def _evaluate_case_set(
    model: GeometryAwareModelParameters,
    curves: tuple[CurveData, ...],
    geometries: pd.DataFrame,
    *,
    condition: Any,
    limits: ValidationLimits,
    training_case_ids: list[str],
    training_curve_count: int,
    scope: str,
) -> pd.DataFrame:
    curve_metrics, predictions = evaluate_model_curves(
        model,
        curves,
        condition=condition,
    )
    summary = summarize_case_validation(
        curve_metrics,
        predictions,
        limits=limits,
        measurement_contract=condition,
    )
    ratios = pd.DataFrame(
        [_gate_ratios(row._asdict(), limits) for row in summary.itertuples(index=False)]
    )
    summary = pd.concat([summary.reset_index(drop=True), ratios], axis=1)
    summary = summary.merge(geometries, on="case_id", validate="one_to_one")
    summary.insert(0, "model_family", model.geometry_family)
    summary.insert(1, "validation_scope", scope)
    summary.insert(2, "split_type", scope)
    summary.insert(3, "fold", np.arange(len(summary), dtype=int))
    summary.insert(
        4,
        "split_key",
        [f"{scope}:{index}" for index in range(len(summary))],
    )
    summary["holdout_case_ids"] = summary["case_id"]
    summary["training_case_ids"] = ";".join(sorted(training_case_ids))
    summary["training_case_count"] = len(training_case_ids)
    summary["holdout_case_count"] = 1
    summary["training_curve_count"] = int(training_curve_count)
    curve_counts = pd.Series([curve.case_id for curve in curves]).value_counts()
    summary["holdout_curve_count"] = summary["case_id"].map(curve_counts)
    summary["optimizer_iterations"] = np.nan
    summary["objective_before"] = np.nan
    summary["objective_after"] = np.nan
    summary["boundary_parameters"] = ""
    return summary.drop(columns=["case_id"])


def _holdout_split(
    inputs: CalibrationInputs,
    *,
    family: str,
    split_type: str,
    fold: int,
    holdout_case_ids: set[str],
    settings: OptimizationSettings,
    limits: ValidationLimits,
) -> dict[str, Any]:
    all_case_ids = set(str(value) for value in inputs.case_summary["case_id"])
    training_case_ids = all_case_ids - holdout_case_ids
    training_curves = tuple(c for c in inputs.curves if c.case_id in training_case_ids)
    holdout_curves = tuple(c for c in inputs.curves if c.case_id in holdout_case_ids)
    if not training_curves or not holdout_curves:
        raise ValueError(f"{split_type} fold {fold} has an empty train or holdout set.")
    try:
        initial = build_initial_model(
            inputs,
            training_case_ids=training_case_ids,
            settings=settings,
            model_family=family,
        )
    except ValueError as error:
        raise ValueError(f"{family} {split_type} fold {fold}: {error}") from error
    fitted, fit_result = fit_geometry_aware_model(
        training_curves,
        initial,
        settings=settings,
    )
    curve_metrics, predictions = evaluate_model_curves(
        fitted,
        holdout_curves,
        condition=inputs.condition,
    )
    case_metrics = summarize_case_validation(
        curve_metrics,
        predictions,
        limits=limits,
        measurement_contract=inputs.condition,
    )
    if set(case_metrics["case_id"].astype(str)) != holdout_case_ids:
        raise RuntimeError(f"{split_type} fold {fold} holdout coverage is incomplete.")
    ratios = [_gate_ratios(row._asdict(), limits) for row in case_metrics.itertuples(index=False)]
    ratio_frame = pd.DataFrame(ratios)
    errors = {
        name: float(case_metrics[name].max())
        for name in (
            "idvg_log_rmse_dec",
            "ion_error_percent",
            "vth_error_mV",
            "ss_error_mV_per_dec",
            "dibl_error_mV_per_V",
            "idvd_nrmse",
        )
    }
    return {
        "model_family": family,
        "validation_scope": "training_grouped",
        "split_type": split_type,
        "fold": int(fold),
        "split_key": f"{split_type}:{fold}",
        "holdout_case_ids": ";".join(sorted(holdout_case_ids)),
        "training_case_ids": ";".join(sorted(training_case_ids)),
        "training_case_count": len(training_case_ids),
        "holdout_case_count": len(holdout_case_ids),
        "training_curve_count": len(training_curves),
        "holdout_curve_count": len(holdout_curves),
        "optimizer_iterations": fit_result.iteration_count,
        "objective_before": fit_result.objective_before,
        "objective_after": fit_result.objective_after,
        "boundary_parameters": ";".join(fit_result.boundary_parameters),
        **errors,
        "worst_normalized_gate_ratio": float(
            ratio_frame["worst_normalized_gate_ratio"].max()
        ),
        "validation_status": (
            "PASS" if (case_metrics["validation_status"] == "PASS").all() else "FAIL"
        ),
    }


def _split_definitions(inputs: CalibrationInputs) -> dict[str, list[set[str]]]:
    cases = inputs.case_summary.copy()
    loco = [{str(case_id)} for case_id in sorted(cases["case_id"].astype(str))]
    lg_rows = [
        set(cases.loc[np.isclose(cases["length_m"], value, rtol=1.0e-9, atol=0.0), "case_id"].astype(str))
        for value in sorted(cases["length_m"].unique())
    ]
    tox_columns = [
        set(
            cases.loc[
                np.isclose(cases["oxide_thickness_m"], value, rtol=1.0e-9, atol=0.0), "case_id"
            ].astype(str)
        )
        for value in sorted(cases["oxide_thickness_m"].unique())
    ]
    return {"loco": loco, "lg_row": lg_rows, "tox_column": tox_columns}


def _scan_values(start: float, stop: float, step: float) -> np.ndarray:
    if not np.isfinite((start, stop, step)).all() or step <= 0.0 or stop < start:
        raise ValueError("Envelope scan bounds and step are invalid.")
    count = int(round((stop - start) / step)) + 1
    values = start + np.arange(count, dtype=float) * step
    if not np.isclose(values[-1], stop, rtol=0.0, atol=1.0e-10):
        raise ValueError("Envelope scan step does not land on the stop value.")
    values[-1] = stop
    return values


def envelope_scan(
    model: GeometryAwareModelParameters,
    config: Mapping[str, Any],
    *,
    condition: Any,
) -> pd.DataFrame:
    envelope = model.envelope
    length_step_nm = (envelope.length_max_m - envelope.length_min_m) * 1e9 / 10
    tox_step_nm = (envelope.tox_max_m - envelope.tox_min_m) * 1e9 / 10
    lengths = _scan_values(
        float(config.get("length_start_nm", envelope.length_min_m * 1e9)),
        float(config.get("length_stop_nm", envelope.length_max_m * 1e9)),
        float(config.get("length_step_nm", max(length_step_nm, 1e-9))),
    )
    toxes = _scan_values(
        float(config.get("tox_start_nm", envelope.tox_min_m * 1e9)),
        float(config.get("tox_stop_nm", envelope.tox_max_m * 1e9)),
        float(config.get("tox_step_nm", max(tox_step_nm, 1e-9))),
    )
    scan_bounds = (lengths[0], lengths[-1], toxes[0], toxes[-1])
    model_bounds = np.array((
        envelope.length_min_m, envelope.length_max_m,
        envelope.tox_min_m, envelope.tox_max_m,
    )) * 1e9
    if not np.allclose(scan_bounds, model_bounds, rtol=1e-9, atol=1e-10):
        raise ValueError("Envelope scan must cover the full declared model envelope.")
    abs_tol = float(config.get("monotonic_absolute_tolerance_A", 1.0e-15))
    rel_tol = float(config.get("monotonic_relative_tolerance", 1.0e-9))
    if abs_tol < 0.0 or rel_tol < 0.0:
        raise ValueError("Envelope monotonic tolerances must be non-negative.")
    vgs_grid = np.linspace(
        condition.transfer.vgs_start_V,
        condition.transfer.vgs_stop_V,
        condition.transfer.num_points,
    )
    vds_grid = np.linspace(0.0, condition.ion.vds_V, condition.transfer.num_points)
    rows: list[dict[str, Any]] = []
    for length_nm in lengths:
        for tox_nm in toxes:
            length_m = float(length_nm * 1.0e-9)
            tox_m = float(tox_nm * 1.0e-9)
            violations: list[str] = []
            try:
                params = model.effective_parameters(
                    length_m,
                    tox_m,
                    device_width_m=model.width_m,
                )
                parameter_values = np.asarray(
                    [
                        params.vth,
                        params.mu,
                        params.subthreshold_n,
                        params.dibl_coeff,
                        params.lambda_clm,
                    ],
                    dtype=float,
                )
                parameter_pass = bool(
                    np.isfinite(parameter_values).all()
                    and np.all(parameter_values[1:] > 0.0)
                )
                min_idvg_diff = np.inf
                min_idvd_diff = np.inf
                current_pass = True
                monotonic_pass = True
                for vds in (condition.transfer.vds_V, condition.ion.vds_V):
                    current = np.asarray(
                        model.ids(
                            vgs=vgs_grid,
                            vds=vds,
                            length_m=length_m,
                            tox_m=tox_m,
                            device_width_m=model.width_m,
                        ),
                        dtype=float,
                    )
                    current_pass &= bool(np.isfinite(current).all() and np.all(current >= 0.0))
                    diff = np.diff(current)
                    min_idvg_diff = min(min_idvg_diff, float(diff.min()))
                    tolerance = abs_tol + rel_tol * max(float(np.max(np.abs(current))), 1.0e-30)
                    monotonic_pass &= bool(np.all(diff >= -tolerance))
                for vgs in np.linspace(condition.transfer.vgs_start_V, condition.transfer.vgs_stop_V, 5):
                    current = np.asarray(
                        model.ids(
                            vgs=vgs,
                            vds=vds_grid,
                            length_m=length_m,
                            tox_m=tox_m,
                            device_width_m=model.width_m,
                        ),
                        dtype=float,
                    )
                    current_pass &= bool(np.isfinite(current).all() and np.all(current >= 0.0))
                    diff = np.diff(current)
                    min_idvd_diff = min(min_idvd_diff, float(diff.min()))
                    tolerance = abs_tol + rel_tol * max(float(np.max(np.abs(current))), 1.0e-30)
                    monotonic_pass &= bool(np.all(diff >= -tolerance))
                if not parameter_pass:
                    violations.append("invalid_effective_parameters")
                if not current_pass:
                    violations.append("invalid_current")
                if not monotonic_pass:
                    violations.append("nonmonotonic_current")
            except (ValueError, FloatingPointError, OverflowError) as error:
                params = None
                parameter_pass = current_pass = monotonic_pass = False
                min_idvg_diff = min_idvd_diff = np.nan
                violations.append(f"evaluation_error:{type(error).__name__}")
            rows.append(
                {
                    "model_family": model.geometry_family,
                    "length_nm": float(length_nm),
                    "tox_nm": float(tox_nm),
                    "vth_V": float(params.vth) if params is not None else np.nan,
                    "mu_m2_per_Vs": float(params.mu) if params is not None else np.nan,
                    "subthreshold_n": float(params.subthreshold_n) if params is not None else np.nan,
                    "dibl_coeff_V_per_V": float(params.dibl_coeff) if params is not None else np.nan,
                    "lambda_1_per_V": float(params.lambda_clm) if params is not None else np.nan,
                    "min_idvg_delta_A": min_idvg_diff,
                    "min_idvd_delta_A": min_idvd_diff,
                    "parameter_pass": parameter_pass,
                    "current_pass": current_pass,
                    "monotonic_pass": monotonic_pass,
                    "physical_pass": parameter_pass and current_pass and monotonic_pass,
                    "violations": ";".join(violations),
                }
            )
    frame = pd.DataFrame(rows)
    return frame


def _run_candidate(
    inputs: CalibrationInputs,
    *,
    family: str,
    model_generalization: Mapping[str, Any],
) -> CandidateArtifacts:
    settings = optimization_settings_from_config(inputs.calibration_config)
    settings = replace(
        settings,
        interaction_regularization_weight=float(
            model_generalization.get("interaction_regularization_weight", 1.0e-3)
        ),
    )
    limits = validation_limits_from_config(inputs.calibration_config)
    definitions = (
        _split_definitions(inputs)
        if model_generalization.get("cross_validation", "disabled") == "grouped"
        else {name: [] for name in ("loco", "lg_row", "tox_column")}
    )
    frames: dict[str, pd.DataFrame] = {}
    for split_type, groups in definitions.items():
        rows = [
            _holdout_split(
                inputs,
                family=family,
                split_type=split_type,
                fold=fold,
                holdout_case_ids=holdout,
                settings=settings,
                limits=limits,
            )
            for fold, holdout in enumerate(groups)
        ]
        frames[split_type] = pd.DataFrame(rows)
    initial = build_initial_model(inputs, settings=settings, model_family=family)
    model, fit_result = fit_geometry_aware_model(inputs.curves, initial, settings=settings)
    training_cases = _evaluate_case_set(
        model, inputs.curves,
        inputs.case_summary[["case_id", "length_m", "oxide_thickness_m"]],
        condition=inputs.condition, limits=limits,
        training_case_ids=sorted(inputs.case_summary["case_id"]),
        training_curve_count=len(inputs.curves), scope="training_fit",
    )
    scan_raw = model_generalization.get("envelope_scan", {})
    if not isinstance(scan_raw, Mapping):
        raise ValueError("model_generalization.envelope_scan must be a mapping.")
    scan = envelope_scan(model, scan_raw, condition=inputs.condition)
    return CandidateArtifacts(
        family=family,
        model=model,
        fit_result=fit_result,
        loco_cases=frames["loco"],
        lg_rows=frames["lg_row"],
        tox_columns=frames["tox_column"],
        envelope=scan,
        training_cases=training_cases,
    )


def _select_candidate(
    additive: CandidateArtifacts,
    interaction: CandidateArtifacts,
    model_generalization: Mapping[str, Any],
) -> tuple[CandidateArtifacts, str, float, float]:
    improvement = (
        (additive.worst_ratio - interaction.worst_ratio) / additive.worst_ratio
        if additive.worst_ratio > 0.0
        else 0.0
    )
    additive_groups = additive.validation_rows[["split_key", "worst_normalized_gate_ratio"]].rename(
        columns={"worst_normalized_gate_ratio": "additive_ratio"}
    )
    interaction_groups = interaction.validation_rows[["split_key", "worst_normalized_gate_ratio"]].rename(
        columns={"worst_normalized_gate_ratio": "interaction_ratio"}
    )
    comparison = additive_groups.merge(interaction_groups, on="split_key", validate="one_to_one")
    comparison["worsening_fraction"] = (
        (comparison["interaction_ratio"] - comparison["additive_ratio"])
        / comparison["additive_ratio"].clip(lower=1.0e-30)
    )
    max_worsening = float(comparison["worsening_fraction"].max())
    improvement_min = float(model_generalization.get("interaction_improvement_min_fraction", 0.10))
    worsening_max = float(model_generalization.get("group_worsening_max_fraction", 0.05))
    additive_pass = additive.validation_pass and additive.envelope_pass
    interaction_pass = interaction.validation_pass and interaction.envelope_pass
    if not additive_pass and interaction_pass:
        return interaction, "interaction_passes_while_additive_fails", improvement, max_worsening
    if (
        additive_pass
        and interaction_pass
        and improvement >= improvement_min
        and max_worsening <= worsening_max
    ):
        return interaction, "interaction_meets_conservative_promotion_rule", improvement, max_worsening
    if additive_pass:
        return additive, "additive_retained_by_default", improvement, max_worsening
    return additive, "both_candidates_fail_additive_retained_for_diagnosis", improvement, max_worsening


def run_model_generalization(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> GeneralizationResult:
    root = Path(project_root or Path.cwd()).resolve()
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Project config must contain a YAML mapping.")
    config = dict(raw)
    settings = _section(config, "model_generalization")
    policies = {
        "model_family": ("additive", {"additive", "interaction", "auto"}),
        "cross_validation": ("disabled", {"disabled", "grouped"}),
        "independent_validation": ("disabled", {"disabled", "required"}),
    }
    for name, (default, allowed) in policies.items():
        value = settings.get(name, default)
        if value not in allowed:
            raise ValueError(f"model_generalization.{name} must be one of {sorted(allowed)}.")
        settings[name] = value

    paths = _paths(_project_path(settings.get("output_dir", "outputs/model_generalization"), root))
    for directory in (paths.table_dir, paths.model_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for path in asdict(paths).values():
        if path.is_file():
            path.unlink()
    run_id = f"model_generalization_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    inputs = load_calibration_inputs(config_path, project_root=root)
    case_manifest_path = _project_path(config["comsol"]["case_manifest"], root)
    before_hashes = {
        str(row.input_path): str(row.sha256)
        for row in inputs.inventory.itertuples(index=False)
    }
    independent_curves: tuple[CurveData, ...] = ()
    independent_geometry = pd.DataFrame()
    independent_input_files: list[dict[str, Any]] = []
    independent_diagnostic_count = 0
    if settings["independent_validation"] == "required":
        (
            independent_curves, independent_geometry, independent_input_files,
            independent_diagnostic_count, case_manifest_path,
        ) = _load_independent_validation_curves(
            config, root=root,
            numerical_zero_current_A=float(inputs.characterization_manifest["numerical_zero_current_A"]),
        )
        before_hashes.update({str(row["path"]): str(row["sha256"]) for row in independent_input_files})

    families = {
        "additive": ("additive",), "interaction": ("interaction",),
        "auto": ("additive", "interaction"),
    }[settings["model_family"]]
    candidates = [
        _run_candidate(inputs, family=family, model_generalization=settings)
        for family in families
    ]
    selected = candidates[0]
    reason = f"configured_{selected.family}"
    improvement = max_worsening = 0.0
    if len(candidates) == 2:
        selected, reason, improvement, max_worsening = _select_candidate(*candidates, settings)

    independent = pd.DataFrame()
    independent_status = "NOT_RUN"
    if settings["independent_validation"] == "required":
        for case in independent_geometry.itertuples(index=False):
            selected.model.envelope.require_contains(case.length_m, case.oxide_thickness_m)
        independent = _evaluate_case_set(
            selected.model, independent_curves, independent_geometry,
            condition=inputs.condition,
            limits=validation_limits_from_config(inputs.calibration_config),
            training_case_ids=sorted(inputs.case_summary["case_id"].astype(str)),
            training_curve_count=len(inputs.curves), scope="independent",
        )
        independent_status = "PASS" if independent["validation_status"].eq("PASS").all() else "FAIL"

    validation = pd.concat(
        [candidate.validation_rows for candidate in candidates] + [independent],
        ignore_index=True, sort=False,
    )
    counts = inventory_counts(inputs.inventory)
    counts.update({
        "independent_validation_cases": len(independent),
        "independent_validation_formal_curves": len(independent_curves),
        "independent_validation_diagnostic_curves": independent_diagnostic_count,
        "candidate_models": len(candidates),
        "training_fit_rows": sum(len(candidate.training_cases) for candidate in candidates),
        "loco_rows": sum(len(candidate.loco_cases) for candidate in candidates),
        "lg_row_holdout_rows": sum(len(candidate.lg_rows) for candidate in candidates),
        "tox_column_holdout_rows": sum(len(candidate.tox_columns) for candidate in candidates),
        "independent_validation_rows": len(independent),
        "validation_rows": len(validation),
        "selected_envelope_rows": len(selected.envelope),
        "selected_envelope_violation_rows": int((~selected.envelope["physical_pass"]).sum()),
    })
    warnings = list(inputs.warnings)
    if independent_status == "NOT_RUN":
        warnings.append("Independent validation was not requested; model predictions are not independently qualified.")
    if selected.fit_result.boundary_parameters:
        warnings.append("Selected-model parameters near a configured bound: " + ", ".join(selected.fit_result.boundary_parameters))
    errors = [
        {"scope": "input_integrity", "error_type": "GeneralizationContractError",
         "error_message": f"Raw input changed during model fitting: {input_path}."}
        for input_path, digest in before_hashes.items()
        if file_sha256(root / input_path) != digest
    ]
    selected_rows = pd.concat([selected.validation_rows, independent], ignore_index=True)
    for row in selected_rows.loc[selected_rows["validation_status"].eq("FAIL")].itertuples(index=False):
        errors.append({
            "scope": row.validation_scope, "case_id": row.holdout_case_ids,
            "error_type": "ModelValidationError",
            "error_message": (
                f"{row.split_type} validation failed for {row.holdout_case_ids}; "
                f"worst normalized error={row.worst_normalized_gate_ratio:.6g}."
            ),
        })
    if not selected.envelope_pass:
        errors.append({
            "scope": "envelope_scan", "error_type": "ModelValidationError",
            "error_message": "Model failed the physical scan of its declared geometry envelope.",
        })
    prediction_ready = bool(
        selected.validation_pass and selected.envelope_pass
        and independent_status != "FAIL" and not errors
    )
    qualification_ready = prediction_ready and independent_status == "PASS"
    validation_status = "PASS" if selected.validation_pass else "FAIL"
    validation_scope = "training_grouped" if settings["cross_validation"] == "grouped" else "training_fit"
    status = "FAIL" if not prediction_ready else ("WARNING" if warnings else "PASS")
    qualification = {
        "validation_status": validation_status,
        "validation_scope": validation_scope,
        "envelope_status": "PASS" if selected.envelope_pass else "FAIL",
        "independent_validation_policy": settings["independent_validation"],
        "independent_validation_status": independent_status,
        "prediction_ready": prediction_ready,
        "qualification_ready": qualification_ready,
    }
    model_payload = {
        "schema_version": 1, "formal": True,
        "workflow_class": "global_geometry_calibration",
        "model": {"type": "geometry_aware_enhanced", "family": selected.family,
                  "parameters": selected.model.to_mapping()},
        "optimization": asdict(selected.fit_result),
        "selection": {"reason": reason, "interaction_improvement_fraction": improvement,
                      "max_group_worsening_fraction": max_worsening},
        "qualification": {**qualification, "limits": asdict(validation_limits_from_config(inputs.calibration_config))},
        "provenance": {
            "run_id": run_id,
            "case_manifest": project_relative_path(case_manifest_path, root),
            "case_manifest_sha256": file_sha256(case_manifest_path),
            "characterization_manifest": project_relative_path(inputs.characterization_manifest_path, root),
            "characterization_manifest_sha256": file_sha256(inputs.characterization_manifest_path),
        },
    }
    validation.to_csv(paths.validation_summary, index=False)
    paths.selected_model.write_text(yaml.safe_dump(model_payload, sort_keys=False), encoding="utf-8")
    output_files = [paths.validation_summary, paths.selected_model]
    manifest = {
        "workflow": "model_generalization", "formal": True,
        "workflow_class": "global_geometry_calibration", "run_id": run_id,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(), "status": status,
        **qualification,
        "selected_family": selected.family, "selection_reason": reason,
        "software_version": software_version(), "git_commit": git_commit(root),
        "counts": counts, "warnings": warnings, "errors": errors,
        "config_files": [
            {"path": project_relative_path(path, root), "sha256": file_sha256(path)}
            for path in (config_path, case_manifest_path)
        ],
        "characterization_manifest": {
            "path": project_relative_path(inputs.characterization_manifest_path, root),
            "sha256": file_sha256(inputs.characterization_manifest_path),
        },
        "input_files": [
            {"path": str(row.input_path), "sha256": str(row.sha256)}
            for row in inputs.inventory.itertuples(index=False)
        ] + independent_input_files,
        "output_files": [
            {"path": project_relative_path(path, root), "sha256": file_sha256(path)}
            for path in output_files
        ],
        "interpretation_limit": (
            f"Fit uses {counts['cases']} cases and {counts['formal_curves']} formal curves. "
            f"Validation scope: {validation_scope}; independent validation: {independent_status}. "
            "The envelope scan checks model behavior, not measured coverage."
        ),
    }
    paths.workflow_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return GeneralizationResult(
        run_id=run_id, status=status, validation_status=validation_status,
        prediction_ready=prediction_ready, qualification_ready=qualification_ready,
        selected_family=selected.family, paths=paths, counts=counts,
        warnings=tuple(warnings), errors=tuple(errors),
    )
