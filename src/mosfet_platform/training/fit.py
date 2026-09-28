from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from mosfet_platform.training._data_contract import (
    CalibrationContractError,
    CalibrationInputs,
)
from mosfet_platform.training._optimizer import (
    GeometryCalibrationResult,
    OptimizationSettings,
    fit_geometry_aware_model,
)
from mosfet_platform.training._validation_metrics import (
    ValidationLimits,
    evaluate_model_curves,
    summarize_case_validation,
)
from mosfet_platform.model.enhanced_model import (
    EnhancedModelParameters,
    ids_nmos_enhanced,
)
from mosfet_platform.model.geometry_aware import (
    GeometryAwareModelParameters,
    GeometryEnvelope,
)
from mosfet_platform.model.temperature import thermal_voltage


@dataclass(frozen=True)
class GeometryCalibrationArtifacts:
    initial_model: GeometryAwareModelParameters
    final_model: GeometryAwareModelParameters
    final_result: GeometryCalibrationResult
    training_curve_metrics: pd.DataFrame
    training_residuals: pd.DataFrame
    qc_diagnostic_summary: pd.DataFrame
    qc_diagnostic_residuals: pd.DataFrame
    loco_case_validation: pd.DataFrame
    loco_curve_validation: pd.DataFrame
    loco_predictions: pd.DataFrame
    validation_status: str


def _pair(
    raw: Mapping[str, Any],
    name: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    value = raw.get(name, default)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"geometry_calibration.bounds.{name} must contain two values.")
    return float(value[0]), float(value[1])


def optimization_settings_from_config(
    calibration: Mapping[str, Any],
) -> OptimizationSettings:
    loss = calibration.get("loss", {})
    bounds = calibration.get("bounds", {})
    if not isinstance(loss, Mapping) or not isinstance(bounds, Mapping):
        raise ValueError("geometry_calibration loss and bounds must be YAML mappings.")
    return OptimizationSettings(
        idvg_log_weight=float(loss.get("idvg_log_weight", 1.0)),
        idvg_linear_weight=float(loss.get("idvg_linear_weight", 1.0)),
        idvd_linear_weight=float(loss.get("idvd_linear_weight", 1.0)),
        regularization_weight=float(
            loss.get("regularization_weight", 1.0e-3)
        ),
        interaction_regularization_weight=float(
            loss.get("interaction_regularization_weight", 1.0e-3)
        ),
        max_nfev=int(calibration.get("max_nfev", 5000)),
        vth_ref_bounds=_pair(bounds, "vth_ref", (-0.5, 1.35)),
        mu_ref_bounds=_pair(bounds, "mu_ref", (1.0e-7, 10.0)),
        subthreshold_n_ref_bounds=_pair(
            bounds,
            "subthreshold_n_ref",
            (1.000001, 5.0),
        ),
        i0_bounds=_pair(bounds, "i0", (1.0e-20, 1.0e-2)),
        dibl_ref_bounds=_pair(bounds, "dibl_ref", (1.0e-6, 1.0)),
        lambda_ref_bounds=_pair(bounds, "lambda_ref", (1.0e-6, 5.0)),
        vth_slope_bounds=_pair(bounds, "vth_slope", (-1.0, 1.0)),
        log_slope_bounds=_pair(bounds, "log_slope", (-5.0, 5.0)),
    )


def validation_limits_from_config(
    calibration: Mapping[str, Any],
) -> ValidationLimits:
    raw = calibration.get("validation_limits", {})
    if not isinstance(raw, Mapping):
        raise ValueError("geometry_calibration.validation_limits must be a YAML mapping.")
    return ValidationLimits.from_mapping(raw)


def _load_base_parameters(path: Path) -> EnhancedModelParameters:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Geometry calibration base model config must be a mapping.")
    device = raw.get("device")
    enhanced = raw.get("enhanced", {})
    if not isinstance(device, Mapping) or not isinstance(enhanced, Mapping):
        raise ValueError("Geometry calibration base model must contain device/enhanced mappings.")
    values = dict(device)
    values.update(dict(enhanced))
    return EnhancedModelParameters(**values)


def _surface_coefficients(
    values: np.ndarray,
    *,
    lengths_m: np.ndarray,
    tox_m: np.ndarray,
    length_ref_m: float,
    tox_ref_m: float,
    include_interaction: bool = False,
) -> tuple[float, float, float, float]:
    x_l = (np.asarray(lengths_m, dtype=float) - length_ref_m) / length_ref_m
    x_t = (np.asarray(tox_m, dtype=float) - tox_ref_m) / tox_ref_m
    columns = [np.ones_like(x_l), x_l, x_t]
    if include_interaction:
        columns.append(x_l * x_t)
    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        label = "Lg, tox, and interaction" if include_interaction else "Lg and tox"
        raise CalibrationContractError(
            f"Training cases do not identify independent {label} slopes."
        )
    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        np.asarray(values, dtype=float),
        rcond=None,
    )
    if not np.isfinite(coefficients).all():
        raise ValueError("Geometry calibration initialization surface is non-finite.")
    result = [float(value) for value in coefficients]
    if not include_interaction:
        result.append(0.0)
    return tuple(result)
def _clip(value: float, bounds: tuple[float, float]) -> float:
    return float(np.clip(float(value), float(bounds[0]), float(bounds[1])))


def build_initial_model(
    inputs: CalibrationInputs,
    *,
    training_case_ids: set[str] | None = None,
    settings: OptimizationSettings | None = None,
    model_family: str = "additive",
) -> GeometryAwareModelParameters:
    """Build deterministic metric initial values without holdout leakage."""

    settings = settings or optimization_settings_from_config(inputs.calibration_config)
    if model_family not in {"additive", "interaction"}:
        raise ValueError("model_family must be additive or interaction.")
    include_interaction = model_family == "interaction"
    cases = inputs.case_summary.copy()
    idvd = inputs.idvd_metrics.copy()
    if training_case_ids is not None:
        cases = cases[cases["case_id"].isin(training_case_ids)].copy()
        idvd = idvd[idvd["case_id"].isin(training_case_ids)].copy()
    if len(cases) < 3:
        raise CalibrationContractError(
            "At least three geometrically independent cases are required."
        )
    base = _load_base_parameters(inputs.base_config_path)
    calibration = inputs.calibration_config
    length_ref = float(calibration.get("reference_length_m", cases["length_m"].median()))
    tox_ref = float(calibration.get("reference_tox_m", cases["oxide_thickness_m"].median()))
    if not np.isfinite((length_ref, tox_ref)).all() or min(length_ref, tox_ref) <= 0.0:
        raise ValueError("Reference geometry must be finite and positive.")
    envelope_raw = calibration.get("geometry_envelope", {})
    if not isinstance(envelope_raw, Mapping):
        raise ValueError("geometry_calibration.geometry_envelope must be a mapping.")
    envelope = GeometryEnvelope(
        length_min_m=float(envelope_raw.get("length_min_m", inputs.case_summary["length_m"].min())),
        length_max_m=float(envelope_raw.get("length_max_m", inputs.case_summary["length_m"].max())),
        tox_min_m=float(envelope_raw.get("tox_min_m", inputs.case_summary["oxide_thickness_m"].min())),
        tox_max_m=float(envelope_raw.get("tox_max_m", inputs.case_summary["oxide_thickness_m"].max())),
    )
    for row in inputs.case_summary.itertuples(index=False):
        envelope.require_contains(row.length_m, row.oxide_thickness_m)
    widths = cases["width_m"].to_numpy(dtype=float)
    temperatures = cases["temperature_K"].to_numpy(dtype=float)
    if not np.allclose(widths, widths[0], rtol=1.0e-9, atol=0.0):
        raise CalibrationContractError("Geometry calibration requires one common device width.")
    if not np.allclose(
        temperatures,
        temperatures[0],
        rtol=1.0e-9,
        atol=0.0,
    ):
        raise CalibrationContractError("Geometry calibration requires one common temperature.")

    lengths = cases["length_m"].to_numpy(dtype=float)
    tox_values = cases["oxide_thickness_m"].to_numpy(dtype=float)
    dibl_measured = cases["dibl_V_per_V"].to_numpy(dtype=float)
    characterization = inputs.config.get("characterization", {})
    if not isinstance(characterization, Mapping):
        raise ValueError("characterization config must be a mapping.")
    low_vds = inputs.condition.transfer.vds_V
    vth0 = cases["vth_V"].to_numpy(dtype=float) + dibl_measured * low_vds
    vth_ref, vth_l, vth_t, vth_lt = _surface_coefficients(
        vth0,
        lengths_m=lengths,
        tox_m=tox_values,
        length_ref_m=length_ref,
        tox_ref_m=tox_ref,
        include_interaction=include_interaction,
    )
    n_values = cases["ss_mV_per_dec"].to_numpy(dtype=float) / (
        1000.0 * np.log(10.0) * thermal_voltage(float(temperatures[0]))
    )
    n_values = np.clip(
        n_values,
        settings.subthreshold_n_ref_bounds[0],
        settings.subthreshold_n_ref_bounds[1],
    )
    n_intercept_log, n_l, n_t, n_lt = _surface_coefficients(
        np.log(np.maximum(n_values - 1.0, 1.0e-6)),
        lengths_m=lengths,
        tox_m=tox_values,
        length_ref_m=length_ref,
        tox_ref_m=tox_ref,
        include_interaction=include_interaction,
    )
    dibl_coeff = np.maximum(dibl_measured * lengths / length_ref, 1.0e-12)
    dibl_log, dibl_l, dibl_t, dibl_lt = _surface_coefficients(
        np.log(dibl_coeff),
        lengths_m=lengths,
        tox_m=tox_values,
        length_ref_m=length_ref,
        tox_ref_m=tox_ref,
        include_interaction=include_interaction,
    )

    summary_vgs = float(
        characterization.get("summary_vgs_V", inputs.condition.ion.vgs_V)
    )
    lambda_rows: list[float] = []
    for case_id in cases["case_id"]:
        selected = idvd[
            (idvd["case_id"] == case_id)
            & np.isclose(
                idvd["vgs_V"].astype(float),
                summary_vgs,
                rtol=0.0,
                atol=1.0e-12,
            )
        ]
        if len(selected) != 1:
            raise CalibrationContractError(
                f"Case {case_id} lacks one summary-Vgs lambda metric."
            )
        value = float(selected.iloc[0]["lambda_estimate_1_V"])
        lambda_rows.append(max(value, settings.lambda_ref_bounds[0]))
    lambda_log, lambda_l, lambda_t, lambda_lt = _surface_coefficients(
        np.log(np.asarray(lambda_rows, dtype=float)),
        lengths_m=lengths,
        tox_m=tox_values,
        length_ref_m=length_ref,
        tox_ref_m=tox_ref,
        include_interaction=include_interaction,
    )

    staged_mu: list[float] = []
    for index, row in cases.reset_index(drop=True).iterrows():
        trial = EnhancedModelParameters(
            w=float(row["width_m"]),
            l=float(row["length_m"]),
            tox=float(row["oxide_thickness_m"]),
            mu=float(base.mu),
            vth=float(vth0[index]),
            lambda_clm=float(lambda_rows[index]),
            subthreshold_n=float(n_values[index]),
            temperature=float(row["temperature_K"]),
            i0=float(base.i0),
            dibl_coeff=float(dibl_coeff[index]),
            l_ref=length_ref,
            theta_mobility=float(base.theta_mobility),
        )
        predicted_ion = float(
            ids_nmos_enhanced(
                inputs.condition.ion.vgs_V,
                inputs.condition.ion.vds_V,
                trial,
            )
        )
        scale = float(row["ion_A"]) / max(predicted_ion, 1.0e-30)
        staged_mu.append(_clip(base.mu * scale, settings.mu_ref_bounds))
    mu_log, mu_l, mu_t, mu_lt = _surface_coefficients(
        np.log(np.asarray(staged_mu, dtype=float)),
        lengths_m=lengths,
        tox_m=tox_values,
        length_ref_m=length_ref,
        tox_ref_m=tox_ref,
        include_interaction=include_interaction,
    )

    return GeometryAwareModelParameters(
        width_m=float(widths[0]),
        temperature_K=float(temperatures[0]),
        length_ref_m=length_ref,
        tox_ref_m=tox_ref,
        theta_mobility=float(base.theta_mobility),
        vth_ref_V=_clip(vth_ref, settings.vth_ref_bounds),
        mu_ref_m2_per_Vs=_clip(np.exp(mu_log), settings.mu_ref_bounds),
        subthreshold_n_ref=_clip(
            1.0 + np.exp(n_intercept_log),
            settings.subthreshold_n_ref_bounds,
        ),
        i0_A=_clip(base.i0, settings.i0_bounds),
        dibl_ref_V_per_V=_clip(
            np.exp(dibl_log),
            settings.dibl_ref_bounds,
        ),
        lambda_ref_1_per_V=_clip(
            np.exp(lambda_log),
            settings.lambda_ref_bounds,
        ),
        vth_length_slope_V=_clip(vth_l, settings.vth_slope_bounds),
        vth_tox_slope_V=_clip(vth_t, settings.vth_slope_bounds),
        log_mu_length_slope=_clip(mu_l, settings.log_slope_bounds),
        log_mu_tox_slope=_clip(mu_t, settings.log_slope_bounds),
        log_n_minus_one_length_slope=_clip(
            n_l,
            settings.log_slope_bounds,
        ),
        log_n_minus_one_tox_slope=_clip(
            n_t,
            settings.log_slope_bounds,
        ),
        log_dibl_length_slope=_clip(
            dibl_l,
            settings.log_slope_bounds,
        ),
        log_dibl_tox_slope=_clip(
            dibl_t,
            settings.log_slope_bounds,
        ),
        log_lambda_length_slope=_clip(
            lambda_l,
            settings.log_slope_bounds,
        ),
        log_lambda_tox_slope=_clip(
            lambda_t,
            settings.log_slope_bounds,
        ),
        envelope=envelope,
        geometry_family=model_family,
        vth_interaction_slope_V=_clip(vth_lt, settings.vth_slope_bounds),
        log_mu_interaction_slope=_clip(mu_lt, settings.log_slope_bounds),
        log_n_minus_one_interaction_slope=_clip(n_lt, settings.log_slope_bounds),
        log_dibl_interaction_slope=_clip(dibl_lt, settings.log_slope_bounds),
        log_lambda_interaction_slope=_clip(lambda_lt, settings.log_slope_bounds),
    )


def calibrate_and_validate(
    inputs: CalibrationInputs,
) -> GeometryCalibrationArtifacts:
    """Hold out each registered case, then fit the final model."""

    settings = optimization_settings_from_config(inputs.calibration_config)
    limits = validation_limits_from_config(inputs.calibration_config)
    case_ids = sorted(str(value) for value in inputs.case_summary["case_id"])

    loco_case_frames: list[pd.DataFrame] = []
    loco_curve_frames: list[pd.DataFrame] = []
    loco_prediction_frames: list[pd.DataFrame] = []
    for fold, holdout_case_id in enumerate(case_ids):
        training_case_ids = set(case_ids) - {holdout_case_id}
        training_curves = tuple(
            curve
            for curve in inputs.curves
            if curve.case_id in training_case_ids
        )
        holdout_curves = tuple(
            curve
            for curve in inputs.curves
            if curve.case_id == holdout_case_id
        )
        initial = build_initial_model(
            inputs,
            training_case_ids=training_case_ids,
            settings=settings,
        )
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
        if len(case_metrics) != 1:
            raise RuntimeError("Each LOCO fold must produce one Case row.")
        common = {
            "fold": int(fold),
            "holdout_case_id": holdout_case_id,
            "training_case_ids": ";".join(sorted(training_case_ids)),
            "training_case_count": len(training_case_ids),
            "training_curve_count": len(training_curves),
            "holdout_curve_count": len(holdout_curves),
            "optimizer_method": fit_result.method,
            "optimizer_iterations": fit_result.iteration_count,
            "objective_before": fit_result.objective_before,
            "objective_after": fit_result.objective_after,
            "boundary_parameters": ";".join(fit_result.boundary_parameters),
        }
        for key, value in common.items():
            case_metrics[key] = value
            curve_metrics[key] = value
            predictions[key] = value
        loco_case_frames.append(case_metrics)
        loco_curve_frames.append(curve_metrics)
        loco_prediction_frames.append(predictions)

    loco_cases = pd.concat(loco_case_frames, ignore_index=True)
    loco_curves = pd.concat(loco_curve_frames, ignore_index=True)
    loco_predictions = pd.concat(loco_prediction_frames, ignore_index=True)
    if len(loco_cases) != len(case_ids) or len(loco_curves) != len(inputs.curves):
        raise RuntimeError("Geometry calibration LOCO coverage contract is incomplete.")
    if set(loco_cases["holdout_case_id"]) != set(case_ids):
        raise RuntimeError("Geometry calibration LOCO did not hold out every Case once.")

    initial_model = build_initial_model(inputs, settings=settings)
    final_model, final_result = fit_geometry_aware_model(
        inputs.curves,
        initial_model,
        settings=settings,
    )
    training_curves, training_points = evaluate_model_curves(
        final_model,
        inputs.curves,
        condition=inputs.condition,
    )
    qc_summary = pd.DataFrame()
    qc_points = pd.DataFrame()
    if inputs.diagnostic_curves:
        qc_summary, qc_points = evaluate_model_curves(
            final_model,
            inputs.diagnostic_curves,
            condition=inputs.condition,
        )
        diagnostic_inventory = inputs.inventory[
            inputs.inventory["analysis_role"] == "diagnostic_only"
        ][["input_path", "qc_status", "analysis_role", "qc_reason"]]
        qc_summary = qc_summary.merge(
            diagnostic_inventory,
            on="input_path",
            how="left",
            validate="one_to_one",
        )
        qc_points = qc_points.merge(
            diagnostic_inventory,
            on="input_path",
            how="left",
            validate="many_to_one",
        )
        mean_log_error = (
            qc_points.groupby("curve_key", as_index=False)["log_error_dec"]
            .mean()
            .rename(columns={"log_error_dec": "mean_log_error_dec"})
        )
        qc_summary = qc_summary.merge(
            mean_log_error,
            on="curve_key",
            how="left",
            validate="one_to_one",
        )
        qc_summary["bias_direction"] = np.where(
            qc_summary["mean_log_error_dec"] > 0.0,
            "overpredict",
            np.where(
                qc_summary["mean_log_error_dec"] < 0.0,
                "underpredict",
                "balanced",
            ),
        )
    validation_status = (
        "PASS"
        if (loco_cases["validation_status"] == "PASS").all()
        else "FAIL"
    )
    return GeometryCalibrationArtifacts(
        initial_model=initial_model,
        final_model=final_model,
        final_result=final_result,
        training_curve_metrics=training_curves,
        training_residuals=training_points,
        qc_diagnostic_summary=qc_summary,
        qc_diagnostic_residuals=qc_points,
        loco_case_validation=loco_cases,
        loco_curve_validation=loco_curves,
        loco_predictions=loco_predictions,
        validation_status=validation_status,
    )
