from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from mosfet_platform.model.geometry_aware import GeometryAwareModelParameters


BASE_PARAMETER_NAMES = (
    "vth_ref_V", "mu_ref_m2_per_Vs", "subthreshold_n_ref", "i0_A",
    "dibl_ref_V_per_V", "lambda_ref_1_per_V", "vth_length_slope_V",
    "vth_tox_slope_V", "log_mu_length_slope", "log_mu_tox_slope",
    "log_n_minus_one_length_slope", "log_n_minus_one_tox_slope",
    "log_dibl_length_slope", "log_dibl_tox_slope",
    "log_lambda_length_slope", "log_lambda_tox_slope",
)
INTERACTION_PARAMETER_NAMES = (
    "vth_interaction_slope_V", "log_mu_interaction_slope",
    "log_n_minus_one_interaction_slope", "log_dibl_interaction_slope",
    "log_lambda_interaction_slope",
)
LOG_PARAMETER_NAMES = {
    "mu_ref_m2_per_Vs", "i0_A", "dibl_ref_V_per_V", "lambda_ref_1_per_V",
}


class GeometryCalibrationError(RuntimeError):
    """Raised when the deterministic geometry-calibration optimizer cannot produce a model."""


@dataclass(frozen=True)
class CurveData:
    """One active COMSOL curve with its characterization provenance."""

    case_id: str
    curve_type: str
    fixed_bias_V: float
    width_m: float
    length_m: float
    tox_m: float
    temperature_K: float
    input_path: str
    input_sha256: str
    sweep_values: np.ndarray
    reference_current_A: np.ndarray
    current_status: str = "signed_consistent"
    numerical_zero_clipped_count: int = 0

    def __post_init__(self) -> None:
        if self.curve_type not in {"idvg", "idvd"}:
            raise ValueError("curve_type must be 'idvg' or 'idvd'.")
        if not self.case_id or not self.input_path or not self.input_sha256:
            raise ValueError("Curve provenance fields must not be empty.")
        if min(
            self.width_m,
            self.length_m,
            self.tox_m,
            self.temperature_K,
        ) <= 0.0:
            raise ValueError("Curve geometry and temperature must be positive.")
        sweep = np.asarray(self.sweep_values, dtype=float)
        current = np.asarray(self.reference_current_A, dtype=float)
        if sweep.ndim != 1 or current.ndim != 1 or sweep.shape != current.shape:
            raise ValueError("Curve sweep and current must be equal-length 1D arrays.")
        if sweep.size < 3 or np.any(np.diff(sweep) <= 0.0):
            raise ValueError("Curve sweep must contain at least three increasing points.")
        if (
            not np.isfinite(sweep).all()
            or not np.isfinite(current).all()
            or np.any(current < 0.0)
        ):
            raise ValueError("Curve values must be finite with non-negative current.")
        if self.numerical_zero_clipped_count < 0:
            raise ValueError("numerical_zero_clipped_count must be non-negative.")

    @property
    def curve_key(self) -> str:
        return (
            f"{self.case_id}|{self.curve_type}|"
            f"{float(self.fixed_bias_V):.12g}|{self.input_path}"
        )

    @property
    def point_count(self) -> int:
        return int(np.asarray(self.sweep_values).size)


@dataclass(frozen=True)
class OptimizationSettings:
    idvg_log_weight: float = 1.0
    idvg_linear_weight: float = 1.0
    idvd_linear_weight: float = 1.0
    regularization_weight: float = 1.0e-3
    interaction_regularization_weight: float = 1.0e-3
    max_nfev: int = 5000
    vth_ref_bounds: tuple[float, float] = (-0.5, 1.35)
    mu_ref_bounds: tuple[float, float] = (1.0e-7, 10.0)
    subthreshold_n_ref_bounds: tuple[float, float] = (1.000001, 5.0)
    i0_bounds: tuple[float, float] = (1.0e-20, 1.0e-2)
    dibl_ref_bounds: tuple[float, float] = (1.0e-6, 1.0)
    lambda_ref_bounds: tuple[float, float] = (1.0e-6, 5.0)
    vth_slope_bounds: tuple[float, float] = (-1.0, 1.0)
    log_slope_bounds: tuple[float, float] = (-5.0, 5.0)

    def __post_init__(self) -> None:
        weights = (
            self.idvg_log_weight,
            self.idvg_linear_weight,
            self.idvd_linear_weight,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in weights):
            raise ValueError("Geometry-calibration data weights must be finite and positive.")
        for name, value in (
            ("regularization_weight", self.regularization_weight),
            (
                "interaction_regularization_weight",
                self.interaction_regularization_weight,
            ),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.max_nfev < 1:
            raise ValueError("max_nfev must be positive.")
        for bounds in (
            self.vth_ref_bounds,
            self.mu_ref_bounds,
            self.subthreshold_n_ref_bounds,
            self.i0_bounds,
            self.dibl_ref_bounds,
            self.lambda_ref_bounds,
            self.vth_slope_bounds,
            self.log_slope_bounds,
        ):
            if (
                len(bounds) != 2
                or not np.isfinite(bounds).all()
                or float(bounds[1]) <= float(bounds[0])
            ):
                raise ValueError("Every geometry-calibration parameter bound must be increasing.")


@dataclass(frozen=True)
class GeometryCalibrationResult:
    converged: bool
    iteration_count: int
    objective_before: float
    objective_after: float
    message: str
    case_count: int
    curve_count: int
    point_count: int
    boundary_parameters: tuple[str, ...]
    method: str = "global_geometry_aware_joint_least_squares"


def _parameter_names(model: GeometryAwareModelParameters) -> tuple[str, ...]:
    if model.geometry_family == "interaction":
        return BASE_PARAMETER_NAMES + INTERACTION_PARAMETER_NAMES
    return BASE_PARAMETER_NAMES


def _encode_parameter(name: str, value: float) -> float:
    if name == "subthreshold_n_ref":
        return float(np.log(max(value - 1.0, 1.0e-6)))
    if name in LOG_PARAMETER_NAMES:
        return float(np.log(value))
    return float(value)


def _decode_parameter(name: str, value: float) -> float:
    if name == "subthreshold_n_ref":
        return float(1.0 + np.exp(value))
    if name in LOG_PARAMETER_NAMES:
        return float(np.exp(value))
    return float(value)


def _model_to_vector(model: GeometryAwareModelParameters) -> np.ndarray:
    return np.asarray(
        [
            _encode_parameter(name, float(getattr(model, name)))
            for name in _parameter_names(model)
        ],
        dtype=float,
    )


def _vector_to_model(
    vector: np.ndarray,
    template: GeometryAwareModelParameters,
) -> GeometryAwareModelParameters:
    values = np.asarray(vector, dtype=float)
    names = _parameter_names(template)
    expected_size = len(names)
    if values.shape != (expected_size,) or not np.isfinite(values).all():
        raise ValueError(
            f"Geometry-calibration optimizer vector must contain {expected_size} finite values."
        )
    updates = {
        name: _decode_parameter(name, float(value))
        for name, value in zip(names, values)
    }
    if template.geometry_family == "additive":
        updates.update({name: 0.0 for name in INTERACTION_PARAMETER_NAMES})
    return replace(template, **updates)


def _vector_bounds(
    settings: OptimizationSettings,
    geometry_family: str = "additive",
) -> tuple[np.ndarray, np.ndarray]:
    lower_values = [
        settings.vth_ref_bounds[0],
        np.log(settings.mu_ref_bounds[0]),
        np.log(settings.subthreshold_n_ref_bounds[0] - 1.0),
        np.log(settings.i0_bounds[0]),
        np.log(settings.dibl_ref_bounds[0]),
        np.log(settings.lambda_ref_bounds[0]),
        settings.vth_slope_bounds[0],
        settings.vth_slope_bounds[0],
        *([settings.log_slope_bounds[0]] * 8),
    ]
    upper_values = [
        settings.vth_ref_bounds[1],
        np.log(settings.mu_ref_bounds[1]),
        np.log(settings.subthreshold_n_ref_bounds[1] - 1.0),
        np.log(settings.i0_bounds[1]),
        np.log(settings.dibl_ref_bounds[1]),
        np.log(settings.lambda_ref_bounds[1]),
        settings.vth_slope_bounds[1],
        settings.vth_slope_bounds[1],
        *([settings.log_slope_bounds[1]] * 8),
    ]
    if geometry_family == "interaction":
        lower_values.extend(
            [settings.vth_slope_bounds[0], *([settings.log_slope_bounds[0]] * 4)]
        )
        upper_values.extend(
            [settings.vth_slope_bounds[1], *([settings.log_slope_bounds[1]] * 4)]
        )
    elif geometry_family != "additive":
        raise ValueError("geometry_family must be additive or interaction.")
    return np.asarray(lower_values, dtype=float), np.asarray(upper_values, dtype=float)
def _predict_curve(
    model: GeometryAwareModelParameters,
    curve: CurveData,
) -> np.ndarray:
    sweep = np.asarray(curve.sweep_values, dtype=float)
    if curve.curve_type == "idvg":
        predicted = model.ids(
            vgs=sweep,
            vds=float(curve.fixed_bias_V),
            length_m=curve.length_m,
            tox_m=curve.tox_m,
            device_width_m=curve.width_m,
        )
    else:
        predicted = model.ids(
            vgs=float(curve.fixed_bias_V),
            vds=sweep,
            length_m=curve.length_m,
            tox_m=curve.tox_m,
            device_width_m=curve.width_m,
        )
    return np.asarray(predicted, dtype=float)


def _curve_scales(curve: CurveData) -> tuple[float, float]:
    reference = np.asarray(curve.reference_current_A, dtype=float)
    positive = reference[reference > 0.0]
    floor = max(
        float(np.min(positive)) * 0.1 if positive.size else 1.0e-30,
        1.0e-30,
    )
    linear_scale = max(float(np.percentile(reference, 95.0)), floor)
    return floor, linear_scale


def _residual_vector(
    vector: np.ndarray,
    *,
    template: GeometryAwareModelParameters,
    curves: tuple[CurveData, ...],
    settings: OptimizationSettings,
) -> np.ndarray:
    model = _vector_to_model(vector, template)
    case_ids = {curve.case_id for curve in curves}
    case_count = len(case_ids)
    counts = {
        (case_id, curve_type): sum(
            curve.case_id == case_id and curve.curve_type == curve_type
            for curve in curves
        )
        for case_id in case_ids
        for curve_type in ("idvg", "idvd")
    }
    residuals: list[np.ndarray] = []
    for curve in curves:
        predicted = _predict_curve(model, curve)
        reference = np.asarray(curve.reference_current_A, dtype=float)
        floor, linear_scale = _curve_scales(curve)
        curve_count = counts[(curve.case_id, curve.curve_type)]
        base = 1.0 / np.sqrt(
            max(case_count, 1) * max(curve_count, 1) * curve.point_count
        )
        if curve.curve_type == "idvg":
            log_error = (
                np.log10(np.maximum(predicted, floor))
                - np.log10(np.maximum(reference, floor))
            )
            linear_error = (predicted - reference) / linear_scale
            residuals.append(
                log_error * base * np.sqrt(settings.idvg_log_weight)
            )
            residuals.append(
                linear_error * base * np.sqrt(settings.idvg_linear_weight)
            )
        else:
            linear_error = (predicted - reference) / linear_scale
            residuals.append(
                linear_error * base * np.sqrt(settings.idvd_linear_weight)
            )
    if settings.regularization_weight > 0.0:
        residuals.append(
            np.asarray(vector[6 : len(BASE_PARAMETER_NAMES)], dtype=float)
            * np.sqrt(settings.regularization_weight)
        )
    if (
        template.geometry_family == "interaction"
        and settings.interaction_regularization_weight > 0.0
    ):
        residuals.append(
            np.asarray(vector[len(BASE_PARAMETER_NAMES) :], dtype=float)
            * np.sqrt(settings.interaction_regularization_weight)
        )
    return np.concatenate(residuals)


def _boundary_parameters(
    model: GeometryAwareModelParameters,
    settings: OptimizationSettings,
) -> tuple[str, ...]:
    vector = _model_to_vector(model)
    lower, upper = _vector_bounds(settings, model.geometry_family)
    names: list[str] = []
    for index, name in enumerate(_parameter_names(model)):
        tolerance = 0.01 * float(upper[index] - lower[index])
        if (
            float(vector[index] - lower[index]) <= tolerance
            or float(upper[index] - vector[index]) <= tolerance
        ):
            names.append(name)
    return tuple(names)


def fit_geometry_aware_model(
    curves: Iterable[CurveData],
    initial_model: GeometryAwareModelParameters,
    *,
    settings: OptimizationSettings | None = None,
) -> tuple[GeometryAwareModelParameters, GeometryCalibrationResult]:
    """Jointly fit one geometry-aware model to all supplied curves."""

    settings = settings or OptimizationSettings()
    curve_tuple = tuple(curves)
    if not curve_tuple:
        raise ValueError("Geometry calibration requires at least one curve.")
    case_ids = {curve.case_id for curve in curve_tuple}
    curve_types = {curve.curve_type for curve in curve_tuple}
    if curve_types != {"idvg", "idvd"}:
        raise ValueError("Geometry calibration requires Id-Vg and Id-Vd curves.")
    for curve in curve_tuple:
        if not np.isclose(
            curve.width_m,
            initial_model.width_m,
            rtol=1.0e-9,
            atol=0.0,
        ) or not np.isclose(
            curve.temperature_K,
            initial_model.temperature_K,
            rtol=1.0e-9,
            atol=0.0,
        ):
            raise ValueError(
                "All curves must match the model width and temperature context."
            )

    lower, upper = _vector_bounds(settings, initial_model.geometry_family)
    initial = np.clip(
        _model_to_vector(initial_model),
        lower + 1.0e-12,
        upper - 1.0e-12,
    )
    kwargs = {
        "template": initial_model,
        "curves": curve_tuple,
        "settings": settings,
    }
    before = _residual_vector(initial, **kwargs)
    objective_before = float(np.mean(np.square(before)))
    try:
        optimization = least_squares(
            _residual_vector,
            initial,
            bounds=(lower, upper),
            kwargs=kwargs,
            max_nfev=settings.max_nfev,
            x_scale="jac",
        )
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        raise GeometryCalibrationError(
            "Global geometry-aware optimizer failed numerically."
        ) from error
    if not bool(optimization.success):
        raise GeometryCalibrationError(
            f"Global geometry-aware optimizer failed: {optimization.message}"
        )
    optimized = np.asarray(optimization.x, dtype=float)
    if not np.isfinite(optimized).all():
        raise GeometryCalibrationError(
            "Global geometry-aware optimizer returned non-finite parameters."
        )
    fitted = _vector_to_model(optimized, initial_model)
    after = _residual_vector(optimized, **kwargs)
    objective_after = float(np.mean(np.square(after)))
    if not np.isfinite(objective_after) or objective_after >= objective_before:
        raise GeometryCalibrationError(
            "Global geometry-aware optimizer did not improve the objective."
        )
    predictions = prediction_frame(fitted, curve_tuple)
    if (
        not np.isfinite(predictions["predicted_current_A"]).all()
        or (predictions["predicted_current_A"] < 0.0).any()
    ):
        raise GeometryCalibrationError(
            "Global geometry-aware optimizer produced invalid predictions."
        )
    result = GeometryCalibrationResult(
        converged=True,
        iteration_count=int(optimization.nfev),
        objective_before=objective_before,
        objective_after=objective_after,
        message=str(optimization.message),
        case_count=len(case_ids),
        curve_count=len(curve_tuple),
        point_count=sum(curve.point_count for curve in curve_tuple),
        boundary_parameters=_boundary_parameters(fitted, settings),
        method=f"global_geometry_aware_{fitted.geometry_family}_joint_least_squares",
    )
    return fitted, result


def prediction_frame(
    model: GeometryAwareModelParameters,
    curves: Iterable[CurveData],
) -> pd.DataFrame:
    """Return one residual row per raw COMSOL point."""

    rows: list[dict[str, object]] = []
    for curve in curves:
        sweep = np.asarray(curve.sweep_values, dtype=float)
        reference = np.asarray(curve.reference_current_A, dtype=float)
        predicted = _predict_curve(model, curve)
        floor, linear_scale = _curve_scales(curve)
        for index, (bias, reference_value, predicted_value) in enumerate(
            zip(sweep, reference, predicted)
        ):
            rows.append(
                {
                    "case_id": curve.case_id,
                    "curve_key": curve.curve_key,
                    "curve_type": curve.curve_type,
                    "fixed_bias_V": float(curve.fixed_bias_V),
                    "point_index": int(index),
                    "vgs_V": (
                        float(bias)
                        if curve.curve_type == "idvg"
                        else float(curve.fixed_bias_V)
                    ),
                    "vds_V": (
                        float(curve.fixed_bias_V)
                        if curve.curve_type == "idvg"
                        else float(bias)
                    ),
                    "length_m": float(curve.length_m),
                    "oxide_thickness_m": float(curve.tox_m),
                    "reference_current_A": float(reference_value),
                    "predicted_current_A": float(predicted_value),
                    "linear_error_A": float(predicted_value - reference_value),
                    "normalized_linear_error": float(
                        (predicted_value - reference_value) / linear_scale
                    ),
                    "log_error_dec": float(
                        np.log10(max(float(predicted_value), floor))
                        - np.log10(max(float(reference_value), floor))
                    ),
                    "input_path": curve.input_path,
                    "input_sha256": curve.input_sha256,
                }
            )
    return pd.DataFrame(rows)
