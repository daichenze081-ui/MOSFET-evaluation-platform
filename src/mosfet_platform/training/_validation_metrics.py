from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from mosfet_platform.training._optimizer import (
    CurveData,
    prediction_frame,
)
from mosfet_platform.extraction.metrics import (
    extract_ion_sample_at_bias,
    extract_transfer_metrics,
    extract_vth_constant_current_robust,
)
from mosfet_platform.model.geometry_aware import GeometryAwareModelParameters
from mosfet_platform.extraction.selection import select_exact_curve_row
from mosfet_platform.measurement import MeasurementContract

@dataclass(frozen=True)
class ValidationLimits:
    idvg_log_rmse_dec_max: float = 0.30
    ion_error_percent_max: float = 20.0
    vth_error_mV_max: float = 30.0
    ss_error_mV_per_dec_max: float = 15.0
    dibl_error_mV_per_V_max: float = 15.0
    idvd_nrmse_max: float = 0.25

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "ValidationLimits":
        return cls(
            **{
                name: float(raw.get(name, default))
                for name, default in cls().__dict__.items()
            }
        )

def _idvg_metrics(
    *,
    vgs: np.ndarray,
    current_A: np.ndarray,
    vds_V: float,
    width_m: float,
    length_m: float,
    condition: MeasurementContract,
) -> dict[str, object]:
    vgs = np.asarray(vgs, dtype=float)
    current_A = np.asarray(current_A, dtype=float)
    vds_V = float(vds_V)
    target_current_A = condition.target_current_A(
        width_m=width_m,
        length_m=length_m,
    )
    vth, vth_status = extract_vth_constant_current_robust(
        vgs=vgs,
        ids=current_A,
        target_current=target_current_A,
    )
    if np.isclose(
        vds_V,
        condition.transfer.vds_V,
        rtol=0.0,
        atol=1.0e-12,
    ):
        transfer = extract_transfer_metrics(
            vgs=vgs,
            ids=current_A,
            vds=vds_V,
            measurement_contract=condition,
            width_m=width_m,
            length_m=length_m,
        )
        same_vds_on_current = extract_ion_sample_at_bias(
            sweep_values=vgs,
            ids=current_A,
            curve_type="idvg",
            fixed_bias_V=vds_V,
            target_vgs_V=condition.transfer.vgs_on_V,
            target_vds_V=condition.transfer.vds_V,
            measurement_contract=condition,
        ).current_A
        return {**transfer, "ion": same_vds_on_current}
    if np.isclose(
        vds_V,
        condition.ion.vds_V,
        rtol=0.0,
        atol=1.0e-12,
    ):
        ion = extract_ion_sample_at_bias(
            sweep_values=vgs,
            ids=current_A,
            curve_type="idvg",
            fixed_bias_V=vds_V,
            target_vgs_V=condition.ion.vgs_V,
            target_vds_V=condition.ion.vds_V,
            measurement_contract=condition,
        ).current_A
        return {
            "ion": ion,
            "vth": float(vth),
            "vth_status": str(vth_status),
            "ss_mv_dec": float("nan"),
            "ss_status": "not_applicable_high_vds",
        }
    raise ValueError(
        f"Id-Vg Vds={vds_V} V is not declared by MeasurementContract."
    )

def evaluate_model_curves(
    model: GeometryAwareModelParameters,
    curves: Iterable[CurveData],
    *,
    condition: MeasurementContract,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one model with one row per curve and one row per point."""

    curve_tuple = tuple(curves)
    points = prediction_frame(model, curve_tuple)
    rows: list[dict[str, object]] = []
    for curve in curve_tuple:
        selected = points[points["curve_key"] == curve.curve_key]
        nrmse = float(
            np.sqrt(np.mean(np.square(selected["normalized_linear_error"])))
        )
        log_rmse = float(
            np.sqrt(np.mean(np.square(selected["log_error_dec"])))
        )
        row: dict[str, object] = {
            "condition_id": condition.condition_id,
            "case_id": curve.case_id,
            "curve_key": curve.curve_key,
            "curve_type": curve.curve_type,
            "fixed_bias_V": float(curve.fixed_bias_V),
            "length_m": float(curve.length_m),
            "oxide_thickness_m": float(curve.tox_m),
            "point_count": curve.point_count,
            "linear_nrmse": nrmse,
            "log_rmse_dec": log_rmse,
            "input_path": curve.input_path,
            "input_sha256": curve.input_sha256,
        }
        if curve.curve_type == "idvg":
            vgs = np.asarray(curve.sweep_values, dtype=float)
            reference = _idvg_metrics(
                vgs=vgs,
                current_A=np.asarray(curve.reference_current_A, dtype=float),
                vds_V=curve.fixed_bias_V,
                width_m=curve.width_m,
                length_m=curve.length_m,
                condition=condition,
            )
            predicted = _idvg_metrics(
                vgs=vgs,
                current_A=selected["predicted_current_A"].to_numpy(dtype=float),
                vds_V=curve.fixed_bias_V,
                width_m=curve.width_m,
                length_m=curve.length_m,
                condition=condition,
            )
            row.update(
                {
                    "reference_ion_A": float(reference["ion"]),
                    "predicted_ion_A": float(predicted["ion"]),
                    "reference_vth_V": float(reference["vth"]),
                    "predicted_vth_V": float(predicted["vth"]),
                    "reference_ss_mV_per_dec": float(reference["ss_mv_dec"]),
                    "predicted_ss_mV_per_dec": float(predicted["ss_mv_dec"]),
                    "reference_vth_status": str(reference["vth_status"]),
                    "predicted_vth_status": str(predicted["vth_status"]),
                    "reference_ss_status": str(reference["ss_status"]),
                    "predicted_ss_status": str(predicted["ss_status"]),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows), points

def summarize_case_validation(
    curve_metrics: pd.DataFrame,
    point_predictions: pd.DataFrame,
    *,
    limits: ValidationLimits,
    measurement_contract: MeasurementContract,
) -> pd.DataFrame:
    """Apply all six gates independently to every held-out Case."""

    rows: list[dict[str, object]] = []
    for case_id in sorted(curve_metrics["case_id"].unique()):
        curves = curve_metrics[curve_metrics["case_id"] == case_id]
        points = point_predictions[point_predictions["case_id"] == case_id]
        low = select_exact_curve_row(
            curves,
            case_id=str(case_id),
            curve_type="idvg",
            fixed_bias_V=measurement_contract.dibl.low_vds_V,
            condition_id=measurement_contract.condition_id,
        )
        high = select_exact_curve_row(
            curves,
            case_id=str(case_id),
            curve_type="idvg",
            fixed_bias_V=measurement_contract.dibl.high_vds_V,
            condition_id=measurement_contract.condition_id,
        )
        idvd_curve = select_exact_curve_row(
            curves,
            case_id=str(case_id),
            curve_type="idvd",
            fixed_bias_V=measurement_contract.ion.vgs_V,
            condition_id=measurement_contract.condition_id,
        )
        idvg_points = points[points["curve_type"] == "idvg"]
        metric_eligibility_pass = all(
            str(row[field]) == "ok"
            for row in (low, high)
            for field in ("reference_vth_status", "predicted_vth_status")
        ) and all(
            str(low[field]) == "ok"
            for field in ("reference_ss_status", "predicted_ss_status")
        )
        delta_vds = float(high["fixed_bias_V"] - low["fixed_bias_V"])
        if delta_vds <= 0.0:
            raise ValueError(f"Case {case_id} has an invalid DIBL Vds pair.")
        reference_dibl = (
            float(low["reference_vth_V"]) - float(high["reference_vth_V"])
        ) / delta_vds
        predicted_dibl = (
            float(low["predicted_vth_V"]) - float(high["predicted_vth_V"])
        ) / delta_vds
        idvg_log_rmse = float(
            np.sqrt(np.mean(np.square(idvg_points["log_error_dec"])))
        )
        ion_reference = float(high["reference_ion_A"])
        ion_error_percent = (
            abs(float(high["predicted_ion_A"]) - ion_reference)
            / max(abs(ion_reference), 1.0e-30)
            * 100.0
        )
        vth_error_mV = (
            abs(float(low["predicted_vth_V"]) - float(low["reference_vth_V"]))
            * 1000.0
        )
        ss_error = abs(
            float(low["predicted_ss_mV_per_dec"])
            - float(low["reference_ss_mV_per_dec"])
        )
        dibl_error = abs(predicted_dibl - reference_dibl) * 1000.0
        idvd_nrmse = float(idvd_curve["linear_nrmse"])
        gates = {
            "metric_eligibility_pass": metric_eligibility_pass,
            "idvg_log_rmse_pass": (
                idvg_log_rmse <= limits.idvg_log_rmse_dec_max
            ),
            "ion_error_pass": (
                ion_error_percent <= limits.ion_error_percent_max
            ),
            "vth_error_pass": vth_error_mV <= limits.vth_error_mV_max,
            "ss_error_pass": ss_error <= limits.ss_error_mV_per_dec_max,
            "dibl_error_pass": (
                dibl_error <= limits.dibl_error_mV_per_V_max
            ),
            "idvd_nrmse_pass": idvd_nrmse <= limits.idvd_nrmse_max,
        }
        rows.append(
            {
                "case_id": case_id,
                "idvg_log_rmse_dec": idvg_log_rmse,
                "ion_error_percent": ion_error_percent,
                "vth_error_mV": vth_error_mV,
                "ss_error_mV_per_dec": ss_error,
                "reference_dibl_mV_per_V": reference_dibl * 1000.0,
                "predicted_dibl_mV_per_V": predicted_dibl * 1000.0,
                "dibl_error_mV_per_V": dibl_error,
                "idvd_nrmse": idvd_nrmse,
                **gates,
                "validation_status": (
                    "PASS" if all(gates.values()) else "FAIL"
                ),
            }
        )
    return pd.DataFrame(rows)
