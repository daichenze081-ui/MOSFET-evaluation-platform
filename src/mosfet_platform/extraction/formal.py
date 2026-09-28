from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

import numpy as np

from mosfet_platform.extraction.metrics import (
    extract_ion_ioff_ratio,
    extract_ion_sample_at_bias,
    extract_transfer_metrics,
    extract_vth_constant_current_robust,
    require_ion_consistency,
)
from mosfet_platform.measurement import MeasurementContract


@dataclass(frozen=True)
class CurveTrace:
    curve_type: Literal["idvg", "idvd"]
    fixed_bias_V: float
    sweep: np.ndarray
    current_A: np.ndarray

    def __post_init__(self) -> None:
        sweep = np.asarray(self.sweep, dtype=float)
        current = np.asarray(self.current_A, dtype=float)
        if sweep.ndim != 1 or current.ndim != 1 or sweep.shape != current.shape:
            raise ValueError("Curve sweep and current must be equal-length 1D arrays.")
        if sweep.size < 2:
            raise ValueError("Curve requires at least two points.")
        if not np.isfinite(sweep).all() or not np.isfinite(current).all():
            raise ValueError("Curve values must be finite.")
        if np.any(current < 0.0) or np.any(np.diff(sweep) <= 0.0):
            raise ValueError("Curve current must be non-negative and sweep increasing.")
        object.__setattr__(self, "sweep", sweep)
        object.__setattr__(self, "current_A", current)


@dataclass(frozen=True)
class DeviceCurveBundle:
    device_id: str
    condition_id: str
    width_m: float
    length_m: float
    oxide_thickness_m: float
    transfer: CurveTrace
    ion: CurveTrace
    verification: CurveTrace
    source: Literal["manifest_validated", "generated"]

    @property
    def curves(self) -> tuple[CurveTrace, CurveTrace, CurveTrace]:
        return self.transfer, self.ion, self.verification


@dataclass(frozen=True)
class CurveIssue:
    code: str
    message: str
    recoverable: bool


@dataclass(frozen=True)
class BundleBuildResult:
    bundle: DeviceCurveBundle | None
    issues: tuple[CurveIssue, ...] = ()
    diagnostic_curves: tuple[CurveTrace, ...] = ()


def curve_rows(
    *,
    device_id: str,
    condition_id: str,
    geometry: tuple[float, float, float],
    curves: Iterable[CurveTrace],
    meta: Mapping[str, Any],
) -> list[dict[str, Any]]:
    width, length, tox = geometry
    rows: list[dict[str, Any]] = []
    for curve in curves:
        for sweep, current in zip(curve.sweep, curve.current_A):
            rows.append(
                {
                    "device_id": device_id,
                    "condition_id": condition_id,
                    "curve_type": curve.curve_type,
                    "vgs_V": float(sweep if curve.curve_type == "idvg" else curve.fixed_bias_V),
                    "vds_V": float(curve.fixed_bias_V if curve.curve_type == "idvg" else sweep),
                    "id_A": float(current),
                    "width_m": float(width),
                    "length_m": float(length),
                    "oxide_thickness_m": float(tox),
                    **meta,
                }
            )
    return rows


def _pick(
    curves: tuple[CurveTrace, ...],
    kind: str,
    bias: float,
    tol: float,
) -> CurveTrace:
    found = [
        curve
        for curve in curves
        if curve.curve_type == kind
        and np.isclose(curve.fixed_bias_V, bias, rtol=0.0, atol=tol)
    ]
    if len(found) != 1:
        state = "missing" if not found else "duplicated"
        raise ValueError(f"Formal {kind} curve at {bias:g} V is {state}.")
    return found[0]


def build_bundle(
    *,
    device_id: str,
    condition_id: str,
    geometry: tuple[float, float, float],
    curves: Iterable[CurveTrace],
    contract: MeasurementContract,
    source: Literal["manifest_validated", "generated"],
) -> DeviceCurveBundle:
    traces = tuple(curves)
    width, length, tox = (float(value) for value in geometry)
    if not device_id.strip() or condition_id != contract.condition_id:
        raise ValueError("Device identity or condition does not match the contract.")
    if not np.isfinite((width, length, tox)).all() or min(width, length, tox) <= 0.0:
        raise ValueError("Device geometry must be finite and positive.")
    tol = contract.current.bias_voltage_tolerance_V
    bundle = DeviceCurveBundle(
        device_id=device_id,
        condition_id=condition_id,
        width_m=width,
        length_m=length,
        oxide_thickness_m=tox,
        transfer=_pick(traces, "idvg", contract.transfer.vds_V, tol),
        ion=_pick(traces, "idvg", contract.ion.vds_V, tol),
        verification=_pick(traces, "idvd", contract.ion.vgs_V, tol),
        source=source,
    )
    if len(traces) != 3:
        raise ValueError("Formal bundle must contain exactly three curves.")
    return bundle


def extract_bundle_metrics(
    bundle: DeviceCurveBundle,
    contract: MeasurementContract,
) -> dict[str, float | str | bool]:
    if bundle.condition_id != contract.condition_id:
        raise ValueError("Bundle condition does not match the measurement contract.")

    metrics = extract_transfer_metrics(
        vgs=bundle.transfer.sweep,
        ids=bundle.transfer.current_A,
        vds=bundle.transfer.fixed_bias_V,
        measurement_contract=contract,
        width_m=bundle.width_m,
        length_m=bundle.length_m,
    )
    ion = extract_ion_sample_at_bias(
        sweep_values=bundle.ion.sweep,
        ids=bundle.ion.current_A,
        curve_type=bundle.ion.curve_type,
        fixed_bias_V=bundle.ion.fixed_bias_V,
        target_vgs_V=contract.ion.vgs_V,
        target_vds_V=contract.ion.vds_V,
        measurement_contract=contract,
    )
    verification = extract_ion_sample_at_bias(
        sweep_values=bundle.verification.sweep,
        ids=bundle.verification.current_A,
        curve_type=bundle.verification.curve_type,
        fixed_bias_V=bundle.verification.fixed_bias_V,
        target_vgs_V=contract.ion.vgs_V,
        target_vds_V=contract.ion.vds_V,
        measurement_contract=contract,
    )
    consistency = require_ion_consistency(
        ion.current_A,
        verification.current_A,
        measurement_contract=contract,
    )
    same_vds = extract_ion_sample_at_bias(
        sweep_values=bundle.transfer.sweep,
        ids=bundle.transfer.current_A,
        curve_type=bundle.transfer.curve_type,
        fixed_bias_V=bundle.transfer.fixed_bias_V,
        target_vgs_V=contract.transfer.vgs_on_V,
        target_vds_V=contract.transfer.vds_V,
        measurement_contract=contract,
    )
    ioff = float(metrics["ioff"])
    same_bias_ratio = extract_ion_ioff_ratio(same_vds.current_A, ioff)
    metrics.update(
        {
            "device_id": bundle.device_id,
            "ion": float(ion.current_A),
            "ion_idvg_A": float(ion.current_A),
            "ion_idvd_A": float(verification.current_A),
            "ion_consistency_error": float(consistency),
            "ion_consistency_status": "ok",
            "ion_idvg_requested_vgs_V": float(ion.requested_voltage_V),
            "ion_idvg_actual_vgs_V": float(ion.actual_voltage_V),
            "ion_idvg_sampling_method": str(ion.sampling_method),
            "ion_idvg_interpolated": bool(ion.interpolated),
            "ion_idvd_requested_vds_V": float(verification.requested_voltage_V),
            "ion_idvd_actual_vds_V": float(verification.actual_voltage_V),
            "ion_idvd_sampling_method": str(verification.sampling_method),
            "ion_idvd_interpolated": bool(verification.interpolated),
            "same_vds_on_current_A": float(same_vds.current_A),
            "same_vds_requested_vgs_V": float(same_vds.requested_voltage_V),
            "same_vds_actual_vgs_V": float(same_vds.actual_voltage_V),
            "same_vds_sampling_method": str(same_vds.sampling_method),
            "same_vds_interpolated": bool(same_vds.interpolated),
            "ion_ioff_cross_bias": extract_ion_ioff_ratio(ion.current_A, ioff),
            "ion_ioff": same_bias_ratio,
            "ion_ioff_same_vds_0p1": same_bias_ratio,
            "ion_vgs_V": float(contract.ion.vgs_V),
            "ion_vds_V": float(contract.ion.vds_V),
            "same_vds_ratio_vgs_V": float(contract.transfer.vgs_on_V),
            "same_vds_ratio_vds_V": float(contract.transfer.vds_V),
            "condition_id": contract.condition_id,
            "condition_id_source": bundle.source,
            "temperature_K": float(contract.temperature_K),
            "width_m": bundle.width_m,
            "length_m": bundle.length_m,
            "data_quality_status": (
                "ok" if bool(metrics["formal_eligible"]) else "invalid"
            ),
        }
    )
    high_vth, high_status = extract_vth_constant_current_robust(
        bundle.ion.sweep,
        bundle.ion.current_A,
        target_current=contract.target_current_A(
            width_m=bundle.width_m,
            length_m=bundle.length_m,
        ),
    )
    delta_vds = contract.dibl.high_vds_V - contract.dibl.low_vds_V
    metrics["dibl_mV_per_V"] = float(
        (float(metrics["vth"]) - high_vth) / delta_vds * 1000.0
        if high_status == "ok"
        else np.nan
    )
    metrics["oxide_thickness_m"] = bundle.oxide_thickness_m
    return metrics
