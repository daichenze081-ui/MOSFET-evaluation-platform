from dataclasses import dataclass
from typing import Literal

import numpy as np

SamplingMethod = Literal["exact", "linear", "nearest"]

@dataclass(frozen=True)
class BiasSample:
    """Current sampled at one requested bias with auditable provenance."""
    current_A: float
    requested_voltage_V: float
    actual_voltage_V: float
    sampling_method: SamplingMethod
    interpolated: bool
    lower_voltage_V: float | None = None
    lower_current_A: float | None = None
    upper_voltage_V: float | None = None
    upper_current_A: float | None = None

def _as_1d_float_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} must not contain NaN or Inf.")

    return array

def _validate_idvg_inputs(vgs, ids) -> tuple[np.ndarray, np.ndarray]:
    vgs_array = _as_1d_float_array(vgs, "vgs")
    ids_array = _as_1d_float_array(ids, "ids")

    if vgs_array.shape != ids_array.shape:
        raise ValueError("vgs and ids must have the same shape.")

    if np.any(ids_array < 0.0):
        raise ValueError("ids must be non-negative.")

    return vgs_array, ids_array

def _validate_strictly_increasing_vgs(vgs: np.ndarray) -> None:
    if np.any(np.diff(vgs) <= 0.0):
        raise ValueError("vgs must be strictly increasing.")

def sample_current_at_bias(
    sweep_values,
    ids,
    *,
    target_voltage_V: float,
    method: SamplingMethod,
    voltage_tolerance_V: float = 0.0,
    allow_interpolation: bool = True,
) -> BiasSample:
    """Sample one current while preserving exact/interpolated bias provenance."""
    sweep_array, ids_array = _validate_idvg_inputs(sweep_values, ids)
    target = float(target_voltage_V)
    tolerance = float(voltage_tolerance_V)
    if method not in {"exact", "linear", "nearest"}:
        raise ValueError("method must be 'exact', 'linear', or 'nearest'.")
    if not np.isfinite(target):
        raise ValueError("target_voltage_V must be finite.")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("voltage_tolerance_V must be finite and non-negative.")
    if method == "exact" and allow_interpolation:
        raise ValueError("Exact current sampling cannot allow interpolation.")

    order = np.argsort(sweep_array)
    sorted_sweep = sweep_array[order]
    sorted_ids = ids_array[order]
    _validate_strictly_increasing_vgs(sorted_sweep)

    matches = np.flatnonzero(np.abs(sorted_sweep - target) <= tolerance)
    if matches.size > 1:
        raise ValueError("Multiple samples match the requested bias within tolerance.")
    if matches.size == 1:
        index = int(matches[0])
        return BiasSample(
            current_A=float(sorted_ids[index]),
            requested_voltage_V=target,
            actual_voltage_V=float(sorted_sweep[index]),
            sampling_method=method,
            interpolated=False,
        )

    lower_bound = float(sorted_sweep[0])
    upper_bound = float(sorted_sweep[-1])
    if target < lower_bound or target > upper_bound:
        raise ValueError("target_voltage_V must lie within the sweep range.")
    if method == "exact":
        raise ValueError("No exact sample matches the requested bias within tolerance.")
    if method == "nearest":
        index = int(np.argmin(np.abs(sorted_sweep - target)))
        return BiasSample(
            current_A=float(sorted_ids[index]),
            requested_voltage_V=target,
            actual_voltage_V=float(sorted_sweep[index]),
            sampling_method=method,
            interpolated=False,
        )
    if not allow_interpolation:
        raise ValueError("Interpolation is disabled by the current sampling policy.")

    upper_index = int(np.searchsorted(sorted_sweep, target, side="right"))
    lower_index = upper_index - 1
    lower_voltage = float(sorted_sweep[lower_index])
    upper_voltage = float(sorted_sweep[upper_index])
    lower_current = float(sorted_ids[lower_index])
    upper_current = float(sorted_ids[upper_index])
    fraction = (target - lower_voltage) / (upper_voltage - lower_voltage)
    current = lower_current + fraction * (upper_current - lower_current)
    return BiasSample(
        current_A=float(current),
        requested_voltage_V=target,
        actual_voltage_V=target,
        sampling_method=method,
        interpolated=True,
        lower_voltage_V=lower_voltage,
        lower_current_A=lower_current,
        upper_voltage_V=upper_voltage,
        upper_current_A=upper_current,
    )

def extract_ion_ioff_ratio(
    ion: float,
    ioff: float,
) -> float:
    ion = float(ion)
    ioff = float(ioff)

    if ion < 0.0:
        raise ValueError("ion must be non-negative.")

    if ioff <= 0.0:
        raise ValueError("ioff must be positive.")

    return ion / ioff

def extract_gm(
    vgs,
    ids,
) -> np.ndarray:
    vgs_array, ids_array = _validate_idvg_inputs(vgs, ids)
    _validate_strictly_increasing_vgs(vgs_array)

    if vgs_array.size < 2:
        raise ValueError("at least two points are required to extract gm.")

    gm = np.gradient(ids_array, vgs_array)

    if np.isnan(gm).any():
        raise ValueError("gm calculation produced NaN.")

    return gm

def extract_gm_max(
    vgs,
    ids,
) -> float:
    gm = extract_gm(vgs=vgs, ids=ids)
    return float(np.max(gm))

def extract_vth_constant_current(
    vgs,
    ids,
    target_current: float = 1e-7,
) -> float:
    vgs_array, ids_array = _validate_idvg_inputs(vgs, ids)
    _validate_strictly_increasing_vgs(vgs_array)

    target_current = float(target_current)

    if target_current <= 0.0:
        raise ValueError("target_current must be positive.")

    if np.any(ids_array <= 0.0):
        raise ValueError("ids must be positive for log-current Vth extraction.")

    ids_min = float(np.min(ids_array))
    ids_max = float(np.max(ids_array))

    if target_current < ids_min or target_current > ids_max:
        raise ValueError(
            "target_current must lie within the Id range of the curve."
        )

    exact_indices = np.where(
        np.isclose(ids_array, target_current, rtol=1e-9, atol=0.0)
    )[0]
    if exact_indices.size > 0:
        return float(vgs_array[int(exact_indices[0])])

    crossing_indices = np.where(ids_array >= target_current)[0]
    if crossing_indices.size == 0:
        raise ValueError(
            "target_current must lie within the Id range of the curve."
        )

    high_index = int(crossing_indices[0])
    if high_index == 0:
        return float(vgs_array[0])

    low_index = high_index - 1

    v_low = float(vgs_array[low_index])
    v_high = float(vgs_array[high_index])

    log_i_low = float(np.log10(ids_array[low_index]))
    log_i_high = float(np.log10(ids_array[high_index]))
    log_i_target = float(np.log10(target_current))

    if log_i_high == log_i_low:
        return v_low

    fraction = (log_i_target - log_i_low) / (log_i_high - log_i_low)

    return float(v_low + fraction * (v_high - v_low))

def extract_vth_constant_current_robust(
    vgs,
    ids,
    target_current: float = 1e-7,
) -> tuple[float, str]:
    vgs_array, ids_array = _validate_idvg_inputs(vgs, ids)
    _validate_strictly_increasing_vgs(vgs_array)
    target_current = float(target_current)
    if target_current <= 0.0:
        raise ValueError("target_current must be positive.")
    if np.any(ids_array <= 0.0):
        raise ValueError("ids must be positive for log-current Vth extraction.")

    ids_min = float(np.min(ids_array))
    ids_max = float(np.max(ids_array))
    if target_current < ids_min:
        return float(vgs_array[0]), "below_current_range"
    if target_current > ids_max:
        return float(vgs_array[-1]), "above_current_range"

    return (
        extract_vth_constant_current(
            vgs=vgs_array,
            ids=ids_array,
            target_current=target_current,
        ),
        "ok",
    )

def extract_subthreshold_swing(
    vgs,
    ids,
    current_min: float = 1e-12,
    current_max: float = 1e-7,
    minimum_points: int = 2,
) -> float:
    vgs_array, ids_array = _validate_idvg_inputs(vgs, ids)
    _validate_strictly_increasing_vgs(vgs_array)

    current_min = float(current_min)
    current_max = float(current_max)

    if current_min <= 0.0:
        raise ValueError("current_min must be positive.")

    if current_max <= 0.0:
        raise ValueError("current_max must be positive.")

    if current_max <= current_min:
        raise ValueError("current_max must be larger than current_min.")

    if np.any(ids_array <= 0.0):
        raise ValueError("ids must be positive for subthreshold swing extraction.")

    mask = (ids_array >= current_min) & (ids_array <= current_max)

    if np.count_nonzero(mask) < minimum_points:
        raise ValueError(
            f"at least two points are required; configured minimum is {minimum_points}."
        )

    vgs_fit = vgs_array[mask]
    log_ids_fit = np.log10(ids_array[mask])

    slope, _ = np.polyfit(vgs_fit, log_ids_fit, deg=1)

    if slope <= 0.0:
        raise ValueError("subthreshold log-current slope must be positive.")

    ss = 1.0 / slope

    return float(ss)

def extract_subthreshold_swing_robust(
    vgs,
    ids,
    current_min: float = 1e-12,
    current_max: float = 1e-7,
    minimum_points: int = 3,
) -> tuple[float, str]:
    try:
        return (
            extract_subthreshold_swing(
                vgs=vgs,
                ids=ids,
                current_min=current_min,
                current_max=current_max,
                minimum_points=minimum_points,
            ),
            "ok",
        )
    except ValueError:
        vgs_array, ids_array = _validate_idvg_inputs(vgs, ids)
        _validate_strictly_increasing_vgs(vgs_array)

        positive_ids = ids_array[ids_array > 0.0]
        if positive_ids.size < 2:
            return float("nan"), "insufficient_points"

        fallback_min = float(np.min(positive_ids))
        fallback_max = float(np.max(positive_ids))
        if fallback_max <= fallback_min:
            return float("nan"), "insufficient_points"

        try:
            return (
                extract_subthreshold_swing(
                    vgs=vgs_array,
                    ids=ids_array,
                    current_min=fallback_min,
                    current_max=fallback_max,
                    minimum_points=minimum_points,
                ),
                "expanded_window",
            )
        except ValueError:
            return float("nan"), "insufficient_points"

def extract_transfer_metrics(
    *,
    vgs,
    ids,
    vds: float,
    measurement_contract,
    width_m: float,
    length_m: float,
) -> dict[str, float | str]:
    """Extract only the formal low-Vds transfer-curve metrics."""

    vds = float(vds)
    expected_vds = float(measurement_contract.transfer.vds_V)
    if not np.isclose(vds, expected_vds, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "Transfer curve Vds does not match MeasurementContract: "
            f"expected {expected_vds}, got {vds}."
        )
    vgs_array, ids_array = _validate_idvg_inputs(vgs, ids)
    _validate_strictly_increasing_vgs(vgs_array)
    target_current = measurement_contract.target_current_A(
        width_m=float(width_m),
        length_m=float(length_m),
    )
    ioff_sample = sample_current_at_bias(
        vgs_array,
        ids_array,
        target_voltage_V=measurement_contract.transfer.vgs_off_V,
        method=measurement_contract.current.sampling_method,
        voltage_tolerance_V=measurement_contract.current.bias_voltage_tolerance_V,
        allow_interpolation=measurement_contract.current.allow_interpolation,
    )
    vth, vth_status = extract_vth_constant_current_robust(
        vgs_array,
        ids_array,
        target_current=target_current,
    )
    ss_v_dec, ss_status = extract_subthreshold_swing_robust(
        vgs_array,
        ids_array,
        current_min=measurement_contract.ss.current_min_A,
        current_max=measurement_contract.ss.current_max_A,
        minimum_points=measurement_contract.ss.minimum_points,
    )
    ineligible_reasons = []
    if vth_status != "ok":
        ineligible_reasons.append(f"vth_status:{vth_status}")
    if ss_status != "ok":
        ineligible_reasons.append(f"ss_status:{ss_status}")
    return {
        "ioff": float(ioff_sample.current_A),
        "gm_max": extract_gm_max(vgs_array, ids_array),
        "vth": float(vth),
        "ss_v_dec": float(ss_v_dec),
        "ss_mv_dec": float(ss_v_dec * 1000.0),
        "vth_target_current_A": float(target_current),
        "vth_status": str(vth_status),
        "vth_method": str(measurement_contract.vth.method),
        "ss_status": str(ss_status),
        "formal_eligible": not ineligible_reasons,
        "formal_ineligibility_reason": ";".join(ineligible_reasons),
        "transfer_vds_V": vds,
        "ioff_vgs_V": float(measurement_contract.transfer.vgs_off_V),
        "ioff_vds_V": vds,
        "ioff_requested_vgs_V": float(ioff_sample.requested_voltage_V),
        "ioff_actual_vgs_V": float(ioff_sample.actual_voltage_V),
        "ioff_sampling_method": str(ioff_sample.sampling_method),
        "ioff_interpolated": bool(ioff_sample.interpolated),
    }

def extract_ion_sample_at_bias(
    *,
    sweep_values,
    ids,
    curve_type: str,
    fixed_bias_V: float,
    target_vgs_V: float,
    target_vds_V: float,
    measurement_contract,
) -> BiasSample:
    """Extract a provenance-bearing current at a declared contract bias."""
    target_vgs_V = float(target_vgs_V)
    target_vds_V = float(target_vds_V)
    allowed_points = (
        (
            float(measurement_contract.ion.vgs_V),
            float(measurement_contract.ion.vds_V),
        ),
        (
            float(measurement_contract.transfer.vgs_on_V),
            float(measurement_contract.transfer.vds_V),
        ),
    )
    if not any(
        np.isclose(target_vgs_V, vgs, rtol=0.0, atol=1.0e-12)
        and np.isclose(target_vds_V, vds, rtol=0.0, atol=1.0e-12)
        for vgs, vds in allowed_points
    ):
        raise ValueError("Requested Ion bias is not declared by MeasurementContract.")

    curve_type = str(curve_type).lower()
    fixed_bias_V = float(fixed_bias_V)
    if curve_type == "idvg":
        if not np.isclose(fixed_bias_V, target_vds_V, rtol=0.0, atol=1.0e-12):
            raise ValueError("Id-Vg fixed Vds does not match requested Ion Vds.")
        target_sweep_value = target_vgs_V
    elif curve_type == "idvd":
        if not np.isclose(fixed_bias_V, target_vgs_V, rtol=0.0, atol=1.0e-12):
            raise ValueError("Id-Vd fixed Vgs does not match requested Ion Vgs.")
        target_sweep_value = target_vds_V
    else:
        raise ValueError("curve_type must be 'idvg' or 'idvd'.")
    return sample_current_at_bias(
        sweep_values=sweep_values,
        ids=ids,
        target_voltage_V=target_sweep_value,
        method=measurement_contract.current.sampling_method,
        voltage_tolerance_V=measurement_contract.current.bias_voltage_tolerance_V,
        allow_interpolation=measurement_contract.current.allow_interpolation,
    )

def ion_consistency_error(primary_ion_A: float, verification_ion_A: float, *, measurement_contract) -> float:
    """Return symmetric relative error for the two formal Ion paths."""

    primary = float(primary_ion_A)
    verification = float(verification_ion_A)
    if not np.isfinite(primary) or not np.isfinite(verification):
        raise ValueError("Ion consistency values must be finite.")
    if primary < 0.0 or verification < 0.0:
        raise ValueError("Ion consistency values must be non-negative.")
    floor = float(measurement_contract.ion.consistency_absolute_tolerance_A)
    return float(2.0 * abs(primary - verification) / max(abs(primary) + abs(verification), floor))

def require_ion_consistency(primary_ion_A: float, verification_ion_A: float, *, measurement_contract) -> float:
    """Return the consistency error or reject a mismatch above the contract limits."""

    error = ion_consistency_error(
        primary_ion_A,
        verification_ion_A,
        measurement_contract=measurement_contract,
    )
    absolute_error = abs(float(primary_ion_A) - float(verification_ion_A))
    relative_limit = float(measurement_contract.ion.consistency_relative_tolerance)
    absolute_limit = float(measurement_contract.ion.consistency_absolute_tolerance_A)
    if absolute_error > absolute_limit and error > relative_limit:
        raise ValueError(
            "Ion Id-Vg/Id-Vd consistency check failed: "
            f"symmetric_relative_error={error:.6g}, limit={relative_limit:.6g}, "
            f"absolute_error_A={absolute_error:.6g}, absolute_limit_A={absolute_limit:.6g}."
        )
    return error
