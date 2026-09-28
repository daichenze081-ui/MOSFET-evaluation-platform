from __future__ import annotations

import numpy as np
import pandas as pd

def _current_at_vds(vds: np.ndarray, ids: np.ndarray, target_vds: float) -> float:
    """Linearly interpolate current inside the measured Vds range only."""

    vds = np.asarray(vds, dtype=float)
    ids = np.asarray(ids, dtype=float)
    target_vds = float(target_vds)
    if target_vds < float(vds[0]) or target_vds > float(vds[-1]):
        raise ValueError("target Vds lies outside the measured curve; extrapolation is forbidden.")
    return float(np.interp(target_vds, vds, ids))


def _slope_from_window(vds: np.ndarray, ids: np.ndarray, mask: np.ndarray) -> float:
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")
    slope, _ = np.polyfit(vds[mask], ids[mask], deg=1)
    return float(slope)


def extract_idvd_metrics_from_dataframe(
    df: pd.DataFrame,
    *,
    id_at_vds_values: tuple[float, ...] = (0.05, 0.2),
    ron_vds: float = 0.05,
    low_vds_max: float = 0.05,
    high_vds_min: float = 0.2,
) -> pd.DataFrame:

    missing = {"vgs", "vds", "id"} - set(df.columns)
    if missing:
        raise ValueError("Id-Vd dataframe is missing required columns: " + ", ".join(sorted(missing)))
    rows: list[dict[str, float | int | str]] = []
    for vgs_value, group in df.groupby("vgs"):
        curve = group.sort_values("vds").reset_index(drop=True)
        vds = curve["vds"].to_numpy(dtype=float)
        ids = curve["id"].abs().to_numpy(dtype=float)
        if vds.size < 2 or np.any(~np.isfinite(vds)) or np.any(~np.isfinite(ids)):
            raise ValueError("Each Id-Vd curve needs at least two Vds points with finite values.")
        if np.any(np.diff(vds) <= 0.0):
            raise ValueError("Vds values must be strictly increasing within each Vgs curve.")
        row: dict[str, float | int | str] = {
            "vgs": float(vgs_value), "point_count": int(len(curve)),
            "vds_min": float(vds[0]), "vds_max": float(vds[-1]),
        }
        for target in id_at_vds_values:
            label = str(target).replace(".", "p")
            row[f"id_at_vd_{label}"] = _current_at_vds(vds, ids, target)
        ron_current = _current_at_vds(vds, ids, ron_vds)
        row["ron_vds"] = float(ron_vds)
        row["ron_ohm"] = float(ron_vds / ron_current) if ron_current > 0.0 else float("inf")

        low_mask = (vds >= 0.0) & (vds <= float(low_vds_max))
        high_mask = vds >= float(high_vds_min)
        low_slope = _slope_from_window(vds, ids, low_mask)
        high_slope = _slope_from_window(vds, ids, high_mask)
        row["low_vd_slope_S"] = low_slope
        row["gds_S"] = high_slope
        row["gds_vds_min"] = float(np.min(vds[high_mask])) if np.any(high_mask) else float("nan")
        row["gds_vds_max"] = float(np.max(vds[high_mask])) if np.any(high_mask) else float("nan")
        row["gds_point_count"] = int(np.count_nonzero(high_mask))
        if np.count_nonzero(high_mask) >= 2 and np.isfinite(high_slope):
            high_reference_vds = float(np.mean(vds[high_mask]))
            high_reference_id = _current_at_vds(vds, ids, high_reference_vds)
            row["lambda_estimate_1_V"] = (
                float(high_slope / high_reference_id) if high_reference_id > 0.0 else float("nan")
            )
            row["lambda_reference_vds"] = high_reference_vds
            row["lambda_note"] = "approximate_gds_over_id_same_high_vds_window"
        else:
            row["lambda_estimate_1_V"] = float("nan")
            row["lambda_reference_vds"] = float("nan")
            row["lambda_note"] = "insufficient_high_vds_points"
        quasi = (
            float(high_slope / low_slope)
            if np.isfinite(high_slope) and np.isfinite(low_slope) and low_slope > 0.0
            else float("nan")
        )
        row["quasi_saturation_index"] = quasi
        row["quasi_saturation_rule"] = "project_heuristic_thresholds_0p3_0p8"
        if np.isfinite(quasi):
            row["saturation_note"] = (
                "clear_saturation_trend" if quasi < 0.3
                else "quasi_saturation" if quasi < 0.8
                else "slowly_rising_high_vd"
            )
        else:
            row["saturation_note"] = "insufficient_slope_window"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("vgs").reset_index(drop=True)
