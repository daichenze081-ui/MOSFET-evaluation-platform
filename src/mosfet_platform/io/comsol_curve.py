"""Validated loading for one COMSOL current-voltage curve."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mosfet_platform.io.csv_loader import (
    CurrentSignChangeError,
    load_iv_csv,
)

def load_comsol_curve(
    path: Path,
    *,
    curve_type: str,
    fixed_bias_V: float,
    numerical_zero_current_A: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate one Id-Vg or Id-Vd curve registered in the manifest."""
    primary = "vgs" if curve_type == "idvg" else "vds"
    secondary = "vds" if curve_type == "idvg" else "vgs"
    clipped_count = 0
    current_status = "signed_consistent"
    frame = load_iv_csv(
        path,
        required_columns=(primary, "id"),
        optional_columns=(secondary,),
        include_current_metadata=True,
    )
    sign_available = bool(frame["current_sign_available"].iloc[0])
    if not sign_available:
        current_status = "source_current_is_magnitude"
    else:
        raw = frame["id_raw"].to_numpy(dtype=float)
        nonzero = raw[raw != 0.0]
        if np.unique(np.sign(nonzero)).size > 1:
            sign_error = CurrentSignChangeError(
                "Signed current changes sign within one IV curve.",
                path=path,
                column="id",
            )
            above_tolerance = raw[np.abs(raw) > numerical_zero_current_A]
            if np.unique(np.sign(above_tolerance)).size > 1:
                raise sign_error
            clip_mask = (raw != 0.0) & (np.abs(raw) <= numerical_zero_current_A)
            if not np.any(clip_mask):
                raise sign_error
            sanitized = raw.copy()
            sanitized[clip_mask] = 0.0
            frame["id_magnitude"] = np.abs(sanitized)
            frame["id"] = np.abs(sanitized)
            clipped_count = int(np.count_nonzero(clip_mask))
            current_status = "numerical_zero_clipped"

    if secondary in frame:
        values = frame[secondary].dropna().unique()
        if len(values) != 1 or not np.isclose(
            float(values[0]), fixed_bias_V, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                f"{secondary} in the CSV does not match manifest fixed bias "
                f"{fixed_bias_V:g} V."
            )
    else:
        frame[secondary] = float(fixed_bias_V)

    frame = frame.sort_values(primary).reset_index(drop=True)
    return frame, {
        "current_status": current_status,
        "current_sign_available": bool(frame["current_sign_available"].iloc[0]),
        "numerical_zero_clipped_count": clipped_count,
    }
