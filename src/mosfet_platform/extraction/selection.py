from __future__ import annotations

import numpy as np
import pandas as pd

BIAS_ATOL_V = 1.0e-12

def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")

def select_exact_metric_row(
    frame: pd.DataFrame,
    *,
    metric: str,
    vgs_V: float | None,
    vds_V: float,
    condition_id: str,
) -> pd.Series:
    """Select one metric by its complete semantic key without fallback."""
    _require_columns(
        frame,
        {"metric", "vgs_V", "vds_V", "condition_id"},
        "Metric dataframe",
    )
    vgs_values = pd.to_numeric(frame["vgs_V"], errors="coerce")
    vds_values = pd.to_numeric(frame["vds_V"], errors="coerce")
    vgs_mask = (
        vgs_values.isna()
        if vgs_V is None
        else np.isclose(vgs_values, float(vgs_V), rtol=0.0, atol=BIAS_ATOL_V)
    )
    mask = (
        frame["metric"].astype(str).eq(str(metric))
        & frame["condition_id"].astype(str).eq(str(condition_id))
        & vgs_mask
        & np.isclose(vds_values, float(vds_V), rtol=0.0, atol=BIAS_ATOL_V)
    )
    selected = frame.loc[mask]
    if selected.empty:
        raise ValueError(
            "Metric dataframe has no exact match for "
            f"metric={metric}, Vgs={vgs_V}, Vds={vds_V}, condition_id={condition_id}."
        )
    if len(selected) > 1:
        raise ValueError(
            "Metric dataframe has multiple exact matches for "
            f"metric={metric}, Vgs={vgs_V}, Vds={vds_V}, condition_id={condition_id}."
        )
    return selected.iloc[0]

def select_exact_curve_row(
    frame: pd.DataFrame,
    *,
    curve_type: str,
    fixed_bias_V: float,
    condition_id: str,
    case_id: str | None = None,
) -> pd.Series:
    required = {"curve_type", "fixed_bias_V", "condition_id"}
    if case_id is not None:
        required.add("case_id")
    _require_columns(frame, required, "Curve dataframe")
    fixed_values = pd.to_numeric(frame["fixed_bias_V"], errors="coerce")
    mask = (
        frame["curve_type"].astype(str).eq(str(curve_type))
        & frame["condition_id"].astype(str).eq(str(condition_id))
        & np.isclose(
            fixed_values,
            float(fixed_bias_V),
            rtol=0.0,
            atol=BIAS_ATOL_V,
        )
    )
    if case_id is not None:
        mask &= frame["case_id"].astype(str).eq(str(case_id))
    selected = frame.loc[mask]
    key = (
        f"curve_type={curve_type}, fixed_bias={fixed_bias_V}, "
        f"condition_id={condition_id}, case_id={case_id}"
    )
    if selected.empty:
        raise ValueError(f"Curve dataframe has no exact match for {key}.")
    if len(selected) > 1:
        raise ValueError(f"Curve dataframe has multiple exact matches for {key}.")
    return selected.iloc[0]
