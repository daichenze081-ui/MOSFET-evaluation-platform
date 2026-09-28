"""Expected characterization coverage derived from registered curves."""

from __future__ import annotations

import pandas as pd


def inventory_counts(inventory: pd.DataFrame) -> dict[str, int]:
    formal = inventory["analysis_role"].eq("formal")
    diagnostic = inventory["analysis_role"].eq("diagnostic_only")
    cases = int(inventory["case_id"].nunique())
    return {
        "cases": cases,
        "curves": len(inventory),
        "active_curves": int(inventory["qc_status"].eq("active").sum()),
        "isolated_curves": int(inventory["qc_status"].eq("isolated").sum()),
        "formal_curves": int(formal.sum()),
        "diagnostic_only_curves": int(diagnostic.sum()),
        "inventory_only_curves": int(inventory["analysis_role"].eq("inventory_only").sum()),
        "idvg_metric_rows": int((formal & inventory["curve_type"].eq("idvg")).sum()),
        "idvd_metric_rows": int((formal & inventory["curve_type"].eq("idvd")).sum()),
        "isolated_diagnostic_rows": int(diagnostic.sum()),
        "case_summary_rows": cases,
    }
