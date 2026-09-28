from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from mosfet_platform.analysis.spec import judge_metrics_dataframe
_COLUMNS = (
    "section",
    "group",
    "item",
    "count",
    "denominator_name",
    "denominator_count",
    "rate",
)

def judge_results(
    metrics: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    predicted: bool = False,
) -> pd.DataFrame:
    judged = judge_metrics_dataframe(metrics, spec)
    if predicted:
        judged["status"] = judged["status"].map(
            {"PASS": "PREDICTED_PASS", "FAIL": "PREDICTED_FAIL"}
        ).fillna("INVALID_INPUT")
        judged["pass_fail"] = judged["status"]
    return judged


def order_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    status = result["status"].astype(str).str.upper()
    rank = {
        "PASS": 0,
        "PREDICTED_PASS": 0,
        "FAIL": 1,
        "PREDICTED_FAIL": 1,
        "RETEST": 2,
        "INVALID": 2,
        "OUT_OF_ENVELOPE": 2,
        "INVALID_INPUT": 2,
    }
    result["_rank"] = status.map(rank).fillna(2)
    result["_severity"] = pd.to_numeric(
        result.get("normalized_exceedance", 0.0), errors="coerce"
    ).fillna(0.0)
    fail = result["_rank"].eq(1)
    group_max = (
        result.loc[fail]
        .groupby("primary_failure_reason")["_severity"]
        .max()
        .to_dict()
    )
    result["_group_severity"] = result["primary_failure_reason"].map(group_max).fillna(0.0)
    result = result.sort_values(
        ["_rank", "_group_severity", "primary_failure_reason", "_severity", "device_id"],
        ascending=[True, False, True, False, True],
        kind="stable",
    )
    return result.drop(columns=["_rank", "_severity", "_group_severity"]).reset_index(drop=True)


def _row(
    section: str,
    group: str,
    item: str,
    count: int,
    denominator_name: str,
    denominator: int,
) -> dict[str, Any]:
    return {
        "section": section,
        "group": group,
        "item": item,
        "count": int(count),
        "denominator_name": denominator_name,
        "denominator_count": int(denominator),
        "rate": float(count / denominator) if denominator else np.nan,
    }


def summarize_results(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=_COLUMNS)
    status = frame["status"].astype(str).str.upper()
    total = len(frame)
    formal = status.isin({"PASS", "FAIL", "PREDICTED_PASS", "PREDICTED_FAIL"})
    failed = status.isin({"FAIL", "PREDICTED_FAIL"})
    rows = [
        _row("judgement", "status", item, int((status == item).sum()), "all_inputs", total)
        for item in status.drop_duplicates()
    ]
    formal_count = int(formal.sum())
    for item in ("PASS", "FAIL", "PREDICTED_PASS", "PREDICTED_FAIL"):
        count = int((status == item).sum())
        if count or item in set(status):
            rows.append(
                _row("judgement", "formal", item, count, "formal_evaluated", formal_count)
            )
    fail_count = int(failed.sum())
    if fail_count:
        primary = frame.loc[failed, "primary_failure_reason"].fillna("").astype(str)
        for item, count in primary[primary.ne("")].value_counts().items():
            rows.append(
                _row(
                    "performance",
                    "primary_failure",
                    item,
                    int(count),
                    "formal_failures",
                    fail_count,
                )
            )
        labels = frame.loc[formal, "violations"].fillna("").astype(str)
        for item in sorted({label for cell in labels for label in cell.split(";") if label}):
            count = int(labels.map(lambda cell: item in cell.split(";")).sum())
            rows.append(
                _row(
                    "performance",
                    "metric_violation",
                    item,
                    count,
                    "formal_evaluated",
                    formal_count,
                )
            )
    issue_mask = ~formal
    if issue_mask.any():
        issues = frame.loc[issue_mask, "fail_reason"].fillna("").astype(str)
        for item in sorted({label for cell in issues for label in cell.split(";") if label}):
            count = int(issues.map(lambda cell: item in cell.split(";")).sum())
            rows.append(_row("data_quality", "issue", item, count, "all_inputs", total))
    return pd.DataFrame(rows, columns=_COLUMNS)
