"""Per-metric Spec decisions and source-separated descriptive statistics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from itertools import combinations
import math
from typing import Any, Mapping, Sequence

from mosfet_platform.analysis.spec import SPEC_RULES, parse_spec_document
from mosfet_platform.io.metric_table import METRIC_UNITS

SOURCE_ALIASES = {
    "measured": "measured", "measurement": "measured", "experimental": "measured",
    "comsol": "comsol", "synthetic": "synthetic", "estimated": "estimated",
    "predicted": "estimated", "prediction": "estimated", "model": "estimated",
}
GROUP_FIELDS = ("temperature_K", "length_m", "oxide_thickness_m", "width_m", "batch_id")
VALID = {"PASS", "FAIL"}


def canonical_source(value: Any) -> str:
    name = str(value or "").strip().lower()
    if name not in SOURCE_ALIASES:
        raise ValueError("Declare source_type as measured, comsol, synthetic, or estimated.")
    return SOURCE_ALIASES[name]


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _metric_state(row: Mapping, metric: str) -> tuple[str, float | None, str]:
    value = row.get(metric)
    number = finite_number(value)
    errors = row.get("context_errors", [])
    if errors:
        return "INVALID", number, ";".join(errors)
    issue = row.get("metric_issues", {}).get(metric)
    if issue:
        return issue["status"], number, issue["reason"]
    if value is None or (isinstance(value, str) and not value.strip()):
        return "MISSING", None, "missing_value"
    if number is None:
        return "INVALID", None, "non_numeric_or_non_finite"
    if metric != "vth" and number < 0:
        return "INVALID", number, "negative_metric"
    status_field = {"vth": "vth_status", "ss_mv_dec": "ss_status"}.get(metric, metric + "_status")
    status = row.get(status_field)
    if status is None and metric in {"vth", "ss_mv_dec"}:
        return "INVALID", number, "missing_extraction_status"
    if status is not None and str(status).strip().lower() != "ok":
        return "INVALID", number, f"{status_field}:{status}"
    return "VALID", number, ""


def _summary(devices: Sequence[dict], details: Sequence[dict], metrics: Sequence[str]) -> list[dict]:
    failed = {d["device_id"] for d in devices if d["status"] == "FAIL"}
    by_id = {d["device_id"]: d for d in devices}
    result = []
    for metric in metrics:
        rows = [r for r in details if r["metric"] == metric]
        valid = [r for r in rows if r["status"] in VALID]
        violations = [r for r in rows if r["status"] == "FAIL"]
        exclusive = [r["device_id"] for r in violations
                     if by_id[r["device_id"]]["complete"]
                     and len(by_id[r["device_id"]]["failed_metrics"]) == 1]
        result.append({
            "metric": metric, "unit": METRIC_UNITS[metric],
            "sample_count": len(devices), "evaluated_count": len(valid),
            "fail_count": len(violations), "exceedance_rate": _ratio(len(violations), len(valid)),
            "missing_count": sum(r["status"] == "MISSING" for r in rows),
            "invalid_count": sum(r["status"] == "INVALID" for r in rows),
            "failed_sample_count": len(failed),
            "failed_sample_evaluated_count": sum(r["device_id"] in failed for r in valid),
            "failure_coverage": _ratio(len(violations), len(failed)),
            "exclusive_fail_count": len(exclusive), "exclusive_device_ids": sorted(exclusive),
            "max_normalized_exceedance": max((r["normalized_exceedance"] for r in violations
                                              if r["normalized_exceedance"] is not None), default=None),
            "failed_device_ids": sorted(r["device_id"] for r in violations),
        })
    return sorted(result, key=lambda r: (-(r["exceedance_rate"] or 0), -r["fail_count"], r["metric"]))


def diagnose_records(
    records: Sequence[Mapping[str, Any]], raw_spec: Mapping[str, Any],
    *, group_by: Sequence[str] = ("length_m", "oxide_thickness_m"),
) -> dict[str, Any]:
    """Diagnose selected observations; duplicate identities must be resolved upstream."""
    document = parse_spec_document(raw_spec)
    if not records:
        raise ValueError("No observations selected for diagnosis.")
    if any(field not in GROUP_FIELDS for field in group_by):
        raise ValueError(f"group_by supports only {GROUP_FIELDS}.")
    rules = [rule for rule in SPEC_RULES if rule[0] in document.limits]
    metrics = list(dict.fromkeys(rule[1] for rule in rules))
    devices, details, rule_details = [], [], []
    seen = set()
    for original in records:
        row = dict(original)
        device = str(row.get("device_id", "")).strip()
        source = canonical_source(row.get("source_type"))
        condition = str(row.get("condition_id", ""))
        if not device:
            raise ValueError("device_id must be non-empty.")
        identity = (device, condition, source)
        if identity in seen:
            raise ValueError(f"Select one current record for device {device}, condition {condition}, source {source}.")
        seen.add(identity)
        row["context_errors"] = list(row.get("context_errors", []))
        if condition != document.metadata.condition_id:
            row["context_errors"].append("condition_id_mismatch")
        if str(row.get("device_type", "")).lower() != document.metadata.device_type.lower():
            row["context_errors"].append("device_type_mismatch")
        if str(row.get("formal_eligible", True)).lower() not in {"true", "1"}:
            row["context_errors"].append("declared_ineligible")
        if str(row.get("data_quality_status", "ok")).lower() not in {"ok", "valid"}:
            row["context_errors"].append("declared_quality_invalid")
        context = {"device_id": device, "condition_id": condition, "source_type": source}
        metric_rows = []
        for metric in metrics:
            state, value, reason = _metric_state(row, metric)
            checks = []
            for key, name, failure, direction in rules:
                if name != metric:
                    continue
                limit = document.limits[key]
                margin = None if state != "VALID" else (value - limit if direction == "minimum" else limit - value)
                fails = state == "VALID" and (value < limit if direction == "minimum" else value > limit)
                status = state if state != "VALID" else ("FAIL" if fails else "PASS")
                span = document.limits.get("vth_max", 0) - document.limits.get("vth_min", 0)
                scale = span if metric == "vth" and span > 0 else abs(limit)
                normalized = max(0.0, -margin / scale) if margin is not None and scale else None
                checks.append({**context, "metric": metric, "rule": key, "value": value,
                               "unit": METRIC_UNITS[metric], "limit": limit, "direction": direction,
                               "status": status, "reason": failure if status == "FAIL" else reason,
                               "margin": finite_number(margin), "normalized_exceedance": finite_number(normalized)})
            failed_checks = [c for c in checks if c["status"] == "FAIL"]
            status = state if state != "VALID" else ("FAIL" if failed_checks else "PASS")
            item = {**context, "metric": metric, "value": value, "unit": METRIC_UNITS[metric],
                    "status": status, "reason": ";".join(c["reason"] for c in failed_checks) or reason,
                    "failed_rules": [c["rule"] for c in failed_checks],
                    "normalized_exceedance": max((c["normalized_exceedance"] for c in checks
                                                  if c["normalized_exceedance"] is not None), default=None),
                    "evidence": row.get("metric_evidence", {}).get(metric, row.get("evidence", [])),
                    "method": row.get("method", "imported_metrics")}
            metric_rows.append(item)
            rule_details.extend(checks)
        failed_metrics = [r["metric"] for r in metric_rows if r["status"] == "FAIL"]
        complete = all(r["status"] in VALID for r in metric_rows)
        status = "FAIL" if failed_metrics else ("PASS" if complete else "INCOMPLETE")
        devices.append({**context, "status": status, "complete": complete,
                        "conclusion_basis": source.upper(), "estimated": source == "estimated",
                        "failed_metrics": failed_metrics,
                        "unavailable_metrics": [r["metric"] for r in metric_rows if r["status"] not in VALID],
                        **{field: (str(row[field]) if field == "batch_id" else finite_number(row[field]))
                           for field in GROUP_FIELDS if field in row},
                        "issues": row.get("issues", [])})
        details.extend(metric_rows)

    partitions = defaultdict(list)
    for device in devices:
        partitions[(device["source_type"], device["condition_id"])].append(device)
    summaries, source_counts, pairs, groups = [], [], [], []
    for (source, condition), selected in sorted(partitions.items()):
        context = {"source_type": source, "condition_id": condition}
        selected_details = [r for r in details if r["source_type"] == source and r["condition_id"] == condition]
        summaries.extend({**context, **r} for r in _summary(selected, selected_details, metrics))
        source_counts.append({**context, "observations": len(selected),
                              **{s: sum(d["status"] == s for d in selected) for s in ("PASS", "FAIL", "INCOMPLETE")},
                              "incomplete_evidence": sum(not d["complete"] for d in selected)})
        indexed = {(r["device_id"], r["metric"]): r for r in selected_details}
        for left, right in combinations(metrics, 2):
            available = [d["device_id"] for d in selected
                         if indexed[d["device_id"], left]["status"] in VALID
                         and indexed[d["device_id"], right]["status"] in VALID]
            both = [d for d in available if indexed[d, left]["status"] == "FAIL"
                    and indexed[d, right]["status"] == "FAIL"]
            pairs.append({**context, "left_metric": left, "right_metric": right,
                          "evaluated_count": len(available), "both_fail_count": len(both),
                          "rate": _ratio(len(both), len(available)), "device_ids": sorted(both)})
        for field in dict.fromkeys(group_by):
            values = {d[field] for d in selected if d.get(field) is not None}
            for value in sorted(values, key=str):
                subset = [d for d in selected if d.get(field) == value]
                ids = {d["device_id"] for d in subset}
                rows = [r for r in selected_details if r["device_id"] in ids]
                groups.extend({**context, "group_by": field, "group_value": value, **r}
                              for r in _summary(subset, rows, metrics))
    return {"spec": {"metadata": asdict(document.metadata), "limits": document.limits},
            "observation_count": len(devices), "device_count": len({d["device_id"] for d in devices}),
            "source_summaries": source_counts, "devices": devices, "metric_details": details,
            "rule_details": rule_details, "metric_summary": summaries,
            "cooccurrence": pairs, "groups": groups,
            "gaps": [r for r in details if r["status"] not in VALID]}
