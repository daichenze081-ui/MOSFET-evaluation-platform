from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

REQUIRED_METRICS = ("ion", "ioff", "vth", "ss_mv_dec", "gm_max")
ALLOWED_CONDITION_ID_SOURCES = {"csv", "manifest_validated", "generated"}
SPEC_RULES = (
    ("ion_min", "ion", "ion_low", "minimum"),
    ("ioff_max", "ioff", "ioff_high", "maximum"),
    ("ion_ioff_cross_bias_min", "ion_ioff_cross_bias", "ion_ioff_cross_bias_low", "minimum"),
    ("ion_ioff_min", "ion_ioff", "ion_ioff_low", "minimum"),
    ("gm_min", "gm_max", "gm_low", "minimum"),
    ("vth_min", "vth", "vth_low", "minimum"),
    ("vth_max", "vth", "vth_high", "maximum"),
    ("ss_max_mv_dec", "ss_mv_dec", "ss_high", "maximum"),
)
SUPPORTED_SPEC_KEYS = {rule for rule, _, _, _ in SPEC_RULES}

@dataclass(frozen=True)
class MetricViolation:
    metric: str
    rule: str
    reason: str
    value: float
    limit: float
    normalized_exceedance: float

@dataclass(frozen=True)
class SpecMetadata:
    spec_id: str
    version: str
    source: str
    effective_date: str
    device_type: str
    condition_id: str
    approval_status: str = "prototype"
    qualification_level: str = "engineering_demo"

@dataclass(frozen=True)
class SpecDocument:
    metadata: SpecMetadata
    limits: dict[str, float]

def normalize_spec(raw_spec: Mapping[str, Any]) -> dict[str, float]:
    spec = raw_spec.get("spec")
    if not isinstance(spec, Mapping):
        raise ValueError("spec must be a dictionary.")
    limits: dict[str, float] = {}
    for key, value in spec.items():
        if key not in SUPPORTED_SPEC_KEYS:
            raise ValueError(
                f"Unsupported spec key: {key}. Supported keys are: "
                f"{sorted(SUPPORTED_SPEC_KEYS)}"
            )
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{key} must be finite.")
        limits[key] = numeric
    if not limits:
        raise ValueError("spec cannot be empty.")
    if (
        "vth_min" in limits
        and "vth_max" in limits
        and limits["vth_min"] >= limits["vth_max"]
    ):
        raise ValueError("vth_max must be larger than vth_min.")
    positive = {
        "ion_min",
        "ioff_max",
        "ion_ioff_min",
        "ion_ioff_cross_bias_min",
        "gm_min",
        "ss_max_mv_dec",
    }
    for key in positive & limits.keys():
        if limits[key] <= 0.0:
            raise ValueError(f"{key} must be positive.")
    return limits

def parse_spec_document(raw_spec: Mapping[str, Any]) -> SpecDocument:
    metadata = raw_spec.get("spec_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("spec_metadata must be a dictionary.")
    required = (
        "spec_id",
        "version",
        "source",
        "effective_date",
        "device_type",
        "condition_id",
    )
    missing = [key for key in required if not str(metadata.get(key, "")).strip()]
    if missing:
        raise ValueError(f"spec_metadata is missing: {', '.join(missing)}.")
    return SpecDocument(
        metadata=SpecMetadata(
            spec_id=str(metadata["spec_id"]),
            version=str(metadata["version"]),
            source=str(metadata["source"]),
            effective_date=str(metadata["effective_date"]),
            device_type=str(metadata["device_type"]),
            condition_id=str(metadata["condition_id"]),
            approval_status=str(metadata.get("approval_status", "prototype")),
            qualification_level=str(
                metadata.get("qualification_level", "engineering_demo")
            ),
        ),
        limits=normalize_spec(raw_spec),
    )

def _metric_violations(
    metrics: Mapping[str, Any],
    limits: Mapping[str, float],
) -> tuple[MetricViolation, ...]:
    vth_scale = float(limits.get("vth_max", 0.0) - limits.get("vth_min", 0.0))
    violations: list[MetricViolation] = []
    for rule, metric, reason, direction in SPEC_RULES:
        if rule not in limits:
            continue
        value = float(metrics[metric])
        limit = float(limits[rule])
        margin = value - limit if direction == "minimum" else limit - value
        if margin >= 0.0:
            continue
        scale = vth_scale if metric == "vth" and vth_scale > 0.0 else abs(limit)
        if scale <= 0.0:
            raise ValueError(f"Spec rule {rule} needs a positive normalization scale.")
        violations.append(
            MetricViolation(
                metric=metric,
                rule=rule,
                reason=reason,
                value=value,
                limit=limit,
                normalized_exceedance=-margin / scale,
            )
        )
    return tuple(violations)

def _invalid_reasons(
    metrics: Mapping[str, Any],
    document: SpecDocument,
) -> list[str]:
    reasons: list[str] = []
    required = set(REQUIRED_METRICS) | {
        metric for rule, metric, _, _ in SPEC_RULES if rule in document.limits
    }
    for metric in sorted(required):
        raw = metrics.get(metric)
        if raw is None or str(raw).strip() == "":
            reasons.append(f"missing_metric:{metric}")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            reasons.append(f"non_numeric_metric:{metric}")
            continue
        if not np.isfinite(value):
            reasons.append(f"non_finite_metric:{metric}")

    for name in ("vth_status", "ss_status"):
        if name not in metrics:
            reasons.append(f"missing_status:{name}")
        elif str(metrics[name]).strip().lower() != "ok":
            reasons.append(f"{name}_not_ok:{metrics[name]}")

    for name in ("width_m", "length_m"):
        raw = metrics.get(name)
        if raw is None or str(raw).strip() == "":
            reasons.append(f"missing_geometry:{name}")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            reasons.append(f"non_numeric_geometry:{name}")
            continue
        if not np.isfinite(value):
            reasons.append(f"non_finite_geometry:{name}")
        elif value <= 0.0:
            reasons.append(f"non_positive_geometry:{name}")

    eligibility = metrics.get("formal_eligible")
    if eligibility is None or str(eligibility).strip() == "":
        reasons.append("missing_formal_eligibility")
    elif isinstance(eligibility, (bool, np.bool_)):
        if not bool(eligibility):
            reasons.append("formal_eligible_not_true")
    elif str(eligibility).strip().lower() not in {"true", "1"}:
        reasons.append("formal_eligible_not_true")

    quality = metrics.get("data_quality_status")
    if quality is not None and str(quality).strip().lower() not in {"ok", "valid"}:
        reasons.append(f"data_quality_status_not_ok:{quality}")

    actual_condition = metrics.get("condition_id")
    if actual_condition is None or str(actual_condition).strip() == "":
        reasons.append("missing_condition_id")
    elif str(actual_condition) != document.metadata.condition_id:
        reasons.append("condition_id_mismatch")
    source = metrics.get("condition_id_source")
    declared = {item.strip() for item in str(source or "").split(";") if item.strip()}
    unknown = sorted(declared - ALLOWED_CONDITION_ID_SOURCES)
    if not declared:
        reasons.append("missing_condition_id_source")
    elif unknown:
        reasons.append("unknown_condition_id_source:" + ",".join(unknown))
    return reasons

def _judge_row(
    metrics: Mapping[str, Any],
    document: SpecDocument,
) -> tuple[str, str, tuple[MetricViolation, ...]]:
    invalid = _invalid_reasons(metrics, document)
    if invalid:
        return "INVALID", ";".join(invalid), ()
    violations = _metric_violations(metrics, document.limits)
    reasons = ";".join(item.reason for item in violations)
    return ("FAIL", reasons, violations) if violations else ("PASS", "", ())

def judge_single_device(
    metrics: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> tuple[str, str]:
    status, reason, _ = _judge_row(metrics, parse_spec_document(spec))
    return status, reason

def judge_metrics_dataframe(
    metrics_df: pd.DataFrame,
    raw_spec: Mapping[str, Any],
) -> pd.DataFrame:
    document = parse_spec_document(raw_spec)
    judged = metrics_df.copy()
    results = [_judge_row(row, document) for row in judged.to_dict(orient="records")]
    primary = [
        max(violations, key=lambda item: item.normalized_exceedance, default=None)
        for _, _, violations in results
    ]
    judged["status"] = [status for status, _, _ in results]
    judged["pass_fail"] = judged["status"]
    judged["fail_reason"] = [reason for _, reason, _ in results]
    judged["primary_failure_reason"] = [item.reason if item else "" for item in primary]
    judged["primary_metric"] = [item.metric if item else "" for item in primary]
    judged["primary_value"] = [item.value if item else np.nan for item in primary]
    judged["primary_limit"] = [item.limit if item else np.nan for item in primary]
    judged["normalized_exceedance"] = [
        item.normalized_exceedance if item else 0.0 for item in primary
    ]
    judged["violations"] = [
        ";".join(item.reason for item in violations)
        for _, _, violations in results
    ]
    return judged
