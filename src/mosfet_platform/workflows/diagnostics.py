from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def write_diagnostics(
    root: Path,
    *,
    row: Mapping[str, Any],
    curves: pd.DataFrame,
    source: Mapping[str, Any],
    issues: Sequence[Any] = (),
) -> Path:
    device_id = str(row["device_id"])
    if not device_id or any(char in device_id for char in "/\\"):
        raise ValueError("Diagnostic device_id is invalid.")
    target = root / "diagnostics" / "devices" / device_id
    target.mkdir(parents=True, exist_ok=True)
    curves.to_csv(target / "curves.csv", index=False)

    metrics = {
        name: _value(row.get(name))
        for name in ("ion", "ioff", "ion_ioff_cross_bias", "vth", "ss_mv_dec", "dibl_mV_per_V")
    }
    (target / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    raw_labels = row.get("violations", "")
    labels = [] if pd.isna(raw_labels) else [item for item in str(raw_labels).split(";") if item]
    records = [{"type": "violation", "code": item} for item in labels]
    records.extend(
        {"type": "issue", "code": issue.code, "message": issue.message}
        for issue in issues
    )
    pd.DataFrame(records, columns=("type", "code", "message")).to_csv(
        target / "violations.csv", index=False
    )
    (target / "provenance.json").write_text(
        json.dumps(
            {name: _value(value) for name, value in source.items()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    lines = [
        f"# {device_id}",
        "",
        f"Status: {row.get('status', '')}",
        f"Condition: {row.get('condition_id', '')}",
        f"Source: {row.get('source_id', '')}",
        f"Primary failure: {row.get('primary_failure_reason', '')}",
        f"Curve points: {len(curves)}",
    ]
    (target / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
