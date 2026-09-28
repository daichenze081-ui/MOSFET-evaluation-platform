"""Read declared electrical metrics without inferring or rescaling units."""

from __future__ import annotations

import csv
from pathlib import Path
import re

import numpy as np
import pandas as pd

from mosfet_platform.measurement import MeasurementContract
from mosfet_platform.provenance import file_sha256


METRIC_UNITS = {
    "ion": "A", "ioff": "A", "gm_max": "S", "vth": "V",
    "ss_mv_dec": "mV/dec", "dibl_mV_per_V": "mV/V",
    "ion_ioff": "1", "ion_ioff_cross_bias": "1",
    "width_m": "m", "length_m": "m", "oxide_thickness_m": "m",
    "temperature_K": "K",
}
_ALIASES = {name.lower(): name for name in METRIC_UNITS}
_ALIASES.update({"ss": "ss_mv_dec", "gm": "gm_max", "dibl": "dibl_mV_per_V"})
_ALIASES.update({
    "width": "width_m", "length": "length_m", "tox": "oxide_thickness_m",
    "temperature": "temperature_K",
})
_ALIASES.update({
    f"{name}_{unit}".lower(): name
    for name, unit in METRIC_UNITS.items()
    if unit in {"A", "V", "S"}
})


def _column_name(value: str) -> str:
    name = value.strip()
    decorated = re.fullmatch(r"(.+?)\s*(?:\(([^()]*)\)|\[([^\[\]]*)\])", name)
    if decorated:
        base, round_unit, square_unit = decorated.groups()
        name = _ALIASES.get(base.strip().lower(), base.strip())
        unit = (round_unit or square_unit or "").strip()
        if name not in METRIC_UNITS or unit != METRIC_UNITS[name]:
            raise ValueError(f"Unsupported metric column or unit: {value}.")
        return name
    return _ALIASES.get(name.lower(), name)


def _context_errors(row: dict, contract: MeasurementContract) -> list[str]:
    errors = []
    for name in ("width_m", "length_m", "oxide_thickness_m", "temperature_K"):
        try:
            value = float(row.get(name, ""))
        except (TypeError, ValueError):
            errors.append(f"missing_or_non_numeric:{name}")
            continue
        if not np.isfinite(value) or value <= 0.0:
            errors.append(f"invalid_positive_value:{name}")
    conditions = {
        "width_m": contract.geometry.width_m,
        "temperature_K": contract.temperature_K,
    }
    optional_biases = {
        "ion_vgs_V": contract.ion.vgs_V,
        "ion_vds_V": contract.ion.vds_V,
        "transfer_vds_V": contract.transfer.vds_V,
        "ioff_vgs_V": contract.transfer.vgs_off_V,
        "ioff_vds_V": contract.transfer.vds_V,
        "same_vds_ratio_vgs_V": contract.transfer.vgs_on_V,
        "same_vds_ratio_vds_V": contract.transfer.vds_V,
    }
    conditions.update({name: value for name, value in optional_biases.items() if name in row})
    for name, expected in conditions.items():
        value = pd.to_numeric(row.get(name, ""), errors="coerce")
        if not np.isclose(value, expected, rtol=1e-9, atol=0.0):
            errors.append(f"contract_mismatch:{name}")
    if str(row.get("device_type", "")).lower() != contract.device_type.lower():
        errors.append("contract_mismatch:device_type")
    for name in ("ion", "ioff", "gm_max", "ss_mv_dec", "ion_ioff", "ion_ioff_cross_bias"):
        value = pd.to_numeric(row.get(name, ""), errors="coerce")
        if value < 0.0:
            errors.append(f"negative_metric:{name}")
    return errors


def read_metric_table(path: Path, *, unique_device: bool = True) -> pd.DataFrame:
    """Read declared columns without imposing training or complete-row eligibility."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        records = [row for row in reader if row]
    columns = [_column_name(name) for name in header]
    if not columns or any(not name for name in columns) or len(set(columns)) != len(columns):
        raise ValueError("Metric table must have unique, non-empty columns.")
    if any(len(row) != len(columns) for row in records):
        raise ValueError("Metric table rows must match the header column count.")
    frame = pd.DataFrame(records, columns=columns)
    if frame.empty or "device_id" not in frame:
        raise ValueError("Metric table must contain devices and a device_id column.")
    frame["device_id"] = frame["device_id"].str.strip()
    if frame["device_id"].eq("").any() or (unique_device and frame["device_id"].duplicated().any()):
        raise ValueError("Metric table device_id values must be non-empty and unique.")

    return frame


def load_metric_table(path: Path, contract: MeasurementContract) -> pd.DataFrame:
    frame = read_metric_table(path)

    rows = []
    digest = file_sha256(path)
    for row in frame.to_dict(orient="records"):
        errors = _context_errors(row, contract)
        quality = str(row.get("data_quality_status", "ok"))
        row.update({
            "upstream_result_origin": row.get("upstream_result_origin", row.get("result_origin", "external")),
            "upstream_source_type": row.get("upstream_source_type", row.get("source_type", "external")),
            "data_quality_status": ";".join(["invalid", quality, *errors]) if errors else quality,
            "formal_eligible": row.get("formal_eligible", True),
            "condition_id_source": "csv",
            "result_origin": "imported_metrics",
            "source_type": "metric_table",
            "source_id": path.name,
            "source_sha256": digest,
        })
        rows.append(row)
    return pd.DataFrame(rows)
