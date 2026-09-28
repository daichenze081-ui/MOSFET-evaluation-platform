from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "vgs": ("vgs", "vg", "gate_voltage", "gate voltage", "vgs (v)", "vg (v)"),
    "vds": ("vds", "vd", "drain_voltage", "drain voltage", "vds (v)", "vd (v)"),
    "id": (
        "id", "ids", "drain_current", "drain current",
        "abs(semi.i0_1) (a)", "terminal current", "terminal current (a)",
        "终端电流", "终端电流 (a)", "semi.i0_1", "i0_1",
    ),
}
_COLUMN_UNITS = {"vgs": "v", "vds": "v", "id": "a"}
# Match a unit suffix without consuming the parentheses in abs(semi.I0_1).
_UNIT_SUFFIX = re.compile(r"\s*\(([a-zµμ]+)\)$")

class IVCSVError(ValueError):
    """Base class for machine-readable IV CSV content errors."""

    code = "iv_csv_error"

    def __init__(self, message: str, *, path: str | Path, column: str | None = None) -> None:
        super().__init__(message)
        self.path = Path(path)
        self.column = column

class EmptyIVCSVError(IVCSVError):
    code = "empty_input"

class MissingColumnError(IVCSVError):
    code = "missing_column"

    def __init__(self, *, path: str | Path, column: str, aliases: Iterable[str]) -> None:
        aliases_text = ", ".join(aliases)
        super().__init__(
            f"{Path(path)} must contain a '{column}' column or one of its aliases: {aliases_text}.",
            path=path,
            column=column,
        )
        self.aliases = tuple(aliases)

class InvalidNumericValueError(IVCSVError):
    code = "invalid_numeric_value"

class NonFiniteColumnError(IVCSVError):
    code = "non_finite_value"

class CurrentSignChangeError(IVCSVError):
    code = "current_sign_change"

class MalformedIVCSVError(IVCSVError):
    code = "invalid_csv_format"

class UnsupportedUnitError(IVCSVError):
    code = "unsupported_unit"

def _column_label(name: str) -> str:
    return " ".join(str(name).lstrip("\ufeff% \t").strip().lower().split())

def _normalized_column_name(name: str) -> str:
    return _UNIT_SUFFIX.sub("", _column_label(name)).strip()

def _find_column(df: pd.DataFrame, canonical_name: str) -> str | None:
    aliases = COLUMN_ALIASES.get(canonical_name, (canonical_name,))
    normalized_to_original = {_normalized_column_name(column): column for column in df.columns}
    for alias in aliases:
        normalized_alias = _normalized_column_name(alias)
        if normalized_alias in normalized_to_original:
            return str(normalized_to_original[normalized_alias])
    return None

def _has_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> bool:
    return all(_find_column(df, column) is not None for column in required_columns)

def _read_csv_with_optional_comsol_header(
    input_path: Path,
    required_columns: tuple[str, ...],
) -> pd.DataFrame:
    try:
        with input_path.open(encoding="utf-8-sig", newline="") as stream:
            for row_index, line in enumerate(stream):
                cleaned = line.strip()
                if not cleaned:
                    continue
                columns = next(csv.reader([cleaned.lstrip("%").strip()], strict=True))
                if not cleaned.startswith("%") or _has_required_columns(
                    pd.DataFrame(columns=columns), required_columns
                ):
                    break
            else:
                raise pd.errors.EmptyDataError("No CSV header or data found.")
        return pd.read_csv(
            input_path,
            skiprows=row_index + 1,
            names=[column.strip() for column in columns],
            encoding="utf-8-sig",
        )
    except pd.errors.EmptyDataError as error:
        raise EmptyIVCSVError(
            "Input CSV contains no data rows.",
            path=input_path,
        ) from error
    except (pd.errors.ParserError, csv.Error) as error:
        raise MalformedIVCSVError(
            "Input CSV has an invalid CSV structure.",
            path=input_path,
        ) from error

def _source_is_magnitude(column_name: str) -> bool:
    normalized = str(column_name).strip().lower().replace(" ", "")
    return normalized.startswith("abs(") or "magnitude" in normalized

def load_iv_csv(
    input_path: str | Path,
    required_columns: Iterable[str] = ("vgs", "vds", "id"),
    optional_columns: Iterable[str] = (),
    current_magnitude: bool = True,
    *,
    include_current_metadata: bool = False,
) -> pd.DataFrame:
    """Load V/A-valued IV columns without rescaling their numerical values.

    Unitless headers assume V/A. Explicit suffixes must agree with these units.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {input_path}")
    required_columns = tuple(dict.fromkeys(required_columns))
    optional_columns = tuple(
        column
        for column in dict.fromkeys(optional_columns)
        if column not in required_columns
    )
    raw_df = _read_csv_with_optional_comsol_header(input_path, required_columns)
    if raw_df.empty:
        raise EmptyIVCSVError(
            "Input CSV contains no data rows.",
            path=input_path,
        )
    output_df = pd.DataFrame()
    selected_columns = required_columns + tuple(
        column
        for column in optional_columns
        if _find_column(raw_df, column) is not None
    )
    for canonical_name in selected_columns:
        source_column = _find_column(raw_df, canonical_name)
        if source_column is None:
            raise MissingColumnError(
                path=input_path,
                column=canonical_name,
                aliases=COLUMN_ALIASES.get(canonical_name, (canonical_name,)),
            )
        unit = _UNIT_SUFFIX.search(_column_label(source_column))
        expected_unit = _COLUMN_UNITS.get(canonical_name)
        if unit and expected_unit and unit[1] != expected_unit:
            raise UnsupportedUnitError(
                f"Column {source_column!r} has unit {unit[1]!r}; expected "
                f"{expected_unit.upper()}. Supply values in that unit; no automatic scaling is applied.",
                path=input_path,
                column=canonical_name,
            )
        try:
            values = pd.to_numeric(raw_df[source_column], errors="raise")
        except (TypeError, ValueError) as error:
            raise InvalidNumericValueError(
                f"Input CSV contains a non-numeric value in column: {canonical_name}",
                path=input_path,
                column=canonical_name,
            ) from error
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise NonFiniteColumnError(
                f"Input CSV contains NaN or Inf in column: {canonical_name}",
                path=input_path,
                column=canonical_name,
            )
        if canonical_name == "id":
            raw_current = values.astype(float)
            magnitude = raw_current.abs()
            output_df["id"] = magnitude if current_magnitude else raw_current
            if include_current_metadata:
                output_df["id_raw"] = raw_current
                output_df["id_magnitude"] = magnitude
                sign_available = not _source_is_magnitude(source_column)
                output_df["current_sign_available"] = bool(sign_available)
        else:
            output_df[canonical_name] = values
    return output_df
