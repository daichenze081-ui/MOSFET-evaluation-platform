from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import yaml


_CURVE_TYPES = ("idvg", "idvd")
_QC_STATUSES = {"active", "isolated"}
_ANALYSIS_ROLES = {"formal", "diagnostic_only", "inventory_only"}
_ALLOWED_CURVE_ROLES = {
    ("active", "formal"),
    ("isolated", "diagnostic_only"),
    ("isolated", "inventory_only"),
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping.")
    return value


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric.") from error
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


@dataclass(frozen=True)
class COMSOLCaseManifest:
    path: Path
    project_root: Path
    source_root: Path
    nominal_case_id: str
    cases: tuple[Mapping[str, Any], ...]
    independent_validation_cases: tuple[Mapping[str, Any], ...]
    raw: Mapping[str, Any]

    @property
    def nominal_case(self) -> Mapping[str, Any]:
        return next(case for case in self.cases if case["case_id"] == self.nominal_case_id)

    def iter_curves(
        self,
        *,
        include_isolated: bool = True,
    ) -> Iterator[tuple[Mapping[str, Any], str, Mapping[str, Any]]]:
        for case in self.cases:
            for curve_type in _CURVE_TYPES:
                for curve in case[curve_type]:
                    if include_isolated or curve["qc_status"] == "active":
                        yield case, curve_type, curve

    def iter_independent_validation_curves(
        self,
        *,
        include_isolated: bool = True,
    ) -> Iterator[tuple[Mapping[str, Any], str, Mapping[str, Any]]]:
        for case in self.independent_validation_cases:
            for curve_type in _CURVE_TYPES:
                for curve in case[curve_type]:
                    if include_isolated or curve["qc_status"] == "active":
                        yield case, curve_type, curve

    def require_nominal_curve(self, curve_type: str, path: str | Path) -> Mapping[str, Any]:
        if curve_type not in _CURVE_TYPES:
            raise ValueError(f"Unsupported curve type: {curve_type}")
        requested = (self.project_root / Path(path)).resolve()
        for curve in self.nominal_case[curve_type]:
            candidate = (self.project_root / Path(str(curve["path"]))).resolve()
            if candidate == requested and curve["qc_status"] == "active":
                return curve
        raise ValueError(
            f"Configured {curve_type} path is not an active curve in nominal case "
            f"'{self.nominal_case_id}': {path}"
        )


def load_comsol_case_manifest(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    include_independent_validation: bool = True,
) -> COMSOLCaseManifest:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"COMSOL case manifest not found: {manifest_path}")
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "COMSOL case manifest")
    if int(root.get("schema_version", 0)) != 1:
        raise ValueError("COMSOL case manifest schema_version must be 1.")

    source = _mapping(root.get("source"), "source")
    source_root_value = source.get("root")
    if not isinstance(source_root_value, str) or not source_root_value.strip():
        raise ValueError("source.root must be a non-empty relative path.")
    if Path(source_root_value).is_absolute():
        raise ValueError("source.root must be relative to the project root.")

    resolved_project_root = Path(project_root or Path.cwd()).resolve()
    resolved_source_root = (resolved_project_root / source_root_value).resolve()
    try:
        resolved_source_root.relative_to(resolved_project_root)
    except ValueError as error:
        raise ValueError("source.root must stay within the project root.") from error
    if not resolved_source_root.is_dir():
        raise FileNotFoundError(f"COMSOL source root not found: {resolved_source_root}")

    common = _mapping(root.get("common_conditions"), "common_conditions")
    _positive_float(common.get("width_m"), "common_conditions.width_m")
    _positive_float(common.get("temperature_K"), "common_conditions.temperature_K")

    cases_value = root.get("cases")
    if not isinstance(cases_value, list) or not cases_value:
        raise ValueError("cases must be a non-empty YAML list.")
    validation_cases_value = root.get("independent_validation_cases", [])
    if not isinstance(validation_cases_value, list):
        raise ValueError("independent_validation_cases must be a YAML list.")
    if not include_independent_validation:
        validation_cases_value = []
    training_case_count = len(cases_value)
    cases_value = [*cases_value, *validation_cases_value]


    cases: list[Mapping[str, Any]] = []
    case_ids: set[str] = set()
    curve_paths: set[Path] = set()
    for index, case_value in enumerate(cases_value):
        case = _mapping(case_value, f"cases[{index}]")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"cases[{index}].case_id must be non-empty.")
        if case_id in case_ids:
            raise ValueError(f"Duplicate COMSOL case_id: {case_id}")
        case_ids.add(case_id)
        if (
            index >= training_case_count
            and case.get("dataset_role") != "independent_validation"
        ):
            raise ValueError(
                f"Independent validation case '{case_id}' must declare "
                "dataset_role: independent_validation."
            )

        geometry = _mapping(case.get("geometry"), f"case '{case_id}'.geometry")
        _positive_float(geometry.get("length_m"), f"case '{case_id}'.geometry.length_m")
        _positive_float(
            geometry.get("oxide_thickness_m"),
            f"case '{case_id}'.geometry.oxide_thickness_m",
        )

        for curve_type in _CURVE_TYPES:
            curves = case.get(curve_type)
            if not isinstance(curves, list) or not curves:
                raise ValueError(f"case '{case_id}'.{curve_type} must be a non-empty list.")
            bias_key = "vds_V" if curve_type == "idvg" else "vgs_V"
            seen_biases: set[float] = set()
            for curve_index, curve_value in enumerate(curves):
                curve = dict(
                    _mapping(
                        curve_value,
                        f"case '{case_id}'.{curve_type}[{curve_index}]",
                    )
                )
                curves[curve_index] = curve
                try:
                    bias = float(curve[bias_key])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"case '{case_id}'.{curve_type}[{curve_index}].{bias_key} "
                        "must be numeric."
                    ) from error
                if bias in seen_biases:
                    raise ValueError(
                        f"case '{case_id}'.{curve_type} has duplicate {bias_key}={bias}."
                    )
                seen_biases.add(bias)

                status = curve.get("qc_status")
                if status not in _QC_STATUSES:
                    raise ValueError(
                        f"case '{case_id}'.{curve_type}[{curve_index}].qc_status "
                        f"must be one of {sorted(_QC_STATUSES)}."
                    )
                if status == "isolated" and not curve.get("qc_reason"):
                    raise ValueError("Isolated curves must provide qc_reason.")
                analysis_role = curve.get(
                    "analysis_role",
                    "formal" if status == "active" else "inventory_only",
                )
                if analysis_role not in _ANALYSIS_ROLES:
                    raise ValueError(
                        f"case '{case_id}'.{curve_type}[{curve_index}].analysis_role "
                        f"must be one of {sorted(_ANALYSIS_ROLES)}."
                    )
                if (status, analysis_role) not in _ALLOWED_CURVE_ROLES:
                    raise ValueError(
                        "Invalid qc_status/analysis_role combination for "
                        f"case '{case_id}'.{curve_type}[{curve_index}]: "
                        f"{status}/{analysis_role}."
                    )
                curve["analysis_role"] = analysis_role

                path_value = curve.get("path")
                if not isinstance(path_value, str) or not path_value.strip():
                    raise ValueError("Every COMSOL curve must provide a non-empty path.")
                relative_path = Path(path_value)
                if relative_path.is_absolute():
                    raise ValueError("COMSOL curve paths must be project-relative.")
                resolved_path = (resolved_project_root / relative_path).resolve()
                try:
                    resolved_path.relative_to(resolved_source_root)
                except ValueError as error:
                    raise ValueError(
                        f"COMSOL curve path must stay under source.root: {path_value}"
                    ) from error
                if resolved_path in curve_paths:
                    raise ValueError(f"Duplicate COMSOL curve path: {path_value}")
                curve_paths.add(resolved_path)
                if not resolved_path.is_file():
                    raise FileNotFoundError(f"COMSOL curve file not found: {resolved_path}")
        cases.append(case)

    nominal_case_id = root.get("nominal_case_id")
    training_cases = cases[:training_case_count]
    independent_validation_cases = cases[training_case_count:]
    if nominal_case_id not in {case["case_id"] for case in training_cases}:
        raise ValueError("nominal_case_id must identify one of the training cases.")

    return COMSOLCaseManifest(
        path=manifest_path,
        project_root=resolved_project_root,
        source_root=resolved_source_root,
        nominal_case_id=str(nominal_case_id),
        cases=tuple(training_cases),
        independent_validation_cases=tuple(independent_validation_cases),
        raw=root,
    )
