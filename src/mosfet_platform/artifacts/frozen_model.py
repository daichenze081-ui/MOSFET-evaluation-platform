from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from mosfet_platform.provenance import file_sha256
from mosfet_platform.model.geometry_aware import GeometryAwareModelParameters

@dataclass(frozen=True)
class FrozenModel:
    manifest_path: Path
    model_path: Path
    model_sha256: str
    run_id: str
    family: str
    status: str
    validation_status: str
    independent_validation_status: str
    model: GeometryAwareModelParameters

    @property
    def selected_family(self) -> str:
        return self.family

    @property
    def model_generalization_status(self) -> str:
        return self.status

def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()

def _mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must contain a mapping.")
    return raw

def load_frozen_model(
    manifest: str | Path,
    *,
    root: str | Path = ".",
) -> FrozenModel:
    project = Path(root).resolve()
    manifest_path = _resolve(project, manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen-model manifest not found: {manifest_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Frozen-model manifest must contain an object.")
    if raw.get("workflow") not in {"fit", "model_generalization"}:
        raise ValueError("Artifact must be a fit or model-generalization manifest.")
    if raw.get("formal") is not True:
        raise ValueError("Frozen-model manifest requires formal=true.")
    if raw.get("workflow_class") != "global_geometry_calibration":
        raise ValueError("Frozen-model workflow_class is unsupported.")
    if raw.get("prediction_ready", raw.get("qualification_ready")) is not True:
        raise ValueError("Frozen-model prediction_ready must be true.")
    if str(raw.get("validation_status", "")).upper() != "PASS":
        raise ValueError("Frozen-model validation did not pass.")
    if str(raw.get("status", "")).upper() not in {"PASS", "WARNING"}:
        raise ValueError("Frozen-model workflow did not pass.")
    independent_status = str(raw.get("independent_validation_status", "")).upper()
    independent_policy = raw.get("independent_validation_policy", "required")
    if independent_policy not in {"disabled", "required"}:
        raise ValueError("Frozen-model independent validation policy is unsupported.")
    # SYNTHETIC is the existing demonstration status, not engineering validation.
    if independent_status not in {"PASS", "SYNTHETIC"} and not (
        independent_status == "NOT_RUN" and independent_policy == "disabled"
        and raw.get("prediction_ready") is True and raw.get("qualification_ready") is False
    ):
        raise ValueError("Frozen-model independent validation did not pass.")

    family = str(raw.get("selected_family", "")).lower()
    if family not in {"additive", "interaction"}:
        raise ValueError("Frozen-model family is unsupported.")
    outputs = raw.get("output_files")
    if not isinstance(outputs, list):
        raise ValueError("Frozen-model manifest has no output inventory.")
    models = [
        item
        for item in outputs
        if isinstance(item, Mapping)
        and str(item.get("path", "")).endswith("selected_geometry_aware_model.yaml")
    ]
    if len(models) != 1:
        raise ValueError("Frozen-model manifest must declare one selected model.")
    entry = models[0]
    expected = str(entry.get("sha256", "")).lower()
    if len(expected) != 64:
        raise ValueError("Frozen-model SHA256 is missing or malformed.")
    model_path = _resolve(project, str(entry["path"]))
    actual = file_sha256(model_path)
    if actual != expected:
        raise ValueError(f"Frozen-model SHA256 mismatch: expected {expected}, actual {actual}.")

    model_raw = _mapping(model_path, "Frozen model")
    section = model_raw.get("model")
    qualification = model_raw.get("qualification")
    if not isinstance(section, Mapping) or not isinstance(qualification, Mapping):
        raise ValueError("Frozen-model metadata is incomplete.")
    if section.get("type") != "geometry_aware_enhanced":
        raise ValueError("Frozen-model type is unsupported.")
    if str(section.get("family", "")).lower() != family:
        raise ValueError("Frozen-model family does not match its manifest.")
    params = section.get("parameters")
    if not isinstance(params, Mapping):
        raise ValueError("Frozen-model parameters are missing.")
    if (
        qualification.get("prediction_ready", qualification.get("qualification_ready")) is not True
        or qualification.get("qualification_ready") != raw.get("qualification_ready")
        or str(qualification.get("validation_status", "")).upper() != "PASS"
        or str(qualification.get("envelope_status", "")).upper() != "PASS"
        or str(qualification.get("independent_validation_status", "")).upper()
        != independent_status
        or qualification.get("independent_validation_policy", "required") != independent_policy
    ):
        raise ValueError("Frozen-model qualification gates did not pass.")
    model = GeometryAwareModelParameters.from_mapping(params)
    if model.geometry_family != family:
        raise ValueError("Frozen-model parameter family is inconsistent.")
    return FrozenModel(
        manifest_path=manifest_path,
        model_path=model_path,
        model_sha256=actual,
        run_id=str(raw.get("run_id", "")),
        family=family,
        status=str(raw.get("status", "UNKNOWN")).upper(),
        validation_status=str(raw.get("validation_status", "")).upper(),
        independent_validation_status=independent_status,
        model=model,
    )
