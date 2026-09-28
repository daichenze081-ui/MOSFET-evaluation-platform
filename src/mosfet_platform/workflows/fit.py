"""Validate and freeze the geometry-aware surrogate model."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Mapping

import yaml

from mosfet_platform.provenance import file_sha256, project_relative_path
from mosfet_platform.training.data import run_characterization
from mosfet_platform.training.validate import run_model_generalization


@dataclass(frozen=True)
class FitResult:
    model: Path
    manifest: Path
    validation: Path
    errors: Path
    family: str


def _resolve(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must contain a YAML mapping.")
    return dict(raw)


def _section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Config must contain a '{name}' mapping.")
    return dict(value)


def _staged_config(source: Path, target: Path, stage_root: Path) -> None:
    raw = _load_yaml(source, "Project config")
    characterization = dict(raw.get("characterization", {}))
    calibration = dict(raw.get("geometry_calibration", {}))
    generalization = dict(raw.get("model_generalization", {}))
    characterization["output_dir"] = str(stage_root / "characterization")
    calibration["characterization_input_dir"] = characterization["output_dir"]
    calibration["output_dir"] = str(stage_root / "calibration")
    generalization["output_dir"] = str(stage_root / "generalization")
    raw["characterization"] = characterization
    raw["geometry_calibration"] = calibration
    raw["model_generalization"] = generalization
    target.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _require_stage(name: str, status: str, ready: bool = True) -> None:
    if str(status).upper() == "FAIL" or not ready:
        raise RuntimeError(f"{name} did not pass.")


def _error_rows(stage: str, errors: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for error in errors or ():
        item = dict(error) if isinstance(error, Mapping) else {"error_message": error}
        rows.append(
            {
                "stage": stage,
                "scope": str(item.get("scope", "")),
                "case_id": str(item.get("case_id", "")),
                "curve_type": str(item.get("curve_type", "")),
                "input_path": str(item.get("input_path", "")),
                "error_type": str(item.get("error_type", "")),
                "error_message": str(item.get("error_message", "")),
            }
        )
    return rows


def _write_errors(path: Path, rows: list[dict[str, str]]) -> None:
    fields = (
        "stage",
        "scope",
        "case_id",
        "curve_type",
        "input_path",
        "error_type",
        "error_message",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(
    *,
    source: Mapping[str, Any],
    source_manifest: Path,
    model: Path,
    validation: Path,
    errors: Path,
    manifest: Path,
    training_config: Path,
    project_config: Path,
    root: Path,
) -> None:
    frozen = dict(source)
    frozen.pop("characterization_manifest", None)
    frozen["workflow"] = "fit"
    frozen["output_files"] = [
        {
            "path": project_relative_path(path, root),
            "sha256": file_sha256(path),
        }
        for path in (model, validation, errors)
    ]
    frozen["artifact"] = {
        "type": "frozen_surrogate_model",
        "source_workflow_manifest_sha256": file_sha256(source_manifest),
        "training_config": project_relative_path(training_config, root),
        "training_config_sha256": file_sha256(training_config),
        "project_config": project_relative_path(project_config, root),
        "project_config_sha256": file_sha256(project_config),
    }
    manifest.write_text(
        json.dumps(frozen, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_fit(
    *,
    config: str | Path,
    root: str | Path | None = None,
) -> FitResult:
    project_root = Path(root or Path.cwd()).resolve()
    config_path = _resolve(config, project_root)
    training = _section(_load_yaml(config_path, "Training config"), "training")
    project_value = training.get("project_config")
    if not isinstance(project_value, str) or not project_value.strip():
        raise ValueError("training.project_config is required.")
    project_config = _resolve(project_value, project_root)
    output = _resolve(
        str(training.get("output_dir", "outputs/frozen_models/default")),
        project_root,
    )
    model = output / "model" / "selected_geometry_aware_model.yaml"
    manifest = output / "workflow_manifest.json"
    validation = output / "validation_summary.csv"
    errors = output / "errors.csv"
    model.parent.mkdir(parents=True, exist_ok=True)
    for path in (model, manifest, validation, errors):
        if path.is_file():
            path.unlink()

    stage = "setup"
    collected: list[dict[str, str]] = []
    try:
        with TemporaryDirectory(prefix="mosfet_fit_") as temp:
            stage_root = Path(temp)
            staged_config = stage_root / "project.yaml"
            _staged_config(project_config, staged_config, stage_root)

            stage = "characterization"
            characterized = run_characterization(
                config_path=staged_config,
                project_root=project_root,
            )
            collected.extend(_error_rows(stage, getattr(characterized, "errors", ())))
            _require_stage("Characterization", characterized.status)

            stage = "validation"
            generalized = run_model_generalization(
                config_path=staged_config,
                project_root=project_root,
            )
            collected.extend(_error_rows(stage, getattr(generalized, "errors", ())))
            _require_stage(
                "Model validation",
                generalized.status,
                generalized.prediction_ready,
            )

            source_manifest = generalized.paths.workflow_manifest
            source_raw = json.loads(source_manifest.read_text(encoding="utf-8"))
            if not isinstance(source_raw, Mapping):
                raise ValueError("Validation manifest must contain a JSON object.")

            shutil.copy2(generalized.paths.selected_model, model)
            shutil.copy2(generalized.paths.validation_summary, validation)
            _write_errors(errors, collected)
            _write_manifest(
                source=source_raw,
                source_manifest=source_manifest,
                model=model,
                validation=validation,
                errors=errors,
                manifest=manifest,
                training_config=config_path,
                project_config=project_config,
                root=project_root,
            )
            family = str(generalized.selected_family)
    except Exception as error:
        collected.append(
            {
                "stage": stage,
                "scope": "run_fit",
                "case_id": "",
                "curve_type": "",
                "input_path": "",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        output.mkdir(parents=True, exist_ok=True)
        _write_errors(errors, collected)
        raise

    return FitResult(
        model=model,
        manifest=manifest,
        validation=validation,
        errors=errors,
        family=family,
    )
