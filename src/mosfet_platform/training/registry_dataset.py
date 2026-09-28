"""Materialize a reproducible training dataset from measured registry records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from mosfet_platform.artifacts.registry import content_hash


def training_fingerprint(records, project, base_bytes, contract_hash):
    settings = {
        section: {key: value for key, value in project.get(section, {}).items()
                  if not key.endswith("output_dir") and key != "characterization_input_dir"}
        for section in ("characterization", "extraction", "geometry_calibration", "model_generalization")
    }
    return content_hash({
        "records": [(row["device_id"], row["payload_hash"]) for row in records],
        "settings": settings, "base_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "contract_hash": contract_hash,
    })


def write_dataset(connection, records, directory: Path, project, contract_bytes, base_bytes):
    directory.mkdir(parents=True)
    training, independent = [], []
    for row in records:
        case = {**json.loads(row["case_json"]), "case_id": row["device_id"], "idvg": [], "idvd": []}
        curves = connection.execute(
            "SELECT * FROM curves WHERE observation_id=? ORDER BY curve_type,sha256", (row["id"],),
        ).fetchall()
        for index, curve in enumerate(curves):
            data = bytes(curve["data"])
            if hashlib.sha256(data).hexdigest() != curve["sha256"]:
                raise ValueError(f"Stored curve checksum failed for {row['device_id']}.")
            path = Path("curves") / str(row["id"]) / f"{index}.csv"
            (directory / path).parent.mkdir(parents=True, exist_ok=True)
            (directory / path).write_bytes(data)
            case[curve["curve_type"]].append({**json.loads(curve["metadata_json"]), "path": path.as_posix()})
        if row["role"] == "independent_validation":
            case["dataset_role"] = "independent_validation"
            independent.append(case)
        else:
            training.append(case)
    if not training:
        raise ValueError("Registry has no complete training curves.")
    first_metrics = json.loads(next(row["metrics_json"] for row in records if row["role"] == "training"))
    manifest = {
        "schema_version": 1,
        "source": {"type": "registry", "root": "curves"},
        "common_conditions": {key: first_metrics[key] for key in ("width_m", "temperature_K", "device_type")},
        "nominal_case_id": training[0]["case_id"],
        "cases": training, "independent_validation_cases": independent,
    }
    (directory / "cases.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    (directory / "measurement_contract.yaml").write_bytes(contract_bytes)
    (directory / "base_model.yaml").write_bytes(base_bytes)
    config = dict(project)
    config["comsol"] = {"case_manifest": "cases.yaml", "nominal_case_id": training[0]["case_id"]}
    config["measurement_contract"] = "measurement_contract.yaml"
    config["model"] = {"base_config": "base_model.yaml"}
    (directory / "project.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (directory / "training.yaml").write_text(yaml.safe_dump({"training": {
        "project_config": "project.yaml", "output_dir": "frozen",
    }}), encoding="utf-8")
    return directory / "training.yaml"
