"""Local diagnostic workflow with durable JSON results and no training side effects."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from importlib.metadata import version
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

import pandas as pd
import yaml

from mosfet_platform.analysis.diagnosis import diagnose_records
from mosfet_platform.analysis.completion import complete_diagnosis
from mosfet_platform.analysis.diagnosis_report import render_diagnosis
from mosfet_platform.case_manifest import load_comsol_case_manifest
from mosfet_platform.io.diagnosis import curve_records, metric_records
from mosfet_platform.measurement import load_measurement_contract
from mosfet_platform.provenance import file_sha256
from mosfet_platform.workflows.evaluate import _check_contract, _check_spec_contract


def run_diagnosis(
    *, contract, spec, cases=None, metrics=None, root=None, output=None,
    source_type=None, device_id=None, group_by=("length_m", "oxide_thickness_m"),
    completion=None,
) -> dict:
    """Return a strict-JSON diagnostic result; exceptions represent execution failure."""
    if (cases is None) == (metrics is None):
        raise ValueError("Provide exactly one of cases or metrics.")
    project = Path(root or Path.cwd()).resolve()

    def resolve(path):
        value = Path(path)
        return value.resolve() if value.is_absolute() else (project / value).resolve()

    contract_path, spec_path = resolve(contract), resolve(spec)
    source_path = resolve(cases if cases is not None else metrics)
    hashes = {p: file_sha256(p) for p in (contract_path, spec_path, source_path)}
    contract_doc = load_measurement_contract(contract_path)
    spec_doc = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec_doc, dict):
        raise ValueError("Spec must contain a YAML mapping.")
    _check_spec_contract(contract_doc, spec_doc)
    if cases is not None:
        manifest = load_comsol_case_manifest(source_path, project_root=project)
        _check_contract(manifest, contract_doc, spec_doc)
        for case in (*manifest.cases, *manifest.independent_validation_cases):
            for kind in ("idvg", "idvd"):
                for curve in case[kind]:
                    path = resolve(curve["path"])
                    hashes[path] = file_sha256(path)
        records = curve_records(manifest, contract_doc, project, source_type=source_type)
        source_metadata = {"source": manifest.raw["source"], "common_conditions": manifest.raw["common_conditions"]}
    else:
        records = metric_records(source_path, contract_doc, source_type=source_type)
        source_metadata = {"input_format": "metric_table"}
    if device_id is not None:
        records = [row for row in records if row["device_id"] == device_id]
        if not records:
            raise ValueError(f"Device not found: {device_id}.")
    data = diagnose_records(records, spec_doc, group_by=group_by)
    completion_performed = False
    if completion is not None:
        completion_performed = complete_diagnosis(records, spec_doc, contract_doc, data, completion, project, hashes)
    if any(file_sha256(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("Input changed during diagnosis; no result was published.")

    run_id = uuid4().hex
    parent = Path(output).resolve() if output is not None else Path.cwd().resolve() / "outputs" / "diagnosis"
    directory = parent / run_id
    artifacts = {"result": str(directory / "result.json"), "report": str(directory / "result.html")}
    tables = ("devices", "metric_details", "rule_details", "metric_summary", "source_summaries", "cooccurrence", "groups", "gaps")
    if completion is not None:
        tables += ("combined_devices", "combined_metric_details", "model_checks")
    artifacts.update({name: str(directory / f"{name}.csv") for name in tables})
    if set(map(Path, artifacts.values())) & set(hashes):
        raise ValueError("Diagnostic outputs must not overwrite inputs.")
    package = Path(__file__).resolve().parents[1]
    result = {
        "schema_version": "1.1" if completion is not None else "1.0", "run_id": run_id, "status": "SUCCEEDED", "data": data,
        "provenance": {
            "created_at": datetime.now(timezone.utc).isoformat(), "algorithm": "diagnosis-v1",
            "environment": {"python": sys.version.split()[0],
                            **{name: version(name) for name in ("numpy", "pandas", "scipy", "pyyaml")}},
            "input_files": [{"path": str(p), "sha256": digest} for p, digest in sorted(hashes.items())],
            "code_files": [{"path": p.relative_to(package).as_posix(), "sha256": file_sha256(p)}
                           for p in sorted(package.rglob("*.py"))],
            "spec": data["spec"], "measurement_contract": asdict(contract_doc),
            "source_metadata": source_metadata,
            "selection": {"device_id": device_id, "group_by": list(group_by), "source_type": source_type},
            "completion_requested": completion is not None,
            "completion_performed": completion_performed,
        },
        "artifacts": artifacts,
    }
    # Validate serialization before creating outputs; absent values must be null, never NaN.
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".diagnosis-", dir=parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        (stage / "result.json").write_text(serialized, encoding="utf-8")
        (stage / "result.html").write_text(render_diagnosis(result), encoding="utf-8")
        for name in tables:
            frame = pd.DataFrame(data[name])
            if frame.empty:
                columns = (list(data["metric_details"][0]) if name == "gaps" else
                           ["source_type", "condition_id", "left_metric", "right_metric", "evaluated_count", "both_fail_count", "rate", "device_ids"]
                           if name == "cooccurrence" else
                           ["group_by", "group_value", *data["metric_summary"][0]])
                frame = pd.DataFrame(columns=columns)
            for column in frame:
                frame[column] = frame[column].map(lambda v: json.dumps(v, ensure_ascii=False, allow_nan=False)
                                                  if isinstance(v, (list, dict)) else v)
            frame.to_csv(stage / f"{name}.csv", index=False, encoding="utf-8-sig")
        stage.rename(directory)
    return result
