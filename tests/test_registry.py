from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pandas as pd
import pytest
import yaml

from examples.train_demo import write_training_inputs
from mosfet_platform.artifacts.frozen_model import load_frozen_model
from mosfet_platform.provenance import file_sha256
from mosfet_platform.workflows import run_evaluation, run_metric_evaluation, run_update


def update(root, **options):
    return run_update(project_config="project.yaml", spec="spec.yaml", database="store/catalog.sqlite3", root=root, **options)


def query(root, sql):
    with sqlite3.connect(root / "store/catalog.sqlite3") as connection:
        return connection.execute(sql).fetchall()


def test_actual_admission_training_and_idempotent_update(tmp_path):
    write_training_inputs(tmp_path)
    first = update(tmp_path)
    assert first.training_status == "TRAINED"
    assert first.counts == {"inserted": 5, "duplicates": 0, "rejected": 0, "stored": 5}
    assert query(tmp_path, "SELECT measured_status,count(*) FROM observations GROUP BY measured_status") == [("FAIL", 2), ("PASS", 3)]
    assert query(tmp_path, "SELECT count(*) FROM curves") == [(15,)]
    assert load_frozen_model(first.model_manifest, root=first.model_root).independent_validation_status == "NOT_RUN"
    html = first.report_path.read_text(encoding="utf-8")
    assert "FALSE_PASS" in html and "MEASUREMENT" in html
    second = update(tmp_path)
    assert second.training_status == "REUSED"
    assert second.counts["duplicates"] == 5
    assert second.model_manifest == first.model_manifest
    assert query(tmp_path, "SELECT count(*) FROM observations") == [(5,)]
    assert len(list((tmp_path / "store/models").iterdir())) == 1


def test_failed_training_retains_model_and_new_measurement_revision(tmp_path, monkeypatch):
    write_training_inputs(tmp_path)
    first = update(tmp_path)
    original_hash = file_sha256(first.model_manifest)
    for path in (tmp_path / "curves/training_4").glob("*.csv"):
        frame = pd.read_csv(path)
        frame["id"] *= 1.01
        frame.to_csv(path, index=False)
    project_path = tmp_path / "project.yaml"
    project = yaml.safe_load(project_path.read_text())
    project["model_generalization"]["cross_validation"] = "grouped"
    project_path.write_text(yaml.safe_dump(project))
    failed = update(tmp_path)
    assert failed.training_status == "FAILED"
    assert failed.counts["inserted"] == 1
    assert failed.model_manifest == first.model_manifest
    assert file_sha256(first.model_manifest) == original_hash
    assert query(tmp_path, "SELECT count(*) FROM observations") == [(6,)]
    assert len(list((tmp_path / "store/models").iterdir())) == 1
    assert "FAILED" in failed.report_path.read_text(encoding="utf-8")
    details = json.loads(query(tmp_path, "SELECT details_json FROM runs ORDER BY rowid DESC LIMIT 1")[0][0])
    assert "training_0" in details["training_error"]
    calls = []

    def forced_failure(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("explicit retry failure")

    monkeypatch.setattr("mosfet_platform.workflows.update.run_fit", forced_failure)
    repeated = update(tmp_path)
    assert repeated.training_status == "FAILED" and not calls
    forced = update(tmp_path, force_retrain=True)
    assert forced.training_status == "FAILED" and len(calls) == 1
    assert forced.model_manifest == first.model_manifest


def test_metric_only_import_does_not_train_and_rejects_predictions(tmp_path):
    write_training_inputs(tmp_path)
    measured = run_evaluation(cases="cases.yaml", contract="measurement_contract.yaml", spec="spec.yaml", output="measured", root=tmp_path)
    frame = pd.read_csv(measured.metrics_path)
    frame.loc[0, "result_origin"] = "predicted"
    frame.loc[1, "ss_status"] = "bad"
    path = tmp_path / "metrics.csv"
    frame.to_csv(path, index=False)
    exported = run_metric_evaluation(
        metrics=path, contract="measurement_contract.yaml", spec="spec.yaml",
        output="reexported", root=tmp_path,
    )
    result = update(tmp_path, metrics=exported.metrics_path)
    assert result.training_status == "NO_TRAINING_DATA"
    assert result.model_manifest is None
    assert result.counts["inserted"] == 3
    assert result.counts["rejected"] == 2
    assert query(tmp_path, "SELECT count(*) FROM curves") == [(0,)]
    assert query(tmp_path, "SELECT count(*) FROM models") == [(0,)]
    assert "PREDICTIONS_ARE_NOT_MEASUREMENTS" in result.report_path.read_text(encoding="utf-8")
    assert "库内实测结论" in result.report_path.read_text(encoding="utf-8")


def test_insufficient_geometry_is_stored_without_publishing_model(tmp_path):
    write_training_inputs(tmp_path, geometries=((0.8e-6, 8e-9), (1.2e-6, 12e-9)))
    result = update(tmp_path)
    assert result.counts["inserted"] == 2
    assert result.training_status == "FAILED"
    assert result.model_manifest is None
    assert query(tmp_path, "SELECT count(*) FROM observations") == [(2,)]
    assert query(tmp_path, "SELECT count(*) FROM models") == [(0,)]


def test_independent_records_never_enter_training_set(tmp_path):
    write_training_inputs(tmp_path, independent=True)
    result = update(tmp_path)
    assert result.training_status == "TRAINED"
    assert result.counts["inserted"] == 6
    manifest = json.loads(result.model_manifest.read_text())
    assert manifest["counts"]["cases"] == 5
    assert manifest["counts"]["independent_validation_cases"] == 1
    assert manifest["qualification_ready"] is True
    assert query(tmp_path, "SELECT count(*) FROM observations WHERE role='independent_validation'") == [(1,)]


def test_new_batch_retrains_on_previous_and_new_curves(tmp_path):
    write_training_inputs(tmp_path)
    path = tmp_path / "cases.yaml"
    all_cases = yaml.safe_load(path.read_text())
    first_batch = {**all_cases, "cases": all_cases["cases"][:4]}
    path.write_text(yaml.safe_dump(first_batch))
    first = update(tmp_path)
    assert first.training_status == "TRAINED"
    first_hash = file_sha256(first.model_manifest)
    second_batch = {**all_cases, "cases": all_cases["cases"][4:], "nominal_case_id": "training_4"}
    path.write_text(yaml.safe_dump(second_batch))
    second = update(tmp_path)
    assert second.training_status == "TRAINED"
    assert second.counts["inserted"] == 1
    assert second.counts["stored"] == 5
    assert second.model_manifest != first.model_manifest
    assert file_sha256(first.model_manifest) == first_hash
    assert json.loads(second.model_manifest.read_text())["counts"]["cases"] == 5


def test_metrics_and_spec_changes_reuse_curve_model(tmp_path):
    write_training_inputs(tmp_path)
    first = update(tmp_path)
    measured = run_evaluation(cases="cases.yaml", contract="measurement_contract.yaml", spec="spec.yaml", output="measured", root=tmp_path)
    frame = pd.read_csv(measured.metrics_path).iloc[:1].copy()
    frame["device_id"] = "external_metric"
    path = tmp_path / "metrics.csv"
    frame.to_csv(path, index=False)
    spec_path = tmp_path / "spec.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    spec["spec"]["ion_min"] = 1e-3
    spec_path.write_text(yaml.safe_dump(spec))
    second = update(tmp_path, metrics=path)
    assert second.training_status == "REUSED"
    assert second.counts["inserted"] == 1
    assert second.model_manifest == first.model_manifest
    assert query(tmp_path, "SELECT count(*) FROM curves") == [(15,)]
    assert "external_metric" in second.report_path.read_text(encoding="utf-8")


def test_invalid_curve_bundle_is_not_admitted(tmp_path):
    write_training_inputs(tmp_path)
    path = tmp_path / "cases.yaml"
    cases = yaml.safe_load(path.read_text())
    cases["cases"][0]["idvg"].pop()
    path.write_text(yaml.safe_dump(cases))
    result = update(tmp_path)
    assert result.counts["rejected"] == 1
    assert result.counts["inserted"] == 4
    assert query(tmp_path, "SELECT count(*) FROM observations WHERE device_id='training_0'") == [(0,)]


def test_copied_training_data_cannot_qualify_as_independent(tmp_path):
    write_training_inputs(tmp_path)
    path = tmp_path / "cases.yaml"
    cases = yaml.safe_load(path.read_text())
    validation = json.loads(json.dumps(cases["cases"][0]))
    validation.update(case_id="copied_validation", dataset_role="independent_validation")
    for kind in ("idvg", "idvd"):
        for curve in validation[kind]:
            original = tmp_path / curve["path"]
            destination = original.with_name("copy_" + original.name)
            destination.write_bytes(original.read_bytes())
            curve["path"] = destination.relative_to(tmp_path).as_posix()
    cases["independent_validation_cases"] = [validation]
    path.write_text(yaml.safe_dump(cases))
    project_path = tmp_path / "project.yaml"
    project = yaml.safe_load(project_path.read_text())
    project["model_generalization"]["independent_validation"] = "required"
    project_path.write_text(yaml.safe_dump(project))
    result = update(tmp_path)
    assert result.training_status == "FAILED"
    assert result.counts["duplicates"] == 1
    assert result.model_manifest is None
    assert query(tmp_path, "SELECT count(*) FROM observations WHERE role='independent_validation'") == [(0,)]
