"""Real curve fitting and artifact tests for data-driven training."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from examples.demo import _model, _write_curve
from examples.train_demo import TRAINING_GEOMETRIES, write_training_inputs
from mosfet_platform.artifacts.frozen_model import load_frozen_model
from mosfet_platform.training.data import run_characterization
from mosfet_platform.training.validate import run_model_generalization
from mosfet_platform.workflows.fit import run_fit
from mosfet_platform.workflows.predict import run_prediction


def update_project(root, **settings):
    path = root / "project.yaml"
    project = yaml.safe_load(path.read_text(encoding="utf-8"))
    project["model_generalization"].update(settings)
    path.write_text(yaml.safe_dump(project), encoding="utf-8")


@pytest.mark.parametrize("case_count", [4, 5])
def test_fit_uses_input_geometry_and_counts(tmp_path, case_count):
    config = write_training_inputs(tmp_path, geometries=TRAINING_GEOMETRIES[:case_count])
    # Exercise a non-default scan size, independent of training case count.
    update_project(tmp_path, envelope_scan={"length_step_nm": 50, "tox_step_nm": 0.5})
    fitted = run_fit(config=config, root=tmp_path)
    manifest = json.loads(fitted.manifest.read_text(encoding="utf-8"))
    loaded = load_frozen_model(fitted.manifest, root=tmp_path)
    model_doc = yaml.safe_load(fitted.model.read_text(encoding="utf-8"))
    report = pd.read_csv(fitted.validation)

    assert manifest["counts"]["cases"] == case_count
    assert manifest["counts"]["formal_curves"] == case_count * 3
    assert manifest["counts"]["diagnostic_only_curves"] == 0
    assert manifest["counts"]["candidate_models"] == 1
    assert manifest["counts"]["selected_envelope_rows"] == {4: 56, 5: 81}[case_count]
    assert manifest["prediction_ready"] is True
    assert manifest["qualification_ready"] is False
    assert manifest["independent_validation_status"] == "NOT_RUN"
    assert model_doc["qualification"]["qualification_ready"] is False
    assert report["validation_scope"].eq("training_fit").all()
    assert report["validation_status"].eq("PASS").all()
    assert loaded.model.envelope.length_min_m == pytest.approx(0.8e-6)
    assert loaded.model.envelope.length_max_m == pytest.approx(TRAINING_GEOMETRIES[case_count - 1][0])
    assert model_doc["optimization"]["objective_after"] < model_doc["optimization"]["objective_before"]
    predicted = run_prediction(config="prediction.yaml", root=tmp_path)
    assert predicted.counts["devices"] == case_count
    assert predicted.counts["INVALID_INPUT"] == 0
    assert predicted.counts["OUT_OF_ENVELOPE"] == 0


def test_required_validation_uses_one_arbitrary_geometry(tmp_path):
    config = write_training_inputs(tmp_path, independent=True)
    fitted = run_fit(config=config, root=tmp_path)
    manifest = json.loads(fitted.manifest.read_text(encoding="utf-8"))
    report = pd.read_csv(fitted.validation)
    held_out = report.loc[report["validation_scope"] == "independent"]
    assert manifest["counts"]["independent_validation_cases"] == 1
    assert manifest["counts"]["independent_validation_formal_curves"] == 3
    assert manifest["qualification_ready"] is True
    assert manifest["independent_validation_status"] == "PASS"
    assert held_out["length_m"].iloc[0] == pytest.approx(1.05e-6)
    assert held_out["holdout_curve_count"].iloc[0] == 3
    assert "validation_0" not in held_out["training_case_ids"].iloc[0]
    assert load_frozen_model(fitted.manifest, root=tmp_path).independent_validation_status == "PASS"


def test_unequal_curve_counts_and_optional_diagnostic(tmp_path):
    config = write_training_inputs(tmp_path)
    manifest_path = tmp_path / "cases.yaml"
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    case = raw["cases"][0]
    length_m, tox_m = TRAINING_GEOMETRIES[0]
    for name, bias, role in (
        ("extra_output", 0.9, "formal"),
        ("diagnostic", 0.0, "diagnostic_only"),
    ):
        path = tmp_path / "curves/training_0" / f"{name}.csv"
        _write_curve(
            path, curve_type="idvd", fixed_bias=bias, model=_model(),
            length_m=length_m, tox_m=tox_m,
        )
        case["idvd"].append({
            "path": path.relative_to(tmp_path).as_posix(),
            "vgs_V": bias,
            "analysis_role": role,
            "qc_status": "active" if role == "formal" else "isolated",
            "qc_reason": "Test optional diagnostic coverage.",
        })
    manifest_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    fitted = run_fit(config=config, root=tmp_path)
    manifest = json.loads(fitted.manifest.read_text(encoding="utf-8"))
    model = yaml.safe_load(fitted.model.read_text(encoding="utf-8"))
    assert manifest["counts"]["curves"] == 17
    assert manifest["counts"]["formal_curves"] == 16
    assert manifest["counts"]["diagnostic_only_curves"] == 1
    assert model["optimization"]["curve_count"] == 16
    assert manifest["prediction_ready"] is True


def test_failed_required_validation_is_not_prediction_ready(tmp_path):
    write_training_inputs(tmp_path, independent=True)
    for path in (tmp_path / "curves/validation_0").glob("*.csv"):
        frame = pd.read_csv(path)
        frame["id"] *= 10
        frame.to_csv(path, index=False)
    config = tmp_path / "project.yaml"
    run_characterization(config, project_root=tmp_path)
    result = run_model_generalization(config, project_root=tmp_path)
    manifest = json.loads(result.paths.workflow_manifest.read_text(encoding="utf-8"))
    model = yaml.safe_load(result.paths.selected_model.read_text(encoding="utf-8"))
    assert result.status == "FAIL"
    assert result.prediction_ready is False
    assert manifest["independent_validation_status"] == "FAIL"
    assert manifest["qualification_ready"] is False
    assert model["qualification"]["prediction_ready"] is False
    assert model["qualification"]["qualification_ready"] is False
    with pytest.raises(ValueError):
        load_frozen_model(result.paths.workflow_manifest, root=tmp_path)


def test_required_validation_cannot_run_without_cases(tmp_path):
    config = write_training_inputs(tmp_path)
    update_project(tmp_path, independent_validation="required")
    with pytest.raises(ValueError, match="no registered cases"):
        run_fit(config=config, root=tmp_path)


def test_collinear_geometries_still_cannot_identify_model(tmp_path):
    config = write_training_inputs(tmp_path, geometries=((0.8e-6, 8e-9), (1e-6, 10e-9), (1.2e-6, 12e-9)))
    with pytest.raises(ValueError, match="do not identify"):
        run_fit(config=config, root=tmp_path)


def test_requested_holdout_requires_identifiable_training_folds(tmp_path):
    config = write_training_inputs(tmp_path, geometries=TRAINING_GEOMETRIES[:3])
    update_project(tmp_path, cross_validation="grouped")
    with pytest.raises(ValueError, match="loco fold 0"):
        run_fit(config=config, root=tmp_path)


def test_partial_scan_cannot_qualify_a_larger_envelope(tmp_path):
    config = write_training_inputs(tmp_path)
    update_project(tmp_path, envelope_scan={"length_start_nm": 1000})
    with pytest.raises(ValueError, match="full declared model envelope"):
        run_fit(config=config, root=tmp_path)


def test_missing_required_curve_is_not_hidden_by_variable_counts(tmp_path):
    config = write_training_inputs(tmp_path)
    path = tmp_path / "cases.yaml"
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest["cases"][0]["idvg"].pop()
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Characterization did not pass"):
        run_fit(config=config, root=tmp_path)


def test_disabled_policy_is_required_to_import_not_run_model(tmp_path):
    config = write_training_inputs(tmp_path)
    fitted = run_fit(config=config, root=tmp_path)
    raw = json.loads(fitted.manifest.read_text(encoding="utf-8"))
    raw["independent_validation_policy"] = "required"
    fitted.manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="independent validation"):
        load_frozen_model(fitted.manifest, root=tmp_path)


def test_automatic_model_family_selection(tmp_path):
    config = write_training_inputs(tmp_path)
    update_project(tmp_path, model_family="auto")
    fitted = run_fit(config=config, root=tmp_path)
    manifest = json.loads(fitted.manifest.read_text(encoding="utf-8"))
    assert manifest["counts"]["candidate_models"] == 2
    assert manifest["prediction_ready"] is True


def test_grouped_validation_records_and_rejects_bad_holdouts(tmp_path):
    write_training_inputs(tmp_path)
    update_project(tmp_path, cross_validation="grouped")
    config = tmp_path / "project.yaml"
    run_characterization(config, project_root=tmp_path)
    result = run_model_generalization(config, project_root=tmp_path)
    rows = pd.read_csv(result.paths.validation_summary)
    assert result.counts["loco_rows"] == 5
    assert result.counts["lg_row_holdout_rows"] == 5
    assert result.counts["tox_column_holdout_rows"] == 5
    held_out = rows.loc[rows["validation_scope"] == "training_grouped"]
    for row in held_out.itertuples(index=False):
        assert set(row.holdout_case_ids.split(";")).isdisjoint(row.training_case_ids.split(";"))
    # The edge case exceeds the SS gate. Variable group sizes must not bypass it.
    assert held_out["validation_status"].eq("FAIL").any()
    assert result.status == "FAIL"
    assert result.prediction_ready is False
    assert any(error["error_type"] == "ModelValidationError" for error in result.errors)
    with pytest.raises(ValueError):
        load_frozen_model(result.paths.workflow_manifest, root=tmp_path)


def test_disabled_independent_validation_does_not_read_its_files(tmp_path):
    config = write_training_inputs(tmp_path, independent=True)
    manifest_path = tmp_path / "cases.yaml"
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    raw["independent_validation_cases"][0]["idvg"][0]["path"] = "curves/not_available.csv"
    manifest_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    update_project(tmp_path, independent_validation="disabled")
    fitted = run_fit(config=config, root=tmp_path)
    assert load_frozen_model(fitted.manifest, root=tmp_path).independent_validation_status == "NOT_RUN"
    update_project(tmp_path, independent_validation="required")
    with pytest.raises(FileNotFoundError):
        run_fit(config=config, root=tmp_path)
