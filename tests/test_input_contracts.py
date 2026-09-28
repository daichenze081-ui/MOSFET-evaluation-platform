from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from examples.demo import _spec, _write_inputs, _write_yaml
from mosfet_platform.analysis.spec import judge_single_device
from mosfet_platform.artifacts.frozen_model import load_frozen_model
from mosfet_platform.io.csv_loader import (
    EmptyIVCSVError,
    IVCSVError,
    MalformedIVCSVError,
    load_iv_csv,
)
from mosfet_platform.provenance import file_sha256
from mosfet_platform.workflows.fit import _write_manifest
from mosfet_platform.workflows.predict import run_prediction


@pytest.fixture
def model_inputs(tmp_path):
    _write_inputs(tmp_path)
    return tmp_path


def test_fit_manifest_can_be_imported_for_prediction(model_inputs):
    root = model_inputs
    source = root / "synthetic_model/workflow_manifest.json"
    manifest = root / "fit_manifest.json"
    validation = root / "validation.csv"
    errors = root / "errors.csv"
    validation.write_text("validation_status\nPASS\n", encoding="utf-8")
    errors.write_text("error_message\n", encoding="utf-8")
    _write_yaml(root / "training.yaml", {"training": {"project_config": "project.yaml"}})
    _write_yaml(root / "project.yaml", {"measurement_contract": "measurement_contract.yaml"})
    _write_manifest(
        source=json.loads(source.read_text(encoding="utf-8")),
        source_manifest=source,
        model=root / "synthetic_model/selected_geometry_aware_model.yaml",
        validation=validation,
        errors=errors,
        manifest=manifest,
        training_config=root / "training.yaml",
        project_config=root / "project.yaml",
        root=root,
    )
    _write_yaml(
        root / "prediction.yaml",
        {"prediction": {
            "model_manifest": "fit_manifest.json",
            "measurement_contract": "measurement_contract.yaml",
            "spec": "spec.yaml",
            "input": "prediction_devices.csv",
        }},
    )

    result = run_prediction(config="prediction.yaml", root=root)

    assert result.counts["PREDICTED_PASS"] == 1
    assert result.counts["PREDICTED_FAIL"] == 2
    # Accepting fit manifests must retain integrity checks.
    model = root / "synthetic_model/selected_geometry_aware_model.yaml"
    model.write_text(model.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run_prediction(config="prediction.yaml", root=root)


@pytest.mark.parametrize(
    ("field", "value"),
    [("status", "FAIL"), ("independent_validation_status", "FAIL"),
     ("independent_validation_status", "NOT_RUN"), ("workflow", "evaluate")],
)
def test_prediction_rejects_failed_or_unvalidated_manifest(model_inputs, field, value):
    manifest = model_inputs / "synthetic_model/workflow_manifest.json"
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw[field] = value
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError):
        run_prediction(config="prediction.yaml", root=model_inputs)


def test_model_qualification_must_match_manifest(model_inputs):
    model = model_inputs / "synthetic_model/selected_geometry_aware_model.yaml"
    raw_model = yaml.safe_load(model.read_text(encoding="utf-8"))
    raw_model["qualification"]["independent_validation_status"] = "FAIL"
    _write_yaml(model, raw_model)
    manifest = model_inputs / "synthetic_model/workflow_manifest.json"
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["output_files"][0]["sha256"] = file_sha256(model)
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="qualification"):
        load_frozen_model(manifest, root=model_inputs)


@pytest.fixture
def valid_metrics():
    return {
        "ion": 1e-3, "ioff": 1e-10, "vth": 0.4, "ss_mv_dec": 70.0,
        "gm_max": 1e-3, "ion_ioff_cross_bias": 1e7, "ion_ioff": 1e7,
        "vth_status": "ok", "ss_status": "ok", "width_m": 1e-5,
        "length_m": 1e-6, "formal_eligible": True,
        "condition_id": _spec()["spec_metadata"]["condition_id"],
        "condition_id_source": "generated",
    }


@pytest.mark.parametrize("metric", ["ion_ioff", "ion_ioff_cross_bias"])
@pytest.mark.parametrize(
    ("value", "reason"),
    [(None, "missing_metric"), ("bad", "non_numeric_metric"),
     (np.nan, "non_finite_metric"), (np.inf, "non_finite_metric")],
)
def test_all_configured_spec_metrics_are_required(valid_metrics, metric, value, reason):
    spec = _spec()
    spec["spec"]["ion_ioff_min"] = 1e6
    valid_metrics[metric] = value

    status, detail = judge_single_device(valid_metrics, spec)

    assert status == "INVALID"
    assert f"{reason}:{metric}" in detail


def test_missing_second_spec_ratio_is_not_silently_skipped(valid_metrics):
    spec = _spec()
    spec["spec"]["ion_ioff_min"] = 1e6
    del valid_metrics["ion_ioff"]

    assert judge_single_device(valid_metrics, spec) == ("INVALID", "missing_metric:ion_ioff")
    # A metric is optional only when its rule is absent.
    del spec["spec"]["ion_ioff_min"]
    assert judge_single_device(valid_metrics, spec) == ("PASS", "")


def test_prediction_evaluates_same_bias_spec_ratio(model_inputs):
    spec = _spec()
    spec["spec"]["ion_ioff_min"] = 1e30
    _write_yaml(model_inputs / "spec.yaml", spec)

    result = run_prediction(config="prediction.yaml", root=model_inputs)

    assert result.counts["INVALID_INPUT"] == 0
    assert result.counts["PREDICTED_FAIL"] == 3
    assert result.results["fail_reason"].str.contains("ion_ioff_low").all()


@pytest.mark.parametrize("metric,reason", [
    ("ion_ioff", "ion_ioff_low"),
    ("ion_ioff_cross_bias", "ion_ioff_cross_bias_low"),
])
def test_both_spec_ratio_limits_are_judged(valid_metrics, metric, reason):
    spec = _spec()
    spec["spec"]["ion_ioff_min"] = 1e6
    assert judge_single_device(valid_metrics, spec) == ("PASS", "")
    valid_metrics[metric] = 1.0
    assert judge_single_device(valid_metrics, spec) == ("FAIL", reason)


@pytest.mark.parametrize("header", [
    "vgs,vds,id",
    '" Vg (V) ","Vd(V)","drain current (A)"',
    '% "Vgs (V)","Vds (V)","abs(semi.I0_1) (A)"',
    "% Vgs (V),Vds (V),abs(semi.I0_1) (A)",
    "% Vgs (V),Vds (V),abs(semi.I0_1)",
    "gate voltage (V),drain voltage (V),终端电流 (A)",
])
@pytest.mark.parametrize("preamble", ["", "\ufeff% Model: synthetic\n% Version: synthetic\n\n"])
def test_csv_header_formats_preserve_scientific_values(tmp_path, header, preamble):
    path = tmp_path / "curve.csv"
    path.write_text(preamble + header + "\n0,1.0e-1,1.2e-12\n1.2,1.0e-1,2.5e-4\n", encoding="utf-8")

    frame = load_iv_csv(path, include_current_metadata=True)

    np.testing.assert_allclose(frame["vgs"], [0.0, 1.2], rtol=1e-12, atol=0)
    np.testing.assert_allclose(frame["vds"], [0.1, 0.1], rtol=1e-12, atol=0)
    np.testing.assert_allclose(frame["id"], [1.2e-12, 2.5e-4], rtol=1e-12, atol=0)
    pd.testing.assert_series_equal(frame["id"], frame["id_raw"], check_names=False)
    assert frame["current_sign_available"].all() == ("abs(" not in header)


@pytest.mark.parametrize("header", ["vgs (mV),vds,id", "vgs,vds (mV),id", "vgs,vds,id (mA)"])
def test_csv_does_not_silently_discard_a_unit_scale(tmp_path, header):
    path = tmp_path / "curve.csv"
    path.write_text(header + "\n1,1,1\n", encoding="utf-8")

    with pytest.raises(IVCSVError, match="unit"):
        load_iv_csv(path, required_columns=("vgs", "id"), optional_columns=("vds",))


@pytest.mark.parametrize("content", ["", "% Model: synthetic\n", "vgs,vds,id\n"])
def test_empty_csv_is_reported(tmp_path, content):
    path = tmp_path / "curve.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(EmptyIVCSVError):
        load_iv_csv(path)


def test_csv_metadata_does_not_hide_malformed_data(tmp_path):
    path = tmp_path / "curve.csv"
    path.write_text("% Model: synthetic\n% vgs,vds,id\n0,0.1,1e-12\n1,0.1,1e-4,extra\n", encoding="utf-8")
    with pytest.raises(MalformedIVCSVError):
        load_iv_csv(path)
