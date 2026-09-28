"""Behavioral checks for missing-only model completion and evidence separation."""

from copy import deepcopy
import json

import pandas as pd
import pytest

from examples.train_demo import write_training_inputs
from mosfet_platform.api import diagnose
from mosfet_platform.workflows.fit import run_fit
from mosfet_platform.workflows.diagnose import run_diagnosis


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp("completion-model")
    config = write_training_inputs(root, independent=True)
    fitted = run_fit(config=config, root=root)
    request = {"root": str(root), "cases": "cases.yaml", "contract": "measurement_contract.yaml",
               "spec": "spec.yaml", "output": str(root / "reports")}
    baseline = run_diagnosis(**request)
    from mosfet_platform.case_manifest import load_comsol_case_manifest
    from mosfet_platform.io.diagnosis import curve_records
    from mosfet_platform.measurement import load_measurement_contract
    records = curve_records(load_comsol_case_manifest(root / "cases.yaml", project_root=root),
                            load_measurement_contract(root / "measurement_contract.yaml"), root)
    # Synthetic data exercise the pipeline, not evidence for real engineering accuracy.
    return root, str(fitted.manifest), request, records, baseline


def run_rows(fitted, tmp_path, rows, **settings):
    root, manifest, request, _, _ = fitted
    pd.DataFrame(rows).drop(columns=["metric_issues", "metric_evidence", "issues"], errors="ignore").to_csv(tmp_path / "input.csv", index=False)
    request = {**request, "metrics": str(tmp_path / "input.csv"), "output": str(tmp_path / "reports"),
               "completion": {"model_manifest": manifest, **settings}}
    request.pop("cases")
    return diagnose(request)


def test_new_id_missing_completion_preserves_raw_and_report(fitted, tmp_path):
    original = deepcopy(fitted[3][-1])
    original.update(device_id="new-device-not-in-training", ion=None)
    result = run_rows(fitted, tmp_path, [original])
    assert result["status"] == "SUCCEEDED", result
    data = result["data"]
    assert data["devices"][0]["status"] == "INCOMPLETE"
    assert data["combined_devices"][0]["status"] == "PREDICTED_PASS"
    ion = next(r for r in data["combined_metric_details"] if r["metric"] == "ion")
    assert ion["predicted"] and ion["original_value"] is None
    assert ion["value_source"] == "estimated" and ion["value"] > 0
    assert next(r for r in data["metric_summary"] if r["metric"] == "ion")["evaluated_count"] == 0
    assert original["ion"] is None
    assert result["provenance"]["completion_performed"]
    assert data["model_checks"][0]["consistency"] == "NOT_ASSESSED"
    from pathlib import Path
    assert "(predicted)" in Path(result["artifacts"]["report"]).read_text(encoding="utf-8")
    json.dumps(result, allow_nan=False)


def test_observed_failure_and_invalid_values_are_not_replaced(fitted, tmp_path):
    record = deepcopy(fitted[3][-1])
    record.update(ion=0, ioff=-1, ss_mv_dec=None)
    result = run_rows(fitted, tmp_path, [record])
    data = result["data"]
    assert data["combined_devices"][0]["status"] == "FAIL"
    metrics = {r["metric"]: r for r in data["combined_metric_details"]}
    assert metrics["ion"]["value"] == 0 and not metrics["ion"]["predicted"]
    assert metrics["ioff"]["status"] == "INVALID" and not metrics["ioff"]["predicted"]
    assert metrics["ss_mv_dec"]["predicted"]


def test_configured_mismatch_blocks_missing_completion(fitted, tmp_path):
    record = deepcopy(fitted[3][-1])
    record.update(ion=0, ss_mv_dec=None)
    result = run_rows(fitted, tmp_path, [record], absolute_tolerances={"ion": 0})
    data = result["data"]
    assert data["model_checks"][0]["consistency"] == "MISMATCH"
    assert not result["provenance"]["completion_performed"]
    assert data["combined_devices"][0]["status"] == "FAIL"
    assert "ss_mv_dec" in data["combined_devices"][0]["unavailable_metrics"]


@pytest.mark.parametrize("field,value,reason", [("length_m", 1e-3, "OUT_OF_ENVELOPE"),
                                                ("temperature_K", 350, "TEMPERATURE_K_MISMATCH"),
                                                ("width_m", 2e-5, "WIDTH_M_MISMATCH")])
def test_unsupported_conditions_keep_raw_diagnosis(fitted, tmp_path, field, value, reason):
    record = deepcopy(fitted[3][-1])
    record.update(ion=None)
    record[field] = value
    result = run_rows(fitted, tmp_path, [record])
    assert result["status"] == "SUCCEEDED"
    assert reason in result["data"]["model_checks"][0]["blockers"]
    assert not result["provenance"]["completion_performed"]


def test_missing_model_does_not_abort_raw_analysis(fitted, tmp_path):
    request = {**fitted[2], "completion": {"model_manifest": "absent.json"}}
    result = diagnose(request)
    assert result["status"] == "SUCCEEDED"
    assert result["data"]["devices"] == fitted[4]["data"]["devices"]
    assert not result["data"]["model_evidence"]["available"]


def test_complete_data_retained_and_sources_separated(fitted, tmp_path):
    first = deepcopy(fitted[3][-1])
    second = {**first, "source_type": "measured"}
    result = run_rows(fitted, tmp_path, [first, second])
    assert result["status"] == "SUCCEEDED"
    assert len(result["data"]["combined_devices"]) == 2
    assert not result["provenance"]["completion_performed"]
    assert all(not r["predicted"] for r in result["data"]["combined_metric_details"])


def test_predicted_failure_is_not_presented_as_observed_failure(fitted, tmp_path, monkeypatch):
    import mosfet_platform.analysis.completion as module
    extract = module.extract_bundle_metrics
    monkeypatch.setattr(module, "extract_bundle_metrics", lambda *args: {**extract(*args), "ion": 0.0})
    record = {**fitted[3][-1], "ion": None}
    result = run_rows(fitted, tmp_path, [record])
    assert result["data"]["devices"][0]["status"] == "INCOMPLETE"
    assert result["data"]["combined_devices"][0]["status"] == "PREDICTED_FAIL"


def test_unvalidated_model_is_reference_only(fitted, tmp_path, monkeypatch):
    from dataclasses import replace
    import mosfet_platform.analysis.completion as module
    load = module.load_frozen_model
    monkeypatch.setattr(module, "load_frozen_model", lambda *args, **kwargs: replace(load(*args, **kwargs), independent_validation_status="NOT_RUN"))
    result = run_rows(fitted, tmp_path, [{**fitted[3][-1], "ion": None}])
    assert not result["provenance"]["completion_performed"]
    assert "NO_INDEPENDENT_ENGINEERING_VALIDATION" in result["data"]["model_checks"][0]["blockers"]
    assert next(r for r in result["data"]["combined_metric_details"] if r["metric"] == "ion")["prediction"] is not None


def test_tolerance_can_confirm_only_the_compared_metrics(fitted, tmp_path):
    result = run_rows(fitted, tmp_path, [{**fitted[3][-1], "ion": None}], absolute_tolerances={"vth": 1.0})
    assert result["data"]["model_checks"][0]["consistency"] == "WITHIN_CONFIGURED_TOLERANCES"
    assert result["provenance"]["completion_performed"]
    assert next(r for r in result["data"]["combined_metric_details"] if r["metric"] == "ion")["comparison"] == "NOT_ASSESSED"


@pytest.mark.parametrize("completion", [{}, {"model_manifest": "x", "other": True},
                                        {"model_manifest": "x", "absolute_tolerances": {"ion": -1}},
                                        {"model_manifest": "x", "absolute_tolerances": {"ion": True}}])
def test_bad_completion_request_is_rejected(completion):
    result = diagnose({"contract": "unused", "spec": "unused", "metrics": "unused", "completion": completion})
    assert result["error"]["code"] == "INVALID_REQUEST"
