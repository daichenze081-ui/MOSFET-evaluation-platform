from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from examples.demo import _contract, _spec, _write_yaml
from examples.train_demo import write_training_inputs
from mosfet_platform.analysis.diagnosis import diagnose_records
from mosfet_platform.api import diagnose
from mosfet_platform.provenance import file_sha256
from mosfet_platform.workflows import run_diagnosis, run_evaluation


def spec(limits=None):
    value = _spec()
    value["spec"] = limits or {"ion_min": 1.0, "ioff_max": 2.0, "vth_min": 0.2, "vth_max": 0.6}
    return value


def row(device, **updates):
    return {"device_id": device, "condition_id": _spec()["spec_metadata"]["condition_id"],
            "device_type": "nmos", "source_type": "measured", "ion": 2.0, "ioff": 1.0,
            "vth": 0.4, "vth_status": "ok", "width_m": 1e-5, "length_m": 1e-6,
            "oxide_thickness_m": 1e-8, "temperature_K": 300.0, **updates}


def metric_request(root, rows, limits=None):
    _write_yaml(root / "spec.yaml", spec(limits))
    _write_yaml(root / "contract.yaml", _contract())
    pd.DataFrame(rows).to_csv(root / "metrics.csv", index=False)
    return {"root": str(root), "metrics": "metrics.csv", "contract": "contract.yaml",
            "spec": "spec.yaml", "output": str(root / "reports")}


def test_partial_failure_keeps_evidence_and_uses_metric_denominators():
    data = diagnose_records([
        row("pass"), row("both", ion=0.5, ioff=3), row("partial", ion=0.5, ioff=None),
        row("missing", ion=None, ioff=None, vth=None),
    ], spec())
    devices = {r["device_id"]: r for r in data["devices"]}
    assert devices["partial"]["status"] == "FAIL"
    assert devices["partial"]["complete"] is False
    assert devices["missing"]["status"] == "INCOMPLETE"
    assert devices["missing"]["failed_metrics"] == []
    summary = {r["metric"]: r for r in data["metric_summary"]}
    assert summary["ion"]["fail_count"] == 2
    assert summary["ion"]["evaluated_count"] == 3
    assert summary["ion"]["exceedance_rate"] == pytest.approx(2 / 3)
    assert summary["ion"]["failure_coverage"] == 1
    assert summary["ion"]["exclusive_fail_count"] == 0
    assert summary["ioff"]["failed_sample_evaluated_count"] == 1
    pair = next(r for r in data["cooccurrence"] if {r["left_metric"], r["right_metric"]} == {"ion", "ioff"})
    assert pair["evaluated_count"] == 2 and pair["both_fail_count"] == 1
    assert pair["rate"] == 0.5


def test_two_sided_vth_is_one_metric_and_exclusive_failure_is_exact():
    data = diagnose_records([row("low", vth=0.1), row("high", vth=0.7)], spec())
    summary = next(r for r in data["metric_summary"] if r["metric"] == "vth")
    assert summary["evaluated_count"] == 2
    assert summary["fail_count"] == summary["exclusive_fail_count"] == 2
    assert len([r for r in data["rule_details"] if r["metric"] == "vth"]) == 4


def test_spec_only_requires_enabled_metrics_and_zero_denominator_is_null():
    data = diagnose_records([row("missing", ion=None)], spec({"ion_min": 1.0}))
    assert len(data["metric_details"]) == 1
    assert data["metric_summary"][0]["exceedance_rate"] is None
    assert data["metric_summary"][0]["failure_coverage"] is None
    result = diagnose_records([row("ok", vth=None, vth_status=None)], spec({"ion_min": 1.0}))
    assert result["devices"][0]["status"] == "PASS"
    json.dumps(data, allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "bad", -1.0])
def test_bad_metric_does_not_hide_other_valid_metric(value):
    data = diagnose_records([row("bad", ion=0.5, ioff=value)], spec())
    by_metric = {r["metric"]: r for r in data["metric_details"]}
    assert by_metric["ion"]["status"] == "FAIL"
    assert by_metric["ioff"]["status"] == "INVALID"
    assert data["devices"][0]["status"] == "FAIL"
    assert not data["devices"][0]["complete"]
    json.dumps(data, allow_nan=False)


def test_sources_and_groups_are_never_pooled():
    data = diagnose_records([row("same", ion=0.5), row("same", source_type="comsol", length_m=2e-6)], spec())
    assert data["device_count"] == 1 and data["observation_count"] == 2
    measured = next(r for r in data["metric_summary"] if r["source_type"] == "measured" and r["metric"] == "ion")
    simulated = next(r for r in data["metric_summary"] if r["source_type"] == "comsol" and r["metric"] == "ion")
    assert measured["exceedance_rate"] == 1 and simulated["exceedance_rate"] == 0
    assert all(r["sample_count"] == 1 for r in data["groups"])
    with pytest.raises(ValueError, match="Select one"):
        diagnose_records([row("same"), row("same")], spec())


def test_condition_mismatch_never_produces_valid_spec_decision():
    data = diagnose_records([row("wrong", condition_id="different")], spec())
    assert all(r["status"] == "INVALID" for r in data["metric_details"])
    assert data["source_summaries"][0]["INCOMPLETE"] == 1


def test_local_api_keeps_inputs_and_history_and_escapes_html(tmp_path):
    request = metric_request(tmp_path, [row('<script>alert("x")</script>', ion=0.5, ioff=None)])
    original_hash = file_sha256(tmp_path / "metrics.csv")
    first = diagnose(request)
    assert first["status"] == "SUCCEEDED"
    assert first["data"]["devices"][0]["status"] == "FAIL"
    result_path = Path(first["artifacts"]["result"])
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved == first
    html = Path(first["artifacts"]["report"]).read_text(encoding="utf-8")
    assert "<script>" not in html and "&lt;script&gt;" in html
    before = file_sha256(result_path)
    second = diagnose(request)
    assert second["run_id"] != first["run_id"]
    assert second["data"] == first["data"]
    assert file_sha256(result_path) == before
    assert file_sha256(tmp_path / "metrics.csv") == original_hash
    assert first["provenance"]["completion_performed"] is False


def test_source_required_and_cannot_relabel_prediction(tmp_path):
    request = metric_request(tmp_path, [row("d")])
    data = pd.read_csv(tmp_path / "metrics.csv").drop(columns="source_type")
    data.to_csv(tmp_path / "metrics.csv", index=False)
    assert diagnose(request)["status"] == "FAILED"
    assert diagnose({**request, "source_type": "measured"})["status"] == "SUCCEEDED"
    data["result_origin"] = "predicted"
    data.to_csv(tmp_path / "metrics.csv", index=False)
    failed = diagnose({**request, "source_type": "measured"})
    assert failed["status"] == "FAILED"
    assert "Conflicting source" in failed["error"]["message"]


def test_missing_unused_gm_is_allowed_by_local_service(tmp_path):
    request = metric_request(tmp_path, [row("001", vth_status="bad")], {"ion_min": 1.0})
    result = diagnose(request)
    assert result["status"] == "SUCCEEDED"
    assert result["data"]["devices"][0]["status"] == "PASS"
    assert result["data"]["devices"][0]["device_id"] == "001"
    assert pd.read_csv(result["artifacts"]["gaps"]).empty
    assert pd.read_csv(result["artifacts"]["cooccurrence"]).empty


def test_curve_diagnosis_matches_existing_complete_evaluation(tmp_path):
    write_training_inputs(tmp_path)
    options = dict(cases="cases.yaml", contract="measurement_contract.yaml", spec="spec.yaml", root=tmp_path)
    old = run_evaluation(**options)
    new = run_diagnosis(**options, output=tmp_path / "diagnosis")
    assert {r["device_id"]: r["status"] for r in new["data"]["devices"]} == old.results.set_index("device_id")["status"].to_dict()
    for row_ in new["data"]["metric_details"]:
        assert row_["value"] == pytest.approx(old.results.set_index("device_id").loc[row_["device_id"], row_["metric"]])
    assert new["data"]["source_summaries"][0]["source_type"] == "synthetic"


def test_partial_curves_keep_independent_metrics_and_never_invent_ion(tmp_path):
    write_training_inputs(tmp_path)
    path = tmp_path / "cases.yaml"
    cases = yaml.safe_load(path.read_text())
    for case in cases["cases"]:
        case["idvg"].pop()
    _write_yaml(path, cases)
    result = run_diagnosis(cases=path, contract="measurement_contract.yaml", spec="spec.yaml",
                           root=tmp_path, output=tmp_path / "diagnosis")
    details = result["data"]["metric_details"]
    assert all(r["status"] == "MISSING" and r["value"] is None for r in details if r["metric"] == "ion")
    assert all(r["status"] in {"PASS", "FAIL"} for r in details if r["metric"] == "ioff")
    assert all(not d["complete"] for d in result["data"]["devices"])


def test_bad_ion_consistency_leaves_low_bias_metrics_available(tmp_path):
    write_training_inputs(tmp_path)
    cases = yaml.safe_load((tmp_path / "cases.yaml").read_text())
    target = tmp_path / cases["cases"][0]["idvd"][0]["path"]
    curve = pd.read_csv(target)
    curve["id"] *= 2
    curve.to_csv(target, index=False)
    result = run_diagnosis(cases="cases.yaml", contract="measurement_contract.yaml", spec="spec.yaml",
                           root=tmp_path, output=tmp_path / "diagnosis")
    first = {r["metric"]: r for r in result["data"]["metric_details"] if r["device_id"] == "training_0"}
    assert first["ion"]["status"] == "INVALID"
    assert first["ioff"]["status"] in {"PASS", "FAIL"}


def test_cli_json_matches_service_and_invalid_request_has_nonzero_exit(tmp_path):
    request = metric_request(tmp_path, [row("d", ion=0.5)])
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    cli = subprocess.run([sys.executable, "-m", "mosfet_platform.cli", "diagnose", "--request", str(path)],
                         capture_output=True, text=True, encoding="utf-8")
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)["data"] == diagnose(request)["data"]
    path.write_text('{"spec":"missing"}', encoding="utf-8")
    bad = subprocess.run([sys.executable, "-m", "mosfet_platform.cli", "diagnose", "--request", str(path)],
                         capture_output=True, text=True, encoding="utf-8")
    assert bad.returncode == 1
    assert json.loads(bad.stdout)["status"] == "FAILED"


@pytest.mark.parametrize("payload", [{}, {"contract": "x", "spec": "y", "group_by": "length_m"},
                                     {"contract": "x", "spec": "y", "unknown": 1}])
def test_service_validation_returns_structured_error(payload):
    result = diagnose(payload)
    assert result["status"] == "FAILED"
    assert result["error"]["code"] == "INVALID_REQUEST"
