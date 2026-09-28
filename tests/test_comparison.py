from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from examples.train_demo import write_training_inputs
from mosfet_platform.analysis.comparison import compare_results
from mosfet_platform.provenance import file_sha256
from mosfet_platform.cli import main
from mosfet_platform.workflows import run_comparison, run_evaluation, run_fit


def result_row(device="a", status="PASS", **values):
    return {"device_id": device, "status": status, "condition_id": "c",
            "width_m": 1e-5, "length_m": 1e-6, "oxide_thickness_m": 1e-8,
            "ion": 2.0, "ioff": 0.0, **values}


def test_errors_and_zero_reference_are_explicit():
    devices, metrics, counts = compare_results(
        pd.DataFrame([result_row()]),
        pd.DataFrame([result_row(status="PREDICTED_FAIL", ion=3.0, ioff=1e-12)]),
    )
    assert counts == {"devices": 1, "comparable": 1, "match": 0, "mismatch": 1, "not_comparable": 0}
    ion = metrics.set_index("metric").loc["ion"]
    assert ion["signed_error"] == 1.0
    assert ion["relative_error_percent"] == 50.0
    ioff = metrics.set_index("metric").loc["ioff"]
    assert ioff["absolute_error"] == 1e-12
    assert np.isnan(ioff["relative_error_percent"])
    assert ioff["relative_error_status"] == "ZERO_REFERENCE"
    assert metrics.set_index("metric").loc["gm_max", "metric_status"] == "MISSING_METRIC"


@pytest.mark.parametrize("measured,predicted,conclusion,assessment", [
    ("FAIL", "PREDICTED_PASS", "FAIL", "FALSE_PASS"),
    ("PASS", "PREDICTED_FAIL", "PASS", "FALSE_FAIL"),
    ("PASS", "PREDICTED_PASS", "PASS", "CORRECT_PASS"),
    ("FAIL", "PREDICTED_FAIL", "FAIL", "CORRECT_FAIL"),
    ("PASS", "OUT_OF_ENVELOPE", "PASS", "NOT_EVALUATED"),
    ("FAIL", "INVALID_INPUT", "FAIL", "NOT_EVALUATED"),
    ("INVALID", "PREDICTED_PASS", "UNDETERMINED", "NOT_EVALUATED"),
    ("RETEST", "PREDICTED_PASS", "UNDETERMINED", "NOT_EVALUATED"),
])
def test_measurement_controls_device_conclusion(measured, predicted, conclusion, assessment):
    devices, _, _ = compare_results(
        pd.DataFrame([result_row(status=measured)]),
        pd.DataFrame([result_row(status=predicted)]),
    )
    row = devices.iloc[0]
    assert row["device_conclusion"] == conclusion
    assert row["model_assessment"] == assessment
    assert row["conclusion_basis"] == (
        "NO_VALID_MEASUREMENT" if conclusion == "UNDETERMINED" else "MEASUREMENT"
    )


@pytest.mark.parametrize("measured,predicted,expected", [
    (result_row(status="INVALID"), result_row(status="PREDICTED_PASS"), "MEASUREMENT_INVALID"),
    (result_row(status="RETEST"), result_row(status="PREDICTED_PASS"), "MEASUREMENT_INVALID"),
    (result_row(), result_row(status="OUT_OF_ENVELOPE"), "OUT_OF_ENVELOPE"),
    (result_row(), result_row(status="INVALID_INPUT"), "PREDICTION_INVALID"),
    (result_row(), result_row(status="PREDICTED_PASS", condition_id="other"), "CONTEXT_MISMATCH"),
    (result_row(), result_row(status="PREDICTED_PASS", length_m=1.001e-6), "CONTEXT_MISMATCH"),
])
def test_uncomparable_rows_never_become_zero_errors(measured, predicted, expected):
    devices, metrics, counts = compare_results(pd.DataFrame([measured]), pd.DataFrame([predicted]))
    assert devices.iloc[0]["comparison_status"] == expected
    assert counts["comparable"] == 0
    assert metrics["absolute_error"].isna().all()
    assert devices.iloc[0]["spec_agreement"] == "NOT_COMPARABLE"


def test_missing_devices_are_preserved_and_duplicates_rejected():
    measured = pd.DataFrame([result_row("a")])
    predicted = pd.DataFrame([result_row("b", status="PREDICTED_PASS")])
    devices, _, counts = compare_results(measured, predicted)
    assert set(devices["comparison_status"]) == {"MISSING_MEASUREMENT", "MISSING_PREDICTION"}
    assert counts["devices"] == 2
    with pytest.raises(ValueError, match="unique"):
        compare_results(pd.concat([measured, measured]), predicted)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    root = tmp_path_factory.mktemp("comparison")
    config = write_training_inputs(root)
    fit = run_fit(config=config, root=root)
    measured = run_evaluation(
        cases="cases.yaml", contract="measurement_contract.yaml", spec="spec.yaml",
        root=root, output="measured",
    )
    return root, fit, measured


def compare(root, fit, **options):
    return run_comparison(
        root=root, model=fit.manifest, contract="measurement_contract.yaml", spec="spec.yaml", **options,
    )


def test_actual_fit_both_input_paths_and_report_provenance(fitted):
    root, fit, measured = fitted
    curves = compare(root, fit, cases="cases.yaml", output="curve_report")
    metrics = compare(root, fit, metrics=measured.metrics_path, output="metric_report")
    assert curves.counts == metrics.counts
    columns = ["device_id", "measured_status", "predicted_status", "comparison_status", "spec_agreement"]
    pd.testing.assert_frame_equal(curves.devices[columns], metrics.devices[columns])
    pd.testing.assert_frame_equal(curves.metrics, metrics.metrics)
    assert curves.counts["comparable"] == 5
    manifest = json.loads(curves.manifest_path.read_text())
    assert manifest["independent_validation_status"] == "NOT_RUN"
    assert manifest["qualification_ready"] is False
    assert manifest["model_sha256"] == file_sha256(fit.model)
    assert len(manifest["input_files"]) >= 20
    for item in manifest["output_files"]:
        assert file_sha256(curves.report_path.parent / item["path"]) == item["sha256"]
    html = curves.report_path.read_text(encoding="utf-8")
    assert "NOT_RUN" in html and "absolute_error" in html and "gm_max" in html


def test_bad_metrics_and_outside_geometry_stay_in_report(fitted):
    root, fit, measured = fitted
    frame = pd.read_csv(measured.metrics_path)
    frame.loc[0, "ss_status"] = "bad"
    frame.loc[1, "length_m"] = 2e-6
    path = root / "bad_metrics.csv"
    frame.to_csv(path, index=False)
    result = compare(root, fit, metrics=path, output="bad_report")
    assert result.counts["devices"] == 5
    assert result.counts["comparable"] == 3
    assert {"MEASUREMENT_INVALID", "OUT_OF_ENVELOPE"} <= set(result.devices["comparison_status"])


@pytest.mark.parametrize("label", ['<script>alert("device")</script>', "NA", "001"])
def test_report_escapes_device_labels_and_selects_one(fitted, label):
    root, fit, measured = fitted
    frame = pd.read_csv(measured.metrics_path)
    frame.loc[0, "device_id"] = label
    path = root / "labels.csv"
    frame.to_csv(path, index=False)
    result = compare(root, fit, metrics=path, output="escaped_report", device_id=label)
    assert result.counts["devices"] == 1
    assert result.counts["comparable"] == 1
    assert result.devices.iloc[0]["device_id"] == label
    html = result.report_path.read_text(encoding="utf-8")
    assert "<script>" not in html
    assert label.replace("<", "&lt;").replace(">", "&gt;") in html


def test_relative_overflow_does_not_emit_infinite_error():
    _, metrics, _ = compare_results(
        pd.DataFrame([result_row(ion=1e-300)]),
        pd.DataFrame([result_row(status="PREDICTED_PASS", ion=1e100)]),
    )
    ion = metrics.set_index("metric").loc["ion"]
    assert ion["absolute_error"] == 1e100
    assert ion["relative_error_status"] == "NUMERIC_OVERFLOW"
    assert np.isnan(ion["relative_error_percent"])


def test_comparison_cli_and_input_exclusivity(fitted, monkeypatch, capsys):
    root, fit, measured = fitted
    monkeypatch.chdir(root)
    monkeypatch.setattr("sys.argv", [
        "mosfet-platform", "compare", "--metrics", str(measured.metrics_path),
        "--model", str(fit.manifest), "--contract", "measurement_contract.yaml",
        "--spec", "spec.yaml", "--out", "cli_report",
    ])
    main()
    assert "comparable: 5" in capsys.readouterr().out
    assert (root / "cli_report/comparison_report.html").is_file()
    with pytest.raises(ValueError, match="exactly one"):
        compare(root, fit, cases="cases.yaml", metrics=measured.metrics_path, output="invalid")


def test_invalid_model_cannot_replace_a_report(fitted):
    root, fit, _ = fitted
    path = root / "invalid_model.json"
    raw = json.loads(fit.manifest.read_text())
    raw["prediction_ready"] = False
    path.write_text(json.dumps(raw))
    output = root / "unchanged_report"
    output.mkdir()
    report = output / "comparison_report.html"
    report.write_text("previous report")
    with pytest.raises(ValueError, match="prediction_ready"):
        run_comparison(
            model=path, cases="cases.yaml", contract="measurement_contract.yaml",
            spec="spec.yaml", output=output, root=root,
        )
    assert report.read_text() == "previous report"
