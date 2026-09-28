from __future__ import annotations

from pathlib import Path

import pytest

from examples.demo import format_report, run_demo


def test_synthetic_demo_runs_both_public_workflows(tmp_path):
    result = run_demo(tmp_path)

    measured = result.measured.set_index("device_id")
    predicted = result.predicted.set_index("device_id")
    assert set(measured.index) == {
        "synthetic_reference",
        "synthetic_low_ion",
        "synthetic_high_ss",
    }
    assert measured.loc["synthetic_reference", "status"] == "PASS"
    assert measured.loc["synthetic_low_ion", "status"] == "FAIL"
    assert measured.loc["synthetic_high_ss", "status"] == "FAIL"
    assert predicted.loc["synthetic_reference", "status"] == "PREDICTED_PASS"
    assert predicted.loc["synthetic_low_ion", "status"] == "PREDICTED_FAIL"
    assert predicted.loc["synthetic_high_ss", "status"] == "PREDICTED_FAIL"
    assert measured.loc["synthetic_low_ion", "primary_metric"] == "ion"
    assert measured.loc["synthetic_high_ss", "primary_metric"] == "ss_mv_dec"
    assert predicted.loc["synthetic_low_ion", "primary_metric"] == "ion"
    assert predicted.loc["synthetic_high_ss", "primary_metric"] == "ss_mv_dec"
    assert result.batch == {
        "devices": 3,
        "pass": 1,
        "fail": 2,
        "pass_rate": pytest.approx(1.0 / 3.0),
    }
    assert result.passes["device_id"].tolist() == ["synthetic_reference"]
    assert result.failures["group"].tolist() == ["Ion", "SS"]
    assert result.failures["deviation_percent"].tolist() == pytest.approx(
        [40.9537736, 5.9264149], rel=1.0e-6
    )
    assert (tmp_path / "measured" / "measured_results.csv").is_file()
    assert (tmp_path / "predicted" / "predicted_results.csv").is_file()
    assert (tmp_path / "predicted" / "predicted_curves.csv").is_file()
    assert result.figure_path == tmp_path / "readme_showcase.png"
    assert result.figure_path.stat().st_size > 10_000

    report = format_report(result)
    assert "Devices: 3    Pass: 1    Fail: 2    Pass rate: 33.3%" in report
    assert "FAIL - Ion" in report
    assert "FAIL - SS" in report
    assert "Synthetic demonstration only" in report


def test_readme_documents_the_complete_workflow():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    expected = ("python -m examples.platform_demo", "result.html", "docs/workflow.md")
    for evidence in expected:
        assert evidence in readme
    assert (root / "docs/workflow.md").is_file()
    assert (root / "examples/platform_demo.py").is_file()
