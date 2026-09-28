from __future__ import annotations

import pandas as pd
import pytest
import yaml

from examples.train_demo import write_training_inputs
from mosfet_platform.cli import build_parser, main
from mosfet_platform.workflows import run_evaluation, run_metric_evaluation


@pytest.fixture
def measured(tmp_path):
    write_training_inputs(tmp_path)
    result = run_evaluation(
        cases="cases.yaml", contract="measurement_contract.yaml", spec="spec.yaml",
        root=tmp_path, output="curves_result",
    )
    return tmp_path, result


def import_table(root, path, **kwargs):
    return run_metric_evaluation(
        metrics=path, contract="measurement_contract.yaml", spec="spec.yaml",
        root=root, **kwargs,
    )


def test_curve_and_metric_paths_have_identical_judgements(measured):
    root, curves = measured
    imported = import_table(root, curves.metrics_path, output="metrics_result")
    columns = [
        "device_id", "status", "primary_failure_reason", "primary_metric",
        "fail_reason", "normalized_exceedance",
    ]
    pd.testing.assert_frame_equal(curves.results[columns], imported.results[columns])
    pd.testing.assert_frame_equal(curves.summary, imported.summary)
    assert imported.counts == curves.counts
    assert imported.results["result_origin"].eq("imported_metrics").all()
    assert imported.metrics_path.is_file()


@pytest.mark.parametrize("field,value,reason", [
    ("ion", "", "missing_metric:ion"),
    ("gm_max", "garbage", "non_numeric_metric:gm_max"),
    ("ss_mv_dec", "inf", "non_finite_metric:ss_mv_dec"),
    ("vth_status", "unreliable", "vth_status_not_ok"),
    ("formal_eligible", "False", "formal_eligible_not_true"),
    ("condition_id", "different", "condition_id_mismatch"),
    ("width_m", 2e-6, "contract_mismatch:width_m"),
    ("temperature_K", 350, "contract_mismatch:temperature_K"),
    ("device_type", "pmos", "contract_mismatch:device_type"),
    ("oxide_thickness_m", -1, "invalid_positive_value:oxide_thickness_m"),
    ("ion_vds_V", 0.1, "contract_mismatch:ion_vds_V"),
    ("ioff", -1e-12, "negative_metric:ioff"),
])
def test_bad_rows_are_invalid_without_hiding_other_devices(measured, field, value, reason):
    root, curves = measured
    frame = pd.read_csv(curves.metrics_path, dtype=str, keep_default_na=False)
    device = frame.loc[0, "device_id"]
    frame.loc[0, field] = str(value)
    path = root / "changed.csv"
    frame.to_csv(path, index=False)
    result = import_table(root, path)
    row = result.results.set_index("device_id").loc[device]
    assert row["status"] == "INVALID"
    assert reason in row["fail_reason"]
    assert result.counts["devices"] == 5
    assert result.counts["INVALID"] == 1


def test_missing_columns_remain_visible_as_invalid(measured):
    root, curves = measured
    frame = pd.read_csv(curves.metrics_path).drop(columns=["ioff", "temperature_K", "ss_status"])
    path = root / "missing.csv"
    frame.to_csv(path, index=False)
    result = import_table(root, path)
    assert result.counts["INVALID"] == 5
    assert result.results["fail_reason"].str.contains("missing_metric:ioff").all()


def test_declared_units_do_not_change_values_or_scientific_notation(measured):
    root, curves = measured
    frame = pd.read_csv(curves.metrics_path, dtype=str, keep_default_na=False)
    frame["device_id"] = ["001", "002", "003", "004", "005"]
    frame["ioff"] = "1.234e-30"
    frame = frame.rename(columns={"ion": "Ion (A)", "ioff": "Ioff [A]", "ss_mv_dec": "SS (mV/dec)"})
    path = root / "units.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    result = import_table(root, path, device_id="001")
    assert result.counts["devices"] == 1
    assert float(result.results.iloc[0]["ioff"]) == 1.234e-30
    assert result.results.iloc[0]["device_id"] == "001"


@pytest.mark.parametrize("column", ["Ion (mA)", "SS (V/dec)"])
def test_incompatible_units_are_rejected(measured, column):
    root, curves = measured
    frame = pd.read_csv(curves.metrics_path).rename(columns={"ion" if column.startswith("Ion") else "ss_mv_dec": column})
    path = root / "units.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unit"):
        import_table(root, path)


def test_duplicate_alias_columns_and_ids_are_rejected(measured):
    root, curves = measured
    frame = pd.read_csv(curves.metrics_path)
    frame["Ion (A)"] = frame["ion"]
    path = root / "duplicates.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique.*columns"):
        import_table(root, path)
    frame = frame.drop(columns=["Ion (A)"])
    frame.loc[1, "device_id"] = frame.loc[0, "device_id"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="device_id"):
        import_table(root, path)


def test_enabled_ratio_cannot_be_omitted(measured):
    root, curves = measured
    spec_path = root / "spec.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    spec["spec"].update(ion_ioff_min=10, ion_ioff_cross_bias_min=10)
    spec_path.write_text(yaml.safe_dump(spec))
    frame = pd.read_csv(curves.metrics_path)
    frame["ion_ioff"] = 100
    frame = frame.drop(columns=["ion_ioff_cross_bias"])
    path = root / "ratio.csv"
    frame.to_csv(path, index=False)
    result = import_table(root, path)
    assert result.counts["INVALID"] == 5
    assert result.results["fail_reason"].str.contains("missing_metric:ion_ioff_cross_bias").all()


def test_cli_metric_entry_point_and_mutual_exclusion(measured, monkeypatch, capsys):
    root, curves = measured
    monkeypatch.chdir(root)
    monkeypatch.setattr("sys.argv", [
        "mosfet-platform", "evaluate", "--metrics", str(curves.metrics_path),
        "--contract", "measurement_contract.yaml", "--spec", "spec.yaml", "--out", "cli",
    ])
    main()
    assert "devices: 5" in capsys.readouterr().out
    assert (root / "cli/measured_results.csv").is_file()
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "evaluate", "--cases", "cases.yaml", "--metrics", "metrics.csv",
            "--contract", "c.yaml", "--spec", "s.yaml",
        ])


def test_both_ratio_rules_match_curve_evaluation(measured):
    root, _ = measured
    path = root / "spec.yaml"
    spec = yaml.safe_load(path.read_text())
    spec["spec"].update(ion_ioff_min=1e30, ion_ioff_cross_bias_min=1e30)
    path.write_text(yaml.safe_dump(spec))
    curves = run_evaluation(
        cases="cases.yaml", contract="measurement_contract.yaml", spec="spec.yaml",
        root=root, output="both_ratios",
    )
    imported = import_table(root, curves.metrics_path)
    assert imported.counts["FAIL"] == 5
    assert imported.results["fail_reason"].str.contains("ion_ioff_low").all()
    assert imported.results["fail_reason"].str.contains("ion_ioff_cross_bias_low").all()
    pd.testing.assert_series_equal(curves.results["fail_reason"], imported.results["fail_reason"])


def test_input_table_cannot_be_overwritten(measured):
    root, curves = measured
    original = curves.metrics_path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        import_table(root, curves.metrics_path, output=curves.metrics_path.parent)
    assert curves.metrics_path.read_bytes() == original


def test_extra_cells_cannot_shift_the_input_columns(measured):
    root, _ = measured
    path = root / "ragged.csv"
    path.write_text("device_id,ion\n001,1e-4,unexpected\n")
    with pytest.raises(ValueError, match="column count"):
        import_table(root, path)


def test_curve_width_uses_relative_tolerance(measured):
    root, _ = measured
    path = root / "cases.yaml"
    cases = yaml.safe_load(path.read_text())
    cases["common_conditions"]["width_m"] += 5e-9
    path.write_text(yaml.safe_dump(cases))
    with pytest.raises(ValueError, match="width"):
        run_evaluation(
            cases="cases.yaml", contract="measurement_contract.yaml", spec="spec.yaml", root=root,
        )


def test_missing_curves_retain_retest_rows_and_empty_metric_schema(measured):
    root, _ = measured
    path = root / "cases.yaml"
    cases = yaml.safe_load(path.read_text())
    for case in cases["cases"]:
        case["idvg"].pop()
    path.write_text(yaml.safe_dump(cases))
    result = run_evaluation(
        cases="cases.yaml", contract="measurement_contract.yaml", spec="spec.yaml",
        root=root, output="incomplete",
    )
    assert result.counts["RETEST"] == 5
    metrics = pd.read_csv(result.metrics_path)
    assert metrics.empty
    assert {"device_id", "ion", "vth_status"} <= set(metrics.columns)
