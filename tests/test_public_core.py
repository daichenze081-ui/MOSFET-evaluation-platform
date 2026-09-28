from __future__ import annotations

import pandas as pd
import pytest

from examples.demo import _model
from mosfet_platform.analysis.results import order_results, summarize_results
from mosfet_platform.io.csv_loader import load_iv_csv


def test_csv_loader_normalizes_public_example_columns(tmp_path):
    path = tmp_path / "curve.csv"
    pd.DataFrame(
        {
            "gate voltage": [0.0, 0.5, 1.0],
            "drain voltage": [0.1, 0.1, 0.1],
            "drain current": [1.0e-12, 1.0e-7, 1.0e-4],
        }
    ).to_csv(path, index=False)

    frame = load_iv_csv(path)

    assert list(frame.columns) == ["vgs", "vds", "id"]
    assert frame["id"].iloc[-1] == pytest.approx(1.0e-4)


def test_pass_precedes_failure_groups_ordered_by_severity():
    frame = pd.DataFrame(
        [
            {
                "device_id": "pass",
                "status": "PASS",
                "primary_failure_reason": "",
                "normalized_exceedance": 0.0,
            },
            {
                "device_id": "ion_minor",
                "status": "FAIL",
                "primary_failure_reason": "ion_low",
                "normalized_exceedance": 0.10,
            },
            {
                "device_id": "ss_major",
                "status": "FAIL",
                "primary_failure_reason": "ss_high",
                "normalized_exceedance": 0.40,
            },
        ]
    )

    ordered = order_results(frame)

    assert ordered["device_id"].tolist() == ["pass", "ss_major", "ion_minor"]


def test_summary_keeps_data_quality_outside_formal_denominator():
    frame = pd.DataFrame(
        [
            {
                "status": "PASS",
                "primary_failure_reason": "",
                "violations": "",
                "fail_reason": "",
            },
            {
                "status": "FAIL",
                "primary_failure_reason": "ion_low",
                "violations": "ion_low",
                "fail_reason": "ion_low",
            },
            {
                "status": "INVALID",
                "primary_failure_reason": "",
                "violations": "",
                "fail_reason": "invalid_curve",
            },
        ]
    )

    summary = summarize_results(frame)
    formal_fail = summary.loc[
        (summary["section"] == "judgement")
        & (summary["group"] == "formal")
        & (summary["item"] == "FAIL")
    ].iloc[0]

    assert formal_fail["count"] == 1
    assert formal_fail["denominator_count"] == 2


def test_synthetic_model_rejects_geometry_outside_demo_envelope():
    model = _model()

    with pytest.raises(ValueError, match="outside"):
        model.ids(
            vgs=1.0,
            vds=0.5,
            length_m=2.0e-6,
            tox_m=10.0e-9,
            device_width_m=10.0e-6,
        )
