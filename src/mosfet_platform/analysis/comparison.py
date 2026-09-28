"""Pair measured and predicted devices without dropping uncomparable inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


COMPARISON_METRICS = {
    "ion": "A", "ioff": "A", "vth": "V", "ss_mv_dec": "mV/dec",
    "gm_max": "S", "ion_ioff": "1", "ion_ioff_cross_bias": "1",
    "dibl_mV_per_V": "mV/V",
}
_GEOMETRY = ("width_m", "length_m", "oxide_thickness_m")


def compare_results(
    measured: pd.DataFrame, predicted: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    for label, frame in (("measured", measured), ("predicted", predicted)):
        required = {"device_id", "status", "condition_id", *_GEOMETRY}
        if not required.issubset(frame.columns):
            raise ValueError(f"{label} results lack identity, status, or geometry columns.")
        ids = frame["device_id"]
        if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.duplicated().any():
            raise ValueError(f"{label} device IDs must be non-empty and unique.")

    devices = measured.rename(columns=lambda name: name if name == "device_id" else f"measured_{name}").merge(
        predicted.rename(columns=lambda name: name if name == "device_id" else f"predicted_{name}"),
        on="device_id", how="outer", validate="one_to_one", indicator=True,
    )
    context_matches = devices["measured_condition_id"].eq(devices["predicted_condition_id"])
    for name in _GEOMETRY:
        left = pd.to_numeric(devices[f"measured_{name}"], errors="coerce")
        right = pd.to_numeric(devices[f"predicted_{name}"], errors="coerce")
        context_matches &= np.isfinite(left) & np.isfinite(right) & (left > 0) & (right > 0)
        context_matches &= np.isclose(left, right, rtol=1e-9, atol=0.0)
    devices["comparison_status"] = np.select(
        [
            devices["_merge"].eq("right_only"),
            devices["_merge"].eq("left_only"),
            ~devices["measured_status"].isin(["PASS", "FAIL"]),
            devices["predicted_status"].eq("OUT_OF_ENVELOPE"),
            ~devices["predicted_status"].isin(["PREDICTED_PASS", "PREDICTED_FAIL"]),
            ~context_matches,
        ],
        ["MISSING_MEASUREMENT", "MISSING_PREDICTION", "MEASUREMENT_INVALID",
         "OUT_OF_ENVELOPE", "PREDICTION_INVALID", "CONTEXT_MISMATCH"],
        default="COMPARABLE",
    )
    comparable = devices["comparison_status"].eq("COMPARABLE")
    agreement = devices["measured_status"].eq(devices["predicted_status"].str.removeprefix("PREDICTED_"))
    devices["spec_agreement"] = np.select(
        [~comparable, agreement], ["NOT_COMPARABLE", "MATCH"], default="MISMATCH",
    )
    measured_valid = devices["measured_status"].isin(["PASS", "FAIL"])
    devices["device_conclusion"] = devices["measured_status"].where(measured_valid, "UNDETERMINED")
    devices["conclusion_basis"] = np.where(measured_valid, "MEASUREMENT", "NO_VALID_MEASUREMENT")
    model_assessment = {
        ("PASS", "PREDICTED_PASS"): "CORRECT_PASS",
        ("FAIL", "PREDICTED_FAIL"): "CORRECT_FAIL",
        ("FAIL", "PREDICTED_PASS"): "FALSE_PASS",
        ("PASS", "PREDICTED_FAIL"): "FALSE_FAIL",
    }
    devices["model_assessment"] = [
        model_assessment.get((measured, predicted), "NOT_EVALUATED") if eligible else "NOT_EVALUATED"
        for measured, predicted, eligible in zip(
            devices["measured_status"], devices["predicted_status"], comparable,
        )
    ]
    devices = devices.drop(columns="_merge")

    rows = []
    for metric, unit in COMPARISON_METRICS.items():
        pair = devices.reindex(columns=["device_id", "comparison_status", f"measured_{metric}", f"predicted_{metric}"]).copy()
        pair = pair.rename(columns={f"measured_{metric}": "measured", f"predicted_{metric}": "predicted"})
        pair["metric"] = metric
        pair["unit"] = unit
        for name in ("measured", "predicted"):
            pair[name] = pd.to_numeric(pair[name], errors="coerce")
        finite = np.isfinite(pair["measured"]) & np.isfinite(pair["predicted"])
        with np.errstate(over="ignore", invalid="ignore"):
            delta = pair["predicted"] - pair["measured"]
        overflow = finite & ~np.isfinite(delta)
        available = comparable & finite & ~overflow
        pair["metric_status"] = np.select(
            [~comparable, ~finite, overflow],
            ["DEVICE_NOT_COMPARABLE", "MISSING_METRIC", "NUMERIC_OVERFLOW"], default="AVAILABLE",
        )
        pair["signed_error"] = delta.where(available)
        pair["absolute_error"] = pair["signed_error"].abs()
        denominator = pair["measured"].abs().where(available & pair["measured"].ne(0))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            relative = pair["absolute_error"] / denominator * 100
        pair["relative_error_percent"] = relative.where(np.isfinite(relative))
        pair["relative_error_status"] = np.select(
            [~available, pair["measured"].eq(0), ~np.isfinite(relative)],
            ["NOT_AVAILABLE", "ZERO_REFERENCE", "NUMERIC_OVERFLOW"], default="AVAILABLE",
        )
        rows.append(pair)
    metrics = pd.concat(rows, ignore_index=True)
    counts = {"devices": len(devices), "comparable": int(comparable.sum())}
    counts.update({name.lower(): int(devices["spec_agreement"].eq(name).sum()) for name in ("MATCH", "MISMATCH", "NOT_COMPARABLE")})
    return devices, metrics, counts
