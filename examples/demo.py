"""Run a synthetic measured-versus-predicted portfolio demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from mosfet_platform.model.geometry_aware import (
    GeometryAwareModelParameters,
    GeometryEnvelope,
)
from mosfet_platform.workflows.evaluate import run_evaluation
from mosfet_platform.workflows.predict import run_prediction


CONDITION_ID = "synthetic_demo_condition_v1"
WIDTH_M = 10.0e-6
DEVICES = (
    ("synthetic_reference", 1.00e-6, 10.0e-9),
    ("synthetic_low_ion", 1.20e-6, 12.0e-9),
    ("synthetic_high_ss", 0.80e-6, 8.0e-9),
)


@dataclass(frozen=True)
class ShowcaseResult:
    measured: pd.DataFrame
    predicted: pd.DataFrame
    batch: dict[str, float | int]
    passes: pd.DataFrame
    failures: pd.DataFrame
    figure_path: Path


def _model() -> GeometryAwareModelParameters:
    return GeometryAwareModelParameters(
        width_m=WIDTH_M,
        temperature_K=300.0,
        length_ref_m=1.0e-6,
        tox_ref_m=10.0e-9,
        theta_mobility=0.15,
        vth_ref_V=0.40,
        mu_ref_m2_per_Vs=0.020,
        subthreshold_n_ref=1.50,
        i0_A=2.0e-9,
        dibl_ref_V_per_V=0.030,
        lambda_ref_1_per_V=0.050,
        vth_length_slope_V=0.15,
        vth_tox_slope_V=0.10,
        log_mu_length_slope=-0.35,
        log_mu_tox_slope=-0.25,
        log_n_minus_one_length_slope=-2.50,
        log_n_minus_one_tox_slope=-2.50,
        log_dibl_length_slope=-0.30,
        log_dibl_tox_slope=0.10,
        log_lambda_length_slope=-0.10,
        log_lambda_tox_slope=0.10,
        envelope=GeometryEnvelope(
            length_min_m=0.80e-6,
            length_max_m=1.20e-6,
            tox_min_m=8.0e-9,
            tox_max_m=12.0e-9,
        ),
    )


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )


def _contract() -> dict[str, Any]:
    return {
        "measurement_contract": {
            "condition_id": CONDITION_ID,
            "device_type": "nmos",
            "temperature_K": 300.0,
            "geometry": {"width_m": WIDTH_M, "length_m": 1.0e-6},
            "transfer": {
                "vds_V": 0.10,
                "vgs_start_V": 0.0,
                "vgs_stop_V": 1.20,
                "vgs_off_V": 0.0,
                "vgs_on_V": 1.20,
                "num_points": 61,
            },
            "ion": {
                "vgs_V": 1.20,
                "vds_V": 1.00,
                "primary_curve_type": "idvg",
                "verification_curve_type": "idvd",
                "consistency_relative_tolerance": 0.01,
                "consistency_absolute_tolerance_A": 1.0e-15,
            },
            "dibl": {"low_vds_V": 0.10, "high_vds_V": 1.00},
            "vth": {
                "method": "constant_current_w_over_l",
                "reference_current_A_per_W_over_L": 1.0e-7,
            },
            "ss": {
                "method": "log_linear_fit",
                "current_min_A": 1.0e-12,
                "current_max_A": 1.0e-7,
                "minimum_points": 3,
            },
            "current": {
                "metric_source": "magnitude",
                "preserve_signed_current": True,
                "sampling_method": "exact",
                "bias_voltage_tolerance_V": 1.0e-12,
                "allow_interpolation": False,
            },
        }
    }


def _spec() -> dict[str, Any]:
    return {
        "spec_metadata": {
            "spec_id": "SYNTHETIC_PORTFOLIO_001",
            "version": "1.0",
            "source": "generated_for_portfolio",
            "effective_date": "2026-09-01",
            "device_type": "nmos",
            "condition_id": CONDITION_ID,
            "approval_status": "illustrative_only",
            "qualification_level": "synthetic_demo",
        },
        "spec": {
            "ion_min": 2.0e-4,
            "ioff_max": 1.0e-7,
            "ion_ioff_cross_bias_min": 1.0e3,
            "vth_min": 0.20,
            "vth_max": 0.65,
            "ss_max_mv_dec": 120.0,
        },
    }


def _write_curve(
    path: Path,
    *,
    curve_type: str,
    fixed_bias: float,
    model: GeometryAwareModelParameters,
    length_m: float,
    tox_m: float,
) -> pd.DataFrame:
    vgs = np.linspace(0.0, 1.20, 61)
    vds = np.linspace(0.0, 1.00, 61)
    if curve_type == "idvg":
        frame = pd.DataFrame(
            {
                "vgs": vgs,
                "vds": fixed_bias,
                "id": model.ids(
                    vgs=vgs,
                    vds=np.full_like(vgs, fixed_bias),
                    length_m=length_m,
                    tox_m=tox_m,
                    device_width_m=WIDTH_M,
                ),
            }
        )
    else:
        frame = pd.DataFrame(
            {
                "vgs": fixed_bias,
                "vds": vds,
                "id": model.ids(
                    vgs=np.full_like(vds, fixed_bias),
                    vds=vds,
                    length_m=length_m,
                    tox_m=tox_m,
                    device_width_m=WIDTH_M,
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _write_inputs(root: Path) -> pd.DataFrame:
    model = _model()
    measured_model = replace(
        model,
        mu_ref_m2_per_Vs=model.mu_ref_m2_per_Vs * 0.985,
        subthreshold_n_ref=1.51,
    )
    curves_root = root / "synthetic_curves"
    cases: list[dict[str, Any]] = []
    curve_frames: list[pd.DataFrame] = []
    for device_id, length_m, tox_m in DEVICES:
        directory = curves_root / device_id
        low = directory / "idvg_low.csv"
        high = directory / "idvg_high.csv"
        verify = directory / "idvd_verify.csv"
        low_frame = _write_curve(
            low,
            curve_type="idvg",
            fixed_bias=0.10,
            model=measured_model,
            length_m=length_m,
            tox_m=tox_m,
        )
        high_frame = _write_curve(
            high,
            curve_type="idvg",
            fixed_bias=1.00,
            model=measured_model,
            length_m=length_m,
            tox_m=tox_m,
        )
        verify_frame = _write_curve(
            verify,
            curve_type="idvd",
            fixed_bias=1.20,
            model=measured_model,
            length_m=length_m,
            tox_m=tox_m,
        )
        for name, frame in (
            ("idvg_low", low_frame),
            ("idvg_high", high_frame),
            ("idvd_verify", verify_frame),
        ):
            curve_frames.append(
                frame.assign(device_id=device_id, curve_name=name)
            )
        cases.append(
            {
                "case_id": device_id,
                "geometry": {
                    "length_m": length_m,
                    "oxide_thickness_m": tox_m,
                },
                "idvg": [
                    {
                        "vds_V": 0.10,
                        "path": low.relative_to(root).as_posix(),
                        "qc_status": "active",
                    },
                    {
                        "vds_V": 1.00,
                        "path": high.relative_to(root).as_posix(),
                        "qc_status": "active",
                    },
                ],
                "idvd": [
                    {
                        "vgs_V": 1.20,
                        "path": verify.relative_to(root).as_posix(),
                        "qc_status": "active",
                    }
                ],
            }
        )

    _write_yaml(root / "measurement_contract.yaml", _contract())
    _write_yaml(root / "spec.yaml", _spec())
    _write_yaml(
        root / "cases.yaml",
        {
            "schema_version": 1,
            "source": {
                "type": "synthetic",
                "root": "synthetic_curves",
                "authoritative": False,
            },
            "common_conditions": {
                "device_type": "nmos",
                "width_m": WIDTH_M,
                "temperature_K": 300.0,
            },
            "nominal_case_id": DEVICES[0][0],
            "cases": cases,
        },
    )
    pd.DataFrame(
        [
            {
                "device_id": device_id,
                "width_m": WIDTH_M,
                "length_m": length_m,
                "oxide_thickness_m": tox_m,
            }
            for device_id, length_m, tox_m in DEVICES
        ]
    ).to_csv(root / "prediction_devices.csv", index=False)

    model_path = root / "synthetic_model" / "selected_geometry_aware_model.yaml"
    _write_yaml(
        model_path,
        {
            "schema_version": 1,
            "formal": True,
            "workflow_class": "global_geometry_calibration",
            "model": {
                "type": "geometry_aware_enhanced",
                "family": "additive",
                "parameters": model.to_mapping(),
            },
            "qualification": {
                "validation_status": "PASS",
                "envelope_status": "PASS",
                "independent_validation_status": "SYNTHETIC",
                "qualification_ready": True,
            },
            "provenance": {
                "run_id": "synthetic_portfolio_model",
                "source": "generated_non_engineering_example",
            },
        },
    )
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest_path = root / "synthetic_model" / "workflow_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "workflow": "model_generalization",
                "formal": True,
                "workflow_class": "global_geometry_calibration",
                "run_id": "synthetic_portfolio_model",
                "status": "PASS",
                "validation_status": "PASS",
                "independent_validation_status": "SYNTHETIC",
                "qualification_ready": True,
                "selected_family": "additive",
                "output_files": [
                    {
                        "path": model_path.relative_to(root).as_posix(),
                        "sha256": digest,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_yaml(
        root / "prediction.yaml",
        {
            "prediction": {
                "model_manifest": manifest_path.relative_to(root).as_posix(),
                "measurement_contract": "measurement_contract.yaml",
                "spec": "spec.yaml",
                "input": "prediction_devices.csv",
            }
        },
    )
    return pd.concat(curve_frames, ignore_index=True)


def _validate_results(measured: pd.DataFrame, predicted: pd.DataFrame) -> None:
    expected = {
        "synthetic_reference": ("PASS", "PREDICTED_PASS", ""),
        "synthetic_low_ion": ("FAIL", "PREDICTED_FAIL", "ion"),
        "synthetic_high_ss": ("FAIL", "PREDICTED_FAIL", "ss_mv_dec"),
    }
    measured_by_id = measured.set_index("device_id")
    predicted_by_id = predicted.set_index("device_id")
    if set(measured_by_id.index) != set(expected) or set(predicted_by_id.index) != set(expected):
        raise RuntimeError("Synthetic showcase returned an unexpected device set.")
    for device_id, (measured_status, predicted_status, metric) in expected.items():
        measured_row = measured_by_id.loc[device_id]
        predicted_row = predicted_by_id.loc[device_id]
        actual = (
            str(measured_row["status"]),
            str(predicted_row["status"]),
            str(measured_row["primary_metric"]),
        )
        if actual != (measured_status, predicted_status, metric):
            raise RuntimeError(f"Synthetic showcase outcome drifted for {device_id}: {actual}")
        if str(predicted_row["primary_metric"]) != metric:
            raise RuntimeError(f"Measured and predicted failure groups differ for {device_id}.")
    if not measured["source_type"].astype(str).eq("synthetic").all():
        raise RuntimeError("Showcase measured inputs must be classified as synthetic.")
    if not predicted["source_id"].astype(str).str.startswith("synthetic_").all():
        raise RuntimeError("Showcase predictions must use the synthetic model.")


def _showcase_tables(
    measured: pd.DataFrame,
    predicted: pd.DataFrame,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    measured_by_id = measured.set_index("device_id")
    predicted_by_id = predicted.set_index("device_id")
    passed = measured_by_id["status"].eq("PASS")
    device_count = len(measured_by_id)
    pass_count = int(passed.sum())
    batch: dict[str, float | int] = {
        "devices": device_count,
        "pass": pass_count,
        "fail": int((measured_by_id["status"] == "FAIL").sum()),
        "pass_rate": pass_count / device_count,
    }
    pass_rows = [
        {
            "device_id": device_id,
            "measured": measured_by_id.loc[device_id, "status"],
            "predicted": predicted_by_id.loc[device_id, "status"],
        }
        for device_id in measured_by_id.index[passed]
    ]
    labels = {
        "ion": ("Ion", "A", ">="),
        "ss_mv_dec": ("SS", "mV/dec", "<="),
    }
    failure_rows: list[dict[str, Any]] = []
    for device_id, measured_row in measured_by_id.loc[~passed].iterrows():
        predicted_row = predicted_by_id.loc[device_id]
        metric = str(measured_row["primary_metric"])
        label, unit, relation = labels[metric]
        failure_rows.append(
            {
                "group": label,
                "device_id": device_id,
                "measured_value": float(measured_row[metric]),
                "predicted_value": float(predicted_row[metric]),
                "limit": float(measured_row["primary_limit"]),
                "relation": relation,
                "unit": unit,
                "deviation_percent": 100.0 * float(measured_row["normalized_exceedance"]),
            }
        )
    failures = pd.DataFrame(failure_rows).sort_values(
        ["deviation_percent", "device_id"],
        ascending=[False, True],
        kind="stable",
    )
    return batch, pd.DataFrame(pass_rows), failures.reset_index(drop=True)


def _curve(
    frame: pd.DataFrame,
    *,
    device_id: str,
    curve_name: str | None = None,
    vds_V: float | None = None,
) -> pd.DataFrame:
    selected = frame.loc[frame["device_id"] == device_id]
    if curve_name is not None:
        selected = selected.loc[selected["curve_name"] == curve_name]
    if vds_V is not None:
        selected = selected.loc[
            (selected["curve_type"] == "idvg")
            & np.isclose(selected["vds_V"], vds_V, rtol=0.0, atol=1.0e-12)
        ]
    if selected.empty:
        raise RuntimeError(f"Missing showcase curve for {device_id}.")
    return selected.sort_values("vgs" if "vgs" in selected else "vgs_V")


def _write_figure(
    measured_curves: pd.DataFrame,
    predicted_curves: pd.DataFrame,
    failures: pd.DataFrame,
    path: Path,
) -> Path:
    ion_measured = _curve(
        measured_curves,
        device_id="synthetic_low_ion",
        curve_name="idvg_high",
    )
    ion_predicted = _curve(
        predicted_curves,
        device_id="synthetic_low_ion",
        vds_V=1.0,
    )
    ss_measured = _curve(
        measured_curves,
        device_id="synthetic_high_ss",
        curve_name="idvg_low",
    )
    ss_predicted = _curve(
        predicted_curves,
        device_id="synthetic_high_ss",
        vds_V=0.1,
    )
    ion = failures.loc[failures["group"] == "Ion"].iloc[0]
    ss = failures.loc[failures["group"] == "SS"].iloc[0]

    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold"})
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), constrained_layout=True)
    ion_axis, ss_axis = axes
    ion_color = "#2563eb"
    ss_color = "#d97706"

    ion_axis.plot(
        ion_measured["vgs"],
        ion_measured["id"] * 1.0e3,
        color=ion_color,
        linewidth=2.2,
        label="Measured",
    )
    ion_axis.plot(
        ion_predicted["vgs_V"],
        ion_predicted["id_A"] * 1.0e3,
        color=ion_color,
        linewidth=2.0,
        linestyle="--",
        label="Predicted",
    )
    ion_axis.axhline(
        ion["limit"] * 1.0e3,
        color="#dc2626",
        linewidth=1.4,
        linestyle=":",
        label="Ion minimum",
    )
    ion_axis.scatter(
        [1.2, 1.2],
        [ion["measured_value"] * 1.0e3, ion["predicted_value"] * 1.0e3],
        color=[ion_color, "#0f172a"],
        zorder=4,
    )
    ion_axis.set(
        title="Ion failure · high-bias Id–Vg",
        xlabel="Gate voltage, Vg (V)",
        ylabel="Drain current, Id (mA)",
        xlim=(0.0, 1.2),
        ylim=(0.0, None),
    )
    ion_axis.grid(alpha=0.22)
    ion_axis.legend(frameon=False, loc="upper left")

    ss_axis.semilogy(
        ss_measured["vgs"],
        ss_measured["id"],
        color=ss_color,
        linewidth=2.2,
        label="Measured",
    )
    ss_axis.semilogy(
        ss_predicted["vgs_V"],
        ss_predicted["id_A"],
        color=ss_color,
        linewidth=2.0,
        linestyle="--",
        label="Predicted",
    )
    ss_axis.axhspan(1.0e-12, 1.0e-7, color="#fde68a", alpha=0.28, label="SS fit window")
    ss_axis.text(
        0.98,
        0.06,
        f"SS limit ≤ {ss['limit']:.0f} mV/dec",
        color="#b91c1c",
        ha="right",
        va="bottom",
        transform=ss_axis.transAxes,
        bbox={"facecolor": "white", "edgecolor": "#fecaca", "alpha": 0.92},
    )
    ss_axis.set(
        title="SS failure · low-bias Id–Vg",
        xlabel="Gate voltage, Vg (V)",
        ylabel="Drain current, Id (A)",
        xlim=(0.0, 1.2),
    )
    ss_axis.grid(alpha=0.22, which="both")
    ss_axis.legend(frameon=False, loc="upper left")
    figure.suptitle("Synthetic failure evidence — measured vs predicted", fontsize=12, fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=170,
        bbox_inches="tight",
        metadata={"Software": "mosfet-device-analytics-portfolio"},
    )
    plt.close(figure)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Showcase figure was not created.")
    return path


def run_demo(
    output: str | Path = "outputs/demo",
    *,
    figure: str | Path | None = None,
) -> ShowcaseResult:
    output_root = Path(output).resolve()
    figure_path = Path(figure).resolve() if figure else output_root / "readme_showcase.png"
    with TemporaryDirectory(prefix="mosfet_synthetic_") as temporary:
        root = Path(temporary)
        measured_curves = _write_inputs(root)
        measured = run_evaluation(
            cases="cases.yaml",
            contract="measurement_contract.yaml",
            spec="spec.yaml",
            output=output_root / "measured",
            root=root,
        )
        predicted = run_prediction(
            config="prediction.yaml",
            output=output_root / "predicted",
            root=root,
            include_curves=True,
        )
        _validate_results(measured.results, predicted.results)
        batch, passes, failures = _showcase_tables(measured.results, predicted.results)
        _write_figure(measured_curves, predicted.curves, failures, figure_path)
    return ShowcaseResult(
        measured.results,
        predicted.results,
        batch,
        passes,
        failures,
        figure_path,
    )


def _failure_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["measured"] = rows.apply(
        lambda row: f"{row.measured_value:.3e} A"
        if row.unit == "A"
        else f"{row.measured_value:.1f} mV/dec",
        axis=1,
    )
    rows["predicted"] = rows.apply(
        lambda row: f"{row.predicted_value:.3e} A"
        if row.unit == "A"
        else f"{row.predicted_value:.1f} mV/dec",
        axis=1,
    )
    rows["specification"] = rows.apply(
        lambda row: f"{row.relation} {row.limit:.3e} A"
        if row.unit == "A"
        else f"{row.relation} {row.limit:.1f} mV/dec",
        axis=1,
    )
    rows["deviation"] = rows["deviation_percent"].map(lambda value: f"{value:+.1f}%")
    return rows.loc[:, ["device_id", "measured", "predicted", "specification", "deviation"]]


def format_report(result: ShowcaseResult) -> str:
    batch = result.batch
    lines = [
        "Synthetic demonstration only; not measurement or qualification evidence.",
        "",
        "BATCH SUMMARY",
        (
            f"Devices: {batch['devices']}    Pass: {batch['pass']}    "
            f"Fail: {batch['fail']}    Pass rate: {100.0 * batch['pass_rate']:.1f}%"
        ),
        "",
        "PASS",
        result.passes.to_string(index=False),
    ]
    for group, frame in result.failures.groupby("group", sort=False):
        lines.extend(
            [
                "",
                f"FAIL - {group}",
                _failure_table(frame).to_string(index=False),
            ]
        )
    lines.extend(["", f"Figure: {result.figure_path}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/demo")
    parser.add_argument("--figure")
    args = parser.parse_args()
    print(format_report(run_demo(args.output, figure=args.figure)))


if __name__ == "__main__":
    main()
