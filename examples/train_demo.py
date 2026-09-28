"""Fit and import a compact model from reproducible synthetic I-V curves."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from examples.demo import WIDTH_M, _contract, _model, _spec, _write_curve, _write_yaml
from mosfet_platform.workflows.fit import run_fit
from mosfet_platform.workflows.predict import run_prediction


TRAINING_GEOMETRIES = (
    (0.80e-6, 8.0e-9),
    (0.85e-6, 11.0e-9),
    (1.00e-6, 10.0e-9),
    (1.15e-6, 8.5e-9),
    (1.20e-6, 12.0e-9),
)


def write_training_inputs(
    root: Path,
    *,
    geometries: tuple[tuple[float, float], ...] = TRAINING_GEOMETRIES,
    independent: bool = False,
) -> Path:
    """Persist fictional training inputs; no qualification is preassigned."""
    root = root.resolve()
    model = _model()

    def write_case(case_id: str, length_m: float, tox_m: float) -> dict:
        curves = {"idvg": [], "idvd": []}
        for kind, bias, name in (
            ("idvg", 0.1, "low"),
            ("idvg", 1.0, "high"),
            ("idvd", 1.2, "verify"),
        ):
            path = root / "curves" / case_id / f"{name}.csv"
            _write_curve(
                path, curve_type=kind, fixed_bias=bias, model=model,
                length_m=length_m, tox_m=tox_m,
            )
            bias_key = {"idvg": "vds_V", "idvd": "vgs_V"}[kind]
            curves[kind].append({
                bias_key: bias,
                "path": path.relative_to(root).as_posix(),
                "qc_status": "active",
            })
        return {
            "case_id": case_id,
            "geometry": {"length_m": length_m, "oxide_thickness_m": tox_m},
            **curves,
        }

    cases = [
        write_case(f"training_{index}", length, tox)
        for index, (length, tox) in enumerate(geometries)
    ]
    validation = []
    if independent:
        validation.append({
            **write_case("validation_0", 1.05e-6, 9.5e-9),
            "dataset_role": "independent_validation",
        })
    _write_yaml(root / "cases.yaml", {
        "schema_version": 1,
        "source": {"type": "synthetic", "root": "curves", "authoritative": False},
        "common_conditions": {
            "device_type": "nmos", "width_m": WIDTH_M, "temperature_K": 300.0,
        },
        "nominal_case_id": cases[0]["case_id"],
        "cases": cases,
        "independent_validation_cases": validation,
    })
    _write_yaml(root / "measurement_contract.yaml", _contract())
    _write_yaml(root / "spec.yaml", _spec())
    base = model.effective_parameters(
        model.length_ref_m, model.tox_ref_m, device_width_m=WIDTH_M,
    )
    _write_yaml(root / "base_model.yaml", {"device": asdict(base)})
    _write_yaml(root / "project.yaml", {
        "comsol": {"case_manifest": "cases.yaml", "nominal_case_id": cases[0]["case_id"]},
        "measurement_contract": "measurement_contract.yaml",
        "model": {"base_config": "base_model.yaml"},
        "model_generalization": {
            "model_family": "additive",
            "cross_validation": "disabled",
            "independent_validation": "required" if independent else "disabled",
        },
    })
    _write_yaml(root / "training.yaml", {"training": {
        "project_config": "project.yaml", "output_dir": "frozen",
    }})
    pd.DataFrame([
        {"device_id": case["case_id"], "width_m": WIDTH_M, **case["geometry"]}
        for case in cases
    ]).to_csv(root / "prediction_devices.csv", index=False)
    _write_yaml(root / "prediction.yaml", {"prediction": {
        "model_manifest": "frozen/workflow_manifest.json",
        "measurement_contract": "measurement_contract.yaml",
        "spec": "spec.yaml",
        "input": "prediction_devices.csv",
    }})
    return root / "training.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/training_demo")
    parser.add_argument("--independent-validation", action="store_true")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    config = write_training_inputs(root, independent=args.independent_validation)
    fitted = run_fit(config=config, root=root)
    prediction = run_prediction(
        config="prediction.yaml", root=root, output="prediction", include_curves=True,
    )
    print("Synthetic training example; not engineering qualification evidence.")
    print(f"Model: {fitted.model}")
    print(f"Validation: {fitted.validation}")
    print(prediction.counts)


if __name__ == "__main__":
    main()
