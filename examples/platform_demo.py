"""Run the complete synthetic admission, training, and reporting workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from examples.train_demo import write_training_inputs
from mosfet_platform.workflows import run_update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/platform")
    directory = Path(parser.parse_args().output).resolve()
    inputs = directory / "input"
    write_training_inputs(inputs)
    for name in ("training.yaml", "prediction.yaml", "prediction_devices.csv"):
        (inputs / name).unlink()
    result = run_update(
        project_config="project.yaml", spec="spec.yaml", root=inputs,
        database=directory / "catalog.sqlite3",
    )
    print("Synthetic data only; measured-reference decisions remain authoritative.")
    print(result.counts)
    print(f"Training: {result.training_status}")
    print(f"Result: {result.report_path}")
    if result.training_status == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
