"""Train a synthetic compact model and report measured/predicted differences."""

from __future__ import annotations

import argparse
from pathlib import Path

from examples.train_demo import write_training_inputs
from mosfet_platform.workflows import run_comparison, run_evaluation, run_fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/comparison_demo")
    root = Path(parser.parse_args().output).resolve()
    config = write_training_inputs(root)
    fit = run_fit(config=config, root=root)
    options = dict(root=root, contract="measurement_contract.yaml", spec="spec.yaml")
    measured = run_evaluation(cases="cases.yaml", output="measured", **options)
    curves = run_comparison(model=fit.manifest, cases="cases.yaml", output="from_curves", **options)
    metrics = run_comparison(model=fit.manifest, metrics=measured.metrics_path, output="from_metrics", **options)
    assert curves.counts == metrics.counts
    print("Synthetic software demonstration; not independent model qualification.")
    print(curves.counts)
    print(f"Curve report: {curves.report_path}")
    print(f"Metric report: {metrics.report_path}")


if __name__ == "__main__":
    main()
