"""Compare curve extraction with reimporting the same calculated metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from examples.train_demo import write_training_inputs
from mosfet_platform.workflows.evaluate import run_evaluation, run_metric_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/metric_import_demo")
    root = Path(parser.parse_args().output).resolve()
    write_training_inputs(root)
    options = dict(root=root, contract="measurement_contract.yaml", spec="spec.yaml")
    curves = run_evaluation(cases="cases.yaml", output="from_curves", **options)
    imported = run_metric_evaluation(
        metrics=curves.metrics_path, output="from_metrics", **options,
    )
    judgement = ["device_id", "status", "fail_reason", "primary_metric", "normalized_exceedance"]
    pd.testing.assert_frame_equal(curves.results[judgement], imported.results[judgement])
    pd.testing.assert_frame_equal(curves.summary, imported.summary)
    print("Synthetic example: curve and metric-table judgements match.")
    print(imported.counts)
    print(f"Metrics: {curves.metrics_path}")
    print(f"Results: {imported.results_path}")


if __name__ == "__main__":
    main()
