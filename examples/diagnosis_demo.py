"""Run the local diagnostic service with synthetic, fully declared inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.train_demo import write_training_inputs
from mosfet_platform.api import diagnose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/diagnosis_demo")
    output = Path(parser.parse_args().output).resolve()
    inputs = output / "inputs"
    write_training_inputs(inputs)
    request = {"root": str(inputs), "cases": "cases.yaml", "contract": "measurement_contract.yaml",
               "spec": "spec.yaml", "output": str(output / "runs")}
    (output / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    result = diagnose(request)
    if result["status"] != "SUCCEEDED":
        raise SystemExit(result["error"]["message"])
    print("Synthetic diagnostic example; not measured production yield.")
    print(result["data"]["source_summaries"])
    print(f"Report: {result['artifacts']['report']}")


if __name__ == "__main__":
    main()
