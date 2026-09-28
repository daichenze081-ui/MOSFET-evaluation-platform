from __future__ import annotations

import argparse
import json
from pathlib import Path

from mosfet_platform.workflows.evaluate import run_evaluation, run_metric_evaluation
from mosfet_platform.workflows.fit import run_fit
from mosfet_platform.workflows.predict import run_prediction
from mosfet_platform.workflows.compare import run_comparison
from mosfet_platform.workflows.update import run_update


def _path(value: Path | None) -> str:
    return str(value) if value else "not exported"


def _counts(counts: dict[str, int]) -> None:
    print(" | ".join(f"{name}: {count}" for name, count in counts.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate measured MOSFET curves or predict device performance."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    diagnose = commands.add_parser("diagnose", help="Diagnose per-metric Spec violations and missing data.")
    diagnose.add_argument("--request", required=True, help="JSON request for the local diagnostic service.")

    evaluate = commands.add_parser("evaluate", help="Evaluate curves or calculated metrics.")
    inputs = evaluate.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--cases")
    inputs.add_argument("--metrics")
    evaluate.add_argument("--contract", required=True)
    evaluate.add_argument("--spec", required=True)
    scope = evaluate.add_mutually_exclusive_group()
    scope.add_argument("--device")
    scope.add_argument("--batch")
    evaluate.add_argument("--out")
    evaluate.add_argument("--diagnostics", action="store_true")

    fit = commands.add_parser("fit", help="Validate and freeze a model.")
    fit.add_argument("--config", required=True)

    predict = commands.add_parser("predict", help="Predict device performance.")
    predict.add_argument("--config", required=True)
    predict.add_argument("--device")
    predict.add_argument("--out")
    predict.add_argument("--curves", action="store_true")
    predict.add_argument("--diagnostics", action="store_true")

    compare = commands.add_parser("compare", help="Compare measured metrics with frozen-model predictions.")
    source = compare.add_mutually_exclusive_group(required=True)
    source.add_argument("--cases")
    source.add_argument("--metrics")
    compare.add_argument("--model", required=True, help="Frozen-model manifest.")
    compare.add_argument("--contract", required=True)
    compare.add_argument("--spec", required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--device")
    update = commands.add_parser("update", help="Admit measurements, retrain when needed, and publish one result.")
    update.add_argument("--project", required=True)
    update.add_argument("--spec", required=True)
    update.add_argument("--database", default="outputs/platform/catalog.sqlite3")
    update.add_argument("--metrics", help="Import calculated metrics without adding training curves.")
    update.add_argument("--retrain", action="store_true", help="Retry training even if data and configuration are unchanged.")
    return parser


def _require_device(args: argparse.Namespace) -> None:
    if args.diagnostics and not args.device:
        raise ValueError("Diagnostics require --device.")
    if args.diagnostics and not args.out:
        raise ValueError("Diagnostics require --out.")


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "diagnose":
        from mosfet_platform.api import diagnose
        try:
            request = json.loads(Path(args.request).read_text(encoding="utf-8-sig"))
        except (ValueError, OSError) as error:
            print(json.dumps({"status": "FAILED", "error": {"code": "INVALID_REQUEST", "message": str(error)}}, ensure_ascii=False))
            raise SystemExit(1)
        result = diagnose(request)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        if result["status"] == "FAILED":
            raise SystemExit(1)
        return

    if args.command == "update":
        result = run_update(
            project_config=args.project, spec=args.spec, database=args.database,
            metrics=args.metrics, force_retrain=args.retrain,
        )
        _counts(result.counts)
        print(f"Training: {result.training_status}")
        print(f"Result: {result.report_path}")
        if result.training_status == "FAILED":
            raise SystemExit(1)
        return

    if args.command == "compare":
        result = run_comparison(
            model=args.model, contract=args.contract, spec=args.spec, output=args.out,
            cases=args.cases, metrics=args.metrics, device_id=args.device,
        )
        _counts(result.counts)
        print(f"Report: {result.report_path}")
        return

    if args.command == "evaluate":
        _require_device(args)
        options = dict(
            contract=args.contract,
            spec=args.spec,
            device_id=args.device,
            batch_id=args.batch,
            output=args.out,
        )
        if args.metrics:
            if args.diagnostics:
                raise ValueError("Curve diagnostics require --cases; metric tables contain no curves.")
            result = run_metric_evaluation(metrics=args.metrics, **options)
        else:
            result = run_evaluation(cases=args.cases, diagnostics=args.diagnostics, **options)
        _counts(result.counts)
        print(f"Results: {_path(result.results_path)}")
        print(f"Metrics: {_path(result.metrics_path)}")
        return

    if args.command == "fit":
        result = run_fit(config=args.config)
        print(f"Family: {result.family}")
        print(f"Model: {result.model}")
        print(f"Manifest: {result.manifest}")
        print(f"Validation: {result.validation}")
        print(f"Errors: {result.errors}")
        return

    _require_device(args)
    result = run_prediction(
        config=args.config,
        device_id=args.device,
        output=args.out,
        include_curves=args.curves,
        diagnostics=args.diagnostics,
    )
    _counts(result.counts)
    print(f"Results: {_path(result.results_path)}")
    if args.curves:
        print(f"Curves: {_path(result.curves_path)}")


if __name__ == "__main__":
    main()
