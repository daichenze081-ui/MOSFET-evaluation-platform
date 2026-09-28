"""Evaluate and predict the same inputs, then export an auditable comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import pandas as pd
import yaml

from mosfet_platform.analysis.comparison import compare_results
from mosfet_platform.analysis.spec import parse_spec_document
from mosfet_platform.artifacts.frozen_model import load_frozen_model
from mosfet_platform.case_manifest import load_comsol_case_manifest
from mosfet_platform.provenance import file_sha256
from mosfet_platform.measurement import load_measurement_contract
from mosfet_platform.workflows.evaluate import run_evaluation, run_metric_evaluation
from mosfet_platform.workflows.predict import run_prediction


@dataclass(frozen=True)
class ComparisonResult:
    devices: pd.DataFrame
    metrics: pd.DataFrame
    counts: dict[str, int]
    report_path: Path
    manifest_path: Path


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _report(devices: pd.DataFrame, metrics: pd.DataFrame, summary: pd.DataFrame, metadata: dict) -> str:
    columns = [
        "device_id", "device_conclusion", "conclusion_basis", "model_assessment",
        "measured_result_origin", "measured_source_type", "measured_status", "predicted_status",
        "comparison_status", "spec_agreement", "measured_fail_reason", "predicted_fail_reason",
    ]
    table_options = dict(index=False, escape=True, na_rep="—", float_format=lambda value: f"{value:.6g}")
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>MOSFET 实测与预测对比</title>
<style>body{font-family:system-ui,sans-serif;margin:32px;color:#18212d}
table{border-collapse:collapse;font-size:13px;margin:16px 0}th,td{padding:8px;border:1px solid #d6dce3;text-align:left}
th{background:#eef3f8}.scroll{overflow-x:auto}pre{white-space:pre-wrap;background:#f4f6f8;padding:16px}
</style><h1>MOSFET 实测与预测对比</h1>
<p><strong>器件结论以有效实测数据为准，预测结果不能推翻实测结论。</strong>
FALSE_PASS 表示实测不达标而模型误判为达标；FALSE_FAIL 表示实测达标而模型误判为不达标。
没有有效实测时，器件结论为 UNDETERMINED，不能用预测补为达标。</p>
<p>预测误差以实测为参考。Spec 判断一致不等于模型误差达标，本报告不自动授予模型资格。</p>
<p>仅 COMPARABLE 器件计算误差；原始状态与异常原因保留。相对误差以实测值绝对值为分母，
实测为零时标记 ZERO_REFERENCE，不使用小常数替代。INVALID/RETEST、超范围或条件不匹配不计入一致率。</p>
<h2>汇总</h2>""" + summary.to_html(**table_options) + "<h2>模型与依据</h2><pre>" + escape(
        json.dumps(metadata, ensure_ascii=False, indent=2)
    ) + "</pre><h2>器件判断</h2><div class='scroll'>" + devices.reindex(columns=columns).to_html(
        **table_options
    ) + "</div><h2>指标偏差</h2><p>signed_error = predicted − measured；absolute_error 与指标单位相同。</p><div class='scroll'>" + metrics.to_html(
        **table_options
    ) + "</div></html>"


def run_comparison(
    *,
    model: str | Path,
    contract: str | Path,
    spec: str | Path,
    output: str | Path,
    cases: str | Path | None = None,
    metrics: str | Path | None = None,
    root: str | Path | None = None,
    device_id: str | None = None,
) -> ComparisonResult:
    if (cases is None) == (metrics is None):
        raise ValueError("Provide exactly one of cases or metrics.")
    project = Path(root or Path.cwd()).resolve()
    model_path, contract_path, spec_path, output_path = (
        _resolve(value, project) for value in (model, contract, spec, output)
    )
    source_path = _resolve(cases if cases is not None else metrics, project)
    frozen = load_frozen_model(model_path, root=project)
    inputs = {source_path, model_path, frozen.model_path, contract_path, spec_path}
    if cases is not None:
        registered = load_comsol_case_manifest(source_path, project_root=project)
        for case in (*registered.cases, *registered.independent_validation_cases):
            for kind in ("idvg", "idvd"):
                inputs.update(_resolve(curve["path"], project) for curve in case[kind])
    names = ("comparison_devices.csv", "comparison_metrics.csv", "comparison_summary.csv",
             "comparison_report.html", "comparison_manifest.json")
    if inputs & {output_path / name for name in names}:
        raise ValueError("Comparison output must not overwrite its inputs.")
    hashes = {path: file_sha256(path) for path in sorted(inputs)}
    options = dict(contract=contract_path, spec=spec_path, root=project, device_id=device_id)
    measured = (
        run_evaluation(cases=source_path, **options)
        if cases is not None else run_metric_evaluation(metrics=source_path, **options)
    )
    with TemporaryDirectory(prefix="mosfet_compare_") as directory:
        stage = Path(directory)
        device_path = stage / "devices.csv"
        measured.results[["device_id", "width_m", "length_m", "oxide_thickness_m"]].to_csv(device_path, index=False)
        prediction_config = stage / "prediction.yaml"
        prediction_config.write_text(yaml.safe_dump({"prediction": {
            "model_manifest": str(model_path), "measurement_contract": str(contract_path),
            "spec": str(spec_path), "input": str(device_path),
        }}), encoding="utf-8")
        predicted = run_prediction(config=prediction_config, root=project)
        devices, metric_rows, counts = compare_results(measured.results, predicted.results)
        summary = pd.DataFrame([
            {"item": name, "count": value,
             "denominator": counts["comparable"] if name in {"match", "mismatch"} else counts["devices"],
             "rate": value / counts["comparable"] if name in {"match", "mismatch"} and counts["comparable"] else None}
            for name, value in counts.items()
        ])
        model_manifest = json.loads(model_path.read_text(encoding="utf-8"))
        spec_document = parse_spec_document(yaml.safe_load(spec_path.read_text(encoding="utf-8")))
        metadata = {
            "model_id": frozen.run_id, "model_family": frozen.family,
            "model_sha256": frozen.model_sha256,
            "model_status": frozen.status,
            "validation_status": frozen.validation_status,
            "prediction_ready": model_manifest.get("prediction_ready", model_manifest.get("qualification_ready")),
            "independent_validation_status": frozen.independent_validation_status,
            "qualification_ready": model_manifest.get("qualification_ready"),
            "validation_scope": model_manifest.get("validation_scope", "legacy"),
            "spec": asdict(spec_document.metadata),
            "spec_limits": spec_document.limits,
            "measurement_contract": asdict(load_measurement_contract(contract_path)),
            "input_kind": "curves" if cases is not None else "imported_metrics",
            "decision_policy": "valid_measurement_is_authoritative",
        }
        devices.to_csv(stage / names[0], index=False)
        metric_rows.to_csv(stage / names[1], index=False)
        summary.to_csv(stage / names[2], index=False)
        (stage / names[3]).write_text(_report(devices, metric_rows, summary, metadata), encoding="utf-8")
        if any(file_sha256(path) != digest for path, digest in hashes.items()):
            raise RuntimeError("Comparison input changed during evaluation.")
        manifest = {
            "workflow": "comparison", "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "counts": counts, **metadata,
            "input_files": [{"path": str(path), "sha256": digest} for path, digest in hashes.items()],
            "output_files": [{"path": name, "sha256": file_sha256(stage / name)} for name in names[:-1]],
        }
        (stage / names[4]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        output_path.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(stage / name, output_path / name)
    return ComparisonResult(devices, metric_rows, counts, output_path / names[3], output_path / names[4])
