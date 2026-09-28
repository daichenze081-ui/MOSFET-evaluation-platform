"""Admit measured data, retrain an immutable candidate, and publish one result."""

from __future__ import annotations

from contextlib import ExitStack, closing
from dataclasses import asdict, dataclass
from html import escape
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from uuid import uuid4

import pandas as pd
import yaml

from mosfet_platform.analysis.spec import parse_spec_document
from mosfet_platform.artifacts.registry import (
    admit_observation, canonical_json, connect_registry, content_hash, current_observations,
)
from mosfet_platform.case_manifest import load_comsol_case_manifest
from mosfet_platform.io.comsol_curve import load_comsol_curve
from mosfet_platform.measurement import load_measurement_contract
from mosfet_platform.provenance import file_sha256
from mosfet_platform.training.registry_dataset import training_fingerprint, write_dataset
from mosfet_platform.workflows.compare import run_comparison
from mosfet_platform.workflows.evaluate import run_evaluation, run_metric_evaluation
from mosfet_platform.workflows.fit import run_fit


@dataclass(frozen=True)
class UpdateResult:
    report_path: Path
    database_path: Path
    training_status: str
    counts: dict[str, int]
    model_manifest: Path | None
    model_root: Path | None


def _resolve(value, root):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _remove_candidate(directory: Path, parent: Path):
    if directory.resolve().parent != parent.resolve():
        raise ValueError("Candidate cleanup must stay inside the model store.")
    if directory.exists():
        shutil.rmtree(directory)


def _train(connection, records, *, contract_hash, project, contract_bytes, base_bytes, models, force):
    active = connection.execute("SELECT * FROM models WHERE contract_hash=?", (contract_hash,)).fetchone()
    active_version = active["version"] if active else None
    curves = sorted((row for row in records if row["kind"] == "curves"), key=lambda row: row["device_id"])
    if not any(row["role"] == "training" for row in curves):
        return active_version, "NO_TRAINING_DATA", "", None
    fingerprint = training_fingerprint(curves, project, base_bytes, contract_hash)
    if active and active["fingerprint"] == fingerprint and not force:
        return active_version, "REUSED", "", fingerprint
    previous = connection.execute(
        "SELECT details_json FROM runs WHERE contract_hash=? AND training_status='FAILED' ORDER BY rowid DESC LIMIT 1",
        (contract_hash,),
    ).fetchone()
    if previous and not force:
        details = json.loads(previous["details_json"])
        if details.get("training_fingerprint") == fingerprint:
            return active_version, "FAILED", details["training_error"], fingerprint
    version = uuid4().hex
    directory = models / version
    try:
        config = write_dataset(connection, curves, directory, project, contract_bytes, base_bytes)
        run_fit(config=config, root=directory)
    except (ValueError, RuntimeError, OSError) as error:
        errors_path = directory / "frozen/errors.csv"
        detail = str(error)
        if errors_path.is_file():
            messages = pd.read_csv(errors_path)["error_message"].dropna().drop_duplicates()
            detail = "; ".join(messages.astype(str)) or detail
        _remove_candidate(directory, models)
        return active_version, "FAILED", f"{type(error).__name__}: {detail}", fingerprint
    return version, "TRAINED", "", fingerprint


def _panel(incoming, counts, training_status, error, active):
    return (
        "<h1>数据更新结果</h1><p><strong>器件结论以有效实测为准，预测不能覆盖实测。</strong></p>"
        f"<p>新增记录：{counts['inserted']}；重复：{counts['duplicates']}；拒绝：{counts['rejected']}。</p>"
        f"<p>训练状态：{escape(training_status)}；当前模型：{escape(active or '无')}。</p>"
        f"<p>{escape(error)}</p>"
        "<p>FAILED 表示本次重训未发布，有效实测记录已保留，原模型保持不变。"
        "NO_TRAINING_DATA 表示仅保存和评价指标；REUSED 表示沿用现有模型。</p>"
        "<h2>本次输入与入库结果</h2>" + incoming.to_html(index=False, escape=True, na_rep="—")
    )


def run_update(
    *, project_config: str | Path, spec: str | Path, database: str | Path,
    metrics: str | Path | None = None, root: str | Path | None = None,
    force_retrain: bool = False,
) -> UpdateResult:
    root = Path(root or Path.cwd()).resolve()
    project_path, spec_path, database_path = (_resolve(value, root) for value in (project_config, spec, database))
    project = yaml.safe_load(project_path.read_text(encoding="utf-8"))
    contract_path = _resolve(project["measurement_contract"], root)
    contract = load_measurement_contract(contract_path)
    contract_hash = content_hash(asdict(contract))
    contract_bytes = contract_path.read_bytes()
    spec_doc = parse_spec_document(yaml.safe_load(spec_path.read_text(encoding="utf-8")))
    spec_snapshot = {"spec_metadata": asdict(spec_doc.metadata), "spec": spec_doc.limits}
    source_path = _resolve(metrics if metrics is not None else project["comsol"]["case_manifest"], root)
    kind = "metrics" if metrics is not None else "curves"
    base_path = _resolve(project["model"]["base_config"], root)
    base_bytes = base_path.read_bytes()
    input_paths = {project_path, contract_path, spec_path, source_path, base_path}
    cases = {}
    predicted_ids = set()
    if kind == "curves":
        manifest = load_comsol_case_manifest(source_path, project_root=root)
        cases = {
            case["case_id"]: {**case, "dataset_role": role}
            for role, group in (("training", manifest.cases), ("independent_validation", manifest.independent_validation_cases))
            for case in group
        }
        for case in cases.values():
            for curve_type in ("idvg", "idvd"):
                input_paths.update(_resolve(curve["path"], root) for curve in case[curve_type])
        if str(manifest.raw["source"]["type"]).lower() in {"predicted", "prediction", "model", "generated"}:
            predicted_ids.update(cases)
    else:
        raw = pd.read_csv(source_path, dtype=str, keep_default_na=False)
        raw.columns = raw.columns.str.strip()
        for row in raw.to_dict(orient="records"):
            origins = [str(row.get(key, "")).lower() for key in ("result_origin", "upstream_result_origin")]
            source_types = {str(row.get(key, "")).lower() for key in ("source_type", "upstream_source_type")}
            if any(origin.startswith("predict") for origin in origins) or source_types & {"predicted", "prediction", "model", "generated"}:
                predicted_ids.add(str(row.get("device_id", "")).strip())
    hashes = {path: file_sha256(path) for path in input_paths}
    report_path = database_path.parent / "result.html"
    if database_path in input_paths or report_path in input_paths:
        raise ValueError("Registry outputs must not overwrite input files.")
    models = database_path.parent / "models"
    with TemporaryDirectory(prefix="mosfet_update_") as temporary:
        stage = Path(temporary)
        options = dict(contract=contract_path, spec=spec_path, root=root, output=stage / "evaluation")
        evaluated = (run_evaluation(cases=source_path, **options) if kind == "curves"
                     else run_metric_evaluation(metrics=source_path, **options))
        frame = pd.read_csv(evaluated.metrics_path, dtype={"device_id": str}, keep_default_na=False)
        metric_rows = {row["device_id"]: row for row in json.loads(frame.to_json(orient="records"))}
        incoming = evaluated.results[["device_id", "status", "fail_reason"]].copy()
        incoming["admission"] = "REJECTED"
        prepared = {}
        for row in incoming.itertuples(index=False):
            if row.status not in {"PASS", "FAIL"} or row.device_id in predicted_ids:
                continue
            case = cases.get(row.device_id, {})
            curve_data = []
            try:
                for curve_type in ("idvg", "idvd"):
                    bias_name = "vds_V" if curve_type == "idvg" else "vgs_V"
                    for curve in case.get(curve_type, []):
                        path = _resolve(curve["path"], root)
                        if curve["analysis_role"] == "formal":
                            load_comsol_curve(path, curve_type=curve_type, fixed_bias_V=float(curve[bias_name]), numerical_zero_current_A=1e-14)
                        curve_data.append((curve_type, {key: value for key, value in curve.items() if key != "path"}, path.read_bytes()))
            except (ValueError, OSError) as error:
                incoming.loc[incoming.device_id.eq(row.device_id), "fail_reason"] = str(error)
                continue
            prepared[row.device_id] = (case, curve_data)
        incoming.loc[incoming.device_id.isin(predicted_ids), "fail_reason"] = "PREDICTIONS_ARE_NOT_MEASUREMENTS"
        incoming.loc[incoming.device_id.isin(predicted_ids), "status"] = "INVALID"
        if any(file_sha256(path) != digest for path, digest in hashes.items()):
            raise RuntimeError("Input changed during admission checks.")
        with closing(connect_registry(database_path)) as connection, connection, ExitStack() as cleanup:
            connection.execute("BEGIN IMMEDIATE")
            # Admit training first so renamed copies cannot take the validation role.
            admission_order = sorted(incoming.index, key=lambda index: cases.get(
                incoming.loc[index, "device_id"], {}
            ).get("dataset_role") == "independent_validation")
            for index in admission_order:
                row = incoming.loc[index]
                if row.device_id not in prepared:
                    continue
                case, curve_data = prepared[row.device_id]
                inserted = admit_observation(
                    connection, contract_hash=contract_hash, device_id=row.device_id, kind=kind,
                    role=case.get("dataset_role", "training") if kind == "curves" else "metrics",
                    metrics=metric_rows[row.device_id], case={"geometry": case["geometry"]} if case else {},
                    spec=spec_snapshot, status=row.status, curves=curve_data,
                )
                incoming.loc[index, "admission"] = "INSERTED" if inserted else "DUPLICATE"
            records = current_observations(connection, contract_hash)
            counts = {"inserted": int(incoming.admission.eq("INSERTED").sum()),
                      "duplicates": int(incoming.admission.eq("DUPLICATE").sum()),
                      "rejected": int(incoming.admission.eq("REJECTED").sum()), "stored": len(records)}
            version, training_status, error, fingerprint = _train(
                connection, records, contract_hash=contract_hash, project=project, contract_bytes=contract_bytes,
                base_bytes=base_bytes, models=models, force=force_retrain,
            )
            if training_status == "TRAINED":
                cleanup.callback(_remove_candidate, models / version, models)
            latest = {row["device_id"]: json.loads(row["metrics_json"]) for row in records}
            metric_path = stage / "stored_metrics.csv"
            pd.DataFrame(list(latest.values())).to_csv(metric_path, index=False)
            panel = _panel(incoming, counts, training_status, error, version)
            if any(row.get("source_type") == "synthetic" for row in latest.values()):
                panel = "<p><strong>包含合成示例数据，仅用于软件流程演示。</strong></p>" + panel
            model_root = models / version if version else None
            model_manifest = model_root / "frozen/workflow_manifest.json" if version else None
            if version and latest:
                compared = run_comparison(
                    model=model_manifest, metrics=metric_path, contract=contract_path, spec=spec_path,
                    root=model_root, output=stage / "comparison",
                )
                html = compared.report_path.read_text(encoding="utf-8").replace("<h1>", panel + "<h1>", 1)
            else:
                html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MOSFET 数据更新</title>' + panel + '</html>'
                if latest:
                    assessed = run_metric_evaluation(
                        metrics=metric_path, contract=contract_path, spec=spec_path,
                        root=root, output=stage / "stored_evaluation",
                    )
                    html = html.replace('</html>', '<h2>库内实测结论</h2>' + assessed.results.to_html(index=False, escape=True) + '</html>')
            if training_status == "TRAINED":
                connection.execute("""INSERT INTO models VALUES (?,?,?) ON CONFLICT(contract_hash)
                    DO UPDATE SET fingerprint=excluded.fingerprint,version=excluded.version""", (contract_hash, fingerprint, version))
            details = {"counts": counts, "training_fingerprint": fingerprint, "training_error": error,
                       "active_version": version, "input_files": [{"path": str(path), "sha256": digest} for path, digest in hashes.items()]}
            connection.execute("INSERT INTO runs (id,contract_hash,training_status,details_json,report_html) VALUES (?,?,?,?,?)",
                               (uuid4().hex, contract_hash, training_status, canonical_json(details), html))
            connection.commit()
            cleanup.pop_all()
            pending = report_path.with_name(f".result-{uuid4().hex}.html")
            try:
                pending.write_text(html, encoding="utf-8")
                pending.replace(report_path)
            finally:
                pending.unlink(missing_ok=True)
    return UpdateResult(report_path, database_path, training_status, counts, model_manifest, model_root)
