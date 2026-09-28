"""Render the same structured diagnosis consumed by local clients and agents."""

from __future__ import annotations

from html import escape
import json

import pandas as pd


def render_diagnosis(result: dict) -> str:
    data = result["data"]

    def table(rows, columns=None):
        if not rows:
            return "<p>无记录。</p>"
        frame = pd.DataFrame(rows)
        if columns:
            frame = frame.reindex(columns=columns)
        for col in frame:
            frame[col] = frame[col].map(lambda v: ", ".join(map(str, v)) if isinstance(v, list) else v)
        return "<div class='scroll'>" + frame.to_html(index=False, escape=True, na_rep="—", float_format=lambda v: f"{v:.6g}") + "</div>"

    summary = []
    labels = {"measured": "实测", "comsol": "COMSOL 仿真", "synthetic": "合成示例", "estimated": "模型估计"}
    for row in data["metric_summary"]:
        summary.append({"来源": labels[row["source_type"]], "条件": row["condition_id"], "指标": row["metric"],
                        "超限数": row["fail_count"], "有效受检数": row["evaluated_count"],
                        "超限率": f"{row['exceedance_rate']:.1%}" if row["exceedance_rate"] is not None else "不可计算",
                        "失败覆盖率": f"{row['failure_coverage']:.1%}" if row["failure_coverage"] is not None else "不可计算",
                        "仅该项超限": row["exclusive_fail_count"], "缺失": row["missing_count"], "无效": row["invalid_count"],
                        "超限器件": row["failed_device_ids"]})
    spec = data["spec"]["metadata"]
    completion_html = ""
    if "combined_devices" in data:
        labels_status = {"PASS": "满足当前 Spec", "FAIL": "不满足当前 Spec",
                         "PREDICTED_PASS": "预计满足（含预测，暂定）", "PREDICTED_FAIL": "预计不满足（含预测）",
                         "INCOMPLETE": "待确认：仍有缺口"}
        devices = [{"器件": r["device_id"], "来源": r["source_type"], "条件": r["condition_id"],
                    "综合结论": labels_status[r["status"]], "必需项已覆盖": r["complete"],
                    "预测补齐项": r["predicted_metrics"], "超限项": r["failed_metrics"], "未完成项": r["unavailable_metrics"]}
                   for r in data["combined_devices"]]
        metrics = [{"器件": r["device_id"], "观测来源": r["source_type"], "条件": r["condition_id"], "指标": r["metric"],
                    "用于判断的值": (f"{r['value']:.6g}" + (" (predicted)" if r["predicted"] or r["value_source"] == "estimated" else "")) if r["value"] is not None else "—",
                    "单位": r["unit"], "值来源": r["value_source"],
                    "Spec": "; ".join(f"{check['rule']} {check['limit']:.6g}" for check in data["rule_details"]
                                      if all(check[k] == r[k] for k in ("device_id", "source_type", "condition_id", "metric"))),
                    "判断": r["status"],
                    "原始状态": r["original_status"], "模型预测": r["prediction"], "实测/输入减预测": r["delta"],
                    "偏差检查": r["comparison"], "补齐说明": r["completion_reason"]}
                   for r in data["combined_metric_details"]]
        completion_html = ("<h2>综合判断</h2><p>predicted 为模型估计；预测结论为暂定，不等于实测验收。"
                           "未提供经过验证的预测区间；NOT_ASSESSED 表示没有可用的偏差判据或比较数据。"
                           "下方异常统计仍仅使用原始输入，不计入本次补齐值。</p>" + table(devices)
                           + "<h2>核心指标与预测对比</h2>" + table(metrics)
                           + "<h2>模型适用性与补齐限制</h2>" + table(data["model_checks"])
                           + "<pre>" + escape(json.dumps(data["model_evidence"], ensure_ascii=False, indent=2)) + "</pre>")
    return """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>MOSFET 指标诊断</title><style>
body{font-family:system-ui,sans-serif;margin:32px;color:#18212d;line-height:1.6}
table{border-collapse:collapse;font-size:13px}th,td{padding:8px;border:1px solid #d6dce3;text-align:left}
th{background:#eef3f8}.scroll{overflow-x:auto;margin:16px 0}pre{white-space:pre-wrap;background:#f4f6f8;padding:16px}
</style><h1>MOSFET 指标诊断</h1>""" + (
        f"<p>运行：{escape(result['run_id'])}；标准：{escape(spec['spec_id'])} / {escape(spec['version'])}。</p>"
        f"<p>器件：{data['device_count']}；选中观测：{data['observation_count']}。</p>"
        "<p>以下为所选样本的超限统计，不是物理根因概率。实测、仿真、合成示例和模型估计分别统计。"
        "同一器件可多项超限，失败覆盖率之和可超过 100%。仿真扫描比例不代表实际制造良率。</p>"
        "<p>FAIL 表示至少一项有效指标超限；complete=false 表示仍有必需项未完成检查。"
        "最大相对超限程度只表示偏离标准的程度。</p>"
        + (completion_html or "<p>本报告未运行模型补齐，缺口保留。</p>")
        + "<h2>原始证据评价概况</h2>" + table(data["source_summaries"])
        + "<h2>异常指标排名</h2>" + table(summary)
        + ("<details><summary>原始器件判定（追溯）</summary>" + table(data["devices"]) + "</details>" if completion_html else "<h2>器件判定</h2>" + table(data["devices"]))
        + "<h2>共同超限</h2>" + table([r for r in data["cooccurrence"] if r["both_fail_count"]])
        + "<h2>分组统计</h2>" + table(data["groups"], ["source_type", "group_by", "group_value", "metric", "fail_count", "evaluated_count", "exceedance_rate", "missing_count", "invalid_count"])
        + "<h2>缺口与不可用指标</h2>" + table(data["gaps"], ["device_id", "source_type", "metric", "status", "reason", "evidence"])
        + "<h2>逐规则明细</h2>" + table(data["rule_details"])
        + "<h2>证据与运行依据</h2><pre>" + escape(json.dumps(result["provenance"], ensure_ascii=False, indent=2))
        + "</pre></html>"
    )
