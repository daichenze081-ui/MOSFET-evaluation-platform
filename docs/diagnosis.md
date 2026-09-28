# 本地指标诊断

`diagnose` 使用现有 Spec 和曲线/指标数据输出逐指标判定、异常排名、分组、共同超限及缺口。默认不运行模型；可显式配置 `completion`，复用既有模型进行预测对比及缺失指标补齐，不重新训练。

## 运行自己的本地数据

建立一个本地 JSON 请求文件，例如放入 Git 忽略的 `outputs/local/`：

```json
{
  "root": "C:/data/device_project",
  "cases": "cases.yaml",
  "contract": "measurement_contract.yaml",
  "spec": "spec.yaml",
  "output": "outputs/diagnosis",
  "group_by": ["length_m", "oxide_thickness_m"]
}
```

```powershell
python -m mosfet_platform.cli diagnose --request outputs/local/request.json
```

`root` 是输入项目根目录，`cases`、`contract`、`spec` 相对该目录解析。`output` 相对运行命令的当前目录解析，建议本地自动化使用绝对路径。程序读取原始数据，输出写入新的独立运行目录。

使用指标表时，以 `"metrics": "metrics.csv"` 替换 `cases`。二者只能选一个。指标表需要 `device_id`、`condition_id`、`device_type`、`width_m`、`length_m`、`oxide_thickness_m`、`temperature_K` 等上下文；仅客户 Spec 启用的指标参与判定，缺失值可以留空。Vth 和 SS 使用时仍须声明有效的 `vth_status`、`ss_status`。

来源由清单或指标表的 `source_type` 声明。指标表无来源字段时，可以在请求中提供 `source_type`。支持 `measured`、`comsol`、`synthetic`、`estimated`。请求与数据的来源声明冲突时拒绝执行，不能靠参数把预测改为实测。每个器件、条件、来源组合只能有一条选定记录；同一组合重复会报错。不同来源可以保留同一器件，分别分析；同一请求仍只对当前合约和 Spec 的条件进行有效判定。

`device_id` 可选择一个器件。`group_by` 支持 `temperature_K`、`length_m`、`oxide_thickness_m`、`width_m`、`batch_id`；每个字段单独分组，来源和条件始终分开。

## Python / agent 调用边界

```python
import json
from pathlib import Path
from mosfet_platform.api import diagnose, DIAGNOSIS_REQUEST_SCHEMA

request = json.loads(Path("outputs/local/request.json").read_text(encoding="utf-8"))
response = diagnose(request)
if response["status"] == "SUCCEEDED":
    print(response["data"]["metric_summary"])
    print(response["artifacts"]["report"])
else:
    print(response["error"])
```

`DIAGNOSIS_REQUEST_SCHEMA` 是可用于注册工具的请求 JSON Schema。返回值只含可序列化的 JSON 数据，未知数值使用 `null`，不会返回 NaN 或要求调用方解析 HTML。CLI 调用同一接口。

顶层字段：

| 字段 | 含义 |
| --- | --- |
| schema_version | 原始诊断为 1.0；启用补齐为 1.1 |
| run_id | 本次执行 ID |
| status | SUCCEEDED 或 FAILED，表示执行状态 |
| data | 器件、逐指标/逐规则明细、统计、共同超限、分组、缺口 |
| provenance | Spec/合约快照、源文件与代码哈希、来源元数据及选择范围 |
| artifacts | JSON、HTML 和 CSV 的绝对路径 |
| error | 执行失败时的错误代码与消息 |

器件 `FAIL` 是诊断结果，不是执行失败；CLI 在成功分析出失败器件时仍返回退出码 0。请求、输入或运行错误返回非零退出码。

## 结果文件

每次成功运行写入 `output/<run_id>/`：

- `result.html`：用户阅读入口。
- `result.json`：完整结构化结果与运行依据。
- `devices.csv`：逐器件判定、完整性和来源。
- `metric_details.csv`、`rule_details.csv`：逐指标与逐规则证据。
- `metric_summary.csv`、`source_summaries.csv`：指标排名与各来源计数。
- `cooccurrence.csv`、`groups.csv`：共同超限及分组统计。
- `gaps.csv`：缺失或无效的指标及原因。

新运行不会覆盖历史报告。原始数据文件不被改写。原始文件仍由输入项目管理；结果保存其路径和哈希，长期复现时应同时保留原始文件。

## 判定与统计含义

- 逐指标状态为 PASS、FAIL、MISSING、INVALID。
- 器件有有效超限项即 FAIL；必需项全部有效且通过才 PASS；其余为 INCOMPLETE。
- `complete=false` 可以与 FAIL 同时出现，表示已发现超限但仍有检查缺口。
- 没有某指标所需曲线时，保留可独立提取的其他指标。Ion 必须通过合约要求的双曲线一致性检查。
- 登记清单引用不存在的文件属于输入配置错误；主动未登记某条所需曲线可以形成诊断缺口。二者不混淆。
- 指标超限率以该指标有效受检数为分母；共同超限率以双方均有效受检数为分母。
- Vth 的上下限分别保留规则，但按一个指标统计。
- 仅该指标超限要求其他必需指标全部检查完成并通过。
- 各来源统计分别展示。合成与仿真场景的比例不等于实测制造良率。
- 本批次异常频率不表示未来失效概率或物理根因概率。

## 可选：旧模型对比与缺失补齐

在原请求中增加以下字段（模型路径相对 `root`）：

```json
{
  "completion": {
    "model_manifest": "models/released/workflow_manifest.json"
  }
}
```

无需新器件 ID 曾出现在训练集中。模型加载沿用已有版本、哈希与验证检查；额外检查固定温度、固定宽度、器件类型和几何范围。当前支持已有 NMOS 模型，绝对电流不做宽度换算。范围内只表示满足已声明边界，不保证稀疏参数组合附近有充分训练数据，也不自动认证新的工艺或材料。

同一请求同时给出已测指标的预测和差值。若有经验证或客户明确指定的绝对误差容差，可在 `completion.absolute_tolerances` 中配置，例如 `{"vth": 0.01}` 表示 0.01 V。这只是格式示例，不是推荐阈值。容差使用指标原单位；任何可比较指标超出已配置容差都会暂停该器件的缺失补齐。未配置或无有效比较值时返回 `NOT_ASSESSED`，不会虚构模型一致性或置信区间。

仅 MISSING 可补齐，INVALID 和已有值不覆盖。模型缺失、损坏、不适用或没有通过独立验证时，原始诊断仍完成，补齐限制写入 `model_evidence` 和 `model_checks`。无法读取原始输入仍属于请求失败。加载后输入变化会终止发布，防止证据与结果不一致。

新增数据与 CSV：

- `combined_devices`：综合判定、原始状态、预测指标、完整性和暂定标记。
- `combined_metric_details`：采用值、原始值/状态、值来源、预测、差值及采用理由。
- `model_checks`：逐器件适用性、输入参数、偏差检查与阻断原因。
- `model_evidence`：模型版本、哈希、验证状态、声明范围与限制（JSON 内）。

报告只有一个主综合结论与核心指标表，估计值标注 `predicted`；原始判定放在追溯区。PASS / FAIL 保留原始证据含义；PREDICTED_PASS / PREDICTED_FAIL 为包含预测的暂定结论；INCOMPLETE 表示仍不能完整判断。原始有效 FAIL 优先保留，预测不能推翻它。

目前没有校准的预测区间，暂定预测通过不等于正式实测验收。原始 `devices`、`metric_summary`、`groups` 等数据保持不变；补齐值不计入这些原始统计。`provenance.completion_requested` 表示请求过模型，`completion_performed` 表示实际采用过至少一个补齐值。

该版本不自动合并不同来源记录、不查库选择 COMSOL 替代记录，不执行局部校准、重训、根因推断或 Spec 修订。原有严格训练与导入入口继续要求唯一器件 ID；诊断入口才按器件/条件/来源组合区分观测。

## 合成示例与验证

```powershell
python -m examples.diagnosis_demo
python -m pytest -q
```

如 Windows 默认测试临时目录无权限，可指定一个尚不存在的工作区临时目录：

```powershell
python -m pytest -q --basetemp outputs/pytest-local-run
```

pytest 会管理该临时目录，不要将 `--basetemp` 指向已有业务数据或报告目录。
