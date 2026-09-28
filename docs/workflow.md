# 数据、模型与结果流程

项目以有效实测为器件判断依据，模型预测用于估计与误差分析，不覆盖实测结论。
日常使用 `update` 完成数据准入、必要的重训和结果输出。最终阅读入口只有数据库旁的
`result.html`。数据库与已发布模型版本是运行资产，不是阶段性 review 输出。

## 运行

```powershell
python -m examples.platform_demo
mosfet-platform update --project project.yaml --spec spec.yaml --database outputs/platform/catalog.sqlite3
mosfet-platform update --project project.yaml --spec spec.yaml --metrics metrics.csv --database outputs/platform/catalog.sqlite3
```

示例使用运行时生成的合成 I–V，仅验证软件流程。第一次运行建立本地库并训练模型；
再次导入相同内容会识别为重复，不重训。阅读 `outputs/platform/result.html`。

`project.yaml` 指定以下输入，路径相对于当前项目目录：

```yaml
comsol:
  case_manifest: cases.yaml
  nominal_case_id: device_1
measurement_contract: measurement_contract.yaml
model:
  base_config: base_model.yaml
model_generalization:
  model_family: additive
  cross_validation: disabled
  independent_validation: disabled
```

标准文件由 `--spec` 单独指定。`--metrics` 导入已计算指标；省略时使用登记的完整曲线。
Python 入口为 `mosfet_platform.workflows.run_update`，支持显式 `root`。

## 数据准入

完整曲线根据测量合约提取 Ion、Ioff、Vth、SS、gm、DIBL 和开关比，再使用 Spec 判断。
每个器件必须覆盖合约要求的低漏压转移、高漏压转移和输出验证曲线。额外正式曲线也会
检查可读性。数据质量异常的器件被拒绝，原因保留在结果页。

有效数据的 PASS 和 FAIL 都可以入库。FAIL 是器件指标不达标，不代表实测数据无效；
只训练达标器件会丢失失效区域。预测数据不能作为实测训练依据：显式声明为预测或模型
输出的导入记录会被拒绝。系统不能证明外部来源声明的真实性，不能通过改标签把预测
变成实测。

数据库记录原始曲线字节及 SHA256、提取指标、数据角色和入库时的 Spec 判断。
同一测量合约、器件、输入类型的更新形成新记录，旧记录保留；重复内容不新增记录，
也不会把旧版本重新切换为当前数据。当前报告用当前 Spec 重新评价有效记录。

## 指标表格式

UTF-8 CSV，允许 BOM。每行一个唯一器件 ID，`001` 和 `NA` 按字面保留。

| 必需字段 | 单位或含义 |
| --- | --- |
| device_id, condition_id, device_type | 器件 ID、测量条件版本、器件类型 |
| width_m, length_m, oxide_thickness_m | m，有限正数 |
| temperature_K | K，须与合约一致 |
| ion, ioff | A，电流幅值 |
| vth, ss_mv_dec, gm_max | V、mV/dec、S |
| vth_status, ss_status | 实际提取状态，须明确为 ok |

Spec 启用开关比规则时，需要 `ion_ioff`（同偏置）或 `ion_ioff_cross_bias`（跨偏置）
对应字段，不互相代替。支持 `Ion (A)`、`Ioff [A]`、`SS (mV/dec)` 等明确单位表头；
不兼容单位会报错，不按量级推断或缩放。缺失、非数值或非有限必需指标为 INVALID。

`formal_eligible` 可省略，默认仅表示允许进入检查，false 会使数据无效。
`data_quality_status` 可省略，提供时只接受 ok/valid。Vth/SS 状态不自动补成 ok。
提供偏置列时还会检查与合约是否矛盾。没有原始曲线的指标只用于存储、判断和对比，
不会伪造 I–V 来训练模型。

## 训练和版本切换

完整测量合约的内容决定数据分组；不同宽度、温度、测量方法或条件分组不混训。
当前模型拟合长度与氧化层厚度影响。additive 至少需要 3 个几何独立器件，interaction
至少需要 4 个，且设计矩阵满秩。参考几何与默认范围从输入推导，不要求固定器件数量或网格。

独立验证记录与训练记录隔离，既有器件的数据角色不可随意切换。启用 required 独立验证
时必须有足够的对应数据并通过检查。默认 disabled 表示 NOT_RUN，不等于验证通过。
相同实测曲线不能通过换一个器件 ID 同时充当训练和独立验证数据。

只有当前曲线数据、训练配置或基础参数发生变化时，才重新拟合。Spec 改变或仅新增指标
不会单独触发训练。`--retrain` 可以明确重试。几何不可辨识、优化或误差门槛失败时，
有效实测记录仍保留，本次模型不发布，原模型和结果依据不会被覆写。

每次候选训练使用新的模型目录。成功后，SQLite 事务切换当前版本；失败的候选目录清除，
错误保存在运行记录和结果页。已发布版本保留以便追溯。重复失败的数据和配置不会自动
反复训练，须修改输入/配置或指定 `--retrain`。

训练状态为 TRAINED、REUSED、NO_TRAINING_DATA 或 FAILED。FAILED 的命令退出码为 1，
但数据准入结果和错误页仍可查看。没有模型时不会用虚构预测补齐结果。

## 结果语义

`device_conclusion` 由有效实测 PASS/FAIL 决定，依据为 MEASUREMENT；实测无效或缺失时为
UNDETERMINED。预测失败、超范围或不同意实测都不能改变有效实测结论。
FALSE_PASS 表示模型误放行，FALSE_FAIL 表示模型误拒绝；应以实测检查模型误差。

只有双方有效且几何、条件一致才计算误差。带符号误差为预测减实测，绝对误差使用原指标
单位，相对误差为绝对误差除以实测绝对值。零实测值的相对误差不定义，数值溢出明确标记。
无效或不可比行不填零误差，也不计入判断一致率的分母。

## 内部文件与使用边界

数据库目录保存 `catalog.sqlite3`、`models/` 和唯一的 `result.html` 阅读入口。
模型目录包含训练输入快照和冻结产物，是可追溯版本；不能当作临时文件删除。
数据和模型指针由事务一起提交。结果页也存入数据库，外部 HTML 用临时文件替换；若文件
系统写出失败，再次执行相同更新可以重新生成结果页，无须重复训练。

系统是单机 SQLite 工作流，写入串行执行。可整体备份数据库目录；不要只复制 HTML
或在更新进行时单独复制数据库文件。尚未实现云端多用户服务或后台定时训练。

`evaluate`、`fit`、`predict`、`compare` 仍可分别用于分析与调试。
训练产物的发布应通过 `update` 的版本管理；不要让定时任务直接覆盖正在使用的独立 fit 目录。
