# 问题一第 1 小问：数据质量评价

PHASE 3 的 A1 标定 Z 与 PHASE 4A 的 Pearson-CRITIC 权重已通过人工确认。
当前 PHASE 4B 使用锁定完整精度权重，计算样本 Q、领域质量、抽样/扩展对照与敏感性。
范围仅为问题一第 1 小问，不进行质量冲突消解、领域配比或其他题目。
qurater 的内部四维按确认规则单独截尾和标准化后平均，最终仍仅为一个指标。
问题一后续小问及问题二至四不在当前范围内。

Python 3.10+，普通 CPU 即可。项目根目录由脚本文件位置自动识别。

```powershell
python -m pip install -r requirements.txt
python scripts/01_inspect_quality_fields.py
```

脚本使用 lzma 逐行读取 A1，以及扩展目录中全部 arxiv_*.jsonl.xz 和
github_*.jsonl.xz，不抽样、不写入 real_attachments。对输入压缩文件读前读后
计算 SHA-256 验证一致性。不累计 content，只保留数值数组用于精确分位数，
以及每字段前 20 个原始非空样例。完整样例同时打印并保存。

22 个官方指标及方向集中定义在 configs/quality_metrics.yaml。
真实字段 rps_lines_ending_with_terminal_punctution_mark 的拼写差异通过
显式 raw_aliases 记录，官方指标名不变；原文件不改。如果规范名与别名同时
出现且取值冲突，检查返回失败。FIELD_ALIASES 由配置自动构建，原件不改。
PHASE 2 的报告是当时待确认状态的历史快照；当前规则以人工确认后的配置为准。

outputs/inspection 输出：

- list_field_samples.txt：A1/A2/A3 各列表字段前 20 个原始非空值，含来源和行号。
- list_field_summary.csv：列表长度、缺失/非法/解析失败、向量和分布、概率诊断。
- dataset_schema_summary.csv：27 个指标/辅助字段的存在性、类型、原始键和来源规模。
- list_field_position_summary.csv：各位置的 count/min/max/mean/median。
- input_file_manifest.csv：实际输入路径、压缩大小、记录数、哈希。
- json_parse_failures.csv：逐行 JSON 解析失败；无失败时只有表头。
- inspection.log：进度和诊断日志。
- inspection_summary.md：自动生成的本阶段结构检查报告和待确认事项。

计数口径：记录数指成功解析的 JSON 对象；物理行数及记录解析失败另计。
列表 missing = 字段缺失 + None + 空列表。非法类型/元素与字段字符串解析
失败分别计数，unusable_rate 包含所有不能用作数值列表的记录。
长度分布包含可解析列表（包括空列表），valid_length_distribution 仅含合法
非空有限数值列表；按位置统计也仅用这些合法列表。字符串形式的 JSON 列表
如果存在会单列解析成功数，原始类型和样例仍保留，后续需确认接受策略。
sum≈1 的分母为合法列表数，判据严格为 abs(sum-1)<0.001。
概率诊断要求合法列表长度稳定且至少为 2、全部元素在 [0,1]，以及至少 99%
合法列表满足上述和约束。此诊断不依赖字段名称，不能证明类别含义。
分位数用 numpy 的线性插值，位置索引从 0 开始。

结构检查完成后必须先人工确认向量语义和类别顺序，才能进入下一阶段。
其余输出目录已建立但本轮不生成对应结果。

## PHASE 2：补充向量语义诊断

```powershell
python scripts/01b_inspect_vector_semantics.py
```

全量读取相同 A1/A2/A3。对 finite 的分类原向量执行 float64 稳定 Softmax，
逐条验证范围、和、NaN/Inf 与 argmax 一致性。NaN/Inf 输入不填补、不转换，
单独输出来源。按固定文件/行顺序，每数据集每个 modernbert 字段输出 20 条
真实诊断样例、每个二分类字段 30 条、qurater 20 条。
modernbert 的 0…5 等级期望只是诊断，不生成正式标量指标。
qurater 不平均、不做 Softmax；只计算四维原分布与 Pearson/Spearman。

领域通过 quality.io.infer_source_domain 解析：先原 _source_domain，缺失时
只接受 arxiv_/github_ 文件名前缀；已有领域与前缀冲突时报错。
quality.io.resolve_metric_value 使用配置中的唯一官方名和明确别名，遇到
别名值冲突报错。辅助字段始终不进入数值诊断指标。

报告 outputs/inspection/vector_semantics_diagnosis.md 及对应 CSV/样例由
脚本自动生成。原始字段检查文件保留。检索本地 PDF 文本、Word XML、README
和元数据，不改动原件；PDF 字体映射警告数会记录，文本未命中不能证明原件
没有定义。上游 README 的来源证据单独记录在 local_evidence 下，包含检索
日期与观察到的版本；不自动批准为正式附件转换规则。
类别或尺度仍需确认的字段 status=NEEDS_CONFIRMATION。整轮不启动 02/03/04。

## 历史 02：人工确认后的标量化与描述统计

```powershell
python scripts/02_profile_quality_data.py
```

当前已获人工确认：fineweb_edu 取 [0]；modernbert 四项 argmax/5；
fluency_en argmax([not_fluent,fluent])；ad_en argmax([has_ad,no_ad])，是正向指标。
qurater 四维各自 P01-P99 Winsorization，再各自 Min-Max，最后平均。
历史 02 阈值在 A1/A2/A3 全部有限记录上拟合，保留全部输入行且不按 id 去重。
该历史结果现在仅为 pooled descriptive / sensitivity analysis，不作为主模型参数。
归一化只是 qurater 的内部标量化，不是最终 22 维 Z；其他连续指标保留原值。
Softmax 期望仅作为 softmax_expected_rating() 辅助函数，主流程不调用。

NaN 分类向量对应 scalar=NaN，整条记录保留；当前只统计和记录缺失 ID，
不提前填补。同领域中位数、总体中位数回退将在后续正式预处理时执行。
列表类型或长度与确认结构冲突则报错停止；数值 NaN/Inf 按缺失保留。

outputs/profile/quality_indicator_profile.csv 包含总体和三个数据集，每组 22 行。
domain_indicator_profile.csv 明确区分合并领域与数据集内领域。
count 为有限数，n 为全部记录数；std 为样本标准差 ddof=1；分位数线性插值。
missing_summary.csv 保留各组缺失数与比例；分组间有交集，不把整表相加当作事件总数。
scalar_missing_records.csv 记录实际缺失的 ID、领域、来源和原因；不含临时 qurater 占位。
quality_profile_summary.md 自动说明方法与真实结果；scalar_validation.csv 核验维数、
分类取值和缺失审计一致性；profile_input_manifest.csv 核验原件哈希。
outputs/preprocess/qurater_transform_parameters.csv 记录四维实际阈值、截尾后范围和数量。
profile_config_snapshot.yaml 保存本次运行配置以便复现。全程 CPU，不缓存全文 content。

执行 02 后停止，不执行 03/04，不计算 CRITIC 或 Q。

## PHASE 2C：重复、ID 冲突、缺失记录及分类天花板

```powershell
python scripts/02c_diagnose_records.py
```

重新流式读取全部 A1/A2/A3，诊断完整原记录 exact duplicates、重复 ID、
原始 22 个质量字段及可比 content/sub_path/domain 冲突、两组跨数据集重叠。
原始 logits 不用 argmax 结果替代比较。同位置 NaN 相等；质量字段别名统一；
单侧缺少 content 记 NOT_COMPARABLE。完整 exact 哈希包含所有原始字段，
因此字段结构不同的 A1 与扩展记录可有相同 ID/质量字段而不构成完整 exact duplicate。
不选代表记录，不去重、不平均、不填补。完整比较口径与计数单位见自动报告。

数值 blob 和哈希存入 outputs/inspection 下临时 SQLite 索引，不缓存 content。
SQLite 缓存约 64 MiB，临时索引运行后关闭移除；清理路径验证在 inspection 目录内。
临时目录采用 pathlib.mkdir 继承工作区权限，避免 Windows mkdtemp 的 0700 权限
与受限执行令牌冲突；只移除本脚本数据库及空目录，不执行递归删除。
缺失记录逐项对照已通过的 02 ID 清单，给出来源记录、unique ID、完整记录哈希
和质量记录哈希等不同唯一性口径；高缺失阈值严格为 >20%。
分类等级比例与 ceiling_rate 的分母为有限标量数，缺失单列；另给全部记录分母。
合并领域与数据集内领域分开统计，ad_en 的 1 表示模型判为无广告。

outputs/inspection/data_cleaning_diagnosis.md 及要求的六个诊断文件自动生成。
classification_level_distribution.csv 与 classification_ceiling_summary.csv 保存
完整等级频率和天花板率。
raw_quality_duplicates_different_ids.csv 另行追踪不同 ID 但原始 22 质量字段相同的记录，
不据此认定文本相同。prior_artifacts_unchanged.csv 核验所有 02 输出与
qurater 参数未变；原件哈希另存 cleaning_diagnosis_input_manifest.csv。
运行后停止，不执行 03/04，不计算 CRITIC 或 Q。

## PHASE 3：A1 标定的 22 维 Z

```powershell
python scripts/03_build_quality_features.py
```

只以 A1 拟合所有参数。A2/A3 仅应用重新载入的 preprocessing_model_A1.json，
不使用自己的中位数、分位数或范围。旧 profile、inspection 和 pooled qurater 参数
原样保留，并核验全部哈希；historical_artifact_roles.csv 标明旧 pooled 结果的用途。
中位数用 A1 有限观测拟合，然后填补，再计算连续分位数；适中型分位数取截尾前数据。
modernbert、fluency、ad、qurater 七项已经有界，不再次 Min-Max 或整体 Winsorize。
其余 15 个连续指标按 A1 P01/P99 截尾，其中三个适中型使用 P10/P90 梯形平台，
其他正负向指标使用 A1 截尾后 Min-Max。连续 RPS 比例不是分类概率，按连续指标处理。
方向、平台分位数、退化常数和报告提示阈值集中于 configs/quality_metrics.yaml。

outputs/preprocess 保存三个全量 Z CSV、A1 模型与五类参数表、逐条填补日志、
截尾比例、66 行指标范围核验及两个非重叠 ID 索引。只保留数值与来源，不缓存 content。
不从任何附件删除记录；跨附件同 ID 的 Z 要求完全相同，非重叠索引不改变 Z 文件。
outputs/report/phase3_preprocessing_summary.md 自动记录实际参数、样本数及所有诊断。
运行 03 后停止，等待人工确认；不运行 04、不计算 CRITIC 或 Q。

## PHASE 4A：A1 CRITIC 权重与诊断

```powershell
python scripts/04a_diagnose_critic_weights.py
```

只从 A1_quality_features.csv 的 22 个指标拟合主权重。sigma 使用 ddof=1，
Pearson R=sum(1-r)，C=sigma*R，w=C/sum(C)。负相关原样参与公式，不改权重。
Spearman 为单独敏感性诊断，使用平均秩相关和原 Z 的标准差。
常数列相关或总信息量为零时停止，不虚构相关系数或回退等权。
configs/critic_diagnostics.yaml 保存诊断阈值，不更改已确认的 PHASE 3 配置或参数。
全部 PHASE 3 文件在运行前后核验 SHA-256；A2/A3 不参与权重估计。

outputs/quality 保存 critic_weights_A1.csv、两套相关矩阵、负相关指标对、
Spearman 敏感性权重、等权权重表，以及适中/分类指标诊断、熵和集中度摘要。
critic_weight_diagnostics.md 自动汇总真实计算结果。少取值对分类指标是预期结构，
不自动认定无效；所有诊断标记均不导致改权。
运行后停止，等待人工确认再进入 PHASE 4B；不计算任何 Q。

## PHASE 4B：固定权重的 Q 与完整验证

```powershell
python scripts/04_build_quality_score.py
python scripts/04b_verify_quality_score.py
```

从 critic_weights_A1.csv 读取完整精度，按指标名对齐后应用到三个全量 Z；
不重拟合 CRITIC，不改 PHASE 3 参数。样本级 Q=sum(w*z)，领域主质量为每侧
截去 floor(0.10*n) 后的均值。95% CI 针对这个 trimmed mean，以根种子2026、
固定领域顺序派生独立随机流，每组2000次完整 n-out-of-n 重采样；批处理控制内存，
无大样本缩减或近似重采样。bootstrap_trimmed_means.csv 保存全部22000次真实统计量。

完整扩展与非重叠余集分别比较。差值为扩展减抽样；relative_difference 是截尾均值
差除以抽样截尾均值；Cliff's delta 正值表示扩展倾向更高。全量组共享记录，
KS p仅作描述性参考；所有比较同时记录实际差值、KS、Wasserstein和Cliff效应。
等权与保存的 Spearman 权重仅做敏感性，领域排名均按各方案的 trimmed mean。

outputs/quality 保存三个全量 Q、数据集/领域/扩展/非重叠摘要、比较、敏感性及核验表。
outputs/figures/question1_part1 保存11张真实图；散点图固定随机展示最多15000点，
所有统计始终使用全量。原件仅为恢复40条极端文本回读 A1；不缓存全部 content。
既有 Z、模型、参数和 PHASE 4A 输出经哈希核验不变。

高低各20条的完整文本导出到 A1_top20_quality_samples.csv、A1_bottom20_quality_samples.csv。
助手实际阅读后的注释为 A1_text_review.csv，并明确 full_text 或 head_middle_tail_only。
该记录不声称独立人工确认，代码结构正常和多语言消歧等反例也如实记录。
人工/助手注释与统计生成分离，可用下面命令仅刷新报告，不重复评分/重采样：

```powershell
python scripts/04_build_quality_score.py --refresh-report
```

最终报告为 outputs/report/question1_part1_final.md，涵盖真实数据、全部方法和统计、
权重、敏感性、文本核验及局限。运行后停止，不开始问题一第2或第3小问。

## 问题一第2小问：冲突诊断第一阶段

第一小问已人工确认并锁定；当前仅执行冲突诊断。直接使用已有三份质量 CSV
中的完整精度 Q 和 22 个 Z，以及 A1 已拟合的权重，不运行前面的预处理/赋权脚本。

```powershell
python scripts/05_analyze_quality_conflict.py
python scripts/05_verify_quality_conflict.py
```

配置位于 configs/conflict_analysis.yaml。K 是固定权重的绝对偏离；K_equal 和 V
仅作辅助诊断。高/极高阈值来自 A1 的 P90/P95，扩展集统一应用这些阈值。
逐对汇总 231 对指标差异，不保存全部记录×231 明细。并列极端指标的驱动组合
均分单条记录贡献；无定义的常量相关系数留空，不替换为零。

输出位于 outputs/conflict；九幅图位于 outputs/figures/question1_part2。
详细方法报告为 outputs/report/question1_part2_conflict_stage1.md，实际发现与解释
边界见 outputs/report/question1_part2_findings.md。脚本对第一小问全部产物与
官方附件核对前后 SHA-256，任何输入不一致即停止，不修复或重拟合锁定数据。

完整核验文本单独保存；原始长文本 CSV 用 Python 读取前需设置
csv.field_size_limit(2**31-1)。文本分层采用高冲突组内部 Q 三分位，明确标注
相对分组；全局高 Q/高 K 数量不足时照实输出，绝不补足虚构记录。
AI 片段复核只作为待人工核验线索，不是最终人工标注或正式实验依据。

当前停止于冲突规律与赋权敏感性诊断；未建立 Q*、未测试或选择 lambda。
待人工确认后，再决定 Q+K 双指标或修正评分。

## PHASE 5B：高冲突稳健重聚合验证

```powershell
python scripts/06_validate_robust_reaggregation.py
python scripts/06_verify_robust_reaggregation.py
```

配置为 configs/robust_reaggregation.yaml。仅对已有 K>A1 完整精度 P90 的记录，
围绕原 Q 做一次局部权重重聚合；普通记录精确复制原 Q。五个 delta 只在 A1
比较，保守候选固定应用 A2/A3，不重新拟合 Q、K、权重或阈值。

新输出位于 outputs/conflict/phase5b，四幅图位于
outputs/figures/question1_part2/phase5b，报告为
outputs/report/question1_part2_phase5b_robust_validation.md。
候选 CSV 明确 candidate_not_activated；本轮建议 B：保留原 (Q,K)。
局部系数只是残差规则，尚未验证为真实可靠性；arxiv 排名变化及辅助等权
敏感性未改善已如实报告。未采用指数惩罚，未进入领域配比建模。

## 问题一第3小问：阶段1领域配比与 Loss

```powershell
python scripts/07_inspect_mixture_loss_data.py
python scripts/08_model_mixture_loss.py
python scripts/08_verify_mixture_models.py
```

本阶段使用 real_attachments/A_data_value/regmix_tables 中十二份官方 CSV。
只在 A4/A5 上拟合及嵌套CV选择；1M/60M/1B直接检验，10B/70B只做数据诊断。
不引入 Q/K、不重新运行前两问、不做最终大尺度预测或 Scaling Law。

派生配比先闭合，再以训练最小正配比的一半执行乘法零替换。主模型使用
16维 Helmert ILR，全部预处理位于CV训练折内。比较常数基线、一阶 Ridge、
一阶 ElasticNet、ILR两两交叉项 ElasticNet，L_mean 为主响应，L_trim 作敏感性。

输出位于 outputs/mixture，报告为 outputs/report/question1_part3_stage1_modeling.md，
图表位于 outputs/figures/question1_part3/stage1。数值最佳对照由训练嵌套CV决定；
按解释优先要求保留一阶 ElasticNet 主候选，不因检验结果改参数。
绝对 Loss 迁移与配方排序迁移分别报告；不同配方不能按跨文件 index 强行配对。
当前阶段完成后停止，等待人工确认再进入 Q_p 与估算表外推。

## 问题一第3小问：最终配比解释与质量特征检验

新增脚本 `scripts/09_analyze_mixture_effects.py` 已完成实际运行。结果分开放在
`outputs/mixture/stage2`，所要求的总表仍在 `outputs/mixture`，最终报告为
`outputs/report/question1_part3_final.md`。配置为 `configs/mixture_analysis.yaml`。

```powershell
python scripts/09_verify_mixture_analysis.py
python scripts/09_analyze_mixture_effects.py --report-only
python scripts/09_analyze_mixture_effects.py --figures-only
```

完成后禁止再次整轮拟合；脚本会拒绝对已完成阶段运行全流程。上述报告和图表
刷新命令只读取已有结果，不重新选参、拟合或校准模型。500 次 MC 有单独检查点。

原一阶 ElasticNet/Ridge 只加载与应用。新增 M3/M4 使用 A4/A5 内相同外折嵌套 CV，
13 个验证域各自的一阶辅助模型仅在训练数据内五折选参。A16 的 11 个 inferred
关联原本是 `(none)`，代理假设及理由明确写入新配置和映射表；CI 扰动不能覆盖
代理语义错误，另有替代代理检验。Kp 是冲突描述，未作为质量罚分。

最终保留冻结的一阶 ILR-ElasticNet 主模型，Qp/Kp 不进入主模型。10B/70B 表属于
估算训练配方子集；本次发现与主模型排序负相关，已如实报告，不宣称外推成功。
本轮结束于问题一第3小问，没有开始问题二。
