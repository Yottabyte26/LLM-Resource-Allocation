# 论文附录 Python 源程序

只提取既有代码。本次没有读取质量附件记录、拟合、训练、优化、运行实验或绘图，也没有改变论文正文、源代码、配置和冻结结果。
questionX_source.py 是按真实文件调用链排序的论文展示汇编，**不是独立可执行的合并脚本**。原 import、相对 import、多个 __future__ 和 main 入口全部保留；请勿直接执行该汇编。full_source 也只备份 Python，不打包附件、参数、配置或依赖环境。此处入口说明仅作溯源，不表示允许重跑冻结实验。
questionX_compact_source.py 和 compact TeX 是原文件函数/调用片段，省略位置由原行号标注。它们同样用于论文展示。source_traceability.csv 和 source_function_traceability.csv 给出一一对应；full_source_hashes.csv 给出原始与备份 SHA256。

使用 LaTeX 时选择 full 或 compact 中的一份，在正文末尾使用 \input{python_appendix_compact.tex}（按论文工程路径调整）。这是附录片段，不是独立文档；主文档需 XeLaTeX/ctex、listings、xcolor。正文已含 \appendix 时合并时保留一次即可。长文件在原语法边界附近切为通常不超过 100 行的列表，行号仍是原文件行号。

本次只做源程序静态核验，未编译完整论文；页数为纸张/字号/折行假设下的估计，最终排版取决于论文模板。

## 问题一

A1冻结预处理→Pearson-CRITIC Q；K为加权绝对偏离；17域Helmert ILR＋一阶ElasticNet；Ridge为数值对照。

入口顺序：`scripts/03_build_quality_features.py` → `scripts/04a_diagnose_critic_weights.py` → `scripts/04_build_quality_score.py` → `scripts/05_analyze_quality_conflict.py` → `scripts/08_model_mixture_loss.py` → `scripts/09_analyze_mixture_effects.py`。
读取的现有配置：`configs/conflict_analysis.yaml`, `configs/critic_diagnostics.yaml`, `configs/mixture_analysis.yaml`, `configs/mixture_modeling.yaml`, `configs/quality_metrics.yaml`, `configs/quality_scoring.yaml`。

原始代码文件：

- `src/quality/io.py`：真实附件读取、索引/字段兼容与输入单位
- `src/quality/normalize.py`：正负向Min-Max及适中型梯形效用
- `src/quality/vector_diagnostics.py`：共用JSON/CSV和数值辅助；softmax不是主分类评分
- `src/quality/vector_transform.py`：8个列表字段真实标量化与qurater分量参数
- `src/quality/preprocess.py`：只拟合A1的填补/截尾参数，原样映射A2/A3
- `scripts/03_build_quality_features.py`：最终阶段实际执行入口：03_build_quality_features.py
- `src/quality/critic.py`：Pearson-CRITIC ddof=1及Spearman敏感性
- `scripts/04a_diagnose_critic_weights.py`：最终阶段实际执行入口：04a_diagnose_critic_weights.py
- `src/quality/aggregate.py`：10%截尾均值、模型记录bootstrap、分布比较
- `src/quality/plotting.py`：最终CSV/图表/报告呈现，compact省略纯绘图和排版
- `src/quality/reporting.py`：最终CSV/图表/报告呈现，compact省略纯绘图和排版
- `scripts/04_build_quality_score.py`：最终阶段实际执行入口：04_build_quality_score.py
- `src/quality/conflict.py`：K=加权绝对偏离、冲突阈值和驱动组合
- `src/quality/conflict_plotting.py`：最终CSV/图表/报告呈现，compact省略纯绘图和排版
- `scripts/05_analyze_quality_conflict.py`：最终阶段实际执行入口：05_analyze_quality_conflict.py
- `src/mixture/io.py`：真实附件读取、索引/字段兼容与输入单位
- `src/mixture/composition.py`：训练折内乘法零替换与Helmert ILR
- `src/mixture/modeling.py`：Ridge/ElasticNet候选、训练内调参与分组CV
- `scripts/08_model_mixture_loss.py`：最终阶段实际执行入口：08_model_mixture_loss.py
- `src/mixture/effects.py`：真实领域配比1个百分点扰动及联合边际效应
- `src/mixture/augmented.py`：Qp/Kp候选对照；最终没有替代一阶ElasticNet
- `scripts/09_analyze_mixture_effects.py`：最终阶段实际执行入口：09_analyze_mixture_effects.py

compact 包含：

- `src/quality/io.py`：infer_source_domain, resolve_metric_value, discover_quality_files, iter_jsonl
- `src/quality/vector_transform.py`：parse_vector, scalarize_record, fit_qurater_parameters, transform_qurater
- `src/quality/preprocess.py`：impute_matrix, fit_preprocessing_model, apply_preprocessing
- `src/quality/normalize.py`：moderate_utility_transform, minmax_transform
- `src/quality/critic.py`：critic_weights
- `src/quality/aggregate.py`：finite_scores, score_profile, bootstrap_trimmed_mean, domain_summary, compare_distributions
- `scripts/03_build_quality_features.py`：read_dataset
- `scripts/04_build_quality_score.py`：load_weights, score_dataset
- `src/quality/conflict.py`：conflict_indices, fit_conflict_thresholds
- `src/mixture/io.py`：read_table, load_datasets, signatures
- `src/mixture/composition.py`：ILRTransformer
- `src/mixture/modeling.py`：scores, group_folds, candidate, tune
- `scripts/08_model_mixture_loss.py`：metadata, fit_checked
- `src/mixture/effects.py`：perturb, direction, summary, baselines, analyze
- `scripts/09_analyze_mixture_effects.py`：scale_diagnostics, extrapolation

仅 full_source 保存（前置诊断、独立核验、初始化及其依赖）：

- `src/quality/profiling.py`：历史描述统计，非A1主预处理参数
- `scripts/02_profile_quality_data.py`：历史pooled描述统计；03使用其缺失记录核对，不复用pooled阈值
- `src/quality/record_diagnostics.py`：预处理前重复/ID冲突及缺失诊断
- `scripts/02c_diagnose_records.py`：03所需重复/重叠/天花板诊断输入的实际生成入口
- `scripts/07_inspect_mixture_loss_data.py`：08所需A4–A15真实数据诊断与input_manifest生成入口
- `scripts/03_verify_quality_features.py`：独立公式/输出/冻结哈希验证；只归档，不在本轮执行
- `scripts/04b_verify_quality_score.py`：独立公式/输出/冻结哈希验证；只归档，不在本轮执行
- `scripts/04b_record_text_review.py`：既有文本核验材料归档，不当作独立人工实验
- `scripts/05_verify_quality_conflict.py`：独立公式/输出/冻结哈希验证；只归档，不在本轮执行
- `scripts/08_verify_mixture_models.py`：独立公式/输出/冻结哈希验证；只归档，不在本轮执行
- `scripts/09_verify_mixture_analysis.py`：独立公式/输出/冻结哈希验证；只归档，不在本轮执行
- `src/quality/__init__.py`：调用链共用依赖或包初始化
- `src/mixture/__init__.py`：调用链共用依赖或包初始化

未作为最终主模型纳入：

- `scripts/01b_inspect_vector_semantics.py`：只作早期结构/描述性诊断，不在最终数值生成依赖链
- `scripts/01_inspect_quality_fields.py`：只作早期结构/描述性诊断，不在最终数值生成依赖链
- `scripts/06_validate_robust_reaggregation.py`：确实执行过，但最终选择(Q,K)，未激活Q_robust/Q_final；默认不纳入主附录或最终调用链备份
- `scripts/06_verify_robust_reaggregation.py`：确实执行过，但最终选择(Q,K)，未激活Q_robust/Q_final；默认不纳入主附录或最终调用链备份
- `src/quality/robust_reaggregation.py`：确实执行过，但最终选择(Q,K)，未激活Q_robust/Q_final；默认不纳入主附录或最终调用链备份
- 同一文件中的候选比较保持原样，并加文件块注释；具体见 source_manifest.md 和 candidate_note 列。
- 完整目录级输出映射与逐个冻结结果路径见 source_manifest.md / execution_evidence.csv。

## 问题二

经典L0＋B6局部Delta_Q＋lnN分段线性lambda_p乘冻结M(p)差值；主边界S-A。

入口顺序：`scripts/11_fit_classical_scaling.py` → `scripts/14_fit_scale_dependent_quality.py` → `scripts/15_calibrate_mixture_scaling.py` → `scripts/16_analyze_generalized_scaling.py`。
读取的现有配置：`configs/scaling_mixture_phase3.yaml`, `configs/scaling_phase1.yaml`, `configs/scaling_phase4.yaml`, `configs/scaling_quality_phase2B.yaml`, `configs/scaling_quality_phase2C.yaml`。

原始代码文件：

- `src/scaling/io.py`：真实附件读取、索引/字段兼容与输入单位
- `src/scaling/classical.py`：经典五参数非线性拟合、解析Jacobian、误差指标
- `src/scaling/bootstrap.py`：整条trajectory重采样拟合
- `scripts/10_diagnose_scaling_data.py`：最终阶段实际执行入口：10_diagnose_scaling_data.py
- `scripts/11_fit_classical_scaling.py`：最终阶段实际执行入口：11_fit_classical_scaling.py
- `src/scaling/quality_extension.py`：最终链复用读入/reference工具；常数gamma MQ候选不是最终主模型
- `src/scaling/scale_quality.py`：S3原参数来源；S1/S2只为比较候选，最终去v且不重拟合
- `src/scaling/scale_quality_analysis.py`：ND分组CV、Q层留出、整块bootstrap及B6/B7差分检验
- `src/scaling/quality_analysis.py`：Phase2C复用B8压力检验工具；MQ候选不是最终模型
- `src/scaling/quality_reporting.py`：Phase2C复用绘图和表格工具
- `src/scaling/scale_quality_reporting.py`：最终CSV/图表/报告呈现，compact省略纯绘图和排版
- `scripts/14_fit_scale_dependent_quality.py`：最终阶段实际执行入口：14_fit_scale_dependent_quality.py
- `src/scaling/mixture_transfer.py`：冻结ILR预测、逐尺度lambda与配方bootstrap；P1/P2曲线仅比较候选
- `src/scaling/mixture_transfer_reporting.py`：最终CSV/图表/报告呈现，compact省略纯绘图和排版
- `scripts/15_calibrate_mixture_scaling.py`：最终阶段实际执行入口：15_calibrate_mixture_scaling.py
- `src/scaling/marginal_analysis.py`：最终广义标度律、边界策略、解析导数/弹性及Q+0.1精确求根
- `src/scaling/marginal_reporting.py`：最终CSV/图表/报告呈现，compact省略纯绘图和排版
- `scripts/16_analyze_generalized_scaling.py`：最终阶段实际执行入口：16_analyze_generalized_scaling.py

compact 包含：

- `src/scaling/io.py`：load_csv, combine_trajectories
- `src/scaling/classical.py`：weights, predict, prediction_jacobian, initial_values, fit, metrics
- `src/scaling/bootstrap.py`：initialize, repetition
- `scripts/11_fit_classical_scaling.py`：fit_and_cv, cluster_bootstrap, validate
- `src/scaling/quality_extension.py`：read_parameters, read_data, subset, group_reference
- `src/scaling/quality_analysis.py`：stress_test
- `src/scaling/scale_quality.py`：centers, components, physical, encoded, response_jacobian, predict_scale, initial_parameters, fit_scale
- `src/scaling/scale_quality_analysis.py`：evaluate, group_cv, q_level_cv, initialize_worker, bootstrap_worker, resampling_jobs, bootstrap_models
- `scripts/14_fit_scale_dependent_quality.py`：basics
- `src/scaling/mixture_transfer.py`：frozen_metadata, predict_ilr, predict_recipe, read_frozen_recipes, transfer, bootstrap_transfers, empirical_lambda
- `src/scaling/marginal_analysis.py`：FrozenGeneralizedScaling, finite_difference
- `scripts/16_analyze_generalized_scaling.py`：frozen_model, large_validation, lambda_sensitivity

仅 full_source 保存（前置诊断、独立核验、初始化及其依赖）：

- `scripts/11_verify_classical_scaling.py`：独立公式/输出/冻结哈希验证；只归档，不在本轮执行
- `scripts/14_verify_scale_dependent_quality.py`：独立公式/输出/冻结哈希验证；只归档，不在本轮执行
- `scripts/15_verify_mixture_scaling.py`：独立公式/输出/冻结哈希验证；只归档，不在本轮执行
- `scripts/16_verify_generalized_scaling.py`：独立公式/输出/冻结哈希验证；只归档，不在本轮执行
- `src/scaling/__init__.py`：调用链共用依赖或包初始化

未作为最终主模型纳入：

- `scripts/12_diagnose_quality_scaling.py`：质量扩展前的诊断阶段；无最终参数拟合，最终链由14重新严格读入B6/B7/B8
- `scripts/13_fit_quality_scaling.py`：常数gamma MQ1/MQ2候选未冻结为最终模型；最终局部Delta_Q来自14阶段S3
- `scripts/13_verify_quality_scaling.py`：常数gamma MQ1/MQ2候选未冻结为最终模型；最终局部Delta_Q来自14阶段S3
- 同一文件中的候选比较保持原样，并加文件块注释；具体见 source_manifest.md 和 candidate_note 列。
- 完整目录级输出映射与逐个冻结结果路径见 source_manifest.md / execution_evidence.csv。

## 问题三

解析消去D_B；P0主方案、P1经验增强；P2-A为理论外推，P2-B无有限最优解。

入口顺序：`scripts/17_setup_compute_optimization.py` → `scripts/18_optimize_compute_resources.py` → `scripts/19_finalize_compute_optimization.py`。
读取的现有配置：`configs/optimization_phase1.yaml`, `configs/optimization_phase2.yaml`, `configs/optimization_phase3.yaml`, `configs/scaling_phase4.yaml`。

原始代码文件：

- `src/optimization/costs.py`：6ND、注意力成本与exp/power/log质量增量成本
- `src/scaling/mixture_transfer.py`：冻结ILR预测、逐尺度lambda与配方bootstrap；P1/P2曲线仅比较候选
- `src/optimization/setup.py`：官方成本、Q0、决策边界与832经验候选的来源
- `scripts/17_setup_compute_optimization.py`：最终阶段实际执行入口：17_setup_compute_optimization.py
- `src/scaling/marginal_analysis.py`：最终广义标度律、边界策略、解析导数/弹性及Q+0.1精确求根
- `src/optimization/resource_solver.py`：D_B解析消元、可行性、网格/SLSQP/DE及P0/P1求解
- `src/optimization/transitions.py`：连续预算扫描路径上的Type I/II转移检验与区间
- `src/optimization/resource_reporting.py`：最终CSV/图表/报告呈现，compact省略纯绘图和排版
- `scripts/18_optimize_compute_resources.py`：最终阶段实际执行入口：18_optimize_compute_resources.py
- `src/optimization/simplex_sensitivity.py`：P2-A配方证书与数值交叉检验；P2-B边界无界性，仅理论敏感性
- `src/optimization/final_reporting.py`：冻结P0/P1复用、证据分层与问题三最终表/报告
- `scripts/19_finalize_compute_optimization.py`：最终阶段实际执行入口：19_finalize_compute_optimization.py

compact 包含：

- `src/optimization/costs.py`：_validate_quality, quality_increment_per_token, marginal_quality_per_token, cost_components
- `src/scaling/marginal_analysis.py`：FrozenGeneralizedScaling
- `src/optimization/setup.py`：decision_domains, mixture_sets, budget_scenarios
- `src/optimization/resource_solver.py`：load, lambda_zero, ResourceProblem, three_dimensional_checks, run_solve
- `src/optimization/simplex_sensitivity.py`：log_contrast, exact_extreme, simplex_search, evidence_level, official_comparison
- `src/optimization/transitions.py`：bisect_status, soft_fit, run_diagnose

仅 full_source 保存（前置诊断、独立核验、初始化及其依赖）：

- `scripts/17_verify_compute_setup.py`：独立公式/输出/冻结哈希验证；只归档，不在本轮执行
- `scripts/18_verify_compute_optimization.py`：独立公式/输出/冻结哈希验证；只归档，不在本轮执行
- `scripts/19_verify_final_optimization.py`：独立公式/输出/冻结哈希验证；只归档，不在本轮执行
- `src/optimization/__init__.py`：调用链共用依赖或包初始化
- `src/scaling/__init__.py`：调用链共用依赖或包初始化

未作为最终主模型纳入：

- 同一文件中的候选比较保持原样，并加文件块注释；具体见 source_manifest.md 和 candidate_note 列。
- 完整目录级输出映射与逐个冻结结果路径见 source_manifest.md / execution_evidence.csv。

## 问题四

固定六任务A_z；M1/M2-C/M3＋双因素Shapley；桥接失败负结果；已执行Phase3情景投影及风险诊断。

入口顺序：`scripts/20_build_evolution_ability_bridge.py` → `scripts/21_build_evolution_contribution.py` → `scripts/22_build_evolution_projection.py`。
读取的现有配置：`configs/evolution_phase1.yaml`, `configs/evolution_phase2.yaml`, `configs/evolution_phase3.yaml`。

原始代码文件：

- `src/evolution/common.py`：模型名规范、CSV和相关系数辅助
- `src/evolution/metadata.py`：C1/C2去重、S0/S1类型证据及C4元数据匹配
- `src/evolution/ability.py`：固定六Benchmark Winsor/median/IQR、A_z、PCA/Average敏感性
- `src/evolution/detailed.py`：C8 JSON全遍历、损坏文件记录与偏科/弱尾
- `src/evolution/bridge.py`：B0/B1/B2、LOMO/bootstrap和桥接失败诊断
- `src/evolution/reporting.py`：最终CSV/图表/报告呈现，compact省略纯绘图和排版
- `scripts/20_build_evolution_ability_bridge.py`：最终阶段实际执行入口：20_build_evolution_ability_bridge.py
- `src/evolution/contribution_regression.py`：OLS/HC3/Huber、LOOCV、Cook/VIF、双因素Shapley与5000模型bootstrap
- `src/evolution/contribution_analysis.py`：T0-T3、M1/M2-C/M3、组织/时间轴对照、技术残差和分解执行
- `src/evolution/contribution_reporting.py`：最终CSV/图表/报告呈现，compact省略纯绘图和排版
- `scripts/21_build_evolution_contribution.py`：最终阶段实际执行入口：21_build_evolution_contribution.py
- `src/evolution/projection_data.py`：实际执行Phase3：锚点、C4去重滚动前沿、增长bootstrap
- `src/evolution/projection_models.py`：冻结成对系数的情景Monte Carlo、回测、补偿、风险标记
- `src/evolution/projection_reporting.py`：最终CSV/图表/报告呈现，compact省略纯绘图和排版
- `scripts/22_build_evolution_projection.py`：最终阶段实际执行入口：22_build_evolution_projection.py

compact 包含：

- `src/evolution/common.py`：normalized_name
- `src/evolution/metadata.py`：canonical_models, epoch_join
- `src/evolution/ability.py`：transform, construct_ability
- `src/evolution/detailed.py`：value, normalize_score, extract, analyze_details
- `src/evolution/bridge.py`：fit_candidate, predict, metrics, decreasing, analyze_bridge
- `src/evolution/contribution_regression.py`：design, accuracy, fit_regression, diagnostics, shapley, endpoints, contribution, bootstrap
- `src/evolution/contribution_analysis.py`：prepare, model_data, fit_models, make_technology_tables, run_decomposition
- `src/evolution/projection_data.py`：load_frozen, select_scope, frontier_stats, build_anchors, benchmark_series, compute_candidates, select_compute_window, compute_grid, rolling_compute, slope, estimate_growth, c3_direction
- `src/evolution/projection_models.py`：frozen_m2, linear_fit, build_hindcast_and_trends, quantiles, level, build_projections

仅 full_source 保存（前置诊断、独立核验、初始化及其依赖）：

- `scripts/20_verify_evolution_phase1.py`：独立公式/输出/冻结哈希验证；只归档，不在本轮执行
- `scripts/21_verify_evolution_contribution.py`：独立公式/输出/冻结哈希验证；只归档，不在本轮执行
- `scripts/22_verify_evolution_projection.py`：独立公式/输出/冻结哈希验证；只归档，不在本轮执行
- `src/evolution/__init__.py`：调用链共用依赖或包初始化

未作为最终主模型纳入：

- 同一文件中的候选比较保持原样，并加文件块注释；具体见 source_manifest.md 和 candidate_note 列。
- 完整目录级输出映射与逐个冻结结果路径见 source_manifest.md / execution_evidence.csv。

## 行数和页数

| 问题 | 完整调用链文件 | 原样备份文件 | 整理版行数 | compact 展示行数 | compact 原代码行数 |
|---|---:|---:|---:|---:|---:|
| 问题1 | 22 | 35 | 3785 | 1453 | 1250 |
| 问题2 | 18 | 23 | 4065 | 1470 | 1265 |
| 问题3 | 12 | 17 | 2835 | 1090 | 1004 |
| 问题4 | 15 | 19 | 2629 | 1572 | 1455 |

完整 TeX 估计 243–308 页；154 个列表。假设：A4, text area about160x235mm, scriptsize mono, 55-68 effective lines/page; UTF8/long-line wrap depends on actual thesis class; NOT rendered。

compact TeX 估计 94–122 页；100 个列表。假设：A4, text area about160x235mm, scriptsize mono, 55-68 effective lines/page; UTF8/long-line wrap depends on actual thesis class; NOT rendered。

compact 保留原函数完整正文及必要调用片段，因此部分候选分支、检查和日志仍随核心函数保留；未为缩页数改变实现。若投稿篇幅更紧，可按 source_manifest 的原函数范围继续人工取舍；本次不改论文正文。
