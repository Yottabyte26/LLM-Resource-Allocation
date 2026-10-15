# 四问最终源程序映射

本索引依据最终报告、冻结结果/参数、历史实现哈希及实际 import 关系建立。文件级调用链包含真实运行入口所导入的候选比较和呈现函数；被导入不等于其中所有分支都被最终模型采用。
完整整理版不删除任何原文件正文；full_source 保持逐字节原样。独立核验/前置诊断在完整备份中保存，不当作主模型。

# 问题一

## 最终实际使用文件

A1冻结预处理→Pearson-CRITIC Q；K为加权绝对偏离；17域Helmert ILR＋一阶ElasticNet；Ridge为数值对照。

实际执行入口：`scripts/03_build_quality_features.py`, `scripts/04a_diagnose_critic_weights.py`, `scripts/04_build_quality_score.py`, `scripts/05_analyze_quality_conflict.py`, `scripts/08_model_mixture_loss.py`, `scripts/09_analyze_mixture_effects.py`。
结果来源：`outputs/preprocess/preprocessing_model_A1.json`, `outputs/quality/critic_weights_A1.csv`, `outputs/quality/A1_sample_quality.csv`, `outputs/quality/domain_quality_summary_A1.csv`, `outputs/conflict/A1_sample_conflict.csv`, `outputs/mixture/final_model_metadata.json`, `outputs/mixture/stage2/final_model_policy.json`, `outputs/mixture/domain_marginal_effects.csv`。
报告证据：`outputs/report/question1_part1_final.md`, `outputs/report/question1_part2_findings.md`, `outputs/report/question1_part2_phase5b_robust_validation.md`, `outputs/report/question1_part3_final.md`。

| 原始文件 | 用途 | 主模型/验证/辅助角色 | 对应最终输出 |
|---|---|---|---|
| `src/quality/io.py` | 真实附件读取、索引/字段兼容与输入单位 | model_or_shared_dependency；主调用链 | outputs/preprocess/; outputs/quality/ |
| `src/quality/normalize.py` | 正负向Min-Max及适中型梯形效用 | model_or_shared_dependency；主调用链 | outputs/preprocess/; outputs/quality/ |
| `src/quality/vector_diagnostics.py` | 共用JSON/CSV和数值辅助；softmax不是主分类评分 | model_or_shared_dependency；主调用链 | outputs/preprocess/; outputs/quality/ |
| `src/quality/vector_transform.py` | 8个列表字段真实标量化与qurater分量参数；softmax_expected_rating is auxiliary, not the final argmax rating model. | model_or_shared_dependency；主调用链 | outputs/preprocess/; outputs/quality/ |
| `src/quality/preprocess.py` | 只拟合A1的填补/截尾参数，原样映射A2/A3 | model_or_shared_dependency；主调用链 | outputs/preprocess/; outputs/quality/ |
| `scripts/03_build_quality_features.py` | 最终阶段实际执行入口：03_build_quality_features.py | execution_entry；主调用链 | outputs/preprocess/; outputs/quality/ |
| `src/quality/critic.py` | Pearson-CRITIC ddof=1及Spearman敏感性 | model_or_shared_dependency；主调用链 | outputs/preprocess/; outputs/quality/ |
| `scripts/04a_diagnose_critic_weights.py` | 最终阶段实际执行入口：04a_diagnose_critic_weights.py | execution_entry；主调用链 | outputs/preprocess/; outputs/quality/ |
| `src/quality/aggregate.py` | 10%截尾均值、模型记录bootstrap、分布比较 | model_or_shared_dependency；主调用链 | outputs/preprocess/; outputs/quality/ |
| `src/quality/plotting.py` | 最终CSV/图表/报告呈现，compact省略纯绘图和排版 | output_presentation；主调用链 | outputs/preprocess/; outputs/quality/ |
| `src/quality/reporting.py` | 最终CSV/图表/报告呈现，compact省略纯绘图和排版 | output_presentation；主调用链 | outputs/preprocess/; outputs/quality/ |
| `scripts/04_build_quality_score.py` | 最终阶段实际执行入口：04_build_quality_score.py | execution_entry；主调用链 | outputs/preprocess/; outputs/quality/ |
| `src/quality/conflict.py` | K=加权绝对偏离、冲突阈值和驱动组合 | model_or_shared_dependency；主调用链 | outputs/conflict/ |
| `src/quality/conflict_plotting.py` | 最终CSV/图表/报告呈现，compact省略纯绘图和排版 | output_presentation；主调用链 | outputs/conflict/ |
| `scripts/05_analyze_quality_conflict.py` | 最终阶段实际执行入口：05_analyze_quality_conflict.py | execution_entry；主调用链 | outputs/conflict/ |
| `src/mixture/io.py` | 真实附件读取、索引/字段兼容与输入单位 | model_or_shared_dependency；主调用链 | outputs/mixture/; outputs/mixture/stage2/ |
| `src/mixture/composition.py` | 训练折内乘法零替换与Helmert ILR | model_or_shared_dependency；主调用链 | outputs/mixture/; outputs/mixture/stage2/ |
| `src/mixture/modeling.py` | Ridge/ElasticNet候选、训练内调参与分组CV；M0 and second-order M2 are candidate / sensitivity models; not used as final main model. M1_Ridge is the numerical comparator; M1_ElasticNet is selected. | model_or_shared_dependency；主调用链 | outputs/mixture/; outputs/mixture/stage2/ |
| `scripts/08_model_mixture_loss.py` | 最终阶段实际执行入口：08_model_mixture_loss.py | execution_entry；主调用链 | outputs/mixture/; outputs/mixture/stage2/ |
| `src/mixture/effects.py` | 真实领域配比1个百分点扰动及联合边际效应 | model_or_shared_dependency；主调用链 | outputs/mixture/; outputs/mixture/stage2/ |
| `src/mixture/augmented.py` | Qp/Kp候选对照；最终没有替代一阶ElasticNet；candidate / sensitivity model; Qp/Kp augmented models are not used as final main model. | candidate_comparison；主调用链 | outputs/mixture/; outputs/mixture/stage2/ |
| `scripts/09_analyze_mixture_effects.py` | 最终阶段实际执行入口：09_analyze_mixture_effects.py；M3_Qp / M4_Qp_Kp are candidate / sensitivity models, not final main model. The saved final_model_policy selects M1_ElasticNet. | execution_entry；主调用链 | outputs/mixture/; outputs/mixture/stage2/ |
| `src/quality/profiling.py` | 历史描述统计，非A1主预处理参数 | model_or_shared_dependency；full_source only | outputs/profile/ （03只复用缺失记录清单） |
| `scripts/02_profile_quality_data.py` | 历史pooled描述统计；03使用其缺失记录核对，不复用pooled阈值 | execution_entry；full_source only | outputs/profile/ （03只复用缺失记录清单） |
| `src/quality/record_diagnostics.py` | 预处理前重复/ID冲突及缺失诊断 | model_or_shared_dependency；full_source only | outputs/inspection/ （03前置诊断证据） |
| `scripts/02c_diagnose_records.py` | 03所需重复/重叠/天花板诊断输入的实际生成入口 | execution_entry；full_source only | outputs/inspection/ （03前置诊断证据） |
| `scripts/07_inspect_mixture_loss_data.py` | 08所需A4–A15真实数据诊断与input_manifest生成入口 | execution_entry；full_source only | outputs/mixture/; outputs/mixture/stage2/ |
| `scripts/03_verify_quality_features.py` | 独立公式/输出/冻结哈希验证；只归档，不在本轮执行 | validation；full_source only | outputs/preprocess/; outputs/quality/ |
| `scripts/04b_verify_quality_score.py` | 独立公式/输出/冻结哈希验证；只归档，不在本轮执行 | validation；full_source only | outputs/preprocess/; outputs/quality/ |
| `scripts/04b_record_text_review.py` | 既有文本核验材料归档，不当作独立人工实验 | execution_entry；full_source only | outputs/preprocess/; outputs/quality/ |
| `scripts/05_verify_quality_conflict.py` | 独立公式/输出/冻结哈希验证；只归档，不在本轮执行 | validation；full_source only | outputs/conflict/ |
| `scripts/08_verify_mixture_models.py` | 独立公式/输出/冻结哈希验证；只归档，不在本轮执行 | validation；full_source only | outputs/mixture/; outputs/mixture/stage2/ |
| `scripts/09_verify_mixture_analysis.py` | 独立公式/输出/冻结哈希验证；只归档，不在本轮执行 | validation；full_source only | outputs/mixture/; outputs/mixture/stage2/ |
| `src/quality/__init__.py` | 调用链共用依赖或包初始化 | model_or_shared_dependency；full_source only | outputs/preprocess/; outputs/quality/ |
| `src/mixture/__init__.py` | 调用链共用依赖或包初始化 | model_or_shared_dependency；full_source only | outputs/mixture/; outputs/mixture/stage2/ |

## 未纳入的相近文件

- `scripts/01b_inspect_vector_semantics.py`：只作早期结构/描述性诊断，不在最终数值生成依赖链
- `scripts/01_inspect_quality_fields.py`：只作早期结构/描述性诊断，不在最终数值生成依赖链
- `scripts/06_validate_robust_reaggregation.py`：确实执行过，但最终选择(Q,K)，未激活Q_robust/Q_final；默认不纳入主附录或最终调用链备份
- `scripts/06_verify_robust_reaggregation.py`：确实执行过，但最终选择(Q,K)，未激活Q_robust/Q_final；默认不纳入主附录或最终调用链备份
- `src/quality/robust_reaggregation.py`：确实执行过，但最终选择(Q,K)，未激活Q_robust/Q_final；默认不纳入主附录或最终调用链备份

## compact 摘录范围

- `src/quality/io.py`：`infer_source_domain`, `resolve_metric_value`, `discover_quality_files`, `iter_jsonl`
- `src/quality/vector_transform.py`：`parse_vector`, `scalarize_record`, `fit_qurater_parameters`, `transform_qurater`
- `src/quality/preprocess.py`：`impute_matrix`, `fit_preprocessing_model`, `apply_preprocessing`
- `src/quality/normalize.py`：`moderate_utility_transform`, `minmax_transform`
- `src/quality/critic.py`：`critic_weights`
- `src/quality/aggregate.py`：`finite_scores`, `score_profile`, `bootstrap_trimmed_mean`, `domain_summary`, `compare_distributions`
- `scripts/03_build_quality_features.py`：`read_dataset`
- `scripts/04_build_quality_score.py`：`load_weights`, `score_dataset`
- `src/quality/conflict.py`：`conflict_indices`, `fit_conflict_thresholds`
- `src/mixture/io.py`：`read_table`, `load_datasets`, `signatures`
- `src/mixture/composition.py`：`ILRTransformer`
- `src/mixture/modeling.py`：`scores`, `group_folds`, `candidate`, `tune`
- `scripts/08_model_mixture_loss.py`：`metadata`, `fit_checked`
- `src/mixture/effects.py`：`perturb`, `direction`, `summary`, `baselines`, `analyze`
- `scripts/09_analyze_mixture_effects.py`：`scale_diagnostics`, `extrapolation`
- 入口中的冻结选择/拟合/CV/外部验证调用片段另按原行号摘录，详见 source_traceability.csv。

# 问题二

## 最终实际使用文件

经典L0＋B6局部Delta_Q＋lnN分段线性lambda_p乘冻结M(p)差值；主边界S-A。

实际执行入口：`scripts/11_fit_classical_scaling.py`, `scripts/14_fit_scale_dependent_quality.py`, `scripts/15_calibrate_mixture_scaling.py`, `scripts/16_analyze_generalized_scaling.py`。
结果来源：`outputs/scaling/phase1/classical_scaling_parameters.csv`, `outputs/scaling/phase1/B1_group_cv.csv`, `outputs/scaling/phase2C/model_parameters.csv`, `outputs/scaling/phase3/lambda_empirical_knots.csv`, `outputs/scaling/phase4/generalized_scaling_final.json`, `outputs/scaling/phase4/marginal_derivative_checks.csv`。
报告证据：`outputs/report/question2_phase1_classical_scaling.md`, `outputs/report/question2_phase2C_scale_dependent_quality.md`, `outputs/report/question2_phase3_modular_mixture_scaling.md`, `outputs/report/question2_final.md`。

| 原始文件 | 用途 | 主模型/验证/辅助角色 | 对应最终输出 |
|---|---|---|---|
| `src/scaling/io.py` | 真实附件读取、索引/字段兼容与输入单位 | model_or_shared_dependency；主调用链 | outputs/scaling/phase4/ |
| `src/scaling/classical.py` | 经典五参数非线性拟合、解析Jacobian、误差指标 | model_or_shared_dependency；主调用链 | outputs/scaling/phase1/ |
| `src/scaling/bootstrap.py` | 整条trajectory重采样拟合 | model_or_shared_dependency；主调用链 | outputs/scaling/phase1/ |
| `scripts/10_diagnose_scaling_data.py` | 最终阶段实际执行入口：10_diagnose_scaling_data.py | execution_entry；主调用链 | outputs/scaling/phase1/ |
| `scripts/11_fit_classical_scaling.py` | 最终阶段实际执行入口：11_fit_classical_scaling.py | execution_entry；主调用链 | outputs/scaling/phase1/ |
| `src/scaling/quality_extension.py` | 最终链复用读入/reference工具；常数gamma MQ候选不是最终主模型；MQ1/MQ2/gamma fitting: candidate / sensitivity model; not used as final main model. Final chain uses read_parameters, read_data, subset and group_reference. | model_or_shared_dependency；主调用链 | outputs/scaling/phase2C/ |
| `src/scaling/scale_quality.py` | S3原参数来源；S1/S2只为比较候选，最终去v且不重拟合；S1/S2 are candidate / sensitivity models, not used as final main model. S3 C,u are frozen inputs to local Delta_Q; the near-zero v is removed without refitting. | model_or_shared_dependency；主调用链 | outputs/scaling/phase2C/ |
| `src/scaling/scale_quality_analysis.py` | ND分组CV、Q层留出、整块bootstrap及B6/B7差分检验 | model_or_shared_dependency；主调用链 | outputs/scaling/phase2C/ |
| `src/scaling/quality_analysis.py` | Phase2C复用B8压力检验工具；MQ候选不是最终模型；MQ candidate functions are not final main model; Phase2C calls stress_test only. | model_or_shared_dependency；主调用链 | outputs/scaling/phase2C/ |
| `src/scaling/quality_reporting.py` | Phase2C复用绘图和表格工具 | output_presentation；主调用链 | outputs/scaling/phase2C/ |
| `src/scaling/scale_quality_reporting.py` | 最终CSV/图表/报告呈现，compact省略纯绘图和排版 | output_presentation；主调用链 | outputs/scaling/phase2C/ |
| `scripts/14_fit_scale_dependent_quality.py` | 最终阶段实际执行入口：14_fit_scale_dependent_quality.py | execution_entry；主调用链 | outputs/scaling/phase2C/ |
| `src/scaling/mixture_transfer.py` | 冻结ILR预测、逐尺度lambda与配方bootstrap；P1/P2曲线仅比较候选；P1/P2 smooth lambda curves: candidate / sensitivity models, not final main model. Final lambda is piecewise linear in ln N; these curve labels differ from Question3 resource policies. | model_or_shared_dependency；主调用链 | outputs/scaling/phase3/ |
| `src/scaling/mixture_transfer_reporting.py` | 最终CSV/图表/报告呈现，compact省略纯绘图和排版 | output_presentation；主调用链 | outputs/scaling/phase3/ |
| `scripts/15_calibrate_mixture_scaling.py` | 最终阶段实际执行入口：15_calibrate_mixture_scaling.py | execution_entry；主调用链 | outputs/scaling/phase3/ |
| `src/scaling/marginal_analysis.py` | 最终广义标度律、边界策略、解析导数/弹性及Q+0.1精确求根 | model_or_shared_dependency；主调用链 | outputs/scaling/phase4/ |
| `src/scaling/marginal_reporting.py` | 最终CSV/图表/报告呈现，compact省略纯绘图和排版 | output_presentation；主调用链 | outputs/scaling/phase4/ |
| `scripts/16_analyze_generalized_scaling.py` | 最终阶段实际执行入口：16_analyze_generalized_scaling.py | execution_entry；主调用链 | outputs/scaling/phase4/ |
| `scripts/11_verify_classical_scaling.py` | 独立公式/输出/冻结哈希验证；只归档，不在本轮执行 | validation；full_source only | outputs/scaling/phase1/ |
| `scripts/14_verify_scale_dependent_quality.py` | 独立公式/输出/冻结哈希验证；只归档，不在本轮执行 | validation；full_source only | outputs/scaling/phase2C/ |
| `scripts/15_verify_mixture_scaling.py` | 独立公式/输出/冻结哈希验证；只归档，不在本轮执行 | validation；full_source only | outputs/scaling/phase3/ |
| `scripts/16_verify_generalized_scaling.py` | 独立公式/输出/冻结哈希验证；只归档，不在本轮执行 | validation；full_source only | outputs/scaling/phase4/ |
| `src/scaling/__init__.py` | 调用链共用依赖或包初始化 | model_or_shared_dependency；full_source only | outputs/scaling/phase4/ |

## 未纳入的相近文件

- `scripts/12_diagnose_quality_scaling.py`：质量扩展前的诊断阶段；无最终参数拟合，最终链由14重新严格读入B6/B7/B8
- `scripts/13_fit_quality_scaling.py`：常数gamma MQ1/MQ2候选未冻结为最终模型；最终局部Delta_Q来自14阶段S3
- `scripts/13_verify_quality_scaling.py`：常数gamma MQ1/MQ2候选未冻结为最终模型；最终局部Delta_Q来自14阶段S3

## compact 摘录范围

- `src/scaling/io.py`：`load_csv`, `combine_trajectories`
- `src/scaling/classical.py`：`weights`, `predict`, `prediction_jacobian`, `initial_values`, `fit`, `metrics`
- `src/scaling/bootstrap.py`：`initialize`, `repetition`
- `scripts/11_fit_classical_scaling.py`：`fit_and_cv`, `cluster_bootstrap`, `validate`
- `src/scaling/quality_extension.py`：`read_parameters`, `read_data`, `subset`, `group_reference`
- `src/scaling/quality_analysis.py`：`stress_test`
- `src/scaling/scale_quality.py`：`centers`, `components`, `physical`, `encoded`, `response_jacobian`, `predict_scale`, `initial_parameters`, `fit_scale`
- `src/scaling/scale_quality_analysis.py`：`evaluate`, `group_cv`, `q_level_cv`, `initialize_worker`, `bootstrap_worker`, `resampling_jobs`, `bootstrap_models`
- `scripts/14_fit_scale_dependent_quality.py`：`basics`
- `src/scaling/mixture_transfer.py`：`frozen_metadata`, `predict_ilr`, `predict_recipe`, `read_frozen_recipes`, `transfer`, `bootstrap_transfers`, `empirical_lambda`
- `src/scaling/marginal_analysis.py`：`FrozenGeneralizedScaling`, `finite_difference`
- `scripts/16_analyze_generalized_scaling.py`：`frozen_model`, `large_validation`, `lambda_sensitivity`
- 入口中的冻结选择/拟合/CV/外部验证调用片段另按原行号摘录，详见 source_traceability.csv。

# 问题三

## 最终实际使用文件

解析消去D_B；P0主方案、P1经验增强；P2-A为理论外推，P2-B无有限最优解。

实际执行入口：`scripts/17_setup_compute_optimization.py`, `scripts/18_optimize_compute_resources.py`, `scripts/19_finalize_compute_optimization.py`。
结果来源：`outputs/optimization/phase1/optimization_setup.json`, `outputs/optimization/phase1/empirical_recipe_candidates.csv`, `outputs/optimization/phase2/official_budget_results.csv`, `outputs/optimization/phase2/P0_main_budget_path.csv`, `outputs/optimization/phase3/P0_P1_P2_official_comparison.csv`, `outputs/optimization/phase3/P2_global_search.csv`, `outputs/optimization/phase3/P2_B_boundary_limit_witness.csv`。
报告证据：`outputs/report/question3_phase1_setup.md`, `outputs/report/question3_phase2_optimization.md`, `outputs/report/question3_final.md`。

| 原始文件 | 用途 | 主模型/验证/辅助角色 | 对应最终输出 |
|---|---|---|---|
| `src/optimization/costs.py` | 6ND、注意力成本与exp/power/log质量增量成本 | model_or_shared_dependency；主调用链 | outputs/optimization/phase1/; phase2/; phase3/ （依来源模块） |
| `src/scaling/mixture_transfer.py` | 冻结ILR预测、逐尺度lambda与配方bootstrap；P1/P2曲线仅比较候选；P1/P2 smooth lambda curves: candidate / sensitivity models, not final main model. Final lambda is piecewise linear in ln N; these curve labels differ from Question3 resource policies. | model_or_shared_dependency；主调用链 | outputs/optimization/phase1/; phase2/; phase3/ （依来源模块） |
| `src/optimization/setup.py` | 官方成本、Q0、决策边界与832经验候选的来源 | model_or_shared_dependency；主调用链 | outputs/optimization/phase1/; phase2/; phase3/ （依来源模块） |
| `scripts/17_setup_compute_optimization.py` | 最终阶段实际执行入口：17_setup_compute_optimization.py | execution_entry；主调用链 | outputs/optimization/phase1/; phase2/; phase3/ （依来源模块） |
| `src/scaling/marginal_analysis.py` | 最终广义标度律、边界策略、解析导数/弹性及Q+0.1精确求根 | model_or_shared_dependency；主调用链 | outputs/optimization/phase1/; phase2/; phase3/ （依来源模块） |
| `src/optimization/resource_solver.py` | D_B解析消元、可行性、网格/SLSQP/DE及P0/P1求解 | model_or_shared_dependency；主调用链 | outputs/optimization/phase1/; phase2/; phase3/ （依来源模块） |
| `src/optimization/transitions.py` | 连续预算扫描路径上的Type I/II转移检验与区间 | model_or_shared_dependency；主调用链 | outputs/optimization/phase1/; phase2/; phase3/ （依来源模块） |
| `src/optimization/resource_reporting.py` | 最终CSV/图表/报告呈现，compact省略纯绘图和排版 | output_presentation；主调用链 | outputs/optimization/phase1/; phase2/; phase3/ （依来源模块） |
| `scripts/18_optimize_compute_resources.py` | 最终阶段实际执行入口：18_optimize_compute_resources.py | execution_entry；主调用链 | outputs/optimization/phase1/; phase2/; phase3/ （依来源模块） |
| `src/optimization/simplex_sensitivity.py` | P2-A配方证书与数值交叉检验；P2-B边界无界性，仅理论敏感性；P2-A/P2-B are theory / sensitivity only. Final engineering hierarchy: P0 main, P1 empirical enhancement; P2-B has no finite attained optimum. | model_or_shared_dependency；主调用链 | outputs/optimization/phase1/; phase2/; phase3/ （依来源模块） |
| `src/optimization/final_reporting.py` | 冻结P0/P1复用、证据分层与问题三最终表/报告 | output_presentation；主调用链 | outputs/optimization/phase1/; phase2/; phase3/ （依来源模块） |
| `scripts/19_finalize_compute_optimization.py` | 最终阶段实际执行入口：19_finalize_compute_optimization.py | execution_entry；主调用链 | outputs/optimization/phase1/; phase2/; phase3/ （依来源模块） |
| `scripts/17_verify_compute_setup.py` | 独立公式/输出/冻结哈希验证；只归档，不在本轮执行 | validation；full_source only | outputs/optimization/phase1/; phase2/; phase3/ （依来源模块） |
| `scripts/18_verify_compute_optimization.py` | 独立公式/输出/冻结哈希验证；只归档，不在本轮执行 | validation；full_source only | outputs/optimization/phase1/; phase2/; phase3/ （依来源模块） |
| `scripts/19_verify_final_optimization.py` | 独立公式/输出/冻结哈希验证；只归档，不在本轮执行 | validation；full_source only | outputs/optimization/phase1/; phase2/; phase3/ （依来源模块） |
| `src/optimization/__init__.py` | 调用链共用依赖或包初始化 | model_or_shared_dependency；full_source only | outputs/optimization/phase1/; phase2/; phase3/ （依来源模块） |
| `src/scaling/__init__.py` | 调用链共用依赖或包初始化 | model_or_shared_dependency；full_source only | outputs/optimization/phase1/; phase2/; phase3/ （依来源模块） |

## 未纳入的相近文件

无其他相近执行入口被排除。

## compact 摘录范围

- `src/optimization/costs.py`：`_validate_quality`, `quality_increment_per_token`, `marginal_quality_per_token`, `cost_components`
- `src/scaling/marginal_analysis.py`：`FrozenGeneralizedScaling`
- `src/optimization/setup.py`：`decision_domains`, `mixture_sets`, `budget_scenarios`
- `src/optimization/resource_solver.py`：`load`, `lambda_zero`, `ResourceProblem`, `three_dimensional_checks`, `run_solve`
- `src/optimization/simplex_sensitivity.py`：`log_contrast`, `exact_extreme`, `simplex_search`, `evidence_level`, `official_comparison`
- `src/optimization/transitions.py`：`bisect_status`, `soft_fit`, `run_diagnose`
- 入口中的冻结选择/拟合/CV/外部验证调用片段另按原行号摘录，详见 source_traceability.csv。

# 问题四

## 最终实际使用文件

固定六任务A_z；M1/M2-C/M3＋双因素Shapley；桥接失败负结果；已执行Phase3情景投影及风险诊断。

实际执行入口：`scripts/20_build_evolution_ability_bridge.py`, `scripts/21_build_evolution_contribution.py`, `scripts/22_build_evolution_projection.py`。
结果来源：`outputs/evolution/phase1/evolution_model_table.csv`, `outputs/evolution/phase1/ability_transform_parameters.json`, `outputs/evolution/phase1/bridge_model.json`, `outputs/evolution/phase2/model_definitions.json`, `outputs/evolution/phase2/contribution_decomposition.csv`, `outputs/evolution/phase3/future_frontier_scenarios.csv`, `outputs/evolution/phase3/verification_summary.json`。
报告证据：`outputs/report/question4_phase1_data_ability_bridge.md`, `outputs/report/question4_phase2_contribution_decomposition.md`, `outputs/report/question4_final.md`。

| 原始文件 | 用途 | 主模型/验证/辅助角色 | 对应最终输出 |
|---|---|---|---|
| `src/evolution/common.py` | 模型名规范、CSV和相关系数辅助 | model_or_shared_dependency；主调用链 | outputs/evolution/phase1/ |
| `src/evolution/metadata.py` | C1/C2去重、S0/S1类型证据及C4元数据匹配 | model_or_shared_dependency；主调用链 | outputs/evolution/phase1/ |
| `src/evolution/ability.py` | 固定六Benchmark Winsor/median/IQR、A_z、PCA/Average敏感性 | model_or_shared_dependency；主调用链 | outputs/evolution/phase1/ |
| `src/evolution/detailed.py` | C8 JSON全遍历、损坏文件记录与偏科/弱尾 | model_or_shared_dependency；主调用链 | outputs/evolution/phase1/ |
| `src/evolution/bridge.py` | B0/B1/B2、LOMO/bootstrap和桥接失败诊断；B0/B1/B2 are failed bridge candidates. Preserve bridge_valid_for_conversion=False; do not convert Loss to Benchmark. | model_or_shared_dependency；主调用链 | outputs/evolution/phase1/ |
| `src/evolution/reporting.py` | 最终CSV/图表/报告呈现，compact省略纯绘图和排版 | output_presentation；主调用链 | outputs/evolution/phase1/ |
| `scripts/20_build_evolution_ability_bridge.py` | 最终阶段实际执行入口：20_build_evolution_ability_bridge.py | execution_entry；主调用链 | outputs/evolution/phase1/ |
| `src/evolution/contribution_regression.py` | OLS/HC3/Huber、LOOCV、Cook/VIF、双因素Shapley与5000模型bootstrap；Huber, quadratic time and month FE are sensitivity/candidate comparisons. Shapley signed components are not future growth weights. | model_or_shared_dependency；主调用链 | outputs/evolution/phase2/ |
| `src/evolution/contribution_analysis.py` | T0-T3、M1/M2-C/M3、组织/时间轴对照、技术残差和分解执行 | model_or_shared_dependency；主调用链 | outputs/evolution/phase2/ |
| `src/evolution/contribution_reporting.py` | 最终CSV/图表/报告呈现，compact省略纯绘图和排版 | output_presentation；主调用链 | outputs/evolution/phase2/ |
| `scripts/21_build_evolution_contribution.py` | 最终阶段实际执行入口：21_build_evolution_contribution.py | execution_entry；主调用链 | outputs/evolution/phase2/ |
| `src/evolution/projection_data.py` | 实际执行Phase3：锚点、C4去重滚动前沿、增长bootstrap | model_or_shared_dependency；主调用链 | outputs/evolution/phase3/ |
| `src/evolution/projection_models.py` | 冻结成对系数的情景Monte Carlo、回测、补偿、风险标记；Executed scenario projection, not a reliable long-term forecast. Frozen full-period calibration/model lookahead is disclosed. No Loss bridge conversion. | model_or_shared_dependency；主调用链 | outputs/evolution/phase3/ |
| `src/evolution/projection_reporting.py` | 最终CSV/图表/报告呈现，compact省略纯绘图和排版 | output_presentation；主调用链 | outputs/evolution/phase3/ |
| `scripts/22_build_evolution_projection.py` | 最终阶段实际执行入口：22_build_evolution_projection.py | execution_entry；主调用链 | outputs/evolution/phase3/ |
| `scripts/20_verify_evolution_phase1.py` | 独立公式/输出/冻结哈希验证；只归档，不在本轮执行 | validation；full_source only | outputs/evolution/phase1/ |
| `scripts/21_verify_evolution_contribution.py` | 独立公式/输出/冻结哈希验证；只归档，不在本轮执行 | validation；full_source only | outputs/evolution/phase2/ |
| `scripts/22_verify_evolution_projection.py` | 独立公式/输出/冻结哈希验证；只归档，不在本轮执行 | validation；full_source only | outputs/evolution/phase3/ |
| `src/evolution/__init__.py` | 调用链共用依赖或包初始化 | model_or_shared_dependency；full_source only | outputs/evolution/phase1/ |

## 未纳入的相近文件

无其他相近执行入口被排除。

## compact 摘录范围

- `src/evolution/common.py`：`normalized_name`
- `src/evolution/metadata.py`：`canonical_models`, `epoch_join`
- `src/evolution/ability.py`：`transform`, `construct_ability`
- `src/evolution/detailed.py`：`value`, `normalize_score`, `extract`, `analyze_details`
- `src/evolution/bridge.py`：`fit_candidate`, `predict`, `metrics`, `decreasing`, `analyze_bridge`
- `src/evolution/contribution_regression.py`：`design`, `accuracy`, `fit_regression`, `diagnostics`, `shapley`, `endpoints`, `contribution`, `bootstrap`
- `src/evolution/contribution_analysis.py`：`prepare`, `model_data`, `fit_models`, `make_technology_tables`, `run_decomposition`
- `src/evolution/projection_data.py`：`load_frozen`, `select_scope`, `frontier_stats`, `build_anchors`, `benchmark_series`, `compute_candidates`, `select_compute_window`, `compute_grid`, `rolling_compute`, `slope`, `estimate_growth`, `c3_direction`
- `src/evolution/projection_models.py`：`frozen_m2`, `linear_fit`, `build_hindcast_and_trends`, `quantiles`, `level`, `build_projections`
- 入口中的冻结选择/拟合/CV/外部验证调用片段另按原行号摘录，详见 source_traceability.csv。

