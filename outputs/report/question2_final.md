# 问题二最终报告：模块化广义标度律、边际收益与参数等价

问题二按人工确认冻结的模块化形式完成。所有骨架、质量和配比参数保持原值，仅分析、推导、数值求解及授权的外推一致性检查。主 λ 规则为 ln N 分段线性插值，低于 1M 保持首节点，高于 70B 保持末节点。没有进入问题三。

## 1. 经典标度律

$L_0(N,D)=E+AN^{-\alpha}+BD^{-\beta}$。N、D 均以十亿为单位。

| parameter | value |
| --- | --- |
| E | 1.6897976 |
| A | 0.35398032 |
| B | 1.2403056 |
| alpha | 0.33997658 |
| beta | 0.27987813 |

读取 PHASE 1 的 trajectory_balanced 主行，程序使用完整精度。主拟合数据是 B1，未使用 B9/B10 回调参数。

## 2. 质量模块及验证边界

$\Delta_Q=C_Q(N/N_0)^{-u_Q}(-\ln Q)$，$N_0=1$B。

C_Q=0.17064090046126643，u_Q=0.16190329425272901，从冻结 PHASE 2C S3 读取；删除已确认 v≈0 的 D 项，未重拟合。

仅称“B6 校准的局部质量修正项”。Q 对 B6 Loss 有增量解释能力，但 B7 非重叠的组内差分幅值没有稳定复现。B8 与此前机制不兼容，不宣称存在全局统一质量弹性。第一问领域 Q 仅提供数值工作点，不证明与 B6 的 Q 具有同一生成语义。

B6/B7 数值支持范围：{'N_min_B': 0.07, 'N_max_B': 11.97, 'D_min_B': 10.0, 'D_max_B': 600.0, 'Q_min': 0.1, 'Q_max': 1.0}。12B 略高于真实上限 11.97B，已标记轻微外推；70B、100B、1000B 为明显外推。

## 3. 配比模块与 λ 的尺度变化

M(p) 是冻结 ILR + 一阶 ElasticNet 对配方的 1M L_mean 预测。参考配方为冻结训练 ILR 中心，未重新读取或拟合 A4–A15。

| N_params_B | lambda_s | CI_low | CI_high | evidence |
| --- | --- | --- | --- | --- |
| 0.001 | 0.93905515 | 0.74859391 | 1.1411218 | observed |
| 0.06 | 0.68974808 | 0.53894611 | 0.85879678 | observed |
| 1 | 0.29032089 | 0.18945514 | 0.40489143 | observed |
| 10 | -0.3110438 | -0.44607347 | -0.19417264 | estimated |
| 70 | -0.3727259 | -0.51420778 | -0.25107689 | estimated |

三个真实尺度同为正，两个估算尺度同为负。方向翻转及过零结论依赖估算表。超过 1B 的配比段已经包含 Class D 证据，不能把 70B 以内的所有配比结果都称为真实观测支持。

## 4. λ 的最终分段与边界规则

$s_i=(\lambda_{i+1}-\lambda_i)/\ln(N_{i+1}/N_i)$；内部 $\lambda(N)=\lambda_i+s_i\ln(N/N_i)$，$\lambda\prime(N)=s_i/N$。

N<1M：λ=λ_1M，导数为 0。N>70B：主方案 λ=λ_70B，导数为 0。节点上 λ 连续，但 N 导数通常不连续，表中同时记录左右弹性；主口径使用右导数。

特别是 1B 节点本身的 λ 来自真实检验，但其右导数使用 1B→10B 段，依赖估算表。非参考配方在 1B 的参数弹性、优势因素和局部等价判断因此含 Class D 条件性证据。

| segment | N_low | N_high | lambda_low | lambda_high | s_i |
| --- | --- | --- | --- | --- | --- |
| lower_clamp | 0 | 0.001 | 0.93905515 | 0.93905515 | 0 |
| interior_1 | 0.001 | 0.06 | 0.93905515 | 0.68974808 | -0.060890594 |
| interior_2 | 0.06 | 1 | 0.68974808 | 0.29032089 | -0.14197258 |
| interior_3 | 1 | 10 | 0.29032089 | -0.3110438 | -0.26116936 |
| interior_4 | 10 | 70 | -0.3110438 | -0.3727259 | -0.031698331 |
| upper_clamp | 70 | n.a. | -0.3727259 | -0.3727259 | 0 |

## 5. 模块化广义标度律及退化条件

$L_{gen}=E+AN^{-\alpha}+BD^{-\beta}+C_Q(N/N_0)^{-u_Q}(-\ln Q)+\lambda_p(N)[M(p)-M(p_{ref})]$。

M(p_ref)=5.3446567761711776。17 维参考配方保存在冻结 PHASE 3 reference_recipe.json。

$Q=1$、$p=p_{ref}$ 时两个修正项均为 0，$L_{gen}=L_0$。该退化是结构恒等，不是观测到的大模型质量或配方。

## 6. N/D/Q 边际效应及有限差分验证

令 $\Delta M=M(p)-M(p_{ref})$：

$\partial_D L=-\beta B D^{-\beta-1}$；$\partial_Q L=-C_Q(N/N_0)^{-u_Q}/Q$。

$\partial_N L=-\alpha A N^{-\alpha-1}-(u_Q/N)\Delta_Q+(s_i/N)\Delta M$。

边界保持区令 s_i=0。在折点处分别取左右 s_i。使用四阶中心差分检查内部，四阶单侧差分检查折点；不使用跨折点中心差分验证单侧导数。

marginal_derivative_checks.csv 全部通过。最大相对误差=8.3702395e-10。判据为相对误差≤1e−5 或绝对误差≤1e−8。

## 7. 可约 Loss 的降低弹性

$R=L_{gen}-E$，$\mathcal E_x=-\partial\ln R/\partial\ln x$。正值表示增加资源降低可约 Loss。

$\mathcal E_D=\beta BD^{-\beta}/R$；$\mathcal E_Q=C_Q(N/N_0)^{-u_Q}/R$。

$\mathcal E_N=[\alpha AN^{-\alpha}+u_Q\Delta_Q-s_i\Delta M]/R$。

仅 R>0 时计算对数弹性。传统 x/L·∂L/∂x 保留在 elasticity_grid.csv 的 traditional_N/D/Q 附表字段，正文使用上述降低弹性。弹性比较的是相同相对变化比例，不等于相同成本或相同绝对增量的收益。

## 8. 工作点与优势区域

D 从 B6 的 45 个不同 N-D 块取 P25/median/P75：{'P25': 50.0, 'median': 150.0, 'P75': 300.0}B。这避免把重复 Q 层当作额外 D 设计；不是删去任何原始数据。三个值均在 B6 支持范围内。
Q 为七个领域 Q_domain 的最小、中位、最大值，另加 0.7：{'domain_min': 0.5867366961905537, 'domain_median': 0.7475281055884827, 'domain_max': 0.7651065297732803, 'Q_0.7': 0.7}。

配方为 p_ref、冻结 M(p) P10 邻近配方 index=142、P90 邻近配方 index=242。完整工作点 N×D×Q×p 共 216 条，不创造配方。

| N_params_B | D_tokens_B | Q | E_N | E_D | E_Q | dominant_factor | extrapolative |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.07 | 150 | 0.7 | 0.24538778 | 0.06708755 | 0.20618173 | N | False |
| 1 | 150 | 0.7 | 0.18083795 | 0.1186149 | 0.23700913 | Q | False |
| 12 | 150 | 0.7 | 0.11707701 | 0.17151336 | 0.2291927 | Q | True |
| 70 | 150 | 0.7 | 0.079530756 | 0.20370831 | 0.2046007 | Q | True |
| 100 | 150 | 0.7 | 0.073096785 | 0.20932671 | 0.1984467 | D | True |
| 1000 | 150 | 0.7 | 0.041008067 | 0.23799324 | 0.15541058 | D | True |

| N_params_B | recipe | factor | dominant_count | n_workpoints | dominant_fraction |
| --- | --- | --- | --- | --- | --- |
| 0.07 | P10 | N | 12 | 12 | 1 |
| 0.07 | P90 | N | 12 | 12 | 1 |
| 0.07 | p_ref | N | 12 | 12 | 1 |
| 1 | P10 | Q | 12 | 12 | 1 |
| 1 | P90 | N | 12 | 12 | 1 |
| 1 | p_ref | Q | 12 | 12 | 1 |
| 12 | P10 | D | 4 | 12 | 0.33333333 |
| 12 | P10 | Q | 8 | 12 | 0.66666667 |
| 12 | P90 | D | 4 | 12 | 0.33333333 |
| 12 | P90 | Q | 8 | 12 | 0.66666667 |
| 12 | p_ref | D | 4 | 12 | 0.33333333 |
| 12 | p_ref | Q | 8 | 12 | 0.66666667 |
| 70 | P10 | D | 4 | 12 | 0.33333333 |
| 70 | P10 | Q | 8 | 12 | 0.66666667 |
| 70 | P90 | D | 4 | 12 | 0.33333333 |
| 70 | P90 | Q | 8 | 12 | 0.66666667 |
| 70 | p_ref | D | 4 | 12 | 0.33333333 |
| 70 | p_ref | Q | 8 | 12 | 0.66666667 |
| 100 | P10 | D | 8 | 12 | 0.66666667 |
| 100 | P10 | Q | 4 | 12 | 0.33333333 |
| 100 | P90 | D | 8 | 12 | 0.66666667 |
| 100 | P90 | Q | 4 | 12 | 0.33333333 |
| 100 | p_ref | D | 8 | 12 | 0.66666667 |
| 100 | p_ref | Q | 4 | 12 | 0.33333333 |
| 1000 | P10 | D | 12 | 12 | 1 |
| 1000 | P90 | D | 12 | 12 | 1 |
| 1000 | p_ref | D | 12 | 12 | 1 |

上述优势区是该冻结候选的结构结果。估算配比段及质量外推的证据等级不会因弹性值较大而提高。

## 9. Q 提升 0.1 的精确参数等价

固定 D、p，求首个 $N_{eq}>N$：$L_{gen}(N_{eq},D,Q,p)=L_{gen}(N,D,Q+0.1,p)$。

等价参数增加为 $N_{eq}/N-1$。E 和 $BD^{-\beta}$ 从方程两侧严格抵消，因此本模型的等价解不随固定 D 改变。

正式结果使用 brentq。搜索上限 max(100N,10000B)。每个分段内加入全部 λ 节点及 Gain_N=0 的驻点，检查所有单调子区间，返回第一根，避免在非单调情况下漏掉根。

| N_params_B | D_tokens_B | Q | Q_new | N_eq_B | equivalent_parameter_gain_pct | local_approx_gain_pct | quality_support | equivalent_N_outside_quality_support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.07 | 150 | 0.5867367 | 0.6867367 | 0.079872643 | 14.103776 | 13.98458 | supported_numeric_range | False |
| 0.07 | 150 | 0.74752811 | 0.84752811 | 0.07801258 | 11.446542 | 11.34144 | supported_numeric_range | False |
| 0.07 | 150 | 0.76510653 | 0.86510653 | 0.077853869 | 11.219812 | 11.116334 | supported_numeric_range | False |
| 1 | 150 | 0.5867367 | 0.6867367 | 1.2280471 | 22.804713 | 21.530982 | supported_numeric_range | False |
| 1 | 150 | 0.74752811 | 0.84752811 | 1.1872366 | 18.723657 | 17.780515 | supported_numeric_range | False |
| 1 | 150 | 0.76510653 | 0.86510653 | 1.1837568 | 18.375679 | 17.459332 | supported_numeric_range | False |
| 12 | 150 | 0.5867367 | 0.6867367 | 16.293461 | 35.778838 | 31.596993 | extrapolation | True |
| 12 | 150 | 0.74752811 | 0.84752811 | 15.590733 | 29.922772 | 26.744803 | extrapolation | True |
| 12 | 150 | 0.76510653 | 0.86510653 | 15.531057 | 29.425472 | 26.328419 | extrapolation | True |

主表仅九条，12B 的原工作点及其等价 N 已超质量拟合上限。Q 数值取自真实附件输出，但上述等价结论仍条件于 B6 局部质量模块。

## 10. 等价存在条件与局部解析近似

$Gain_N=-N\partial_NL=\alpha AN^{-\alpha}+u_Q\Delta_Q-s_i\Delta M$。

Gain_N≤0 时标记 no_local_equivalence，不强行求根。Gain_N>0 但搜索范围内无根时标记 no_finite_equivalence_within_range。后者仅表示指定有限范围内无解，不能断言任意无穷范围都无解。折点采用已声明的右侧局部判据。

$\Delta N/N\approx[C_Q(N/N_0)^{-u_Q}/(Q\,Gain_N)]\Delta Q$，这里 $\Delta Q=0.1$。

0.1 不一定是严格小扰动；局部式只解释替代条件，不能替代数值解。

216 个代表网格的状态：{'solved': 216}；局部近似平均绝对相对误差=13.326528%，最大=26.589128%。

进一步复用全部 512 个真实训练配方在相同 N/Q 工作点诊断，状态计数：{'no_finite_equivalence_within_range': 0, 'no_local_equivalence': 7, 'solved': 12281}。training_recipe_equivalence_diagnosis.csv 保留全部记录，no_equivalence_records.csv 保留无局部/无有限解情况。这仅是模型结构诊断，不是新的大模型观测或优化。

本次无局部等价发生的 N=[1.0]B，涉及配方 index=[84, 323]。其中 1B 右侧的判断依赖估算支持的 1B→10B 斜率。这些工作点按指令没有求全局等价根，因此不能说它们在任意更大 N 都无有限解。其余完成搜索的工作点未检测到范围内无根。

| recipe | recipe_index | N_params_B | Q | N_eq_B | equivalent_parameter_gain_pct | status |
| --- | --- | --- | --- | --- | --- | --- |
| P10 | 142 | 0.07 | 0.5867367 | 0.08103581 | 15.765443 | solved |
| P90 | 242 | 0.07 | 0.5867367 | 0.07894918 | 12.784542 | solved |
| P10 | 142 | 0.07 | 0.74752811 | 0.078971252 | 12.816074 | solved |
| P90 | 242 | 0.07 | 0.74752811 | 0.077253339 | 10.361913 | solved |
| P10 | 142 | 0.07 | 0.76510653 | 0.078795185 | 12.564551 | solved |
| P90 | 242 | 0.07 | 0.76510653 | 0.07710859 | 10.155129 | solved |
| P10 | 142 | 1 | 0.5867367 | 1.4537645 | 45.376454 | solved |
| P90 | 242 | 1 | 0.5867367 | 1.1547991 | 15.479905 | solved |
| P10 | 142 | 1 | 0.74752811 | 1.3816274 | 38.16274 | solved |
| P90 | 242 | 1 | 0.74752811 | 1.126034 | 12.603395 | solved |
| P10 | 142 | 1 | 0.76510653 | 1.3755705 | 37.557053 | solved |
| P90 | 242 | 1 | 0.76510653 | 1.1235765 | 12.357653 | solved |
| P10 | 142 | 12 | 0.5867367 | 16.995366 | 41.628046 | solved |
| P90 | 242 | 12 | 0.5867367 | 15.776832 | 31.473596 | solved |
| P10 | 142 | 12 | 0.74752811 | 16.20861 | 35.071754 | solved |
| P90 | 242 | 12 | 0.74752811 | 15.14165 | 26.180419 | solved |
| P10 | 142 | 12 | 0.76510653 | 16.142095 | 34.517461 | solved |
| P90 | 242 | 12 | 0.76510653 | 15.087576 | 25.729798 | solved |

## 11. 十七领域的尺度化边际效应

$\Delta L_j(N)=\lambda_p(N)\Delta M_j$，复用问题一每域增加 1 个百分点的冻结结果。

主表的 ΔM_j 为五个代表基准的等权平均，另保留全部基准案例、零激活及最小/最大范围。存在基准依赖的领域不能按平均值宣称普遍有利或不利。

| domain | DeltaM_j | source_baseline_direction |
| --- | --- | --- |
| train_the_pile_stackexchange | -0.03132934 | lower_loss |
| train_the_pile_ubuntu_irc | -0.030288144 | lower_loss |
| train_the_pile_wikipedia_en | -0.012327216 | lower_loss |
| train_the_pile_arxiv | -0.008899186 | uncertain |
| train_the_pile_dm_mathematics | -0.0085929599 | lower_loss |
| train_the_pile_pile_cc | -0.0065627098 | uncertain |
| train_the_pile_github | -0.004544587 | uncertain |
| train_the_pile_gutenberg_pg_19 | -0.0036215149 | uncertain |
| train_the_pile_pubmed_central | -0.003594814 | uncertain |
| train_the_pile_pubmed_abstracts | -0.0029562979 | uncertain |
| train_the_pile_freelaw | -0.0011570052 | uncertain |
| train_the_pile_uspto_backgrounds | -0.00064375076 | uncertain |
| train_the_pile_hackernews | 0.016683587 | higher_loss |
| train_the_pile_europarl | 0.034131615 | higher_loss |
| train_the_pile_philpapers | 0.037655669 | higher_loss |
| train_the_pile_nih_exporter | 0.048447822 | higher_loss |
| train_the_pile_enron_emails | 0.12091945 | higher_loss |

1M/60M/1B 的正 λ 保持原方向且逐步衰减；10B/70B 的负 λ 使对应方向反转，但仅是 estimated-supported 的条件性结果。完整 17×5 结果见 domain_effect_by_scale.csv。

## 12. 领域组合趋势

$I_{jk}(N)=\lambda_p(N)I_{jk}$。每尺度 136 对、17×17 对称矩阵，主对角线不适用而非真实零偏离。

| scale | negative_mean | positive_mean | stable_negative_across_five_baselines | stable_positive_across_five_baselines | baseline_dependent |
| --- | --- | --- | --- | --- | --- |
| 1M | 70 | 66 | 37 | 25 | 74 |
| 60M | 70 | 66 | 37 | 25 | 74 |
| 1B | 70 | 66 | 37 | 25 | 74 |
| 10B | 66 | 70 | 25 | 37 | 74 |
| 70B | 66 | 70 | 25 | 37 | 74 |

λ 变号时，组合偏离符号同步翻转，基准依赖程度保留。负向称条件性互补趋势，正向称条件性竞争趋势。ILR 与闭合操作本身也会形成组合偏离，不作为因果互补/竞争或真实交互的证据。

## 13. B9/B10 百亿/千亿参数级外推

B9：132 条元数据，N=100.0–10000.0B，D=0.0–36000.0B。无 Loss，非正 D=4 条，全部保留。

B10：128 条估算 Loss。B9.model_name 与 B10.family 精确一对一对应，128 条匹配，4 条仅 B9，匹配 N/D 相同。缺失、重复及全部字段见 B9_B10_diagnosis.md。

参考情景设置 Q=1、p=p_ref，故只预测 L0(N,D)。没有给真实大模型虚构观测 Q 或配方。B9 用于现实尺度锚点，B10 用于估算 Loss 对照，不能验证完整 N-D-Q-p 模型。

| bucket | n | RMSE | MAE | R2 | Pearson | Spearman | mean_bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 128 | 0.0010938278 | 0.00061404883 | 0.99999242 | 0.99999998 | 0.99999428 | 0.00061404883 |
| 100-300B | 65 | 0.0006865892 | 0.00050850145 | 0.99998575 | 0.99999997 | 1 | 0.00050850145 |
| 300-1000B | 43 | 0.00097412178 | 0.00056093114 | 0.99999132 | 0.99999998 | 0.99988664 | 0.00056093114 |
| 1000B_and_above | 20 | 0.0020211657 | 0.0010712809 | 0.99999338 | 1 | 1 | 0.0010712809 |

分桶采用 [100,300)、[300,1000)、[1000,∞)B。统一称“百亿/千亿参数级估算一致性检验”。参数总量/激活参数、稠密/稀疏模型和不同评测口径未统一，因此高一致性也不等于真实同口径泛化性能。

## 14. 超 70B 的 λ 结构敏感性

S-A 主方案：保持 λ_70B。S-B：延续 10B→70B 的 ln N 斜率。S-C：70B 后在 ln N 上线性收缩，到 1000B 为 0，之后保持 0。未按结果好看程度更换主方案。

| N_params_B | recipe | policy | lambda_p | DeltaM | Delta_p | role |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | P10 | S-A | -0.3727259 | -0.21260021 | 0.079241605 | main |
| 100 | P10 | S-B | -0.3840319 | -0.21260021 | 0.081645263 | structural_sensitivity_only |
| 100 | P10 | S-C | -0.3227338 | -0.21260021 | 0.068613274 | structural_sensitivity_only |
| 100 | P90 | S-A | -0.3727259 | 0.20897323 | -0.077889735 | main |
| 100 | P90 | S-B | -0.3840319 | 0.20897323 | -0.080252386 | structural_sensitivity_only |
| 100 | P90 | S-C | -0.3227338 | 0.20897323 | -0.067442725 | structural_sensitivity_only |
| 300 | P10 | S-A | -0.3727259 | -0.21260021 | 0.079241605 | main |
| 300 | P10 | S-B | -0.41885608 | -0.21260021 | 0.08904889 | structural_sensitivity_only |
| 300 | P10 | S-C | -0.16875065 | -0.21260021 | 0.035876423 | structural_sensitivity_only |
| 300 | P90 | S-A | -0.3727259 | 0.20897323 | -0.077889735 | main |
| 300 | P90 | S-B | -0.41885608 | 0.20897323 | -0.087529707 | structural_sensitivity_only |
| 300 | P90 | S-C | -0.16875065 | 0.20897323 | -0.035264367 | structural_sensitivity_only |
| 1000 | P10 | S-A | -0.3727259 | -0.21260021 | 0.079241605 | main |
| 1000 | P10 | S-B | -0.45702001 | -0.21260021 | 0.097162549 | structural_sensitivity_only |
| 1000 | P10 | S-C | -0 | -0.21260021 | 0 | structural_sensitivity_only |
| 1000 | P90 | S-A | -0.3727259 | 0.20897323 | -0.077889735 | main |
| 1000 | P90 | S-B | -0.45702001 | 0.20897323 | -0.095504946 | structural_sensitivity_only |
| 1000 | P90 | S-C | -0 | 0.20897323 | -0 | structural_sensitivity_only |
| 3000 | P10 | S-A | -0.3727259 | -0.21260021 | 0.079241605 | main |
| 3000 | P10 | S-B | -0.49184418 | -0.21260021 | 0.10456618 | structural_sensitivity_only |
| 3000 | P10 | S-C | -0 | -0.21260021 | 0 | structural_sensitivity_only |
| 3000 | P90 | S-A | -0.3727259 | 0.20897323 | -0.077889735 | main |
| 3000 | P90 | S-B | -0.49184418 | 0.20897323 | -0.10278227 | structural_sensitivity_only |
| 3000 | P90 | S-C | -0 | 0.20897323 | -0 | structural_sensitivity_only |
| 10000 | P10 | S-A | -0.3727259 | -0.21260021 | 0.079241605 | main |
| 10000 | P10 | S-B | -0.53000811 | -0.21260021 | 0.11267984 | structural_sensitivity_only |
| 10000 | P10 | S-C | -0 | -0.21260021 | 0 | structural_sensitivity_only |
| 10000 | P90 | S-A | -0.3727259 | 0.20897323 | -0.077889735 | main |
| 10000 | P90 | S-B | -0.53000811 | 0.20897323 | -0.11075751 | structural_sensitivity_only |
| 10000 | P90 | S-C | -0 | 0.20897323 | -0 | structural_sensitivity_only |

代表配方的三规则最大 Δp 跨方案差值为 0.11267984。**超70B领域配比修正具有高外推不确定性。**一类规则保留偏移，一类持续增大，一类收缩至零；B9/B10 无法验证 p 项，不能据此选出真实机制。

## 15. 证据等级与分模块验证

| module | parameters | evidence | support |
| --- | --- | --- | --- |
| L0 | ["E", "A", "B", "alpha", "beta"] | Class A | B1 observations; frozen B3/B4/B5 validations with original comparability limits |
| Delta_Q | ["C_Q", "u_Q"] | Class B | B6 semisynthetic local calibration; B7 difference validation failed |
| M(p) | ["frozen ILR first-order ElasticNet"] | Class C | Question 1 training and true mixture checks at 1M/60M/1B |
| lambda_p large N | ["10B knot", "70B knot", "segments above 1B"] | Class D | Estimated recipe tables; no observed large-scale reversal |

| module | dataset | stage | target | n | RMSE | R2 | limitation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L0 | n.a. | n.a. | n.a. | 147 | 0.00022023037 | 0.99999923 | Frozen B1 leave-trajectory-out validation; no refitting in this stage |
| L0 | n.a. | n.a. | n.a. | 147 | 0.00018181804 | 0.99999947 | Frozen B1 leave-trajectory-out validation; no refitting in this stage |
| L0 | n.a. | n.a. | n.a. | 147 | 0.00013511027 | 0.99999971 | Frozen B1 leave-trajectory-out validation; no refitting in this stage |
| L0 | n.a. | n.a. | n.a. | 147 | 0.00015459771 | 0.99999962 | Frozen B1 leave-trajectory-out validation; no refitting in this stage |
| L0 | n.a. | n.a. | n.a. | 147 | 0.00013191675 | 0.99999972 | Frozen B1 leave-trajectory-out validation; no refitting in this stage |
| L0 | n.a. | n.a. | n.a. | 147 | 0.00010197532 | 0.99999983 | Frozen B1 leave-trajectory-out validation; no refitting in this stage |
| L0 | n.a. | n.a. | n.a. | 147 | 0.00011441463 | 0.99999979 | Frozen B1 leave-trajectory-out validation; no refitting in this stage |
| L0 | n.a. | n.a. | n.a. | 147 | 0.0001350424 | 0.99999971 | Frozen B1 leave-trajectory-out validation; no refitting in this stage |
| L0 | B3 | n.a. | n.a. | 4000 | 0.0038212821 | 0.99995774 | B3 interpolation consistency; B4/B5 evaluation scopes not absolutely comparable |
| L0 | B4 | n.a. | n.a. | 57 | 0.29266668 | 0.60450489 | B3 interpolation consistency; B4/B5 evaluation scopes not absolutely comparable |
| L0 | B5 | n.a. | n.a. | 44 | 0.19759507 | 0.73055911 | B3 interpolation consistency; B4/B5 evaluation scopes not absolutely comparable |
| Delta_Q | n.a. | B6_5fold_GroupCV_OOF | absolute_Loss | 360 | 0.069920473 | 0.95893904 | B6 calibrated local correction only; frozen S3 v near zero is removed without refitting |
| Delta_Q | n.a. | B6_5fold_GroupCV_OOF | within_ND_DeltaL | 315 | 0.087072519 | 0.59491474 | B6 calibrated local correction only; frozen S3 v near zero is removed without refitting |
| Delta_Q | n.a. | B7_fixed_B6_parameters | absolute_Loss | 90 | 0.071513543 | 0.95060039 | B7 nonoverlap difference R2 fails; not a global quality law |
| Delta_Q | n.a. | B7_fixed_B6_parameters | within_ND_DeltaL | 90 | 0.083817946 | -0.040868046 | B7 nonoverlap difference R2 fails; not a global quality law |
| M(p) | test_1M | n.a. | n.a. | 256 | 0.24206725 | 0.24014133 | Question 1 frozen absolute predictions; centered lambda here is fitted calibration, not independent validation |
| M(p) | test_60M | n.a. | n.a. | 256 | 1.5247194 | -47.52535 | Question 1 frozen absolute predictions; centered lambda here is fitted calibration, not independent validation |
| M(p) | test_1B | n.a. | n.a. | 64 | 2.9148113 | -3148.2884 | Question 1 frozen absolute predictions; centered lambda here is fitted calibration, not independent validation |
| M(p) | estimated_10B | n.a. | n.a. | 63 | 3.5555545 | -1367.4667 | 10B/70B estimated training subsets, not independent observed experiments |
| M(p) | estimated_70B | n.a. | n.a. | 63 | 3.9383733 | -1428.7504 | 10B/70B estimated training subsets, not independent observed experiments |

Class A：B1 主拟合经典五参数。Class B：B6 半合成局部质量校准。Class C：第一问真实配比实验的 M(p)。Class D：10B/70B 估算 λ 节点及其大尺度段。大尺度质量项更属于外推，不能借 Class D 配比证据补强。

## 16. 局限性

不存在完整 (N,D,Q,p,L) 联合直接观测。采用 modular calibration + component-wise validation，无法进行单一完整联合外部验证。相加结构未验证 Q×p 等联合交互。不同尺度配方覆盖和评测分布可能混杂传递系数变化。第一问领域 Q 与 B6 Q 的语义一致性没有联合观测支持。估算表不会变成真实实验。等价参数数值是冻结模型内的条件性替代，不是训练实测承诺，也未纳入质量改善成本或算力约束优化。

## 17. 结论与停止点

经典骨架、局部质量修正、尺度调制配比项可按明确锚点组合，所有拟合参数保留冻结值。质量提升与参数扩张的等价使用精确求根，局部近似只解释机制。配比及组合方向反转依赖估算表，超 70B 边界规则具有结构不确定性。B10 只支持经典骨架的估算一致性，不能补充缺失的完整联合验证。问题二完成后停止，不进入问题三。

复现：python -B scripts/16_analyze_generalized_scaling.py。独立核验：python -B scripts/16_verify_generalized_scaling.py。全部结果位于 outputs/scaling/phase4，图表位于 outputs/figures/question2_phase4。
