# 问题一第3小问最终报告



最终主模型：**M1_ElasticNet**。第一小问 Q、第二小问 K 和上一轮 ILR/权重/模型文件均保持锁定。



## 1. 输入、训练边界与冻结核验



A4/A5 为唯一训练数据；A6–A11 只作原值检验；A12–A15 的 10B/70B Loss 为估算值，且配方是训练集子集。没有用检验 Loss 调参数、截距或尺度。

实际配方数：train_1M=512, test_1M=256, test_60M=256, test_1B=64, estimated_10B=63, estimated_70B=63。受保护文件 2351 个，SHA-256 全部一致。



已有真实同配方 1M/60M 的 256 条 L_mean：Spearman=0.9657980945296404。这是排序跨尺度稳定性的核心观测证据，不能把不同配方的 1B 表按 index 强行对应。



## 2. 为什么配比不能直接 OLS；ILR、零配比与主模型



17 配比和为 1，原始配比加截距的 OLS 存在精确共线性；删去一域虽可计算，但解释依赖参考域且忽略相对比例几何。闭合后采用 16 维正交 Helmert ILR，ILR(p)=H log(p)。

零项按训练折最小正闭合配比的一半进行乘法替换，非零项同比收缩，保持正比值。主模型沿用上一轮固定参数：ElasticNet alpha=0.03、l1_ratio=0.1；Ridge alpha=10 作数值对照。新模型中的 ILR 零参数、X/y 标准化均在每个 CV 训练折内拟合。

这是数值正则化，不代表结构零对应真实正质量。ILR 中心由冻结零替换后的几何中心求得，分量中位数基准由原闭合配比求得。

ElasticNet 已按人工确认锁定为主模型，提供收缩与有限稀疏性；Ridge 上轮 CV RMSE 略低，故不声称 ElasticNet 数值最优。上轮二阶模型没有获得嵌套 CV 收益，本轮没有恢复二阶主模型。



## 3. 17 域边际关联与稳定性



基准为 A4 ILR 中心、闭合分量中位数、冻结 ElasticNet 预测 P10/P50/P90 的真实配方，共五个。基准不使用检验 Loss 选择。对目标域增加绝对 0.005/0.01/0.02（0.5/1/2 个百分点），其余域按原比例缩减。

Delta L_j=f(p perturb j)-f(p)。以下为 ElasticNet 增加 1 个百分点的五基准均值与范围。strict_direction 只有两个模型、三个步长、五个基准共 30 个效应全部同向才成立；uncertain 不给总体有利/不利结论。数值零容差 1e-10 仅防浮点误判，不是显著性阈值。



| domain | mean_effect | min_effect | max_effect | strict_direction |
| --- | --- | --- | --- | --- |
| arxiv | -0.008899 | -0.017529 | 0.001116 | uncertain |
| freelaw | -0.001157 | -0.004803 | 0.001484 | uncertain |
| nih_exporter | 0.048448 | 0.009112 | 0.074645 | higher_loss |
| pubmed_central | -0.003595 | -0.017504 | 0.001587 | uncertain |
| wikipedia_en | -0.012327 | -0.029577 | -0.000630 | lower_loss |
| dm_mathematics | -0.008593 | -0.014011 | -0.004669 | lower_loss |
| github | -0.004545 | -0.012712 | 0.001148 | uncertain |
| philpapers | 0.037656 | 0.005166 | 0.078981 | higher_loss |
| stackexchange | -0.031329 | -0.085865 | -0.001694 | lower_loss |
| enron_emails | 0.120919 | 0.051960 | 0.234141 | higher_loss |
| gutenberg_pg_19 | -0.003622 | -0.009403 | 0.000784 | uncertain |
| pile_cc | -0.006563 | -0.032238 | 0.001220 | uncertain |
| ubuntu_irc | -0.030288 | -0.114828 | -0.003394 | lower_loss |
| europarl | 0.034132 | 0.005103 | 0.044591 | higher_loss |
| hackernews | 0.016684 | 0.003066 | 0.025548 | higher_loss |
| pubmed_abstracts | -0.002956 | -0.013444 | 0.000423 | uncertain |
| uspto_backgrounds | -0.000644 | -0.003722 | 0.001076 | uncertain |



这些是相对于按比例减小其余领域的模型关联，不是因果作用或孤立单域贡献。均值是五个代表基准的等权摘要，不是全体 A4 配方的平均处理效应。零激活效应包含从替换质量到实际正配比的变化，不能当作在 0 处的导数。



## 4. 双域联合扰动与组合偏离



同时给 j,k 各增加相同步长，其余领域按比例缩减。I_jk=Delta L_jk-Delta L_j-Delta L_k，仅称组合偏离指数。即使一阶 ILR 模型也因闭合和对数变换产生非加性，不能据此证明真实交互或因果互补。

136 对中严格同向组合偏离 56 对：负向 36，正向 20。完整结果在 domain_joint_effects.csv、stage2/joint_effect_stability.csv。以下仅列绝对平均偏离最大的同向对：



| domain_1 | domain_2 | direction | mean_deviation | min_deviation | max_deviation |
| --- | --- | --- | --- | --- | --- |
| enron_emails | uspto_backgrounds | higher_loss | 0.001159 | 0.000030 | 0.003461 |
| enron_emails | pile_cc | higher_loss | 0.001068 | 0.000124 | 0.002924 |
| philpapers | enron_emails | higher_loss | 0.000998 | 0.000056 | 0.002388 |
| enron_emails | pubmed_abstracts | higher_loss | 0.000936 | 0.000008 | 0.002894 |
| pubmed_central | enron_emails | higher_loss | 0.000934 | 0.000175 | 0.002030 |
| nih_exporter | enron_emails | higher_loss | 0.000910 | 0.000055 | 0.002273 |
| enron_emails | gutenberg_pg_19 | higher_loss | 0.000904 | 0.000014 | 0.002762 |
| dm_mathematics | ubuntu_irc | lower_loss | -0.000873 | -0.001891 | -0.000201 |
| github | enron_emails | higher_loss | 0.000865 | 0.000065 | 0.002304 |
| enron_emails | europarl | higher_loss | 0.000852 | 0.000050 | 0.002065 |
| freelaw | enron_emails | higher_loss | 0.000824 | 0.000070 | 0.001927 |
| enron_emails | hackernews | higher_loss | 0.000821 | 0.000047 | 0.001913 |



负向同向对可描述为扰动情景下的互补趋势，正向为竞争趋势。小偏离未做抽样显著性检验，不把数值同号等同于稳定的真实交互证据。



## 5. 13 个验证域的异质性



对 A5 的每个验证 Loss 分别在 A4/A5 内训练一阶 ElasticNet/Ridge；五折选择超参数，得到 26 个辅助诊断模型，没有用检验表选择。选择 CV 的最小 RMSE 不是无偏嵌套泛化评估。

17×13 共 221 个训练域/验证域单元中，两模型均值方向不一致 0 个；ElasticNet 在不同验证域出现正负混合方向的训练领域 15 个。因此宏观 L_mean 会掩盖部分领域差异。矩阵和两模型热图已输出。



## 6. 17 域质量映射与 Qp



Qp=sum(p_j Q_j)，使用原配比闭合值，不使用零替换后的虚拟质量。Q_j 读取第一小问已锁定的 A1 各领域 10% 双侧截尾均值及 bootstrap 95% CI。Qp 是配比 p 的确定函数，是外部质量评价构造的质量加权配比特征，不是独立实验变量。

**A16 只有 3 个 direct、3 个 near_direct 的确定关联，11 个 inferred 的 quality_domain 实际为 (none)。Scheme A 对这 11 个添加有明确理由的研究者代理假设，原 A16 字段、类型、note 都保留。绝不能把代理 Q_j 写成该语料实测质量。**



| mixture_domain_short | A16_quality_domain | quality_domain | mapping_type | Q_domain | Q_CI_low | Q_CI_high |
| --- | --- | --- | --- | --- | --- | --- |
| arxiv | arxiv | arxiv | direct | 0.765107 | 0.762056 | 0.768049 |
| freelaw | (none) | wikipedia | inferred | 0.681264 | 0.679775 | 0.682750 |
| nih_exporter | (none) | arxiv | inferred | 0.765107 | 0.762056 | 0.768049 |
| pubmed_central | (none) | arxiv | inferred | 0.765107 | 0.762056 | 0.768049 |
| wikipedia_en | wikipedia | wikipedia | near_direct | 0.681264 | 0.679775 | 0.682750 |
| dm_mathematics | (none) | arxiv | inferred | 0.765107 | 0.762056 | 0.768049 |
| github | github | github | direct | 0.648929 | 0.647252 | 0.650606 |
| philpapers | (none) | arxiv | inferred | 0.765107 | 0.762056 | 0.768049 |
| stackexchange | stackexchange | stackexchange | direct | 0.747528 | 0.746350 | 0.748648 |
| enron_emails | (none) | commoncrawl | inferred | 0.759731 | 0.758372 | 0.761150 |
| gutenberg_pg_19 | book | book | near_direct | 0.586737 | 0.575536 | 0.597500 |
| pile_cc | commoncrawl | commoncrawl | near_direct | 0.759731 | 0.758372 | 0.761150 |
| ubuntu_irc | (none) | stackexchange | inferred | 0.747528 | 0.746350 | 0.748648 |
| europarl | (none) | wikipedia | inferred | 0.681264 | 0.679775 | 0.682750 |
| hackernews | (none) | commoncrawl | inferred | 0.759731 | 0.758372 | 0.761150 |
| pubmed_abstracts | (none) | arxiv | inferred | 0.765107 | 0.762056 | 0.768049 |
| uspto_backgrounds | (none) | arxiv | inferred | 0.765107 | 0.762056 | 0.768049 |



## 7. M3 的嵌套 CV 与直接检验



M3=一阶 ILR ElasticNet + Qp。沿用原模型标准化与同一 ElasticNet 网格；五个外折严格复用上一轮划分，各外训练内再四折选参，最后仅全 A4/A5 五折选最终超参数。M1 使用冻结的原 OOF 预测，没有重调。

M3 最终参数：{'regressor__regression__alpha': 0.03, 'regressor__regression__l1_ratio': 0.1}。gamma 原 Loss/Qp 单位=-1.007148338；还原公式 gamma=s_y beta_Qp/s_Qp。该系数受确定性配比特征与正则化共线性影响，不具有因果解释。



| model | RMSE | MAE | R2 | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- |
| M1_ElasticNet | 0.281252 | 0.218983 | 0.248673 | 0.498914 | 0.517726 |
| M3_Qp | 0.280460 | 0.217559 | 0.252900 | 0.502968 | 0.520152 |
| M4_Qp_Kp | 0.274681 | 0.211986 | 0.283374 | 0.533123 | 0.553545 |



| model | dataset | RMSE | MAE | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- |
| M1_ElasticNet | test_1M | 0.242067 | 0.194978 | 0.526487 | 0.500614 |
| M1_ElasticNet | test_60M | 1.524719 | 1.511972 | 0.490627 | 0.474158 |
| M1_ElasticNet | test_1B | 2.914811 | 2.913280 | 0.681197 | 0.642170 |
| M1_Ridge | test_1M | 0.242175 | 0.194783 | 0.526805 | 0.501667 |
| M1_Ridge | test_60M | 1.524597 | 1.511739 | 0.492346 | 0.476981 |
| M1_Ridge | test_1B | 2.906459 | 2.904826 | 0.684098 | 0.639973 |
| M3_Qp | test_1M | 0.241442 | 0.195401 | 0.529971 | 0.504607 |
| M3_Qp | test_60M | 1.524305 | 1.511650 | 0.497287 | 0.481905 |
| M3_Qp | test_1B | 2.911559 | 2.910014 | 0.677554 | 0.646429 |
| M4_Qp_Kp | test_1M | 0.245954 | 0.196832 | 0.514149 | 0.496733 |
| M4_Qp_Kp | test_60M | 1.522121 | 1.508359 | 0.489363 | 0.483030 |
| M4_Qp_Kp | test_1B | 2.894148 | 2.892345 | 0.642130 | 0.588553 |



最终判据：M3 nested RMSE difference=-0.000792262; 1M test RMSE difference=-0.000625345; outer RMSE improved 4/5; three observed scales ranking improved 3/3; conservative retention requires consistent outer-fold and observed-scale improvement。Qp 保留为辅助解释特征，不进入最终主模型。全部预测未用 test bias 校准。

M3 相对 M1 的嵌套 CV RMSE 降幅 0.2817%。改善外折 4/5；1M test MAE 差 +0.000422169。综合改善幅度和外折变化，本轮不足以证明质量特征具有稳定的实用增益。保守规则要求各外折方向一致，属于操作性稳定标准，不是统计显著性检验。



## 8. 映射不确定性：CI 扰动与替代语义代理



Scheme B：seed=2026，对每个 inferred 域独立 Uniform(Q_CI_low,Q_CI_high) 抽样 500 次，direct/near_direct 固定。每次重算 Qp、用 Scheme A 已由训练 CV 选定的各外折和全训练超参数重拟合 M3，重新计算 OOF 与检验预测。

这是条件于已选择超参数的 Monte Carlo，不重复选择超参数，不包含模型选择不确定性。独立抽样假设允许同代理域的不同 inferred 域有所区别，未建模来源相关性。

500 次 gamma>0 比例 0.000000；gamma<0 比例 1.000000；OOF RMSE 优于冻结 M1 的比例 1.000000。



| metric | mean | std | P025 | median | P975 |
| --- | --- | --- | --- | --- | --- |
| gamma | -1.010709 | 0.035706 | -1.081415 | -1.007923 | -0.953397 |
| CV_RMSE_difference_from_M1 | -0.000796 | 0.000043 | -0.000880 | -0.000795 | -0.000724 |
| test_1M_RMSE | 0.241434 | 0.000029 | 0.241368 | 0.241440 | 0.241477 |
| test_1M_Spearman | 0.504599 | 0.000270 | 0.503998 | 0.504607 | 0.505147 |
| test_60M_Spearman | 0.481935 | 0.000213 | 0.481464 | 0.481936 | 0.482337 |
| test_1B_Spearman | 0.646961 | 0.001357 | 0.645421 | 0.646429 | 0.649634 |



CI 仅刻画已有代理质量域截尾均值的抽样波动，不包含“语义关联选错”的偏差，不能由窄 CI 下稳定推出映射可靠。另设未经检验 Loss 选择的替代语义代理组合，在同一 A4/A5 嵌套 CV 内分析：



| scheme | gamma_original_loss_per_Qp | CV_RMSE | CV_MAE | CV_R2 | CV_Spearman |
| --- | --- | --- | --- | --- | --- |
| A_assumed_proxy | -1.007148 | 0.280460 | 0.217559 | 0.252900 | 0.520152 |
| alternative_assumed_proxy | -0.510826 | 0.281212 | 0.218675 | 0.248890 | 0.516190 |



替代代理下 gamma 相对 Scheme A 比值 0.507200；替代方案相对 M1 的 CV RMSE 差为 -0.000040573。因此 CI 内稳定与代理语义变化稳定必须分开，不能由前者推断后者具有相同增益幅度。

替代映射列出在 stage2/alternative_domain_quality_mapping.csv。各方案不按 test 结果择优替换主映射。直接映射也只代表来源域名对应，不证明 A1 文本质量均值准确代表 A4 中全部同名训练语料。



## 9. Kp 仅作辅助检查



读取已有 A1_sample_conflict.csv 的 K，分别按原领域作两端各 10% 截尾均值，不重新计算 K。通过相同显式代理映射形成 K_j 和 Kp=sum(p_j K_j)。M4=M3+eta Kp，仍在 A4/A5 内嵌套 CV。

M4 eta 原 Loss/Kp 单位=-10.483044856，与 M3 的 OOF RMSE 差为 -0.005779387。Kp 仅保留辅助敏感性结果；CV 获得改善，但泛化检验未稳定复现，不进入本轮最终主模型。K 是评价冲突，不是低质量程度，未作为惩罚项。

M4 的 1M test RMSE=0.245954，M3=0.241442；1B Spearman M4=0.588553，M3=0.646429。eta 符号由训练拟合自由决定，不能解释为质量惩罚。M4 gamma=-7.233363，加入 Kp 后系数幅度变化，提示确定性配比描述符间的依赖与正则化分配影响系数，不能解释为独立因果贡献。



## 10. 尺度迁移诊断，不做偏置校准



残差 e=y-yhat；centered RMSE=sqrt(mean((e-mean(e))^2)) 仅解释偏移贡献。它没有提供可在未知目标尺度应用的偏置，不能称为校准后的预测性能。



| model | dataset | mean_bias | median_bias | residual_std | RMSE | centered_RMSE_diagnostic_only | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1_ElasticNet | test_1M | -0.052601 | -0.062827 | 0.236746 | 0.242067 | 0.236283 | 0.526487 | 0.500614 |
| M1_ElasticNet | test_60M | -1.511972 | -1.521390 | 0.197132 | 1.524719 | 0.196747 | 0.490627 | 0.474158 |
| M1_ElasticNet | test_1B | -2.913280 | -2.913219 | 0.095226 | 2.914811 | 0.094479 | 0.681197 | 0.642170 |
| M1_Ridge | test_1M | -0.052368 | -0.063474 | 0.236908 | 0.242175 | 0.236445 | 0.526805 | 0.501667 |
| M1_Ridge | test_60M | -1.511739 | -1.518194 | 0.197975 | 1.524597 | 0.197587 | 0.492346 | 0.476981 |
| M1_Ridge | test_1B | -2.904826 | -2.903083 | 0.098175 | 2.906459 | 0.097405 | 0.684098 | 0.639973 |



绝对误差跨尺度显著增大，去除样本均值残差后的诊断误差远小于原 RMSE，表明整体 Loss 水平偏移是主要来源；正排序相关仍显示有限的关联迁移。60M 与 1B 的配方集合不同，不能把其相关大小直接解读为规模单调变化。



## 11. 10B/70B 外推一致性与证据边界



所有超参数选择在查看估算性能之前完成并冻结；按闭合配比向量匹配训练子集，不按跨表 index 假定相同配方。10B、70B 各 63 个配方均属于 A4，因此只称估算训练配方子集的外推一致性检验。



| dataset | model | RMSE | MAE | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- |
| estimated_10B | M1_ElasticNet | 3.555555 | 3.548236 | -0.525356 | -0.532258 |
| estimated_10B | M1_Ridge | 3.555853 | 3.548282 | -0.526625 | -0.532018 |
| estimated_10B | M3_Qp | 3.554782 | 3.547176 | -0.526293 | -0.558948 |
| estimated_70B | M1_ElasticNet | 3.938373 | 3.931149 | -0.580932 | -0.580693 |
| estimated_70B | M1_Ridge | 3.938655 | 3.931195 | -0.581410 | -0.579445 |
| estimated_70B | M3_Qp | 3.937590 | 3.930089 | -0.583005 | -0.607239 |



| comparison | n | Spearman | diagnostic_ratio_estimated_over_1M | diagnostic_max_abs_scaling_deviation |
| --- | --- | --- | --- | --- |
| M1_ElasticNet_prediction_vs_estimated_10B | 63 | -0.532258 |  |  |
| M1_Ridge_prediction_vs_estimated_10B | 63 | -0.532018 |  |  |
| M3_Qp_prediction_vs_estimated_10B | 63 | -0.558948 |  |  |
| observed_train_1M_vs_estimated_10B | 63 | -0.739247 | 0.332755 | 0.515502 |
| M1_ElasticNet_prediction_vs_estimated_70B | 63 | -0.580693 |  |  |
| M1_Ridge_prediction_vs_estimated_70B | 63 | -0.579445 |  |  |
| M3_Qp_prediction_vs_estimated_70B | 63 | -0.607239 |  |  |
| observed_train_1M_vs_estimated_70B | 63 | -0.813892 | 0.261480 | 0.470502 |
| estimated_10B_vs_estimated_70B_same_recipes | 63 | 0.990015 |  |  |



独立核验后的实际结论：主模型与 10B/70B 估算宏观 Loss 的 Spearman 分别为 -0.532258/-0.580693，排序发生反向关联，**不支持主模型在这两张估算表上的排序迁移**。M3 同样没有修复反向关系。10B 与 70B 彼此 Spearman=0.990015，仅说明两张估算表之间排序相近，不能代替“1M 模型预测与大规模估算表一致”。

同配方 1M 实测宏观 Loss 与 10B/70B 估算宏观 Loss 的 Spearman 分别为 -0.739247/-0.813892，说明反向关系并非只由模型拟合误差造成。配方已按向量确认一致；但这些是估算值，不能据此断言真实大模型发生相同反转。

逐验证域关系与宏观均值关系不同，估算值也不是原 1M 宏观 Loss 的简单正比例缩放。生成关系诊断在 stage2/estimated_*_generation_relationship.csv；其中比例系数仅是对已有表的诊断数，没有用于拟合、校准或构造新 Loss。现有附件不足以识别估算生成机制，不能把诊断比例写成已知 Scaling Law。



## 12. 最终选择、局限性与复用文件



主模型采用 M1_ElasticNet；Ridge 保留数值对照。原 Q 与 K 分别表示综合质量与评价冲突。参数与选择理由记录在 stage2/final_model_policy.json，冻结阶段清单记录在 stage2/freeze_before_extrapolation.json。

局限性：512 个小规模训练配方、较多零配比、对零激活与基准敏感；17 域与 13 验证域不齐；没有因果实验识别；新增代理映射缺直接质量观测；Qp/Kp 均由 p 确定且相互相关；嵌套 CV 差异未证明显著；估算表不是外部观测，且配方属于训练子集。

复用：domain_marginal_effects.csv、domain_effect_stability.csv、domain_joint_effects.csv、train_domain_vs_validation_loss_effects.csv、domain_quality_mapping.csv、recipe_quality_conflict_descriptors.csv、stage2/model_comparison_nested_CV.csv、stage2/model_comparison_validation.csv、estimated_scale_consistency.csv。

**本轮完成后停止，未开始问题二，未重新拟合前两小问结果。**

