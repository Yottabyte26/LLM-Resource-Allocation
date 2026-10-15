# 问题一第3小问：阶段1配比/Loss建模与检验

本轮已完成 A4–A15 数据诊断、响应变量、单纯形/零值处理、ILR、M0/M1/M2 与 A6–A11 检验。原 Q、K、预处理及 CRITIC 全部冻结；未引入 Q/K，未读取 A16 映射用于建模，未预测10B/70B，未建立模型规模方程或 Scaling Law。

## 真实结构

| dataset | n | provenance |
| --- | --- | --- |
| train_1M | 512 | observed |
| test_1M | 256 | observed |
| test_60M | 256 | observed |
| test_1B | 64 | observed |
| estimated_10B | 63 | estimated |
| estimated_70B | 63 | estimated |

详细十二文件、17配比列、13Loss列、逐域分布、零值/相关矩阵与行和诊断见 outputs/mixture/phase1_data_diagnosis.md。所有对应表的 index 唯一并在同一实验内一一配对。

## 响应与组合变换

主响应为13域等权 L_mean；敏感性响应 L_trim 去掉最高最低各一个后对11域取均值。两个响应各自拟合相同候选和CV流程，未选择某个有利Loss域。

先闭合 p/sum(p)。零替换 δ=min(0.5×闭合训练配比最小正值,1/(2×17))；A4全量 min_positive=0.000998003992016，最终 δ=0.000499001996008。每行零坐标变为 δ，非零坐标乘 (1−零个数×δ)，保留非零比值。测试统一使用训练参数；CV中则仅从当前折训练部分拟合 δ。原始零可能为结构性缺席或舍入，替换不是真实观测恢复，因此结论依赖此处理。

ILR 用固定列顺序的16×17 Helmert正交对比矩阵 H：x=log(p_replaced)Hᵀ，HHᵀ=I，H1=0。第k个坐标为 sqrt(k/(k+1))×log(前k域几何均值/第k+1域)，k=1..16。单纯形约束得到16维独立坐标；CLR另存仅供解释，不直接对17原始p做OLS。

## 候选模型与无泄漏CV

M0=训练平均Loss；M1-Ridge/M1-ElasticNet=16维ILR一阶模型；M2-ElasticNet=16个一阶坐标+120个两两交叉乘积，共136个特征，没有无约束OLS。所有X标准化在管道内，y也在每个训练折内标准化；预测还原到原Loss单位。L1筛选依赖所选ILR基，不将某个ILR交叉系数直接称为某对原领域因果互补。

固定种子 2026。外层5折、内层4折嵌套CV；同配方组不得跨折。最终参数在全部A4/A5上用5折选取并重拟合，只用于之后的直接检验。最佳模型按训练嵌套OOF RMSE选为 M1_Ridge，测试结果不参与选择。所有CV折、完整网格及外层模型参数已保存。

Ridge alpha 网格：[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]；ElasticNet alpha：[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]；l1_ratio：[0.1, 0.5, 0.9].

### L_mean 嵌套CV

| model | RMSE | MAE | R2 | Pearson | Spearman | mean_fold_RMSE | std_fold_RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | 0.325398 | 0.256870 | -0.005695 | -0.100875 | -0.091317 | 0.324716 | 0.022639 |
| M1_Ridge | 0.280522 | 0.218211 | 0.252568 | 0.503294 | 0.517338 | 0.279551 | 0.025481 |
| M1_ElasticNet | 0.281252 | 0.218983 | 0.248673 | 0.498914 | 0.517726 | 0.280193 | 0.026708 |
| M2_ElasticNet | 0.288406 | 0.223565 | 0.209965 | 0.462172 | 0.484307 | 0.287283 | 0.027826 |

### L_trim 敏感性

| model | RMSE | MAE | R2 | Pearson | Spearman | mean_fold_RMSE | std_fold_RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | 0.307486 | 0.240903 | -0.004509 | -0.089709 | -0.083765 | 0.306516 | 0.026590 |
| M1_Ridge | 0.272805 | 0.210054 | 0.209307 | 0.458764 | 0.473577 | 0.271730 | 0.026541 |
| M1_ElasticNet | 0.272892 | 0.210369 | 0.208804 | 0.456976 | 0.474765 | 0.271731 | 0.027622 |
| M2_ElasticNet | 0.280706 | 0.216967 | 0.162840 | 0.404877 | 0.426410 | 0.279484 | 0.028805 |

数值上 M1-Ridge 的 L_mean 嵌套 RMSE=0.280522，M1-ElasticNet=0.281252，差值仅 0.000730；不能宣称显著优越。按用户解释优先的要求，保留 M1-ElasticNet 为可解释主方案候选，Ridge 为数值最佳对照，等待人工确认，不凭检验集选择最终模型。M2-ElasticNet 的嵌套 RMSE=0.288406，暂未支持加入交互项。

### 最终训练拟合与复杂度

| response | model | training_RMSE | training_R2 | nested_OOF_RMSE | n_model_features | n_nonzero_coefficients | parameters |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L_mean | M0 | 0.324475 | 0.000000 | 0.325398 | 0.000000 | 0.000000 | {} |
| L_mean | M1_Ridge | 0.272215 | 0.296181 | 0.280522 | 16.000000 | 16.000000 | {"regressor__regression__alpha": 10.0} |
| L_mean | M1_ElasticNet | 0.272350 | 0.295481 | 0.281252 | 16.000000 | 15.000000 | {"regressor__regression__alpha": 0.03, "regressor__regression__l1_ratio": 0.1} |
| L_mean | M2_ElasticNet | 0.273070 | 0.291753 | 0.288406 | 136.000000 | 28.000000 | {"regressor__regression__alpha": 0.1, "regressor__regression__l1_ratio": 0.5} |
| L_trim | M0 | 0.306795 | 0.000000 | 0.307486 | 0.000000 | 0.000000 | {} |
| L_trim | M1_Ridge | 0.264965 | 0.254097 | 0.272805 | 16.000000 | 16.000000 | {"regressor__regression__alpha": 10.0} |
| L_trim | M1_ElasticNet | 0.265099 | 0.253346 | 0.272892 | 16.000000 | 16.000000 | {"regressor__regression__alpha": 0.03, "regressor__regression__l1_ratio": 0.1} |
| L_trim | M2_ElasticNet | 0.258561 | 0.289721 | 0.280706 | 136.000000 | 66.000000 | {"regressor__regression__alpha": 0.3, "regressor__regression__l1_ratio": 0.1} |

训练误差小于嵌套OOF误差是乐观偏差诊断；M2是否增加泛化能力须看嵌套CV和同尺度1M检验，不能仅看训练拟合。跨模型CV差距很小时不声称统计显著优越；这里只给出真实误差及折间离散。

一阶 Ridge 的训练 RMSE=0.272215、嵌套 OOF=0.280522，未见严重训练拟合崩溃。M2 在 L_trim 下训练 RMSE=0.258561、嵌套 OOF=0.280706；二阶复杂度增加但泛化未改善，表现出额外拟合乐观/过拟合风险，当前不作为优选。跨尺度绝对偏差不能单独归因于过拟合，模型没有任何规模变量。

## 三尺度直接检验

| dataset | model | RMSE | MAE | R2 | Pearson | Spearman | prediction_bias_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test_1M | M0 | 0.284615 | 0.225599 | -0.050452 | 未定义 | 未定义 | 0.062375 |
| test_60M | M0 | 1.537406 | 1.521746 | -48.336249 | 未定义 | 未定义 | 1.521746 |
| test_1B | M0 | 3.119519 | 3.119086 | -3606.171591 | 未定义 | 未定义 | 3.119086 |
| test_1M | M1_Ridge | 0.242175 | 0.194783 | 0.239465 | 0.526805 | 0.501667 | 0.052368 |
| test_60M | M1_Ridge | 1.524597 | 1.511739 | -47.517569 | 0.492346 | 0.476981 | 1.511739 |
| test_1B | M1_Ridge | 2.906459 | 2.904826 | -3130.265543 | 0.684098 | 0.639973 | 2.904826 |
| test_1M | M1_ElasticNet | 0.242067 | 0.194978 | 0.240141 | 0.526487 | 0.500614 | 0.052601 |
| test_60M | M1_ElasticNet | 1.524719 | 1.511972 | -47.525350 | 0.490627 | 0.474158 | 1.511972 |
| test_1B | M1_ElasticNet | 2.914811 | 2.913280 | -3148.288428 | 0.681197 | 0.642170 | 2.913280 |
| test_1M | M2_ElasticNet | 0.246009 | 0.197204 | 0.215195 | 0.504913 | 0.499739 | 0.050655 |
| test_60M | M2_ElasticNet | 1.522673 | 1.510026 | -47.395164 | 0.454407 | 0.451252 | 1.510026 |
| test_1B | M2_ElasticNet | 2.953377 | 2.952250 | -3232.176900 | 0.609738 | 0.596703 | 2.952250 |

L_trim 的同样指标完整保存在 model_validation.csv。图展示固定1M模型的原预测，不以测试真值调整均值、截距、尺度或重新训练。外部检验的baseline相关系数因恒定预测而未定义，留空而非置零；嵌套OOF的M0跨折采用不同训练均值，因此聚合OOF可能有非零相关，不代表配比解释力。

### 训练CV选定模型

| dataset | model | n | RMSE | MAE | R2 | Pearson | Spearman | prediction_bias_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test_1M | M1_Ridge | 256.000000 | 0.242175 | 0.194783 | 0.239465 | 0.526805 | 0.501667 | 0.052368 |
| test_60M | M1_Ridge | 256.000000 | 1.524597 | 1.511739 | -47.517569 | 0.492346 | 0.476981 | 1.511739 |
| test_1B | M1_Ridge | 64.000000 | 2.906459 | 2.904826 | -3130.265543 | 0.684098 | 0.639973 | 2.904826 |

绝对误差/R²与配方优劣排序相关须分开判断。若跨尺度偏差大而排序较好，仅能称排序迁移有效，不能声称1M模型无条件解释60M/1B绝对Loss。额外去均值RMSE只是描述性分解，不是可部署校准或另一组实际验证结果。

## 实际跨尺度配方排序

| response | dataset_1 | dataset_2 | shared_recipe_n | actual_Pearson | actual_Spearman | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| L_mean | test_1M | test_60M | 256.000000 | 0.967096 | 0.965798 | direct observed ordering on matched recipes |
| L_mean | test_1M | test_1B | 0.000000 | 未定义 | 未定义 | no shared recipes; direct observed rank stability undefined |
| L_mean | test_60M | test_1B | 0.000000 | 未定义 | 未定义 | no shared recipes; direct observed rank stability undefined |
| L_trim | test_1M | test_60M | 256.000000 | 0.984106 | 0.982340 | direct observed ordering on matched recipes |
| L_trim | test_1M | test_1B | 0.000000 | 未定义 | 未定义 | no shared recipes; direct observed rank stability undefined |
| L_trim | test_60M | test_1B | 0.000000 | 未定义 | 未定义 | no shared recipes; direct observed rank stability undefined |

共同配方以闭合向量匹配；同名index不一定为同配方。没有相同配方时，直接实际排序一致性未定义，不虚构相关值。1B上的预测/真值Spearman仅是固定1M模型的间接迁移证据。

## 范围与局限

本轮捕获收敛/运行警告数 0，逐条见 convergence_warnings.csv；所选ElasticNet达到迭代上限则停止，不隐瞒失败。原件及先前锁定产物/代码 2239 个文件前后 SHA-256 保持一致。独立验证见 mixture_verification.csv。

对稀疏配方的log-ratio零替换会影响效应，未声称唯一零处理结论。训练配方数有限，二阶项更易过拟合；ILR基与L1惩罚的选择影响稀疏解释。后续仍需按用户下一阶段做原领域扰动、互补趋势、Q映射及 estimated 外推讨论，当前没有提前生成这些结论。

10B/70B是estimated材料，本轮只读结构和Loss描述，未用于CV、拟合、选择或预测。模型规模方程尚未建立。Q/K保持锁定，未进入本轮任何解释变量。

## 当前停止点

可复用 ILR_basis.csv、composition_transform_A4.json、四个候选的final_model_metadata.json/joblib、嵌套OOF与测试预测CSV。等待人工确认阶段1后再进入 Q_p 与外推分析。
