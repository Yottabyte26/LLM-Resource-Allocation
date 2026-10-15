# 问题二 PHASE 3：领域配比的尺度调制与模块化广义标度律

本阶段完成冻结模块的组合和逐尺度配比传递校准。最终候选使用分尺度经验系数及 ln N 上的分段线性插值。不冻结新的全局 N-D-Q 函数，不重新训练配比模型，不进行联合最小二乘，不使用 B9/B10。

## 1. 冻结输入与证据等级

第一问主模型为 L_mean 的 ILR + 一阶 ElasticNet。M(p) 是其 1M 尺度预测 Loss。只解析此前输出，未解析 A4–A15 原始附件。现成逐配方预测与冻结模型参数、保存 ILR 坐标逐行核对。L_trim、Ridge、增强 Qp/Kp 模型均未替代主模型。

| module | parameters | evidence | support |
| --- | --- | --- | --- |
| L0 | ["E", "A", "B", "alpha", "beta"] | Class A | B1 observations; frozen B3/B4/B5 validations with original comparability limits |
| Delta_Q | ["C_Q", "u_Q"] | Class B | B6 semisynthetic local calibration; B7 difference validation failed |
| M(p) | ["frozen ILR first-order ElasticNet"] | Class C | Question 1 training and true mixture checks at 1M/60M/1B |
| lambda_p large N | ["10B knot", "70B knot", "segments above 1B"] | Class D | Estimated recipe tables; no observed large-scale reversal |

Class A 是骨架参数的直接数据支持，不表示 B4/B5 已统一评测口径。Class B 是 B6 半合成局部证据。Class C 是问题一真实配比实验。Class D 是 10B/70B 估算表的条件性证据。

## 2. 逐尺度中心化与传递系数

$m_{si}=M(p_{si})-\overline{M}_s$, $y_{si}=L_{si}-\overline{L}_s$。

$\hat\lambda_s=\sum_i m_{si}y_{si}/\sum_i m_{si}^2$，中心化坐标不设截距。

standard error 使用原坐标含截距 OLS 等价的 n−2 残差自由度，因为两侧均值从数据估计。OLS CI 为 t 区间，CI_low/CI_high 为主要报告的配方 Bootstrap 百分位区间。R² 是校准后的组内拟合 R²，不是未经目标尺度校准的外部预测性能。Pearson/Spearman 比较原始 M(p) 与 Loss，保留估算表负相关。

| scale | evidence | n | lambda_s | standard_error | OLS_CI_low | OLS_CI_high | CI_low | CI_high | R2 | Pearson | Spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1M | observed | 256 | 0.93905515 | 0.095147756 | 0.75167615 | 1.1264341 | 0.74859391 | 1.1411218 | 0.27718906 | 0.52648748 | 0.5006137 |
| 60M | observed | 256 | 0.68974808 | 0.076864284 | 0.53837559 | 0.84112057 | 0.53894611 | 0.85879678 | 0.24071491 | 0.49062705 | 0.47415813 |
| 1B | observed | 64 | 0.29032089 | 0.039626012 | 0.21110962 | 0.36953216 | 0.18945514 | 0.40489143 | 0.46402918 | 0.68119687 | 0.64217033 |
| 10B | estimated | 63 | -0.3110438 | 0.064501825 | -0.44002317 | -0.18206443 | -0.44607347 | -0.19417264 | 0.27599942 | -0.52535647 | -0.53225806 |
| 70B | estimated | 63 | -0.3727259 | 0.066864838 | -0.50643041 | -0.23902139 | -0.51420778 | -0.25107689 | 0.33748241 | -0.58093236 | -0.58069316 |

真实三尺度的系数同为正：True（三个 Bootstrap 区间均不跨 0）。估算两尺度同为负：True（两个区间均不跨 0）。

定量结果支持真实尺度内配比方向一致，并在估算表中出现反向。真实实验尚未观测到符号翻转。**配比反向及临界规模结论依赖估算表。**

## 3. 配方 Bootstrap

固定种子 2026，每尺度 2000 次按配方 index 有放回抽样。每次重新计算样本内两个均值，再估计无截距传递系数。1M/60M 的完整 256 个配方相同，10B/70B 的完整 63 个配方相同，分别共享抽样索引。1B 的 64 个配方独立，三类配方集合互不重叠。不能仅按同名 index 关联不同配方。因此 1B 与其他组的配方覆盖不同，传递系数变化不能完全归因于 N；评测口径、配方覆盖和估算生成方式也可能影响系数。此 Bootstrap 条件于冻结 M(p)，不包含第一问模型再训练不确定性和估算表生成误差。

| scale | bootstrap_mean | bootstrap_median | bootstrap_std | CI_low | CI_high | positive_fraction | negative_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1M | 0.93906377 | 0.93621917 | 0.099335518 | 0.74859391 | 1.1411218 | 1 | 0 |
| 60M | 0.69016342 | 0.68684103 | 0.080467254 | 0.53894611 | 0.85879678 | 1 | 0 |
| 1B | 0.28925879 | 0.28772297 | 0.056296029 | 0.18945514 | 0.40489143 | 1 | 0 |
| 10B | -0.3164511 | -0.31593555 | 0.06360967 | -0.44607347 | -0.19417264 | 0 | 1 |
| 70B | -0.37864239 | -0.37770761 | 0.066749203 | -0.51420778 | -0.25107689 | 0 | 1 |

## 4. P1/P2 的真实数据拟合与估算敏感性

P1：$\lambda(N)=a+b\ln(N/N_{ref})$，$N_{ref}=1$B。

P2：$\lambda(N)=a\tanh[b(\ln N_c-\ln N)]$，$a,b>0$。

N 使用十亿参数为单位，所以 P2 的对数为 ln(N/B)。各尺度等权拟合，避免小标准误的尺度自动主导。Fit A 仅三项真实检验，Fit B 加两项估算表。P2 全拟合 16 初值，Bootstrap 每次 3 初值，正参数有界拟合。AICc 仅在 n>k+1 时定义。三点三参数的 P2 Fit A 是饱和拟合，残差自由度为 0；留出一尺度后只有两点，禁止解释成可识别的三参数模型。

| fit | model | parameters | RMSE | LOSO_RMSE | AICc | residual_df | jacobian_condition | saturated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_real_only | P1 | {"a": 0.3431424950283471, "b": -0.0915215959374177} | 0.063375873 | 0.2453079 | n.a. | 1 | 6.7426556 | False |
| A_real_only | P2 | {"a": 1.0201940506195122, "b": 0.18811099500138348, "log_Nc_B": 1.5557427917860869} | 0 | n.a. | n.a. | 0 | 17.716123 | True |
| B_with_estimated | P1 | {"a": 0.16516617088168273, "b": -0.1291837514438167} | 0.13246767 | 0.24855928 | -10.214166 | 3 | 4.0329847 | False |
| B_with_estimated | P2 | {"a": 1.03735559751279, "b": 0.17867431745081885, "log_Nc_B": 1.3984878738546271} | 0.085111817 | 0.17814108 | 5.3621061 | 2 | 5.4545585 | False |

Fit B 中 P2 的拟合 RMSE 和留尺度 RMSE 均低于 P1；P1 参数更少且 AICc 更低，适合作为简约趋势对照。数值性能和简约准则未一致选择同一形式，P2 的较小误差也没有足够真实尺度验证支撑。因此不把任一光滑函数冻结为全局配比尺度律，保留分尺度经验系数。

## 5. 临界规模的条件性与辨识限制

对 P1 求 ln(Nc/B)=−a/b，P2 使用拟合 ln(Nc/B)。每次使用上述配方 Bootstrap 系数重新拟合曲线。表中 valid_fraction 报告有限根比例，boundary_fraction 报告触边比例。conditional_interval_stable 要求有限根比例≥95%、触边比例≤5%，且 95% CI 完整位于数据中相反符号节点的括区间内。Fit A 没有符号括区间，不能判为稳定临界规模。这个条件性标记不表示真实实验已确认物理临界规模。负向趋势与反向传递不是因果机制证明。

| fit | model | valid_fraction | boundary_fraction | Nc_B_median | Nc_B_CI_low | Nc_B_CI_high | conditional_interval_stable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_real_only | P1 | 1 | 0 | n.a. | n.a. | n.a. | False |
| A_real_only | P2 | 1 | 0.004 | n.a. | n.a. | n.a. | False |
| B_with_estimated | P1 | 1 | 0 | 3.4637202 | 1.8468936 | 6.8928204 | True |
| B_with_estimated | P2 | 1 | 0 | 3.9494731 | 2.4694586 | 6.8461621 | True |
| B_with_estimated | empirical_log_linear | 1 | 0 | 2.9951181 | 2.2355277 | 4.2280502 | True |

**不报告已被真实实验稳定识别的具体临界规模。**真实尺度均为正，没有过零括区间；Fit A 的 P1/P2 外推根还存在结构依赖。Fit B 和经验插值的过零位置仅作为依赖估算表的条件性诊断，不能用于宣称普适临界规模。

## 6. 最终配比项：经验系数及分段形式

| N_params_B | lambda_s | CI_low | CI_high | evidence |
| --- | --- | --- | --- | --- |
| 0.001 | 0.93905515 | 0.74859391 | 1.1411218 | observed |
| 0.06 | 0.68974808 | 0.53894611 | 0.85879678 | observed |
| 1 | 0.29032089 | 0.18945514 | 0.40489143 | observed |
| 10 | -0.3110438 | -0.44607347 | -0.19417264 | estimated |
| 70 | -0.3727259 | -0.51420778 | -0.25107689 | estimated |

令 $x=\ln(N/B)$、$x_s=\ln(N_s/B)$。在 $N_s\le N\le N_{s+1}$ 内：

$\lambda_p(N)=\lambda_s+(\lambda_{s+1}-\lambda_s)(x-x_s)/(x_{s+1}-x_s)$。

只在节点范围 0.001B–70B 内插值，接口拒绝范围外外推。0.001B–1B 的节点来自真实检验，超过 1B 的插值依赖 Class D 估算证据。在 ln N 而非原 N 上线性插值，保持与尺度趋势图一致。节点处可连续，但导数一般有左右两值。

参考配方直接读取冻结的 A4_ILR_center，M(p_ref)=5.3446567761711776。中心是在冻结零替换后的 ILR 空间取均值并反变换，17 维配方完整存于 reference_recipe.json。

$\Delta_p(N,p)=\lambda_p(N)[M(p)-M(p_{ref})]$，所以参考配方修正严格为 0。

逐尺度中心化只校准斜率，不改变经典绝对基线。参考配方锚定不会证明各源的绝对 Loss 基线已对齐。

## 7. 模块化广义标度律候选

$L_{gen}=E+AN^{-\alpha}+BD^{-\beta}+C_Q(N/N_0)^{-u_Q}(-\ln Q)+\lambda_p(N)[M(p)-M(p_{ref})]$。

N、D 以 B 为单位，N0=1B。完整冻结参数：{"E": 1.6897975629216038, "A": 0.35398032066343604, "B": 1.2403055835502586, "alpha": 0.3399765818976516, "beta": 0.27987812854408783}。

C_Q=0.17064090046126643，u_Q=0.16190329425272901，直接读取 PHASE 2C S3，未重新拟合。原 v≈0 且触边，按人工确认删除 D^v 项。仅称“B6 校准的局部质量修正项”。

$Q=1$ 且 $p=p_{ref}$ 时，$L_{gen}=L_0$。anchor_identity_checks.csv 是公式恒等核验，不是联合观测测试集。

质量与配比相加采用模块独立性假设，尚未验证 Q×p、N×D×p 的联合交互。配比表没有共同 N/D/Q/p/L 联合观测。模型是在各自证据源独立标定后组合的候选，不提供统一外部预测精度。

## 8. 边际效应公式与 ILR 有限差分

记 $G(p)=M(p)-M(p_{ref})$，$\Delta_Q=C_Q(N/N_0)^{-u_Q}(-\ln Q)$。

$\partial_N L=-\alpha A N^{-\alpha-1}-(u_Q/N)\Delta_Q+\lambda_p\prime(N)G(p)$。

$\partial_D L=-\beta B D^{-\beta-1}$。

$\partial_Q L=-C_Q(N/N_0)^{-u_Q}/Q$。

分段内部：$\lambda_p\prime(N)=(\lambda_{s+1}-\lambda_s)/[N\ln(N_{s+1}/N_s)]$。

P1 对照的导数为 $b/N$；P2 对照为 $-ab[1-\tanh^2(b(\ln N_c-\ln N))]/N$。

令 $t=ILR(p)$，$p_{\pm}=ILR^{-1}(t\pm h e_k)$。

$[\Delta_p(N,p_+)-\Delta_p(N,p_-)]/(2h)=\lambda_p(N)[M(p_+)-M(p_-)]/(2h)$。

冻结一阶模型下，若 $g_k=s_y\beta_k/s_{X,k}$，上述有限差分等于 $\lambda_p(N)g_k$。

此关系针对严格正配方的 ILR 扰动。原始零配比先按冻结约定替换，不把对替换组成的有限差分解释为原始结构零处的导数。本轮仅推导公式及实现核验，不计算弹性、Q 提升 0.1 的参数等价。

## 9. 分模块验证及局限性

| module | evidence | dataset | stage | target | n | RMSE | R2 | Pearson | Spearman | limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0 | Class A | n.a. | n.a. | n.a. | 147 | 0.00022023037 | 0.99999923 | 0.99999967 | 1 | Frozen B1 leave-trajectory-out validation; no refitting in this stage |
| L0 | Class A | n.a. | n.a. | n.a. | 147 | 0.00018181804 | 0.99999947 | 0.99999976 | 1 | Frozen B1 leave-trajectory-out validation; no refitting in this stage |
| L0 | Class A | n.a. | n.a. | n.a. | 147 | 0.00013511027 | 0.99999971 | 0.99999985 | 1 | Frozen B1 leave-trajectory-out validation; no refitting in this stage |
| L0 | Class A | n.a. | n.a. | n.a. | 147 | 0.00015459771 | 0.99999962 | 0.99999984 | 1 | Frozen B1 leave-trajectory-out validation; no refitting in this stage |
| L0 | Class A | n.a. | n.a. | n.a. | 147 | 0.00013191675 | 0.99999972 | 0.99999988 | 1 | Frozen B1 leave-trajectory-out validation; no refitting in this stage |
| L0 | Class A | n.a. | n.a. | n.a. | 147 | 0.00010197532 | 0.99999983 | 0.99999992 | 1 | Frozen B1 leave-trajectory-out validation; no refitting in this stage |
| L0 | Class A | n.a. | n.a. | n.a. | 147 | 0.00011441463 | 0.99999979 | 0.9999999 | 1 | Frozen B1 leave-trajectory-out validation; no refitting in this stage |
| L0 | Class A | n.a. | n.a. | n.a. | 147 | 0.0001350424 | 0.99999971 | 0.99999986 | 1 | Frozen B1 leave-trajectory-out validation; no refitting in this stage |
| L0 | Class A | B3 | n.a. | n.a. | 4000 | 0.0038212821 | 0.99995774 | 0.99998994 | 0.99998979 | B3 interpolation consistency; B4/B5 evaluation scopes not absolutely comparable |
| L0 | Class A | B4 | n.a. | n.a. | 57 | 0.29266668 | 0.60450489 | 0.97507581 | 0.98298165 | B3 interpolation consistency; B4/B5 evaluation scopes not absolutely comparable |
| L0 | Class A | B5 | n.a. | n.a. | 44 | 0.19759507 | 0.73055911 | 0.90762639 | 0.95880757 | B3 interpolation consistency; B4/B5 evaluation scopes not absolutely comparable |
| Delta_Q | Class B | n.a. | B6_5fold_GroupCV_OOF | absolute_Loss | 360 | 0.069920473 | 0.95893904 | 0.98083336 | 0.98040256 | B6 calibrated local correction only; frozen S3 v near zero is removed without refitting |
| Delta_Q | Class B | n.a. | B6_5fold_GroupCV_OOF | within_ND_DeltaL | 315 | 0.087072519 | 0.59491474 | 0.81034668 | 0.8372232 | B6 calibrated local correction only; frozen S3 v near zero is removed without refitting |
| Delta_Q | Class B | n.a. | B7_fixed_B6_parameters | absolute_Loss | 90 | 0.071513543 | 0.95060039 | 0.99186771 | 0.99129519 | B7 nonoverlap difference R2 fails; not a global quality law |
| Delta_Q | Class B | n.a. | B7_fixed_B6_parameters | within_ND_DeltaL | 90 | 0.083817946 | -0.040868046 | 0.60739764 | 0.59573533 | B7 nonoverlap difference R2 fails; not a global quality law |
| M(p) | Class C | test_1M | n.a. | n.a. | 256 | 0.24206725 | 0.24014133 | 0.52648748 | 0.5006137 | Question 1 frozen absolute predictions; centered lambda here is fitted calibration, not independent validation |
| M(p) | Class C | test_60M | n.a. | n.a. | 256 | 1.5247194 | -47.52535 | 0.49062705 | 0.47415813 | Question 1 frozen absolute predictions; centered lambda here is fitted calibration, not independent validation |
| M(p) | Class C | test_1B | n.a. | n.a. | 64 | 2.9148113 | -3148.2884 | 0.68119687 | 0.64217033 | Question 1 frozen absolute predictions; centered lambda here is fitted calibration, not independent validation |
| M(p) | Class D | estimated_10B | n.a. | n.a. | 63 | 3.5555545 | -1367.4667 | -0.52535647 | -0.53225806 | 10B/70B estimated training subsets, not independent observed experiments |
| M(p) | Class D | estimated_70B | n.a. | n.a. | 63 | 3.9383733 | -1428.7504 | -0.58093236 | -0.58069316 | 10B/70B estimated training subsets, not independent observed experiments |

L0：B1 的验证来自此前留轨迹 CV，B3 是插值轨迹一致性，B4/B5 保留不同评测口径限制。Delta_Q：B6 质量修正有效，但 B7 差分 R² 未通过；此失败在组合模型中仍然保留。Delta_p：问题一原预测检验与本轮中心化斜率校准分别报告，后者不能声称独立外部验证。10B/70B：估算且为训练配方子集，不是大模型真实联合实验。

**广义模型采用模块化证据融合，无法进行单一完整联合外部验证。**未制造联合测试集，未重拟合经典或质量参数。此前 B8 机制不兼容结论保持冻结，本轮未用 B8 校准配比项。

## 10. 输出与停止点

主要输出：mixture_transfer_by_scale.csv、mixture_transfer_bootstrap.csv、lambda_scale_model_comparison.csv、critical_scale_diagnosis.csv、lambda_empirical_knots.csv、generalized_scaling_candidate.json、component_validation.csv。复现：python -B scripts/15_calibrate_mixture_scaling.py；核验：python -B scripts/15_verify_mixture_scaling.py。

可以提交本阶段候选进行人工确认。尚未授权进入下一阶段。对于超过 70B 的 B9/B10 目标，本轮分段接口没有外推策略，下一阶段须先明确外推规则和 Class D 情景身份。不把本阶段称为完整联合模型验证通过。已停止，不执行弹性、参数等价或 B9/B10 外推。

## 11. 实际解析文件

| path | bytes | sha256 | source_role |
| --- | --- | --- | --- |
| outputs/mixture/test_predictions.csv | 305565 | d742bb1ccb37bf5226717ea356a12acad5ebdb95a848e9b5657526c11a9c00d7 | frozen_output |
| outputs/mixture/estimated_scale_predictions.csv | 28950 | 824f464ea57af05c4900d488eddcc059107ccd901b99efac11a939f6719eabe2 | frozen_output |
| outputs/mixture/stage2/model_comparison_test_predictions.csv | 135231 | bfee95ee5bce2e1536582c249518677ad86b73b2dc95418de712906666a659ea | frozen_output |
| outputs/mixture/final_model_metadata.json | 206875 | 34eafee82f719df42b63c0d406e33efd41ba900e721f2261249e830c1e142009 | frozen_output |
| outputs/mixture/stage2/final_model_policy.json | 514 | ed8903742fc224ad249954d35bc4043f291e5f88c8cd8e7992a2fe9d51328f62 | frozen_output |
| outputs/mixture/stage2/representative_baselines.csv | 1774 | 6e984421dcb277f98f4860a38368df7aa6b631ac420a7cfd5b9403ab84091e4a | frozen_output |
| outputs/mixture/train_1M_closed_replaced_composition.csv | 182779 | cbaa83a14c5d08205c43fc52c300d5b5f076898a1a63c0db875818a944b21818 | frozen_output |
| outputs/mixture/train_1M_ILR_features.csv | 159758 | 1f9e1406521aab43aff50049eac587bed65c1fd79621f34d4bdc014e956558d9 | frozen_output |
| outputs/scaling/phase1/classical_scaling_parameters.csv | 585 | 85f9a0f461895bb37e419c2a1d7a4dabbca7d99d7635164a9fe4ed0edfdb2e6f | frozen_output |
| outputs/scaling/phase2C/model_parameters.csv | 1079 | 9d0bbb7d57891a8442016a949c8f63947ab223b9206b2ab917ac109b202a95ca | frozen_output |
| configs/scaling_mixture_phase3.yaml | 522 | 825a52d1451a2b37704bfb0011a40777829d090d8843f5524df9fa0575829634 | configuration |
| outputs/mixture/test_1M_closed_replaced_composition.csv | 91621 | 530a0c6437b051a421e281ad6ba2a19f9a12fb8dc742a5ccac311fa18a1eea6d | frozen_output |
| outputs/mixture/test_1M_ILR_features.csv | 79639 | 1676f2385f113db2539885d6f487fb7909e2740f599c7115a4745ce27fac478b | frozen_output |
| outputs/mixture/test_60M_closed_replaced_composition.csv | 91621 | 530a0c6437b051a421e281ad6ba2a19f9a12fb8dc742a5ccac311fa18a1eea6d | frozen_output |
| outputs/mixture/test_60M_ILR_features.csv | 79639 | 1676f2385f113db2539885d6f487fb7909e2740f599c7115a4745ce27fac478b | frozen_output |
| outputs/mixture/test_1B_closed_replaced_composition.csv | 23029 | 4b858e23aa70419d587562c70bc9b29ee39fd694687e1dca9f2a470c8a04e79e | frozen_output |
| outputs/mixture/test_1B_ILR_features.csv | 19951 | 1ff2cea6019e652e758bc0936dd578b526de8bd11d27e14b72fa9b4c0814d49e | frozen_output |
| outputs/mixture/estimated_scale_consistency.csv | 1679 | c8cc56dfa6a22aeaad5a3c4914277b898856dd0b6e0e544d7aa3d080495ac92b | frozen_output |
| outputs/mixture/model_validation.csv | 5229 | 00524fd9fa48e202fb689ab7ef6e58c757a845387fe5ccce50d5c8fa2a4745d6 | frozen_output |
| outputs/scaling/phase1/B1_group_cv.csv | 7067 | 86cf68fd218e4879109c2a69f656c5cbe3e1248a501b061ce083c8d69fb95384 | frozen_output |
| outputs/scaling/phase1/B3_validation.csv | 2720 | 8953908d354af99a19a5a64503e11845edfa6ff69a1b7300d58c74db5b729d7b | frozen_output |
| outputs/scaling/phase1/B4_cross_family_validation.csv | 2845 | 7d48cb852e9c0fde504730607cfef9a45b00193968cfbdcfae45b272978f1434 | frozen_output |
| outputs/scaling/phase1/B5_literature_validation.csv | 6553 | a3a145d2a3d694b96346c5b68b184e83dafdbf497db730c1dc5657929bc46b97 | frozen_output |
| outputs/scaling/phase2C/B6_group_cv.csv | 1993 | 5d8888da6048eca82004ce16e89e65c30f757efe6ee26d120837a7f226c8cf85 | frozen_output |
| outputs/scaling/phase2C/B7_nonoverlap_validation.csv | 2100 | 477406142115fbe5f760c23f01cca8ecf6b2f2697194f4fdd0df43817d547193 | frozen_output |
