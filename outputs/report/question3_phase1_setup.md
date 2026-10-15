# 问题三 PHASE 1 算力约束优化输入标定

本阶段完成官方公式核验、实际C7诊断、成本与支持域整理、数学形式及未来转移判据。未拟合任何模型，未运行资源优化或结构转移检测。问题一/二冻结文件完整性与独立算术核验见本阶段验证表。

## 1 官方三类成本及预算

C_train=6ND；CQ=D[g(Q)-g(Q0)]+；C_attn=etaNDL_ctx。eta=0.0002，来源官方问题三原始OMML；训练/质量/注意力使用同一D。预算建议[1e+19, 1e+22, 1e+24] FLOPs，非强制唯一取值。附加41点研究对数网格从最低/10到最高×10，仅列场景。

## 2 C7上下文及注意力临界值

| min | P01 | P05 | P10 | P25 | median | P75 | P90 | P95 | P99 | max | rows | valid_count | invalid_count | unique_value_count | mean | std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 2048 | 2048 | 2048 | 2048 | 4096 | 32768 | 32768 | 32768 | 87818.24 | 131072 | 45 | 45 | 0 | 5 | 13744.35556 | 21934.65376 |

| L_ctx | support_count | support_fraction | scenario_type | evidence_source | attention_to_training_ratio |
| --- | --- | --- | --- | --- | --- |
| 2048 | 18 | 0.4 | short | real_attachments/C_efficiency_evolution/model_architecture_metadata.csv | 0.06826666667 |
| 4096 | 8 | 0.1777777778 | medium_median | real_attachments/C_efficiency_evolution/model_architecture_metadata.csv | 0.1365333333 |
| 8192 | 7 | 0.1555555556 | observed_frequent | real_attachments/C_efficiency_evolution/model_architecture_metadata.csv | 0.2730666667 |
| 32768 | 11 | 0.2444444444 | observed_frequent | real_attachments/C_efficiency_evolution/model_architecture_metadata.csv | 1.092266667 |
| 131072 | 1 | 0.02222222222 | long | real_attachments/C_efficiency_evolution/model_architecture_metadata.csv | 4.369066667 |

C7实际45行7列，完整字段与缺失/重复见C7_diagnosis.md；没有独立N或明确tokens字段，training_data_TB未换算为D。eta使用题给值，无另估区间；L_crit=6/eta=30000，在观测范围内但非观测离散值；超临界12/45。上下文取五个全部实际窗口作为外生场景。架构最大窗口、训练真实序列与自由N/D组合存在差异；题给成本为简化代理。

## 3 冻结目标与证据等级

L_gen=E+A N_B^(-alpha)+B D_B^(-beta)+C_Q(N_B/1B)^(-u_Q)(-lnQ)+lambda_p(N_B)[M(p)-M_ref]。这里N_B/1B表示相对N0_B=1的数值比例，不能再把以B计的N乘10^9代入目标系数。

Class A经典骨架、Class B局部质量、Class C冻结ILR+一阶ElasticNet配方模型、Class D大尺度配方转移。所有完整参数直接读取最终JSON，见frozen_objective_summary.csv。没有联合外部N-D-Q-p验证；B6是局部半合成质量校准，B7差分扩展失败，不改冻结参数也不夸大证据。

## 4 基线质量

| candidate | Q0 | definition | mapping_needed | recommended |
| --- | --- | --- | --- | --- |
| Q0_A | 0.721575982 | A1全部样本Q的双侧各10%截尾均值 | False | False |
| Q0_B | 0.7280227165 | 冻结p_ref点乘冻结17→7代理映射Q_domain | True | True |
| Q0_C | 0.7475281056 | 七领域Q_domain中位数（等域敏感性） | False | False |
| Q0_B_alternative | 0.6925361864 | 冻结替代映射下p_ref加权质量 | True | False |
| Q0_B_direct_only | 0.7118360081 | 仅直接/近直接映射，按覆盖质量质量归一化 | True | False |

建议Q0_B=0.72802271649780836以保持与p_ref一致，但仅在冻结代理映射假设下。依赖代理映射的配比权重为48.424184%，替代映射Q0=0.69253618640575432；未识别域包络[0.6512576874986107, 0.7376318233356024]不是置信区间。七质量域聚合权重见baseline_quality_domain_weights.csv；c4无直接对应的17域配比，当前主映射聚合权重为0，不补造配比。主/替代映射的结构敏感性不可被七域bootstrap CI替代；不继承Q0=.8，不重算样本Q。

## 5 三种官方质量成本与边际成本

| family | g_formula | gamma | lambda_Q | marginal_formula | endpoint_finite | shape |
| --- | --- | --- | --- | --- | --- | --- |
| exponential | γ exp(λ_Q Q) | 10000000 | 6 | D × γ λ_Q exp(λ_Q Q) | True | 凸 |
| power | γ Q^λ_Q | 5000000000 | 4 | D × γ λ_Q Q^(λ_Q-1) | True | 凸 |
| logarithmic | γ ln(1+λ_Q Q) | 2000000000 | 10 | D × γ λ_Q/(1+λ_Q Q) | True | 凹，边际成本递减 |

统一乘实际D与正部差分，Q=Q0增量0，Q>Q0边际正，Q=1均有限。对数型边际递减，不能改成发散模型。三种全部保留待后续比较，未提前择优。

## 6 N/D/Q支持域

| scheme | variable | lower | upper | unit | extrapolation |
| --- | --- | --- | --- | --- | --- |
| main | N | 0.070542 | 11.965825 | billion parameters | False |
| main | D | 10 | 299.893 | billion tokens | False |
| classical_only_reference | N | 0.070542 | 11.965825 | billion parameters | False |
| classical_only_reference | D | 0.134 | 299.893 | billion tokens | False |
| extrapolation | N | 0.070542 | 10000 | billion parameters | True |
| extrapolation | D | 10 | 36000 | billion tokens | True |
| main | Q | 0.7280227165 | 1 | dimensionless | True |
| extrapolation | Q | 0.7280227165 | 1 | dimensionless | True |

主域不是未经论证直接指定B1：比较B1–B5范围与已冻结验证局限后，经典骨架采用B1高可信包络，再与B6质量数值支持求交集，因此D下界10B而非0.134B。B3同源插值不能扩证据，B4/B5跨评测Loss口径未确认、仅列扩展覆盖，不扩主域。该连续矩形仍是插值建模选择，真实B1只含8个固定N轨迹。质量模块没有直接实验支持，表中“非外推”仅指N/D各模块数值范围。外推域将N上界扩至10000B、D上界36000B，保留主下界；B9 D=0是未知而不是有效可行下界。总/激活参数、稠密/稀疏等尚未统一，不能把B10估算Loss当验证。

Q∈[Q0,1]；A1样本质量范围[0.30649962546558407, 0.9511217995313428]，七领域质量范围[0.5867366961905537, 0.7651065297732803]；B6数值Q范围[0.1,1.0]。质量范围覆盖不等于直接验证；Q超过A1样本最大值或领域最大值分别标记，Q≈1不能宣称测得验证。

## 7 p可行集

P0固定p_ref；P1读取真实训练/检验配方已冻结闭合零替换模型输入，1088条来源记录对应832个候选，完整membership保存。不新建邻域，不做17维联立寻优；P1有限候选外层选择有据可查，但模型输入的微小零替换质量不是原始训练比例。P2理论simplex只作敏感性，极端角点不能作建议。固定Q0与可独立改善Q的假设对P1也需保留。N>1B使用Class D，不因N在B1范围而升级配比证据。

## 8 单位

C_train=6N_BD_B10^18；C_attn=etaN_BD_BL_ctx10^18；CQ=D_B10^9Δg，均为FLOPs。gamma按官方总成本维度推定FLOPs/token，lambda_Q无量纲；没有货币系数或额外M。12个独立Decimal手工算式与程序核验通过，见unit_audit.md及unit_numeric_checks.csv。

## 9 数学形式与预算绑定

P0 min_{N_B,D_B,Q}L_gen(p_ref)，受三个成本之和≤预算及上述box约束；P1逐配方同样求解再外层选最优。约束保持不等式。P0各资源边际Loss都负，内域预算绑定或全部资源到上界；P1的N导数可能因分段lambda与配方项改变符号，须左右检查，D/Q仍严格递减。边界饱和可出现预算松弛。尤其低预算配长上下文时主域可能不可行：预算可行性只按主域成本解析下界筛查，未求Loss最优。详见budget_feasibility_screen.csv。

## 10 结构性转移定义

采用活跃集、以预算为分母的三项成本份额、最优N/D对预算log弹性与Q对ln预算导数。候选变化须后续局部加密、左右导数、数值与情景稳定性验证，不在本阶段识别断点。同一上下文下C_attn/C_train=etaL_ctx/6恒定，不能制造由预算驱动的注意力/训练比例交叉。

## 11 真实输入、解析推导与假设

真实输入：官方Word/PDF、C7、问题一冻结质量/配方结果、B1以及B9公开元数据。B6局部校准为半合成，B10及10B/70B配方转移为估算；ILR配方函数是冻结回归模型。推导：临界窗口、边际质量成本、FLOPs换算、预算最低可行成本及单调性。假设/敏感性：17→7代理映射、Q0选取、矩形支持域及其B6交集、跨架构上下文自由组合、p/Q独立改变、零替换配方空间、41点预算网格与未来识别容差。旧优化稿未作为数据输入。

## 12 进入PHASE 2的条件

已具备按已审计公式进行条件化数值求解的输入与数学定义，等待人工确认Q0代理映射及主/外推域。无公式/单位缺失；条件化模型不等于完整联合真实验证。后续不得强行求不可行预算场景，不得把估算大模型配比反转或Q=1端点当直接证据。当前停止于PHASE 1，未求低中高最优解、未检测结构转移、未开始问题四。

## 图表

- [C7实测上下文经验分布](../../outputs/figures/question3_phase1/01_C7_context_distribution.png)
- [C7实际离散窗口频数](../../outputs/figures/question3_phase1/02_C7_context_frequency.png)
- [Official incremental quality costs，不是最优结果](../../outputs/figures/question3_phase1/03_quality_cost_functions.png)
- [Official marginal quality costs，不是最优结果](../../outputs/figures/question3_phase1/04_marginal_quality_costs.png)
- [官方三档与研究预算网格](../../outputs/figures/question3_phase1/05_budget_scenarios.png)
- [数据范围对照；B4/B5可比性有限，B9非Loss验证](../../outputs/figures/question3_phase1/06_N_D_evidence_support.png)
