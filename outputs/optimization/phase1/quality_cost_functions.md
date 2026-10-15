# 官方质量成本函数

统一 CQ(D,Q;Q0,θ)=D[gθ(Q)-gθ(Q0)]+，D为实际token数，θ=(gamma,lambda_Q)。推荐Q0=0.72802271649780836；三个官方参数全部从Word OMML读取。

| family | g_formula | gamma | lambda_Q | gamma_unit | marginal_formula | cost_at_Q0_per_token | cost_at_Q1_per_token | endpoint_finite | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exponential | γ exp(λ_Q Q) | 10000000 | 6 | FLOPs/token | D × γ λ_Q exp(λ_Q Q) | 0 | 3245323383 | True | 凸 |
| power | γ Q^λ_Q | 5000000000 | 4 | FLOPs/token | D × γ λ_Q Q^(λ_Q-1) | 0 | 3595409497 | True | 凸 |
| logarithmic | γ ln(1+λ_Q Q) | 2000000000 | 10 | FLOPs/token | D × γ λ_Q/(1+λ_Q Q) | 0 | 568049738.8 | True | 凹，边际成本递减 |

在Q>Q0时 dCQ/dQ=D g′(Q)>0；Q<Q0时为零；Q=Q0处有左导数0与右导数Dg′(Q0)，主可行域采用右导数。对D的导数为[g(Q)-g(Q0)]+，非负；三类均显式乘同一训练D。Q=Q0增量为零；Q→1时三类均有限，不排除Q=1。对数渐进型是ln(1+10Q)，不是-ln(1-Q)，也不是越趋完美越发散的模型。

指数/幂的质量边际成本递增，对数型边际成本递减，因此后续可能产生不同活跃集，不能先套同一凸性或内点结论。本阶段三类均保留，未依据旧最优结果选主成本。曲线以CQ/D和(dCQ/dQ)/D显示，避免主观选D。没有额外最大值M；Q=1有限上界是由给定函数和Q0解析导出的，不是新的调参上限。
