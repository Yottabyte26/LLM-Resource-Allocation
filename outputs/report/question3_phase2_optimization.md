# 问题三 PHASE 2 正式资源优化与结构性转移

冻结问题一/二与PHASE 1保持只读。P0为正文主方案，三类官方成本平行运行；P1为经验配方增强，主域和扩展域分开。没有P2 simplex优化或问题四。

## 1 解析消去D与单位

固定N/Q/p/Lctx时，∂L/∂D_B=-beta B D_B^(-beta-1)<0；成本线性为D_B c_D。因此D*=min(D_max,C_budget/c_D)，c_D=N_B(6+etaLctx)10^18+10^9[g(Q)-g(Q0)]。D*<D_min则该N/Q不可行。这使连续优化降为logN/Q二维，没有强设预算等号；D达到上界时允许预算松弛。目标仍用B单位，成本显式转换实际参数/token。新15组Decimal单位核验通过，未重拟合参数。

冻结目标：L_gen=E+A N_B^(-alpha)+B D_B^(-beta)+C_Qloss(N_B/N0_B)^(-u_Q)(-lnQ)+lambda_p(N_B)[M(p)-M_ref]。C_Qloss对应冻结JSON的quality.C_Q字段，只是符号区分，不改系数；它与质量算力成本C_quality不同。

## 2 求解与独立最优性核验

每个光滑分段采用121×81网格及可行边界/预算D上界折线、至少10个不同初值SLSQP，显式评估节点两侧和节点本身。官方以及Q0敏感性可行场景全部使用全域differential_evolution独立复核，再细化其候选。最大DE细化与最终Loss差=2.992202e-10；完整三维核验30/30通过，832配方全枚举核验20/20通过。全枚举是在每个随机场景的31×23资源点上对832配方逐一评价，验证点态及整个资源网格最优一致，非逐配方重复连续求解。主/扩展路径Loss随预算不恶化的检查见monotonicity_checks.csv；这提供强数值复核，有限多优化器不构成任意非凸问题的形式化全局证明。

## 3 官方低中高预算 主域P0

| budget | cost_function | L_ctx | status | N_star | D_star | Q_star | L_star |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e+19 | exponential | 2048 | SOLVED | 0.156016 | 10 | 0.7280227 | 3.079781 |
| 1e+19 | exponential | 4096 | SOLVED | 0.1466448 | 10 | 0.7280227 | 3.094687 |
| 1e+19 | exponential | 8192 | SOLVED | 0.1309175 | 10 | 0.7280227 | 3.122792 |
| 1e+19 | exponential | 32768 | SOLVED | 0.07965842 | 10 | 0.7280227 | 3.25912 |
| 1e+19 | exponential | 131072 | INFEASIBLE | — | — | — | — |
| 1e+19 | logarithmic | 2048 | SOLVED | 0.156016 | 10 | 0.7280227 | 3.079781 |
| 1e+19 | logarithmic | 4096 | SOLVED | 0.1466448 | 10 | 0.7280227 | 3.094687 |
| 1e+19 | logarithmic | 8192 | SOLVED | 0.1309175 | 10 | 0.7280227 | 3.122792 |
| 1e+19 | logarithmic | 32768 | SOLVED | 0.07965842 | 10 | 0.7280227 | 3.25912 |
| 1e+19 | logarithmic | 131072 | INFEASIBLE | — | — | — | — |
| 1e+19 | power | 2048 | SOLVED | 0.156016 | 10 | 0.7280227 | 3.079781 |
| 1e+19 | power | 4096 | SOLVED | 0.1466448 | 10 | 0.7280227 | 3.094687 |
| 1e+19 | power | 8192 | SOLVED | 0.1309175 | 10 | 0.7280227 | 3.122792 |
| 1e+19 | power | 32768 | SOLVED | 0.07965842 | 10 | 0.7280227 | 3.25912 |
| 1e+19 | power | 131072 | INFEASIBLE | — | — | — | — |
| 1e+22 | exponential | 2048 | SOLVED | 5.5421 | 257.9449 | 1 | 2.149739 |
| 1e+22 | exponential | 4096 | SOLVED | 5.373241 | 250.7112 | 1 | 2.153926 |
| 1e+22 | exponential | 8192 | SOLVED | 5.07822 | 237.8981 | 1 | 2.161708 |
| 1e+22 | exponential | 32768 | SOLVED | 3.976729 | 188.0845 | 1 | 2.197595 |
| 1e+22 | exponential | 131072 | SOLVED | 2.527375 | 118.1151 | 1 | 2.274311 |
| 1e+22 | logarithmic | 2048 | SOLVED | 5.113763 | 299.893 | 1 | 2.144395 |
| 1e+22 | logarithmic | 4096 | SOLVED | 4.963806 | 290.5521 | 1 | 2.148698 |
| 1e+22 | logarithmic | 8192 | SOLVED | 4.710588 | 273.6023 | 1 | 2.156688 |
| 1e+22 | logarithmic | 32768 | SOLVED | 3.748035 | 209.9985 | 1 | 2.193399 |
| 1e+22 | logarithmic | 131072 | SOLVED | 2.435504 | 126.5401 | 1 | 2.271352 |
| 1e+22 | power | 2048 | SOLVED | 5.594667 | 253.4534 | 1 | 2.150397 |
| 1e+22 | power | 4096 | SOLVED | 5.422909 | 246.4553 | 1 | 2.154571 |
| 1e+22 | power | 8192 | SOLVED | 5.122975 | 234.0455 | 1 | 2.16233 |
| 1e+22 | power | 32768 | SOLVED | 4.004961 | 185.6249 | 1 | 2.19812 |
| 1e+22 | power | 131072 | SOLVED | 2.538952 | 117.1149 | 1 | 2.274687 |
| 1e+24 | exponential | 2048 | SOLVED | 11.96583 | 299.893 | 1 | 2.093379 |
| 1e+24 | exponential | 4096 | SOLVED | 11.96583 | 299.893 | 1 | 2.093379 |
| 1e+24 | exponential | 8192 | SOLVED | 11.96583 | 299.893 | 1 | 2.093379 |
| 1e+24 | exponential | 32768 | SOLVED | 11.96583 | 299.893 | 1 | 2.093379 |
| 1e+24 | exponential | 131072 | SOLVED | 11.96583 | 299.893 | 1 | 2.093379 |
| 1e+24 | logarithmic | 2048 | SOLVED | 11.96583 | 299.893 | 1 | 2.093379 |
| 1e+24 | logarithmic | 4096 | SOLVED | 11.96583 | 299.893 | 1 | 2.093379 |
| 1e+24 | logarithmic | 8192 | SOLVED | 11.96583 | 299.893 | 1 | 2.093379 |
| 1e+24 | logarithmic | 32768 | SOLVED | 11.96583 | 299.893 | 1 | 2.093379 |
| 1e+24 | logarithmic | 131072 | SOLVED | 11.96583 | 299.893 | 1 | 2.093379 |
| 1e+24 | power | 2048 | SOLVED | 11.96583 | 299.893 | 1 | 2.093379 |
| 1e+24 | power | 4096 | SOLVED | 11.96583 | 299.893 | 1 | 2.093379 |
| 1e+24 | power | 8192 | SOLVED | 11.96583 | 299.893 | 1 | 2.093379 |
| 1e+24 | power | 32768 | SOLVED | 11.96583 | 299.893 | 1 | 2.093379 |
| 1e+24 | power | 131072 | SOLVED | 11.96583 | 299.893 | 1 | 2.093379 |

成本/未使用预算均为FLOPs：

| budget | cost_function | L_ctx | C_train | C_quality | C_attn | unused_budget | budget_binding | evidence_domain_saturated | extrapolative_Q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1e+19 | exponential | 2048 | 9.360959e+18 | 0 | 6.390414e+17 | 2048 | True | False | False |
| 1e+19 | exponential | 4096 | 8.798686e+18 | 0 | 1.201314e+18 | 2048 | True | False | False |
| 1e+19 | exponential | 8192 | 7.855048e+18 | 0 | 2.144952e+18 | 0 | True | False | False |
| 1e+19 | exponential | 32768 | 4.779505e+18 | 0 | 5.220495e+18 | 0 | True | False | False |
| 1e+19 | exponential | 131072 | — | — | — | — | — | — | — |
| 1e+19 | logarithmic | 2048 | 9.360959e+18 | 0 | 6.390414e+17 | 2048 | True | False | False |
| 1e+19 | logarithmic | 4096 | 8.798686e+18 | 0 | 1.201314e+18 | 2048 | True | False | False |
| 1e+19 | logarithmic | 8192 | 7.855048e+18 | 0 | 2.144952e+18 | 0 | True | False | False |
| 1e+19 | logarithmic | 32768 | 4.779505e+18 | 0 | 5.220495e+18 | 0 | True | False | False |
| 1e+19 | logarithmic | 131072 | — | — | — | — | — | — | — |
| 1e+19 | power | 2048 | 9.360959e+18 | 0 | 6.390414e+17 | 2048 | True | False | False |
| 1e+19 | power | 4096 | 8.798686e+18 | 0 | 1.201314e+18 | 2048 | True | False | False |
| 1e+19 | power | 8192 | 7.855048e+18 | 0 | 2.144952e+18 | 0 | True | False | False |
| 1e+19 | power | 32768 | 4.779505e+18 | 0 | 5.220495e+18 | 0 | True | False | False |
| 1e+19 | power | 131072 | — | — | — | — | — | — | — |
| 1e+22 | exponential | 2048 | 8.577339e+21 | 8.371147e+20 | 5.855463e+20 | 0 | True | False | True |
| 1e+22 | exponential | 4096 | 8.082791e+21 | 8.13639e+20 | 1.10357e+21 | 0 | True | False | True |
| 1e+22 | exponential | 8192 | 7.248594e+21 | 7.720564e+20 | 1.979349e+21 | 0 | True | False | True |
| 1e+22 | exponential | 32768 | 4.487767e+21 | 6.10395e+20 | 4.901838e+21 | 0 | True | False | True |
| 1e+22 | exponential | 131072 | 1.791127e+21 | 3.833216e+20 | 7.825552e+21 | 0 | True | False | True |
| 1e+22 | logarithmic | 2048 | 9.201491e+21 | 1.703541e+20 | 6.281551e+20 | 0 | True | False | True |
| 1e+22 | logarithmic | 4096 | 8.653465e+21 | 1.65048e+20 | 1.181486e+21 | 0 | True | False | True |
| 1e+22 | logarithmic | 8192 | 7.732965e+21 | 1.554197e+20 | 2.111615e+21 | 0 | True | False | True |
| 1e+22 | logarithmic | 32768 | 4.722491e+21 | 1.192896e+20 | 5.158219e+21 | 0 | True | False | True |
| 1e+22 | logarithmic | 131072 | 1.849133e+21 | 7.188106e+19 | 8.078986e+21 | 0 | True | False | True |
| 1e+22 | power | 2048 | 8.507924e+21 | 9.112688e+20 | 5.808076e+20 | 0 | True | False | True |
| 1e+22 | power | 4096 | 8.019028e+21 | 8.861078e+20 | 1.094865e+21 | 2097152 | True | False | True |
| 1e+22 | power | 8192 | 7.194054e+21 | 8.414893e+20 | 1.964456e+21 | 0 | True | False | True |
| 1e+22 | power | 32768 | 4.460523e+21 | 6.673974e+20 | 4.87208e+21 | 0 | True | False | True |
| 1e+22 | power | 131072 | 1.784095e+21 | 4.210761e+20 | 7.794829e+21 | 0 | True | False | True |
| 1e+24 | exponential | 2048 | 2.15308e+22 | 9.732498e+20 | 1.469836e+21 | 9.760261e+23 | False | True | True |
| 1e+24 | exponential | 4096 | 2.15308e+22 | 9.732498e+20 | 2.939672e+21 | 9.745563e+23 | False | True | True |
| 1e+24 | exponential | 8192 | 2.15308e+22 | 9.732498e+20 | 5.879345e+21 | 9.716166e+23 | False | True | True |
| 1e+24 | exponential | 32768 | 2.15308e+22 | 9.732498e+20 | 2.351738e+22 | 9.539786e+23 | False | True | True |
| 1e+24 | exponential | 131072 | 2.15308e+22 | 9.732498e+20 | 9.406951e+22 | 8.834264e+23 | False | True | True |
| 1e+24 | logarithmic | 2048 | 2.15308e+22 | 1.703541e+20 | 1.469836e+21 | 9.76829e+23 | False | True | True |
| 1e+24 | logarithmic | 4096 | 2.15308e+22 | 1.703541e+20 | 2.939672e+21 | 9.753592e+23 | False | True | True |
| 1e+24 | logarithmic | 8192 | 2.15308e+22 | 1.703541e+20 | 5.879345e+21 | 9.724195e+23 | False | True | True |
| 1e+24 | logarithmic | 32768 | 2.15308e+22 | 1.703541e+20 | 2.351738e+22 | 9.547815e+23 | False | True | True |
| 1e+24 | logarithmic | 131072 | 2.15308e+22 | 1.703541e+20 | 9.406951e+22 | 8.842293e+23 | False | True | True |
| 1e+24 | power | 2048 | 2.15308e+22 | 1.078238e+21 | 1.469836e+21 | 9.759211e+23 | False | True | True |
| 1e+24 | power | 4096 | 2.15308e+22 | 1.078238e+21 | 2.939672e+21 | 9.744513e+23 | False | True | True |
| 1e+24 | power | 8192 | 2.15308e+22 | 1.078238e+21 | 5.879345e+21 | 9.715116e+23 | False | True | True |
| 1e+24 | power | 32768 | 2.15308e+22 | 1.078238e+21 | 2.351738e+22 | 9.538736e+23 | False | True | True |
| 1e+24 | power | 131072 | 2.15308e+22 | 1.078238e+21 | 9.406951e+22 | 8.833214e+23 | False | True | True |

最低预算×131072上下文最低成本约2.27247e19 FLOPs，故1e19证据支持域内不可行；不可行行保留状态、没有伪造决策。Q=1超过问题一实际样本质量上界，虽在B6半合成数值范围内，不能称直接验证。

## 4 主域饱和与扩展情景

P0各资源边际Loss均负。主域饱和预算可解析为D_max[(6+etaLctx)N_max+Δg(1)/10^9]10^18；达到此预算后P0取三个上界且剩余预算不再改变冻结目标。不是求解失败，也不能覆盖此结果而偷偷扩大边界。

| cost_function | L_ctx | exact_saturation_budget | first_saturated_grid_budget |
| --- | --- | --- | --- |
| exponential | 2048 | 2.397389e+22 | 3.548134e+22 |
| exponential | 4096 | 2.544373e+22 | 3.548134e+22 |
| exponential | 8192 | 2.83834e+22 | 3.548134e+22 |
| exponential | 32768 | 4.602143e+22 | 5.308844e+22 |
| exponential | 131072 | 1.165736e+23 | 1.188502e+23 |
| power | 2048 | 2.407888e+22 | 3.548134e+22 |
| power | 4096 | 2.554871e+22 | 3.548134e+22 |
| power | 8192 | 2.848839e+22 | 3.548134e+22 |
| power | 32768 | 4.612642e+22 | 5.308844e+22 |
| power | 131072 | 1.166786e+23 | 1.188502e+23 |
| logarithmic | 2048 | 2.317099e+22 | 2.371374e+22 |
| logarithmic | 4096 | 2.464083e+22 | 3.548134e+22 |
| logarithmic | 8192 | 2.75805e+22 | 3.548134e+22 |
| logarithmic | 32768 | 4.521854e+22 | 5.308844e+22 |
| logarithmic | 131072 | 1.157707e+23 | 1.188502e+23 |

以下是独立扩展域P0情景，所有超出主N/D/Q实测范围的点分别标记，scenario_extrapolation始终为True：

| budget | cost_function | L_ctx | status | N_star | D_star | Q_star | L_star |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e+19 | exponential | 2048 | SOLVED | 0.156016 | 10 | 0.7280227 | 3.079781 |
| 1e+19 | exponential | 4096 | SOLVED | 0.1466448 | 10 | 0.7280227 | 3.094687 |
| 1e+19 | exponential | 8192 | SOLVED | 0.1309175 | 10 | 0.7280227 | 3.122792 |
| 1e+19 | exponential | 32768 | SOLVED | 0.07965842 | 10 | 0.7280227 | 3.25912 |
| 1e+19 | exponential | 131072 | INFEASIBLE | — | — | — | — |
| 1e+19 | logarithmic | 2048 | SOLVED | 0.156016 | 10 | 0.7280227 | 3.079781 |
| 1e+19 | logarithmic | 4096 | SOLVED | 0.1466448 | 10 | 0.7280227 | 3.094687 |
| 1e+19 | logarithmic | 8192 | SOLVED | 0.1309175 | 10 | 0.7280227 | 3.122792 |
| 1e+19 | logarithmic | 32768 | SOLVED | 0.07965842 | 10 | 0.7280227 | 3.25912 |
| 1e+19 | logarithmic | 131072 | INFEASIBLE | — | — | — | — |
| 1e+19 | power | 2048 | SOLVED | 0.156016 | 10 | 0.7280227 | 3.079781 |
| 1e+19 | power | 4096 | SOLVED | 0.1466448 | 10 | 0.7280227 | 3.094687 |
| 1e+19 | power | 8192 | SOLVED | 0.1309175 | 10 | 0.7280227 | 3.122792 |
| 1e+19 | power | 32768 | SOLVED | 0.07965842 | 10 | 0.7280227 | 3.25912 |
| 1e+19 | power | 131072 | INFEASIBLE | — | — | — | — |
| 1e+22 | exponential | 2048 | SOLVED | 5.5421 | 257.9449 | 1 | 2.149739 |
| 1e+22 | exponential | 4096 | SOLVED | 5.373241 | 250.7112 | 1 | 2.153926 |
| 1e+22 | exponential | 8192 | SOLVED | 5.078221 | 237.8981 | 1 | 2.161708 |
| 1e+22 | exponential | 32768 | SOLVED | 3.976729 | 188.0845 | 1 | 2.197595 |
| 1e+22 | exponential | 131072 | SOLVED | 2.527375 | 118.1151 | 1 | 2.274311 |
| 1e+22 | logarithmic | 2048 | SOLVED | 5.107919 | 300.2303 | 1 | 2.144395 |
| 1e+22 | logarithmic | 4096 | SOLVED | 4.963806 | 290.5521 | 1 | 2.148698 |
| 1e+22 | logarithmic | 8192 | SOLVED | 4.710587 | 273.6023 | 1 | 2.156688 |
| 1e+22 | logarithmic | 32768 | SOLVED | 3.748035 | 209.9985 | 1 | 2.193399 |
| 1e+22 | logarithmic | 131072 | SOLVED | 2.435504 | 126.5401 | 1 | 2.271352 |
| 1e+22 | power | 2048 | SOLVED | 5.594667 | 253.4534 | 1 | 2.150397 |
| 1e+22 | power | 4096 | SOLVED | 5.422908 | 246.4553 | 1 | 2.154571 |
| 1e+22 | power | 8192 | SOLVED | 5.122974 | 234.0455 | 1 | 2.16233 |
| 1e+22 | power | 32768 | SOLVED | 4.004961 | 185.6249 | 1 | 2.19812 |
| 1e+22 | power | 131072 | SOLVED | 2.538952 | 117.1149 | 1 | 2.274687 |
| 1e+24 | exponential | 2048 | SOLVED | 40.63103 | 3792.563 | 1 | 1.913816 |
| 1e+24 | exponential | 4096 | SOLVED | 39.49172 | 3669.088 | 1 | 1.915942 |
| 1e+24 | exponential | 8192 | SOLVED | 37.48885 | 3453.037 | 1 | 1.91989 |
| 1e+24 | exponential | 32768 | SOLVED | 29.86358 | 2644.517 | 1 | 1.938021 |
| 1e+24 | exponential | 131072 | SOLVED | 19.43577 | 1588.923 | 1 | 1.976505 |
| 1e+24 | logarithmic | 2048 | SOLVED | 40.15334 | 3876.947 | 1 | 1.913462 |
| 1e+24 | logarithmic | 4096 | SOLVED | 39.04249 | 3748.033 | 1 | 1.915597 |
| 1e+24 | logarithmic | 8192 | SOLVED | 37.08744 | 3522.904 | 1 | 1.919559 |
| 1e+24 | logarithmic | 32768 | SOLVED | 29.61852 | 2685.378 | 1 | 1.937749 |
| 1e+24 | logarithmic | 131072 | SOLVED | 19.33985 | 1603.618 | 1 | 1.976316 |
| 1e+24 | power | 2048 | SOLVED | 40.69273 | 3781.869 | 1 | 1.913862 |
| 1e+24 | power | 4096 | SOLVED | 39.54977 | 3659.074 | 1 | 1.915987 |
| 1e+24 | power | 8192 | SOLVED | 37.54076 | 3444.158 | 1 | 1.919932 |
| 1e+24 | power | 32768 | SOLVED | 29.89536 | 2639.29 | 1 | 1.938057 |
| 1e+24 | power | 131072 | SOLVED | 19.44825 | 1587.027 | 1 | 1.976529 |

1e24主域P0全部15个成本×上下文场景饱和；扩展域对应15个结果在N或D上超出主域。B9元数据、B10估算Loss及总/激活参数差异不能为这些联合最优点提供直接验证。

## 5 三种质量成本与投资节奏

三类公式与参数均保持题给：1e7exp(6Q)、5e9Q^4、2e9ln(1+10Q)，均扣除g(Q0)、乘实际D。报告不按结果是否漂亮选择主成本；具体Q脱离Q0+0.01、达到0.9/0.95/0.99/1的局部二分预算见quality_threshold_budgets.csv。“明显离开”以绝对0.01定义（研究者阈值）；满分判定容差1e-7。所有转移预算是冻结模型条件结果。

| cost_function | threshold | budget_threshold |
| --- | --- | --- |
| exponential | departure_Q0_plus_0.01 | 4.55039e+19 |
| exponential | 0.9 | 3.008725e+20 |
| exponential | 0.95 | 5.639256e+20 |
| exponential | 0.99 | 9.381875e+20 |
| exponential | 1.0 | 1.066196e+21 |
| power | departure_Q0_plus_0.01 | 1.088916e+20 |
| power | 0.9 | 3.304855e+20 |
| power | 0.95 | 4.650936e+20 |
| power | 0.99 | 6.07899e+20 |
| power | 1.0 | 6.49413e+20 |
| logarithmic | departure_Q0_plus_0.01 | 1.689709e+19 |
| logarithmic | 0.9 | 2.025025e+19 |
| logarithmic | 0.95 | 2.11653e+19 |
| logarithmic | 0.99 | 2.186397e+19 |
| logarithmic | 1.0 | 2.203428e+19 |

## 6 上下文与30k临界值

| L_ctx | ratio_attn_train | L_ctx_crit | attention_dominant |
| --- | --- | --- | --- |
| 2048 | 0.06826667 | 30000 | False |
| 4096 | 0.1365333 | 30000 | False |
| 8192 | 0.2730667 | 30000 | False |
| 32768 | 1.092267 | 30000 | True |
| 131072 | 4.369067 | 30000 | True |

C_attn/C_train=etaLctx/6与预算/决策/配方无关。8192→32768时比例从0.27307增至1.09227；跨临界表示成本代理的注意力超过基础训练，不是由预算驱动的两个份额交叉。不能要求N/D/Q全部同向变化：它们存在替代、边界和质量成本效应。官方预算对照：

| cost_function | budget | N_8192 | N_32768 | D_8192 | D_32768 | Q_8192 | Q_32768 | Loss_change | ratio_8192 | ratio_32768 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exponential | 1e+19 | 0.1309175 | 0.07965842 | 10 | 10 | 0.7280227 | 0.7280227 | 0.1363286 | 0.2730667 | 1.092267 |
| exponential | 1e+22 | 5.07822 | 3.976729 | 237.8981 | 188.0845 | 1 | 1 | 0.03588645 | 0.2730667 | 1.092267 |
| exponential | 1e+24 | 11.96583 | 11.96583 | 299.893 | 299.893 | 1 | 1 | 0 | 0.2730667 | 1.092267 |
| logarithmic | 1e+19 | 0.1309175 | 0.07965842 | 10 | 10 | 0.7280227 | 0.7280227 | 0.1363286 | 0.2730667 | 1.092267 |
| logarithmic | 1e+22 | 4.710588 | 3.748035 | 273.6023 | 209.9985 | 1 | 1 | 0.03671087 | 0.2730667 | 1.092267 |
| logarithmic | 1e+24 | 11.96583 | 11.96583 | 299.893 | 299.893 | 1 | 1 | 0 | 0.2730667 | 1.092267 |
| power | 1e+19 | 0.1309175 | 0.07965842 | 10 | 10 | 0.7280227 | 0.7280227 | 0.1363286 | 0.2730667 | 1.092267 |
| power | 1e+22 | 5.122975 | 4.004961 | 234.0455 | 185.6249 | 1 | 1 | 0.03579032 | 0.2730667 | 1.092267 |
| power | 1e+24 | 11.96583 | 11.96583 | 299.893 | 299.893 | 1 | 1 | 0 | 0.2730667 | 1.092267 |

## 7 Type I硬转移

活跃集采用1e-6归一化容差，先从44点路径发现变化，再局部log预算二分至括区宽≤1e-5。可行性阈值另由解析最低成本给出。并发N上界与预算松弛可对应同一转移，不能数作两个独立经济机制。代表2048上下文的主域事件：

| cost_function | constraint | transition | budget_critical | bracket_low | bracket_high |
| --- | --- | --- | --- | --- | --- |
| exponential | feasibility | INFEASIBLE→FEASIBLE | 4.52146e+18 | 4.52146e+18 | 4.52146e+18 |
| exponential | active_N_upper | False→True | 2.397396e+22 | 2.397381e+22 | 2.397411e+22 |
| exponential | active_D_lower | True→False | 2.248123e+19 | 2.248109e+19 | 2.248136e+19 |
| exponential | active_D_upper | False→True | 1.275177e+22 | 1.27517e+22 | 1.275185e+22 |
| exponential | active_Q_lower | True→False | 4.108309e+19 | 4.108284e+19 | 4.108334e+19 |
| exponential | active_Q_upper | False→True | 1.066183e+21 | 1.066176e+21 | 1.066189e+21 |
| exponential | budget_binding | True→False | 2.397396e+22 | 2.397381e+22 | 2.397411e+22 |
| power | feasibility | INFEASIBLE→FEASIBLE | 4.52146e+18 | 4.52146e+18 | 4.52146e+18 |
| power | active_N_upper | False→True | 2.407885e+22 | 2.40787e+22 | 2.407899e+22 |
| power | active_D_lower | True→False | 2.248123e+19 | 2.248109e+19 | 2.248136e+19 |
| power | active_D_upper | False→True | 1.30873e+22 | 1.308721e+22 | 1.308738e+22 |
| power | active_Q_lower | True→False | 1.021641e+20 | 1.021635e+20 | 1.021647e+20 |
| power | active_Q_upper | False→True | 6.49413e+20 | 6.49409e+20 | 6.49417e+20 |
| power | budget_binding | True→False | 2.407885e+22 | 2.40787e+22 | 2.407899e+22 |
| logarithmic | feasibility | INFEASIBLE→FEASIBLE | 4.52146e+18 | 4.52146e+18 | 4.52146e+18 |
| logarithmic | active_N_upper | False→True | 2.317108e+22 | 2.317094e+22 | 2.317123e+22 |
| logarithmic | active_D_lower | True→False | 3.669977e+19 | 3.669954e+19 | 3.669999e+19 |
| logarithmic | active_D_upper | False→True | 9.980309e+21 | 9.980204e+21 | 9.980414e+21 |
| logarithmic | active_Q_lower | True→False | 1.666699e+19 | 1.666688e+19 | 1.666709e+19 |
| logarithmic | active_Q_upper | False→True | 2.203428e+19 | 2.203414e+19 | 2.203441e+19 |
| logarithmic | budget_binding | True→False | 2.317108e+22 | 2.317094e+22 | 2.317123e+22 |
| exponential | active_N_lower | True→False | 4.521488e+18 | 4.52146e+18 | 4.521517e+18 |
| logarithmic | active_N_lower | True→False | 4.521488e+18 | 4.52146e+18 | 4.521517e+18 |
| power | active_N_lower | True→False | 4.521488e+18 | 4.52146e+18 | 4.521517e+18 |

完整五上下文及扩展域见typeI_active_set_transitions.csv。数值活跃容差下的精确括区，不表示真实系统可支持同样精度。

## 8 Type II软转移及影子价格

对log10预算拟合单线性与连续单断点hinge，两侧至少5点，BIC参数数目分别2/4；只有DeltaBIC≥10才标显著。300次长度3循环块残差扰动提供断点区间；这是确定性模型曲线拟合/网格与近似误差的区间，不是现实转移预算的统计置信区间。主域有89条指标曲线满足门槛；不能当作89个独立物理转移。训练与注意力份额固定成比例，其BIC与断点重复，不是独立证据；单断点也只是对含多次边界变化曲线的近似。

| cost_function | metric | delta_BIC | break_budget | break_interval_low | break_interval_high |
| --- | --- | --- | --- | --- | --- |
| exponential | s_train | 25.61833 | 8.413951e+21 | 2.790134e+20 | 2.175204e+23 |
| exponential | s_quality | 60.59783 | 5.158222e+20 | 1.864769e+20 | 1.727826e+21 |
| exponential | s_attn | 25.61833 | 8.413951e+21 | 3.447466e+20 | 1.796284e+23 |
| exponential | ln_N | 98.60874 | 2.900681e+22 | 1.569007e+22 | 4.340103e+22 |
| exponential | ln_D | 80.97323 | 1.938653e+22 | 8.413951e+21 | 4.340103e+22 |
| exponential | Q | 83.91152 | 1.727826e+21 | 7.717915e+20 | 3.868121e+21 |
| power | s_train | 23.45985 | 8.413951e+21 | 3.447466e+20 | 1.200535e+23 |
| power | s_quality | 47.36393 | 5.158222e+20 | 1.539927e+20 | 2.134888e+21 |
| power | s_attn | 23.45985 | 8.413951e+21 | 3.447466e+20 | 6.493816e+22 |
| power | ln_N | 100.01 | 2.900681e+22 | 1.938653e+22 | 4.340103e+22 |
| power | ln_D | 79.27099 | 2.900681e+22 | 1.295687e+22 | 6.493816e+22 |
| power | Q | 52.11497 | 2.585235e+21 | 7.717915e+20 | 1.295687e+22 |
| logarithmic | s_train | 33.0204 | 5.78762e+21 | 7.717915e+20 | 6.493816e+22 |
| logarithmic | s_quality | 17.4674 | 3.072557e+19 | 2.053525e+19 | 5.158222e+20 |
| logarithmic | s_attn | 33.0204 | 5.78762e+21 | 2.790134e+20 | 4.340103e+22 |
| logarithmic | ln_N | 97.03578 | 4.340103e+22 | 2.347605e+22 | 6.493816e+22 |
| logarithmic | ln_D | 87.45944 | 1.295687e+22 | 5.78762e+21 | 2.900681e+22 |
| logarithmic | Q | 69.6953 | 4.59727e+19 | 3.072557e+19 | 6.878599e+19 |

mu=-dL*/dC以及-E的剩余Loss预算弹性由相邻预算有限差分计算，见shadow_price_budget_elasticity.csv。Type I/II附近左右预算点的mu与弹性变化另列transition_shadow_support.csv，用于逐个事件核对；节点附近有限网格导数只作佐证；饱和后的影子价格应为零。不能单凭mu或弹性定义转移。

## 9 质量成本份额反转

| cost_function | L_ctx | quality_share_reversal | C_peak_Q | peak_s_quality | active_Q_upper_at_peak | nearest_active_transition |
| --- | --- | --- | --- | --- | --- | --- |
| exponential | 2048 | True | 1.066193e+21 | 0.1806272 | True | active_Q_upper |
| exponential | 4096 | True | 1.014021e+21 | 0.1792455 | True | active_Q_upper |
| exponential | 8192 | True | 9.249146e+20 | 0.1767366 | True | active_Q_upper |
| exponential | 32768 | True | 6.174647e+20 | 0.1660736 | True | active_Q_upper |
| exponential | 131072 | True | 2.853517e+20 | 0.1472714 | True | active_Q_upper |
| power | 2048 | True | 6.494165e+20 | 0.2231066 | True | active_Q_upper |
| power | 4096 | True | 6.182638e+20 | 0.2214877 | True | active_Q_upper |
| power | 8192 | True | 5.649834e+20 | 0.2185451 | True | active_Q_upper |
| power | 32768 | True | 3.801713e+20 | 0.2059909 | True | active_Q_upper |
| power | 131072 | True | 1.957586e+20 | 0.1836644 | False | active_Q_upper |
| logarithmic | 2048 | True | 2.203422e+19 | 0.2578016 | False | active_Q_upper |
| logarithmic | 4096 | True | 2.218833e+19 | 0.2560129 | True | active_Q_upper |
| logarithmic | 8192 | True | 2.247378e+19 | 0.2527611 | True | active_Q_upper |
| logarithmic | 32768 | True | 2.378338e+19 | 0.2388432 | True | active_Q_upper |
| logarithmic | 131072 | True | 2.840532e+19 | 0.19998 | True | active_N_lower |

峰值在预算网格邻域用一维搜索细化；是否先升后降按两侧分别至少变化1e-4判断（数值判据）。Q上界后质量处理总成本仍可随D增长，所以质量份额峰值不必与Q达到1完全同时。

## 10 P1精确配方降维及证据边界

成本与p无关，固定N时lambda>0取最小M、lambda<0取最大M、零点全部等价。因此经验全集合的点态下包络等于两个极值配方的包络；再对N/Q优化是数学精确降维。| role | candidate_id | M_p | DeltaM | source_members |
| --- | --- | --- | --- | --- |
| minM | P1_0083 | 4.811609 | -0.533048 | [{"source_dataset": "train_1M", "index": 84}] |
| maxM | P1_0278 | 5.8438 | 0.4991433 | [{"source_dataset": "train_1M", "index": 279}] |

N区间按冻结节点与条件过零点分割；过零是冻结估算段的结果，不是新观测的尺度反转。P1官方预算Loss改善范围：

| budget | min | max | mean |
| --- | --- | --- | --- |
| 1e+19 | 0.2968421 | 0.3462214 | 0.3131663 |
| 1e+22 | 0.1306276 | 0.1476083 | 0.1402828 |
| 1e+24 | 0.158095 | 0.158095 | 0.158095 |

完整N/D/Q变化及选择配方见P0_vs_P1_official_comparison.csv。N>1B的配比效应明确标mixture_transfer_uses_estimated_support=True、Class D；它只是增强情景，不能当高可信真实配方建议。冻结模型输入配方经过已有零替换，不再预处理；物理部署需回溯原始来源。

## 11 Q0敏感性

仅针对三官方预算×三个指定上下文×三成本运行三候选，共81行；未重跑完整连续路径或改变主Q0。| Q0 | Loss_change_min | Loss_change_max | N_change_min | N_change_max | active_structure_changed |
| --- | --- | --- | --- | --- | --- |
| 0.6925362 | 0 | 0.01284405 | 0 | 0.03769479 | 1 |
| 0.721576 | 0 | 0.002286137 | -0.002438843 | 0.007305319 | 0 |
| 0.7475281 | -0.006795682 | 0 | -0.02342339 | 0.007265197 | 0 |

基线改变可能影响低预算质量与Loss以及中预算配置，不能说所有数值不变；最低可行预算不变（CQ(Q0)=0），高预算主域饱和判定需按新的Δg(1)重新计算，但官方1e24依然可覆盖上界成本。敏感性只验证这些官方点，不据此宣称所有连续转移预算对Q0不敏感。

## 12 结论范围与停止点

以上是已冻结模块、成本代理、跨体系Q0映射及N/D矩形支持域下的最优配置；质量局部校准为半合成、估算lambda大尺度段及扩展域不可升级为直接证据。主域高预算饱和是支持范围受限。具备进入下一阶段P2理论敏感性及第三问收尾的技术条件，需人工确认本阶段结果；本轮没有执行P2或开始问题四。

## 图表

- [P0 N_star：主域实线、扩展域虚线；Lctx=2048](../../outputs/figures/question3_phase2/01_N_star_vs_budget.png)
- [P0 D_star：主域实线、扩展域虚线；Lctx=2048](../../outputs/figures/question3_phase2/02_D_star_vs_budget.png)
- [P0 Q_star：主域实线、扩展域虚线；Lctx=2048](../../outputs/figures/question3_phase2/03_Q_star_vs_budget.png)
- [P0 L_star：主域实线、扩展域虚线；Lctx=2048](../../outputs/figures/question3_phase2/04_L_star_vs_budget.png)
- [预算分母下三类成本与未使用份额](../../outputs/figures/question3_phase2/05_cost_shares.png)
- [有限差分影子价格；饱和后的零值不画在对数轴上](../../outputs/figures/question3_phase2/06_shadow_price.png)
- [五上下文下三种官方成本的质量投资路径](../../outputs/figures/question3_phase2/07_quality_cost_function_comparison.png)
- [主域五上下文对照：N_star](../../outputs/figures/question3_phase2/08_five_context_N_star.png)
- [主域五上下文对照：D_star](../../outputs/figures/question3_phase2/09_five_context_D_star.png)
- [主域五上下文对照：L_star](../../outputs/figures/question3_phase2/10_five_context_L_star.png)
- [P1相对P0的模型Loss改善，非新观测实验收益](../../outputs/figures/question3_phase2/11_P0_vs_P1_improvement.png)
- [2048上下文硬转移与显著软断点示意；软线不是额外独立物理相变](../../outputs/figures/question3_phase2/12_structural_transitions.png)
