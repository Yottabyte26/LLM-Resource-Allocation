# 优化数学形式

所有目标参数、Q/K/M(p)/lambda_p均冻结。N_B、D_B为十亿单位，物理N=10^9N_B、D=10^9D_B。

L_gen=E+A N_B^(-alpha)+B D_B^(-beta)+C_Q (N_B/N0_B)^(-u_Q)(-lnQ)+lambda_p(N_B)[M(p)-M_ref]。

P0：min_{N_B,D_B,Q} L_gen(N_B,D_B,Q,p_ref)。约束

(6+eta L_ctx)N_B D_B 10^18 + D_B 10^9 [g(Q)-g(Q0)]+ ≤ C_budget，
N_min≤N_B≤N_max，D_min≤D_B≤D_max，Q0≤Q≤1。

L_ctx为外生C7离散场景，成本函数类型与预算也为外生场景。P1对每个已保存经验候选p建立同样三维问题，再外层比较候选；本阶段不求解。P2只作为后续敏感性。

## 单调性与预算松弛

P0的DeltaM=0：
dL/dD_B=-beta B D_B^(-beta-1)<0；
dL/dQ=-C_Q (N_B/N0_B)^(-u_Q)/Q<0；
dL/dN_B=-alpha A N_B^(-alpha-1)-u_Q DeltaQ/N_B<0。

成本对N_B的导数=(6+eta L_ctx)D_B 10^18>0；对D_B为(6+eta L_ctx)N_B 10^18+10^9Δg>0；
对Q在Q≥Q0取右导数D_B10^9 g′(Q)>0。
因此可行域内若有任一变量尚未触达上界且预算严格松弛，就可以小幅提高该变量降低目标。
P0最优预算必须绑定，或全部N/D/Q达到各自上界。全部上界仍成本小于预算时允许剩余预算，不能硬编码等号。

P1中记DeltaM=M(p)-M_ref，s=d lambda_p/d lnN_B：
dL/dN_B=[-alpha A N_B^(-alpha)-u_Q DeltaQ+s DeltaM]/N_B。
该导数未保证为负，必须保留冻结分段lambda并检查左右导数；在节点主约定为右导数。
不过D/Q在固定p下仍严格递减：若任意一个未达到上界，预算松弛仍可被排除。
若D和Q已触上界，N局部不单调、位于节点或达到上界，可存在预算松弛。
质量模块外推不自动改变公式的单调符号，但降低上述数学结论的经验解释可信度。

## 后续数值求解要求

对每种成本/上下文/预算/域方案先检查可行性，不以调整边界或Q0掩盖不可行；记录不等式松弛。
分段lambda、对数型凹质量成本、有限候选p可能导致非凸/多分支，后续需比较边界和分段解，而非假定一个KKT内点就是全局解。
在质量模块局部证据外，最优解只解释为冻结模型情景结果。P1在N>1B时保留Class D标记。
