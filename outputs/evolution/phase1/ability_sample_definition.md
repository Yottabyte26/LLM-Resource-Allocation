# Ability sample definition

原始评测4576条；每模型一行4497个。主标定S1=422；S0=4497，S2=0。

重复模型按最新Submission Date选评测集合；同日多条逐任务取中位数，不选最高分，不修改官方CSV。模型类型元数据冲突保留unknown。

六维完整且0–100有效是S0。S1要求C2明确Epoch开权重Yes或保守C4匹配Yes，出现C2/C4冲突则不纳入。原始parquet的Available on hub不等同于研究许可，也不单独作为open weights证据。

S2需要明确授权条款；附件只有许可证标识，因此license_unknown，当前严格敏感性样本为空，不将其伪造成已确认开放许可。

A_z=A_equal=sum((clip(x,P01,P99)-median)/IQR)/6；全部参数一次由S1计算，跨月固定。退化IQR为0的任务置中性0并记录；缺任务不填补，不与完整指数混合。A_100只作展示，基于所有S0的固定P01/P99映射。

本阶段用完整已观测时期定义固定口径；后续如做时间外验证，应只在相应训练期拟合新的分析版本，不能将此全期标定冒称无前视验证。

N取排行榜实际评测参数量，C4 N另列；超过25%相对差异标记conflict，不覆盖。D优先C4明确token证据，否则可用C6明确D_tokens_B并保留来源；不推导compute=6ND，不把C7 TB换成token。