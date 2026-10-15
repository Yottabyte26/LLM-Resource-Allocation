# 源程序提取静态核验

执行范围：源码 AST/原行切片、文件复制与哈希、既有报告/JSON/清单读取。没有加载项目模型模块，没有重新训练/拟合/优化/预测/绘图，也没有重新读取 A1/A2/A3 质量记录。原数据仅作字节哈希完整性核对。
逐文件原始正文保持完整。备份为逐字节复制；汇编只统一文本行结束并增加来源/用途/候选说明。compact 为原片段，不能当作完整可执行脚本。没有为凑篇幅或贴合文字改写函数、参数、公式、变量或模型。

## 原文件存在、原始备份字节一致

94 个按问题归档副本全部 SHA256 匹配

## 逐行追溯

151 个原文件/函数/调用片段逐行匹配；505 个函数/类节点原位置可查

## 最终源版本证据

51 条历史 implementation_manifest 源码哈希全部匹配；其余调用链由实际 import、最终报告、输出参数及冻结文件哈希支持

## 问题一最终指标

Q=Z@w；K=abs(Z-Q)@w。最终保留(Q,K)，不采用Q*、Q_robust或分组Huber作为主评分；保留ILR＋一阶ElasticNet，Ridge只是数值对照

## 问题二公式

L0+ C_Q*(N_B/1)^(-u_Q)*(-lnQ) + lambda_p(N_B)*(M(p)-M_ref)。S3的C/u原参数读入，v去掉且不重拟合；B6局部拟合不宣称统一N-D-Q法则；S-A为主边界

## 问题二证据分层

1M/60M/1B观测配方与10B/70B估算Loss明确区分；后两者不作为真实实验或调参/模型选择证据；同脚本S1/S2/MQ及平滑lambda候选有明确注释

## 问题三D解析消元

实现 d=min(Dmax_B,(C/1e18)/(N_B*(6+eta*Lctx)+[g(Q)-g(Q0)]_+/1e9))，与用户分母N_B*(6+eta*Lctx)*1e18+1e9*[g(Q)-g(Q0)]_+等价，输出是D_B而非原始token数；保留D_min约束

## 问题三证据层级

P0正文主方案；P1经验候选增强；P2-A理论外推/敏感性，P2-B正配方边界极限无界，不能报告有限最优解；解析消元、实际优化器和连续预算识别均有原源码

## 问题四回归与分解

M1含lnN/t，M2-C含lnC/t，M3含lnN/lnD/t；OLS+HC3及敏感性原样，双因素Shapley保留符号，Delta_scale+Delta_tech=预测增量，负项不能当作非负增长占比

## 问题四桥接

冻结 bridge_valid_for_conversion=False 保留；禁止把失败桥接用于Loss到Benchmark数值转换

## 问题四Phase3实际完成

已有独立核验verified=True，question4_complete=True，566项检查、90情景行；因此纳入真实锚点/C4增长/MC/回测/风险代码；本次没有重运行

## 问题四预测口径

冻结六任务Winsor/median/IQR使实际A_z为有操作性上界的变换；24个月情景越界应标风险，不称无限能力/可靠长期预测；保留全期变换及冻结M2的前视限制说明

## 冻结完整性

3161 个任务前已存在文件逐一SHA256与大小核对，无修改；本提取器只写appendix。额外观察到Word临时锁文件：~$约束下提升大语言模型能力的资源配置建模.docx；不删除、不修改，不作为源码或冻结结果

## 版本模糊性

没有无法确认最终使用入口的代码；文件包含未采用候选分支是实际比较过程的保留，不等于独立最终模型

## 论文公式一致性检查范围

已核对用户列明公式、最终Phase报告和冻结模型。未获得最终整篇论文稿，不能据此声称所有正文公式逐字核对；未发现所列主公式实质冲突。D*单位及A_z操作性上界需保持明确

## 证据文件

- source_manifest.csv / .md：问题→源文件→输出→用途→候选/排除规则。
- extraction_plan.json / import_graph.json：实际项目 import 闭包与入口顺序。
- source_traceability.csv / source_function_traceability.csv：完整/精简展示片段及原行号。
- full_source_hashes.csv：每问备份与原文件 SHA256。
- recorded_source_hash_checks.json：51条历史实现清单版本对照。
- execution_evidence.csv：报告/冻结结果/配置真实路径、大小与SHA256。
- static_formula_checks.csv：静态公式标记；不执行数值模型。
- original_files_before_extraction.json / frozen_integrity_check.json：任务前后原文件完整性。
- extraction_summary.json：真实行数和明确未渲染的页数估计。

LaTeX只核验列表成对、代码片段和原文件一致性，没有编译论文或生成新图表；页数不是渲染测量值。纯绘图/报告及独立审计从compact省略；关键拟合、CV、bootstrap、优化约束及预测/分解保留。
附录完成后停止，不修改论文正文。

## 独立静态复核

94份原始备份均通过AST解析；两份TeX的列表起止成对。直接从compact TeX按四问提取的原代码行数依次为1250、1265、1004、1455，与源片段清单完全相同。没有执行项目代码。详细结果见independent_static_verification.json。
