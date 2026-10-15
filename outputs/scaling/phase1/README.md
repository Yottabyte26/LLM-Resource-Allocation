# 问题二 PHASE 1

本阶段独立于问题一，仅使用 B1 拟合经典加性 N–D–Loss 模型。输入 N、D 直接
使用 CSV 的十亿单位，训练目标使用原始 Loss，未取 log(Loss)。所有前阶段文件
保持只读，完整冻结哈希记录在 `frozen_project_before.json`。

代码入口：

```powershell
python -B scripts/10_diagnose_scaling_data.py
python -B scripts/11_fit_classical_scaling.py
python -B scripts/11_verify_classical_scaling.py
```

全流程完成后拟合脚本会拒绝再次运行，以保护本阶段冻结参数。复核只需运行
验证脚本，不必重新读取问题一的全量质量数据或重复 bootstrap。

只刷新当前阶段已有结果的报告或图表，可分别使用
`python -B scripts/11_fit_classical_scaling.py --report-only`、
`python -B scripts/11_fit_classical_scaling.py --figures-only`；两者均不拟合模型。

32 初值用于各完整拟合、分组 CV 和每次 cluster bootstrap。bootstrap 以完整
轨迹抽样 2,000 次，固定种子 2026，使用两个 CPU 工作进程。Windows 沙箱可能
阻止进程通信命名管道；本次运行已获得脚本范围的沙箱外执行授权。

bootstrap 中断后，在尚未完成阶段且输入/配置不变的前提下，拟合脚本可从
`bootstrap_checkpoint.json` 和预测数组检查点继续。抽样计划、主模型与配置
的 SHA-256 必须一致，否则拒绝恢复。

B2 为半合成辅助检验；B3 为插值轨迹一致性；B4 为跨族表数值诊断；B5 分 source
与 family 检验，缺少评测口径，所以跨文献绝对误差仅作描述。上述输入不用于
参数选择或校准。没有使用 B6–B10，没有构造质量扩展，也没有开始问题三。

主要结果：`classical_scaling_parameters.csv`、`B1_group_cv.csv`、
`parameter_bootstrap.csv`、`parameter_correlation.csv`、
`terminal_checkpoint_sensitivity.csv`、`B2_validation.csv`、`B3_validation.csv`、
`B4_cross_family_validation.csv`、`B5_literature_validation.csv`。

报告：`outputs/report/question2_phase1_classical_scaling.md`。
图表：`outputs/figures/question2_phase1/`。
