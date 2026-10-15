# B9/B10 实际数据诊断

B9 是模型元数据，无直接可比较 Loss。B10 的磁盘字段名为 val_loss，但按赛题口径是估算 Loss。

## B9

文件：real_attachments/B_scaling_laws/supplementary_large_models.csv
实际字段：['model_name', 'N_params_B', 'D_tokens_B', 'FLOPs', 'publication_date', 'organization', 'accessibility', 'country']
行数：132；N 范围：100.0–10000.0B；D 范围：0.0–36000.0B。
缺失字段计数：{'model_name': 0, 'N_params_B': 0, 'D_tokens_B': 0, 'FLOPs': 11, 'publication_date': 0, 'organization': 1, 'accessibility': 2, 'country': 1}；完全重复：0；名称重复：0。
非正 N：0；非正 D：4。

## B10

文件：real_attachments/B_scaling_laws/supplementary_large_baseline.csv
实际字段：['family', 'N_params_B', 'D_tokens_B', 'val_loss', 'is_converged']
行数：128；N 范围：100.0–10000.0B；D 范围：0.1–36000.0B。
缺失字段计数：{'family': 0, 'N_params_B': 0, 'D_tokens_B': 0, 'val_loss': 0, 'is_converged': 0}；完全重复：0；名称重复：0。
非正 N：0；非正 D：0。

## 对应关系

按 B9.model_name = B10.family 精确一对一关联：匹配 128，仅 B9 4，仅 B10 0。
匹配行 N、D 完全一致。B9 中 D=0 的行完整保留，D=0 不可代入经典公式，不能视作实际零训练或擅自填补。
B9 只提供所报告参数规模和训练数据规模的现实锚点。元数据未统一稠密/稀疏模型、总/激活参数及各家评测口径。
B10 对照仅设置结构参考 Q=1、p=p_ref，计算 L0；这不是测得的大模型真实 Q/p，也不验证完整广义模型。
