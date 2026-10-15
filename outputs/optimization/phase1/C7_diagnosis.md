# C7 真实结构诊断

文件：`real_attachments/C_efficiency_evolution/model_architecture_metadata.csv`。实际 45 行 × 7 列。

字段：model_name, n_layers, n_heads, d_model, vocab_size, max_position_embeddings, training_data_TB。模型名重复 0；完全重复 0。总缺失单元格 0；有效上下文 45。

hidden_size 对应磁盘 d_model。没有独立参数量 N 字段，不从模型名猜参数量；没有明确 training tokens 字段。training_data_TB 字段含有数值，但单位语义不能由列名可靠判定为 token，不能转换成 D_tokens_B。参数量和训练数据上下界来自冻结问题二，不从该列反推。

| min | P01 | P05 | P10 | P25 | median | P75 | P90 | P95 | P99 | max | rows | valid_count | invalid_count | unique_value_count | mean | std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 2048 | 2048 | 2048 | 2048 | 4096 | 32768 | 32768 | 32768 | 87818.24 | 131072 | 45 | 45 | 0 | 5 | 13744.35556 | 21934.65376 |

分位数采用线性插值，分位数可能不是实际窗口；场景只取实际观测离散值。

| L_ctx | support_count | support_fraction | scenario_type | evidence_source | attention_to_training_ratio |
| --- | --- | --- | --- | --- | --- |
| 2048 | 18 | 0.4 | short | real_attachments/C_efficiency_evolution/model_architecture_metadata.csv | 0.06826666667 |
| 4096 | 8 | 0.1777777778 | medium_median | real_attachments/C_efficiency_evolution/model_architecture_metadata.csv | 0.1365333333 |
| 8192 | 7 | 0.1555555556 | observed_frequent | real_attachments/C_efficiency_evolution/model_architecture_metadata.csv | 0.2730666667 |
| 32768 | 11 | 0.2444444444 | observed_frequent | real_attachments/C_efficiency_evolution/model_architecture_metadata.csv | 1.092266667 |
| 131072 | 1 | 0.02222222222 | long | real_attachments/C_efficiency_evolution/model_architecture_metadata.csv | 4.369066667 |

eta=0.0002 是题给简化代理，不需要也没有用 C7 重新估计。单位为每参数、每训练 token、每窗口位置的 FLOPs 系数。临界窗口 30000 在观测最小最大值之间，但不是观测离散值；超临界模型 12。C7 频数描述本附件，不代表模型市场总体分布；最大支持窗口不证明所有训练序列都达到该长度。
