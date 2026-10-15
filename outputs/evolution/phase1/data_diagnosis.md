# Question 4 data diagnosis

| dataset | file | rows | columns | exact_duplicate_rows | duplicated_model_rows |
| --- | --- | --- | --- | --- | --- |
| C1 | leaderboard_cleaned.csv | 4576 | 12 | 0 | 79 |
| C2 | leaderboard_enhanced.csv | 4576 | 15 | 0 | 79 |
| C3 | leaderboard_extended_timeseries.csv | 4599 | 11 | 0 | 79 |
| C4 | epoch_all_ai_models.csv | 3523 | 57 | 0 | 5 |
| C5 | loss_benchmark_bridge.csv | 43 | 13 | 0 | 0 |
| C6 | loss_benchmark_bridge_expanded.csv | 75 | 13 | 0 | 0 |
| C7 | model_architecture_metadata.csv | 45 | 7 | 0 | 0 |

C2与C1核心逐行数值一致（比较容差1e-12，仅用于CSV舍入审计），但若干浮点并非逐位相等，最大差异见一致性表。C2新增Epoch_AI_Publication_Date, Epoch_AI_Organization, Epoch_AI_Open_Weights。这些列标为Epoch AI增强元数据，但没有Epoch模型ID或显式匹配模型名；不能据此静默恢复唯一C4记录。

全部字段及缺失/数值/日期范围见field_diagnosis.csv；重复模型名不等同于重复评测。原始parquet用于架构与原始得分溯源，不读取任何旧问题四拟合结果。

许可证只有标识符，没有逐模型授权条款。遵照本轮要求，不能仅据标识符确认研究/复现授权，统一license_unknown；没有自动做法律解释。

C7 training_data_TB与token数不同，禁止转换为D_tokens。C4的Dataset字段也不是统一token单位，只在notes明确标注token时提取D_tokens。