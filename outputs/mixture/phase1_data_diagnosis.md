# A4–A15 数据诊断（先于回归）

实际路径为 real_attachments/A_data_value/regmix_tables。读取十二个官方 CSV，原件只读；无回归、无 Q/K 引入。index 仅在同一个实验的 mixture/loss 对内关联，不能跨尺度直接按 index 判断配方相同。

| dataset | kind | rows | unique_index_count | numeric_columns | nan_count | inf_count | negative_count | provenance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_1M | mixture | 512 | 512 | 17 | 0 | 0 | 0 | observed |
| train_1M | loss | 512 | 512 | 13 | 0 | 0 | 0 | observed |
| test_1M | mixture | 256 | 256 | 17 | 0 | 0 | 0 | observed |
| test_1M | loss | 256 | 256 | 13 | 0 | 0 | 0 | observed |
| test_60M | mixture | 256 | 256 | 17 | 0 | 0 | 0 | observed |
| test_60M | loss | 256 | 256 | 13 | 0 | 0 | 0 | observed |
| test_1B | mixture | 64 | 64 | 17 | 0 | 0 | 0 | observed |
| test_1B | loss | 64 | 64 | 13 | 0 | 0 | 0 | observed |
| estimated_10B | mixture | 63 | 63 | 17 | 0 | 0 | 0 | estimated |
| estimated_10B | loss | 63 | 63 | 13 | 0 | 0 | 0 | estimated |
| estimated_70B | mixture | 63 | 63 | 17 | 0 | 0 | 0 | estimated |
| estimated_70B | loss | 63 | 63 | 13 | 0 | 0 | 0 | estimated |

## 17 个训练配比列

1. train_the_pile_arxiv
2. train_the_pile_freelaw
3. train_the_pile_nih_exporter
4. train_the_pile_pubmed_central
5. train_the_pile_wikipedia_en
6. train_the_pile_dm_mathematics
7. train_the_pile_github
8. train_the_pile_philpapers
9. train_the_pile_stackexchange
10. train_the_pile_enron_emails
11. train_the_pile_gutenberg_pg_19
12. train_the_pile_pile_cc
13. train_the_pile_ubuntu_irc
14. train_the_pile_europarl
15. train_the_pile_hackernews
16. train_the_pile_pubmed_abstracts
17. train_the_pile_uspto_backgrounds

## 13 个验证 Loss 列

1. metric/the_pile_arxiv_val_loss
2. metric/the_pile_freelaw_val_loss
3. metric/the_pile_pubmed_central_val_loss
4. metric/the_pile_wikipedia_en_val_loss
5. metric/the_pile_dm_mathematics_val_loss
6. metric/the_pile_github_val_loss
7. metric/the_pile_stackexchange_val_loss
8. metric/the_pile_gutenberg_pg_19_val_loss
9. metric/the_pile_pile_cc_val_loss
10. metric/the_pile_ubuntu_irc_val_loss
11. metric/the_pile_hackernews_val_loss
12. metric/the_pile_pubmed_abstracts_val_loss
13. metric/the_pile_uspto_backgrounds_val_loss

## 单纯形与零值

| dataset | mixture_n | one_to_one | raw_sum_min | raw_sum_max | max_abs_sum_error | zero_cell_rate | all_zero_rows | unique_recipe_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_1M | 512 | True | 0.996000 | 1.003000 | 0.004000 | 0.451287 | 0 | 512 |
| test_1M | 256 | True | 0.997000 | 1.003000 | 0.003000 | 0.429917 | 0 | 256 |
| test_60M | 256 | True | 0.997000 | 1.003000 | 0.003000 | 0.429917 | 0 | 256 |
| test_1B | 64 | True | 0.998000 | 1.002000 | 0.002000 | 0.330882 | 0 | 64 |
| estimated_10B | 63 | True | 0.998000 | 1.002000 | 0.002000 | 0.454715 | 0 | 63 |
| estimated_70B | 63 | True | 0.998000 | 1.002000 | 0.002000 | 0.454715 | 0 | 63 |

原始配比为有限小数导出，因此行和不完全为1。误差界由实际配比列数×0.5×10^(−文件最大小数位数) 推导；所有行均在该界内。后续仅对派生数据执行闭合 p/sum(p)，保持非零比例，绝不修改原件；不存在负值、NaN/Inf或全零配方。逐域分布和零比例见 mixture_domain_profile.csv、mixture_zero_summary.csv；逐行闭合幅度见 simplex_closure_diagnosis.csv。

十七域相关矩阵按数据集分别保存于 correlations/。原始比例相关受总和约束，不直接解释为领域互补或因果替代。各验证域 Loss 分布见 loss_domain_profile.csv。

## Loss 响应

主响应 L_mean=13 域等权均值。敏感性 L_trim=排序后去掉最低/最高各一个，再对11域取均值。另保存中位数与全部原13域 Loss，未按观察结果选择偏好的验证域。完整对齐响应见 loss_response_summary.csv。

## 配方重叠与跨尺度比较

| dataset_1 | dataset_2 | n_1 | n_2 | same_composition_count |
| --- | --- | --- | --- | --- |
| train_1M | test_1M | 512 | 256 | 0 |
| train_1M | test_60M | 512 | 256 | 0 |
| train_1M | test_1B | 512 | 64 | 0 |
| train_1M | estimated_10B | 512 | 63 | 63 |
| train_1M | estimated_70B | 512 | 63 | 63 |
| test_1M | test_60M | 256 | 256 | 256 |
| test_1M | test_1B | 256 | 64 | 0 |
| test_1M | estimated_10B | 256 | 63 | 0 |
| test_1M | estimated_70B | 256 | 63 | 0 |
| test_60M | test_1B | 256 | 64 | 0 |
| test_60M | estimated_10B | 256 | 63 | 0 |
| test_60M | estimated_70B | 256 | 63 | 0 |
| test_1B | estimated_10B | 64 | 63 | 0 |
| test_1B | estimated_70B | 64 | 63 | 0 |
| estimated_10B | estimated_70B | 63 | 63 | 63 |

相同配方使用闭合组成向量匹配，不使用跨文件 index。1M/60M 可以在共同配方上直接比较实际排序；若1B无共同配方，则只报告1M模型在1B上的排序迁移能力，不能虚构逐配方跨尺度一致性。

## 估算材料的范围

10B/70B Loss 只作为 estimated 数据诊断，绝非本轮观测检验或模型选择数据。本轮不预测这两组，不拟合模型规模或 Scaling Law；Q、K 也不进入当前模型。

结构核验通过后允许继续 ILR 与 M0/M1/M2。所有数据驱动预处理参数须在相应训练/CV折内拟合。
