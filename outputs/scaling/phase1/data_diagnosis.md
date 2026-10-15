# 问题二 PHASE 1 数据诊断（尚未拟合）



N_params_B、D_tokens_B 直接按 1B 单位输入模型，不乘 1e9。FLOPs 核验采用 6ND/1e21=0.006 N_B D_B。原始数据不填补、不删除、不覆盖。



| dataset | file_path | rows | columns | missing_or_nan | inf_count | exact_duplicate_excess_rows | duplicate_checkpoint_excess_rows | N_min | N_max | D_min | D_max | Loss_min | Loss_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | real_attachments/B_scaling_laws/pythia_training_log_existing.csv | 1176 | 15 | 0 | 0 | 0 | 0 | 0.070542 | 11.9658 | 0.134 | 299.893 | 2.0933 | 4.7388 |
| B2 | real_attachments/B_scaling_laws/cerebras_training_log.csv | 1029 | 15 | 3087 | 0 | 0 | 0 | 0.111 | 13 | 14 | 2050 | 2.8354 | 6.4979 |
| B4 | real_attachments/B_scaling_laws/scaling_baseline.csv | 57 | 5 | 0 | 0 | 0 | 不适用/未提供 | 0.07 | 72 | 6 | 33739 | 1.71 | 4.3629 |
| B5 | real_attachments/B_scaling_laws/published_scaling_data.csv | 44 | 6 | 0 | 0 | 0 | 不适用/未提供 | 0.125 | 540 | 5 | 2000 | 1.473 | 3.015 |
| B3 | real_attachments/B_scaling_laws/training_trajectories/pythia_0.070542B_trajectory.csv | 500 | 5 | 0 | 0 | 0 | 383 | 0.070542 | 0.070542 | 0.134 | 299.893 | 2.8127 | 4.7385 |
| B3 | real_attachments/B_scaling_laws/training_trajectories/pythia_0.162405B_trajectory.csv | 500 | 5 | 0 | 0 | 0 | 383 | 0.162405 | 0.162405 | 0.134 | 299.893 | 2.5959 | 4.5229 |
| B3 | real_attachments/B_scaling_laws/training_trajectories/pythia_0.409009B_trajectory.csv | 500 | 5 | 0 | 0 | 0 | 383 | 0.409009 | 0.409009 | 0.134 | 299.893 | 2.4196 | 4.3484 |
| B3 | real_attachments/B_scaling_laws/training_trajectories/pythia_1.040867B_trajectory.csv | 500 | 5 | 0 | 0 | 0 | 383 | 1.04087 | 1.04087 | 0.134 | 299.893 | 2.2923 | 4.2162 |
| B3 | real_attachments/B_scaling_laws/training_trajectories/pythia_1.416184B_trajectory.csv | 500 | 5 | 0 | 0 | 0 | 383 | 1.41618 | 1.41618 | 0.134 | 299.893 | 2.2523 | 4.1845 |
| B3 | real_attachments/B_scaling_laws/training_trajectories/pythia_11.965825B_trajectory.csv | 500 | 5 | 0 | 0 | 0 | 383 | 11.9658 | 11.9658 | 0.134 | 299.893 | 2.0936 | 4.0229 |
| B3 | real_attachments/B_scaling_laws/training_trajectories/pythia_2.782831B_trajectory.csv | 500 | 5 | 0 | 0 | 0 | 383 | 2.78283 | 2.78283 | 0.134 | 299.893 | 2.1897 | 4.1185 |
| B3 | real_attachments/B_scaling_laws/training_trajectories/pythia_6.86104B_trajectory.csv | 500 | 5 | 0 | 0 | 0 | 383 | 6.86104 | 6.86104 | 0.134 | 299.893 | 2.1224 | 4.0498 |



字段完整列表见 data_file_summary.csv；逐字段 NaN/空缺与 Inf 见 field_diagnosis.csv。B2 的可选元数据空缺不等于 N/D/Loss 缺失，未做任何填补。无 checkpoint 字段的 B4/B5 记“不适用”，不虚报重复 checkpoint 为零。



B1 实际 8 个规模、1176 条 checkpoint。run_id 有 1176 个唯一值，是 checkpoint 标识，不是模型轨迹 ID。轨迹按来源 B1 + 精确规范化 N_params_B 分组；本文件每个规模只有一条候选轨迹，无法识别未标注的同规模重复运行。



| trajectory_id | N_params_B | n_checkpoints | D_strictly_increasing | loss_rebound_count | D_max | matched_B12_nominal_size | B12_matching_checkpoint_steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1:N=0.070542 | 0.070542 | 147 | True | 0 | 299.893 | 70m | 147 |
| B1:N=0.162405 | 0.162405 | 147 | True | 0 | 299.893 | 160m | 147 |
| B1:N=0.409009 | 0.409009 | 147 | True | 0 | 299.893 | 410m | 147 |
| B1:N=1.040867 | 1.04087 | 147 | True | 0 | 299.893 | 1b | 147 |
| B1:N=1.416184 | 1.41618 | 147 | True | 0 | 299.893 | 1.4b | 147 |
| B1:N=11.965825 | 11.9658 | 147 | True | 0 | 299.893 | 12b | 147 |
| B1:N=2.782831 | 2.78283 | 147 | True | 0 | 299.893 | 2.8b | 147 |
| B1:N=6.86104 | 6.86104 | 147 | True | 0 | 299.893 | 6.9b | 147 |



B12 的名义规模用于辅助对照，参数计数不一定等于命名规模；多出的名义模型不补造 B1 数据。run_id 不同不能使同一轨迹 checkpoint 成为独立观测。Group CV 留出整条轨迹，bootstrap 以轨迹整体重采样。



B1 的 C 与 6ND：1176/1176 条在实际 CSV 导出精度容差内一致。相对误差在极小 C 处可能因四舍五入很大，不能只看相对值判异常。



B3 各文件含 interpolated 字段，只称插值轨迹一致性检验；B2 按本轮给定的半合成来源定义，仅作辅助敏感性。B4 跨模型族表未提供完整评测数据集/tokenizer 口径，数值误差只反映给定表上的一致性。



| source | n | families | absolute_loss_comparability |
| --- | --- | --- | --- |
| Chowdhery et al. 2022 | 3 | ['PaLM'] | unconfirmed; source-layer trend primary, pooled errors diagnostic only |
| Hoffmann et al. 2022 | 16 | ['Chinchilla', 'Gopher'] | unconfirmed; source-layer trend primary, pooled errors diagnostic only |
| Kaplan et al. 2020 | 9 | ['GPT-2', 'GPT-3'] | unconfirmed; source-layer trend primary, pooled errors diagnostic only |
| Scao et al. 2022 | 4 | ['BLOOM'] | unconfirmed; source-layer trend primary, pooled errors diagnostic only |
| Touvron et al. 2023 | 7 | ['LLaMA', 'LLaMA-2'] | unconfirmed; source-layer trend primary, pooled errors diagnostic only |
| Zhang et al. 2022 | 5 | ['OPT'] | unconfirmed; source-layer trend primary, pooled errors diagnostic only |



B5 有文献 source 和 family，没有 loss_type、dataset、tokenizer。跨文献绝对 Loss 可比性无法由附件确认，所以 source 内排序是核心文献证据，绝对误差只作分层描述，不当作统一口径精度。



末端 D 的实际取值：[299.893]。若所有模型的 terminal D 相同，则 E 与 B D_terminal^-beta 混淆，终点数据不能分别识别 E/B/beta；仍输出约束拟合对照，但不将 beta 的数值当作终点实证结论。



2478 个既有文件的哈希已建立并核验。只读取 B1-B5、B11/B12 作为数值/元数据输入，B6-B10 未用于拟合。

