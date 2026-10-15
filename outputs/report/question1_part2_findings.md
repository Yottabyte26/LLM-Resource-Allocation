# 问题一第2小问冲突诊断：实际发现与解释边界

本页只汇总已经生成的真实统计；未重新计算 CRITIC、修改预处理/Q 或构造 Q*。详细表、定义与九幅图索引见 `question1_part2_conflict_stage1.md`。

A1 n=51230，K 均值 0.261613，中位数 0.260928，标准差 0.055566，范围 [0.072923, 0.462123]。P90=0.333206，P95=0.353981，高冲突 5123 条。

## 领域与主要组合

| domain | n | high_conflict_rate | mean_K |
| --- | --- | --- | --- |
| book | 171 | 0.8888888888888888 | 0.3773015491437268 |
| github | 10000 | 0.2324 | 0.30149494013660877 |
| arxiv | 1419 | 0.14376321353065538 | 0.25283644335743577 |
| c4 | 10000 | 0.1175 | 0.2602240889418539 |
| wikipedia | 10000 | 0.0929 | 0.2817608662430283 |
| commoncrawl | 9640 | 0.02925311203319502 | 0.23747474275225794 |
| stackexchange | 10000 | 0.0057 | 0.22550851589442572 |

book 的高冲突率最高，但仅 171 条；不能根据描述排序宣称统计显著或代表整个书籍领域。stackexchange 与 commoncrawl 的高冲突率相对较低。

A1 Top10 指标对见下表：严重率指 P(|z_j-z_k|≥0.5)，按该比例降序。大量涉及 reasoning，但全局指标对差异与高 K 的极端驱动并非同一个统计量。

| rank | metric_1 | metric_2 | severe_pair_conflict_rate | pearson_r |
| --- | --- | --- | --- | --- |
| 1 | dsir_wiki | modernbert_reasoning | 0.8184071832910405 | -0.4231988077675179 |
| 2 | dsir_books | modernbert_reasoning | 0.8176068709740386 | -0.4288684159633733 |
| 3 | dsir_math | modernbert_reasoning | 0.7980089791138005 | -0.42212555446313815 |
| 4 | modernbert_reasoning | ad_en | 0.7731407378489167 | 0.10649503736071876 |
| 5 | modernbert_reasoning | rps_doc_word_count | 0.7712082764005466 | -0.014906958451330702 |
| 6 | modernbert_reasoning | rps_doc_num_sentences | 0.7631465937926996 | -0.038945769969713176 |
| 7 | modernbert_reasoning | rps_doc_mean_word_length | 0.7251024790162014 | 0.11559793056416297 |
| 8 | fineweb_edu | ad_en | 0.699121608432559 | 0.16021417609030505 |
| 9 | modernbert_reasoning | rps_lines_numerical_chars_fraction | 0.6925824712082764 | -0.06908583699407848 |
| 10 | qurater | ad_en | 0.6847940659769666 | 0.0503399482912283 |

| top_positive_metric | top_negative_metric | fractional_rate | possible_sample_count |
| --- | --- | --- | --- |
| ad_en | fluency_en | 0.07585040108936214 | 3086 |
| rps_doc_num_sentences | fluency_en | 0.060362097805246435 | 2356 |
| rps_doc_mean_word_length | fluency_en | 0.05382249146432348 | 2158 |
| rps_doc_word_count | fluency_en | 0.0509131709160986 | 1804 |
| rps_doc_frac_chars_top_3gram | fluency_en | 0.03524267042470968 | 1831 |
| ad_en | modernbert_reasoning | 0.03498401107811377 | 2054 |
| rps_doc_num_sentences | modernbert_reasoning | 0.02405172452239587 | 1297 |
| rps_doc_frac_chars_top_3gram | modernbert_reasoning | 0.023422627464221954 | 1542 |
| rps_lines_ending_with_terminal_punctuation_mark | fluency_en | 0.0233864021783698 | 869 |
| rps_doc_frac_chars_top_2gram | fluency_en | 0.02163593577952137 | 1277 |

高冲突组 fluency_en 平均值 0.320906，普通组 0.834298。主要极端组合表现为无广告/长度效用等高分与英语流畅低分、推理低分并存。并列值均分贡献后的比例不能当成原始唯一记录比例；possible_sample_count 可以重叠。

数据支持‘评价构念不同、格式规则与内容评价不同、代码/非英语文本与英语分类适用范围不同’作为待核验机制；不将这些解释作为已经证实的因果成因。

## 扩展集复现

| domain | view | high_rate_A1 | high_rate_extended | KS_statistic | Wasserstein_distance | cliffs_delta_extended_vs_A1 |
| --- | --- | --- | --- | --- | --- | --- |
| arxiv | full | 0.14376321353065538 | 0.16464075786109683 | 0.04741940492827368 | 0.004653592033465104 | 0.04130916310656161 |
| arxiv | nonoverlap | 0.14376321353065538 | 0.16648037754595132 | 0.05159775413301915 | 0.00506364215116797 | 0.044949109855705355 |
| github | full | 0.2324 | 0.2309866897012054 | 0.012085445050846133 | 0.0004966372040658053 | 0.0054582914523538415 |
| github | nonoverlap | 0.2324 | 0.23091374540649903 | 0.01270920351789917 | 0.0005222698274227666 | 0.0057400068128329 |

| domain | view | pair_rate_pearson | pair_rate_spearman | top10_pair_overlap_n | driver_rate_pearson | top10_driver_overlap_n |
| --- | --- | --- | --- | --- | --- | --- |
| arxiv | full | 0.9984894064741553 | 0.9959660799212173 | 10 | 0.9990825470208378 | 9 |
| arxiv | nonoverlap | 0.998219396371578 | 0.995454192701415 | 10 | 0.9989382592933773 | 9 |
| github | full | 0.9999235918008994 | 0.9998396089869337 | 10 | 0.9996605719715282 | 10 |
| github | nonoverlap | 0.9999154760207367 | 0.9998293867979199 | 10 | 0.9996239970323699 | 10 |

非重叠视图仍保留很高的指标对/驱动模式一致性；github 的分布差异更小，arxiv 的高冲突率略上升。这支持主要冲突结构复现，不能宣称分布完全相同，也不能推广到附件以外的数据。

## 评分敏感性与不能跳过的控制结果

Q 与 K：Pearson=-0.880584，Spearman=-0.896023。K 与 ΔQ：Pearson=0.612246，Spearman=0.610678。

高冲突组 mean ΔQ=0.038559，普通组 0.018831，比值 2.0476，Cliff's delta=0.575529。这是总体赋权敏感性差异，尚非质量误差证据。

但高冲突组内部 Pearson(K,ΔQ) 仅 0.052626；Q 最低四分位内部仅 0.077023，Spearman=0.099538。在该 Q 层中高/普通组 ΔQ 均值比降至 1.0582、Cliff's delta=0.084711。总体差异不能直接解释为冲突的独立影响。

Q、K、ΔQ 都是同一组 Z 的函数，Q/K 还受到 K≤2Q(1−Q) 的数学约束。需要人工文本核验和独立质量标签才能评估真实错误/收益。目前仅确认内部意见差异与一种权重敏感性，不足以据此自动确定惩罚系数或选择 Q*。

## 文本材料与初步比较（非正式证据）

完整文本材料包含最大 K 样本、高冲突组内相对高/中/低 Q 样本及全局四象限控制样本，组间可复用，共 79 条唯一记录。实际选取数量见 text_selection_summary.csv；组内相对高 Q 材料不得称为全局高 Q。

另对其中 12 条开头片段记录 AI 辅助复核，见 `A1_conflict_text_AI_review.csv`。明确标记**AI辅助复核，非最终人工标注**，不把片段观察作为全文质量或正式实验结论。控制组可见新闻/百科材料，高 Q+高 K 可见商品文案和带公式论文，低 Q+高 K 可见重复关键词和短代码；这些是选定案例，不是各组总体占比。

原始全文有长字段，Python 读取文本 CSV 时应先设置 `csv.field_size_limit(2**31-1)`。完整输出未截断 content，主冲突 CSV 未写入 content。

## 停止点

独立核验结果见 `conflict_verification.csv`。第一小问/官方附件的 2141 个文件 SHA-256 均保持一致。当前诊断阶段已完成；未构造 Q*、未测试 lambda，等待人工确认后再决定采用 Q+K 还是修正评分。
