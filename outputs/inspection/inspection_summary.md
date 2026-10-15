# PHASE 1 + PHASE 2：真实数据结构检查

本报告只含结构检查；未标量化、填补、截尾、归一化、计算 CRITIC 或 Q。

## 实际文件和记录数

| 数据集 | 实际路径 | 记录数 | JSON 解析失败 | 原文件哈希一致 |
|---|---|---:|---:|---|
| A1 | `real_attachments/A_data_value/slimpajama_quality_signal_sample.jsonl.xz` | 51230 | 0 | True |
| A2 | `real_attachments/A_data_value/slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz` | 17523 | 0 | True |
| A3 | `real_attachments/A_data_value/slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz` | 203752 | 0 | True |

共读取 272,505 条 JSON 对象。
目录与用户提供的文件规则一致。A2/A3 的全量指磁盘已提供的全部匹配文件。

## 八个列表字段

| 数据集 | 字段 | 长度分布 | 有效数 | 缺失率 | 非法数 | 解析失败 | sum≈1 比例 | 概率诊断 |
|---|---|---|---:|---:|---:|---:|---:|---|
| A1 | fineweb_edu | {"1": 51230} | 51230 | 0.000000% | 0 | 0 | 0.183486% | False |
| A1 | qurater | {"4": 51230} | 51230 | 0.000000% | 0 | 0 | 0.007808% | False |
| A1 | modernbert_cleanliness | {"6": 51230} | 51230 | 0.000000% | 0 | 0 | 0.003904% | False |
| A1 | modernbert_readability | {"6": 51230} | 51230 | 0.000000% | 0 | 0 | 0.019520% | False |
| A1 | modernbert_reasoning | {"6": 51230} | 51217 | 0.000000% | 13 | 0 | 0.064432% | False |
| A1 | modernbert_professionalism | {"6": 51230} | 51225 | 0.000000% | 5 | 0 | 0.011713% | False |
| A1 | fluency_en | {"2": 51230} | 51230 | 0.000000% | 0 | 0 | 0.017568% | False |
| A1 | ad_en | {"2": 51230} | 51230 | 0.000000% | 0 | 0 | 0.011712% | False |
| A2 | fineweb_edu | {"1": 17523} | 17523 | 0.000000% | 0 | 0 | 0.039947% | False |
| A2 | qurater | {"4": 17523} | 17523 | 0.000000% | 0 | 0 | 0.000000% | False |
| A2 | modernbert_cleanliness | {"6": 17523} | 17523 | 0.000000% | 0 | 0 | 0.000000% | False |
| A2 | modernbert_readability | {"6": 17523} | 17523 | 0.000000% | 0 | 0 | 0.000000% | False |
| A2 | modernbert_reasoning | {"6": 17523} | 17523 | 0.000000% | 0 | 0 | 0.000000% | False |
| A2 | modernbert_professionalism | {"6": 17523} | 17523 | 0.000000% | 0 | 0 | 0.000000% | False |
| A2 | fluency_en | {"2": 17523} | 17523 | 0.000000% | 0 | 0 | 0.000000% | False |
| A2 | ad_en | {"2": 17523} | 17523 | 0.000000% | 0 | 0 | 0.000000% | False |
| A3 | fineweb_edu | {"1": 203752} | 203752 | 0.000000% | 0 | 0 | 0.203679% | False |
| A3 | qurater | {"4": 203752} | 203752 | 0.000000% | 0 | 0 | 0.017669% | False |
| A3 | modernbert_cleanliness | {"6": 203752} | 203752 | 0.000000% | 0 | 0 | 0.016687% | False |
| A3 | modernbert_readability | {"6": 203752} | 203752 | 0.000000% | 0 | 0 | 0.045153% | False |
| A3 | modernbert_reasoning | {"6": 203752} | 203752 | 0.000000% | 0 | 0 | 0.076073% | False |
| A3 | modernbert_professionalism | {"6": 203752} | 203751 | 0.000000% | 1 | 0 | 0.009816% | False |
| A3 | fluency_en | {"2": 203752} | 203752 | 0.000000% | 0 | 0 | 0.000982% | False |
| A3 | ad_en | {"2": 203752} | 203752 | 0.000000% | 0 | 0 | 0.022576% | False |

缺失=字段不存在+None+空列表；非法值另计。sum≈1 的分母为有效有限数值列表数，
条件为 abs(sum−1)<0.001。NaN 按非法数值记录；Python JSON 解码器可读取 NaN，
所以这类值不计作 JSON 语法解析失败。

完整前 20 个真实非空值及非法样例见 `list_field_samples.txt`；
每个位置的 min/max/mean/median 见 `list_field_position_summary.csv`；
向量和的 min/P25/median/mean/P75/max 见 `list_field_summary.csv`。

## 22 指标与辅助字段核验

| 数据集 | 规范字段 | 缺失字段数 | None | 非法数 | 标量最小值 | 标量最大值 |
|---|---|---:|---:|---:|---:|---:|
| A1 | fineweb_edu | 0 | 0 | 0 |  |  |
| A1 | dsir_books | 0 | 0 | 0 | -1242823.60370116 | -24.02237722 |
| A1 | dsir_wiki | 0 | 0 | 0 | -1561611.03426511 | -25.0120389 |
| A1 | dsir_math | 0 | 0 | 0 | -1668348.05111975 | -22.86740911 |
| A1 | qurater | 0 | 0 | 0 |  |  |
| A1 | modernbert_cleanliness | 0 | 0 | 0 |  |  |
| A1 | modernbert_readability | 0 | 0 | 0 |  |  |
| A1 | modernbert_reasoning | 0 | 0 | 13 |  |  |
| A1 | modernbert_professionalism | 0 | 0 | 5 |  |  |
| A1 | fluency_en | 0 | 0 | 0 |  |  |
| A1 | ad_en | 0 | 0 | 0 |  |  |
| A1 | rps_doc_word_count | 0 | 0 | 0 | 1.0 | 552671.0 |
| A1 | rps_doc_num_sentences | 0 | 0 | 0 | 1 | 25078 |
| A1 | rps_doc_unigram_entropy | 0 | 0 | 0 | 0.0 | 7.84673154 |
| A1 | rps_doc_frac_unique_words | 0 | 0 | 0 | 5.088363331179848 | 100.0 |
| A1 | rps_doc_frac_no_alph_words | 0 | 0 | 0 | 4.347826086956516 | 100.0 |
| A1 | rps_doc_frac_chars_top_2gram | 0 | 0 | 0 | 0.0 | 135.13338722716247 |
| A1 | rps_doc_frac_chars_top_3gram | 0 | 0 | 0 | 0.0 | 201.6814874696847 |
| A1 | rps_lines_uppercase_letter_fraction | 0 | 0 | 0 | 0.0 | 86.95652173913044 |
| A1 | rps_lines_ending_with_terminal_punctuation_mark | 0 | 0 | 0 | 0.0 | 100.0 |
| A1 | rps_lines_numerical_chars_fraction | 0 | 0 | 0 | 0.0 | 72.34424037417816 |
| A1 | rps_doc_mean_word_length | 0 | 0 | 0 | 1.9272030651340997 | 83.0 |
| A1 | id | 0 | 0 | 0 |  |  |
| A1 | content | 0 | 0 | 0 |  |  |
| A1 | sub_path | 0 | 0 | 0 |  |  |
| A1 | _source_domain | 0 | 0 | 0 |  |  |
| A1 | _source_path | 0 | 0 | 0 |  |  |
| A2 | fineweb_edu | 0 | 0 | 0 |  |  |
| A2 | dsir_books | 0 | 0 | 0 | -2026325.80102772 | -316.68028944 |
| A2 | dsir_wiki | 0 | 0 | 0 | -1966147.71309439 | -313.93355077 |
| A2 | dsir_math | 0 | 0 | 0 | -1508740.10117446 | -306.94026452 |
| A2 | qurater | 0 | 0 | 0 |  |  |
| A2 | modernbert_cleanliness | 0 | 0 | 0 |  |  |
| A2 | modernbert_readability | 0 | 0 | 0 |  |  |
| A2 | modernbert_reasoning | 0 | 0 | 0 |  |  |
| A2 | modernbert_professionalism | 0 | 0 | 0 |  |  |
| A2 | fluency_en | 0 | 0 | 0 |  |  |
| A2 | ad_en | 0 | 0 | 0 |  |  |
| A2 | rps_doc_word_count | 0 | 0 | 0 | 10.0 | 217176.0 |
| A2 | rps_doc_num_sentences | 0 | 0 | 0 | 1 | 30449 |
| A2 | rps_doc_unigram_entropy | 0 | 0 | 0 | 2.30258509 | 7.80240899 |
| A2 | rps_doc_frac_unique_words | 0 | 0 | 0 | 2.720757022773438 | 100.0 |
| A2 | rps_doc_frac_no_alph_words | 0 | 0 | 0 | 15.985130111524157 | 91.47592509058103 |
| A2 | rps_doc_frac_chars_top_2gram | 0 | 0 | 0 | 0.0 | 21.883763659220577 |
| A2 | rps_doc_frac_chars_top_3gram | 0 | 0 | 0 | 0.0 | 35.37375495600038 |
| A2 | rps_lines_uppercase_letter_fraction | 0 | 0 | 0 | 0.5336562412395783 | 87.8054690204223 |
| A2 | rps_lines_ending_with_terminal_punctuation_mark | 0 | 0 | 0 | 0.20883054892601433 | 100.0 |
| A2 | rps_lines_numerical_chars_fraction | 0 | 0 | 0 | 0.0 | 85.67810452791292 |
| A2 | rps_doc_mean_word_length | 0 | 0 | 0 | 2.986633821449751 | 39.0 |
| A2 | id | 0 | 0 | 0 |  |  |
| A2 | content | 17523 | 0 | 0 |  |  |
| A2 | sub_path | 0 | 0 | 0 |  |  |
| A2 | _source_domain | 17523 | 0 | 0 |  |  |
| A2 | _source_path | 17523 | 0 | 0 |  |  |
| A3 | fineweb_edu | 0 | 0 | 0 |  |  |
| A3 | dsir_books | 0 | 0 | 0 | -1875504.79852465 | -16.82294988 |
| A3 | dsir_wiki | 0 | 0 | 0 | -1906833.36424408 | -17.66218046 |
| A3 | dsir_math | 0 | 0 | 0 | -1754034.28491041 | -15.05893869 |
| A3 | qurater | 0 | 0 | 0 |  |  |
| A3 | modernbert_cleanliness | 0 | 0 | 0 |  |  |
| A3 | modernbert_readability | 0 | 0 | 0 |  |  |
| A3 | modernbert_reasoning | 0 | 0 | 0 |  |  |
| A3 | modernbert_professionalism | 0 | 0 | 1 |  |  |
| A3 | fluency_en | 0 | 0 | 0 |  |  |
| A3 | ad_en | 0 | 0 | 0 |  |  |
| A3 | rps_doc_word_count | 0 | 0 | 0 | 1.0 | 93541.0 |
| A3 | rps_doc_num_sentences | 0 | 0 | 0 | 1 | 26727 |
| A3 | rps_doc_unigram_entropy | 0 | 0 | 0 | 0.0 | 7.40034161 |
| A3 | rps_doc_frac_unique_words | 0 | 0 | 0 | 1.2174256383258752 | 100.0 |
| A3 | rps_doc_frac_no_alph_words | 0 | 0 | 0 | 4.347826086956516 | 99.58013873676525 |
| A3 | rps_doc_frac_chars_top_2gram | 0 | 0 | 0 | 0 | 194 |
| A3 | rps_doc_frac_chars_top_3gram | 0 | 0 | 0 | 0 | 284 |
| A3 | rps_lines_uppercase_letter_fraction | 0 | 0 | 0 | 0.0 | 96.49122807017544 |
| A3 | rps_lines_ending_with_terminal_punctuation_mark | 0 | 0 | 0 | 0.0 | 100.0 |
| A3 | rps_lines_numerical_chars_fraction | 0 | 0 | 0 | 0.0 | 63.01441377528455 |
| A3 | rps_doc_mean_word_length | 0 | 0 | 0 | 2.5 | 87.0 |
| A3 | id | 0 | 0 | 0 |  |  |
| A3 | content | 203752 | 0 | 0 |  |  |
| A3 | sub_path | 0 | 0 | 0 |  |  |
| A3 | _source_domain | 203752 | 0 | 0 |  |  |
| A3 | _source_path | 203752 | 0 | 0 |  |  |

真实终止标点字段拼作 `rps_lines_ending_with_terminal_punctution_mark`，
通过配置中的显式别名对应官方规范名；不修改官方文件，也不增加指标维数。

## 待确认的含义与处理规则

- fineweb_edu：若长度检查继续确认始终为 1，结构上支持取唯一分量；原值不是 [0,1] 概率。
- qurater：长度 4 不能证明四项同尺度、同方向或独立。需确认模型输出和四个位置的定义后，才能决定均值是否适用。
- modernbert 四项：原向量含负数且和不接近 1，不能直接代入概率期望。疑似 logits 是统计推测，需要模型定义证实；六个位置的等级顺序也需确认。
- fluency_en/ad_en：不能直接取某个分量作概率。需确认是否为 logits，以及两个位置对应的实际类别。未经确认不执行 softmax 或反向。
- RPS 原值中的若干 fraction 指标达到 100，重复 n-gram 字段还可超过 100。字段名称不能证明 [0,1] 或统一百分比口径；当前仅记录真实范围，不剪裁、不换算。

下一步先人工确认以上结构和语义，再进入 PHASE 4。当前没有生成任何质量权重或评分。
