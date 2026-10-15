# PHASE 3：22 维标量指标描述性统计

已按人工确认规则完成标量化，仅做描述性统计。未填补缺失，未对其他连续指标截尾，
未构造统一方向的 Z，未运行 03/04，未计算 CRITIC 或 Q。

## 数据与口径

| 数据集 | 实际文件 | 全部记录数 | 输入哈希一致 |
|---|---|---:|---|
| A1 | `real_attachments/A_data_value/slimpajama_quality_signal_sample.jsonl.xz` | 51230 | True |
| A2 | `real_attachments/A_data_value/slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz` | 17523 | True |
| A3 | `real_attachments/A_data_value/slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz` | 203752 | True |

合计 272,505 条，严格保留 22 个指标，不按 ID 去重。content 未累计或写入结果。
A1 采用原领域；A2/A3 由文件名前缀推断。总体与合并领域统计使用全部记录；
同时保留数据集内领域统计用于区分抽样集与扩展集，不应把重叠记录视作独立实验。

count 是有限非缺失数，n 是该分组全部记录数，missing=n−count，std 使用 ddof=1。
分位数采用线性插值。空组不生成；指标全缺失时统计值留空，不填 0。

## 正式标量化

- fineweb_edu：取唯一分量，保留原值。
- modernbert 四项：argmax(logits)/5，取值为 {0,0.2,0.4,0.6,0.8,1}。
- fluency_en：argmax([not_fluent,fluent])，0/1 正向。
- ad_en：argmax([has_ad,no_ad])，0/1 正向；未反向。
- qurater：四个子维度分别 P01-P99 截尾、独立 Min-Max，之后平均；最终只保留一个指标。
- 分类向量含 NaN/Inf 时该标量为 NaN，不删除整条记录。
- 并列 argmax 取最小索引；Softmax 期望辅助函数保留但主流程不调用。

方向已更新为 14 正向、5 负向、3 适中；当前只记录方向，不执行反向或适中效用函数。

## qurater 统一拟合参数

参数使用 A1/A2/A3 全部有限子维度记录合并拟合。A1 若与扩展文件出现相同 ID，仍按原始记录保留。
未分别对数据集或领域拟合，保证相同原值在各组中得到同一转换。

| 子维度 | 有效数 | P01 | P99 | 截尾后 min | 截尾后 max | 下侧截尾数 | 上侧截尾数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Writing Style | 272505 | -5.3046875 | 4.5546875 | -5.3046875 | 4.5546875 | 2719 | 2718 |
| Required Expertise | 272505 | -3.908203125 | 9 | -3.908203125 | 9 | 2718 | 2725 |
| Facts and Trivia | 272505 | -4.296875 | 4.09765625 | -4.296875 | 4.09765625 | 2705 | 2722 |
| Educational Value | 272505 | -4.12890625 | 7.9296875 | -4.12890625 | 7.9296875 | 2721 | 2720 |

若某子维度常数，配置将其有效值记为 0.5 以避免除零；实际是否触发见参数表，不推断额外区分度。

## 总体 22 指标概览

| 指标 | count | missing | missing_rate | mean | std | min | P50 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fineweb_edu | 272505 | 0 | 0.00000000% | 1.237804417 | 0.5680122511 | -0.652137816 | 1.181002736 | 4.694155216 |
| dsir_books | 272505 | 0 | 0.00000000% | -8248.158574 | 25753.94802 | -2026325.801 | -2395.400116 | -16.82294988 |
| dsir_wiki | 272505 | 0 | 0.00000000% | -8252.028347 | 25813.47507 | -1966147.713 | -2410.213233 | -17.66218046 |
| dsir_math | 272505 | 0 | 0.00000000% | -6737.716628 | 20205.09596 | -1754034.285 | -2302.117697 | -15.05893869 |
| qurater | 272505 | 0 | 0.00000000% | 0.3839056988 | 0.1763166787 | 0 | 0.3598615998 | 1 |
| modernbert_cleanliness | 272505 | 0 | 0.00000000% | 0.5036619512 | 0.2739539607 | 0.2 | 0.4 | 1 |
| modernbert_readability | 272505 | 0 | 0.00000000% | 0.6366877672 | 0.2920826145 | 0 | 0.8 | 1 |
| modernbert_reasoning | 272492 | 13 | 0.00477055% | 0.3002326674 | 0.3017140303 | 0 | 0.2 | 1 |
| modernbert_professionalism | 272499 | 6 | 0.00220179% | 0.8103237076 | 0.2391037676 | 0.2 | 0.8 | 1 |
| fluency_en | 272505 | 0 | 0.00000000% | 0.6283334251 | 0.4832508551 | 0 | 1 | 1 |
| ad_en | 272505 | 0 | 0.00000000% | 0.9868589567 | 0.1138789 | 0 | 1 | 1 |
| rps_doc_word_count | 272505 | 0 | 0.00000000% | 891.7652704 | 3733.465109 | 1 | 152 | 552671 |
| rps_doc_num_sentences | 272505 | 0 | 0.00000000% | 96.59773215 | 346.1629896 | 1 | 24 | 30449 |
| rps_doc_unigram_entropy | 272505 | 0 | 0.00000000% | 4.108856808 | 1.07843965 | 0 | 4.13445314 | 7.84673154 |
| rps_doc_frac_unique_words | 272505 | 0 | 0.00000000% | 57.20038613 | 20.75385949 | 1.217425638 | 56.89655172 | 100 |
| rps_doc_frac_no_alph_words | 272505 | 0 | 0.00000000% | 33.716198 | 10.12215061 | 4.347826087 | 33.16261204 | 100 |
| rps_doc_frac_chars_top_2gram | 272505 | 0 | 0.00000000% | 3.605443106 | 5.045856694 | 0 | 2 | 194 |
| rps_doc_frac_chars_top_3gram | 272505 | 0 | 0.00000000% | 2.979328641 | 5.056313388 | 0 | 1 | 284 |
| rps_lines_uppercase_letter_fraction | 272505 | 0 | 0.00000000% | 22.13410027 | 12.3183163 | 0 | 20.76840611 | 96.49122807 |
| rps_lines_ending_with_terminal_punctuation_mark | 272505 | 0 | 0.00000000% | 30.89317635 | 21.82184323 | 0 | 25 | 100 |
| rps_lines_numerical_chars_fraction | 272505 | 0 | 0.00000000% | 1.858027778 | 3.10076668 | 0 | 0.8166723873 | 85.67810453 |
| rps_doc_mean_word_length | 272505 | 0 | 0.00000000% | 8.824037357 | 3.812932747 | 1.927203065 | 8.056603774 | 87 |

## 领域规模

| scope | dataset | domain | n |
|---|---|---|---:|
| domain | ALL | arxiv | 18942 |
| dataset_domain | A1 | arxiv | 1419 |
| dataset_domain | A2 | arxiv | 17523 |
| domain | ALL | book | 171 |
| dataset_domain | A1 | book | 171 |
| domain | ALL | c4 | 10000 |
| dataset_domain | A1 | c4 | 10000 |
| domain | ALL | commoncrawl | 9640 |
| dataset_domain | A1 | commoncrawl | 9640 |
| domain | ALL | github | 213752 |
| dataset_domain | A1 | github | 10000 |
| dataset_domain | A3 | github | 203752 |
| domain | ALL | stackexchange | 10000 |
| dataset_domain | A1 | stackexchange | 10000 |
| domain | ALL | wikipedia | 10000 |
| dataset_domain | A1 | wikipedia | 10000 |

## 缺失审计与下一阶段边界

实际 19 个标量缺失事件；涉及 19 条来源记录。
所有事件保留 id、dataset、domain、source_path、line_number、metric、reason。
scalar_missing_records.csv 中 imputation_performed=False；后续按领域中位数优先、总体中位数回退，
到实际填补时再另行记录填补值、方法及数量。本轮没有删除样本或填补缺失。

## 输出

- quality_indicator_profile.csv：总体与 A1/A2-arxiv/A3-github，每组 22 行。
- domain_indicator_profile.csv：scope=domain 为全数据合并领域；scope=dataset_domain 为数据集内领域。
- missing_summary.csv：对应所有分组的缺失汇总；分组有交集，不能把整表缺失数相加当作独立事件数。
- scalar_missing_records.csv：原始缺失事件及真实 ID，不包含人为的 qurater 拟合前占位。
- group_record_counts.csv / profile_input_manifest.csv / scalar_validation.csv：规模、哈希、22 维和取值核验。
- ../preprocess/qurater_transform_parameters.csv：四个子维度实际阈值和截尾后范围。

统计描述的是转换后的指标分布，不是文本综合质量或领域质量结论。RPS 原始数值口径保留，
DSIR 负值保留；本轮不执行连续指标的全局截尾或统一 [0,1] 归一化。
