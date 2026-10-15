# PHASE 3：A1 标定的统一 22 维质量向量 Z

本阶段完成原始标量→缺失填补→异常截尾→方向统一→[0,1] 标准化。未运行 04，未计算 CRITIC 或 Q。

## 标定范围与历史结果

A1 是七领域主样本。全部参数只用 A1 拟合，A2/A3 仅应用保存的参数；这使抽样集与扩展集共享质量尺度，
并避免扩展集规模与分布反过来改变主样本的评价尺度。

原 outputs/profile/ 全部文件和 qurater_transform_parameters.csv 保留且哈希不变，角色是
**pooled descriptive / sensitivity analysis**，不是主模型最终参数。角色另存 historical_artifact_roles.csv。

不对任一数据集或跨数据集去重；不建立 pooled deduplicated view。A3 不同 ID 的质量字段相同记录两条都保留，
因为缺少 content 无法认定文本重复。跨附件重叠记录仍存在；非重叠索引仅用于以后比较，不从 Z 删除记录。

## 实际输入与规模

| dataset | source_file | n | source_unchanged |
|---|---|---|---|
| A1 | real_attachments/A_data_value/slimpajama_quality_signal_sample.jsonl.xz | 51230 | True |
| A2 | real_attachments/A_data_value/slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz | 17523 | True |
| A3 | real_attachments/A_data_value/slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz | 203752 | True |

## 22 个指标的正式转换规则

fineweb_edu 取唯一分量；qurater 四维使用 A1 各自 P01/P99 截尾、Min-Max 后平均；
modernbert 四项 argmax/5；fluency_en argmax([not_fluent,fluent])；ad_en argmax([has_ad,no_ad])，1 为无广告。
后三类已在 [0,1]，包含 qurater 在内的七个指标不再次 Min-Max，不再整体截尾。
Softmax 期望只保留为辅助函数，本模型未调用。并列 argmax 取最小索引；非有限向量先作为缺失处理。

其余 15 个连续指标（fineweb、DSIR、RPS，包括连续比例统计）使用 A1 P01/P99 Winsorization。
这些 RPS 比例是统计特征，不是分类概率；分类概率/评级不参与连续截尾。
其中 12 个按方向 Min-Max；三个适中指标使用梯形效用。所有指标方向只在 YAML 中定义。

| metric | direction | scalarization | final_transform |
|---|---|---|---|
| fineweb_edu | positive | single_value | A1 Winsor + directional Min-Max |
| dsir_books | positive | raw_scalar | A1 Winsor + directional Min-Max |
| dsir_wiki | positive | raw_scalar | A1 Winsor + directional Min-Max |
| dsir_math | positive | raw_scalar | A1 Winsor + directional Min-Max |
| qurater | positive | component_winsor_minmax_mean | bounded identity (no second Min-Max) |
| modernbert_cleanliness | positive | argmax_scaled | bounded identity (no second Min-Max) |
| modernbert_readability | positive | argmax_scaled | bounded identity (no second Min-Max) |
| modernbert_reasoning | positive | argmax_scaled | bounded identity (no second Min-Max) |
| modernbert_professionalism | positive | argmax_scaled | bounded identity (no second Min-Max) |
| fluency_en | positive | argmax_index | bounded identity (no second Min-Max) |
| ad_en | positive | argmax_index | bounded identity (no second Min-Max) |
| rps_doc_word_count | moderate | raw_scalar | A1 Winsor + trapezoid |
| rps_doc_num_sentences | moderate | raw_scalar | A1 Winsor + trapezoid |
| rps_doc_unigram_entropy | positive | raw_scalar | A1 Winsor + directional Min-Max |
| rps_doc_frac_unique_words | positive | raw_scalar | A1 Winsor + directional Min-Max |
| rps_doc_frac_no_alph_words | negative | raw_scalar | A1 Winsor + directional Min-Max |
| rps_doc_frac_chars_top_2gram | negative | raw_scalar | A1 Winsor + directional Min-Max |
| rps_doc_frac_chars_top_3gram | negative | raw_scalar | A1 Winsor + directional Min-Max |
| rps_lines_uppercase_letter_fraction | negative | raw_scalar | A1 Winsor + directional Min-Max |
| rps_lines_ending_with_terminal_punctuation_mark | positive | raw_scalar | A1 Winsor + directional Min-Max |
| rps_lines_numerical_chars_fraction | negative | raw_scalar | A1 Winsor + directional Min-Max |
| rps_doc_mean_word_length | moderate | raw_scalar | A1 Winsor + trapezoid |

正向 14、负向 5、适中 3，严格为 22 维。辅助 ID、dataset、domain、来源、imputed_count 不参与向量。

## 缺失填补与记录保留

实际填补 19 个标量，涉及 19 条记录。
中位数由 A1 有限观测拟合；优先 A1 同领域同指标，无法取得时回退 A1 全体该指标。
A2/A3 不使用自身中位数。填充值、来源领域、ID、行号逐项记录于 imputation_log.csv。
这些记录只有少量指标缺失，保留样本与其他已知指标，不删除整行；高缺失记录单独留档而不自动删除。

| dataset | domain | id | metric | imputed_value | imputation_method |
|---|---|---|---|---|---|
| A1 | commoncrawl | BkiUdKI4eILhQJnKqVYf | modernbert_professionalism | 0.4 | A1_domain_median |
| A1 | commoncrawl | BkiUeN3xK6wB9mpb3Bjn | modernbert_professionalism | 0.4 | A1_domain_median |
| A1 | commoncrawl | BkiUecLxK7IACwaaAoRW | modernbert_professionalism | 0.4 | A1_domain_median |
| A1 | commoncrawl | BkiUcp3xK6Ot9TMSqSKV | modernbert_professionalism | 0.4 | A1_domain_median |
| A1 | wikipedia | BkiUdSQ5qU2Ap3mnjRLV | modernbert_reasoning | 0.2 | A1_domain_median |
| A1 | wikipedia | BkiUcNvxK4tBVgEFUGsD | modernbert_reasoning | 0.2 | A1_domain_median |
| A1 | wikipedia | BkiUdnI5qWTD6lXPdKWn | modernbert_reasoning | 0.2 | A1_domain_median |
| A1 | wikipedia | BkiUd7I4eIXh4qTE6B0_ | modernbert_reasoning | 0.2 | A1_domain_median |
| A1 | wikipedia | BkiUco_xK7ICUuXekmA1 | modernbert_reasoning | 0.2 | A1_domain_median |
| A1 | wikipedia | BkiUe8jxK1yAga6JtoEE | modernbert_reasoning | 0.2 | A1_domain_median |
| A1 | wikipedia | BkiUdcU4ubnhBx1Jqx5c | modernbert_reasoning | 0.2 | A1_domain_median |
| A1 | wikipedia | BkiUbBg241xg-EsBtW7L | modernbert_reasoning | 0.2 | A1_domain_median |
| A1 | wikipedia | BkiUdIM5jDKDyKYv5QE8 | modernbert_reasoning | 0.2 | A1_domain_median |
| A1 | wikipedia | BkiUcCbxK1ThhCdy7RAd | modernbert_reasoning | 0.2 | A1_domain_median |
| A1 | wikipedia | BkiUeYzxK7IAHYL5-XVI | modernbert_professionalism | 0.4 | A1_domain_median |
| A1 | wikipedia | BkiUfZQ5qhDCd8S7ey87 | modernbert_reasoning | 0.2 | A1_domain_median |
| A1 | wikipedia | BkiUc9A5qhLA3caTagV_ | modernbert_reasoning | 0.2 | A1_domain_median |
| A1 | wikipedia | BkiUbO7xK6-gDz87OFCT | modernbert_reasoning | 0.2 | A1_domain_median |
| A3 | github | BkiUdg_xK3YB9m7_-uDT | modernbert_professionalism | 0.8 | A1_domain_median |

## A1 完整拟合参数

preprocessing_model_A1.json 保存全部参数、别名、指标顺序、官方类别顺序及配置。
分位数使用 numpy 线性插值；顺序为标量化→用观测拟合中位数→填补→计算连续分位数。
适中型各分位数取自填补后的 A1、连续截尾前。所有文件中的 fitting_dataset/fit_scope 都为 A1。

### qurater 四维

| component_name | P01 | P99 | min_after_winsorization | max_after_winsorization |
|---|---|---|---|---|
| Writing Style | -5.32421875 | 5.609375 | -5.32421875 | 5.609375 |
| Required Expertise | -5.11328125 | 8.234375 | -5.11328125 | 8.234375 |
| Facts and Trivia | -4.217617188 | 5.045742187 | -4.217617188 | 5.045742187 |
| Educational Value | -4.671875 | 8.4375 | -4.671875 | 8.4375 |

### 中位数（所有指标和 A1 七领域）

| metric | domain | median | valid_count |
|---|---|---|---|
| fineweb_edu | ALL | 1.214874566 | 51230 |
| fineweb_edu | arxiv | 2.178714752 | 1419 |
| fineweb_edu | book | 1.413538694 | 171 |
| fineweb_edu | c4 | 0.9851182103 | 10000 |
| fineweb_edu | commoncrawl | 1.225038886 | 9640 |
| fineweb_edu | github | 1.135771215 | 10000 |
| fineweb_edu | stackexchange | 1.472918987 | 10000 |
| fineweb_edu | wikipedia | 1.107888043 | 10000 |
| dsir_books | ALL | -2104.226416 | 51230 |
| dsir_books | arxiv | -33433.10006 | 1419 |
| dsir_books | book | -42714.16883 | 171 |
| dsir_books | c4 | -1340.533639 | 10000 |
| dsir_books | commoncrawl | -3044.159595 | 9640 |
| dsir_books | github | -2239.865081 | 10000 |
| dsir_books | stackexchange | -2091.01414 | 10000 |
| dsir_books | wikipedia | -1462.411928 | 10000 |
| dsir_wiki | ALL | -2129.00643 | 51230 |
| dsir_wiki | arxiv | -32999.96831 | 1419 |
| dsir_wiki | book | -67472.84584 | 171 |
| dsir_wiki | c4 | -1366.717009 | 10000 |
| dsir_wiki | commoncrawl | -3079.754116 | 9640 |
| dsir_wiki | github | -2250.572146 | 10000 |
| dsir_wiki | stackexchange | -2165.976723 | 10000 |
| dsir_wiki | wikipedia | -1441.253077 | 10000 |
| dsir_math | ALL | -2126.006141 | 51230 |
| dsir_math | arxiv | -22776.5876 | 1419 |
| dsir_math | book | -86116.17716 | 171 |
| dsir_math | c4 | -1396.361236 | 10000 |
| dsir_math | commoncrawl | -3274.08237 | 9640 |
| dsir_math | github | -2114.330347 | 10000 |
| dsir_math | stackexchange | -2014.283295 | 10000 |
| dsir_math | wikipedia | -1493.879194 | 10000 |
| qurater | ALL | 0.4063532794 | 51230 |
| qurater | arxiv | 0.7999560094 | 1419 |
| qurater | book | 0.4355544133 | 171 |
| qurater | c4 | 0.4531000546 | 10000 |
| qurater | commoncrawl | 0.4867442398 | 9640 |
| qurater | github | 0.3484519739 | 10000 |
| qurater | stackexchange | 0.4005888376 | 10000 |
| qurater | wikipedia | 0.3728379116 | 10000 |
| modernbert_cleanliness | ALL | 0.8 | 51230 |
| modernbert_cleanliness | arxiv | 1 | 1419 |
| modernbert_cleanliness | book | 1 | 171 |
| modernbert_cleanliness | c4 | 0.8 | 10000 |
| modernbert_cleanliness | commoncrawl | 0.8 | 9640 |
| modernbert_cleanliness | github | 0.4 | 10000 |
| modernbert_cleanliness | stackexchange | 0.8 | 10000 |
| modernbert_cleanliness | wikipedia | 0.8 | 10000 |
| modernbert_readability | ALL | 0.8 | 51230 |
| modernbert_readability | arxiv | 1 | 1419 |
| modernbert_readability | book | 1 | 171 |
| modernbert_readability | c4 | 0.8 | 10000 |
| modernbert_readability | commoncrawl | 1 | 9640 |
| modernbert_readability | github | 0.8 | 10000 |
| modernbert_readability | stackexchange | 0.8 | 10000 |
| modernbert_readability | wikipedia | 0.8 | 10000 |
| modernbert_reasoning | ALL | 0.2 | 51217 |
| modernbert_reasoning | arxiv | 1 | 1419 |
| modernbert_reasoning | book | 1 | 171 |
| modernbert_reasoning | c4 | 0.2 | 10000 |
| modernbert_reasoning | commoncrawl | 0.2 | 9640 |
| modernbert_reasoning | github | 0.2 | 10000 |
| modernbert_reasoning | stackexchange | 0.4 | 10000 |
| modernbert_reasoning | wikipedia | 0.2 | 9987 |
| modernbert_professionalism | ALL | 0.6 | 51225 |
| modernbert_professionalism | arxiv | 1 | 1419 |
| modernbert_professionalism | book | 0.4 | 171 |
| modernbert_professionalism | c4 | 0.2 | 10000 |
| modernbert_professionalism | commoncrawl | 0.4 | 9636 |
| modernbert_professionalism | github | 0.8 | 10000 |
| modernbert_professionalism | stackexchange | 0.8 | 10000 |
| modernbert_professionalism | wikipedia | 0.4 | 9999 |
| fluency_en | ALL | 1 | 51230 |
| fluency_en | arxiv | 1 | 1419 |
| fluency_en | book | 1 | 171 |
| fluency_en | c4 | 1 | 10000 |
| fluency_en | commoncrawl | 1 | 9640 |
| fluency_en | github | 1 | 10000 |
| fluency_en | stackexchange | 1 | 10000 |
| fluency_en | wikipedia | 1 | 10000 |
| ad_en | ALL | 1 | 51230 |
| ad_en | arxiv | 1 | 1419 |
| ad_en | book | 1 | 171 |
| ad_en | c4 | 1 | 10000 |
| ad_en | commoncrawl | 1 | 9640 |
| ad_en | github | 1 | 10000 |
| ad_en | stackexchange | 1 | 10000 |
| ad_en | wikipedia | 1 | 10000 |
| rps_doc_word_count | ALL | 253 | 51230 |
| rps_doc_word_count | arxiv | 5310 | 1419 |
| rps_doc_word_count | book | 69024 | 171 |
| rps_doc_word_count | c4 | 205 | 10000 |
| rps_doc_word_count | commoncrawl | 660 | 9640 |
| rps_doc_word_count | github | 115 | 10000 |
| rps_doc_word_count | stackexchange | 223 | 10000 |
| rps_doc_word_count | wikipedia | 175 | 10000 |
| rps_doc_num_sentences | ALL | 20 | 51230 |
| rps_doc_num_sentences | arxiv | 334 | 1419 |
| rps_doc_num_sentences | book | 4397 | 171 |
| rps_doc_num_sentences | c4 | 13 | 10000 |
| rps_doc_num_sentences | commoncrawl | 34 | 9640 |
| rps_doc_num_sentences | github | 23 | 10000 |
| rps_doc_num_sentences | stackexchange | 20 | 10000 |
| rps_doc_num_sentences | wikipedia | 11 | 10000 |
| rps_doc_unigram_entropy | ALL | 4.60656079 | 51230 |
| rps_doc_unigram_entropy | arxiv | 5.92790847 | 1419 |
| rps_doc_unigram_entropy | book | 6.60434536 | 171 |
| rps_doc_unigram_entropy | c4 | 4.53099923 | 10000 |
| rps_doc_unigram_entropy | commoncrawl | 5.240156765 | 9640 |
| rps_doc_unigram_entropy | github | 3.920187395 | 10000 |
| rps_doc_unigram_entropy | stackexchange | 4.51415478 | 10000 |
| rps_doc_unigram_entropy | wikipedia | 4.430807815 | 10000 |
| rps_doc_frac_unique_words | ALL | 55.73770492 | 51230 |
| rps_doc_frac_unique_words | arxiv | 24.5553889 | 1419 |
| rps_doc_frac_unique_words | book | 10.66643627 | 171 |
| rps_doc_frac_unique_words | c4 | 60.3960396 | 10000 |
| rps_doc_frac_unique_words | commoncrawl | 49.04032567 | 9640 |
| rps_doc_frac_unique_words | github | 59.68992248 | 10000 |
| rps_doc_frac_unique_words | stackexchange | 55 | 10000 |
| rps_doc_frac_unique_words | wikipedia | 62.5 | 10000 |
| rps_doc_frac_no_alph_words | ALL | 23.45679012 | 51230 |
| rps_doc_frac_no_alph_words | arxiv | 29.57569892 | 1419 |
| rps_doc_frac_no_alph_words | book | 22.65331167 | 171 |
| rps_doc_frac_no_alph_words | c4 | 20.28348497 | 10000 |
| rps_doc_frac_no_alph_words | commoncrawl | 20.65708465 | 9640 |
| rps_doc_frac_no_alph_words | github | 34.81541388 | 10000 |
| rps_doc_frac_no_alph_words | stackexchange | 28.5740968 | 10000 |
| rps_doc_frac_no_alph_words | wikipedia | 23.70536038 | 10000 |
| rps_doc_frac_chars_top_2gram | ALL | 2.270326054 | 51230 |
| rps_doc_frac_chars_top_2gram | arxiv | 0.9997093868 | 1419 |
| rps_doc_frac_chars_top_2gram | book | 0.6856390613 | 171 |
| rps_doc_frac_chars_top_2gram | c4 | 2.468627959 | 10000 |
| rps_doc_frac_chars_top_2gram | commoncrawl | 1.772002375 | 9640 |
| rps_doc_frac_chars_top_2gram | github | 2 | 10000 |
| rps_doc_frac_chars_top_2gram | stackexchange | 2.60258235 | 10000 |
| rps_doc_frac_chars_top_2gram | wikipedia | 3.046060792 | 10000 |
| rps_doc_frac_chars_top_3gram | ALL | 1.753446013 | 51230 |
| rps_doc_frac_chars_top_3gram | arxiv | 0.6724949563 | 1419 |
| rps_doc_frac_chars_top_3gram | book | 0.1711565197 | 171 |
| rps_doc_frac_chars_top_3gram | c4 | 1.642822925 | 10000 |
| rps_doc_frac_chars_top_3gram | commoncrawl | 1.588888251 | 9640 |
| rps_doc_frac_chars_top_3gram | github | 1 | 10000 |
| rps_doc_frac_chars_top_3gram | stackexchange | 2.634641151 | 10000 |
| rps_doc_frac_chars_top_3gram | wikipedia | 1 | 10000 |
| rps_lines_uppercase_letter_fraction | ALL | 18.14584024 | 51230 |
| rps_lines_uppercase_letter_fraction | arxiv | 17.50302138 | 1419 |
| rps_lines_uppercase_letter_fraction | book | 51.72992686 | 171 |
| rps_lines_uppercase_letter_fraction | c4 | 3.407711325 | 10000 |
| rps_lines_uppercase_letter_fraction | commoncrawl | 7.484406933 | 9640 |
| rps_lines_uppercase_letter_fraction | github | 21.22182648 | 10000 |
| rps_lines_uppercase_letter_fraction | stackexchange | 27.54230789 | 10000 |
| rps_lines_uppercase_letter_fraction | wikipedia | 36.93411892 | 10000 |
| rps_lines_ending_with_terminal_punctuation_mark | ALL | 50 | 51230 |
| rps_lines_ending_with_terminal_punctuation_mark | arxiv | 33.57664234 | 1419 |
| rps_lines_ending_with_terminal_punctuation_mark | book | 79.08064617 | 171 |
| rps_lines_ending_with_terminal_punctuation_mark | c4 | 100 | 10000 |
| rps_lines_ending_with_terminal_punctuation_mark | commoncrawl | 52.63157895 | 9640 |
| rps_lines_ending_with_terminal_punctuation_mark | github | 21.875 | 10000 |
| rps_lines_ending_with_terminal_punctuation_mark | stackexchange | 41.17647059 | 10000 |
| rps_lines_ending_with_terminal_punctuation_mark | wikipedia | 52.94117647 | 10000 |
| rps_lines_numerical_chars_fraction | ALL | 1.197585613 | 51230 |
| rps_lines_numerical_chars_fraction | arxiv | 3.031805247 | 1419 |
| rps_lines_numerical_chars_fraction | book | 1.539734617 | 171 |
| rps_lines_numerical_chars_fraction | c4 | 0.4907375197 | 10000 |
| rps_lines_numerical_chars_fraction | commoncrawl | 1.825366381 | 9640 |
| rps_lines_numerical_chars_fraction | github | 0.5999236834 | 10000 |
| rps_lines_numerical_chars_fraction | stackexchange | 1.218071088 | 10000 |
| rps_lines_numerical_chars_fraction | wikipedia | 2.128708577 | 10000 |
| rps_doc_mean_word_length | ALL | 5.322730203 | 51230 |
| rps_doc_mean_word_length | arxiv | 5.734962644 | 1419 |
| rps_doc_mean_word_length | book | 4.503931562 | 171 |
| rps_doc_mean_word_length | c4 | 4.815275246 | 10000 |
| rps_doc_mean_word_length | commoncrawl | 4.958840466 | 9640 |
| rps_doc_mean_word_length | github | 9.009479264 | 10000 |
| rps_doc_mean_word_length | stackexchange | 5.174843769 | 10000 |
| rps_doc_mean_word_length | wikipedia | 5.631578947 | 10000 |

### 连续指标截尾

| metric | P01 | P99 | min_after_winsorization | max_after_winsorization |
|---|---|---|---|---|
| fineweb_edu | 0.03446993351 | 3.240450001 | 0.03446993351 | 3.240450001 |
| dsir_books | -56918.40923 | -232.5487817 | -56918.40923 | -232.5487817 |
| dsir_wiki | -60860.82795 | -228.2681829 | -60860.82795 | -228.2681829 |
| dsir_math | -45615.73244 | -227.7430869 | -45615.73244 | -227.7430869 |
| rps_doc_word_count | 10 | 10201.97 | 10 | 10201.97 |
| rps_doc_num_sentences | 1 | 758.13 | 1 | 758.13 |
| rps_doc_unigram_entropy | 2.137130405 | 6.32564844 | 2.137130405 | 6.32564844 |
| rps_doc_frac_unique_words | 17.20475965 | 100 | 17.20475965 | 100 |
| rps_doc_frac_no_alph_words | 16.63689768 | 99.90756178 | 16.63689768 | 99.90756178 |
| rps_doc_frac_chars_top_2gram | 0 | 21 | 0 | 21 |
| rps_doc_frac_chars_top_3gram | 0 | 20.63594321 | 0 | 20.63594321 |
| rps_lines_uppercase_letter_fraction | 1.127057714 | 52.75775062 | 1.127057714 | 52.75775062 |
| rps_lines_ending_with_terminal_punctuation_mark | 0.430056358 | 100 | 0.430056358 | 100 |
| rps_lines_numerical_chars_fraction | 0 | 19.17733033 | 0 | 19.17733033 |
| rps_doc_mean_word_length | 3.954545455 | 16.64097403 | 3.954545455 | 16.64097403 |

### 正负向 Min-Max 范围

正向 z=(x−L)/(U−L)，负向 z=(U−x)/(U−L)，然后 clip 到 [0,1]。
U=L 时使用配置常数 0.5，不制造区分度；实际触发状态见 constant 列。

| metric | direction | L | U | constant |
|---|---|---|---|---|
| fineweb_edu | positive | 0.03446993351 | 3.240450001 | False |
| dsir_books | positive | -56918.40923 | -232.5487817 | False |
| dsir_wiki | positive | -60860.82795 | -228.2681829 | False |
| dsir_math | positive | -45615.73244 | -227.7430869 | False |
| rps_doc_unigram_entropy | positive | 2.137130405 | 6.32564844 | False |
| rps_doc_frac_unique_words | positive | 17.20475965 | 100 | False |
| rps_doc_frac_no_alph_words | negative | 16.63689768 | 99.90756178 | False |
| rps_doc_frac_chars_top_2gram | negative | 0 | 21 | False |
| rps_doc_frac_chars_top_3gram | negative | 0 | 20.63594321 | False |
| rps_lines_uppercase_letter_fraction | negative | 1.127057714 | 52.75775062 | False |
| rps_lines_ending_with_terminal_punctuation_mark | positive | 0.430056358 | 100 | False |
| rps_lines_numerical_chars_fraction | negative | 0 | 19.17733033 | False |

## 适中型梯形效用

x≤a 或 x≥d 时 z=0；a<x<b 线性上升；b≤x≤c 时 z=1；c<x<d 线性下降。
主模型 a=P01、b=P10、c=P90、d=P99。平台分位点由配置控制，可另行拟合 P05/P95 或 P20/P80 敏感性模型。
当前不计算这些额外模型。零宽斜坡按阶跃处理，外端点 0 优先；a=b=c=d 时全为 0，避免除零。

| metric | P01 | P05 | P10 | P20 | P50 | P80 | P90 | P95 | P99 | a | b | c | d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rps_doc_word_count | 10 | 30 | 48 | 87 | 253 | 698 | 1289 | 2669.1 | 10201.97 | 10 | 48 | 1289 | 10201.97 |
| rps_doc_num_sentences | 1 | 3 | 4 | 7 | 20 | 56 | 109 | 219 | 758.13 | 1 | 4 | 109 | 758.13 |
| rps_doc_mean_word_length | 3.954545455 | 4.27197954 | 4.458578247 | 4.708317338 | 5.322730203 | 6.941176471 | 9.22238806 | 11.44985546 | 16.64097403 | 3.954545455 | 4.458578247 | 9.22238806 | 16.64097403 |

## A2/A3 Clip rate

分母为该附件全部保留记录，含按 A1 填补的记录。low/high 指方向统一前的原始指标低于/高于 A1 边界，
对负向指标不代表最终 z 的低/高方向。适中指标报告原值超出外端点的比例，端点等值不计截尾。
qurater 指四个原分量中至少一个越界；低/高事件可在同一条记录发生，不能直接把比例相加。
四维单独统计另存 external_qurater_component_clip_rates.csv。分类指标未经再次截尾，clip rate 为 0。
large_clip_flag 的诊断阈值为配置项 10%，是报告提示阈值，不是伪造的拟合参数。

| dataset | metric | clip_low_rate | clip_high_rate | large_clip_flag | clip_stage |
|---|---|---|---|---|---|
| A2 | fineweb_edu | 5.706785368e-05 | 0.04183073675 | False | A1_continuous_winsor_bounds |
| A2 | dsir_books | 0.2787764652 | 0 | True | A1_continuous_winsor_bounds |
| A2 | dsir_wiki | 0.2491011813 | 0 | True | A1_continuous_winsor_bounds |
| A2 | dsir_math | 0.1922045312 | 0 | True | A1_continuous_winsor_bounds |
| A2 | qurater | 0.002739256977 | 0.3664897563 | True | any_raw_qurater_component_outside_A1_bounds |
| A2 | modernbert_cleanliness | 0 | 0 | False | bounded_identity_no_clipping |
| A2 | modernbert_readability | 0 | 0 | False | bounded_identity_no_clipping |
| A2 | modernbert_reasoning | 0 | 0 | False | bounded_identity_no_clipping |
| A2 | modernbert_professionalism | 0 | 0 | False | bounded_identity_no_clipping |
| A2 | fluency_en | 0 | 0 | False | bounded_identity_no_clipping |
| A2 | ad_en | 0 | 0 | False | bounded_identity_no_clipping |
| A2 | rps_doc_word_count | 0 | 0.1762826 | True | A1_continuous_winsor_bounds |
| A2 | rps_doc_num_sentences | 0 | 0.1327968955 | True | A1_continuous_winsor_bounds |
| A2 | rps_doc_unigram_entropy | 0 | 0.08485989842 | False | A1_continuous_winsor_bounds |
| A2 | rps_doc_frac_unique_words | 0.1276037208 | 0 | True | A1_continuous_winsor_bounds |
| A2 | rps_doc_frac_no_alph_words | 0.000171203561 | 0 | False | A1_continuous_winsor_bounds |
| A2 | rps_doc_frac_chars_top_2gram | 0 | 5.706785368e-05 | False | A1_continuous_winsor_bounds |
| A2 | rps_doc_frac_chars_top_3gram | 0 | 5.706785368e-05 | False | A1_continuous_winsor_bounds |
| A2 | rps_lines_uppercase_letter_fraction | 0.000171203561 | 0.005535581807 | False | A1_continuous_winsor_bounds |
| A2 | rps_lines_ending_with_terminal_punctuation_mark | 5.706785368e-05 | 0 | False | A1_continuous_winsor_bounds |
| A2 | rps_lines_numerical_chars_fraction | 0 | 0.009359128003 | False | A1_continuous_winsor_bounds |
| A2 | rps_doc_mean_word_length | 0.0006277463905 | 0.0003424071221 | False | A1_continuous_winsor_bounds |
| A3 | fineweb_edu | 0.01124896934 | 0.00113863913 | False | A1_continuous_winsor_bounds |
| A3 | dsir_books | 0.008161883074 | 0.03970513173 | False | A1_continuous_winsor_bounds |
| A3 | dsir_wiki | 0.007317719581 | 0.03870391456 | False | A1_continuous_winsor_bounds |
| A3 | dsir_math | 0.01006615886 | 0.04061309828 | False | A1_continuous_winsor_bounds |
| A3 | qurater | 0.02481938828 | 0.0006134909105 | False | any_raw_qurater_component_outside_A1_bounds |
| A3 | modernbert_cleanliness | 0 | 0 | False | bounded_identity_no_clipping |
| A3 | modernbert_readability | 0 | 0 | False | bounded_identity_no_clipping |
| A3 | modernbert_reasoning | 0 | 0 | False | bounded_identity_no_clipping |
| A3 | modernbert_professionalism | 0 | 0 | False | bounded_identity_no_clipping |
| A3 | fluency_en | 0 | 0 | False | bounded_identity_no_clipping |
| A3 | ad_en | 0 | 0 | False | bounded_identity_no_clipping |
| A3 | rps_doc_word_count | 0.04331245828 | 0.001840472732 | False | A1_continuous_winsor_bounds |
| A3 | rps_doc_num_sentences | 0 | 0.009393772822 | False | A1_continuous_winsor_bounds |
| A3 | rps_doc_unigram_entropy | 0.04687561349 | 0.001776669677 | False | A1_continuous_winsor_bounds |
| A3 | rps_doc_frac_unique_words | 0.009398680749 | 0 | False | A1_continuous_winsor_bounds |
| A3 | rps_doc_frac_no_alph_words | 0.007700537909 | 0 | False | A1_continuous_winsor_bounds |
| A3 | rps_doc_frac_chars_top_2gram | 0 | 0.01626487102 | False | A1_continuous_winsor_bounds |
| A3 | rps_doc_frac_chars_top_3gram | 0 | 0.01756547175 | False | A1_continuous_winsor_bounds |
| A3 | rps_lines_uppercase_letter_fraction | 0.007872315364 | 0.01896423103 | False | A1_continuous_winsor_bounds |
| A3 | rps_lines_ending_with_terminal_punctuation_mark | 0.02118752209 | 0 | False | A1_continuous_winsor_bounds |
| A3 | rps_lines_numerical_chars_fraction | 0 | 0.002503042915 | False | A1_continuous_winsor_bounds |
| A3 | rps_doc_mean_word_length | 0.001040480584 | 0.04767069771 | False | A1_continuous_winsor_bounds |

## 分类天花板（保留 PHASE 2C 已有结果）

以下 ceiling_rate 分母为非缺失标量，不重估分类规则。A2-arxiv 的多个分类指标存在明显顶部集中，
cleanliness/reasoning/professionalism 等最高档占比接近 99%；当前仍按官方 argmax 主方案。
二分类顶部占比表示分类标签比例，不是分类准确率；ad_en=1 表示无广告。

| dataset | metric | valid_count | missing | ceiling_rate |
|---|---|---|---|---|
| A1 | modernbert_cleanliness | 51230 | 0 | 0.29238727308217843 |
| A1 | modernbert_readability | 51230 | 0 | 0.2974624243607261 |
| A1 | modernbert_reasoning | 51217 | 13 | 0.09010680047640432 |
| A1 | modernbert_professionalism | 51225 | 5 | 0.17749145924841386 |
| A1 | fluency_en | 51230 | 0 | 0.7829592035916455 |
| A1 | ad_en | 51230 | 0 | 0.9428069490532891 |
| A2 | modernbert_cleanliness | 17523 | 0 | 0.9913256862409405 |
| A2 | modernbert_readability | 17523 | 0 | 0.913827540946185 |
| A2 | modernbert_reasoning | 17523 | 0 | 0.992752382582891 |
| A2 | modernbert_professionalism | 17523 | 0 | 0.995263368144724 |
| A2 | fluency_en | 17523 | 0 | 0.9919534326313987 |
| A2 | ad_en | 17523 | 0 | 0.9990869143411516 |
| A3 | modernbert_cleanliness | 203752 | 0 | 0.068053319722015 |
| A3 | modernbert_readability | 203752 | 0 | 0.07294161529702776 |
| A3 | modernbert_reasoning | 203752 | 0 | 0.05087557422749225 |
| A3 | modernbert_professionalism | 203751 | 1 | 0.4801792383841061 |
| A3 | fluency_en | 203752 | 0 | 0.5581834779535907 |
| A3 | ad_en | 203752 | 0 | 0.9968834661745651 |

## 输出验证与非重叠索引

| dataset | n | dimensions | NaN_count | Inf_count | all_in_0_1 |
|---|---|---|---|---|---|
| A1 | 51230 | 22 | 0 | 0 | True |
| A2 | 17523 | 22 | 0 | 0 | True |
| A3 | 203752 | 22 | 0 | 0 | True |

每指标实际 min/max 及取值校验见 feature_validation.csv。

| dataset | domain | n | overlap_ids | nonoverlap_ids | overlap_Z_exactly_equal | records_removed |
|---|---|---|---|---|---|---|
| A2 | arxiv | 17523 | 1419 | 16104 | True | 0 |
| A3 | github | 203752 | 10000 | 193752 | True | 0 |

ID 索引只列非重叠 ID 与来源，不复制完整 Z；A1 抽样、扩展全量、扩展非重叠余集可供后续一致性检验。

已有历史结果 52 个文件经 SHA-256 核验未改变，见 prior_artifacts_unchanged_phase3.csv；原始附件读前读后哈希一致。

## 阶段局限与停止点

Z 表示按确认方向定义的指标效用，尚无综合权重或质量分数。梯形平台内的差异被有意压平，
A1 范围外值发生饱和；明显扩展截尾与分类天花板需在后续解释、敏感性分析中保留。
领域中位数保留主样本尺度，但可能给缺失评级赋予小数；本次具体填补值以日志为准。
依赖官方质量字段而非文本人工验证，不能把向量一致推断为文本完全一致。

本阶段到 Z 为止。未运行 04_build_quality_score.py，未计算 CRITIC、Q 或其他题目。等待人工确认。
