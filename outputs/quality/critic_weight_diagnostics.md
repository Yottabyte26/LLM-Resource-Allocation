# PHASE 4A：A1 CRITIC 权重诊断

本阶段只拟合权重并诊断相关性，不计算任何样本 Q、领域 Q 或 A1/A2/A3 的 Q 对照。

## 输入与计算口径

唯一拟合输入：outputs/preprocess/A1_quality_features.csv，实际 51,230×22 个 Z 值。
A2/A3 未参与标准差、相关矩阵或权重拟合。指标顺序和类型取自已确认的 preprocessing_model_A1.json。
PHASE 3 配置、模型、参数与输出文件读前读后 SHA-256 一致，见 phase3_artifacts_unchanged_4A.csv。

主方案：σ_j=std(Z_j,ddof=1)，r_jk 为 Pearson，R_j=Σ_k(1−r_jk)，C_j=σ_jR_j，w_j=C_j/Σ_jC_j。
保留负相关及其信息量，未取相关系数绝对值、未剔除指标、未人工调整权重。
若常数列使相关系数无定义，或总信息量为零，脚本停止而不填造相关系数/权重。本次没有这类退化。

权重和=1；所有权重非负=True。

## 22 个 Pearson 主权重

| metric | metric_type | mean_z | std_z | min_z | max_z | unique_value_count | conflict_information | information_content | critic_weight | weight_rank |
|---|---|---|---|---|---|---|---|---|---|---|
| fluency_en | binary | 0.7829592036 | 0.4122346495 | 0 | 1 | 2 | 18.62358698 | 7.677287851 | 0.08178702499 | 1 |
| modernbert_professionalism | modernbert_ordinal | 0.5679367558 | 0.2946334938 | 0.2 | 1 | 5 | 22.23870821 | 6.552268297 | 0.06980206309 | 2 |
| rps_lines_ending_with_terminal_punctuation_mark | continuous_positive | 0.5440028271 | 0.3050620566 | 0 | 1 | 6614 | 19.00605766 | 5.79802704 | 0.06176704477 | 3 |
| rps_lines_uppercase_letter_fraction | continuous_negative | 0.633767156 | 0.2804956786 | 0 | 1 | 49958 | 20.12705916 | 5.645553118 | 0.06014272266 | 4 |
| modernbert_reasoning | modernbert_ordinal | 0.3506343939 | 0.2672400636 | 0 | 1 | 6 | 19.61188609 | 5.241081687 | 0.05583384227 | 5 |
| ad_en | binary | 0.9428069491 | 0.2322133899 | 0 | 1 | 2 | 21.56768342 | 5.008304878 | 0.05335404432 | 6 |
| modernbert_readability | modernbert_ordinal | 0.7592972867 | 0.2551423802 | 0 | 1 | 6 | 18.1839557 | 4.639497739 | 0.04942509971 | 7 |
| modernbert_cleanliness | modernbert_ordinal | 0.7275073199 | 0.2534571012 | 0.2 | 1 | 5 | 17.5338843 | 4.444087488 | 0.04734337197 | 8 |
| rps_doc_frac_unique_words | continuous_positive | 0.4732139342 | 0.1960147409 | 0 | 1 | 23354 | 22.45898878 | 4.402292866 | 0.04689812908 | 9 |
| rps_doc_mean_word_length | moderate | 0.9180148908 | 0.2157286914 | 0 | 1 | 7631 | 19.90993029 | 4.295143208 | 0.04575665154 | 10 |
| rps_doc_word_count | moderate | 0.9200499877 | 0.2180529369 | 0 | 1 | 2836 | 18.29197463 | 3.988618789 | 0.04249121186 | 11 |
| rps_doc_num_sentences | moderate | 0.9257081092 | 0.2150043842 | 0 | 1 | 587 | 18.02996025 | 3.8765205 | 0.04129701598 | 12 |
| rps_doc_unigram_entropy | continuous_positive | 0.5733218526 | 0.20173063 | 0 | 1 | 46200 | 19.09484876 | 3.852015871 | 0.04103596536 | 13 |
| fineweb_edu | continuous_positive | 0.3959958858 | 0.1981190305 | 0 | 1 | 50124 | 18.83320326 | 3.731215972 | 0.03974907023 | 14 |
| rps_lines_numerical_chars_fraction | continuous_negative | 0.8752744927 | 0.1742131783 | 0 | 1 | 39722 | 20.42401378 | 3.558132354 | 0.03790519066 | 15 |
| rps_doc_frac_chars_top_2gram | continuous_negative | 0.8328020837 | 0.181687401 | 0 | 1 | 25368 | 19.18024155 | 3.484808239 | 0.03712406049 | 16 |
| rps_doc_frac_chars_top_3gram | continuous_negative | 0.862919509 | 0.1773536422 | 0 | 1 | 21206 | 19.6338781 | 3.482139791 | 0.0370956332 | 17 |
| rps_doc_frac_no_alph_words | continuous_negative | 0.8598906538 | 0.1669247292 | 0 | 1 | 43102 | 19.15641621 | 3.197679587 | 0.03406524614 | 18 |
| qurater | qurater | 0.4257465845 | 0.1689171129 | 0 | 1 | 51174 | 17.69743777 | 2.989400093 | 0.03184642088 | 19 |
| dsir_math | continuous_positive | 0.9202763612 | 0.139572035 | 0 | 1 | 50206 | 19.46582806 | 2.716885235 | 0.02894328895 | 20 |
| dsir_books | continuous_positive | 0.931532914 | 0.1373814672 | 0 | 1 | 50206 | 19.45233107 | 2.672389783 | 0.02846927381 | 21 |
| dsir_wiki | continuous_positive | 0.934204135 | 0.1348037173 | 0 | 1 | 50206 | 19.40535314 | 2.615913738 | 0.02786762805 | 22 |

## 权重集中度与标记

最大：fluency_en，0.08178702499；最小：dsir_wiki，0.02786762805。
前 5 累计=32.93326978%；前 10 累计=57.21099944%。
H_w=−Σw_jln(w_j)=3.048875704；H_norm=H_w/ln(22)=0.9863584051。
熵越接近 1 越均匀；熵与最高权重、累计权重一起解读，不以熵单独证明权重合理。

异常标记只提示复核，不改权重。配置独立保存于 configs/critic_diagnostics.yaml，不改 PHASE 3 参数。
weight>0.1；weight>2.0×(1/22)=0.09090909091；
weight<0.01；std<0.01；unique≤6。
后两项是明示的诊断阈值，不是从附件伪造的统计结果；少取值对 ordinal/binary 是预期结构，不等于指标无效。

| metric | critic_weight | std_z | unique_value_count | flag_weight_gt_0_10 | flag_weight_gt_2_equal | flag_weight_lt_0_01 | flag_low_std | flag_low_unique_values | low_unique_expected_for_type |
|---|---|---|---|---|---|---|---|---|---|
| modernbert_cleanliness | 0.04734337197 | 0.2534571012 | 5 | False | False | False | False | True | True |
| modernbert_readability | 0.04942509971 | 0.2551423802 | 6 | False | False | False | False | True | True |
| modernbert_reasoning | 0.05583384227 | 0.2672400636 | 6 | False | False | False | False | True | True |
| modernbert_professionalism | 0.06980206309 | 0.2946334938 | 5 | False | False | False | False | True | True |
| fluency_en | 0.08178702499 | 0.4122346495 | 2 | False | False | False | False | True | True |
| ad_en | 0.05335404432 | 0.2322133899 | 2 | False | False | False | False | True | True |

标记计数：{"flag_weight_gt_0_10": 0, "flag_weight_gt_2_equal": 0, "flag_weight_lt_0_01": 0, "flag_low_std": 0, "flag_low_unique_values": 6}。

主模型是否有明显集中，应结合上述数值及 >0.10、>2×等权的标记；不能把高权重解释为指标更真实或更重要。

未触发两类高权重标记；结合归一化熵和累计权重，本次未见单一指标明显主导的权重集中。

## 前五大与后五小

| metric | critic_weight | weight_rank |
|---|---|---|
| fluency_en | 0.08178702499 | 1 |
| modernbert_professionalism | 0.06980206309 | 2 |
| rps_lines_ending_with_terminal_punctuation_mark | 0.06176704477 | 3 |
| rps_lines_uppercase_letter_fraction | 0.06014272266 | 4 |
| modernbert_reasoning | 0.05583384227 | 5 |
| rps_doc_frac_no_alph_words | 0.03406524614 | 18 |
| qurater | 0.03184642088 | 19 |
| dsir_math | 0.02894328895 | 20 |
| dsir_books | 0.02846927381 | 21 |
| dsir_wiki | 0.02786762805 | 22 |

## 负相关指标对

不重复计数、不计对角线：共 84 对 Pearson r<0；
r<−0.2 有 24 对，r<−0.4 有 9 对。
全部越大越好并不保证统计正相关：领域组成、指标取舍和梯形效用会影响联合分布。
负相关在标准 CRITIC 中增加冲突信息，本轮如实保留；它不自动证明方向错误，也不自动证明质量语义正确。

### Pearson 最负的前 20 对

| metric_1 | metric_2 | pearson_r | spearman_r |
|---|---|---|---|
| rps_doc_unigram_entropy | rps_doc_frac_unique_words | -0.6646239762 | -0.6479777934 |
| dsir_math | rps_doc_unigram_entropy | -0.5071707697 | -0.8310669281 |
| modernbert_reasoning | rps_doc_frac_unique_words | -0.502759215 | -0.4564458314 |
| modernbert_professionalism | rps_lines_ending_with_terminal_punctuation_mark | -0.4932015658 | -0.4979277517 |
| dsir_books | rps_doc_unigram_entropy | -0.460114691 | -0.8045662714 |
| dsir_wiki | rps_doc_unigram_entropy | -0.4577301958 | -0.8076062508 |
| dsir_books | modernbert_reasoning | -0.428868416 | -0.46118835 |
| dsir_wiki | modernbert_reasoning | -0.4231988078 | -0.4702323896 |
| dsir_math | modernbert_reasoning | -0.4221255545 | -0.4496413474 |
| fineweb_edu | rps_doc_frac_unique_words | -0.3929796011 | -0.3749569911 |
| dsir_books | modernbert_professionalism | -0.3333932538 | -0.454389101 |
| qurater | rps_doc_frac_unique_words | -0.3203909953 | -0.3011860696 |
| dsir_wiki | modernbert_professionalism | -0.3183869346 | -0.4534834706 |
| dsir_math | modernbert_professionalism | -0.3158079925 | -0.410618785 |
| modernbert_readability | rps_doc_frac_unique_words | -0.3021722679 | -0.3138911001 |
| modernbert_professionalism | rps_doc_frac_unique_words | -0.292953372 | -0.3013226682 |
| ad_en | rps_lines_ending_with_terminal_punctuation_mark | -0.2781201446 | -0.2588220027 |
| modernbert_professionalism | rps_doc_frac_no_alph_words | -0.2581434567 | -0.4705821977 |
| ad_en | rps_lines_uppercase_letter_fraction | -0.2423172648 | -0.2688759659 |
| dsir_books | qurater | -0.2383793322 | -0.24178652 |

## 三个适中型指标：保持 P10–P90 平台

| metric | mean_z | std_z | z_eq_1_rate | z_eq_0_rate | critic_weight |
|---|---|---|---|---|---|
| rps_doc_word_count | 0.9200499877 | 0.2180529369 | 0.8008198321 | 0.02014444661 | 0.04249121186 |
| rps_doc_num_sentences | 0.9257081092 | 0.2150043842 | 0.8268983018 | 0.0312707398 | 0.04129701598 |
| rps_doc_mean_word_length | 0.9180148908 | 0.2157286914 | 0.8 | 0.02004684755 | 0.04575665154 |

平台会把内部原值差异压为 1，但是否导致低方差须结合实际 std 与权重；当前不改变效用函数。

本次三个适中型指标均未触发极低标准差标记；平台集中存在，但未发生方差退化。

## 六个离散分类指标（A1）

| metric | unique_values | std_z | ceiling_rate | critic_weight |
|---|---|---|---|---|
| modernbert_cleanliness | [0.2, 0.4, 0.6, 0.8, 1.0] | 0.2534571012 | 0.2923872731 | 0.04734337197 |
| modernbert_readability | [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] | 0.2551423802 | 0.2974624244 | 0.04942509971 |
| modernbert_reasoning | [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] | 0.2672400636 | 0.09008393519 | 0.05583384227 |
| modernbert_professionalism | [0.2, 0.4, 0.6, 0.8, 1.0] | 0.2946334938 | 0.1774741362 | 0.06980206309 |
| fluency_en | [0.0, 1.0] | 0.4122346495 | 0.7829592036 | 0.08178702499 |
| ad_en | [0.0, 1.0] | 0.2322133899 | 0.9428069491 | 0.05335404432 |

所有值来自已填补的 A1 Z，ceiling_rate 分母为 A1 全部记录。PHASE 2C 的 ceiling_rate 分母为原始非缺失数，
两者可能因 18 次 A1 填补略有不同；本报告不覆盖或改写 PHASE 2C 结果。
σ 越小在给定 R 下会减少 CRITIC 信息量，但最终权重还受相关结构影响，不能单由天花板率推出权重。
A2 的强天花板不参与本次 A1 拟合，也不会反过来压低 A1 权重。

## Spearman 敏感性（主模型仍为 Pearson）

Spearman 按每列平均秩处理并列值；σ 仍使用原始 Z 的样本标准差，不使用秩标准差。

两组权重 Pearson correlation=0.9907224324；
Spearman rank correlation=0.9830604178。
Pearson Top-5：fluency_en, modernbert_professionalism, rps_lines_ending_with_terminal_punctuation_mark, rps_lines_uppercase_letter_fraction, modernbert_reasoning。
Spearman Top-5：fluency_en, modernbert_professionalism, rps_lines_uppercase_letter_fraction, rps_lines_ending_with_terminal_punctuation_mark, modernbert_reasoning。
Top-5 集合一致=True，顺序一致=False，交集数=5。

| metric | pearson_critic_weight | critic_weight | pearson_weight_rank | weight_rank | weight_difference | rank_change |
|---|---|---|---|---|---|---|
| fineweb_edu | 0.03974907023 | 0.04065611186 | 14 | 14 | 0.0009070416267 | 0 |
| dsir_books | 0.02846927381 | 0.02945823793 | 21 | 21 | 0.0009889641146 | 0 |
| dsir_wiki | 0.02786762805 | 0.0289458997 | 22 | 22 | 0.001078271644 | 0 |
| dsir_math | 0.02894328895 | 0.03014283137 | 20 | 20 | 0.001199542418 | 0 |
| qurater | 0.03184642088 | 0.03125022531 | 19 | 19 | -0.0005961955625 | 0 |
| modernbert_cleanliness | 0.04734337197 | 0.04746010281 | 8 | 8 | 0.0001167308381 | 0 |
| modernbert_readability | 0.04942509971 | 0.04998927016 | 7 | 7 | 0.0005641704474 | 0 |
| modernbert_reasoning | 0.05583384227 | 0.05509234939 | 5 | 5 | -0.0007414928838 | 0 |
| modernbert_professionalism | 0.06980206309 | 0.07011508306 | 2 | 2 | 0.0003130199732 | 0 |
| fluency_en | 0.08178702499 | 0.08034278481 | 1 | 1 | -0.001444240182 | 0 |
| ad_en | 0.05335404432 | 0.05214610767 | 6 | 6 | -0.001207936653 | 0 |
| rps_doc_word_count | 0.04249121186 | 0.04623302223 | 11 | 9 | 0.003741810362 | -2 |
| rps_doc_num_sentences | 0.04129701598 | 0.04409050103 | 12 | 11 | 0.002793485048 | -1 |
| rps_doc_unigram_entropy | 0.04103596536 | 0.04372163641 | 13 | 12 | 0.002685671058 | -1 |
| rps_doc_frac_unique_words | 0.04689812908 | 0.04285015786 | 9 | 13 | -0.004047971228 | 4 |
| rps_doc_frac_no_alph_words | 0.03406524614 | 0.0315487434 | 18 | 18 | -0.00251650274 | 0 |
| rps_doc_frac_chars_top_2gram | 0.03712406049 | 0.03895576864 | 16 | 15 | 0.001831708152 | -1 |
| rps_doc_frac_chars_top_3gram | 0.0370956332 | 0.03783579431 | 17 | 16 | 0.0007401611017 | -1 |
| rps_lines_uppercase_letter_fraction | 0.06014272266 | 0.05930023828 | 4 | 3 | -0.0008424843724 | -1 |
| rps_lines_ending_with_terminal_punctuation_mark | 0.06176704477 | 0.05851828763 | 3 | 4 | -0.003248757138 | 1 |
| rps_lines_numerical_chars_fraction | 0.03790519066 | 0.03686450012 | 15 | 17 | -0.001040690542 | 2 |
| rps_doc_mean_word_length | 0.04575665154 | 0.04448234605 | 10 | 10 | -0.001274305482 | 0 |

等权表仅保存 1/22，不用于计算 Q。

## 停止点

权重与诊断表已生成。本轮未运行完整 04_build_quality_score.py，未计算 Q_i、Q_domain 或评分对照。
等待人工确认权重后才进入 PHASE 4B。
