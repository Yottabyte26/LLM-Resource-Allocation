# PHASE 5B：高冲突样本稳健重聚合验证

第一小问 Q/Z/全局权重、第一阶段 K/P90/P95 全部固定。输入直接来自三份已有冲突 CSV；没有调用预处理、CRITIC、Q/K 拟合或指数惩罚。输出中的 Q_robust 仅为候选，不覆盖原 Q。

## 算法与统计口径

高冲突使用阈值文件的完整精度 A1 P90，不使用显示四舍五入的 0.333206。普通样本 Q_robust=原 Q（精确复制）。高冲突：r=z−原Q，a=min(1,δ/(|r|+1e−8))，局部权重=w×a/Σ(w×a)，候选评分=Σ局部权重×z。只做一次重聚合，中心固定为原 Q；不迭代。

a 只是残差启发式系数，尚未校准为真实可靠性。K 继续表示内部指标分歧，不作为额外低质量程度。方向统计以 |变化|≤1e−10 为数值近似不变；这个容差不是实际意义上的最小重要差异。低冲突为 K≤原P50，普通为 K≤原P90，极高为 K>原P95。

## 五个 δ 的 A1 全量结果

| delta | pearson_Q_Qrobust | spearman_Q_Qrobust | spearman_high_Q_Qrobust | mean_abs_change_low | mean_abs_change_high | mean_abs_change_P95_conflict | P95_abs_change_high | max_abs_change_all | up_rate_high | down_rate_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.150000 | 0.989857 | 0.990618 | 0.979889 | 0.000000 | 0.031492 | 0.030157 | 0.076688 | 0.130946 | 0.659770 | 0.340230 |
| 0.200000 | 0.989759 | 0.990294 | 0.981484 | 0.000000 | 0.031477 | 0.029990 | 0.078882 | 0.129472 | 0.684755 | 0.315245 |
| 0.250000 | 0.989811 | 0.990134 | 0.984778 | 0.000000 | 0.031406 | 0.029926 | 0.079191 | 0.126985 | 0.713840 | 0.286160 |
| 0.300000 | 0.990127 | 0.990155 | 0.988560 | 0.000000 | 0.030845 | 0.029099 | 0.078606 | 0.122208 | 0.727503 | 0.272497 |
| 0.350000 | 0.991801 | 0.992432 | 0.991541 | 0.000000 | 0.028429 | 0.026911 | 0.070365 | 0.105379 | 0.747414 | 0.252586 |

全体/高冲突的变化方向计数、全体排名变化、最大调整、低/普通组是否精确不变均保存于 delta_sensitivity_A1.csv。全局稳定性部分源自只调整原 P90 以上样本，不能代替高冲突组内稳定性。

## 保守候选 δ=0.35

按照用户较大 δ 优先原则，推荐 δ=0.35 作为验证候选；它同时取得网格内最小高冲突平均绝对变化和最小最大变化。未按最大效果选择，未增加拍脑袋的接受门槛。高冲突平均 |变化|=0.028429，P95 |变化|=0.070365，最大 |变化|=0.105379。最大调整不能称为对每个样本都很小，需逐例检查。

## 极端指标降权是否实际发生

| delta | mean_local_weight_L1_change | original_effective_metric_count | mean_robust_effective_metric_count_high | extreme_residual_normalized_weight_down_rate | extreme_residual_significant_downweight_rate | max_local_weight_high |
| --- | --- | --- | --- | --- | --- | --- |
| 0.150000 | 0.362770 | 20.201776 | 17.065213 | 1.000000 | 0.234115 | 0.180064 |
| 0.200000 | 0.298460 | 20.201776 | 18.250440 | 1.000000 | 0.134418 | 0.141414 |
| 0.250000 | 0.238636 | 20.201776 | 19.083520 | 1.000000 | 0.066188 | 0.116499 |
| 0.300000 | 0.184292 | 20.201776 | 19.630742 | 1.000000 | 0.019440 | 0.103906 |
| 0.350000 | 0.135213 | 20.201776 | 19.971192 | 1.000000 | 0.000000 | 0.094453 |

显著降权的操作定义为归一化局部权重/原权重≤0.5，见配置；这是供查看的诊断界线，不是优化参数。a<1 不保证归一化后的权重比原来小，因此单独记录局部实际权重。最大绝对残差并列时逐个计数。有效指标数为 1/Σw²；它下降表明局部权重可能更集中，不能把结果自动称为全面更稳健。权重表仅证明极端指标权重变化，不证明真实质量误差变小。

## 典型核验材料

| group | available_n | requested_n | selected_n | definition |
| --- | --- | --- | --- | --- |
| up | 51.000000 | 10.000000 | 10.000000 | saved text material; unchanged uses stated numeric tolerance |
| down | 18.000000 | 10.000000 | 10.000000 | saved text material; unchanged uses stated numeric tolerance |
| unchanged | 10.000000 | 10.000000 | 10.000000 | saved text material; unchanged uses stated numeric tolerance |
| high_conflict_smallest_change | 69.000000 | 10.000000 | 10.000000 | high-conflict smallest absolute changes; not claimed near-zero |

上调/下调各选已有文本中变化最大的真实案例；近似不变按数值容差筛选。普通样本精确不变是算法规定，不构成高冲突重聚合有效性证据。另保存高冲突组中调整最小的十条，明确标为 smallest_change，不声称其变化接近零。案例分组可复用同一文本。

| case_group | id | domain | Q | K | Q_robust | change | most_downweighted_metrics |
| --- | --- | --- | --- | --- | --- | --- | --- |
| up | BkiUfI_xK1yAgXBVGnXn | book | 0.651560 | 0.399773 | 0.747023 | 0.095463 | dsir_math;rps_doc_word_count;rps_doc_num_sentences |
| up | BkiUdcg4dbjiVCOrALlv | book | 0.652486 | 0.403648 | 0.746536 | 0.094050 | dsir_books;rps_doc_word_count;dsir_wiki |
| up | BkiUabvxK0wg05VB4taD | book | 0.631139 | 0.405985 | 0.720013 | 0.088873 | dsir_math;dsir_books;dsir_wiki |
| up | BkiUd885qoTBG9o64qmS | book | 0.645506 | 0.398956 | 0.733488 | 0.087982 | dsir_math;rps_doc_word_count;rps_doc_num_sentences |
| up | BkiUbvjxK6wB9mn49q52 | arxiv | 0.637361 | 0.411294 | 0.722443 | 0.085082 | dsir_wiki;dsir_math;rps_doc_word_count |
| up | BkiUd_E4uzlh53v23gzI | book | 0.628033 | 0.402171 | 0.712384 | 0.084351 | dsir_books;dsir_wiki;dsir_math |
| up | BkiUdmg5qYVBXTJ3BsEF | arxiv | 0.736102 | 0.334513 | 0.815665 | 0.079563 | rps_doc_word_count;rps_doc_frac_unique_words;rps_lines_uppercase_letter_fraction |
| up | BkiUcV3xK6Ot9WA5gtvG | github | 0.636686 | 0.407472 | 0.715650 | 0.078964 | rps_doc_unigram_entropy;rps_lines_ending_with_terminal_punctuation_mark;modernbert_reasoning |
| up | BkiUc_g4eIZjkiQT3F92 | book | 0.613805 | 0.408006 | 0.690462 | 0.076657 | dsir_math;rps_doc_word_count;rps_doc_num_sentences |
| up | BkiUcsHxK7FjYCv2RUDM | book | 0.639604 | 0.404349 | 0.716095 | 0.076491 | dsir_wiki;rps_doc_word_count;rps_doc_num_sentences |
| down | BkiUdPY5qWTD6fkJfvzP | github | 0.434240 | 0.445241 | 0.368694 | -0.065546 | dsir_wiki;ad_en;rps_doc_frac_unique_words |
| down | BkiUcC3xK6-gDz87O0dB | c4 | 0.460850 | 0.435179 | 0.408792 | -0.052058 | rps_doc_word_count;rps_lines_ending_with_terminal_punctuation_mark;rps_doc_frac_no_alph_words |
| down | BkiUb1zxK19JmejM8KYA | github | 0.491399 | 0.445150 | 0.460540 | -0.030859 | ad_en;rps_doc_frac_unique_words;rps_lines_numerical_chars_fraction |
| down | BkiUbRzxK3xgpdbchQki | github | 0.468656 | 0.457998 | 0.442408 | -0.026248 | rps_lines_numerical_chars_fraction;dsir_wiki;dsir_math |
| down | BkiUboE5qg5A5-klCrDP | wikipedia | 0.495612 | 0.434500 | 0.469909 | -0.025703 | ad_en;rps_doc_word_count;rps_doc_frac_no_alph_words |
| down | BkiUdpU5qoYArXHsxfh_ | book | 0.486911 | 0.446577 | 0.461321 | -0.025590 | modernbert_cleanliness;ad_en;rps_doc_unigram_entropy |
| down | BkiUdBY5jDKDyJ94W75k | c4 | 0.512534 | 0.441840 | 0.488816 | -0.023718 | modernbert_reasoning;rps_doc_frac_chars_top_3gram;fluency_en |
| down | BkiUainxK1Thg9qFa8nv | c4 | 0.496860 | 0.462123 | 0.473438 | -0.023422 | rps_doc_word_count;rps_lines_uppercase_letter_fraction;rps_lines_numerical_chars_fraction |
| down | BkiUctk5qoTAhudwxeH1 | c4 | 0.518232 | 0.436977 | 0.495410 | -0.022822 | fluency_en;rps_doc_frac_chars_top_2gram;rps_doc_frac_unique_words |
| down | BkiUatvxaJJQnMsbx8XA | github | 0.476037 | 0.441055 | 0.454284 | -0.021753 | rps_doc_frac_chars_top_2gram;rps_doc_frac_unique_words;ad_en |
| unchanged | BkiUa-c4eIZijWO-ljYW | wikipedia | 0.773964 | 0.260901 | 0.773964 | 0.000000 | 未定义 |
| unchanged | BkiUb7zxaJJQnKrAjVb7 | wikipedia | 0.710535 | 0.260914 | 0.710535 | 0.000000 | 未定义 |
| unchanged | BkiUc8nxaKgTvXx06mnP | c4 | 0.749614 | 0.260895 | 0.749614 | 0.000000 | 未定义 |
| unchanged | BkiUcyg5qoYAmdCYN0tV | github | 0.720616 | 0.260912 | 0.720616 | 0.000000 | 未定义 |
| unchanged | BkiUd-U4eIXh7gjUqssm | wikipedia | 0.737734 | 0.260911 | 0.737734 | 0.000000 | 未定义 |
| unchanged | BkiUdOg5qhDBZQieouQK | c4 | 0.773123 | 0.260913 | 0.773123 | 0.000000 | 未定义 |
| unchanged | BkiUd_3xaJJQnK0WJoXz | wikipedia | 0.715818 | 0.260927 | 0.715818 | 0.000000 | 未定义 |
| unchanged | BkiUdlbxK7IDPjMdK1YM | github | 0.727162 | 0.260912 | 0.727162 | 0.000000 | 未定义 |
| unchanged | BkiUdmw5qoYAwaI90jqY | commoncrawl | 0.759734 | 0.260902 | 0.759734 | 0.000000 | 未定义 |
| unchanged | BkiUduvxK7IAEwOJJChw | wikipedia | 0.708650 | 0.260915 | 0.708650 | 0.000000 | 未定义 |

typical_cases.csv 保留完整原22维Z、Q/K、候选变化、降权指标、全部局部权重/系数以及已有 content；不重新读取任何原始 JSONL。typical_case_local_weights.csv 给出逐指标审计明细。当前仅有第一阶段的人工确认/AI片段线索，尚无对本轮重聚合解释改善的独立人工标注。

下表与 typical_case_AI_observations.csv 是明确标记的 AI 辅助观察；短文本检查全文，长文仅看开头片段。不是最终人工标注，不能作为外部实验标签。

| id | domain | Q | Q_robust | change | inspection_scope | observation |
| --- | --- | --- | --- | --- | --- | --- |
| BkiUfI_xK1yAgXBVGnXn | book | 0.651560 | 0.747023 | 0.095463 | 开头片段 | 开头为瑞典语书籍简介。低分 DSIR/词数/句数被降低权重后上调；可能降低长度效用的影响，但不能据片段确认真实质量提高。 |
| BkiUbvjxK6wB9mn49q52 | arxiv | 0.637361 | 0.722443 | 0.085082 | 开头片段 | 开头为含公式/引用的介子跃迁论文；词数、数值字符及 DSIR 低分被减权。专业用途可解释评分分歧，但不足以证明上调幅度正确。 |
| BkiUdmg5qYVBXTJ3BsEF | arxiv | 0.736102 | 0.815665 | 0.079563 | 开头片段 | 开头为天文学 LaTeX 论文，词数/独特词/大写比例低分被减权；原K仅略高于阈值，却产生较大上调，提醒分段激活可能形成边界跳变。 |
| BkiUdPY5qWTD6fkJfvzP | github | 0.434240 | 0.368694 | -0.065546 | 完整短文本 | 全文为23字符的重复大写串；降低部分高分指标后下调，方向与缺乏自然语言内容的观察相符，但不等于已获独立准确性标签。 |
| BkiUbRzxK3xgpdbchQki | github | 0.468656 | 0.442408 | -0.026248 | 完整短文本 | 全文是短 PHP 配置语句。下调不一定改善代码用途评价，说明不能把偏离较远的指标自动视为错误。 |
| BkiUainxK1Thg9qFa8nv | c4 | 0.496860 | 0.473438 | -0.023422 | 完整短文本 | 全文为重复沙发关键词行。部分高分格式/长度指标减权后下调，解释方向合理；只能作为案例线索。 |
| BkiUdwo5qhLBXeFpceh2 | github | 0.497385 | 0.497479 | 0.000094 | 完整短文本 | 全文为 Android XML 颜色选择配置，调整很小；两侧指标减权接近平衡，不足以验证代码实际质量。 |

## A2/A3 固定候选验证

δ 只从 A1 选择，A2/A3 固定同一 δ 与 epsilon、全局权重和 A1 P90。full 保留跨集重叠；nonoverlap 用既有索引独立对照，不删除数据。

| dataset | domain | view | delta | n | n_high | pearson_Q_Qrobust | spearman_Q_Qrobust | spearman_high_Q_Qrobust | mean_abs_change_high | mean_abs_change_P95_conflict | max_abs_change_all | up_rate_high | down_rate_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A2 | arxiv | full | 0.350000 | 17523.000000 | 2885.000000 | 0.847876 | 0.854660 | 0.898833 | 0.077694 | 0.080674 | 0.105831 | 0.996880 | 0.003120 |
| A2 | arxiv | nonoverlap | 0.350000 | 16104.000000 | 2681.000000 | 0.846838 | 0.851526 | 0.898841 | 0.077652 | 0.080577 | 0.105831 | 0.996643 | 0.003357 |
| A3 | github | full | 0.350000 | 203752.000000 | 47064.000000 | 0.988203 | 0.988754 | 0.991760 | 0.021415 | 0.020354 | 0.093743 | 0.685790 | 0.314210 |
| A3 | github | nonoverlap | 0.350000 | 193752.000000 | 44740.000000 | 0.988200 | 0.988735 | 0.991777 | 0.021426 | 0.020357 | 0.093743 | 0.685338 | 0.314662 |

| dataset | shared_n | nonoverlap_n | max_shared_candidate_abs_difference |
| --- | --- | --- | --- |
| A2 | 1419.000000 | 16104.000000 | 0.000000 |
| A3 | 10000.000000 | 193752.000000 | 0.000000 |

实际复现具有领域差异：A2-arxiv 全体 Spearman=0.854660、高冲突平均 |变化|=0.077694、上调率=0.996880；A3-github 分别为 0.988754、0.021415、0.685790。arxiv 的单向较大上调也已存在于 A1 域内，并非仅扩展集造成；不能用混合领域 A1 全体的高相关概括每个领域的排名稳定性。

## 辅助赋权敏感性

仅对选定 δ 作等权辅助对照：中心为原冲突表已保存 Q_equal，采用同一高冲突掩码和 δ 做一次等权局部重聚合。不会替代主权重、原 Q 或阈值。比较主候选与等权候选的距离是否比原有 Delta_Q 小；该距离也不是真实误差。

| dataset | n_high | mean_original_Delta_Q_high | mean_candidate_Delta_Q_high | candidate_over_original_ratio | candidate_difference_reduced_rate_high |
| --- | --- | --- | --- | --- | --- |
| A1 | 5123.000000 | 0.038559 | 0.051673 | 1.340095 | 0.093695 |
| A2 | 2885.000000 | 0.049294 | 0.052802 | 1.071161 | 0.378510 |
| A3 | 47064.000000 | 0.038869 | 0.052978 | 1.362995 | 0.059897 |

在这个辅助对照下，A1 高冲突组差异均值由 0.038559 增至 0.051673，比值=1.340095。只对 0.093695 的样本减小了该距离。因此不能声称此算法已经降低对赋权方法的敏感性。此结论限于已声明的等权候选对照。

## 当前最终建议：B，保留 (Q,K)

本轮可以确认普通样本不变、候选整体排名稳定及极端残差降权；这些是算法/内部统计性质。没有充分证据证明典型文本解释或外部评价准确性得到改善。局部重聚合会放大剩余指标的相对权重，最大调整和高冲突组排名需要单独审查，不能将局部系数视为已验证可靠性。

结合 arxiv 较大的单向上调、域内排名变化、辅助赋权敏感性未下降以及尚无独立人工解释改善证据，本阶段建议采用 B：(Q,K)。δ 候选及 Q_robust 单独保留供人工复核，不激活分段 Q_final。部分短噪声案例下调方向可解释，但不能以少数案例覆盖域级不稳定性；不会反向修改第一小问。

原始附件、第一小问和冲突第一阶段的 2182 个已有文件均进行前后 SHA-256 核验。独立检查见 verification.csv。未测试指数惩罚或 lambda；不开始问题一第3小问。
