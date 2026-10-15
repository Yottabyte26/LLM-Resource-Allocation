# PHASE 2C：重复记录、ID 冲突、缺失记录与分类天花板诊断

本轮只诊断，不删除、选择、合并、平均或填补任何记录；02 结果保持不变。未运行 03/04，未计算 CRITIC 或 Q。

## 实际输入与只读核验

| 数据集 | 实际来源文件 | 全量记录数 | SHA-256 一致 |
|---|---|---:|---|
| A1 | `real_attachments/A_data_value/slimpajama_quality_signal_sample.jsonl.xz` | 51230 | True |
| A2 | `real_attachments/A_data_value/slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz` | 17523 | True |
| A3 | `real_attachments/A_data_value/slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz` | 203752 | True |

所有来源逐行 XZ 读取；完整 content 只参与当前记录哈希，不长期保存；临时 SQLite 索引不保存全文，运行后移除。

## 比较口径

- exact duplicate：全部原始 JSON 键值的规范 JSON SHA-256 相同。忽略字典键顺序，保留字段存在性、原键名、字符串、数字类型及来源元数据。不是仅比较 id。
- duplicated id：相同非空字符串 ID 出现至少两次；空/缺失 ID 另计，不把缺失 ID 聚成同一 ID。
- SAME_ID_SAME_DATA：同 ID 的原始 22 质量字段一致，且双方可比较的 content、sub_path、推断/原始 domain 均无差异。
- SAME_ID_DIFFERENT_DATA：上述任何可比较字段不同。原始质量比较使用原标量和 logits，未使用 argmax、截尾或归一化后的值。
- quality-only 口径单独提供 same_id_same_quality / different_quality 列，便于只研究质量字段一致性。
- A2/A3 缺少 content：content_same=NOT_COMPARABLE，不是 True，也不能证明两条文本相同。单侧缺失字段不会强制判成冲突。
- _source_path 和实际 source_file 是溯源信息，不参与 comparable-data 相等判据；完整 exact duplicate 哈希仍包括原始来源字段。
- 别名在质量比较中映射到规范名；同位置 NaN 与 NaN 相等；数值 1 与 1.0、正负零在质量比较中等价。质量 blob 的等同性逐字节检查，不仅依赖哈希。
- max_absolute_difference 仅对不同的有限数值分量计算，不同向量字段计为一个质量字段。缺失/非有限值不可计算有限绝对差的部分另列，无法定义时留空。各指标原单位不同，最大差值不是统一尺度的质量效应。

## 重复和 ID 冲突汇总

| scope | n | unique id | exact 重复组 | exact 多余出现数 | 重复 ID 组 | ID 多余出现数 | 同数据 ID 组 | 不同数据 ID 组 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 51230 | 51230 | 0 | 0 | 0 | 0 | 0 | 0 |
| A2 | 17523 | 17523 | 0 | 0 | 0 | 0 | 0 | 0 |
| A3 | 203752 | 203752 | 0 | 0 | 0 | 0 | 0 | 0 |
| A1+A2+A3 | 272505 | 261086 | 0 | 0 | 11419 | 11419 | 11419 | 0 |

组数指重复键的数量；records 包含重复组的全部成员；excess_records=Σ(组大小−1)，只是计数，不表示删过记录。
同数据组要求组内所有配对均无差异；任何配对有差异则整组记不同数据组。混合组中的同/不同配对数另列，所以组数与配对数不能互换。
id_conflicts.csv 共 0 个不同数据配对，按总体范围只写一次，并保留两端 dataset/source_file/行号/id/sub_path/content 比较结果和质量差异。

| scope | 原始 22 质量字段重复组 | 原始质量字段多余出现数 |
|---|---:|---:|
| A1 | 0 | 0 |
| A2 | 0 | 0 |
| A3 | 1 | 1 |
| A1+A2+A3 | 11420 | 11420 |

质量字段相同不意味着完整记录或文本相同；不同 ID 的此类组单独追踪于 raw_quality_duplicates_different_ids.csv，不自动删除。

## 跨数据集重叠

| 对照 | 左记录数 | 右记录数 | same id | same id+same data | same id+different data | content 不可比较配对 |
|---|---:|---:|---:|---:|---:|---:|
| A1-arxiv vs A2-arxiv | 1419 | 17523 | 1419 | 1419 | 0 | 1419 |
| A1-github vs A3-github | 10000 | 203752 | 10000 | 10000 | 0 | 10000 |

same_id 是交集的 unique ID 数；same/different data 是至少有一个对应配对的 ID 数。
如果一 ID 有两种配对，两个计数可交叠，same_id_both_types 会明确记录；所有笛卡尔配对数也分别给出。
完整原记录字段结构不同，可能没有 exact duplicate，但仍有相同 ID 和相同原始质量字段。未从 A2/A3 移除任何重叠。

## scalar missing 记录诊断

- 缺失标量事件：19。
- 来源记录出现数（dataset+source_file+line_number）：19。
- unique ID：19；unique 完整记录哈希：19；unique 原始质量记录：19。
- 每条记录缺失数量分布：{1: 19}。
- 同时缺 reasoning 与 professionalism：0 条。
- 缺失比例严格 >20%：0 条（本轮阈值 20%）。

缺失诊断重新检查全部 22 个原始字段，在已批准的标量化规则下有限性与标量缺失对应；已逐事件核对 02 的缺失 ID 清单。
未把 qurater 正式拟合前的临时占位当成缺失。所有缺失记录保留，不填补、不删除。

## 分类指标天花板

每个等级的 proportion=count/valid_count。ceiling_rate=score==1 的数量/有限分类标量数量；同时提供除以全部记录数的 rate_all_records。
modernbert 主方案为 argmax/5；fluency 和 ad 为已确认类别顺序下的 argmax；ad 的 1 表示模型判为无广告。
原始 NaN 不属于任何离散等级，单列 missing。下表数值用于判断区分度，不自动改变转换或权重。

| scope | dataset | domain | 指标 | 有效数 | 缺失 | score=1 数 | ceiling_rate |
|---|---|---|---|---:|---:|---:|---:|
| overall | ALL | ALL | modernbert_cleanliness | 272505 | 0 | 46216 | 16.959689% |
| overall | ALL | ALL | modernbert_readability | 272505 | 0 | 46114 | 16.922258% |
| overall | ALL | ALL | modernbert_reasoning | 272492 | 13 | 32377 | 11.881817% |
| overall | ALL | ALL | modernbert_professionalism | 272499 | 6 | 124369 | 45.640167% |
| overall | ALL | ALL | fluency_en | 272505 | 0 | 171224 | 62.833343% |
| overall | ALL | ALL | ad_en | 272505 | 0 | 268924 | 98.685896% |
| dataset | A1 | ALL | modernbert_cleanliness | 51230 | 0 | 14979 | 29.238727% |
| dataset | A1 | ALL | modernbert_readability | 51230 | 0 | 15239 | 29.746242% |
| dataset | A1 | ALL | modernbert_reasoning | 51217 | 13 | 4615 | 9.010680% |
| dataset | A1 | ALL | modernbert_professionalism | 51225 | 5 | 9092 | 17.749146% |
| dataset | A1 | ALL | fluency_en | 51230 | 0 | 40111 | 78.295920% |
| dataset | A1 | ALL | ad_en | 51230 | 0 | 48300 | 94.280695% |
| domain | ALL | arxiv | modernbert_cleanliness | 18942 | 0 | 18782 | 99.155316% |
| domain | ALL | arxiv | modernbert_readability | 18942 | 0 | 17315 | 91.410622% |
| domain | ALL | arxiv | modernbert_reasoning | 18942 | 0 | 18809 | 99.297857% |
| domain | ALL | arxiv | modernbert_professionalism | 18942 | 0 | 18853 | 99.530145% |
| domain | ALL | arxiv | fluency_en | 18942 | 0 | 18792 | 99.208109% |
| domain | ALL | arxiv | ad_en | 18942 | 0 | 18926 | 99.915532% |
| dataset_domain | A1 | arxiv | modernbert_cleanliness | 1419 | 0 | 1411 | 99.436223% |
| dataset_domain | A1 | arxiv | modernbert_readability | 1419 | 0 | 1302 | 91.754757% |
| dataset_domain | A1 | arxiv | modernbert_reasoning | 1419 | 0 | 1413 | 99.577167% |
| dataset_domain | A1 | arxiv | modernbert_professionalism | 1419 | 0 | 1413 | 99.577167% |
| dataset_domain | A1 | arxiv | fluency_en | 1419 | 0 | 1410 | 99.365751% |
| dataset_domain | A1 | arxiv | ad_en | 1419 | 0 | 1419 | 100.000000% |
| domain | ALL | book | modernbert_cleanliness | 171 | 0 | 142 | 83.040936% |
| domain | ALL | book | modernbert_readability | 171 | 0 | 100 | 58.479532% |
| domain | ALL | book | modernbert_reasoning | 171 | 0 | 89 | 52.046784% |
| domain | ALL | book | modernbert_professionalism | 171 | 0 | 15 | 8.771930% |
| domain | ALL | book | fluency_en | 171 | 0 | 124 | 72.514620% |
| domain | ALL | book | ad_en | 171 | 0 | 169 | 98.830409% |
| dataset_domain | A1 | book | modernbert_cleanliness | 171 | 0 | 142 | 83.040936% |
| dataset_domain | A1 | book | modernbert_readability | 171 | 0 | 100 | 58.479532% |
| dataset_domain | A1 | book | modernbert_reasoning | 171 | 0 | 89 | 52.046784% |
| dataset_domain | A1 | book | modernbert_professionalism | 171 | 0 | 15 | 8.771930% |
| dataset_domain | A1 | book | fluency_en | 171 | 0 | 124 | 72.514620% |
| dataset_domain | A1 | book | ad_en | 171 | 0 | 169 | 98.830409% |
| domain | ALL | c4 | modernbert_cleanliness | 10000 | 0 | 4206 | 42.060000% |
| domain | ALL | c4 | modernbert_readability | 10000 | 0 | 3385 | 33.850000% |
| domain | ALL | c4 | modernbert_reasoning | 10000 | 0 | 330 | 3.300000% |
| domain | ALL | c4 | modernbert_professionalism | 10000 | 0 | 195 | 1.950000% |
| domain | ALL | c4 | fluency_en | 10000 | 0 | 8840 | 88.400000% |
| domain | ALL | c4 | ad_en | 10000 | 0 | 7712 | 77.120000% |
| dataset_domain | A1 | c4 | modernbert_cleanliness | 10000 | 0 | 4206 | 42.060000% |
| dataset_domain | A1 | c4 | modernbert_readability | 10000 | 0 | 3385 | 33.850000% |
| dataset_domain | A1 | c4 | modernbert_reasoning | 10000 | 0 | 330 | 3.300000% |
| dataset_domain | A1 | c4 | modernbert_professionalism | 10000 | 0 | 195 | 1.950000% |
| dataset_domain | A1 | c4 | fluency_en | 10000 | 0 | 8840 | 88.400000% |
| dataset_domain | A1 | c4 | ad_en | 10000 | 0 | 7712 | 77.120000% |
| domain | ALL | commoncrawl | modernbert_cleanliness | 9640 | 0 | 3697 | 38.350622% |
| domain | ALL | commoncrawl | modernbert_readability | 9640 | 0 | 5847 | 60.653527% |
| domain | ALL | commoncrawl | modernbert_reasoning | 9640 | 0 | 1092 | 11.327801% |
| domain | ALL | commoncrawl | modernbert_professionalism | 9636 | 4 | 517 | 5.365297% |
| domain | ALL | commoncrawl | fluency_en | 9640 | 0 | 8415 | 87.292531% |
| domain | ALL | commoncrawl | ad_en | 9640 | 0 | 9060 | 93.983402% |
| dataset_domain | A1 | commoncrawl | modernbert_cleanliness | 9640 | 0 | 3697 | 38.350622% |
| dataset_domain | A1 | commoncrawl | modernbert_readability | 9640 | 0 | 5847 | 60.653527% |
| dataset_domain | A1 | commoncrawl | modernbert_reasoning | 9640 | 0 | 1092 | 11.327801% |
| dataset_domain | A1 | commoncrawl | modernbert_professionalism | 9636 | 4 | 517 | 5.365297% |
| dataset_domain | A1 | commoncrawl | fluency_en | 9640 | 0 | 8415 | 87.292531% |
| dataset_domain | A1 | commoncrawl | ad_en | 9640 | 0 | 9060 | 93.983402% |
| domain | ALL | github | modernbert_cleanliness | 213752 | 0 | 14551 | 6.807422% |
| domain | ALL | github | modernbert_readability | 213752 | 0 | 15551 | 7.275254% |
| domain | ALL | github | modernbert_reasoning | 213752 | 0 | 10839 | 5.070830% |
| domain | ALL | github | modernbert_professionalism | 213751 | 1 | 102705 | 48.048898% |
| domain | ALL | github | fluency_en | 213752 | 0 | 119321 | 55.822168% |
| domain | ALL | github | ad_en | 213752 | 0 | 213088 | 99.689360% |
| dataset_domain | A1 | github | modernbert_cleanliness | 10000 | 0 | 685 | 6.850000% |
| dataset_domain | A1 | github | modernbert_readability | 10000 | 0 | 689 | 6.890000% |
| dataset_domain | A1 | github | modernbert_reasoning | 10000 | 0 | 473 | 4.730000% |
| dataset_domain | A1 | github | modernbert_professionalism | 10000 | 0 | 4868 | 48.680000% |
| dataset_domain | A1 | github | fluency_en | 10000 | 0 | 5590 | 55.900000% |
| dataset_domain | A1 | github | ad_en | 10000 | 0 | 9971 | 99.710000% |
| domain | ALL | stackexchange | modernbert_cleanliness | 10000 | 0 | 2210 | 22.100000% |
| domain | ALL | stackexchange | modernbert_readability | 10000 | 0 | 860 | 8.600000% |
| domain | ALL | stackexchange | modernbert_reasoning | 10000 | 0 | 865 | 8.650000% |
| domain | ALL | stackexchange | modernbert_professionalism | 10000 | 0 | 1906 | 19.060000% |
| domain | ALL | stackexchange | fluency_en | 10000 | 0 | 8436 | 84.360000% |
| domain | ALL | stackexchange | ad_en | 10000 | 0 | 9995 | 99.950000% |
| dataset_domain | A1 | stackexchange | modernbert_cleanliness | 10000 | 0 | 2210 | 22.100000% |
| dataset_domain | A1 | stackexchange | modernbert_readability | 10000 | 0 | 860 | 8.600000% |
| dataset_domain | A1 | stackexchange | modernbert_reasoning | 10000 | 0 | 865 | 8.650000% |
| dataset_domain | A1 | stackexchange | modernbert_professionalism | 10000 | 0 | 1906 | 19.060000% |
| dataset_domain | A1 | stackexchange | fluency_en | 10000 | 0 | 8436 | 84.360000% |
| dataset_domain | A1 | stackexchange | ad_en | 10000 | 0 | 9995 | 99.950000% |
| domain | ALL | wikipedia | modernbert_cleanliness | 10000 | 0 | 2628 | 26.280000% |
| domain | ALL | wikipedia | modernbert_readability | 10000 | 0 | 3056 | 30.560000% |
| domain | ALL | wikipedia | modernbert_reasoning | 9987 | 13 | 353 | 3.534595% |
| domain | ALL | wikipedia | modernbert_professionalism | 9999 | 1 | 178 | 1.780178% |
| domain | ALL | wikipedia | fluency_en | 10000 | 0 | 7296 | 72.960000% |
| domain | ALL | wikipedia | ad_en | 10000 | 0 | 9974 | 99.740000% |
| dataset_domain | A1 | wikipedia | modernbert_cleanliness | 10000 | 0 | 2628 | 26.280000% |
| dataset_domain | A1 | wikipedia | modernbert_readability | 10000 | 0 | 3056 | 30.560000% |
| dataset_domain | A1 | wikipedia | modernbert_reasoning | 9987 | 13 | 353 | 3.534595% |
| dataset_domain | A1 | wikipedia | modernbert_professionalism | 9999 | 1 | 178 | 1.780178% |
| dataset_domain | A1 | wikipedia | fluency_en | 10000 | 0 | 7296 | 72.960000% |
| dataset_domain | A1 | wikipedia | ad_en | 10000 | 0 | 9974 | 99.740000% |
| dataset | A2 | ALL | modernbert_cleanliness | 17523 | 0 | 17371 | 99.132569% |
| dataset | A2 | ALL | modernbert_readability | 17523 | 0 | 16013 | 91.382754% |
| dataset | A2 | ALL | modernbert_reasoning | 17523 | 0 | 17396 | 99.275238% |
| dataset | A2 | ALL | modernbert_professionalism | 17523 | 0 | 17440 | 99.526337% |
| dataset | A2 | ALL | fluency_en | 17523 | 0 | 17382 | 99.195343% |
| dataset | A2 | ALL | ad_en | 17523 | 0 | 17507 | 99.908691% |
| dataset_domain | A2 | arxiv | modernbert_cleanliness | 17523 | 0 | 17371 | 99.132569% |
| dataset_domain | A2 | arxiv | modernbert_readability | 17523 | 0 | 16013 | 91.382754% |
| dataset_domain | A2 | arxiv | modernbert_reasoning | 17523 | 0 | 17396 | 99.275238% |
| dataset_domain | A2 | arxiv | modernbert_professionalism | 17523 | 0 | 17440 | 99.526337% |
| dataset_domain | A2 | arxiv | fluency_en | 17523 | 0 | 17382 | 99.195343% |
| dataset_domain | A2 | arxiv | ad_en | 17523 | 0 | 17507 | 99.908691% |
| dataset | A3 | ALL | modernbert_cleanliness | 203752 | 0 | 13866 | 6.805332% |
| dataset | A3 | ALL | modernbert_readability | 203752 | 0 | 14862 | 7.294162% |
| dataset | A3 | ALL | modernbert_reasoning | 203752 | 0 | 10366 | 5.087557% |
| dataset | A3 | ALL | modernbert_professionalism | 203751 | 1 | 97837 | 48.017924% |
| dataset | A3 | ALL | fluency_en | 203752 | 0 | 113731 | 55.818348% |
| dataset | A3 | ALL | ad_en | 203752 | 0 | 203117 | 99.688347% |
| dataset_domain | A3 | github | modernbert_cleanliness | 203752 | 0 | 13866 | 6.805332% |
| dataset_domain | A3 | github | modernbert_readability | 203752 | 0 | 14862 | 7.294162% |
| dataset_domain | A3 | github | modernbert_reasoning | 203752 | 0 | 10366 | 5.087557% |
| dataset_domain | A3 | github | modernbert_professionalism | 203751 | 1 | 97837 | 48.017924% |
| dataset_domain | A3 | github | fluency_en | 203752 | 0 | 113731 | 55.818348% |
| dataset_domain | A3 | github | ad_en | 203752 | 0 | 203117 | 99.688347% |

scope=domain 是全量合并领域；scope=dataset_domain 区分 A1 中领域与扩展领域；总体也单列供核验。
相同记录出现会影响合并频率，当前全部记录保留；天花板统计不是对分类器正确率的评估。

## 输出与阶段边界

- duplicate_summary.csv / exact_duplicates.csv / id_conflicts.csv：全部组和冲突诊断（无事件时 CSV 仍有表头）。
- cross_dataset_overlap.csv：两组跨数据集对照。
- nan_record_diagnosis.csv：每条实际缺失来源记录及缺失字段/比例/双缺失标记。
- classification_level_distribution.csv：所有离散等级的数量和比例，包括零占比等级。
- classification_ceiling_summary.csv：各数据集、各领域的 ceiling_rate。
- cleaning_diagnosis_input_manifest.csv / prior_artifacts_unchanged.csv：原附件哈希及 02 输出未变核验。

重复、冲突和天花板仅为诊断；记录处理策略仍待后续决定。本轮停止，不运行 03，不计算 CRITIC 或 Q。
