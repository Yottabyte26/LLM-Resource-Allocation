# 向量语义诊断：PHASE 2

本轮全量读取 A1/A2/A3，仅检查结构、数值和来源证据。不生成质量特征、CRITIC 或 Q。

Softmax 数值有效不等于原值必为 logits，也不证明概率经过校准。原始 NaN 向量无法转换，单独记录。

## 实际输入

| 数据集 | 文件 | 记录数 | 来源领域计数 | 原件 SHA-256 一致 |
|---|---|---:|---|---|
| A1 | `real_attachments/A_data_value/slimpajama_quality_signal_sample.jsonl.xz` | 51230 | {"arxiv": 1419, "book": 171, "c4": 10000, "commoncrawl": 9640, "github": 10000, "stackexchange": 10000, "wikipedia": 10000} | True |
| A2 | `real_attachments/A_data_value/slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz` | 17523 | {"arxiv": 17523} | True |
| A3 | `real_attachments/A_data_value/slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz` | 203752 | {"github": 203752} | True |

终止标点字段别名集中定义在 configs/quality_metrics.yaml；不改原文件。
A1 采用原 _source_domain；A2/A3 采用实际文件名 arxiv_/github_ 前缀。规则、领域计数和原始别名使用数均写入清单。

## 字段诊断与状态

| 数据集 | 字段 | 实际长度分布 | 实际值域（有限值） | 含负数 | 原始 sum≈1 | 疑似 logits | Softmax 有效 | 类别顺序 | 推荐转换 | 允许下一阶段 | status |
|---|---|---|---|---|---:|---|---|---|---|---|---|
| A1 | fineweb_edu | {"1": 51230} | [-0.652137815952, 4.3755698204] | True | 0.183486% | NOT_ESTABLISHED | NOT_APPLICABLE | NOT_APPLICABLE | 可直接取 v1 | True | READY_FOR_EXTRACTION |
| A1 | qurater | {"4": 51230} | [-8.375, 12.8203125] | True | 0.007808% | NOT_ESTABLISHED | NOT_APPLICABLE | 类别顺序未确认（本地）；上游定义另见证据表 | 保留四项诊断；不得自动平均，需确认同尺度与方向 | False | NEEDS_CONFIRMATION |
| A1 | modernbert_cleanliness | {"6": 51230} | [-21.015625, 20.8125] | True | 0.003904% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |
| A1 | modernbert_readability | {"6": 51230} | [-13.296875, 16.234375] | True | 0.019520% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |
| A1 | modernbert_reasoning | {"6": 51230} | [-10.3515625, 17.953125] | True | 0.064432% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |
| A1 | modernbert_professionalism | {"6": 51230} | [-11, 18.484375] | True | 0.011713% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |
| A1 | fluency_en | {"2": 51230} | [-3.921875, 3.716796875] | True | 0.017568% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |
| A1 | ad_en | {"2": 51230} | [-3.97265625, 4.6796875] | True | 0.011712% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |
| A2 | fineweb_edu | {"1": 17523} | [-0.137066423893, 4.34501218796] | True | 0.039947% | NOT_ESTABLISHED | NOT_APPLICABLE | NOT_APPLICABLE | 可直接取 v1 | True | READY_FOR_EXTRACTION |
| A2 | qurater | {"4": 17523} | [-5.296875, 12.53125] | True | 0.000000% | NOT_ESTABLISHED | NOT_APPLICABLE | 类别顺序未确认（本地）；上游定义另见证据表 | 保留四项诊断；不得自动平均，需确认同尺度与方向 | False | NEEDS_CONFIRMATION |
| A2 | modernbert_cleanliness | {"6": 17523} | [-16.8125, 16.71875] | True | 0.000000% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |
| A2 | modernbert_readability | {"6": 17523} | [-10.109375, 12.5390625] | True | 0.000000% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |
| A2 | modernbert_reasoning | {"6": 17523} | [-8.375, 15.703125] | True | 0.000000% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |
| A2 | modernbert_professionalism | {"6": 17523} | [-6.75390625, 13.953125] | True | 0.000000% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |
| A2 | fluency_en | {"2": 17523} | [-3.783203125, 3.44921875] | True | 0.000000% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |
| A2 | ad_en | {"2": 17523} | [-3.9296875, 4.53515625] | True | 0.000000% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |
| A3 | fineweb_edu | {"1": 203752} | [-0.652137815952, 4.69415521622] | True | 0.203679% | NOT_ESTABLISHED | NOT_APPLICABLE | NOT_APPLICABLE | 可直接取 v1 | True | READY_FOR_EXTRACTION |
| A3 | qurater | {"4": 203752} | [-7.66796875, 11.1796875] | True | 0.017669% | NOT_ESTABLISHED | NOT_APPLICABLE | 类别顺序未确认（本地）；上游定义另见证据表 | 保留四项诊断；不得自动平均，需确认同尺度与方向 | False | NEEDS_CONFIRMATION |
| A3 | modernbert_cleanliness | {"6": 203752} | [-19.8125, 21.65625] | True | 0.016687% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |
| A3 | modernbert_readability | {"6": 203752} | [-12.1015625, 14.546875] | True | 0.045153% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |
| A3 | modernbert_reasoning | {"6": 203752} | [-9.4296875, 18.421875] | True | 0.076073% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |
| A3 | modernbert_professionalism | {"6": 203752} | [-8.7890625, 16.09375] | True | 0.009816% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |
| A3 | fluency_en | {"2": 203752} | [-3.650390625, 3.392578125] | True | 0.000982% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |
| A3 | ad_en | {"2": 203752} | [-3.91015625, 4.61328125] | True | 0.022576% | True | True | 类别顺序未确认（本地）；上游定义另见证据表 | 仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用 | False | NEEDS_CONFIRMATION |

缺失、非法与非有限值分开统计；sum≈1 比例仅使用合法、有限、预期长度的向量。
仅 fineweb_edu 标为可提取；其他字段均待确认。READY_FOR_EXTRACTION 不代表允许运行后续脚本。

## 原始分布与位置统计

| 数据集 | 字段 | 位置 | min | max | mean | median | std |
|---|---|---:|---:|---:|---:|---:|---:|
| A1 | fineweb_edu | 0 | -0.652137816 | 4.37556982 | 1.30556072 | 1.21487457 | 0.647202775 |
| A1 | qurater | 0 | -7.74609375 | 9.4921875 | -1.01889199 | -1.62109375 | 2.59152267 |
| A1 | qurater | 1 | -8.375 | 10.875 | 0.724461698 | 0.634765625 | 2.70045582 |
| A1 | qurater | 2 | -6.73828125 | 7.78125 | -0.307493064 | -0.567382812 | 2.06094941 |
| A1 | qurater | 3 | -7.828125 | 12.8203125 | 1.23233112 | 1.01074219 | 2.99130033 |
| A1 | modernbert_cleanliness | 0 | -16.03125 | 0.796386719 | -7.7710951 | -7.9609375 | 2.41113172 |
| A1 | modernbert_cleanliness | 1 | -15.34375 | 20.8125 | -4.07791033 | -5.4765625 | 5.46470756 |
| A1 | modernbert_cleanliness | 2 | -7.65234375 | 17.625 | 1.97439409 | 1.24316406 | 4.81963161 |
| A1 | modernbert_cleanliness | 3 | -13.015625 | 11.8203125 | -1.91039371 | -2.50195312 | 5.07631348 |
| A1 | modernbert_cleanliness | 4 | -8.5625 | 16.96875 | 5.05946033 | 5.5703125 | 4.32769953 |
| A1 | modernbert_cleanliness | 5 | -21.015625 | 18.609375 | 1.22006839 | 1.40039062 | 8.38608769 |
| A1 | modernbert_readability | 0 | -13.296875 | 6.625 | -6.63596581 | -6.8984375 | 1.8247274 |
| A1 | modernbert_readability | 1 | -9.5234375 | 16.234375 | -1.55257644 | -1.83007812 | 3.5070595 |
| A1 | modernbert_readability | 2 | -6.890625 | 6.07421875 | -0.876805381 | -1.09326172 | 2.29138231 |
| A1 | modernbert_readability | 3 | -5.59765625 | 4.8671875 | 0.3204236 | 0.579101562 | 1.65823758 |
| A1 | modernbert_readability | 4 | -4.02734375 | 10.1640625 | 4.14074656 | 4.0859375 | 2.15879413 |
| A1 | modernbert_readability | 5 | -10.90625 | 15.3046875 | 2.3597449 | 2.30859375 | 4.81790426 |
| A1 | modernbert_reasoning | 0 | -10.3515625 | 17.953125 | -0.541858232 | -1.32421875 | 5.16488306 |
| A1 | modernbert_reasoning | 1 | -9.9375 | 14.046875 | 4.41171333 | 5.3671875 | 5.09309758 |
| A1 | modernbert_reasoning | 2 | -6.6875 | 11.375 | 3.36069967 | 3.50585938 | 2.80901314 |
| A1 | modernbert_reasoning | 3 | -8.515625 | 6.34765625 | -1.12317912 | -1.36816406 | 2.9333091 |
| A1 | modernbert_reasoning | 4 | -8.265625 | 8.1171875 | -2.36960859 | -3.16796875 | 3.34564464 |
| A1 | modernbert_reasoning | 5 | -9.0234375 | 15.703125 | -3.07206728 | -4.65625 | 4.14938116 |
| A1 | modernbert_professionalism | 0 | -7.1796875 | -0.903320312 | -4.22789356 | -4.25 | 0.82506722 |
| A1 | modernbert_professionalism | 1 | -7.73828125 | 18.484375 | 0.919956047 | -0.355712891 | 5.58532943 |
| A1 | modernbert_professionalism | 2 | -5.61328125 | 9.0859375 | 1.91781748 | 2.52929688 | 3.6977412 |
| A1 | modernbert_professionalism | 3 | -6.37890625 | 6.30078125 | 0.830869988 | 1.13085938 | 2.51639464 |
| A1 | modernbert_professionalism | 4 | -8.8203125 | 7.375 | 0.779623462 | 1.27929688 | 3.8019396 |
| A1 | modernbert_professionalism | 5 | -11 | 14 | -1.13460154 | -2.87304688 | 5.69155379 |
| A1 | fluency_en | 0 | -3.921875 | 3.12109375 | -1.12346213 | -0.981445312 | 1.52588066 |
| A1 | fluency_en | 1 | -3.1328125 | 3.71679688 | 1.30158648 | 1.2109375 | 1.30560615 |
| A1 | ad_en | 0 | -3.97265625 | 3.30664062 | -2.16406186 | -2.37695312 | 1.14737705 |
| A1 | ad_en | 1 | -3.9609375 | 4.6796875 | 2.47074208 | 2.765625 | 1.45073473 |
| A2 | fineweb_edu | 0 | -0.137066424 | 4.34501219 | 2.14826465 | 2.14141178 | 0.618343163 |
| A2 | qurater | 0 | -4.984375 | 6.796875 | 2.74302726 | 2.93554688 | 1.48383035 |
| A2 | qurater | 1 | -1.59375 | 11.578125 | 7.46523671 | 7.64453125 | 1.60482393 |
| A2 | qurater | 2 | -5.296875 | 6.69140625 | 1.93039783 | 2.07226562 | 1.63326245 |
| A2 | qurater | 3 | -3.94335938 | 12.53125 | 5.84028352 | 5.94921875 | 1.76599533 |
| A2 | modernbert_cleanliness | 0 | -13.4609375 | -0.841308594 | -3.20835404 | -3.03515625 | 1.01622457 |
| A2 | modernbert_cleanliness | 1 | -10.9921875 | 11.1171875 | 0.365612165 | 0.551757812 | 1.29400572 |
| A2 | modernbert_cleanliness | 2 | -5.0703125 | 12.78125 | -1.9569758 | -2.02929688 | 0.778409309 |
| A2 | modernbert_cleanliness | 3 | -11.390625 | 4.99609375 | -5.62977425 | -5.6640625 | 1.12987306 |
| A2 | modernbert_cleanliness | 4 | -5.35546875 | 10.46875 | -0.646329769 | -0.792480469 | 0.968245317 |
| A2 | modernbert_cleanliness | 5 | -16.8125 | 16.71875 | 8.04520752 | 8.09375 | 1.87375736 |
| A2 | modernbert_readability | 0 | -10.109375 | -1.14648438 | -5.71216496 | -5.6484375 | 0.540184763 |
| A2 | modernbert_readability | 1 | -6.96875 | 3.70507812 | -2.16346361 | -2.09765625 | 0.856108967 |
| A2 | modernbert_readability | 2 | -4.984375 | 1.86816406 | -2.27026521 | -2.26171875 | 0.50277117 |
| A2 | modernbert_readability | 3 | -4.17578125 | 1.98535156 | -1.45700821 | -1.44824219 | 0.554492755 |
| A2 | modernbert_readability | 4 | 0.0725097656 | 8.1015625 | 3.6362186 | 3.62304688 | 0.587627659 |
| A2 | modernbert_readability | 5 | -4.73828125 | 12.5390625 | 4.86054033 | 4.796875 | 1.47620761 |
| A2 | modernbert_reasoning | 0 | -8.375 | 12.109375 | -5.73711431 | -5.86328125 | 1.1377634 |
| A2 | modernbert_reasoning | 1 | -8.3046875 | 9.6484375 | -3.78869269 | -3.94726562 | 1.24013444 |
| A2 | modernbert_reasoning | 2 | -6.6875 | 5.77734375 | -4.11716748 | -4.26953125 | 1.04254659 |
| A2 | modernbert_reasoning | 3 | -6.68359375 | 3.0546875 | -2.14719841 | -2.234375 | 0.543571608 |
| A2 | modernbert_reasoning | 4 | -7.1171875 | 5.76953125 | 1.82965819 | 1.86035156 | 0.834367046 |
| A2 | modernbert_reasoning | 5 | -6.67578125 | 15.703125 | 11.5332427 | 11.90625 | 2.14090529 |
| A2 | modernbert_professionalism | 0 | -5.49609375 | -0.884277344 | -2.68748392 | -2.68164062 | 0.697304261 |
| A2 | modernbert_professionalism | 1 | -6.75390625 | 5.9453125 | -3.49536105 | -3.54882812 | 0.521119886 |
| A2 | modernbert_professionalism | 2 | -5.18359375 | 4.21484375 | -3.53324942 | -3.65429688 | 0.652902742 |
| A2 | modernbert_professionalism | 3 | -6.53125 | 2.35351562 | -4.60715175 | -4.7421875 | 0.855278387 |
| A2 | modernbert_professionalism | 4 | -2.1015625 | 5.45703125 | 2.36166885 | 2.37304688 | 0.366610283 |
| A2 | modernbert_professionalism | 5 | -5.265625 | 13.953125 | 11.5117963 | 11.8203125 | 1.51336246 |
| A2 | fluency_en | 0 | -3.78320312 | 2.26757812 | -2.51711202 | -2.70703125 | 0.819550548 |
| A2 | fluency_en | 1 | -1.77246094 | 3.44921875 | 2.3619321 | 2.5078125 | 0.671173438 |
| A2 | ad_en | 0 | -3.9296875 | 1.01464844 | -2.89234808 | -3.11328125 | 0.678365289 |
| A2 | ad_en | 1 | -1.16699219 | 4.53515625 | 3.25276607 | 3.57226562 | 0.916861808 |
| A3 | fineweb_edu | 0 | -0.652137816 | 4.69415522 | 1.14246719 | 1.13994461 | 0.460191638 |
| A3 | qurater | 0 | -7.66796875 | 5.4453125 | -2.95212823 | -3.01171875 | 1.10985935 |
| A3 | qurater | 1 | -5.7265625 | 8.28125 | 1.90606355 | 2.0859375 | 2.04930329 |
| A3 | qurater | 2 | -5.76171875 | 6.05859375 | -1.87906592 | -1.90625 | 1.18440293 |
| A3 | qurater | 3 | -7.2109375 | 11.1796875 | 0.284664731 | 0.253173828 | 1.99023623 |
| A3 | modernbert_cleanliness | 0 | -14.453125 | 1.54101562 | -5.67832725 | -5.8125 | 2.16000496 |
| A3 | modernbert_cleanliness | 1 | -14.7890625 | 21.65625 | 2.91452921 | 2.27929688 | 6.37446002 |
| A3 | modernbert_cleanliness | 2 | -5.58984375 | 17 | 6.5610568 | 6.3984375 | 2.85282447 |
| A3 | modernbert_cleanliness | 3 | -12.3359375 | 11.734375 | -2.61110902 | -3.24609375 | 2.53436159 |
| A3 | modernbert_cleanliness | 4 | -9.03125 | 16.875 | -0.313578006 | -0.555664062 | 3.10547931 |
| A3 | modernbert_cleanliness | 5 | -19.8125 | 17.328125 | -2.48473358 | -1.54882812 | 5.04596393 |
| A3 | modernbert_readability | 0 | -12.1015625 | 7.3515625 | -4.61875984 | -5.0703125 | 2.39379069 |
| A3 | modernbert_readability | 1 | -8.640625 | 13.8125 | 0.839843589 | 0.96484375 | 2.33943498 |
| A3 | modernbert_readability | 2 | -6.421875 | 5.57421875 | 1.31999066 | 1.51757812 | 1.56598865 |
| A3 | modernbert_readability | 3 | -4.375 | 5.91796875 | 0.922413726 | 1.03613281 | 0.96958649 |
| A3 | modernbert_readability | 4 | -3.640625 | 9.703125 | 2.35407083 | 2.33398438 | 1.27617177 |
| A3 | modernbert_readability | 5 | -10.7421875 | 14.546875 | -1.05237501 | -1.48828125 | 3.06969973 |
| A3 | modernbert_reasoning | 0 | -8.890625 | 18.421875 | 3.50126804 | 3.18359375 | 4.45138344 |
| A3 | modernbert_reasoning | 1 | -8.5 | 14.09375 | 4.39820378 | 4.46484375 | 2.73324355 |
| A3 | modernbert_reasoning | 2 | -4.5859375 | 10.078125 | 2.40843071 | 2.44335938 | 1.65283845 |
| A3 | modernbert_reasoning | 3 | -9.1171875 | 5.38671875 | -2.44689584 | -2.22851562 | 2.10830977 |
| A3 | modernbert_reasoning | 4 | -8.90625 | 6.875 | -3.37154489 | -3.6015625 | 1.93912004 |
| A3 | modernbert_reasoning | 5 | -9.4296875 | 12.640625 | -3.8687577 | -4.52734375 | 2.36202003 |
| A3 | modernbert_professionalism | 0 | -7.00390625 | -0.698730469 | -4.22945197 | -4.2734375 | 0.587423698 |
| A3 | modernbert_professionalism | 1 | -7.09765625 | 16.09375 | -3.1106008 | -3.76953125 | 2.25155399 |
| A3 | modernbert_professionalism | 2 | -6.0078125 | 7.828125 | -1.84361593 | -2.5234375 | 2.42399275 |
| A3 | modernbert_professionalism | 3 | -6.27734375 | 5.69921875 | -0.655342373 | -0.307861328 | 2.30503054 |
| A3 | modernbert_professionalism | 4 | -6.6953125 | 8.0546875 | 4.30352822 | 4.6171875 | 1.68127225 |
| A3 | modernbert_professionalism | 5 | -8.7890625 | 14.421875 | 4.63162598 | 5.0234375 | 4.8383533 |
| A3 | fluency_en | 0 | -3.65039062 | 3.1953125 | 0.0359224595 | 0.124267578 | 0.941626093 |
| A3 | fluency_en | 1 | -2.94140625 | 3.39257812 | 0.362621443 | 0.364746094 | 0.762751469 |
| A3 | ad_en | 0 | -3.91015625 | 2.13671875 | -2.30189408 | -2.37890625 | 0.687774457 |
| A3 | ad_en | 1 | -2.53125 | 4.61328125 | 2.75207877 | 2.90039062 | 0.912718806 |

| 数据集 | 字段 | 分布量 | min | P25 | median | mean | P75 | max |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A1 | fineweb_edu | raw_sum | -0.652137816 | 0.882679358 | 1.21487457 | 1.30556072 | 1.64116997 | 4.37556982 |
| A1 | qurater | raw_sum | -27.875 | -4.71546936 | -0.174530029 | 0.630407765 | 5.14453125 | 31.1328125 |
| A1 | modernbert_cleanliness | raw_sum | -18.3632812 | -7.75720215 | -5.7578125 | -5.50547634 | -3.41415405 | 8.46972656 |
| A1 | modernbert_readability | raw_sum | -9.05664062 | -3.40994263 | -2.45898438 | -2.24443258 | -1.2646637 | 6.40161133 |
| A1 | modernbert_reasoning | raw_sum | -3.91308594 | -0.409179688 | 0.631103516 | 0.665699787 | 1.68566895 | 7.36425781 |
| A1 | modernbert_professionalism | raw_sum | -4.57983398 | -1.64770508 | -0.9453125 | -0.914228117 | -0.198730469 | 3.703125 |
| A1 | fluency_en | raw_sum | -0.815429688 | -0.106445312 | 0.158569336 | 0.178124348 | 0.467557907 | 1.39967346 |
| A1 | fluency_en | l1_minus_l0 | -6.22460938 | 0.264694214 | 2.17700195 | 2.42504861 | 4.88085938 | 7.53320312 |
| A1 | ad_en | raw_sum | -1.05761719 | 0.01171875 | 0.33984375 | 0.306680223 | 0.615234375 | 1.36132812 |
| A1 | ad_en | l1_minus_l0 | -7.13867188 | 3.12207031 | 5.15820312 | 4.63480394 | 6.67578125 | 8.53710938 |
| A2 | fineweb_edu | raw_sum | -0.137066424 | 1.71245712 | 2.14141178 | 2.14826465 | 2.57301331 | 4.34501219 |
| A2 | qurater | raw_sum | -14.9140625 | 15.9628906 | 18.7958984 | 17.9789453 | 21.2382812 | 31.1328125 |
| A2 | modernbert_cleanliness | raw_sum | -14.3271484 | -3.31469727 | -2.83911133 | -3.03061417 | -2.40014648 | 1.82421875 |
| A2 | modernbert_readability | raw_sum | -7.953125 | -3.43945312 | -3.08789062 | -3.10614306 | -2.75024414 | 1.25628662 |
| A2 | modernbert_reasoning | raw_sum | -4.13623047 | -2.77075195 | -2.44140625 | -2.42727205 | -2.11914062 | 2.39550781 |
| A2 | modernbert_professionalism | raw_sum | -4.04589844 | -0.69921875 | -0.412109375 | -0.449781037 | -0.115234375 | 1.01269531 |
| A2 | fluency_en | raw_sum | -0.818359375 | -0.298828125 | -0.2109375 | -0.155179921 | -0.0703125 | 0.847473145 |
| A2 | fluency_en | l1_minus_l0 | -4.04003906 | 4.20019531 | 5.2109375 | 4.87904411 | 5.96484375 | 7.23242188 |
| A2 | ad_en | raw_sum | -0.655761719 | 0.18359375 | 0.423828125 | 0.360417993 | 0.5703125 | 1.03710938 |
| A2 | ad_en | l1_minus_l0 | -2.18164062 | 5.23242188 | 6.68945312 | 6.14511414 | 7.38671875 | 8.46484375 |
| A3 | fineweb_edu | raw_sum | -0.652137816 | 0.88405323 | 1.13994461 | 1.14246719 | 1.39781344 | 4.69415522 |
| A3 | qurater | raw_sum | -24.2265625 | -5.92248535 | -2.38082886 | -2.64046587 | 0.78515625 | 25.7128906 |
| A3 | modernbert_cleanliness | raw_sum | -13.2111816 | -3.50927734 | -1.86572266 | -1.61216185 | 0.0293884277 | 9.765625 |
| A3 | modernbert_readability | raw_sum | -7.09472656 | -1.35987091 | -0.26171875 | -0.234816035 | 0.860992432 | 6.78747559 |
| A3 | modernbert_reasoning | raw_sum | -4.8359375 | -0.270874023 | 0.557617188 | 0.620704096 | 1.4375 | 7.18359375 |
| A3 | modernbert_professionalism | raw_sum | -4.1081543 | -1.45751953 | -0.970703125 | -0.903856874 | -0.421875 | 3.43554688 |
| A3 | fluency_en | raw_sum | -0.856445312 | 0.264160156 | 0.463623047 | 0.398543903 | 0.581542969 | 1.20336914 |
| A3 | fluency_en | l1_minus_l0 | -6.13671875 | -0.849441528 | 0.244384766 | 0.326698984 | 1.41845703 | 7.04296875 |
| A3 | ad_en | raw_sum | -0.699584961 | 0.245117188 | 0.48828125 | 0.450184687 | 0.673828125 | 1.36132812 |
| A3 | ad_en | l1_minus_l0 | -4.66796875 | 4.0390625 | 5.28125 | 5.05397285 | 6.28125 | 8.43554688 |

完整分位数包含 P01/P05/P10/P25/P50/P75/P90/P95/P99；使用线性插值。

## Argmax 类别占比（不赋予二分类语义）

| 数据集 | 字段 | index | count | 占比 |
|---|---|---:|---:|---:|
| A1 | modernbert_cleanliness | 0 | 0 | 0.000000% |
| A1 | modernbert_cleanliness | 1 | 2755 | 5.377708% |
| A1 | modernbert_cleanliness | 2 | 11974 | 23.373024% |
| A1 | modernbert_cleanliness | 3 | 1335 | 2.605895% |
| A1 | modernbert_cleanliness | 4 | 20187 | 39.404646% |
| A1 | modernbert_cleanliness | 5 | 14979 | 29.238727% |
| A1 | modernbert_readability | 0 | 52 | 0.101503% |
| A1 | modernbert_readability | 1 | 6628 | 12.937732% |
| A1 | modernbert_readability | 2 | 2099 | 4.097209% |
| A1 | modernbert_readability | 3 | 1375 | 2.683974% |
| A1 | modernbert_readability | 4 | 25837 | 50.433340% |
| A1 | modernbert_readability | 5 | 15239 | 29.746242% |
| A1 | modernbert_reasoning | 0 | 4068 | 7.942675% |
| A1 | modernbert_reasoning | 1 | 25084 | 48.975926% |
| A1 | modernbert_reasoning | 2 | 13177 | 25.727786% |
| A1 | modernbert_reasoning | 3 | 1803 | 3.520316% |
| A1 | modernbert_reasoning | 4 | 2470 | 4.822617% |
| A1 | modernbert_reasoning | 5 | 4615 | 9.010680% |
| A1 | modernbert_professionalism | 0 | 0 | 0.000000% |
| A1 | modernbert_professionalism | 1 | 13502 | 26.358224% |
| A1 | modernbert_professionalism | 2 | 10751 | 20.987799% |
| A1 | modernbert_professionalism | 3 | 6517 | 12.722304% |
| A1 | modernbert_professionalism | 4 | 11363 | 22.182528% |
| A1 | modernbert_professionalism | 5 | 9092 | 17.749146% |
| A1 | fluency_en | 0 | 11119 | 21.704080% |
| A1 | fluency_en | 1 | 40111 | 78.295920% |
| A1 | ad_en | 0 | 2930 | 5.719305% |
| A1 | ad_en | 1 | 48300 | 94.280695% |
| A2 | modernbert_cleanliness | 0 | 0 | 0.000000% |
| A2 | modernbert_cleanliness | 1 | 57 | 0.325287% |
| A2 | modernbert_cleanliness | 2 | 34 | 0.194031% |
| A2 | modernbert_cleanliness | 3 | 0 | 0.000000% |
| A2 | modernbert_cleanliness | 4 | 61 | 0.348114% |
| A2 | modernbert_cleanliness | 5 | 17371 | 99.132569% |
| A2 | modernbert_readability | 0 | 0 | 0.000000% |
| A2 | modernbert_readability | 1 | 43 | 0.245392% |
| A2 | modernbert_readability | 2 | 1 | 0.005707% |
| A2 | modernbert_readability | 3 | 0 | 0.000000% |
| A2 | modernbert_readability | 4 | 1466 | 8.366147% |
| A2 | modernbert_readability | 5 | 16013 | 91.382754% |
| A2 | modernbert_reasoning | 0 | 51 | 0.291046% |
| A2 | modernbert_reasoning | 1 | 67 | 0.382355% |
| A2 | modernbert_reasoning | 2 | 6 | 0.034241% |
| A2 | modernbert_reasoning | 3 | 0 | 0.000000% |
| A2 | modernbert_reasoning | 4 | 3 | 0.017120% |
| A2 | modernbert_reasoning | 5 | 17396 | 99.275238% |
| A2 | modernbert_professionalism | 0 | 0 | 0.000000% |
| A2 | modernbert_professionalism | 1 | 33 | 0.188324% |
| A2 | modernbert_professionalism | 2 | 13 | 0.074188% |
| A2 | modernbert_professionalism | 3 | 5 | 0.028534% |
| A2 | modernbert_professionalism | 4 | 32 | 0.182617% |
| A2 | modernbert_professionalism | 5 | 17440 | 99.526337% |
| A2 | fluency_en | 0 | 141 | 0.804657% |
| A2 | fluency_en | 1 | 17382 | 99.195343% |
| A2 | ad_en | 0 | 16 | 0.091309% |
| A2 | ad_en | 1 | 17507 | 99.908691% |
| A3 | modernbert_cleanliness | 0 | 0 | 0.000000% |
| A3 | modernbert_cleanliness | 1 | 52195 | 25.616926% |
| A3 | modernbert_cleanliness | 2 | 129646 | 63.629314% |
| A3 | modernbert_cleanliness | 3 | 320 | 0.157054% |
| A3 | modernbert_cleanliness | 4 | 7725 | 3.791374% |
| A3 | modernbert_cleanliness | 5 | 13866 | 6.805332% |
| A3 | modernbert_readability | 0 | 1107 | 0.543308% |
| A3 | modernbert_readability | 1 | 51772 | 25.409321% |
| A3 | modernbert_readability | 2 | 39693 | 19.481036% |
| A3 | modernbert_readability | 3 | 3705 | 1.818387% |
| A3 | modernbert_readability | 4 | 92613 | 45.453787% |
| A3 | modernbert_readability | 5 | 14862 | 7.294162% |
| A3 | modernbert_reasoning | 0 | 65994 | 32.389375% |
| A3 | modernbert_reasoning | 1 | 76270 | 37.432761% |
| A3 | modernbert_reasoning | 2 | 49853 | 24.467490% |
| A3 | modernbert_reasoning | 3 | 700 | 0.343555% |
| A3 | modernbert_reasoning | 4 | 569 | 0.279261% |
| A3 | modernbert_reasoning | 5 | 10366 | 5.087557% |
| A3 | modernbert_professionalism | 0 | 0 | 0.000000% |
| A3 | modernbert_professionalism | 1 | 4935 | 2.422074% |
| A3 | modernbert_professionalism | 2 | 11589 | 5.687825% |
| A3 | modernbert_professionalism | 3 | 3665 | 1.798764% |
| A3 | modernbert_professionalism | 4 | 85725 | 42.073413% |
| A3 | modernbert_professionalism | 5 | 97837 | 48.017924% |
| A3 | fluency_en | 0 | 90021 | 44.181652% |
| A3 | fluency_en | 1 | 113731 | 55.818348% |
| A3 | ad_en | 0 | 635 | 0.311653% |
| A3 | ad_en | 1 | 203117 | 99.688347% |

## Softmax 全量数值验证

| 数据集 | 字段 | 已转换有限向量 | 原始 NaN/Inf 向量 | 越界 | 输出 NaN/Inf | 和失败 | argmax 不一致 | 最大和误差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | modernbert_cleanliness | 51230 | 0 | 0 | 0 | 0 | 0 | 5.55e-16 |
| A1 | modernbert_readability | 51230 | 0 | 0 | 0 | 0 | 0 | 4.44e-16 |
| A1 | modernbert_reasoning | 51217 | 13 | 0 | 0 | 0 | 0 | 5.55e-16 |
| A1 | modernbert_professionalism | 51225 | 5 | 0 | 0 | 0 | 0 | 5.55e-16 |
| A1 | fluency_en | 51230 | 0 | 0 | 0 | 0 | 0 | 2.22e-16 |
| A1 | ad_en | 51230 | 0 | 0 | 0 | 0 | 0 | 2.22e-16 |
| A2 | modernbert_cleanliness | 17523 | 0 | 0 | 0 | 0 | 0 | 4.44e-16 |
| A2 | modernbert_readability | 17523 | 0 | 0 | 0 | 0 | 0 | 2.22e-16 |
| A2 | modernbert_reasoning | 17523 | 0 | 0 | 0 | 0 | 0 | 3.33e-16 |
| A2 | modernbert_professionalism | 17523 | 0 | 0 | 0 | 0 | 0 | 3.33e-16 |
| A2 | fluency_en | 17523 | 0 | 0 | 0 | 0 | 0 | 2.22e-16 |
| A2 | ad_en | 17523 | 0 | 0 | 0 | 0 | 0 | 2.22e-16 |
| A3 | modernbert_cleanliness | 203752 | 0 | 0 | 0 | 0 | 0 | 6.66e-16 |
| A3 | modernbert_readability | 203752 | 0 | 0 | 0 | 0 | 0 | 5.55e-16 |
| A3 | modernbert_reasoning | 203752 | 0 | 0 | 0 | 0 | 0 | 5.55e-16 |
| A3 | modernbert_professionalism | 203751 | 1 | 0 | 0 | 0 | 0 | 4.44e-16 |
| A3 | fluency_en | 203752 | 0 | 0 | 0 | 0 | 0 | 2.22e-16 |
| A3 | ad_en | 203752 | 0 | 0 | 0 | 0 | 0 | 2.22e-16 |

采用 exp(l−max(l))/sum(exp(l−max(l)))，float64；和误差严格小于配置容差 1e-12。
NaN/Inf 原始向量未填补、未转换，其来源和原值见 vector_invalid_records.csv；该文件用字符串 NaN/Inf 表示非有限原值。
argmax 若并列取最小索引，并另记原始并列数。现代分类期望按假设 index=等级 0…5 计算，范围为 0…5，仅用于诊断。

## qurater：尺度与相关性

- A1：四维极差最大/最小比=1.42211，标准差最大/最小比=1.45142。
- A2：四维极差最大/最小比=1.39838，标准差最大/最小比=1.19016。
- A3：四维极差最大/最小比=1.55585，标准差最大/最小比=1.84645。

这些数字衡量观察到的尺度差异；不能证明评分量纲、方向或语义可交换。每维位置分布见上表。
四维极差和标准差处于接近的数量级，但不完全一致；位置均值、中位数和相关性随数据集变化。
四项是不同评分维度，统计相似性不构成自动平均的依据；未对 qurater 应用 Softmax 或均值。

### Pearson

| 数据集 | 位置 | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|---:|
| A1 | 0 | 1.000000 | 0.192880 | 0.592545 | 0.489463 |
| A1 | 1 | 0.192880 | 1.000000 | 0.355494 | 0.635801 |
| A1 | 2 | 0.592545 | 0.355494 | 1.000000 | 0.475791 |
| A1 | 3 | 0.489463 | 0.635801 | 0.475791 | 1.000000 |
| A2 | 0 | 1.000000 | 0.564380 | 0.661961 | 0.735453 |
| A2 | 1 | 0.564380 | 1.000000 | 0.317339 | 0.410217 |
| A2 | 2 | 0.661961 | 0.317339 | 1.000000 | 0.554387 |
| A2 | 3 | 0.735453 | 0.410217 | 0.554387 | 1.000000 |
| A3 | 0 | 1.000000 | 0.171758 | 0.495472 | 0.664195 |
| A3 | 1 | 0.171758 | 1.000000 | 0.593127 | 0.560055 |
| A3 | 2 | 0.495472 | 0.593127 | 1.000000 | 0.714376 |
| A3 | 3 | 0.664195 | 0.560055 | 0.714376 | 1.000000 |

### Spearman（并列值采用平均秩）

| 数据集 | 位置 | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|---:|
| A1 | 0 | 1.000000 | 0.127114 | 0.620287 | 0.460734 |
| A1 | 1 | 0.127114 | 1.000000 | 0.286507 | 0.608642 |
| A1 | 2 | 0.620287 | 0.286507 | 1.000000 | 0.425368 |
| A1 | 3 | 0.460734 | 0.608642 | 0.425368 | 1.000000 |
| A2 | 0 | 1.000000 | 0.401251 | 0.560661 | 0.653409 |
| A2 | 1 | 0.401251 | 1.000000 | 0.102599 | 0.173465 |
| A2 | 2 | 0.560661 | 0.102599 | 1.000000 | 0.415508 |
| A2 | 3 | 0.653409 | 0.173465 | 0.415508 | 1.000000 |
| A3 | 0 | 1.000000 | 0.218955 | 0.456159 | 0.644622 |
| A3 | 1 | 0.218955 | 1.000000 | 0.625372 | 0.547496 |
| A3 | 2 | 0.456159 | 0.625372 | 1.000000 | 0.716474 |
| A3 | 3 | 0.644622 | 0.547496 | 0.716474 | 1.000000 |

## 类别证据与未确认项

本地 PDF 可提取文本、Word XML 文本、附件 README 和来源/模型元数据中未发现这八项的明确类别索引定义。
PDF 存在字体映射提取警告；不能用关键词未命中证明原件绝无定义。因此本地类别顺序记录为未确认。
完整本地检索范围和提取警告数见 local_semantics_evidence_inventory.csv。

附件 source_manifest.json 指向的[上游数据集 README](https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater/blob/main/README.md)（检索日期 2026-09-26）说明：

- modernbert 四项为六级 logits，文档按索引 0…5 表示等级；文档推荐 argmax，Softmax 期望是本轮另行诊断的候选方式。
- fluency_en 的上游索引为 [不流畅, 流畅]；ad_en 的上游索引为 [有广告, 无广告]。
- qurater 四项依次涉及写作、专业知识要求、事实知识、教育价值；未证实相同量纲和独立性。

上游语义证据和本地确认分开记录。按用户要求，本轮不把上游映射自动批准为附件级转换规则。
因此这些字段 status=NEEDS_CONFIRMATION；不能据此启动正式标量化。尤其 ad_en 不应误取 p1 当作广告概率。

## 真实诊断样例与输出

- vector_semantics_samples.txt：每数据集每个现代分类字段 20 条；每个二分类字段 30 条；qurater 20 条。
- modernbert_softmax_samples.csv：原值、Softmax、argmax、0…5 假设期望（diagnostic_only=True）。
- binary_softmax_samples.csv：原值、[p0,p1]、argmax，不在数据列中绑定实际类别。
- qurater_raw_samples.csv：真实四维值，不平均。
- vector_semantics_summary.csv / vector_position_profile.csv / vector_raw_distributions.csv：全量统计。
- vector_argmax_distribution.csv / qurater_pearson.csv / qurater_spearman.csv：占比与四维相关。

本轮结束并停止，未运行 02/03/04；未计算 CRITIC 或 Q。
