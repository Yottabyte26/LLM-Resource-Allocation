"""Assemble the final report from actual persisted results and reviewed texts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def read_table(path: Path) -> list[dict]:
    """Read a generated audit/statistics table with no computation or fitting."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
    """Preserve output values in Markdown, escaping table separators."""
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")).replace("|", "\\|").replace("\n", " ") for k in columns) + " |")
    return lines


def write_final_report(root: Path, settings: dict) -> None:
    """Write only claims supported by existing score/audit/review files."""
    quality, preprocess = root/"outputs"/"quality", root/"outputs"/"preprocess"
    model = json.loads((preprocess/"preprocessing_model_A1.json").read_text(encoding="utf-8"))
    weight_summary = json.loads((quality/"critic_diagnostics_summary_A1.json").read_text(encoding="utf-8"))
    weights = sorted(read_table(quality/"critic_weights_A1.csv"), key=lambda r: int(r["weight_rank"]))
    dataset = read_table(quality/"Q_dataset_summary.csv")
    domains = read_table(quality/"domain_quality_summary_A1.csv")
    extended = read_table(quality/"extended_domain_quality_summary.csv")
    remainder = read_table(quality/"nonoverlap_quality_summary.csv")
    consistency = read_table(quality/"sample_extended_consistency.csv")
    equal = read_table(quality/"Q_weighting_robustness.csv")
    spearman = read_table(quality/"Q_spearman_weighting_sensitivity.csv")
    imputed = read_table(preprocess/"imputation_log.csv")
    duplicate = read_table(root/"outputs"/"inspection"/"duplicate_summary.csv")
    overlaps = read_table(root/"outputs"/"inspection"/"cross_dataset_overlap.csv")
    inputs = read_table(quality/"scoring_input_manifest.csv")
    validations = read_table(quality/"Q_validation.csv")
    lines = ["# 问题一第 1 小问：数据质量评价最终报告", "",
        "范围仅为数据质量评价。没有进行质量冲突消解、领域配比或其他问题。", "",
        "## 数据规模、真实附件与清洗流程", "",
        "官方 XZ 经 lzma 流式读取，22 维指标独立于 id/content/来源等辅助字段。",
        "结构与类别顺序人工确认后，先标量化、按 A1 中位数填补、按 A1 阈值截尾，再统一方向构造 Z。",
        "本阶段读取保存的 Z，不重新拟合预处理或 CRITIC；仅为 40 条核验文本回读原始 A1。", ""]
    lines += markdown_table(read_table(preprocess/"preprocessing_input_manifest.csv"), ["dataset", "source_file", "n", "source_unchanged"])
    lines += ["", "### 阶段 4B 输入文件与完整精度权重", ""]
    lines += markdown_table(inputs, ["role", "path", "sha256_before", "sha256_after", "unchanged"])
    lines += ["", "## 22 个指标处理与 A1 标定策略", "",
        "全部填补值、分位数、qurater 四维范围、适中型阈值与连续 Min-Max 范围只由 A1 标定；A2/A3 共用固定模型。",
        "旧 pooled profile 与 qurater_transform_parameters.csv 仅用于描述/敏感性，未删除、未作主模型参数。",
        "原始 terminal_punctution 拼写通过 FIELD_ALIASES 映射到内部 terminal_punctuation，附件未改。", ""]
    rules = []
    for metric in model["metric_definitions"]:
        name = metric["name"]
        final = "A1 P01/P99 + " + ("trapezoid" if metric["direction"] == "moderate" else "Min-Max") if name in model["winsor_thresholds"] else "already bounded; no second Min-Max"
        rules.append({"metric": name, "direction": metric["direction"], "scalarization": metric.get("scalarization", "raw scalar"), "normalization": final})
    lines += markdown_table(rules, ["metric", "direction", "scalarization", "normalization"])
    lines += ["", "fineweb 取唯一值；qurater 四语义维度独立 P01/P99 截尾、Min-Max 后平均，最终仍为一个指标；",
        "modernbert 四项按等级 0…5 argmax/5，fluency 按 [not_fluent,fluent]、ad 按 [has_ad,no_ad] 取 argmax。",
        "ad=1 表示无广告，为正向。主方案不使用 Softmax 期望。14 正向、5 负向、3 适中，严格 22 维。",
        "适中梯形外端点为 P01/P99，满分平台 P10/P90；所有阈值保持 PHASE 3 已确认值。", "",
        "## 缺失与重复记录", "",
        f"实际 {len(imputed)} 个 scalar missing 涉及 {len({(r['dataset'],r['id']) for r in imputed})} 条记录，每条仅缺 1 项。",
        "19 项均采用 A1 同领域中位数，没有删除记录，没有触发总体回退；原始 NaN 向量不直接 argmax。",
        "imputation_log.csv 可追溯填补 ID、指标、来源和填充值。", ""]
    lines += markdown_table(imputed, ["dataset", "domain", "id", "metric", "imputed_value", "imputation_method"])
    lines += markdown_table(duplicate, ["scope", "n", "exact_duplicate_groups", "duplicated_id_groups", "SAME_ID_SAME_DATA_groups", "SAME_ID_DIFFERENT_DATA_groups"])
    lines += ["", "跨附件同 ID 记录保留。A3 不同 ID 但原 22 质量字段相同的那组两条记录也保留，",
        "缺 content 不能证明文本重复。完整 exact duplicate 与质量字段一致的口径不同。", ""]
    lines += markdown_table(overlaps, ["comparison", "same_id", "same_id_same_quality_pairs", "same_id_different_quality_pairs", "content_not_comparable_pairs"])
    lines += ["", "## CRITIC 主权重与稳定性", "",
        "只用 A1：σ_j=std(Z_j,ddof=1)，r_jk=Pearson(Z_j,Z_k)，R_j=Σ_k(1−r_jk)，C_j=σ_jR_j，w_j=C_j/ΣC_j。",
        "负相关保留，不取绝对值、不人工修正。读取 critic_weights_A1.csv 完整浮点精度，未用展示时四舍五入权重。", ""]
    lines += markdown_table(weights, ["metric", "critic_weight", "weight_rank"])
    lines += ["", f"权重和={weight_summary['weight_sum']}；H_norm={weight_summary['normalized_weight_entropy']:.9f}；",
        f"前五权重累计={weight_summary['top5_weight_sum']:.6%}。无高于 0.10、低于 0.01 权重及常数退化指标。",
        f"Pearson 与 Spearman 权重相关={weight_summary['weight_vector_pearson']:.9f}，权重秩相关={weight_summary['weight_vector_spearman']:.9f}。", "",
        "## 样本级 Q 与数据集描述统计", "",
        "Q_i=Σ_{j=1}^{22}w_jz_ij。A1/A2/A3 使用同一组 A1 Pearson-CRITIC 权重。",
        "因为 z∈[0,1]、w≥0、Σw=1，Q∈[0,1]；实际写出文件逐项验证有限性和范围。主 CSV 不包含 content。", ""]
    lines += markdown_table(dataset, ["dataset", "n", "mean_Q", "std_Q", "min_Q", "P01", "P05", "P10", "P25", "median_Q", "P75", "P90", "P95", "P99", "max_Q"])
    lines += markdown_table(validations, ["dataset", "n", "min_Q", "max_Q", "nan_count", "inf_count", "all_in_0_1"])
    lines += ["", "## A1 主领域质量与 Bootstrap", "",
        "Q_domain=TrimmedMean_10%(Q_i)，每侧截去 floor(0.10n) 个值，剩余约 80% 求均值。",
        f"95% CI 针对同一个 trimmed mean：固定根种子 {settings['random_state']}、每组 {settings['bootstrap_repetitions']} 次普通完整 n-out-of-n 重采样，",
        "每次实际抽取 n 个值并精确截尾，以 bootstrap 分布的 2.5%/97.5% 分位点得到区间。",
        "没有对大扩展集缩减样本量或用正态/有效样本数近似。各组由固定顺序 SeedSequence 子流生成；",
        "bootstrap_trimmed_means.csv 保存全部复制统计量，置信区间反映所用抽样/重采样假设而非因果可靠性。", ""]
    lines += markdown_table(domains, ["domain", "n", "mean_Q", "median_Q", "std_Q", "P10", "P25", "P75", "P90", "Q_domain", "ci95_low", "ci95_high", "quality_rank"])
    overlapping = [f"{a['domain']}/{b['domain']}" for a,b in zip(domains, domains[1:])
                   if max(float(a["ci95_low"]),float(b["ci95_low"])) <= min(float(a["ci95_high"]),float(b["ci95_high"]))]
    if overlapping:
        lines += ["", "相邻排名中置信区间重叠的领域："+", ".join(overlapping)+"。点估计排序不能直接解释为前者显著优于后者，CI重叠也不等同于正式差值检验。"]
    lines += ["", "## 扩展全量与非重叠余集领域质量", ""]
    lines += markdown_table(extended + remainder, ["dataset", "domain", "view", "n", "mean_Q", "median_Q", "std_Q", "trimmed_mean_Q", "ci95_low", "ci95_high"])
    lines += ["", "非重叠索引只选择 A2 IDs−A1-arxiv IDs、A3 IDs−A1-github IDs，不删除全量质量输出，也不复制完整 Z。", "",
        "## 抽样—扩展一致性与效应量", "",
        "所有有符号差值为扩展减 A1 抽样；relative_difference 是 trimmed mean 差除以抽样 trimmed mean 的绝对值。",
        "Cliff's delta=P(Q_extended>Q_sample)−P(Q_extended<Q_sample)，并列贡献为零；正值表示扩展倾向更高。",
        "Wasserstein 距离使用 Q 原尺度，KS 衡量最大 CDF 差。不能只用 p-value 宣称分布完全相同/不同。",
        "full 扩展包含 A1 记录，两组不独立；其 KS p-value 仅作描述性参考，不能作为独立双样本推断。",
        "nonoverlap 消除共享记录，但未知的抽样设计与潜在来源相关性仍限制 iid 推断。", ""]
    lines += markdown_table(consistency, ["comparison", "view", "sample_n", "extended_n", "mean_difference", "median_difference", "trimmed_mean_difference", "relative_difference", "KS_statistic", "KS_p_value", "Wasserstein_distance", "Cliffs_delta", "shared_id_count"])
    for r in consistency:
        lines += ["", f"- {r['comparison']}：trimmed mean 差 {float(r['trimmed_mean_difference']):+.8f}（相对 {float(r['relative_difference']):+.4%}），",
            f"  KS={float(r['KS_statistic']):.6f}，Wasserstein={float(r['Wasserstein_distance']):.6f}，Cliff's delta={float(r['Cliffs_delta']):+.6f}。",
            "  这些量分别描述位置、CDF 形态和成对排序；差异必须结合 Q 的模型含义解释，不能据显著性自动判定实质质量差异。"]
    lines += ["", "## 等权与 Spearman-CRITIC 敏感性", "",
        "Q_equal=mean(Z_i)；Q_spearman 使用 PHASE 4A 已保存的 Spearman 权重，仅为敏感性，均不替代主模型。",
        "领域排名都依据各方案的 10% trimmed mean，而不是混用平均数排名。rank_change=敏感性排名−主排名。", ""]
    lines += markdown_table(equal + spearman, ["scope", "alternative", "domain", "n", "pearson_correlation", "spearman_correlation", "main_Q_domain", "alternative_Q_domain", "main_rank", "alternative_rank", "rank_change"])
    for label, rows in [("等权",equal),("Spearman-CRITIC",spearman)]:
        changed=[r for r in rows if r["scope"]=="domain" and int(r["rank_change"])!=0]
        descriptions=[f"{r['domain']} {r['main_rank']}→{r['alternative_rank']}" for r in changed]
        lines += ["", f"{label}有 {len(changed)} 个领域发生排名变化："+(", ".join(descriptions) if descriptions else "无")+"。样本评分相关性高不能自动推出领域排名完全稳定。"]
    lines += ["", "## 高低 Q 文本核验", "",
        "仅 A1 有 content。按 Q 排序选最高和最低各 20 条，遇相同 Q 用 ID 字典序稳定排序，",
        "再依据 id 和来源行号回读原件；只保存这 40 条文本到 top20/bottom20 CSV。content 没有进入评分。"]
    review_path = quality/"A1_text_review.csv"
    if review_path.is_file():
        review = read_table(review_path)
        if len(review) != 40 or len({r["id"] for r in review}) != 40:
            raise ValueError("Text review must cover exactly 40 distinct selected records")
        lines += ["", "以下为助手逐条阅读后的文本复核记录，不冒称独立人工确认。判读使用材料中注明的阅读范围；",
            "完整 content 已导出供人工复核，文体/领域差异也会影响指标，应避免把代码天然判为低质。", ""]
        lines += markdown_table(review, ["set", "rank", "id", "domain", "Q", "review_scope", "observations", "judgement", "limitations"])
        for label in ["top20", "bottom20"]:
            counts = {}
            for row in review:
                if row["set"] == label:
                    counts[row["judgement"]] = counts.get(row["judgement"], 0) + 1
            lines += ["", f"{label} 助手复核分布：{json.dumps(counts, ensure_ascii=False)}。"]
            source_counts={}
            for row in review:
                if row["set"]==label:source_counts[row["domain"]]=source_counts.get(row["domain"],0)+1
            lines += ["", f"{label} 来源领域：{json.dumps(source_counts,ensure_ascii=False)}。"]
        lines += ["", "所见高分端多为专业、连贯英文，但部分正文混入评论/推广/跨主题拼接；低分端也包含功能正常的短多语言消歧页、天文表格和结构化代码。",
            "XML样本全文件解析成功，代码样本未编译；因此低Q主要反映本指标体系对文体/语言的适配程度，不能统一当作事实错误或内容无用。",
            "本次核验提供支持及反例，未反向修正已锁定的预处理或权重。"]
    else:
        lines += ["", "状态：文本材料已导出，逐条复核尚未完成；此状态不代表已通过人工验证。"]
    lines += ["", "## 图表", ""]
    for row in read_table(quality/"quality_figure_manifest.csv"):
        lines += ["", f"![{row['figure']}](../figures/question1_part1/{row['figure']})"]
    lines += ["", "## 模型局限性与可复用文件", "",
        "1. Q 是这 22 个已确认指标的加权效用，不是训练收益或事实正确性的直接测量。CRITIC 的方差/相关结构反映统计信息，不能证明指标语义有效。",
        "2. A1 统一尺度使扩展集可比，也造成范围外截尾饱和；A2 的 DSIR/qurater 等大量截尾，以及分类天花板仍然存在。",
        "3. P10–P90 平台压平内部差异，领域中位数填补会降低局部不确定性；当前未改变这些经确认的主方案。",
        "4. 跨附件共享记录不应当作额外独立证据；非重叠对照改善这一点，仍不能证明真实训练数据总体代表性。",
        "5. bootstrap 区间以当前记录独立重采样为基础，未建模文档来源聚类；显著性不能替代效应量与实际意义。",
        "6. 高低各 20 条是有选择的极端材料，助手文本判读不是盲评或独立人工确认；不用于反向修正本次权重。", "",
        "后续可直接复用 A1_sample_quality.csv、A2_arxiv_quality.csv、A3_github_quality.csv、A1 领域摘要、",
        "固定 A1 模型/权重、非重叠 ID 索引和敏感性评分。但本阶段没有开始质量冲突消解、领域配比或其他问题。", ""]
    (root/"outputs"/"report"/"question1_part1_final.md").write_text("\n".join(lines), encoding="utf-8")
