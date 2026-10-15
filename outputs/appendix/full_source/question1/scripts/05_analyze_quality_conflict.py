"""Question 1 part 2 stage 1: diagnose disagreement using frozen part 1.

No preprocessing, CRITIC fitting, Q adjustment or lambda model is invoked.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import sys

import numpy as np
import yaml
from scipy.stats import ks_2samp, wasserstein_distance, pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quality.conflict import (conflict_indices, fit_conflict_thresholds,
    conflict_profile, pair_conflict_summary, deviation_driver_summary)
from quality.conflict_plotting import make_conflict_figures
from quality.io import iter_jsonl, sha256_file
from quality.vector_diagnostics import write_rows

OUT = ROOT / "outputs" / "conflict"
META = ["id", "dataset", "domain", "sub_path", "source_file", "line_number", "imputed_count"]
INPUTS = {"A1": "A1_sample_quality.csv", "A2": "A2_arxiv_quality.csv", "A3": "A3_github_quality.csv"}


def read_csv(path):
    """Read a small table, never the content-bearing original dataset."""
    csv.field_size_limit(2**31-1)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def protected_files():
    """Snapshot all finished part 1 artifacts and official attachment bytes."""
    files = []
    for relative in ["outputs/preprocess", "outputs/quality", "outputs/profile", "outputs/inspection",
                     "outputs/figures/question1_part1", "real_attachments"]:
        files.extend(p for p in (ROOT / relative).rglob("*") if p.is_file())
    files.extend(ROOT / "configs" / n for n in ["quality_metrics.yaml", "critic_diagnostics.yaml", "quality_scoring.yaml"])
    files.extend(p for p in (ROOT / "outputs/report").glob("*") if p.is_file() and not p.name.startswith("question1_part2"))
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(set(files))}


def load_scores(dataset, names, expected_n):
    """Read compact numerical arrays; keep metadata only, never original content."""
    values = np.empty((expected_n, 22)); q = np.empty(expected_n); metadata = []
    path = ROOT / "outputs/quality" / INPUTS[dataset]
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != set(META + ["Q"] + names):
            raise ValueError(f"Changed score schema: {path}")
        for i, row in enumerate(reader):
            if i >= expected_n or row["dataset"] != dataset:
                raise ValueError(f"Changed record scope: {path}")
            values[i] = [float(row[n]) for n in names]; q[i] = float(row["Q"])
            metadata.append({n: row[n] for n in META})
    if len(metadata) != expected_n or len({r['id'] for r in metadata}) != expected_n:
        raise ValueError(f"Unexpected rows/within-dataset duplicated IDs: {path}")
    return {"Z": values, "Q": q, "metadata": metadata, "names": names}


def safe_corr(a, b, method="pearson"):
    """Undefined correlations remain empty, especially tiny/constant strata."""
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float((pearsonr if method == "pearson" else spearmanr)(a, b).statistic)


def cliffs_delta(a, b):
    """Signed effect size P(a>b)-P(a<b), with exact tie handling."""
    ordered = np.sort(b)
    lower = np.searchsorted(ordered, a, side="left")
    upper = len(b) - np.searchsorted(ordered, a, side="right")
    return float((lower.sum(dtype=np.int64) - upper.sum(dtype=np.int64)) / (len(a) * len(b)))


def save_samples(dataset, data, thresholds):
    """Write separate conflict outputs, preserving frozen Q/Z and every record."""
    filename = {"A1":"A1_sample_conflict.csv", "A2":"A2_arxiv_conflict.csv", "A3":"A3_github_conflict.csv"}[dataset]
    keys = META + ["Q", "K", "K_equal", "V", "Q_equal", "Delta_Q", "conflict_level", "high_conflict", "extreme_conflict"] + data["names"]
    with (OUT / filename).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        for i, meta in enumerate(data["metadata"]):
            k = data["K"][i]
            level = "low" if k <= thresholds["P50"] else "medium" if k <= thresholds["P90"] else "high"
            writer.writerow({**meta, **{n: float(data[n][i]) for n in ["Q", "K", "K_equal", "V", "Q_equal", "Delta_Q"]},
                "conflict_level": level, "high_conflict": k > thresholds["P90"], "extreme_conflict": k > thresholds["P95"],
                **dict(zip(data["names"], data["Z"][i]))})


def relationship(data, mask, meta):
    """Describe association with locked Q, alternative dispersion and weighting sensitivity."""
    k, q, delta = [data[n][mask] for n in ["K", "Q", "Delta_Q"]]
    return {**meta, "n": len(k), "pearson_Q_K": safe_corr(q,k), "spearman_Q_K": safe_corr(q,k,"spearman"),
        "pearson_K_Delta_Q": safe_corr(k,delta), "spearman_K_Delta_Q": safe_corr(k,delta,"spearman"),
        "pearson_K_K_equal": safe_corr(k,data["K_equal"][mask]),
        "spearman_K_K_equal": safe_corr(k,data["K_equal"][mask],"spearman"),
        "pearson_K_V": safe_corr(k,data["V"][mask]), "spearman_K_V": safe_corr(k,data["V"][mask],"spearman"),
        "mean_Delta_Q": float(delta.mean()), "median_Delta_Q": float(np.median(delta))}


def sensitivity_comparison(data, mask, high, meta):
    """Compare weighting differences high versus ordinary; no causal/reliability claim."""
    a, b = data["Delta_Q"][mask & high], data["Delta_Q"][mask & ~high]
    if not len(a) or not len(b):
        return {**meta,"n_high":len(a),"n_ordinary":len(b),"comparison_defined":False}
    return {**meta,"n_high":len(a),"n_ordinary":len(b),"comparison_defined":True,
        "mean_Delta_Q_high":float(a.mean()),"mean_Delta_Q_ordinary":float(b.mean()),
        "median_Delta_Q_high":float(np.median(a)),"median_Delta_Q_ordinary":float(np.median(b)),
        "mean_difference_high_minus_ordinary":float(a.mean()-b.mean()),
        "mean_ratio_high_over_ordinary":float(a.mean()/b.mean()) if b.mean()>0 else None,
        "cliffs_delta_high_vs_ordinary":cliffs_delta(a,b),"KS_statistic":float(ks_2samp(a,b).statistic)}


def select_texts(a1, weights, thresholds, settings, manifest):
    """Retrieve selected A1 contents once; disclose conditional Q bands and scarce global groups."""
    q,k,z=a1["Q"],a1["K"],a1["Z"];high=k>thresholds["P90"]
    qlo,qhi=np.quantile(q[high],settings["conditional_Q_band_quantiles"],method=settings["quantile_method"])
    qmedian=float(np.quantile(q,settings["quadrant_Q_boundary_quantile"])); low=k<=thresholds["P50"]
    bands={"top_K_30":(np.ones(len(k),dtype=bool),settings["text_top_K_count"],"All A1; largest K"),
        "highK_relative_lowQ":(high & (q<=qlo),settings["text_per_conditional_Q_band"],f"K>A1P90; Q<=highK Q P33={qlo}"),
        "highK_relative_middleQ":(high & (q>qlo) & (q<=qhi),settings["text_per_conditional_Q_band"],f"K>A1P90; {qlo}<Q<=highK Q P67={qhi}"),
        "highK_relative_highQ":(high & (q>qhi),settings["text_per_conditional_Q_band"],f"K>A1P90; Q>highK Q P67={qhi}; NOT globally high Q")}
    for label,mask in {"global_highQ_lowK":(q>qmedian)&low,"global_highQ_highK":(q>qmedian)&high,
                       "global_lowQ_lowK":(q<=qmedian)&low,"global_lowQ_highK":(q<=qmedian)&high}.items():
        bands[label]=(mask,settings["text_per_quadrant_max"],f"A1 Q median={qmedian}; low K<=P50, high K>P90; middle K excluded")
    selection={}; rows=[]
    for label,(mask,requested,definition) in bands.items():
        eligible=np.flatnonzero(mask)
        order=sorted(eligible,key=lambda i:(-k[i],a1["metadata"][i]["id"]))[:requested]
        rows.append({"selection_group":label,"eligible_n":len(eligible),"requested_n":requested,"selected_n":len(order),"definition":definition})
        for i in order:selection.setdefault(int(i),[]).append(label)
    write_rows(OUT/"text_selection_summary.csv",rows)
    selected={a1["metadata"][i]["id"]:i for i in selection}; texts={}
    source_rows=[r for r in manifest if r["dataset"]=="A1"]
    for entry in source_rows:
        path=ROOT/entry["source_file"]
        if sha256_file(path)!=entry["sha256_before"]:raise ValueError("A1 source differs from frozen preprocessing manifest")
        n=0
        for line,record,error in iter_jsonl(path):
            n+=1
            if error:raise ValueError(f"Cannot recover text at {path.name}:{line}: {error}")
            record_id=record.get("id")
            if record_id in selected:
                i=selected[record_id]
                if str(line)!=a1["metadata"][i]["line_number"]:raise ValueError("Selected text row mismatch")
                content=record.get("content")
                if not isinstance(content,str):raise ValueError("Selected A1 content unavailable")
                texts[record_id]=content
        if n!=int(entry["n"]):raise ValueError("Changed original A1 record count")
    if set(texts)!=set(selected):raise ValueError("Selected A1 texts not fully recovered")
    text_rows=[]
    for i in sorted(selection,key=lambda i:(-k[i],a1["metadata"][i]["id"])):
        pos=np.flatnonzero(z[i]==z[i].max());neg=np.flatnonzero(z[i]==z[i].min())
        p=min(pos,key=lambda j:(-weights[j],j));m=min(neg,key=lambda j:(-weights[j],j))
        text_rows.append({**a1["metadata"][i],"Q":float(q[i]),"K":float(k[i]),"selection_groups":";".join(selection[i]),
            "top_positive_metric":a1["names"][p],"top_positive_value":float(z[i,p]),"top_positive_deviation":float(z[i,p]-q[i]),
            "top_negative_metric":a1["names"][m],"top_negative_value":float(z[i,m]),"top_negative_deviation":float(z[i,m]-q[i]),
            "all_tied_positive_metrics":";".join(a1["names"][j] for j in pos),
            "all_tied_negative_metrics":";".join(a1["names"][j] for j in neg),
            "representative_tie_rule":settings["text_representative_tie_rule"],"review_status":"待人工核验; no AI annotation used as evidence",
            "content":texts[a1["metadata"][i]["id"]]})
    write_rows(OUT/"A1_high_conflict_text_samples.csv",text_rows)
    return rows


def markdown_table(rows, columns):
    """Render actual numeric output as compact Markdown, without invented findings."""
    def cell(v):
        if v is None:return "未定义"
        if isinstance(v,(float,np.floating)):return f"{v:.6f}"
        return str(v).replace("|","\\|").replace("\n"," ")
    return "\n".join(["| "+" | ".join(columns)+" |","| "+" | ".join("---" for _ in columns)+" |"]+
        ["| "+" | ".join(cell(row.get(c)) for c in columns)+" |" for row in rows])


def build_report(context):
    """Write diagnostics and limitations; leave Q+K/Q* choice pending human confirmation."""
    c=context;top=c["top_pairs"][:10];a1=c["profiles"][0];rel=c["relationships"][0]
    sensitivities=c["sensitivity"];driver=c["drivers_global"][:10]
    report=["# 问题一第2小问：冲突诊断第一阶段（未构造 Q*）",
        "第一小问全部锁定。直接读取三份质量 CSV 内的原有 Q 和 22 个 Z；权重来自原有 `critic_weights_A1.csv`。未重新拟合预处理、CRITIC、领域质量或评分参数。新输出全部独立保存。",
        "## 定义与边界",
        "主指标 Kᵢ=Σⱼwⱼ|zᵢⱼ−Qᵢ|；辅助 K_equal=mean|Z−mean Z|、Vᵢ=Σⱼwⱼ(zᵢⱼ−Qᵢ)²。Q 使用已保存的第一小问结果；重新求和仅校验输入一致性。",
        "K 衡量不同评价构念的内部离散，不能自动解释为 Q 错误、预测误差或统计置信区间。由于 0≤z≤1 且 Q 为固定加权均值，必有 K≤2Q(1−Q)、V≤Q(1−Q)。因此高 Q 与高 K 的稀少及其负相关部分有数学约束，不能单独作为低可靠性的证据。",
        "阈值仅拟合 A1 的 K 分布，线性分位数；低冲突 K≤P50，中等 P50<K≤P90，高 K>P90，极高 K>P95（高冲突子集）。‘显著冲突’是分位阈值操作定义，不是显著性检验结果。A2/A3 直接应用 A1 阈值。",
        markdown_table(c["threshold_rows"],["quantile","K_threshold","fitting_dataset","n"]),
        "## 实际数据与总体分布",
        markdown_table(c["profiles"],["dataset","n","mean_K","median_K","std_K","min_K","max_K","P95_K","high_conflict_rate","extreme_conflict_rate"]),
        "仅为恢复被选中文本，对 A1 原始压缩文件流式读取一次；A2/A3 原始 JSONL 未重新解压。原始附件与全部第一小问产物进行前后 SHA-256 核验。",
        "## 领域冲突",
        markdown_table(c["domains"],["domain","n","mean_K","median_K","std_K","P90_K","P95_K","high_conflict_rate","extreme_conflict_rate","high_rate_wilson95_low","high_rate_wilson95_high"]),
        "以上排序是描述排序，不声称领域差异显著。Wilson 区间仅描述二项比例的不确定性；book 样本量小，区间较宽，不宜据此推断整体领域。附件采样机制、同源文档相关性也限制总体推断。",
        "## 指标对冲突",
        "Dᵢⱼₖ=|zᵢⱼ−zᵢₖ|，严重率 P(D≥0.5)。每个范围逐对聚合 231 对，不保存 n×231 明细；Top20 按严重率降序、平均差异降序排序。相关性另列，恒定指标相关系数留空。",
        markdown_table(top,["metric_1","metric_2","severe_pair_conflict_rate","mean_abs_difference","median_abs_difference","pearson_r","spearman_r"]),
        markdown_table(c["correlation_conflict"],["correlation_group","pair_count","mean_severe_pair_rate","mean_pair_difference"]),
        "负相关描述跨样本共同变化，D 描述同一样本的数值差异，两者不等同。即使正相关，位置/尺度分布不同也可产生大量高 D；全部数值来自固定归一化，不回调第一小问参数。",
        "## 数据驱动成因材料",
        "高冲突组与普通组的 22 指标平均值、中位数、P10/P90、KS、Wasserstein 与 Cliff's delta（高减普通）见 `high_vs_ordinary_indicator_summary.csv`。这些比较由 K 选择分组，存在定义带来的选择效应；不能当成独立因果验证。下表按均值差绝对值排序。",
        markdown_table(sorted(c["feature_comparison"],key=lambda r:-abs(r["mean_difference_high_minus_ordinary"]))[:12],
            ["metric","mean_high","mean_ordinary","median_high","median_ordinary","mean_difference_high_minus_ordinary","cliffs_delta_high_vs_ordinary"]),
        "正/负极端偏离的指标分别是 max(Z) 与 min(Z)（减去同一个 Q 不改变排序）。大量指标并列为 1 或 0；统计将每条样本均分给全部并列最大×最小组合，确保各组合的 fractional_sample_count 之和等于高冲突样本数，不偏向指标排列顺序。文本表只为展示选固定权重最大的并列指标，同时保留全部并列名称。",
        markdown_table(driver,["top_positive_metric","top_negative_metric","fractional_sample_count","fractional_rate","possible_sample_count"]),
        "分领域及扩展集同样计算驱动组合，见 `high_conflict_driver_summary.csv`。定义包含的数学关系、量表离散化/天花板、效用平台与评价构念差异是解释时必须考虑的机制；具体文本成因须人工对照。尚未作正式人工标注或因果判断。",
        "## 文本核验材料及全局高 Q 边界",
        markdown_table(c["text_selection"],["selection_group","eligible_n","requested_n","selected_n","definition"]),
        f"A1 全局 Q 上四分位以上且高 K 的记录数为 {c['upper_quartile_high_n']}；全局 Q 中位数以上且高 K 的记录数为 {c['median_high_n']}。不能凑出不存在的 20 条全局高 Q+高 K 文本。另使用**高冲突组内部 Q 三分位**取得相对高/中/低各 20 条（不同分组允许同一文本复用），清楚标明组内相对等级；最大 K 30 条也完整保留。",
        "四类全局 Q 高/低×K 高/低控制组以 Q 中位数及 K 的 P50/P90 划分，中间 K 不属于四类；完整六组计数另存，不删除任何样本。文本见 `A1_high_conflict_text_samples.csv`，全部标注待人工核验；当前没有将 AI 判断作为实验数据。",
        markdown_table(c["quadrants"],["Q_group","K_group","n","mean_Q","mean_K","mean_Delta_Q"]),
        "## A2/A3 扩展验证",
        "所有视图使用原权重、Q、K 公式及 A1 阈值。full 保留跨集重叠，nonoverlap 仅作为对照索引，不修改/删除原输出。Cliff's delta 定义为扩展减 A1；KS 为描述统计，不用 p 值宣称相同。",
        markdown_table(c["validation"],["domain","extended_dataset","view","n_A1","n_extended","mean_K_A1","mean_K_extended","median_K_A1","median_K_extended","high_rate_A1","high_rate_extended","P95_K_A1","P95_K_extended","KS_statistic","Wasserstein_distance","cliffs_delta_extended_vs_A1"]),
        "不仅比较 K 分布，还比较全部 231 对严重率向量及高冲突驱动分布；Top10 重合仅是描述性指标。某域高冲突样本很少时，驱动组合复现结论不稳定。",
        markdown_table(c["replication"],["domain","extended_dataset","view","pair_rate_pearson","pair_rate_spearman","top10_pair_overlap_n","driver_rate_pearson","top10_driver_overlap_n","high_n_A1","high_n_extended"]),
        "## Q 与 K、赋权敏感性",
        markdown_table(c["relationships"],["dataset","domain","view","n","pearson_Q_K","spearman_Q_K","pearson_K_Delta_Q","spearman_K_Delta_Q","pearson_K_K_equal","spearman_K_V","mean_Delta_Q"]),
        "Delta_Q=|原有 Q_CRITIC−mean Z|。等权只作对照，未替代锁定 Q。高冲突与普通组敏感性比较及分领域/分 Q 四分位对照如下：",
        markdown_table(sensitivities,["scope","domain","n_high","n_ordinary","mean_Delta_Q_high","mean_Delta_Q_ordinary","mean_ratio_high_over_ordinary","cliffs_delta_high_vs_ordinary"]),
        "总体相关性必须结合分领域与 Q 分层结果解释。Delta_Q 衡量对一种替代权重的变化，不是真实质量误差；增大也不自动证明应惩罚 Q。当前没有评分正确性的外部标签、冲突校准或修正收益证据。",
        "## 当前停止点与复用",
        "已完成 K、阈值、指标对、领域、驱动、文本材料、扩展验证及评分敏感性诊断。九幅图位于 `outputs/figures/question1_part2/`。未计算 Q*，未测试/选择 lambda，未生成冲突修正评分。等待人工确认冲突规律，后续再决定 Q+K 或 Q*；第一小问保持锁定。",
        "后续可直接读取三份 `*_conflict.csv`、`conflict_thresholds_A1.csv`、指标对/驱动/验证 CSV。可运行 `python scripts/05_analyze_quality_conflict.py` 重现此诊断阶段；运行前后核验锁定文件，一旦输入结构或内容不符即停止。",
        "全部输出为实际附件结果；文本复核待人工进行；图中散点仅固定随机种子抽取展示，所有统计使用全部记录。"]
    path=ROOT/"outputs/report/question1_part2_conflict_stage1.md"
    path.write_text("\n\n".join(report)+"\n",encoding="utf-8")


def record_findings():
    """Summarize already-saved diagnostics only; no new fit or raw reread."""
    def read(name):return read_csv(OUT/name)
    def number(row,key):return float(row[key])
    profile=read('dataset_conflict_summary.csv')[0]
    relationships=read('conflict_score_relationships.csv')
    overall=relationships[0]
    within=next(r for r in relationships if r['dataset']=='A1' and r['view']=='high_K')
    quartile=next(r for r in relationships if r['dataset']=='A1' and r['view']=='Q_quartile_1')
    sensitivity=read('score_sensitivity_group_comparison.csv')
    first=sensitivity[0];lowest=next(r for r in sensitivity if r['scope']=='A1_Q_quartile_1')
    domains=read('domain_conflict_summary_A1.csv')
    book_n=next(r['n'] for r in domains if r['domain']=='book')
    protected_n=len(read('locked_inputs_before.csv'))
    drivers=[r for r in read('high_conflict_driver_summary.csv') if r['dataset']=='A1' and r['domain']=='all']
    features=read('high_vs_ordinary_indicator_summary.csv')
    fluency=next(r for r in features if r['metric']=='fluency_en')
    text=read('A1_high_conflict_text_samples.csv')
    # These are explicitly excerpt-only observations, never full-text judgments,
    # final human labels, or input to an adjustment model.
    observations={
        'BkiUbvjxK6wB9mn49q52':'开头为带 LaTeX 命令及文献引用的介子跃迁论文；流畅为高端指标而词数效用为低端，提示专业文本与长度平台存在差异。',
        'BkiUdavxK7FjYAb_7N4b':'开头包含版权页、目录与小说序章；最低指标为独特词比例，需人工核验长篇叙述与重复词的区别。',
        'BkiUc0nxK6wB9k0iLK8R':'全文仅 47 字符，为压缩 JavaScript 的命令行；英语自然语言流畅分类不宜直接视为代码用途正确性的标签。',
        'BkiUbXU5jDKDyDF8vuRy':'开头为社会学书籍标题、出版社及版权材料；专业性高、英语流畅低的判定需要结合长篇结构人工检查。',
        'BkiUainxK1Thg9qFa8nv':'可见重复的沙发关键词行，每行带句号；句末标点高可以与流畅低共存，体现格式规则与文本意义的差异。',
        'BkiUbRzxK3xgpdbchQki':'全文为 76 字符的 PHP 配置语句；大写比例反向分数高、英语流畅低，体现结构化代码与语言分类的适用性差异。',
        'BkiUdOI5qsJBjuAa5ywn':'全文为流畅的珠宝商品介绍和规格选择；流畅高、推理低可以共存，需人工核验推广性，不能凭流畅认定全面高质。',
        'BkiUdmg5qYVBXTJ3BsEF':'开头为带 LaTeX、数学表达式和文献引用的天文学论文；大写比例反向分数低需结合符号/格式核验，不能自动判为正文质量差。',
        'BkiUdOg5qhDBZQieouQK':'开头为英语篮球比赛新闻，含比分和球员；流畅高、专业性相对低，可能反映新闻体裁与专业性构念差异。',
        'BkiUd-U4eIXh7gjUqssm':'开头及可见全文为波兰语村庄百科材料，却有 fluent=1；分类输出本身不是语言/质量真值。',
        'BkiUd_3xaJJQnK0WJoXz':'可见全文为德语小行星百科条目；流畅高、推理低可体现描述性事实文本与推理维度差异。',
        'BkiUduvxK7IAEwOJJChw':'开头为德语音乐术语解释；无广告高、英语流畅低，提示语言适用范围须与内容用途区分。'}
    annotated=[]
    for row in text:
        if row['id'] not in observations:continue
        annotated.append({k:row[k] for k in ['id','domain','Q','K','selection_groups','top_positive_metric','top_negative_metric']}|
            {'annotation_status':'AI辅助复核，非最终人工标注','inspection_scope':'仅检查开头片段最多1200字符；短文本可能覆盖全文；非正式实验证据',
             'full_content_characters':len(row['content']),'content_excerpt':row['content'][:1200],
             'preliminary_observation':observations[row['id']]})
    write_rows(OUT/'A1_conflict_text_AI_review.csv',annotated)
    notes=["# 问题一第2小问冲突诊断：实际发现与解释边界",
        "本页只汇总已经生成的真实统计；未重新计算 CRITIC、修改预处理/Q 或构造 Q*。详细表、定义与九幅图索引见 `question1_part2_conflict_stage1.md`。",
        f"A1 n={profile['n']}，K 均值 {number(profile,'mean_K'):.6f}，中位数 {number(profile,'median_K'):.6f}，标准差 {number(profile,'std_K'):.6f}，范围 [{number(profile,'min_K'):.6f}, {number(profile,'max_K'):.6f}]。P90={number(profile,'high_threshold_A1_P90'):.6f}，P95={number(profile,'extreme_threshold_A1_P95'):.6f}，高冲突 {profile['high_conflict_count']} 条。",
        "## 领域与主要组合",
        markdown_table(domains,['domain','n','high_conflict_rate','mean_K']),
        f"book 的高冲突率最高，但仅 {book_n} 条；不能根据描述排序宣称统计显著或代表整个书籍领域。stackexchange 与 commoncrawl 的高冲突率相对较低。",
        "A1 Top10 指标对见下表：严重率指 P(|z_j-z_k|≥0.5)，按该比例降序。大量涉及 reasoning，但全局指标对差异与高 K 的极端驱动并非同一个统计量。",
        markdown_table(read('top20_conflicting_pairs.csv')[:10],['rank','metric_1','metric_2','severe_pair_conflict_rate','pearson_r']),
        markdown_table(drivers[:10],['top_positive_metric','top_negative_metric','fractional_rate','possible_sample_count']),
        f"高冲突组 fluency_en 平均值 {number(fluency,'mean_high'):.6f}，普通组 {number(fluency,'mean_ordinary'):.6f}。主要极端组合表现为无广告/长度效用等高分与英语流畅低分、推理低分并存。并列值均分贡献后的比例不能当成原始唯一记录比例；possible_sample_count 可以重叠。",
        "数据支持‘评价构念不同、格式规则与内容评价不同、代码/非英语文本与英语分类适用范围不同’作为待核验机制；不将这些解释作为已经证实的因果成因。",
        "## 扩展集复现",
        markdown_table(read('sample_extended_conflict_validation.csv'),['domain','view','high_rate_A1','high_rate_extended','KS_statistic','Wasserstein_distance','cliffs_delta_extended_vs_A1']),
        markdown_table(read('conflict_pattern_replication.csv'),['domain','view','pair_rate_pearson','pair_rate_spearman','top10_pair_overlap_n','driver_rate_pearson','top10_driver_overlap_n']),
        "非重叠视图仍保留很高的指标对/驱动模式一致性；github 的分布差异更小，arxiv 的高冲突率略上升。这支持主要冲突结构复现，不能宣称分布完全相同，也不能推广到附件以外的数据。",
        "## 评分敏感性与不能跳过的控制结果",
        f"Q 与 K：Pearson={number(overall,'pearson_Q_K'):.6f}，Spearman={number(overall,'spearman_Q_K'):.6f}。K 与 ΔQ：Pearson={number(overall,'pearson_K_Delta_Q'):.6f}，Spearman={number(overall,'spearman_K_Delta_Q'):.6f}。",
        f"高冲突组 mean ΔQ={number(first,'mean_Delta_Q_high'):.6f}，普通组 {number(first,'mean_Delta_Q_ordinary'):.6f}，比值 {number(first,'mean_ratio_high_over_ordinary'):.4f}，Cliff's delta={number(first,'cliffs_delta_high_vs_ordinary'):.6f}。这是总体赋权敏感性差异，尚非质量误差证据。",
        f"但高冲突组内部 Pearson(K,ΔQ) 仅 {number(within,'pearson_K_Delta_Q'):.6f}；Q 最低四分位内部仅 {number(quartile,'pearson_K_Delta_Q'):.6f}，Spearman={number(quartile,'spearman_K_Delta_Q'):.6f}。在该 Q 层中高/普通组 ΔQ 均值比降至 {number(lowest,'mean_ratio_high_over_ordinary'):.4f}、Cliff's delta={number(lowest,'cliffs_delta_high_vs_ordinary'):.6f}。总体差异不能直接解释为冲突的独立影响。",
        "Q、K、ΔQ 都是同一组 Z 的函数，Q/K 还受到 K≤2Q(1−Q) 的数学约束。需要人工文本核验和独立质量标签才能评估真实错误/收益。目前仅确认内部意见差异与一种权重敏感性，不足以据此自动确定惩罚系数或选择 Q*。",
        "## 文本材料与初步比较（非正式证据）",
        f"完整文本材料包含最大 K 样本、高冲突组内相对高/中/低 Q 样本及全局四象限控制样本，组间可复用，共 {len(text)} 条唯一记录。实际选取数量见 text_selection_summary.csv；组内相对高 Q 材料不得称为全局高 Q。",
        f"另对其中 {len(annotated)} 条开头片段记录 AI 辅助复核，见 `A1_conflict_text_AI_review.csv`。明确标记**AI辅助复核，非最终人工标注**，不把片段观察作为全文质量或正式实验结论。控制组可见新闻/百科材料，高 Q+高 K 可见商品文案和带公式论文，低 Q+高 K 可见重复关键词和短代码；这些是选定案例，不是各组总体占比。",
        "原始全文有长字段，Python 读取文本 CSV 时应先设置 `csv.field_size_limit(2**31-1)`。完整输出未截断 content，主冲突 CSV 未写入 content。",
        "## 停止点",
        f"独立核验结果见 `conflict_verification.csv`。第一小问/官方附件的 {protected_n} 个文件 SHA-256 均保持一致。当前诊断阶段已完成；未构造 Q*、未测试 lambda，等待人工确认后再决定采用 Q+K 还是修正评分。"]
    (ROOT/'outputs/report/question1_part2_findings.md').write_text('\n\n'.join(notes)+'\n',encoding='utf-8')


def main():
    """Run the authorized diagnostic stage and stop before any Q adjustment."""
    OUT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),logging.FileHandler(OUT/"conflict_analysis.log",encoding="utf-8")])
    settings=yaml.safe_load((ROOT/"configs/conflict_analysis.yaml").read_text(encoding="utf-8"))
    if settings["stage_stop"]!="diagnostics_only_no_Q_star" or settings["threshold_fitting_dataset"]!="A1":
        raise ValueError("Unauthorized model stage or threshold fitting scope")
    logging.info("Snapshot protected part 1 artifacts and raw attachment hashes")
    before=protected_files()
    write_rows(OUT/"locked_inputs_before.csv",[{"path":p,"sha256":h} for p,h in before.items()])
    model=json.loads((ROOT/"outputs/preprocess/preprocessing_model_A1.json").read_text(encoding="utf-8"))
    names=model["metric_order"]
    if len(names)!=22 or len(set(names))!=22:raise ValueError("Expected exactly 22 locked metrics")
    wrows=read_csv(ROOT/"outputs/quality/critic_weights_A1.csv")
    if {r["metric"] for r in wrows}!=set(names) or len(wrows)!=22:raise ValueError("Locked weight schema mismatch")
    wmap={r["metric"]:float(r["critic_weight"]) for r in wrows};weights=np.array([wmap[n] for n in names])
    manifest=read_csv(ROOT/"outputs/preprocess/preprocessing_input_manifest.csv")
    datasets={}
    for dataset in INPUTS:
        n=sum(int(r["n"]) for r in manifest if r["dataset"]==dataset)
        logging.info("Read locked %s scores: %d records",dataset,n)
        data=load_scores(dataset,names,n);data.update(conflict_indices(data["Z"],data["Q"],weights));datasets[dataset]=data
    a1=datasets["A1"];domains=np.array([r["domain"] for r in a1["metadata"]]);thresholds=fit_conflict_thresholds(a1["K"],settings)
    thresholds_rows=[{"quantile":p,"K_threshold":v,"fitting_dataset":"A1","n":len(a1["K"]),"quantile_method":settings["quantile_method"]} for p,v in thresholds.items()]
    write_rows(OUT/"conflict_thresholds_A1.csv",thresholds_rows)
    profiles=[{"dataset":d,**conflict_profile(data["K"],thresholds,settings)} for d,data in datasets.items()]
    write_rows(OUT/"dataset_conflict_summary.csv",profiles)
    domain_rows=[{"dataset":"A1","domain":domain,**conflict_profile(a1["K"][domains==domain],thresholds,settings)} for domain in sorted(set(domains))]
    domain_rows.sort(key=lambda r:(-r["high_conflict_rate"],-r["mean_K"]))
    write_rows(OUT/"domain_conflict_summary_A1.csv",domain_rows)
    for d,data in datasets.items():
        logging.info("Write %s conflict output",d);save_samples(d,data,thresholds)
    overlap_rows=[]
    for d,domain in [("A2","arxiv"),("A3","github")]:
        data=datasets[d];a1ids={r["id"]:i for i,r in enumerate(a1["metadata"]) if r["domain"]==domain}
        ids={r["id"] for r in data["metadata"]}
        expected={r["id"] for r in read_csv(ROOT/f"outputs/preprocess/{d}_{domain}_nonoverlap_ids.csv")}
        if expected!=ids-set(a1ids) or not set(a1ids)<=ids:raise ValueError("Frozen nonoverlap index mismatch")
        mask=np.array([r["id"] in expected for r in data["metadata"]]);data["nonoverlap_mask"]=mask
        shared=[(a1ids[r["id"]],i) for i,r in enumerate(data["metadata"]) if r["id"] in a1ids]
        differences=[abs(a1["K"][i]-data["K"][j]) for i,j in shared]
        if max(differences)>1e-14:raise ValueError("Shared record K mismatch")
        overlap_rows.append({"dataset":d,"domain":domain,"shared_n":len(shared),"nonoverlap_n":int(mask.sum()),"max_shared_K_abs_diff":max(differences)})
    write_rows(OUT/"overlap_conflict_consistency.csv",overlap_rows)
    # Each group is processed sequentially, only one numerical pair vector at a time.
    scopes=[("A1","all","full",np.ones(len(a1["Q"]),dtype=bool))]
    scopes += [("A1",d,"full",domains==d) for d in sorted(set(domains))]
    scopes += [(d,domain,view,np.ones(len(datasets[d]["Q"]),dtype=bool) if view=="full" else datasets[d]["nonoverlap_mask"])
        for d,domain in [("A2","arxiv"),("A3","github")] for view in ["full","nonoverlap"]]
    pair_rows=[];driver_rows=[];relationships=[];by_scope={};drivers_by_scope={}
    for d,domain,view,mask in scopes:
        logging.info("Pair/driver diagnostics %s %s %s n=%d",d,domain,view,int(mask.sum()))
        data=datasets[d];meta={"dataset":d,"domain":domain,"view":view};key=(d,domain,view)
        rows=pair_conflict_summary(data["Z"][mask],names,settings,meta);pair_rows+=rows;by_scope[key]=rows
        drivers=deviation_driver_summary(data["Z"][mask],data["K"][mask]>thresholds["P90"],names,meta)
        driver_rows+=drivers;drivers_by_scope[key]=drivers
        relationships.append(relationship(data,mask,meta))
    write_rows(OUT/"indicator_pair_conflict.csv",pair_rows)
    pairs=by_scope[("A1","all","full")]
    top_pairs=sorted(pairs,key=lambda r:(-r["severe_pair_conflict_rate"],-r["mean_abs_difference"]))[:20]
    write_rows(OUT/"top20_conflicting_pairs.csv",[{"rank":i,**r} for i,r in enumerate(top_pairs,1)])
    write_rows(OUT/"high_conflict_driver_summary.csv",driver_rows)
    high=a1["K"]>thresholds["P90"];feature_rows=[]
    for j,name in enumerate(names):
        x,y=a1["Z"][high,j],a1["Z"][~high,j]
        feature_rows.append({"metric":name,"n_high":len(x),"n_ordinary":len(y),"mean_high":float(x.mean()),"mean_ordinary":float(y.mean()),
            "median_high":float(np.median(x)),"median_ordinary":float(np.median(y)),
            "P10_high":float(np.quantile(x,.1)),"P90_high":float(np.quantile(x,.9)),"P10_ordinary":float(np.quantile(y,.1)),"P90_ordinary":float(np.quantile(y,.9)),
            "mean_difference_high_minus_ordinary":float(x.mean()-y.mean()),"KS_statistic":float(ks_2samp(x,y).statistic),
            "Wasserstein_distance":float(wasserstein_distance(x,y)),"cliffs_delta_high_vs_ordinary":cliffs_delta(x,y),
            "mean_weighted_abs_deviation_high":float((weights[j]*np.abs(x-a1["Q"][high])).mean()),
            "mean_weighted_abs_deviation_ordinary":float((weights[j]*np.abs(y-a1["Q"][~high])).mean())})
    write_rows(OUT/"high_vs_ordinary_indicator_summary.csv",feature_rows)
    sensitivities=[sensitivity_comparison(a1,np.ones(len(high),dtype=bool),high,{"scope":"A1_all","domain":"all"})]
    sensitivities += [sensitivity_comparison(a1,domains==d,high,{"scope":"A1_domain","domain":d}) for d in sorted(set(domains))]
    qbounds=np.quantile(a1["Q"],[.25,.5,.75]);qgroups=np.searchsorted(qbounds,a1["Q"],side="left")
    for i in range(4):
        mask=qgroups==i
        sensitivities.append(sensitivity_comparison(a1,mask,high,{"scope":f"A1_Q_quartile_{i+1}","domain":"all"}))
        relationships.append(relationship(a1,mask,{"dataset":"A1","domain":"all","view":f"Q_quartile_{i+1}"}))
    for label,mask in [("high_K",high),("ordinary_K",~high)]:
        relationships.append(relationship(a1,mask,{"dataset":"A1","domain":"all","view":label}))
    write_rows(OUT/"score_sensitivity_group_comparison.csv",sensitivities)
    write_rows(OUT/"conflict_score_relationships.csv",relationships)
    quadrants=[]
    for qlabel,qmask in [("high_Q",a1["Q"]>qbounds[1]),("low_Q",a1["Q"]<=qbounds[1])]:
        for klabel,kmask in [("low_K",a1["K"]<=thresholds["P50"]),("medium_K",(a1["K"]>thresholds["P50"])&~high),("high_K",high)]:
            mask=qmask&kmask
            quadrants.append({"Q_group":qlabel,"K_group":klabel,"n":int(mask.sum()),
                "mean_Q":float(a1["Q"][mask].mean()) if mask.any() else None,
                "mean_K":float(a1["K"][mask].mean()) if mask.any() else None,
                "mean_Delta_Q":float(a1["Delta_Q"][mask].mean()) if mask.any() else None,
                "A1_Q_median":float(qbounds[1]),"A1_K_P50":thresholds["P50"],"A1_K_P90":thresholds["P90"]})
    write_rows(OUT/"quality_conflict_quadrant_summary.csv",quadrants)
    validation=[];replication=[];pair_replication=[]
    for d,domain in [("A2","arxiv"),("A3","github")]:
        reference=a1["K"][domains==domain];ref_profile=conflict_profile(reference,thresholds,settings)
        refpairs=by_scope[("A1",domain,"full")];refdrivers=drivers_by_scope[("A1",domain,"full")]
        for view in ["full","nonoverlap"]:
            mask=np.ones(len(datasets[d]["K"]),dtype=bool) if view=="full" else datasets[d]["nonoverlap_mask"]
            extension=datasets[d]["K"][mask];ep=conflict_profile(extension,thresholds,settings)
            meta={"domain":domain,"extended_dataset":d,"view":view}
            validation.append({**meta,"n_A1":len(reference),"n_extended":len(extension),
                "mean_K_A1":ref_profile["mean_K"],"mean_K_extended":ep["mean_K"],"median_K_A1":ref_profile["median_K"],"median_K_extended":ep["median_K"],
                "high_rate_A1":ref_profile["high_conflict_rate"],"high_rate_extended":ep["high_conflict_rate"],
                "P95_K_A1":ref_profile["P95_K"],"P95_K_extended":ep["P95_K"],"KS_statistic":float(ks_2samp(reference,extension).statistic),
                "Wasserstein_distance":float(wasserstein_distance(reference,extension)),"cliffs_delta_extended_vs_A1":cliffs_delta(extension,reference),
                "shared_records_included":view=="full","fixed_high_K_threshold":thresholds["P90"]})
            extpairs=by_scope[(d,domain,view)];extdrivers=drivers_by_scope[(d,domain,view)]
            rates1=np.array([r["severe_pair_conflict_rate"] for r in refpairs]);rates2=np.array([r["severe_pair_conflict_rate"] for r in extpairs])
            def top_pair_set(rows):return {(r["metric_1"],r["metric_2"]) for r in sorted(rows,key=lambda r:(-r["severe_pair_conflict_rate"],-r["mean_abs_difference"]))[:10]}
            def driver_vector(rows):
                mapping={(r["top_positive_metric"],r["top_negative_metric"]):r["fractional_rate"] for r in rows}
                return np.array([mapping.get((j,k),0.) for j in names for k in names])
            def top_driver_set(rows):return {(r["top_positive_metric"],r["top_negative_metric"]) for r in rows[:10]}
            replication.append({**meta,"pair_rate_pearson":safe_corr(rates1,rates2),"pair_rate_spearman":safe_corr(rates1,rates2,"spearman"),
                "top10_pair_overlap_n":len(top_pair_set(refpairs)&top_pair_set(extpairs)),
                "driver_rate_pearson":safe_corr(driver_vector(refdrivers),driver_vector(extdrivers)),
                "top10_driver_overlap_n":len(top_driver_set(refdrivers)&top_driver_set(extdrivers)),
                "high_n_A1":ref_profile["high_conflict_count"],"high_n_extended":ep["high_conflict_count"]})
            pair_replication += [{**meta,"metric_1":a["metric_1"],"metric_2":a["metric_2"],"severe_rate_A1":a["severe_pair_conflict_rate"],
                "severe_rate_extended":b["severe_pair_conflict_rate"],"rate_difference_extended_minus_A1":b["severe_pair_conflict_rate"]-a["severe_pair_conflict_rate"]} for a,b in zip(refpairs,extpairs)]
    write_rows(OUT/"sample_extended_conflict_validation.csv",validation)
    write_rows(OUT/"conflict_pattern_replication.csv",replication)
    write_rows(OUT/"extended_pair_conflict_comparison.csv",pair_replication)
    correlation_conflict=[]
    for label,condition in [("negative",lambda r:r["pearson_r"] is not None and r["pearson_r"]<0),
                            ("nonnegative",lambda r:r["pearson_r"] is not None and r["pearson_r"]>=0)]:
        selected=[r for r in pairs if condition(r)]
        correlation_conflict.append({"correlation_group":label,"pair_count":len(selected),
            "mean_severe_pair_rate":float(np.mean([r["severe_pair_conflict_rate"] for r in selected])),
            "mean_pair_difference":float(np.mean([r["mean_abs_difference"] for r in selected]))})
    write_rows(OUT/"correlation_vs_conflict_summary.csv",correlation_conflict)
    logging.info("Recover selected A1 text records by one streaming pass")
    text_selection=select_texts(a1,weights,thresholds,settings,manifest)
    logging.info("Generate nine diagnostic figures")
    figures=make_conflict_figures(ROOT/"outputs/figures/question1_part2",datasets,domains,domain_rows,pairs,thresholds,settings)
    write_rows(OUT/"conflict_figure_manifest.csv",[{"path":f"outputs/figures/question1_part2/{n}","statistics_scope":"full records; scatter displays fixed-seed subset"} for n in figures])
    build_report({"profiles":profiles,"domains":domain_rows,"threshold_rows":thresholds_rows,"top_pairs":top_pairs,
        "relationships":relationships,"sensitivity":sensitivities,"drivers_global":drivers_by_scope[("A1","all","full")],
        "feature_comparison":feature_rows,"text_selection":text_selection,"quadrants":quadrants,"validation":validation,
        "replication":replication,"correlation_conflict":correlation_conflict,
        "upper_quartile_high_n":int((high&(a1["Q"]>=qbounds[2])).sum()),"median_high_n":int((high&(a1["Q"]>qbounds[1])).sum())})
    record_findings()
    logging.info("Verify all protected files remain byte-for-byte unchanged")
    after=protected_files()
    if before!=after:raise ValueError("Protected part 1/raw attachment snapshot changed; stop")
    write_rows(OUT/"locked_inputs_integrity.csv",[{"path":p,"sha256_before":h,"sha256_after":after[p],"unchanged":True} for p,h in before.items()])
    (OUT/"conflict_analysis_config_snapshot.yaml").write_text(yaml.safe_dump(settings,allow_unicode=True,sort_keys=False),encoding="utf-8")
    logging.info("COMPLETED conflict diagnosis stage 1; no Q_star/lambda; locked files unchanged: %d",len(before))
    print(json.dumps({"thresholds":thresholds,"A1_profile":profiles[0],"relationship_A1":relationships[0],"sensitivity_A1":sensitivities[0]},ensure_ascii=False,allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("STOPPED: conflict diagnosis failed; no frozen input is repaired or rewritten")
        raise
