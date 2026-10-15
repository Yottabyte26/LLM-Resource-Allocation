"""Continue Phase 2: all-record vector diagnostics and local evidence review.

This script produces no scalar features, normalized indicators, CRITIC or Q.
Only hypothetical index expectations are printed for ModernBERT diagnostics.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from array import array
from collections import Counter
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quality.io import discover_quality_files, infer_source_domain, iter_jsonl, resolve_metric_value, sha256_file
from quality.vector_diagnostics import correlation_tables, distribution, json_cell, stable_softmax, write_rows

LOGGER = logging.getLogger("quality.vector_semantics")
MODERN = ("modernbert_cleanliness", "modernbert_readability", "modernbert_reasoning", "modernbert_professionalism")
BINARY = ("fluency_en", "ad_en")


class WarningCounter(logging.Handler):
    """Count PDF text-extraction warnings without flooding logs with font tables."""
    def __init__(self):
        super().__init__()
        self.count = 0

    def emit(self, record):
        self.count += 1


def inspect_local_evidence(output: Path) -> list[dict]:
    """Extract local field descriptions read-only and record search coverage/limits."""
    evidence_dir = output / "local_evidence"
    evidence_dir.mkdir(exist_ok=True)
    rows = []
    keywords = (*MODERN, *BINARY, "qurater", "fineweb_edu", "id2label", "label2id", "logits", "softmax")
    pdf_logger = logging.getLogger("pypdf")
    counter = WarningCounter()
    previous_handlers, previous_propagate = pdf_logger.handlers[:], pdf_logger.propagate
    pdf_logger.handlers, pdf_logger.propagate = [counter], False
    try:
        for path in sorted(PROJECT_ROOT.glob("*.pdf")):
            before = sha256_file(path)
            initial_warnings = counter.count
            reader = PdfReader(path)
            text = "\n".join(f"=== PAGE {i} ===\n{p.extract_text() or ''}" for i,p in enumerate(reader.pages,1))
            dest = evidence_dir / (path.stem + "_extracted.txt")
            dest.write_text(text, encoding="utf-8")
            if sha256_file(path) != before:
                raise RuntimeError(f"Evidence source changed: {path.name}")
            rows.append({"source_path": path.name, "source_kind": "pdf_extracted_text", "sha256": before,
                         "page_count": len(reader.pages), "text_characters": len(text),
                         "extraction_warning_count": counter.count-initial_warnings,
                         "keyword_hits": json_cell([k for k in keywords if k.lower() in text.lower()])})
    finally:
        pdf_logger.handlers, pdf_logger.propagate = previous_handlers, previous_propagate
    for path in sorted(PROJECT_ROOT.glob("*.docx")):
        before = sha256_file(path)
        with ZipFile(path) as archive:
            xml = ET.fromstring(archive.read("word/document.xml"))
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        text = "\n".join("".join(p.itertext()) for p in xml.findall(".//w:p", namespace))
        (evidence_dir / (path.stem+"_extracted.txt")).write_text(text, encoding="utf-8")
        if sha256_file(path) != before:
            raise RuntimeError(f"Evidence source changed: {path.name}")
        rows.append({"source_path": path.name, "source_kind": "docx_xml_text", "sha256": before,
                     "text_characters": len(text), "extraction_warning_count": 0,
                     "keyword_hits": json_cell([k for k in keywords if k.lower() in text.lower()])})
    attachments = PROJECT_ROOT / "real_attachments"
    paths = {attachments/"source_manifest.json", attachments/"attachment_size_summary.csv"}
    paths.update(attachments.rglob("*.md"))
    for pattern in ("*metadata*.csv", "*config*.json", "*.yaml", "*.yml"):
        paths.update(attachments.rglob(pattern))
    for path in sorted(paths):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8-sig")
        rows.append({"source_path": path.relative_to(PROJECT_ROOT).as_posix(), "source_kind": "local_metadata",
                     "sha256": sha256_file(path), "text_characters": len(text), "extraction_warning_count": 0,
                     "keyword_hits": json_cell([k for k in keywords if k.lower() in text.lower()])})
    write_rows(output/"local_semantics_evidence_inventory.csv", rows)
    LOGGER.info("Reviewed %s local documents/metadata files", len(rows))
    return rows


@dataclass
class VectorAudit:
    """Retain only compact numeric vectors and bounded, provenance-tagged examples."""
    metric: str
    length: int
    sample_limit: int
    counts: Counter = field(default_factory=Counter)
    lengths: Counter = field(default_factory=Counter)
    values: array = field(default_factory=lambda: array("d"))
    examples: list[dict] = field(default_factory=list)

    def add(self, value, raw_name, provenance, invalid_rows):
        """Classify raw vector; no replacement of NaN and no length guessing."""
        self.counts["records"] += 1
        if raw_name is None or value is None or value == []:
            self.counts["missing"] += 1
            return
        if not isinstance(value, list):
            self.counts["invalid_type"] += 1
            invalid_rows.append({**provenance, "metric": self.metric, "reason": "not a list"})
            return
        self.lengths[len(value)] += 1
        if len(value) != self.length:
            self.counts["length_mismatch"] += 1
            invalid_rows.append({**provenance,"metric":self.metric,"reason":"unexpected length",
                                 "raw_vector":json_cell(value)})
            return
        if not all(isinstance(v,Real) and not isinstance(v,bool) for v in value):
            self.counts["invalid_element"] += 1
            invalid_rows.append({**provenance,"metric":self.metric,"reason":"nonnumeric element"})
            return
        if not all(math.isfinite(v) for v in value):
            self.counts["nonfinite"] += 1
            self.counts["nan_vector"] += any(math.isnan(v) for v in value)
            self.counts["inf_vector"] += any(math.isinf(v) for v in value)
            printable = [v if math.isfinite(v) else "NaN" if math.isnan(v) else "Inf" if v>0 else "-Inf" for v in value]
            invalid_rows.append({**provenance,"metric":self.metric,"reason":"nonfinite raw vector; Softmax not defined",
                                 "raw_vector":json_cell(printable)})
            return
        self.counts["valid"] += 1
        self.values.extend(value)
        if len(self.examples) < self.sample_limit:
            self.examples.append({**provenance,"raw_vector":value.copy()})

    def summarize(self, dataset: str, settings: dict, metadata: dict):
        """Compute raw distributions, argmax and batched numerical Softmax checks."""
        if self.counts["length_mismatch"] or self.counts["invalid_type"]:
            raise ValueError(f"{dataset}/{self.metric}: incompatible vector structure; observed lengths={dict(self.lengths)}")
        if not self.counts["valid"]:
            raise ValueError(f"{dataset}/{self.metric}: no valid vectors")
        matrix = np.frombuffer(self.values,dtype=np.float64).reshape(-1,self.length)
        raw_sums = matrix.sum(axis=1)
        negative_count = int(np.any(matrix<0,axis=1).sum())
        summary = {"dataset":dataset,"metric":self.metric,"record_count":self.counts["records"],
                   "valid_count":len(matrix),"missing_count":self.counts["missing"],
                   "missing_rate":self.counts["missing"]/self.counts["records"],
                   "invalid_count":self.counts["records"]-len(matrix)-self.counts["missing"],
                   "nonfinite_vector_count":self.counts["nonfinite"],
                   "raw_nan_vector_count":self.counts["nan_vector"],"raw_inf_vector_count":self.counts["inf_vector"],
                   "length_mismatch_count":self.counts["length_mismatch"],
                   "length_distribution":json_cell(dict(sorted(self.lengths.items()))),
                   "actual_min":float(matrix.min()),"actual_max":float(matrix.max()),
                   "negative_vector_count":negative_count,"negative_vector_ratio":negative_count/len(matrix),
                   "raw_sum_near_one_count":int((np.abs(raw_sums-1)<settings["raw_sum_tolerance"]).sum()),
                   "raw_sum_near_one_ratio":float((np.abs(raw_sums-1)<settings["raw_sum_tolerance"]).mean()),
                   "upstream_representation":metadata["fields"][self.metric]["upstream_representation"],
                   "upstream_index_labels":json_cell(metadata["fields"][self.metric]["upstream_index_labels"]),
                   "local_class_order_confirmed":False,"attachment_class_order_confirmed":False,
                   "class_order_status":"类别顺序未确认（本地）；上游定义另见证据表",
                   "allowed_next_stage":False,"status":"NEEDS_CONFIRMATION"}
        position_rows=[{"dataset":dataset,"metric":self.metric,"position_zero_based":p,
                        **distribution(matrix[:,p])} for p in range(self.length)]
        dist_rows=[{"dataset":dataset,"metric":self.metric,"quantity":"raw_sum",**distribution(raw_sums)}]
        if self.metric in BINARY:
            dist_rows.append({"dataset":dataset,"metric":self.metric,"quantity":"l1_minus_l0",
                              **distribution(matrix[:,1]-matrix[:,0])})
        argmax=np.argmax(matrix,axis=1)
        if self.metric in (*MODERN,*BINARY):
            summary.update({"suspected_logits":negative_count>0,"softmax_tested_count":0,
                            "softmax_nonfinite_count":0,"softmax_out_of_bounds_count":0,
                            "softmax_sum_failure_count":0,"softmax_argmax_mismatch_count":0,
                            "softmax_max_sum_error":0.0,"softmax_min":1.0,"softmax_max":0.0,
                            "softmax_unavailable_raw_count":summary["invalid_count"],
                            "raw_argmax_tie_count":int(((matrix==matrix.max(axis=1,keepdims=True)).sum(axis=1)>1).sum())})
            for start in range(0,len(matrix),settings["batch_size"]):
                raw=matrix[start:start+settings["batch_size"]]
                probabilities=stable_softmax(raw)
                errors=np.abs(probabilities.sum(axis=1)-1)
                summary["softmax_tested_count"]+=len(raw)
                summary["softmax_nonfinite_count"]+=int((~np.isfinite(probabilities).all(axis=1)).sum())
                summary["softmax_out_of_bounds_count"]+=int(((probabilities<0)|(probabilities>1)).any(axis=1).sum())
                summary["softmax_sum_failure_count"]+=int((errors>=settings["softmax_sum_tolerance"]).sum())
                summary["softmax_argmax_mismatch_count"]+=int((np.argmax(raw,axis=1)!=np.argmax(probabilities,axis=1)).sum())
                summary["softmax_max_sum_error"]=max(summary["softmax_max_sum_error"],float(errors.max()))
                summary["softmax_min"]=min(summary["softmax_min"],float(probabilities.min()))
                summary["softmax_max"]=max(summary["softmax_max"],float(probabilities.max()))
            summary["softmax_valid"]=all(summary[key]==0 for key in ["softmax_nonfinite_count",
                "softmax_out_of_bounds_count","softmax_sum_failure_count","softmax_argmax_mismatch_count"])
            summary["recommended_conversion"]="仅诊断 stable Softmax；类别确认后再决定标量化，不正式使用"
        else:
            summary["suspected_logits"]="NOT_ESTABLISHED"
            summary["softmax_valid"]="NOT_APPLICABLE"
            if self.metric=="fineweb_edu":
                ready=self.lengths==Counter({1:self.counts["records"]}) and summary["invalid_count"]==0 and not self.counts["missing"]
                summary.update({"allowed_next_stage":ready,"status":"READY_FOR_EXTRACTION" if ready else "NEEDS_CONFIRMATION",
                                "class_order_status":"NOT_APPLICABLE","recommended_conversion":"可直接取 v1" if ready else "停止并核验单分量结构"})
            else:
                ranges=np.ptp(matrix,axis=0);stds=matrix.std(axis=0,ddof=1)
                summary.update({"component_range_ratio":float(ranges.max()/ranges.min()) if ranges.min()>0 else None,
                                "component_std_ratio":float(stds.max()/stds.min()) if stds.min()>0 else None,
                                "same_scale_semantics_confirmed":False,
                                "recommended_conversion":"保留四项诊断；不得自动平均，需确认同尺度与方向"})
        argmax_rows=[{"dataset":dataset,"metric":self.metric,"index":i,"count":int((argmax==i).sum()),
                      "ratio":float((argmax==i).mean()),"tie_policy":"first_index"} for i in range(self.length)]
        samples=[]
        for example in self.examples:
            row={"dataset":dataset,"metric":self.metric,**example}
            raw=row.pop("raw_vector")
            row["raw_vector"]=json_cell(raw)
            if self.metric in (*MODERN,*BINARY):
                p=stable_softmax(np.asarray([raw]))[0]
                row.update({"softmax_probabilities":json_cell(p.tolist()),"argmax_index":int(np.argmax(raw)),
                            "class_order_confirmed":False,"diagnostic_only":True})
                if self.metric in MODERN:
                    row["expected_grade_0_to_5_diagnostic"]=float(p@np.arange(6))
            samples.append(row)
        pearson_rows=[];spearman_rows=[]
        if self.metric=="qurater":
            pearson,spearman=correlation_tables(matrix)
            for result,target in [(pearson,pearson_rows),(spearman,spearman_rows)]:
                target.extend({"dataset":dataset,"position_zero_based":i,
                               **{f"position_{j}":float(result[i,j]) for j in range(4)}} for i in range(4))
        return summary,position_rows,dist_rows,argmax_rows,samples,pearson_rows,spearman_rows


def write_report(output, summary, positions, distributions, argmax, files, local_evidence, metadata):
    """Write the required measured semantics report with explicit pending status."""
    lines=["# 向量语义诊断：PHASE 2", "",
           "本轮全量读取 A1/A2/A3，仅检查结构、数值和来源证据。不生成质量特征、CRITIC 或 Q。", "",
           "Softmax 数值有效不等于原值必为 logits，也不证明概率经过校准。原始 NaN 向量无法转换，单独记录。", "",
           "## 实际输入", "", "| 数据集 | 文件 | 记录数 | 来源领域计数 | 原件 SHA-256 一致 |",
           "|---|---|---:|---|---|"]
    for r in files:
        lines.append(f"| {r['dataset']} | `{r['source_path']}` | {r['record_count']} | {r['domain_counts']} | {r['source_unchanged']} |")
    lines += ["", "终止标点字段别名集中定义在 configs/quality_metrics.yaml；不改原文件。",
              "A1 采用原 _source_domain；A2/A3 采用实际文件名 arxiv_/github_ 前缀。规则、领域计数和原始别名使用数均写入清单。", "",
              "## 字段诊断与状态", "",
              "| 数据集 | 字段 | 实际长度分布 | 实际值域（有限值） | 含负数 | 原始 sum≈1 | 疑似 logits | Softmax 有效 | 类别顺序 | 推荐转换 | 允许下一阶段 | status |",
              "|---|---|---|---|---|---:|---|---|---|---|---|---|"]
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['metric']} | {r['length_distribution']} | [{r['actual_min']:.12g}, {r['actual_max']:.12g}] | "
                     f"{r['negative_vector_count']>0} | {r['raw_sum_near_one_ratio']:.6%} | {r['suspected_logits']} | "
                     f"{r['softmax_valid']} | {r['class_order_status']} | {r['recommended_conversion']} | {r['allowed_next_stage']} | {r['status']} |")
    lines += ["", "缺失、非法与非有限值分开统计；sum≈1 比例仅使用合法、有限、预期长度的向量。",
              "仅 fineweb_edu 标为可提取；其他字段均待确认。READY_FOR_EXTRACTION 不代表允许运行后续脚本。", "",
              "## 原始分布与位置统计", "",
              "| 数据集 | 字段 | 位置 | min | max | mean | median | std |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in positions:
        lines.append(f"| {r['dataset']} | {r['metric']} | {r['position_zero_based']} | {r['min']:.9g} | {r['max']:.9g} | {r['mean']:.9g} | {r['median']:.9g} | {r['std']:.9g} |")
    lines += ["", "| 数据集 | 字段 | 分布量 | min | P25 | median | mean | P75 | max |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in distributions:
        lines.append(f"| {r['dataset']} | {r['metric']} | {r['quantity']} | {r['min']:.9g} | {r['p25']:.9g} | {r['median']:.9g} | {r['mean']:.9g} | {r['p75']:.9g} | {r['max']:.9g} |")
    lines += ["", "完整分位数包含 P01/P05/P10/P25/P50/P75/P90/P95/P99；使用线性插值。", "",
              "## Argmax 类别占比（不赋予二分类语义）", "", "| 数据集 | 字段 | index | count | 占比 |", "|---|---|---:|---:|---:|"]
    for r in argmax:
        if r['metric'] in (*MODERN,*BINARY):
            lines.append(f"| {r['dataset']} | {r['metric']} | {r['index']} | {r['count']} | {r['ratio']:.6%} |")
    lines += ["", "## Softmax 全量数值验证", "",
              "| 数据集 | 字段 | 已转换有限向量 | 原始 NaN/Inf 向量 | 越界 | 输出 NaN/Inf | 和失败 | argmax 不一致 | 最大和误差 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        if r['metric'] not in (*MODERN,*BINARY): continue
        lines.append(f"| {r['dataset']} | {r['metric']} | {r['softmax_tested_count']} | {r['nonfinite_vector_count']} | "
                     f"{r['softmax_out_of_bounds_count']} | {r['softmax_nonfinite_count']} | {r['softmax_sum_failure_count']} | "
                     f"{r['softmax_argmax_mismatch_count']} | {r['softmax_max_sum_error']:.3g} |")
    lines += ["", "采用 exp(l−max(l))/sum(exp(l−max(l)))，float64；和误差严格小于配置容差 1e-12。",
              "NaN/Inf 原始向量未填补、未转换，其来源和原值见 vector_invalid_records.csv；该文件用字符串 NaN/Inf 表示非有限原值。",
              "argmax 若并列取最小索引，并另记原始并列数。现代分类期望按假设 index=等级 0…5 计算，范围为 0…5，仅用于诊断。", "",
              "## qurater：尺度与相关性", ""]
    for r in summary:
        if r['metric']=='qurater':
            lines.append(f"- {r['dataset']}：四维极差最大/最小比={r['component_range_ratio']:.6g}，标准差最大/最小比={r['component_std_ratio']:.6g}。")
    lines += ["", "这些数字衡量观察到的尺度差异；不能证明评分量纲、方向或语义可交换。每维位置分布见上表。",
              "四维极差和标准差处于接近的数量级，但不完全一致；位置均值、中位数和相关性随数据集变化。",
              "四项是不同评分维度，统计相似性不构成自动平均的依据；未对 qurater 应用 Softmax 或均值。", ""]
    for filename,label in [("qurater_pearson.csv","Pearson"),("qurater_spearman.csv","Spearman（并列值采用平均秩）")]:
        import csv
        with (output/filename).open(encoding='utf-8-sig',newline='') as stream: correlations=list(csv.DictReader(stream))
        lines += [f"### {label}", "", "| 数据集 | 位置 | 0 | 1 | 2 | 3 |", "|---|---:|---:|---:|---:|---:|"]
        for r in correlations:
            lines.append(f"| {r['dataset']} | {r['position_zero_based']} | " + " | ".join(f"{float(r[f'position_{j}']):.6f}" for j in range(4))+" |")
        lines.append("")
    lines += ["## 类别证据与未确认项", "",
              "本地 PDF 可提取文本、Word XML 文本、附件 README 和来源/模型元数据中未发现这八项的明确类别索引定义。",
              "PDF 存在字体映射提取警告；不能用关键词未命中证明原件绝无定义。因此本地类别顺序记录为未确认。",
              "完整本地检索范围和提取警告数见 local_semantics_evidence_inventory.csv。", "",
              f"附件 source_manifest.json 指向的[上游数据集 README]({metadata['source_url']})（检索日期 {metadata['retrieved_date']}）说明：", "",
              "- modernbert 四项为六级 logits，文档按索引 0…5 表示等级；文档推荐 argmax，Softmax 期望是本轮另行诊断的候选方式。",
              "- fluency_en 的上游索引为 [不流畅, 流畅]；ad_en 的上游索引为 [有广告, 无广告]。",
              "- qurater 四项依次涉及写作、专业知识要求、事实知识、教育价值；未证实相同量纲和独立性。", "",
              "上游语义证据和本地确认分开记录。按用户要求，本轮不把上游映射自动批准为附件级转换规则。",
              "因此这些字段 status=NEEDS_CONFIRMATION；不能据此启动正式标量化。尤其 ad_en 不应误取 p1 当作广告概率。", "",
              "## 真实诊断样例与输出", "",
              "- vector_semantics_samples.txt：每数据集每个现代分类字段 20 条；每个二分类字段 30 条；qurater 20 条。",
              "- modernbert_softmax_samples.csv：原值、Softmax、argmax、0…5 假设期望（diagnostic_only=True）。",
              "- binary_softmax_samples.csv：原值、[p0,p1]、argmax，不在数据列中绑定实际类别。",
              "- qurater_raw_samples.csv：真实四维值，不平均。",
              "- vector_semantics_summary.csv / vector_position_profile.csv / vector_raw_distributions.csv：全量统计。",
              "- vector_argmax_distribution.csv / qurater_pearson.csv / qurater_spearman.csv：占比与四维相关。", "",
              "本轮结束并停止，未运行 02/03/04；未计算 CRITIC 或 Q。"]
    (output/"vector_semantics_diagnosis.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main() -> int:
    """Run all Phase 2 diagnostics, emit a report, stop on structural mismatch."""
    output=PROJECT_ROOT/"outputs"/"inspection";output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),logging.FileHandler(output/"vector_semantics.log",mode="w",encoding="utf-8")])
    with (PROJECT_ROOT/"configs"/"quality_metrics.yaml").open(encoding="utf-8") as stream: config=yaml.safe_load(stream)
    settings={**config["vector_diagnosis"],"raw_sum_tolerance":config["inspection"]["sum_tolerance"]}
    with (output/"local_evidence"/"source_metadata_notes.json").open(encoding="utf-8") as stream: metadata=json.load(stream)
    local_evidence=inspect_local_evidence(output)
    specs={m["name"]:m for m in config["metrics"] if m["kind"]=="list"}
    if len(specs)!=8: raise ValueError("Exactly eight list fields are required")
    summary=[];positions=[];distributions=[];argmax=[];samples=[];pearson=[];spearman=[];files=[];invalid=[];parse_failures=[]
    punctuation=next(m for m in config["metrics"] if m.get("raw_aliases"))
    for dataset,paths in discover_quality_files(PROJECT_ROOT).items():
        audits={name:VectorAudit(name,settings["expected_lengths"][name],
            settings["modernbert_sample_count"] if name in MODERN else settings["binary_sample_count"] if name in BINARY else
            settings["qurater_sample_count"] if name=="qurater" else 0) for name in specs}
        for path in paths:
            relative=path.relative_to(PROJECT_ROOT).as_posix();before=sha256_file(path)
            decoded=0;lines=0;domains=Counter();domain_rules=Counter();alias_names=Counter()
            LOGGER.info("Reading %s: %s",dataset,relative)
            for line_number,record,error in iter_jsonl(path):
                lines+=1
                if error:
                    parse_failures.append({"dataset":dataset,"source_path":relative,"line_number":line_number,"error":error})
                    continue
                decoded+=1
                domain,rule=infer_source_domain(path,record);domains[str(domain)]+=1;domain_rules[rule]+=1
                _,raw_name=resolve_metric_value(record,punctuation);alias_names[str(raw_name)]+=1
                provenance={"dataset":dataset,"source_path":relative,"line_number":line_number,"id":record.get("id"),"domain":domain,"domain_method":rule}
                for name,spec in specs.items():
                    value,raw_name=resolve_metric_value(record,spec)
                    audits[name].add(value,raw_name,provenance,invalid)
                if decoded%config["inspection"]["progress_every"]==0: LOGGER.info("%s: %s records",dataset,decoded)
            after=sha256_file(path)
            if before!=after: raise RuntimeError(f"Original file changed: {relative}")
            files.append({"dataset":dataset,"source_path":relative,"physical_lines":lines,"record_count":decoded,
                          "json_parse_failure_count":lines-decoded,"domain_counts":json_cell(domains),"domain_rule_counts":json_cell(domain_rules),
                          "punctuation_raw_field_counts":json_cell(alias_names),"sha256_before":before,"sha256_after":after,"source_unchanged":True})
        for name,audit in audits.items():
            result=audit.summarize(dataset,settings,metadata)
            for target,rows in zip([summary,positions,distributions,argmax,samples,pearson,spearman],result):
                target.append(rows) if isinstance(rows,dict) else target.extend(rows)
            LOGGER.info("%s/%s: %s valid, %s invalid; status=%s",dataset,name,result[0]["valid_count"],result[0]["invalid_count"],result[0]["status"])
        del audits
    outputs={"vector_semantics_summary.csv":summary,"vector_position_profile.csv":positions,"vector_raw_distributions.csv":distributions,
             "vector_argmax_distribution.csv":argmax,"qurater_pearson.csv":pearson,"qurater_spearman.csv":spearman,
             "vector_semantics_input_manifest.csv":files,"vector_invalid_records.csv":invalid,"vector_json_parse_failures.csv":parse_failures,
             "modernbert_softmax_samples.csv":[r for r in samples if r["metric"] in MODERN],
             "binary_softmax_samples.csv":[r for r in samples if r["metric"] in BINARY],
             "qurater_raw_samples.csv":[r for r in samples if r["metric"]=="qurater"]}
    for filename,rows in outputs.items(): write_rows(output/filename,rows,["dataset","source_path","line_number","error"])
    with (output/"vector_semantics_samples.txt").open("w",encoding="utf-8") as stream:
        stream.write("Phase 2 only. Hypothetical ModernBERT levels 0..5; no confirmed attachment class labels.\n")
        for row in samples:
            line=json_cell(row);print(line);stream.write(line+"\n")
    write_report(output,summary,positions,distributions,argmax,files,local_evidence,metadata)
    LOGGER.info("Completed %s records; STOP before scalar features, CRITIC or Q",sum(r['record_count'] for r in files))
    return 1 if parse_failures or any(r["length_mismatch_count"] for r in summary) else 0


if __name__=="__main__":
    try: sys.exit(main())
    except Exception:
        LOGGER.exception("Vector diagnosis failed; do not proceed to any later stage.")
        sys.exit(1)
