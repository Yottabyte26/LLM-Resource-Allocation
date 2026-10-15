"""Phase 3: approved 22-indicator scalarization and descriptive profiling only.

Raw XZ records are streamed. Only numeric arrays and missing-value provenance
are retained. No content, imputation, direction normalization, CRITIC or Q.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quality.io import discover_quality_files, infer_source_domain, iter_jsonl, resolve_metric_value, sha256_file
from quality.profiling import profile_matrix
from quality.vector_diagnostics import json_cell, write_rows
from quality.vector_transform import fit_qurater_parameters, scalarize_record, transform_qurater

LOGGER = logging.getLogger("quality.profile")


def validate_configuration(config: dict) -> None:
    """Enforce approved scope and prevent accidental reversal or dimension drift."""
    metrics = config["metrics"]
    if len(metrics) != 22 or len({m["name"] for m in metrics}) != 22:
        raise ValueError("Exactly 22 distinct official metrics required")
    if sum(m["kind"]=="list" for m in metrics)!=8:
        raise ValueError("Exactly 8 list fields required")
    directions = Counter(m["direction"] for m in metrics)
    if directions != Counter(positive=14,negative=5,moderate=3):
        raise ValueError(f"Wrong approved direction counts: {dict(directions)}")
    ad = next(m for m in metrics if m["name"]=="ad_en")
    if ad["direction"]!="positive" or ad["index_labels"]!=["has_ad","no_ad"]:
        raise ValueError("ad_en must remain argmax([has_ad,no_ad]), positive")
    if config["scalarization"]["status"]!="USER_CONFIRMED":
        raise ValueError("Vector semantics have not been approved")
    if config["scalarization"]["softmax_expectation_main_result"]:
        raise ValueError("Softmax expectation is forbidden in the main result")
    if config["scalarization"]["argmax_tie_policy"]!="first_index":
        raise ValueError("Current argmax tie policy is the first index")


def read_scalar_records(config: dict):
    """Stream every official row into compact scalar/raw-qurater arrays."""
    scalar_buffer, qurater_buffer = array("d"), array("d")
    dataset_buffer, domain_buffer = array("B"), array("H")
    domains, manifests, missing_records = {}, [], []
    dataset_names = []
    metrics = config["metrics"]
    punctuation = next(m for m in metrics if m.get("raw_aliases"))
    for dataset_code, (dataset, paths) in enumerate(discover_quality_files(PROJECT_ROOT).items()):
        dataset_names.append(dataset)
        for path in paths:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            before = sha256_file(path)
            decoded = 0
            domain_counts, domain_methods, aliases = Counter(), Counter(), Counter()
            LOGGER.info("Reading %s: %s",dataset,relative)
            for line_number, record, error in iter_jsonl(path):
                if error:
                    raise ValueError(f"JSON parsing failed at {relative}:{line_number}: {error}")
                domain, domain_method = infer_source_domain(path,record)
                domain = domain if domain is not None else "<unknown>"
                if domain not in domains:
                    domains[domain] = len(domains)
                _, raw_name = resolve_metric_value(record,punctuation)
                aliases[str(raw_name)] += 1
                provenance = {"dataset":dataset,"source_path":relative,"line_number":line_number,
                              "id":record.get("id"),"domain":domain,"domain_method":domain_method}
                try:
                    scalars, components, issues = scalarize_record(record,metrics)
                except ValueError as exc:
                    LOGGER.error("Structure conflict: %s",json_cell(provenance))
                    raise ValueError(f"{relative}:{line_number}: {exc}") from exc
                if len(scalars)!=22:
                    raise ValueError("Scalar dimensionality changed")
                scalar_buffer.extend(scalars)
                qurater_buffer.extend(components)
                dataset_buffer.append(dataset_code)
                domain_buffer.append(domains[domain])
                missing_records.extend({**provenance,**issue,"scalar_state":"NaN_before_imputation",
                                        "imputation_performed":False,
                                        "planned_imputation":"domain_median_then_global_median"} for issue in issues)
                decoded += 1
                domain_counts[domain] += 1
                domain_methods[domain_method] += 1
                if decoded%config["profile"]["progress_every"]==0:
                    LOGGER.info("%s: %s records scalarized",dataset,decoded)
            after = sha256_file(path)
            if before!=after:
                raise RuntimeError(f"Source changed while reading: {relative}")
            manifests.append({"dataset":dataset,"source_path":relative,"record_count":decoded,
                              "domain_counts":json_cell(domain_counts),"domain_method_counts":json_cell(domain_methods),
                              "punctuation_raw_field_counts":json_cell(aliases),
                              "sha256_before":before,"sha256_after":after,"source_unchanged":True,
                              "compressed_bytes":path.stat().st_size})
    n = len(dataset_buffer)
    matrix = np.frombuffer(scalar_buffer,dtype=np.float64).reshape(n,22)
    components = np.frombuffer(qurater_buffer,dtype=np.float64).reshape(n,4)
    dataset_codes = np.frombuffer(dataset_buffer,dtype=np.uint8)
    domain_codes = np.frombuffer(domain_buffer,dtype=np.uint16)
    return matrix,components,dataset_codes,domain_codes,dataset_names,domains,manifests,missing_records


def build_profiles(matrix, dataset_codes, domain_codes, dataset_names, domains, config):
    """Profile overall, datasets, pooled domains and within-dataset domains."""
    names = [m["name"] for m in config["metrics"]]
    settings = config["profile"]
    quality_rows = profile_matrix(matrix,names,{"scope":"overall","group":"ALL","dataset":"ALL","domain":"ALL"},settings)
    domain_rows, group_counts = [], []
    for code,dataset in enumerate(dataset_names):
        subset = matrix[dataset_codes==code]
        label = dataset if dataset=="A1" else f"{dataset}-{'arxiv' if dataset=='A2' else 'github'}"
        quality_rows.extend(profile_matrix(subset,names,{"scope":"dataset","group":label,"dataset":dataset,"domain":"ALL"},settings))
        group_counts.append({"scope":"dataset","dataset":dataset,"domain":"ALL","n":len(subset)})
    for domain,code in domains.items():
        mask = domain_codes==code
        subset = matrix[mask]
        domain_rows.extend(profile_matrix(subset,names,{"scope":"domain","group":domain,"dataset":"ALL","domain":domain},settings))
        group_counts.append({"scope":"domain","dataset":"ALL","domain":domain,"n":len(subset)})
        for dataset_code,dataset in enumerate(dataset_names):
            within_mask = mask & (dataset_codes==dataset_code)
            if not within_mask.any():
                continue
            subset = matrix[within_mask]
            domain_rows.extend(profile_matrix(subset,names,{"scope":"dataset_domain","group":f"{dataset}-{domain}",
                                                          "dataset":dataset,"domain":domain},settings))
            group_counts.append({"scope":"dataset_domain","dataset":dataset,"domain":domain,"n":len(subset)})
    return quality_rows,domain_rows,group_counts


def validate_scalar_results(matrix: np.ndarray, config: dict, missing_records: list[dict]) -> list[dict]:
    """Check final dimensions, argmax lattices and a one-to-one missing audit."""
    if matrix.ndim!=2 or matrix.shape[1]!=22 or np.isinf(matrix).any():
        raise ValueError("Invalid final 22-scalar matrix")
    missing = int(np.isnan(matrix).sum())
    if missing!=len(missing_records):
        raise ValueError(f"Missing provenance mismatch: matrix={missing}, audit={len(missing_records)}")
    issues = Counter(r["metric"] for r in missing_records)
    rows = []
    for j,metric in enumerate(config["metrics"]):
        values = matrix[np.isfinite(matrix[:,j]),j]
        nan_count = int(np.isnan(matrix[:,j]).sum())
        if nan_count!=issues[metric["name"]]:
            raise ValueError(f"Missing audit mismatch for {metric['name']}")
        method = metric.get("scalarization","raw_scalar")
        if method=="argmax_scaled":
            if not np.isin(values,np.arange(metric["expected_length"])/metric["divisor"]).all():
                raise ValueError("ModernBERT scalar is not on the approved argmax lattice")
        elif method=="argmax_index":
            if not np.isin(values,[0.0,1.0]).all():
                raise ValueError("Binary scalar is not 0/1")
        elif method=="component_winsor_minmax_mean":
            if not ((values>=0)&(values<=1)).all():
                raise ValueError("qurater scalar outside [0,1]")
        rows.append({"metric":metric["name"],"direction":metric["direction"],"method":method,
                     "n":len(matrix),"valid_count":len(values),"missing":nan_count,
                     "dimension_check":True,"value_check":True,"missing_audit_check":True})
    return rows


def write_summary(output: Path, quality_rows, counts, parameters, manifest, missing_records, config):
    """Generate the current descriptive-statistics report using actual outputs."""
    lines = ["# PHASE 3：22 维标量指标描述性统计", "",
             "已按人工确认规则完成标量化，仅做描述性统计。未填补缺失，未对其他连续指标截尾，",
             "未构造统一方向的 Z，未运行 03/04，未计算 CRITIC 或 Q。", "",
             "## 数据与口径", "", "| 数据集 | 实际文件 | 全部记录数 | 输入哈希一致 |", "|---|---|---:|---|"]
    for r in manifest:
        lines.append(f"| {r['dataset']} | `{r['source_path']}` | {r['record_count']} | {r['source_unchanged']} |")
    total = sum(r["record_count"] for r in manifest)
    lines += ["",f"合计 {total:,} 条，严格保留 22 个指标，不按 ID 去重。content 未累计或写入结果。",
              "A1 采用原领域；A2/A3 由文件名前缀推断。总体与合并领域统计使用全部记录；",
              "同时保留数据集内领域统计用于区分抽样集与扩展集，不应把重叠记录视作独立实验。", "",
              "count 是有限非缺失数，n 是该分组全部记录数，missing=n−count，std 使用 ddof=1。",
              "分位数采用线性插值。空组不生成；指标全缺失时统计值留空，不填 0。", "",
              "## 正式标量化", "",
              "- fineweb_edu：取唯一分量，保留原值。",
              "- modernbert 四项：argmax(logits)/5，取值为 {0,0.2,0.4,0.6,0.8,1}。",
              "- fluency_en：argmax([not_fluent,fluent])，0/1 正向。",
              "- ad_en：argmax([has_ad,no_ad])，0/1 正向；未反向。",
              "- qurater：四个子维度分别 P01-P99 截尾、独立 Min-Max，之后平均；最终只保留一个指标。",
              "- 分类向量含 NaN/Inf 时该标量为 NaN，不删除整条记录。",
              "- 并列 argmax 取最小索引；Softmax 期望辅助函数保留但主流程不调用。", "",
              "方向已更新为 14 正向、5 负向、3 适中；当前只记录方向，不执行反向或适中效用函数。", "",
              "## qurater 统一拟合参数", "",
              "参数使用 A1/A2/A3 全部有限子维度记录合并拟合。A1 若与扩展文件出现相同 ID，仍按原始记录保留。",
              "未分别对数据集或领域拟合，保证相同原值在各组中得到同一转换。", "",
              "| 子维度 | 有效数 | P01 | P99 | 截尾后 min | 截尾后 max | 下侧截尾数 | 上侧截尾数 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in parameters:
        lines.append(f"| {r['component_name']} | {r['valid_count']} | {r['P01']:.12g} | {r['P99']:.12g} | "
                     f"{r['min_after_winsorization']:.12g} | {r['max_after_winsorization']:.12g} | {r['clipped_lower_count']} | {r['clipped_upper_count']} |")
    lines += ["", "若某子维度常数，配置将其有效值记为 0.5 以避免除零；实际是否触发见参数表，不推断额外区分度。", "",
              "## 总体 22 指标概览", "", "| 指标 | count | missing | missing_rate | mean | std | min | P50 | max |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in quality_rows:
        if r['scope']!='overall': continue
        fields=" | ".join("" if r[k] is None else f"{r[k]:.10g}" for k in ['mean','std','min','P50','max'])
        lines.append(f"| {r['metric']} | {r['count']} | {r['missing']} | {r['missing_rate']:.8%} | {fields} |")
    lines += ["", "## 领域规模", "", "| scope | dataset | domain | n |", "|---|---|---|---:|"]
    for r in counts:
        if r['scope']!='dataset': lines.append(f"| {r['scope']} | {r['dataset']} | {r['domain']} | {r['n']} |")
    lines += ["", "## 缺失审计与下一阶段边界", "",
              f"实际 {len(missing_records)} 个标量缺失事件；涉及 {len({(r['dataset'],r['source_path'],r['line_number']) for r in missing_records})} 条来源记录。",
              "所有事件保留 id、dataset、domain、source_path、line_number、metric、reason。",
              "scalar_missing_records.csv 中 imputation_performed=False；后续按领域中位数优先、总体中位数回退，",
              "到实际填补时再另行记录填补值、方法及数量。本轮没有删除样本或填补缺失。", "",
              "## 输出", "",
              "- quality_indicator_profile.csv：总体与 A1/A2-arxiv/A3-github，每组 22 行。",
              "- domain_indicator_profile.csv：scope=domain 为全数据合并领域；scope=dataset_domain 为数据集内领域。",
              "- missing_summary.csv：对应所有分组的缺失汇总；分组有交集，不能把整表缺失数相加当作独立事件数。",
              "- scalar_missing_records.csv：原始缺失事件及真实 ID，不包含人为的 qurater 拟合前占位。",
              "- group_record_counts.csv / profile_input_manifest.csv / scalar_validation.csv：规模、哈希、22 维和取值核验。",
              "- ../preprocess/qurater_transform_parameters.csv：四个子维度实际阈值和截尾后范围。", "",
              "统计描述的是转换后的指标分布，不是文本综合质量或领域质量结论。RPS 原始数值口径保留，",
              "DSIR 负值保留；本轮不执行连续指标的全局截尾或统一 [0,1] 归一化。"]
    (output/"quality_profile_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main() -> int:
    """Produce the three requested profile tables and stop before feature building."""
    output = PROJECT_ROOT/"outputs"/"profile"
    preprocess = PROJECT_ROOT/"outputs"/"preprocess"
    output.mkdir(parents=True,exist_ok=True)
    preprocess.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),logging.FileHandler(output/"profile.log",mode="w",encoding="utf-8")])
    config_path = PROJECT_ROOT/"configs"/"quality_metrics.yaml"
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes.decode("utf-8"))
    validate_configuration(config)
    matrix,components,dataset_codes,domain_codes,datasets,domains,manifest,missing_records = read_scalar_records(config)
    names = [m["name"] for m in config["metrics"]]
    parameters = fit_qurater_parameters(components,config["scalarization"]["qurater"])
    matrix[:,names.index("qurater")] = transform_qurater(components,parameters)
    checks = validate_scalar_results(matrix,config,missing_records)
    quality_rows,domain_rows,counts = build_profiles(matrix,dataset_codes,domain_codes,datasets,domains,config)
    missing_rows = [{key:r[key] for key in ["scope","group","dataset","domain","metric","n","count","missing","missing_rate"]}
                    for r in quality_rows+domain_rows]
    write_rows(preprocess/"qurater_transform_parameters.csv",parameters)
    outputs = {"quality_indicator_profile.csv":quality_rows,"domain_indicator_profile.csv":domain_rows,
               "missing_summary.csv":missing_rows,"scalar_missing_records.csv":missing_records,
               "group_record_counts.csv":counts,"profile_input_manifest.csv":manifest,"scalar_validation.csv":checks}
    for filename,rows in outputs.items():
        write_rows(output/filename,rows,["dataset","source_path","line_number","id","domain","metric","reason"])
    write_summary(output,quality_rows,counts,parameters,manifest,missing_records,config)
    (output/"profile_config_snapshot.yaml").write_bytes(config_bytes)
    LOGGER.info("Configuration SHA-256: %s",hashlib.sha256(config_bytes).hexdigest())
    LOGGER.info("Completed: %s records × 22 scalar indicators, %s domains, %s actual missing scalar events",
                len(matrix),len(domains),len(missing_records))
    LOGGER.info("STOP: no imputation, feature normalization, CRITIC or Q; scripts 03/04 not run.")
    return 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception:
        LOGGER.exception("Profiling failed; dependent steps must not proceed.")
        sys.exit(1)
