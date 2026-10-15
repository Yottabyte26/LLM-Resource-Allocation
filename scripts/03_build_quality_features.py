"""Build only 22-dimensional Z with an A1-only calibration model (CPU)."""

from __future__ import annotations

import csv
import json
import logging
import sys
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quality.io import discover_quality_files, infer_source_domain, iter_jsonl, sha256_file
from quality.preprocess import apply_preprocessing, fit_preprocessing_model, parameter_tables, validate_config
from quality.vector_diagnostics import write_rows
from quality.vector_transform import scalarize_record

LOGGER = logging.getLogger("quality.features")


def load_csv(path: Path) -> list[dict]:
    """Read a small existing diagnostic or parameter table."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_dataset(dataset: str, paths: list[Path], config: dict):
    """Stream official XZ files; retain numeric features and provenance, no text."""
    scalar_buffer, component_buffer = array("d"), array("d")
    metadata, issues, manifest, ids = [], [], [], set()
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        before = sha256_file(path)
        count = 0
        LOGGER.info("Reading %s: %s", dataset, relative)
        for line_number, record, error in iter_jsonl(path):
            if error:
                raise ValueError(f"{relative}:{line_number}: {error}")
            domain, domain_rule = infer_source_domain(path, record)
            if dataset == "A1" and not domain:
                LOGGER.warning("A1 domain unresolved; A1 overall median fallback will be used")
            if dataset != "A1" and not domain:
                raise ValueError(f"Cannot reliably infer external domain: {relative}")
            identity = record.get("id")
            if not isinstance(identity, str) or not identity or identity in ids:
                raise ValueError(f"Missing/duplicate ID in {dataset}, {relative}:{line_number}: {identity!r}")
            ids.add(identity)
            meta = {"id": identity, "dataset": dataset, "domain": domain or "<unknown>",
                    "sub_path": record.get("sub_path", ""), "source_file": relative,
                    "line_number": line_number}
            try:
                scalars, components, missing = scalarize_record(record, config["metrics"])
            except ValueError as exc:
                raise ValueError(f"Structure conflict at {relative}:{line_number}: {exc}") from exc
            metadata.append(meta)
            scalar_buffer.extend(scalars)
            component_buffer.extend(components)
            issues.extend({**meta, **item} for item in missing)
            count += 1
            if count % config["preprocessing"]["progress_every"] == 0:
                LOGGER.info("%s: %s records read", dataset, count)
            del record
        after = sha256_file(path)
        if before != after:
            raise RuntimeError(f"Official source bytes changed: {relative}")
        manifest.append({"dataset": dataset, "source_file": relative, "n": count,
                         "sha256_before": before, "sha256_after": after, "source_unchanged": True})
    if not metadata:
        raise ValueError(f"Empty input dataset {dataset}")
    return (np.frombuffer(scalar_buffer, dtype=np.float64).reshape(-1, 22),
            np.frombuffer(component_buffer, dtype=np.float64).reshape(-1, 4), metadata, issues, manifest)


def write_features(path: Path, z: np.ndarray, metadata: list[dict], counts, names: list[str]) -> None:
    """Write all original occurrences; only the 22 named columns are features."""
    columns = ["id", "dataset", "domain", "sub_path", "source_file", "line_number", "imputed_count", *names]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for meta, row, n in zip(metadata, z, counts):
            writer.writerow({**meta, "imputed_count": int(n), **dict(zip(names, map(float, row)))})
    temporary.replace(path)


def markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
    """Render a compact exact-results table without hardcoding statistics."""
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        values = []
        for col in columns:
            v = row.get(col)
            values.append("" if v is None else f"{v:.10g}" if isinstance(v, float) else str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(model, tables, validations, imputations, clips, overlap, manifest, ceilings, preserved):
    """Describe the actual A1 calibration and external transformations only."""
    report = PROJECT_ROOT / "outputs" / "report" / "phase3_preprocessing_summary.md"
    lines = ["# PHASE 3：A1 标定的统一 22 维质量向量 Z", "",
        "本阶段完成原始标量→缺失填补→异常截尾→方向统一→[0,1] 标准化。未运行 04，未计算 CRITIC 或 Q。", "",
        "## 标定范围与历史结果", "",
        "A1 是七领域主样本。全部参数只用 A1 拟合，A2/A3 仅应用保存的参数；这使抽样集与扩展集共享质量尺度，",
        "并避免扩展集规模与分布反过来改变主样本的评价尺度。", "",
        "原 outputs/profile/ 全部文件和 qurater_transform_parameters.csv 保留且哈希不变，角色是",
        "**pooled descriptive / sensitivity analysis**，不是主模型最终参数。角色另存 historical_artifact_roles.csv。", "",
        "不对任一数据集或跨数据集去重；不建立 pooled deduplicated view。A3 不同 ID 的质量字段相同记录两条都保留，",
        "因为缺少 content 无法认定文本重复。跨附件重叠记录仍存在；非重叠索引仅用于以后比较，不从 Z 删除记录。", "",
        "## 实际输入与规模", ""]
    lines += markdown_table(manifest, ["dataset", "source_file", "n", "source_unchanged"])
    lines += ["", "## 22 个指标的正式转换规则", "",
        "fineweb_edu 取唯一分量；qurater 四维使用 A1 各自 P01/P99 截尾、Min-Max 后平均；",
        "modernbert 四项 argmax/5；fluency_en argmax([not_fluent,fluent])；ad_en argmax([has_ad,no_ad])，1 为无广告。",
        "后三类已在 [0,1]，包含 qurater 在内的七个指标不再次 Min-Max，不再整体截尾。",
        "Softmax 期望只保留为辅助函数，本模型未调用。并列 argmax 取最小索引；非有限向量先作为缺失处理。", "",
        "其余 15 个连续指标（fineweb、DSIR、RPS，包括连续比例统计）使用 A1 P01/P99 Winsorization。",
        "这些 RPS 比例是统计特征，不是分类概率；分类概率/评级不参与连续截尾。",
        "其中 12 个按方向 Min-Max；三个适中指标使用梯形效用。所有指标方向只在 YAML 中定义。", ""]
    rules = []
    for metric in model["metric_definitions"]:
        name = metric["name"]
        final = "A1 Winsor + " + ("trapezoid" if metric["direction"] == "moderate" else "directional Min-Max") if name in model["winsor_thresholds"] else "bounded identity (no second Min-Max)"
        rules.append({"metric": name, "direction": metric["direction"], "scalarization": metric.get("scalarization", "raw_scalar"), "final_transform": final})
    lines += markdown_table(rules, ["metric", "direction", "scalarization", "final_transform"])
    lines += ["", "正向 14、负向 5、适中 3，严格为 22 维。辅助 ID、dataset、domain、来源、imputed_count 不参与向量。", "",
        "## 缺失填补与记录保留", "",
        f"实际填补 {len(imputations)} 个标量，涉及 {len({(r['dataset'], r['source_file'], r['line_number']) for r in imputations})} 条记录。",
        "中位数由 A1 有限观测拟合；优先 A1 同领域同指标，无法取得时回退 A1 全体该指标。",
        "A2/A3 不使用自身中位数。填充值、来源领域、ID、行号逐项记录于 imputation_log.csv。",
        "这些记录只有少量指标缺失，保留样本与其他已知指标，不删除整行；高缺失记录单独留档而不自动删除。", ""]
    lines += markdown_table(imputations, ["dataset", "domain", "id", "metric", "imputed_value", "imputation_method"])
    lines += ["", "## A1 完整拟合参数", "",
        "preprocessing_model_A1.json 保存全部参数、别名、指标顺序、官方类别顺序及配置。",
        "分位数使用 numpy 线性插值；顺序为标量化→用观测拟合中位数→填补→计算连续分位数。",
        "适中型各分位数取自填补后的 A1、连续截尾前。所有文件中的 fitting_dataset/fit_scope 都为 A1。", "",
        "### qurater 四维", ""]
    lines += markdown_table(tables["qurater_transform_parameters_A1.csv"], ["component_name", "P01", "P99", "min_after_winsorization", "max_after_winsorization"])
    lines += ["", "### 中位数（所有指标和 A1 七领域）", ""]
    lines += markdown_table(tables["imputation_parameters_A1.csv"], ["metric", "domain", "median", "valid_count"])
    lines += ["", "### 连续指标截尾", ""]
    lines += markdown_table(tables["winsor_parameters_A1.csv"], ["metric", "P01", "P99", "min_after_winsorization", "max_after_winsorization"])
    lines += ["", "### 正负向 Min-Max 范围", "",
        "正向 z=(x−L)/(U−L)，负向 z=(U−x)/(U−L)，然后 clip 到 [0,1]。",
        f"U=L 时使用配置常数 {model['settings']['constant_minmax_value']}，不制造区分度；实际触发状态见 constant 列。", ""]
    lines += markdown_table(tables["minmax_parameters_A1.csv"], ["metric", "direction", "L", "U", "constant"])
    lines += ["", "## 适中型梯形效用", "",
        "x≤a 或 x≥d 时 z=0；a<x<b 线性上升；b≤x≤c 时 z=1；c<x<d 线性下降。",
        "主模型 a=P01、b=P10、c=P90、d=P99。平台分位点由配置控制，可另行拟合 P05/P95 或 P20/P80 敏感性模型。",
        "当前不计算这些额外模型。零宽斜坡按阶跃处理，外端点 0 优先；a=b=c=d 时全为 0，避免除零。", ""]
    lines += markdown_table(tables["moderate_metric_parameters_A1.csv"], ["metric", "P01", "P05", "P10", "P20", "P50", "P80", "P90", "P95", "P99", "a", "b", "c", "d"])
    lines += ["", "## A2/A3 Clip rate", "",
        "分母为该附件全部保留记录，含按 A1 填补的记录。low/high 指方向统一前的原始指标低于/高于 A1 边界，",
        "对负向指标不代表最终 z 的低/高方向。适中指标报告原值超出外端点的比例，端点等值不计截尾。",
        "qurater 指四个原分量中至少一个越界；低/高事件可在同一条记录发生，不能直接把比例相加。",
        "四维单独统计另存 external_qurater_component_clip_rates.csv。分类指标未经再次截尾，clip rate 为 0。",
        f"large_clip_flag 的诊断阈值为配置项 {model['settings']['large_clip_rate_threshold']:.0%}，是报告提示阈值，不是伪造的拟合参数。", ""]
    lines += markdown_table(clips, ["dataset", "metric", "clip_low_rate", "clip_high_rate", "large_clip_flag", "clip_stage"])
    lines += ["", "## 分类天花板（保留 PHASE 2C 已有结果）", "",
        "以下 ceiling_rate 分母为非缺失标量，不重估分类规则。A2-arxiv 的多个分类指标存在明显顶部集中，",
        "cleanliness/reasoning/professionalism 等最高档占比接近 99%；当前仍按官方 argmax 主方案。",
        "二分类顶部占比表示分类标签比例，不是分类准确率；ad_en=1 表示无广告。", ""]
    lines += markdown_table([r for r in ceilings if r["scope"] == "dataset"], ["dataset", "metric", "valid_count", "missing", "ceiling_rate"])
    lines += ["", "## 输出验证与非重叠索引", ""]
    group_results = []
    for dataset in ["A1", "A2", "A3"]:
        group = [r for r in validations if r["dataset"] == dataset]
        group_results.append({"dataset": dataset, "n": group[0]["n"], "dimensions": len(group),
                              "NaN_count": sum(r["nan_count"] for r in group), "Inf_count": sum(r["inf_count"] for r in group),
                              "all_in_0_1": all(r["in_0_1"] for r in group)})
    lines += markdown_table(group_results, ["dataset", "n", "dimensions", "NaN_count", "Inf_count", "all_in_0_1"])
    lines += ["", "每指标实际 min/max 及取值校验见 feature_validation.csv。", ""]
    lines += markdown_table(overlap, ["dataset", "domain", "n", "overlap_ids", "nonoverlap_ids", "overlap_Z_exactly_equal", "records_removed"])
    lines += ["", "ID 索引只列非重叠 ID 与来源，不复制完整 Z；A1 抽样、扩展全量、扩展非重叠余集可供后续一致性检验。", "",
        f"已有历史结果 {len(preserved)} 个文件经 SHA-256 核验未改变，见 prior_artifacts_unchanged_phase3.csv；原始附件读前读后哈希一致。", "",
        "## 阶段局限与停止点", "",
        "Z 表示按确认方向定义的指标效用，尚无综合权重或质量分数。梯形平台内的差异被有意压平，",
        "A1 范围外值发生饱和；明显扩展截尾与分类天花板需在后续解释、敏感性分析中保留。",
        "领域中位数保留主样本尺度，但可能给缺失评级赋予小数；本次具体填补值以日志为准。",
        "依赖官方质量字段而非文本人工验证，不能把向量一致推断为文本完全一致。", "",
        "本阶段到 Z 为止。未运行 04_build_quality_score.py，未计算 CRITIC、Q 或其他题目。等待人工确认。", ""]
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Fit A1, transform every retained row, audit and stop before scoring."""
    output, inspection = PROJECT_ROOT / "outputs" / "preprocess", PROJECT_ROOT / "outputs" / "inspection"
    output.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs" / "report").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "features.log", mode="w", encoding="utf-8")])
    config_path = PROJECT_ROOT / "configs" / "quality_metrics.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    np.random.seed(config["preprocessing"]["random_state"])
    files = discover_quality_files(PROJECT_ROOT)
    historical = sorted(p for folder in [PROJECT_ROOT / "outputs" / "profile", inspection] for p in folder.rglob("*") if p.is_file())
    pooled_parameter = output / "qurater_transform_parameters.csv"
    if pooled_parameter.is_file():
        historical.append(pooled_parameter)
    before = {p: sha256_file(p) for p in historical}
    baseline = {r["scope"]: int(r["n"]) for r in load_csv(inspection / "duplicate_summary.csv")}
    previous_cross = {r["right_dataset"]: r for r in load_csv(inspection / "cross_dataset_overlap.csv")}
    ceilings = load_csv(inspection / "classification_ceiling_summary.csv")
    previous_missing = load_csv(PROJECT_ROOT / "outputs" / "profile" / "scalar_missing_records.csv")
    expected_missing = {(r["dataset"], r["source_path"], int(r["line_number"]), r["id"], r["metric"]) for r in previous_missing}
    all_imputations, validations, manifest, high_missing, clips, component_clips, overlap = [], [], [], [], [], [], []
    all_issues = set()
    model, tables, a1_reference = None, None, {}
    for dataset, paths in files.items():
        matrix, components, metadata, issues, source_manifest = read_dataset(dataset, paths, config)
        if len(metadata) != baseline[dataset]:
            raise ValueError(f"Record count differs from accepted PHASE 2C: {dataset}")
        all_issues.update((r["dataset"], r["source_file"], r["line_number"], r["id"], r["metric"]) for r in issues)
        if dataset == "A1":
            model, _ = fit_preprocessing_model(matrix, components, metadata, config)
            model["fitting_inputs"] = source_manifest
            model["fitting_config_sha256"] = sha256_file(config_path)
            model["historical_pooled_role"] = config["preprocessing"]["historical_pooled_role"]
            serialized = json.dumps(model, ensure_ascii=False, indent=2, allow_nan=False)
            (output / "preprocessing_model_A1.json").write_text(serialized + "\n", encoding="utf-8")
            # All transforms, including A1, use the model reloaded from disk.
            model = json.loads((output / "preprocessing_model_A1.json").read_text(encoding="utf-8"))
            tables = parameter_tables(model)
            for filename, rows in tables.items():
                write_rows(output / filename, rows)
        model_before = json.dumps(model, sort_keys=True, allow_nan=False)
        z, imputations, missing_counts, high, dataset_clips, q_clips = apply_preprocessing(matrix, components, metadata, model)
        if model_before != json.dumps(model, sort_keys=True, allow_nan=False):
            raise RuntimeError("An external transformation mutated fitted model parameters")
        if int(missing_counts.sum()) != len(issues) or len(imputations) != len(issues):
            raise ValueError("Scalar missing and imputation provenance do not reconcile")
        reasons = {(r["source_file"], r["line_number"], r["metric"]): r["reason"] for r in issues}
        for r in imputations:
            r["missing_reason"] = reasons[(r["source_file"], r["line_number"], r["metric"])]
        all_imputations.extend(imputations)
        high_missing.extend(high)
        manifest.extend(source_manifest)
        for j, name in enumerate(model["metric_order"]):
            validations.append({"dataset": dataset, "metric": name, "n": len(z), "min": float(z[:, j].min()),
                "max": float(z[:, j].max()), "nan_count": int(np.isnan(z[:, j]).sum()),
                "inf_count": int(np.isinf(z[:, j]).sum()), "in_0_1": bool(((z[:, j] >= 0) & (z[:, j] <= 1)).all())})
        if dataset == "A1":
            a1_reference = {m["id"]: (m["domain"], row.copy()) for m, row in zip(metadata, z)
                            if m["domain"] in {r["domain"] for r in previous_cross.values()}}
        else:
            domain = previous_cross[dataset]["domain"]
            reference_ids = {identity for identity, (d, _) in a1_reference.items() if d == domain}
            actual_ids = {m["id"] for m in metadata}
            shared = reference_ids & actual_ids
            if shared != reference_ids or len(shared) != int(previous_cross[dataset]["same_id"]):
                raise ValueError("Cross-dataset overlap differs from accepted PHASE 2C")
            nonoverlap = []
            for m, row in zip(metadata, z):
                if m["id"] in shared:
                    if not np.array_equal(a1_reference[m["id"]][1], row):
                        raise ValueError(f"Same raw quality failed to yield identical Z: {m['id']}")
                else:
                    nonoverlap.append(m)
            prefix = "A2_arxiv" if dataset == "A2" else "A3_github"
            write_rows(output / f"{prefix}_nonoverlap_ids.csv", nonoverlap)
            overlap.append({"dataset": dataset, "domain": domain, "n": len(z), "overlap_ids": len(shared),
                "nonoverlap_ids": len(nonoverlap), "overlap_Z_exactly_equal": True, "records_removed": 0})
            clips.extend(dataset_clips)
            component_clips.extend(q_clips)
        filename = {"A1": "A1_quality_features.csv", "A2": "A2_arxiv_quality_features.csv", "A3": "A3_github_quality_features.csv"}[dataset]
        write_features(output / filename, z, metadata, missing_counts, model["metric_order"])
        LOGGER.info("%s Z complete: %s rows x 22; imputed=%s; all finite, all [0,1]", dataset, len(z), len(imputations))
        del matrix, components, metadata, z
    if all_issues != expected_missing:
        raise ValueError("Raw missing events differ from accepted 02 scalar missing ID list")
    write_rows(output / "imputation_log.csv", all_imputations)
    write_rows(output / "high_missing_samples.csv", high_missing, ["id", "dataset", "domain", "source_file", "line_number", "missing_count", "missing_rate", "missing_metrics", "action"])
    imputation_summary = []
    for dataset in files:
        for name in model["metric_order"]:
            events = [r for r in all_imputations if r["dataset"] == dataset and r["metric"] == name]
            imputation_summary.append({"dataset": dataset, "metric": name, "n": baseline[dataset], "missing_before": len(events),
                "imputed_count": len(events), "imputed_rate": len(events)/baseline[dataset],
                "imputation_method": "A1_domain_median_then_A1_overall_median", "fitting_dataset": "A1"})
    write_rows(output / "imputation_summary.csv", imputation_summary)
    write_rows(output / "external_clip_rates.csv", clips)
    write_rows(output / "external_qurater_component_clip_rates.csv", component_clips)
    write_rows(output / "feature_validation.csv", validations)
    write_rows(output / "nonoverlap_index_summary.csv", overlap)
    write_rows(output / "preprocessing_input_manifest.csv", manifest)
    preserved = []
    for path, original in before.items():
        after = sha256_file(path)
        if after != original:
            raise RuntimeError(f"An accepted historical output changed: {path}")
        preserved.append({"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256_before": original,
                          "sha256_after": after, "unchanged": True})
    write_rows(output / "prior_artifacts_unchanged_phase3.csv", preserved)
    roles = [{"path": p.relative_to(PROJECT_ROOT).as_posix(), "role": config["preprocessing"]["historical_pooled_role"],
              "main_model_parameter": False} for p in historical if p.parent == PROJECT_ROOT / "outputs" / "profile" or p == pooled_parameter]
    write_rows(output / "historical_artifact_roles.csv", roles)
    (output / "preprocessing_config_A1.yaml").write_bytes(config_path.read_bytes())
    write_report(model, tables, validations, all_imputations, clips, overlap, manifest, ceilings, preserved)
    LOGGER.info("Completed: %s records retained; %s imputations; 22 finite features per row", sum(baseline[d] for d in files), len(all_imputations))
    LOGGER.info("STOP: script 04 not run; no CRITIC, no Q; waiting for human confirmation")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOGGER.exception("Feature construction failed; dependent steps must stop")
        sys.exit(1)
