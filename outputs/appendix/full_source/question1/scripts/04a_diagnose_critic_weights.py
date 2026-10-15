"""PHASE 4A: fit and diagnose A1 CRITIC weights, then stop before any Q."""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quality.critic import critic_weights, weight_entropy
from quality.io import sha256_file
from quality.vector_diagnostics import write_rows

LOGGER = logging.getLogger("quality.critic")


def read_a1_features(path: Path, names: list[str], expected_n: int) -> np.ndarray:
    """Read only configured A1 indicator columns; exclude every auxiliary field."""
    values, identities = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or any(name not in reader.fieldnames for name in names):
            raise ValueError("A1 feature columns missing")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Duplicate A1 column names")
        for line_number, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Truncated/invalid A1 CSV row {line_number}")
            if row["dataset"] != "A1" or not row["id"] or row["id"] in identities:
                raise ValueError(f"Non-A1 or duplicated/missing ID at row {line_number}")
            identities.add(row["id"])
            values.append([float(row[name]) for name in names])
    z = np.asarray(values, dtype=np.float64)
    if z.shape != (expected_n, 22) or not np.isfinite(z).all() or (z < 0).any() or (z > 1).any():
        raise ValueError("A1 must have expected n x 22 finite normalized features")
    return z


def metric_type(metric: dict) -> str:
    """Derive diagnostic types from saved PHASE 3 definitions, not new directions."""
    method = metric.get("scalarization")
    if method == "argmax_scaled":
        return "modernbert_ordinal"
    if method == "argmax_index":
        return "binary"
    if method == "component_winsor_minmax_mean":
        return "qurater"
    if metric["direction"] == "moderate":
        return "moderate"
    return f"continuous_{metric['direction']}"


def write_matrix(path: Path, matrix: np.ndarray, names: list[str]) -> None:
    """Write a labeled 22x22 matrix, keeping official metric order."""
    write_rows(path, [{"metric": name, **dict(zip(names, map(float, matrix[j])))} for j, name in enumerate(names)])


def table(rows: list[dict], columns: list[str]) -> list[str]:
    """Format observed values as a readable Markdown table."""
    result = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        cells = []
        for key in columns:
            v = row.get(key)
            cells.append(f"{v:.10g}" if isinstance(v, float) else "" if v is None else str(v))
        result.append("| " + " | ".join(cells) + " |")
    return result


def write_diagnostics(path, weights, negative_pairs, moderate, discrete, sensitivity, summary, settings):
    """Explain flags and concentration using actual A1 weights; no corrections."""
    ordered = sorted(weights, key=lambda r: r["weight_rank"])
    lines = ["# PHASE 4A：A1 CRITIC 权重诊断", "",
        "本阶段只拟合权重并诊断相关性，不计算任何样本 Q、领域 Q 或 A1/A2/A3 的 Q 对照。", "",
        "## 输入与计算口径", "",
        f"唯一拟合输入：outputs/preprocess/A1_quality_features.csv，实际 {summary['n']:,}×22 个 Z 值。",
        "A2/A3 未参与标准差、相关矩阵或权重拟合。指标顺序和类型取自已确认的 preprocessing_model_A1.json。",
        "PHASE 3 配置、模型、参数与输出文件读前读后 SHA-256 一致，见 phase3_artifacts_unchanged_4A.csv。", "",
        "主方案：σ_j=std(Z_j,ddof=1)，r_jk 为 Pearson，R_j=Σ_k(1−r_jk)，C_j=σ_jR_j，w_j=C_j/Σ_jC_j。",
        "保留负相关及其信息量，未取相关系数绝对值、未剔除指标、未人工调整权重。",
        "若常数列使相关系数无定义，或总信息量为零，脚本停止而不填造相关系数/权重。本次没有这类退化。", "",
        f"权重和={summary['weight_sum']:.17g}；所有权重非负={summary['all_weights_nonnegative']}。", "",
        "## 22 个 Pearson 主权重", ""]
    lines += table(ordered, ["metric", "metric_type", "mean_z", "std_z", "min_z", "max_z", "unique_value_count", "conflict_information", "information_content", "critic_weight", "weight_rank"])
    lines += ["", "## 权重集中度与标记", "",
        f"最大：{summary['max_weight_metric']}，{summary['max_weight']:.10g}；最小：{summary['min_weight_metric']}，{summary['min_weight']:.10g}。",
        f"前 5 累计={summary['top5_weight_sum']:.8%}；前 10 累计={summary['top10_weight_sum']:.8%}。",
        f"H_w=−Σw_jln(w_j)={summary['weight_entropy']:.10g}；H_norm=H_w/ln(22)={summary['normalized_weight_entropy']:.10g}。",
        "熵越接近 1 越均匀；熵与最高权重、累计权重一起解读，不以熵单独证明权重合理。", "",
        "异常标记只提示复核，不改权重。配置独立保存于 configs/critic_diagnostics.yaml，不改 PHASE 3 参数。",
        f"weight>{settings['high_weight_threshold']}；weight>{settings['equal_weight_multiplier']}×(1/22)={settings['equal_weight_multiplier']/22:.10g}；",
        f"weight<{settings['low_weight_threshold']}；std<{settings['low_std_threshold']}；unique≤{settings['low_unique_value_count_threshold']}。",
        "后两项是明示的诊断阈值，不是从附件伪造的统计结果；少取值对 ordinal/binary 是预期结构，不等于指标无效。", ""]
    flagged = [r for r in weights if any(r[k] for k in ["flag_weight_gt_0_10", "flag_weight_gt_2_equal", "flag_weight_lt_0_01", "flag_low_std", "flag_low_unique_values"])]
    lines += table(flagged, ["metric", "critic_weight", "std_z", "unique_value_count", "flag_weight_gt_0_10", "flag_weight_gt_2_equal", "flag_weight_lt_0_01", "flag_low_std", "flag_low_unique_values", "low_unique_expected_for_type"])
    lines += ["", f"标记计数：{json.dumps(summary['flag_counts'], ensure_ascii=False)}。", "",
        "主模型是否有明显集中，应结合上述数值及 >0.10、>2×等权的标记；不能把高权重解释为指标更真实或更重要。", "",
        "## 前五大与后五小", ""]
    if summary["flag_counts"]["flag_weight_gt_0_10"] == 0 and summary["flag_counts"]["flag_weight_gt_2_equal"] == 0:
        lines[-2:-2] = ["未触发两类高权重标记；结合归一化熵和累计权重，本次未见单一指标明显主导的权重集中。", ""]
    else:
        lines[-2:-2] = ["本次触发高权重标记，存在需要人工复核的集中情况；本轮仍保留主公式原权重。", ""]
    lines += table(ordered[:5] + ordered[-5:], ["metric", "critic_weight", "weight_rank"])
    lines += ["", "## 负相关指标对", "",
        f"不重复计数、不计对角线：共 {summary['negative_pair_count']} 对 Pearson r<0；",
        f"r<−0.2 有 {summary['pearson_lt_minus_0_2_count']} 对，r<−0.4 有 {summary['pearson_lt_minus_0_4_count']} 对。",
        "全部越大越好并不保证统计正相关：领域组成、指标取舍和梯形效用会影响联合分布。",
        "负相关在标准 CRITIC 中增加冲突信息，本轮如实保留；它不自动证明方向错误，也不自动证明质量语义正确。", "",
        "### Pearson 最负的前 20 对", ""]
    lines += table(negative_pairs[:20], ["metric_1", "metric_2", "pearson_r", "spearman_r"])
    lines += ["", "## 三个适中型指标：保持 P10–P90 平台", ""]
    lines += table(moderate, ["metric", "mean_z", "std_z", "z_eq_1_rate", "z_eq_0_rate", "critic_weight"])
    lines += ["", "平台会把内部原值差异压为 1，但是否导致低方差须结合实际 std 与权重；当前不改变效用函数。", "",
        "## 六个离散分类指标（A1）", ""]
    if all(row["std_z"] >= settings["low_std_threshold"] for row in moderate):
        lines[-2:-2] = ["本次三个适中型指标均未触发极低标准差标记；平台集中存在，但未发生方差退化。", ""]
    lines += table(discrete, ["metric", "unique_values", "std_z", "ceiling_rate", "critic_weight"])
    lines += ["", "所有值来自已填补的 A1 Z，ceiling_rate 分母为 A1 全部记录。PHASE 2C 的 ceiling_rate 分母为原始非缺失数，",
        "两者可能因 18 次 A1 填补略有不同；本报告不覆盖或改写 PHASE 2C 结果。",
        "σ 越小在给定 R 下会减少 CRITIC 信息量，但最终权重还受相关结构影响，不能单由天花板率推出权重。",
        "A2 的强天花板不参与本次 A1 拟合，也不会反过来压低 A1 权重。", "",
        "## Spearman 敏感性（主模型仍为 Pearson）", "",
        "Spearman 按每列平均秩处理并列值；σ 仍使用原始 Z 的样本标准差，不使用秩标准差。", "",
        f"两组权重 Pearson correlation={summary['weight_vector_pearson']:.10g}；",
        f"Spearman rank correlation={summary['weight_vector_spearman']:.10g}。",
        f"Pearson Top-5：{', '.join(summary['pearson_top5'])}。",
        f"Spearman Top-5：{', '.join(summary['spearman_top5'])}。",
        f"Top-5 集合一致={summary['top5_same_set']}，顺序一致={summary['top5_same_order']}，交集数={summary['top5_intersection_count']}。", ""]
    lines += table(sensitivity, ["metric", "pearson_critic_weight", "critic_weight", "pearson_weight_rank", "weight_rank", "weight_difference", "rank_change"])
    lines += ["", "等权表仅保存 1/22，不用于计算 Q。", "",
        "## 停止点", "",
        "权重与诊断表已生成。本轮未运行完整 04_build_quality_score.py，未计算 Q_i、Q_domain 或评分对照。",
        "等待人工确认权重后才进入 PHASE 4B。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Fit A1 weights and sensitivity diagnostics, preserve all PHASE 3 files."""
    output = PROJECT_ROOT / "outputs" / "quality"
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "critic_A1.log", mode="w", encoding="utf-8")])
    settings_path = PROJECT_ROOT / "configs" / "critic_diagnostics.yaml"
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if settings["fitting_dataset"] != "A1" or settings["std_ddof"] != 1 or settings["main_correlation"] != "pearson" or settings["spearman_tie_method"] != "average":
        raise ValueError("Only approved A1/Pearson/ddof=1/average-tie configuration allowed")
    preprocess = PROJECT_ROOT / "outputs" / "preprocess"
    protected = sorted(p for p in preprocess.rglob("*") if p.is_file())
    protected += [PROJECT_ROOT / "configs" / "quality_metrics.yaml", PROJECT_ROOT / "outputs" / "report" / "phase3_preprocessing_summary.md"]
    before = {p: sha256_file(p) for p in protected}
    model = json.loads((preprocess / "preprocessing_model_A1.json").read_text(encoding="utf-8"))
    names, metrics = model["metric_order"], model["metric_definitions"]
    if len(names) != len(set(names)) or len(names) != 22 or [m["name"] for m in metrics] != names or model["fitting_dataset"] != "A1":
        raise ValueError("Invalid approved 22-indicator A1 model")
    input_path = preprocess / "A1_quality_features.csv"
    z = read_a1_features(input_path, names, model["fit_record_count"])
    LOGGER.info("A1 loaded: %s rows x 22 finite normalized indicators", len(z))
    main_fit = critic_weights(z, "pearson")
    sensitivity_fit = critic_weights(z, "spearman")
    weights, sensitivity, moderate, discrete, negative = [], [], [], [], []
    mean, minimum, maximum = z.mean(axis=0), z.min(axis=0), z.max(axis=0)
    for j, metric in enumerate(metrics):
        unique = np.unique(z[:, j])
        kind, w = metric_type(metric), float(main_fit["weights"][j])
        row = {"metric": names[j], "metric_type": kind, "mean_z": float(mean[j]),
            "std_z": float(main_fit["std"][j]), "min_z": float(minimum[j]), "max_z": float(maximum[j]),
            "unique_value_count": len(unique), "conflict_information": float(main_fit["conflict_information"][j]),
            "information_content": float(main_fit["information_content"][j]), "critic_weight": w,
            "weight_rank": int(main_fit["weight_rank"][j]),
            "flag_weight_gt_0_10": w > settings["high_weight_threshold"],
            "flag_weight_gt_2_equal": w > settings["equal_weight_multiplier"]/22,
            "flag_weight_lt_0_01": w < settings["low_weight_threshold"],
            "flag_low_std": bool(main_fit["std"][j] < settings["low_std_threshold"]),
            "flag_low_unique_values": len(unique) <= settings["low_unique_value_count_threshold"],
            "low_unique_expected_for_type": kind in {"modernbert_ordinal", "binary"}}
        weights.append(row)
        sensitivity.append({"metric": names[j], "metric_type": kind, "std_z": row["std_z"],
            "conflict_information": float(sensitivity_fit["conflict_information"][j]),
            "information_content": float(sensitivity_fit["information_content"][j]),
            "critic_weight": float(sensitivity_fit["weights"][j]), "weight_rank": int(sensitivity_fit["weight_rank"][j]),
            "pearson_critic_weight": w, "pearson_weight_rank": row["weight_rank"],
            "weight_difference": float(sensitivity_fit["weights"][j] - w),
            "rank_change": int(sensitivity_fit["weight_rank"][j] - row["weight_rank"])})
        if kind == "moderate":
            moderate.append({k: row[k] for k in ["metric", "mean_z", "std_z", "critic_weight"]})
            moderate[-1].update({"z_eq_1_rate": float((z[:, j] == 1).mean()), "z_eq_0_rate": float((z[:, j] == 0).mean())})
        elif kind in {"modernbert_ordinal", "binary"}:
            discrete.append({"metric": names[j], "unique_values": json.dumps(unique.tolist()), "unique_value_count": len(unique),
                "std_z": row["std_z"], "ceiling_rate": float((z[:, j] == 1).mean()), "critic_weight": w})
    for j in range(22):
        for k in range(j+1, 22):
            if main_fit["correlation"][j, k] < 0:
                negative.append({"metric_1": names[j], "metric_2": names[k],
                    "pearson_r": float(main_fit["correlation"][j, k]), "spearman_r": float(sensitivity_fit["correlation"][j, k])})
    negative.sort(key=lambda r: r["pearson_r"])
    order = np.argsort(-main_fit["weights"], kind="stable")
    sorder = np.argsort(-sensitivity_fit["weights"], kind="stable")
    top5, stop5 = [names[j] for j in order[:5]], [names[j] for j in sorder[:5]]
    entropy, normalized_entropy = weight_entropy(main_fit["weights"])
    summary = {"fitting_dataset": "A1", "n": len(z), "dimensions": 22, "std_ddof": 1,
        "weight_sum": float(main_fit["weights"].sum()), "all_weights_nonnegative": bool((main_fit["weights"] >= 0).all()),
        "max_weight_metric": names[order[0]], "max_weight": float(main_fit["weights"][order[0]]),
        "min_weight_metric": names[order[-1]], "min_weight": float(main_fit["weights"][order[-1]]),
        "top5_weight_sum": float(main_fit["weights"][order[:5]].sum()),
        "top10_weight_sum": float(main_fit["weights"][order[:10]].sum()),
        "weight_entropy": entropy, "normalized_weight_entropy": normalized_entropy,
        "negative_pair_count": len(negative), "pearson_lt_minus_0_2_count": sum(r["pearson_r"] < -.2 for r in negative),
        "pearson_lt_minus_0_4_count": sum(r["pearson_r"] < -.4 for r in negative),
        "weight_vector_pearson": float(pearsonr(main_fit["weights"], sensitivity_fit["weights"]).statistic),
        "weight_vector_spearman": float(spearmanr(main_fit["weights"], sensitivity_fit["weights"]).statistic),
        "pearson_top5": top5, "spearman_top5": stop5, "top5_same_set": set(top5) == set(stop5),
        "top5_same_order": top5 == stop5, "top5_intersection_count": len(set(top5) & set(stop5)),
        "flag_counts": {key: sum(r[key] for r in weights) for key in ["flag_weight_gt_0_10", "flag_weight_gt_2_equal", "flag_weight_lt_0_01", "flag_low_std", "flag_low_unique_values"]}}
    if abs(summary["weight_sum"] - 1) > settings["weight_sum_tolerance"]:
        raise ValueError("Weights do not sum to one")
    write_rows(output / "critic_weights_A1.csv", weights)
    write_matrix(output / "critic_pearson_correlation_A1.csv", main_fit["correlation"], names)
    write_matrix(output / "critic_spearman_correlation_A1.csv", sensitivity_fit["correlation"], names)
    write_rows(output / "negative_correlation_pairs_A1.csv", negative, ["metric_1", "metric_2", "pearson_r", "spearman_r"])
    write_rows(output / "critic_weights_spearman_sensitivity.csv", sensitivity)
    write_rows(output / "equal_weights.csv", [{"metric": name, "equal_weight": 1/22} for name in names])
    write_rows(output / "moderate_indicator_diagnostics_A1.csv", moderate)
    write_rows(output / "discrete_indicator_diagnostics_A1.csv", discrete)
    (output / "critic_diagnostics_summary_A1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    preserved = []
    for path, original in before.items():
        current = sha256_file(path)
        if original != current:
            raise RuntimeError(f"Protected PHASE 3 artifact changed: {path}")
        preserved.append({"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256_before": original, "sha256_after": current, "unchanged": True})
    write_rows(output / "phase3_artifacts_unchanged_4A.csv", preserved)
    write_rows(output / "critic_inputs_manifest.csv", [{"fitting_dataset": "A1", "source": input_path.relative_to(PROJECT_ROOT).as_posix(),
        "n": len(z), "dimensions": 22, "sha256_before": before[input_path], "sha256_after": sha256_file(input_path),
        "model_source": "outputs/preprocess/preprocessing_model_A1.json", "std_ddof": 1, "main_correlation": "pearson"}])
    (output / "critic_diagnostics_config_snapshot.yaml").write_bytes(settings_path.read_bytes())
    write_diagnostics(output / "critic_weight_diagnostics.md", weights, negative, moderate, discrete, sensitivity, summary, settings)
    LOGGER.info("Completed: 22 A1 Pearson weights, sum=%.17g, H_norm=%.9f", summary["weight_sum"], normalized_entropy)
    LOGGER.info("Negative pairs=%s, r<-0.2=%s, r<-0.4=%s", len(negative), summary["pearson_lt_minus_0_2_count"], summary["pearson_lt_minus_0_4_count"])
    LOGGER.info("STOP: weights and diagnostics only; no sample Q, no domain Q; no complete script 04 run")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOGGER.exception("CRITIC fit/diagnosis failed; dependent steps must stop")
        sys.exit(1)
