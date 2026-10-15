"""Phase 2: inspect every A1/A2/A3 record; no transforms, weights or Q.

Exact quantiles use compact numeric arrays. Only one JSON record (and at most
20 raw field examples per field) is retained; content is never collected.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
from array import array
from collections import Counter
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quality.io import discover_quality_files, iter_jsonl, sha256_file

LOGGER = logging.getLogger("quality.inspection")
ABSENT = object()


def finite_number(value: object) -> bool:
    """Reject booleans, nonnumeric elements and NaN/Inf without score assumptions."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def numeric_stats(values: array) -> dict:
    """Compute exact observed summary statistics (linear-interpolated quantiles)."""
    if not values:
        return dict.fromkeys(["min", "p25", "median", "mean", "p75", "max"])
    data = np.frombuffer(values, dtype=np.float64)
    quartiles = np.quantile(data, [0.25, 0.5, 0.75], method="linear")
    return {"min": float(data.min()), "p25": float(quartiles[0]),
            "median": float(quartiles[1]), "mean": float(data.mean()),
            "p75": float(quartiles[2]), "max": float(data.max())}


def json_text(value: object) -> str:
    """Serialize structures in CSV cells, retaining raw values in examples."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write Excel-readable UTF-8 CSV with a consistent union of columns."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class FieldAudit:
    """Accumulate raw schema counts and exact list statistics for one field."""

    name: str
    kind: str
    sample_count: int
    sum_tolerance: float
    probability_min_ratio: float
    counts: Counter = field(default_factory=Counter)
    types: Counter = field(default_factory=Counter)
    raw_names: Counter = field(default_factory=Counter)
    lengths: Counter = field(default_factory=Counter)
    valid_lengths: Counter = field(default_factory=Counter)
    positions: dict[int, array] = field(default_factory=dict)
    sums: array = field(default_factory=lambda: array("d"))
    samples: list[dict] = field(default_factory=list)
    invalid_examples: list[dict] = field(default_factory=list)
    scalar_min: float | None = None
    scalar_max: float | None = None

    def observe(self, value: object, raw_name: str | None, provenance: dict) -> None:
        """Classify one raw value; missing/empty/invalid/parse-failed are disjoint."""
        self.counts["total"] += 1
        if value is ABSENT:
            self.counts["absent"] += 1
            return
        self.raw_names[raw_name] += 1
        self.types[type(value).__name__] += 1
        if value is None:
            self.counts["none"] += 1
            return
        if self.kind == "list":
            if value != [] and len(self.samples) < self.sample_count:
                self.samples.append({**provenance, "value": value})
            parsed = value
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    self.counts["string_parse_success"] += 1
                except (ValueError, TypeError) as exc:
                    self.counts["parse_failure"] += 1
                    self.remember_invalid(value, provenance, str(exc))
                    return
            if isinstance(parsed, list):
                self.lengths[len(parsed)] += 1
                if not parsed:
                    self.counts["empty_list"] += 1
                    return
                if all(finite_number(v) for v in parsed):
                    total = sum(parsed)
                    if math.isfinite(total):
                        self.counts["valid"] += 1
                        self.valid_lengths[len(parsed)] += 1
                        self.sums.append(total)
                        for position, number in enumerate(parsed):
                            self.positions.setdefault(position, array("d")).append(number)
                        self.counts["sum_near_one"] += abs(total - 1) < self.sum_tolerance
                        self.counts["within_unit_interval"] += all(0 <= v <= 1 for v in parsed)
                        self.counts["negative_vector"] += any(v < 0 for v in parsed)
                        self.counts["above_one_vector"] += any(v > 1 for v in parsed)
                        return
            self.counts["invalid"] += 1
            self.remember_invalid(value, provenance, "not a nonempty finite numeric list")
        elif self.kind == "scalar":
            if finite_number(value):
                self.counts["valid"] += 1
                self.scalar_min = value if self.scalar_min is None else min(self.scalar_min, value)
                self.scalar_max = value if self.scalar_max is None else max(self.scalar_max, value)
            else:
                self.counts["invalid"] += 1
                self.remember_invalid(value, provenance, "not a finite numeric scalar")
        else:
            self.counts["non_null"] += 1

    def remember_invalid(self, value: object, provenance: dict, reason: str) -> None:
        """Keep a bounded number of field-specific failure examples, without content."""
        if len(self.invalid_examples) < 3:
            self.invalid_examples.append({**provenance, "value": value, "reason": reason})

    def schema_row(self, dataset: str, meta: dict) -> dict:
        """Summarize field presence/types; aliases remain visible and explicit."""
        total = self.counts["total"]
        return {"dataset": dataset, **meta, "field": self.name, "kind": self.kind,
                "observed_raw_names": json_text(self.raw_names),
                "raw_types": json_text(self.types), "present_count": total-self.counts["absent"],
                "absent_count": self.counts["absent"], "none_count": self.counts["none"],
                "empty_list_count": self.counts["empty_list"],
                "invalid_count": self.counts["invalid"],
                "field_parse_failure_count": self.counts["parse_failure"],
                "valid_count": self.counts["valid"], "scalar_min": self.scalar_min,
                "scalar_max": self.scalar_max, "alias_conflict_count": self.counts["alias_conflict"]}

    def list_row(self, dataset: str) -> dict:
        """Infer probability likeness only from length, bounds and observed sums."""
        c = self.counts
        valid = c["valid"]
        dominant = self.lengths.most_common(1)[0][0] if self.lengths else None
        near_ratio = c["sum_near_one"] / valid if valid else None
        unit_ratio = c["within_unit_interval"] / valid if valid else None
        stable = len(self.valid_lengths) == 1
        likely = bool(valid and stable and next(iter(self.valid_lengths)) >= 2
                      and unit_ratio == 1 and near_ratio >= self.probability_min_ratio)
        missing = c["absent"] + c["none"] + c["empty_list"]
        row = {"dataset": dataset, "field": self.name, "record_count": c["total"],
               "absent_count": c["absent"], "none_count": c["none"],
               "empty_list_count": c["empty_list"], "invalid_count": c["invalid"],
               "parse_failure_count": c["parse_failure"],
               "string_parse_success_count": c["string_parse_success"],
               "valid_count": valid, "missing_count": missing,
               "missing_rate": missing / c["total"] if c["total"] else None,
               "unusable_rate": (c["total"]-valid) / c["total"] if c["total"] else None,
               "length_distribution": json_text(self.lengths),
               "valid_length_distribution": json_text(self.valid_lengths),
               "dominant_length": dominant, "stable_valid_length": stable,
               "sum_near_1_count": c["sum_near_one"], "sum_near_1_ratio": near_ratio,
               "unit_interval_vector_ratio": unit_ratio,
               "negative_vector_count": c["negative_vector"],
               "above_one_vector_count": c["above_one_vector"],
               "likely_probability_vector": likely,
               "position_stats": json_text({p: numeric_stats(v) for p,v in self.positions.items()})}
        row.update({f"sum_{k}": v for k,v in numeric_stats(self.sums).items()})
        # Ensure every decoded record has exactly one classification.
        assert valid + missing + c["invalid"] + c["parse_failure"] == c["total"]
        return row


def inspect_dataset(dataset: str, files: list[Path], config: dict, samples_stream):
    """Stream all shards of one dataset, report raw examples and schema statistics."""
    settings = config["inspection"]
    specs = {m["name"]: m for m in config["metrics"]}
    specs.update({name: {"name": name, "kind": "auxiliary"} for name in config["auxiliary_fields"]})
    audits = {name: FieldAudit(name, spec["kind"], settings["sample_count"],
              settings["sum_tolerance"], settings["probability_min_near_one_ratio"])
              for name, spec in specs.items()}
    domains, raw_keys = Counter(), Counter()
    file_rows, failures = [], []
    valid_records = 0
    for path in files:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        LOGGER.info("Reading %s: %s (%s compressed bytes)", dataset, relative, path.stat().st_size)
        before = sha256_file(path)
        lines = decoded = failed = 0
        for line_number, record, error in iter_jsonl(path):
            lines += 1
            if error is not None:
                failed += 1
                failures.append({"dataset": dataset, "source_path": relative,
                                 "line_number": line_number, "error": error})
                LOGGER.error("JSON failure %s:%s: %s", relative, line_number, error)
                continue
            decoded += 1
            valid_records += 1
            raw_keys.update(record.keys())
            domain = record.get("_source_domain")
            domains[str(domain) if domain is not None else "<absent_or_null>"] += 1
            provenance = {"source_path": relative, "line_number": line_number, "id": record.get("id")}
            for name, spec in specs.items():
                names = [key for key in [name, *spec.get("raw_aliases", [])] if key in record]
                raw_name = names[0] if names else None
                if len(names) > 1 and any(record[k] != record[raw_name] for k in names[1:]):
                    audits[name].counts["alias_conflict"] += 1
                audits[name].observe(record[raw_name] if names else ABSENT, raw_name, provenance)
            if valid_records % settings["progress_every"] == 0:
                LOGGER.info("%s: %s decoded records", dataset, valid_records)
        after = sha256_file(path)
        if before != after:
            raise RuntimeError(f"Source hash changed while inspecting: {relative}")
        file_rows.append({"dataset": dataset, "source_path": relative,
                          "compressed_bytes": path.stat().st_size, "physical_lines": lines,
                          "record_count": decoded, "json_parse_failure_count": failed,
                          "sha256_before": before, "sha256_after": after, "source_unchanged": True})
    meta = {"file_count": len(files), "record_count": valid_records,
            "physical_lines": sum(r["physical_lines"] for r in file_rows),
            "json_parse_failure_count": len(failures),
            "source_files": json_text([r["source_path"] for r in file_rows]),
            "observed_domain_counts": json_text(domains), "raw_key_counts": json_text(raw_keys)}
    schema_rows = [a.schema_row(dataset, meta) for a in audits.values()]
    list_rows, position_rows = [], []
    for name, audit in audits.items():
        if audit.kind != "list":
            continue
        row = audit.list_row(dataset)
        list_rows.append(row)
        header = f"\n[{dataset}] {name}: first {len(audit.samples)} non-null/nonempty raw values\n"
        samples_stream.write(header)
        print(header, end="")
        for example in audit.samples:
            text = json_text(example)
            samples_stream.write(text + "\n")
            print(text)
        samples_stream.write("diagnostic: " + json_text(row) + "\n")
        if audit.invalid_examples:
            samples_stream.write("invalid_examples: " + json_text(audit.invalid_examples) + "\n")
        for position, values in audit.positions.items():
            stats = numeric_stats(values)
            position_rows.append({"dataset": dataset, "field": name, "position_zero_based": position,
                                  "count": len(values), "min": stats["min"], "max": stats["max"],
                                  "mean": stats["mean"], "median": stats["median"]})
        LOGGER.info("%s %s: length=%s valid=%s missing_rate=%.6f sum_near_1=%s probability=%s",
                    dataset, name, row["dominant_length"], row["valid_count"], row["missing_rate"],
                    row["sum_near_1_ratio"], row["likely_probability_vector"])
    LOGGER.info("Completed %s: %s records; observed domains=%s", dataset, valid_records, dict(domains))
    return list_rows, schema_rows, position_rows, file_rows, failures


def write_inspection_report(output: Path, combined: list[list[dict]]) -> None:
    """Generate a reviewable Phase 2 report from measured outputs only."""
    lists, schema, positions, files, failures = combined
    lines = ["# PHASE 1 + PHASE 2：真实数据结构检查", "",
             "本报告只含结构检查；未标量化、填补、截尾、归一化、计算 CRITIC 或 Q。", "",
             "## 实际文件和记录数", "",
             "| 数据集 | 实际路径 | 记录数 | JSON 解析失败 | 原文件哈希一致 |",
             "|---|---|---:|---:|---|"]
    for row in files:
        lines.append(f"| {row['dataset']} | `{row['source_path']}` | {row['record_count']} | "
                     f"{row['json_parse_failure_count']} | {row['source_unchanged']} |")
    lines += ["", f"共读取 {sum(int(r['record_count']) for r in files):,} 条 JSON 对象。",
              "目录与用户提供的文件规则一致。A2/A3 的全量指磁盘已提供的全部匹配文件。", "",
              "## 八个列表字段", "",
              "| 数据集 | 字段 | 长度分布 | 有效数 | 缺失率 | 非法数 | 解析失败 | sum≈1 比例 | 概率诊断 |",
              "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for row in lists:
        ratio = row["sum_near_1_ratio"]
        ratio_text = f"{float(ratio):.6%}" if ratio is not None and ratio != "" else "无有效值"
        lines.append(f"| {row['dataset']} | {row['field']} | {row['length_distribution']} | "
                     f"{row['valid_count']} | {float(row['missing_rate']):.6%} | {row['invalid_count']} | "
                     f"{row['parse_failure_count']} | {ratio_text} | {row['likely_probability_vector']} |")
    lines += ["", "缺失=字段不存在+None+空列表；非法值另计。sum≈1 的分母为有效有限数值列表数，",
              "条件为 abs(sum−1)<0.001。NaN 按非法数值记录；Python JSON 解码器可读取 NaN，",
              "所以这类值不计作 JSON 语法解析失败。", "",
              "完整前 20 个真实非空值及非法样例见 `list_field_samples.txt`；",
              "每个位置的 min/max/mean/median 见 `list_field_position_summary.csv`；",
              "向量和的 min/P25/median/mean/P75/max 见 `list_field_summary.csv`。", "",
              "## 22 指标与辅助字段核验", "",
              "| 数据集 | 规范字段 | 缺失字段数 | None | 非法数 | 标量最小值 | 标量最大值 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in schema:
        lines.append(f"| {row['dataset']} | {row['field']} | {row['absent_count']} | {row['none_count']} | "
                     f"{row['invalid_count']} | {row['scalar_min'] if row['scalar_min'] is not None else ''} | "
                     f"{row['scalar_max'] if row['scalar_max'] is not None else ''} |")
    lines += ["", "真实终止标点字段拼作 `rps_lines_ending_with_terminal_punctution_mark`，",
              "通过配置中的显式别名对应官方规范名；不修改官方文件，也不增加指标维数。", "",
              "## 待确认的含义与处理规则", "",
              "- fineweb_edu：若长度检查继续确认始终为 1，结构上支持取唯一分量；原值不是 [0,1] 概率。",
              "- qurater：长度 4 不能证明四项同尺度、同方向或独立。需确认模型输出和四个位置的定义后，才能决定均值是否适用。",
              "- modernbert 四项：原向量含负数且和不接近 1，不能直接代入概率期望。疑似 logits 是统计推测，需要模型定义证实；六个位置的等级顺序也需确认。",
              "- fluency_en/ad_en：不能直接取某个分量作概率。需确认是否为 logits，以及两个位置对应的实际类别。未经确认不执行 softmax 或反向。",
              "- RPS 原值中的若干 fraction 指标达到 100，重复 n-gram 字段还可超过 100。字段名称不能证明 [0,1] 或统一百分比口径；当前仅记录真实范围，不剪裁、不换算。", "",
              "下一步先人工确认以上结构和语义，再进入 PHASE 4。当前没有生成任何质量权重或评分。"]
    (output / "inspection_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the structure-only stage and stop before any feature/score processing."""
    output = PROJECT_ROOT / "outputs" / "inspection"
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(output / "inspection.log", mode="w", encoding="utf-8")])
    with (PROJECT_ROOT / "configs" / "quality_metrics.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    metrics = config["metrics"]
    if len(metrics) != 22 or len({m["name"] for m in metrics}) != 22:
        raise ValueError("Configuration must contain exactly 22 unique quality metrics")
    if sum(m["kind"] == "list" for m in metrics) != 8:
        raise ValueError("Configuration must contain exactly 8 list metrics")
    datasets = discover_quality_files(PROJECT_ROOT)
    combined = [[] for _ in range(5)]
    with (output / "list_field_samples.txt").open("w", encoding="utf-8") as stream:
        stream.write("Full-data structure inspection; no scalarization, normalization, CRITIC or Q.\n")
        stream.write("Sample order: sorted filenames then physical line order. Positions are zero-based.\n")
        stream.write("Missing=absent+None+empty list; invalid and parse failure counted separately.\n")
        stream.write("Sum statistics and near-one ratio use valid finite numeric lists only.\n")
        stream.write("Likely probability: stable length >=2, all vectors in [0,1], >=99% sums within 0.001.\n")
        for dataset, files in datasets.items():
            result = inspect_dataset(dataset, files, config, stream)
            for target, rows in zip(combined, result):
                target.extend(rows)
    for filename, rows in zip(["list_field_summary.csv", "dataset_schema_summary.csv",
                               "list_field_position_summary.csv", "input_file_manifest.csv",
                               "json_parse_failures.csv"], combined):
        if rows or filename != "json_parse_failures.csv":
            write_csv(output / filename, rows)
        else:
            (output / filename).write_text("dataset,source_path,line_number,error\n", encoding="utf-8-sig")
    write_inspection_report(output, combined)
    conflicts = sum(r["alias_conflict_count"] for r in combined[1])
    LOGGER.info("Inspection outputs written to %s", output)
    LOGGER.info("STOP: vector meaning/class order must be confirmed before any later stage.")
    return 1 if combined[4] or conflicts else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOGGER.exception("Inspection failed; downstream stages must not proceed.")
        sys.exit(1)
