"""PHASE 2C: duplicates, ID conflicts, scalar-missing records and ceilings.

This is diagnosis only: no row removal/selection, imputation, averaging,
feature normalization, CRITIC or Q. SQLite stores numeric blobs and hashes,
never content, in a bounded temporary directory under outputs/inspection.
"""

from __future__ import annotations

import csv
import logging
import sqlite3
import sys
import uuid
from collections import Counter
from contextlib import contextmanager
from itertools import combinations, product
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quality.io import discover_quality_files, infer_source_domain, iter_jsonl, sha256_file
from quality.record_diagnostics import compare_record_pair, raw_record_diagnostics
from quality.vector_diagnostics import json_cell, write_rows

LOGGER = logging.getLogger("quality.cleaning_diagnosis")
COMPARE_BASIS = "raw_22_quality_fields_and_comparable_content_sub_path_resolved_domain"


@contextmanager
def diagnosis_workspace(output: Path):
    """Use inherited Windows workspace ACLs and remove only our database/empty dir.

    Windows Python's mkdtemp(mode=0700) denies the restricted execution token
    access even inside an allowed workspace. A normal pathlib mkdir inherits
    the explicitly writable parent permissions. No recursive deletion occurs.
    """
    directory = output / ("phase2c_" + uuid.uuid4().hex)
    if directory.resolve().parent != output.resolve():
        raise ValueError("Temporary index escaped the inspection directory")
    directory.mkdir()
    try:
        yield directory
    finally:
        database = directory / "records.sqlite"
        if database.exists():
            database.unlink()
        directory.rmdir()


def remove_empty_failed_temp(name: str) -> None:
    """Remove only a named, empty legacy mkdtemp directory from this script."""
    import re
    if not re.fullmatch(r"phase2c_[a-z0-9_]{8}",name):
        raise ValueError("Only a literal legacy phase2c temporary basename is accepted")
    parent = PROJECT_ROOT/"outputs"/"inspection"
    directory = parent/name
    if directory.is_symlink() or directory.resolve().parent!=parent.resolve():
        raise ValueError("Unsafe empty temporary directory target")
    directory.rmdir()  # Never recursive; fails rather than touching any contained file.


def initialize_database(connection: sqlite3.Connection) -> None:
    """Create a disposable index with bounded cache and no full text."""
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("""CREATE TABLE records (
        row_no INTEGER PRIMARY KEY, dataset TEXT NOT NULL, source_file TEXT NOT NULL,
        line_number INTEGER NOT NULL, id TEXT, domain TEXT NOT NULL, domain_method TEXT,
        sub_path_json TEXT, sub_path_present INTEGER NOT NULL, content_present INTEGER NOT NULL,
        content_hash BLOB, content_length INTEGER, quality_blob BLOB NOT NULL,
        quality_hash BLOB NOT NULL, exact_hash BLOB NOT NULL,
        missing_metrics_json TEXT NOT NULL, missing_count INTEGER NOT NULL)""")


def index_records(connection: sqlite3.Connection, config: dict):
    """Stream all rows, store raw fingerprints, and count approved argmax levels."""
    metrics = config["metrics"]
    if len(metrics)!=22 or config["scalarization"]["status"]!="USER_CONFIRMED":
        raise ValueError("Require the approved 22-field scalarization configuration")
    categorical = [m for m in metrics if m.get("scalarization") in ("argmax_scaled","argmax_index")]
    if len(categorical)!=6:
        raise ValueError("Exactly six approved classification indicators required")
    group_n, grade_counts = Counter(), Counter()
    manifests, batch, row_no = [], [], 0
    statement = "INSERT INTO records VALUES ("+",".join("?" for _ in range(17))+")"
    for dataset,files in discover_quality_files(PROJECT_ROOT).items():
        for path in files:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            before = sha256_file(path)
            count, domain_counts = 0, Counter()
            LOGGER.info("Indexing %s: %s",dataset,relative)
            for line_number,record,error in iter_jsonl(path):
                if error:
                    raise ValueError(f"{relative}:{line_number}: {error}")
                domain,domain_method = infer_source_domain(path,record)
                domain = domain if domain is not None else "<unknown>"
                identifier = record.get("id")
                if identifier is not None and not isinstance(identifier,str):
                    raise ValueError(f"{relative}:{line_number}: non-string ID {identifier!r}")
                identifier = identifier if identifier else None
                try:
                    audit = raw_record_diagnostics(record,metrics)
                except ValueError as exc:
                    raise ValueError(f"{relative}:{line_number}, id={identifier}: {exc}") from exc
                row_no += 1
                batch.append((row_no,dataset,relative,line_number,identifier,domain,domain_method,
                              audit["sub_path_json"],audit["sub_path_present"],audit["content_present"],
                              audit["content_hash"],audit["content_length"],audit["quality_blob"],
                              audit["quality_hash"],audit["exact_hash"],json_cell(audit["missing_metrics"]),
                              len(audit["missing_metrics"])))
                groups = [("overall","ALL","ALL"),("dataset",dataset,"ALL"),
                          ("domain","ALL",domain),("dataset_domain",dataset,domain)]
                for group in groups:
                    group_n[group] += 1
                    for name,grade in audit["grades"].items():
                        grade_counts[(*group,name,grade)] += 1
                if len(batch)>=2000:
                    connection.executemany(statement,batch)
                    batch.clear()
                count += 1
                domain_counts[domain] += 1
                if count%10000==0:
                    LOGGER.info("%s: %s rows indexed",dataset,count)
            if batch:
                connection.executemany(statement,batch)
                batch.clear()
            after = sha256_file(path)
            if before!=after:
                raise RuntimeError(f"Original input changed: {relative}")
            manifests.append({"dataset":dataset,"source_file":relative,"record_count":count,
                              "domain_counts":json_cell(domain_counts),"sha256_before":before,
                              "sha256_after":after,"source_unchanged":True})
    connection.commit()
    for name,columns in [("idx_id","id"),("idx_dataset_id","dataset,id"),
                         ("idx_exact","exact_hash"),("idx_dataset_exact","dataset,exact_hash"),
                         ("idx_quality","quality_hash")]:
        connection.execute(f"CREATE INDEX {name} ON records ({columns})")
    LOGGER.info("Index complete: %s records; analyzing all same-ID pairs",row_no)
    return manifests,group_n,grade_counts,categorical


def scope_clause(dataset: str) -> tuple[str,tuple]:
    """Restrict to one dataset or all records, using SQL parameters."""
    return ("",()) if dataset=="ALL" else (" WHERE dataset=?",(dataset,))


def pair_export(left: dict,right: dict,result: dict) -> dict:
    """Export a conflict pair with both provenances and original field differences."""
    return {"dataset":json_cell([left["dataset"],right["dataset"]]),
            "source_file":json_cell([left["source_file"],right["source_file"]]),
            "id":left["id"],"dataset_left":left["dataset"],"dataset_right":right["dataset"],
            "source_file_left":left["source_file"],"source_file_right":right["source_file"],
            "line_number_left":left["line_number"],"line_number_right":right["line_number"],
            "domain_left":left["domain"],"domain_right":right["domain"],
            "sub_path":json_cell([left["sub_path_json"],right["sub_path_json"]]),
            "sub_path_left":left["sub_path_json"],"sub_path_right":right["sub_path_json"],
            "content_present_left":bool(left["content_present"]),"content_present_right":bool(right["content_present"]),
            "content_same":result["content_same"],"sub_path_same":result["sub_path_same"],
            "domain_same":result["domain_same"],"classification":result["classification"],
            "quality_classification":result["quality_classification"],"comparison_basis":COMPARE_BASIS,
            "different_quality_fields":json_cell(result["different_quality_fields"]),
            "different_quality_field_count":result["different_quality_field_count"],
            "max_absolute_difference":result["max_absolute_difference"],
            "nonfinite_component_mismatch_count":result["nonfinite_component_mismatch_count"],
            "exact_record_same":result["exact_record_same"]}


def duplicate_diagnosis(connection,metrics):
    """Summarize all groups and pairs; never select a representative record."""
    summaries,exact_rows,conflicts = [],[],[]
    for dataset in ("A1","A2","A3","ALL"):
        where,params = scope_clause(dataset)
        n = connection.execute("SELECT COUNT(*) FROM records"+where,params).fetchone()[0]
        nonmissing = connection.execute("SELECT COUNT(id) FROM records"+where,params).fetchone()[0]
        unique = connection.execute("SELECT COUNT(DISTINCT id) FROM records"+where,params).fetchone()[0]
        exact_groups = list(connection.execute("SELECT exact_hash,COUNT(*) AS n FROM records"+where+" GROUP BY exact_hash HAVING COUNT(*)>1",params))
        quality_groups = list(connection.execute("SELECT quality_hash,COUNT(*) AS n FROM records"+where+" GROUP BY quality_hash HAVING COUNT(*)>1",params))
        condition = where+(" AND " if where else " WHERE ")+"id IS NOT NULL"
        id_groups = list(connection.execute("SELECT id,COUNT(*) AS n FROM records"+condition+" GROUP BY id HAVING COUNT(*)>1",params))
        totals = Counter()
        for group in id_groups:
            predicate = "id=?" if dataset=="ALL" else "id=? AND dataset=?"
            values = (group["id"],) if dataset=="ALL" else (group["id"],dataset)
            records = [dict(r) for r in connection.execute("SELECT * FROM records WHERE "+predicate+" ORDER BY row_no",values)]
            any_conflict = any_quality_conflict = False
            for left,right in combinations(records,2):
                result = compare_record_pair(left,right,metrics)
                different = result["classification"]=="SAME_ID_DIFFERENT_DATA"
                any_conflict |= different
                any_quality_conflict |= not result["quality_same"]
                totals["same_pairs" if not different else "different_pairs"] += 1
                totals["same_quality_pairs" if result["quality_same"] else "different_quality_pairs"] += 1
                totals["content_not_comparable_pairs"] += result["content_same"]=="NOT_COMPARABLE"
                if different and dataset=="ALL":
                    conflicts.append(pair_export(left,right,result))
            totals["different_groups" if any_conflict else "same_groups"] += 1
            totals["different_quality_groups" if any_quality_conflict else "same_quality_groups"] += 1
        for group in exact_groups:
            predicate = "exact_hash=?" if dataset=="ALL" else "exact_hash=? AND dataset=?"
            values = (group["exact_hash"],) if dataset=="ALL" else (group["exact_hash"],dataset)
            for r in connection.execute("SELECT * FROM records WHERE "+predicate+" ORDER BY row_no",values):
                exact_rows.append({"scope":dataset,"exact_group_hash":group["exact_hash"].hex(),"group_size":group["n"],
                                   "dataset":r["dataset"],"source_file":r["source_file"],"line_number":r["line_number"],
                                   "id":r["id"],"domain":r["domain"],"sub_path":r["sub_path_json"],
                                   "content_present":bool(r["content_present"])})
        summaries.append({"scope":dataset if dataset!="ALL" else "A1+A2+A3","n":n,"unique_id_count":unique,
                          "missing_id_count":n-nonmissing,"exact_duplicate_groups":len(exact_groups),
                          "exact_duplicate_records":sum(r["n"] for r in exact_groups),
                          "exact_duplicate_excess_records":sum(r["n"]-1 for r in exact_groups),
                          "duplicated_id_groups":len(id_groups),"duplicated_id_records":sum(r["n"] for r in id_groups),
                          "duplicated_id_excess_records":nonmissing-unique,
                          "SAME_ID_SAME_DATA_groups":totals["same_groups"],"SAME_ID_DIFFERENT_DATA_groups":totals["different_groups"],
                          "SAME_ID_SAME_DATA_pairs":totals["same_pairs"],"SAME_ID_DIFFERENT_DATA_pairs":totals["different_pairs"],
                          "same_id_same_quality_groups":totals["same_quality_groups"],"same_id_different_quality_groups":totals["different_quality_groups"],
                          "same_id_same_quality_pairs":totals["same_quality_pairs"],"same_id_different_quality_pairs":totals["different_quality_pairs"],
                          "content_not_comparable_pairs":totals["content_not_comparable_pairs"],
                          "raw_quality_duplicate_groups":len(quality_groups),
                          "raw_quality_duplicate_excess_records":sum(r["n"]-1 for r in quality_groups),
                          "comparison_basis":COMPARE_BASIS})
        LOGGER.info("%s: exact groups=%s, duplicated-ID groups=%s, different-data groups=%s",dataset,len(exact_groups),len(id_groups),totals["different_groups"])
    return summaries,exact_rows,conflicts


def cross_dataset_diagnosis(connection,metrics):
    """Compare every cross-dataset pair for each shared ID; never remove overlap."""
    rows = []
    for left_dataset,right_dataset,domain in [("A1","A2","arxiv"),("A1","A3","github")]:
        shared = list(connection.execute("""SELECT DISTINCT a.id FROM records a JOIN records b ON a.id=b.id
            WHERE a.dataset=? AND b.dataset=? AND a.domain=? AND b.domain=? ORDER BY a.id""",
            (left_dataset,right_dataset,domain,domain)))
        totals = Counter()
        for shared_row in shared:
            identifier = shared_row["id"]
            left = [dict(r) for r in connection.execute("SELECT * FROM records WHERE id=? AND dataset=? AND domain=?",(identifier,left_dataset,domain))]
            right = [dict(r) for r in connection.execute("SELECT * FROM records WHERE id=? AND dataset=? AND domain=?",(identifier,right_dataset,domain))]
            totals["left_overlap_records"] += len(left)
            totals["right_overlap_records"] += len(right)
            has_same = has_different = False
            for a,b in product(left,right):
                result = compare_record_pair(a,b,metrics)
                same = result["classification"]=="SAME_ID_SAME_DATA"
                has_same |= same
                has_different |= not same
                totals["same_data_pairs" if same else "different_data_pairs"] += 1
                totals["same_quality_pairs" if result["quality_same"] else "different_quality_pairs"] += 1
                totals["exact_record_pairs"] += result["exact_record_same"]
                totals["content_not_comparable_pairs"] += result["content_same"]=="NOT_COMPARABLE"
            totals["ids_with_same_data_pair"] += has_same
            totals["ids_with_different_data_pair"] += has_different
            totals["ids_with_both_pair_types"] += has_same and has_different
        left_n = connection.execute("SELECT COUNT(*) FROM records WHERE dataset=? AND domain=?",(left_dataset,domain)).fetchone()[0]
        right_n = connection.execute("SELECT COUNT(*) FROM records WHERE dataset=? AND domain=?",(right_dataset,domain)).fetchone()[0]
        rows.append({"comparison":f"{left_dataset}-{domain} vs {right_dataset}-{domain}","left_dataset":left_dataset,
                     "right_dataset":right_dataset,"domain":domain,"left_n":left_n,"right_n":right_n,"same_id":len(shared),
                     "same_id_same_data":totals["ids_with_same_data_pair"],"same_id_different_data":totals["ids_with_different_data_pair"],
                     "same_id_both_types":totals["ids_with_both_pair_types"],"same_id_same_data_pairs":totals["same_data_pairs"],
                     "same_id_different_data_pairs":totals["different_data_pairs"],"same_id_same_quality_pairs":totals["same_quality_pairs"],
                     "same_id_different_quality_pairs":totals["different_quality_pairs"],"exact_record_same_pairs":totals["exact_record_pairs"],
                     "content_not_comparable_pairs":totals["content_not_comparable_pairs"],"left_overlap_records":totals["left_overlap_records"],
                     "right_overlap_records":totals["right_overlap_records"],"left_overlap_rate":totals["left_overlap_records"]/left_n if left_n else None,
                     "right_overlap_rate":totals["right_overlap_records"]/right_n if right_n else None,
                     "comparison_basis":COMPARE_BASIS,"records_removed":0})
    return rows


def different_id_quality_diagnosis(connection):
    """Trace raw-quality matches across different IDs without calling them text duplicates."""
    rows = []
    groups = list(connection.execute("SELECT quality_hash,COUNT(DISTINCT id) AS ids FROM records GROUP BY quality_hash HAVING COUNT(DISTINCT id)>1"))
    for group in groups:
        for r in connection.execute("SELECT * FROM records WHERE quality_hash=? ORDER BY row_no",(group["quality_hash"],)):
            rows.append({"quality_group_hash":r["quality_hash"].hex(),"distinct_id_count":group["ids"],
                         "dataset":r["dataset"],"source_file":r["source_file"],"line_number":r["line_number"],
                         "id":r["id"],"domain":r["domain"],"sub_path":r["sub_path_json"],
                         "content_present":bool(r["content_present"]),"exact_record_hash":r["exact_hash"].hex(),
                         "comparison_basis":"original_22_quality_fields_only","records_removed":0})
    return rows


def missing_diagnosis(connection,threshold):
    """Diagnose each incomplete source occurrence and distinguish uniqueness bases."""
    import json
    rows = []
    for r in connection.execute("SELECT * FROM records WHERE missing_count>0 ORDER BY row_no"):
        names = json.loads(r["missing_metrics_json"])
        count = len(names)
        rows.append({"dataset":r["dataset"],"source_file":r["source_file"],"line_number":r["line_number"],"id":r["id"],
                     "domain":r["domain"],"sub_path":r["sub_path_json"],"missing_metrics":json_cell(names),
                     "missing_count":count,"indicator_count":22,"missing_rate":count/22,
                     "reasoning_missing":"modernbert_reasoning" in names,"professionalism_missing":"modernbert_professionalism" in names,
                     "reasoning_and_professionalism_missing":all(m in names for m in ["modernbert_reasoning","modernbert_professionalism"]),
                     "missing_rate_gt_20pct":count/22>threshold,"exact_record_hash":r["exact_hash"].hex(),
                     "raw_quality_hash":r["quality_hash"].hex(),
                     "id_occurrences_all":connection.execute("SELECT COUNT(*) FROM records WHERE id=?",(r["id"],)).fetchone()[0],
                     "imputed":False,"deleted":False})
    summary = {"scalar_missing_events":sum(r["missing_count"] for r in rows),"source_occurrence_records":len(rows),
               "unique_ids":len({r["id"] for r in rows}),"unique_exact_records":len({r["exact_record_hash"] for r in rows}),
               "unique_quality_records":len({r["raw_quality_hash"] for r in rows}),
               "missing_count_distribution":dict(Counter(r["missing_count"] for r in rows)),
               "both_reasoning_and_professionalism_records":sum(r["reasoning_and_professionalism_missing"] for r in rows),
               "missing_rate_gt_20pct_records":sum(r["missing_rate_gt_20pct"] for r in rows),
               "threshold":threshold}
    return rows,summary


def ceiling_diagnosis(group_n,grade_counts,metrics):
    """Report every possible discrete level, including observed zero counts."""
    level_rows,ceiling_rows = [],[]
    for (scope,dataset,domain),n in group_n.items():
        for metric in metrics:
            name = metric["name"]
            divisor = float(metric.get("divisor",1))
            levels = metric["expected_length"]
            counts = [grade_counts[(scope,dataset,domain,name,k)] for k in range(levels)]
            valid = sum(counts)
            common = {"scope":scope,"dataset":dataset,"domain":domain,"metric":name,"n":n,"valid_count":valid,"missing":n-valid}
            for k,count in enumerate(counts):
                level_rows.append({**common,"rating_index":k,"score":k/divisor,"count":count,
                                   "proportion":count/valid if valid else None,"proportion_all_records":count/n})
            top = next(k for k in range(levels) if k/divisor==1)
            ceiling_rows.append({**common,"ceiling_count":counts[top],"ceiling_rate":counts[top]/valid if valid else None,
                                 "ceiling_rate_all_records":counts[top]/n,"denominator":"finite_classification_scalars"})
    return level_rows,ceiling_rows


def reconcile_missing_with_phase3(rows: list[dict]) -> None:
    """Require every re-read missing field to match the accepted 02 ID audit."""
    import json
    with (PROJECT_ROOT/"outputs"/"profile"/"scalar_missing_records.csv").open(encoding="utf-8-sig",newline="") as stream:
        previous = list(csv.DictReader(stream))
    old = Counter((r["dataset"],r["source_path"],int(r["line_number"]),r["id"],r["metric"]) for r in previous)
    actual = Counter((r["dataset"],r["source_file"],r["line_number"],r["id"],name)
                     for r in rows for name in json.loads(r["missing_metrics"]))
    if actual!=old:
        raise ValueError("Re-read scalar-missing events differ from accepted Phase 3 audit")


def write_report(output,summaries,cross,missing_summary,ceiling_rows,manifest,conflicts):
    """Write actual counts and explain definitions without prescribing cleaning."""
    lines = ["# PHASE 2C：重复记录、ID 冲突、缺失记录与分类天花板诊断", "",
             "本轮只诊断，不删除、选择、合并、平均或填补任何记录；02 结果保持不变。未运行 03/04，未计算 CRITIC 或 Q。", "",
             "## 实际输入与只读核验", "", "| 数据集 | 实际来源文件 | 全量记录数 | SHA-256 一致 |", "|---|---|---:|---|"]
    for r in manifest:
        lines.append(f"| {r['dataset']} | `{r['source_file']}` | {r['record_count']} | {r['source_unchanged']} |")
    lines += ["", "所有来源逐行 XZ 读取；完整 content 只参与当前记录哈希，不长期保存；临时 SQLite 索引不保存全文，运行后移除。", "",
              "## 比较口径", "",
              "- exact duplicate：全部原始 JSON 键值的规范 JSON SHA-256 相同。忽略字典键顺序，保留字段存在性、原键名、字符串、数字类型及来源元数据。不是仅比较 id。",
              "- duplicated id：相同非空字符串 ID 出现至少两次；空/缺失 ID 另计，不把缺失 ID 聚成同一 ID。",
              "- SAME_ID_SAME_DATA：同 ID 的原始 22 质量字段一致，且双方可比较的 content、sub_path、推断/原始 domain 均无差异。",
              "- SAME_ID_DIFFERENT_DATA：上述任何可比较字段不同。原始质量比较使用原标量和 logits，未使用 argmax、截尾或归一化后的值。",
              "- quality-only 口径单独提供 same_id_same_quality / different_quality 列，便于只研究质量字段一致性。",
              "- A2/A3 缺少 content：content_same=NOT_COMPARABLE，不是 True，也不能证明两条文本相同。单侧缺失字段不会强制判成冲突。",
              "- _source_path 和实际 source_file 是溯源信息，不参与 comparable-data 相等判据；完整 exact duplicate 哈希仍包括原始来源字段。",
              "- 别名在质量比较中映射到规范名；同位置 NaN 与 NaN 相等；数值 1 与 1.0、正负零在质量比较中等价。质量 blob 的等同性逐字节检查，不仅依赖哈希。",
              "- max_absolute_difference 仅对不同的有限数值分量计算，不同向量字段计为一个质量字段。缺失/非有限值不可计算有限绝对差的部分另列，无法定义时留空。各指标原单位不同，最大差值不是统一尺度的质量效应。", "",
              "## 重复和 ID 冲突汇总", "",
              "| scope | n | unique id | exact 重复组 | exact 多余出现数 | 重复 ID 组 | ID 多余出现数 | 同数据 ID 组 | 不同数据 ID 组 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['scope']} | {r['n']} | {r['unique_id_count']} | {r['exact_duplicate_groups']} | {r['exact_duplicate_excess_records']} | "
                     f"{r['duplicated_id_groups']} | {r['duplicated_id_excess_records']} | {r['SAME_ID_SAME_DATA_groups']} | {r['SAME_ID_DIFFERENT_DATA_groups']} |")
    lines += ["", "组数指重复键的数量；records 包含重复组的全部成员；excess_records=Σ(组大小−1)，只是计数，不表示删过记录。",
              "同数据组要求组内所有配对均无差异；任何配对有差异则整组记不同数据组。混合组中的同/不同配对数另列，所以组数与配对数不能互换。",
              f"id_conflicts.csv 共 {len(conflicts)} 个不同数据配对，按总体范围只写一次，并保留两端 dataset/source_file/行号/id/sub_path/content 比较结果和质量差异。", "",
              "| scope | 原始 22 质量字段重复组 | 原始质量字段多余出现数 |", "|---|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['scope']} | {r['raw_quality_duplicate_groups']} | {r['raw_quality_duplicate_excess_records']} |")
    lines += ["", "质量字段相同不意味着完整记录或文本相同；不同 ID 的此类组单独追踪于 raw_quality_duplicates_different_ids.csv，不自动删除。", "",
              "## 跨数据集重叠", "", "| 对照 | 左记录数 | 右记录数 | same id | same id+same data | same id+different data | content 不可比较配对 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in cross:
        lines.append(f"| {r['comparison']} | {r['left_n']} | {r['right_n']} | {r['same_id']} | {r['same_id_same_data']} | "
                     f"{r['same_id_different_data']} | {r['content_not_comparable_pairs']} |")
    lines += ["", "same_id 是交集的 unique ID 数；same/different data 是至少有一个对应配对的 ID 数。",
              "如果一 ID 有两种配对，两个计数可交叠，same_id_both_types 会明确记录；所有笛卡尔配对数也分别给出。",
              "完整原记录字段结构不同，可能没有 exact duplicate，但仍有相同 ID 和相同原始质量字段。未从 A2/A3 移除任何重叠。", "",
              "## scalar missing 记录诊断", "",
              f"- 缺失标量事件：{missing_summary['scalar_missing_events']}。",
              f"- 来源记录出现数（dataset+source_file+line_number）：{missing_summary['source_occurrence_records']}。",
              f"- unique ID：{missing_summary['unique_ids']}；unique 完整记录哈希：{missing_summary['unique_exact_records']}；unique 原始质量记录：{missing_summary['unique_quality_records']}。",
              f"- 每条记录缺失数量分布：{missing_summary['missing_count_distribution']}。",
              f"- 同时缺 reasoning 与 professionalism：{missing_summary['both_reasoning_and_professionalism_records']} 条。",
              f"- 缺失比例严格 >20%：{missing_summary['missing_rate_gt_20pct_records']} 条（本轮阈值 {missing_summary['threshold']:.0%}）。", "",
              "缺失诊断重新检查全部 22 个原始字段，在已批准的标量化规则下有限性与标量缺失对应；已逐事件核对 02 的缺失 ID 清单。",
              "未把 qurater 正式拟合前的临时占位当成缺失。所有缺失记录保留，不填补、不删除。", "",
              "## 分类指标天花板", "",
              "每个等级的 proportion=count/valid_count。ceiling_rate=score==1 的数量/有限分类标量数量；同时提供除以全部记录数的 rate_all_records。",
              "modernbert 主方案为 argmax/5；fluency 和 ad 为已确认类别顺序下的 argmax；ad 的 1 表示模型判为无广告。",
              "原始 NaN 不属于任何离散等级，单列 missing。下表数值用于判断区分度，不自动改变转换或权重。", "",
              "| scope | dataset | domain | 指标 | 有效数 | 缺失 | score=1 数 | ceiling_rate |",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for r in ceiling_rows:
        rate = f"{r['ceiling_rate']:.6%}" if r['ceiling_rate'] is not None else "无有效值"
        lines.append(f"| {r['scope']} | {r['dataset']} | {r['domain']} | {r['metric']} | {r['valid_count']} | {r['missing']} | {r['ceiling_count']} | {rate} |")
    lines += ["", "scope=domain 是全量合并领域；scope=dataset_domain 区分 A1 中领域与扩展领域；总体也单列供核验。",
              "相同记录出现会影响合并频率，当前全部记录保留；天花板统计不是对分类器正确率的评估。", "",
              "## 输出与阶段边界", "",
              "- duplicate_summary.csv / exact_duplicates.csv / id_conflicts.csv：全部组和冲突诊断（无事件时 CSV 仍有表头）。",
              "- cross_dataset_overlap.csv：两组跨数据集对照。",
              "- nan_record_diagnosis.csv：每条实际缺失来源记录及缺失字段/比例/双缺失标记。",
              "- classification_level_distribution.csv：所有离散等级的数量和比例，包括零占比等级。",
              "- classification_ceiling_summary.csv：各数据集、各领域的 ceiling_rate。",
              "- cleaning_diagnosis_input_manifest.csv / prior_artifacts_unchanged.csv：原附件哈希及 02 输出未变核验。", "",
              "重复、冲突和天花板仅为诊断；记录处理策略仍待后续决定。本轮停止，不运行 03，不计算 CRITIC 或 Q。"]
    (output/"data_cleaning_diagnosis.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main() -> int:
    """Run the diagnosis only and preserve all accepted profiling artifacts."""
    output = PROJECT_ROOT/"outputs"/"inspection"
    output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),logging.FileHandler(output/"cleaning_diagnosis.log",mode="w",encoding="utf-8")])
    with (PROJECT_ROOT/"configs"/"quality_metrics.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    protected = sorted(p for p in (PROJECT_ROOT/"outputs"/"profile").iterdir() if p.is_file())
    protected.append(PROJECT_ROOT/"outputs"/"preprocess"/"qurater_transform_parameters.csv")
    before = {p:sha256_file(p) for p in protected}
    with diagnosis_workspace(output) as temp_path:
        if not temp_path.is_relative_to(output.resolve()):
            raise RuntimeError("Temporary cleanup target escaped the inspection workspace")
        connection = sqlite3.connect(temp_path/"records.sqlite")
        try:
            initialize_database(connection)
            manifest,group_n,grades,categorical = index_records(connection,config)
            summaries,exact_rows,conflicts = duplicate_diagnosis(connection,config["metrics"])
            different_id_quality = different_id_quality_diagnosis(connection)
            cross = cross_dataset_diagnosis(connection,config["metrics"])
            nan_rows,nan_summary = missing_diagnosis(connection,config["record_diagnosis"]["high_missing_rate_threshold"])
            reconcile_missing_with_phase3(nan_rows)
            level_rows,ceiling_rows = ceiling_diagnosis(group_n,grades,categorical)
        finally:
            connection.close()
    unchanged = []
    for path,old_hash in before.items():
        after = sha256_file(path)
        if old_hash!=after:
            raise RuntimeError(f"Accepted 02 artifact changed: {path.name}")
        unchanged.append({"path":path.relative_to(PROJECT_ROOT).as_posix(),"sha256_before":old_hash,"sha256_after":after,"unchanged":True})
    conflict_headers = ["dataset","source_file","id","dataset_left","dataset_right","source_file_left","source_file_right",
                        "line_number_left","line_number_right","domain_left","domain_right","sub_path","sub_path_left","sub_path_right",
                        "content_present_left","content_present_right","content_same","sub_path_same","domain_same","classification",
                        "quality_classification","comparison_basis","different_quality_fields","different_quality_field_count",
                        "max_absolute_difference","nonfinite_component_mismatch_count","exact_record_same"]
    outputs = {"duplicate_summary.csv":(summaries,None),"exact_duplicates.csv":(exact_rows,["scope","exact_group_hash","group_size","dataset","source_file","line_number","id","domain","sub_path","content_present"]),
               "id_conflicts.csv":(conflicts,conflict_headers),"cross_dataset_overlap.csv":(cross,None),
               "nan_record_diagnosis.csv":(nan_rows,["dataset","source_file","line_number","id","domain","missing_metrics","missing_count","missing_rate"]),
               "classification_level_distribution.csv":(level_rows,None),"classification_ceiling_summary.csv":(ceiling_rows,None),
               "raw_quality_duplicates_different_ids.csv":(different_id_quality,["quality_group_hash","distinct_id_count","dataset","source_file","line_number","id","domain","sub_path","content_present","exact_record_hash","comparison_basis","records_removed"]),
               "cleaning_diagnosis_input_manifest.csv":(manifest,None),"prior_artifacts_unchanged.csv":(unchanged,None)}
    for filename,(rows,headers) in outputs.items():
        write_rows(output/filename,rows,headers)
    write_report(output,summaries,cross,nan_summary,ceiling_rows,manifest,conflicts)
    LOGGER.info("Completed: %s records; %s ID conflict pairs; missing summary=%s",
                sum(r["record_count"] for r in manifest),len(conflicts),json_cell(nan_summary))
    LOGGER.info("STOP: all records retained, no imputation, scripts 03/04 not run, no CRITIC or Q.")
    return 0


if __name__=="__main__":
    try:
        if len(sys.argv)==3 and sys.argv[1]=="--remove-empty-failed-temp":
            remove_empty_failed_temp(sys.argv[2])
            print("Removed empty legacy temporary directory; no records or results changed.")
            sys.exit(0)
        if len(sys.argv)!=1:
            raise ValueError("Unknown diagnostic command arguments")
        sys.exit(main())
    except Exception:
        LOGGER.exception("PHASE 2C diagnosis failed; dependent stages must not proceed.")
        sys.exit(1)
