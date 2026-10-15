"""PHASE 4B: apply locked A1 weights, summarize, compare and export real Q.

No CRITIC or preprocessing fitting. --refresh-report reads saved results only.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from array import array
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import pearsonr, rankdata, spearmanr, trim_mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quality.aggregate import compare_distributions, domain_summary, score_profile
from quality.io import discover_quality_files, iter_jsonl, sha256_file
from quality.plotting import make_quality_figures
from quality.reporting import read_table, write_final_report
from quality.vector_diagnostics import write_rows

LOGGER = logging.getLogger("quality.score")


def load_weights(path: Path, names: list[str]) -> np.ndarray:
    """Read complete-precision saved weights, align by metric and never refit."""
    rows = read_table(path)
    lookup = {row["metric"]: float(row["critic_weight"]) for row in rows}
    if len(rows) != 22 or set(lookup) != set(names):
        raise ValueError(f"Invalid 22-metric weight file: {path}")
    weights = np.array([lookup[name] for name in names])
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("Locked weights must be finite/nonnegative and sum to one")
    return weights


def score_dataset(source: Path, target: Path, dataset: str, names: list[str], weights, spearman_weights, settings):
    """Stream numeric CSV batches; write Z/provenance/Q without any content."""
    q_buffer, equal_buffer, spearman_buffer = array("d"), array("d"), array("d")
    metadata, identities = [], set()
    temporary = target.with_suffix(".csv.tmp")
    with source.open(encoding="utf-8-sig", newline="") as stream, temporary.open("w", encoding="utf-8-sig", newline="") as dest:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or any(name not in reader.fieldnames for name in names):
            raise ValueError("Missing feature columns")
        extra = [name for name in reader.fieldnames if name not in names]
        if "content" in extra or "Q" in extra or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Unexpected content/score/duplicate feature columns")
        writer = csv.DictWriter(dest, fieldnames=[*extra, "Q", *names])
        writer.writeheader()
        batch = []

        def consume(records):
            values = np.array([[float(r[name]) for name in names] for r in records])
            if values.shape[1] != 22 or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                raise ValueError("Invalid saved Z")
            scores = values @ weights
            if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                raise ValueError("Invalid Q outside finite [0,1]")
            q_buffer.extend(scores)
            if dataset == "A1":
                equal_buffer.extend(values.mean(axis=1))
                spearman_buffer.extend(values @ spearman_weights)
            for record, score in zip(records, scores):
                writer.writerow({**record, "Q": float(score)})
                metadata.append({k: record[k] for k in extra})

        for row in reader:
            if None in row or any(v is None for v in row.values()) or row["dataset"] != dataset or row["id"] in identities:
                raise ValueError(f"Malformed/wrong-dataset/duplicate-ID feature record in {source.name}")
            identities.add(row["id"])
            batch.append(row)
            if len(batch) == settings["feature_batch_size"]:
                consume(batch); batch = []
        if batch:
            consume(batch)
    temporary.replace(target)
    LOGGER.info("%s scores saved: %s rows; Q min=%.8f max=%.8f", dataset, len(q_buffer), min(q_buffer), max(q_buffer))
    return {"Q": np.frombuffer(q_buffer, dtype=np.float64), "metadata": metadata,
            "Q_equal": np.frombuffer(equal_buffer, dtype=np.float64),
            "Q_spearman": np.frombuffer(spearman_buffer, dtype=np.float64)}


def sensitivity_summary(main, alternative, domains, domain_rows, label: str, settings: dict):
    """Compare all A1 scores and ranks based on each method's trimmed mean."""
    rows = [{"scope": "overall", "alternative": label, "domain": "ALL", "n": len(main),
             "pearson_correlation": float(pearsonr(main, alternative).statistic),
             "spearman_correlation": float(spearmanr(main, alternative).statistic),
             "mean_difference": float((alternative-main).mean()), "mean_absolute_difference": float(np.abs(alternative-main).mean())}]
    domain_names = [r["domain"] for r in domain_rows]
    estimates = [float(trim_mean(alternative[domains == name], settings["trim_fraction_per_tail"])) for name in domain_names]
    ranks = rankdata(-np.array(estimates), method="min").astype(int)
    for name, estimate, rank in zip(domain_names, estimates, ranks):
        primary = next(r for r in domain_rows if r["domain"] == name)
        mask = domains == name
        rows.append({"scope": "domain", "alternative": label, "domain": name, "n": int(mask.sum()),
            "pearson_correlation": float(pearsonr(main[mask], alternative[mask]).statistic),
            "spearman_correlation": float(spearmanr(main[mask], alternative[mask]).statistic),
            "main_Q_domain": primary["Q_domain"], "alternative_Q_domain": estimate,
            "main_rank": primary["quality_rank"], "alternative_rank": int(rank), "rank_change": int(rank-primary["quality_rank"])})
    return rows


def fetch_extreme_texts(a1, settings, output: Path):
    """Read raw A1 once for just top/bottom selected IDs, never store all content."""
    identities = np.array([r["id"] for r in a1["metadata"]])
    count = settings["text_review_top_bottom_count"]
    groups = {"top20": np.lexsort((identities, -a1["Q"]))[:count],
              "bottom20": np.lexsort((identities, a1["Q"]))[:count]}
    targets = {a1["metadata"][i]["id"]: a1["metadata"][i] for indices in groups.values() for i in indices}
    if len(targets) != 2*count:
        raise ValueError("Top/bottom selections overlap; text verification cannot proceed")
    sources = discover_quality_files(PROJECT_ROOT)["A1"]
    texts, manifests = {}, []
    for source in sources:
        original = sha256_file(source)
        relative = source.relative_to(PROJECT_ROOT).as_posix()
        LOGGER.info("Fetching only selected 40 A1 contents from %s", relative)
        for line_number, record, error in iter_jsonl(source):
            if error:
                raise ValueError(f"{relative}:{line_number}: {error}")
            identity = record.get("id")
            if identity in targets:
                meta = targets[identity]
                if identity in texts or meta["source_file"] != relative or int(meta["line_number"]) != line_number:
                    raise ValueError("Selected text ID/source association mismatch")
                if not isinstance(record.get("content"), str):
                    raise ValueError(f"Selected A1 content missing: {identity}")
                texts[identity] = record["content"]
            del record
        after = sha256_file(source)
        if after != original:
            raise RuntimeError("Raw official A1 changed during read")
        manifests.append({"role": "selected_text_source", "path": relative, "sha256_before": original,
                          "sha256_after": after, "unchanged": True})
    if set(texts) != set(targets):
        raise ValueError("Some selected contents not recovered")
    samples = []
    for label, indices in groups.items():
        rows = [{"id": a1["metadata"][i]["id"], "domain": a1["metadata"][i]["domain"],
                 "Q": float(a1["Q"][i]), "content": texts[a1["metadata"][i]["id"]]} for i in indices]
        write_rows(output/f"A1_{label}_quality_samples.csv", rows)
        samples.extend({"set": label, "rank": rank, "id": r["id"], "domain": r["domain"], "Q": r["Q"],
                        "content_characters": len(r["content"]), "human_confirmation": "pending_independent_review"}
                       for rank, r in enumerate(rows, 1))
    write_rows(output/"A1_extreme_text_manifest.csv", samples)
    return manifests


def main(refresh_report: bool = False) -> int:
    """Apply fixed weights, compute exact summaries and stop at part 1."""
    quality = PROJECT_ROOT/"outputs"/"quality"
    quality.mkdir(parents=True, exist_ok=True)
    settings_path = PROJECT_ROOT/"configs"/"quality_scoring.yaml"
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if refresh_report:
        write_final_report(PROJECT_ROOT, settings)
        print("Report refreshed from saved results only; no rescoring/bootstrap/source read")
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(quality/"quality_scoring.log", mode="w", encoding="utf-8")])
    if settings["std_ddof"] != 1 or settings["bootstrap_repetitions"] < 2000 or settings["random_state"] != 2026:
        raise ValueError("Scoring must use sample std, >=2000 bootstrap and seed 2026")
    model_path = PROJECT_ROOT/"outputs"/"preprocess"/"preprocessing_model_A1.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    names = model["metric_order"]
    if len(names) != len(set(names)) or len(names) != 22 or model["fitting_dataset"] != "A1":
        raise ValueError("Invalid fixed A1 model")
    weight_path, sensitivity_path = PROJECT_ROOT/settings["weight_source"], PROJECT_ROOT/settings["spearman_weight_source"]
    protected = sorted(p for p in (PROJECT_ROOT/"outputs"/"preprocess").rglob("*") if p.is_file())
    phase4a_names = ["critic_weights_A1.csv", "critic_weights_spearman_sensitivity.csv", "critic_pearson_correlation_A1.csv",
        "critic_spearman_correlation_A1.csv", "equal_weights.csv", "critic_diagnostics_summary_A1.json", "critic_weight_diagnostics.md",
        "critic_inputs_manifest.csv", "critic_diagnostics_config_snapshot.yaml", "negative_correlation_pairs_A1.csv",
        "moderate_indicator_diagnostics_A1.csv", "discrete_indicator_diagnostics_A1.csv", "phase3_artifacts_unchanged_4A.csv", "critic_A1.log"]
    protected += [quality/name for name in phase4a_names]
    protected += [PROJECT_ROOT/"configs"/"quality_metrics.yaml", PROJECT_ROOT/"configs"/"critic_diagnostics.yaml"]
    before = {path: sha256_file(path) for path in protected}
    weights, spearman_weights = load_weights(weight_path, names), load_weights(sensitivity_path, names)
    feature_names = {"A1": "A1_quality_features.csv", "A2": "A2_arxiv_quality_features.csv", "A3": "A3_github_quality_features.csv"}
    target_names = {"A1": "A1_sample_quality.csv", "A2": "A2_arxiv_quality.csv", "A3": "A3_github_quality.csv"}
    accepted_counts = {r["scope"]: int(r["n"]) for r in read_table(PROJECT_ROOT/"outputs"/"inspection"/"duplicate_summary.csv")}
    datasets = {}
    dataset_rows, validations = [], []
    for dataset, filename in feature_names.items():
        source = PROJECT_ROOT/"outputs"/"preprocess"/filename
        result = score_dataset(source, quality/target_names[dataset], dataset, names, weights, spearman_weights, settings)
        if len(result["Q"]) != accepted_counts[dataset]:
            raise ValueError("Score record count differs from accepted source count")
        datasets[dataset] = result
        dataset_rows.append({"dataset": dataset, **score_profile(result["Q"], settings)})
        validations.append({"dataset": dataset, "n": len(result["Q"]), "min_Q": float(result["Q"].min()),
            "max_Q": float(result["Q"].max()), "nan_count": int(np.isnan(result["Q"]).sum()),
            "inf_count": int(np.isinf(result["Q"]).sum()), "all_in_0_1": bool(((result["Q"]>=0)&(result["Q"]<=1)).all())})
    write_rows(quality/"Q_dataset_summary.csv", dataset_rows)
    write_rows(quality/"Q_validation.csv", validations)
    domains = np.array([r["domain"] for r in datasets["A1"]["metadata"]])
    domain_names = sorted(set(domains))
    seeds = np.random.SeedSequence(settings["random_state"]).spawn(len(domain_names)+4)
    bootstrap_audit, domain_rows, extended_rows, remainder_rows, comparisons = [], [], [], [], []
    for i, domain in enumerate(domain_names):
        rows = datasets["A1"]["Q"][domains==domain]
        domain_rows.append({"dataset": "A1", "domain": domain, **domain_summary(rows, settings, np.random.default_rng(seeds[i]), f"A1:{domain}", bootstrap_audit),
                            "bootstrap_stream_index": i})
    domain_rows.sort(key=lambda r:r["Q_domain"], reverse=True)
    ranks = rankdata(-np.array([r["Q_domain"] for r in domain_rows]), method="min").astype(int)
    for r, rank in zip(domain_rows, ranks):r["quality_rank"] = int(rank)
    remainders = {}
    a1_lookup = {r["id"]: (r["domain"], q) for r,q in zip(datasets["A1"]["metadata"], datasets["A1"]["Q"])}
    for offset, (dataset, domain) in enumerate([("A2", "arxiv"), ("A3", "github")]):
        index = read_table(PROJECT_ROOT/"outputs"/"preprocess"/f"{dataset}_{domain}_nonoverlap_ids.csv")
        nonoverlap = {r["id"] for r in index}
        if len(nonoverlap) != len(index):raise ValueError("Duplicate ID in nonoverlap index")
        actual_ids = {r["id"] for r in datasets[dataset]["metadata"]}
        a1_ids = {identity for identity,(d,_) in a1_lookup.items() if d==domain}
        if nonoverlap != actual_ids-a1_ids or not a1_ids<=actual_ids:
            raise ValueError("Nonoverlap ID index differs from score IDs")
        for meta,q in zip(datasets[dataset]["metadata"], datasets[dataset]["Q"]):
            if meta["id"] in a1_ids and not np.isclose(q,a1_lookup[meta["id"]][1],atol=1e-14,rtol=0):
                raise ValueError("Cross-source identical Z failed equal Q check")
        mask = np.array([r["id"] in nonoverlap for r in datasets[dataset]["metadata"]])
        remainders[dataset] = datasets[dataset]["Q"][mask]
        full_i, remainder_i = len(domain_names)+offset, len(domain_names)+2+offset
        full = {"dataset":dataset,"domain":domain,"view":"full",**domain_summary(datasets[dataset]["Q"],settings,np.random.default_rng(seeds[full_i]),f"{dataset}:{domain}:full",bootstrap_audit),"bootstrap_stream_index":full_i}
        remainder = {"dataset":dataset,"domain":domain,"view":"nonoverlap",**domain_summary(remainders[dataset],settings,np.random.default_rng(seeds[remainder_i]),f"{dataset}:{domain}:nonoverlap",bootstrap_audit),"bootstrap_stream_index":remainder_i}
        extended_rows.append(full);remainder_rows.append(remainder)
        sample = datasets["A1"]["Q"][domains==domain]
        primary = next(r for r in domain_rows if r["domain"]==domain)
        for view, values, summary, shared in [("full",datasets[dataset]["Q"],full,len(a1_ids)),("nonoverlap",remainders[dataset],remainder,0)]:
            comparisons.append({"comparison":f"A1-{domain} vs {dataset}-{view}","domain":domain,"view":view,
                **compare_distributions(sample,values,primary,summary,settings),"shared_id_count":shared,
                "KS_inference_note":"overlapping_records_p_descriptive_only" if shared else "disjoint_ids_sampling_design_unknown"})
    write_rows(quality/"domain_quality_summary_A1.csv",domain_rows)
    write_rows(quality/"extended_domain_quality_summary.csv",extended_rows)
    write_rows(quality/"nonoverlap_quality_summary.csv",remainder_rows)
    write_rows(quality/"sample_extended_consistency.csv",comparisons)
    write_rows(quality/"bootstrap_trimmed_means.csv",bootstrap_audit)
    a1=datasets["A1"]
    equal=sensitivity_summary(a1["Q"],a1["Q_equal"],domains,domain_rows,"equal",settings)
    spearman=sensitivity_summary(a1["Q"],a1["Q_spearman"],domains,domain_rows,"spearman_critic",settings)
    write_rows(quality/"Q_weighting_robustness.csv",equal)
    write_rows(quality/"Q_spearman_weighting_sensitivity.csv",spearman)
    write_rows(quality/"A1_weighting_sensitivity_scores.csv",[{"id":r["id"],"domain":r["domain"],"Q_CRITIC":float(q),"Q_equal":float(e),"Q_spearman":float(s)} for r,q,e,s in zip(a1["metadata"],a1["Q"],a1["Q_equal"],a1["Q_spearman"])])
    text_manifest=fetch_extreme_texts(a1,settings,quality)
    correlation_rows=read_table(quality/"critic_pearson_correlation_A1.csv")
    correlation=np.array([[float(r[n]) for n in names] for r in correlation_rows])
    figure_names=make_quality_figures(PROJECT_ROOT/"outputs"/"figures"/"question1_part1",read_table(weight_path),datasets,domains,domain_rows,remainders,a1["Q_equal"],a1["Q_spearman"],settings,correlation,read_table(PROJECT_ROOT/"outputs"/"preprocess"/"imputation_summary.csv"))
    write_rows(quality/"quality_figure_manifest.csv",[{"figure":n,"path":f"outputs/figures/question1_part1/{n}"} for n in figure_names])
    protected_rows=[]
    for path, original in before.items():
        current=sha256_file(path)
        if original!=current:raise RuntimeError(f"Locked prior artifact changed: {path}")
        protected_rows.append({"path":path.relative_to(PROJECT_ROOT).as_posix(),"sha256_before":original,"sha256_after":current,"unchanged":True})
    write_rows(quality/"prior_artifacts_unchanged_4B.csv",protected_rows)
    sources=[(PROJECT_ROOT/"outputs"/"preprocess"/n,"feature_input") for n in feature_names.values()]+[(weight_path,"locked_main_weight"),(sensitivity_path,"sensitivity_weight"),(model_path,"locked_A1_model")]
    inputs=[{"role":role,"path":p.relative_to(PROJECT_ROOT).as_posix(),"sha256_before":before[p],"sha256_after":sha256_file(p),"unchanged":True} for p,role in sources]+text_manifest
    write_rows(quality/"scoring_input_manifest.csv",inputs)
    (quality/"quality_scoring_config_snapshot.yaml").write_bytes(settings_path.read_bytes())
    write_final_report(PROJECT_ROOT,settings)
    LOGGER.info("Completed: %s real Q records, %s domain/view summaries, %s bootstrap statistics, %s figures",sum(len(r['Q']) for r in datasets.values()),len(domain_rows)+len(extended_rows)+len(remainder_rows),len(bootstrap_audit),len(figure_names))
    LOGGER.info("STOP: question 1 part 1 only; no conflict resolution or domain allocation. Selected texts await explicit review annotations.")
    return 0


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-report",action="store_true",help="Rebuild report from saved tables and text review only")
    args=parser.parse_args()
    try:sys.exit(main(args.refresh_report))
    except Exception:
        LOGGER.exception("Quality scoring failed; no dependent task may proceed")
        sys.exit(1)
