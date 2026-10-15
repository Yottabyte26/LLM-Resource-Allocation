"""Independently verify saved PHASE 3 artifacts against accepted 02/2C tables.

Reads generated CSV/JSON and small historical tables only, not official XZ.
Does not fit parameters, impute, deduplicate, score or mutate any output.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quality.io import sha256_file


def rows(path: Path) -> list[dict]:
    """Load a small audit table."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    """Check independent A1 parameters, every saved Z and overlap indices."""
    output = PROJECT_ROOT / "outputs" / "preprocess"
    historical = PROJECT_ROOT / "outputs" / "inspection"
    model = json.loads((output / "preprocessing_model_A1.json").read_text(encoding="utf-8"))
    names = model["metric_order"]
    assert len(names) == len(set(names)) == 22
    assert model["fitting_dataset"] == "A1"
    assert [len(model["indicator_directions"][d]) for d in ["positive", "negative", "moderate"]] == [14, 5, 3]
    assert all(p["fit_scope"] == "A1" for p in model["qurater_parameters"])
    assert sum(r["n"] for r in model["fitting_inputs"]) == model["fit_record_count"]
    assert all(r["dataset"] == "A1" for r in model["fitting_inputs"])
    profiles = rows(PROJECT_ROOT / "outputs" / "profile" / "quality_indicator_profile.csv")
    a1_profiles = {r["metric"]: r for r in profiles if r["dataset"] == "A1" and r["scope"] == "dataset"}
    # Continuous scalars had no missing observations, so historical A1-only
    # descriptive quantiles are an independent reference unaffected by pooled q.
    for metric, p in model["winsor_thresholds"].items():
        expected = a1_profiles[metric]
        assert int(expected["missing"]) == 0
        assert p["P01"] == float(expected["P01"]) and p["P99"] == float(expected["P99"]), metric
    for metric, p in model["moderate_utility_thresholds"].items():
        for key in ["P01", "P05", "P10", "P50", "P90", "P95", "P99"]:
            assert p[key] == float(a1_profiles[metric][key]), (metric, key)
    domain_profiles = {(r["domain"], r["metric"]): r for r in rows(PROJECT_ROOT / "outputs" / "profile" / "domain_indicator_profile.csv")
                       if r["scope"] == "dataset_domain" and r["dataset"] == "A1"}
    for metric, p in model["imputation_medians"].items():
        if metric == "qurater":
            continue  # Historical q uses pooled internal bounds; it is not main q.
        assert p["overall"] == float(a1_profiles[metric]["P50"])
        for domain, median in p["domains"].items():
            assert median == float(domain_profiles[(domain, metric)]["P50"])
    baseline = {r["scope"]: int(r["n"]) for r in rows(historical / "duplicate_summary.csv")}
    saved_checks = {(r["dataset"], r["metric"]): r for r in rows(output / "feature_validation.csv")}
    imputation_log = rows(output / "imputation_log.csv")
    previous_missing = rows(PROJECT_ROOT / "outputs" / "profile" / "scalar_missing_records.csv")
    assert {(r["dataset"], r["source_file"], int(r["line_number"]), r["id"], r["metric"]) for r in imputation_log} == {
        (r["dataset"], r["source_path"], int(r["line_number"]), r["id"], r["metric"]) for r in previous_missing}
    by_record = {}
    for r in imputation_log:
        by_record.setdefault((r["dataset"], r["id"]), []).append(r)
        assert r["fitting_dataset"] == "A1"
        assert r["imputation_method"] == "A1_domain_median"
        assert float(r["imputed_value"]) == float(domain_profiles[(r["domain"], r["metric"])]["P50"])
    lattice = {m["name"]: np.arange(m["expected_length"]) / m.get("divisor", 1)
               for m in model["metric_definitions"] if m.get("scalarization") in ["argmax_scaled", "argmax_index"]}
    level_reference = {(r["dataset"], r["metric"], float(r["score"])): int(r["count"])
                       for r in rows(historical / "classification_level_distribution.csv") if r["scope"] == "dataset"}
    filenames = {"A1": "A1_quality_features.csv", "A2": "A2_arxiv_quality_features.csv", "A3": "A3_github_quality_features.csv"}
    ids, a1_overlap, grade_counts = {}, {}, Counter()
    quality_pair = {r["id"] for r in rows(historical / "raw_quality_duplicates_different_ids.csv")}
    pair_vectors = []
    for dataset, filename in filenames.items():
        seen, n = set(), 0
        minimum, maximum = np.full(22, np.inf), np.full(22, -np.inf)
        total_imputed = 0
        with (output / filename).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == ["id", "dataset", "domain", "sub_path", "source_file", "line_number", "imputed_count", *names]
            for r in reader:
                assert None not in r and all(v is not None for v in r.values())
                assert r["dataset"] == dataset and r["id"] not in seen
                seen.add(r["id"])
                vector = np.array([float(r[name]) for name in names])
                assert np.isfinite(vector).all() and ((vector >= 0) & (vector <= 1)).all()
                minimum, maximum = np.minimum(minimum, vector), np.maximum(maximum, vector)
                events = by_record.get((dataset, r["id"]), [])
                assert int(r["imputed_count"]) == len(events)
                total_imputed += int(r["imputed_count"])
                for event in events:
                    # Actual missing events are bounded argmax classifications.
                    assert vector[names.index(event["metric"])] == float(event["imputed_value"])
                for metric, allowed in lattice.items():
                    value = vector[names.index(metric)]
                    if metric not in [e["metric"] for e in events]:
                        assert value in allowed
                    grade_counts[dataset, metric, value] += 1
                if dataset == "A1" and r["domain"] in {"arxiv", "github"}:
                    a1_overlap[r["id"]] = (r["domain"], vector.copy())
                if dataset != "A1" and r["id"] in a1_overlap:
                    assert r["domain"] == a1_overlap[r["id"]][0]
                    assert np.array_equal(vector, a1_overlap[r["id"]][1])
                if dataset == "A3" and r["id"] in quality_pair:
                    pair_vectors.append(vector.copy())
                n += 1
        assert n == baseline[dataset]
        for j, name in enumerate(names):
            saved = saved_checks[dataset, name]
            assert n == int(saved["n"]) and minimum[j] == float(saved["min"]) and maximum[j] == float(saved["max"])
            assert int(saved["nan_count"]) == int(saved["inf_count"]) == 0
        assert total_imputed == sum(r["dataset"] == dataset for r in imputation_log)
        ids[dataset] = seen
        print(f"PASS {dataset}: {n} saved rows, 22 finite [0,1] features, {total_imputed} logged imputations")
    for key, reference_count in level_reference.items():
        extra = sum(r["dataset"] == key[0] and r["metric"] == key[1] and float(r["imputed_value"]) == key[2] for r in imputation_log)
        assert grade_counts[key] == reference_count + extra, key
    assert len(pair_vectors) == 2 and np.array_equal(*pair_vectors)
    for dataset, domain in [("A2", "arxiv"), ("A3", "github")]:
        reference_ids = {i for i, (d, _) in a1_overlap.items() if d == domain}
        index = rows(output / f"{dataset}_{domain}_nonoverlap_ids.csv")
        nonoverlap = {r["id"] for r in index}
        assert len(nonoverlap) == len(index)
        assert nonoverlap == ids[dataset] - reference_ids
        assert reference_ids <= ids[dataset]
        print(f"PASS {dataset}: overlap {len(reference_ids)}, nonoverlap {len(nonoverlap)}")
    for r in rows(output / "prior_artifacts_unchanged_phase3.csv"):
        assert sha256_file(PROJECT_ROOT / r["path"]) == r["sha256_before"] == r["sha256_after"]
    for filename, expected_n in [("imputation_parameters_A1.csv", 176), ("winsor_parameters_A1.csv", 15),
        ("qurater_transform_parameters_A1.csv", 4), ("moderate_metric_parameters_A1.csv", 3),
        ("minmax_parameters_A1.csv", 12), ("external_clip_rates.csv", 44), ("external_qurater_component_clip_rates.csv", 8)]:
        assert len(rows(output / filename)) == expected_n
    for r in rows(output / "external_clip_rates.csv"):
        assert r["fitting_dataset"] == "A1"
        assert int(r["n"]) == baseline[r["dataset"]]
        assert float(r["clip_low_rate"]) == int(r["clip_low_count"])/int(r["n"])
        assert float(r["clip_high_rate"]) == int(r["clip_high_count"])/int(r["n"])
    assert rows(output / "high_missing_samples.csv") == []
    assert (PROJECT_ROOT / "outputs" / "report" / "phase3_preprocessing_summary.md").read_text(encoding="utf-8").strip()
    print("PASS: independent historical A1 quantiles/medians, saved Z round-trip, grade preservation, overlap equality, both quality-only duplicates retained, historical hashes, complete parameter and audit tables")


if __name__ == "__main__":
    main()
