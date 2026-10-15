"""Verify real persisted PHASE 4B outputs independently; never score/refit inputs."""

from __future__ import annotations

import csv
import json
import math
import sys
from itertools import zip_longest
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu, pearsonr, spearmanr, trim_mean, wasserstein_distance

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/"src"))
from quality.io import sha256_file
from quality.reporting import read_table


def main():
    """Check every saved sample Q, summaries, exact CI draws and fixed inputs."""
    quality,preprocess=PROJECT_ROOT/"outputs"/"quality",PROJECT_ROOT/"outputs"/"preprocess"
    model=json.loads((preprocess/"preprocessing_model_A1.json").read_text(encoding="utf-8"))
    names=model["metric_order"]
    weights={r["metric"]:float(r["critic_weight"]) for r in read_table(quality/"critic_weights_A1.csv")}
    sensitivity={r["metric"]:float(r["critic_weight"]) for r in read_table(quality/"critic_weights_spearman_sensitivity.csv")}
    q,ids,domains,secondary={}, {}, {}, {}
    filenames={"A1":("A1_quality_features.csv","A1_sample_quality.csv"),"A2":("A2_arxiv_quality_features.csv","A2_arxiv_quality.csv"),"A3":("A3_github_quality_features.csv","A3_github_quality.csv")}
    summary={r["dataset"]:r for r in read_table(quality/"Q_dataset_summary.csv")}
    for dataset,(original,scored) in filenames.items():
        scores,seen,domain_list,equal,spearman=[],set(),[],[],[]
        with (preprocess/original).open(encoding="utf-8-sig",newline="") as a,(quality/scored).open(encoding="utf-8-sig",newline="") as b:
            source,out=csv.DictReader(a),csv.DictReader(b)
            assert "content" not in out.fieldnames and len(names)==22
            for raw,r in zip_longest(source,out):
                assert raw is not None and r is not None and None not in r
                assert all(raw[k]==r[k] for k in raw),"Features/provenance changed"
                assert r["id"] not in seen;seen.add(r["id"])
                actual=float(r["Q"])
                expected=math.fsum(weights[name]*float(raw[name]) for name in names)
                assert math.isfinite(actual) and 0<=actual<=1 and abs(actual-expected)<1e-14
                scores.append(actual);domain_list.append(r["domain"])
                if dataset=="A1":
                    equal.append(math.fsum(float(raw[name]) for name in names)/22)
                    spearman.append(math.fsum(sensitivity[name]*float(raw[name]) for name in names))
        q[dataset]=np.array(scores);ids[dataset]=seen;domains[dataset]=np.array(domain_list)
        expected_q=np.quantile(q[dataset],[.01,.05,.10,.25,.5,.75,.9,.95,.99])
        r=summary[dataset]
        assert int(r["n"])==len(scores)
        for key,value in zip(["P01","P05","P10","P25","median_Q","P75","P90","P95","P99"],expected_q):assert abs(float(r[key])-value)<1e-14
        assert abs(float(r["mean_Q"])-np.mean(scores))<1e-14
        assert abs(float(r["std_Q"])-np.std(scores,ddof=1))<1e-14
        if dataset=="A1":secondary={"equal":np.array(equal),"spearman_critic":np.array(spearman)}
        print(f"PASS {dataset}: every one of {len(scores)} Q values, identical retained Z/provenance, finite [0,1] and dataset statistics")
    a1_domains=read_table(quality/"domain_quality_summary_A1.csv")
    extended=read_table(quality/"extended_domain_quality_summary.csv")
    remainder=read_table(quality/"nonoverlap_quality_summary.csv")
    assert [float(r["Q_domain"]) for r in a1_domains]==sorted([float(r["Q_domain"]) for r in a1_domains],reverse=True)
    nonoverlap={}
    for dataset,domain in [("A2","arxiv"),("A3","github")]:
        index={r["id"] for r in read_table(preprocess/f"{dataset}_{domain}_nonoverlap_ids.csv")}
        with (quality/filenames[dataset][1]).open(encoding="utf-8-sig",newline="") as f:
            nonoverlap[dataset]=np.array([float(r["Q"]) for r in csv.DictReader(f) if r["id"] in index])
        assert len(nonoverlap[dataset])==len(index)
    groups={f"A1:{r['domain']}":q["A1"][domains["A1"]==r["domain"]] for r in a1_domains}
    groups.update({f"{r['dataset']}:{r['domain']}:full":q[r["dataset"]] for r in extended})
    groups.update({f"{r['dataset']}:{r['domain']}:nonoverlap":nonoverlap[r["dataset"]] for r in remainder})
    statistics={group:[] for group in groups}
    for r in read_table(quality/"bootstrap_trimmed_means.csv"):
        assert int(r["replicate"])==len(statistics[r["group"]])+1
        statistics[r["group"]].append(float(r["trimmed_mean_Q"]))
    for r in a1_domains+extended+remainder:
        group=f"A1:{r['domain']}" if r["dataset"]=="A1" else f"{r['dataset']}:{r['domain']}:{r['view']}"
        values=groups[group]
        assert int(r["n"])==len(values) and len(statistics[group])==int(r["bootstrap_repetitions"])==2000
        assert abs(float(r["Q_domain"])-trim_mean(values,.1))<1e-14
        low,high=np.quantile(statistics[group],[.025,.975])
        assert abs(float(r["ci95_low"])-low)<1e-14 and abs(float(r["ci95_high"])-high)<1e-14
    for r in read_table(quality/"sample_extended_consistency.csv"):
        a=q["A1"][domains["A1"]==r["domain"]];dataset="A2" if r["domain"]=="arxiv" else "A3"
        b=q[dataset] if r["view"]=="full" else nonoverlap[dataset]
        ks=ks_2samp(a,b,method="asymp")
        assert abs(float(r["KS_statistic"])-ks.statistic)<1e-14
        assert abs(float(r["KS_p_value"])-ks.pvalue)<1e-14
        assert abs(float(r["Wasserstein_distance"])-wasserstein_distance(a,b))<1e-14
        # Mann-Whitney U gives an independent tie-aware Cliff's delta identity.
        delta=2*mannwhitneyu(b,a,method="asymptotic").statistic/(len(a)*len(b))-1
        assert abs(float(r["Cliffs_delta"])-delta)<1e-12
        assert abs(float(r["mean_difference"])-(b.mean()-a.mean()))<1e-14
        assert abs(float(r["median_difference"])-(np.median(b)-np.median(a)))<1e-14
        diff=trim_mean(b,.1)-trim_mean(a,.1)
        assert abs(float(r["trimmed_mean_difference"])-diff)<1e-14
        assert abs(float(r["relative_difference"])-diff/abs(trim_mean(a,.1)))<1e-14
    for file,alternative in [("Q_weighting_robustness.csv","equal"),("Q_spearman_weighting_sensitivity.csv","spearman_critic")]:
        rows=read_table(quality/file);assert len(rows)==8
        for r in rows:
            mask=np.ones(len(q["A1"]),dtype=bool) if r["scope"]=="overall" else domains["A1"]==r["domain"]
            a,b=q["A1"][mask],secondary[alternative][mask]
            assert abs(float(r["pearson_correlation"])-pearsonr(a,b).statistic)<1e-12
            assert abs(float(r["spearman_correlation"])-spearmanr(a,b).statistic)<1e-12
            if r["scope"]=="domain":assert abs(float(r["alternative_Q_domain"])-trim_mean(b,.1))<1e-14
    csv.field_size_limit(100000000)
    with (quality/"A1_sample_quality.csv").open(encoding="utf-8-sig",newline="") as f:a1=list(csv.DictReader(f))
    for label,reverse in [("top20",True),("bottom20",False)]:
        expected=sorted(a1,key=lambda r:((-float(r['Q']) if reverse else float(r['Q'])),r['id']))[:20]
        samples=read_table(quality/f"A1_{label}_quality_samples.csv")
        assert len(samples)==20 and [r["id"] for r in samples]==[r["id"] for r in expected]
        assert all(isinstance(r['content'],str) and len(r['content'])>0 for r in samples)
    for file in ["prior_artifacts_unchanged_4B.csv","scoring_input_manifest.csv"]:
        for r in read_table(quality/file):assert sha256_file(PROJECT_ROOT/r["path"])==r["sha256_before"]==r["sha256_after"]
    figures=read_table(quality/"quality_figure_manifest.csv");assert len(figures)>=8
    for r in figures:
        path=PROJECT_ROOT/r["path"]
        assert path.stat().st_size>0 and path.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
    print("PASS: 11 full/remaining domain summaries, 22000 actual bootstrap statistics and CIs; four KS/Wasserstein/Mann-Whitney effect checks; sensitivities; true extreme selections; immutable prior artifacts; all figure files")


if __name__=="__main__":main()
