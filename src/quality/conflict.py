"""Within-record disagreement, pair differences and tie-aware driver summaries.

Reads locked Q and weights; never fits preprocessing/CRITIC or changes scores.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, norm


def conflict_indices(z: np.ndarray, q: np.ndarray, weights: np.ndarray) -> dict:
    """Calculate weighted MAD K, unweighted MAD and weighted variance V."""
    values,q,w=np.asarray(z,dtype=float),np.asarray(q,dtype=float),np.asarray(weights,dtype=float)
    if values.ndim!=2 or values.shape[1]!=22 or len(q)!=len(values) or w.shape!=(22,):
        raise ValueError("Expected aligned n x 22 Z, n Q and 22 locked weights")
    if not np.isfinite(values).all() or not np.isfinite(q).all() or not np.isfinite(w).all():
        raise ValueError("Nonfinite locked inputs")
    if (values<0).any() or (values>1).any() or (w<0).any() or not np.isclose(w.sum(),1,atol=1e-12,rtol=0):
        raise ValueError("Invalid normalized inputs or locked weight sum")
    if not np.allclose(values@w,q,rtol=0,atol=1e-14):
        raise ValueError("Saved Q differs from fixed weighted consensus; do not repair it")
    deviation=values-q[:,None]
    k=np.abs(deviation)@w
    variance=np.square(deviation)@w
    equal_q=values.mean(axis=1)
    equal_k=np.abs(values-equal_q[:,None]).mean(axis=1)
    if (k<0).any() or (k>2*q*(1-q)+1e-12).any() or (variance>q*(1-q)+1e-12).any():
        raise ValueError("Conflict indices violate exact bounded-variable inequalities")
    return {"K":k,"K_equal":equal_k,"V":variance,"Q_equal":equal_q,"Delta_Q":np.abs(q-equal_q)}


def fit_conflict_thresholds(k: np.ndarray, settings: dict) -> dict:
    """Fit A1 K thresholds only; external observations must never be passed."""
    values=np.asarray(k,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():raise ValueError("Invalid A1 K")
    quantiles=settings["threshold_quantiles"]
    return {f"P{int(round(p*100)):02d}":float(v) for p,v in zip(quantiles,np.quantile(values,quantiles,method=settings["quantile_method"]))}


def wilson_interval(count: int,n: int) -> tuple[float,float]:
    """Descriptive 95% binomial rate interval; not a domain-significance test."""
    if n<=0:return float('nan'),float('nan')
    z=float(norm.ppf(.975));p=count/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(mid-half),float(mid+half)


def conflict_profile(k,thresholds: dict,settings: dict) -> dict:
    """Describe K and high/extreme rates relative to fixed A1 boundaries."""
    x=np.asarray(k,dtype=float)
    high=int((x>thresholds["P90"]).sum());extreme=int((x>thresholds["P95"]).sum())
    lo,hi=wilson_interval(high,len(x))
    quantiles=np.quantile(x,[.01,.05,.1,.25,.5,.75,.9,.95,.99],method=settings["quantile_method"])
    return {"n":len(x),"mean_K":float(x.mean()),"median_K":float(np.median(x)),
        "std_K":float(x.std(ddof=settings["std_ddof"])) if len(x)>1 else 0.0,
        "min_K":float(x.min()),"max_K":float(x.max()),
        **{name:float(v) for name,v in zip(["P01_K","P05_K","P10_K","P25_K","P50_K","P75_K","P90_K","P95_K","P99_K"],quantiles)},
        "high_conflict_count":high,"high_conflict_rate":high/len(x),
        "extreme_conflict_count":extreme,"extreme_conflict_rate":extreme/len(x),
        "high_rate_wilson95_low":lo,"high_rate_wilson95_high":hi,
        "high_threshold_A1_P90":thresholds["P90"],"extreme_threshold_A1_P95":thresholds["P95"]}


def pair_conflict_summary(z, names: list[str], settings: dict, metadata: dict) -> list[dict]:
    """Aggregate one pair at a time; never materialize/save n x 231 differences."""
    values=np.asarray(z,dtype=float)
    ranks=rankdata(values,axis=0,method="average")
    def correlations(matrix):
        centered=matrix-matrix.mean(axis=0);sums=np.square(centered).sum(axis=0)
        denominator=np.sqrt(sums[:,None]*sums[None,:])
        result=np.full((22,22),np.nan)
        # Exact constant levels can have a tiny centering residual after summation.
        # They are still constant: do not invent a correlation from roundoff.
        variable=np.ptp(matrix,axis=0)>0
        defined=(denominator>0)&variable[:,None]&variable[None,:]
        np.divide(centered.T@centered,denominator,out=result,where=defined)
        return np.clip(result,-1,1)
    pearson,spearman=correlations(values),correlations(ranks)
    output=[]
    for j in range(22):
        for k in range(j+1,22):
            difference=np.abs(values[:,j]-values[:,k])
            median,p90,p95=np.quantile(difference,[.5,.9,.95],method=settings["quantile_method"])
            severe=int((difference>=settings["severe_pair_difference"]).sum())
            output.append({**metadata,"metric_1":names[j],"metric_2":names[k],"n":len(values),
                "mean_abs_difference":float(difference.mean()),"median_abs_difference":float(median),
                "P90_abs_difference":float(p90),"P95_abs_difference":float(p95),
                "severe_pair_conflict_count":severe,"severe_pair_conflict_rate":severe/len(values),
                "severe_difference_threshold":settings["severe_pair_difference"],
                "pearson_r":float(pearson[j,k]) if np.isfinite(pearson[j,k]) else None,
                "spearman_r":float(spearman[j,k]) if np.isfinite(spearman[j,k]) else None,
                "correlation_defined":bool(np.isfinite(pearson[j,k]) and np.isfinite(spearman[j,k]))})
    return output


def deviation_driver_summary(z, high_mask, names: list[str], metadata: dict) -> list[dict]:
    """Distribute one sample's count uniformly across all tied max/min pairs.

    Each significant-conflict record contributes exactly one fractional count.
    Raw possible counts can overlap and must not be summed as distinct records.
    """
    values=np.asarray(z,dtype=float)[high_mask]
    fractional=np.zeros((22,22));possible=np.zeros((22,22),dtype=np.int64)
    for row in values:
        positive=np.where(row==row.max())[0];negative=np.where(row==row.min())[0]
        fractional[np.ix_(positive,negative)]+=1/(len(positive)*len(negative))
        possible[np.ix_(positive,negative)]+=1
    if not np.isclose(fractional.sum(),len(values),atol=1e-8,rtol=0):raise ValueError("Driver tie counts do not reconcile")
    output=[{**metadata,"high_conflict_n":len(values),"top_positive_metric":names[j],"top_negative_metric":names[k],
        "fractional_sample_count":float(fractional[j,k]),"fractional_rate":float(fractional[j,k]/len(values)) if len(values) else 0.0,
        "possible_sample_count":int(possible[j,k]),"tie_policy":"fractional_all_max_min_pairs"}
        for j in range(22) for k in range(22) if possible[j,k]>0]
    output.sort(key=lambda r:(-r["fractional_rate"],r["top_positive_metric"],r["top_negative_metric"]))
    for rank,row in enumerate(output,1):row["driver_rank"]=rank
    return output
