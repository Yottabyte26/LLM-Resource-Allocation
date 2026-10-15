"""Descriptive statistics of approved scalar indicators, with explicit scopes."""

from __future__ import annotations

import numpy as np


def indicator_profile(values: np.ndarray, ddof: int = 1, quantile_method: str = "linear") -> dict:
    """Report finite count and missing before imputation; std is sample std."""
    data = np.asarray(values, dtype=np.float64)
    valid = data[np.isfinite(data)]
    n, count = len(data), len(valid)
    result = {"n": n, "count": count, "missing": n-count, "missing_rate": (n-count)/n if n else 0.0}
    if not count:
        result.update(dict.fromkeys(["mean","std","min","P01","P05","P10","P25","P50","P75","P90","P95","P99","max"]))
        return result
    quantiles = np.quantile(valid, [0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99], method=quantile_method)
    result.update({"mean": float(valid.mean()), "std": float(valid.std(ddof=ddof)) if count>ddof else None,
                   "min": float(valid.min()), "max": float(valid.max())})
    result.update({name:float(value) for name,value in zip(["P01","P05","P10","P25","P50","P75","P90","P95","P99"],quantiles)})
    return result


def profile_matrix(matrix: np.ndarray, metric_names: list[str], metadata: dict, settings: dict) -> list[dict]:
    """Create one row per official indicator, never auxiliary or subcomponent rows."""
    if matrix.ndim != 2 or matrix.shape[1] != 22 or len(metric_names) != 22:
        raise ValueError("Profile input must have exactly 22 indicator columns")
    return [{**metadata, "metric":name,
             **indicator_profile(matrix[:,j], settings["std_ddof"], settings["quantile_method"])}
            for j,name in enumerate(metric_names)]
