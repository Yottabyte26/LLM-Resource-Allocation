"""CRITIC objective weighting only; no sample or domain quality scoring."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def critic_weights(z: np.ndarray, correlation_method: str = "pearson") -> dict:
    """Fit CRITIC with sample std and Pearson (or sensitivity Spearman).

    Spearman ranks use average ties, while sigma always comes from original Z.
    Undefined constant-column correlations and zero total information stop the
    fit; no replacement correlations, equal-weight fallback or reweighting.
    """
    values = np.asarray(z, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("CRITIC requires at least two rows and two indicators")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("CRITIC input must be finite Z within [0,1]")
    sigma = values.std(axis=0, ddof=1)
    constant = np.where((sigma == 0) | (np.ptp(values, axis=0) == 0))[0]
    if len(constant):
        raise ValueError(f"Constant indicators {constant.tolist()}: correlations undefined; fit stopped")
    if correlation_method == "pearson":
        correlation_values = values
    elif correlation_method == "spearman":
        correlation_values = rankdata(values, axis=0, method="average")
    else:
        raise ValueError("Only Pearson main or Spearman sensitivity supported")
    correlation = np.corrcoef(correlation_values, rowvar=False)
    if not np.isfinite(correlation).all() or np.abs(correlation).max() > 1 + 1e-12:
        raise ValueError("Invalid correlation matrix")
    # Remove only floating-point drift outside theoretical correlation bounds.
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    conflict = (1.0 - correlation).sum(axis=1)
    information = sigma * conflict
    total = float(information.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Zero/invalid total CRITIC information; fit stopped")
    weights = information / total
    if (weights < 0).any() or not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("CRITIC weight validation failed")
    return {"std": sigma, "correlation": correlation, "conflict_information": conflict,
            "information_content": information, "weights": weights,
            "weight_rank": rankdata(-weights, method="min").astype(int)}


def weight_entropy(weights: np.ndarray) -> tuple[float, float]:
    """Return natural-log entropy and entropy divided by ln(m), with 0ln0=0."""
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid weight vector")
    if not np.isclose(values.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("Weight vector does not sum to 1")
    positive = values[values > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    return entropy, float(entropy / np.log(len(values)))
