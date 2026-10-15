"""Quality summaries, exact full-size bootstrap and distribution comparisons."""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.stats import ks_2samp, trim_mean, wasserstein_distance

LOGGER = logging.getLogger("quality.score")


def finite_scores(values) -> np.ndarray:
    """Validate a nonempty quality-score array without discarding observations."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError("Quality scores must be a nonempty finite [0,1] vector")
    return x


def score_profile(values, settings: dict) -> dict:
    """Summarize all scores using sample std and linear quantiles."""
    x = finite_scores(values)
    quantiles = [.01, .05, .10, .25, .50, .75, .90, .95, .99]
    q = np.quantile(x, quantiles, method=settings["quantile_method"])
    return {"n": len(x), "mean_Q": float(x.mean()),
        "std_Q": float(x.std(ddof=settings["std_ddof"])) if len(x) > settings["std_ddof"] else 0.0,
        "min_Q": float(x.min()), **{name: float(v) for name, v in zip(
            ["P01", "P05", "P10", "P25", "median_Q", "P75", "P90", "P95", "P99"], q)}, "max_Q": float(x.max())}


def bootstrap_trimmed_mean(values, settings: dict, rng: np.random.Generator, label: str = "", audit: list | None = None) -> tuple[float, float]:
    """Percentile CI from B ordinary iid n-out-of-n bootstrap replicates.

    Every replicate draws the full original n with replacement, then removes
    floor(n*trim_fraction) scores at each tail. In-place partition is exact for
    the retained mean and avoids sorting/resampling approximations. Batching
    controls memory; no data subsampling or effective-n shortcut is used.
    """
    x = finite_scores(values)
    n, repetitions = len(x), int(settings["bootstrap_repetitions"])
    fraction, confidence = settings["trim_fraction_per_tail"], settings["bootstrap_confidence"]
    if repetitions < 2000 or not 0 <= fraction < .5 or not 0 < confidence < 1:
        raise ValueError("At least 2000 bootstrap draws and valid trim/confidence required")
    cut = math.floor(n * fraction)
    budget = int(settings["bootstrap_working_memory_mb"] * 1024 * 1024)
    batch = max(1, min(settings["bootstrap_max_batch"], budget // (n * 12)))
    statistics = np.empty(repetitions)
    for start in range(0, repetitions, batch):
        count = min(batch, repetitions - start)
        indices = rng.integers(0, n, size=(count, n), dtype=np.int32)
        samples = x[indices]
        del indices
        if cut:
            samples.partition((cut, n - cut - 1), axis=1)
        statistics[start:start+count] = samples[:, cut:n-cut].mean(axis=1)
        if start == 0 or (start + count) // 500 != start // 500 or start + count == repetitions:
            LOGGER.info("Bootstrap %s: %s/%s full-n=%s replicates", label, start + count, repetitions, n)
    alpha = (1-confidence)/2
    low, high = np.quantile(statistics, [alpha, 1-alpha], method=settings["quantile_method"])
    if audit is not None:
        audit.extend({"group": label, "replicate": i+1, "trimmed_mean_Q": float(v)} for i, v in enumerate(statistics))
    return float(low), float(high)


def domain_summary(values, settings: dict, rng: np.random.Generator, label: str = "", audit: list | None = None) -> dict:
    """Aggregate quality; CI estimates the same trimmed-mean domain estimand."""
    profile = score_profile(values, settings)
    estimate = float(trim_mean(values, settings["trim_fraction_per_tail"]))
    low, high = bootstrap_trimmed_mean(values, settings, rng, label, audit)
    return {**profile, "trimmed_mean_Q": estimate, "Q_domain": estimate,
        "ci95_low": low, "ci95_high": high, "bootstrap_repetitions": settings["bootstrap_repetitions"],
        "random_state": settings["random_state"], "trim_fraction_per_tail": settings["trim_fraction_per_tail"],
        "bootstrap_method": "ordinary_full_n_percentile_CI_of_trimmed_mean"}


def cliffs_delta(first, second) -> float:
    """P(first>second)-P(first<second), treating ties as zero; no n*m matrix."""
    x, y = finite_scores(first), np.sort(finite_scores(second))
    less = np.searchsorted(y, x, side="left").sum(dtype=np.int64)
    greater = (len(y) - np.searchsorted(y, x, side="right")).sum(dtype=np.int64)
    return float((int(less) - int(greater)) / (len(x) * len(y)))


def compare_distributions(sample, extended, sample_summary: dict, extended_summary: dict, settings: dict) -> dict:
    """Compare location and shape; positive signed effects favor extension."""
    a, b = finite_scores(sample), finite_scores(extended)
    ks = ks_2samp(a, b, alternative="two-sided", method=settings["ks_method"])
    mean_difference = extended_summary["mean_Q"] - sample_summary["mean_Q"]
    median_difference = extended_summary["median_Q"] - sample_summary["median_Q"]
    trimmed_difference = extended_summary["trimmed_mean_Q"] - sample_summary["trimmed_mean_Q"]
    baseline = sample_summary["trimmed_mean_Q"]
    return {"sample_n": len(a), "extended_n": len(b),
        **{f"sample_{k}": sample_summary[k] for k in ["mean_Q", "median_Q", "std_Q", "trimmed_mean_Q", "ci95_low", "ci95_high"]},
        **{f"extended_{k}": extended_summary[k] for k in ["mean_Q", "median_Q", "std_Q", "trimmed_mean_Q", "ci95_low", "ci95_high"]},
        "mean_difference": mean_difference, "median_difference": median_difference,
        "trimmed_mean_difference": trimmed_difference,
        "relative_difference": trimmed_difference / abs(baseline) if baseline != 0 else None,
        "relative_difference_reference": "sample_trimmed_mean",
        "KS_statistic": float(ks.statistic), "KS_p_value": float(ks.pvalue),
        "KS_method": settings["ks_method"], "Wasserstein_distance": float(wasserstein_distance(a, b)),
        "Cliffs_delta": cliffs_delta(b, a), "difference_direction": "extended_minus_sample"}
