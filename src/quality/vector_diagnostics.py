"""Raw-vector and Softmax numerical diagnostics, never production features."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute row-wise float64 Softmax for finite nonempty vectors only.

    Numerical validity cannot establish logits provenance or calibrated
    probabilities. Nonfinite raw vectors must remain explicitly unusable.
    """
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Softmax requires a 2D finite matrix with nonempty rows")
    with np.errstate(over="ignore", under="ignore", invalid="raise", divide="raise"):
        shifted = values - values.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=1, keepdims=True)


def distribution(values: np.ndarray) -> dict:
    """Exact descriptive statistics on finite raw values; no Winsorization."""
    data = np.asarray(values, dtype=np.float64)
    if not data.size or not np.isfinite(data).all():
        raise ValueError("Distribution requires nonempty finite observations")
    quantiles = np.quantile(data, [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
    result = {"count": int(data.size), "min": float(data.min()), "max": float(data.max()),
              "mean": float(data.mean()), "std": float(data.std(ddof=1)) if len(data)>1 else 0.0}
    result.update({key: float(value) for key,value in zip(
        ["p01","p05","p10","p25","median","p75","p90","p95","p99"], quantiles)})
    return result


def correlation_tables(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pearson and tie-aware Spearman for the four aligned qurater components."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or len(values)<2:
        raise ValueError("qurater correlation needs at least two complete length-4 rows")
    if np.any(values.std(axis=0) == 0):
        raise ValueError("Constant qurater component: correlations are undefined")
    pearson = np.corrcoef(values, rowvar=False)
    ranks = rankdata(values, axis=0, method="average")
    spearman = np.corrcoef(ranks, rowvar=False)
    return pearson, spearman


def json_cell(value: object) -> str:
    """Strict JSON for finite results and source metadata cells."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def write_rows(path: Path, rows: list[dict], empty_columns: list[str] | None = None) -> None:
    """Write a UTF-8-BOM CSV with stable union columns, including empty logs."""
    keys = list(dict.fromkeys(k for row in rows for k in row)) or empty_columns or []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
