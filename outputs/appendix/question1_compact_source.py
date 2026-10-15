# ============================================================
# Source: src/quality/io.py
# Original lines: 3-11
# Purpose: 真实附件读取、索引/字段兼容与输入单位
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path
from typing import Iterator

import yaml

# ============================================================
# Source: src/quality/io.py
# Original lines: 22-67
# Purpose: 真实附件读取、索引/字段兼容与输入单位
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
FIELD_ALIASES = load_field_aliases()


def infer_source_domain(path: Path, record: dict) -> tuple[str | None, str]:
    """Resolve an existing domain or infer only documented extension prefixes.

    Returns (domain, provenance rule). Conflicting metadata raises rather than
    silently choosing a domain. Does not modify the input record.
    """
    inferred = next((d for d in ("arxiv", "github") if path.name.startswith(f"{d}_")), None)
    domain = record.get("_source_domain")
    if domain is not None and not isinstance(domain, str):
        raise ValueError(f"Invalid _source_domain type in {path.name}")
    if domain:
        if inferred and domain != inferred:
            raise ValueError(f"Conflicting domain in {path.name}: {domain} vs {inferred}")
        return domain, "record._source_domain"
    if inferred:
        return inferred, "filename_prefix"
    return None, "unresolved"


def resolve_metric_value(record: dict, metric: dict) -> tuple[object, str | None]:
    """Resolve a configured official name or explicit alias without source edits."""
    aliases = metric.get("raw_aliases", FIELD_ALIASES.get(metric["name"], ()))
    names = [name for name in [metric["name"], *aliases] if name in record]
    if not names:
        return None, None
    if len(names) > 1 and any(record[k] != record[names[0]] for k in names[1:]):
        raise ValueError(f"Conflicting raw aliases for {metric['name']}: {names}")
    return record[names[0]], names[0]


def discover_quality_files(root: Path) -> dict[str, list[Path]]:
    """Discover exact A1 and every actual A2/A3 shard; fail on absent input."""
    base = root / "real_attachments" / "A_data_value"
    extended = base / "slimpajama_quality_extended"
    datasets = {
        "A1": [base / "slimpajama_quality_signal_sample.jsonl.xz"],
        "A2": sorted(extended.glob("arxiv_*.jsonl.xz")),
        "A3": sorted(extended.glob("github_*.jsonl.xz")),
    }
    for dataset, files in datasets.items():
        if not files or any(not p.is_file() for p in files):
            raise FileNotFoundError(f"{dataset}: no readable files under {base}")
    return datasets

# ============================================================
# Source: src/quality/io.py
# Original lines: 79-90
# Purpose: 真实附件读取、索引/字段兼容与输入单位
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def iter_jsonl(path: Path) -> Iterator[tuple[int, dict | None, str | None]]:
    """Yield physical line number, object or failure; never retain content."""
    with lzma.open(path, "rt", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("JSON record is not an object")
                yield line_number, record, None
            except (json.JSONDecodeError, ValueError) as exc:
                # Do not copy raw lines: a failed record could contain huge text.
                yield line_number, None, str(exc)

# ============================================================
# Source: src/quality/vector_transform.py
# Original lines: 7-23
# Purpose: 8个列表字段真实标量化与qurater分量参数
# softmax_expected_rating is auxiliary, not the final argmax rating model.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations

from numbers import Real

import numpy as np

from .io import resolve_metric_value
from .vector_diagnostics import stable_softmax


def parse_vector(value: object, expected_length: int) -> np.ndarray:
    """Validate the confirmed structure while retaining numeric NaN/Inf markers."""
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(f"Expected list length {expected_length}, observed {value!r}")
    if not all(isinstance(v, Real) and not isinstance(v, bool) for v in value):
        raise ValueError(f"Expected numeric vector elements, observed {value!r}")
    return np.asarray(value, dtype=np.float64)

# ============================================================
# Source: src/quality/vector_transform.py
# Original lines: 38-136
# Purpose: 8个列表字段真实标量化与qurater分量参数
# softmax_expected_rating is auxiliary, not the final argmax rating model.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def scalarize_record(record: dict, metrics: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Convert one record without retaining content or deleting incomplete rows.

    qurater is a temporary NaN placeholder until the chosen calibration fit.
    A legitimate qurater placeholder is not counted as a missing observation.
    Returns 22 scalar slots, 4 raw qurater components, and actual missing issues.
    """
    scalars = np.full(len(metrics), np.nan, dtype=np.float64)
    qurater_components = np.full(4, np.nan, dtype=np.float64)
    issues = []
    for position, metric in enumerate(metrics):
        name = metric["name"]
        value, raw_name = resolve_metric_value(record, metric)
        if raw_name is None or value is None:
            issues.append({"metric": name, "reason": "absent_field" if raw_name is None else "raw_none"})
            continue
        if metric["kind"] == "scalar":
            if not isinstance(value, Real) or isinstance(value, bool):
                raise ValueError(f"{name}: nonnumeric scalar {value!r}")
            if np.isfinite(value):
                scalars[position] = value
            else:
                issues.append({"metric": name, "reason": "nonfinite_raw_scalar"})
            continue
        if value == []:
            issues.append({"metric": name, "reason": "empty_list"})
            continue
        vector = parse_vector(value, metric["expected_length"])
        method = metric["scalarization"]
        if method == "component_winsor_minmax_mean":
            qurater_components[:] = vector
            qurater_components[~np.isfinite(qurater_components)] = np.nan
        if not np.isfinite(vector).all():
            issues.append({"metric": name, "reason": "nonfinite_raw_vector"})
            continue
        if method == "single_value":
            scalars[position] = vector[0]
        elif method == "argmax_scaled":
            scalars[position] = int(np.argmax(vector)) / float(metric["divisor"])
        elif method == "argmax_index":
            scalars[position] = int(np.argmax(vector))
        elif method != "component_winsor_minmax_mean":
            raise ValueError(f"{name}: unknown scalarization method {method}")
    return scalars, qurater_components, issues


def fit_qurater_parameters(components: np.ndarray, settings: dict) -> list[dict]:
    """Fit per-component P01/P99 on the supplied calibration observations only."""
    values = np.asarray(components, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("qurater requires an n×4 matrix")
    lower_q, upper_q = settings["winsor_lower_quantile"], settings["winsor_upper_quantile"]
    if not 0 <= lower_q < upper_q <= 1:
        raise ValueError("Invalid qurater quantile configuration")
    if lower_q != 0.01 or upper_q != 0.99:
        raise ValueError("Current approved qurater method is strictly P01-P99")
    parameters = []
    for k, name in enumerate(settings["component_names"]):
        finite = values[np.isfinite(values[:, k]), k]
        if not len(finite):
            raise ValueError(f"qurater component {k} entirely missing; cannot fit bounds")
        p01, p99 = np.quantile(finite, [lower_q, upper_q], method=settings["quantile_method"])
        clipped = np.clip(finite, p01, p99)
        low, high = float(clipped.min()), float(clipped.max())
        parameters.append({"component_index": k, "component_name": name, "fit_scope": settings["fit_scope"],
                           "total_count": len(values), "valid_count": len(finite),
                           "missing_count": len(values)-len(finite), "P01": float(p01), "P99": float(p99),
                           "min_before": float(finite.min()), "max_before": float(finite.max()),
                           "min_after_winsorization": low, "max_after_winsorization": high,
                           "L": low, "U": high, "clipped_lower_count": int((finite < p01).sum()),
                           "clipped_upper_count": int((finite > p99).sum()),
                           "constant_component": low == high,
                           "constant_component_value": settings["constant_component_value"],
                           "quantile_method": settings["quantile_method"]})
    return parameters


def transform_qurater(components: np.ndarray, parameters: list[dict]) -> np.ndarray:
    """Average independently scaled components; any missing component stays NaN."""
    values = np.asarray(components, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or len(parameters) != 4:
        raise ValueError("qurater needs exactly four components and fitted parameter rows")
    normalized = np.full_like(values, np.nan)
    for k, parameter in enumerate(parameters):
        if parameter["component_index"] != k:
            raise ValueError("Misordered qurater parameters")
        valid = np.isfinite(values[:, k])
        low, high = parameter["L"], parameter["U"]
        if high < low:
            raise ValueError("Reversed qurater bounds")
        if high == low:
            normalized[valid, k] = parameter["constant_component_value"]
        else:
            clipped = np.clip(values[valid, k], parameter["P01"], parameter["P99"])
            normalized[valid, k] = np.clip((clipped-low)/(high-low), 0, 1)
    complete = np.isfinite(normalized).all(axis=1)
    result = np.full(len(values), np.nan)
    result[complete] = normalized[complete].mean(axis=1)
    return result

# ============================================================
# Source: src/quality/preprocess.py
# Original lines: 3-10
# Purpose: 只拟合A1的填补/截尾参数，原样映射A2/A3
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations

from collections import Counter

import numpy as np

from .normalize import minmax_transform, moderate_utility_transform
from .vector_transform import fit_qurater_parameters, transform_qurater

# ============================================================
# Source: src/quality/preprocess.py
# Original lines: 36-180
# Purpose: 只拟合A1的填补/截尾参数，原样映射A2/A3
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def impute_matrix(matrix: np.ndarray, metadata: list[dict], model: dict):
    """Fill missing scalars using saved A1 domain medians, then A1 overall."""
    result = np.asarray(matrix, dtype=np.float64).copy()
    logs = []
    for i, j in zip(*np.where(~np.isfinite(result))):
        name = model["metric_order"][j]
        domain = metadata[i]["domain"]
        parameter = model["imputation_medians"][name]
        domain_value = parameter["domains"].get(domain)
        method = "A1_domain_median" if domain_value is not None else "A1_overall_median_fallback"
        value = parameter["overall"] if domain_value is None else domain_value
        if value is None or not np.isfinite(value):
            raise ValueError(f"No A1 imputation value for {name}")
        result[i, j] = value
        logs.append({**metadata[i], "metric": name, "missing_before": True,
                     "imputed_value": value, "imputation_method": method,
                     "fitting_dataset": "A1"})
    return result, logs


def fit_preprocessing_model(a1_matrix: np.ndarray, a1_components: np.ndarray,
                            metadata: list[dict], config: dict) -> tuple[dict, np.ndarray]:
    """Fit every threshold from A1; no other dataset is accepted as input."""
    validate_config(config)
    if not metadata or len(metadata) != len(a1_matrix) or any(m["dataset"] != "A1" for m in metadata):
        raise ValueError("Fitting input must consist exclusively of A1 records")
    metrics, settings = config["metrics"], config["preprocessing"]
    names = [m["name"] for m in metrics]
    if a1_matrix.shape != (len(metadata), 22):
        raise ValueError("Invalid A1 matrix dimensions")
    qsettings = dict(config["scalarization"]["qurater"], fit_scope="A1")
    qparameters = fit_qurater_parameters(a1_components, qsettings)
    scalars = a1_matrix.copy()
    scalars[:, names.index("qurater")] = transform_qurater(a1_components, qparameters)
    domains = sorted({m["domain"] for m in metadata})
    domain_array = np.array([m["domain"] for m in metadata])
    model = {"schema_version": 1, "fitting_dataset": "A1", "fit_record_count": len(metadata),
             "metric_order": names, "field_aliases": {m["name"]: m["raw_aliases"] for m in metrics if m.get("raw_aliases")},
             "metric_definitions": metrics, "settings": settings,
             "indicator_directions": {direction: [m["name"] for m in metrics if m["direction"] == direction]
                                      for direction in ["positive", "negative", "moderate"]},
             "classification_conversion_rules": {m["name"]: m for m in metrics if m.get("scalarization") in {"argmax_scaled", "argmax_index"}},
             "argmax_tie_policy": config["scalarization"]["argmax_tie_policy"],
             "qurater_parameters": qparameters, "imputation_medians": {},
             "winsor_thresholds": {}, "moderate_utility_thresholds": {}, "minmax_bounds": {},
             "quantile_fitting_order": "A1 scalarization -> observed-value medians -> imputation -> quantiles",
             "moderate_quantiles_source": "A1 imputed scalars before Winsorization",
             "constant_indicator_policy": {"minmax": settings["constant_minmax_value"],
                                           "moderate": "outer zero endpoints take precedence; zero-width ramps are steps"}}
    for j, name in enumerate(names):
        observed = scalars[np.isfinite(scalars[:, j]), j]
        if not len(observed):
            raise ValueError(f"{name} entirely missing in A1; cannot fit imputation")
        values = {d: scalars[(domain_array == d) & np.isfinite(scalars[:, j]), j] for d in domains}
        model["imputation_medians"][name] = {"overall": float(np.median(observed)), "overall_valid_count": len(observed),
            "domains": {d: float(np.median(v)) if len(v) else None for d, v in values.items()},
            "domain_valid_counts": {d: len(v) for d, v in values.items()}}
    filled, _ = impute_matrix(scalars, metadata, model)
    for j, metric in enumerate(metrics):
        name, values = metric["name"], filled[:, j]
        if metric.get("scalarization") in settings["bounded_scalarization_methods"]:
            if ((values < 0) | (values > 1)).any():
                raise ValueError(f"Bounded scalar {name} outside [0,1]")
            continue
        lo, hi = np.quantile(values, [settings["winsor_lower_quantile"], settings["winsor_upper_quantile"]],
                             method=settings["quantile_method"])
        clipped = np.clip(values, lo, hi)
        model["winsor_thresholds"][name] = {"P01": float(lo), "P99": float(hi),
            "min_before": float(values.min()), "max_before": float(values.max()),
            "min_after_winsorization": float(clipped.min()), "max_after_winsorization": float(clipped.max())}
        if metric["direction"] == "moderate":
            quantiles = [.01, .05, .10, .20, .50, .80, .90, .95, .99]
            data = np.quantile(values, quantiles, method=settings["quantile_method"])
            b, c = np.quantile(values, [settings["moderate_plateau_lower_quantile"], settings["moderate_plateau_upper_quantile"]],
                               method=settings["quantile_method"])
            model["moderate_utility_thresholds"][name] = {
                **{f"P{int(round(100*q)):02d}": float(v) for q, v in zip(quantiles, data)},
                "a": float(lo), "b": float(b), "c": float(c), "d": float(hi),
                "plateau_lower_quantile": settings["moderate_plateau_lower_quantile"],
                "plateau_upper_quantile": settings["moderate_plateau_upper_quantile"]}
        else:
            model["minmax_bounds"][name] = {"L": float(clipped.min()), "U": float(clipped.max()),
                                           "direction": metric["direction"], "constant": bool(clipped.max() == clipped.min())}
    return model, scalars


def apply_preprocessing(matrix: np.ndarray, components: np.ndarray, metadata: list[dict], model: dict):
    """Return Z and audits without estimating any parameters from these rows."""
    if model["fitting_dataset"] != "A1" or matrix.shape != (len(metadata), 22):
        raise ValueError("Invalid fixed A1 model or input shape")
    names, settings = model["metric_order"], model["settings"]
    scalars = matrix.copy()
    scalars[:, names.index("qurater")] = transform_qurater(components, model["qurater_parameters"])
    missing_counts = (~np.isfinite(scalars)).sum(axis=1)
    high_missing = [{**metadata[i], "missing_count": int(n), "missing_rate": float(n/22),
                     "missing_metrics": [names[j] for j in np.where(~np.isfinite(scalars[i]))[0]], "action": "retained"}
                    for i, n in enumerate(missing_counts) if n/22 > settings["high_missing_rate_threshold"]]
    filled, logs = impute_matrix(scalars, metadata, model)
    z = np.empty_like(filled)
    clips, component_clips = [], []
    for j, metric in enumerate(model["metric_definitions"]):
        name, x = metric["name"], filled[:, j]
        low_mask = np.zeros(len(x), dtype=bool)
        high_mask = low_mask.copy()
        lower = upper = None
        method = metric.get("scalarization", "raw_scalar")
        if name in model["winsor_thresholds"]:
            threshold = model["winsor_thresholds"][name]
            lower, upper = threshold["P01"], threshold["P99"]
            low_mask, high_mask = x < lower, x > upper
            clipped = np.clip(x, lower, upper)
            stage = "A1_continuous_winsor_bounds"
            if metric["direction"] == "moderate":
                p = model["moderate_utility_thresholds"][name]
                z[:, j] = moderate_utility_transform(clipped, p["a"], p["b"], p["c"], p["d"])
            else:
                p = model["minmax_bounds"][name]
                z[:, j] = minmax_transform(clipped, p["L"], p["U"], p["direction"], settings["constant_minmax_value"])
        else:
            # Seven already-bounded indicators are passed through, not re-scaled.
            z[:, j] = x
            stage = "bounded_identity_no_clipping"
            if method == "component_winsor_minmax_mean":
                stage = "any_raw_qurater_component_outside_A1_bounds"
                valid_q = np.isfinite(components).all(axis=1)
                for k, p in enumerate(model["qurater_parameters"]):
                    component_low = valid_q & (components[:, k] < p["P01"])
                    component_high = valid_q & (components[:, k] > p["P99"])
                    low_mask |= component_low
                    high_mask |= component_high
                    component_clips.append({"dataset": metadata[0]["dataset"], "component_index": k,
                        "component_name": p["component_name"], "n": len(x), "valid_count": int(valid_q.sum()),
                        "P01": p["P01"], "P99": p["P99"],
                        "clip_low_count": int(component_low.sum()), "clip_high_count": int(component_high.sum()),
                        "clip_low_rate": float(component_low.sum()/valid_q.sum()) if valid_q.any() else 0.0,
                        "clip_high_rate": float(component_high.sum()/valid_q.sum()) if valid_q.any() else 0.0,
                        "fitting_dataset": "A1"})
        clips.append({"dataset": metadata[0]["dataset"], "metric": name, "n": len(x),
            "clip_low_count": int(low_mask.sum()), "clip_high_count": int(high_mask.sum()),
            "clip_low_rate": float(low_mask.mean()), "clip_high_rate": float(high_mask.mean()),
            "clip_stage": stage, "lower_bound": lower, "upper_bound": upper,
            "fitting_dataset": "A1", "large_clip_flag": bool(max(low_mask.mean(), high_mask.mean()) >= settings["large_clip_rate_threshold"])})
    if not np.isfinite(z).all() or (z < 0).any() or (z > 1).any():
        raise ValueError("Z must have 22 finite indicators all in [0,1]")
    return z, logs, missing_counts, high_missing, clips, component_clips

# ============================================================
# Source: src/quality/normalize.py
# Original lines: 3-44
# Purpose: 正负向Min-Max及适中型梯形效用
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations

import numpy as np


def moderate_utility_transform(values, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Trapezoid with zero at outer endpoints and a unit central plateau.

    Zero-width ramps become steps. If all thresholds coincide, utility is zero:
    the documented outer-boundary rule takes precedence over the plateau.
    """
    if not np.isfinite([a, b, c, d]).all() or not a <= b <= c <= d:
        raise ValueError("Invalid moderate utility thresholds")
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Imputation must precede utility transformation")
    z = np.zeros_like(x)
    interior = (x > a) & (x < d)
    z[interior & (x >= b) & (x <= c)] = 1.0
    if b > a:
        ramp = interior & (x < b)
        z[ramp] = (x[ramp] - a) / (b - a)
    if d > c:
        ramp = interior & (x > c)
        z[ramp] = (d - x[ramp]) / (d - c)
    return np.clip(z, 0.0, 1.0)


def minmax_transform(values, lower: float, upper: float, direction: str,
                     constant_value: float = 0.5) -> np.ndarray:
    """Map finite values using fixed positive/negative bounds, then clip."""
    x = np.asarray(values, dtype=np.float64)
    if direction not in {"positive", "negative"}:
        raise ValueError("Min-Max only applies to positive/negative indicators")
    if not np.isfinite(x).all() or not np.isfinite([lower, upper]).all() or upper < lower:
        raise ValueError("Nonfinite values or reversed Min-Max bounds")
    if not 0 <= constant_value <= 1:
        raise ValueError("Constant indicator utility must be in [0,1]")
    if upper == lower:
        return np.full_like(x, constant_value)
    z = (x - lower) / (upper - lower) if direction == "positive" else (upper - x) / (upper - lower)
    return np.clip(z, 0.0, 1.0)

# ============================================================
# Source: src/quality/critic.py
# Original lines: 3-47
# Purpose: Pearson-CRITIC ddof=1及Spearman敏感性
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: src/quality/aggregate.py
# Original lines: 3-75
# Purpose: 10%截尾均值、模型记录bootstrap、分布比较
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: src/quality/aggregate.py
# Original lines: 86-103
# Purpose: 10%截尾均值、模型记录bootstrap、分布比较
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: scripts/03_build_quality_features.py
# Original lines: 3-23
# Purpose: 最终阶段实际执行入口：03_build_quality_features.py
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations

import csv
import json
import logging
import sys
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quality.io import discover_quality_files, infer_source_domain, iter_jsonl, sha256_file
from quality.preprocess import apply_preprocessing, fit_preprocessing_model, parameter_tables, validate_config
from quality.vector_diagnostics import write_rows
from quality.vector_transform import scalarize_record

LOGGER = logging.getLogger("quality.features")

# ============================================================
# Source: scripts/03_build_quality_features.py
# Original lines: 32-76
# Purpose: 最终阶段实际执行入口：03_build_quality_features.py
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def read_dataset(dataset: str, paths: list[Path], config: dict):
    """Stream official XZ files; retain numeric features and provenance, no text."""
    scalar_buffer, component_buffer = array("d"), array("d")
    metadata, issues, manifest, ids = [], [], [], set()
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        before = sha256_file(path)
        count = 0
        LOGGER.info("Reading %s: %s", dataset, relative)
        for line_number, record, error in iter_jsonl(path):
            if error:
                raise ValueError(f"{relative}:{line_number}: {error}")
            domain, domain_rule = infer_source_domain(path, record)
            if dataset == "A1" and not domain:
                LOGGER.warning("A1 domain unresolved; A1 overall median fallback will be used")
            if dataset != "A1" and not domain:
                raise ValueError(f"Cannot reliably infer external domain: {relative}")
            identity = record.get("id")
            if not isinstance(identity, str) or not identity or identity in ids:
                raise ValueError(f"Missing/duplicate ID in {dataset}, {relative}:{line_number}: {identity!r}")
            ids.add(identity)
            meta = {"id": identity, "dataset": dataset, "domain": domain or "<unknown>",
                    "sub_path": record.get("sub_path", ""), "source_file": relative,
                    "line_number": line_number}
            try:
                scalars, components, missing = scalarize_record(record, config["metrics"])
            except ValueError as exc:
                raise ValueError(f"Structure conflict at {relative}:{line_number}: {exc}") from exc
            metadata.append(meta)
            scalar_buffer.extend(scalars)
            component_buffer.extend(components)
            issues.extend({**meta, **item} for item in missing)
            count += 1
            if count % config["preprocessing"]["progress_every"] == 0:
                LOGGER.info("%s: %s records read", dataset, count)
            del record
        after = sha256_file(path)
        if before != after:
            raise RuntimeError(f"Official source bytes changed: {relative}")
        manifest.append({"dataset": dataset, "source_file": relative, "n": count,
                         "sha256_before": before, "sha256_after": after, "source_unchanged": True})
    if not metadata:
        raise ValueError(f"Empty input dataset {dataset}")
    return (np.frombuffer(scalar_buffer, dtype=np.float64).reshape(-1, 22),
            np.frombuffer(component_buffer, dtype=np.float64).reshape(-1, 4), metadata, issues, manifest)

# ============================================================
# Source: scripts/04_build_quality_score.py
# Original lines: 6-87
# Purpose: 最终阶段实际执行入口：04_build_quality_score.py
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: src/quality/conflict.py
# Original lines: 5-37
# Purpose: K=加权绝对偏离、冲突阈值和驱动组合
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: src/mixture/io.py
# Original lines: 2-54
# Purpose: 真实附件读取、索引/字段兼容与输入单位
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
import csv
from decimal import Decimal
from pathlib import Path
import numpy as np

PAIRS={
    'train_1M':('train_mixture_1m.csv','train_pile_loss_1m.csv','observed'),
    'test_1M':('test_mixture_1m.csv','test_pile_loss_1m.csv','observed'),
    'test_60M':('test_mixture_60m.csv','test_pile_loss_60m.csv','observed'),
    'test_1B':('test_mixture_1B.csv','test_pile_loss_1B.csv','observed'),
    'estimated_10B':('est_mixture_10b.csv','est_pile_loss_10b.csv','estimated'),
    'estimated_70B':('est_mixture_70b.csv','est_pile_loss_70b.csv','estimated')}


def read_table(path: Path, expected_width: int):
    """Stop for malformed fields, duplicate indices or nonfinite numeric values."""
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);columns=reader.fieldnames
        if not columns or columns[0]!='index' or len(columns)!=expected_width+1 or len(set(columns))!=len(columns):
            raise ValueError(f'Unexpected schema: {path.name}')
        records=list(reader)
    if not records or any(None in r or any(v is None for v in r.values()) for r in records):raise ValueError(f'Malformed CSV: {path}')
    indices=[r['index'] for r in records]
    if any(not i for i in indices) or len(set(indices))!=len(indices):raise ValueError(f'Duplicate/empty index: {path}')
    values=np.array([[float(r[k]) for k in columns[1:]] for r in records])
    if not np.isfinite(values).all():raise ValueError(f'Nonfinite values: {path}')
    decimals=max(max(-Decimal(r[k]).as_tuple().exponent,0) for r in records for k in columns[1:])
    return {'path':path,'columns':columns[1:],'indices':indices,'values':values,'decimal_places':decimals,'records':records}


def load_datasets(root: Path):
    """Pair all A4–A15 by index, without joining indices across experiments."""
    base=root/'real_attachments/A_data_value/regmix_tables';datasets={};canonical_p=None;canonical_l=None
    for label,(mixture,loss,provenance) in PAIRS.items():
        p=read_table(base/mixture,17);l=read_table(base/loss,13)
        if set(p['indices'])!=set(l['indices']):raise ValueError(f'Non-bijective mixture/Loss indices: {label}')
        if canonical_p is None:canonical_p,canonical_l=p['columns'],l['columns']
        if p['columns']!=canonical_p or l['columns']!=canonical_l:raise ValueError(f'Changed domain column order: {label}')
        lookup={index:i for i,index in enumerate(l['indices'])};order=[lookup[index] for index in p['indices']]
        y=l['values'][order];x=p['values']
        if (x<0).any() or (x.sum(axis=1)<=0).any():raise ValueError(f'Negative or all-zero mixture: {label}')
        if (y<0).any():raise ValueError(f'Negative cross entropy: {label}')
        datasets[label]={'p_table':p,'loss_table':l,'P':x,'losses':y,'indices':p['indices'],
            'provenance':provenance,'L_mean':y.mean(axis=1),'L_trim':np.sort(y,axis=1)[:,1:-1].mean(axis=1),
            'L_median':np.median(y,axis=1)}
    return datasets,canonical_p,canonical_l


def signatures(values):
    """Composition keys after closure, rounded only for matching exported recipes."""
    closed=values/values.sum(axis=1,keepdims=True)
    return [tuple(row) for row in np.round(closed,12)]

# ============================================================
# Source: src/mixture/composition.py
# Original lines: 2-51
# Purpose: 训练折内乘法零替换与Helmert ILR
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
import numpy as np
from scipy.linalg import helmert
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ILRTransformer(TransformerMixin,BaseEstimator):
    """Fit zero mass only on the current training fold; basis is deterministic.

    δ=min(fraction × smallest positive closed training share, 1/(2D)).
    Zeros receive δ; nonzeros are multiplied by 1−number_of_zeros×δ.
    Positive-to-positive ratios are preserved. No structural-zero semantics are
    inferred: this is a modeling regularization, not recovered positive mass.
    """
    def __init__(self,zero_fraction=0.5):self.zero_fraction=zero_fraction

    @staticmethod
    def closure(X):
        values=np.asarray(X,dtype=float)
        if values.ndim!=2 or not np.isfinite(values).all() or (values<0).any() or (values.sum(axis=1)<=0).any():
            raise ValueError('Invalid simplex input')
        return values/values.sum(axis=1,keepdims=True)

    def fit(self,X,y=None):
        closed=self.closure(X);self.n_features_in_=closed.shape[1]
        if self.n_features_in_!=17 or not 0<self.zero_fraction<1:raise ValueError('Expected 17 domains and fractional replacement')
        self.min_positive_training_=float(closed[closed>0].min())
        self.zero_delta_=min(self.zero_fraction*self.min_positive_training_,1/(2*self.n_features_in_))
        self.basis_=helmert(self.n_features_in_,full=False)
        return self

    def replace(self,X):
        """Apply fixed training δ and preserve all nonzero ratios."""
        check_is_fitted(self,'zero_delta_');closed=self.closure(X)
        if closed.shape[1]!=self.n_features_in_:raise ValueError('Changed composition width')
        zero=closed==0;count=zero.sum(axis=1)
        result=closed*(1-count*self.zero_delta_)[:,None];result[zero]=self.zero_delta_
        if (result<=0).any() or not np.allclose(result.sum(axis=1),1,atol=1e-14,rtol=0):raise ValueError('Zero replacement failed')
        return result

    def transform(self,X):
        """ILR=log(replaced P) × Hᵀ, giving exactly sixteen coordinates."""
        return np.log(self.replace(X))@self.basis_.T

    def inverse_transform(self,X):
        """Stable inverse to the replacement composition, not original zeros."""
        check_is_fitted(self,'basis_');log_values=np.asarray(X)@self.basis_
        values=np.exp(log_values-log_values.max(axis=1,keepdims=True))
        return values/values.sum(axis=1,keepdims=True)

# ============================================================
# Source: src/mixture/modeling.py
# Original lines: 2-61
# Purpose: Ridge/ElasticNet候选、训练内调参与分组CV
# M0 and second-order M2 are candidate / sensitivity models; not used as final main model. M1_Ridge is the numerical comparator; M1_ElasticNet is selected.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold,GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from scipy.stats import pearsonr,spearmanr
from .composition import ILRTransformer

MODELS=['M0','M1_Ridge','M1_ElasticNet','M2_ElasticNet']


def scores(truth,prediction):
    """Separate absolute errors, offset bias and rankings; constant ranks undefined."""
    y,p=np.asarray(truth),np.asarray(prediction);defined=len(y)>1 and np.ptp(y)>0 and np.ptp(p)>0
    return {'n':len(y),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'MAE':float(mean_absolute_error(y,p)),
        'R2':float(r2_score(y,p)),'Pearson':float(pearsonr(y,p).statistic) if defined else None,
        'Spearman':float(spearmanr(y,p).statistic) if defined else None,
        'prediction_bias_mean':float((p-y).mean()),'true_mean':float(y.mean()),'predicted_mean':float(p.mean()),
        'RMSE_after_removing_mean_error_diagnostic_only':float(np.sqrt(np.mean(((p-y)-(p-y).mean())**2)))}


def group_folds(P,n_splits,seed):
    """Keep repeated composition groups in the same fold, regardless of index."""
    from .io import signatures
    keys=signatures(P);mapping={};groups=np.array([mapping.setdefault(k,len(mapping)) for k in keys])
    if len(mapping)<n_splits:raise ValueError('Too few unique training recipes')
    return list(GroupKFold(n_splits=n_splits,shuffle=True,random_state=seed).split(P,groups=groups))


def candidate(model,settings):
    """Build ILR→(cross-products)→scaler→regressor, with fold-specific y scale."""
    if model=='M0':return DummyRegressor(strategy='mean'),{}
    steps=[('ilr',ILRTransformer(settings['zero_replacement_fraction_of_training_min']))]
    if model=='M2_ElasticNet':steps.append(('interactions',PolynomialFeatures(degree=2,include_bias=False,interaction_only=settings['polynomial_interaction_only'])))
    steps.append(('scale',StandardScaler()))
    if model=='M1_Ridge':
        steps.append(('regression',Ridge()));grid={'regressor__regression__alpha':settings['ridge_alphas']}
    else:
        steps.append(('regression',ElasticNet(max_iter=settings['elastic_max_iter'],tol=settings['elastic_tolerance'],selection='cyclic')))
        grid={'regressor__regression__alpha':settings['elastic_alphas'],'regressor__regression__l1_ratio':settings['elastic_l1_ratios']}
    return TransformedTargetRegressor(regressor=Pipeline(steps),transformer=StandardScaler()),grid


def tune(model,P,y,settings,folds):
    """Select hyperparameters only from the supplied training partition."""
    estimator,grid=candidate(model,settings)
    if not grid:return estimator.fit(P,y),{},None
    search=GridSearchCV(estimator,grid,cv=folds,scoring='neg_root_mean_squared_error',refit=True,
        n_jobs=settings['parallel_jobs'],return_train_score=True,error_score='raise')
    search.fit(P,y)
    records=[]
    for i,parameters in enumerate(search.cv_results_['params']):
        records.append({'parameters':parameters,'mean_CV_RMSE':float(-search.cv_results_['mean_test_score'][i]),
            'std_CV_RMSE':float(search.cv_results_['std_test_score'][i]),'mean_training_RMSE':float(-search.cv_results_['mean_train_score'][i]),
            'selection_rank':int(search.cv_results_['rank_test_score'][i])})
    return search.best_estimator_,search.best_params_,records

# ============================================================
# Source: scripts/08_model_mixture_loss.py
# Original lines: 2-25
# Purpose: 最终阶段实际执行入口：08_model_mixture_loss.py
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
import json
import logging
from itertools import combinations
from pathlib import Path
import sys
import warnings
import joblib
import numpy as np
import yaml
from scipy.stats import pearsonr,spearmanr
from sklearn.exceptions import ConvergenceWarning
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mixture.io import load_datasets,signatures
from mixture.composition import ILRTransformer
from mixture.modeling import MODELS,scores,group_folds,tune
from quality.io import sha256_file
from quality.vector_diagnostics import write_rows
OUT=ROOT/'outputs/mixture';FIG=ROOT/'outputs/figures/question1_part3/stage1'

# ============================================================
# Source: scripts/08_model_mixture_loss.py
# Original lines: 39-64
# Purpose: 最终阶段实际执行入口：08_model_mixture_loss.py
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def metadata(estimator,model,names,training_indices,parameters):
    """Export a complete numerical predictor for independent reconstruction."""
    result={'model':model,'fitting_recipe_indices':training_indices,'n_fit':len(training_indices),'parameters':parameters}
    if model=='M0':return {**result,'constant':float(estimator.constant_[0,0])}
    pipeline=estimator.regressor_;ilr=pipeline.named_steps['ilr'];scaler=pipeline.named_steps['scale'];regression=pipeline.named_steps['regression']
    iterations=getattr(regression,'n_iter_',None)
    if iterations is not None and int(iterations)>=regression.max_iter:raise ValueError('Selected ElasticNet did not converge')
    powers=pipeline.named_steps['interactions'].powers_ if 'interactions' in pipeline.named_steps else np.eye(16,dtype=int)
    feature_names=pipeline.named_steps['interactions'].get_feature_names_out([f'ilr_{i+1:02d}' for i in range(16)]).tolist() if 'interactions' in pipeline.named_steps else [f'ilr_{i+1:02d}' for i in range(16)]
    return {**result,'domain_order':names,'zero_delta':ilr.zero_delta_,'training_min_positive':ilr.min_positive_training_,
        'ilr_basis':ilr.basis_.tolist(),'feature_powers':powers.tolist(),'feature_names':feature_names,
        'X_mean':scaler.mean_.tolist(),'X_scale':scaler.scale_.tolist(),'coefficients':regression.coef_.tolist(),
        'intercept':float(regression.intercept_),'y_mean':float(estimator.transformer_.mean_[0]),'y_scale':float(estimator.transformer_.scale_[0]),
        'n_nonzero_coefficients':int((regression.coef_!=0).sum()),'n_model_features':len(regression.coef_),
        'n_iter':int(iterations) if iterations is not None else None}


def fit_checked(model,P,y,settings,folds,context,warning_rows):
    """Capture convergence warnings and verify selected fit before prediction."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always',ConvergenceWarning)
        estimator,parameters,grid=tune(model,P,y,settings,folds)
    for item in captured:
        warning_rows.append({**context,'category':item.category.__name__,'message':str(item.message)})
    if captured:logging.info('%s %s: captured %d warnings; selected-fit convergence checked',context.get('response'),model,len(captured))
    return estimator,parameters,grid

# ============================================================
# Source: src/mixture/effects.py
# Original lines: 2-109
# Purpose: 真实领域配比1个百分点扰动及联合边际效应
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
from itertools import combinations
import numpy as np
from .composition import ILRTransformer


def perturb(p, targets, step):
    """Add absolute share step to each target; shrink all other shares pro rata.

    A zero target may be activated. Its response includes the fitted zero-
    replacement convention, so it is not an infinitesimal derivative at zero.
    """
    p = np.asarray(p, dtype=float)
    if p.shape != (17,) or not np.isfinite(p).all() or np.any(p < 0) or abs(p.sum()-1) > 1e-12:
        raise ValueError('Expected a closed 17-domain composition')
    targets = list(targets)
    if len(set(targets)) != len(targets) or step <= 0:
        raise ValueError('Invalid perturbation targets or step')
    rest = np.ones(17, dtype=bool); rest[targets] = False
    donor = float(p[rest].sum()); mass = step*len(targets)
    if donor <= 0 or mass > donor:
        raise ValueError('Insufficient donor mass; perturbation unavailable')
    out = p.copy(); out[targets] += step; out[rest] *= (donor-mass)/donor
    if (out < 0).any() or abs(out.sum()-1) > 1e-12:
        raise ArithmeticError('Perturbation violated simplex')
    return out


def direction(values, tolerance=1e-10):
    """Strict numerical sign agreement, not statistical significance."""
    x=np.asarray(values)
    if np.all(x < -tolerance): return 'lower_loss'
    if np.all(x > tolerance): return 'higher_loss'
    if np.all(abs(x) <= tolerance): return 'numerically_zero'
    return 'uncertain'


def summary(values):
    """Equal-baseline summaries, not a population-average causal estimand."""
    x=np.asarray(values)
    return dict(mean_effect=float(x.mean()), median_effect=float(np.median(x)),
                std_effect=float(x.std(ddof=1)), min_effect=float(x.min()), max_effect=float(x.max()))


def baselines(P, indices, frozen_model):
    """A4 ILR center, closed component median, and predicted P10/P50/P90 recipes."""
    closed=ILRTransformer.closure(P)
    ilr=frozen_model.regressor_.named_steps['ilr']
    center=ilr.inverse_transform(ilr.transform(P).mean(axis=0,keepdims=True))[0]
    median=np.median(closed,axis=0)
    if median.sum()<=0: raise ValueError('Componentwise median cannot be closed')
    median/=median.sum()
    result=[('A4_ILR_center', '', center), ('A4_closed_component_median', '', median)]
    pred=frozen_model.predict(P)
    for label,q in [('low',.1),('middle',.5),('high',.9)]:
        i=int(np.argmin(abs(pred-np.quantile(pred,q))))
        result.append((f'A4_predicted_{label}',indices[i],closed[i].copy()))
    return result


def analyze(models, bases, names, steps, tolerance):
    """Evaluate every model, baseline and step for singles and joint deviations."""
    singles=[]; joint_cases=[]; marginal=[]; joints=[]; pairs=list(combinations(range(17),2))
    for model_name,model in models.items():
        for step in steps:
            for label,index,p0 in bases:
                x=np.stack([p0]+[perturb(p0,[j],step) for j in range(17)]+
                           [perturb(p0,[j,k],step) for j,k in pairs])
                pred=model.predict(x); effects=pred[1:18]-pred[0]
                for j,name in enumerate(names):
                    singles.append(dict(model=model_name,step=step,baseline=label,baseline_recipe_index=index,
                        mixture_domain=name,baseline_share=float(p0[j]),zero_activation=bool(p0[j]==0),
                        baseline_prediction=float(pred[0]),perturbed_prediction=float(pred[1+j]),effect=float(effects[j])))
                for idx,(j,k) in enumerate(pairs):
                    dj=float(pred[18+idx]-pred[0]); deviation=dj-effects[j]-effects[k]
                    joint_cases.append(dict(model=model_name,step=step,baseline=label,domain_1=names[j],domain_2=names[k],
                        joint_effect=dj,sum_single_effects=float(effects[j]+effects[k]),joint_perturbation_deviation=float(deviation)))
            for name in names:
                subset=[r for r in singles if r['model']==model_name and r['step']==step and r['mixture_domain']==name]
                marginal.append(dict(model=model_name,step=step,mixture_domain=name,n_baselines=len(bases),
                    zero_activation_baselines=sum(r['zero_activation'] for r in subset),
                    **summary([r['effect'] for r in subset]),baseline_direction=direction([r['effect'] for r in subset],tolerance)))
            for j,k in pairs:
                subset=[r for r in joint_cases if r['model']==model_name and r['step']==step and r['domain_1']==names[j] and r['domain_2']==names[k]]
                deviations=[r['joint_perturbation_deviation'] for r in subset]
                joints.append(dict(model=model_name,step=step,domain_1=names[j],domain_2=names[k],n_baselines=len(bases),
                    mean_joint_effect=float(np.mean([r['joint_effect'] for r in subset])),
                    mean_sum_single_effects=float(np.mean([r['sum_single_effects'] for r in subset])),
                    **summary(deviations),baseline_deviation_direction=direction(deviations,tolerance)))
    stability=[]
    for name in names:
        values=[r['effect'] for r in singles if r['mixture_domain']==name]
        means=[r['mean_effect'] for r in marginal if r['mixture_domain']==name]
        en=[r['effect'] for r in singles if r['mixture_domain']==name and r['model']=='M1_ElasticNet']
        ridge=[r['effect'] for r in singles if r['mixture_domain']==name and r['model']=='M1_Ridge']
        model_agreement=bool(np.all(np.sign(en)==np.sign(ridge)))
        d=direction(values,tolerance)
        stability.append(dict(mixture_domain=name,stable_direction=bool(model_agreement and d!='uncertain'),
            direction=d,models_casewise_agree=model_agreement,aggregate_direction=direction(means,tolerance),
            n_cases=len(values),lower_loss_cases=sum(v < -tolerance for v in values),higher_loss_cases=sum(v > tolerance for v in values),
            min_case_effect=float(min(values)),max_case_effect=float(max(values))))
    joint_stability=[]
    for j,k in pairs:
        values=[r['joint_perturbation_deviation'] for r in joint_cases if r['domain_1']==names[j] and r['domain_2']==names[k]]
        d=direction(values,tolerance)
        joint_stability.append(dict(domain_1=names[j],domain_2=names[k],stable_deviation=d!='uncertain',direction=d,
            mean_deviation=float(np.mean(values)),min_deviation=float(min(values)),max_deviation=float(max(values))))
    return singles,marginal,stability,joint_cases,joints,joint_stability

# ============================================================
# Source: scripts/09_analyze_mixture_effects.py
# Original lines: 6-35
# Purpose: 最终阶段实际执行入口：09_analyze_mixture_effects.py
# M3_Qp / M4_Qp_Kp are candidate / sensitivity models, not final main model. The saved final_model_policy selects M1_ElasticNet.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
import sys
import warnings
import joblib
import numpy as np
import yaml
from scipy.stats import spearmanr,rankdata,trim_mean
from sklearn.exceptions import ConvergenceWarning
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mixture.io import load_datasets,signatures
from mixture.composition import ILRTransformer
from mixture.modeling import scores,group_folds,tune as tune_base
from mixture.effects import baselines,analyze,perturb,direction,summary
from mixture.augmented import tune,candidate,metadata,checked
from quality.io import sha256_file
from quality.vector_diagnostics import write_rows

OUT=ROOT/'outputs/mixture'
STAGE=OUT/'stage2'
FIG=ROOT/'outputs/figures/question1_part3/stage2'

# ============================================================
# Source: scripts/09_analyze_mixture_effects.py
# Original lines: 102-113
# Purpose: 最终阶段实际执行入口：09_analyze_mixture_effects.py
# M3_Qp / M4_Qp_Kp are candidate / sensitivity models, not final main model. The saved final_model_policy selects M1_ElasticNet.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def scale_diagnostics(datasets,models):
    """Residual is y-p; centered RMSE is explanatory only and never a calibration."""
    rows=[]
    for model_name,model in models.items():
        for label in ['test_1M','test_60M','test_1B']:
            d=datasets[label];pred=model.predict(d['P']);e=d['L_mean']-pred
            rows.append(dict(model=model_name,dataset=label,residual_convention='observed_minus_prediction',
                mean_bias=float(e.mean()),median_bias=float(np.median(e)),residual_std=float(e.std(ddof=1)),
                centered_RMSE_diagnostic_only=float(np.sqrt(np.mean((e-e.mean())**2))),
                calibration_applied=False,**scores(d['L_mean'],pred)))
    write_rows(OUT/'scale_transfer_diagnostics.csv',rows)
    return rows

# ============================================================
# Source: scripts/09_analyze_mixture_effects.py
# Original lines: 275-306
# Purpose: 最终阶段实际执行入口：09_analyze_mixture_effects.py
# M3_Qp / M4_Qp_Kp are candidate / sensitivity models, not final main model. The saved final_model_policy selects M1_ElasticNet.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def extrapolation(datasets,models):
    """Estimated recipe subset consistency; no calibration or new external evidence."""
    train=datasets['train_1M'];lookup={key:i for i,key in enumerate(signatures(train['P']))}
    rows=[];pred_rows=[];matches={};links=[]
    for label in ['estimated_10B','estimated_70B']:
        d=datasets[label];positions=np.array([lookup[key] for key in signatures(d['P'])]);matches[label]=positions
        for name,model in models.items():
            pred=model.predict(d['P']);m=scores(d['L_mean'],pred)
            rows.append(dict(dataset=label,model=name,validation_kind='estimated_training_recipe_subset_consistency',
                calibration_applied=False,**m))
            pred_rows += [dict(dataset=label,model=name,index=index,matched_train_index=train['indices'][positions[i]],
                estimated_loss=float(d['L_mean'][i]),uncalibrated_prediction=float(pred[i]),
                estimated_rank=float(rankdata(d['L_mean'])[i]),predicted_rank=float(rankdata(pred)[i])) for i,index in enumerate(d['indices'])]
            links.append(dict(comparison=f'{name}_prediction_vs_{label}',n=len(pred),Spearman=m['Spearman']))
        y1=train['L_mean'][positions];ye=d['L_mean'];slope=np.dot(y1,ye)/np.dot(y1,y1)
        pair_ranks=[]
        for col in range(13):
            actual=train['losses'][positions,col];estimated=d['losses'][:,col]
            ratio=np.dot(actual,estimated)/np.dot(actual,actual)
            pair_ranks.append(dict(dataset=label,validation_loss=d['loss_table']['columns'][col],
                through_origin_diagnostic_ratio=float(ratio),max_abs_deviation=float(abs(estimated-ratio*actual).max()),
                observed_vs_estimated_Spearman=float(spearmanr(actual,estimated).statistic)))
        write_rows(STAGE/f'{label}_generation_relationship.csv',pair_ranks)
        links.append(dict(comparison=f'observed_train_1M_vs_{label}',n=len(ye),Spearman=float(spearmanr(y1,ye).statistic),
            diagnostic_ratio_estimated_over_1M=float(slope),diagnostic_max_abs_scaling_deviation=float(abs(ye-slope*y1).max())))
    a=datasets['estimated_10B'];b=datasets['estimated_70B'];blookup={key:i for i,key in enumerate(signatures(b['P']))}
    order=[blookup[key] for key in signatures(a['P'])]
    links.append(dict(comparison='estimated_10B_vs_estimated_70B_same_recipes',n=len(order),Spearman=float(spearmanr(a['L_mean'],b['L_mean'][order]).statistic)))
    write_rows(OUT/'estimated_scale_consistency.csv',rows)
    write_rows(OUT/'estimated_scale_predictions.csv',pred_rows)
    write_rows(OUT/'matched_recipe_rank_consistency.csv',links)
    return rows,links

# ============================================================
# Source: scripts/03_build_quality_features.py
# Original lines: 215-271
# Purpose: 最终阶段实际执行入口：03_build_quality_features.py
# Original call-site excerpt, not a standalone function.
# ============================================================
    for dataset, paths in files.items():
        matrix, components, metadata, issues, source_manifest = read_dataset(dataset, paths, config)
        if len(metadata) != baseline[dataset]:
            raise ValueError(f"Record count differs from accepted PHASE 2C: {dataset}")
        all_issues.update((r["dataset"], r["source_file"], r["line_number"], r["id"], r["metric"]) for r in issues)
        if dataset == "A1":
            model, _ = fit_preprocessing_model(matrix, components, metadata, config)
            model["fitting_inputs"] = source_manifest
            model["fitting_config_sha256"] = sha256_file(config_path)
            model["historical_pooled_role"] = config["preprocessing"]["historical_pooled_role"]
            serialized = json.dumps(model, ensure_ascii=False, indent=2, allow_nan=False)
            (output / "preprocessing_model_A1.json").write_text(serialized + "\n", encoding="utf-8")
            # All transforms, including A1, use the model reloaded from disk.
            model = json.loads((output / "preprocessing_model_A1.json").read_text(encoding="utf-8"))
            tables = parameter_tables(model)
            for filename, rows in tables.items():
                write_rows(output / filename, rows)
        model_before = json.dumps(model, sort_keys=True, allow_nan=False)
        z, imputations, missing_counts, high, dataset_clips, q_clips = apply_preprocessing(matrix, components, metadata, model)
        if model_before != json.dumps(model, sort_keys=True, allow_nan=False):
            raise RuntimeError("An external transformation mutated fitted model parameters")
        if int(missing_counts.sum()) != len(issues) or len(imputations) != len(issues):
            raise ValueError("Scalar missing and imputation provenance do not reconcile")
        reasons = {(r["source_file"], r["line_number"], r["metric"]): r["reason"] for r in issues}
        for r in imputations:
            r["missing_reason"] = reasons[(r["source_file"], r["line_number"], r["metric"])]
        all_imputations.extend(imputations)
        high_missing.extend(high)
        manifest.extend(source_manifest)
        for j, name in enumerate(model["metric_order"]):
            validations.append({"dataset": dataset, "metric": name, "n": len(z), "min": float(z[:, j].min()),
                "max": float(z[:, j].max()), "nan_count": int(np.isnan(z[:, j]).sum()),
                "inf_count": int(np.isinf(z[:, j]).sum()), "in_0_1": bool(((z[:, j] >= 0) & (z[:, j] <= 1)).all())})
        if dataset == "A1":
            a1_reference = {m["id"]: (m["domain"], row.copy()) for m, row in zip(metadata, z)
                            if m["domain"] in {r["domain"] for r in previous_cross.values()}}
        else:
            domain = previous_cross[dataset]["domain"]
            reference_ids = {identity for identity, (d, _) in a1_reference.items() if d == domain}
            actual_ids = {m["id"] for m in metadata}
            shared = reference_ids & actual_ids
            if shared != reference_ids or len(shared) != int(previous_cross[dataset]["same_id"]):
                raise ValueError("Cross-dataset overlap differs from accepted PHASE 2C")
            nonoverlap = []
            for m, row in zip(metadata, z):
                if m["id"] in shared:
                    if not np.array_equal(a1_reference[m["id"]][1], row):
                        raise ValueError(f"Same raw quality failed to yield identical Z: {m['id']}")
                else:
                    nonoverlap.append(m)
            prefix = "A2_arxiv" if dataset == "A2" else "A3_github"
            write_rows(output / f"{prefix}_nonoverlap_ids.csv", nonoverlap)
            overlap.append({"dataset": dataset, "domain": domain, "n": len(z), "overlap_ids": len(shared),
                "nonoverlap_ids": len(nonoverlap), "overlap_Z_exactly_equal": True, "records_removed": 0})
            clips.extend(dataset_clips)
            component_clips.extend(q_clips)
        filename = {"A1": "A1_quality_features.csv", "A2": "A2_arxiv_quality_features.csv", "A3": "A3_github_quality_features.csv"}[dataset]

# ============================================================
# Source: scripts/04_build_quality_score.py
# Original lines: 205-242
# Purpose: 最终阶段实际执行入口：04_build_quality_score.py
# Original call-site excerpt, not a standalone function.
# ============================================================
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

# ============================================================
# Source: scripts/08_model_mixture_loss.py
# Original lines: 180-224
# Purpose: 最终阶段实际执行入口：08_model_mixture_loss.py
# Original call-site excerpt, not a standalone function.
# ============================================================
    nested=[];oof_rows=[];fold_scores=[];grid_rows=[];final_rows=[];final_metadata=[];validation=[];predictions=[];coefficients=[];warning_rows=[]
    for response in ['L_mean','L_trim']:
        y=train[response]
        for model in MODELS:
            oof=np.full(len(P),np.nan);fold_rmse=[]
            for fold,(fit,hold) in enumerate(outer,1):
                logging.info('%s %s outer fold %d/%d',response,model,fold,len(outer))
                inner=group_folds(P[fit],settings['inner_folds'],settings['random_state']+fold)
                context={'response':response,'model':model,'phase':'outer_inner','outer_fold':fold}
                estimator,parameters,grid=fit_checked(model,P[fit],y[fit],settings,inner,context,warning_rows)
                meta=metadata(estimator,model,names,[indices[i] for i in fit],parameters)
                path=OUT/'models/outer'/f'{response}_{model}_fold{fold}.json';path.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
                prediction=estimator.predict(P[hold]);oof[hold]=prediction
                metric=scores(y[hold],prediction);fold_rmse.append(metric['RMSE']);fold_scores.append({**context,**metric,'parameters':json.dumps(parameters,sort_keys=True)})
                grid_rows += [{**context,**r,'parameters':json.dumps(r['parameters'],sort_keys=True)} for r in grid or []]
                oof_rows += [{'response':response,'model':model,'index':indices[i],'outer_fold':fold,'true_loss':float(y[i]),'predicted_loss':float(p)} for i,p in zip(hold,prediction)]
            if not np.isfinite(oof).all():raise ValueError('Incomplete nested OOF predictions')
            nested_metric=scores(y,oof);nested.append({'response':response,'model':model,**nested_metric,
                'mean_fold_RMSE':float(np.mean(fold_rmse)),'std_fold_RMSE':float(np.std(fold_rmse,ddof=1))})
            finalfolds=group_folds(P,settings['final_selection_folds'],settings['random_state']+100)
            context={'response':response,'model':model,'phase':'final_training_CV','outer_fold':None}
            estimator,parameters,grid=fit_checked(model,P,y,settings,finalfolds,context,warning_rows)
            meta=metadata(estimator,model,names,indices,parameters);meta['response']=response;final_metadata.append(meta)
            joblib.dump(estimator,OUT/'models'/f'{response}_{model}.joblib')
            fitted=scores(y,estimator.predict(P))
            final_rows.append({'response':response,'model':model,'parameters':json.dumps(parameters,sort_keys=True),
                'training_RMSE':fitted['RMSE'],'training_R2':fitted['R2'],'nested_OOF_RMSE':nested_metric['RMSE'],
                'n_model_features':meta.get('n_model_features',0),'n_nonzero_coefficients':meta.get('n_nonzero_coefficients',0)})
            grid_rows += [{**context,**r,'parameters':json.dumps(r['parameters'],sort_keys=True)} for r in grid or []]
            if model!='M0':coefficients += [{'response':response,'model':model,'feature':feature,'standardized_y_coefficient':coefficient,
                'selected_nonzero':coefficient!=0} for feature,coefficient in zip(meta['feature_names'],meta['coefficients'])]
            for d in ['test_1M','test_60M','test_1B']:
                values=datasets[d];prediction=estimator.predict(values['P'])
                if not np.isfinite(prediction).all():raise ValueError('Nonfinite direct validation prediction')
                validation.append({'response':response,'model':model,'dataset':d,**scores(values[response],prediction),
                    'negative_predicted_losses':int((prediction<0).sum()),'fit_dataset':'train_1M','test_used_for_fitting':False})
                predictions += [{'response':response,'model':model,'dataset':d,'index':index,'true_loss':float(truth),'predicted_loss':float(predicted)}
                    for index,truth,predicted in zip(values['indices'],values[response],prediction)]
            logging.info('%s %s completed; nested RMSE=%.6f, full-training RMSE=%.6f',response,model,nested_metric['RMSE'],fitted['RMSE'])
            # Checkpoint results permit recovery without re-running completed fits.
            for filename,data in [('nested_CV_summary.csv',nested),('nested_OOF_predictions.csv',oof_rows),('outer_fold_scores.csv',fold_scores),
                ('hyperparameter_search.csv',grid_rows),('final_training_summary.csv',final_rows),('model_validation.csv',validation),('test_predictions.csv',predictions)]:write_rows(OUT/filename,data)
    best=min([r for r in nested if r['response']=='L_mean'],key=lambda r:r['RMSE'])['model']
    (OUT/'model_selection.json').write_text(json.dumps({'primary_response':'L_mean','recommended_model':best,
        'recommended_model_role':'numerical_nested_CV_benchmark_not_final_main_model',

# ============================================================
# Source: scripts/09_analyze_mixture_effects.py
# Original lines: 543-562
# Purpose: 最终阶段实际执行入口：09_analyze_mixture_effects.py
# M3_Qp / M4_Qp_Kp are candidate / sensitivity models, not final main model. The saved final_model_policy selects M1_ElasticNet.
# Original call-site excerpt, not a standalone function.
# ============================================================
    # No test-based hyperparameter change: only the user's predeclared retain/reject rule.
    b1=next(r for r in validation if r['model']=='M1_ElasticNet' and r['dataset']=='test_1M')
    q1=next(r for r in validation if r['model']=='M3_Qp' and r['dataset']=='test_1M')
    fold_improvements=[m3['fold_rows'][i]['RMSE']<folds[i]['RMSE'] for i in range(5)]
    rank_improvements=[next(r['Spearman'] for r in validation if r['model']=='M3_Qp' and r['dataset']==d)>
                       next(r['Spearman'] for r in validation if r['model']=='M1_ElasticNet' and r['dataset']==d) for d in ['test_1M','test_60M','test_1B']]
    retain=bool(m3['cv']['RMSE']<base_cv['RMSE'] and q1['RMSE']<b1['RMSE'] and all(rank_improvements) and all(fold_improvements))
    selected='M3_Qp' if retain else 'M1_ElasticNet'
    policy=dict(selected_model=selected,Qp_retained=retain,Kp_retained=False,
        reason=f"M3 nested RMSE difference={m3['cv']['RMSE']-base_cv['RMSE']:+.9f}; 1M test RMSE difference={q1['RMSE']-b1['RMSE']:+.9f}; "
               f"outer RMSE improved {sum(fold_improvements)}/5; three observed scales ranking improved {sum(rank_improvements)}/3; "
               'conservative retention requires consistent outer-fold and observed-scale improvement',
        model_selection_uses_no_estimated_loss=True,hyperparameters_selected_on='A4/A5_only',
        no_test_calibration=True,Qp_mapping_assumptions_explicit=True)
    dump(STAGE/'final_model_policy.json',policy)
    protected_n=verify_protected(protected)
    # Declare selected predictor and all newly trained params frozen BEFORE estimated diagnostics.
    freeze={str(p.relative_to(ROOT)):sha256_file(p) for p in sorted((STAGE/'models').glob('*')) if p.is_file()}
    freeze.update({str(p.relative_to(ROOT)):sha256_file(p) for p in sorted((OUT/'models').glob('L_mean_M1*'))})
    dump(STAGE/'freeze_before_extrapolation.json',dict(selected_model=selected,model_files=freeze,estimated_loss_used_for_selection=False))

