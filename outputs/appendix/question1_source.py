# 问题一：论文源程序逐文件汇编
# For appendix display only; NOT a standalone merged executable.
# Source bodies, imports, formulas, branches, parameters and entry guards are unchanged.
# Use the original scripts from the project root for the original workflow; DO NOT run this listing.
# This extraction task did not rerun any model or experiment.

# ============================================================
# Source: src/quality/io.py
# Original lines: 1-90
# Purpose: 真实附件读取、索引/字段兼容与输入单位
# ============================================================
"""Read official compressed JSONL without mutating files or retaining text."""

from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path
from typing import Iterator

import yaml


def load_field_aliases() -> dict[str, tuple[str, ...]]:
    """Build FIELD_ALIASES exclusively from the project's central configuration."""
    root = Path(__file__).resolve().parents[2]
    with (root / "configs" / "quality_metrics.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return {m["name"]: tuple(m["raw_aliases"]) for m in config["metrics"] if m.get("raw_aliases")}


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


def sha256_file(path: Path) -> str:
    """Hash compressed source bytes in bounded memory for read-only verification."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
# Source: src/quality/normalize.py
# Original lines: 1-44
# Purpose: 正负向Min-Max及适中型梯形效用
# ============================================================
"""Apply saved calibration bounds; never fit on an external dataset."""

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
# Source: src/quality/vector_diagnostics.py
# Original lines: 1-65
# Purpose: 共用JSON/CSV和数值辅助；softmax不是主分类评分
# ============================================================
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

# ============================================================
# Source: src/quality/vector_transform.py
# Original lines: 1-136
# Purpose: 8个列表字段真实标量化与qurater分量参数
# softmax_expected_rating is auxiliary, not the final argmax rating model.
# ============================================================
"""User-approved list scalarization, retaining exactly 22 final indicators.

Only qurater's four internal components are Winsorized and Min-Max scaled here.
Other continuous metrics remain raw; no missing imputation or quality scoring.
"""

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


def softmax_expected_rating(logits: object) -> float:
    """Return a hypothetical 0..5 Softmax expectation for sensitivity analysis.

    This helper is never called by main scalarization or the profiling script.
    Nonfinite logits yield NaN; they are never converted with np.argmax.
    """
    values = parse_vector(logits, 6)
    if not np.isfinite(values).all():
        return float("nan")
    return float(stable_softmax(values[None, :])[0] @ np.arange(6))


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
# Original lines: 1-195
# Purpose: 只拟合A1的填补/截尾参数，原样映射A2/A3
# ============================================================
"""Fit exclusively on A1 and apply an explicit, serializable 22-feature model."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .normalize import minmax_transform, moderate_utility_transform
from .vector_transform import fit_qurater_parameters, transform_qurater


def validate_config(config: dict) -> None:
    """Reject unconfirmed semantics, dimension drift and non-A1 fitting."""
    metrics = config["metrics"]
    if len(metrics) != 22 or len({m["name"] for m in metrics}) != 22:
        raise ValueError("Exactly 22 unique official indicators required")
    if Counter(m["direction"] for m in metrics) != Counter(positive=14, negative=5, moderate=3):
        raise ValueError("Direction counts must be 14 positive, 5 negative, 3 moderate")
    if config["preprocessing"]["fitting_dataset"] != "A1":
        raise ValueError("Only A1 may fit preprocessing parameters")
    settings = config["preprocessing"]
    lo, hi = settings["winsor_lower_quantile"], settings["winsor_upper_quantile"]
    b, c = settings["moderate_plateau_lower_quantile"], settings["moderate_plateau_upper_quantile"]
    if not 0 <= lo < b <= c < hi <= 1:
        raise ValueError("Invalid outer/plateau quantile ordering")
    if config["scalarization"]["status"] != "USER_CONFIRMED" or config["scalarization"]["softmax_expectation_main_result"]:
        raise ValueError("Only confirmed argmax main rules allowed")
    if config["scalarization"]["argmax_tie_policy"] != "first_index":
        raise ValueError("Expected deterministic first-index ties")
    ad = next(m for m in metrics if m["name"] == "ad_en")
    if ad["direction"] != "positive" or ad["index_labels"] != ["has_ad", "no_ad"]:
        raise ValueError("ad_en must remain positive with 1 = no ad")


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


def parameter_tables(model: dict) -> dict[str, list[dict]]:
    """Flatten the saved model into readable A1 parameter tables."""
    imputation = []
    for metric, p in model["imputation_medians"].items():
        imputation.append({"metric": metric, "domain": "ALL", "median": p["overall"],
                           "valid_count": p["overall_valid_count"], "fitting_dataset": "A1"})
        imputation.extend({"metric": metric, "domain": d, "median": v, "valid_count": p["domain_valid_counts"][d],
                           "fitting_dataset": "A1"} for d, v in p["domains"].items())
    return {"imputation_parameters_A1.csv": imputation,
            "qurater_transform_parameters_A1.csv": model["qurater_parameters"],
            "winsor_parameters_A1.csv": [{"metric": k, "fitting_dataset": "A1", **v} for k, v in model["winsor_thresholds"].items()],
            "moderate_metric_parameters_A1.csv": [{"metric": k, "fitting_dataset": "A1", **v} for k, v in model["moderate_utility_thresholds"].items()],
            "minmax_parameters_A1.csv": [{"metric": k, "fitting_dataset": "A1", **v} for k, v in model["minmax_bounds"].items()]}

# ============================================================
# Source: scripts/03_build_quality_features.py
# Original lines: 1-315
# Purpose: 最终阶段实际执行入口：03_build_quality_features.py
# ============================================================
"""Build only 22-dimensional Z with an A1-only calibration model (CPU)."""

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


def load_csv(path: Path) -> list[dict]:
    """Read a small existing diagnostic or parameter table."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


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


def write_features(path: Path, z: np.ndarray, metadata: list[dict], counts, names: list[str]) -> None:
    """Write all original occurrences; only the 22 named columns are features."""
    columns = ["id", "dataset", "domain", "sub_path", "source_file", "line_number", "imputed_count", *names]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for meta, row, n in zip(metadata, z, counts):
            writer.writerow({**meta, "imputed_count": int(n), **dict(zip(names, map(float, row)))})
    temporary.replace(path)


def markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
    """Render a compact exact-results table without hardcoding statistics."""
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        values = []
        for col in columns:
            v = row.get(col)
            values.append("" if v is None else f"{v:.10g}" if isinstance(v, float) else str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(model, tables, validations, imputations, clips, overlap, manifest, ceilings, preserved):
    """Describe the actual A1 calibration and external transformations only."""
    report = PROJECT_ROOT / "outputs" / "report" / "phase3_preprocessing_summary.md"
    lines = ["# PHASE 3：A1 标定的统一 22 维质量向量 Z", "",
        "本阶段完成原始标量→缺失填补→异常截尾→方向统一→[0,1] 标准化。未运行 04，未计算 CRITIC 或 Q。", "",
        "## 标定范围与历史结果", "",
        "A1 是七领域主样本。全部参数只用 A1 拟合，A2/A3 仅应用保存的参数；这使抽样集与扩展集共享质量尺度，",
        "并避免扩展集规模与分布反过来改变主样本的评价尺度。", "",
        "原 outputs/profile/ 全部文件和 qurater_transform_parameters.csv 保留且哈希不变，角色是",
        "**pooled descriptive / sensitivity analysis**，不是主模型最终参数。角色另存 historical_artifact_roles.csv。", "",
        "不对任一数据集或跨数据集去重；不建立 pooled deduplicated view。A3 不同 ID 的质量字段相同记录两条都保留，",
        "因为缺少 content 无法认定文本重复。跨附件重叠记录仍存在；非重叠索引仅用于以后比较，不从 Z 删除记录。", "",
        "## 实际输入与规模", ""]
    lines += markdown_table(manifest, ["dataset", "source_file", "n", "source_unchanged"])
    lines += ["", "## 22 个指标的正式转换规则", "",
        "fineweb_edu 取唯一分量；qurater 四维使用 A1 各自 P01/P99 截尾、Min-Max 后平均；",
        "modernbert 四项 argmax/5；fluency_en argmax([not_fluent,fluent])；ad_en argmax([has_ad,no_ad])，1 为无广告。",
        "后三类已在 [0,1]，包含 qurater 在内的七个指标不再次 Min-Max，不再整体截尾。",
        "Softmax 期望只保留为辅助函数，本模型未调用。并列 argmax 取最小索引；非有限向量先作为缺失处理。", "",
        "其余 15 个连续指标（fineweb、DSIR、RPS，包括连续比例统计）使用 A1 P01/P99 Winsorization。",
        "这些 RPS 比例是统计特征，不是分类概率；分类概率/评级不参与连续截尾。",
        "其中 12 个按方向 Min-Max；三个适中指标使用梯形效用。所有指标方向只在 YAML 中定义。", ""]
    rules = []
    for metric in model["metric_definitions"]:
        name = metric["name"]
        final = "A1 Winsor + " + ("trapezoid" if metric["direction"] == "moderate" else "directional Min-Max") if name in model["winsor_thresholds"] else "bounded identity (no second Min-Max)"
        rules.append({"metric": name, "direction": metric["direction"], "scalarization": metric.get("scalarization", "raw_scalar"), "final_transform": final})
    lines += markdown_table(rules, ["metric", "direction", "scalarization", "final_transform"])
    lines += ["", "正向 14、负向 5、适中 3，严格为 22 维。辅助 ID、dataset、domain、来源、imputed_count 不参与向量。", "",
        "## 缺失填补与记录保留", "",
        f"实际填补 {len(imputations)} 个标量，涉及 {len({(r['dataset'], r['source_file'], r['line_number']) for r in imputations})} 条记录。",
        "中位数由 A1 有限观测拟合；优先 A1 同领域同指标，无法取得时回退 A1 全体该指标。",
        "A2/A3 不使用自身中位数。填充值、来源领域、ID、行号逐项记录于 imputation_log.csv。",
        "这些记录只有少量指标缺失，保留样本与其他已知指标，不删除整行；高缺失记录单独留档而不自动删除。", ""]
    lines += markdown_table(imputations, ["dataset", "domain", "id", "metric", "imputed_value", "imputation_method"])
    lines += ["", "## A1 完整拟合参数", "",
        "preprocessing_model_A1.json 保存全部参数、别名、指标顺序、官方类别顺序及配置。",
        "分位数使用 numpy 线性插值；顺序为标量化→用观测拟合中位数→填补→计算连续分位数。",
        "适中型各分位数取自填补后的 A1、连续截尾前。所有文件中的 fitting_dataset/fit_scope 都为 A1。", "",
        "### qurater 四维", ""]
    lines += markdown_table(tables["qurater_transform_parameters_A1.csv"], ["component_name", "P01", "P99", "min_after_winsorization", "max_after_winsorization"])
    lines += ["", "### 中位数（所有指标和 A1 七领域）", ""]
    lines += markdown_table(tables["imputation_parameters_A1.csv"], ["metric", "domain", "median", "valid_count"])
    lines += ["", "### 连续指标截尾", ""]
    lines += markdown_table(tables["winsor_parameters_A1.csv"], ["metric", "P01", "P99", "min_after_winsorization", "max_after_winsorization"])
    lines += ["", "### 正负向 Min-Max 范围", "",
        "正向 z=(x−L)/(U−L)，负向 z=(U−x)/(U−L)，然后 clip 到 [0,1]。",
        f"U=L 时使用配置常数 {model['settings']['constant_minmax_value']}，不制造区分度；实际触发状态见 constant 列。", ""]
    lines += markdown_table(tables["minmax_parameters_A1.csv"], ["metric", "direction", "L", "U", "constant"])
    lines += ["", "## 适中型梯形效用", "",
        "x≤a 或 x≥d 时 z=0；a<x<b 线性上升；b≤x≤c 时 z=1；c<x<d 线性下降。",
        "主模型 a=P01、b=P10、c=P90、d=P99。平台分位点由配置控制，可另行拟合 P05/P95 或 P20/P80 敏感性模型。",
        "当前不计算这些额外模型。零宽斜坡按阶跃处理，外端点 0 优先；a=b=c=d 时全为 0，避免除零。", ""]
    lines += markdown_table(tables["moderate_metric_parameters_A1.csv"], ["metric", "P01", "P05", "P10", "P20", "P50", "P80", "P90", "P95", "P99", "a", "b", "c", "d"])
    lines += ["", "## A2/A3 Clip rate", "",
        "分母为该附件全部保留记录，含按 A1 填补的记录。low/high 指方向统一前的原始指标低于/高于 A1 边界，",
        "对负向指标不代表最终 z 的低/高方向。适中指标报告原值超出外端点的比例，端点等值不计截尾。",
        "qurater 指四个原分量中至少一个越界；低/高事件可在同一条记录发生，不能直接把比例相加。",
        "四维单独统计另存 external_qurater_component_clip_rates.csv。分类指标未经再次截尾，clip rate 为 0。",
        f"large_clip_flag 的诊断阈值为配置项 {model['settings']['large_clip_rate_threshold']:.0%}，是报告提示阈值，不是伪造的拟合参数。", ""]
    lines += markdown_table(clips, ["dataset", "metric", "clip_low_rate", "clip_high_rate", "large_clip_flag", "clip_stage"])
    lines += ["", "## 分类天花板（保留 PHASE 2C 已有结果）", "",
        "以下 ceiling_rate 分母为非缺失标量，不重估分类规则。A2-arxiv 的多个分类指标存在明显顶部集中，",
        "cleanliness/reasoning/professionalism 等最高档占比接近 99%；当前仍按官方 argmax 主方案。",
        "二分类顶部占比表示分类标签比例，不是分类准确率；ad_en=1 表示无广告。", ""]
    lines += markdown_table([r for r in ceilings if r["scope"] == "dataset"], ["dataset", "metric", "valid_count", "missing", "ceiling_rate"])
    lines += ["", "## 输出验证与非重叠索引", ""]
    group_results = []
    for dataset in ["A1", "A2", "A3"]:
        group = [r for r in validations if r["dataset"] == dataset]
        group_results.append({"dataset": dataset, "n": group[0]["n"], "dimensions": len(group),
                              "NaN_count": sum(r["nan_count"] for r in group), "Inf_count": sum(r["inf_count"] for r in group),
                              "all_in_0_1": all(r["in_0_1"] for r in group)})
    lines += markdown_table(group_results, ["dataset", "n", "dimensions", "NaN_count", "Inf_count", "all_in_0_1"])
    lines += ["", "每指标实际 min/max 及取值校验见 feature_validation.csv。", ""]
    lines += markdown_table(overlap, ["dataset", "domain", "n", "overlap_ids", "nonoverlap_ids", "overlap_Z_exactly_equal", "records_removed"])
    lines += ["", "ID 索引只列非重叠 ID 与来源，不复制完整 Z；A1 抽样、扩展全量、扩展非重叠余集可供后续一致性检验。", "",
        f"已有历史结果 {len(preserved)} 个文件经 SHA-256 核验未改变，见 prior_artifacts_unchanged_phase3.csv；原始附件读前读后哈希一致。", "",
        "## 阶段局限与停止点", "",
        "Z 表示按确认方向定义的指标效用，尚无综合权重或质量分数。梯形平台内的差异被有意压平，",
        "A1 范围外值发生饱和；明显扩展截尾与分类天花板需在后续解释、敏感性分析中保留。",
        "领域中位数保留主样本尺度，但可能给缺失评级赋予小数；本次具体填补值以日志为准。",
        "依赖官方质量字段而非文本人工验证，不能把向量一致推断为文本完全一致。", "",
        "本阶段到 Z 为止。未运行 04_build_quality_score.py，未计算 CRITIC、Q 或其他题目。等待人工确认。", ""]
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Fit A1, transform every retained row, audit and stop before scoring."""
    output, inspection = PROJECT_ROOT / "outputs" / "preprocess", PROJECT_ROOT / "outputs" / "inspection"
    output.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs" / "report").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "features.log", mode="w", encoding="utf-8")])
    config_path = PROJECT_ROOT / "configs" / "quality_metrics.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    np.random.seed(config["preprocessing"]["random_state"])
    files = discover_quality_files(PROJECT_ROOT)
    historical = sorted(p for folder in [PROJECT_ROOT / "outputs" / "profile", inspection] for p in folder.rglob("*") if p.is_file())
    pooled_parameter = output / "qurater_transform_parameters.csv"
    if pooled_parameter.is_file():
        historical.append(pooled_parameter)
    before = {p: sha256_file(p) for p in historical}
    baseline = {r["scope"]: int(r["n"]) for r in load_csv(inspection / "duplicate_summary.csv")}
    previous_cross = {r["right_dataset"]: r for r in load_csv(inspection / "cross_dataset_overlap.csv")}
    ceilings = load_csv(inspection / "classification_ceiling_summary.csv")
    previous_missing = load_csv(PROJECT_ROOT / "outputs" / "profile" / "scalar_missing_records.csv")
    expected_missing = {(r["dataset"], r["source_path"], int(r["line_number"]), r["id"], r["metric"]) for r in previous_missing}
    all_imputations, validations, manifest, high_missing, clips, component_clips, overlap = [], [], [], [], [], [], []
    all_issues = set()
    model, tables, a1_reference = None, None, {}
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
        write_features(output / filename, z, metadata, missing_counts, model["metric_order"])
        LOGGER.info("%s Z complete: %s rows x 22; imputed=%s; all finite, all [0,1]", dataset, len(z), len(imputations))
        del matrix, components, metadata, z
    if all_issues != expected_missing:
        raise ValueError("Raw missing events differ from accepted 02 scalar missing ID list")
    write_rows(output / "imputation_log.csv", all_imputations)
    write_rows(output / "high_missing_samples.csv", high_missing, ["id", "dataset", "domain", "source_file", "line_number", "missing_count", "missing_rate", "missing_metrics", "action"])
    imputation_summary = []
    for dataset in files:
        for name in model["metric_order"]:
            events = [r for r in all_imputations if r["dataset"] == dataset and r["metric"] == name]
            imputation_summary.append({"dataset": dataset, "metric": name, "n": baseline[dataset], "missing_before": len(events),
                "imputed_count": len(events), "imputed_rate": len(events)/baseline[dataset],
                "imputation_method": "A1_domain_median_then_A1_overall_median", "fitting_dataset": "A1"})
    write_rows(output / "imputation_summary.csv", imputation_summary)
    write_rows(output / "external_clip_rates.csv", clips)
    write_rows(output / "external_qurater_component_clip_rates.csv", component_clips)
    write_rows(output / "feature_validation.csv", validations)
    write_rows(output / "nonoverlap_index_summary.csv", overlap)
    write_rows(output / "preprocessing_input_manifest.csv", manifest)
    preserved = []
    for path, original in before.items():
        after = sha256_file(path)
        if after != original:
            raise RuntimeError(f"An accepted historical output changed: {path}")
        preserved.append({"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256_before": original,
                          "sha256_after": after, "unchanged": True})
    write_rows(output / "prior_artifacts_unchanged_phase3.csv", preserved)
    roles = [{"path": p.relative_to(PROJECT_ROOT).as_posix(), "role": config["preprocessing"]["historical_pooled_role"],
              "main_model_parameter": False} for p in historical if p.parent == PROJECT_ROOT / "outputs" / "profile" or p == pooled_parameter]
    write_rows(output / "historical_artifact_roles.csv", roles)
    (output / "preprocessing_config_A1.yaml").write_bytes(config_path.read_bytes())
    write_report(model, tables, validations, all_imputations, clips, overlap, manifest, ceilings, preserved)
    LOGGER.info("Completed: %s records retained; %s imputations; 22 finite features per row", sum(baseline[d] for d in files), len(all_imputations))
    LOGGER.info("STOP: script 04 not run; no CRITIC, no Q; waiting for human confirmation")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOGGER.exception("Feature construction failed; dependent steps must stop")
        sys.exit(1)

# ============================================================
# Source: src/quality/critic.py
# Original lines: 1-59
# Purpose: Pearson-CRITIC ddof=1及Spearman敏感性
# ============================================================
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

# ============================================================
# Source: scripts/04a_diagnose_critic_weights.py
# Original lines: 1-255
# Purpose: 最终阶段实际执行入口：04a_diagnose_critic_weights.py
# ============================================================
"""PHASE 4A: fit and diagnose A1 CRITIC weights, then stop before any Q."""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from quality.critic import critic_weights, weight_entropy
from quality.io import sha256_file
from quality.vector_diagnostics import write_rows

LOGGER = logging.getLogger("quality.critic")


def read_a1_features(path: Path, names: list[str], expected_n: int) -> np.ndarray:
    """Read only configured A1 indicator columns; exclude every auxiliary field."""
    values, identities = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or any(name not in reader.fieldnames for name in names):
            raise ValueError("A1 feature columns missing")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Duplicate A1 column names")
        for line_number, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Truncated/invalid A1 CSV row {line_number}")
            if row["dataset"] != "A1" or not row["id"] or row["id"] in identities:
                raise ValueError(f"Non-A1 or duplicated/missing ID at row {line_number}")
            identities.add(row["id"])
            values.append([float(row[name]) for name in names])
    z = np.asarray(values, dtype=np.float64)
    if z.shape != (expected_n, 22) or not np.isfinite(z).all() or (z < 0).any() or (z > 1).any():
        raise ValueError("A1 must have expected n x 22 finite normalized features")
    return z


def metric_type(metric: dict) -> str:
    """Derive diagnostic types from saved PHASE 3 definitions, not new directions."""
    method = metric.get("scalarization")
    if method == "argmax_scaled":
        return "modernbert_ordinal"
    if method == "argmax_index":
        return "binary"
    if method == "component_winsor_minmax_mean":
        return "qurater"
    if metric["direction"] == "moderate":
        return "moderate"
    return f"continuous_{metric['direction']}"


def write_matrix(path: Path, matrix: np.ndarray, names: list[str]) -> None:
    """Write a labeled 22x22 matrix, keeping official metric order."""
    write_rows(path, [{"metric": name, **dict(zip(names, map(float, matrix[j])))} for j, name in enumerate(names)])


def table(rows: list[dict], columns: list[str]) -> list[str]:
    """Format observed values as a readable Markdown table."""
    result = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        cells = []
        for key in columns:
            v = row.get(key)
            cells.append(f"{v:.10g}" if isinstance(v, float) else "" if v is None else str(v))
        result.append("| " + " | ".join(cells) + " |")
    return result


def write_diagnostics(path, weights, negative_pairs, moderate, discrete, sensitivity, summary, settings):
    """Explain flags and concentration using actual A1 weights; no corrections."""
    ordered = sorted(weights, key=lambda r: r["weight_rank"])
    lines = ["# PHASE 4A：A1 CRITIC 权重诊断", "",
        "本阶段只拟合权重并诊断相关性，不计算任何样本 Q、领域 Q 或 A1/A2/A3 的 Q 对照。", "",
        "## 输入与计算口径", "",
        f"唯一拟合输入：outputs/preprocess/A1_quality_features.csv，实际 {summary['n']:,}×22 个 Z 值。",
        "A2/A3 未参与标准差、相关矩阵或权重拟合。指标顺序和类型取自已确认的 preprocessing_model_A1.json。",
        "PHASE 3 配置、模型、参数与输出文件读前读后 SHA-256 一致，见 phase3_artifacts_unchanged_4A.csv。", "",
        "主方案：σ_j=std(Z_j,ddof=1)，r_jk 为 Pearson，R_j=Σ_k(1−r_jk)，C_j=σ_jR_j，w_j=C_j/Σ_jC_j。",
        "保留负相关及其信息量，未取相关系数绝对值、未剔除指标、未人工调整权重。",
        "若常数列使相关系数无定义，或总信息量为零，脚本停止而不填造相关系数/权重。本次没有这类退化。", "",
        f"权重和={summary['weight_sum']:.17g}；所有权重非负={summary['all_weights_nonnegative']}。", "",
        "## 22 个 Pearson 主权重", ""]
    lines += table(ordered, ["metric", "metric_type", "mean_z", "std_z", "min_z", "max_z", "unique_value_count", "conflict_information", "information_content", "critic_weight", "weight_rank"])
    lines += ["", "## 权重集中度与标记", "",
        f"最大：{summary['max_weight_metric']}，{summary['max_weight']:.10g}；最小：{summary['min_weight_metric']}，{summary['min_weight']:.10g}。",
        f"前 5 累计={summary['top5_weight_sum']:.8%}；前 10 累计={summary['top10_weight_sum']:.8%}。",
        f"H_w=−Σw_jln(w_j)={summary['weight_entropy']:.10g}；H_norm=H_w/ln(22)={summary['normalized_weight_entropy']:.10g}。",
        "熵越接近 1 越均匀；熵与最高权重、累计权重一起解读，不以熵单独证明权重合理。", "",
        "异常标记只提示复核，不改权重。配置独立保存于 configs/critic_diagnostics.yaml，不改 PHASE 3 参数。",
        f"weight>{settings['high_weight_threshold']}；weight>{settings['equal_weight_multiplier']}×(1/22)={settings['equal_weight_multiplier']/22:.10g}；",
        f"weight<{settings['low_weight_threshold']}；std<{settings['low_std_threshold']}；unique≤{settings['low_unique_value_count_threshold']}。",
        "后两项是明示的诊断阈值，不是从附件伪造的统计结果；少取值对 ordinal/binary 是预期结构，不等于指标无效。", ""]
    flagged = [r for r in weights if any(r[k] for k in ["flag_weight_gt_0_10", "flag_weight_gt_2_equal", "flag_weight_lt_0_01", "flag_low_std", "flag_low_unique_values"])]
    lines += table(flagged, ["metric", "critic_weight", "std_z", "unique_value_count", "flag_weight_gt_0_10", "flag_weight_gt_2_equal", "flag_weight_lt_0_01", "flag_low_std", "flag_low_unique_values", "low_unique_expected_for_type"])
    lines += ["", f"标记计数：{json.dumps(summary['flag_counts'], ensure_ascii=False)}。", "",
        "主模型是否有明显集中，应结合上述数值及 >0.10、>2×等权的标记；不能把高权重解释为指标更真实或更重要。", "",
        "## 前五大与后五小", ""]
    if summary["flag_counts"]["flag_weight_gt_0_10"] == 0 and summary["flag_counts"]["flag_weight_gt_2_equal"] == 0:
        lines[-2:-2] = ["未触发两类高权重标记；结合归一化熵和累计权重，本次未见单一指标明显主导的权重集中。", ""]
    else:
        lines[-2:-2] = ["本次触发高权重标记，存在需要人工复核的集中情况；本轮仍保留主公式原权重。", ""]
    lines += table(ordered[:5] + ordered[-5:], ["metric", "critic_weight", "weight_rank"])
    lines += ["", "## 负相关指标对", "",
        f"不重复计数、不计对角线：共 {summary['negative_pair_count']} 对 Pearson r<0；",
        f"r<−0.2 有 {summary['pearson_lt_minus_0_2_count']} 对，r<−0.4 有 {summary['pearson_lt_minus_0_4_count']} 对。",
        "全部越大越好并不保证统计正相关：领域组成、指标取舍和梯形效用会影响联合分布。",
        "负相关在标准 CRITIC 中增加冲突信息，本轮如实保留；它不自动证明方向错误，也不自动证明质量语义正确。", "",
        "### Pearson 最负的前 20 对", ""]
    lines += table(negative_pairs[:20], ["metric_1", "metric_2", "pearson_r", "spearman_r"])
    lines += ["", "## 三个适中型指标：保持 P10–P90 平台", ""]
    lines += table(moderate, ["metric", "mean_z", "std_z", "z_eq_1_rate", "z_eq_0_rate", "critic_weight"])
    lines += ["", "平台会把内部原值差异压为 1，但是否导致低方差须结合实际 std 与权重；当前不改变效用函数。", "",
        "## 六个离散分类指标（A1）", ""]
    if all(row["std_z"] >= settings["low_std_threshold"] for row in moderate):
        lines[-2:-2] = ["本次三个适中型指标均未触发极低标准差标记；平台集中存在，但未发生方差退化。", ""]
    lines += table(discrete, ["metric", "unique_values", "std_z", "ceiling_rate", "critic_weight"])
    lines += ["", "所有值来自已填补的 A1 Z，ceiling_rate 分母为 A1 全部记录。PHASE 2C 的 ceiling_rate 分母为原始非缺失数，",
        "两者可能因 18 次 A1 填补略有不同；本报告不覆盖或改写 PHASE 2C 结果。",
        "σ 越小在给定 R 下会减少 CRITIC 信息量，但最终权重还受相关结构影响，不能单由天花板率推出权重。",
        "A2 的强天花板不参与本次 A1 拟合，也不会反过来压低 A1 权重。", "",
        "## Spearman 敏感性（主模型仍为 Pearson）", "",
        "Spearman 按每列平均秩处理并列值；σ 仍使用原始 Z 的样本标准差，不使用秩标准差。", "",
        f"两组权重 Pearson correlation={summary['weight_vector_pearson']:.10g}；",
        f"Spearman rank correlation={summary['weight_vector_spearman']:.10g}。",
        f"Pearson Top-5：{', '.join(summary['pearson_top5'])}。",
        f"Spearman Top-5：{', '.join(summary['spearman_top5'])}。",
        f"Top-5 集合一致={summary['top5_same_set']}，顺序一致={summary['top5_same_order']}，交集数={summary['top5_intersection_count']}。", ""]
    lines += table(sensitivity, ["metric", "pearson_critic_weight", "critic_weight", "pearson_weight_rank", "weight_rank", "weight_difference", "rank_change"])
    lines += ["", "等权表仅保存 1/22，不用于计算 Q。", "",
        "## 停止点", "",
        "权重与诊断表已生成。本轮未运行完整 04_build_quality_score.py，未计算 Q_i、Q_domain 或评分对照。",
        "等待人工确认权重后才进入 PHASE 4B。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Fit A1 weights and sensitivity diagnostics, preserve all PHASE 3 files."""
    output = PROJECT_ROOT / "outputs" / "quality"
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "critic_A1.log", mode="w", encoding="utf-8")])
    settings_path = PROJECT_ROOT / "configs" / "critic_diagnostics.yaml"
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    if settings["fitting_dataset"] != "A1" or settings["std_ddof"] != 1 or settings["main_correlation"] != "pearson" or settings["spearman_tie_method"] != "average":
        raise ValueError("Only approved A1/Pearson/ddof=1/average-tie configuration allowed")
    preprocess = PROJECT_ROOT / "outputs" / "preprocess"
    protected = sorted(p for p in preprocess.rglob("*") if p.is_file())
    protected += [PROJECT_ROOT / "configs" / "quality_metrics.yaml", PROJECT_ROOT / "outputs" / "report" / "phase3_preprocessing_summary.md"]
    before = {p: sha256_file(p) for p in protected}
    model = json.loads((preprocess / "preprocessing_model_A1.json").read_text(encoding="utf-8"))
    names, metrics = model["metric_order"], model["metric_definitions"]
    if len(names) != len(set(names)) or len(names) != 22 or [m["name"] for m in metrics] != names or model["fitting_dataset"] != "A1":
        raise ValueError("Invalid approved 22-indicator A1 model")
    input_path = preprocess / "A1_quality_features.csv"
    z = read_a1_features(input_path, names, model["fit_record_count"])
    LOGGER.info("A1 loaded: %s rows x 22 finite normalized indicators", len(z))
    main_fit = critic_weights(z, "pearson")
    sensitivity_fit = critic_weights(z, "spearman")
    weights, sensitivity, moderate, discrete, negative = [], [], [], [], []
    mean, minimum, maximum = z.mean(axis=0), z.min(axis=0), z.max(axis=0)
    for j, metric in enumerate(metrics):
        unique = np.unique(z[:, j])
        kind, w = metric_type(metric), float(main_fit["weights"][j])
        row = {"metric": names[j], "metric_type": kind, "mean_z": float(mean[j]),
            "std_z": float(main_fit["std"][j]), "min_z": float(minimum[j]), "max_z": float(maximum[j]),
            "unique_value_count": len(unique), "conflict_information": float(main_fit["conflict_information"][j]),
            "information_content": float(main_fit["information_content"][j]), "critic_weight": w,
            "weight_rank": int(main_fit["weight_rank"][j]),
            "flag_weight_gt_0_10": w > settings["high_weight_threshold"],
            "flag_weight_gt_2_equal": w > settings["equal_weight_multiplier"]/22,
            "flag_weight_lt_0_01": w < settings["low_weight_threshold"],
            "flag_low_std": bool(main_fit["std"][j] < settings["low_std_threshold"]),
            "flag_low_unique_values": len(unique) <= settings["low_unique_value_count_threshold"],
            "low_unique_expected_for_type": kind in {"modernbert_ordinal", "binary"}}
        weights.append(row)
        sensitivity.append({"metric": names[j], "metric_type": kind, "std_z": row["std_z"],
            "conflict_information": float(sensitivity_fit["conflict_information"][j]),
            "information_content": float(sensitivity_fit["information_content"][j]),
            "critic_weight": float(sensitivity_fit["weights"][j]), "weight_rank": int(sensitivity_fit["weight_rank"][j]),
            "pearson_critic_weight": w, "pearson_weight_rank": row["weight_rank"],
            "weight_difference": float(sensitivity_fit["weights"][j] - w),
            "rank_change": int(sensitivity_fit["weight_rank"][j] - row["weight_rank"])})
        if kind == "moderate":
            moderate.append({k: row[k] for k in ["metric", "mean_z", "std_z", "critic_weight"]})
            moderate[-1].update({"z_eq_1_rate": float((z[:, j] == 1).mean()), "z_eq_0_rate": float((z[:, j] == 0).mean())})
        elif kind in {"modernbert_ordinal", "binary"}:
            discrete.append({"metric": names[j], "unique_values": json.dumps(unique.tolist()), "unique_value_count": len(unique),
                "std_z": row["std_z"], "ceiling_rate": float((z[:, j] == 1).mean()), "critic_weight": w})
    for j in range(22):
        for k in range(j+1, 22):
            if main_fit["correlation"][j, k] < 0:
                negative.append({"metric_1": names[j], "metric_2": names[k],
                    "pearson_r": float(main_fit["correlation"][j, k]), "spearman_r": float(sensitivity_fit["correlation"][j, k])})
    negative.sort(key=lambda r: r["pearson_r"])
    order = np.argsort(-main_fit["weights"], kind="stable")
    sorder = np.argsort(-sensitivity_fit["weights"], kind="stable")
    top5, stop5 = [names[j] for j in order[:5]], [names[j] for j in sorder[:5]]
    entropy, normalized_entropy = weight_entropy(main_fit["weights"])
    summary = {"fitting_dataset": "A1", "n": len(z), "dimensions": 22, "std_ddof": 1,
        "weight_sum": float(main_fit["weights"].sum()), "all_weights_nonnegative": bool((main_fit["weights"] >= 0).all()),
        "max_weight_metric": names[order[0]], "max_weight": float(main_fit["weights"][order[0]]),
        "min_weight_metric": names[order[-1]], "min_weight": float(main_fit["weights"][order[-1]]),
        "top5_weight_sum": float(main_fit["weights"][order[:5]].sum()),
        "top10_weight_sum": float(main_fit["weights"][order[:10]].sum()),
        "weight_entropy": entropy, "normalized_weight_entropy": normalized_entropy,
        "negative_pair_count": len(negative), "pearson_lt_minus_0_2_count": sum(r["pearson_r"] < -.2 for r in negative),
        "pearson_lt_minus_0_4_count": sum(r["pearson_r"] < -.4 for r in negative),
        "weight_vector_pearson": float(pearsonr(main_fit["weights"], sensitivity_fit["weights"]).statistic),
        "weight_vector_spearman": float(spearmanr(main_fit["weights"], sensitivity_fit["weights"]).statistic),
        "pearson_top5": top5, "spearman_top5": stop5, "top5_same_set": set(top5) == set(stop5),
        "top5_same_order": top5 == stop5, "top5_intersection_count": len(set(top5) & set(stop5)),
        "flag_counts": {key: sum(r[key] for r in weights) for key in ["flag_weight_gt_0_10", "flag_weight_gt_2_equal", "flag_weight_lt_0_01", "flag_low_std", "flag_low_unique_values"]}}
    if abs(summary["weight_sum"] - 1) > settings["weight_sum_tolerance"]:
        raise ValueError("Weights do not sum to one")
    write_rows(output / "critic_weights_A1.csv", weights)
    write_matrix(output / "critic_pearson_correlation_A1.csv", main_fit["correlation"], names)
    write_matrix(output / "critic_spearman_correlation_A1.csv", sensitivity_fit["correlation"], names)
    write_rows(output / "negative_correlation_pairs_A1.csv", negative, ["metric_1", "metric_2", "pearson_r", "spearman_r"])
    write_rows(output / "critic_weights_spearman_sensitivity.csv", sensitivity)
    write_rows(output / "equal_weights.csv", [{"metric": name, "equal_weight": 1/22} for name in names])
    write_rows(output / "moderate_indicator_diagnostics_A1.csv", moderate)
    write_rows(output / "discrete_indicator_diagnostics_A1.csv", discrete)
    (output / "critic_diagnostics_summary_A1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    preserved = []
    for path, original in before.items():
        current = sha256_file(path)
        if original != current:
            raise RuntimeError(f"Protected PHASE 3 artifact changed: {path}")
        preserved.append({"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256_before": original, "sha256_after": current, "unchanged": True})
    write_rows(output / "phase3_artifacts_unchanged_4A.csv", preserved)
    write_rows(output / "critic_inputs_manifest.csv", [{"fitting_dataset": "A1", "source": input_path.relative_to(PROJECT_ROOT).as_posix(),
        "n": len(z), "dimensions": 22, "sha256_before": before[input_path], "sha256_after": sha256_file(input_path),
        "model_source": "outputs/preprocess/preprocessing_model_A1.json", "std_ddof": 1, "main_correlation": "pearson"}])
    (output / "critic_diagnostics_config_snapshot.yaml").write_bytes(settings_path.read_bytes())
    write_diagnostics(output / "critic_weight_diagnostics.md", weights, negative, moderate, discrete, sensitivity, summary, settings)
    LOGGER.info("Completed: 22 A1 Pearson weights, sum=%.17g, H_norm=%.9f", summary["weight_sum"], normalized_entropy)
    LOGGER.info("Negative pairs=%s, r<-0.2=%s, r<-0.4=%s", len(negative), summary["pearson_lt_minus_0_2_count"], summary["pearson_lt_minus_0_4_count"])
    LOGGER.info("STOP: weights and diagnostics only; no sample Q, no domain Q; no complete script 04 run")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        LOGGER.exception("CRITIC fit/diagnosis failed; dependent steps must stop")
        sys.exit(1)

# ============================================================
# Source: src/quality/aggregate.py
# Original lines: 1-103
# Purpose: 10%截尾均值、模型记录bootstrap、分布比较
# ============================================================
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

# ============================================================
# Source: src/quality/plotting.py
# Original lines: 1-110
# Purpose: 最终CSV/图表/报告呈现，compact省略纯绘图和排版
# ============================================================
"""CPU-only publication/export figures drawn entirely from real score outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig, folder: Path, name: str, settings: dict) -> str:
    """Save a clean PNG and close it to release memory."""
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder/name, dpi=settings["figure_dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name


def ecdf(ax, values, label: str, color: str) -> None:
    """Plot exact empirical CDF; plotting can rasterize dense segments."""
    x = np.sort(values)
    ax.plot(x, np.arange(1, len(x)+1)/len(x), label=label, color=color, linewidth=1.5)


def make_quality_figures(folder, weights, datasets, a1_domains, domain_rows,
                         remainders, equal, spearman, settings, correlation, missing_rows):
    """Generate required figures plus missingness, correlation and mean plots."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    files = []
    ordered = sorted(weights, key=lambda x: float(x["critic_weight"]))
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh([r["metric"] for r in ordered], [float(r["critic_weight"]) for r in ordered], color="#4777a8")
    ax.set(xlabel="A1 Pearson-CRITIC weight", title="Locked objective weights (22 indicators)")
    ax.axvline(1/22, color="#666666", linestyle="--", label="Equal weight = 1/22")
    ax.legend(); fig.tight_layout()
    files.append(save_figure(fig, folder, "01_critic_weights.png", settings))
    domains = [r["domain"] for r in domain_rows]
    estimate = np.array([r["Q_domain"] for r in domain_rows])
    ci = np.array([[r["ci95_low"] for r in domain_rows], [r["ci95_high"] for r in domain_rows]])
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(domains))
    ax.vlines(positions, ci[0], ci[1], color="#4777a8", linewidth=2)
    ax.scatter(positions, estimate, color="#203f60", s=35, zorder=3)
    ax.set(xticks=positions, xticklabels=domains, ylabel="10% trimmed mean Q",
           title="A1 domain quality with 95% bootstrap CI (B=2000)")
    ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=.2); fig.tight_layout()
    files.append(save_figure(fig, folder, "02_A1_domain_quality_ci.png", settings))
    groups = [datasets["A1"]["Q"][a1_domains == d] for d in domains]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(groups, tick_labels=domains, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#d7e3ef"}, medianprops={"color": "#243e59"})
    ax.set(ylabel="Sample Q", ylim=(0, 1), title="A1 domain Q distributions (outliers hidden)")
    ax.tick_params(axis="x", rotation=20); fig.tight_layout()
    files.append(save_figure(fig, folder, "03_A1_domain_Q_boxplots.png", settings))
    for dataset, domain, number in [("A2", "arxiv", 4), ("A3", "github", 5)]:
        a = datasets["A1"]["Q"][a1_domains == domain]
        b = datasets[dataset]["Q"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for values, label, color in [(a, f"A1-{domain} (n={len(a):,})", "#4777a8"),
                                     (b, f"{dataset}-full (n={len(b):,})", "#c07842")]:
            axes[0].hist(values, bins=np.linspace(0, 1, 61), density=True, histtype="step", linewidth=1.5, label=label, color=color)
            ecdf(axes[1], values, label, color)
        axes[0].set(xlabel="Q", ylabel="Density", title=f"{domain}: sample vs full extension")
        axes[1].set(xlabel="Q", ylabel="Cumulative probability", title="Empirical CDF")
        for ax in axes: ax.legend(fontsize=8)
        fig.tight_layout()
        files.append(save_figure(fig, folder, f"0{number}_{domain}_sample_full.png", settings))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, dataset, domain in zip(axes, ["A2", "A3"], ["arxiv", "github"]):
        a = datasets["A1"]["Q"][a1_domains == domain]
        b = remainders[dataset]
        ecdf(ax, a, f"A1-{domain} (n={len(a):,})", "#4777a8")
        ecdf(ax, b, f"Nonoverlap extension (n={len(b):,})", "#c07842")
        ax.set(xlabel="Q", ylabel="Cumulative probability", title=f"{domain}: disjoint record sets")
        ax.legend(fontsize=8)
    fig.tight_layout()
    files.append(save_figure(fig, folder, "06_nonoverlap_comparison.png", settings))
    rng = np.random.default_rng(settings["random_state"])
    q = datasets["A1"]["Q"]
    chosen = np.sort(rng.choice(len(q), min(len(q), settings["scatter_max_points"]), replace=False))
    for other, label, name in [(equal, "Equal-weight Q", "07_critic_vs_equal.png"),
                               (spearman, "Spearman-CRITIC Q (sensitivity)", "08_pearson_vs_spearman_Q.png")]:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(q[chosen], other[chosen], s=4, alpha=.2, color="#4777a8", rasterized=True)
        ax.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1)
        ax.set(xlabel="Pearson-CRITIC Q (main)", ylabel=label, xlim=(0, 1), ylim=(0, 1),
               title=f"A1 sensitivity scores ({len(chosen):,} points displayed)")
        fig.tight_layout(); files.append(save_figure(fig, folder, name, settings))
    names = [r["metric"] for r in weights]
    fig, ax = plt.subplots(figsize=(12, 9))
    missing = {r["metric"]: int(r["missing_before"]) for r in missing_rows if r["dataset"] == "A1"}
    ax.barh(names[::-1], [100*missing[n]/len(q) for n in names[::-1]], color="#4777a8")
    ax.set(xlabel="Missing scalars before imputation (%)", title="A1 missingness across 22 indicators")
    fig.tight_layout(); files.append(save_figure(fig, folder, "09_A1_indicator_missingness.png", settings))
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set(xticks=range(22), yticks=range(22), xticklabels=names, yticklabels=names,
           title="A1 Pearson correlation of normalized indicators")
    ax.tick_params(axis="x", labelrotation=90, labelsize=8); ax.tick_params(axis="y", labelsize=8)
    fig.colorbar(im, ax=ax, shrink=.7, label="Pearson r"); fig.tight_layout()
    files.append(save_figure(fig, folder, "10_indicator_correlation.png", settings))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(positions-.18, [r["mean_Q"] for r in domain_rows], width=.36, color="#9cb4ca", label="Mean Q")
    ax.bar(positions+.18, estimate, width=.36, color="#4777a8", label="10% trimmed mean Q")
    ax.set(xticks=positions, xticklabels=domains, ylabel="Q", title="A1 mean and trimmed mean by domain")
    ax.tick_params(axis="x", rotation=20); ax.legend(); fig.tight_layout()
    files.append(save_figure(fig, folder, "11_A1_mean_trimmed_mean.png", settings))
    return files

# ============================================================
# Source: src/quality/reporting.py
# Original lines: 1-159
# Purpose: 最终CSV/图表/报告呈现，compact省略纯绘图和排版
# ============================================================
"""Assemble the final report from actual persisted results and reviewed texts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def read_table(path: Path) -> list[dict]:
    """Read a generated audit/statistics table with no computation or fitting."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
    """Preserve output values in Markdown, escaping table separators."""
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")).replace("|", "\\|").replace("\n", " ") for k in columns) + " |")
    return lines


def write_final_report(root: Path, settings: dict) -> None:
    """Write only claims supported by existing score/audit/review files."""
    quality, preprocess = root/"outputs"/"quality", root/"outputs"/"preprocess"
    model = json.loads((preprocess/"preprocessing_model_A1.json").read_text(encoding="utf-8"))
    weight_summary = json.loads((quality/"critic_diagnostics_summary_A1.json").read_text(encoding="utf-8"))
    weights = sorted(read_table(quality/"critic_weights_A1.csv"), key=lambda r: int(r["weight_rank"]))
    dataset = read_table(quality/"Q_dataset_summary.csv")
    domains = read_table(quality/"domain_quality_summary_A1.csv")
    extended = read_table(quality/"extended_domain_quality_summary.csv")
    remainder = read_table(quality/"nonoverlap_quality_summary.csv")
    consistency = read_table(quality/"sample_extended_consistency.csv")
    equal = read_table(quality/"Q_weighting_robustness.csv")
    spearman = read_table(quality/"Q_spearman_weighting_sensitivity.csv")
    imputed = read_table(preprocess/"imputation_log.csv")
    duplicate = read_table(root/"outputs"/"inspection"/"duplicate_summary.csv")
    overlaps = read_table(root/"outputs"/"inspection"/"cross_dataset_overlap.csv")
    inputs = read_table(quality/"scoring_input_manifest.csv")
    validations = read_table(quality/"Q_validation.csv")
    lines = ["# 问题一第 1 小问：数据质量评价最终报告", "",
        "范围仅为数据质量评价。没有进行质量冲突消解、领域配比或其他问题。", "",
        "## 数据规模、真实附件与清洗流程", "",
        "官方 XZ 经 lzma 流式读取，22 维指标独立于 id/content/来源等辅助字段。",
        "结构与类别顺序人工确认后，先标量化、按 A1 中位数填补、按 A1 阈值截尾，再统一方向构造 Z。",
        "本阶段读取保存的 Z，不重新拟合预处理或 CRITIC；仅为 40 条核验文本回读原始 A1。", ""]
    lines += markdown_table(read_table(preprocess/"preprocessing_input_manifest.csv"), ["dataset", "source_file", "n", "source_unchanged"])
    lines += ["", "### 阶段 4B 输入文件与完整精度权重", ""]
    lines += markdown_table(inputs, ["role", "path", "sha256_before", "sha256_after", "unchanged"])
    lines += ["", "## 22 个指标处理与 A1 标定策略", "",
        "全部填补值、分位数、qurater 四维范围、适中型阈值与连续 Min-Max 范围只由 A1 标定；A2/A3 共用固定模型。",
        "旧 pooled profile 与 qurater_transform_parameters.csv 仅用于描述/敏感性，未删除、未作主模型参数。",
        "原始 terminal_punctution 拼写通过 FIELD_ALIASES 映射到内部 terminal_punctuation，附件未改。", ""]
    rules = []
    for metric in model["metric_definitions"]:
        name = metric["name"]
        final = "A1 P01/P99 + " + ("trapezoid" if metric["direction"] == "moderate" else "Min-Max") if name in model["winsor_thresholds"] else "already bounded; no second Min-Max"
        rules.append({"metric": name, "direction": metric["direction"], "scalarization": metric.get("scalarization", "raw scalar"), "normalization": final})
    lines += markdown_table(rules, ["metric", "direction", "scalarization", "normalization"])
    lines += ["", "fineweb 取唯一值；qurater 四语义维度独立 P01/P99 截尾、Min-Max 后平均，最终仍为一个指标；",
        "modernbert 四项按等级 0…5 argmax/5，fluency 按 [not_fluent,fluent]、ad 按 [has_ad,no_ad] 取 argmax。",
        "ad=1 表示无广告，为正向。主方案不使用 Softmax 期望。14 正向、5 负向、3 适中，严格 22 维。",
        "适中梯形外端点为 P01/P99，满分平台 P10/P90；所有阈值保持 PHASE 3 已确认值。", "",
        "## 缺失与重复记录", "",
        f"实际 {len(imputed)} 个 scalar missing 涉及 {len({(r['dataset'],r['id']) for r in imputed})} 条记录，每条仅缺 1 项。",
        "19 项均采用 A1 同领域中位数，没有删除记录，没有触发总体回退；原始 NaN 向量不直接 argmax。",
        "imputation_log.csv 可追溯填补 ID、指标、来源和填充值。", ""]
    lines += markdown_table(imputed, ["dataset", "domain", "id", "metric", "imputed_value", "imputation_method"])
    lines += markdown_table(duplicate, ["scope", "n", "exact_duplicate_groups", "duplicated_id_groups", "SAME_ID_SAME_DATA_groups", "SAME_ID_DIFFERENT_DATA_groups"])
    lines += ["", "跨附件同 ID 记录保留。A3 不同 ID 但原 22 质量字段相同的那组两条记录也保留，",
        "缺 content 不能证明文本重复。完整 exact duplicate 与质量字段一致的口径不同。", ""]
    lines += markdown_table(overlaps, ["comparison", "same_id", "same_id_same_quality_pairs", "same_id_different_quality_pairs", "content_not_comparable_pairs"])
    lines += ["", "## CRITIC 主权重与稳定性", "",
        "只用 A1：σ_j=std(Z_j,ddof=1)，r_jk=Pearson(Z_j,Z_k)，R_j=Σ_k(1−r_jk)，C_j=σ_jR_j，w_j=C_j/ΣC_j。",
        "负相关保留，不取绝对值、不人工修正。读取 critic_weights_A1.csv 完整浮点精度，未用展示时四舍五入权重。", ""]
    lines += markdown_table(weights, ["metric", "critic_weight", "weight_rank"])
    lines += ["", f"权重和={weight_summary['weight_sum']}；H_norm={weight_summary['normalized_weight_entropy']:.9f}；",
        f"前五权重累计={weight_summary['top5_weight_sum']:.6%}。无高于 0.10、低于 0.01 权重及常数退化指标。",
        f"Pearson 与 Spearman 权重相关={weight_summary['weight_vector_pearson']:.9f}，权重秩相关={weight_summary['weight_vector_spearman']:.9f}。", "",
        "## 样本级 Q 与数据集描述统计", "",
        "Q_i=Σ_{j=1}^{22}w_jz_ij。A1/A2/A3 使用同一组 A1 Pearson-CRITIC 权重。",
        "因为 z∈[0,1]、w≥0、Σw=1，Q∈[0,1]；实际写出文件逐项验证有限性和范围。主 CSV 不包含 content。", ""]
    lines += markdown_table(dataset, ["dataset", "n", "mean_Q", "std_Q", "min_Q", "P01", "P05", "P10", "P25", "median_Q", "P75", "P90", "P95", "P99", "max_Q"])
    lines += markdown_table(validations, ["dataset", "n", "min_Q", "max_Q", "nan_count", "inf_count", "all_in_0_1"])
    lines += ["", "## A1 主领域质量与 Bootstrap", "",
        "Q_domain=TrimmedMean_10%(Q_i)，每侧截去 floor(0.10n) 个值，剩余约 80% 求均值。",
        f"95% CI 针对同一个 trimmed mean：固定根种子 {settings['random_state']}、每组 {settings['bootstrap_repetitions']} 次普通完整 n-out-of-n 重采样，",
        "每次实际抽取 n 个值并精确截尾，以 bootstrap 分布的 2.5%/97.5% 分位点得到区间。",
        "没有对大扩展集缩减样本量或用正态/有效样本数近似。各组由固定顺序 SeedSequence 子流生成；",
        "bootstrap_trimmed_means.csv 保存全部复制统计量，置信区间反映所用抽样/重采样假设而非因果可靠性。", ""]
    lines += markdown_table(domains, ["domain", "n", "mean_Q", "median_Q", "std_Q", "P10", "P25", "P75", "P90", "Q_domain", "ci95_low", "ci95_high", "quality_rank"])
    overlapping = [f"{a['domain']}/{b['domain']}" for a,b in zip(domains, domains[1:])
                   if max(float(a["ci95_low"]),float(b["ci95_low"])) <= min(float(a["ci95_high"]),float(b["ci95_high"]))]
    if overlapping:
        lines += ["", "相邻排名中置信区间重叠的领域："+", ".join(overlapping)+"。点估计排序不能直接解释为前者显著优于后者，CI重叠也不等同于正式差值检验。"]
    lines += ["", "## 扩展全量与非重叠余集领域质量", ""]
    lines += markdown_table(extended + remainder, ["dataset", "domain", "view", "n", "mean_Q", "median_Q", "std_Q", "trimmed_mean_Q", "ci95_low", "ci95_high"])
    lines += ["", "非重叠索引只选择 A2 IDs−A1-arxiv IDs、A3 IDs−A1-github IDs，不删除全量质量输出，也不复制完整 Z。", "",
        "## 抽样—扩展一致性与效应量", "",
        "所有有符号差值为扩展减 A1 抽样；relative_difference 是 trimmed mean 差除以抽样 trimmed mean 的绝对值。",
        "Cliff's delta=P(Q_extended>Q_sample)−P(Q_extended<Q_sample)，并列贡献为零；正值表示扩展倾向更高。",
        "Wasserstein 距离使用 Q 原尺度，KS 衡量最大 CDF 差。不能只用 p-value 宣称分布完全相同/不同。",
        "full 扩展包含 A1 记录，两组不独立；其 KS p-value 仅作描述性参考，不能作为独立双样本推断。",
        "nonoverlap 消除共享记录，但未知的抽样设计与潜在来源相关性仍限制 iid 推断。", ""]
    lines += markdown_table(consistency, ["comparison", "view", "sample_n", "extended_n", "mean_difference", "median_difference", "trimmed_mean_difference", "relative_difference", "KS_statistic", "KS_p_value", "Wasserstein_distance", "Cliffs_delta", "shared_id_count"])
    for r in consistency:
        lines += ["", f"- {r['comparison']}：trimmed mean 差 {float(r['trimmed_mean_difference']):+.8f}（相对 {float(r['relative_difference']):+.4%}），",
            f"  KS={float(r['KS_statistic']):.6f}，Wasserstein={float(r['Wasserstein_distance']):.6f}，Cliff's delta={float(r['Cliffs_delta']):+.6f}。",
            "  这些量分别描述位置、CDF 形态和成对排序；差异必须结合 Q 的模型含义解释，不能据显著性自动判定实质质量差异。"]
    lines += ["", "## 等权与 Spearman-CRITIC 敏感性", "",
        "Q_equal=mean(Z_i)；Q_spearman 使用 PHASE 4A 已保存的 Spearman 权重，仅为敏感性，均不替代主模型。",
        "领域排名都依据各方案的 10% trimmed mean，而不是混用平均数排名。rank_change=敏感性排名−主排名。", ""]
    lines += markdown_table(equal + spearman, ["scope", "alternative", "domain", "n", "pearson_correlation", "spearman_correlation", "main_Q_domain", "alternative_Q_domain", "main_rank", "alternative_rank", "rank_change"])
    for label, rows in [("等权",equal),("Spearman-CRITIC",spearman)]:
        changed=[r for r in rows if r["scope"]=="domain" and int(r["rank_change"])!=0]
        descriptions=[f"{r['domain']} {r['main_rank']}→{r['alternative_rank']}" for r in changed]
        lines += ["", f"{label}有 {len(changed)} 个领域发生排名变化："+(", ".join(descriptions) if descriptions else "无")+"。样本评分相关性高不能自动推出领域排名完全稳定。"]
    lines += ["", "## 高低 Q 文本核验", "",
        "仅 A1 有 content。按 Q 排序选最高和最低各 20 条，遇相同 Q 用 ID 字典序稳定排序，",
        "再依据 id 和来源行号回读原件；只保存这 40 条文本到 top20/bottom20 CSV。content 没有进入评分。"]
    review_path = quality/"A1_text_review.csv"
    if review_path.is_file():
        review = read_table(review_path)
        if len(review) != 40 or len({r["id"] for r in review}) != 40:
            raise ValueError("Text review must cover exactly 40 distinct selected records")
        lines += ["", "以下为助手逐条阅读后的文本复核记录，不冒称独立人工确认。判读使用材料中注明的阅读范围；",
            "完整 content 已导出供人工复核，文体/领域差异也会影响指标，应避免把代码天然判为低质。", ""]
        lines += markdown_table(review, ["set", "rank", "id", "domain", "Q", "review_scope", "observations", "judgement", "limitations"])
        for label in ["top20", "bottom20"]:
            counts = {}
            for row in review:
                if row["set"] == label:
                    counts[row["judgement"]] = counts.get(row["judgement"], 0) + 1
            lines += ["", f"{label} 助手复核分布：{json.dumps(counts, ensure_ascii=False)}。"]
            source_counts={}
            for row in review:
                if row["set"]==label:source_counts[row["domain"]]=source_counts.get(row["domain"],0)+1
            lines += ["", f"{label} 来源领域：{json.dumps(source_counts,ensure_ascii=False)}。"]
        lines += ["", "所见高分端多为专业、连贯英文，但部分正文混入评论/推广/跨主题拼接；低分端也包含功能正常的短多语言消歧页、天文表格和结构化代码。",
            "XML样本全文件解析成功，代码样本未编译；因此低Q主要反映本指标体系对文体/语言的适配程度，不能统一当作事实错误或内容无用。",
            "本次核验提供支持及反例，未反向修正已锁定的预处理或权重。"]
    else:
        lines += ["", "状态：文本材料已导出，逐条复核尚未完成；此状态不代表已通过人工验证。"]
    lines += ["", "## 图表", ""]
    for row in read_table(quality/"quality_figure_manifest.csv"):
        lines += ["", f"![{row['figure']}](../figures/question1_part1/{row['figure']})"]
    lines += ["", "## 模型局限性与可复用文件", "",
        "1. Q 是这 22 个已确认指标的加权效用，不是训练收益或事实正确性的直接测量。CRITIC 的方差/相关结构反映统计信息，不能证明指标语义有效。",
        "2. A1 统一尺度使扩展集可比，也造成范围外截尾饱和；A2 的 DSIR/qurater 等大量截尾，以及分类天花板仍然存在。",
        "3. P10–P90 平台压平内部差异，领域中位数填补会降低局部不确定性；当前未改变这些经确认的主方案。",
        "4. 跨附件共享记录不应当作额外独立证据；非重叠对照改善这一点，仍不能证明真实训练数据总体代表性。",
        "5. bootstrap 区间以当前记录独立重采样为基础，未建模文档来源聚类；显著性不能替代效应量与实际意义。",
        "6. 高低各 20 条是有选择的极端材料，助手文本判读不是盲评或独立人工确认；不用于反向修正本次权重。", "",
        "后续可直接复用 A1_sample_quality.csv、A2_arxiv_quality.csv、A3_github_quality.csv、A1 领域摘要、",
        "固定 A1 模型/权重、非重叠 ID 索引和敏感性评分。但本阶段没有开始质量冲突消解、领域配比或其他问题。", ""]
    (root/"outputs"/"report"/"question1_part1_final.md").write_text("\n".join(lines), encoding="utf-8")

# ============================================================
# Source: scripts/04_build_quality_score.py
# Original lines: 1-277
# Purpose: 最终阶段实际执行入口：04_build_quality_score.py
# ============================================================
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

# ============================================================
# Source: src/quality/conflict.py
# Original lines: 1-115
# Purpose: K=加权绝对偏离、冲突阈值和驱动组合
# ============================================================
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

# ============================================================
# Source: src/quality/conflict_plotting.py
# Original lines: 1-57
# Purpose: 最终CSV/图表/报告呈现，compact省略纯绘图和排版
# ============================================================
"""Real conflict-diagnostic figures, separate from all locked part 1 figures."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_conflict_figures(folder,datasets,domains,domain_rows,pairs,thresholds,settings):
    """Draw all requested diagnostics, with plotting subsamples clearly labeled."""
    folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    files=[]
    def save(fig,name):
        fig.tight_layout();fig.savefig(folder/name,dpi=settings["figure_dpi"],bbox_inches="tight",facecolor="white");plt.close(fig);files.append(name)
    a1=datasets["A1"]
    fig,ax=plt.subplots(figsize=(8,4))
    ax.hist(a1["K"],bins=60,color="#4777a8",alpha=.8)
    for name,color in [("P50","#666666"),("P90","#c07842"),("P95","#9b3b36")]:ax.axvline(thresholds[name],color=color,linestyle="--",label=f"A1 {name}={thresholds[name]:.4f}")
    ax.set(xlabel="Weighted disagreement K",ylabel="Record count",title="A1 within-record conflict distribution");ax.legend();save(fig,"01_A1_K_distribution.png")
    order=[r["domain"] for r in domain_rows]
    fig,ax=plt.subplots(figsize=(10,5))
    ax.boxplot([a1["K"][domains==d] for d in order],tick_labels=order,showfliers=False,patch_artist=True,boxprops={"facecolor":"#d7e3ef"})
    ax.axhline(thresholds["P90"],linestyle="--",color="#c07842",label="A1 global P90")
    ax.set(ylabel="K",title="A1 domains: K (outliers hidden)");ax.tick_params(axis="x",rotation=20);ax.legend();save(fig,"02_domain_K_boxplots.png")
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(order))
    rate=np.array([r["high_conflict_rate"] for r in domain_rows])
    low=np.array([r["high_rate_wilson95_low"] for r in domain_rows]);high=np.array([r["high_rate_wilson95_high"] for r in domain_rows])
    ax.bar(x,100*rate,color="#4777a8");ax.errorbar(x,100*rate,yerr=100*np.array([rate-low,high-rate]),fmt="none",color="#333333",capsize=3)
    ax.set(xticks=x,xticklabels=order,ylabel="K > A1 global P90 (%)",title="A1 high-conflict rates (95% Wilson intervals)")
    ax.tick_params(axis="x",rotation=20);save(fig,"03_domain_high_conflict_rates.png")
    names=a1["names"];matrix=np.zeros((22,22))
    for r in pairs:
        j,k=names.index(r["metric_1"]),names.index(r["metric_2"]);matrix[j,k]=matrix[k,j]=r["severe_pair_conflict_rate"]
    fig,ax=plt.subplots(figsize=(14,12));im=ax.imshow(matrix,vmin=0,vmax=1,cmap="YlOrRd")
    ax.set(xticks=range(22),yticks=range(22),xticklabels=names,yticklabels=names,title="A1 pair disagreement: P(|z_j-z_k| >= 0.5)")
    ax.tick_params(axis="x",labelrotation=90,labelsize=8);ax.tick_params(axis="y",labelsize=8);fig.colorbar(im,ax=ax,shrink=.7,label="Severe pair rate");save(fig,"04_pair_conflict_heatmap.png")
    ranked=sorted(pairs,key=lambda r:(-r["severe_pair_conflict_rate"],-r["mean_abs_difference"]))[:20][::-1]
    fig,ax=plt.subplots(figsize=(14,9));ax.barh([f"{r['metric_1']} vs {r['metric_2']}" for r in ranked],[100*r["severe_pair_conflict_rate"] for r in ranked],color="#4777a8")
    ax.set(xlabel="Severe pair conflict rate (%)",title="A1 top 20 indicator pairs (rate, then mean difference)");save(fig,"05_top20_pair_conflicts.png")
    rng=np.random.default_rng(settings["random_state"]);index=np.sort(rng.choice(len(a1["Q"]),min(settings["scatter_max_points"],len(a1["Q"])),replace=False))
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(a1["Q"][index],a1["K"][index],s=4,alpha=.2,color="#4777a8")
    grid=np.linspace(0,1,300);ax.plot(grid,2*grid*(1-grid),color="#666666",linestyle="--",label="Mathematical bound K <= 2Q(1-Q)")
    ax.axhline(thresholds["P90"],color="#c07842",linestyle="--",label="A1 P90");ax.set(xlabel="Locked Q",ylabel="K",title=f"A1 Q vs K ({len(index):,} points shown)",xlim=(0,1),ylim=(0,.52));ax.legend(fontsize=8);save(fig,"06_Q_vs_K.png")
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(a1["K"][index],a1["Delta_Q"][index],s=4,alpha=.2,color="#4777a8")
    ax.axvline(thresholds["P90"],color="#c07842",linestyle="--",label="A1 P90");ax.set(xlabel="K",ylabel="|Q_CRITIC - Q_equal|",title=f"A1 disagreement vs weighting sensitivity ({len(index):,} shown)");ax.legend();save(fig,"07_K_vs_weighting_sensitivity.png")
    for dataset,domain,number in [("A2","arxiv",8),("A3","github",9)]:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        groups=[(a1["K"][domains==domain],f"A1-{domain}","#4777a8"),(datasets[dataset]["K"],f"{dataset}-full","#c07842"),(datasets[dataset]["K"][datasets[dataset]["nonoverlap_mask"]],f"{dataset}-nonoverlap","#41927d")]
        for values,label,color in groups:
            axes[0].hist(values,bins=np.linspace(0,.5,61),density=True,histtype="step",label=f"{label} (n={len(values):,})",color=color,linewidth=1.3)
            ordered=np.sort(values);axes[1].plot(ordered,np.arange(1,len(values)+1)/len(values),label=label,color=color,linewidth=1.3)
        axes[0].set(xlabel="K",ylabel="Density",title=f"{domain}: sample / full / nonoverlap")
        axes[1].set(xlabel="K",ylabel="Cumulative probability",title="Empirical CDF")
        for ax in axes:ax.axvline(thresholds["P90"],color="#777777",linestyle="--",linewidth=.8);ax.legend(fontsize=8)
        save(fig,f"0{number}_{domain}_sample_extended_K.png")
    return files

# ============================================================
# Source: scripts/05_analyze_quality_conflict.py
# Original lines: 1-462
# Purpose: 最终阶段实际执行入口：05_analyze_quality_conflict.py
# ============================================================
"""Question 1 part 2 stage 1: diagnose disagreement using frozen part 1.

No preprocessing, CRITIC fitting, Q adjustment or lambda model is invoked.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import sys

import numpy as np
import yaml
from scipy.stats import ks_2samp, wasserstein_distance, pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quality.conflict import (conflict_indices, fit_conflict_thresholds,
    conflict_profile, pair_conflict_summary, deviation_driver_summary)
from quality.conflict_plotting import make_conflict_figures
from quality.io import iter_jsonl, sha256_file
from quality.vector_diagnostics import write_rows

OUT = ROOT / "outputs" / "conflict"
META = ["id", "dataset", "domain", "sub_path", "source_file", "line_number", "imputed_count"]
INPUTS = {"A1": "A1_sample_quality.csv", "A2": "A2_arxiv_quality.csv", "A3": "A3_github_quality.csv"}


def read_csv(path):
    """Read a small table, never the content-bearing original dataset."""
    csv.field_size_limit(2**31-1)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def protected_files():
    """Snapshot all finished part 1 artifacts and official attachment bytes."""
    files = []
    for relative in ["outputs/preprocess", "outputs/quality", "outputs/profile", "outputs/inspection",
                     "outputs/figures/question1_part1", "real_attachments"]:
        files.extend(p for p in (ROOT / relative).rglob("*") if p.is_file())
    files.extend(ROOT / "configs" / n for n in ["quality_metrics.yaml", "critic_diagnostics.yaml", "quality_scoring.yaml"])
    files.extend(p for p in (ROOT / "outputs/report").glob("*") if p.is_file() and not p.name.startswith("question1_part2"))
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(set(files))}


def load_scores(dataset, names, expected_n):
    """Read compact numerical arrays; keep metadata only, never original content."""
    values = np.empty((expected_n, 22)); q = np.empty(expected_n); metadata = []
    path = ROOT / "outputs/quality" / INPUTS[dataset]
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != set(META + ["Q"] + names):
            raise ValueError(f"Changed score schema: {path}")
        for i, row in enumerate(reader):
            if i >= expected_n or row["dataset"] != dataset:
                raise ValueError(f"Changed record scope: {path}")
            values[i] = [float(row[n]) for n in names]; q[i] = float(row["Q"])
            metadata.append({n: row[n] for n in META})
    if len(metadata) != expected_n or len({r['id'] for r in metadata}) != expected_n:
        raise ValueError(f"Unexpected rows/within-dataset duplicated IDs: {path}")
    return {"Z": values, "Q": q, "metadata": metadata, "names": names}


def safe_corr(a, b, method="pearson"):
    """Undefined correlations remain empty, especially tiny/constant strata."""
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float((pearsonr if method == "pearson" else spearmanr)(a, b).statistic)


def cliffs_delta(a, b):
    """Signed effect size P(a>b)-P(a<b), with exact tie handling."""
    ordered = np.sort(b)
    lower = np.searchsorted(ordered, a, side="left")
    upper = len(b) - np.searchsorted(ordered, a, side="right")
    return float((lower.sum(dtype=np.int64) - upper.sum(dtype=np.int64)) / (len(a) * len(b)))


def save_samples(dataset, data, thresholds):
    """Write separate conflict outputs, preserving frozen Q/Z and every record."""
    filename = {"A1":"A1_sample_conflict.csv", "A2":"A2_arxiv_conflict.csv", "A3":"A3_github_conflict.csv"}[dataset]
    keys = META + ["Q", "K", "K_equal", "V", "Q_equal", "Delta_Q", "conflict_level", "high_conflict", "extreme_conflict"] + data["names"]
    with (OUT / filename).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        for i, meta in enumerate(data["metadata"]):
            k = data["K"][i]
            level = "low" if k <= thresholds["P50"] else "medium" if k <= thresholds["P90"] else "high"
            writer.writerow({**meta, **{n: float(data[n][i]) for n in ["Q", "K", "K_equal", "V", "Q_equal", "Delta_Q"]},
                "conflict_level": level, "high_conflict": k > thresholds["P90"], "extreme_conflict": k > thresholds["P95"],
                **dict(zip(data["names"], data["Z"][i]))})


def relationship(data, mask, meta):
    """Describe association with locked Q, alternative dispersion and weighting sensitivity."""
    k, q, delta = [data[n][mask] for n in ["K", "Q", "Delta_Q"]]
    return {**meta, "n": len(k), "pearson_Q_K": safe_corr(q,k), "spearman_Q_K": safe_corr(q,k,"spearman"),
        "pearson_K_Delta_Q": safe_corr(k,delta), "spearman_K_Delta_Q": safe_corr(k,delta,"spearman"),
        "pearson_K_K_equal": safe_corr(k,data["K_equal"][mask]),
        "spearman_K_K_equal": safe_corr(k,data["K_equal"][mask],"spearman"),
        "pearson_K_V": safe_corr(k,data["V"][mask]), "spearman_K_V": safe_corr(k,data["V"][mask],"spearman"),
        "mean_Delta_Q": float(delta.mean()), "median_Delta_Q": float(np.median(delta))}


def sensitivity_comparison(data, mask, high, meta):
    """Compare weighting differences high versus ordinary; no causal/reliability claim."""
    a, b = data["Delta_Q"][mask & high], data["Delta_Q"][mask & ~high]
    if not len(a) or not len(b):
        return {**meta,"n_high":len(a),"n_ordinary":len(b),"comparison_defined":False}
    return {**meta,"n_high":len(a),"n_ordinary":len(b),"comparison_defined":True,
        "mean_Delta_Q_high":float(a.mean()),"mean_Delta_Q_ordinary":float(b.mean()),
        "median_Delta_Q_high":float(np.median(a)),"median_Delta_Q_ordinary":float(np.median(b)),
        "mean_difference_high_minus_ordinary":float(a.mean()-b.mean()),
        "mean_ratio_high_over_ordinary":float(a.mean()/b.mean()) if b.mean()>0 else None,
        "cliffs_delta_high_vs_ordinary":cliffs_delta(a,b),"KS_statistic":float(ks_2samp(a,b).statistic)}


def select_texts(a1, weights, thresholds, settings, manifest):
    """Retrieve selected A1 contents once; disclose conditional Q bands and scarce global groups."""
    q,k,z=a1["Q"],a1["K"],a1["Z"];high=k>thresholds["P90"]
    qlo,qhi=np.quantile(q[high],settings["conditional_Q_band_quantiles"],method=settings["quantile_method"])
    qmedian=float(np.quantile(q,settings["quadrant_Q_boundary_quantile"])); low=k<=thresholds["P50"]
    bands={"top_K_30":(np.ones(len(k),dtype=bool),settings["text_top_K_count"],"All A1; largest K"),
        "highK_relative_lowQ":(high & (q<=qlo),settings["text_per_conditional_Q_band"],f"K>A1P90; Q<=highK Q P33={qlo}"),
        "highK_relative_middleQ":(high & (q>qlo) & (q<=qhi),settings["text_per_conditional_Q_band"],f"K>A1P90; {qlo}<Q<=highK Q P67={qhi}"),
        "highK_relative_highQ":(high & (q>qhi),settings["text_per_conditional_Q_band"],f"K>A1P90; Q>highK Q P67={qhi}; NOT globally high Q")}
    for label,mask in {"global_highQ_lowK":(q>qmedian)&low,"global_highQ_highK":(q>qmedian)&high,
                       "global_lowQ_lowK":(q<=qmedian)&low,"global_lowQ_highK":(q<=qmedian)&high}.items():
        bands[label]=(mask,settings["text_per_quadrant_max"],f"A1 Q median={qmedian}; low K<=P50, high K>P90; middle K excluded")
    selection={}; rows=[]
    for label,(mask,requested,definition) in bands.items():
        eligible=np.flatnonzero(mask)
        order=sorted(eligible,key=lambda i:(-k[i],a1["metadata"][i]["id"]))[:requested]
        rows.append({"selection_group":label,"eligible_n":len(eligible),"requested_n":requested,"selected_n":len(order),"definition":definition})
        for i in order:selection.setdefault(int(i),[]).append(label)
    write_rows(OUT/"text_selection_summary.csv",rows)
    selected={a1["metadata"][i]["id"]:i for i in selection}; texts={}
    source_rows=[r for r in manifest if r["dataset"]=="A1"]
    for entry in source_rows:
        path=ROOT/entry["source_file"]
        if sha256_file(path)!=entry["sha256_before"]:raise ValueError("A1 source differs from frozen preprocessing manifest")
        n=0
        for line,record,error in iter_jsonl(path):
            n+=1
            if error:raise ValueError(f"Cannot recover text at {path.name}:{line}: {error}")
            record_id=record.get("id")
            if record_id in selected:
                i=selected[record_id]
                if str(line)!=a1["metadata"][i]["line_number"]:raise ValueError("Selected text row mismatch")
                content=record.get("content")
                if not isinstance(content,str):raise ValueError("Selected A1 content unavailable")
                texts[record_id]=content
        if n!=int(entry["n"]):raise ValueError("Changed original A1 record count")
    if set(texts)!=set(selected):raise ValueError("Selected A1 texts not fully recovered")
    text_rows=[]
    for i in sorted(selection,key=lambda i:(-k[i],a1["metadata"][i]["id"])):
        pos=np.flatnonzero(z[i]==z[i].max());neg=np.flatnonzero(z[i]==z[i].min())
        p=min(pos,key=lambda j:(-weights[j],j));m=min(neg,key=lambda j:(-weights[j],j))
        text_rows.append({**a1["metadata"][i],"Q":float(q[i]),"K":float(k[i]),"selection_groups":";".join(selection[i]),
            "top_positive_metric":a1["names"][p],"top_positive_value":float(z[i,p]),"top_positive_deviation":float(z[i,p]-q[i]),
            "top_negative_metric":a1["names"][m],"top_negative_value":float(z[i,m]),"top_negative_deviation":float(z[i,m]-q[i]),
            "all_tied_positive_metrics":";".join(a1["names"][j] for j in pos),
            "all_tied_negative_metrics":";".join(a1["names"][j] for j in neg),
            "representative_tie_rule":settings["text_representative_tie_rule"],"review_status":"待人工核验; no AI annotation used as evidence",
            "content":texts[a1["metadata"][i]["id"]]})
    write_rows(OUT/"A1_high_conflict_text_samples.csv",text_rows)
    return rows


def markdown_table(rows, columns):
    """Render actual numeric output as compact Markdown, without invented findings."""
    def cell(v):
        if v is None:return "未定义"
        if isinstance(v,(float,np.floating)):return f"{v:.6f}"
        return str(v).replace("|","\\|").replace("\n"," ")
    return "\n".join(["| "+" | ".join(columns)+" |","| "+" | ".join("---" for _ in columns)+" |"]+
        ["| "+" | ".join(cell(row.get(c)) for c in columns)+" |" for row in rows])


def build_report(context):
    """Write diagnostics and limitations; leave Q+K/Q* choice pending human confirmation."""
    c=context;top=c["top_pairs"][:10];a1=c["profiles"][0];rel=c["relationships"][0]
    sensitivities=c["sensitivity"];driver=c["drivers_global"][:10]
    report=["# 问题一第2小问：冲突诊断第一阶段（未构造 Q*）",
        "第一小问全部锁定。直接读取三份质量 CSV 内的原有 Q 和 22 个 Z；权重来自原有 `critic_weights_A1.csv`。未重新拟合预处理、CRITIC、领域质量或评分参数。新输出全部独立保存。",
        "## 定义与边界",
        "主指标 Kᵢ=Σⱼwⱼ|zᵢⱼ−Qᵢ|；辅助 K_equal=mean|Z−mean Z|、Vᵢ=Σⱼwⱼ(zᵢⱼ−Qᵢ)²。Q 使用已保存的第一小问结果；重新求和仅校验输入一致性。",
        "K 衡量不同评价构念的内部离散，不能自动解释为 Q 错误、预测误差或统计置信区间。由于 0≤z≤1 且 Q 为固定加权均值，必有 K≤2Q(1−Q)、V≤Q(1−Q)。因此高 Q 与高 K 的稀少及其负相关部分有数学约束，不能单独作为低可靠性的证据。",
        "阈值仅拟合 A1 的 K 分布，线性分位数；低冲突 K≤P50，中等 P50<K≤P90，高 K>P90，极高 K>P95（高冲突子集）。‘显著冲突’是分位阈值操作定义，不是显著性检验结果。A2/A3 直接应用 A1 阈值。",
        markdown_table(c["threshold_rows"],["quantile","K_threshold","fitting_dataset","n"]),
        "## 实际数据与总体分布",
        markdown_table(c["profiles"],["dataset","n","mean_K","median_K","std_K","min_K","max_K","P95_K","high_conflict_rate","extreme_conflict_rate"]),
        "仅为恢复被选中文本，对 A1 原始压缩文件流式读取一次；A2/A3 原始 JSONL 未重新解压。原始附件与全部第一小问产物进行前后 SHA-256 核验。",
        "## 领域冲突",
        markdown_table(c["domains"],["domain","n","mean_K","median_K","std_K","P90_K","P95_K","high_conflict_rate","extreme_conflict_rate","high_rate_wilson95_low","high_rate_wilson95_high"]),
        "以上排序是描述排序，不声称领域差异显著。Wilson 区间仅描述二项比例的不确定性；book 样本量小，区间较宽，不宜据此推断整体领域。附件采样机制、同源文档相关性也限制总体推断。",
        "## 指标对冲突",
        "Dᵢⱼₖ=|zᵢⱼ−zᵢₖ|，严重率 P(D≥0.5)。每个范围逐对聚合 231 对，不保存 n×231 明细；Top20 按严重率降序、平均差异降序排序。相关性另列，恒定指标相关系数留空。",
        markdown_table(top,["metric_1","metric_2","severe_pair_conflict_rate","mean_abs_difference","median_abs_difference","pearson_r","spearman_r"]),
        markdown_table(c["correlation_conflict"],["correlation_group","pair_count","mean_severe_pair_rate","mean_pair_difference"]),
        "负相关描述跨样本共同变化，D 描述同一样本的数值差异，两者不等同。即使正相关，位置/尺度分布不同也可产生大量高 D；全部数值来自固定归一化，不回调第一小问参数。",
        "## 数据驱动成因材料",
        "高冲突组与普通组的 22 指标平均值、中位数、P10/P90、KS、Wasserstein 与 Cliff's delta（高减普通）见 `high_vs_ordinary_indicator_summary.csv`。这些比较由 K 选择分组，存在定义带来的选择效应；不能当成独立因果验证。下表按均值差绝对值排序。",
        markdown_table(sorted(c["feature_comparison"],key=lambda r:-abs(r["mean_difference_high_minus_ordinary"]))[:12],
            ["metric","mean_high","mean_ordinary","median_high","median_ordinary","mean_difference_high_minus_ordinary","cliffs_delta_high_vs_ordinary"]),
        "正/负极端偏离的指标分别是 max(Z) 与 min(Z)（减去同一个 Q 不改变排序）。大量指标并列为 1 或 0；统计将每条样本均分给全部并列最大×最小组合，确保各组合的 fractional_sample_count 之和等于高冲突样本数，不偏向指标排列顺序。文本表只为展示选固定权重最大的并列指标，同时保留全部并列名称。",
        markdown_table(driver,["top_positive_metric","top_negative_metric","fractional_sample_count","fractional_rate","possible_sample_count"]),
        "分领域及扩展集同样计算驱动组合，见 `high_conflict_driver_summary.csv`。定义包含的数学关系、量表离散化/天花板、效用平台与评价构念差异是解释时必须考虑的机制；具体文本成因须人工对照。尚未作正式人工标注或因果判断。",
        "## 文本核验材料及全局高 Q 边界",
        markdown_table(c["text_selection"],["selection_group","eligible_n","requested_n","selected_n","definition"]),
        f"A1 全局 Q 上四分位以上且高 K 的记录数为 {c['upper_quartile_high_n']}；全局 Q 中位数以上且高 K 的记录数为 {c['median_high_n']}。不能凑出不存在的 20 条全局高 Q+高 K 文本。另使用**高冲突组内部 Q 三分位**取得相对高/中/低各 20 条（不同分组允许同一文本复用），清楚标明组内相对等级；最大 K 30 条也完整保留。",
        "四类全局 Q 高/低×K 高/低控制组以 Q 中位数及 K 的 P50/P90 划分，中间 K 不属于四类；完整六组计数另存，不删除任何样本。文本见 `A1_high_conflict_text_samples.csv`，全部标注待人工核验；当前没有将 AI 判断作为实验数据。",
        markdown_table(c["quadrants"],["Q_group","K_group","n","mean_Q","mean_K","mean_Delta_Q"]),
        "## A2/A3 扩展验证",
        "所有视图使用原权重、Q、K 公式及 A1 阈值。full 保留跨集重叠，nonoverlap 仅作为对照索引，不修改/删除原输出。Cliff's delta 定义为扩展减 A1；KS 为描述统计，不用 p 值宣称相同。",
        markdown_table(c["validation"],["domain","extended_dataset","view","n_A1","n_extended","mean_K_A1","mean_K_extended","median_K_A1","median_K_extended","high_rate_A1","high_rate_extended","P95_K_A1","P95_K_extended","KS_statistic","Wasserstein_distance","cliffs_delta_extended_vs_A1"]),
        "不仅比较 K 分布，还比较全部 231 对严重率向量及高冲突驱动分布；Top10 重合仅是描述性指标。某域高冲突样本很少时，驱动组合复现结论不稳定。",
        markdown_table(c["replication"],["domain","extended_dataset","view","pair_rate_pearson","pair_rate_spearman","top10_pair_overlap_n","driver_rate_pearson","top10_driver_overlap_n","high_n_A1","high_n_extended"]),
        "## Q 与 K、赋权敏感性",
        markdown_table(c["relationships"],["dataset","domain","view","n","pearson_Q_K","spearman_Q_K","pearson_K_Delta_Q","spearman_K_Delta_Q","pearson_K_K_equal","spearman_K_V","mean_Delta_Q"]),
        "Delta_Q=|原有 Q_CRITIC−mean Z|。等权只作对照，未替代锁定 Q。高冲突与普通组敏感性比较及分领域/分 Q 四分位对照如下：",
        markdown_table(sensitivities,["scope","domain","n_high","n_ordinary","mean_Delta_Q_high","mean_Delta_Q_ordinary","mean_ratio_high_over_ordinary","cliffs_delta_high_vs_ordinary"]),
        "总体相关性必须结合分领域与 Q 分层结果解释。Delta_Q 衡量对一种替代权重的变化，不是真实质量误差；增大也不自动证明应惩罚 Q。当前没有评分正确性的外部标签、冲突校准或修正收益证据。",
        "## 当前停止点与复用",
        "已完成 K、阈值、指标对、领域、驱动、文本材料、扩展验证及评分敏感性诊断。九幅图位于 `outputs/figures/question1_part2/`。未计算 Q*，未测试/选择 lambda，未生成冲突修正评分。等待人工确认冲突规律，后续再决定 Q+K 或 Q*；第一小问保持锁定。",
        "后续可直接读取三份 `*_conflict.csv`、`conflict_thresholds_A1.csv`、指标对/驱动/验证 CSV。可运行 `python scripts/05_analyze_quality_conflict.py` 重现此诊断阶段；运行前后核验锁定文件，一旦输入结构或内容不符即停止。",
        "全部输出为实际附件结果；文本复核待人工进行；图中散点仅固定随机种子抽取展示，所有统计使用全部记录。"]
    path=ROOT/"outputs/report/question1_part2_conflict_stage1.md"
    path.write_text("\n\n".join(report)+"\n",encoding="utf-8")


def record_findings():
    """Summarize already-saved diagnostics only; no new fit or raw reread."""
    def read(name):return read_csv(OUT/name)
    def number(row,key):return float(row[key])
    profile=read('dataset_conflict_summary.csv')[0]
    relationships=read('conflict_score_relationships.csv')
    overall=relationships[0]
    within=next(r for r in relationships if r['dataset']=='A1' and r['view']=='high_K')
    quartile=next(r for r in relationships if r['dataset']=='A1' and r['view']=='Q_quartile_1')
    sensitivity=read('score_sensitivity_group_comparison.csv')
    first=sensitivity[0];lowest=next(r for r in sensitivity if r['scope']=='A1_Q_quartile_1')
    domains=read('domain_conflict_summary_A1.csv')
    book_n=next(r['n'] for r in domains if r['domain']=='book')
    protected_n=len(read('locked_inputs_before.csv'))
    drivers=[r for r in read('high_conflict_driver_summary.csv') if r['dataset']=='A1' and r['domain']=='all']
    features=read('high_vs_ordinary_indicator_summary.csv')
    fluency=next(r for r in features if r['metric']=='fluency_en')
    text=read('A1_high_conflict_text_samples.csv')
    # These are explicitly excerpt-only observations, never full-text judgments,
    # final human labels, or input to an adjustment model.
    observations={
        'BkiUbvjxK6wB9mn49q52':'开头为带 LaTeX 命令及文献引用的介子跃迁论文；流畅为高端指标而词数效用为低端，提示专业文本与长度平台存在差异。',
        'BkiUdavxK7FjYAb_7N4b':'开头包含版权页、目录与小说序章；最低指标为独特词比例，需人工核验长篇叙述与重复词的区别。',
        'BkiUc0nxK6wB9k0iLK8R':'全文仅 47 字符，为压缩 JavaScript 的命令行；英语自然语言流畅分类不宜直接视为代码用途正确性的标签。',
        'BkiUbXU5jDKDyDF8vuRy':'开头为社会学书籍标题、出版社及版权材料；专业性高、英语流畅低的判定需要结合长篇结构人工检查。',
        'BkiUainxK1Thg9qFa8nv':'可见重复的沙发关键词行，每行带句号；句末标点高可以与流畅低共存，体现格式规则与文本意义的差异。',
        'BkiUbRzxK3xgpdbchQki':'全文为 76 字符的 PHP 配置语句；大写比例反向分数高、英语流畅低，体现结构化代码与语言分类的适用性差异。',
        'BkiUdOI5qsJBjuAa5ywn':'全文为流畅的珠宝商品介绍和规格选择；流畅高、推理低可以共存，需人工核验推广性，不能凭流畅认定全面高质。',
        'BkiUdmg5qYVBXTJ3BsEF':'开头为带 LaTeX、数学表达式和文献引用的天文学论文；大写比例反向分数低需结合符号/格式核验，不能自动判为正文质量差。',
        'BkiUdOg5qhDBZQieouQK':'开头为英语篮球比赛新闻，含比分和球员；流畅高、专业性相对低，可能反映新闻体裁与专业性构念差异。',
        'BkiUd-U4eIXh7gjUqssm':'开头及可见全文为波兰语村庄百科材料，却有 fluent=1；分类输出本身不是语言/质量真值。',
        'BkiUd_3xaJJQnK0WJoXz':'可见全文为德语小行星百科条目；流畅高、推理低可体现描述性事实文本与推理维度差异。',
        'BkiUduvxK7IAEwOJJChw':'开头为德语音乐术语解释；无广告高、英语流畅低，提示语言适用范围须与内容用途区分。'}
    annotated=[]
    for row in text:
        if row['id'] not in observations:continue
        annotated.append({k:row[k] for k in ['id','domain','Q','K','selection_groups','top_positive_metric','top_negative_metric']}|
            {'annotation_status':'AI辅助复核，非最终人工标注','inspection_scope':'仅检查开头片段最多1200字符；短文本可能覆盖全文；非正式实验证据',
             'full_content_characters':len(row['content']),'content_excerpt':row['content'][:1200],
             'preliminary_observation':observations[row['id']]})
    write_rows(OUT/'A1_conflict_text_AI_review.csv',annotated)
    notes=["# 问题一第2小问冲突诊断：实际发现与解释边界",
        "本页只汇总已经生成的真实统计；未重新计算 CRITIC、修改预处理/Q 或构造 Q*。详细表、定义与九幅图索引见 `question1_part2_conflict_stage1.md`。",
        f"A1 n={profile['n']}，K 均值 {number(profile,'mean_K'):.6f}，中位数 {number(profile,'median_K'):.6f}，标准差 {number(profile,'std_K'):.6f}，范围 [{number(profile,'min_K'):.6f}, {number(profile,'max_K'):.6f}]。P90={number(profile,'high_threshold_A1_P90'):.6f}，P95={number(profile,'extreme_threshold_A1_P95'):.6f}，高冲突 {profile['high_conflict_count']} 条。",
        "## 领域与主要组合",
        markdown_table(domains,['domain','n','high_conflict_rate','mean_K']),
        f"book 的高冲突率最高，但仅 {book_n} 条；不能根据描述排序宣称统计显著或代表整个书籍领域。stackexchange 与 commoncrawl 的高冲突率相对较低。",
        "A1 Top10 指标对见下表：严重率指 P(|z_j-z_k|≥0.5)，按该比例降序。大量涉及 reasoning，但全局指标对差异与高 K 的极端驱动并非同一个统计量。",
        markdown_table(read('top20_conflicting_pairs.csv')[:10],['rank','metric_1','metric_2','severe_pair_conflict_rate','pearson_r']),
        markdown_table(drivers[:10],['top_positive_metric','top_negative_metric','fractional_rate','possible_sample_count']),
        f"高冲突组 fluency_en 平均值 {number(fluency,'mean_high'):.6f}，普通组 {number(fluency,'mean_ordinary'):.6f}。主要极端组合表现为无广告/长度效用等高分与英语流畅低分、推理低分并存。并列值均分贡献后的比例不能当成原始唯一记录比例；possible_sample_count 可以重叠。",
        "数据支持‘评价构念不同、格式规则与内容评价不同、代码/非英语文本与英语分类适用范围不同’作为待核验机制；不将这些解释作为已经证实的因果成因。",
        "## 扩展集复现",
        markdown_table(read('sample_extended_conflict_validation.csv'),['domain','view','high_rate_A1','high_rate_extended','KS_statistic','Wasserstein_distance','cliffs_delta_extended_vs_A1']),
        markdown_table(read('conflict_pattern_replication.csv'),['domain','view','pair_rate_pearson','pair_rate_spearman','top10_pair_overlap_n','driver_rate_pearson','top10_driver_overlap_n']),
        "非重叠视图仍保留很高的指标对/驱动模式一致性；github 的分布差异更小，arxiv 的高冲突率略上升。这支持主要冲突结构复现，不能宣称分布完全相同，也不能推广到附件以外的数据。",
        "## 评分敏感性与不能跳过的控制结果",
        f"Q 与 K：Pearson={number(overall,'pearson_Q_K'):.6f}，Spearman={number(overall,'spearman_Q_K'):.6f}。K 与 ΔQ：Pearson={number(overall,'pearson_K_Delta_Q'):.6f}，Spearman={number(overall,'spearman_K_Delta_Q'):.6f}。",
        f"高冲突组 mean ΔQ={number(first,'mean_Delta_Q_high'):.6f}，普通组 {number(first,'mean_Delta_Q_ordinary'):.6f}，比值 {number(first,'mean_ratio_high_over_ordinary'):.4f}，Cliff's delta={number(first,'cliffs_delta_high_vs_ordinary'):.6f}。这是总体赋权敏感性差异，尚非质量误差证据。",
        f"但高冲突组内部 Pearson(K,ΔQ) 仅 {number(within,'pearson_K_Delta_Q'):.6f}；Q 最低四分位内部仅 {number(quartile,'pearson_K_Delta_Q'):.6f}，Spearman={number(quartile,'spearman_K_Delta_Q'):.6f}。在该 Q 层中高/普通组 ΔQ 均值比降至 {number(lowest,'mean_ratio_high_over_ordinary'):.4f}、Cliff's delta={number(lowest,'cliffs_delta_high_vs_ordinary'):.6f}。总体差异不能直接解释为冲突的独立影响。",
        "Q、K、ΔQ 都是同一组 Z 的函数，Q/K 还受到 K≤2Q(1−Q) 的数学约束。需要人工文本核验和独立质量标签才能评估真实错误/收益。目前仅确认内部意见差异与一种权重敏感性，不足以据此自动确定惩罚系数或选择 Q*。",
        "## 文本材料与初步比较（非正式证据）",
        f"完整文本材料包含最大 K 样本、高冲突组内相对高/中/低 Q 样本及全局四象限控制样本，组间可复用，共 {len(text)} 条唯一记录。实际选取数量见 text_selection_summary.csv；组内相对高 Q 材料不得称为全局高 Q。",
        f"另对其中 {len(annotated)} 条开头片段记录 AI 辅助复核，见 `A1_conflict_text_AI_review.csv`。明确标记**AI辅助复核，非最终人工标注**，不把片段观察作为全文质量或正式实验结论。控制组可见新闻/百科材料，高 Q+高 K 可见商品文案和带公式论文，低 Q+高 K 可见重复关键词和短代码；这些是选定案例，不是各组总体占比。",
        "原始全文有长字段，Python 读取文本 CSV 时应先设置 `csv.field_size_limit(2**31-1)`。完整输出未截断 content，主冲突 CSV 未写入 content。",
        "## 停止点",
        f"独立核验结果见 `conflict_verification.csv`。第一小问/官方附件的 {protected_n} 个文件 SHA-256 均保持一致。当前诊断阶段已完成；未构造 Q*、未测试 lambda，等待人工确认后再决定采用 Q+K 还是修正评分。"]
    (ROOT/'outputs/report/question1_part2_findings.md').write_text('\n\n'.join(notes)+'\n',encoding='utf-8')


def main():
    """Run the authorized diagnostic stage and stop before any Q adjustment."""
    OUT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),logging.FileHandler(OUT/"conflict_analysis.log",encoding="utf-8")])
    settings=yaml.safe_load((ROOT/"configs/conflict_analysis.yaml").read_text(encoding="utf-8"))
    if settings["stage_stop"]!="diagnostics_only_no_Q_star" or settings["threshold_fitting_dataset"]!="A1":
        raise ValueError("Unauthorized model stage or threshold fitting scope")
    logging.info("Snapshot protected part 1 artifacts and raw attachment hashes")
    before=protected_files()
    write_rows(OUT/"locked_inputs_before.csv",[{"path":p,"sha256":h} for p,h in before.items()])
    model=json.loads((ROOT/"outputs/preprocess/preprocessing_model_A1.json").read_text(encoding="utf-8"))
    names=model["metric_order"]
    if len(names)!=22 or len(set(names))!=22:raise ValueError("Expected exactly 22 locked metrics")
    wrows=read_csv(ROOT/"outputs/quality/critic_weights_A1.csv")
    if {r["metric"] for r in wrows}!=set(names) or len(wrows)!=22:raise ValueError("Locked weight schema mismatch")
    wmap={r["metric"]:float(r["critic_weight"]) for r in wrows};weights=np.array([wmap[n] for n in names])
    manifest=read_csv(ROOT/"outputs/preprocess/preprocessing_input_manifest.csv")
    datasets={}
    for dataset in INPUTS:
        n=sum(int(r["n"]) for r in manifest if r["dataset"]==dataset)
        logging.info("Read locked %s scores: %d records",dataset,n)
        data=load_scores(dataset,names,n);data.update(conflict_indices(data["Z"],data["Q"],weights));datasets[dataset]=data
    a1=datasets["A1"];domains=np.array([r["domain"] for r in a1["metadata"]]);thresholds=fit_conflict_thresholds(a1["K"],settings)
    thresholds_rows=[{"quantile":p,"K_threshold":v,"fitting_dataset":"A1","n":len(a1["K"]),"quantile_method":settings["quantile_method"]} for p,v in thresholds.items()]
    write_rows(OUT/"conflict_thresholds_A1.csv",thresholds_rows)
    profiles=[{"dataset":d,**conflict_profile(data["K"],thresholds,settings)} for d,data in datasets.items()]
    write_rows(OUT/"dataset_conflict_summary.csv",profiles)
    domain_rows=[{"dataset":"A1","domain":domain,**conflict_profile(a1["K"][domains==domain],thresholds,settings)} for domain in sorted(set(domains))]
    domain_rows.sort(key=lambda r:(-r["high_conflict_rate"],-r["mean_K"]))
    write_rows(OUT/"domain_conflict_summary_A1.csv",domain_rows)
    for d,data in datasets.items():
        logging.info("Write %s conflict output",d);save_samples(d,data,thresholds)
    overlap_rows=[]
    for d,domain in [("A2","arxiv"),("A3","github")]:
        data=datasets[d];a1ids={r["id"]:i for i,r in enumerate(a1["metadata"]) if r["domain"]==domain}
        ids={r["id"] for r in data["metadata"]}
        expected={r["id"] for r in read_csv(ROOT/f"outputs/preprocess/{d}_{domain}_nonoverlap_ids.csv")}
        if expected!=ids-set(a1ids) or not set(a1ids)<=ids:raise ValueError("Frozen nonoverlap index mismatch")
        mask=np.array([r["id"] in expected for r in data["metadata"]]);data["nonoverlap_mask"]=mask
        shared=[(a1ids[r["id"]],i) for i,r in enumerate(data["metadata"]) if r["id"] in a1ids]
        differences=[abs(a1["K"][i]-data["K"][j]) for i,j in shared]
        if max(differences)>1e-14:raise ValueError("Shared record K mismatch")
        overlap_rows.append({"dataset":d,"domain":domain,"shared_n":len(shared),"nonoverlap_n":int(mask.sum()),"max_shared_K_abs_diff":max(differences)})
    write_rows(OUT/"overlap_conflict_consistency.csv",overlap_rows)
    # Each group is processed sequentially, only one numerical pair vector at a time.
    scopes=[("A1","all","full",np.ones(len(a1["Q"]),dtype=bool))]
    scopes += [("A1",d,"full",domains==d) for d in sorted(set(domains))]
    scopes += [(d,domain,view,np.ones(len(datasets[d]["Q"]),dtype=bool) if view=="full" else datasets[d]["nonoverlap_mask"])
        for d,domain in [("A2","arxiv"),("A3","github")] for view in ["full","nonoverlap"]]
    pair_rows=[];driver_rows=[];relationships=[];by_scope={};drivers_by_scope={}
    for d,domain,view,mask in scopes:
        logging.info("Pair/driver diagnostics %s %s %s n=%d",d,domain,view,int(mask.sum()))
        data=datasets[d];meta={"dataset":d,"domain":domain,"view":view};key=(d,domain,view)
        rows=pair_conflict_summary(data["Z"][mask],names,settings,meta);pair_rows+=rows;by_scope[key]=rows
        drivers=deviation_driver_summary(data["Z"][mask],data["K"][mask]>thresholds["P90"],names,meta)
        driver_rows+=drivers;drivers_by_scope[key]=drivers
        relationships.append(relationship(data,mask,meta))
    write_rows(OUT/"indicator_pair_conflict.csv",pair_rows)
    pairs=by_scope[("A1","all","full")]
    top_pairs=sorted(pairs,key=lambda r:(-r["severe_pair_conflict_rate"],-r["mean_abs_difference"]))[:20]
    write_rows(OUT/"top20_conflicting_pairs.csv",[{"rank":i,**r} for i,r in enumerate(top_pairs,1)])
    write_rows(OUT/"high_conflict_driver_summary.csv",driver_rows)
    high=a1["K"]>thresholds["P90"];feature_rows=[]
    for j,name in enumerate(names):
        x,y=a1["Z"][high,j],a1["Z"][~high,j]
        feature_rows.append({"metric":name,"n_high":len(x),"n_ordinary":len(y),"mean_high":float(x.mean()),"mean_ordinary":float(y.mean()),
            "median_high":float(np.median(x)),"median_ordinary":float(np.median(y)),
            "P10_high":float(np.quantile(x,.1)),"P90_high":float(np.quantile(x,.9)),"P10_ordinary":float(np.quantile(y,.1)),"P90_ordinary":float(np.quantile(y,.9)),
            "mean_difference_high_minus_ordinary":float(x.mean()-y.mean()),"KS_statistic":float(ks_2samp(x,y).statistic),
            "Wasserstein_distance":float(wasserstein_distance(x,y)),"cliffs_delta_high_vs_ordinary":cliffs_delta(x,y),
            "mean_weighted_abs_deviation_high":float((weights[j]*np.abs(x-a1["Q"][high])).mean()),
            "mean_weighted_abs_deviation_ordinary":float((weights[j]*np.abs(y-a1["Q"][~high])).mean())})
    write_rows(OUT/"high_vs_ordinary_indicator_summary.csv",feature_rows)
    sensitivities=[sensitivity_comparison(a1,np.ones(len(high),dtype=bool),high,{"scope":"A1_all","domain":"all"})]
    sensitivities += [sensitivity_comparison(a1,domains==d,high,{"scope":"A1_domain","domain":d}) for d in sorted(set(domains))]
    qbounds=np.quantile(a1["Q"],[.25,.5,.75]);qgroups=np.searchsorted(qbounds,a1["Q"],side="left")
    for i in range(4):
        mask=qgroups==i
        sensitivities.append(sensitivity_comparison(a1,mask,high,{"scope":f"A1_Q_quartile_{i+1}","domain":"all"}))
        relationships.append(relationship(a1,mask,{"dataset":"A1","domain":"all","view":f"Q_quartile_{i+1}"}))
    for label,mask in [("high_K",high),("ordinary_K",~high)]:
        relationships.append(relationship(a1,mask,{"dataset":"A1","domain":"all","view":label}))
    write_rows(OUT/"score_sensitivity_group_comparison.csv",sensitivities)
    write_rows(OUT/"conflict_score_relationships.csv",relationships)
    quadrants=[]
    for qlabel,qmask in [("high_Q",a1["Q"]>qbounds[1]),("low_Q",a1["Q"]<=qbounds[1])]:
        for klabel,kmask in [("low_K",a1["K"]<=thresholds["P50"]),("medium_K",(a1["K"]>thresholds["P50"])&~high),("high_K",high)]:
            mask=qmask&kmask
            quadrants.append({"Q_group":qlabel,"K_group":klabel,"n":int(mask.sum()),
                "mean_Q":float(a1["Q"][mask].mean()) if mask.any() else None,
                "mean_K":float(a1["K"][mask].mean()) if mask.any() else None,
                "mean_Delta_Q":float(a1["Delta_Q"][mask].mean()) if mask.any() else None,
                "A1_Q_median":float(qbounds[1]),"A1_K_P50":thresholds["P50"],"A1_K_P90":thresholds["P90"]})
    write_rows(OUT/"quality_conflict_quadrant_summary.csv",quadrants)
    validation=[];replication=[];pair_replication=[]
    for d,domain in [("A2","arxiv"),("A3","github")]:
        reference=a1["K"][domains==domain];ref_profile=conflict_profile(reference,thresholds,settings)
        refpairs=by_scope[("A1",domain,"full")];refdrivers=drivers_by_scope[("A1",domain,"full")]
        for view in ["full","nonoverlap"]:
            mask=np.ones(len(datasets[d]["K"]),dtype=bool) if view=="full" else datasets[d]["nonoverlap_mask"]
            extension=datasets[d]["K"][mask];ep=conflict_profile(extension,thresholds,settings)
            meta={"domain":domain,"extended_dataset":d,"view":view}
            validation.append({**meta,"n_A1":len(reference),"n_extended":len(extension),
                "mean_K_A1":ref_profile["mean_K"],"mean_K_extended":ep["mean_K"],"median_K_A1":ref_profile["median_K"],"median_K_extended":ep["median_K"],
                "high_rate_A1":ref_profile["high_conflict_rate"],"high_rate_extended":ep["high_conflict_rate"],
                "P95_K_A1":ref_profile["P95_K"],"P95_K_extended":ep["P95_K"],"KS_statistic":float(ks_2samp(reference,extension).statistic),
                "Wasserstein_distance":float(wasserstein_distance(reference,extension)),"cliffs_delta_extended_vs_A1":cliffs_delta(extension,reference),
                "shared_records_included":view=="full","fixed_high_K_threshold":thresholds["P90"]})
            extpairs=by_scope[(d,domain,view)];extdrivers=drivers_by_scope[(d,domain,view)]
            rates1=np.array([r["severe_pair_conflict_rate"] for r in refpairs]);rates2=np.array([r["severe_pair_conflict_rate"] for r in extpairs])
            def top_pair_set(rows):return {(r["metric_1"],r["metric_2"]) for r in sorted(rows,key=lambda r:(-r["severe_pair_conflict_rate"],-r["mean_abs_difference"]))[:10]}
            def driver_vector(rows):
                mapping={(r["top_positive_metric"],r["top_negative_metric"]):r["fractional_rate"] for r in rows}
                return np.array([mapping.get((j,k),0.) for j in names for k in names])
            def top_driver_set(rows):return {(r["top_positive_metric"],r["top_negative_metric"]) for r in rows[:10]}
            replication.append({**meta,"pair_rate_pearson":safe_corr(rates1,rates2),"pair_rate_spearman":safe_corr(rates1,rates2,"spearman"),
                "top10_pair_overlap_n":len(top_pair_set(refpairs)&top_pair_set(extpairs)),
                "driver_rate_pearson":safe_corr(driver_vector(refdrivers),driver_vector(extdrivers)),
                "top10_driver_overlap_n":len(top_driver_set(refdrivers)&top_driver_set(extdrivers)),
                "high_n_A1":ref_profile["high_conflict_count"],"high_n_extended":ep["high_conflict_count"]})
            pair_replication += [{**meta,"metric_1":a["metric_1"],"metric_2":a["metric_2"],"severe_rate_A1":a["severe_pair_conflict_rate"],
                "severe_rate_extended":b["severe_pair_conflict_rate"],"rate_difference_extended_minus_A1":b["severe_pair_conflict_rate"]-a["severe_pair_conflict_rate"]} for a,b in zip(refpairs,extpairs)]
    write_rows(OUT/"sample_extended_conflict_validation.csv",validation)
    write_rows(OUT/"conflict_pattern_replication.csv",replication)
    write_rows(OUT/"extended_pair_conflict_comparison.csv",pair_replication)
    correlation_conflict=[]
    for label,condition in [("negative",lambda r:r["pearson_r"] is not None and r["pearson_r"]<0),
                            ("nonnegative",lambda r:r["pearson_r"] is not None and r["pearson_r"]>=0)]:
        selected=[r for r in pairs if condition(r)]
        correlation_conflict.append({"correlation_group":label,"pair_count":len(selected),
            "mean_severe_pair_rate":float(np.mean([r["severe_pair_conflict_rate"] for r in selected])),
            "mean_pair_difference":float(np.mean([r["mean_abs_difference"] for r in selected]))})
    write_rows(OUT/"correlation_vs_conflict_summary.csv",correlation_conflict)
    logging.info("Recover selected A1 text records by one streaming pass")
    text_selection=select_texts(a1,weights,thresholds,settings,manifest)
    logging.info("Generate nine diagnostic figures")
    figures=make_conflict_figures(ROOT/"outputs/figures/question1_part2",datasets,domains,domain_rows,pairs,thresholds,settings)
    write_rows(OUT/"conflict_figure_manifest.csv",[{"path":f"outputs/figures/question1_part2/{n}","statistics_scope":"full records; scatter displays fixed-seed subset"} for n in figures])
    build_report({"profiles":profiles,"domains":domain_rows,"threshold_rows":thresholds_rows,"top_pairs":top_pairs,
        "relationships":relationships,"sensitivity":sensitivities,"drivers_global":drivers_by_scope[("A1","all","full")],
        "feature_comparison":feature_rows,"text_selection":text_selection,"quadrants":quadrants,"validation":validation,
        "replication":replication,"correlation_conflict":correlation_conflict,
        "upper_quartile_high_n":int((high&(a1["Q"]>=qbounds[2])).sum()),"median_high_n":int((high&(a1["Q"]>qbounds[1])).sum())})
    record_findings()
    logging.info("Verify all protected files remain byte-for-byte unchanged")
    after=protected_files()
    if before!=after:raise ValueError("Protected part 1/raw attachment snapshot changed; stop")
    write_rows(OUT/"locked_inputs_integrity.csv",[{"path":p,"sha256_before":h,"sha256_after":after[p],"unchanged":True} for p,h in before.items()])
    (OUT/"conflict_analysis_config_snapshot.yaml").write_text(yaml.safe_dump(settings,allow_unicode=True,sort_keys=False),encoding="utf-8")
    logging.info("COMPLETED conflict diagnosis stage 1; no Q_star/lambda; locked files unchanged: %d",len(before))
    print(json.dumps({"thresholds":thresholds,"A1_profile":profiles[0],"relationship_A1":relationships[0],"sensitivity_A1":sensitivities[0]},ensure_ascii=False,allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("STOPPED: conflict diagnosis failed; no frozen input is repaired or rewritten")
        raise

# ============================================================
# Source: src/mixture/io.py
# Original lines: 1-54
# Purpose: 真实附件读取、索引/字段兼容与输入单位
# ============================================================
"""Read small official RegMix tables with strict index/schema checks."""
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
# Original lines: 1-51
# Purpose: 训练折内乘法零替换与Helmert ILR
# ============================================================
"""Training-only multiplicative zero replacement and orthonormal ILR basis."""
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
# Original lines: 1-61
# Purpose: Ridge/ElasticNet候选、训练内调参与分组CV
# M0 and second-order M2 are candidate / sensitivity models; not used as final main model. M1_Ridge is the numerical comparator; M1_ElasticNet is selected.
# ============================================================
"""Nested CV pipelines; all learned transforms are inside training folds."""
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
# Original lines: 1-256
# Purpose: 最终阶段实际执行入口：08_model_mixture_loss.py
# ============================================================
"""Train-only nested-CV M0/M1/M2 and direct 1M/60M/1B validation."""
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


def protect():
    """Hash all original attachments and locked part 1/2 artifacts/code."""
    paths=list((ROOT/'outputs').rglob('*'))+list((ROOT/'real_attachments').rglob('*'))+list((ROOT/'configs').glob('*'))+list((ROOT/'src/quality').glob('*.py'))
    paths += [p for p in (ROOT/'scripts').glob('*.py') if p.name[:2].isdigit() and int(p.name[:2])<=6]
    result={}
    for p in sorted(paths):
        if not p.is_file() or p.is_relative_to(OUT) or p.is_relative_to(ROOT/'outputs/figures/question1_part3') or p.name.startswith('question1_part3'):continue
        result[p.relative_to(ROOT).as_posix()]=sha256_file(p)
    return result


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


def figure_outputs(validation,nested,predictions,pnames,train,settings):
    """Full-data plots, labeled observed versus direct uncalibrated prediction."""
    FIG.mkdir(parents=True,exist_ok=True);paths=[]
    def save(fig,name):
        fig.tight_layout();fig.savefig(FIG/name,dpi=settings['figure_dpi'],bbox_inches='tight');plt.close(fig);paths.append((FIG/name).relative_to(ROOT).as_posix())
    fig,ax=plt.subplots(figsize=(11,5));ax.bar([n.removeprefix('train_the_pile_') for n in pnames],100*(train['P']==0).mean(axis=0),color='#4777a8')
    ax.set(ylabel='Zero recipe share (%)',title='A4: zeros by training domain');ax.tick_params(axis='x',rotation=60);save(fig,'01_train_zero_rates.png')
    fig,ax=plt.subplots(figsize=(8,4))
    primary=[r for r in nested if r['response']=='L_mean'];ax.bar([r['model'] for r in primary],[r['RMSE'] for r in primary],color='#4777a8')
    ax.set(ylabel='Nested out-of-fold RMSE',title='A4/A5 model comparison: macro Loss');save(fig,'02_nested_CV_comparison.png')
    for response in ['L_mean','L_trim']:
        fig,axes=plt.subplots(3,4,figsize=(15,10))
        for row,dataset in enumerate(['test_1M','test_60M','test_1B']):
            for col,model in enumerate(MODELS):
                values=[r for r in predictions if r['response']==response and r['dataset']==dataset and r['model']==model]
                y=np.array([r['true_loss'] for r in values]);p=np.array([r['predicted_loss'] for r in values]);ax=axes[row,col]
                ax.scatter(y,p,s=12,alpha=.6,color='#4777a8');lo,hi=min(y.min(),p.min()),max(y.max(),p.max())
                ax.plot([lo,hi],[lo,hi],linestyle='--',color='#777777');ax.set(title=f'{dataset} / {model}',xlabel='Observed Loss',ylabel='1M model prediction')
        fig.suptitle(f'{response}: no scale calibration or test-set fitting',y=1.01);save(fig,f'03_{response}_predicted_vs_observed.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(3);labels=['test_1M','test_60M','test_1B']
    for i,model in enumerate(MODELS):
        values=[next(r for r in validation if r['response']=='L_mean' and r['dataset']==d and r['model']==model) for d in labels]
        axes[0].plot(x,[r['RMSE'] for r in values],'o-',label=model)
        if model!='M0':axes[1].plot(x,[r['Spearman'] for r in values],'o-',label=model)
    axes[0].set(xticks=x,xticklabels=labels,ylabel='RMSE',title='Absolute transfer error')
    axes[1].set(xticks=x,xticklabels=labels,ylabel='Spearman',title='Recipe ranking transfer')
    for ax in axes:ax.legend(fontsize=8)
    save(fig,'04_scale_absolute_vs_rank.png')
    write_rows(OUT/'figure_manifest.csv',[{'path':p} for p in paths])


def build_report(datasets,names,settings,nested,validation,final_rows,scale_rows,warnings_rows,best,protected_n):
    """Describe findings without test-driven selection or unsupported causal effects."""
    def table(rows,columns):
        def value(v):
            if v is None:return '未定义'
            if isinstance(v,(float,np.floating)):return f'{v:.6f}'
            return str(v).replace('|','\\|')
        return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |']+
            ['| '+' | '.join(value(r.get(k)) for k in columns)+' |' for r in rows])
    primary=[r for r in nested if r['response']=='L_mean'];secondary=[r for r in nested if r['response']=='L_trim']
    ridge=next(r for r in primary if r['model']=='M1_Ridge')
    elastic=next(r for r in primary if r['model']=='M1_ElasticNet')
    quadratic=next(r for r in primary if r['model']=='M2_ElasticNet')
    ridge_train=next(r for r in final_rows if r['response']=='L_mean' and r['model']=='M1_Ridge')
    quadratic_trim=next(r for r in final_rows if r['response']=='L_trim' and r['model']=='M2_ElasticNet')
    ilr=ILRTransformer(settings['zero_replacement_fraction_of_training_min']).fit(datasets['train_1M']['P'])
    selected=[r for r in validation if r['response']=='L_mean' and r['model']==best]
    text=['# 问题一第3小问：阶段1配比/Loss建模与检验',
        '本轮已完成 A4–A15 数据诊断、响应变量、单纯形/零值处理、ILR、M0/M1/M2 与 A6–A11 检验。原 Q、K、预处理及 CRITIC 全部冻结；未引入 Q/K，未读取 A16 映射用于建模，未预测10B/70B，未建立模型规模方程或 Scaling Law。',
        '## 真实结构',table([{'dataset':d,'n':len(v['P']),'provenance':v['provenance']} for d,v in datasets.items()],['dataset','n','provenance']),
        '详细十二文件、17配比列、13Loss列、逐域分布、零值/相关矩阵与行和诊断见 outputs/mixture/phase1_data_diagnosis.md。所有对应表的 index 唯一并在同一实验内一一配对。',
        '## 响应与组合变换',
        '主响应为13域等权 L_mean；敏感性响应 L_trim 去掉最高最低各一个后对11域取均值。两个响应各自拟合相同候选和CV流程，未选择某个有利Loss域。',
        f'先闭合 p/sum(p)。零替换 δ=min(0.5×闭合训练配比最小正值,1/(2×17))；A4全量 min_positive={ilr.min_positive_training_:.12g}，最终 δ={ilr.zero_delta_:.12g}。每行零坐标变为 δ，非零坐标乘 (1−零个数×δ)，保留非零比值。测试统一使用训练参数；CV中则仅从当前折训练部分拟合 δ。原始零可能为结构性缺席或舍入，替换不是真实观测恢复，因此结论依赖此处理。',
        'ILR 用固定列顺序的16×17 Helmert正交对比矩阵 H：x=log(p_replaced)Hᵀ，HHᵀ=I，H1=0。第k个坐标为 sqrt(k/(k+1))×log(前k域几何均值/第k+1域)，k=1..16。单纯形约束得到16维独立坐标；CLR另存仅供解释，不直接对17原始p做OLS。',
        '## 候选模型与无泄漏CV',
        'M0=训练平均Loss；M1-Ridge/M1-ElasticNet=16维ILR一阶模型；M2-ElasticNet=16个一阶坐标+120个两两交叉乘积，共136个特征，没有无约束OLS。所有X标准化在管道内，y也在每个训练折内标准化；预测还原到原Loss单位。L1筛选依赖所选ILR基，不将某个ILR交叉系数直接称为某对原领域因果互补。',
        f"固定种子 {settings['random_state']}。外层{settings['outer_folds']}折、内层{settings['inner_folds']}折嵌套CV；同配方组不得跨折。最终参数在全部A4/A5上用{settings['final_selection_folds']}折选取并重拟合，只用于之后的直接检验。最佳模型按训练嵌套OOF RMSE选为 {best}，测试结果不参与选择。所有CV折、完整网格及外层模型参数已保存。",
        'Ridge alpha 网格：'+str(settings['ridge_alphas'])+'；ElasticNet alpha：'+str(settings['elastic_alphas'])+'；l1_ratio：'+str(settings['elastic_l1_ratios'])+'.',
        '### L_mean 嵌套CV',table(primary,['model','RMSE','MAE','R2','Pearson','Spearman','mean_fold_RMSE','std_fold_RMSE']),
        '### L_trim 敏感性',table(secondary,['model','RMSE','MAE','R2','Pearson','Spearman','mean_fold_RMSE','std_fold_RMSE']),
        f"数值上 M1-Ridge 的 L_mean 嵌套 RMSE={ridge['RMSE']:.6f}，M1-ElasticNet={elastic['RMSE']:.6f}，差值仅 {elastic['RMSE']-ridge['RMSE']:.6f}；不能宣称显著优越。按用户解释优先的要求，保留 M1-ElasticNet 为可解释主方案候选，Ridge 为数值最佳对照，等待人工确认，不凭检验集选择最终模型。M2-ElasticNet 的嵌套 RMSE={quadratic['RMSE']:.6f}，暂未支持加入交互项。",
        '### 最终训练拟合与复杂度',table(final_rows,['response','model','training_RMSE','training_R2','nested_OOF_RMSE','n_model_features','n_nonzero_coefficients','parameters']),
        '训练误差小于嵌套OOF误差是乐观偏差诊断；M2是否增加泛化能力须看嵌套CV和同尺度1M检验，不能仅看训练拟合。跨模型CV差距很小时不声称统计显著优越；这里只给出真实误差及折间离散。',
        f"一阶 Ridge 的训练 RMSE={ridge_train['training_RMSE']:.6f}、嵌套 OOF={ridge['RMSE']:.6f}，未见严重训练拟合崩溃。M2 在 L_trim 下训练 RMSE={quadratic_trim['training_RMSE']:.6f}、嵌套 OOF={quadratic_trim['nested_OOF_RMSE']:.6f}；二阶复杂度增加但泛化未改善，表现出额外拟合乐观/过拟合风险，当前不作为优选。跨尺度绝对偏差不能单独归因于过拟合，模型没有任何规模变量。",
        '## 三尺度直接检验',table([r for r in validation if r['response']=='L_mean'],['dataset','model','RMSE','MAE','R2','Pearson','Spearman','prediction_bias_mean']),
        'L_trim 的同样指标完整保存在 model_validation.csv。图展示固定1M模型的原预测，不以测试真值调整均值、截距、尺度或重新训练。外部检验的baseline相关系数因恒定预测而未定义，留空而非置零；嵌套OOF的M0跨折采用不同训练均值，因此聚合OOF可能有非零相关，不代表配比解释力。',
        '### 训练CV选定模型',table(selected,['dataset','model','n','RMSE','MAE','R2','Pearson','Spearman','prediction_bias_mean']),
        '绝对误差/R²与配方优劣排序相关须分开判断。若跨尺度偏差大而排序较好，仅能称排序迁移有效，不能声称1M模型无条件解释60M/1B绝对Loss。额外去均值RMSE只是描述性分解，不是可部署校准或另一组实际验证结果。',
        '## 实际跨尺度配方排序',table(scale_rows,['response','dataset_1','dataset_2','shared_recipe_n','actual_Pearson','actual_Spearman','interpretation']),
        '共同配方以闭合向量匹配；同名index不一定为同配方。没有相同配方时，直接实际排序一致性未定义，不虚构相关值。1B上的预测/真值Spearman仅是固定1M模型的间接迁移证据。',
        '## 范围与局限',
        f'本轮捕获收敛/运行警告数 {len(warnings_rows)}，逐条见 convergence_warnings.csv；所选ElasticNet达到迭代上限则停止，不隐瞒失败。原件及先前锁定产物/代码 {protected_n} 个文件前后 SHA-256 保持一致。独立验证见 mixture_verification.csv。',
        '对稀疏配方的log-ratio零替换会影响效应，未声称唯一零处理结论。训练配方数有限，二阶项更易过拟合；ILR基与L1惩罚的选择影响稀疏解释。后续仍需按用户下一阶段做原领域扰动、互补趋势、Q映射及 estimated 外推讨论，当前没有提前生成这些结论。',
        '10B/70B是estimated材料，本轮只读结构和Loss描述，未用于CV、拟合、选择或预测。模型规模方程尚未建立。Q/K保持锁定，未进入本轮任何解释变量。',
        '## 当前停止点',
        '可复用 ILR_basis.csv、composition_transform_A4.json、四个候选的final_model_metadata.json/joblib、嵌套OOF与测试预测CSV。等待人工确认阶段1后再进入 Q_p 与外推分析。']
    (ROOT/'outputs/report/question1_part3_stage1_modeling.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'models/outer').mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'mixture_modeling.log',encoding='utf-8')])
    settings=yaml.safe_load((ROOT/'configs/mixture_modeling.yaml').read_text(encoding='utf-8'))
    if settings['stage_stop']!='M0_M1_M2_validation_only_no_Q_K_or_large_scale_prediction':raise ValueError('Unauthorized phase')
    # Diagnosis must already exist, and its source hashes must still match.
    import csv
    with (OUT/'input_manifest.csv').open(encoding='utf-8-sig') as f:manifest=list(csv.DictReader(f))
    if len(manifest)!=12 or any(sha256_file(ROOT/r['path'])!=r['sha256'] for r in manifest):raise ValueError('Diagnosis absent or original input changed')
    before=protect();write_rows(OUT/'locked_inputs_before.csv',[{'path':p,'sha256':h} for p,h in before.items()])
    datasets,names,loss_names=load_datasets(ROOT);train=datasets['train_1M'];P=train['P'];indices=train['indices']
    # Train/test composition overlap must be disclosed, never hidden by index.
    trainkeys=set(signatures(P))
    if any(trainkeys&set(signatures(datasets[d]['P'])) for d in ['test_1M','test_60M','test_1B']):raise ValueError('Train/test recipe overlap: validation needs explicit policy')
    ilr=ILRTransformer(settings['zero_replacement_fraction_of_training_min']).fit(P)
    write_rows(OUT/'ILR_basis.csv',[{'coordinate':f'ilr_{i+1:02d}',**dict(zip(names,map(float,ilr.basis_[i])))} for i in range(16)])
    (OUT/'composition_transform_A4.json').write_text(json.dumps({'fit_dataset':'train_1M','n_fit':len(P),'domain_order':names,
        'training_min_positive':ilr.min_positive_training_,'zero_delta':ilr.zero_delta_,'zero_fraction':ilr.zero_fraction,
        'closure':'p/sum(p)','basis':ilr.basis_.tolist()},indent=2)+'\n',encoding='utf-8')
    for d in ['train_1M','test_1M','test_60M','test_1B']:
        replaced=ilr.replace(datasets[d]['P']);x=ilr.transform(datasets[d]['P']);clr=np.log(replaced)-np.log(replaced).mean(axis=1,keepdims=True)
        write_rows(OUT/f'{d}_ILR_features.csv',[{'index':index,**{f'ilr_{j+1:02d}':float(x[i,j]) for j in range(16)}} for i,index in enumerate(datasets[d]['indices'])])
        write_rows(OUT/f'{d}_closed_replaced_composition.csv',[{'index':index,**dict(zip(names,map(float,replaced[i])))} for i,index in enumerate(datasets[d]['indices'])])
        write_rows(OUT/f'{d}_CLR_explanation.csv',[{'index':index,**dict(zip(names,map(float,clr[i])))} for i,index in enumerate(datasets[d]['indices'])])
    outer=group_folds(P,settings['outer_folds'],settings['random_state']);outer_assign=[];inner_assign=[]
    for fold,(fit,hold) in enumerate(outer,1):
        outer_assign += [{'index':indices[i],'outer_holdout_fold':fold} for i in hold]
        inner=group_folds(P[fit],settings['inner_folds'],settings['random_state']+fold)
        for innerfold,(_,validation) in enumerate(inner,1):inner_assign += [{'index':indices[fit[i]],'outer_fold':fold,'inner_holdout_fold':innerfold} for i in validation]
    write_rows(OUT/'outer_fold_assignments.csv',outer_assign);write_rows(OUT/'inner_fold_assignments.csv',inner_assign)
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
        'interpretable_main_candidate':'M1_ElasticNet',
        'selection_rule':settings['selection_rule'],'training_dataset':'train_1M','test_used_for_selection':False,
        'Q_or_K_used':False,'estimated_loss_used':False,'phase':'stage1_candidates_pending_human_confirmation'},indent=2)+'\n',encoding='utf-8')
    (OUT/'final_model_metadata.json').write_text(json.dumps(final_metadata,indent=2)+'\n',encoding='utf-8')
    write_rows(OUT/'model_coefficients.csv',coefficients)
    write_rows(OUT/'convergence_warnings.csv',warning_rows,['response','model','phase','outer_fold','category','message'])
    scale_rows=[]
    for response in ['L_mean','L_trim']:
        for left,right in combinations(['test_1M','test_60M','test_1B'],2):
            a={key:i for i,key in enumerate(signatures(datasets[left]['P']))};b={key:i for i,key in enumerate(signatures(datasets[right]['P']))};shared=sorted(set(a)&set(b))
            yl=np.array([datasets[left][response][a[key]] for key in shared]);yr=np.array([datasets[right][response][b[key]] for key in shared])
            defined=len(shared)>1 and np.ptp(yl)>0 and np.ptp(yr)>0
            scale_rows.append({'response':response,'dataset_1':left,'dataset_2':right,'shared_recipe_n':len(shared),
                'actual_Pearson':float(pearsonr(yl,yr).statistic) if defined else None,'actual_Spearman':float(spearmanr(yl,yr).statistic) if defined else None,
                'interpretation':'direct observed ordering on matched recipes' if defined else 'no shared recipes; direct observed rank stability undefined'})
    write_rows(OUT/'cross_scale_recipe_rank_stability.csv',scale_rows)
    figure_outputs(validation,nested,predictions,names,train,settings)
    build_report(datasets,names,settings,nested,validation,final_rows,scale_rows,warning_rows,best,len(before))
    after=protect()
    if before!=after:raise ValueError('Original or frozen part 1/2 input changed')
    write_rows(OUT/'locked_inputs_integrity.csv',[{'path':p,'sha256_before':h,'sha256_after':after[p],'unchanged':True} for p,h in before.items()])
    (OUT/'modeling_config_snapshot.yaml').write_text(yaml.safe_dump(settings,sort_keys=False),encoding='utf-8')
    logging.info('COMPLETED part 3 stage 1; selected %s from train-only nested CV; no Q/K or 10B/70B prediction; frozen files %d',best,len(before))
    print(json.dumps({'best_training_CV_model':best,'nested_primary':[r for r in nested if r['response']=='L_mean'],
        'selected_validation':[r for r in validation if r['response']=='L_mean' and r['model']==best],'cross_scale':scale_rows},ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    try:main()
    except Exception:
        logging.exception('STOPPED modeling; no official or previously locked file is repaired or changed')
        raise

# ============================================================
# Source: src/mixture/effects.py
# Original lines: 1-109
# Purpose: 真实领域配比1个百分点扰动及联合边际效应
# ============================================================
"""Frozen-model compositional perturbations, distinct from causal effects."""
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
# Source: src/mixture/augmented.py
# Original lines: 1-73
# Purpose: Qp/Kp候选对照；最终没有替代一阶ElasticNet
# candidate / sensitivity model; Qp/Kp augmented models are not used as final main model.
# ============================================================
"""Training-fold ILR with fixed external quality/conflict descriptors."""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from .composition import ILRTransformer


class DescriptorILR(TransformerMixin,BaseEstimator):
    """Append Qp (and optionally Kp) of closed ORIGINAL shares, not replacement mass."""
    def __init__(self,q_values,k_values=None,zero_fraction=.5):
        self.q_values=q_values;self.k_values=k_values;self.zero_fraction=zero_fraction

    def fit(self,X,y=None):
        self.ilr_=ILRTransformer(self.zero_fraction).fit(X)
        if len(self.q_values)!=17 or (self.k_values is not None and len(self.k_values)!=17):
            raise ValueError('Descriptor domain order must contain exactly 17 values')
        return self

    def transform(self,X):
        closed=self.ilr_.closure(X)
        parts=[self.ilr_.transform(X), (closed@np.asarray(self.q_values))[:,None]]
        if self.k_values is not None: parts.append((closed@np.asarray(self.k_values))[:,None])
        result=np.concatenate(parts,axis=1)
        if not np.isfinite(result).all(): raise ValueError('Nonfinite augmented feature')
        return result


def candidate(q_values,k_values,settings):
    """Same first-order ElasticNet/scaling and grid as frozen M1, extra descriptors."""
    model=TransformedTargetRegressor(regressor=Pipeline([
        ('features',DescriptorILR(tuple(q_values),None if k_values is None else tuple(k_values),settings['zero_replacement_fraction_of_training_min'])),
        ('scale',StandardScaler()),
        ('regression',ElasticNet(max_iter=settings['elastic_max_iter'],tol=settings['elastic_tolerance'],selection='cyclic'))
    ]),transformer=StandardScaler())
    grid={'regressor__regression__alpha':settings['elastic_alphas'], 'regressor__regression__l1_ratio':settings['elastic_l1_ratios']}
    return model,grid


def tune(q_values,k_values,P,y,settings,folds):
    """Select exclusively on supplied training folds, never on validation/estimated Loss."""
    model,grid=candidate(q_values,k_values,settings)
    search=GridSearchCV(model,grid,cv=folds,scoring='neg_root_mean_squared_error',refit=True,n_jobs=1,error_score='raise')
    search.fit(P,y)
    checked(search.best_estimator_)
    return search.best_estimator_,search.best_params_,[
        dict(parameters=p,mean_CV_RMSE=float(-search.cv_results_['mean_test_score'][i]),
             std_CV_RMSE=float(search.cv_results_['std_test_score'][i])) for i,p in enumerate(search.cv_results_['params'])]


def checked(model):
    """Stop if the selected fit failed its numerical convergence budget."""
    regression=model.regressor_.named_steps['regression']
    if regression.n_iter_>=regression.max_iter: raise RuntimeError('Augmented model did not converge')


def metadata(model,parameters,indices,names,label):
    """Persist all coefficients/scales and original-unit descriptor effects."""
    p=model.regressor_;f=p.named_steps['features'];s=p.named_steps['scale'];r=p.named_steps['regression']
    sy=float(model.transformer_.scale_[0]);raw=sy*r.coef_/s.scale_
    return dict(model=label,parameters=parameters,n_fit=len(indices),fitting_recipe_indices=list(indices),
        fitting_dataset='A4/A5_only',domain_order=list(names),q_values=list(f.q_values),
        k_values=None if f.k_values is None else list(f.k_values),zero_delta=f.ilr_.zero_delta_,
        training_min_positive=f.ilr_.min_positive_training_,ilr_basis=f.ilr_.basis_.tolist(),
        X_mean=s.mean_.tolist(),X_scale=s.scale_.tolist(),coefficients=r.coef_.tolist(),
        intercept=float(r.intercept_),y_mean=float(model.transformer_.mean_[0]),y_scale=sy,
        gamma_original_loss_per_Qp=float(raw[16]),gamma_standardized=float(r.coef_[16]),
        eta_original_loss_per_Kp=None if f.k_values is None else float(raw[17]),
        n_nonzero_coefficients=int(np.count_nonzero(r.coef_)),n_iter=int(r.n_iter_))

# ============================================================
# Source: scripts/09_analyze_mixture_effects.py
# Original lines: 1-597
# Purpose: 最终阶段实际执行入口：09_analyze_mixture_effects.py
# M3_Qp / M4_Qp_Kp are candidate / sensitivity models, not final main model. The saved final_model_policy selects M1_ElasticNet.
# ============================================================
"""Question 1 part 3 final: frozen-model effects and train-only Qp/Kp validation.

Never rerun prior stages, alter attachments, or tune against test/estimated Loss.
All new model fitting uses A4/A5. Long Monte Carlo runs have a separate checkpoint.
"""
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


def read_csv(path):
    """Strict nonempty CSV parse; no silent missing data substitutions."""
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);rows=list(reader)
    if not rows or any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f'Empty/malformed CSV: {path}')
    return rows


def dump(path,data):
    """Write finite JSON and automatically create its directory."""
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')


def short(name):
    """Display only; saved schema retains exact official column names."""
    return name.removeprefix('train_the_pile_').removeprefix('metric/the_pile_').removesuffix('_val_loss')


def capture_protected():
    """Snapshot every existing original/prior result and frozen source before work.

    A subsequent run reuses the original manifest, so it cannot legitimize an
    accidental change to an old file. New outputs are not part of this snapshot.
    """
    manifest=STAGE/'protected_before.json'
    if manifest.exists():
        snapshot=json.loads(manifest.read_text(encoding='utf-8'))
        verify_protected(snapshot)
        return snapshot
    paths=[]
    for folder in ['real_attachments','outputs','configs','src','scripts']:
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.is_relative_to(STAGE):continue
            if p.name in ['mixture_analysis.yaml','effects.py','augmented.py'] or p.name.startswith('09_'):continue
            paths.append(p)
    snapshot={p.relative_to(ROOT).as_posix():sha256_file(p) for p in sorted(paths)}
    dump(manifest,snapshot)
    return snapshot


def verify_protected(snapshot):
    """Stop before publishing if any already locked file changed."""
    rows=[]
    for relative,digest in snapshot.items():
        path=ROOT/relative
        current=sha256_file(path) if path.exists() else 'MISSING'
        rows.append(dict(path=relative,before_sha256=digest,after_sha256=current,unchanged=current==digest))
        if current!=digest: raise RuntimeError(f'Locked input/result changed: {relative}')
    write_rows(STAGE/'locked_files_integrity.csv',rows)
    return len(rows)


def frozen_oof(train):
    """Align already frozen M1 OOF predictions by index, without refitting M1."""
    records=read_csv(OUT/'nested_OOF_predictions.csv')
    lookup={r['index']:r for r in records if r['response']=='L_mean' and r['model']=='M1_ElasticNet'}
    if len(lookup)!=len(train['indices']):raise ValueError('Frozen OOF index mismatch')
    if any(abs(float(lookup[i]['true_loss'])-train['L_mean'][k])>1e-12 for k,i in enumerate(train['indices'])):
        raise ValueError('Frozen OOF response mismatch')
    return np.array([float(lookup[i]['predicted_loss']) for i in train['indices']])


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


def mapping(names,settings):
    """Retain A16 original '(none)' while labeling researcher proxy assumptions."""
    guide=read_csv(ROOT/'real_attachments/A_data_value/domain_mapping_guide.csv')
    guide={r['mixture_domain']:r for r in guide}
    quality={r['domain']:r for r in read_csv(ROOT/'outputs/quality/domain_quality_summary_A1.csv')}
    rows=[]
    for name in names:
        domain=short(name);original=guide[domain];inferred=original['mapping_type']=='inferred'
        if inferred:
            if original['quality_domain']!='(none)':raise ValueError('Changed A16 inferred schema; inspect before proceeding')
            proxy=settings['inferred_mapping'][domain];qd=proxy['quality_domain'];rationale=proxy['rationale']
        else:qd=original['quality_domain'];rationale=original['note']
        q=quality[qd]
        rows.append(dict(mixture_domain=name,mixture_domain_short=domain,quality_domain=qd,
            mapping_type=original['mapping_type'],A16_quality_domain=original['quality_domain'],A16_note=original['note'],
            researcher_assumption=inferred,association_status='assumed_proxy' if inferred else original['mapping_type'],
            mapping_rationale=rationale,Q_domain=float(q['Q_domain']),Q_CI_low=float(q['ci95_low']),Q_CI_high=float(q['ci95_high']),
            Q_source='outputs/quality/domain_quality_summary_A1.csv',quality_n=int(q['n'])))
    if len(rows)!=17:raise ValueError('Mapping must contain exactly 17 domains')
    write_rows(OUT/'domain_quality_mapping.csv',rows)
    return rows,quality


def original_k_domains():
    """Aggregate existing K only; no K recomputation, normalization or penalty."""
    values=defaultdict(list)
    with (ROOT/'outputs/conflict/A1_sample_conflict.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):values[r['domain']].append(float(r['K']))
    if sum(map(len,values.values()))!=51230:raise ValueError('Unexpected locked A1 K record count')
    rows=[dict(domain=d,n=len(v),K_domain=float(trim_mean(v,.1)),trim_fraction_per_tail=.1,
               source='outputs/conflict/A1_sample_conflict.csv') for d,v in sorted(values.items())]
    write_rows(STAGE/'domain_conflict_trimmed_summary_A1.csv',rows)
    return {r['domain']:r['K_domain'] for r in rows}


def descriptors(datasets,q,k):
    """Export fixed Qp/Kp derived from original closed shares; K is descriptive."""
    rows=[]
    for label,d in datasets.items():
        closed=ILRTransformer.closure(d['P']);qp=closed@q;kp=closed@k
        for i,index in enumerate(d['indices']):
            rows.append(dict(dataset=label,index=index,loss_provenance=d['provenance'],Q_p=float(qp[i]),K_p=float(kp[i]),
                composition_used='closed_original_shares',mapping_scheme='A_assumed_proxy'))
    write_rows(OUT/'recipe_quality_conflict_descriptors.csv',rows)


def nested_augmented(label,q,k,train,names,settings,outer,store=True):
    """Matched outer folds, inner fold preprocessing, plus final train-only selection."""
    P,y=train['P'],train['L_mean'];ids=train['indices'];oof=np.empty(len(y));outer_models=[];metas=[];grid_rows=[];fold_rows=[]
    for fold,(tr,va) in enumerate(outer,1):
        inner=group_folds(P[tr],settings['inner_folds'],settings['random_state']+fold)
        model,params,grid=tune(q,k,P[tr],y[tr],settings,inner)
        oof[va]=model.predict(P[va]);outer_models.append((tr,va,params,model))
        metas.append(metadata(model,params,[ids[i] for i in tr],names,label+f'_outer{fold}'))
        fold_rows.append(dict(model=label,outer_fold=fold,**scores(y[va],oof[va])))
        grid_rows += [dict(model=label,outer_fold=fold,**r) for r in grid]
        logging.info('%s outer fold %d/5 RMSE %.6f',label,fold,fold_rows[-1]['RMSE'])
    folds=group_folds(P,settings['final_selection_folds'],settings['random_state']+100)
    model,params,grid=tune(q,k,P,y,settings,folds)
    final_meta=metadata(model,params,ids,names,label)
    grid_rows += [dict(model=label,outer_fold='full_training',**r) for r in grid]
    if store:
        joblib.dump(model,STAGE/'models'/f'{label}.joblib')
        dump(STAGE/'models'/f'{label}.json',final_meta)
        dump(STAGE/'models'/f'{label}_outer.json',metas)
        write_rows(STAGE/f'{label}_hyperparameter_search.csv',grid_rows)
        write_rows(STAGE/f'{label}_OOF_predictions.csv',[dict(index=ids[i],true_loss=float(y[i]),predicted_loss=float(oof[i])) for i in range(len(y))])
    return dict(model=model,params=params,meta=final_meta,oof=oof,outer=outer_models,fold_rows=fold_rows,
                cv=dict(model=label,**scores(y,oof)))


def single_loss_models(train,bases,names,loss_names,settings):
    """26 lightweight first-order models; 5-fold training selection, no test selection."""
    folds=group_folds(train['P'],settings['final_selection_folds'],settings['random_state']+100)
    records=[];model_rows=[]
    x=np.stack([p for _,_,p in bases]);xp=np.stack([perturb(p,[j],.01) for _,_,p in bases for j in range(17)])
    for col,loss_name in enumerate(loss_names):
        y=train['losses'][:,col]
        for label in ['M1_ElasticNet','M1_Ridge']:
            model,params,grid=tune_base(label,train['P'],y,settings,folds)
            pipe=model.regressor_;reg=pipe.named_steps['regression']
            if getattr(reg,'n_iter_',None) is not None and reg.n_iter_>=reg.max_iter:
                raise RuntimeError('Single-loss model did not converge')
            pred=model.predict(x);effects=model.predict(xp).reshape(len(bases),17)-pred[:,None]
            for j,name in enumerate(names):
                records.append(dict(model=label,training_domain=name,validation_loss=loss_name,step=.01,
                    n_baselines=len(bases),**summary(effects[:,j]),baseline_direction=direction(effects[:,j])))
            model_rows.append(dict(validation_loss=loss_name,model=label,parameters=params,
                best_training_5fold_selection_RMSE=min(r['mean_CV_RMSE'] for r in grid),n_fit=len(y),
                zero_delta=pipe.named_steps['ilr'].zero_delta_,selection_dataset='A4/A5_only',
                performance_note='selection_CV_not_nested_unbiased_performance'))
            joblib.dump(model,STAGE/'single_loss_models'/f'{short(loss_name)}_{label}.joblib')
        logging.info('Single-loss diagnostic %d/13: %s',col+1,short(loss_name))
    for r in records:
        other=next(s for s in records if s['model']!=r['model'] and s['training_domain']==r['training_domain'] and s['validation_loss']==r['validation_loss'])
        r['models_mean_direction_agree']=bool(np.sign(r['mean_effect'])==np.sign(other['mean_effect']))
        r['stable_across_models_and_baselines']=r['baseline_direction']==other['baseline_direction'] and r['baseline_direction']!='uncertain'
    write_rows(OUT/'train_domain_vs_validation_loss_effects.csv',records)
    write_rows(STAGE/'single_loss_model_selection.csv',model_rows)
    return records


def monte_carlo(m3,mapping_rows,train,datasets,names,settings,analysis_settings,baseline_oof):
    """500 conditional refits, reusing A4/A5-selected hyperparameters per outer fold.

    Independent uniform draws for each inferred domain. Sampling ranges are
    proxy bootstrap CIs, not credible bounds for semantic mapping error.
    All fold scalers/zero parameters remain learned from their training folds.
    """
    checkpoint=STAGE/'mapping_MC_checkpoint.json'
    seed=analysis_settings['random_state'];n=analysis_settings['monte_carlo_repetitions']
    rng=np.random.default_rng(seed);base_q=np.array([r['Q_domain'] for r in mapping_rows]);inferred=np.array([r['researcher_assumption'] for r in mapping_rows])
    draws=np.tile(base_q,(n,1))
    for j,r in enumerate(mapping_rows):
        if inferred[j]:draws[:,j]=rng.uniform(r['Q_CI_low'],r['Q_CI_high'],size=n)
    write_rows(STAGE/'mapping_MC_draws.csv',[dict(repetition=i+1,**dict(zip(names,row))) for i,row in enumerate(draws)])
    rows=[];prediction_sum={label:np.zeros(len(d['P'])) for label,d in datasets.items()};prediction_square={label:np.zeros(len(d['P'])) for label,d in datasets.items()}
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text(encoding='utf-8'))
        if saved['draw_hash']!=sha256_file(STAGE/'mapping_MC_draws.csv'):raise ValueError('MC configuration changed; cannot resume')
        rows=saved['rows'];prediction_sum={k:np.array(v) for k,v in saved['prediction_sum'].items()};prediction_square={k:np.array(v) for k,v in saved['prediction_square'].items()}
        logging.info('Resuming conditional MC at %d/%d',len(rows),n)
    P,y=train['P'],train['L_mean'];base_cv=scores(y,baseline_oof)
    for i in range(len(rows),n):
        q=draws[i];oof=np.empty(len(y))
        for tr,va,params,_ in m3['outer']:
            model,_=candidate(q,None,settings);model.set_params(**params).fit(P[tr],y[tr]);checked(model)
            oof[va]=model.predict(P[va])
        final,_=candidate(q,None,settings);final.set_params(**m3['params']).fit(P,y);checked(final)
        meta=metadata(final,m3['params'],train['indices'],names,'M3_MC')
        row=dict(repetition=i+1,gamma=meta['gamma_original_loss_per_Qp'],CV_RMSE=scores(y,oof)['RMSE'],
                 CV_MAE=scores(y,oof)['MAE'],CV_R2=scores(y,oof)['R2'],CV_Spearman=scores(y,oof)['Spearman'])
        row['CV_RMSE_difference_from_M1']=row['CV_RMSE']-base_cv['RMSE']
        for label,d in datasets.items():
            pred=final.predict(d['P']);prediction_sum[label]+=pred;prediction_square[label]+=pred**2
            if label!='train_1M':
                s=scores(d['L_mean'],pred)
                row[label+'_RMSE']=s['RMSE'];row[label+'_Spearman']=s['Spearman']
        rows.append(row)
        if (i+1)%25==0 or i+1==n:
            dump(checkpoint,dict(draw_hash=sha256_file(STAGE/'mapping_MC_draws.csv'),rows=rows,
                prediction_sum={k:v.tolist() for k,v in prediction_sum.items()},prediction_square={k:v.tolist() for k,v in prediction_square.items()}))
            logging.info('Mapping conditional MC %d/%d',i+1,n)
    write_rows(OUT/'quality_mapping_monte_carlo_results.csv',rows)
    summary_rows=[]
    for key in rows[0]:
        if key=='repetition':continue
        values=np.array([r[key] for r in rows])
        summary_rows.append(dict(metric=key,repetitions=n,mean=float(values.mean()),std=float(values.std(ddof=1)),
            min=float(values.min()),P025=float(np.quantile(values,.025)),median=float(np.median(values)),P975=float(np.quantile(values,.975)),max=float(values.max())))
    write_rows(OUT/'quality_mapping_monte_carlo_summary.csv',summary_rows)
    prediction_rows=[]
    for label,d in datasets.items():
        mean=prediction_sum[label]/n;std=np.sqrt(np.maximum(0,prediction_square[label]/n-mean**2))
        prediction_rows += [dict(dataset=label,index=index,mean_prediction=float(mean[i]),std_prediction=float(std[i])) for i,index in enumerate(d['indices'])]
    write_rows(STAGE/'mapping_MC_prediction_summary.csv',prediction_rows)
    return rows,summary_rows


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


def figures(names,loss_names,effects,stability,single,cv,validation,mc):
    """Standalone scientific plots; effects are signed changes in uncalibrated Loss."""
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,path):
        fig.tight_layout();fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,9),sharey=True)
    for ax,model in zip(axes,['M1_ElasticNet','M1_Ridge']):
        rows=[next(r for r in effects if r['model']==model and r['step']==.01 and r['mixture_domain']==n) for n in names]
        means=np.array([r['mean_effect'] for r in rows]);lo=np.array([r['min_effect'] for r in rows]);hi=np.array([r['max_effect'] for r in rows])
        ax.barh(np.arange(17),means,color=['#4777a8' if m<0 else '#b77755' for m in means])
        ax.errorbar(means,np.arange(17),xerr=np.stack([means-lo,hi-means]),fmt='none',ecolor='#555555',capsize=3)
        ax.axvline(0,color='#777777',lw=.8);ax.set(yticks=np.arange(17),yticklabels=[short(n) for n in names],
            title=model,xlabel='Predicted Loss change (+1 pp); bars = mean, whiskers = baseline range')
    fig.suptitle('Five A4 baselines; associations, not causal effects');save(fig,FIG/'marginal_effects.png')
    matrices=[]
    for model in ['M1_ElasticNet','M1_Ridge']:
        matrix=np.array([[next(r['mean_effect'] for r in single if r['model']==model and r['training_domain']==n and r['validation_loss']==loss) for loss in loss_names] for n in names]);matrices.append(matrix)
    vmax=max(abs(x).max() for x in matrices)
    fig,axes=plt.subplots(1,2,figsize=(18,11),sharey=True)
    for ax,matrix,label in zip(axes,matrices,['ElasticNet','Ridge']):
        im=ax.imshow(matrix,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
        ax.set(xticks=range(13),xticklabels=[short(n) for n in loss_names],yticks=range(17),yticklabels=[short(n) for n in names],
            title=f'{label}: +1 percentage point',xlabel='Validation Loss domain',ylabel='Training mixture domain')
        ax.tick_params(axis='x',rotation=65)
    fig.subplots_adjust(left=.12,right=.88,bottom=.22,wspace=.16,top=.92)
    color_axis=fig.add_axes([.91,.3,.016,.52])
    fig.colorbar(im,cax=color_axis,label='Mean predicted Loss change')
    path=ROOT/'outputs/figures/question1_part3/train_domain_validation_effect_heatmap.png'
    fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].bar([r['model'] for r in cv],[r['RMSE'] for r in cv],color='#4777a8');axes[0].set(ylabel='Nested OOF RMSE',title='A4/A5 only: matched outer folds')
    for model in ['M1_ElasticNet','M3_Qp','M4_Qp_Kp']:
        rows=[r for r in validation if r['model']==model]
        axes[1].plot(range(3),[r['Spearman'] for r in rows],'o-',label=model)
    axes[1].set(xticks=range(3),xticklabels=['1M','60M','1B'],ylabel='Spearman',title='Uncalibrated observed-table ranking');axes[1].legend()
    save(fig,FIG/'quality_conflict_model_comparison.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].hist([r['gamma'] for r in mc],bins=25,color='#4777a8');axes[0].set(xlabel='Gamma in original Loss / Qp units',ylabel='MC draws',title='500 inferred-proxy CI draws')
    axes[1].hist([r['CV_RMSE_difference_from_M1'] for r in mc],bins=25,color='#4777a8');axes[1].axvline(0,color='#777777');axes[1].set(xlabel='Nested OOF RMSE difference from frozen M1',title='Fixed training-selected hyperparameters')
    save(fig,FIG/'mapping_monte_carlo.png')


def table(rows,columns):
    """Markdown table with finite actual results, identifiers preserved."""
    def fmt(x):
        if x is None:return '未定义'
        if isinstance(x,(float,np.floating)):return f'{x:.6f}'
        return str(x).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |']+[
        '| '+' | '.join(fmt(r.get(c,'')) for c in columns)+' |' for r in rows])


def report(result):
    """Answer all 15 requested modeling questions with quantitative limitations."""
    r=result;stable=r['stability'];marginal=r['marginal'];joint=r['joint_stability']
    main=[dict(domain=short(n),**{k:next(a[k] for a in marginal if a['model']=='M1_ElasticNet' and a['step']==.01 and a['mixture_domain']==n) for k in ['mean_effect','min_effect','max_effect']},
        strict_direction=next(a['direction'] for a in stable if a['mixture_domain']==n)) for n in r['names']]
    stable_j=[a for a in joint if a['stable_deviation']]
    stable_j=sorted(stable_j,key=lambda a:abs(a['mean_deviation']),reverse=True)
    gamma=r['m3']['meta']['gamma_original_loss_per_Qp'];eta=r['m4']['meta']['eta_original_loss_per_Kp']
    def test(model,dataset,key):
        return next(a[key] for a in r['validation'] if a['model']==model and a['dataset']==dataset)
    def link(name):
        return next(a['Spearman'] for a in r['rank_links'] if a['comparison']==name)
    fold_rows=read_csv(STAGE/'model_comparison_outer_fold_scores.csv')
    fold_improved=sum(float(next(a['RMSE'] for a in fold_rows if a['model']=='M3_Qp' and a['outer_fold']==str(f)))<
                      float(next(a['RMSE'] for a in fold_rows if a['model']=='M1_ElasticNet' and a['outer_fold']==str(f))) for f in range(1,6))
    lines=['# 问题一第3小问最终报告','',
        f"最终主模型：**{r['selected_model']}**。第一小问 Q、第二小问 K 和上一轮 ILR/权重/模型文件均保持锁定。",'',
        '## 1. 输入、训练边界与冻结核验','',
        'A4/A5 为唯一训练数据；A6–A11 只作原值检验；A12–A15 的 10B/70B Loss 为估算值，且配方是训练集子集。没有用检验 Loss 调参数、截距或尺度。',
        f"实际配方数：{', '.join(k+'='+str(len(v['P'])) for k,v in r['datasets'].items())}。受保护文件 {r['protected_n']} 个，SHA-256 全部一致。",'',
        '已有真实同配方 1M/60M 的 256 条 L_mean：Spearman=0.9657980945296404。这是排序跨尺度稳定性的核心观测证据，不能把不同配方的 1B 表按 index 强行对应。','',
        '## 2. 为什么配比不能直接 OLS；ILR、零配比与主模型','',
        '17 配比和为 1，原始配比加截距的 OLS 存在精确共线性；删去一域虽可计算，但解释依赖参考域且忽略相对比例几何。闭合后采用 16 维正交 Helmert ILR，ILR(p)=H log(p)。',
        '零项按训练折最小正闭合配比的一半进行乘法替换，非零项同比收缩，保持正比值。主模型沿用上一轮固定参数：ElasticNet alpha=0.03、l1_ratio=0.1；Ridge alpha=10 作数值对照。新模型中的 ILR 零参数、X/y 标准化均在每个 CV 训练折内拟合。',
        '这是数值正则化，不代表结构零对应真实正质量。ILR 中心由冻结零替换后的几何中心求得，分量中位数基准由原闭合配比求得。',
        'ElasticNet 已按人工确认锁定为主模型，提供收缩与有限稀疏性；Ridge 上轮 CV RMSE 略低，故不声称 ElasticNet 数值最优。上轮二阶模型没有获得嵌套 CV 收益，本轮没有恢复二阶主模型。','',
        '## 3. 17 域边际关联与稳定性','',
        '基准为 A4 ILR 中心、闭合分量中位数、冻结 ElasticNet 预测 P10/P50/P90 的真实配方，共五个。基准不使用检验 Loss 选择。对目标域增加绝对 0.005/0.01/0.02（0.5/1/2 个百分点），其余域按原比例缩减。',
        'Delta L_j=f(p perturb j)-f(p)。以下为 ElasticNet 增加 1 个百分点的五基准均值与范围。strict_direction 只有两个模型、三个步长、五个基准共 30 个效应全部同向才成立；uncertain 不给总体有利/不利结论。数值零容差 1e-10 仅防浮点误判，不是显著性阈值。','',table(main,['domain','mean_effect','min_effect','max_effect','strict_direction']),'',
        '这些是相对于按比例减小其余领域的模型关联，不是因果作用或孤立单域贡献。均值是五个代表基准的等权摘要，不是全体 A4 配方的平均处理效应。零激活效应包含从替换质量到实际正配比的变化，不能当作在 0 处的导数。','',
        '## 4. 双域联合扰动与组合偏离','',
        '同时给 j,k 各增加相同步长，其余领域按比例缩减。I_jk=Delta L_jk-Delta L_j-Delta L_k，仅称组合偏离指数。即使一阶 ILR 模型也因闭合和对数变换产生非加性，不能据此证明真实交互或因果互补。',
        f"136 对中严格同向组合偏离 {len(stable_j)} 对：负向 {sum(a['direction']=='lower_loss' for a in stable_j)}，正向 {sum(a['direction']=='higher_loss' for a in stable_j)}。完整结果在 domain_joint_effects.csv、stage2/joint_effect_stability.csv。以下仅列绝对平均偏离最大的同向对：",'',
        table([dict(domain_1=short(a['domain_1']),domain_2=short(a['domain_2']),**{k:a[k] for k in ['direction','mean_deviation','min_deviation','max_deviation']}) for a in stable_j[:12]],['domain_1','domain_2','direction','mean_deviation','min_deviation','max_deviation']),'',
        '负向同向对可描述为扰动情景下的互补趋势，正向为竞争趋势。小偏离未做抽样显著性检验，不把数值同号等同于稳定的真实交互证据。','',
        '## 5. 13 个验证域的异质性','',
        '对 A5 的每个验证 Loss 分别在 A4/A5 内训练一阶 ElasticNet/Ridge；五折选择超参数，得到 26 个辅助诊断模型，没有用检验表选择。选择 CV 的最小 RMSE 不是无偏嵌套泛化评估。',
        f"17×13 共 221 个训练域/验证域单元中，两模型均值方向不一致 {r['single_disagree']} 个；ElasticNet 在不同验证域出现正负混合方向的训练领域 {r['single_mixed_domains']} 个。因此宏观 L_mean 会掩盖部分领域差异。矩阵和两模型热图已输出。",'',
        '## 6. 17 域质量映射与 Qp','',
        'Qp=sum(p_j Q_j)，使用原配比闭合值，不使用零替换后的虚拟质量。Q_j 读取第一小问已锁定的 A1 各领域 10% 双侧截尾均值及 bootstrap 95% CI。Qp 是配比 p 的确定函数，是外部质量评价构造的质量加权配比特征，不是独立实验变量。',
        '**A16 只有 3 个 direct、3 个 near_direct 的确定关联，11 个 inferred 的 quality_domain 实际为 (none)。Scheme A 对这 11 个添加有明确理由的研究者代理假设，原 A16 字段、类型、note 都保留。绝不能把代理 Q_j 写成该语料实测质量。**','',
        table(r['mapping'],['mixture_domain_short','A16_quality_domain','quality_domain','mapping_type','Q_domain','Q_CI_low','Q_CI_high']),'',
        '## 7. M3 的嵌套 CV 与直接检验','',
        'M3=一阶 ILR ElasticNet + Qp。沿用原模型标准化与同一 ElasticNet 网格；五个外折严格复用上一轮划分，各外训练内再四折选参，最后仅全 A4/A5 五折选最终超参数。M1 使用冻结的原 OOF 预测，没有重调。',
        f"M3 最终参数：{r['m3']['params']}。gamma 原 Loss/Qp 单位={gamma:.9f}；还原公式 gamma=s_y beta_Qp/s_Qp。该系数受确定性配比特征与正则化共线性影响，不具有因果解释。",'',
        table(r['cv'],['model','RMSE','MAE','R2','Pearson','Spearman']),'',
        table(r['validation'],['model','dataset','RMSE','MAE','Pearson','Spearman']),'',
        f"最终判据：{r['policy']['reason']}。Qp {'进入最终模型' if r['selected_model']=='M3_Qp' else '保留为辅助解释特征，不进入最终主模型'}。全部预测未用 test bias 校准。",
        f"M3 相对 M1 的嵌套 CV RMSE 降幅 {(r['cv'][0]['RMSE']-r['cv'][1]['RMSE'])/r['cv'][0]['RMSE']*100:.4f}%。改善外折 {fold_improved}/5；1M test MAE 差 {test('M3_Qp','test_1M','MAE')-test('M1_ElasticNet','test_1M','MAE'):+.9f}。综合改善幅度和外折变化，本轮不足以证明质量特征具有稳定的实用增益。保守规则要求各外折方向一致，属于操作性稳定标准，不是统计显著性检验。",'',
        '## 8. 映射不确定性：CI 扰动与替代语义代理','',
        'Scheme B：seed=2026，对每个 inferred 域独立 Uniform(Q_CI_low,Q_CI_high) 抽样 500 次，direct/near_direct 固定。每次重算 Qp、用 Scheme A 已由训练 CV 选定的各外折和全训练超参数重拟合 M3，重新计算 OOF 与检验预测。',
        '这是条件于已选择超参数的 Monte Carlo，不重复选择超参数，不包含模型选择不确定性。独立抽样假设允许同代理域的不同 inferred 域有所区别，未建模来源相关性。',
        f"500 次 gamma>0 比例 {r['mc_gamma_positive_rate']:.6f}；gamma<0 比例 {r['mc_gamma_negative_rate']:.6f}；OOF RMSE 优于冻结 M1 的比例 {r['mc_cv_improve_rate']:.6f}。",'',
        table([a for a in r['mc_summary'] if a['metric'] in ['gamma','CV_RMSE_difference_from_M1','test_1M_RMSE','test_1M_Spearman','test_60M_Spearman','test_1B_Spearman']],['metric','mean','std','P025','median','P975']),'',
        'CI 仅刻画已有代理质量域截尾均值的抽样波动，不包含“语义关联选错”的偏差，不能由窄 CI 下稳定推出映射可靠。另设未经检验 Loss 选择的替代语义代理组合，在同一 A4/A5 嵌套 CV 内分析：','',
        table(r['alternative'],['scheme','gamma_original_loss_per_Qp','CV_RMSE','CV_MAE','CV_R2','CV_Spearman']),'',
        f"替代代理下 gamma 相对 Scheme A 比值 {r['alternative'][1]['gamma_original_loss_per_Qp']/gamma:.6f}；替代方案相对 M1 的 CV RMSE 差为 {r['alternative'][1]['CV_RMSE']-r['cv'][0]['RMSE']:+.9f}。因此 CI 内稳定与代理语义变化稳定必须分开，不能由前者推断后者具有相同增益幅度。",
        '替代映射列出在 stage2/alternative_domain_quality_mapping.csv。各方案不按 test 结果择优替换主映射。直接映射也只代表来源域名对应，不证明 A1 文本质量均值准确代表 A4 中全部同名训练语料。','',
        '## 9. Kp 仅作辅助检查','',
        '读取已有 A1_sample_conflict.csv 的 K，分别按原领域作两端各 10% 截尾均值，不重新计算 K。通过相同显式代理映射形成 K_j 和 Kp=sum(p_j K_j)。M4=M3+eta Kp，仍在 A4/A5 内嵌套 CV。',
        f"M4 eta 原 Loss/Kp 单位={eta:.9f}，与 M3 的 OOF RMSE 差为 {r['m4']['cv']['RMSE']-r['m3']['cv']['RMSE']:+.9f}。Kp {'未获得 CV 改善，舍弃' if r['m4']['cv']['RMSE']>=r['m3']['cv']['RMSE'] else '仅保留辅助敏感性结果；CV 获得改善，但泛化检验未稳定复现'}，不进入本轮最终主模型。K 是评价冲突，不是低质量程度，未作为惩罚项。",
        f"M4 的 1M test RMSE={test('M4_Qp_Kp','test_1M','RMSE'):.6f}，M3={test('M3_Qp','test_1M','RMSE'):.6f}；1B Spearman M4={test('M4_Qp_Kp','test_1B','Spearman'):.6f}，M3={test('M3_Qp','test_1B','Spearman'):.6f}。eta 符号由训练拟合自由决定，不能解释为质量惩罚。M4 gamma={r['m4']['meta']['gamma_original_loss_per_Qp']:.6f}，加入 Kp 后系数幅度变化，提示确定性配比描述符间的依赖与正则化分配影响系数，不能解释为独立因果贡献。", '',
        '## 10. 尺度迁移诊断，不做偏置校准','',
        '残差 e=y-yhat；centered RMSE=sqrt(mean((e-mean(e))^2)) 仅解释偏移贡献。它没有提供可在未知目标尺度应用的偏置，不能称为校准后的预测性能。', '',
        table(r['scale'],['model','dataset','mean_bias','median_bias','residual_std','RMSE','centered_RMSE_diagnostic_only','Pearson','Spearman']),'',
        '绝对误差跨尺度显著增大，去除样本均值残差后的诊断误差远小于原 RMSE，表明整体 Loss 水平偏移是主要来源；正排序相关仍显示有限的关联迁移。60M 与 1B 的配方集合不同，不能把其相关大小直接解读为规模单调变化。','',
        '## 11. 10B/70B 外推一致性与证据边界','',
        '所有超参数选择在查看估算性能之前完成并冻结；按闭合配比向量匹配训练子集，不按跨表 index 假定相同配方。10B、70B 各 63 个配方均属于 A4，因此只称估算训练配方子集的外推一致性检验。', '',
        table(r['estimated'],['dataset','model','RMSE','MAE','Pearson','Spearman']),'',
        table(r['rank_links'],['comparison','n','Spearman','diagnostic_ratio_estimated_over_1M','diagnostic_max_abs_scaling_deviation']),'',
        f"独立核验后的实际结论：主模型与 10B/70B 估算宏观 Loss 的 Spearman 分别为 {link('M1_ElasticNet_prediction_vs_estimated_10B'):.6f}/{link('M1_ElasticNet_prediction_vs_estimated_70B'):.6f}，排序发生反向关联，**不支持主模型在这两张估算表上的排序迁移**。M3 同样没有修复反向关系。10B 与 70B 彼此 Spearman={link('estimated_10B_vs_estimated_70B_same_recipes'):.6f}，仅说明两张估算表之间排序相近，不能代替“1M 模型预测与大规模估算表一致”。",
        f"同配方 1M 实测宏观 Loss 与 10B/70B 估算宏观 Loss 的 Spearman 分别为 {link('observed_train_1M_vs_estimated_10B'):.6f}/{link('observed_train_1M_vs_estimated_70B'):.6f}，说明反向关系并非只由模型拟合误差造成。配方已按向量确认一致；但这些是估算值，不能据此断言真实大模型发生相同反转。",
        '逐验证域关系与宏观均值关系不同，估算值也不是原 1M 宏观 Loss 的简单正比例缩放。生成关系诊断在 stage2/estimated_*_generation_relationship.csv；其中比例系数仅是对已有表的诊断数，没有用于拟合、校准或构造新 Loss。现有附件不足以识别估算生成机制，不能把诊断比例写成已知 Scaling Law。','',
        '## 12. 最终选择、局限性与复用文件','',
        f"主模型采用 {r['selected_model']}；Ridge 保留数值对照。原 Q 与 K 分别表示综合质量与评价冲突。参数与选择理由记录在 stage2/final_model_policy.json，冻结阶段清单记录在 stage2/freeze_before_extrapolation.json。", 
        '局限性：512 个小规模训练配方、较多零配比、对零激活与基准敏感；17 域与 13 验证域不齐；没有因果实验识别；新增代理映射缺直接质量观测；Qp/Kp 均由 p 确定且相互相关；嵌套 CV 差异未证明显著；估算表不是外部观测，且配方属于训练子集。',
        '复用：domain_marginal_effects.csv、domain_effect_stability.csv、domain_joint_effects.csv、train_domain_vs_validation_loss_effects.csv、domain_quality_mapping.csv、recipe_quality_conflict_descriptors.csv、stage2/model_comparison_nested_CV.csv、stage2/model_comparison_validation.csv、estimated_scale_consistency.csv。',
        '**本轮完成后停止，未开始问题二，未重新拟合前两小问结果。**','']
    (ROOT/'outputs/report/question1_part3_final.md').write_text('\n\n'.join(lines),encoding='utf-8')


def refresh_report_only(refresh_figures=False):
    """Refresh prose from completed saved results; no fitting or parameter changes."""
    datasets,names,loss_names=load_datasets(ROOT)
    summary=json.loads((STAGE/'execution_summary.json').read_text(encoding='utf-8'))
    policy=json.loads((STAGE/'final_model_policy.json').read_text(encoding='utf-8'))
    def numeric_rows(path):
        records=read_csv(path)
        for row in records:
            for key,value in row.items():
                if value=='True':row[key]=True
                elif value=='False':row[key]=False
                elif value=='':row[key]=None
                else:
                    try:row[key]=float(value)
                    except ValueError:pass
        return records
    cv=summary['cv'];m3=dict(meta=json.loads((STAGE/'models/M3_Qp.json').read_text(encoding='utf-8')),cv=cv[1])
    m3['params']=m3['meta']['parameters'];m4=dict(meta=json.loads((STAGE/'models/M4_Qp_Kp.json').read_text(encoding='utf-8')),cv=cv[2])
    mc=numeric_rows(OUT/'quality_mapping_monte_carlo_results.csv')
    single=numeric_rows(OUT/'train_domain_vs_validation_loss_effects.csv');en=[r for r in single if r['model']=='M1_ElasticNet']
    result=dict(names=names,datasets=datasets,marginal=numeric_rows(OUT/'domain_marginal_effects.csv'),
        stability=numeric_rows(OUT/'domain_effect_stability.csv'),joint_stability=numeric_rows(STAGE/'joint_effect_stability.csv'),
        mapping=numeric_rows(OUT/'domain_quality_mapping.csv'),m3=m3,m4=m4,cv=cv,validation=summary['validation'],
        scale=numeric_rows(OUT/'scale_transfer_diagnostics.csv'),alternative=numeric_rows(OUT/'quality_proxy_mapping_sensitivity.csv'),
        mc_summary=numeric_rows(OUT/'quality_mapping_monte_carlo_summary.csv'),
        mc_gamma_positive_rate=float(np.mean([r['gamma']>0 for r in mc])),mc_gamma_negative_rate=float(np.mean([r['gamma']<0 for r in mc])),
        mc_cv_improve_rate=float(np.mean([r['CV_RMSE_difference_from_M1']<0 for r in mc])),
        single_disagree=sum(not r['models_mean_direction_agree'] for r in en),
        single_mixed_domains=sum(direction([r['mean_effect'] for r in en if r['training_domain']==n])=='uncertain' for n in names),
        estimated=summary['estimated'],rank_links=summary['rank_links'],protected_n=summary['protected_files_unchanged'],
        selected_model=summary['selected_model'],policy=policy)
    report(result)
    if refresh_figures:
        figures(names,loss_names,result['marginal'],result['stability'],single,cv,result['validation'],mc)
    verify_protected(json.loads((STAGE/'protected_before.json').read_text(encoding='utf-8')))
    print('Report refreshed from existing results; no fitting.')


def main():
    """Run the authorized final round sequentially, fail loudly on inconsistent inputs."""
    if (STAGE/'execution_summary.json').exists():
        raise RuntimeError('This round is complete and frozen. Use --report-only or --figures-only; do not refit completed models.')
    for p in [STAGE,STAGE/'models',STAGE/'single_loss_models',FIG]:p.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(STAGE/'analysis.log',encoding='utf-8')],force=True)
    settings=yaml.safe_load((ROOT/'configs/mixture_modeling.yaml').read_text(encoding='utf-8'))
    a=yaml.safe_load((ROOT/'configs/mixture_analysis.yaml').read_text(encoding='utf-8'))
    protected=capture_protected();logging.info('Protected %d original/locked files',len(protected))
    datasets,names,loss_names=load_datasets(ROOT);train=datasets['train_1M'];y=train['L_mean'];P=train['P']
    models={label:joblib.load(OUT/'models'/f'L_mean_{label}.joblib') for label in ['M1_ElasticNet','M1_Ridge']}
    # Merely applying already frozen predictors; no re-fitting/re-scaling.
    scale=scale_diagnostics(datasets,models)
    bases=baselines(P,train['indices'],models['M1_ElasticNet'])
    write_rows(STAGE/'representative_baselines.csv',[dict(baseline=label,recipe_index=index,
        frozen_EN_prediction=float(models['M1_ElasticNet'].predict(p[None,:])[0]),**dict(zip(names,p))) for label,index,p in bases])
    singles,marginal,stability,joint_cases,joints,joint_stability=analyze(models,bases,names,a['perturbation_steps'],a['numerical_sign_tolerance'])
    write_rows(STAGE/'domain_marginal_effect_cases.csv',singles);write_rows(OUT/'domain_marginal_effects.csv',marginal)
    write_rows(OUT/'domain_effect_stability.csv',stability);write_rows(STAGE/'domain_joint_effect_cases.csv',joint_cases)
    write_rows(OUT/'domain_joint_effects.csv',joints);write_rows(STAGE/'joint_effect_stability.csv',joint_stability)
    logging.info('Frozen effects: %d stable domains, %d stable joint deviations',sum(r['stable_direction'] for r in stability),sum(r['stable_deviation'] for r in joint_stability))
    single=single_loss_models(train,bases,names,loss_names,settings)
    mapping_rows,quality=mapping(names,a);q=np.array([r['Q_domain'] for r in mapping_rows])
    outer=group_folds(P,settings['outer_folds'],settings['random_state'])
    assignments={r['index']:int(r['outer_holdout_fold']) for r in read_csv(OUT/'outer_fold_assignments.csv')}
    for fold,(_,va) in enumerate(outer,1):
        if any(assignments[train['indices'][i]]!=fold for i in va):raise ValueError('Outer folds differ from frozen M1')
    base_oof=frozen_oof(train);base_cv=dict(model='M1_ElasticNet',**scores(y,base_oof))
    logging.info('Starting M3 nested CV on A4/A5 only')
    m3=nested_augmented('M3_Qp',q,None,train,names,settings,outer)
    # Conditional inferred-quality CI sensitivity precedes any Kp model.
    observed={label:d for label,d in datasets.items() if d['provenance']=='observed'}
    mc,mc_summary=monte_carlo(m3,mapping_rows,train,observed,names,settings,a,base_oof)
    alternative_mapping=[]
    q_alt=q.copy()
    for j,r in enumerate(mapping_rows):
        domain=short(names[j]);qd=a['alternative_inferred_mapping'][domain] if r['researcher_assumption'] else r['quality_domain']
        q_alt[j]=float(quality[qd]['Q_domain']);alternative_mapping.append(dict(mixture_domain=names[j],mapping_type=r['mapping_type'],
            quality_domain=qd,Q_domain=q_alt[j],researcher_assumption=r['researcher_assumption']))
    write_rows(STAGE/'alternative_domain_quality_mapping.csv',alternative_mapping)
    alt=nested_augmented('M3_alternative_mapping',q_alt,None,train,names,settings,outer)
    alternative=[dict(scheme='A_assumed_proxy',gamma_original_loss_per_Qp=m3['meta']['gamma_original_loss_per_Qp'],
        **{'CV_'+k:m3['cv'][k] for k in ['RMSE','MAE','R2','Spearman']}),
        dict(scheme='alternative_assumed_proxy',gamma_original_loss_per_Qp=alt['meta']['gamma_original_loss_per_Qp'],
        **{'CV_'+k:alt['cv'][k] for k in ['RMSE','MAE','R2','Spearman']})]
    write_rows(OUT/'quality_proxy_mapping_sensitivity.csv',alternative)
    # Now and only now aggregate fixed K and test M4.
    kd=original_k_domains();k=np.array([kd[r['quality_domain']] for r in mapping_rows])
    descriptors(datasets,q,k)
    write_rows(STAGE/'domain_quality_conflict_mapping.csv',[dict(**r,K_domain=float(k[j])) for j,r in enumerate(mapping_rows)])
    m4=nested_augmented('M4_Qp_Kp',q,k,train,names,settings,outer)
    cv=[base_cv,m3['cv'],m4['cv']];write_rows(STAGE/'model_comparison_nested_CV.csv',cv)
    folds=[]
    for fold,(_,va) in enumerate(outer,1):folds.append(dict(model='M1_ElasticNet',outer_fold=fold,**scores(y[va],base_oof[va])))
    folds+=m3['fold_rows']+m4['fold_rows'];write_rows(STAGE/'model_comparison_outer_fold_scores.csv',folds)
    validation=[];predictions=[]
    test_models={**models,'M3_Qp':m3['model'],'M4_Qp_Kp':m4['model']}
    for name,model in test_models.items():
        for label in ['test_1M','test_60M','test_1B']:
            d=datasets[label];pred=model.predict(d['P']);validation.append(dict(model=name,dataset=label,**scores(d['L_mean'],pred)))
            predictions += [dict(model=name,dataset=label,index=index,true_loss=float(d['L_mean'][i]),predicted_loss=float(pred[i])) for i,index in enumerate(d['indices'])]
    write_rows(STAGE/'model_comparison_validation.csv',validation);write_rows(STAGE/'model_comparison_test_predictions.csv',predictions)
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
    estimated,rank_links=extrapolation(datasets,{'M1_ElasticNet':models['M1_ElasticNet'],'M1_Ridge':models['M1_Ridge'],'M3_Qp':m3['model']})
    # Count heterogeneity based on the actual sensitivity matrices.
    en=[r for r in single if r['model']=='M1_ElasticNet']
    mixed=sum(direction([r['mean_effect'] for r in en if r['training_domain']==n])=='uncertain' for n in names)
    result=dict(names=names,datasets=datasets,marginal=marginal,stability=stability,joint_stability=joint_stability,
        mapping=mapping_rows,m3=m3,m4=m4,cv=cv,validation=validation,scale=scale,alternative=alternative,
        mc_summary=mc_summary,mc_gamma_positive_rate=float(np.mean([r['gamma']>0 for r in mc])),
        mc_gamma_negative_rate=float(np.mean([r['gamma']<0 for r in mc])),mc_cv_improve_rate=float(np.mean([r['CV_RMSE_difference_from_M1']<0 for r in mc])),
        single_disagree=sum(not r['models_mean_direction_agree'] for r in en),single_mixed_domains=mixed,
        estimated=estimated,rank_links=rank_links,protected_n=protected_n,selected_model=selected,policy=policy)
    figures(names,loss_names,marginal,stability,single,cv,validation,mc)
    report(result)
    protected_n=verify_protected(protected)
    for relative,digest in freeze.items():
        if sha256_file(ROOT/relative)!=digest:raise RuntimeError('Model changed after extrapolation freeze')
    dump(STAGE/'execution_summary.json',dict(selected_model=selected,protected_files_unchanged=protected_n,
        stable_domains=[short(r['mixture_domain']) for r in stability if r['stable_direction']],
        stable_joint_pairs=sum(r['stable_deviation'] for r in joint_stability),cv=cv,validation=validation,
        mapping_gamma=gamma if (gamma:=m3['meta']['gamma_original_loss_per_Qp']) else 0,
        MC_repetitions=len(mc),estimated=estimated,rank_links=rank_links))
    logging.info('COMPLETE: question1 part3 final; selected=%s; protected=%d; MC=%d; stopped before question2',selected,protected_n,len(mc))


if __name__=='__main__':
    try:
        if sys.argv[1:] in [['--report-only'],['--figures-only']]:
            refresh_report_only(sys.argv[1:]==['--figures-only'])
            sys.exit(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            main()
        write_rows(STAGE/'convergence_warnings.csv',[dict(category=w.category.__name__,message=str(w.message)) for w in caught],['category','message'])
    except Exception:
        logging.exception('Analysis failed; prior frozen files must remain unchanged')
        raise

