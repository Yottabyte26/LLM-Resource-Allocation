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
