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
