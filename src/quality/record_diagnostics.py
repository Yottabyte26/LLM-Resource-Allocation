"""Exact record fingerprints and raw-quality comparisons; never clean data."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real

import numpy as np

from .io import resolve_metric_value
from .vector_transform import parse_vector


def canonical_json(value: object) -> str:
    """Canonical raw JSON: ignore key order, retain strings/types and NaN tokens."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=True)


def raw_record_diagnostics(record: dict, metrics: list[dict]) -> dict:
    """Fingerprint one record while retaining no full content in the audit index.

    Full-record hash covers all original fields, including content and source
    metadata. The quality blob covers the 22 official raw fields with aliases
    resolved, presence states retained, numeric int/float equivalence, and NaN
    equality. Do not hash transformed argmax scores: distinct logits matter.
    """
    if len(metrics) != 22:
        raise ValueError("Exactly 22 raw quality fields are required")
    states, numeric_values, missing_metrics, grades = [], [], [], {}
    for metric in metrics:
        name = metric["name"]
        width = metric.get("expected_length", 1) if metric["kind"] == "list" else 1
        value, raw_name = resolve_metric_value(record, metric)
        if raw_name is None or value is None or value == []:
            states.append(0 if raw_name is None else 1 if value is None else 2)
            numeric_values.extend([float("nan")] * width)
            missing_metrics.append(name)
            continue
        states.append(3)
        if metric["kind"] == "list":
            vector = parse_vector(value, width)
        else:
            if not isinstance(value, Real) or isinstance(value, bool):
                raise ValueError(f"{name}: invalid scalar {value!r}")
            vector = np.array([value], dtype=np.float64)
        numeric_values.extend(vector)
        if not np.isfinite(vector).all():
            missing_metrics.append(name)
        elif metric.get("scalarization") in ("argmax_scaled", "argmax_index"):
            grades[name] = int(np.argmax(vector))
    values = np.asarray(numeric_values, dtype="<f8")
    values[np.isnan(values)] = np.nan  # canonical NaN payload; NaN equals NaN for this audit.
    values[values == 0] = 0.0  # mathematical equality of signed zero.
    quality_blob = bytes(states) + values.tobytes()
    content_present = "content" in record and record["content"] is not None
    if content_present and not isinstance(record["content"], str):
        raise ValueError("content must be a string when present")
    content_hash = hashlib.sha256(record["content"].encode("utf-8")).digest() if content_present else None
    raw_json = canonical_json(record)
    return {"quality_blob": quality_blob, "quality_hash": hashlib.sha256(quality_blob).digest(),
            "exact_hash": hashlib.sha256(raw_json.encode("utf-8")).digest(),
            "content_present": int(content_present), "content_hash": content_hash,
            "content_length": len(record["content"]) if content_present else None,
            "sub_path_present": int("sub_path" in record),
            "sub_path_json": canonical_json(record.get("sub_path")),
            "missing_metrics": missing_metrics, "grades": grades}


def compare_raw_quality(left: bytes, right: bytes, metrics: list[dict]) -> dict:
    """Compare original scalars/logit components exactly; equal NaN is unchanged.

    max_absolute_difference uses only pairs of finite numeric components. A
    missing/finite or unequal infinity comparison is flagged separately; when
    no changed finite components exist, the numeric maximum is left undefined.
    """
    if len(left) != len(right) or len(left) < len(metrics):
        raise ValueError("Incompatible quality fingerprints")
    if left == right:
        return {"different_quality_fields":[],"different_quality_field_count":0,
                "max_absolute_difference":None,"nonfinite_component_mismatch_count":0,"quality_same":True}
    left_states, right_states = left[:22], right[:22]
    a = np.frombuffer(left, dtype="<f8", offset=22)
    b = np.frombuffer(right, dtype="<f8", offset=22)
    differences, finite_max, nonfinite_mismatch = [], None, 0
    offset = 0
    for k, metric in enumerate(metrics):
        width = metric.get("expected_length", 1) if metric["kind"] == "list" else 1
        x, y = a[offset:offset+width], b[offset:offset+width]
        same_elements = (x == y) | (np.isnan(x) & np.isnan(y))
        if left_states[k] != right_states[k] or not same_elements.all():
            differences.append(metric["name"])
            finite_changed = np.isfinite(x) & np.isfinite(y) & ~same_elements
            if finite_changed.any():
                difference = float(np.abs(x[finite_changed] - y[finite_changed]).max())
                finite_max = difference if finite_max is None else max(finite_max, difference)
            nonfinite_mismatch += int((~same_elements & ~(np.isfinite(x) & np.isfinite(y))).sum())
        offset += width
    if offset != len(a):
        raise ValueError("Fingerprint dimension differs from configured raw fields")
    return {"different_quality_fields": differences, "different_quality_field_count": len(differences),
            "max_absolute_difference": finite_max, "nonfinite_component_mismatch_count": nonfinite_mismatch,
            "quality_same": not differences}


def compare_record_pair(left: dict, right: dict, metrics: list[dict]) -> dict:
    """Compare same-ID observations without choosing, dropping or averaging.

    Comparable-data equality requires all raw quality fields, sub_path when
    present on both sides, content when present on both sides, and resolved
    domain to agree. One-sided absent content is NOT_COMPARABLE, never equal
    text. Source paths are provenance, not part of this comparable-data test.
    """
    quality = compare_raw_quality(left["quality_blob"], right["quality_blob"], metrics)
    content_same = (left["content_hash"] == right["content_hash"]) if left["content_present"] and right["content_present"] else "NOT_COMPARABLE"
    sub_path_same = (left["sub_path_json"] == right["sub_path_json"]) if left["sub_path_present"] and right["sub_path_present"] else "NOT_COMPARABLE"
    domain_same = left["domain"] == right["domain"] if left["domain"] != "<unknown>" and right["domain"] != "<unknown>" else "NOT_COMPARABLE"
    same = quality["quality_same"] and content_same is not False and sub_path_same is not False and domain_same is not False
    return {**quality,"content_same":content_same,"sub_path_same":sub_path_same,"domain_same":domain_same,
            "exact_record_same":left["exact_hash"]==right["exact_hash"],
            "classification":"SAME_ID_SAME_DATA" if same else "SAME_ID_DIFFERENT_DATA",
            "quality_classification":"SAME_ID_SAME_DATA" if quality["quality_same"] else "SAME_ID_DIFFERENT_DATA"}
