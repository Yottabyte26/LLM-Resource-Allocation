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
