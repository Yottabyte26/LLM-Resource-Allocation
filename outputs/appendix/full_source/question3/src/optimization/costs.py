"""Official cost proxies evaluated in FLOPs; model inputs N/D are billions."""
from __future__ import annotations
import numpy as np


def _validate_quality(q, q0):
    """Reject invalid or nonfinite qualities rather than silently clipping them."""
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or np.any((q <= 0) | (q > 1)) or not 0 < q0 <= 1:
        raise ValueError('Q and Q0 must be finite in (0, 1]')
    return q


def quality_increment_per_token(q, q0, family, gamma, lam):
    """Compute [g(Q)-g(Q0)]+, retaining the exact Appendix B families."""
    q = _validate_quality(q, q0)
    if gamma <= 0 or lam <= 0:
        raise ValueError('Positive official cost parameters required')
    if family == 'exponential':
        diff = gamma * np.exp(lam*q0) * np.expm1(lam*(q-q0))
    elif family == 'power':
        diff = gamma * (q**lam-q0**lam)
    elif family == 'logarithmic':
        diff = gamma * np.log1p(lam*(q-q0)/(1+lam*q0))
    else:
        raise ValueError(f'Unknown cost family: {family}')
    return np.maximum(diff, 0)


def marginal_quality_per_token(q, q0, family, gamma, lam):
    """Right derivative at Q0; below Q0 the positive-part cost is zero."""
    q = _validate_quality(q, q0)
    if family == 'exponential':
        derivative = gamma*lam*np.exp(lam*q)
    elif family == 'power':
        derivative = gamma*lam*q**(lam-1)
    elif family == 'logarithmic':
        derivative = gamma*lam/(1+lam*q)
    else:
        raise ValueError(f'Unknown cost family: {family}')
    return np.where(q >= q0, derivative, 0)


def cost_components(n_b, d_b, q, q0, l_ctx, official, family):
    """Return training, incremental quality and attention FLOPs, without fitting."""
    scalars = np.asarray([n_b, d_b, l_ctx], float)
    if not np.isfinite(scalars).all() or (scalars <= 0).any():
        raise ValueError('Finite positive N_B, D_B and L_ctx required')
    pars = official['quality_costs'][family]
    train = official['training_coefficient']*n_b*d_b*1e18
    attention = official['eta']*n_b*d_b*l_ctx*1e18
    quality = d_b*1e9*quality_increment_per_token(q, q0, family, pars['gamma'], pars['lambda'])
    return train, quality, attention
