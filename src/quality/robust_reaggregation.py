"""One-step local downweighting around frozen Q, applied only to frozen high K.

Coefficients are a deterministic residual rule, not calibrated reliability.
Neither Q, K, Z, global weights nor conflict thresholds are fitted here.
"""
from __future__ import annotations
import numpy as np


def reaggregate(z, q, high_mask, weights, delta, epsilon=1e-8):
    """Return candidate score and local weights; ordinary scores remain exact Q.

    Uses original Q once as center; never iterates or estimates a new center.
    Local weights are returned only for high-conflict rows to bound memory.
    """
    values=np.asarray(z,dtype=float);center=np.asarray(q,dtype=float)
    high=np.asarray(high_mask,dtype=bool);w=np.asarray(weights,dtype=float)
    if values.ndim!=2 or values.shape[1]!=22 or center.shape!=(len(values),) or high.shape!=center.shape or w.shape!=(22,):
        raise ValueError('Expected n x 22 Z, aligned Q/high mask, 22 weights')
    if not all(np.isfinite(x).all() for x in [values,center,w]) or (values<0).any() or (values>1).any() or (center<0).any() or (center>1).any():
        raise ValueError('Nonfinite or unbounded frozen inputs')
    if (w<0).any() or not np.isclose(w.sum(),1,rtol=0,atol=1e-12) or not np.isfinite(delta) or delta<=0 or not np.isfinite(epsilon) or epsilon<=0:
        raise ValueError('Invalid fixed weights or candidate parameters')
    residual=values[high]-center[high,None]
    coefficients=np.minimum(1.,float(delta)/(np.abs(residual)+epsilon))
    unnormalized=coefficients*w
    denominators=unnormalized.sum(axis=1,keepdims=True)
    if (denominators<=0).any():raise ValueError('Local normalization denominator is not positive')
    local=unnormalized/denominators
    candidate=np.einsum('ij,ij->i',local,values[high])
    if not np.isfinite(candidate).all() or (candidate < -1e-14).any() or (candidate > 1+1e-14).any():
        raise ValueError('Invalid local convex combination')
    # Endpoint rounding only; frozen ordinary Q is never clipped or rewritten.
    score=center.copy();score[high]=np.clip(candidate,0,1)
    if not np.isfinite(score).all() or (score<0).any() or (score>1).any():raise ValueError('Invalid robust candidate')
    if not np.array_equal(score[~high],center[~high]):raise ValueError('Ordinary Q unexpectedly changed')
    return {'Q_robust':score,'change':score-center,'coefficients_high':coefficients,'local_weights_high':local}
