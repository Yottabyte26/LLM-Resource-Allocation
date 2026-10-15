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
