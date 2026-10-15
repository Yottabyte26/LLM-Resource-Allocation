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
