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
