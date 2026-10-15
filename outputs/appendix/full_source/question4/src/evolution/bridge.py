"""High-only small-sample bridge, LOMO validation and bootstrap uncertainty."""
import json
import logging
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from .ability import transform
from .common import csv,dump,corr


def fit_candidate(kind,x,y):
    """Three simple response models; no medium rows or previous fitted bridge."""
    x=np.asarray(x,float);y=np.asarray(y,float)
    if kind=='B2':
        model=IsotonicRegression(increasing=False,out_of_bounds='clip').fit(x,y)
        return dict(kind=kind,x=model.X_thresholds_.tolist(),y=model.y_thresholds_.tolist())
    degree=1 if kind=='B0' else 2
    coef=np.linalg.lstsq(np.column_stack([x**k for k in range(degree+1)]),y,rcond=None)[0]
    return dict(kind=kind,coefficients=coef.tolist())


def predict(candidate,x):
    """Evaluate persisted coefficients/thresholds without a refit."""
    x=np.asarray(x,float)
    if candidate['kind']=='B2':return np.interp(x,candidate['x'],candidate['y'])
    return sum(v*x**k for k,v in enumerate(candidate['coefficients']))


def metrics(observed,predicted):
    """Undefined R²/correlations remain NaN for a constant response/prediction."""
    observed=np.asarray(observed,float);predicted=np.asarray(predicted,float);error=predicted-observed
    total=np.sum((observed-observed.mean())**2)
    return dict(n=len(observed),RMSE=float(np.sqrt(np.mean(error**2))),MAE=float(np.mean(abs(error))),
        R2=float(1-error@error/total) if total>0 else np.nan,
        Pearson=corr(observed,predicted),Spearman=corr(observed,predicted,'spearman'),bias=float(error.mean()))


def decreasing(candidate,lo,hi):
    """Diagnose fitted direction; never change an empirical slope's sign."""
    if candidate['kind']=='B2':return bool(np.all(np.diff(candidate['y'])<=1e-12))
    c=candidate['coefficients'];derivative=[c[1]+(2*c[2]*x if len(c)==3 else 0) for x in [lo,hi]]
    return bool(max(derivative)<0)


def predict_bridge(loss,model_path):
    """Reusable calibrated-range predictor; extrapolation is explicitly flagged.

    Bootstrap bands are saved separately; do not apply this bridge to future
    resource/benchmark forecasts without carrying that uncertainty.
    """
    metadata=json.loads(model_path.read_text(encoding='utf8'));x=np.atleast_1d(loss).astype(float)
    lo,hi=metadata['calibrated_loss_range']
    return pd.DataFrame(dict(Val_Loss=x,A_pred=predict(metadata['selected_model'],x),
        outside_high_calibration_range=(x<lo)|(x>hi),
        validated_for_loss_to_ability_conversion=metadata['bridge_valid_for_conversion'],
        evidence=metadata['bridge_status']))


def analyze_bridge(root,cfg):
    """High-only fitting; medium external validation; 2000 paired bootstraps."""
    out=root/'outputs/evolution/phase1';base=root/'real_attachments/C_efficiency_evolution'
    data=pd.read_csv(base/'loss_benchmark_bridge_expanded.csv',float_precision='round_trip')
    params=json.loads((out/'ability_transform_parameters.json').read_text(encoding='utf8'))
    mapping={'IFEval':'LB_IFEval','BBH':'LB_BBH','MATH Lvl 5':'LB_MATH','GPQA':'LB_GPQA','MUSR':'LB_MUSR','MMLU-PRO':'LB_MMLU_PRO'}
    tasks=pd.DataFrame({m:pd.to_numeric(data[field],errors='coerce') for m,field in mapping.items()})
    valid=tasks.notna().all(axis=1)&tasks.ge(0).all(axis=1)&tasks.le(100).all(axis=1)
    data['A_bridge']=transform(tasks,params).mean(axis=1,skipna=False);data.loc[~valid,'A_bridge']=np.nan
    data['comparability']=data.Loss_Comparability.str.extract(r'^(High|Medium)',expand=False).fillna('other')
    data['model_family_diagnostic']=np.where(data.Model.str.contains('pythia',case=False),'Pythia',data.Model.str.split('/').str[0])
    csv(out/'C6_bridge_transformed_records.csv',data)
    c5=pd.read_csv(base/'loss_benchmark_bridge.csv',float_precision='round_trip');comparison=[]
    for r in c5.to_dict('records'):
        matches=data[data.Model==r['Model']]
        for field in c5.columns:
            if field=='Model':continue
            same=False;gap=np.nan
            if len(matches)==1:
                v=matches[field].iloc[0]
                if isinstance(r[field],(int,float)) and isinstance(v,(int,float)):
                    gap=abs(r[field]-v);same=bool(np.isclose(r[field],v,rtol=0,atol=1e-12,equal_nan=True))
                else:same=(pd.isna(r[field]) and pd.isna(v)) or str(r[field])==str(v)
            comparison.append(dict(Model=r['Model'],field=field,C6_matches=len(matches),consistent=same,absolute_difference=gap))
    csv(out/'C5_C6_consistency.csv',comparison)
    valid_rows=data.A_bridge.notna()&pd.to_numeric(data.Val_Loss,errors='coerce').gt(0)
    high=data[(data.comparability=='High')&valid_rows].copy();medium=data[(data.comparability=='Medium')&valid_rows].copy()
    if len(high)<4 or high.Model.duplicated().any():raise ValueError('Insufficient/ambiguous independent High models; bridge cannot proceed')
    x=high.Val_Loss.to_numpy();y=high.A_bridge.to_numpy();lo,hi=float(x.min()),float(x.max());grid=np.linspace(lo,hi,cfg['bridge_grid_points'])
    # No family column exists in C6. Inferred High family is solely Pythia,
    # hence true family-holdout validation is impossible, not substituted.
    kinds=['B0','B1','B2'];fitted={};cv={};records=[];scores=[];folds=[]
    for kind in kinds:
        full=fit_candidate(kind,x,y);fitted[kind]=full;oof=np.empty(len(x));directions=[]
        for i in range(len(x)):
            mask=np.arange(len(x))!=i;fold=fit_candidate(kind,x[mask],y[mask]);oof[i]=float(predict(fold,[x[i]])[0])
            direction=decreasing(fold,float(x[mask].min()),float(x[mask].max()));directions.append(direction)
            folds.append(dict(candidate=kind,held_out_Model=high.Model.iloc[i],training_n=int(mask.sum()),
                decreasing_on_fold_training_range=direction,held_out_outside_training_range=bool(x[i]<x[mask].min() or x[i]>x[mask].max()),
                formula=json.dumps(fold),observed_A=float(y[i]),predicted_A=float(oof[i])))
        cv[kind]=oof;valid_direction=decreasing(full,lo,hi)
        scores.append(dict(candidate=kind,evaluation='High_LOMO_CV',**metrics(y,oof),
            decreasing_full=valid_direction,decreasing_folds=sum(directions),folds=len(x),
            CV_method='Leave-One-Model-Out; only one High family; no out-of-family test'))
        external=predict(full,medium.Val_Loss)
        scores.append(dict(candidate=kind,evaluation='Medium_external',**metrics(medium.A_bridge,external),
            outside_High_loss_range=int(((medium.Val_Loss<lo)|(medium.Val_Loss>hi)).sum()),decreasing_full=valid_direction))
        for i,r in enumerate(high.to_dict('records')):
            records.append(dict(candidate=kind,evaluation='High_LOMO_CV',Model=r['Model'],Val_Loss=r['Val_Loss'],
                observed_A=r['A_bridge'],predicted_A=float(oof[i]),comparability='High'))
        for i,r in enumerate(medium.to_dict('records')):
            records.append(dict(candidate=kind,evaluation='Medium_external',Model=r['Model'],Val_Loss=r['Val_Loss'],
                observed_A=r['A_bridge'],predicted_A=float(external[i]),comparability='Medium',
                outside_High_loss_range=bool(r['Val_Loss']<lo or r['Val_Loss']>hi)))
    scores=pd.DataFrame(scores);eligible=scores[(scores.evaluation=='High_LOMO_CV')&scores.decreasing_full].copy()
    selected=str(eligible.sort_values(['RMSE','candidate']).candidate.iloc[0])
    rng=np.random.default_rng(cfg['random_state']);B=cfg['bridge_bootstrap_repetitions'];band=[];arrays={};bootstrap_parameters=[]
    for kind in kinds:
        predictions=[];directions=[];attempts=0
        while len(predictions)<B:
            attempts+=1;ix=rng.integers(0,len(x),len(x));needed=3 if kind=='B1' else 2
            if np.unique(x[ix]).size<needed:continue
            fit=fit_candidate(kind,x[ix],y[ix]);predictions.append(predict(fit,grid));directions.append(decreasing(fit,lo,hi))
            bootstrap_parameters.append(dict(candidate=kind,replicate=len(predictions),decreasing=bool(directions[-1]),
                fitted_parameters=json.dumps(fit),source_model_indices=json.dumps(ix.tolist())))
        curves=np.array(predictions);residual=y-cv[kind];centered=residual-residual.mean()
        # Conditional residual-noise band, using OOF rather than in-sample
        # residuals; seven same-family rows do NOT justify a universal band.
        noise=rng.choice(centered,size=curves.shape);predictive=curves+noise
        ci=np.quantile(curves,[.025,.975],axis=0);pi=np.quantile(predictive,[.025,.975],axis=0);mean=predict(fitted[kind],grid)
        for i,l in enumerate(grid):
            band.append(dict(candidate=kind,selected=kind==selected,Val_Loss=float(l),A_pred=float(mean[i]),
                mean_CI_low=float(ci[0,i]),mean_CI_high=float(ci[1,i]),prediction_band_low=float(pi[0,i]),prediction_band_high=float(pi[1,i]),
                bootstrap_repetitions=B,bootstrap_decreasing_fraction=float(np.mean(directions)),resampling_attempts=attempts,
                outside_High_calibration_range=False,uncertainty_scope='within-Pythia paired bootstrap plus centered High LOMO residual noise'))
        arrays[kind+'_bootstrap_curve']=curves
    np.savez_compressed(out/'bridge_bootstrap_curves.npz',Val_Loss=grid,**arrays)
    csv(out/'loss_benchmark_bridge.csv',band);csv(out/'bridge_bootstrap_parameters.csv',bootstrap_parameters)
    csv(out/'bridge_validation_metrics.csv',scores);csv(out/'bridge_validation_predictions.csv',records);csv(out/'bridge_CV_fold_diagnostics.csv',folds)
    selected_fit=fitted[selected]
    cv_row=scores[(scores.candidate==selected)&(scores.evaluation=='High_LOMO_CV')].iloc[0]
    flat=bool(np.ptp(predict(selected_fit,grid))<1e-12)
    supported=bool(cv_row.R2>0 and cv_row.Pearson>0 and not flat)
    summary=dict(raw_C6_count=len(data),High_count=len(high),Medium_count=len(medium),
        other_count=int((data.comparability=='other').sum()),selected_candidate=selected,selection_rule='minimum High LOMO RMSE among decreasing fits; Medium never used to select',
        candidate_models=fitted,selected_model=selected_fit,calibrated_loss_range=[lo,hi],
        fixed_ability_parameters_source='outputs/evolution/phase1/ability_transform_parameters.json',
        High_families=high.model_family_diagnostic.unique().tolist(),CV='LOMO; Group CV impossible for sole High family',
        bootstrap_repetitions=B,Medium_outside_High_loss_range=int(((medium.Val_Loss<lo)|(medium.Val_Loss>hi)).sum()),
        selected_fit_is_constant=flat,bridge_valid_for_conversion=supported,
        bridge_status='DIAGNOSTIC_ONLY_FLAT_ISOTONIC' if flat else 'DIAGNOSTIC_ONLY_CV_NOT_VALIDATED' if not supported else 'WITHIN_FAMILY_LIMITED_VALIDATION',
        interpretation='numerical candidate ranking only; if CV fails or fit is constant, no useful predictive bridge is validated; no universal cross-family evidence',
        future_forecast_performed=False,contribution_decomposition_performed=False)
    dump(out/'bridge_model.json',summary);logging.info('Bridge selected %s; validation: %s',selected,scores.to_dict('records'))
