"""Small, auditable regression and paired-bootstrap contribution calculations."""
import json
import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import HuberT
from .common import corr


def design(data,terms):
    """Use an explicit intercept and the requested predictors in fixed order."""
    return np.column_stack([np.ones(len(data)),data[terms].to_numpy(float)])


def accuracy(y,pred):
    """Undefined correlations/R² remain missing, rather than become zero."""
    y=np.asarray(y,float);pred=np.asarray(pred,float);ok=np.isfinite(y)&np.isfinite(pred)
    y=y[ok];pred=pred[ok];err=pred-y;ss=np.sum((y-y.mean())**2)
    return dict(n=len(y),RMSE=float(np.sqrt(np.mean(err**2))),MAE=float(np.mean(abs(err))),
                R2=float(1-err@err/ss) if ss>0 else np.nan,
                Pearson=corr(y,pred),Spearman=corr(y,pred,'spearman'))


def fit_regression(data,terms,label,scope,axis='submission',method='OLS'):
    """Fit OLS or HuberT RLM without importing incompatible statsmodels.api."""
    X=design(data,terms);y=data.A_z.to_numpy(float);names=['intercept']+terms
    if len(data)<=X.shape[1]+1 or np.linalg.matrix_rank(X)<X.shape[1]:
        raise ValueError(f'{label}: insufficient full-rank sample')
    if method=='OLS':
        fit=OLS(y,X).fit();leverage=np.einsum('ij,ji->i',X,np.linalg.pinv(X))
        robust=fit.get_robustcov_results(cov_type='HC3',use_t=True) if np.min(1-leverage)>1e-9 else None
        variants={'OLS':fit,'HC3':robust}
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');fit=RLM(y,X,M=HuberT()).fit(maxiter=200,cov='H3')
        variants={'Huber_H3':fit}
        for w in caught:logging.warning('%s: %s',label,str(w.message))
    coefficients=[]
    for inference,v in variants.items():
        if v is None:
            ci=np.full((len(names),2),np.nan);values=np.asarray(fit.params);SE=np.full(len(names),np.nan);pv=SE.copy()
            status='UNIDENTIFIABLE_LEVERAGE_ONE'
        else:
            ci=np.asarray(v.conf_int());values=np.asarray(v.params);SE=np.asarray(v.bse);pv=np.asarray(v.pvalues);status='AVAILABLE'
        for j,name in enumerate(names):
            coefficients.append(dict(model=label,scope=scope,time_axis=axis,method=method,inference=inference,
                term=name,n=len(data),coef=values[j],SE=SE[j],CI_low=ci[j,0],CI_high=ci[j,1],p_value=pv[j],inference_status=status))
    fitted=np.asarray(fit.fittedvalues);evaluation=dict(model=label,scope=scope,time_axis=axis,method=method,
        evaluation='in_sample',**accuracy(y,fitted),adjusted_R2=float(fit.rsquared_adj) if method=='OLS' else np.nan,
        AIC=float(fit.aic) if method=='OLS' else np.nan,BIC=float(fit.bic) if method=='OLS' else np.nan,
        AICc=float(fit.aic+2*X.shape[1]*(X.shape[1]+1)/(len(data)-X.shape[1]-1)) if method=='OLS' else np.nan)
    predictions=[];oof=[];fold_rows=[]
    for i in range(len(data)):
        mask=np.arange(len(data))!=i;Xt=X[mask];yt=y[mask];rank=np.linalg.matrix_rank(Xt)
        identifiable=rank==X.shape[1]
        if identifiable:
            if method=='OLS':beta=np.linalg.lstsq(Xt,yt,rcond=None)[0]
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore');beta=np.asarray(RLM(yt,Xt,M=HuberT()).fit(maxiter=200).params)
            estimate=float(X[i]@beta)
        else:beta=np.full(X.shape[1],np.nan);estimate=np.nan
        oof.append(estimate)
        fold_rows.append(dict(model=label,scope=scope,time_axis=axis,method=method,held_out_Model=data.Model.iloc[i],
            full_rank=identifiable,training_n=len(yt),training_rank=rank,parameters=json.dumps(beta.tolist())))
        predictions.append(dict(model=label,scope=scope,time_axis=axis,method=method,Model=data.Model.iloc[i],
            observed_A=y[i],fitted_A=fitted[i],LOOCV_predicted_A=estimate,residual=y[i]-fitted[i]))
    count=int(np.isfinite(oof).sum());cv=dict(model=label,scope=scope,time_axis=axis,method=method,evaluation='LOOCV',
        **accuracy(y,oof),eligible_n=len(data),unidentifiable_folds=len(data)-count,
        full_LOOCV_available=count==len(data))
    # A partial subset is not substituted for a full CV score in model selection.
    cv['partial_RMSE']=cv['RMSE']
    if count<len(data):
        for k in ['RMSE','MAE','R2','Pearson','Spearman']:cv[k]=np.nan
    return dict(label=label,scope=scope,axis=axis,method=method,terms=terms,names=names,n=len(data),
        data=data.copy(),X=X,y=y,fit=fit,beta=np.asarray(fit.params),
        coefficients=coefficients,evaluations=[evaluation,cv],predictions=predictions,folds=fold_rows)


def diagnostics(model):
    """VIF, standardized condition number, Koenker BP, residuals and Cook's D."""
    X=model['X'];y=model['y'];beta=model['beta'];n,p=X.shape;res=y-X@beta
    hat=np.einsum('ij,ji->i',X,np.linalg.pinv(X));mse=res@res/(n-p)
    cook=np.full(n,np.nan);identifiable=(1-hat)>1e-9
    if mse>0:cook[identifiable]=(res[identifiable]**2/(p*mse))*hat[identifiable]/(1-hat[identifiable])**2
    else:cook[identifiable]=0
    vif=[]
    for j in range(1,p):
        target=X[:,j];other=np.delete(X,j,axis=1);error=target-other@np.linalg.lstsq(other,target,rcond=None)[0]
        total=np.sum((target-target.mean())**2);r2=1-error@error/total if total>0 else 1
        v=1/(1-r2) if r2<1-1e-14 else np.nan
        vif.append(dict(model=model['label'],scope=model['scope'],term=model['names'][j],VIF=v,
            exactly_collinear=r2>=1-1e-14))
    standardized=(X[:,1:]-X[:,1:].mean(axis=0))/X[:,1:].std(axis=0,ddof=0)
    raw_condition=np.linalg.cond(X);scaled_condition=np.linalg.cond(np.column_stack([np.ones(n),standardized]))
    aux=OLS(res**2,X).fit();LM=n*aux.rsquared;df=p-1
    BPp=stats.chi2.sf(LM,df);fvalue=aux.fvalue;fp=aux.f_pvalue
    JB=stats.jarque_bera(res)
    summary=dict(model=model['label'],scope=model['scope'],n=n,p=p,
        condition_number_raw=raw_condition,condition_number_standardized=scaled_condition,
        Koenker_BP_LM=LM,Koenker_BP_p=BPp,Koenker_BP_F=fvalue,Koenker_BP_F_p=fp,
        residual_mean=res.mean(),residual_std=res.std(ddof=1),Jarque_Bera=JB.statistic,JB_p=JB.pvalue,
        Cook_threshold=4/n,Cook_exceed_count=int((cook>4/n).sum()),Cook_unidentifiable_count=int((~identifiable).sum()),max_Cook=float(np.nanmax(cook)),
        max_Cook_Model=model['data'].Model.iloc[int(np.nanargmax(cook))])
    influence=[dict(model=model['label'],Model=row.Model,leverage=hat[i],Cooks_distance=cook[i],
        Cook_warning=bool(cook[i]>4/n),residual=res[i]) for i,row in enumerate(model['data'].itertuples())]
    return vif,summary,influence


def shapley(beta,terms,early,late):
    """Two-factor endpoint Shapley with grouped scale predictors; preserve signs."""
    def f(state,t):return beta[0]+sum(beta[k+1]*(t if term=='t_month' else state[term]) for k,term in enumerate(terms))
    s0={term:early[term] for term in terms if term!='t_month'};s1={term:late[term] for term in terms if term!='t_month'}
    t0=early['t_month'];t1=late['t_month'];f00=f(s0,t0);f10=f(s1,t0);f01=f(s0,t1);f11=f(s1,t1)
    ds=.5*(f10-f00+f11-f01);dt=.5*(f01-f00+f11-f10);total=f11-f00;err=abs(total-ds-dt)
    if err>=1e-10:raise ArithmeticError('Shapley components do not sum to predicted change')
    physical=total>0 and ds>=0 and dt>=0;den=abs(ds)+abs(dt)
    return dict(A_early_pred=f00,A_late_pred=f11,Delta_pred=total,Delta_scale=ds,Delta_tech=dt,
        identity_error=err,nonnegative_growth_shares_valid=physical,cancellation_effect=bool(ds*dt<0),
        share_scale=ds/total if physical else np.nan,share_tech=dt/total if physical else np.nan,
        signed_ratio_scale=ds/total if abs(total)>1e-12 else np.nan,
        signed_ratio_tech=dt/total if abs(total)>1e-12 else np.nan,
        absolute_share_scale=abs(ds)/den if den>1e-12 else np.nan,
        absolute_share_tech=abs(dt)/den if den>1e-12 else np.nan)


def endpoints(data,terms,bounds,fraction):
    """Fixed original time thirds/quarters; endpoint medians use resampled models."""
    lo,hi=bounds
    e=data[data.t_month.between(lo,lo+fraction*(hi-lo))]
    l=data[data.t_month.between(hi-fraction*(hi-lo),hi)]
    if e.empty or l.empty:return None,None,len(e),len(l)
    fields=set(terms+['t_month','lnN','lnD','lnC','A_z','N_B','D_tokens','training_compute'])
    states=[]
    for h in [e,l]:
        state={k:h[k].median() for k in fields}
        for raw,log in [('D_tokens','lnD'),('training_compute','lnC')]:
            state[log]=np.log(state[raw]) if np.isfinite(state[raw]) and state[raw]>0 else np.nan
        states.append(state)
    return states[0],states[1],len(e),len(l)


def contribution(model,bounds,fraction,series):
    """Main fitted contribution plus actual endpoint-state diagnostics."""
    e,l,ne,nl=endpoints(model['data'],model['terms'],bounds,fraction)
    base=dict(series=series,model=model['label'],scope=model['scope'],time_axis=model['axis'],
        window_fraction=fraction,n=model['n'],early_n=ne,late_n=nl,time_bound_min=bounds[0],time_bound_max=bounds[1])
    if e is None:return dict(**base,status='EMPTY_ENDPOINT_WINDOW')
    base.update(status='OK',**shapley(model['beta'],model['terms'],e,l))
    for prefix,s in [('early',e),('late',l)]:
        for k,v in s.items():base[prefix+'_'+k]=v
    base['observed_median_A_change']=l['A_z']-e['A_z']
    return base


def bootstrap(model,bounds,fractions,series,B,seed):
    """Exactly B model draws; unavailable endpoints/rank retained and audited."""
    rng=np.random.default_rng(seed);d=model['data'];X=model['X'];y=model['y'];n,p=X.shape
    allrows=[];coefs=[]
    # Avoid repeated pandas sampling/grouping in 5000 small bootstrap draws.
    fields=['lnN','lnD','lnC','t_month','A_z','N_B','D_tokens','training_compute'];values=d[fields].to_numpy(float);fi={k:i for i,k in enumerate(fields)}
    for rep in range(1,B+1):
        ix=rng.integers(0,n,n);Xt=X[ix];yt=y[ix];full=np.linalg.matrix_rank(Xt)==p
        b=np.linalg.lstsq(Xt,yt,rcond=None)[0] if full else np.full(p,np.nan)
        coefs.append(dict(model=model['label'],series=series,time_axis=model['axis'],replicate=rep,
            full_rank=full,source_model_indices=json.dumps(ix.tolist()),**dict(zip(model['names'],b))))
        v=values[ix];time=v[:,fi['t_month']]
        for frac in fractions:
            lo,hi=bounds;em=(time>=lo)&(time<=lo+frac*(hi-lo));lm=(time>=hi-frac*(hi-lo))&(time<=hi)
            row=dict(series=series,model=model['label'],scope=model['scope'],time_axis=model['axis'],
                window_fraction=frac,replicate=rep,early_n=int(em.sum()),late_n=int(lm.sum()),full_rank=full)
            if not full:row['status']='RANK_DEFICIENT'
            elif not em.any() or not lm.any():row['status']='EMPTY_ENDPOINT_WINDOW'
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    ev=np.nanmedian(v[em],axis=0);lv=np.nanmedian(v[lm],axis=0)
                e=dict(zip(fields,ev));l=dict(zip(fields,lv))
                for state in [e,l]:
                    for raw,log in [('D_tokens','lnD'),('training_compute','lnC')]:
                        state[log]=np.log(state[raw]) if np.isfinite(state[raw]) and state[raw]>0 else np.nan
                row.update(status='OK',**shapley(b,model['terms'],e,l))
            allrows.append(row)
        if rep%1000==0:logging.info('%s bootstrap %d/%d',model['label'],rep,B)
    return allrows,coefs


def bootstrap_summary(rows):
    """Conditional valid percentiles, signed ratios and nonnegative-growth rates."""
    frame=pd.DataFrame(rows);result=[]
    quantities=['Delta_pred','Delta_scale','Delta_tech','signed_ratio_scale','signed_ratio_tech',
        'share_scale','share_tech','absolute_share_scale','absolute_share_tech']
    for keys,g in frame.groupby(['series','model','scope','time_axis','window_fraction'],dropna=False):
        valid=g[g.status=='OK'];base=dict(zip(['series','model','scope','time_axis','window_fraction'],keys))
        for q in quantities:
            x=valid[q].dropna() if q in valid else pd.Series(dtype=float)
            result.append(dict(**base,quantity=q,draws=len(g),valid_decomposition_draws=len(valid),
                valid_quantity_draws=len(x),invalid_window_or_rank_draws=len(g)-len(valid),
                CI_low=x.quantile(.025) if len(x) else np.nan,CI_high=x.quantile(.975) if len(x) else np.nan,
                median=x.median() if len(x) else np.nan,
                nonnegative_growth_fraction=float(valid.nonnegative_growth_shares_valid.mean()) if len(valid) else np.nan,
                positive_scale_fraction=float(valid.Delta_scale.gt(0).mean()) if len(valid) else np.nan,
                positive_tech_fraction=float(valid.Delta_tech.gt(0).mean()) if len(valid) else np.nan,
                cancellation_fraction=float(valid.cancellation_effect.mean()) if len(valid) else np.nan,
                ratio_CI_outside_zero_one=bool(len(x) and q.startswith('signed_ratio') and (x.quantile(.025)<0 or x.quantile(.975)>1)),
                interval_scope='percentile; conditional on identifiable sampled endpoints; physical shares additionally conditional on nonnegative growth'))
    return result
