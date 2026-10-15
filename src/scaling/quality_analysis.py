"""ND-group validation and reproducible block bootstrap for quality extensions."""
from __future__ import annotations
import csv
import logging
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold
from .quality_extension import MODELS, fit_quality, predict_quality, subset, group_reference
from .classical import metrics
from .io import write_csv, write_json


def evaluate(y, pred, **labels):
    """Metrics in raw Loss; bias is actual minus prediction."""
    return dict(**labels, **metrics(y,pred))


def cross_validate(t, p, cfg, out):
    """Each exact ND block appears wholly in one test fold; no full-fit warm starts."""
    n=len(t['L']); oof={m:np.full(n,np.nan) for m in MODELS}
    folds=[];fit_rows=[];starts=[];fold_metrics=[];fold_of_row=np.zeros(n,dtype=int)
    splitter=GroupKFold(n_splits=cfg['group_cv_folds'])
    for fold,(train,test) in enumerate(splitter.split(t['N'],groups=t['group']),1):
        tr,te=subset(t,train),subset(t,test)
        train_groups,test_groups=sorted(set(tr['group'])),sorted(set(te['group']))
        if set(train_groups)&set(test_groups):raise AssertionError('ND leakage')
        fold_of_row[test]=fold
        folds.append(dict(fold=fold,train_rows=len(train),test_rows=len(test),
                          train_groups=train_groups,test_groups=test_groups))
        for model in MODELS:
            if model=='M0':gamma=offset=0.0
            else:
                fit,runs=fit_quality(tr,p,model,cfg);gamma,offset=fit['gamma'],fit['offset']
                fit_rows.append(dict(stage='GroupCV',fold=fold,**fit))
                starts.extend(dict(stage='GroupCV',model=model,fold=fold,**r) for r in runs)
            yp=predict_quality(te,p,model,gamma,offset);oof[model][test]=yp
            fold_metrics.append(evaluate(te['L'],yp,model=model,fold=fold))
    if any(not np.isfinite(v).all() for v in oof.values()):raise AssertionError('Incomplete OOF predictions')
    predictions=[]
    for i in range(n):
        for m in MODELS:
            predictions.append(dict(row_index=i,ND_group=t['group'][i],fold=int(fold_of_row[i]),
                                    N_params_B=float(t['N'][i]),D_tokens_B=float(t['D'][i]),Q_score=float(t['Q'][i]),
                                    model=m,actual_Loss=float(t['L'][i]),predicted_Loss=float(oof[m][i])))
    write_csv(out/'group_cv_predictions.csv',predictions)
    write_csv(out/'group_cv_folds.csv',folds)
    write_csv(out/'group_cv_model_parameters.csv',fit_rows)
    write_csv(out/'group_cv_fold_metrics.csv',fold_metrics)
    comparison=[evaluate(t['L'],oof[m],dataset='B6',evaluation='5fold_ND_GroupCV',model=m,
                         role='source_offset_sensitivity' if m.endswith('_offset') else ('main_candidate' if m=='MQ1' else 'baseline_or_form_sensitivity')) for m in MODELS]
    write_csv(out/'quality_model_comparison.csv',comparison)
    return oof, comparison, starts


def q_level_cv(t,p,cfg,out):
    """All rows at a Q level are held out; endpoint tests are Q extrapolation."""
    rows=[]; starts=[];predictions=[]
    levels=np.unique(t['Q'])
    for q in levels:
        train=t['Q']!=q;test=~train
        tr,te=subset(t,train),subset(t,test)
        for m in MODELS:
            if m=='M0':gamma=offset=0.0
            else:
                fit,runs=fit_quality(tr,p,m,cfg);gamma,offset=fit['gamma'],fit['offset']
                starts.extend(dict(stage='leave_one_Q_out',held_Q=float(q),model=m,**r) for r in runs)
            yp=predict_quality(te,p,m,gamma,offset)
            rows.append(evaluate(te['L'],yp,held_Q=float(q),model=m,gamma=gamma,offset=offset,
                                 Q_test_kind='endpoint_extrapolation' if q in [levels[0],levels[-1]] else 'interior_interpolation'))
            predictions.extend(dict(model=m,held_Q=float(q),ND_group=g,actual_Loss=float(y),predicted_Loss=float(v)) for g,y,v in zip(te['group'],te['L'],yp))
    write_csv(out/'leave_one_Q_level_out.csv',rows)
    write_csv(out/'leave_one_Q_level_out_predictions.csv',predictions)
    return rows,starts


def layer_fit(t,p,cfg,out):
    """Separate one-gamma fits by N and D; offset fits diagnose confounding only."""
    stats=[];starts=[];tables={}
    for axis,file in [('N','gamma_by_scale.csv'),('D','gamma_by_data.csv')]:
        rows=[]
        for level in np.unique(t[axis]):
            a=subset(t,t[axis]==level)
            if len(np.unique(a['Q']))<2 or len(set(a['group']))<2:continue
            for m in ['MQ1','MQ1_offset']:
                fit,runs=fit_quality(a,p,m,cfg)
                yp=predict_quality(a,p,m,fit['gamma'],fit['offset'])
                rows.append(evaluate(a['L'],yp,axis=axis,level=float(level),model=m,
                                     gamma=fit['gamma'],eta=p['beta']*fit['gamma'],offset=fit['offset'],
                                     n_ND_groups=len(set(a['group'])),n_Q_levels=len(np.unique(a['Q']))))
                starts.extend(dict(stage='layer_fit',axis=axis,level=float(level),model=m,**r) for r in runs)
        write_csv(out/file,rows);tables[axis]=rows
        for m in ['MQ1','MQ1_offset']:
            r=[x for x in rows if x['model']==m]
            x=np.log([x['level'] for x in r]);y=np.array([x['gamma'] for x in r])
            stats.append(dict(axis=axis,model=m,n_layers=len(r),gamma_min=float(y.min()),gamma_max=float(y.max()),
                              gamma_mean=float(y.mean()),gamma_std=float(y.std(ddof=1)),
                              Pearson_gamma_logscale=float(pearsonr(x,y).statistic),
                              Spearman_gamma_logscale=float(spearmanr(x,y).statistic)))
    write_csv(out/'gamma_scale_dependence.csv',stats)
    return tables,stats,starts


def bootstrap(t,p,cfg,out,oof):
    """2000 whole ND-block resamples, paired across model forms and offset versions.

    Every draw uses all seven initializations. Rows carry block multiplicity;
    no IID-point bootstrap is used. Classical coefficients remain fixed, so
    these intervals are conditional on the frozen beta and source mechanism.
    """
    groups=np.unique(t['group']);G=len(groups);rng=np.random.default_rng(cfg['random_state'])
    rows=[];draws=[];summary=[];fitmodels=cfg['bootstrap_models'];paired=[]
    startcols=['replicate','model','start','initial_gamma','initial_offset','gamma','offset','objective','successful','legal_solution','status','nfev','optimality','at_bound','message']
    with (out/'bootstrap_multistart_solutions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=startcols);writer.writeheader()
        for rep in range(1,cfg['bootstrap_repetitions']+1):
            counts=rng.multinomial(G,np.full(G,1/G))
            multiplicities=dict(zip(groups,counts))
            draws.append(dict(replicate=rep,ND_groups=groups.tolist(),multiplicities=counts.tolist(),n_sampled_blocks=int(counts.sum()),n_distinct_blocks=int((counts>0).sum())))
            for model in fitmodels:
                try:
                    fit,runs=fit_quality(t,p,model,cfg,multiplicities)
                    writer.writerows(dict(replicate=rep,model=model,**r) for r in runs)
                    rows.append(dict(replicate=rep,model=model,success=True,gamma=fit['gamma'],
                                     eta=p['beta']*fit['gamma'] if model.startswith('MQ1') else None,
                                     offset=fit['offset'],objective=fit['objective'],at_bound=fit['at_bound'],
                                     n_distinct_blocks=int((counts>0).sum()),n_starts=fit['n_starts'],
                                     all_starts_converged=all(r['legal_solution'] for r in runs),error=''))
                except (RuntimeError,ValueError,FloatingPointError) as ex:
                    rows.append(dict(replicate=rep,model=model,success=False,gamma=None,eta=None,offset=None,objective=None,
                                     at_bound=None,n_distinct_blocks=int((counts>0).sum()),n_starts=len(cfg['gamma_starts']),
                                     all_starts_converged=False,error=str(ex)))
            ww=np.array([multiplicities[g] for g in t['group']],dtype=float);ww/=ww.sum()
            rmse={m:float(np.sqrt(np.sum(ww*(t['L']-v)**2))) for m,v in oof.items()}
            paired.append(dict(replicate=rep,**{m+'_RMSE':v for m,v in rmse.items()},
                               MQ1_RMSE_gain_over_M0=rmse['M0']-rmse['MQ1'],
                               MQ2_RMSE_gain_over_MQ1=rmse['MQ1']-rmse['MQ2']))
            if rep%cfg['bootstrap_log_interval']==0:
                f.flush();logging.info('ND block bootstrap %d/%d; %d model fits',rep,cfg['bootstrap_repetitions'],len(rows))
                write_csv(out/'quality_bootstrap_replicates.csv',rows)
    for model in fitmodels:
        valid=[r for r in rows if r['model']==model and r['success']]
        for parameter in ['gamma','eta','offset']:
            if parameter=='eta' and not model.startswith('MQ1'):continue
            if parameter=='offset' and not model.endswith('_offset'):continue
            vals=np.array([r[parameter] for r in valid],dtype=float)
            if len(vals)==0:raise RuntimeError('All bootstrap fits failed')
            summary.append(dict(model=model,parameter=parameter,n_requested=cfg['bootstrap_repetitions'],n_success=len(vals),
                                failure_rate=1-len(vals)/cfg['bootstrap_repetitions'],mean=float(vals.mean()),
                                median=float(np.median(vals)),std=float(vals.std(ddof=1)),P025=float(np.quantile(vals,.025)),
                                P975=float(np.quantile(vals,.975)),bootstrap_unit='exact_ND_pair',random_state=cfg['random_state'],
                                fixed_classical_parameters=True,at_bound_count=sum(r['at_bound'] for r in valid)))
    write_csv(out/'quality_bootstrap_replicates.csv',rows)
    write_csv(out/'quality_parameter_bootstrap.csv',summary)
    write_csv(out/'bootstrap_block_draws.csv',draws)
    write_csv(out/'group_cv_paired_bootstrap.csv',paired)
    gains=[]
    for k in ['MQ1_RMSE_gain_over_M0','MQ2_RMSE_gain_over_MQ1']:
        v=np.array([r[k] for r in paired])
        gains.append(dict(comparison=k,mean=float(v.mean()),P025=float(np.quantile(v,.025)),P975=float(np.quantile(v,.975)),
                          fraction_positive=float((v>0).mean()),method='ND-block resampling of fixed OOF predictions; not a nested refit CI'))
    write_csv(out/'group_cv_improvement_uncertainty.csv',gains)
    return summary,rows,gains


def difference_table(t, predictions, masks, outrows, pointrows, label_prefix=''):
    """Validate differences within full source ND groups, primary excludes references."""
    for m,yp in predictions.items():
        obs,pred,qref,refs=group_reference(t,yp)
        for name,mask in masks.items():
            for include_refs in [False,True]:
                ix=mask if include_refs else mask&~refs
                if ix.sum()<2:continue
                outrows.append(evaluate(obs[ix],pred[ix],dataset=t['label'],subset=name,model=m,
                                        evaluation=label_prefix or 'fixed_B6_parameters',include_reference_zero_rows=include_refs,
                                        q_reference_rule='highest Q in full source ND group; no reference used for refitting'))
            for i in np.flatnonzero(mask&~refs):
                pointrows.append(dict(dataset=t['label'],subset=name,model=m,evaluation=label_prefix or 'fixed_B6_parameters',
                                      row_index=int(i),ND_group=t['group'][i],N_params_B=float(t['N'][i]),D_tokens_B=float(t['D'][i]),
                                      Q_score=float(t['Q'][i]),Q_ref=float(qref[i]),Delta_L_obs=float(obs[i]),Delta_L_pred=float(pred[i])))


def monotonicity_details(t,out):
    """Save B6 local rebounds and magnitude; do not confuse noise with reversed trend."""
    groups=defaultdict(list)
    for i,g in enumerate(t['group']):groups[g].append(i)
    rows=[];viol=[]
    for g,idx in groups.items():
        idx=sorted(idx,key=lambda i:t['Q'][i]);loss=t['L'][idx];q=t['Q'][idx];diff=np.diff(loss)
        positives=diff[diff>1e-12]
        rows.append(dict(ND_group=g,n_Q=len(idx),spearman=float(spearmanr(q,loss).statistic),
                         strictly_decreasing=bool((diff< -1e-12).all()),violation_count=len(positives),
                         max_local_rebound=float(positives.max()) if len(positives) else 0.,
                         mean_local_rebound=float(positives.mean()) if len(positives) else 0.))
        for j in np.flatnonzero(diff>1e-12):
            viol.append(dict(ND_group=g,Q_lower=float(q[j]),Q_higher=float(q[j+1]),Loss_lower=float(loss[j]),
                             Loss_higher=float(loss[j+1]),rebound_magnitude=float(diff[j])))
    write_csv(out/'B6_monotonicity_details.csv',rows)
    write_csv(out/'B6_local_violations.csv',viol)
    return dict(n_groups=len(rows),negative_correlation_group_rate=sum(r['spearman']<0 for r in rows)/len(rows),
                strictly_monotone_group_rate=sum(r['strictly_decreasing'] for r in rows)/len(rows),
                local_violation_count=len(viol),mean_rebound=float(np.mean([r['rebound_magnitude'] for r in viol])),
                max_rebound=max(r['rebound_magnitude'] for r in viol))


def stress_test(t,b6,p,predictions,keys6,keys7,out):
    """B8 is solely an incompatibility pressure test; never used for fitting/selection."""
    inside=np.ones(len(t['L']),dtype=bool)
    for k in ['N','D','Q']:inside&=(t[k]>=b6[k].min())&(t[k]<=b6[k].max())
    nonoverlap=np.array([k not in keys6|keys7 for k in t['keys']])
    masks={'full':np.ones(len(t['L']),dtype=bool),'nonoverlap':nonoverlap,
           'within_B6_numeric_ranges':inside,'outside_B6_numeric_ranges':~inside,
           'nonoverlap_within_ranges':inside&nonoverlap,'nonoverlap_outside_ranges':~inside&nonoverlap}
    diagnostics=[];group_rows=[];performance=[]
    for name,mask in masks.items():
        valid_rhos=[];insufficient=0;local=[];groups=defaultdict(list)
        for i in np.flatnonzero(mask):groups[t['group'][i]].append(i)
        for g,idx in groups.items():
            q,l=t['Q'][idx],t['L'][idx]
            if len(np.unique(q))<2 or np.ptp(l)==0:insufficient+=1;continue
            rho=float(spearmanr(q,l).statistic);valid_rhos.append(rho)
            group_rows.append(dict(subset=name,ND_group=g,n_Q=len(idx),spearman_Q_Loss=rho))
            ix=sorted(idx,key=lambda i:t['Q'][i]);local.extend(np.diff(t['L'][ix]).tolist())
        rv=np.array(valid_rhos);lv=np.array(local)
        diagnostics.append(dict(subset=name,n=int(mask.sum()),n_ND_groups=len(groups),n_groups_with_correlation=len(rv),
                                n_groups_insufficient_Q=insufficient,positive_groups=int((rv>0).sum()),negative_groups=int((rv<0).sum()),
                                positive_group_rate=float((rv>0).mean()) if len(rv) else None,
                                spearman_min=float(rv.min()) if len(rv) else None,spearman_P25=float(np.quantile(rv,.25)) if len(rv) else None,
                                spearman_median=float(np.median(rv)) if len(rv) else None,spearman_mean=float(rv.mean()) if len(rv) else None,
                                spearman_P75=float(np.quantile(rv,.75)) if len(rv) else None,spearman_max=float(rv.max()) if len(rv) else None,
                                loss_below_E_count=int((t['L'][mask]<p['E']).sum()),loss_below_E_rate=float((t['L'][mask]<p['E']).mean()),
                                increasing_adjacent_pair_rate=float((lv>1e-12).mean()) if len(lv) else None,
                                role='incompatible_semisynthetic_stress_test'))
        for m,yp in predictions.items():
            performance.append(evaluate(t['L'][mask],yp[mask],dataset='B8',subset=name,model=m,
                                        role='incompatible_semisynthetic_stress_test'))
    write_csv(out/'B8_incompatibility_diagnostics.csv',diagnostics)
    write_csv(out/'B8_subset_group_correlations.csv',group_rows)
    write_csv(out/'B8_range_validation.csv',performance)
    return diagnostics,performance,masks
