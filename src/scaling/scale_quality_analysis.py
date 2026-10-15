"""Grouped model comparison, audited multi-start bootstrap and sensitivity surfaces."""
from __future__ import annotations
import csv
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from .quality_extension import subset,group_reference
from .scale_quality import PARAMETERS,fit_scale,predict_scale,marginal_quality
from .classical import predict,metrics
from .io import write_csv,csv_rows


def evaluate(t,pred,model,stage,mask=None):
    """Report absolute and reference-excluded within-ND differences together."""
    if mask is None:mask=np.ones(len(t['L']),dtype=bool)
    obs,dp,refq,refs=group_reference(t,pred)
    active=mask&~refs
    if not np.all(refq[mask]==1):raise ValueError('This B6/B7 evaluation requires observed Q=1 references')
    return [dict(model=model,stage=stage,target='absolute_Loss',**metrics(t['L'][mask],pred[mask])),
            dict(model=model,stage=stage,target='within_ND_DeltaL',reference_zero_rows_excluded=True,
                 reference_rule='Q=1 from full source ND group; reference may overlap B6 for B7',
                 **metrics(obs[active],dp[active]))]


def group_cv(t,p,reference,cfg,out):
    """Independent 50-start train-fold fits; ND blocks cannot be split."""
    models=['M0']+cfg['models'];oof={m:np.full(len(t['L']),np.nan) for m in models}
    folds=[];fits=[];starts=[];foldmetrics=[];rowfold=np.zeros(len(t['L']),dtype=int)
    for fold,(train,test) in enumerate(GroupKFold(n_splits=cfg['group_cv_folds']).split(t['N'],groups=t['group']),1):
        tr,te=subset(t,train),subset(t,test);rowfold[test]=fold
        trgroups,tegroups=sorted(set(tr['group'])),sorted(set(te['group']))
        if set(trgroups)&set(tegroups):raise AssertionError('ND group leakage')
        folds.append(dict(fold=fold,train_rows=len(train),test_rows=len(test),train_groups=trgroups,test_groups=tegroups))
        for m in models:
            if m=='M0':yp=predict(p,te['N'],te['D'])
            else:
                f,rs=fit_scale(tr,p,reference,m,cfg,cfg['random_state']+fold)
                fits.append(dict(fold=fold,**f));starts.extend(dict(stage='GroupCV',model=m,fold=fold,**r) for r in rs)
                yp=predict_scale(te,p,reference,m,f['parameters'])
            oof[m][test]=yp;foldmetrics+=evaluate(te,yp,m,'fold_'+str(fold))
        logging.info('Group CV fold %d/5 completed, training groups=%d test groups=%d',fold,len(trgroups),len(tegroups))
    comparison=[]
    for m,yp in oof.items():
        if not np.isfinite(yp).all():raise AssertionError('Incomplete OOF predictions')
        comparison+=evaluate(t,yp,m,'B6_5fold_GroupCV_OOF')
    pointrows=[dict(row_index=i,fold=int(rowfold[i]),ND_group=t['group'][i],N_params_B=float(t['N'][i]),D_tokens_B=float(t['D'][i]),
                    Q_score=float(t['Q'][i]),actual_Loss=float(t['L'][i]),**{m+'_prediction':float(v[i]) for m,v in oof.items()}) for i in range(len(t['L']))]
    write_csv(out/'group_cv_predictions.csv',pointrows);write_csv(out/'group_cv_folds.csv',folds)
    write_csv(out/'group_cv_model_parameters.csv',fits);write_csv(out/'group_cv_fold_metrics.csv',foldmetrics)
    write_csv(out/'B6_group_cv.csv',comparison)
    return oof,comparison,starts


def q_level_cv(t,p,reference,cfg,out):
    """Hold out each entire Q layer; fitting always uses training observations only."""
    rows=[];starts=[];predpoints=[]
    levels=np.unique(t['Q'])
    for j,q in enumerate(levels):
        tr=subset(t,t['Q']!=q);te=subset(t,t['Q']==q)
        for m in cfg['models']:
            f,rs=fit_scale(tr,p,reference,m,cfg,cfg['random_state']+100+j)
            starts.extend(dict(stage='leave_one_Q_out',model=m,held_Q=float(q),**r) for r in rs)
            yp=predict_scale(te,p,reference,m,f['parameters'])
            rows.append(dict(model=m,held_Q=float(q),test_kind='endpoint_extrapolation' if q in [levels[0],levels[-1]] else 'interior_interpolation',
                             parameters=f['parameters'],boundary_parameters=f['boundary_parameters'],**metrics(te['L'],yp)))
            predpoints.extend(dict(model=m,held_Q=float(q),ND_group=g,actual_Loss=float(y),predicted_Loss=float(v)) for g,y,v in zip(te['group'],te['L'],yp))
    write_csv(out/'leave_one_Q_level_out.csv',rows);write_csv(out/'leave_one_Q_predictions.csv',predpoints)
    return rows,starts


_WORKER_DATA=None


def initialize_worker(t,p,reference,cfg):
    """Set small read-only worker arrays; no source/output writes in workers."""
    global _WORKER_DATA
    from threadpoolctl import threadpool_limits
    # Small 360x4 systems benefit from one BLAS thread per CPU worker.
    threadpool_limits(limits=1)
    _WORKER_DATA=(t,p,reference,cfg)


def bootstrap_worker(job):
    """Fit all three models on the same whole-ND bootstrap draw, 50 starts each."""
    rep,groups,counts=job;t,p,reference,cfg=_WORKER_DATA
    mult=dict(zip(groups,counts));rows=[];starts=[]
    for m in cfg['models']:
        try:
            f,rs=fit_scale(t,p,reference,m,cfg,cfg['random_state']+1000+rep,mult,
                           count=cfg['bootstrap_multistart_count'])
            rows.append(dict(replicate=rep,model=m,success=True,parameters=f['parameters'],objective=f['objective'],
                             jacobian_rank=f['jacobian_rank'],jacobian_condition=f['jacobian_condition'],
                             boundary_hit=f['boundary_hit'],boundary_parameters=f['boundary_parameters'],
                             n_starts=f['n_starts'],n_successful_starts=f['n_successful_starts'],error=''))
            starts.extend(dict(replicate=rep,model=m,**r) for r in rs)
        except (RuntimeError,ValueError,FloatingPointError) as ex:
            rows.append(dict(replicate=rep,model=m,success=False,parameters=None,objective=None,jacobian_rank=None,
                             jacobian_condition=None,boundary_hit=None,boundary_parameters=[],
                             n_starts=cfg['bootstrap_multistart_count'],n_successful_starts=0,error=str(ex)))
    return rep,rows,starts


def resampling_jobs(t,cfg):
    """Uniform with-replacement sampling over 45 exact ND blocks."""
    groups=np.unique(t['group']);G=len(groups);rng=np.random.default_rng(cfg['random_state'])
    return [(rep,groups.tolist(),rng.multinomial(G,np.full(G,1/G)).tolist()) for rep in range(1,cfg['bootstrap_repetitions']+1)]


def bootstrap_models(t,p,reference,cfg,out,resume=False):
    """Bounded CPU workers; main process writes an audit of every one of 300k starts."""
    jobs=resampling_jobs(t,cfg);rows=[];params=[];correlations=[]
    write_csv(out/'bootstrap_block_draws.csv',[dict(replicate=r,ND_groups=g,multiplicities=c,n_sampled_blocks=sum(c),n_distinct_blocks=sum(v>0 for v in c)) for r,g,c in jobs])
    cols=['replicate','model','start','initial_parameters','parameters','objective','convergence','legal_solution','status','nfev',
          'jacobian_rank','jacobian_condition','jacobian_singular_values','boundary_hit','boundary_parameters','per_parameter_boundary','message']
    completed=0
    if resume:
        import json
        _,saved=csv_rows(out/'bootstrap_replicates.csv')
        for r in saved:
            rows.append(dict(replicate=int(r['replicate']),model=r['model'],success=r['success']=='True',
                             parameters=json.loads(r['parameters']) if r['parameters'] else None,
                             objective=float(r['objective']) if r['objective'] else None,
                             jacobian_rank=int(r['jacobian_rank']) if r['jacobian_rank'] else None,
                             jacobian_condition=float(r['jacobian_condition']) if r['jacobian_condition'] else None,
                             boundary_hit=r['boundary_hit']=='True',boundary_parameters=json.loads(r['boundary_parameters']),
                             n_starts=int(r['n_starts']),n_successful_starts=int(r['n_successful_starts']),error=r['error']))
        completed=max(r['replicate'] for r in rows)
        if len(rows)!=completed*len(cfg['models']):raise ValueError('Incomplete bootstrap checkpoint')
        if any({r['model'] for r in rows if r['replicate']==i}!=set(cfg['models']) for i in range(1,completed+1)):
            raise ValueError('Checkpoint model/replicate gaps')
        expected=completed*len(cfg['models'])*cfg['bootstrap_multistart_count'];retained=0
        temp=out/'bootstrap_resume_verified_prefix.csv'
        with (out/'bootstrap_multistart_solutions.csv').open(encoding='utf-8-sig',newline='') as f,temp.open('w',encoding='utf-8-sig',newline='') as g:
            rd=csv.DictReader(f);wr=csv.DictWriter(g,fieldnames=cols);wr.writeheader()
            for r in rd:
                if retained==expected:break
                if None in r or any(v is None for v in r.values()) or int(r['replicate'])>completed:raise ValueError('Invalid checkpoint audit prefix')
                wr.writerow(r);retained+=1
        if retained!=expected:raise ValueError('Missing multistart audit in checkpoint')
        temp.replace(out/'bootstrap_multistart_solutions.csv')
        jobs=[job for job in jobs if job[0]>completed]
        logging.info('Resuming from %d completed draws, preserving %d start records; BLAS threads per worker=1',completed,retained)
        from .io import write_json
        write_json(out/'bootstrap_resume_runtime.json',dict(preserved_replicates=completed,preserved_multistart_records=retained,
                   change='BLAS threads per worker reduced to 1; model/bounds/seed/starts/resamples unchanged'))
    initialize_worker(t,p,reference,cfg)
    executor=None
    if cfg['bootstrap_workers']>1:
        executor=ProcessPoolExecutor(max_workers=cfg['bootstrap_workers'],initializer=initialize_worker,initargs=(t,p,reference,cfg))
        results=executor.map(bootstrap_worker,jobs,chunksize=1)
    else:results=map(bootstrap_worker,jobs)
    try:
        with (out/'bootstrap_multistart_solutions.csv').open('a' if resume else 'w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=cols)
            if not resume:writer.writeheader()
            import json
            for rep,rr,starts in results:
                rows.extend(rr)
                writer.writerows({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in starts)
                if rep%cfg['bootstrap_log_interval']==0:
                    f.flush();write_csv(out/'bootstrap_replicates.csv',rows)
                    logging.info('ND block bootstrap %d/%d, %d model fits, %d starts per model',rep,cfg['bootstrap_repetitions'],len(rows),cfg['bootstrap_multistart_count'])
    finally:
        if executor is not None:executor.shutdown(wait=True,cancel_futures=True)
    write_csv(out/'bootstrap_replicates.csv',rows)
    for m in cfg['models']:
        valid=[r for r in rows if r['model']==m and r['success']]
        if not valid:raise RuntimeError('All block-bootstrap fits failed: '+m)
        for key in PARAMETERS[m]:
            v=np.array([r['parameters'][key] for r in valid]);lo,hi=cfg['parameter_bounds'][key]
            hit=np.array([key in r['boundary_parameters'] for r in valid])
            params.append(dict(model=m,parameter=key,n_requested=cfg['bootstrap_repetitions'],n_success=len(v),
                               mean=float(v.mean()),median=float(np.median(v)),std=float(v.std(ddof=1)),
                               P025=float(np.quantile(v,.025)),P975=float(np.quantile(v,.975)),
                               failure_rate=1-len(valid)/cfg['bootstrap_repetitions'],parameter_boundary_rate=float(hit.mean()),
                               model_any_boundary_rate=sum(r['boundary_hit'] for r in valid)/len(valid),
                               lower_bound=lo,upper_bound=hi,bootstrap_unit='exact_ND_pair',fixed_classical_parameters=True))
    correlations=parameter_correlation_rows(rows,cfg)
    write_csv(out/'bootstrap_parameters.csv',params);write_csv(out/'parameter_correlations.csv',correlations)
    return params,rows,correlations


def parameter_correlation_rows(rows,cfg):
    """Do not interpret floating noise at a fixed active bound as covariance."""
    result=[]
    for m in cfg['models']:
        valid=[r for r in rows if r['model']==m and r['success']]
        a=np.array([[r['parameters'][key] for key in PARAMETERS[m]] for r in valid])
        usable={}
        for j,key in enumerate(PARAMETERS[m]):
            fixed_bound=all(key in r['boundary_parameters'] for r in valid) and np.ptp(a[:,j])<=cfg['boundary_absolute_tolerance']
            spread=np.std(a[:,j])>np.finfo(float).eps*max(1,abs(np.mean(a[:,j])))
            usable[key]=not fixed_bound and spread
        for j,key in enumerate(PARAMETERS[m]):
            for l,other in enumerate(PARAMETERS[m]):
                defined=usable[key] and usable[other]
                result.append(dict(model=m,parameter_1=key,parameter_2=other,
                                   Pearson=float(np.corrcoef(a[:,j],a[:,l])[0,1]) if defined else None,
                                   Spearman=float(spearmanr(a[:,j],a[:,l]).statistic) if defined else None,
                                   n=len(valid),defined=bool(defined),
                                   undefined_reason='' if defined else 'At least one parameter is numerically constant or fixed at an active bound'))
    return result


def sensitivity_surfaces(t,p,reference,fitted,out,domain_rows,cfg):
    """Analytic/finite-difference Q gradients and within-design N/D trend checks."""
    rows=[];checks=[];trends=[];domain=[]
    for m,f in fitted.items():
        params=f['parameters'];grad=marginal_quality(t,p,reference,m,params);yp=predict_scale(t,p,reference,m,params)
        qt={**t,'Q':np.ones(len(t['Q']))};refpred=predict_scale(qt,p,reference,m,params)
        step=1e-5
        plus={**t,'Q':np.minimum(t['Q']+step,1.)};minus={**t,'Q':np.maximum(t['Q']-step,t['Q'].min())}
        numeric=(predict_scale(plus,p,reference,m,params)-predict_scale(minus,p,reference,m,params))/(plus['Q']-minus['Q'])
        checks.append(dict(model=m,n=len(grad),analytic_violation_count=int((grad>=0).sum()),numerical_violation_count=int((numeric>=0).sum()),
                           analytic_violation_rate=float((grad>=0).mean()),numerical_violation_rate=float((numeric>=0).mean()),
                           max_abs_analytic_numeric_discrepancy=float(max(abs(grad-numeric))),
                           min_derivative=float(grad.min()),max_derivative=float(grad.max())))
        for i in range(len(t['L'])):
            gamma=params['gamma0']*(t['N'][i]/reference['N0'])**(-params['u'])*(t['D'][i]/reference['D0'])**params['v'] if m=='S1' else None
            rows.append(dict(model=m,N_params_B=float(t['N'][i]),D_tokens_B=float(t['D'][i]),Q_score=float(t['Q'][i]),
                             effective_gamma=gamma,dL_dQ=float(grad[i]),abs_quality_sensitivity=float(abs(grad[i])),
                             numerical_dL_dQ=float(numeric[i]),Delta_Loss_relative_Q1=float(yp[i]-refpred[i])))
        # Assess actual |dL/dQ|; do not mistake increasing gamma for increasing marginal Loss sensitivity.
        for axis in ['N','D']:
            other='D' if axis=='N' else 'N';g=defaultdict(list)
            for i in range(len(grad)):g[(float(t[other][i]),float(t['Q'][i]))].append(i)
            relative=[]
            for idx in g.values():
                idx=sorted(idx,key=lambda i:t[axis][i]);v=abs(grad[idx]);relative.extend((np.diff(v)/np.maximum(v[:-1],1e-300)).tolist())
            a=np.array(relative);tol=1e-8
            trends.append(dict(model=m,axis=axis,n_adjacent_pairs=len(a),decreasing_rate=float((a< -tol).mean()),
                               increasing_rate=float((a>tol).mean()),flat_rate=float((abs(a)<=tol).mean()),
                               min_relative_change=float(a.min()),max_relative_change=float(a.max()),comparison_numerical_tolerance=tol))
        design=[(float(n),float(d)) for n in np.unique(t['N']) for d in np.unique(t['D'])]
        # Fixed domain scores are used only as numeric Q coordinates, never in a mixture.
        coordinates=[dict(domain=r['domain'],Q=float(r['Q_domain'])) for r in domain_rows]
        low=min(x['Q'] for x in coordinates);high=max(x['Q'] for x in coordinates)
        coordinates += [dict(domain='Q_interval_low',Q=low),dict(domain='Q_interval_high',Q=high)]
        for coord in coordinates:
            tt=dict(N=np.array([n for n,d in design]),D=np.array([d for n,d in design]),Q=np.full(len(design),coord['Q']))
            derivative=marginal_quality(tt,p,reference,m,params);prediction=predict_scale(tt,p,reference,m,params)
            baseline=predict(p,tt['N'],tt['D'])
            for i,(n,d) in enumerate(design):
                domain.append(dict(model=m,domain_coordinate=coord['domain'],N_params_B=n,D_tokens_B=d,Q_coordinate=coord['Q'],
                                    dL_dQ=float(derivative[i]),abs_quality_sensitivity=float(abs(derivative[i])),
                                    Delta_Loss_relative_Q1=float(prediction[i]-baseline[i]),
                                    numeric_support_only=True,semantic_Q_equivalence_confirmed=False))
    write_csv(out/'quality_sensitivity_surface.csv',rows);write_csv(out/'monotonicity_constraint_checks.csv',checks)
    write_csv(out/'quality_sensitivity_scale_trends.csv',trends);write_csv(out/'question1_Q_interval_analysis.csv',domain)
    return rows,checks,trends,domain
