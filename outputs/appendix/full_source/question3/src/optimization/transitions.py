"""Type I active-set and Type II curve diagnostics, always on solved P0 paths."""
from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from .resource_solver import load, ResourceProblem, csv

FLAGS=['active_N_lower','active_N_upper','active_D_lower','active_D_upper',
       'active_Q_lower','active_Q_upper','budget_binding']


def bisect_status(fun,lo,hi,initial,cfg):
    """Refine the first bracketed boolean crossing in log10 budget."""
    for i in range(cfg['transition_max_bisections']):
        if hi-lo<=cfg['transition_log10_bracket_tolerance']:break
        mid=(lo+hi)/2
        if bool(fun(10**mid))==initial:lo=mid
        else:hi=mid
    return lo,hi,i+1


def soft_fit(x,y,cfg,rng):
    """Continuous single-break hinge versus one line; block residual perturbation CI."""
    n=len(x);m=cfg['soft_transition_min_points_per_side']
    if n<2*m+1 or np.ptp(y)<1e-10:return dict(status='NO_VARIATION',delta_BIC=np.nan,significant=False)
    xx=x-x.mean();linear=np.c_[np.ones(n),xx]
    coef=np.linalg.lstsq(linear,y,rcond=None)[0];res=y-linear@coef
    sse0=float(res@res);locs=(x[m-1:n-m]+x[m:n-m+1])/2
    designs=np.array([np.c_[np.ones(n),xx,np.maximum(0,x-k)] for k in locs])
    inv=np.array([np.linalg.pinv(a) for a in designs])
    coefs=np.einsum('kjn,n->kj',inv,y);fits=np.einsum('knj,kj->kn',designs,coefs)
    sses=np.sum((fits-y)**2,axis=1);best=int(np.argmin(sses));sse1=float(sses[best])
    floor=max(1e-24,float(np.var(y))*n*1e-14)
    bic0=n*np.log(max(sse0,floor)/n)+2*np.log(n)
    bic1=n*np.log(max(sse1,floor)/n)+4*np.log(n) # intercept, two slopes, selected breakpoint
    delta=float(bic0-bic1);significant=delta>=cfg['soft_transition_delta_BIC']
    result=dict(status='FITTED',n=n,delta_BIC=delta,significant=significant,break_log10C=float(locs[best]),
        break_budget=float(10**locs[best]),slope_left=float(coefs[best,1]),slope_right=float(coefs[best,1]+coefs[best,2]),
        points_left=int((x<locs[best]).sum()),points_right=int((x>locs[best]).sum()),linear_SSE=sse0,piecewise_SSE=sse1)
    if significant:
        residual=y-fits[best];residual-=residual.mean();reps=cfg['soft_transition_bootstrap_repetitions'];block=cfg['soft_transition_bootstrap_block_length']
        perturb=[]
        for _ in range(reps):
            starts=rng.integers(0,n,size=int(np.ceil(n/block)))
            idx=np.concatenate([(s+np.arange(block))%n for s in starts])[:n]
            perturb.append(fits[best]+residual[idx])
        ys=np.array(perturb);boot_coef=np.einsum('kjn,rn->rkj',inv,ys)
        boot_fit=np.einsum('knj,rkj->rkn',designs,boot_coef)
        boot_sse=np.sum((boot_fit-ys[:,None,:])**2,axis=2);chosen=locs[np.argmin(boot_sse,axis=1)]
        lows,highs=np.quantile(chosen,[.025,.975])
        result.update(break_interval_low=10**float(lows),break_interval_high=10**float(highs),
            interval_method='circular_block_residual_perturbation_not_physical_sampling_CI',bootstrap_repetitions=reps,
            breakpoint_grid_resolution='midpoints between solved log-budget points')
    return result


def complete_feasibility_boundary_events(root):
    """Include active releases between the analytic feasibility boundary and first grid point."""
    root=Path(root);out=root/'outputs/optimization/phase2'
    setup,frozen,cfg,model,candidates,ext=load(root)
    hard=pd.read_csv(out/'typeI_active_set_transitions.csv',float_precision='round_trip')
    refined=pd.read_csv(out/'refined_budget_solutions.csv',float_precision='round_trip').to_dict('records')
    extra=[]
    for domain in ('main','extended'):
        path=pd.read_csv(out/f'P0_{domain}_budget_path.csv',float_precision='round_trip')
        for (family,lctx),frame in path[path.feasible==True].groupby(['cost_function','L_ctx']):
            first=frame.sort_values('budget').iloc[0]
            problem=ResourceProblem(setup,model,cfg,ext,first.budget,family,int(lctx),domain,'P0')
            minimum=problem.a*problem.nlo*problem.dlo*1e18
            if first.budget<=minimum*(1+1e-12):continue
            minimum_record,_=ResourceProblem(setup,model,cfg,ext,minimum,family,int(lctx),domain,'P0').solve()
            cache={minimum:minimum_record,float(first.budget):first.to_dict()}
            refined.append(minimum_record)
            def solve(C):
                if C not in cache:
                    cache[C],_=ResourceProblem(setup,model,cfg,ext,C,family,int(lctx),domain,'P0').solve()
                    refined.append(cache[C])
                return cache[C]
            for flag in FLAGS:
                if bool(minimum_record[flag])==bool(first[flag]):continue
                old=hard[(hard.domain==domain)&(hard.cost_function==family)&(hard.L_ctx==lctx)&(hard.constraint==flag)]
                if len(old) and old.budget_critical.min()<=first.budget:continue
                lo,hi,count=bisect_status(lambda C:solve(C)[flag],np.log10(minimum),np.log10(first.budget),bool(minimum_record[flag]),cfg)
                extra.append(dict(domain=domain,cost_function=family,L_ctx=int(lctx),constraint=flag,
                    transition=f'{bool(minimum_record[flag])}→{bool(first[flag])}',budget_critical=10**((lo+hi)/2),
                    bracket_low=10**lo,bracket_high=10**hi,method='feasibility_boundary_log_budget_bisection',
                    active_tolerance=cfg['active_tolerance'],bisections=count,log10_bracket_width=hi-lo))
    if extra:
        csv(out/'typeI_active_set_transitions.csv',pd.concat([hard,pd.DataFrame(extra)],ignore_index=True))
        csv(out/'refined_budget_solutions.csv',refined)
        p=out/'transition_execution_summary.json';summary=json.loads(p.read_text())
        summary['typeI_rows']=len(hard)+len(extra);summary['feasibility_boundary_release_rows']=len(extra)
        p.write_text(json.dumps(summary,indent=2),encoding='utf-8')


def run_diagnose(root):
    """Detect transitions only after solved paths exist; keep main/extrapolation apart."""
    root=Path(root);out=root/'outputs/optimization/phase2'
    setup,frozen,cfg,model,candidates,ext=load(root)
    path=pd.concat([pd.read_csv(out/f'P0_{d}_budget_path.csv',float_precision='round_trip') for d in ('main','extended')],ignore_index=True)
    typeI=[];typeII=[];thresholds=[];peaks=[];saturation=[];shadow=[];refined=[];monotonic=[]
    rng=np.random.default_rng(cfg['random_state']);families=list(setup['official']['quality_costs'])
    for (domain,family,lctx),frame in path.groupby(['domain','cost_function','L_ctx'],sort=False):
        f=frame[frame.feasible==True].sort_values('budget').copy();base=dict(domain=domain,cost_function=family,L_ctx=int(lctx))
        if f.empty:continue
        cache={float(r['budget']):r for r in f.to_dict('records')}
        def solve(C):
            C=float(C)
            if C not in cache:
                rec,_=ResourceProblem(setup,model,cfg,ext,C,family,int(lctx),domain,'P0').solve()
                cache[C]=rec;refined.append(rec)
            return cache[C]
        prob=ResourceProblem(setup,model,cfg,ext,float(f.budget.max()),family,int(lctx),domain,'P0')
        Cmin=prob.a*prob.nlo*prob.dlo*1e18
        Csat=prob.dmax*(prob.a*prob.nmax+float(prob.h(1.)))*1e18
        saturation.append(dict(**base,exact_saturation_budget=Csat,N_upper=prob.nmax,D_upper=prob.dmax,Q_upper=1.,
                               within_budget_grid=Csat<=frame.budget.max(),first_saturated_grid_budget=float(f.loc[f.evidence_domain_saturated==True,'budget'].min()) if (f.evidence_domain_saturated==True).any() else np.nan))
        typeI.append(dict(**base,constraint='feasibility',transition='INFEASIBLE→FEASIBLE',budget_critical=Cmin,
                          bracket_low=Cmin,bracket_high=Cmin,method='analytic_minimum_cost',active_tolerance=cfg['active_tolerance']))
        for flag in FLAGS:
            arr=f[flag].astype(bool).to_numpy();bud=f.budget.to_numpy()
            for i in np.flatnonzero(arr[1:]!=arr[:-1]):
                lo,hi,count=bisect_status(lambda C:solve(C)[flag],np.log10(bud[i]),np.log10(bud[i+1]),bool(arr[i]),cfg)
                C=10**((lo+hi)/2)
                typeI.append(dict(**base,constraint=flag,transition=f'{bool(arr[i])}→{bool(arr[i+1])}',budget_critical=C,
                    bracket_low=10**lo,bracket_high=10**hi,method='local_log_budget_bisection',bisections=count,
                    active_tolerance=cfg['active_tolerance'],log10_bracket_width=hi-lo))
        # First crossing, with domain-minimum lower bracket if already crossed at first grid point.
        targets=[('departure_Q0_plus_0.01',cfg['Q0_main']+cfg['quality_departure_absolute'])]+[(str(q),q) for q in cfg['quality_thresholds']]
        for name,target in targets:
            tolerance=1e-7 if target==1 else 1e-9
            reached=f.Q_star.to_numpy()>=target-tolerance;inds=np.flatnonzero(reached)
            if not len(inds):
                thresholds.append(dict(**base,threshold=name,target_Q=target,status='NOT_REACHED_IN_GRID',budget_threshold=np.nan));continue
            k=int(inds[0]);upper=float(f.budget.iloc[k]);lower=float(f.budget.iloc[k-1]) if k else Cmin
            lo,hi,_=bisect_status(lambda C:solve(C)['Q_star']>=target-tolerance,np.log10(lower),np.log10(upper),False,cfg)
            thresholds.append(dict(**base,threshold=name,target_Q=target,status='REACHED',budget_threshold=10**((lo+hi)/2),
                bracket_low=10**lo,bracket_high=10**hi,Q_tolerance=tolerance,
                Q_path_nonmonotonic=bool((np.diff(f.Q_star)<-1e-5).any())))
        # Shadow price finite differences on actual, unequal budget coordinates.
        Cs=f.budget.to_numpy();Ls=f.L_star.to_numpy();x=np.log10(Cs)
        mu=-np.gradient(Ls,Cs,edge_order=2);elasticity=-np.gradient(np.log(Ls-model.c['E']),np.log(Cs),edge_order=2)
        mu[np.abs(mu)<1e-30]=0
        for i,r in enumerate(f.to_dict('records')):
            shadow.append(dict(**base,budget=Cs[i],L_star=Ls[i],shadow_price=mu[i],budget_elasticity=elasticity[i],
                endpoint_one_sided=i in (0,len(Cs)-1),negative_mu_numerical=bool(mu[i]<-1e-24)))
        deltas=np.diff(Ls);monotonic.append(dict(**base,maximum_loss_increase=float(max(0,deltas.max())),
             passed=bool((deltas<=cfg['loss_validation_tolerance']).all()),n_comparisons=len(deltas)))
        for metric in ['s_train','s_quality','s_attn','ln_N','ln_D','Q']:
            values=np.log(f.N_star.to_numpy()) if metric=='ln_N' else np.log(f.D_star.to_numpy()) if metric=='ln_D' else f.Q_star.to_numpy() if metric=='Q' else f[metric].to_numpy()
            result=soft_fit(x,values,cfg,rng);typeII.append(dict(**base,metric=metric,**result))
        values=f.s_quality.to_numpy();k=int(np.argmax(values));peakC=Cs[k];peak=values[k]
        reversal=k>0 and k<len(Cs)-1 and peak-values[0]>1e-4 and peak-values[-1]>1e-4
        if reversal:
            result=minimize_scalar(lambda logC:-solve(10**logC)['s_quality'],bounds=(np.log10(Cs[k-1]),np.log10(Cs[k+1])),method='bounded',options={'xatol':1e-5})
            peakC=10**result.x;peak=solve(peakC)['s_quality']
        near=[r for r in typeI if r['domain']==domain and r['cost_function']==family and r['L_ctx']==lctx and r['constraint']!='feasibility']
        closest=min(near,key=lambda r:abs(np.log10(r['budget_critical']/peakC))) if near else None
        peaks.append(dict(**base,quality_share_reversal=bool(reversal),C_peak_Q=peakC,peak_s_quality=peak,
              start_s_quality=values[0],end_s_quality=values[-1],nearest_active_transition=closest['constraint'] if closest else '',
              nearest_transition_budget=closest['budget_critical'] if closest else np.nan,
              distance_log10_budget=abs(np.log10(closest['budget_critical']/peakC)) if closest else np.nan,
              active_Q_upper_at_peak=solve(peakC)['active_Q_upper']))
        logging.info('Transitions %s %s L=%s complete; refined %s budget points',domain,family,lctx,len(cache)-len(f))
        # Incremental checkpoint, useful if a desktop connection closes.
        for name,records in [('typeI_active_set_transitions.csv',typeI),('typeII_soft_transitions.csv',typeII),
             ('quality_threshold_budgets.csv',thresholds),('quality_cost_share_peaks.csv',peaks),('domain_saturation_budgets.csv',saturation),
             ('shadow_price_budget_elasticity.csv',shadow),('monotonicity_checks.csv',monotonic),('refined_budget_solutions.csv',refined)]:csv(out/name,records)
    if not all(r['passed'] for r in monotonic):raise ValueError('Loss increases with budget; must diagnose optimizer')
    summary=dict(typeI_rows=len(typeI),typeII_significant_metrics=sum(bool(r['significant']) for r in typeII),
       typeII_rows=len(typeII),quality_share_reversal_scenarios=sum(bool(r['quality_share_reversal']) for r in peaks),
       main_grid_points=path[path.domain=='main'].budget.nunique(),P2_performed=False)
    (out/'transition_execution_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
