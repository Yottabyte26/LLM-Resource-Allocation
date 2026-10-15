"""Frozen log-contrast simplex analysis; never fit or alter previous phases."""
from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq, differential_evolution, minimize
from scipy.special import softmax, xlogy
from scipy.spatial.distance import jensenshannon, pdist, squareform
from scaling.mixture_transfer import frozen_metadata, predict_recipe
from .resource_solver import csv, load, ResourceProblem


def log_contrast(meta):
    """Algebraically expand the frozen standardized linear ILR predictor."""
    coef = np.asarray(meta['coefficients']) / np.asarray(meta['X_scale'])
    b = float(meta['y_scale']) * coef @ np.asarray(meta['ilr_basis'])
    a = float(meta['y_mean']) + float(meta['y_scale']) * (
        float(meta['intercept']) - np.asarray(meta['X_mean']) @ coef)
    if abs(b.sum()) > 1e-12:
        raise ValueError('Frozen basis no longer gives a zero-sum log contrast')
    return a, b


def exact_extreme(b, epsilon, direction):
    """Global certificate: floor unfavorable signs, water-fill favorable signs.

    To minimize, any positive coefficient is at the floor: transferring its
    mass to a negative coefficient strictly decreases the objective. The
    remaining negative-log problem is convex. Maximization is symmetric.
    """
    favor = -b if direction == 'minM' else b
    active = favor > 0
    if not active.any():
        raise ValueError('Constant or unexpected frozen mixture model')
    p = np.full(len(b), epsilon)
    mass = 1 - (~active).sum() * epsilon
    t = brentq(lambda t: np.maximum(epsilon, t * favor[active]).sum() - mass,
               0, 2 / favor[active].min(), xtol=1e-14)
    p[active] = np.maximum(epsilon, t * favor[active])
    if abs(p.sum() - 1) > 1e-12:
        raise ValueError('Analytic simplex closure failed')
    return p


def recipe_summary(p, pref, candidates, names, cfg):
    """Describe concentration and distance in frozen positive model space."""
    a = candidates[names].to_numpy(float)
    entropy = float(-xlogy(p, p).sum())
    l1 = np.abs(a-p).sum(axis=1)
    js = jensenshannon(a, np.broadcast_to(p, a.shape), axis=1)
    i, k = int(l1.argmin()), int(js.argmin())
    outside = (p < a.min(axis=0)-1e-12) | (p > a.max(axis=0)+1e-12)
    top = np.argsort(-p)[:5]
    far = bool(l1[i]>cfg.get('empirical_neighbor_L1_P95',np.inf) or
               js[k]>cfg.get('empirical_neighbor_JS_P95',np.inf))
    return dict(max_share=float(p.max()), entropy=entropy,
        normalized_entropy=entropy/np.log(len(p)), effective_domain_count=float(np.exp(entropy)),
        HHI=float(p@p), L1_to_ref=float(np.abs(p-pref).sum()),
        L2_to_ref=float(np.linalg.norm(p-pref)), JS_distance_to_ref=float(jensenshannon(p, pref)),
        minimum_empirical_L1=float(l1[i]), minimum_empirical_JS_distance=float(js[k]),
        nearest_empirical_recipe=str(candidates.candidate_id.iloc[i]),
        nearest_empirical_recipe_by_JS=str(candidates.candidate_id.iloc[k]),
        out_of_empirical_range_domains=int(outside.sum()),
        outside_domains=json.dumps(np.asarray(names)[outside].tolist()),
        distance_exceeds_empirical_neighbor_P95=far,
        unsupported_mixture_extrapolation=bool(outside.any() or far),
        unsupported_rule='outside coordinate min/max OR nearest distance exceeds empirical leave-one-out nearest-neighbor P95; descriptive diagnostic, not a confidence region',
        highly_concentrated=bool(p.max()>cfg['high_concentration_share'] or
                                np.exp(entropy)<cfg['high_concentration_effective_domains']),
        near_degenerate=bool(p.max()>cfg['near_degenerate_share']),
        top5=json.dumps([dict(domain=names[j], share=float(p[j])) for j in top]))


def support_calibration(candidates,names,cfg,out):
    """Data-derived descriptive spacing benchmark; never refit M(p)."""
    a=candidates[names].to_numpy(float);result=dict(cfg);rows=[]
    for metric,key in [('cityblock','empirical_neighbor_L1_P95'),('jensenshannon','empirical_neighbor_JS_P95')]:
        d=squareform(pdist(a,metric));np.fill_diagonal(d,np.inf);nearest=d.min(axis=1)
        result[key]=float(np.quantile(nearest,.95))
        rows.append(dict(distance=metric,candidate_count=len(a),nearest_neighbor_P50=float(np.median(nearest)),
            nearest_neighbor_P95=result[key],nearest_neighbor_max=float(nearest.max()),
            purpose='descriptive empirical sparsity benchmark; not a statistical support boundary'))
    csv(out/'empirical_distance_calibration.csv',rows)
    return result


def simplex_search(root, cfg):
    """Three numerical methods, analytic certificates, and unboundedness witnesses."""
    root = Path(root); out = root/'outputs/optimization/phase3'
    setup, frozen, solver_cfg, model, candidates, empirical = load(root)
    meta = frozen_metadata(root); names = meta['domain_order']; epsilon = float(meta['zero_delta'])
    a, b = log_contrast(meta); pref = np.asarray(frozen['reference']['p_ref'])
    cfg=support_calibration(candidates,names,cfg,out)
    rng = np.random.default_rng(cfg['random_state'])
    samples = []
    for alpha in cfg['dirichlet_concentrations']:
        block = rng.dirichlet(np.full(17, alpha), cfg['dirichlet_samples']//4)
        # A numerical generator can underflow positive Dirichlet entries to 0.
        # P2-A's explicit floor keeps these values safe, without changing B.
        samples.append(block)
    raw_samples = np.concatenate(samples)
    A_samples = epsilon + (1-17*epsilon)*raw_samples
    exact = {role:exact_extreme(b, epsilon, role) for role in ('minM','maxM')}
    for p in exact.values():
        alpha = cfg['targeted_concentration'] * (p-epsilon)/(1-17*epsilon)
        alpha = np.maximum(alpha, cfg['targeted_alpha_floor'])
        block = rng.dirichlet(alpha, cfg['targeted_dirichlet_samples_per_extreme'])
        A_samples = np.r_[A_samples, epsilon+(1-17*epsilon)*block]
    values = a + np.log(A_samples) @ b
    if not np.allclose(values[:100], predict_recipe(A_samples[:100], meta), atol=1e-12):
        raise ValueError('Algebraic log contrast differs from frozen predictor')
    rows = []; recipes = []; diagnostics = []; slsqp_rows = []
    # Finite logit range only limits this numerical check; analytic/SLSQP
    # solutions use the exact simplex constraints, including the floor.
    bound = -np.log(np.finfo(float).eps)
    def mapping(v): return epsilon+(1-17*epsilon)*softmax(np.r_[v, 0.])
    for role, p in exact.items():
        sign = 1 if role=='minM' else -1
        optimum = float(a+np.log(p)@b)
        logging.info('P2-A %s certified M=%.12f; DE check starts', role, optimum)
        def de_objective(v): return sign*float(a+np.log(mapping(v))@b)
        def de_gradient(v):
            q=softmax(np.r_[v,0.]);x=epsilon+(1-17*epsilon)*q;g=b/x
            return (sign*(1-17*epsilon)*q*(g-q@g))[:16]
        # Broad +/-36 uniform logits mostly saturate softmax before searching.
        # A model-independent moderate initial population keeps coordinates
        # active; the full bounds still allow near-floor solutions.
        population=rng.normal(0,2,size=(cfg['simplex_DE_popsize']*16,16))
        de = differential_evolution(de_objective,
            [(-bound,bound)]*16, seed=cfg['random_state']+(role=='maxM'),
            maxiter=cfg['simplex_DE_maxiter'], popsize=cfg['simplex_DE_popsize'],
            tol=cfg['simplex_DE_tolerance'], atol=1e-12, polish=False,init=population)
        polish=minimize(de_objective,de.x,jac=de_gradient,method='L-BFGS-B',bounds=[(-bound,bound)]*16,
            options=dict(ftol=1e-15,gtol=1e-11,maxiter=2000,maxls=50))
        dp = mapping(polish.x if polish.fun<de.fun else de.x); dv = float(a+np.log(dp)@b)
        starts = [pref] + [A_samples[i] for i in rng.choice(len(A_samples), cfg['simplex_slsqp_starts']-1, replace=False)]
        best = None
        for k, start in enumerate(starts):
            r = minimize(lambda x: sign*float(a+np.log(x)@b), start,
                jac=lambda x: sign*b/x, method='SLSQP', bounds=[(epsilon,1)]*17,
                constraints=[dict(type='eq',fun=lambda x:x.sum()-1,jac=lambda x:np.ones(17))],
                options=dict(ftol=cfg['simplex_slsqp_ftol'],maxiter=cfg['simplex_slsqp_maxiter']))
            legal = abs(r.x.sum()-1)<1e-9 and r.x.min()>=epsilon-1e-12
            val = float(a+np.log(r.x)@b)
            slsqp_rows.append(dict(role=role,start=k+1,success=bool(r.success),feasible=legal,M_p=val,
                certificate_gap=sign*(val-optimum),iterations=r.nit))
            if legal and (best is None or sign*val<sign*best[0]): best=(val,r.x)
        if best is None: raise ValueError('No legal SLSQP point')
        idx = int(np.argmin(sign*values)); rv=float(values[idx])
        for method, val, count, numerical_p in [('analytic_global_certificate',optimum,1,p),
            ('differential_evolution_softmax_polished',dv,de.nfev+polish.nfev,dp),
            ('multistart_SLSQP',best[0],len(starts),best[1]),
            ('Dirichlet_random_search',rv,len(A_samples),A_samples[idx])]:
            gap = sign*(val-optimum)
            rows.append(dict(P2_mode='P2_A',role=role,method=method,M_p=val,certificate_M=optimum,
                gap_to_certificate=gap,epsilon=epsilon,evaluations_or_starts=count,
                maximum_share=float(numerical_p.max()), simplex_sum=float(numerical_p.sum()),
                minimum_share=float(numerical_p.min()),
                guarantee='analytic global optimum' if method.startswith('analytic') else
                'one-sided feasible witness, not a bound certificate' if method.startswith('Dirichlet') else 'numerical cross-check',
                status='FINITE_COMPACT_SIMPLEX'))
            tolerance=cfg['random_witness_gap_tolerance'] if method.startswith('Dirichlet') else cfg['optimizer_agreement_absolute']
            if gap < -1e-10 or gap>tolerance: raise ValueError(f'{role} {method} certificate gap {gap} exceeds {tolerance}')
        rec=dict(candidate_id='P2_A_'+role,role=role,M_p=optimum,DeltaM=optimum-frozen['reference']['M_ref'])
        rec.update(zip(names,p)); recipes.append(rec)
        diagnostics.append(dict(candidate_id=rec['candidate_id'],mode='P2_A',role=role,
            **recipe_summary(p,pref,candidates,names,cfg)))
        logging.info('%s DE/SLSQP/random certificate gaps: %.3g %.3g %.3g',role,sign*(dv-optimum),sign*(best[0]-optimum),sign*(rv-optimum))
    csv(out/'P2_global_search.csv',rows);csv(out/'P2_A_extreme_recipes.csv',recipes)
    csv(out/'simplex_multistart_checks.csv',slsqp_rows)
    # Exact zeros are replaced, but positive t is NOT floored in the frozen rule.
    witness=[]
    for role,j in [('minM',int(b.argmax())),('maxM',int(b.argmin()))]:
        for power in (3,6,12,24,48,96,192,300):
            t=10.**(-power); p=pref.copy();p[j]=0;p=p/p.sum()*(1-t);p[j]=t
            val=float(predict_recipe(p,meta)[0])
            witness.append(dict(mode='P2_B',role=role,t=t,log10_t=-power,vanishing_domain=names[j],
                contrast_coefficient=float(b[j]),M_p=val,zero_replacement_applied=False,status='POSITIVE_BOUNDARY_LIMIT',
                **dict(zip(names,p))))
        endpoint=pref.copy();endpoint[j]=0;endpoint/=endpoint.sum()
        witness.append(dict(mode='P2_B',role=role,t=0,log10_t=np.nan,vanishing_domain=names[j],
            contrast_coefficient=float(b[j]),M_p=float(predict_recipe(endpoint,meta)[0]),
            zero_replacement_applied=True,status='EXACT_ZERO_REGULARIZED_ENDPOINT',**dict(zip(names,endpoint))))
        rows.append(dict(P2_mode='P2_B',role=role,method='analytic_boundary_limit',M_p=np.nan,
            limit='-infinity' if role=='minM' else '+infinity',epsilon=0,status='UNBOUNDED_NO_ATTAINED_OPTIMUM',
            guarantee='nonzero mixed-sign log-contrast; positive entries are never floored'))
    vertices=np.eye(17); vm=predict_recipe(vertices,meta);vertex_records=[]
    for j in range(17):
        modelp=vertices[j]*(1-16*epsilon);modelp[modelp==0]=epsilon
        vertex_records.append(dict(candidate_id='P2_B_vertex_'+str(j+1),raw_domain=names[j],M_p=float(vm[j]),
            diagnostic_only=True,raw_max_share=1.,model_max_share=float(modelp.max()),**dict(zip(names,modelp))))
    csv(out/'P2_B_exact_zero_vertices.csv',vertex_records);csv(out/'P2_B_boundary_limit_witness.csv',witness)
    csv(out/'P2_global_search.csv',rows)
    for r in empirical.to_dict('records'):
        p=np.array([r[name] for name in names]);diagnostics.append(dict(candidate_id=r['candidate_id'],mode='P1',role=r['role'],
            **recipe_summary(p,pref,candidates,names,cfg)))
    csv(out/'recipe_concentration_and_support.csv',diagnostics)
    csv(out/'P1_final_recipes.csv',empirical)
    csv(out/'frozen_log_contrast.csv',[dict(domain=n,coefficient=float(v),intercept=a,zero_delta=epsilon) for n,v in zip(names,b)])
    coordinate=[]
    for r in recipes:
        for j,n in enumerate(names):
            lo,hi=float(candidates[n].min()),float(candidates[n].max());v=r[n]
            coordinate.append(dict(candidate_id=r['candidate_id'],domain=n,share=v,empirical_min=lo,empirical_max=hi,
                outside_empirical_range=bool(v<lo-1e-12 or v>hi+1e-12)))
    csv(out/'P2_empirical_coordinate_support.csv',coordinate)
    summary=dict(epsilon=epsilon,intercept=a,coefficient_sum=float(b.sum()),
        P2_A_minM=recipes[0]['M_p'],P2_A_maxM=recipes[1]['M_p'],
        random_samples_unbiased=len(raw_samples),targeted_samples=len(A_samples)-len(raw_samples),
        P2_B_status='UNBOUNDED_NO_ATTAINED_OPTIMUM',P2_B_reason='exact-zero replacement does not floor positive components',
        no_new_floor_or_refitting=True)
    (out/'simplex_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    return pd.DataFrame(recipes)


def evidence_level(r):
    """Classify transfer evidence with 1e-12 relative node-roundoff tolerance."""
    if r['mode']=='P0':return 'reference_term_zero'
    if pd.isna(r.get('N_star')):return 'no_finite_optimizer'
    n=r['N_star']
    return 'extrapolative-lambda' if n>70*(1+1e-12) else 'estimated-transfer-dependent' if n>1+1e-12 else 'real-scale-supported' if n>=.06*(1-1e-12) else 'outside_real_interpolation_interval'


def official_comparison(root, extremes):
    """Solve only P2-A; read frozen P0/P1 and mark literal P2-B unbounded."""
    root=Path(root);out=root/'outputs/optimization/phase3'
    setup,frozen,cfg,model,candidates,empirical=load(root)
    original=pd.read_csv(root/'outputs/optimization/phase2/official_budget_results.csv',float_precision='round_trip')
    results=[];cost_checks=[]
    for domain in ('main','extended'):
        for family in setup['official']['quality_costs']:
            for lctx in setup['official_contexts']:
                for budget in setup['official']['budgets_FLOPs']:
                    prob=ResourceProblem(setup,model,cfg,extremes,budget,family,lctx,domain,'P1')
                    r,_=prob.solve(independent=True,seed=cfg['random_state']+len(results))
                    r['mode']='P2_A';r['scenario_extrapolation']=True
                    r['unsupported_mixture_extrapolation']=True
                    results.append(r)
                    if r['feasible']:
                        for mode,ext in [('P0',empirical),('P1',empirical),('P2_A',extremes)]:
                            check=ResourceProblem(setup,model,cfg,ext,budget,family,lctx,domain,'P0' if mode=='P0' else 'P1')
                            q=r['Q_star'];n=r['N_star'];d=r['D_star']
                            costs=np.array([6*n*d*1e18,d*float(check.h(q))*1e18,check.eta*n*d*lctx*1e18])
                            gap=float(np.max(np.abs(costs-np.array([r['C_train'],r['C_quality'],r['C_attn']]))))
                            cost_checks.append(dict(domain=domain,cost_function=family,L_ctx=lctx,budget=budget,
                                mode=mode,N_B=n,D_B=d,Q=q,maximum_absolute_cost_gap=gap,passed=gap==0))
                    b=dict(budget=budget,cost_function=family,L_ctx=lctx,domain=domain,mode='P2_B',
                        Q0=cfg['Q0_main'],status='UNBOUNDED_NO_ATTAINED_OPTIMUM' if prob.feasible else 'INFEASIBLE',
                        feasible=prob.feasible,scenario_extrapolation=True,L_star=np.nan,
                        loss_infimum='-infinity' if prob.feasible else 'not feasible',
                        unsupported_mixture_extrapolation=True)
                    results.append(b)
                    logging.info('Official %s %s L=%s C=%.0e P2-A %s Loss=%s',domain,family,lctx,budget,r['status'],r.get('L_star'))
                    csv(out/'P2_official_budget_results.csv',results)
    combined=pd.concat([original,pd.DataFrame(results)],ignore_index=True)
    baseline=original[original['mode']=='P0'][['budget','cost_function','L_ctx','domain','L_star']].rename(columns={'L_star':'P0_Loss'})
    combined=combined.merge(baseline,on=['budget','cost_function','L_ctx','domain'],validate='many_to_one')
    combined['Loss_improvement_vs_P0']=combined.P0_Loss-combined.L_star
    combined['mixing_contribution']=combined.lambda_value*combined.DeltaM
    combined['unused_budget_fraction']=combined.unused_budget/combined.budget
    combined['mixture_evidence_level']=combined.apply(evidence_level,axis=1)
    combined['result_type']=np.where((combined.domain=='main')&combined['mode'].isin(['P0','P1']),
        'support-domain model-conditional optimization','scenario extrapolation')
    csv(out/'P0_P1_P2_official_comparison.csv',combined);csv(out/'fixed_resource_cost_invariance.csv',cost_checks)
    return combined
