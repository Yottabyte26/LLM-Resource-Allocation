"""Independent conservation, objective, optimality and frozen-file checks."""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'outputs/optimization/phase2'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    """Independently recompute allocations and conservation from frozen source numbers."""
    checks=[]
    def check(name,passed,detail=''):checks.append(dict(check=name,passed=bool(passed),detail=str(detail)))
    def read(name):return pd.read_csv(OUT/name,float_precision='round_trip')
    before=json.loads((OUT/'frozen_before_phase2.json').read_text(encoding='utf-8'))
    for rel,v in before.items():
        p=ROOT/rel;check('frozen:'+rel,p.exists() and p.stat().st_size==v['size'] and sha(p)==v['sha256'])
    setup=json.loads((ROOT/'outputs/optimization/phase1/optimization_setup.json').read_text(encoding='utf-8'))
    model=json.loads((ROOT/setup['frozen_objective_source']).read_text(encoding='utf-8'))
    c=model['classical'];ql=model['quality'];eta=setup['official']['eta'];knots=model['lambda_knots']
    Ns=np.array([k['N_params_B'] for k in knots]);lambdas=np.array([k['lambda_s'] for k in knots])
    candidates=pd.read_csv(ROOT/'outputs/optimization/phase1/empirical_recipe_candidates.csv',float_precision='round_trip').set_index('candidate_id')
    for p in OUT.glob('*.csv'):
        f=pd.read_csv(p,float_precision='round_trip');check('parse:'+p.name,len(f)>0 and len(f.columns)>0)
        check('no_inf:'+p.name,not np.isinf(f.select_dtypes(include='number').to_numpy()).any())
    path=pd.concat([read(f'P0_{domain}_budget_path.csv') for domain in ('main','extended')],ignore_index=True)
    allresults=pd.concat([path,read('P1_official_budget_results.csv'),read('Q0_sensitivity_results.csv'),read('refined_budget_solutions.csv')],ignore_index=True)
    grid=read('budget_grid.csv');check('44_distinct_sorted_budgets',len(grid)==44 and len(grid.budget.unique())==44 and (np.diff(grid.budget)>0).all())
    check('official_budgets_present',set([1e19,1e22,1e24]).issubset(set(grid.budget)))
    for domain in ('main','extended'):check('P0_path_count:'+domain,len(path[path.domain==domain])==3*5*len(grid))
    bounds=setup['main_bounds'];violations=0;maxobjectivegap=0.;costgap=0
    for i,r in enumerate(allresults.itertuples()):
        maximumN=bounds['N_max_B'] if r.domain=='main' else 10000.
        maximumD=bounds['D_max_B'] if r.domain=='main' else 36000.
        Cmin=(6+eta*r.L_ctx)*bounds['N_min_B']*bounds['D_min_B']*1e18
        if not r.feasible:
            check(f'infeasible_minimum:{i}',r.budget<Cmin*(1+1e-13))
            check(f'no_fabricated_infeasible_decision:{i}',pd.isna(getattr(r,'N_star',np.nan)) and pd.isna(getattr(r,'L_star',np.nan)))
            continue
        n,d,q=r.N_star,r.D_star,r.Q_star;pars=setup['official']['quality_costs'][r.cost_function]
        gamma,lam=pars['gamma'],pars['lambda']
        g=lambda x:gamma*(math.exp(lam*x) if r.cost_function=='exponential' else x**lam if r.cost_function=='power' else math.log1p(lam*x))
        increment=max(0,g(q)-g(r.Q0));train=6*(n*1e9)*(d*1e9);att=eta*(n*1e9)*(d*1e9)*r.L_ctx;quality=d*1e9*increment
        total=train+att+quality
        check(f'bounds:{i}',bounds['N_min_B']*(1-1e-9)<=n<=maximumN*(1+1e-9) and bounds['D_min_B']*(1-1e-9)<=d<=maximumD*(1+1e-9) and r.Q0-1e-12<=q<=1+1e-12)
        check(f'budget:{i}',total<=r.budget*(1+2e-8))
        check(f'D_analytic:{i}',math.isclose(d,min(maximumD,r.budget/((6+eta*r.L_ctx)*n*1e18+increment*1e9)),rel_tol=2e-13,abs_tol=1e-10))
        for name,x in [('train',train),('attn',att),('quality',quality),('used',total)]:
            y=getattr(r,'C_'+name);check(f'cost_{name}:{i}',math.isclose(x,y,rel_tol=2e-12,abs_tol=1e7))
        lamN=float(np.interp(np.log(n),np.log(Ns),lambdas))
        dm=0. if r.mode=='P0' else float(candidates.loc[r.selected_candidate,'M_p'])-model['reference']['M_ref']
        L=c['E']+c['A']*n**(-c['alpha'])+c['B']*d**(-c['beta'])+ql['C_Q']*(n/ql['N0_B'])**(-ql['u_Q'])*(-math.log(q))+lamN*dm
        gap=abs(L-r.L_star);maxobjectivegap=max(maxobjectivegap,gap);check(f'frozen_objective:{i}',gap<1e-12)
        check(f'cost_shares:{i}',math.isclose(r.s_train+r.s_quality+r.s_attn,total/r.budget,rel_tol=2e-12,abs_tol=1e-12))
        check(f'unused_budget:{i}',math.isclose(r.unused_budget,max(0,r.budget-total),rel_tol=2e-10,abs_tol=r.budget*2e-12))
        check(f'no_N_extrapolation_hidden:{i}',bool(r.extrapolative_N)==(n>bounds['N_max_B']*(1+1e-6)))
        check(f'no_D_extrapolation_hidden:{i}',bool(r.extrapolative_D)==(d>bounds['D_max_B']*(1+1e-6)))
        check(f'no_Q_extrapolation_hidden:{i}',bool(r.extrapolative_Q)==(q>setup['baseline']['A1_sample_Q_range'][1]+1e-6))
        if r.mode=='P1':
            effects=lamN*(candidates.M_p.to_numpy()-model['reference']['M_ref'])
            check(f'P1_full_enumeration_at_optimum:{i}',abs(lamN*dm-effects.min())<1e-12)
            check(f'P1_estimated_support:{i}',bool(r.mixture_transfer_uses_estimated_support)==(n>1+1e-12))
    cross=read('optimizer_cross_checks.csv');legal=cross[cross.status=='SOLVED']
    check('every_official_and_sensitivity_two_optimizers',legal.DE_refined_loss.notna().all())
    check('optimizer_loss_agreement',(legal.DE_minus_final_loss.abs()<2e-7).all())
    check('at_least_ten_local_starts',(legal.local_start_count>=10).all())
    full=read('D_elimination_3D_check.csv');enum=read('P1_candidate_reduction_check.csv')
    check('30_independent_3D_checks',len(full)>=30 and full.passed.all())
    check('20_full_832_enumeration_checks',len(enum)>=20 and enum.passed.all() and (enum.enumerated_candidates==832).all())
    check('15_new_unit_checks',len(read('unit_tests.csv'))>=15 and read('unit_tests.csv').passed.all())
    ratio=read('context_cost_ratio.csv');check('attention_ratio_exact',np.allclose(ratio.ratio_attn_train,eta*ratio.L_ctx/6,rtol=1e-15))
    check('critical_context_30000',(ratio.L_ctx_crit==30000).all())
    official=read('official_budget_results.csv')
    bad=official[(official.budget==1e19)&(official.L_ctx==131072)]
    check('known_low_long_infeasibility',len(bad)==12 and (bad.status=='INFEASIBLE').all())
    for keys,f in path[path.feasible==True].groupby(['domain','cost_function','L_ctx']):
        f=f.sort_values('budget');check('Loss_monotone:'+str(keys),(np.diff(f.L_star)<=2e-7).all())
    check('main_saturation_at_high_budget',official[(official.domain=='main')&(official['mode']=='P0')&(official.budget==1e24)].evidence_domain_saturated.all())
    # Verify each refined active-set bracket from recorded independent resource solutions.
    hard=read('typeI_active_set_transitions.csv');p0=pd.concat([path,read('refined_budget_solutions.csv')],ignore_index=True)
    for i,r in enumerate(hard.itertuples()):
        check(f'typeI_bracket:{i}',r.bracket_low<=r.budget_critical<=r.bracket_high)
        if r.method=='analytic_minimum_cost':continue
        check(f'typeI_refinement_width:{i}',np.log10(r.bracket_high/r.bracket_low)<=1.01e-5)
        states=[]
        for C in [r.bracket_low,r.bracket_high]:
            g=p0[(p0.domain==r.domain)&(p0.cost_function==r.cost_function)&(p0.L_ctx==r.L_ctx)&np.isclose(p0.budget,C,rtol=2e-13,atol=0)]
            states.append(bool(g[r.constraint].iloc[0]) if len(g) else None)
        check(f'typeI_changed_state:{i}',states[0] is not None and states[1] is not None and states[0]!=states[1],states)
    # Independently calculate reported hinge BIC and ensure the acceptance threshold.
    soft=read('typeII_soft_transitions.csv')
    for i,r in enumerate(soft.itertuples()):
        if r.status!='FITTED':continue
        check(f'typeII_minimum_sides:{i}',r.points_left>=5 and r.points_right>=5)
        check(f'typeII_decision:{i}',bool(r.significant)==(r.delta_BIC>=10))
        f=path[(path.domain==r.domain)&(path.cost_function==r.cost_function)&(path.L_ctx==r.L_ctx)&(path.feasible==True)].sort_values('budget')
        x=np.log10(f.budget.to_numpy());y=np.log(f.N_star.to_numpy()) if r.metric=='ln_N' else np.log(f.D_star.to_numpy()) if r.metric=='ln_D' else f.Q_star.to_numpy() if r.metric=='Q' else f[r.metric].to_numpy()
        A=np.c_[np.ones(len(x)),x-x.mean()];B=np.c_[A,np.maximum(0,x-r.break_log10C)]
        e0=y-A@np.linalg.lstsq(A,y,rcond=None)[0];e1=y-B@np.linalg.lstsq(B,y,rcond=None)[0]
        floor=max(1e-24,np.var(y)*len(x)*1e-14);bic=len(x)*np.log(max(e0@e0,floor)/len(x))+2*np.log(len(x))-len(x)*np.log(max(e1@e1,floor)/len(x))-4*np.log(len(x))
        check(f'typeII_BIC_arithmetic:{i}',abs(bic-r.delta_BIC)<1e-7)
        if r.significant:check(f'typeII_interval_finite:{i}',np.isfinite(r.break_interval_low+r.break_interval_high) and r.break_interval_low<=r.break_interval_high)
    figs=read('figure_manifest.csv');check('at_least_12_figures',len(figs)>=12)
    for r in figs.itertuples():check('figure:'+r.figure,(ROOT/r.figure).exists() and (ROOT/r.figure).stat().st_size>1000)
    for r in read('implementation_manifest.csv').itertuples():check('implementation_hash:'+r.path,sha(ROOT/r.path)==r.sha256)
    for r in read('input_manifest.csv').itertuples():check('input_hash:'+r.path,sha(ROOT/r.path)==r.sha256)
    from optimization.resource_solver import load,ResourceProblem
    s,f,config,m,p,ext=load(ROOT)
    for family in s['official']['quality_costs']:
        problem=ResourceProblem(s,m,config,ext,1e25,family,2048)
        with np.errstate(over='raise',invalid='raise'):
            check('bounded_inverse_no_overflow:'+family,np.isfinite(problem.inverse_h([0,1,1e100])).all()
                  and problem.inverse_h(1e100)==1)
    summaryfile=json.loads((OUT/'execution_summary.json').read_text(encoding='utf-8'))
    check('P2_not_performed',summaryfile['P2_performed'] is False);check('Question4_not_started',summaryfile['Question4_started'] is False)
    frame=pd.DataFrame(checks);frame.to_csv(OUT/'independent_verification.csv',index=False,encoding='utf-8-sig')
    summary=dict(total_checks=len(frame),passed=int(frame.passed.sum()),failed=int((~frame.passed).sum()),
                 protected_files_checked=len(before),protected_files_changed=int((~frame[frame.check.str.startswith('frozen:')].passed).sum()),
                 maximum_frozen_objective_recomputation_gap=maxobjectivegap,P2_performed=False,Question4_started=False)
    (OUT/'verification_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    if not frame.passed.all():
        print(frame[~frame.passed].to_string(index=False));raise SystemExit(1)


if __name__=='__main__':main()
