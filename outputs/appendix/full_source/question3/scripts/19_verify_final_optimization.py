"""Independent Phase 3 output/formula/provenance checks; never refit anything."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/'src'))
import hashlib
import json
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scaling.mixture_transfer import frozen_metadata,predict_recipe,predict_ilr


def main():
    """Recompute formulas independently and SHA256-check every frozen file."""
    root=PROJECT_ROOT;out=root/'outputs/optimization/phase3';checks=[]
    def check(name,passed,detail=''):
        checks.append(dict(check=name,passed=bool(passed),detail=str(detail)))
    def read(name):return pd.read_csv(out/name,float_precision='round_trip')
    meta=frozen_metadata(root);names=meta['domain_order']
    frozen=json.loads((root/'outputs/scaling/phase4/generalized_scaling_final.json').read_text(encoding='utf8'))
    eps=float(meta['zero_delta']);coeff=read('frozen_log_contrast.csv');b=coeff.coefficient.to_numpy()
    derived=float(meta['y_scale'])*(np.array(meta['coefficients'])/np.array(meta['X_scale']))@np.array(meta['ilr_basis'])
    check('frozen_log_contrast_algebra',np.allclose(b,derived,rtol=0,atol=1e-15) and abs(b.sum())<1e-12)
    extremes=read('P2_A_extreme_recipes.csv');diag=read('recipe_concentration_and_support.csv')
    for r in extremes.to_dict('records'):
        p=np.array([r[n] for n in names]);check(r['role']+'_simplex',abs(p.sum()-1)<1e-12 and p.min()>=eps-1e-14)
        check(r['role']+'_frozen_prediction',abs(float(predict_recipe(p,meta)[0])-r['M_p'])<1e-12)
        favor=-b if r['role']=='minM' else b
        check(r['role']+'_unfavorable_signs_at_floor',np.allclose(p[favor<=0],eps,rtol=0,atol=1e-14))
        active=(favor>0)&(p>eps+1e-12);ratios=p[active]/favor[active]
        check(r['role']+'_waterfill_KKT',np.ptp(ratios)<1e-10)
        d=diag[(diag['mode']=='P2_A')&(diag.role==r['role'])].iloc[0]
        h=-np.sum(p*np.log(p));check(r['role']+'_entropy',abs(d.entropy-h)<1e-12 and abs(d.effective_domain_count-np.exp(h))<1e-11)
        check(r['role']+'_JS_ref',abs(d.JS_distance_to_ref-jensenshannon(p,np.array(frozen['reference']['p_ref'])))<1e-12)
    searches=read('P2_global_search.csv');sa=searches[searches.P2_mode=='P2_A']
    check('three_numerical_methods_both_extrema',len(sa)==8 and sa.groupby('role').method.nunique().eq(4).all())
    check('Dirichlet_at_least_100000',sa[sa.method=='Dirichlet_random_search'].evaluations_or_starts.ge(100000).all())
    check('DE_SLSQP_agreement',sa[sa.method.str.contains('evolution|SLSQP')].gap_to_certificate.abs().max()<1e-7)
    w=read('P2_B_boundary_limit_witness.csv')
    for role,g in w.groupby('role'):
        g=g[g.t>0].sort_values('log10_t',ascending=False)
        direction=np.diff(g.M_p)<0 if role=='minM' else np.diff(g.M_p)>0
        check('boundary_'+role+'_divergence',direction.all() and np.isfinite(g.M_p).all())
        for r in g.to_dict('records'):
            check('boundary_predict_'+role+str(r['log10_t']),abs(float(predict_recipe([r[n] for n in names],meta)[0])-r['M_p'])<1e-11)
    check('P2_B_explicitly_unbounded',searches[searches.P2_mode=='P2_B'].status.eq('UNBOUNDED_NO_ATTAINED_OPTIMUM').all())
    p1=read('P1_final_recipes.csv');prior=pd.read_csv(root/'outputs/optimization/phase2/P1_extreme_candidates.csv',float_precision='round_trip')
    check('P1_exact_frozen_candidate_reuse',p1[names+['M_p','DeltaM','candidate_id']].equals(prior[names+['M_p','DeltaM','candidate_id']]))
    for r in p1.to_dict('records'):
        p=np.array([r[n] for n in names]);prediction=float(predict_ilr(np.log(p)@np.array(meta['ilr_basis']).T,meta))
        check('P1_frozen_model_'+r['candidate_id'],abs(prediction-r['M_p'])<1e-12)
    combined=read('P0_P1_P2_official_comparison.csv');keys=['budget','cost_function','L_ctx','domain','mode']
    check('official_scenarios_complete',len(combined)==360 and not combined[keys].duplicated().any())
    old=pd.read_csv(root/'outputs/optimization/phase2/official_budget_results.csv',float_precision='round_trip')
    for _,r in old.iterrows():
        row=combined
        for k in keys:row=row[row[k]==r[k]]
        check('frozen_allocation_'+str(tuple(r[k] for k in keys)),len(row)==1 and all(
            pd.isna(r[k]) and pd.isna(row[k].iloc[0]) or r[k]==row[k].iloc[0] for k in ['N_star','D_star','Q_star','L_star']))
    setup=json.loads((root/'outputs/optimization/phase1/optimization_setup.json').read_text(encoding='utf8'))
    c=frozen['classical'];q=frozen['quality'];knots=frozen['lambda_knots']
    # Lambda nodes are read without importing the resource solver.
    Ns=np.array([v['N_params_B'] for v in knots])
    lambdas=np.array([v['lambda_s'] for v in knots])
    for r in combined[combined.status=='SOLVED'].to_dict('records'):
        n,d,Q=r['N_star'],r['D_star'],r['Q_star'];family=r['cost_function'];pars=setup['official']['quality_costs'][family]
        gamma,l=pars['gamma'],pars['lambda'];q0=r['Q0']
        delta=gamma*(np.exp(l*Q)-np.exp(l*q0)) if family=='exponential' else gamma*(Q**l-q0**l) if family=='power' else gamma*np.log((1+l*Q)/(1+l*q0))
        costs=np.array([6*n*d*1e18,d*1e9*max(delta,0),setup['official']['eta']*r['L_ctx']*n*d*1e18])
        saved=np.array([r['C_train'],r['C_quality'],r['C_attn']])
        check('cost_'+str(tuple(r[k] for k in keys)),np.allclose(costs,saved,rtol=1e-12,atol=1e5) and costs.sum()<=r['budget']*(1+2e-8))
        lam=float(np.interp(np.log(n),np.log(Ns),lambdas))
        loss=c['E']+c['A']*n**(-c['alpha'])+c['B']*d**(-c['beta'])+q['C_Q']*(n/q['N0_B'])**(-q['u_Q'])*(-np.log(Q))+lam*r['DeltaM']
        check('objective_'+str(tuple(r[k] for k in keys)),abs(loss-r['L_star'])<1e-11)
        if r['mode']=='P2_A':
            chosen=extremes.iloc[0 if lam>=0 else 1]
            check('P2_sign_selection_'+str(tuple(r[k] for k in keys)),r['selected_candidate']==chosen.candidate_id and abs(r['DeltaM']-chosen.DeltaM)<1e-14)
    invariant=read('fixed_resource_cost_invariance.csv');check('fixed_resource_cost_identity',len(invariant)==252 and invariant.passed.all() and invariant.maximum_absolute_cost_gap.eq(0).all())
    hard=pd.read_csv(root/'outputs/optimization/phase2/typeI_active_set_transitions.csv',float_precision='round_trip')
    timeline=read('structural_transition_timeline.csv');check('all_saved_TypeI_events_included',timeline.evidence_type.eq('Type I active-set change').sum()==len(hard))
    for _,g in timeline.groupby(['domain','cost_function','L_ctx']):check('timeline_order',np.all(np.diff(g.budget)>=0))
    check('quality_peaks_near_saturation',read('quality_share_peak_vs_saturation.csv').query('domain == "main"').relative_gap.max()<1e-4)
    report=(root/'outputs/report/question3_final.md').read_text(encoding='utf8')
    check('nine_report_sections',all('## 3.'+str(k) in report for k in range(1,10)))
    check('exactly_five_core_tables',len(re.findall(r'^\|\s*(?:---\s*\|\s*)+$',report,re.MULTILINE))==5)
    for target in re.findall(r'\]\(([^)]+)\)',report):
        if Path(target).name not in ['independent_verification.csv','verification_summary.json']:
            check('report_link_'+target,(root/'outputs/report'/target).resolve().is_file())
    for manifest_name in ['input_manifest.csv','implementation_manifest.csv']:
        for r in read(manifest_name).to_dict('records'):
            with (root/r['path']).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            check('manifest_'+r['path'],digest==r['sha256'])
    for p in out.glob('*.csv'):
        frame=pd.read_csv(p);check('CSV_parses_'+p.name,p.stat().st_size>0 and len(frame)>0)
    baseline=json.loads((out/'frozen_before_phase3.json').read_text(encoding='utf8'));changes=[]
    for relative,expected in baseline.items():
        p=root/relative
        if not p.is_file():changes.append(relative);continue
        with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if p.stat().st_size!=expected['size'] or digest!=expected['sha256']:changes.append(relative)
    check('all_frozen_files_unchanged',not changes,changes)
    result=pd.DataFrame(checks);result.to_csv(out/'independent_verification.csv',index=False,encoding='utf-8-sig')
    summary=dict(checks=len(checks),passed=int(result.passed.sum()),failed=int((~result.passed).sum()),
        frozen_files_checked=len(baseline),frozen_files_changed=changes,question4_started=False)
    (out/'verification_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if not result.passed.all():
        print(result[~result.passed].to_string(index=False));raise ValueError('Final verification failed')


if __name__=='__main__':main()
