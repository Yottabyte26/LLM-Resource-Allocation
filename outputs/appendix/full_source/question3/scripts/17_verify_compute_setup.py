"""Independent audit of Question 3 setup; no resource optimization or refitting."""
from __future__ import annotations
from pathlib import Path
import ast
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
import zipfile
from decimal import Decimal, localcontext
import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'outputs/optimization/phase1'


def sha(path):
    """Blockwise byte hashing without data parsing or writes."""
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def main():
    """Compare independently calculated data summaries, dimensions and derivatives."""
    checks=[]
    def check(name,passed,detail=''):
        checks.append(dict(check=name,passed=bool(passed),detail=str(detail)))
    def read(name):
        return pd.read_csv(OUT/name,float_precision='round_trip')
    setup=json.loads((OUT/'optimization_setup.json').read_text(encoding='utf-8'))
    official=setup['official'];q0=setup['baseline']['Q0']
    before=json.loads((OUT/'frozen_before_phase1.json').read_text(encoding='utf-8'))
    for rel,info in before.items():
        p=ROOT/rel
        check('frozen_bytes:'+rel,p.exists() and p.stat().st_size==info['size'] and sha(p)==info['sha256'])
    check('no_optimization_performed',setup['optimization_performed'] is False)
    check('no_shift_detection_performed',setup['shift_detection_performed'] is False)
    # Every output table must parse, be nonempty, and contain no infinities.
    for path in sorted(OUT.glob('*.csv')):
        f=pd.read_csv(path,float_precision='round_trip')
        check('csv_parse:'+path.name,len(f)>0 and len(f.columns)>0)
        check('csv_no_inf:'+path.name,not np.isinf(f.select_dtypes(include='number').to_numpy()).any())
    required=['problem3_formula_audit.md','frozen_objective_summary.csv','C7_context_summary.csv','C7_diagnosis.md',
              'context_scenarios.csv','context_critical_analysis.csv','baseline_quality_candidates.csv',
              'quality_cost_functions.md','decision_bounds.csv','mixture_feasible_sets.md',
              'compute_budget_scenarios.csv','unit_audit.md','structural_shift_definition.md','optimization_formulation.md']
    for name in required:
        check('required:'+name,(OUT/name).exists() and (OUT/name).stat().st_size>0)
    # Independent OMML check uses raw base and superscript nodes, not setup's renderer.
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    with zipfile.ZipFile(next(ROOT.glob('*.docx'))) as archive:
        xml=ET.fromstring(archive.read('word/document.xml'))
    paras=xml.findall('.//w:p',ns)
    text=lambda n:''.join(x.text or '' for x in n.iter() if x.tag.endswith('}t'))
    budget=next(p for p in paras if '为便于对照' in text(p))
    rawbudgets=[float(text(s.find('m:e',ns)))**float(text(s.find('m:sup',ns))) for s in budget.findall('.//m:sSup',ns)]
    check('budgets_from_raw_superscripts',rawbudgets==official['budgets_FLOPs'])
    attn=next(p for p in paras if text(p).startswith('长文本注意力开销：'))
    eta_math=next(m for m in attn.findall('.//m:oMath',ns) if text(m).startswith('η='))
    coeff=float(text(eta_math).split('=')[1].split('×')[0])
    sup=eta_math.find('.//m:sSup',ns)
    eta=coeff*float(text(sup.find('m:e',ns)))**float(text(sup.find('m:sup',ns)))
    check('eta_from_raw_OMML',eta==official['eta'])
    for family,phrase in [('exponential','指数型：'),('power','幂函数型：'),('logarithmic','对数渐进型：')]:
        p=next(p for p in paras if text(p).startswith(phrase))
        gamma_math=next(m for m in p.findall('.//m:oMath',ns) if text(m).startswith('γ='))
        s=gamma_math.find('.//m:sSup',ns);prefix=text(gamma_math).split('=')[1].split('×')
        gamma=(float(prefix[0]) if len(prefix)>1 else 1.)*float(text(s.find('m:e',ns)))**float(text(s.find('m:sup',ns)))
        lm=next(m for m in p.findall('.//m:oMath',ns) if text(m).startswith('λ='))
        check('official_gamma:'+family,gamma==official['quality_costs'][family]['gamma'])
        check('official_lambda:'+family,float(text(lm).split('=')[1])==official['quality_costs'][family]['lambda'])
    # Compare C7 directly, including quantiles, source frequency and absent fields.
    c7=pd.read_csv(ROOT/'real_attachments/C_efficiency_evolution/model_architecture_metadata.csv',float_precision='round_trip')
    ctx=c7.max_position_embeddings.to_numpy(float)
    check('C7_context_valid',np.isfinite(ctx).all() and (ctx>0).all() and (ctx==np.floor(ctx)).all())
    stats=read('C7_context_summary.csv').iloc[0]
    for field,quantile in [('min',0),('P01',.01),('P05',.05),('P10',.1),('P25',.25),('median',.5),
                           ('P75',.75),('P90',.9),('P95',.95),('P99',.99),('max',1)]:
        check('C7_quantile:'+field,math.isclose(stats[field],np.quantile(ctx,quantile),rel_tol=1e-14))
    scenarios=read('context_scenarios.csv')
    check('C7_scenarios_are_observed',set(scenarios.L_ctx)==set(ctx))
    for row in scenarios.itertuples():
        check('C7_frequency:'+str(row.L_ctx),row.support_count==int((ctx==row.L_ctx).sum()))
        check('C7_fraction:'+str(row.L_ctx),row.support_fraction==float((ctx==row.L_ctx).mean()))
    critical=read('context_critical_analysis.csv').iloc[0]
    check('critical_formula',critical.L_ctx_crit==6/eta)
    check('critical_exceeding_count',critical.models_above_critical==int((ctx>6/eta).sum()))
    # Freeze all objective values and compare direct source metadata, not rounded report.
    frozen=json.loads((ROOT/'outputs/scaling/phase4/generalized_scaling_final.json').read_text(encoding='utf-8'))
    objective=read('frozen_objective_summary.csv')
    for module,key in [('L0','classical'),('Delta_Q','quality')]:
        for name,value in frozen[key].items():
            matched=objective[(objective.module==module)&(objective.parameter==name)]
            check('objective_precision:'+name,len(matched)==1 and json.loads(matched.value.iloc[0])==value)
    refrow=objective[(objective.module=='M(p)')&(objective.parameter=='reference')]
    check('reference_full_precision',json.loads(refrow.value.iloc[0])==frozen['reference'])
    # Quality baseline A: independently sort and remove floor(10% n) per tail.
    qs=pd.read_csv(ROOT/'outputs/quality/A1_sample_quality.csv',usecols=['Q'],float_precision='round_trip').Q.to_numpy()
    sortedq=np.sort(qs);k=int(.1*len(qs));qa=sortedq[k:-k].mean()
    baselines=read('baseline_quality_candidates.csv').set_index('candidate')
    check('Q0_A_independent_trim',math.isclose(qa,baselines.loc['Q0_A','Q0'],abs_tol=1e-15))
    domain=pd.read_csv(ROOT/'outputs/quality/domain_quality_summary_A1.csv',float_precision='round_trip')
    check('Q0_C_median',baselines.loc['Q0_C','Q0']==np.median(domain.Q_domain))
    mapping=pd.read_csv(ROOT/'outputs/mixture/stage2/domain_quality_conflict_mapping.csv',float_precision='round_trip').set_index('mixture_domain')
    p=np.array(frozen['reference']['p_ref']);names=frozen['reference']['domain_order']
    q0_independent=sum(float(v)*float(mapping.loc[n,'Q_domain']) for n,v in zip(names,p))
    check('Q0_B_independent_sum',abs(q0_independent-q0)<2e-15)
    weights=read('baseline_quality_domain_weights.csv')
    check('seven_quality_domains_present',set(weights.quality_domain)==set(domain.domain))
    check('mapped_weights_sum_one',abs(weights.p_ref_mapped_weight.sum()-1)<2e-15)
    check('mapped_domain_quality_equals_Q0',abs(weights.quality_contribution.sum()-q0)<2e-15)
    # Independent arithmetic from saved case data; do not trust passed column alone.
    units=read('unit_numeric_checks.csv')
    check('at_least_ten_unit_cases',len(units)>=10)
    for r in units.itertuples():
        t=6*(r.N_B*1e9)*(r.D_B*1e9);a=eta*(r.N_B*1e9)*(r.D_B*1e9)*r.L_ctx
        pars=official['quality_costs'][r.family];ga,la=pars['gamma'],pars['lambda']
        with localcontext() as dc:
            dc.prec=50
            q,qb=Decimal(str(r.Q)),Decimal(str(r.Q0));g,l=Decimal(str(ga)),Decimal(str(la))
            if r.family=='exponential':h=(l*q).exp()-(l*qb).exp()
            elif r.family=='power':h=q**int(l)-qb**int(l)
            else:h=(1+l*q).ln()-(1+l*qb).ln()
            qual=float(Decimal(str(r.D_B))*Decimal('1e9')*g*max(Decimal(0),h))
        for label,x,y in [('train',t,r.program_train_FLOPs),('attention',a,r.program_attention_FLOPs),('quality',qual,r.program_quality_FLOPs)]:
            check(f'unit_case_{r.case_id}_{label}',math.isclose(x,y,rel_tol=2e-14,abs_tol=1e-5))
    # Validate analytic quality derivatives with direct central differences.
    from optimization.costs import quality_increment_per_token, marginal_quality_per_token, cost_components
    for family,pars in official['quality_costs'].items():
        gamma,lam=pars['gamma'],pars['lambda']
        qgrid=np.linspace(q0,1,101)
        costs=quality_increment_per_token(qgrid,q0,family,gamma,lam)
        check('cost_endpoint_zero:'+family,costs[0]==0)
        check('cost_monotone_finite:'+family,np.isfinite(costs).all() and (np.diff(costs)>0).all())
        check('cost_below_Q0_zero:'+family,float(quality_increment_per_token(q0*.95,q0,family,gamma,lam))==0)
        for i,q in enumerate(np.linspace(q0+.01, .99,11)):
            h=1e-6
            g=lambda x:gamma*(math.exp(lam*x) if family=='exponential' else x**lam if family=='power' else math.log(1+lam*x))
            approx=(g(q+h)-g(q-h))/(2*h)
            actual=float(marginal_quality_per_token(q,q0,family,gamma,lam))
            check(f'marginal_cost_{family}_{i}',math.isclose(approx,actual,rel_tol=2e-8))
        check('attention_train_ratio:'+family,math.isclose(cost_components(1,100,.9,q0,8192,official,family)[2]/
              cost_components(1,100,.9,q0,8192,official,family)[0],eta*8192/6,rel_tol=1e-15))
    # Main range deliberately intersects classical observed and local quality numeric support.
    diagnosis=json.loads((ROOT/'outputs/scaling/phase1/data_diagnosis.json').read_text(encoding='utf-8'))
    b=setup['main_bounds'];quality=frozen['quality_numeric_support']
    for variable,key in [('N','N_range'),('D','D_range')]:
        check('main_bound_low:'+variable,b[variable+'_min_B']==max(diagnosis[key][0],quality[variable+'_min_B']))
        check('main_bound_high:'+variable,b[variable+'_max_B']==min(diagnosis[key][1],quality[variable+'_max_B']))
    budget=read('compute_budget_scenarios.csv');grid=budget[budget.scenario_type=='research_log_grid']
    check('log_grid_size',30<=len(grid)<=50)
    check('log_grid_spacing',np.allclose(np.diff(np.log(grid.C_budget_FLOPs)),np.diff(np.log(grid.C_budget_FLOPs))[0]))
    check('log_grid_covers_sides',grid.C_budget_FLOPs.min()<min(rawbudgets) and grid.C_budget_FLOPs.max()>max(rawbudgets))
    screen=read('budget_feasibility_screen.csv')
    for r in screen.itertuples():
        lower=(6+eta*r.L_ctx)*b['N_min_B']*b['D_min_B']*1e18
        check('budget_screen:'+r.scenario_id+'_'+str(r.L_ctx),math.isclose(lower,r.minimum_total_cost_FLOPs,rel_tol=1e-15)
              and r.main_box_feasible==(r.C_budget_FLOPs>=lower))
    # The finite set derives only from already frozen empirical model inputs.
    candidates=read('empirical_recipe_candidates.csv');membership=read('empirical_recipe_membership.csv')
    check('empirical_candidate_count',len(candidates)==setup['mixture']['unique_candidates'])
    check('empirical_source_records',len(membership)==setup['mixture']['source_records'])
    check('empirical_source_types',set(membership.source_dataset)=={'train_1M','test_1M','test_60M','test_1B'})
    check('empirical_simplex',np.allclose(candidates[names].sum(axis=1),1,atol=1e-14,rtol=0) and (candidates[names]>0).all().all())
    for path in (ROOT/'src/optimization').glob('*.py'):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        banned=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=getattr(node.func,'attr',None) or getattr(node.func,'id','')
                if name in {'fit','fit_transform','minimize','least_squares','differential_evolution','linprog','milp'}:
                    banned.append(name)
        check('no_fit_or_optimizer_calls:'+path.name,not banned,banned)
    for r in read('input_manifest.csv').itertuples():
        check('input_manifest:'+r.path,sha(ROOT/r.path)==r.sha256)
    for r in read('figure_manifest.csv').itertuples():
        check('figure_exists:'+r.figure,(ROOT/r.figure).exists() and (ROOT/r.figure).stat().st_size>1000)
    results=pd.DataFrame(checks)
    results.to_csv(OUT/'independent_verification.csv',index=False,encoding='utf-8-sig')
    failures=results[~results.passed]
    summary=dict(total_checks=len(checks),passed=int(results.passed.sum()),failed=len(failures),
                 protected_files_checked=len(before),protected_files_changed=int((~results[results.check.str.startswith('frozen_bytes:')].passed).sum()),
                 optimization_performed=False,structural_shift_detection_performed=False)
    (OUT/'verification_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if len(failures):
        print(failures.to_string(index=False));raise SystemExit(1)


if __name__=='__main__':
    main()
