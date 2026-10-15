"""Independent raw-formula verification of scaling phase 1 and frozen Q1 files."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np
import yaml
from scipy.stats import rankdata

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scaling.io import sha256,write_csv
OUT=ROOT/'outputs/scaling/phase1'
KEYS=['E','A','B','alpha','beta']


def rows(path):
    """Inspect saved outputs and raw CSVs without importing any fitting pipeline."""
    with path.open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f);data=list(r)
    if not data or any(None in a or any(v is None for v in a.values()) for a in data):raise ValueError(f'Invalid/truncated CSV {path}')
    return data


def raw_formula(p,N,D,model='M0_N_D'):
    """Independent direct powers of billion-unit N/D, never log of Loss."""
    result=np.full(len(N),p['E'],dtype=float)
    if model!='M2_D_only':result+=p['A']*N**(-p['alpha'])
    if model!='M1_N_only':result+=p['B']*D**(-p['beta'])
    return result


def metric(y,p):
    """Independent direct error/R2/Pearson and tied-rank Spearman formulas."""
    e=y-p;defined=len(y)>1 and np.ptp(y)>0 and np.ptp(p)>0
    return dict(RMSE=float(np.sqrt(np.mean(e**2))),MAE=float(np.mean(abs(e))),
        R2=float(1-np.sum(e**2)/np.sum((y-y.mean())**2)) if len(y)>1 and np.ptp(y)>0 else None,
        Pearson=float(np.corrcoef(y,p)[0,1]) if defined else None,
        Spearman=float(np.corrcoef(rankdata(y),rankdata(p))[0,1]) if defined else None,
        mean_bias=float(e.mean()))


def main():
    """Check actual fits, bootstrap unit, holdouts, predictions and input hashes."""
    checks=[]
    def check(name,condition,detail):
        if not condition:raise AssertionError(f'{name}: {detail}')
        checks.append(dict(check=name,passed=True,detail=detail))
    if not (OUT/'execution_summary.json').exists():raise RuntimeError('Complete phase1 fitting before final verification')
    config=yaml.safe_load((ROOT/'configs/scaling_phase1.yaml').read_text(encoding='utf-8'))
    base=ROOT/'real_attachments/B_scaling_laws';b1=rows(base/'pythia_training_log_existing.csv')
    N=np.array([float(r['N_params_B']) for r in b1]);D=np.array([float(r['D_tokens_B']) for r in b1]);y=np.array([float(r['val_loss']) for r in b1])
    membership=sorted(rows(OUT/'B1_trajectory_membership.csv'),key=lambda r:int(r['source_row']))
    check('membership aligned by source row',
          [int(r['source_row']) for r in membership]==list(range(1,len(b1)+1)) and
          all(float(r['N_params_B'])==N[i] and float(r['D_tokens_B'])==D[i] and r['run_id']==b1[i]['run_id'] for i,r in enumerate(membership)),
          'Diagnostic membership table is grouped; align by source_row, never assume its row order is raw CSV order')
    g=np.array([r['trajectory_id'] for r in membership]);groups=np.unique(g)
    check('B1 actual trajectory grouping',len(groups)==8 and len(g)==1176 and all(len(set(N[g==a]))==1 for a in groups),'8 exact-size trajectories, 147 checkpoints each; run_id is not model ID')
    check('B1 positive finite raw billion units',np.isfinite(np.column_stack([N,D,y])).all() and np.all(N>0) and np.all(D>0) and np.all(y>0),'No 1e9 multiplication for N/D powers')
    protected=json.loads((OUT/'frozen_project_before.json').read_text(encoding='utf-8'))
    check('all prior project files unchanged',all((ROOT/p).exists() and sha256(ROOT/p)==h for p,h in protected.items()),f'{len(protected)} original and Q1 frozen files')
    manifest=rows(OUT/'input_manifest.csv');allowed={'B1','B2','B3','B4','B5','B11','B12'}
    check('source restriction',all(r['dataset'] in allowed and sha256(ROOT/r['path'])==r['sha256'] for r in manifest),'No B6-B10 numerical ingestion; primary fit B1 only')
    main_model=json.loads((OUT/'main_model.json').read_text(encoding='utf-8'));params=main_model['parameters'];pred=raw_formula(params,N,D)
    check('five positive finite main parameters',set(params)==set(KEYS) and all(np.isfinite(v) and v>0 for v in params.values()),str(params))
    check('main predictions finite positive',np.isfinite(pred).all() and np.all(pred>0),'1176 provided checkpoints')
    count=Counter(g);w=np.array([1/count[a] for a in g]);w/=w.sum();e=y-pred
    check('balanced objective reconstructed',abs(np.dot(w,e*e)-main_model['objective'])<1e-18,'Normalized trajectory mass=1/8; raw Loss squared residuals')
    check('equal trajectory total weights',all(abs(w[g==a].sum()-1/8)<1e-14 for a in groups),'No large-checkpoint trajectory dominance')
    starts=rows(OUT/'multistart_results.csv');full=[r for r in starts if r['model']=='M0_N_D' and r['context']=='full_B1_balanced']
    check('at least 30 dispersed primary starts',len(full)>=30 and len({r['initial_parameters'] for r in full})==len(full),'32 independently generated starts, analytic-Jacobian optimization')
    legal=[r for r in full if r['legal_solution']=='True'];best=min(float(r['objective']) for r in legal)
    check('best legal start selected',abs(best-main_model['objective'])<1e-18,'Training objective alone; no validation selection')
    cvmodels=json.loads((OUT/'group_cv_models.json').read_text(encoding='utf-8'));cvpred=rows(OUT/'B1_group_cv_predictions.csv');cvstats=rows(OUT/'B1_group_cv.csv')
    seen={name:[] for name in ['M0_N_D','M1_N_only','M2_D_only']}
    for m in cvmodels:
        tr=np.array(m['training_source_rows'])-1;va=np.array(m['heldout_source_rows'])-1;label=f"{m['model']} fold{m['fold']}"
        check(label+' complete holdout',not set(tr)&set(va) and set(g[va])=={m['heldout_trajectory']} and set(g[tr])==set(m['training_trajectories']) and len(set(g[tr]))==7 and len(va)==147,'Entire heldout trajectory absent from training')
        p=raw_formula(m['fit']['parameters'],N[va],D[va],m['model']);pr=[r for r in cvpred if r['model']==m['model'] and int(r['fold'])==m['fold']]
        check(label+' predictions reconstructed',np.allclose(p,[float(r['predicted_loss']) for r in pr],rtol=0,atol=2e-12) and np.array_equal(va+1,[int(r['source_row']) for r in pr]),'Correct source row order and raw-unit powers')
        seen[m['model']].extend(va.tolist());actual=metric(y[va],p);export=next(r for r in cvstats if r['model']==m['model'] and int(r['fold'])==m['fold'])
        check(label+' metrics',all((export[k]=='' if v is None else abs(float(export[k])-v)<1e-10) for k,v in actual.items()),'Undefined constant-prediction ranks are blank, never fabricated')
        # Recreate first random initialization from the training subset only.
        rng=np.random.default_rng(config['random_state']+m['fold']);lo,hi=config['initial_exponent_range'];a,b=np.exp(rng.uniform(np.log(lo),np.log(hi),size=2))
        initial_E=float(rng.uniform(.05,.95)*y[tr].min());excess=float(np.median(y[tr]))-initial_E
        fraction=float(rng.uniform(.1,.9)) if m['model']=='M0_N_D' else (1.0 if m['model']=='M1_N_only' else 0.0)
        p0=dict(E=initial_E,A=excess*fraction*float(np.median(N[tr]))**a,B=excess*(1-fraction)*float(np.median(D[tr]))**b,alpha=a,beta=b)
        first=next(r for r in starts if r['context']=='group_cv' and r['model']==m['model'] and r['heldout_trajectory']==m['heldout_trajectory'] and r['start']=='1')
        init=json.loads(first['initial_parameters'])
        check(label+' initialization isolation',all(abs(init[k]-p0[k])<1e-12 for k in init),'No all-B1 or heldout-informed CV warm start')
    for model,v in seen.items():check(model+' OOF exact coverage',len(v)==1176 and sorted(v)==list(range(1176)),'Each checkpoint held out exactly once as a complete trajectory')
    comparison=rows(OUT/'baseline_model_comparison.csv')
    for r in comparison:
        actual=sorted([a for a in cvpred if a['model']==r['model']],key=lambda a:int(a['source_row']));p=np.array([float(a['predicted_loss']) for a in actual]);values=metric(y,p)
        check(r['model']+' pooled Group CV metrics',all(abs(float(r[k])-v)<1e-10 for k,v in values.items()),'Pooled heldout predictions, identical group folds')
    weighting=rows(OUT/'weighting_sensitivity.csv')
    pp=[raw_formula({k:float(r[k]) for k in KEYS},N,D) for r in weighting]
    check('balanced/unweighted mathematical equivalence',np.allclose(w,np.full(len(y),1/len(y)),rtol=0,atol=1e-18) and np.max(abs(pp[0]-pp[1]))<1e-8,'All eight checkpoint counts equal; numerical roundoff only')
    draws=rows(OUT/'bootstrap_cluster_draws.csv');boot=rows(OUT/'parameter_bootstrap_draws.csv');rng=np.random.default_rng(config['random_state']);n=config['bootstrap_repetitions']
    check('bootstrap repetition count',len(draws)==len(boot)==n==2000,'No ordinary IID row bootstrap')
    for i,(r,b) in enumerate(zip(draws,boot)):
        expected=rng.choice(groups,size=8,replace=True).tolist();actual=json.loads(r['drawn_trajectories']);counts=Counter(actual)
        if actual!=expected or actual!=json.loads(b['drawn_trajectories']) or dict(counts)!=json.loads(b['multiplicities']):raise AssertionError('Bootstrap cluster plan mismatch')
        if b['converged']=='True':
            if not all(np.isfinite(float(b[k])) and float(b[k])>0 for k in KEYS) or int(b['n_starts'])<30:raise AssertionError('Invalid bootstrap fit or too few starts')
    check('all bootstrap draws reproducible whole clusters',True,'Fixed seed 2026; eight sampled trajectory IDs per repetition, multiplicities retained')
    eligible=[r for r in boot if r['eligible_parameter_CI']=='True'];x=np.array([[float(r[k]) for k in KEYS] for r in eligible]);summary=rows(OUT/'parameter_bootstrap.csv')
    for j,k in enumerate(KEYS):
        values=x[:,j];r=next(a for a in summary if a['parameter']==k);calc=dict(mean=values.mean(),median=np.median(values),std=values.std(ddof=1),CI95_low=np.quantile(values,.025),CI95_high=np.quantile(values,.975))
        check(k+' cluster CI reconstruction',all(abs(float(r[key])-v)<1e-12 for key,v in calc.items()),f'{len(values)} eligible trajectory-resampled fits')
    correlation=rows(OUT/'parameter_correlation.csv');expected=np.corrcoef(x,rowvar=False)
    check('parameter bootstrap correlations',np.allclose([[float(r[k]) for k in KEYS] for r in correlation],expected,atol=1e-12,rtol=0),'Includes alpha-beta and E with every other parameter')
    for tail in rows(OUT/'parameter_tail_diagnostics.csv'):
        j=KEYS.index(tail['parameter']);r=next(a for a in summary if a['parameter']==tail['parameter'])
        ratio=(x[:,j].max()-x[:,j].min())/(np.quantile(x[:,j],.975)-np.quantile(x[:,j],.025))
        check(tail['parameter']+' tail summary',abs(ratio-float(tail['full_range_over_95CI_width']))<1e-10,'Rare supported tails retained, no parameter draw trimming')
    with (OUT/'bootstrap_predictions_checkpoint.npy').open('rb') as f:p_matrix=np.load(f)
    for i,b in enumerate(boot):
        if b['converged']!='True':continue
        p=raw_formula({k:float(b[k]) for k in KEYS},N,D)
        multiplicities=json.loads(b['multiplicities']);wb=np.array([multiplicities.get(a,0)/count[a] for a in g]);wb/=wb.sum()
        if not np.isfinite(p).all() or not np.allclose(p,p_matrix[i],atol=2e-12,rtol=0):raise AssertionError('Invalid/reconstructed bootstrap predictions')
        if abs(np.dot(wb,(y-p)**2)-float(b['objective']))>1e-17:raise AssertionError('Bootstrap objective must weight complete trajectory occurrences')
    check('all bootstrap predictions and weighted objectives',True,'Every converged draw independently reconstructed; whole-cluster multiplicity objective')
    for i in [0,n//2,n-1]:
        if boot[i]['converged']=='True':
            p=raw_formula({k:float(boot[i][k]) for k in KEYS},N,D)
            check(f'bootstrap draw {i+1} full-B1 predictions',np.allclose(p,p_matrix[i],atol=2e-12,rtol=0) and np.isfinite(p).all(),'Parameters and cached predictions aligned')
    freeze=json.loads((OUT/'freeze_before_validation.json').read_text(encoding='utf-8'))
    check('main parameters frozen before external validation',freeze['fit_datasets']==['B1'] and not freeze['validation_used_for_parameter_selection'] and all(sha256(ROOT/p)==h for p,h in freeze['files'].items()),'B2/B3/B4/B5 never calibrate E, exponents or amplitudes')
    for label in ['B2','B3','B4','B5']:
        pr=rows(OUT/f'{label}_direct_predictions.csv');nraw=np.array([float(r['N_params_B']) for r in pr]);draw=np.array([float(r['D_tokens_B']) for r in pr]);yraw=np.array([float(r['true_loss']) for r in pr]);p=raw_formula(params,nraw,draw)
        check(label+' direct finite predictions',np.isfinite(p).all() and np.allclose(p,[float(r['predicted_loss']) for r in pr],atol=2e-12,rtol=0),'Frozen B1 raw-power equation, no correction')
        name=f'{label}_validation.csv' if label in ['B2','B3'] else ('B4_cross_family_validation.csv' if label=='B4' else 'B5_literature_validation.csv');stats=rows(OUT/name)
        for r in stats:
            if r['stratum']=='overall':sel=np.ones(len(pr),dtype=bool)
            elif r['stratum']=='trajectory':sel=np.array([a['trajectory_id']==r['group_key'] for a in pr])
            elif r['stratum']=='family':sel=np.array([a['family']==r['group_key'] for a in pr])
            elif r['stratum']=='source':sel=np.array([a['source']==r['group_key'] for a in pr])
            else:sel=np.array([a['source']+' / '+a['family']==r['group_key'] for a in pr])
            actual=metric(yraw[sel],p[sel]);check(label+' '+r['stratum']+' '+r['group_key']+' metrics',all(r[k]=='' if v is None else abs(float(r[k])-v)<1e-10 for k,v in actual.items()),f'n={int(sel.sum())}; source strata preserved')
        if label=='B3':check('B3 interpolation provenance',all(r['provenance']=='interpolated_trajectory_consistency' for r in pr),'8x500 interpolation points, not independent observed experiments')
        if label=='B2':check('B2 semi-synthetic provenance',all(r['provenance']=='semi_synthetic_auxiliary' for r in pr),'No semi-synthetic data fitting')
        if label=='B5':check('B5 incomparable absolute errors not core evidence',all(r['absolute_loss_comparability']=='False' and r['absolute_error_role']=='descriptive_only' for r in stats),'Missing evaluation dataset, tokenizer and loss_type; primary source-layer ranks')
    terminal=json.loads((OUT/'terminal_only_model.json').read_text(encoding='utf-8'));terminal_rows=rows(OUT/'terminal_checkpoint_sensitivity.csv')
    check('terminal-only rank deficiency diagnosed',terminal['n_positive_weight_rows']==8 and terminal['jacobian_rank']<=3 and len(set(float(r['D_max']) for r in rows(OUT/'B1_trajectory_summary.csv')))==1,'E+B*D0^-beta unidentifiable as separate E/B/beta')
    check('terminal D direction not claimed as data evidence',all(r['D_direction_empirically_testable']=='False' for r in terminal_rows),'Positivity-imposed monotonicity is distinguished from empirical direction test')
    terminal_starts=[r for r in rows(OUT/'terminal_only_multistart.csv') if r['legal_solution']=='True']
    for r in rows(OUT/'terminal_multistart_parameter_spread.csv'):
        vals=[float(a[r['parameter']]) for a in terminal_starts]
        check(r['parameter']+' terminal multistart range',abs(min(vals)-float(r['min']))<1e-12 and abs(max(vals)-float(r['max']))<1e-12,
              'Terminal E/B/beta spread disclosed despite almost identical terminal fits')
    residual=rows(OUT/'B1_residuals.csv');check('B1 residual sign and values',np.allclose([float(r['residual']) for r in residual],e,atol=2e-12,rtol=0),'provided Loss minus prediction')
    fig=rows(OUT/'figure_manifest.csv');check('required scientific figures saved',len(fig)>=10 and all((ROOT/r['path']).is_file() and (ROOT/r['path']).stat().st_size>1000 for r in fig),'Coverage, fit, Group CV, residuals, bootstrap, B3/B4/B5 plots')
    check('final report exists',(ROOT/'outputs/report/question2_phase1_classical_scaling.md').stat().st_size>1000,'Requested method, data trust, identifiability and validation caveats')
    check('original/Q1 freeze still intact',all(sha256(ROOT/p)==h for p,h in protected.items()),f'{len(protected)} byte-identical project files')
    write_csv(OUT/'independent_verification.csv',checks)
    print(json.dumps(dict(passed_checks=len(checks),frozen_files_unchanged=len(protected),bootstrap_count=len(boot),phase='question2_phase1_only'),ensure_ascii=True))


if __name__=='__main__':main()
