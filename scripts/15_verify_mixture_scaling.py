"""Independent Phase 3 checks: frozen sources, joins, resamples and algebra.

Does not refit any model, read original A/B observations, calculate elasticities,
perform Q parameter equivalence or extrapolate B9/B10.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))  # Frozen joblib records use module name mixture.
import hashlib
import json
import numpy as np
import pandas as pd
import yaml
import joblib
from scipy.stats import linregress,pearsonr,spearmanr
from src.scaling.mixture_transfer import (SCALES,frozen_metadata,predict_ilr,predict_recipe,
                                         inverse_ilr,ModularScaling)
from src.scaling.io import write_csv,write_json

OUT=ROOT/'outputs/scaling/phase3'
CHECKS=[]


def check(name,condition,detail=''):
    """Record pass/fail explicitly and stop on failed independent checks."""
    passed=bool(condition)
    CHECKS.append(dict(check=name,passed=passed,detail=detail))
    if not passed:raise AssertionError(name+': '+str(detail))


def near(name,a,b,atol=1e-11,rtol=1e-11):
    """Compare numerical calculations with declared precision tolerances."""
    check(name,np.allclose(a,b,atol=atol,rtol=rtol),float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))


def hash_file(p):
    """Independent SHA-256 bytes, never a parser for official source data."""
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()


def main():
    cfg=yaml.safe_load((ROOT/'configs/scaling_mixture_phase3.yaml').read_text(encoding='utf-8'))
    config=json.loads((OUT/'generalized_scaling_candidate.json').read_text(encoding='utf-8'))
    data=pd.read_csv(OUT/'mixture_recipe_points.csv');rows=pd.read_csv(OUT/'mixture_transfer_by_scale.csv')
    B=cfg['bootstrap_repetitions'];check('requested_2000_bootstrap',B==2000)
    check('recipe_point_count',len(data)==256+256+64+63+63)
    meta=frozen_metadata(ROOT)
    original=pd.read_csv(ROOT/'outputs/mixture/test_predictions.csv')
    original=original[(original.model=='M1_ElasticNet')&(original.response=='L_mean')]
    estimated=pd.read_csv(ROOT/'outputs/mixture/estimated_scale_predictions.csv')
    estimated=estimated[estimated.model=='M1_ElasticNet']
    frozen_pipeline=joblib.load(ROOT/'outputs/mixture/models/L_mean_M1_ElasticNet.joblib')
    for label,dataset,N,evidence in SCALES:
        g=data[data.scale==label].sort_values('index');r=rows[rows.scale==label].iloc[0]
        check(label+'_index_unique',not g['index'].duplicated().any())
        check(label+'_evidence',set(g.evidence)=={evidence})
        check(label+'_scale_exact',np.all(g.N_params_B==N))
        source=original[original.dataset==dataset] if evidence=='observed' else estimated[estimated.dataset==dataset]
        source=source.set_index('index').loc[g['index']]
        y=source.true_loss.to_numpy() if evidence=='observed' else source.estimated_loss.to_numpy()
        m=source.predicted_loss.to_numpy() if evidence=='observed' else source.uncalibrated_prediction.to_numpy()
        near(label+'_original_M_retained',g.M_p,m)
        near(label+'_original_Loss_retained',g.Loss,y)
        near(label+'_centered_M',g.m_centered,m-m.mean())
        near(label+'_centered_Loss',g.y_centered,y-y.mean())
        p=g[[v+'_saved_replaced' for v in meta['domain_order']]].to_numpy()
        # Saved replacement compositions are strictly positive; applying the
        # frozen pipeline does no second zero activation and agrees with M(p).
        near(label+'_frozen_joblib_predict',frozen_pipeline.predict(p),m)
        reg=linregress(m,y)
        near(label+'_independent_slope',reg.slope,r.lambda_s)
        near(label+'_independent_SE',reg.stderr,r.standard_error)
        near(label+'_independent_R2',reg.rvalue**2,r.R2)
        near(label+'_independent_Pearson',pearsonr(m,y).statistic,r.Pearson)
        near(label+'_independent_Spearman',spearmanr(m,y).statistic,r.Spearman)
        check(label+'_CI_sign',r.CI_low>0 if evidence=='observed' else r.CI_high<0)
    # Replay every sampled recipe index and also compare to the seeded RNG.
    draws=pd.read_csv(OUT/'mixture_transfer_bootstrap_draws.csv')
    reps=pd.read_csv(OUT/'mixture_transfer_bootstrap_replicates.csv')
    check('paired_draw_count',len(draws)==B*3)
    check('bootstrap_replicate_count',len(reps)==B*5)
    slope_matrix=np.empty((B,5));rng=np.random.default_rng(cfg['random_state'])
    for cohort in [('1M','60M'),('1B',),('10B','70B')]:
        b=draws[draws.cohort=='+'.join(cohort)].sort_values('replicate')
        base=data[data.scale==cohort[0]].sort_values('index').set_index('index')
        expected=rng.integers(0,len(base),size=(B,len(base)))
        ids=np.array([json.loads(v) for v in b.sampled_recipe_indices])
        check('seeded_'+cohort[0],np.array_equal(ids,base.index.to_numpy()[expected]))
        for s in cohort:
            g=data[data.scale==s].sort_values('index').set_index('index')
            check('paired_signatures_'+s,np.array_equal(base.recipe_signature,g.recipe_signature))
            x=g.loc[ids.ravel(),'M_p'].to_numpy().reshape(ids.shape)
            y=g.loc[ids.ravel(),'Loss'].to_numpy().reshape(ids.shape)
            x=x-x.mean(axis=1,keepdims=True);y=y-y.mean(axis=1,keepdims=True)
            val=np.sum(x*y,axis=1)/np.sum(x*x,axis=1)
            recorded=reps[reps.scale==s].sort_values('replicate').lambda_s.to_numpy()
            near('all_bootstrap_slopes_'+s,val,recorded)
            r=rows[rows.scale==s].iloc[0]
            near('bootstrap_mean_'+s,val.mean(),r.bootstrap_mean)
            near('bootstrap_median_'+s,np.median(val),r.bootstrap_median)
            near('bootstrap_std_'+s,val.std(ddof=1),r.bootstrap_std)
            near('bootstrap_CI_'+s,np.quantile(val,[.025,.975]),[r.CI_low,r.CI_high])
            slope_matrix[:,[t[0] for t in SCALES].index(s)]=val
    # Independently evaluate every bootstrap curve objective and root.
    curve=pd.read_csv(OUT/'lambda_scale_bootstrap_replicates.csv')
    check('curve_bootstrap_record_count',len(curve)==B*4)
    check('curve_keys_unique',not curve[['replicate','fit','model']].duplicated().any())
    xall=np.log([s[2] for s in SCALES])
    for i,r in curve.iterrows():
        if not r.success:continue
        n=3 if r['fit']=='A_real_only' else 5;x=xall[:n];y=slope_matrix[int(r.replicate)-1,:n]
        p=json.loads(r.parameters)
        if r.model=='P1':
            slope=np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2)
            intercept=y.mean()-slope*x.mean()
            near('P1_boot_params_'+str(i),[p['a'],p['b']],[intercept,slope])
            yp=p['a']+p['b']*x;root=-p['a']/p['b']
        else:
            check('P2_positive_'+str(i),p['a']>0 and p['b']>0)
            yp=p['a']*np.tanh(p['b']*(p['log_Nc_B']-x));root=p['log_Nc_B']
        near('curve_boot_objective_'+str(i),np.sum((yp-y)**2),r.SSE,atol=1e-10)
        near('curve_boot_root_'+str(i),root,r.log_Nc_B)
        if np.isfinite(r.Nc_B):near('curve_boot_exp_root_'+str(i),np.exp(root),r.Nc_B,rtol=1e-10)
    diagnostics=pd.read_csv(OUT/'critical_scale_diagnosis.csv')
    for _,r in diagnostics.iterrows():
        if r.model=='empirical_log_linear':
            lr=np.log(1.)+slope_matrix[:,2]/(slope_matrix[:,2]-slope_matrix[:,3])*np.log(10.)
        else:
            lr=curve.loc[(curve['fit']==r['fit'])&(curve.model==r.model),'log_Nc_B'].dropna().to_numpy()
        roots=np.exp(lr)
        near('critical_CI_'+r['fit']+'_'+r.model,np.quantile(roots,[.025,.975]),[r.Nc_B_CI_low,r.Nc_B_CI_high],rtol=1e-10)
    p1=pd.read_csv(ROOT/'outputs/scaling/phase1/classical_scaling_parameters.csv',float_precision='round_trip')
    p1=p1[(p1.model=='M0_N_D')&(p1.weighting=='trajectory_balanced')].iloc[0]
    for k,v in config['classical'].items():check('frozen_classical_'+k,float(p1[k])==v)
    q=pd.read_csv(ROOT/'outputs/scaling/phase2C/model_parameters.csv')
    q=json.loads(q[q.model=='S3'].iloc[0].parameters)
    check('frozen_C_Q',config['quality']['C_Q']==q['C'])
    check('frozen_u_Q',config['quality']['u_Q']==q['u'])
    check('deleted_v_not_refit',config['quality']['v_removed_without_refit'] and q['v']<1e-10)
    ref=config['reference'];p_ref=np.array(ref['p_ref'])
    near('reference_closure',p_ref.sum(),1.)
    near('reference_M',predict_recipe(p_ref,meta)[0],ref['M_ref'])
    knots=config['lambda_knots'];sizes=np.array([r['N_params_B'] for r in knots]);sl=np.array([r['lambda_s'] for r in knots])
    near('empirical_knots',sl,rows.lambda_s)
    model=ModularScaling(config['classical'],config['quality'],meta,p_ref,sizes,sl)
    for N in sizes:
        c=model.components(N,150.,1.,p_ref)
        near('anchor_'+str(N),c['L_gen'],c['L0'],atol=1e-13)
    for N in [sizes[0]/2,sizes[-1]*2]:
        try:model.components(N,150.,.7,p_ref)
        except ValueError:check('reject_extrapolation_'+str(N),True)
        else:check('reject_extrapolation_'+str(N),False)
    # Formula-only numerical checks at interior points. No elasticity or joint
    # validation dataset is fabricated. Store pass/fail, not pseudo-observations.
    z=np.log(p_ref)@np.array(meta['ilr_basis']).T
    delta=z.copy();delta[0]+=.02;p=inverse_ilr(delta,meta)[0]
    for N in np.sqrt(sizes[:-1]*sizes[1:]):
        D,Q=150.,.6;partials=model.partials(N,D,Q,p)
        for dim,base in [('N',N),('D',D),('Q',Q)]:
            h=base*1e-5;plus=dict(N=N,D=D,Q=Q);minus=dict(plus)
            plus[dim]+=h;minus[dim]-=h
            num=(model.components(p=p,**plus)['L_gen'][0]-model.components(p=p,**minus)['L_gen'][0])/(2*h)
            near('partial_formula_'+dim+'_'+str(N),partials['dL_d'+dim],num,atol=1e-8,rtol=1e-6)
    ilr_gradient=float(meta['y_scale'])*np.array(meta['coefficients'])/np.array(meta['X_scale'])
    for j in range(16):
        plus=z.copy();minus=z.copy();h=1e-4;plus[j]+=h;minus[j]-=h
        pp=inverse_ilr(plus,meta)[0];pm=inverse_ilr(minus,meta)[0]
        actual=(frozen_pipeline.predict(pp[None,:])[0]-frozen_pipeline.predict(pm[None,:])[0])/(2*h)
        near('ILR_finite_difference_'+str(j),actual,ilr_gradient[j],atol=1e-10)
    decision=json.loads((OUT/'model_selection.json').read_text(encoding='utf-8'))
    check('no_global_lambda_frozen',not decision['main_global_function_frozen'])
    check('no_physical_Nc_claim',not decision['physical_critical_scale_identified'])
    check('no_joint_validation_claim',not config['joint_validation_available'])
    check('evidence_levels',set(r['evidence'] for r in config['evidence'])=={'Class A','Class B','Class C','Class D'})
    execution=json.loads((OUT/'execution_summary.json').read_text(encoding='utf-8'))
    for flag in ['fitted_Question1','refitted_quality','refitted_classical','parsed_original_A_data','used_B9_B10','joint_global_fit']:
        check('forbidden_action_'+flag,execution[flag] is False)
    manifest=pd.read_csv(OUT/'figure_manifest.csv')
    for name in manifest.filename:
        p=ROOT/'outputs/figures/question2_phase3'/name
        check('figure_nonempty_'+name,p.is_file() and p.stat().st_size>1000)
    snapshot=json.loads((OUT/'frozen_before_phase3.json').read_text(encoding='utf-8'))
    for r in snapshot['files']:
        p=ROOT/r['path']
        check('prior_file_'+r['path'],p.is_file() and p.stat().st_size==r['bytes'] and hash_file(p)==r['sha256'])
    check('report_nonempty',(ROOT/'outputs/report/question2_phase3_modular_mixture_scaling.md').stat().st_size>5000)
    write_csv(OUT/'independent_verification.csv',CHECKS)
    write_json(OUT/'independent_verification_summary.json',dict(passed=True,n_checks=len(CHECKS),
             prior_files_unchanged=len(snapshot['files']),bootstrap_slopes_replayed=B*5,
             curve_bootstrap_records_verified=len(curve),models_refitted=False))
    print('Independent checks passed:',len(CHECKS),'Frozen files unchanged:',len(snapshot['files']))


if __name__=='__main__':
    try:main()
    except Exception:
        if CHECKS:write_csv(OUT/'independent_verification.csv',CHECKS)
        raise
