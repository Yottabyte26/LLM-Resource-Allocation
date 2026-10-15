"""Independent final-phase verification; no fitting, no new observations.

Recompute formulas, compare frozen joblib predictions, and independently scan
and solve every eligible root. Original A files are only hashed for integrity.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import csv
import json
import hashlib
import cmath
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import brentq
from scipy.stats import pearsonr,spearmanr

OUT=ROOT/'outputs/scaling/phase4'
CHECKS=[]


def read(path):
    """Independent round-trip parsing of actual saved records."""
    return pd.read_csv(ROOT/path,float_precision='round_trip')


def check(name,condition,detail=''):
    """Record failure without disguising missing/invalid statistics."""
    flag=bool(condition);CHECKS.append(dict(check=name,passed=flag,detail=detail))
    if not flag:raise AssertionError(name+': '+str(detail))


def near(name,a,b,atol=1e-10,rtol=1e-10):
    """Scale-aware independent numerical comparison."""
    delta=float(np.max(abs(np.asarray(a)-np.asarray(b))))
    check(name,np.allclose(a,b,atol=atol,rtol=rtol),delta)


def sha(p):
    """Independent streamed bytes, not raw-data parsing."""
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()


def main():
    final=json.loads((OUT/'generalized_scaling_final.json').read_text(encoding='utf-8'))
    points=json.loads((OUT/'workpoint_provenance.json').read_text(encoding='utf-8'))
    c,q=final['classical'],final['quality'];Rref=final['reference']['M_ref']
    p1=read('outputs/scaling/phase1/classical_scaling_parameters.csv')
    p1=p1[(p1.model=='M0_N_D')&(p1.weighting=='trajectory_balanced')].iloc[0]
    for key,value in c.items():check('classical_full_precision_'+key,float(p1[key])==value)
    qp=read('outputs/scaling/phase2C/model_parameters.csv');qp=json.loads(qp[qp.model=='S3'].iloc[0].parameters)
    check('quality_C_full_precision',q['C_Q']==qp['C']);check('quality_u_full_precision',q['u_Q']==qp['u'])
    k=read('outputs/scaling/phase3/mixture_transfer_by_scale.csv').sort_values('N_params_B')
    Ns=k.N_params_B.to_numpy();ws=k.lambda_s.to_numpy();ss=np.diff(ws)/np.log(Ns[1:]/Ns[:-1])
    near('frozen_lambda_values',ws,[r['lambda_s'] for r in final['lambda_knots']],atol=0,rtol=0)
    def lam(N):return np.interp(np.log(N),np.log(Ns),ws)
    def slope(N,side='right'):
        if side=='right':i=np.count_nonzero(N>=Ns)-1
        else:i=np.count_nonzero(N>Ns)-1
        return float(ss[i]) if 0<=i<len(ss) else 0.
    def raw_terms(N,D,Q,dm):
        nt=c['A']*np.asarray(N)**(-c['alpha']);dt=c['B']*np.asarray(D)**(-c['beta'])
        quality=q['C_Q']*(np.asarray(N)/q['N0_B'])**(-q['u_Q'])*(-np.log(Q))
        mixture=lam(N)*dm
        return nt,dt,quality,mixture
    def gain(N,Q,dm,side='right'):
        nt,_,quality,_=raw_terms(N,150.,Q,dm)
        return c['alpha']*nt+q['u_Q']*quality-slope(N,side)*dm
    def complex_derivative(N,D,Q,dm,variable,side='right'):
        # Choose the real segment first, then analytically extend its log term.
        s=slope(N,side);base=lam(N);h=1e-20*max(1.,{'N':N,'D':D,'Q':Q}[variable])
        nv,dv,qv=complex(N),complex(D),complex(Q)
        if variable=='N':nv+=1j*h
        elif variable=='D':dv+=1j*h
        else:qv+=1j*h
        mix=(base+s*cmath.log(nv/N))*dm
        val=c['E']+c['A']*nv**(-c['alpha'])+c['B']*dv**(-c['beta'])+q['C_Q']*(nv/q['N0_B'])**(-q['u_Q'])*(-cmath.log(qv))+mix
        return val.imag/h
    pipeline=joblib.load(ROOT/'outputs/mixture/models/L_mean_M1_ElasticNet.joblib')
    bases=read('outputs/mixture/stage2/representative_baselines.csv').set_index('baseline')
    names=final['reference']['domain_order'];saved_recipes={}
    for recipe in points['recipes']:
        b=bases.loc[recipe['baseline']];p=b[names].to_numpy(float)
        actual=float(pipeline.predict(p[None,:])[0])
        near('baseline_M_'+recipe['recipe'],actual,recipe['M_p'])
        near('baseline_recipe_'+recipe['recipe'],p,recipe['p'])
        near('baseline_DeltaM_'+recipe['recipe'],actual-Rref,recipe['DeltaM'])
        saved_recipes[recipe['recipe']]=recipe
    b6=read('outputs/scaling/phase2C/B6_predictions.csv');unique=sorted(set(zip(b6.N_params_B,b6.D_tokens_B)))
    near('actual_D_quantiles',np.quantile([v[1] for v in unique],[.25,.5,.75]),list(points['D_quantiles'].values()))
    qdata=read('outputs/quality/domain_quality_summary_A1.csv').Q_domain
    near('actual_domain_Q_points',[qdata.min(),qdata.median(),qdata.max()],
         [points['domain_Q_points'][n] for n in ['domain_min','domain_median','domain_max']])
    piece=read('outputs/scaling/phase4/lambda_piecewise_definition.csv')
    near('main_segment_slopes',piece[piece.segment.str.startswith('interior')].s_i,ss)
    check('clamp_segment_slopes',np.all(piece[piece.segment.str.contains('clamp')].s_i==0))
    grid=read('outputs/scaling/phase4/elasticity_grid.csv');check('grid_count',len(grid)==216)
    check('grid_unique',not grid[['N_params_B','D_tokens_B','Q','recipe']].duplicated().any())
    for i,r in grid.iterrows():
        N,D,Q,dm=r.N_params_B,r.D_tokens_B,r.Q,r.DeltaM
        nt,dt,dq,dp=raw_terms(N,D,Q,dm);R=nt+dt+dq+dp
        near('terms_'+str(i),[nt,dt,dq,dp,R,c['E']+R],[r.N_term,r.D_term,r.Delta_Q,r.Delta_p,r.R,r.L_gen])
        GN=gain(N,Q,dm)
        near('independent_Gain_N_'+str(i),GN,r.Gain_N)
        if R>0:
            e=np.array([GN,c['beta']*dt,q['C_Q']*(N/q['N0_B'])**(-q['u_Q'])])/R
            near('elasticities_'+str(i),e,[r.E_N,r.E_D,r.E_Q])
            check('dominant_'+str(i),['N','D','Q'][int(np.argmax(e))]==r.dominant_factor)
        check('quality_extrapolation_'+str(i),r.extrapolative==(N<.07 or N>11.97))
        for v in ['N','D','Q']:
            near('complex_step_'+v+'_'+str(i),complex_derivative(N,D,Q,dm,v),r['dL_d'+v],atol=1e-10,rtol=1e-10)
        for side in ['left','right']:
            near('side_elasticity_'+side+'_'+str(i),gain(N,Q,dm,side)/R,r['E_N_'+side])
    fd=read('outputs/scaling/phase4/marginal_derivative_checks.csv')
    check('all_finite_difference_passed',fd.passed.all());check('finite_difference_count',len(fd)==720)
    check('finite_difference_actual_tolerance',np.all((fd.relative_error<=1e-5)|(fd.absolute_error<=1e-8)))
    for i,r in fd.iterrows():
        dm=saved_recipes[r.recipe]['DeltaM'];s=r.derivative_side if r.derivative_side!='central' else 'right'
        near('FD_vs_independent_complex_'+str(i),r.analytic,complex_derivative(r.N_params_B,r.D_tokens_B,r.Q,dm,r.variable,s))
    # Independent source-model evaluation for all 512 real recipes.
    train=read('outputs/mixture/train_1M_closed_replaced_composition.csv')
    predicted=pipeline.predict(train[names].to_numpy());mapping=dict(zip(train['index'],predicted))
    train_eq=read('outputs/scaling/phase4/training_recipe_equivalence_diagnosis.csv')
    check('training_recipe_workpoint_count',len(train_eq)==12288)
    check('training_recipe_indices',set(train_eq.recipe_index)==set(train['index']))
    near('training_frozen_M',train_eq.M_p,[mapping[i] for i in train_eq.recipe_index])
    eq=read('outputs/scaling/phase4/Q01_parameter_equivalence_grid.csv')
    check('equivalence_grid_count',len(eq)==216)
    check('nine_main_equivalences',len(read('outputs/scaling/phase4/Q01_parameter_equivalence.csv'))==9)
    check('eighteen_recipe_sensitivities',len(read('outputs/scaling/phase4/Q01_parameter_equivalence_recipe_sensitivity.csv'))==18)
    def F(N,Q,dm):
        return c['A']*np.asarray(N)**(-c['alpha'])+q['C_Q']*(np.asarray(N)/q['N0_B'])**(-q['u_Q'])*(-np.log(Q))+lam(N)*dm
    for label,table in [('representative',eq),('training',train_eq)]:
        for i,r in table.iterrows():
            N,Q,dm=float(r.N_params_B),float(r.Q),float(r.DeltaM);GN=float(gain(N,Q,dm))
            name=label+'_'+str(i)
            if r.status=='no_local_equivalence':
                check('no_local_'+name,GN<=0 and pd.isna(r.N_eq_B));continue
            check('positive_local_Gain_'+name,GN>0)
            maxN=max(100*N,10000.);near('max_search_'+name,maxN,r.N_max_B)
            target=float(F(N,Q+.1,dm))
            # Independent logarithmic scan (not the production stationary-point
            # segmentation), followed by first-bracket Brent solve.
            scan=np.unique(np.r_[np.geomspace(N,maxN,257),Ns[(Ns>N)&(Ns<maxN)]])
            values=F(scan,Q,dm)-target
            candidates=np.flatnonzero(values[:-1]*values[1:]<=0)
            if r.status=='no_finite_equivalence_within_range':
                check('no_finite_'+name,len(candidates)==0);continue
            check('solved_status_'+name,r.status=='solved' and r.N_eq_B>N)
            near('full_Loss_equation_'+name,F(r.N_eq_B,Q,dm),target,atol=1e-9)
            check('first_root_bracket_'+name,len(candidates)>0)
            j=candidates[0]
            independent=brentq(lambda x:float(F(x,Q,dm)-target),scan[j],scan[j+1],xtol=1e-12,rtol=1e-12)
            near('first_root_independent_'+name,independent,r.N_eq_B,atol=1e-8,rtol=1e-9)
            near('root_gain_'+name,100*(r.N_eq_B/N-1),r.equivalent_parameter_gain_pct)
            approximation=q['C_Q']*(N/q['N0_B'])**(-q['u_Q'])/(Q*GN)*.1
            near('local_approximation_'+name,approximation*100,r.local_approx_gain_pct)
    for key,sub in eq.groupby(['N_params_B','Q','recipe']):
        near('D_independent_equivalence_'+str(key),sub.N_eq_B.to_numpy(),np.full(len(sub),sub.N_eq_B.iloc[0]))
    source=read('outputs/mixture/domain_marginal_effects.csv')
    source=source[(source.model=='M1_ElasticNet')&(source.step==.01)].set_index('mixture_domain')
    effects=read('outputs/scaling/phase4/domain_effect_by_scale.csv');check('17_by_5_effects',len(effects)==85)
    for i,r in effects.iterrows():
        old=source.loc[r.domain];mult=lam(r.N_params_B)
        near('domain_scaled_'+str(i),mult*old.mean_effect,r.DeltaL_j)
        near('domain_scaled_range_'+str(i),np.sort(mult*np.array([old.min_effect,old.max_effect])),[r.min_effect,r.max_effect])
        check('estimated_effect_labels_'+str(i),r.evidence==('estimated-supported' if r.N_params_B>=10 else 'observed-supported'))
    source_joint=read('outputs/mixture/domain_joint_effects.csv')
    source_joint=source_joint[(source_joint.model=='M1_ElasticNet')&(source_joint.step==.01)].set_index(['domain_1','domain_2'])
    joints=read('outputs/scaling/phase4/joint_trend_by_scale.csv');check('136_by_5_pairs',len(joints)==680)
    for scale,sub in joints.groupby('scale'):
        matrix=read('outputs/scaling/phase4/joint_trend_matrix_'+scale+'.csv').set_index('domain')
        check('joint_matrix_shape_'+scale,matrix.shape==(17,17));check('diagonal_NA_'+scale,np.isnan(np.diag(matrix.to_numpy())).all())
        near('matrix_symmetric_'+scale,np.nan_to_num(matrix.to_numpy()),np.nan_to_num(matrix.to_numpy().T))
        for i,r in sub.iterrows():
            old=source_joint.loc[(r.domain_1,r.domain_2)]
            near('joint_scaled_'+str(i),lam(r.N_params_B)*old.mean_effect,r.scaled_I_jk)
            near('matrix_entry_'+str(i),matrix.loc[r.domain_1,r.domain_2],r.scaled_I_jk)
            check('joint_not_causal_'+str(i),r.causal_interpretation==False)
    b9=read('real_attachments/B_scaling_laws/supplementary_large_models.csv')
    b10=read('real_attachments/B_scaling_laws/supplementary_large_baseline.csv')
    output=read('outputs/scaling/phase4/B10_reference_predictions.csv')
    check('B9_no_Loss_field','val_loss' not in b9.columns);check('B10_all_rows_retained',len(output)==len(b10))
    for field in b10.columns:
        if b10[field].dtype.kind in 'fiu':near('B10_raw_'+field,b10[field],output[field],atol=0,rtol=0)
        else:check('B10_raw_'+field,np.array_equal(b10[field],output[field]))
    prediction=c['E']+c['A']*b10.N_params_B.to_numpy()**(-c['alpha'])+c['B']*b10.D_tokens_B.to_numpy()**(-c['beta'])
    near('B10_backbone_only',prediction,output.L0_prediction)
    check('B10_zero_quality_and_mixture',np.all(output.Delta_Q==0)&np.all(output.Delta_p==0))
    metrics=read('outputs/scaling/phase4/B10_extrapolation_metrics.csv')
    buckets={'all':np.ones(len(b10),bool),'100-300B':(b10.N_params_B>=100)&(b10.N_params_B<300),
             '300-1000B':(b10.N_params_B>=300)&(b10.N_params_B<1000),'1000B_and_above':b10.N_params_B>=1000}
    for _,r in metrics.iterrows():
        mask=buckets[r.bucket];y=b10.loc[mask,'val_loss'].to_numpy();pred=prediction[mask]
        check('B10_bucket_count_'+r.bucket,len(y)==r.n)
        near('B10_RMSE_'+r.bucket,np.sqrt(np.mean((pred-y)**2)),r.RMSE)
        near('B10_MAE_'+r.bucket,np.mean(abs(pred-y)),r.MAE)
        near('B10_R2_'+r.bucket,1-np.sum((pred-y)**2)/np.sum((y-y.mean())**2),r.R2)
        near('B10_Pearson_'+r.bucket,pearsonr(y,pred).statistic,r.Pearson)
        near('B10_Spearman_'+r.bucket,spearmanr(y,pred).statistic,r.Spearman)
        near('B10_bias_'+r.bucket,(pred-y).mean(),r.mean_bias)
    joins=read('outputs/scaling/phase4/B9_B10_join.csv')
    check('B9_join_all_metadata_retained',len(joins)==len(b9))
    check('B9_zero_D_not_imputed',int((b9.D_tokens_B==0).sum())==int((joins.D_tokens_B_B9==0).sum()))
    sensitivity=read('outputs/scaling/phase4/lambda_beyond70B_sensitivity.csv')
    check('three_policy_sensitivity_count',len(sensitivity)==30)
    for i,r in sensitivity.iterrows():
        N=r.N_params_B
        expected=ws[-1] if r.policy=='S-A' else ws[-1]+ss[-1]*np.log(N/70.) if r.policy=='S-B' else ws[-1]*max(0,1-np.log(N/70.)/np.log(1000./70.))
        near('lambda_sensitivity_'+str(i),expected,r.lambda_p)
        near('mixture_sensitivity_'+str(i),expected*r.DeltaM,r.Delta_p)
        check('main_policy_unchanged_'+str(i),r.role==('main' if r.policy=='S-A' else 'structural_sensitivity_only'))
    # Core frozen API must implement both clamps and the declared sensitivities.
    from src.scaling.marginal_analysis import FrozenGeneralizedScaling
    import yaml
    cfg=yaml.safe_load((ROOT/'configs/scaling_phase4.yaml').read_text(encoding='utf-8'))
    model=FrozenGeneralizedScaling(c,q,{},Rref,final['lambda_knots'],cfg)
    for N in [.0001,.001,.06,1,10,70,100,1000,10000]:
        near('API_main_lambda_'+str(N),model.lambda_value(N),lam(N))
        for side in ['left','right']:near('API_side_slope_'+str(N)+'_'+side,model.log_slope(N,side=side),slope(N,side))
    for _,r in sensitivity.iterrows():near('API_sensitivity_'+str(r.name),model.lambda_value(r.N_params_B,r.policy),r.lambda_p)
    execution=json.loads((OUT/'execution_summary.json').read_text(encoding='utf-8'))
    for flag in ['model_refit','original_A_parsed','joint_external_test_fabricated','question3_started']:
        check('forbidden_action_'+flag,execution[flag] is False)
    snapshot=json.loads((OUT/'frozen_before_phase4.json').read_text(encoding='utf-8'))
    for r in snapshot['files']:
        p=ROOT/r['path'];check('frozen_'+r['path'],p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'])
    figures=read('outputs/scaling/phase4/figure_manifest.csv')
    for name in figures.filename:check('figure_'+name,(ROOT/'outputs/figures/question2_phase4'/name).stat().st_size>1000)
    check('final_report_complete',(ROOT/'outputs/report/question2_final.md').stat().st_size>10000)
    pd.DataFrame(CHECKS).to_csv(OUT/'independent_verification.csv',index=False,encoding='utf-8-sig')
    result=dict(passed=True,n_checks=len(CHECKS),prior_files_unchanged=len(snapshot['files']),
                all_representative_equivalence_roots_checked=len(eq),training_recipe_points_checked=len(train_eq),models_refitted=False)
    (OUT/'independent_verification_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Independent checks passed:',len(CHECKS),'Prior frozen files unchanged:',len(snapshot['files']))


if __name__=='__main__':
    try:main()
    except Exception:
        if CHECKS:pd.DataFrame(CHECKS).to_csv(OUT/'independent_verification.csv',index=False,encoding='utf-8-sig')
        raise
