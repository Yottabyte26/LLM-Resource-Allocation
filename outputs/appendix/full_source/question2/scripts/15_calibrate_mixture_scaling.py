"""Question 2 Phase 3: reuse frozen mixture predictions, calibrate scale transfer.

Run: python -B scripts/15_calibrate_mixture_scaling.py
Only outputs/mixture and earlier scaling outputs are parsed. Official raw data
are hashed for preservation but never ingested for fitting or preprocessing.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import argparse
import logging
import numpy as np
import pandas as pd
import yaml
from src.scaling.io import sha256,write_csv,write_json
from src.scaling.quality_extension import read_parameters
from src.scaling.mixture_transfer import (SCALES,frozen_metadata,read_frozen_recipes,transfer,
    bootstrap_transfers,fit_curve,curve_root,empirical_lambda,predict_recipe,predict_ilr,ModularScaling)

OUT=ROOT/'outputs/scaling/phase3'
FIG=ROOT/'outputs/figures/question2_phase3'
REPORT=ROOT/'outputs/report/question2_phase3_modular_mixture_scaling.md'
NEW_FILES={'configs/scaling_mixture_phase3.yaml','src/scaling/mixture_transfer.py',
           'src/scaling/mixture_transfer_reporting.py','scripts/15_calibrate_mixture_scaling.py',
           'scripts/15_verify_mixture_scaling.py'}


def freeze_prior():
    """Snapshot every pre-existing project file; exclude only this stage's outputs/code."""
    path=OUT/'frozen_before_phase3.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    rows=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT).as_posix()
        if rel in NEW_FILES or rel.startswith('outputs/scaling/phase3/') or rel.startswith('outputs/figures/question2_phase3/') or p==REPORT:
            continue
        rows.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha256(p)))
    result=dict(files=rows,parsed_original_A_data=False,prior_files=len(rows))
    write_json(path,result)
    return result


def verify_prior(snapshot):
    """Byte-level freeze check, without parsing any original attachment."""
    bad=[]
    for r in snapshot['files']:
        p=ROOT/r['path']
        if not p.is_file() or p.stat().st_size!=r['bytes'] or sha256(p)!=r['sha256']:
            bad.append(r['path'])
    if bad:raise RuntimeError('Frozen files changed: '+str(bad))
    return dict(checked=len(snapshot['files']),changed_files=[],all_unchanged=True)


def root_summary(logroots,boundary,cfg,sign_bracket=None):
    """Finite conditional intervals; retain failures rather than discarding their rate."""
    a=np.array([np.nan if x is None else x for x in logroots],float)
    ok=np.isfinite(a)&(a>-700)&(a<700)
    valid=float(ok.mean()); boundary_rate=float(np.mean(boundary))
    result=dict(valid_fraction=valid,boundary_fraction=boundary_rate,
                sign_change_bracket_available=sign_bracket is not None)
    if ok.any():
        values=np.exp(a[ok]); lo,hi=np.quantile(values,[.025,.975])
        result.update(Nc_B_median=float(np.median(values)),Nc_B_CI_low=float(lo),Nc_B_CI_high=float(hi))
    else:
        result.update(Nc_B_median=None,Nc_B_CI_low=None,Nc_B_CI_high=None)
    inside=bool(sign_bracket is not None and ok.any() and lo>=sign_bracket[0] and hi<=sign_bracket[1])
    result['CI_inside_sign_bracket']=inside
    result['conditional_interval_stable']=bool(valid>=cfg['critical_interval_min_valid_fraction'] and
        boundary_rate<=cfg['critical_interval_max_boundary_fraction'] and inside)
    return result


def scale_curves(slopes,point,cfg):
    """P1/P2, fits A/B, leave-one-scale-out and joint recipe bootstrap."""
    sizes=np.array([s[2] for s in SCALES]); x=np.log(sizes)
    fits={}; comparison=[]; loso=[]; fullstarts=[]
    for label,n in [('A_real_only',3),('B_with_estimated',5)]:
        for model,k in [('P1',2),('P2',3)]:
            f=fit_curve(x[:n],point[:n],model,cfg)
            fits[(label,model)]=f
            if not f['success']:raise RuntimeError('Full scale fit failed')
            root=curve_root(f,model); sse=f['SSE']
            residual_df=n-k
            # AICc is undefined for n<=k+1, and exact P2 Fit A is saturated.
            aicc=float(n*np.log(max(sse/n,np.finfo(float).tiny))+2*k+2*k*(k+1)/(n-k-1)) if n>k+1 else None
            rmses=[]
            for j in range(n):
                keep=np.arange(n)!=j
                held=fit_curve(x[:n][keep],point[:n][keep],model,cfg,starts=[f['theta']])
                if not held['success']:
                    loso.append(dict(fit=label,model=model,held_scale=SCALES[j][0],
                                     success=False,reason=held['reason'],prediction=None,observed=float(point[j])))
                    continue
                p=held['parameters']
                pred=float(p['a']+p['b']*x[j]) if model=='P1' else float(p['a']*np.tanh(p['b']*(p['log_Nc_B']-x[j])))
                err=pred-point[j]; rmses.append(err)
                loso.append(dict(fit=label,model=model,held_scale=SCALES[j][0],success=True,
                                 prediction=pred,observed=float(point[j]),error=float(err),
                                 parameters=p,boundary_hit=held['boundary_hit']))
            comparison.append(dict(fit=label,model=model,n_scales=n,n_parameters=k,residual_df=residual_df,
                 parameters=f['parameters'],RMSE=float(np.sqrt(sse/n)),SSE=sse,AICc=aicc,
                 LOSO_RMSE=float(np.sqrt(np.mean(np.array(rmses)**2))) if len(rmses)==n else None,
                 LOSO_successful_folds=len(rmses),saturated=residual_df==0,
                 jacobian_rank=f['jacobian_rank'],jacobian_condition=f['jacobian_condition'],
                 boundary_hit=f['boundary_hit'],Nc_B=float(np.exp(root)) if root is not None else None,
                 critical_scale_has_observed_sign_bracket=False,
                 critical_scale_requires_estimated_evidence=label=='B_with_estimated'))
    write_csv(OUT/'lambda_scale_model_comparison.csv',comparison)
    write_csv(OUT/'lambda_leave_one_scale_out.csv',loso)
    write_json(OUT/'lambda_scale_full_fits.json',dict((label+'__'+m,f) for (label,m),f in fits.items()))
    records=[]; summaries=[]
    for label,n in [('A_real_only',3),('B_with_estimated',5)]:
        for model in ['P1','P2']:
            f=fits[(label,model)]; roots=[]; boundary=[]; failed=0
            rng=np.random.default_rng(cfg['random_state']+n)
            for b in range(len(slopes)):
                fb=fit_curve(x[:n],slopes[b,:n],model,cfg,starts=[f['theta']],
                             nstarts=cfg['bootstrap_fit_starts'],rng=rng)
                root=curve_root(fb,model); roots.append(root)
                boundary.append(fb.get('boundary_hit',False)); failed+=int(not fb['success'])
                records.append(dict(replicate=b+1,fit=label,model=model,success=fb['success'],
                     parameters=fb.get('parameters'),log_Nc_B=root,
                     Nc_B=float(np.exp(root)) if root is not None and -700<root<700 else None,
                     boundary_hit=fb.get('boundary_hit',False),SSE=fb.get('SSE'),
                     jacobian_rank=fb.get('jacobian_rank'),n_starts=fb.get('n_starts'),
                     reason=fb.get('reason')))
                if (b+1)%250==0:logging.info('Scale bootstrap %s %s %d/%d',label,model,b+1,len(slopes))
            summaries.append(dict(fit=label,model=model,failure_fraction=failed/len(slopes),
                                  **root_summary(roots,boundary,cfg,sign_bracket=(1.,10.) if n==5 else None)))
    # The empirical interpolation's crossing is conditional on the estimated 10B knot.
    logroot=np.log(sizes[2])+point[2]/(point[2]-point[3])*np.log(sizes[3]/sizes[2])
    bootroots=[]
    for row in slopes:
        bootroots.append(float(np.log(sizes[2])+row[2]/(row[2]-row[3])*np.log(sizes[3]/sizes[2])))
    summaries.append(dict(fit='B_with_estimated',model='empirical_log_linear',failure_fraction=0.,
                          point_Nc_B=float(np.exp(logroot)),**root_summary(bootroots,[False]*len(slopes),cfg,sign_bracket=(1.,10.))))
    write_csv(OUT/'lambda_scale_bootstrap_replicates.csv',records)
    write_csv(OUT/'critical_scale_diagnosis.csv',summaries)
    # Three positive observed knots provide no observed sign bracket. Fits A/B
    # are structurally different and estimated points alone supply reversal.
    # Keep the knots instead of freezing an unvalidated continuous scale law.
    decision=dict(selected_lambda_form='piecewise_linear_in_log_N',
         main_global_function_frozen=False,observed_only_support_B=[float(sizes[0]),float(sizes[2])],
         with_estimated_support_B=[float(sizes[0]),float(sizes[-1])],extrapolation_prohibited=True,
         physical_critical_scale_identified=False,estimated_conditional_crossing=True,
         reason='Only three positive observed scales; P2 Fit A saturated; large-scale sign reversal and crossing rely on estimated tables.',
         P1_preferred_over_P2_for_parsimony=True,
         generalized_model_status='modular_candidate_pending_human_confirmation',
         next_stage_requires_human_confirmation=True,
         next_stage_requires_explicit_outside_knot_extrapolation_policy=True)
    write_json(OUT/'model_selection.json',decision)
    return fits,comparison,summaries,decision


def component_evidence():
    """Reuse previous validation metrics, carrying original evidence limitations."""
    records=[]
    def append_csv(relative,module,select,evidence,note):
        t=pd.read_csv(ROOT/relative); t=t[select(t)]
        for _,r in t.iterrows():
            row=r.to_dict(); row={k:(None if pd.isna(v) else v) for k,v in row.items()}
            records.append(dict(module=module,evidence=evidence,source_file=relative,limitation=note,**row))
    append_csv('outputs/scaling/phase1/B1_group_cv.csv','L0',lambda t:t.model=='M0_N_D','Class A',
               'Frozen B1 leave-trajectory-out validation; no refitting in this stage')
    for name in ['B3_validation.csv','B4_cross_family_validation.csv','B5_literature_validation.csv']:
        append_csv('outputs/scaling/phase1/'+name,'L0',lambda t:t.stratum=='overall','Class A',
                   'B3 interpolation consistency; B4/B5 evaluation scopes not absolutely comparable')
    append_csv('outputs/scaling/phase2C/B6_group_cv.csv','Delta_Q',lambda t:t.model=='S3','Class B',
               'B6 calibrated local correction only; frozen S3 v near zero is removed without refitting')
    append_csv('outputs/scaling/phase2C/B7_nonoverlap_validation.csv','Delta_Q',lambda t:t.model=='S3','Class B',
               'B7 nonoverlap difference R2 fails; not a global quality law')
    append_csv('outputs/mixture/model_validation.csv','M(p)',
               lambda t:(t.model=='M1_ElasticNet')&(t.response=='L_mean'),'Class C',
               'Question 1 frozen absolute predictions; centered lambda here is fitted calibration, not independent validation')
    append_csv('outputs/mixture/estimated_scale_consistency.csv','M(p)',
               lambda t:t.model=='M1_ElasticNet','Class D',
               '10B/70B estimated training subsets, not independent observed experiments')
    write_csv(OUT/'component_validation.csv',records)
    return records


def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'execution.log',encoding='utf-8')])
    snapshot=freeze_prior();verify_prior(snapshot)
    cfg=yaml.safe_load((ROOT/'configs/scaling_mixture_phase3.yaml').read_text(encoding='utf-8'))
    logging.info('Phase 3 starts; %d prior files protected; no original A4-A15 parsing',len(snapshot['files']))
    classical,_=read_parameters(ROOT)
    qrows=pd.read_csv(ROOT/'outputs/scaling/phase2C/model_parameters.csv')
    qrow=qrows[qrows.model=='S3']
    if len(qrow)!=1:raise ValueError('Ambiguous frozen S3')
    qp=json.loads(qrow.iloc[0]['parameters'])
    quality=dict(C_Q=qp['C'],u_Q=qp['u'],N0_B=1.,v_removed_without_refit=True,
                 description='B6-calibrated local quality correction')
    meta=frozen_metadata(ROOT);data=read_frozen_recipes(ROOT,meta)
    # Validate against the final Question 1 analysis as well as stage-1 predictions.
    check=pd.read_csv(ROOT/'outputs/mixture/stage2/model_comparison_test_predictions.csv')
    check=check[check.model=='M1_ElasticNet']
    joined=data[data.evidence=='observed'].merge(check,on=['dataset','index'],validate='one_to_one')
    if len(joined)!=len(check) or not np.allclose(joined.M_p,joined.predicted_loss,rtol=0,atol=1e-12):
        raise ValueError('Question 1 final predictions differ from reused predictions')
    data.to_csv(OUT/'mixture_recipe_points.csv',index=False,encoding='utf-8-sig')
    rows=[]
    for s,dataset,N,e in SCALES:
        r=dict(scale=s,dataset=dataset,N_params_B=N,evidence=e,**transfer(data[data.scale==s]))
        rows.append(r);logging.info('%s lambda=%.12g Pearson=%.6g n=%d',s,r['lambda_s'],r['Pearson'],r['n'])
    slopes,summary,replicates,draws=bootstrap_transfers(data,cfg)
    for row,uncertainty in zip(rows,summary):row.update(uncertainty)
    write_csv(OUT/'mixture_transfer_by_scale.csv',rows)
    write_csv(OUT/'mixture_transfer_bootstrap.csv',summary)
    write_csv(OUT/'mixture_transfer_bootstrap_replicates.csv',replicates)
    write_csv(OUT/'mixture_transfer_bootstrap_draws.csv',draws)
    point=np.array([r['lambda_s'] for r in rows]);sizes=np.array([s[2] for s in SCALES])
    fits,comparison,critical,decision=scale_curves(slopes,point,cfg)
    ref=pd.read_csv(ROOT/'outputs/mixture/stage2/representative_baselines.csv')
    reference=ref[ref.baseline=='A4_ILR_center']
    if len(reference)!=1:raise ValueError('No unique frozen ILR center')
    p_ref=reference[meta['domain_order']].to_numpy(float)[0]
    M_ref=float(predict_recipe(p_ref,meta)[0])
    if abs(M_ref-float(reference.iloc[0].frozen_EN_prediction))>1e-12:
        raise ValueError('Reference prediction changed')
    train_ilr=pd.read_csv(ROOT/'outputs/mixture/train_1M_ILR_features.csv').iloc[:,1:].to_numpy(float)
    if not np.allclose(np.log(p_ref)@np.asarray(meta['ilr_basis']).T,train_ilr.mean(axis=0),atol=1e-12):
        raise ValueError('Reference is not frozen training ILR center')
    reference_config=dict(name='A4_ILR_center',domain_order=meta['domain_order'],p_ref=p_ref.tolist(),M_ref=M_ref)
    write_json(OUT/'reference_recipe.json',reference_config)
    knots=[dict(N_params_B=N,lambda_s=float(point[j]),evidence=e,
                CI_low=rows[j]['CI_low'],CI_high=rows[j]['CI_high']) for j,(s,d,N,e) in enumerate(SCALES)]
    write_csv(OUT/'lambda_empirical_knots.csv',knots)
    evidence=[dict(module='L0',parameters=['E','A','B','alpha','beta'],evidence='Class A',support='B1 observations; frozen B3/B4/B5 validations with original comparability limits'),
              dict(module='Delta_Q',parameters=['C_Q','u_Q'],evidence='Class B',support='B6 semisynthetic local calibration; B7 difference validation failed'),
              dict(module='M(p)',parameters=['frozen ILR first-order ElasticNet'],evidence='Class C',support='Question 1 training and true mixture checks at 1M/60M/1B'),
              dict(module='lambda_p large N',parameters=['10B knot','70B knot','segments above 1B'],evidence='Class D',support='Estimated recipe tables; no observed large-scale reversal')]
    write_csv(OUT/'module_evidence_levels.csv',evidence)
    config=dict(classical=classical,quality=quality,reference=reference_config,lambda_knots=knots,
                lambda_form='piecewise_linear_in_log_N',joint_validation_available=False,
                model_status='modular_candidate_pending_human_confirmation',evidence=evidence,
                units='N_B,D_B',outside_knot_extrapolation='raise_ValueError',
                prediction_scope='No global validation; Q is a B6 local correction; above 1B mixture segments use estimated evidence')
    write_json(OUT/'generalized_scaling_candidate.json',config)
    component=component_evidence()
    model=ModularScaling(classical,quality,meta,p_ref,sizes,point)
    # No model-generated joint test set. Only algebraic anchor diagnostics.
    anchor=[]
    for N in sizes:
        c=model.components(N,150.,1.,p_ref)
        anchor.append(dict(N_params_B=float(N),D_tokens_B=150.,Q=1.,
                           L0=float(c['L0']),Delta_Q=float(c['Delta_Q']),Delta_p=float(c['Delta_p'][0]),
                           L_gen=float(c['L_gen'][0]),difference=float(c['L_gen'][0]-c['L0']),
                           role='algebraic_anchor_check_only_no_observed_joint_Loss'))
    write_csv(OUT/'anchor_identity_checks.csv',anchor)
    # Data inventory documents parsed sources, not just the whole-project hash snapshot.
    inputs=['outputs/mixture/test_predictions.csv','outputs/mixture/estimated_scale_predictions.csv',
            'outputs/mixture/stage2/model_comparison_test_predictions.csv','outputs/mixture/final_model_metadata.json',
            'outputs/mixture/stage2/final_model_policy.json','outputs/mixture/stage2/representative_baselines.csv',
            'outputs/mixture/train_1M_closed_replaced_composition.csv','outputs/mixture/train_1M_ILR_features.csv',
            'outputs/scaling/phase1/classical_scaling_parameters.csv','outputs/scaling/phase2C/model_parameters.csv',
            'configs/scaling_mixture_phase3.yaml']
    inputs += ['outputs/mixture/'+d+suffix for s,d,N,e in SCALES if e=='observed'
               for suffix in ['_closed_replaced_composition.csv','_ILR_features.csv']]
    inputs += sorted({r['source_file'] for r in component})
    write_csv(OUT/'input_manifest.csv',[dict(path=p,bytes=(ROOT/p).stat().st_size,sha256=sha256(ROOT/p),
                                          source_role='configuration' if p.startswith('configs/') else 'frozen_output') for p in dict.fromkeys(inputs)])
    from src.scaling.mixture_transfer_reporting import build_report,plot_results
    plot_results(OUT,FIG,rows,data,fits,comparison,critical,cfg)
    build_report(REPORT,OUT,rows,comparison,critical,decision,config,component)
    integrity=verify_prior(snapshot);write_json(OUT/'frozen_integrity_check.json',integrity)
    write_json(OUT/'execution_summary.json',dict(completed=True,prior_files_unchanged=integrity['checked'],
        n_recipe_points=len(data),bootstrap_repetitions=cfg['bootstrap_repetitions'],
        fitted_Question1=False,refitted_quality=False,refitted_classical=False,
        parsed_original_A_data=False,used_B9_B10=False,joint_global_fit=False,
        selected_lambda_form=decision['selected_lambda_form'],physical_Nc_identified=False,
        stopped_before_elasticity_parameter_equivalence_and_extrapolation=True))
    write_json(OUT/'implementation_manifest.json',[dict(path=p,sha256=sha256(ROOT/p)) for p in sorted(NEW_FILES) if (ROOT/p).exists()])
    logging.info('Phase 3 completed normally; prior %d files unchanged; stopped before next stage',integrity['checked'])


def refresh_artifacts(skip_figures=False):
    """Refresh clarified diagnostics/report without rerunning fitted experiments."""
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'execution.log',encoding='utf-8')])
    snapshot=freeze_prior();verify_prior(snapshot)
    cfg=yaml.safe_load((ROOT/'configs/scaling_mixture_phase3.yaml').read_text(encoding='utf-8'))
    kwargs=dict(float_precision='round_trip')
    rows=pd.read_csv(OUT/'mixture_transfer_by_scale.csv',**kwargs).to_dict('records')
    data=pd.read_csv(OUT/'mixture_recipe_points.csv',**kwargs)
    raw=json.loads((OUT/'lambda_scale_full_fits.json').read_text(encoding='utf-8'))
    fits={tuple(k.split('__')):v for k,v in raw.items()}
    comparison=pd.read_csv(OUT/'lambda_scale_model_comparison.csv',**kwargs).to_dict('records')
    for r in comparison:
        r['parameters']=json.loads(r['parameters'])
        for k,v in r.items():
            if isinstance(v,float) and np.isnan(v):r[k]=None
    curve=pd.read_csv(OUT/'lambda_scale_bootstrap_replicates.csv',**kwargs)
    critical=[]
    for fit in ['A_real_only','B_with_estimated']:
        for m in ['P1','P2']:
            c=curve[(curve['fit']==fit)&(curve.model==m)]
            roots=[None if np.isnan(v) else float(v) for v in c.log_Nc_B]
            critical.append(dict(fit=fit,model=m,failure_fraction=float((~c.success).mean()),
                **root_summary(roots,c.boundary_hit.to_numpy(),cfg,
                               sign_bracket=(1.,10.) if fit=='B_with_estimated' else None)))
    reps=pd.read_csv(OUT/'mixture_transfer_bootstrap_replicates.csv',**kwargs)
    slopes=reps.pivot(index='replicate',columns='scale',values='lambda_s')[[s[0] for s in SCALES]].to_numpy()
    roots=slopes[:,2]/(slopes[:,2]-slopes[:,3])*np.log(10.)
    point=np.array([r['lambda_s'] for r in rows]);root=point[2]/(point[2]-point[3])*np.log(10.)
    critical.append(dict(fit='B_with_estimated',model='empirical_log_linear',failure_fraction=0.,
                         point_Nc_B=float(np.exp(root)),**root_summary(roots,[False]*len(roots),cfg,sign_bracket=(1.,10.))))
    write_csv(OUT/'critical_scale_diagnosis.csv',critical)
    decision=json.loads((OUT/'model_selection.json').read_text(encoding='utf-8'))
    config=json.loads((OUT/'generalized_scaling_candidate.json').read_text(encoding='utf-8'))
    component=pd.read_csv(OUT/'component_validation.csv',**kwargs).to_dict('records')
    for r in component:
        for k,v in r.items():
            if isinstance(v,float) and np.isnan(v):r[k]=None
    from src.scaling.mixture_transfer_reporting import plot_results,build_report
    if not skip_figures:plot_results(OUT,FIG,rows,data,fits,comparison,critical,cfg)
    build_report(REPORT,OUT,rows,comparison,critical,decision,config,component)
    anchors=pd.read_csv(OUT/'anchor_identity_checks.csv',**kwargs).to_dict('records')
    write_csv(OUT/'anchor_identity_checks.csv',anchors)
    write_json(OUT/'implementation_manifest.json',[dict(path=p,sha256=sha256(ROOT/p)) for p in sorted(NEW_FILES)])
    write_json(OUT/'frozen_integrity_check.json',verify_prior(snapshot))
    logging.info('Artifact refresh completed normally; saved fits/bootstrap retained; prior files unchanged')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-artifacts',action='store_true')
    parser.add_argument('--skip-figures',action='store_true',help='Keep unchanged figures during report-only refresh')
    args=parser.parse_args()
    try:
        if args.refresh_artifacts:refresh_artifacts(skip_figures=args.skip_figures)
        else:main()
    except Exception:
        logging.exception('Phase 3 stopped on an error; no prior results overwritten')
        raise
