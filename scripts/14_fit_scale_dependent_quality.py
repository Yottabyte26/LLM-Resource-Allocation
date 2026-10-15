"""Question 2 Phase 2C, with separate compute/finalize steps and frozen provenance.

python -B scripts/14_fit_scale_dependent_quality.py --compute
python -B scripts/14_fit_scale_dependent_quality.py --finalize
Only B6 ever enters parameter fitting. B7 is an extension check and B8 solely
an incompatible semisynthetic stress test. No mixture/conflict model is built.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import json
import logging
import copy
import numpy as np
import yaml
from src.scaling.io import csv_rows,write_csv,write_json,sha256
from src.scaling.quality_extension import read_parameters,read_data,group_reference
from src.scaling.classical import predict
from src.scaling.scale_quality import centers,fit_scale,predict_scale,PARAMETERS
from src.scaling.scale_quality_analysis import (group_cv,q_level_cv,bootstrap_models,evaluate,sensitivity_surfaces)

OUT=ROOT/'outputs/scaling/phase2C'
CONFIG=ROOT/'configs/scaling_quality_phase2C.yaml'


def verify_prior():
    """Byte-for-byte preservation of every pre-stage file, including previous diagnostics."""
    snapshot=json.loads((OUT/'frozen_before_phase2C.json').read_text(encoding='utf-8'))
    bad=[]
    for r in snapshot['files']:
        p=ROOT/r['path']
        if not p.is_file() or p.stat().st_size!=r['bytes'] or sha256(p)!=r['sha256']:bad.append(r['path'])
    if bad:raise RuntimeError('Prior frozen files changed: '+repr(bad))
    return len(snapshot['files'])


def basics(cfg,b6,p,ref):
    """Fit/CV only B6; every fit evaluates 50 independent starts."""
    fitted={};starts=[]
    for m in cfg['models']:
        f,rs=fit_scale(b6,p,ref,m,cfg,cfg['random_state']);fitted[m]=f
        starts.extend(dict(stage='full_B6',model=m,**r) for r in rs)
        logging.info('B6 %s params=%s RMSE=%.9g Jacobian rank=%d boundary=%s',m,f['parameters'],np.sqrt(f['objective']),f['jacobian_rank'],f['boundary_parameters'])
    oof,cv,rs=group_cv(b6,p,ref,cfg,OUT);starts+=rs
    qcv,rs=q_level_cv(b6,p,ref,cfg,OUT);starts+=rs
    write_csv(OUT/'multistart_solutions.csv',starts)
    write_csv(OUT/'model_parameters.csv',[dict(model=m,parameters=f['parameters'],objective=f['objective'],
               fit_RMSE=float(np.sqrt(f['objective'])),jacobian_rank=f['jacobian_rank'],jacobian_condition=f['jacobian_condition'],
               boundary_hit=f['boundary_hit'],boundary_parameters=f['boundary_parameters'],
               all_start_boundary_rate=f['all_start_boundary_rate'],n_starts=f['n_starts'],n_successful_starts=f['n_successful_starts'],
               near_optimal_parameter_spread=f['near_optimal_parameter_spread']) for m,f in fitted.items()])
    data=dict(classical_parameters=p,reference_scales=ref,fitted=fitted,CV=cv,Q_level_CV=qcv,
              OOF_predictions={m:v.tolist() for m,v in oof.items()},config_sha256=sha256(CONFIG),
              B6_sha256=sha256(b6['path']),fitting_source='B6_only',B7_B8_used_for_selection=False)
    write_json(OUT/'B6_compute_basics.json',data)
    return data


def compute(cfg,reuse,resume_bootstrap=False):
    """Compute grouped uncertainties, before numerical evaluation of B7/B8."""
    nprotected=verify_prior();p,_=read_parameters(ROOT);b6=read_data(ROOT,'B6');ref=centers(b6)
    if reuse:
        data=json.loads((OUT/'B6_compute_basics.json').read_text(encoding='utf-8'))
        if data['config_sha256']!=sha256(CONFIG) or data['B6_sha256']!=sha256(b6['path']):raise ValueError('Reuse inputs changed')
        logging.info('Reusing completed B6 full fits/Group CV/Q-level fits; no refitting')
    else:data=basics(cfg,b6,p,ref)
    bp,reps,corr=bootstrap_models(b6,p,ref,cfg,OUT,resume=resume_bootstrap)
    data.update(bootstrap_parameters=bp,parameter_correlations=corr,protected_prior_files=nprotected,
                bootstrap_completed=True,n_bootstrap_records=len(reps))
    write_json(OUT/'B6_computation_complete.json',data)
    write_json(OUT/'parameters_before_B7_B8.json',dict(classical_parameters=p,reference_scales=ref,
               model_parameters={m:f['parameters'] for m,f in data['fitted'].items()},fitting_source='B6_only',
               evaluation_sources_not_used_for_fitting=['B7','B8'],config_sha256=sha256(CONFIG),
               B6_sha256=sha256(b6['path']),bootstrap_repetitions=cfg['bootstrap_repetitions'],
               final_selection_priority=cfg['selection_priority'],freeze_status='candidate_only_pending_validation_and_human_confirmation'))
    verify_prior();logging.info('B6 compute step complete: 2000 draws x 3 models x 50 starts')


def finalize(cfg):
    """Evaluate fixed candidates without tuning, then offset sensitivity and reporting."""
    nprotected=verify_prior();data=json.loads((OUT/'B6_computation_complete.json').read_text(encoding='utf-8'))
    if data['config_sha256']!=sha256(CONFIG):raise ValueError('Finalize configuration differs from computed parameters')
    p,parameter_path=read_parameters(ROOT);ref=data['reference_scales'];fits=data['fitted']
    if p!=data['classical_parameters']:raise ValueError('Frozen classical coefficients changed')
    ts={label:read_data(ROOT,label) for label in ['B6','B7','B8']};b6,b7,b8=[ts[k] for k in ['B6','B7','B8']]
    if sha256(b6['path'])!=data['B6_sha256']:raise ValueError('B6 changed after computation')
    oof={m:np.array(v) for m,v in data['OOF_predictions'].items()};predictions={}
    for label,t in ts.items():
        predictions[label]={'M0':predict(p,t['N'],t['D'])}
        for m in cfg['models']:predictions[label][m]=predict_scale(t,p,ref,m,fits[m]['parameters'])
        records=[dict(dataset=label,source_row=i+2,**r,**{m+'_prediction':float(v[i]) for m,v in predictions[label].items()}) for i,r in enumerate(t['records'])]
        write_csv(OUT/(label+'_predictions.csv'),records)
    keys6=set(b6['keys']);keys7=set(b7['keys']);over=np.array([k in keys6 for k in b7['keys']])
    masks7=dict(full=np.ones(len(over),dtype=bool),overlap=over,nonoverlap=~over)
    if int((~over).sum())!=90:raise ValueError('Unexpected B7 nonoverlap size')
    validation=[]
    for name,mask in masks7.items():
        for m,yp in predictions['B7'].items():
            validation.extend(dict(subset=name,**r) for r in evaluate(b7,yp,m,'B7_fixed_B6_parameters',mask))
    write_csv(OUT/'B7_validation_all_subsets.csv',validation)
    write_csv(OUT/'B7_nonoverlap_validation.csv',[r for r in validation if r['subset']=='nonoverlap'])
    comparison=data['CV']+[r for r in validation if r['subset']=='nonoverlap']
    write_csv(OUT/'model_comparison.csv',comparison)
    within=[r for r in comparison if r['target']=='within_ND_DeltaL']
    for m,yp in predictions['B6'].items():within.append(evaluate(b6,yp,m,'B6_in_sample')[1])
    write_csv(OUT/'within_ND_validation.csv',within)
    dp=[]
    for label,source,preds,masks in [('B6','OOF',oof,dict(full=np.ones(len(b6['L']),dtype=bool))),
                                     ('B7','fixed_B6_parameters',predictions['B7'],masks7)]:
        t=ts[label]
        for m,yp in preds.items():
            obs,pp,qref,refs=group_reference(t,yp)
            for name,mask in masks.items():
                for i in np.flatnonzero(mask&~refs):
                    dp.append(dict(dataset=label,evaluation=source,subset=name,model=m,row_index=int(i),ND_group=t['group'][i],
                                   Q_score=float(t['Q'][i]),Q_ref=float(qref[i]),Delta_L_obs=float(obs[i]),Delta_L_pred=float(pp[i])))
    write_csv(OUT/'within_ND_prediction_points.csv',dp)
    # Main candidate acceptance is determined before any offset or B8 pressure result.
    bp=data['bootstrap_parameters'];acceptance=[]
    cv0=next(r for r in data['CV'] if r['model']=='M0' and r['target']=='absolute_Loss')
    v0=next(r for r in validation if r['model']=='M0' and r['subset']=='nonoverlap' and r['target']=='absolute_Loss')
    _,domains=csv_rows(ROOT/'outputs/quality/domain_quality_summary_A1.csv')
    surfaces,mono,trends,domain=sensitivity_surfaces(b6,p,ref,fits,OUT,domains,cfg)
    for m in cfg['models']:
        acv=next(r for r in data['CV'] if r['model']==m and r['target']=='absolute_Loss')
        dcv=next(r for r in data['CV'] if r['model']==m and r['target']=='within_ND_DeltaL')
        av=next(r for r in validation if r['model']==m and r['subset']=='nonoverlap' and r['target']=='absolute_Loss')
        dv=next(r for r in validation if r['model']==m and r['subset']=='nonoverlap' and r['target']=='within_ND_DeltaL')
        br=[r for r in bp if r['model']==m];mon=next(r for r in mono if r['model']==m)
        gates=dict(B6_absolute_CV_improves_M0=acv['RMSE']<cv0['RMSE'],B6_DeltaL_CV_positive_R2=dcv['R2']>0,
                   B7_absolute_improves_M0=av['RMSE']<v0['RMSE'],B7_DeltaL_positive_R2=dv['R2']>0,
                   B7_DeltaL_positive_correlations=dv['Pearson']>0 and dv['Spearman']>0,
                   bootstrap_without_failure_or_boundary_flags=all(r['failure_rate']==0 and r['parameter_boundary_rate']==0 for r in br) and not fits[m]['boundary_hit'],
                   strictly_decreasing_Q=mon['analytic_violation_count']==0 and mon['numerical_violation_count']==0,
                   source_offset_not_used_in_main=True)
        acceptance.append(dict(model=m,accepted=all(gates.values()),**gates,
                               boundary_interpretation='No boundary flags' if not fits[m]['boundary_hit'] else 'Boundary-limited parameter inference; numerical full rank does not establish interior/positive effect identification',
                               B7_DeltaL_R2=dv['R2'],B7_DeltaL_Pearson=dv['Pearson'],B7_DeltaL_Spearman=dv['Spearman']))
    write_csv(OUT/'model_acceptance_checks.csv',acceptance)
    accepted=[m for m in cfg['selection_priority'] if next(r for r in acceptance if r['model']==m)['accepted']]
    selected=accepted[0] if accepted else None
    # These are B6-only FINAL sensitivities, never a correction tuned on B7.
    offset_fits={};offset_rows=[];offset_starts=[]
    for m in cfg['models']:
        f,rs=fit_scale(b6,p,ref,m,cfg,cfg['random_state']+3000,has_offset=True);offset_fits[m]=f
        offset_starts.extend(dict(model=m,stage='final_B6_offset_sensitivity',**r) for r in rs)
        gammachanges={k:(f['parameters'][k]-fits[m]['parameters'][k])/fits[m]['parameters'][k] if abs(fits[m]['parameters'][k])>1e-8 else None for k in PARAMETERS[m]}
        yp=predict_scale(b6,p,ref,m,f['parameters']);q1=b6['Q']==1
        offset_rows.append(dict(model=m,main_parameters=fits[m]['parameters'],offset_parameters=f['parameters'],relative_changes=gammachanges,
                                offset=f['parameters']['offset'],B6_absolute_RMSE=float(np.sqrt(f['objective'])),
                                Q1_RMSE=float(np.sqrt(np.mean((b6['L'][q1]-yp[q1])**2))),boundary_parameters=f['boundary_parameters']))
    write_csv(OUT/'source_offset_sensitivity.csv',offset_rows);write_csv(OUT/'offset_multistart_solutions.csv',offset_starts)
    # Practical C/eta non-identification is checked by changing only the B6 computational bound.
    boundrows=[]
    for upper in [10.,100.,1000.]:
        cc=copy.deepcopy(cfg);cc['parameter_bounds']['C'][1]=upper
        if upper==100.:f=fits['S2']
        else:f,_=fit_scale(b6,p,ref,'S2',cc,cfg['random_state'])
        a=f['parameters'];boundrows.append(dict(C_upper_bound=upper,**a,C_times_eta=a['C']*a['eta'],
                                               objective=f['objective'],jacobian_condition=f['jacobian_condition'],boundary_parameters=f['boundary_parameters']))
    write_csv(OUT/'S2_bound_identifiability_sensitivity.csv',boundrows)
    from src.scaling.quality_analysis import stress_test
    stress,pressure,masks8=stress_test(b8,b6,p,predictions['B8'],keys6,keys7,OUT)
    write_csv(OUT/'B8_stress_test.csv',pressure)
    summary=dict(status='COMPLETE_PENDING_HUMAN_CONFIRMATION',classical_parameters=p,reference_scales=ref,
                 fitted=fits,bootstrap_parameters=bp,parameter_correlations=data['parameter_correlations'],
                 B6_CV=data['CV'],B7_nonoverlap=[r for r in validation if r['subset']=='nonoverlap'],
                 Q_level_CV=data['Q_level_CV'],monotonicity_checks=mono,sensitivity_scale_trends=trends,
                 offset_sensitivity=offset_rows,S2_bound_sensitivity=boundrows,
                 model_acceptance=acceptance,selected_model=selected,freeze_recommended=bool(selected),human_freeze=False,
                 all_B7_DeltaL_failed=all(not r['B7_DeltaL_positive_R2'] for r in acceptance),
                 B8_incompatibility=stress,B8_pressure_performance=pressure,
                 question1_Q_domain_range=[min(float(r['Q_domain']) for r in domains),max(float(r['Q_domain']) for r in domains)],
                 protected_files=nprotected,bootstrap_repetitions=cfg['bootstrap_repetitions'],multistarts_per_fit=cfg['multistart_count'],
                 evidence='B6/B7 calibrated semisynthetic data; no new independent real training verification')
    write_json(OUT/'execution_summary.json',summary)
    inputpaths=[t['path'] for t in ts.values()]+[parameter_path,CONFIG,ROOT/'outputs/quality/domain_quality_summary_A1.csv']
    write_json(OUT/'input_provenance.json',dict(inputs=[dict(path=x.relative_to(ROOT).as_posix(),sha256=sha256(x)) for x in inputpaths],
                                            fitting_sources=['B6'],acceptance_sources=['B6_GroupCV','B7_nonoverlap'],pressure_sources=['B8']))
    from src.scaling.scale_quality_reporting import figures,report
    figures(ROOT,summary,ts,p,ref,oof,predictions,dp,surfaces,domain,cfg)
    report(ROOT,summary,domain)
    verified=verify_prior();write_json(OUT/'frozen_integrity_check.json',dict(passed=True,checked_files=verified,modified_prior_files=[]))
    logging.info('Phase 2C complete. Selected=%s; all B7 Delta-L failed=%s. %d frozen files verified. Stop; no mixture stage.',selected,summary['all_B7_DeltaL_failed'],verified)
    print(json.dumps(dict(selected_model=selected,freeze_recommended=bool(selected),reference_scales=ref,
                          B7_DeltaL_R2={r['model']:r['B7_DeltaL_R2'] for r in acceptance},protected_files=verified),indent=2))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(OUT/'execution.log',encoding='utf-8'),logging.StreamHandler()])
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--compute',action='store_true');mode.add_argument('--finalize',action='store_true')
    parser.add_argument('--reuse-basics',action='store_true',help='Reuse hash-verified B6 full fits and CV after an infrastructure failure')
    parser.add_argument('--resume-bootstrap',action='store_true',help='Preserve complete bootstrap checkpoints and continue remaining draws')
    args=parser.parse_args();cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    if args.compute:compute(cfg,args.reuse_basics,args.resume_bootstrap)
    else:finalize(cfg)


if __name__=='__main__':
    try:main()
    except Exception:
        logging.exception('Phase 2C execution failed; existing data protected; no later stage started')
        raise
