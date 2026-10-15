"""Question 2 Phase 2B: B6-only quality fits, B7 validation, B8 pressure test.

Run with python -B. Existing Question 1, classical Phase 1 and confirmed
Phase 2 diagnoses are protected byte-for-byte by a pre-stage manifest.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import logging
import argparse
import numpy as np
import yaml
from src.scaling.io import csv_rows, sha256, write_csv, write_json
from src.scaling.quality_extension import read_parameters,read_data,predict_quality,fit_quality,MODELS
from src.scaling.quality_analysis import (cross_validate,q_level_cv,layer_fit,bootstrap,
                                         difference_table,monotonicity_details,stress_test,evaluate)

OUT=ROOT/'outputs/scaling/phase2'
CONFIG=ROOT/'configs/scaling_quality_phase2B.yaml'


def verify_existing():
    """Protect all pre-existing files, including the previous diagnosis report."""
    manifest=json.loads((OUT/'frozen_before_phase2B.json').read_text(encoding='utf-8'))
    bad=[]
    for r in manifest['files']:
        p=ROOT/r['path']
        if not p.is_file() or p.stat().st_size!=r['bytes'] or sha256(p)!=r['sha256']:bad.append(r['path'])
    if bad:raise RuntimeError('Protected files changed: '+repr(bad))
    return len(manifest['files'])


def completed_bootstrap(cfg,fitted):
    """Resume validated completed resamples without refitting or changing them."""
    prior=json.loads((OUT/'parameters_before_B7_B8.json').read_text(encoding='utf-8'))
    if prior['config_sha256']!=sha256(CONFIG):raise ValueError('Resume configuration changed')
    for model,fit in fitted.items():
        for key in ['gamma','offset','objective']:
            if fit[key]!=prior['fitted_quality_models'][model][key]:raise ValueError('Resume fit changed')
    def read_result(name):
        _,rows=csv_rows(OUT/name)
        parsed=[]
        for r in rows:
            a={}
            for k,v in r.items():
                if v=='':a[k]=None
                elif v in ['True','False']:a[k]=v=='True'
                else:
                    try:a[k]=float(v)
                    except ValueError:a[k]=v
            parsed.append(a)
        return parsed
    bs=read_result('quality_parameter_bootstrap.csv')
    rows=read_result('quality_bootstrap_replicates.csv')
    gains=read_result('group_cv_improvement_uncertainty.csv')
    if len(rows)!=cfg['bootstrap_repetitions']*len(cfg['bootstrap_models']):raise ValueError('Incomplete bootstrap resume')
    for m in cfg['bootstrap_models']:
        r=[x for x in rows if x['model']==m]
        if len(r)!=cfg['bootstrap_repetitions']:raise ValueError('Incomplete model resamples')
    logging.info('Reused completed 2000 paired ND-block bootstraps; unchanged fitted parameters and configuration')
    return bs,rows,gains


def main():
    """Fit and freeze numerical candidate before ever evaluating B7/B8."""
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(OUT/'quality_fit.log',encoding='utf-8'),logging.StreamHandler()])
    nprotected=verify_existing();cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    p,parameter_path=read_parameters(ROOT)
    b6=read_data(ROOT,'B6')
    mono=monotonicity_details(b6,OUT)
    if mono['negative_correlation_group_rate']==0:raise RuntimeError('B6 does not support the prescribed Q direction')
    fitted={};multistart=[]
    for m in MODELS[1:]:
        fit,runs=fit_quality(b6,p,m,cfg);fitted[m]=fit
        multistart.extend(dict(stage='full_B6',model=m,**r) for r in runs)
        logging.info('B6 %s gamma=%.12g offset=%.12g RMSE=%.9g',m,fit['gamma'],fit['offset'],np.sqrt(fit['objective']))
    logging.info('Starting five-fold exact ND GroupKFold')
    oof,comparison,starts=cross_validate(b6,p,cfg,OUT);multistart+=starts
    loqo,starts=q_level_cv(b6,p,cfg,OUT);multistart+=starts
    layers,scale_stats,starts=layer_fit(b6,p,cfg,OUT);multistart+=starts
    write_csv(OUT/'quality_fit_multistart_solutions.csv',multistart)
    write_csv(OUT/'quality_scaling_parameters.csv',[dict(**f,eta=p['beta']*f['gamma'] if m.startswith('MQ1') else None,
                                                       fit_dataset='B6_only',classical_parameters_frozen=True,
                                                       role='source_offset_sensitivity' if m.endswith('_offset') else ('main_candidate' if m=='MQ1' else 'form_sensitivity')) for m,f in fitted.items()])
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-completed-bootstrap',action='store_true',help='Reuse verified completed resamples after a downstream-only failure')
    args=parser.parse_args()
    if args.reuse_completed_bootstrap:
        bs,brows,gains=completed_bootstrap(cfg,fitted)
    else:
        logging.info('Starting 2000 ND-block bootstraps for four prescribed variants')
        bs,brows,gains=bootstrap(b6,p,cfg,OUT,oof)
    domain_path=ROOT/'outputs/quality/domain_quality_summary_A1.csv'
    _,domains=csv_rows(domain_path)
    form_models=[r for r in comparison if r['model'] in ['MQ1','MQ2']]
    form_winner=min(form_models,key=lambda r:r['RMSE'])['model']
    write_json(OUT/'parameters_before_B7_B8.json',dict(classical_parameters=p,
               fitted_quality_models=fitted,fit_dataset='B6_only',selection_dataset='B6_5fold_ND_GroupCV_only',
               B6_numeric_form_winner=form_winner,primary_candidate='MQ1',
               bootstrap_completed_before_validation=True,config_sha256=sha256(CONFIG),
               parameter_csv_sha256=sha256(OUT/'quality_scaling_parameters.csv'),
               main_parameters_sha256=sha256(parameter_path),
               prohibition='No tuning based on B7/B8; B8 incompatible stress test only'))
    # These evaluation sources are read only AFTER all B6 parameters/CV/bootstrap
    # have been saved. B8 never enters any least-squares call.
    b7=read_data(ROOT,'B7');b8=read_data(ROOT,'B8')
    ts={'B6':b6,'B7':b7,'B8':b8}
    predictions={}
    for label,t in ts.items():
        predictions[label]={m:predict_quality(t,p,m,0.0 if m=='M0' else fitted[m]['gamma'],
                                                0.0 if m=='M0' else fitted[m]['offset']) for m in MODELS}
        pointrows=[]
        for i,r in enumerate(t['records']):
            pointrows.append(dict(dataset=label,source_row=i+2,**r,
                                  **{m+'_predicted_Loss':float(predictions[label][m][i]) for m in MODELS}))
        write_csv(OUT/(label+'_fixed_quality_predictions.csv'),pointrows)
    keys6=set(b6['keys']);keys7=set(b7['keys'])
    b7over=np.array([k in keys6 for k in b7['keys']])
    masks7={'full':np.ones(len(b7['L']),dtype=bool),'overlap':b7over,'nonoverlap':~b7over}
    validation=[]
    for name,mask in masks7.items():
        for m,v in predictions['B7'].items():
            validation.append(evaluate(b7['L'][mask],v[mask],dataset='B7',subset=name,model=m,
                                       role='primary_extension_validation' if name=='nonoverlap' else 'overlapping_descriptive_only'))
    write_csv(OUT/'B7_validation.csv',validation)
    diffrows=[];diffpoints=[]
    masks6={'full':np.ones(len(b6['L']),dtype=bool)}
    difference_table(b6,predictions['B6'],masks6,diffrows,diffpoints,label_prefix='B6_in_sample')
    difference_table(b6,oof,masks6,diffrows,diffpoints,label_prefix='B6_GroupCV_out_of_fold')
    difference_table(b7,predictions['B7'],masks7,diffrows,diffpoints)
    write_csv(OUT/'within_ND_quality_difference_validation.csv',diffrows)
    write_csv(OUT/'within_ND_quality_difference_points.csv',diffpoints)
    diagnostics,pressure,masks8=stress_test(b8,b6,p,predictions['B8'],keys6,keys7,OUT)
    membership=[]
    for i,k in enumerate(b8['keys']):
        flags={axis:bool(b8[axis][i]<b6[axis].min() or b8[axis][i]>b6[axis].max()) for axis in ['N','D','Q']}
        membership.append(dict(source_row=i+2,N_params_B=float(b8['N'][i]),D_tokens_B=float(b8['D'][i]),Q_score=float(b8['Q'][i]),
                               within_B6_numeric_ranges=bool(masks8['within_B6_numeric_ranges'][i]),
                               nonoverlap=bool(masks8['nonoverlap'][i]),outside_N=flags['N'],outside_D=flags['D'],outside_Q=flags['Q'],
                               loss_below_E=bool(b8['L'][i]<p['E'])))
    write_csv(OUT/'B8_range_membership.csv',membership)
    # Summarize original overlap differences without changing the confirmed table.
    _,old_overlap=csv_rows(OUT/'overlap_point_details.csv')
    loss_differences=[]
    for source in ['B6','B7']:
        v=np.array([float(r['loss_difference']) for r in old_overlap if r['dataset_1']==source and r['dataset_2']=='B8'])
        loss_differences.append(dict(comparison='B8_minus_'+source,n=len(v),mean=float(v.mean()),median=float(np.median(v)),
                                     std=float(v.std(ddof=1)),min=float(v.min()),P025=float(np.quantile(v,.025)),
                                     P25=float(np.quantile(v,.25)),P75=float(np.quantile(v,.75)),P975=float(np.quantile(v,.975)),max=float(v.max()),
                                     mean_abs_difference=float(np.mean(abs(v))),max_abs_difference=float(max(abs(v))),same_loss_count=int((v==0).sum())))
    write_csv(OUT/'B8_overlap_loss_difference_distribution.csv',loss_differences)
    anchors=[]
    mask=b6['Q']==1.0
    for m,v in predictions['B6'].items():anchors.append(evaluate(b6['L'][mask],v[mask],dataset='B6',model=m,Q_ref=1.0))
    write_csv(OUT/'source_offset_anchor_sensitivity.csv',anchors)
    sensitivity=[]
    for base in ['MQ1','MQ2']:
        off=fitted[base+'_offset'];main=fitted[base]
        sensitivity.append(dict(model=base,gamma_no_offset=main['gamma'],gamma_with_offset=off['gamma'],
                                gamma_change=off['gamma']-main['gamma'],gamma_relative_change=(off['gamma']/main['gamma']-1),
                                delta_B6=off['offset'],interpretation='all-Q offset may absorb quality-form misspecification; not proof of a pure source zero-point shift'))
    write_csv(OUT/'source_offset_sensitivity.csv',sensitivity)
    qdomain=[float(r['Q_domain']) for r in domains]
    mainci=next(r for r in bs if r['model']=='MQ1' and r['parameter']=='gamma')
    cv0=next(r for r in comparison if r['model']=='M0');cv1=next(r for r in comparison if r['model']=='MQ1')
    difference=next(r for r in diffrows if r['dataset']=='B6' and r['model']=='MQ1' and r['evaluation']=='B6_GroupCV_out_of_fold' and not r['include_reference_zero_rows'])
    b7diff=next(r for r in diffrows if r['dataset']=='B7' and r['subset']=='nonoverlap' and r['model']=='MQ1' and not r['include_reference_zero_rows'])
    b7mq1=next(r for r in validation if r['subset']=='nonoverlap' and r['model']=='MQ1')
    gates=dict(group_cv_improves_M0=cv1['RMSE']<cv0['RMSE'],
               gamma_positive_stable=mainci['P025']>0 and mainci['failure_rate']==0 and mainci['at_bound_count']==0,
               within_ND_difference_positive_R2=difference['R2']>0,
               B7_nonoverlap_quality_difference_positive_R2=b7diff['R2']>0)
    summary=dict(status='COMPLETED_AWAITING_HUMAN_CONFIRMATION',classical_parameters=p,fit_models=fitted,
                 B6_monotonicity=mono,B6_group_cv=comparison,bootstrap=bs,CV_RMSE_gain=1-cv1['RMSE']/cv0['RMSE'],
                 bootstrap_CV_gains=gains,B6_numeric_form_winner=form_winner,
                 primary_candidate='MQ1',source_offset=sensitivity,
                 B6_primary_difference=difference,B7_primary_absolute=b7mq1,B7_primary_difference=b7diff,
                 B7_validation=validation,leave_one_Q_level_out=loqo,gamma_scale_dependence=scale_stats,
                 B8_incompatibility=diagnostics,B8_pressure_performance=pressure,B8_overlap_loss_differences=loss_differences,
                 question1_Q_domain_range=[min(qdomain),max(qdomain)],Q_numeric_coverage_only=True,
                 freeze_evidence_gates=gates,freeze_recommended=all(gates.values()),actual_human_freeze=False,
                 scope='B6/B7 numerical support; semisynthetic mechanism only; source/form confounding explicitly retained',
                 protected_existing_files=nprotected,bootstrap_repetitions=cfg['bootstrap_repetitions'],random_state=cfg['random_state'])
    write_json(OUT/'phase2B_execution_summary.json',summary)
    inputs=[b6['path'],b7['path'],b8['path'],parameter_path,domain_path,CONFIG]
    write_json(OUT/'quality_fit_input_provenance.json',dict(inputs=[dict(path=f.relative_to(ROOT).as_posix(),sha256=sha256(f)) for f in inputs],
                                                         fitting_sources=['B6'],selection_sources=['B6'],validation_sources=['B7'],pressure_sources=['B8']))
    from src.scaling.quality_reporting import make_figures,make_report
    make_figures(ROOT,ts,p,fitted,oof,comparison,layers,brows,masks7,masks8,predictions,diffpoints,cfg)
    make_report(ROOT,summary,bs,diffrows,anchors)
    protected=verify_existing()
    write_json(OUT/'phase2B_frozen_integrity_check.json',dict(passed=True,checked_files=protected,modified_existing_files=[]))
    logging.info('Phase 2B completed; %d protected files unchanged; awaiting human confirmation; no p/K/B9/B10 used',protected)
    print(json.dumps({'main_gamma':fitted['MQ1']['gamma'],'gamma_CI':[mainci['P025'],mainci['P975']],
                      'eta':p['beta']*fitted['MQ1']['gamma'],'B6_CV_form_winner':form_winner,
                      'gates':gates,'freeze_recommended':summary['freeze_recommended'],'protected_files':protected},indent=2))


if __name__=='__main__':
    try:main()
    except Exception:
        logging.exception('Quality pipeline failed; no downstream stage permitted')
        raise
