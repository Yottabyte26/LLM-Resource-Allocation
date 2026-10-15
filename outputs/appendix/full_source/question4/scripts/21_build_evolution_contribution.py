"""Question 4 PHASE 2: contribution analysis only; no ability/bridge refitting."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/'src'))
import argparse
import json
import logging
import pandas as pd
import yaml
from evolution.contribution_analysis import prepare,fit_models,run_decomposition,model_data
from evolution.contribution_regression import design


def load_models(root):
    """Recover this phase's cached estimates, retaining frozen input model IDs."""
    out=root/'outputs/evolution/phase2';d=pd.read_csv(out/'analysis_model_table.csv',float_precision='round_trip')
    for col in ['submission_date','release_date']:d[col]=pd.to_datetime(d[col],errors='coerce')
    definitions=json.loads((out/'model_definitions.json').read_text(encoding='utf8'));models={}
    for item in definitions:
        g=d.set_index('Model',drop=False).loc[item['Models']].reset_index(drop=True)
        g['t_month']=g['t_release' if item['time_axis']=='release' else 't_submission']
        if 't_squared' in item['terms']:g['t_squared']=g.t_month**2
        if any(t.startswith('month_') for t in item['terms']):
            dummy=pd.get_dummies(g.submission_date.dt.to_period('M').astype(str),prefix='month',drop_first=True,dtype=float)
            g=pd.concat([g,dummy],axis=1)
        terms=item['terms'];models[item['model']]=dict(label=item['model'],scope=item['scope'],axis=item['time_axis'],
            method=item['method'],terms=terms,names=['intercept']+terms,n=len(g),data=g,
            X=design(g,terms),y=g.A_z.to_numpy(float),beta=item['beta'])
    metadata=json.loads((out/'sample_definition.json').read_text(encoding='utf8'))
    return models,d,metadata


def main():
    """Checkpoint samples, models, bootstrap and report without touching Phase 1."""
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['samples','models','bootstrap','report','all'],default='all');args=p.parse_args()
    out=PROJECT_ROOT/'outputs/evolution/phase2';out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(out/'execution.log',encoding='utf8')])
    cfg=yaml.safe_load((PROJECT_ROOT/'configs/evolution_phase2.yaml').read_text(encoding='utf8'))
    if cfg['future_prediction_enabled'] or cfg['loss_conversion_enabled']:raise ValueError('Outside authorized Phase 2 scope')
    if not (out/'frozen_before_question4_phase2.json').exists():raise ValueError('Freeze snapshot required before stage execution')
    if args.stage in ['samples','models','all']:
        d,metadata=prepare(PROJECT_ROOT,cfg)
        if args.stage in ['models','all']:models=fit_models(PROJECT_ROOT,cfg,d,metadata)
    if args.stage in ['bootstrap','all']:
        if args.stage=='bootstrap':models,d,metadata=load_models(PROJECT_ROOT)
        run_decomposition(PROJECT_ROOT,cfg,models,metadata)
    if args.stage in ['report','all']:
        if args.stage=='report':models,d,metadata=load_models(PROJECT_ROOT)
        from evolution.contribution_reporting import finalize
        finalize(PROJECT_ROOT,cfg,models,d,metadata)
    logging.info('Question 4 PHASE 2 %s complete; no future forecasting or Loss conversion',args.stage)


if __name__=='__main__':
    try:main()
    except Exception:
        logging.exception('Question 4 PHASE 2 stopped on an error; frozen inputs remain read-only')
        raise
