"""Independent Question 4 PHASE 1; stop before decomposition/forecasting."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/'src'))
import argparse
import logging
import pandas as pd
import yaml
from evolution.metadata import inspect_sources,canonical_models,epoch_join


def main():
    """Run checkpointed source/model/ability, exhaustive C8, bridge, report steps."""
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['ability','C8','bridge','report','all'],default='all');args=p.parse_args()
    out=PROJECT_ROOT/'outputs/evolution/phase1';out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(out/'execution.log',encoding='utf8')])
    cfg=yaml.safe_load((PROJECT_ROOT/'configs/evolution_phase1.yaml').read_text(encoding='utf8'))
    if cfg['future_prediction_enabled'] or cfg['contribution_decomposition_enabled']:raise ValueError('Outside PHASE 1 scope')
    if args.stage in ['ability','all']:
        from evolution.ability import construct_ability
        data=inspect_sources(PROJECT_ROOT,cfg);models=epoch_join(canonical_models(data,PROJECT_ROOT,cfg),data,PROJECT_ROOT,cfg)
        construct_ability(PROJECT_ROOT,cfg,models,data)
    if args.stage in ['C8','all']:
        from evolution.detailed import analyze_details
        analyze_details(PROJECT_ROOT,cfg)
    if args.stage in ['bridge','all']:
        from evolution.bridge import analyze_bridge
        analyze_bridge(PROJECT_ROOT,cfg)
    if args.stage in ['report','all']:
        from evolution.reporting import finalize
        finalize(PROJECT_ROOT,cfg)
    logging.info('Question 4 PHASE 1 %s completed; no forecast or contribution decomposition',args.stage)


if __name__=='__main__':
    try:main()
    except Exception:
        logging.exception('Question 4 PHASE 1 stopped on an error; frozen files unchanged')
        raise
