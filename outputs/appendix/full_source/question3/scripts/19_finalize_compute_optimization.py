"""Question 3 PHASE 3 only: new diagnostics and final reporting; no refitting."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/'src'))
import argparse
import logging
import pandas as pd
import yaml
from optimization.simplex_sensitivity import simplex_search,official_comparison


def main():
    """Run separately checkpointed simplex, official comparisons, and reporting."""
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['simplex','official','report','all'],default='all')
    args=parser.parse_args();out=PROJECT_ROOT/'outputs/optimization/phase3';out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(out/'execution.log',encoding='utf8')])
    cfg=yaml.safe_load((PROJECT_ROOT/'configs/optimization_phase3.yaml').read_text(encoding='utf8'))
    if cfg['question4_enabled']:raise ValueError('Question 4 is outside scope')
    if args.stage in ('simplex','all'):simplex_search(PROJECT_ROOT,cfg)
    if args.stage in ('official','all'):
        extremes=pd.read_csv(out/'P2_A_extreme_recipes.csv',float_precision='round_trip')
        official_comparison(PROJECT_ROOT,extremes)
    if args.stage in ('report','all'):
        from optimization.final_reporting import finalize
        finalize(PROJECT_ROOT,cfg)
    logging.info('PHASE 3 stage %s completed; earlier phases unchanged; Question 4 not started',args.stage)


if __name__=='__main__':main()
