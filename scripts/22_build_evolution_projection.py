"""Q4 Phase 3 only: frozen-input scenario projection, never rerun Phase 1/2."""
import argparse
import logging
import sys
from pathlib import Path
sys.dont_write_bytecode=True
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/'src'))
import yaml
from evolution.projection_data import load_frozen,build_anchors,benchmark_series,compute_candidates,estimate_growth,c3_direction


def main():
    """Allow auditable staged execution with existing frozen inputs."""
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['data','hindcast','project','report','all'],default='all');args=parser.parse_args()
    out=PROJECT_ROOT/'outputs/evolution/phase3';out.mkdir(parents=True,exist_ok=True)
    if not (out/'frozen_before_question4_phase3.json').is_file():raise RuntimeError('Frozen baseline snapshot required before running new analysis')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(out/'execution.log',encoding='utf8')])
    cfg=yaml.safe_load((PROJECT_ROOT/'configs/evolution_phase3.yaml').read_text(encoding='utf8'));d=load_frozen(PROJECT_ROOT)
    logging.info('Phase 3 %s; frozen S1=%d; no ability/M2/Q1-Q3 fitting',args.stage,len(d))
    if args.stage in ['data','all']:
        base,_=build_anchors(PROJECT_ROOT,cfg,d);benchmark_series(PROJECT_ROOT,cfg,d,base)
        data=compute_candidates(PROJECT_ROOT,d,base);estimate_growth(PROJECT_ROOT,cfg,data,base);c3_direction(PROJECT_ROOT)
    if args.stage in ['hindcast','all']:
        from evolution.projection_models import build_hindcast_and_trends
        build_hindcast_and_trends(PROJECT_ROOT,cfg,d)
    if args.stage in ['project','all']:
        from evolution.projection_models import build_projections
        build_projections(PROJECT_ROOT,cfg,d)
    if args.stage in ['report','all']:
        from evolution.projection_reporting import build_report_and_figures
        build_report_and_figures(PROJECT_ROOT,cfg,d)
    logging.info('Phase 3 %s completed normally',args.stage)


if __name__=='__main__':main()
