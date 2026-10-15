"""Question 3 Phase 2: frozen resource optimization and transition diagnosis."""
from pathlib import Path
import argparse
import hashlib
import json
import logging
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'outputs/optimization/phase2'
NEW_FILES={'scripts/18_optimize_compute_resources.py','scripts/18_verify_compute_optimization.py',
           'configs/optimization_phase2.yaml','src/optimization/resource_solver.py',
           'src/optimization/transitions.py','src/optimization/resource_reporting.py',
           'outputs/report/question3_phase2_optimization.md'}
NEW_PREFIXES=('outputs/optimization/phase2/','outputs/figures/question3_phase2/')


def sha(path):
    """Hash protected data without parsing or rewriting it."""
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def snapshot():
    """Save a once-only byte baseline including all of frozen Phase 1."""
    OUT.mkdir(parents=True,exist_ok=True);target=OUT/'frozen_before_phase2.json'
    if target.exists():return
    records={}
    for p in ROOT.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ('.git','.codex','.agents')):continue
        rel=p.relative_to(ROOT).as_posix()
        if rel in NEW_FILES or rel.startswith(NEW_PREFIXES):continue
        records[rel]={'size':p.stat().st_size,'sha256':sha(p)}
    target.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Frozen files protected:',len(records),flush=True)


def main():
    """Run authorized Phase 2 only, with checkpointable outputs and logs."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-only',action='store_true')
    parser.add_argument('--stage',choices=['solve','diagnose','report','all'],default='all')
    args=parser.parse_args();snapshot()
    if args.snapshot_only:return
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(OUT/'execution.log',encoding='utf-8')])
    try:
        if args.stage in ('solve','all'):
            from optimization.resource_solver import run_solve
            run_solve(ROOT)
        if args.stage in ('diagnose','all'):
            from optimization.transitions import run_diagnose
            run_diagnose(ROOT)
        if args.stage in ('report','all'):
            from optimization.resource_reporting import run_report
            run_report(ROOT)
    except Exception:
        logging.exception('Phase 2 failed; protected prior files remain read-only');raise


if __name__=='__main__':main()
