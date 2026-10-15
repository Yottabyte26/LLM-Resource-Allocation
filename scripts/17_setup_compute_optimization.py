"""Question 3 Phase 1 only: audit inputs and formulate, never optimize."""
from pathlib import Path
import argparse
import hashlib
import json
import logging
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
OUT = PROJECT_ROOT / 'outputs/optimization/phase1'
NEW_PREFIXES = ('outputs/optimization/phase1/', 'outputs/figures/question3_phase1/',
                'src/optimization/')
NEW_FILES = {'scripts/17_setup_compute_optimization.py',
             'scripts/17_verify_compute_setup.py', 'configs/optimization_phase1.yaml',
             'outputs/report/question3_phase1_setup.md'}


def file_hash(path):
    """Hash file bytes in blocks, without parsing or changing official data."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def protected_files():
    """Every existing project artifact outside this newly authorized phase."""
    for path in PROJECT_ROOT.rglob('*'):
        if not path.is_file():
            continue
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if rel in NEW_FILES or rel.startswith(NEW_PREFIXES):
            continue
        if any(x in path.parts for x in ('.git', '.codex', '.agents')):
            continue
        yield path, rel


def snapshot():
    """Write the before-state once; never silently replace the freeze baseline."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'frozen_before_phase1.json'
    if target.exists():
        return
    values = {rel: {'size': p.stat().st_size, 'sha256': file_hash(p)}
              for p, rel in protected_files()}
    target.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Protected files snapshotted: {len(values)}', flush=True)


def main():
    """Run setup only, with a persistent log and explicit failure reporting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-only', action='store_true')
    args = parser.parse_args()
    snapshot()
    if args.snapshot_only:
        return
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(OUT/'execution.log', encoding='utf-8')])
    try:
        from optimization.setup import run
        run(PROJECT_ROOT)
    except Exception:
        logging.exception('Phase 1 setup failed; no optimizer has been invoked')
        raise


if __name__ == '__main__':
    main()
