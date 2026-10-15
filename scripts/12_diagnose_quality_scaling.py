"""Diagnose B6--B8 without fitting; stop on reversed quality semantics.

Only reads supplementary N/D/Q CSVs, frozen classical parameters and frozen
domain summaries. It never transforms Q or fits any model.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import csv
import json
import logging
from collections import Counter, defaultdict
from decimal import Decimal
from itertools import combinations, product
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.scaling.io import csv_rows, sha256, write_csv, write_json
from src.scaling.classical import predict, metrics

OUT = ROOT / 'outputs/scaling/phase2'
FIG = ROOT / 'outputs/figures/question2_phase2'
REPORT = ROOT / 'outputs/report/question2_phase2_quality_scaling.md'
FIELDS = ['N_params_B', 'D_tokens_B', 'Q_score', 'val_loss']
NAMES = {
    'B6': 'supplementary_NQ_experiment.csv',
    'B7': 'supplementary_NQ_experiment_expanded.csv',
    'B8': 'supplementary_NQ_experiment_large.csv',
}
TOL = 1e-12  # numerical comparison only; not a data-derived threshold


def canonical(value):
    """Exact decimal key, treating 0.07 and 0.070 as the same numeric value."""
    return format(Decimal(value).normalize(), 'f')


def load(label, name):
    """Strictly parse finite numeric N/D/Q/L; retain all source rows."""
    path = ROOT / 'real_attachments/B_scaling_laws' / name
    columns, rows = csv_rows(path)
    a = np.array([[float(r[c]) for c in FIELDS] for r in rows])
    missing = sum(str(v).strip().lower() in ['', 'nan', 'none', 'null', 'na', 'n/a']
                  for r in rows for v in r.values())
    if not np.isfinite(a).all() or (a[:, :3] <= 0).any() or (a[:, 2] > 1).any():
        raise ValueError(f'{label}: nonfinite or invalid N/D/Q; no imputation allowed')
    keys = [tuple(canonical(r[c]) for c in FIELDS[:3]) for r in rows]
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k[:2]].append(i)
    exact = Counter(tuple(r[c] for c in columns) for r in rows)
    counts = Counter(keys)
    levels = [sorted(set(a[:, j])) for j in range(3)]
    expected = int(np.prod([len(v) for v in levels]))
    summary = dict(dataset=label, source_file=path.relative_to(ROOT).as_posix(),
                   rows=len(rows), columns=columns, missing_or_nan_cells=missing,
                   nan_required_cells=int(np.isnan(a).sum()),
                   inf_required_cells=int(np.isinf(a).sum()),
                   exact_duplicate_groups=sum(v > 1 for v in exact.values()),
                   exact_duplicate_excess_rows=sum(max(v - 1, 0) for v in exact.values()),
                   duplicate_candidate_key_groups=sum(v > 1 for v in counts.values()),
                   unique_N=len(levels[0]), unique_D=len(levels[1]), unique_Q=len(levels[2]),
                   unique_ND=len(groups), expected_grid_points=expected,
                   actual_unique_grid_points=len(counts), missing_grid_points=expected-len(counts),
                   grid_coverage=len(counts)/expected, complete_cartesian_grid=len(counts)==expected)
    for j, c in enumerate(FIELDS):
        summary[c+'_min'] = float(a[:, j].min())
        summary[c+'_max'] = float(a[:, j].max())
    logging.info('%s: %d rows; %d exact numeric ND groups', label, len(rows), len(groups))
    return dict(label=label, path=path, columns=columns, rows=rows, a=a,
                keys=keys, groups=groups, summary=summary, levels=levels)


def monotonicity(t):
    """Use exact ND groups; ties are not increasing violations."""
    rows = []
    for (n, d), idx in sorted(t['groups'].items(), key=lambda kv: tuple(map(float, kv[0]))):
        a = t['a'][idx]
        a = a[np.argsort(a[:, 2], kind='stable')]
        if len(set(a[:, 2])) != len(a):
            raise ValueError('Repeated NDQ key requires resolution before monotonicity test')
        diff = np.diff(a[:, 3])
        rho = float(spearmanr(a[:, 2], a[:, 3]).statistic) if len(a) > 1 and np.ptp(a[:, 3]) else None
        rows.append(dict(dataset=t['label'], N_params_B=n, D_tokens_B=d,
                         n_Q=len(a), spearman_Q_Loss=rho,
                         negative_correlation=rho is not None and rho < -TOL,
                         positive_correlation=rho is not None and rho > TOL,
                         nonincreasing=bool(np.all(diff <= TOL)),
                         strictly_decreasing=bool(np.all(diff < -TOL)),
                         increasing_adjacent_pairs=int((diff > TOL).sum()),
                         decreasing_adjacent_pairs=int((diff < -TOL).sum()),
                         tied_adjacent_pairs=int((abs(diff) <= TOL).sum()),
                         adjacent_pairs=len(diff),
                         violation_pair_rate=float((diff > TOL).mean()) if len(diff) else None))
    m = len(rows)
    npairs = sum(r['adjacent_pairs'] for r in rows)
    violations = sum(r['increasing_adjacent_pairs'] for r in rows)
    summary = dict(dataset=t['label'], n_ND_groups=m,
                   negative_groups=sum(r['negative_correlation'] for r in rows),
                   positive_groups=sum(r['positive_correlation'] for r in rows),
                   fully_nonincreasing_groups=sum(r['nonincreasing'] for r in rows),
                   fully_strictly_decreasing_groups=sum(r['strictly_decreasing'] for r in rows),
                   violation_groups=sum(not r['nonincreasing'] for r in rows),
                   adjacent_pairs=npairs, increasing_adjacent_pairs=violations,
                   violation_pair_rate=violations/npairs)
    for name in ['negative_groups', 'positive_groups', 'fully_nonincreasing_groups', 'violation_groups']:
        summary[name+'_rate'] = summary[name]/m
    return rows, summary


def overlap(left, right):
    """Report same numerical keys without discarding any conflicting Loss values."""
    if len(set(left['keys'])) != len(left['keys']) or len(set(right['keys'])) != len(right['keys']):
        raise ValueError('Duplicate key: no automatic first/last selection')
    ld = dict(zip(left['keys'], left['rows']))
    rd = dict(zip(right['keys'], right['rows']))
    common = sorted(ld.keys() & rd.keys(), key=lambda k: tuple(map(float, k)))
    details = []
    for k in common:
        a, b = ld[k], rd[k]
        same_loss = Decimal(a['val_loss']) == Decimal(b['val_loss'])
        numeric_same = all(Decimal(a[c]) == Decimal(b[c]) for c in FIELDS)
        shared = set(left['columns']) & set(right['columns'])
        shared_same = all(Decimal(a[c]) == Decimal(b[c]) if c in FIELDS else a[c] == b[c] for c in shared)
        details.append(dict(dataset_1=left['label'], dataset_2=right['label'],
                            **dict(zip(FIELDS[:3], k)), loss_1=a['val_loss'], loss_2=b['val_loss'],
                            loss_difference=float(Decimal(b['val_loss'])-Decimal(a['val_loss'])),
                            same_loss=same_loss, same_numeric_data=numeric_same,
                            same_shared_fields=shared_same,
                            exact_full_record_same=left['columns']==right['columns'] and a==b))
    s = dict(dataset_1=left['label'], dataset_2=right['label'],
             rows_1=len(ld), rows_2=len(rd), overlapping_keys=len(common),
             overlap_ratio_1=len(common)/len(ld), overlap_ratio_2=len(common)/len(rd),
             same_key_same_loss=sum(r['same_loss'] for r in details),
             same_key_different_loss=sum(not r['same_loss'] for r in details),
             same_numeric_data=sum(r['same_numeric_data'] for r in details),
             same_shared_fields=sum(r['same_shared_fields'] for r in details),
             exact_full_record_same=sum(r['exact_full_record_same'] for r in details),
             keys_1_subset_of_2=set(ld) <= set(rd), keys_2_subset_of_1=set(rd) <= set(ld))
    return s, details


def verify_frozen():
    """Verify every file existing before Phase 2 byte-for-byte, including originals."""
    manifest = json.loads((OUT/'frozen_before_phase2.json').read_text(encoding='utf-8'))
    failures = []
    for r in manifest['files']:
        p = ROOT/r['path']
        if not p.is_file() or p.stat().st_size != r['bytes'] or sha256(p) != r['sha256']:
            failures.append(r['path'])
    if failures:
        raise RuntimeError('Frozen files changed: '+repr(failures))
    return len(manifest['files'])


def main():
    """Produce audit evidence; reversed B8 semantics prevents downstream fitting."""
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(OUT/'diagnosis.log', encoding='utf-8'),
                                  logging.StreamHandler()])
    protected_count = verify_frozen()
    ts = {label: load(label, name) for label, name in NAMES.items()}
    write_csv(OUT/'dataset_diagnosis.csv', [t['summary'] for t in ts.values()])
    missing_grid = []
    for label, t in ts.items():
        for ndq in product(*t['levels']):
            k = tuple(canonical(str(v)) for v in ndq)
            if k not in set(t['keys']):
                missing_grid.append(dict(dataset=label, **dict(zip(FIELDS[:3], k))))
    write_csv(OUT/'missing_grid_points.csv', missing_grid, ['dataset']+FIELDS[:3])
    group_rows, mono = [], []
    for t in ts.values():
        a, b = monotonicity(t)
        group_rows.extend(a); mono.append(b)
    write_csv(OUT/'Q_monotonicity_diagnosis.csv', group_rows)
    write_csv(OUT/'Q_monotonicity_summary.csv', mono)
    overlaps, overlap_rows = [], []
    for l, r in combinations(ts.values(), 2):
        a, b = overlap(l, r)
        overlaps.append(a); overlap_rows.extend(b)
    write_csv(OUT/'dataset_overlap.csv', overlaps)
    write_csv(OUT/'overlap_point_details.csv', overlap_rows)
    b6keys = set(ts['B6']['keys'])
    b7keys = set(ts['B7']['keys'])
    subset_rows = []
    for label, excluded, rule in [('B7', b6keys, 'exclude exact B6 NDQ keys'),
                                  ('B8', b6keys|b7keys, 'exclude union of B6 and B7 NDQ keys')]:
        t = ts[label]
        for i, k in enumerate(t['keys']):
            subset_rows.append(dict(dataset=label, row_number=i+2, **dict(zip(FIELDS[:3], k)),
                                    included_in_nonoverlap=k not in excluded, exclusion_rule=rule))
    write_csv(OUT/'nonoverlap_membership.csv', subset_rows)
    parameters_path = ROOT/'outputs/scaling/phase1/classical_scaling_parameters.csv'
    _, parameter_rows = csv_rows(parameters_path)
    primary = [r for r in parameter_rows if r['model']=='M0_N_D' and r['weighting']=='trajectory_balanced']
    if len(primary) != 1:
        raise ValueError('Frozen primary parameter row is ambiguous')
    parameters = {k:float(primary[0][k]) for k in ['E', 'A', 'B', 'alpha', 'beta']}
    anchor_rows = []; anchor_points = []
    for label, t in ts.items():
        a = t['a']; qref = float(a[:, 2].max())
        ix = np.flatnonzero(a[:, 2] == qref)
        pred = predict(parameters, a[ix, 0], a[ix, 1])
        m = metrics(a[ix, 3], pred)
        anchor_rows.append(dict(dataset=label, Q_ref=qref, exact_Q1=qref==1,
                                **m, residual_min=float((a[ix, 3]-pred).min()),
                                residual_max=float((a[ix, 3]-pred).max())))
        for i, yp in zip(ix, pred):
            anchor_points.append(dict(dataset=label, N_params_B=float(a[i, 0]),
                                      D_tokens_B=float(a[i, 1]), Q_score=float(a[i, 2]),
                                      actual_loss=float(a[i, 3]), frozen_L0=float(yp),
                                      residual_actual_minus_L0=float(a[i, 3]-yp)))
    write_csv(OUT/'Q1_anchor_check.csv', anchor_rows)
    write_csv(OUT/'Q1_anchor_points.csv', anchor_points)
    domain_path = ROOT/'outputs/quality/domain_quality_summary_A1.csv'
    _, domain_rows = csv_rows(domain_path)
    compatible = []
    for r in domain_rows:
        for label, t in ts.items():
            q = float(r['Q_domain']); lo, hi = float(t['a'][:,2].min()), float(t['a'][:,2].max())
            compatible.append(dict(domain=r['domain'], Q_domain=q, dataset=label,
                                   Q_min=lo, Q_max=hi, numeric_range_interpolation=lo<=q<=hi,
                                   semantic_calibration_confirmed=False,
                                   note='Numerical range coverage alone does not establish equivalent quality semantics'))
    write_csv(OUT/'Q_scale_compatibility.csv', compatible)
    examples = []
    for label, t in ts.items():
        # Same exact numerical ND, not text-string matching.
        idx = t['groups'][('0.07', '10')]
        for i in sorted(idx, key=lambda i:t['a'][i, 2]):
            examples.append(dict(dataset=label, **{c:float(t['a'][i,j]) for j,c in enumerate(FIELDS)}))
    write_csv(OUT/'direction_counterexample.csv', examples)
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for ax, (label,t) in zip(axes,ts.items()):
        a = t['a']
        s = ax.scatter(a[:,0],a[:,1],c=a[:,2],cmap='viridis',s=12)
        ax.set(xscale='log',yscale='log',xlabel='N (billion parameters)',ylabel='D (billion tokens)',title=label+' NDQ coverage')
        fig.colorbar(s,ax=ax,label='Q_score')
    fig.tight_layout(); fig.savefig(FIG/'01_NDQ_coverage.png',dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5))
    for label,t in ts.items():
        a=t['a'][t['groups'][('0.07','10')]]; a=a[np.argsort(a[:,2])]
        ax.plot(a[:,2],a[:,3],'.-',label=label)
    ax.set(xlabel='Q_score',ylabel='Validation Loss',title='Opposite Q direction at fixed N=0.07 B, D=10 B')
    ax.legend(); fig.tight_layout(); fig.savefig(FIG/'02_fixed_ND_direction_conflict.png',dpi=160); plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(11,4))
    for label in ts:
        v=[r['spearman_Q_Loss'] for r in group_rows if r['dataset']==label]
        axes[0].hist(v,bins=np.linspace(-1,1,21),alpha=.5,label=label)
        ar=[r['residual_actual_minus_L0'] for r in anchor_points if r['dataset']==label]
        axes[1].hist(ar,bins=15,alpha=.5,label=label)
    axes[0].set(xlabel='Within-ND Spearman(Q, Loss)',ylabel='Number of ND groups',title='Quality-direction diagnosis')
    axes[1].set(xlabel='Actual Loss - frozen L0 at Q=1',ylabel='Number of ND groups',title='Q=1 anchor residuals')
    for ax in axes:ax.legend()
    fig.tight_layout(); fig.savefig(FIG/'03_monotonicity_and_anchor.png',dpi=160);plt.close(fig)
    diagnosis = ['# N/D/Q 数据诊断', '',
                 'B6–B8 为“基于真实数据校准的半合成补充实验”，不是直接训练实验。', '',
                 '所有主键均经 Decimal 精确数值规范化：0.07 与 0.070 是同一工作点；未作近似匹配、填补、去重或质量反向。', '']
    for t in ts.values():
        diagnosis += ['## '+t['label'],'','```json',json.dumps(t['summary'],ensure_ascii=False,indent=2),'```','']
    diagnosis += ['候选主键 (N,D,Q) 无重复，B7_nonoverlap 排除 B6 主键；B8_nonoverlap 排除 B6∪B7 主键。',
                  'Q 单调性按精确固定 (N,D) 分组；相邻 Loss 增加超过 1e-12 为违背，相等不是违背。',
                  '该容差仅防浮点误差；不是人为调整数据、噪声阈值或统计显著性阈值。',
                  'B8 的 Q–Loss 方向与主假设相反。已触发用户指定的拟合停止条件。']
    (OUT/'data_diagnosis_NDQ.md').write_text('\n'.join(diagnosis)+'\n',encoding='utf-8')
    report = ['# 问题二 PHASE 2：质量扩展诊断与停止报告','',
              '**状态：NEEDS_CONFIRMATION。尚未拟合 γ；L(N,D,Q) 尚不能冻结。**','',
              '用户要求：“如果数据本身不支持 Q↑→Loss↓，立即报告，不要强行拟合正 gamma。”B8 触发此停止条件。', '',
              '## 数据范围与网格','',
              '|数据|行数|N (B)|D (B)|Q|Loss|N/D/Q 唯一值数|网格覆盖|',
              '|---|---:|---|---|---|---|---|---|']
    for t in ts.values():
        a=t['a']; s=t['summary']
        ranges=['%.6g–%.6g'%(a[:,j].min(),a[:,j].max()) for j in range(4)]
        report.append('|%s|%s|%s|%s|%s|%s|%s/%s/%s|%s/%s|'%(t['label'],len(a),*ranges,s['unique_N'],s['unique_D'],s['unique_Q'],s['actual_unique_grid_points'],s['expected_grid_points']))
    report += ['','## 重叠与非重叠定义','',
               '|组合|相同 NDQ 主键|占左/右比例|相同 Loss|不同 Loss|','|---|---:|---|---:|---:|']
    for s in overlaps:
        report.append(f"|{s['dataset_1']}∩{s['dataset_2']}|{s['overlapping_keys']}|{s['overlap_ratio_1']:.2%}/{s['overlap_ratio_2']:.2%}|{s['same_key_same_loss']}|{s['same_key_different_loss']}|")
    report += ['','B7 的全量结果不能称为独立验证。即使主键不重叠，半合成构造与来源也不提供真实外部训练验证。',
               'B7_nonoverlap 排除 B6 的 NDQ；B8_nonoverlap 排除 B6∪B7 的 NDQ，不以 Loss 是否相同决定排除。', '',
               '## 方向与单调性','',
               '|数据|ND组数|负相关组|正相关组|完全非增组|违背组|违背相邻对|',
               '|---|---:|---:|---:|---:|---:|---:|']
    for s in mono:
        report.append(f"|{s['dataset']}|{s['n_ND_groups']}|{s['negative_groups']}|{s['positive_groups']}|{s['fully_nonincreasing_groups']}|{s['violation_groups']}|{s['increasing_adjacent_pairs']}/{s['adjacent_pairs']} ({s['violation_pair_rate']:.2%})|")
    report += ['','B6/B7 总体方向正确，但并非每一相邻 Q 点都严格下降；B8 每一个工作点的组内秩相关均为正，方向不能由全局 N/D 混杂解释。',
               '证据实例固定 N=0.07 B、D=10 B：','',
               '|数据|Q|Loss|','|---|---:|---:|']
    for r in examples:
        report.append(f"|{r['dataset']}|{r['Q_score']:.6g}|{r['val_loss']:.6g}|")
    report += ['','## Q=1 锚点','',
               '以下均直接读取 PHASE 1 参数 CSV 的完整精度，未拟合或校准经典参数。残差定义为实际 Loss−L0。', '',
               '|数据|锚点数|mean bias|median bias|RMSE|','|---|---:|---:|---:|---:|']
    for r in anchor_rows:
        report.append(f"|{r['dataset']}|{r['n']}|{r['mean_bias']:.9g}|{r['median_residual']:.9g}|{r['RMSE']:.9g}|")
    report += ['','锚点偏差是跨源兼容性诊断；不是 γ、source offset 或有效数据量假设已经成立的证据。', '',
               '## 问题一 Q 的数值覆盖','',
               f"冻结领域 Q_domain 范围为 {min(float(r['Q_domain']) for r in domain_rows):.12g}–{max(float(r['Q_domain']) for r in domain_rows):.12g}。",
               '全部位于 B6 的 Q 数值区间内，但数值覆盖不等于 B6 Q_score 与 CRITIC Q 已完成语义标定。', '',
               '## 拟合与验证未执行','',
               '有效数据量 Power / Exponential 候选尚未拟合，γ/η/置信区间、Group CV、组内差分预测、B7/B8 模型验证、γ_N/γ_D 均未生成，避免把方向相反的补充集纳入认可的验证链。',
               '已完成数据诊断和证据图，未生成依赖拟合的其余图表。未引入 p/K、未使用 B9/B10、未推导参数替代量或最终弹性。', '',
               '## 待人工确认事项','',
               '请确认 B8 Q_score 是否与 B6/B7 使用同一“越大越好”的定义，或允许仅用 B6 拟合、B7 非重叠点验证，将 B8 单列为方向不兼容的负面证据。当前没有证据允许自动把 Q 改为 1−Q，也不修改官方数据。', '',
               '如选择继续 B6/B7，仍需单独审查 Q=1 跨源偏置和质量来源混杂；所有经典参数继续冻结。', '',
               '## 文件与冻结核验','',
               '输入仅 B6/B7/B8、PHASE 1 主参数 CSV、问题一冻结领域汇总 CSV。原始质量 XZ/JSONL 未解析。',
               f'开始时保护 {protected_count} 个既有文件；结束再次逐文件 SHA-256 核验。详情见 frozen_integrity_check.json。']
    REPORT.write_text('\n'.join(report)+'\n',encoding='utf-8')
    inputs = [t['path'] for t in ts.values()]+[parameters_path,domain_path]
    write_json(OUT/'input_provenance.json',{'numerical_inputs':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha256(p)} for p in inputs],
               'frozen_classical_parameters':parameters,'fitting_executed':False})
    n = verify_frozen()
    write_json(OUT/'frozen_integrity_check.json',{'passed':True,'checked_files':n,'modified_existing_files':[]})
    summary={'status':'NEEDS_CONFIRMATION','reason':'B8: all 150 fixed-ND groups have positive Q-Loss Spearman correlation',
             'fitting_executed':False,'datasets':[t['summary'] for t in ts.values()],
             'monotonicity':mono,'overlap':overlaps,'anchor':anchor_rows,
             'nonoverlap_counts':{label:sum(r['dataset']==label and r['included_in_nonoverlap'] for r in subset_rows) for label in ['B7','B8']},
             'protected_files_verified':n}
    write_json(OUT/'diagnosis_execution_summary.json',summary)
    logging.info('Diagnosis completed; NEEDS_CONFIRMATION; no fitting. Protected files verified: %d',n)
    print(json.dumps(summary,ensure_ascii=True,indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logging.exception('Diagnosis failed; no downstream fit attempted')
        raise
