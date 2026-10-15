# 问题二：论文源程序逐文件汇编
# For appendix display only; NOT a standalone merged executable.
# Source bodies, imports, formulas, branches, parameters and entry guards are unchanged.
# Use the original scripts from the project root for the original workflow; DO NOT run this listing.
# This extraction task did not rerun any model or experiment.

# ============================================================
# Source: src/scaling/io.py
# Original lines: 1-116
# Purpose: 真实附件读取、索引/字段兼容与输入单位
# ============================================================
"""Strict CSV intake, provenance, trajectory grouping and immutable-input hashing."""
from __future__ import annotations
import csv
from collections import Counter
from decimal import Decimal
import hashlib
from pathlib import Path
import json
import numpy as np

PRIMARY_COLUMNS=['N_params_B','D_tokens_B','val_loss']


def sha256(path):
    """Hash bytes without interpreting or decompressing original attachments."""
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def csv_rows(path):
    """Validate CSV width and headers, preserving original text/row order."""
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);columns=reader.fieldnames;records=list(reader)
    if not columns or len(columns)!=len(set(columns)) or not records:
        raise ValueError(f'Invalid/empty CSV: {path}')
    if any(None in r or any(v is None for v in r.values()) for r in records):
        raise ValueError(f'Truncated CSV: {path}')
    return columns,records


def write_csv(path,records,columns=None):
    """UTF-8-BOM CSVs; arrays/dicts serialized explicitly as JSON cells."""
    path.parent.mkdir(parents=True,exist_ok=True)
    if columns is None:columns=list(dict.fromkeys(k for r in records for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in records:
            writer.writerow({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})


def write_json(path,value):
    """Strict finite JSON; never emit NaN as a numerical result."""
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')


def size_key(value):
    """Stable numeric model-scale key; no floating point size merging tolerance."""
    return format(Decimal(str(value)).normalize(),'f')


def load_csv(path,label):
    """N/D in billions, raw positive cross-entropy Loss; no transformations or imputation."""
    columns,records=csv_rows(path)
    if any(c not in columns for c in PRIMARY_COLUMNS):raise ValueError(f'Missing N/D/Loss: {path}')
    values=np.array([[float(r[c]) for c in PRIMARY_COLUMNS] for r in records])
    if not np.isfinite(values).all() or (values<=0).any():
        raise ValueError(f'Nonfinite/nonpositive required N/D/Loss; inspect rather than impute: {path}')
    N,D,L=values.T
    # run_id is checkpoint-specific in B1, but a repeated trajectory ID in B2.
    group=np.array([f'{label}:N={size_key(r["N_params_B"])}' for r in records])
    return dict(path=path,label=label,columns=columns,records=records,N=N,D=D,L=L,group=group,
        group_rule='dataset/source file + exact canonical N_params_B; run_id not assumed to mean model ID')


def input_paths(root):
    """Only B1-B5 plus auxiliary B11/B12; never ingest B6-B10 for modeling."""
    base=root/'real_attachments/B_scaling_laws'
    files={label:base/name for label,name in [('B1','pythia_training_log_existing.csv'),
        ('B2','cerebras_training_log.csv'),('B4','scaling_baseline.csv'),('B5','published_scaling_data.csv')]}
    for path in files.values():
        if not path.is_file():raise FileNotFoundError(path)
    trajectories=sorted((base/'training_trajectories').glob('*.csv'))
    if not trajectories:raise FileNotFoundError('No actual B3 trajectory CSV files')
    return files,trajectories,[base/'open_model_family_metadata.csv',base/'pythia_checkpoint_index.csv']


def combine_trajectories(trajectories):
    """B3 rows keep source file provenance; filenames are trajectory grouping units."""
    keys=['N','D','L','group'];combined={k:[] for k in keys};records=[];source=[]
    for t in trajectories:
        if len(set(t['N']))!=1:raise ValueError('B3 source file contains multiple model sizes')
        for k in ['N','D','L']:combined[k].extend(t[k])
        combined['group'].extend([t['path'].stem]*len(t['N']));records+=t['records'];source.extend([t['path'].name]*len(t['N']))
    return dict(label='B3',records=records,source=source,**{k:np.array(v) for k,v in combined.items()})


def diagnosis(t,root):
    """Fieldwise missing/nonfinite counts and duplicate design/checkpoint distinctions."""
    field_rows=[]
    for c in t['columns']:
        values=[r[c].strip() for r in t['records']];missing=sum(v.lower() in ['', 'nan','null','none','na','n/a'] for v in values)
        numeric=[];nonnumeric=0;inf=0
        for v in values:
            if v.lower() in ['','nan','null','none','na','n/a']:continue
            try:x=float(v)
            except ValueError:nonnumeric+=1;continue
            if np.isinf(x):inf+=1
            elif np.isfinite(x):numeric.append(x)
        field_rows.append(dict(dataset=t['label'],source_file=t['path'].relative_to(root).as_posix(),field=c,
            missing_or_nan=missing,inf_count=inf,nonnumeric_count=nonnumeric,
            finite_numeric_count=len(numeric),min=min(numeric) if numeric else None,max=max(numeric) if numeric else None))
    exact=Counter(tuple(r[c] for c in t['columns']) for r in t['records'])
    checkpoint_field='steps' if 'steps' in t['columns'] else ('step' if 'step' in t['columns'] else None)
    checkpoints=Counter((g,r[checkpoint_field]) for g,r in zip(t['group'],t['records'])) if checkpoint_field else Counter()
    design=Counter((r.get('family',''),size_key(r['N_params_B']),size_key(r['D_tokens_B'])) for r in t['records'])
    info=dict(dataset=t['label'],file_path=t['path'].relative_to(root).as_posix(),rows=len(t['records']),columns=len(t['columns']),
        field_names=t['columns'],missing_or_nan=sum(r['missing_or_nan'] for r in field_rows),inf_count=sum(r['inf_count'] for r in field_rows),
        exact_duplicate_groups=sum(v>1 for v in exact.values()),exact_duplicate_excess_rows=sum(max(v-1,0) for v in exact.values()),
        checkpoint_key_available=checkpoint_field is not None,duplicate_checkpoint_groups=sum(v>1 for v in checkpoints.values()) if checkpoint_field else None,
        duplicate_checkpoint_excess_rows=sum(max(v-1,0) for v in checkpoints.values()) if checkpoint_field else None,
        repeated_family_N_D_design_groups=sum(v>1 for v in design.values()),N_min=float(t['N'].min()),N_max=float(t['N'].max()),
        D_min=float(t['D'].min()),D_max=float(t['D'].max()),Loss_min=float(t['L'].min()),Loss_max=float(t['L'].max()))
    return info,field_rows

# ============================================================
# Source: src/scaling/classical.py
# Original lines: 1-99
# Purpose: 经典五参数非线性拟合、解析Jacobian、误差指标
# ============================================================
"""Positive nonlinear additive scaling laws, balanced objectives and analytic Jacobians."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import pearsonr,spearmanr

PARAMETERS={'M0_N_D':['E','A','B','alpha','beta'], 'M1_N_only':['E','A','alpha'], 'M2_D_only':['E','B','beta']}


def weights(groups,balanced=True,multiplicities=None):
    """Each selected trajectory occurrence has equal total mass; normalize to one.

    Cluster-bootstrap multiplicity represents resampled whole trajectories,
    never randomly selected checkpoint rows.
    """
    groups=np.asarray(groups);unique,count=np.unique(groups,return_counts=True);counts=dict(zip(unique,count))
    if multiplicities is None:multiplicities={g:1 for g in unique}
    w=np.array([multiplicities.get(g,0)/(counts[g] if balanced else 1) for g in groups],dtype=float)
    if not np.isfinite(w).all() or w.sum()<=0:raise ValueError('Invalid trajectory weights')
    return w/w.sum()


def predict(parameters,N,D,model='M0_N_D'):
    """N and D measured in units of 1B. This does NOT log-transform Loss."""
    result=np.full(np.shape(N),parameters['E'],dtype=float)
    if 'A' in PARAMETERS[model]:result+=parameters['A']*np.exp(-parameters['alpha']*np.log(N))
    if 'B' in PARAMETERS[model]:result+=parameters['B']*np.exp(-parameters['beta']*np.log(D))
    if not np.isfinite(result).all() or (result<=0).any():raise FloatingPointError('Invalid predicted Loss')
    return result


def prediction_jacobian(logtheta,lnN,lnD,model):
    """Analytic derivatives with respect to LOG PARAMETERS, not log Loss."""
    keys=PARAMETERS[model];p=dict(zip(keys,np.exp(logtheta)));n=len(lnN)
    pred=np.full(n,p['E']);parts={'E':np.full(n,p['E'])}
    if 'A' in p:
        a=p['A']*np.exp(-p['alpha']*lnN);pred+=a;parts['A']=a;parts['alpha']=-p['alpha']*lnN*a
    if 'B' in p:
        b=p['B']*np.exp(-p['beta']*lnD);pred+=b;parts['B']=b;parts['beta']=-p['beta']*lnD*b
    return pred,np.column_stack([parts[k] for k in keys])


def initial_values(N,D,L,model,settings,rng,count,warm=None):
    """Reproducible dispersed exponent starts and data-scale positive amplitudes."""
    starts=[];keys=PARAMETERS[model];bounds=settings['parameter_bounds']
    if warm is not None:starts.append(np.array([warm[k] for k in keys]))
    exponent_lo,exponent_hi=settings['initial_exponent_range']
    while len(starts)<count:
        a,b=np.exp(rng.uniform(np.log(exponent_lo),np.log(exponent_hi),size=2))
        E=float(rng.uniform(.05,.95)*L.min());excess=max(float(np.median(L))-E,1e-5)
        fraction=float(rng.uniform(.1,.9)) if model=='M0_N_D' else (1.0 if model=='M1_N_only' else 0.0)
        p=dict(E=E,A=excess*fraction*float(np.median(N))**a,B=excess*(1-fraction)*float(np.median(D))**b,alpha=a,beta=b)
        starts.append(np.array([p[k] for k in keys]))
    lo=np.array([bounds[k][0] for k in keys]);hi=np.array([bounds[k][1] for k in keys])
    return [np.log(np.clip(x,lo*1.01,hi*.99)) for x in starts]


def fit(N,D,L,w,model,settings,seed,warm=None,count=None,save_starts=True):
    """Weighted least squares in ORIGINAL Loss with strict positive parameter bounds.

    Independent multi-start solves; only legal successful finite solutions are
    eligible. No validation data appears in this function or its selection.
    """
    keys=PARAMETERS[model];w=np.asarray(w);sel=w>0
    N,D,L,w=np.asarray(N)[sel],np.asarray(D)[sel],np.asarray(L)[sel],w[sel];w/=w.sum()
    lnN=np.log(N);lnD=np.log(D);sw=np.sqrt(w)
    bounds=settings['parameter_bounds'];lo=np.log([bounds[k][0] for k in keys]);hi=np.log([bounds[k][1] for k in keys])
    def fun(x):return sw*(prediction_jacobian(x,lnN,lnD,model)[0]-L)
    def jac(x):return sw[:,None]*prediction_jacobian(x,lnN,lnD,model)[1]
    starts=initial_values(N,D,L,model,settings,np.random.default_rng(seed),count or settings['multistart_count'],warm)
    runs=[];best=None;tol=settings['optimizer_tolerance']
    for number,x0 in enumerate(starts,1):
        solved=least_squares(fun,x0,jac=jac,bounds=(lo,hi),method='trf',x_scale='jac',loss='linear',
            max_nfev=settings['max_nfev'],ftol=tol,xtol=tol,gtol=tol)
        p=dict(zip(keys,map(float,np.exp(solved.x))));objective=float(np.dot(solved.fun,solved.fun))
        legal=bool(solved.success and np.isfinite(objective) and all(np.isfinite(v) and v>0 for v in p.values()))
        if save_starts:
            runs.append(dict(start=number,initial_parameters=dict(zip(keys,map(float,np.exp(x0)))),**p,
                objective=objective,successful=bool(solved.success),legal_solution=legal,status=int(solved.status),
                nfev=int(solved.nfev),optimality=float(solved.optimality),message=solved.message,
                active_bounds=solved.active_mask.tolist()))
        if legal and (best is None or objective<best['objective']):
            singular=np.linalg.svd(solved.jac,compute_uv=False);rank=int(np.linalg.matrix_rank(solved.jac))
            best=dict(model=model,parameters=p,objective=objective,successful=True,status=int(solved.status),
                nfev=int(solved.nfev),optimality=float(solved.optimality),chosen_start=number,
                active_bounds=solved.active_mask.tolist(),jacobian_singular_values=singular.tolist(),jacobian_rank=rank,
                jacobian_condition_number=float(singular[0]/singular[-1]) if singular[-1]>0 else None,
                log_parameters=solved.x.tolist(),n_positive_weight_rows=len(L),n_starts=len(starts))
    if best is None:raise RuntimeError('All nonlinear starts failed; do not substitute invented parameters')
    return best,runs


def metrics(y,p):
    """Observed-minus-predicted bias; constant ranks and zero-variance R2 undefined."""
    y=np.asarray(y);p=np.asarray(p);e=y-p;defined=len(y)>1 and np.ptp(y)>0 and np.ptp(p)>0
    return dict(n=len(y),RMSE=float(np.sqrt(np.mean(e**2))),MAE=float(np.mean(abs(e))),
        R2=float(1-np.sum(e**2)/np.sum((y-y.mean())**2)) if len(y)>1 and np.ptp(y)>0 else None,
        Pearson=float(pearsonr(y,p).statistic) if defined else None,Spearman=float(spearmanr(y,p).statistic) if defined else None,
        mean_bias=float(e.mean()),median_residual=float(np.median(e)),residual_std=float(e.std(ddof=1)) if len(y)>1 else None)

# ============================================================
# Source: src/scaling/bootstrap.py
# Original lines: 1-34
# Purpose: 整条trajectory重采样拟合
# ============================================================
"""Whole-trajectory resampling; bounded CPU workers and no IID checkpoints."""
from __future__ import annotations
from collections import Counter
import numpy as np
from .classical import weights,fit,predict,metrics

_DATA=None


def initialize(data):
    """Workers receive ONLY B1 arrays, its primary parameters and fixed fit settings."""
    global _DATA
    _DATA=data


def repetition(task):
    """Fit each resampled trajectory multiplicity with 32 positive multi-starts."""
    number,drawn,seed=task;d=_DATA;groups=d['group'];counts=Counter(drawn)
    selected=[g for g in np.unique(groups) if counts[g]>0]
    prefix=dict(repetition=number,drawn_trajectories=list(drawn),multiplicities=dict(counts),
        distinct_trajectories=len(selected),seed=seed,fit_dataset='B1_only',bootstrap_unit='trajectory')
    try:
        w=weights(groups,True,counts)
        best,_=fit(d['N'],d['D'],d['L'],w,'M0_N_D',d['settings'],seed,
            warm=d['primary_parameters'],count=d['settings']['bootstrap_multistart_count'],save_starts=False)
        p=predict(best['parameters'],d['N'],d['D'])
        identified=best['jacobian_rank']==5 and len(selected)>=3
        result=dict(**prefix,**best['parameters'],objective=best['objective'],converged=True,
            identifiable=identified,eligible_parameter_CI=identified,status=best['status'],jacobian_rank=best['jacobian_rank'],
            condition_number=best['jacobian_condition_number'],active_bounds=best['active_bounds'],
            n_starts=best['n_starts'],prediction_RMSE_on_full_B1=metrics(d['L'],p)['RMSE'],error=None)
        return result,p
    except Exception as exc:
        return dict(**prefix,converged=False,identifiable=False,eligible_parameter_CI=False,error=repr(exc)),None

# ============================================================
# Source: scripts/10_diagnose_scaling_data.py
# Original lines: 1-159
# Purpose: 最终阶段实际执行入口：10_diagnose_scaling_data.py
# ============================================================
"""Question 2 phase 1 first step: B1-B5 diagnosis only, no fitting."""
from __future__ import annotations
import logging
from pathlib import Path
import sys
from collections import Counter
from decimal import Decimal
import json
import re
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scaling.io import input_paths,load_csv,diagnosis,csv_rows,write_csv,write_json,sha256,size_key
OUT=ROOT/'outputs/scaling/phase1'


def protected_files():
    """Freeze every preexisting project file, including original attachments and Q1.

    Only the specifically new scaling paths are excluded. Original B6-B10 bytes
    may be hashed for integrity, but their CSV values are never ingested.
    """
    manifest=OUT/'frozen_project_before.json'
    if manifest.exists():return json.loads(manifest.read_text(encoding='utf-8'))
    paths=[]
    excluded=[ROOT/'outputs/scaling',ROOT/'outputs/figures/question2_phase1',ROOT/'src/scaling']
    for p in ROOT.rglob('*'):
        if not p.is_file() or '.git' in p.parts or any(p.is_relative_to(e) for e in excluded):continue
        if p.name=='scaling_phase1.yaml' or p.name.startswith('10_diagnose_scaling') or p.name.startswith('11_'):continue
        if p==ROOT/'outputs/report/question2_phase1_classical_scaling.md':continue
        paths.append(p)
    result={p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(paths)}
    write_json(manifest,result)
    return result


def verify_protected(before):
    """All frozen files stay byte-identical; fail if a file is missing or changed."""
    rows=[]
    for p,h in before.items():
        actual=sha256(ROOT/p) if (ROOT/p).exists() else 'MISSING'
        rows.append(dict(path=p,before_sha256=h,after_sha256=actual,unchanged=h==actual))
        if actual!=h:raise RuntimeError(f'Frozen project file changed: {p}')
    write_csv(OUT/'frozen_project_integrity.csv',rows)
    return len(rows)


def md_table(rows,columns):
    """Readable diagnosis tables with missing applicability distinguished from zero."""
    def value(x):
        if x is None:return '不适用/未提供'
        if isinstance(x,float):return f'{x:.6g}'
        return str(x).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join('---' for c in columns)+' |']+[
        '| '+' | '.join(value(r.get(c)) for c in columns)+' |' for r in rows])


def main():
    """Inspect actual tables and write provenance/grouping before any nonlinear fit."""
    OUT.mkdir(parents=True,exist_ok=True)
    (ROOT/'outputs/figures/question2_phase1').mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'diagnosis.log',encoding='utf-8')])
    frozen=protected_files();verify_protected(frozen)
    files,trajectory_paths,aux=input_paths(ROOT)
    tables={label:load_csv(path,label) for label,path in files.items()}
    trajectories=[load_csv(path,'B3') for path in trajectory_paths]
    summaries=[];fields=[];manifest=[]
    for t in list(tables.values())+trajectories:
        s,f=diagnosis(t,ROOT);summaries.append(s);fields+=f
        manifest.append(dict(dataset=t['label'],path=t['path'].relative_to(ROOT).as_posix(),sha256=sha256(t['path']),
            usage='primary_fit' if t['label']=='B1' else 'validation_only'))
    auxiliary=[]
    for path in aux:
        columns,records=csv_rows(path)
        auxiliary.append(dict(path=path.relative_to(ROOT).as_posix(),rows=len(records),columns=len(columns),field_names=columns))
        manifest.append(dict(dataset='B11' if 'metadata' in path.name else 'B12',path=path.relative_to(ROOT).as_posix(),sha256=sha256(path),usage='grouping_provenance_only'))
    _,checkpoint_rows=csv_rows(aux[1]);_,family_metadata=csv_rows(aux[0])
    nominal={}
    for r in checkpoint_rows:
        label=r['model_size'];match=re.fullmatch(r'([\d.]+)([mb])',label.lower())
        if not match:raise ValueError(f'Unexpected B12 nominal model_size {label}')
        nominal[label]=float(match[1])/(1000 if match[2]=='m' else 1)
    b1=tables['B1'];group_rows=[];membership=[];flops=[];trend_events=[]
    run_counts=Counter(r['run_id'] for r in b1['records'])
    # Quantization bounds from actual exported column decimal precision.
    half={c:.5*10**(-max(max(-Decimal(r[c]).as_tuple().exponent,0) for r in b1['records'])) for c in ['N_params_B','D_tokens_B','C_FLOPs_1e21']}
    for group in np.unique(b1['group']):
        sel=np.where(b1['group']==group)[0];steps=np.array([float(b1['records'][i]['steps']) for i in sel]);order=np.argsort(steps);sel=sel[order];steps=steps[order]
        N,D,L=b1['N'][sel],b1['D'][sel],b1['L'][sel]
        if len(set(N))!=1:raise ValueError('N varies within inferred B1 trajectory')
        size=min(nominal,key=lambda key:abs(np.log(nominal[key]/N[0])))
        actual_steps=set(map(int,steps));b12_steps={int(r['step']) for r in checkpoint_rows if r['model_size']==size}
        metadata_repos={r['model_repo'] for r in family_metadata};repos={r['model_repo'] for r in checkpoint_rows if r['model_size']==size}
        info=dict(trajectory_id=group,N_params_B=float(N[0]),n_checkpoints=len(sel),N_fixed=True,
            unique_checkpoint_count=len(set(steps)),D_strictly_increasing=bool(np.all(np.diff(D)>0)),
            D_min=float(D.min()),D_max=float(D.max()),first_loss=float(L[0]),last_loss=float(L[-1]),
            loss_decrease_first_to_last=float(L[0]-L[-1]),loss_rebound_count=int(np.sum(np.diff(L)>0)),
            maximum_loss_rebound=float(max(0,np.diff(L).max())),matched_B12_nominal_size=size,
            nominal_match_rule='nearest log-scale B12 name; grouping is exact B1 N, not nominal value',
            B12_matching_checkpoint_steps=len(actual_steps&b12_steps),B12_unmatched_B1_steps=sorted(actual_steps-b12_steps),
            B11_repo_metadata_present=bool(repos&metadata_repos))
        group_rows.append(info)
        for position,i in enumerate(sel):
            r=b1['records'][i]
            membership.append(dict(source_row=i+1,run_id=r['run_id'],trajectory_id=group,N_params_B=float(N[position]),
                steps=int(steps[position]),D_tokens_B=float(D[position]),val_loss=float(L[position]),
                training_progress_steps=float(steps[position]/steps[-1]),checkpoint_weight_raw=1/len(sel),
                group_rule='B1 exact canonical N; run_id is unique checkpoint identifier'))
            observed=float(r['C_FLOPs_1e21']);expected=.006*N[position]*D[position]
            tolerance=half['C_FLOPs_1e21']+.006*(abs(D[position])*half['N_params_B']+abs(N[position])*half['D_tokens_B']+half['N_params_B']*half['D_tokens_B'])
            flops.append(dict(source_row=i+1,trajectory_id=group,C_reported_1e21=observed,C_6ND_1e21=float(expected),
                absolute_difference=float(abs(observed-expected)),relative_difference=float(abs(observed-expected)/expected),
                export_rounding_absolute_tolerance=float(tolerance),within_export_rounding=bool(abs(observed-expected)<=tolerance+1e-14)))
            if position and (D[position]<=D[position-1] or L[position]>L[position-1]):
                trend_events.append(dict(source_row=i+1,trajectory_id=group,D_change=float(D[position]-D[position-1]),Loss_change=float(L[position]-L[position-1])))
    # Never deduplicate or repair original rows: stop before dependent fitting if ambiguous.
    b1summary=next(r for r in summaries if r['dataset']=='B1')
    if b1summary['duplicate_checkpoint_groups'] or b1summary['exact_duplicate_groups']:
        raise ValueError('B1 duplicate checkpoints/rows require explicit review; no automatic deletion')
    if not all(r['D_strictly_increasing'] for r in group_rows):raise ValueError('B1 D trajectory not strictly increasing')
    write_csv(OUT/'data_file_summary.csv',summaries);write_csv(OUT/'field_diagnosis.csv',fields)
    write_csv(OUT/'auxiliary_schema_summary.csv',auxiliary);write_csv(OUT/'input_manifest.csv',manifest)
    write_csv(OUT/'B1_trajectory_summary.csv',group_rows);write_csv(OUT/'B1_trajectory_membership.csv',membership)
    write_csv(OUT/'B1_flops_consistency.csv',flops);write_csv(OUT/'B1_trajectory_anomalies.csv',trend_events,['source_row','trajectory_id','D_change','Loss_change'])
    _,literature=csv_rows(files['B5'])
    literature_sources=[dict(source=s,n=sum(r['source']==s for r in literature),families=sorted({r['family'] for r in literature if r['source']==s}),
        evaluation_dataset_provided=False,tokenizer_provided=False,loss_type_provided=False,
        absolute_loss_comparability='unconfirmed; source-layer trend primary, pooled errors diagnostic only') for s in sorted({r['source'] for r in literature})]
    write_csv(OUT/'B5_source_diagnosis.csv',literature_sources)
    state=dict(B1_trajectory_count=len(group_rows),B1_records=len(b1['N']),N_range=[float(b1['N'].min()),float(b1['N'].max())],
        D_range=[float(b1['D'].min()),float(b1['D'].max())],Loss_range=[float(b1['L'].min()),float(b1['L'].max())],
        B1_unique_run_ids=len(run_counts),B1_run_id_is_trajectory=False,group_rule=b1['group_rule'],
        terminal_D_values=sorted(set(r['D_max'] for r in group_rows)),B1_loss_rebounds=sum(r['loss_rebound_count'] for r in group_rows),
        B1_C_6ND_rounding_consistent_count=sum(r['within_export_rounding'] for r in flops),
        frozen_file_count=len(frozen),B5_absolute_loss_comparability_confirmed=False,
        data_sources_used_for_fitting=['B1'],disallowed_B6_B10_ingested=False)
    write_json(OUT/'data_diagnosis.json',state)
    lines=['# 问题二 PHASE 1 数据诊断（尚未拟合）','',
        'N_params_B、D_tokens_B 直接按 1B 单位输入模型，不乘 1e9。FLOPs 核验采用 6ND/1e21=0.006 N_B D_B。原始数据不填补、不删除、不覆盖。','',
        md_table(summaries,['dataset','file_path','rows','columns','missing_or_nan','inf_count','exact_duplicate_excess_rows','duplicate_checkpoint_excess_rows','N_min','N_max','D_min','D_max','Loss_min','Loss_max']),'',
        '字段完整列表见 data_file_summary.csv；逐字段 NaN/空缺与 Inf 见 field_diagnosis.csv。B2 的可选元数据空缺不等于 N/D/Loss 缺失，未做任何填补。无 checkpoint 字段的 B4/B5 记“不适用”，不虚报重复 checkpoint 为零。','',
        f"B1 实际 {len(group_rows)} 个规模、{len(b1['N'])} 条 checkpoint。run_id 有 {len(run_counts)} 个唯一值，是 checkpoint 标识，不是模型轨迹 ID。轨迹按来源 B1 + 精确规范化 N_params_B 分组；本文件每个规模只有一条候选轨迹，无法识别未标注的同规模重复运行。",'',
        md_table(group_rows,['trajectory_id','N_params_B','n_checkpoints','D_strictly_increasing','loss_rebound_count','D_max','matched_B12_nominal_size','B12_matching_checkpoint_steps']),'',
        'B12 的名义规模用于辅助对照，参数计数不一定等于命名规模；多出的名义模型不补造 B1 数据。run_id 不同不能使同一轨迹 checkpoint 成为独立观测。Group CV 留出整条轨迹，bootstrap 以轨迹整体重采样。','',
        f"B1 的 C 与 6ND：{state['B1_C_6ND_rounding_consistent_count']}/{len(flops)} 条在实际 CSV 导出精度容差内一致。相对误差在极小 C 处可能因四舍五入很大，不能只看相对值判异常。",'',
        'B3 各文件含 interpolated 字段，只称插值轨迹一致性检验；B2 按本轮给定的半合成来源定义，仅作辅助敏感性。B4 跨模型族表未提供完整评测数据集/tokenizer 口径，数值误差只反映给定表上的一致性。','',
        md_table(literature_sources,['source','n','families','absolute_loss_comparability']),'',
        'B5 有文献 source 和 family，没有 loss_type、dataset、tokenizer。跨文献绝对 Loss 可比性无法由附件确认，所以 source 内排序是核心文献证据，绝对误差只作分层描述，不当作统一口径精度。','',
        f"末端 D 的实际取值：{state['terminal_D_values']}。若所有模型的 terminal D 相同，则 E 与 B D_terminal^-beta 混淆，终点数据不能分别识别 E/B/beta；仍输出约束拟合对照，但不将 beta 的数值当作终点实证结论。",'',
        f"{len(frozen)} 个既有文件的哈希已建立并核验。只读取 B1-B5、B11/B12 作为数值/元数据输入，B6-B10 未用于拟合。",'']
    (OUT/'data_diagnosis.md').write_text('\n\n'.join(lines),encoding='utf-8')
    verify_protected(frozen)
    logging.info('DIAGNOSIS COMPLETE: B1 %d trajectories / %d checkpoints; no model fitted',len(group_rows),len(b1['N']))
    print(json.dumps(state,ensure_ascii=True))


if __name__=='__main__':main()

# ============================================================
# Source: scripts/11_fit_classical_scaling.py
# Original lines: 1-475
# Purpose: 最终阶段实际执行入口：11_fit_classical_scaling.py
# ============================================================
"""Classical scaling law: B1-only multistart fits, grouped CV and cluster bootstrap.

B2/B3/B4/B5 are never fitted or used for selection. Frozen question 1 is never
imported. N and D remain in billions. No Q, K, mixtures, or effective tokens.
"""
from __future__ import annotations
import csv
import json
import logging
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import time
import numpy as np
import yaml
from scipy.stats import pearsonr,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FormatStrFormatter

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from scaling.io import input_paths,load_csv,combine_trajectories,write_csv,write_json,sha256,csv_rows
from scaling.classical import fit,weights,predict,metrics,PARAMETERS
from scaling.bootstrap import initialize,repetition
from importlib import import_module
diagnostic=import_module('10_diagnose_scaling_data')
OUT=ROOT/'outputs/scaling/phase1';FIG=ROOT/'outputs/figures/question2_phase1'


def read_rows(path):
    """Read derived result rows without changing any training data."""
    return csv_rows(path)[1]


def selected_row(best,weighting,n_trajectories):
    """Export objective=sum normalized weight*(raw Loss residual)^2, no half factor."""
    return dict(model=best['model'],weighting=weighting,**best['parameters'],objective=best['objective'],
        convergence_status=best['status'],converged=best['successful'],n_starts=best['n_starts'],
        n_trajectories=n_trajectories,n_checkpoints=best['n_positive_weight_rows'],chosen_start=best['chosen_start'],
        jacobian_rank=best['jacobian_rank'],jacobian_condition_number=best['jacobian_condition_number'],
        units='N_B,D_B',fit_dataset='B1_only')


def fit_and_cv(b1,settings):
    """32 starts per fit; leave out every checkpoint of one exact-size trajectory."""
    N,D,L,G=[b1[k] for k in ['N','D','L','group']];unique=np.unique(G);models={};starts=[];cv=[];predictions=[];cv_models=[]
    for model in PARAMETERS:
        balanced=weights(G)
        best,runs=fit(N,D,L,balanced,model,settings,settings['random_state'])
        models[model]=best
        starts += [dict(context='full_B1_balanced',model=model,**r) for r in runs]
        logging.info('%s full B1 objective %.9g params=%s',model,best['objective'],best['parameters'])
        for fold,group in enumerate(unique,1):
            va=G==group;tr=~va
            # Even initialization is computed exclusively on the current training
            # trajectories; do not warm-start CV from the all-B1 fitted model.
            fold_best,runs=fit(N[tr],D[tr],L[tr],weights(G[tr]),model,settings,settings['random_state']+fold)
            pred=predict(fold_best['parameters'],N[va],D[va],model)
            cv.append(dict(model=model,fold=fold,heldout_trajectory=group,N_heldout=float(N[va][0]),
                n_training_trajectories=len(unique)-1,n_training_rows=int(tr.sum()),
                **metrics(L[va],pred),fit_dataset='other_B1_trajectories_only',**fold_best['parameters']))
            cv_models.append(dict(model=model,fold=fold,heldout_trajectory=group,
                training_trajectories=unique[unique!=group].tolist(),training_source_rows=(np.where(tr)[0]+1).tolist(),
                heldout_source_rows=(np.where(va)[0]+1).tolist(),fit=fold_best))
            predictions += [dict(model=model,fold=fold,source_row=int(i+1),trajectory_id=group,
                N_params_B=float(N[i]),D_tokens_B=float(D[i]),true_loss=float(L[i]),predicted_loss=float(p)) for i,p in zip(np.where(va)[0],pred)]
            starts += [dict(context='group_cv',model=model,heldout_trajectory=group,**r) for r in runs]
            logging.info('%s leave-trajectory-out %d/%d RMSE %.8g',model,fold,len(unique),cv[-1]['RMSE'])
    # Same raw record weights because all B1 trajectory checkpoint counts are equal.
    unweighted,runs=fit(N,D,L,weights(G,False),'M0_N_D',settings,settings['random_state'])
    starts += [dict(context='full_B1_unweighted',model='M0_N_D',**r) for r in runs]
    write_json(OUT/'main_model.json',models['M0_N_D']);write_json(OUT/'baseline_models.json',models)
    write_json(OUT/'group_cv_models.json',cv_models);write_json(OUT/'unweighted_model.json',unweighted)
    write_csv(OUT/'multistart_results.csv',starts)
    write_csv(OUT/'classical_scaling_parameters.csv',[selected_row(models['M0_N_D'],'trajectory_balanced',len(unique)),selected_row(unweighted,'unweighted',len(unique))])
    write_csv(OUT/'B1_group_cv.csv',cv);write_csv(OUT/'B1_group_cv_predictions.csv',predictions)
    comparison=[]
    for model in PARAMETERS:
        actual=sorted([r for r in predictions if r['model']==model],key=lambda r:r['source_row'])
        p=np.array([r['predicted_loss'] for r in actual])
        comparison.append(dict(model=model,validation='leave_one_trajectory_out',folds=len(unique),**metrics(L,p)))
    write_csv(OUT/'baseline_model_comparison.csv',comparison)
    sensitivity=[]
    for label,best in [('trajectory_balanced',models['M0_N_D']),('unweighted',unweighted)]:
        pred=predict(best['parameters'],N,D)
        sensitivity.append(dict(weighting=label,**best['parameters'],objective=best['objective'],**metrics(L,pred),
            max_abs_prediction_difference_from_balanced=float(abs(pred-predict(models['M0_N_D']['parameters'],N,D)).max()),
            trajectory_checkpoint_counts=dict(Counter(G))))
    write_csv(OUT/'weighting_sensitivity.csv',sensitivity)
    return models,starts,cv,comparison,unweighted


def cluster_bootstrap(b1,primary,settings):
    """2,000 whole-trajectory draws, each with >=30 starts; resumable checkpoints."""
    rng=np.random.default_rng(settings['random_state']);groups=np.unique(b1['group']);n=settings['bootstrap_repetitions']
    tasks=[(i+1,rng.choice(groups,size=len(groups),replace=True).tolist(),settings['random_state']+10000+i) for i in range(n)]
    draw_rows=[dict(repetition=i,drawn_trajectories=draws,seed=seed) for i,draws,seed in tasks]
    write_csv(OUT/'bootstrap_cluster_draws.csv',draw_rows)
    # Resume only under unchanged inputs/config/model/draw plan.
    signature={k:sha256(OUT/k) for k in ['main_model.json','bootstrap_cluster_draws.csv','input_manifest.csv']}
    signature['config']=sha256(ROOT/'configs/scaling_phase1.yaml')
    checkpoint=OUT/'bootstrap_checkpoint.json';rows=[];pred=np.full((n,len(b1['L'])),np.nan,dtype=float)
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text(encoding='utf-8'))
        if saved['signature']!=signature:raise RuntimeError('Bootstrap input/config changed; cannot silently resume')
        rows=saved['rows']
        with (OUT/'bootstrap_predictions_checkpoint.npy').open('rb') as f:loaded=np.load(f)
        if loaded.shape!=pred.shape:raise RuntimeError('Bootstrap checkpoint prediction shape mismatch')
        pred[:len(rows)]=loaded[:len(rows)]
        logging.info('Bootstrap resume %d/%d',len(rows),n)
    data={k:b1[k] for k in ['N','D','L','group']};data.update(settings=settings,primary_parameters=primary['parameters'])
    if len(rows)<n:
        started=time.monotonic()
        with ProcessPoolExecutor(max_workers=settings['bootstrap_workers'],initializer=initialize,initargs=(data,)) as pool:
            for row,p in pool.map(repetition,tasks[len(rows):],chunksize=5):
                i=len(rows)
                if p is not None:pred[i]=p
                rows.append(row)
                if len(rows)%settings['bootstrap_checkpoint_interval']==0 or len(rows)==n:
                    with (OUT/'bootstrap_predictions_checkpoint.npy').open('wb') as f:np.save(f,pred)
                    write_json(checkpoint,dict(signature=signature,rows=rows))
                    logging.info('Cluster bootstrap %d/%d, failures=%d, elapsed=%.1fs',len(rows),n,sum(not r['converged'] for r in rows),time.monotonic()-started)
    write_csv(OUT/'parameter_bootstrap_draws.csv',rows)
    valid=[r for r in rows if r['eligible_parameter_CI']];failed=sum(not r['converged'] for r in rows);unidentified=sum(r['converged'] and not r['identifiable'] for r in rows)
    if not valid:raise RuntimeError('No identifiable bootstrap samples; parameter uncertainty cannot be estimated')
    statistics=[]
    for key in PARAMETERS['M0_N_D']:
        x=np.array([r[key] for r in valid])
        statistics.append(dict(parameter=key,main_estimate=primary['parameters'][key],median=float(np.median(x)),mean=float(x.mean()),std=float(x.std(ddof=1)),
            CI95_low=float(np.quantile(x,.025)),CI95_high=float(np.quantile(x,.975)),min=float(x.min()),max=float(x.max()),
            P01=float(np.quantile(x,.01)),P99=float(np.quantile(x,.99)),successful_identifiable_repetitions=len(valid),
            attempted_repetitions=n,solver_failure_count=failed,solver_failure_rate=failed/n,
            converged_nonidentifiable_count=unidentified,eligible_CI_exclusion_rate=(failed+unidentified)/n,
            bootstrap_unit='whole_trajectory',random_state=settings['random_state'],n_multistart_per_fit=settings['bootstrap_multistart_count']))
    write_csv(OUT/'parameter_bootstrap.csv',statistics)
    values=np.array([[r[k] for k in PARAMETERS['M0_N_D']] for r in valid]);correlation=np.corrcoef(values,rowvar=False)
    write_csv(OUT/'parameter_correlation.csv',[dict(parameter=key,**dict(zip(PARAMETERS['M0_N_D'],row))) for key,row in zip(PARAMETERS['M0_N_D'],correlation)])
    indices=[i for i,r in enumerate(rows) if r['eligible_parameter_CI']];vp=pred[indices]
    prediction_ci=np.quantile(vp,[.025,.5,.975],axis=0)
    write_csv(OUT/'bootstrap_B1_prediction_intervals.csv',[dict(source_row=i+1,trajectory_id=b1['group'][i],
        N_params_B=float(b1['N'][i]),D_tokens_B=float(b1['D'][i]),pred_CI95_low=float(prediction_ci[0,i]),pred_median=float(prediction_ci[1,i]),
        pred_CI95_high=float(prediction_ci[2,i]),CI_width=float(prediction_ci[2,i]-prediction_ci[0,i]),
        interval_kind='fitted_mean_uncertainty_not_future_loss_prediction_interval') for i in range(len(b1['L']))])
    result=dict(attempted=n,converged=n-failed,identifiable=len(valid),solver_failure_rate=failed/n,
        converged_nonidentifiable=unidentified,parameter_correlation=correlation.tolist(),
        parameter_CI_basis='converged_full_Jacobian_rank_5_whole_trajectory_resamples',
        prediction_mean_CI_width=float((prediction_ci[2]-prediction_ci[0]).mean()),
        prediction_max_CI_width=float((prediction_ci[2]-prediction_ci[0]).max()))
    write_json(OUT/'bootstrap_summary.json',result)
    return rows,statistics,result


def terminal_fit(b1,primary,settings):
    """Only final D per model; report rank deficiency and invariant intercept."""
    select=np.array([np.where(b1['group']==g)[0][np.argmax(b1['D'][b1['group']==g])] for g in np.unique(b1['group'])])
    N,D,L=b1['N'][select],b1['D'][select],b1['L'][select]
    best,runs=fit(N,D,L,weights(b1['group'][select]),'M0_N_D',settings,settings['random_state']+500,warm=primary['parameters'])
    write_json(OUT/'terminal_only_model.json',best)
    write_csv(OUT/'terminal_only_multistart.csv',runs)
    rows=[]
    for label,fit_result in [('full_checkpoint_primary',primary),('terminal_only_sensitivity',best)]:
        p=fit_result['parameters'];terminal=predict(p,N,D);full=predict(p,b1['N'],b1['D'])
        rows.append(dict(dataset=label,n_fit=fit_result['n_positive_weight_rows'],**p,objective=fit_result['objective'],
            jacobian_rank=fit_result['jacobian_rank'],parameters_identifiable=fit_result['jacobian_rank']==5,
            terminal_D_unique_count=len(set(D)),E_effective_at_terminal_D=float(p['E']+p['B']*D[0]**(-p['beta'])),
            terminal_RMSE=metrics(L,terminal)['RMSE'],terminal_Pearson=metrics(L,terminal)['Pearson'],
            full_B1_RMSE_descriptive=metrics(b1['L'],full)['RMSE'],
            N_trend='decreasing_by_positive_constraints',D_trend='decreasing_by_positive_constraints',
            D_direction_empirically_testable=len(set(D))>1))
    write_csv(OUT/'terminal_checkpoint_sensitivity.csv',rows)
    return best,rows


def validate(t,parameters,label,group_keys,provenance,absolute_comparability):
    """Direct predictions of frozen B1 parameters; source-layer errors remain descriptive."""
    pred=predict(parameters,t['N'],t['D']);rows=[]
    rows.append(dict(dataset=label,stratum='overall',group_key='all',provenance=provenance,
        absolute_loss_comparability=absolute_comparability,absolute_error_role='descriptive_only' if not absolute_comparability else 'file_scale_direct_validation',
        no_calibration=True,**metrics(t['L'],pred)))
    for group_name,groups in group_keys.items():
        for group in sorted(set(groups)):
            sel=np.array(groups)==group
            rows.append(dict(dataset=label,stratum=group_name,group_key=group,provenance=provenance,
                absolute_loss_comparability=absolute_comparability,absolute_error_role='descriptive_only' if not absolute_comparability else 'file_scale_direct_validation',
                no_calibration=True,**metrics(t['L'][sel],pred[sel])))
    write_csv(OUT/f'{label}_validation.csv' if label in ['B2','B3'] else OUT/('B4_cross_family_validation.csv' if label=='B4' else 'B5_literature_validation.csv'),rows)
    write_csv(OUT/f'{label}_direct_predictions.csv',[dict(source_row=i+1,dataset=label,
        source=t.get('source',[r.get('source','') for r in t['records']])[i],family=t['records'][i].get('family',''),
        trajectory_id=t['group'][i],N_params_B=float(t['N'][i]),D_tokens_B=float(t['D'][i]),true_loss=float(t['L'][i]),
        predicted_loss=float(pred[i]),residual=float(t['L'][i]-pred[i]),provenance=provenance) for i in range(len(pred))])
    return rows,pred


def residual_diagnostics(b1,primary):
    """Group/early-vs-late structure and serial correlations without IID p-values."""
    p=predict(primary['parameters'],b1['N'],b1['D']);e=b1['L']-p
    progress=np.array([float(r['steps']) for r in b1['records']]);progress/=max(progress)
    residual_rows=[dict(source_row=i+1,trajectory_id=b1['group'][i],N_params_B=float(b1['N'][i]),D_tokens_B=float(b1['D'][i]),
        steps=float(b1['records'][i]['steps']),training_progress=float(progress[i]),true_loss=float(b1['L'][i]),predicted_loss=float(p[i]),residual=float(e[i])) for i in range(len(e))]
    write_csv(OUT/'B1_residuals.csv',residual_rows)
    associations=[]
    for label,x in [('log_N',np.log(b1['N'])),('log_D',np.log(b1['D'])),('predicted_Loss',p),('training_progress',progress)]:
        associations.append(dict(variable=label,Pearson=float(pearsonr(e,x).statistic),Spearman=float(spearmanr(e,x).statistic),
            n_checkpoints=len(e),independent_trajectory_count=len(set(b1['group'])),IID_p_value_reported=False))
    write_csv(OUT/'residual_associations.csv',associations)
    summaries=[]
    for g in np.unique(b1['group']):
        sel=np.where(b1['group']==g)[0];sel=sel[np.argsort(b1['D'][sel])];res=e[sel]
        summaries.append(dict(stratum='trajectory',group=g,**metrics(b1['L'][sel],p[sel]),
            lag1_residual_Pearson=float(pearsonr(res[:-1],res[1:]).statistic)))
    # The quantile-derived progress strata are diagnostics, not deleted training rows.
    lo,hi=np.quantile(progress,[.1,.9])
    for label,sel in [('early_P10',progress<=lo),('middle_P10_P90',(progress>lo)&(progress<hi)),('late_P90',progress>=hi)]:
        summaries.append(dict(stratum='training_progress',group=label,**metrics(b1['L'][sel],p[sel]),
            lag1_residual_Pearson=None))
    write_csv(OUT/'residual_structure_summary.csv',summaries)
    return residual_rows,associations,summaries


def plot_all(b1,models,cv_predictions,bootstrap_rows,tables,trajectories,validations,residuals,settings):
    """Paper-style standalone plots; no interpolation plot treated as independent evidence."""
    FIG.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    manifest=[]
    def save(fig,name):
        fig.tight_layout();path=FIG/name;fig.savefig(path,dpi=settings['figure_dpi'],bbox_inches='tight');plt.close(fig)
        manifest.append(dict(path=path.relative_to(ROOT).as_posix()))
    pred=predict(models['M0_N_D']['parameters'],b1['N'],b1['D']);res=b1['L']-pred
    fig,ax=plt.subplots(figsize=(8,5));ax.scatter(b1['N'],b1['D'],s=8,alpha=.5)
    ax.set(xscale='log',yscale='log',xlabel='N (billions of parameters)',ylabel='D (billions of trained tokens)',title='B1 coverage: eight model trajectories');save(fig,'01_B1_N_D_coverage.png')
    groups=np.unique(b1['group']);fig,axes=plt.subplots(2,4,figsize=(15,7))
    for ax,g in zip(axes.flat,groups):
        sel=np.where(b1['group']==g)[0];sel=sel[np.argsort(b1['D'][sel])]
        ax.plot(b1['D'][sel],b1['L'][sel],'.',ms=3,label='Provided B1 Loss');ax.plot(b1['D'][sel],pred[sel],'-',lw=1.3,label='Frozen classical fit')
        ax.set(xscale='log',title=f"N={b1['N'][sel[0]]:.6g}B",xlabel='D (B tokens)',ylabel='Validation Loss')
    axes.flat[0].legend(fontsize=8);save(fig,'02_B1_Loss_D_trajectories.png')
    def observed_plot(y,p,title,name,groups=None):
        fig,ax=plt.subplots(figsize=(7,5))
        if groups is None:ax.scatter(y,p,s=10,alpha=.6)
        else:
            for group in sorted(set(groups)):
                sel=np.array(groups)==group;ax.scatter(y[sel],p[sel],s=25,alpha=.8,label=group)
            ax.legend(fontsize=7,ncol=2)
        lo,hi=min(y.min(),p.min()),max(y.max(),p.max());ax.plot([lo,hi],[lo,hi],'--',color='gray',lw=1)
        ax.set(xlabel='Provided validation Loss',ylabel='B1 model predicted Loss',title=title);save(fig,name)
    observed_plot(b1['L'],pred,'B1 fitted checkpoints (not IID)','03_B1_observed_predicted.png')
    ordered=sorted([r for r in cv_predictions if r['model']=='M0_N_D'],key=lambda r:int(r['source_row']));oof=np.array([float(r['predicted_loss']) for r in ordered])
    observed_plot(b1['L'],oof,'B1 leave-one-trajectory-out predictions','04_group_CV_observed_predicted.png')
    for name,x,title in [('05',np.log(b1['N']),'log N (N measured in B)'),('06',np.log(b1['D']),'log D (D measured in B)'),('07',pred,'Predicted Loss'),('08',np.array([r['training_progress'] for r in residuals]),'Training progress (step / final step)')]:
        fig,ax=plt.subplots(figsize=(8,5));ax.scatter(x,res,s=9,alpha=.5);ax.axhline(0,color='gray',lw=.8)
        ax.set(xlabel=title,ylabel='Residual (provided - predicted)',title='B1 residual structure; no IID significance test');save(fig,f'{name}_residual_structure.png')
    fig,ax=plt.subplots(figsize=(7,4));ax.hist(res,bins=40);ax.set(xlabel='Residual (provided - predicted)',ylabel='Checkpoint count',title='B1 residual histogram');save(fig,'09_residual_histogram.png')
    fig,axes=plt.subplots(2,3,figsize=(13,7));valid=[r for r in bootstrap_rows if r['eligible_parameter_CI']]
    for ax,key in zip(axes.flat,PARAMETERS['M0_N_D']):
        ax.hist([r[key] for r in valid],bins=35);ax.axvline(models['M0_N_D']['parameters'][key],color='gray',linestyle='--');ax.set(xlabel=key,ylabel='Whole-trajectory bootstrap draws')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))
        ax.tick_params(axis='x',labelsize=8)
    axes.flat[-1].axis('off');fig.suptitle('B1 cluster bootstrap parameter distributions (conditional on this table)');save(fig,'10_parameter_bootstrap.png')
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for ax,t in zip(axes.flat,trajectories):
        pp=predict(models['M0_N_D']['parameters'],t['N'],t['D']);order=np.argsort(t['D'])
        ax.plot(t['D'][order],t['L'][order],'.',ms=2,label='B3 interpolated table');ax.plot(t['D'][order],pp[order],'-',lw=1.3,label='B1 frozen model')
        ax.set(xscale='log',title=f"N={t['N'][0]:.6g}B",xlabel='D (B tokens)',ylabel='Validation Loss')
    axes.flat[0].legend(fontsize=8);save(fig,'11_B3_interpolated_trajectory_validation.png')
    observed_plot(tables['B2']['L'],validations['B2'][1],'B2 semi-synthetic auxiliary consistency','12_B2_auxiliary_validation.png')
    observed_plot(tables['B4']['L'],validations['B4'][1],'B4 cross-family table consistency; evaluation scope unspecified',
        '13_B4_cross_family_validation.png',[r['family'] for r in tables['B4']['records']])
    b5=tables['B5'];sources=sorted({r['source'] for r in b5['records']});fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,source in zip(axes.flat,sources):
        sel=np.array([r['source']==source for r in b5['records']]);yy=b5['L'][sel];pp=validations['B5'][1][sel]
        ax.scatter(yy,pp,s=28);lo,hi=min(yy.min(),pp.min()),max(yy.max(),pp.max());ax.plot([lo,hi],[lo,hi],'--',color='gray',lw=1)
        ax.set(title=source,xlabel='Reported val_loss (scope unspecified)',ylabel='B1 model prediction')
    fig.suptitle('B5 literature source layers: absolute Loss comparability unconfirmed');save(fig,'14_B5_literature_comparison.png')
    write_csv(OUT/'figure_manifest.csv',manifest)


def md_table(rows,columns):
    """Precise estimates/intervals retain meaningful digits even for narrow CIs."""
    def fmt(x):
        if x is None:return '未定义/不适用'
        if isinstance(x,(float,np.floating)):return f'{x:.10g}'
        return str(x).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join('---' for c in columns)+' |']+[
        '| '+' | '.join(fmt(r.get(c)) for c in columns)+' |' for r in rows])


def build_report(b1,models,starts,cv,comparison,unweighted,boot_stats,boot_summary,terminal_rows,validations,associations,residual_summary,settings,protected_n):
    """Full requested report, separating conditional fit from external scientific evidence."""
    primary=models['M0_N_D'];params=primary['parameters'];d=json.loads((OUT/'data_diagnosis.json').read_text(encoding='utf-8'))
    all_starts=[r for r in starts if r['context']=='full_B1_balanced' and r['model']=='M0_N_D'];successful=[r for r in all_starts if r['legal_solution']]
    spread=[dict(parameter=k,min=float(min(r[k] for r in successful)),max=float(max(r[k] for r in successful)),
        relative_range=float((max(r[k] for r in successful)-min(r[k] for r in successful))/params[k])) for k in params]
    main_cv=[r for r in cv if r['model']=='M0_N_D'];weighting=read_rows(OUT/'weighting_sensitivity.csv')
    corr=np.array(boot_summary['parameter_correlation']);key_order=PARAMETERS['M0_N_D'];pairs=[]
    for i in range(5):
        for j in range(i+1,5):pairs.append(dict(parameter_1=key_order[i],parameter_2=key_order[j],bootstrap_Pearson=float(corr[i,j])))
    source=[r for r in validations['B5'][0] if r['stratum']=='source'];family=[r for r in validations['B4'][0] if r['stratum']=='family']
    b3=[r for r in validations['B3'][0] if r['stratum']=='trajectory'];b2=validations['B2'][0]
    early=next(r for r in residual_summary if r['group']=='early_P10');late=next(r for r in residual_summary if r['group']=='late_P90')
    fitted=metrics(b1['L'],predict(params,b1['N'],b1['D']))
    boot_draws=read_rows(OUT/'parameter_bootstrap_draws.csv')
    tails=[]
    for stat in boot_stats:
        key=stat['parameter'];eligible=[r for r in boot_draws if r['eligible_parameter_CI']=='True']
        lo=min(eligible,key=lambda r:float(r[key]));hi=max(eligible,key=lambda r:float(r[key]))
        tails.append(dict(parameter=key,min=stat['min'],max=stat['max'],
            full_range_over_95CI_width=(stat['max']-stat['min'])/(stat['CI95_high']-stat['CI95_low']),
            min_extreme_distinct_trajectories=int(lo['distinct_trajectories']),max_extreme_distinct_trajectories=int(hi['distinct_trajectories'])))
    touching=sum(any(v!=0 for v in json.loads(r['active_bounds'])) for r in boot_draws if r['converged']=='True')
    terminal_starts=[r for r in read_rows(OUT/'terminal_only_multistart.csv') if r['legal_solution']=='True']
    terminal_spread=[dict(parameter=k,min=min(float(a[k]) for a in terminal_starts),max=max(float(a[k]) for a in terminal_starts)) for k in params]
    lines=['# 问题二 PHASE 1：经典 Scaling Law 基线拟合与跨数据源验证','',
        '## 1. 数据范围与可信度','',
        '本阶段只拟合 L(N,D)=E+A N^(-alpha)+B D^(-beta)。仅 B1 参与参数拟合、LOTO CV、bootstrap 和敏感性。B2/B3/B4/B5 仅应用冻结参数，无截距/尺度校准。没有引入 Q、K、p 或 D_eff，没有使用 B6–B10，没有进入问题三。',
        'B1 为附件中的 Pythia existing training log，B11/B12 提供元数据与 checkpoint 来源辅助；这些索引本身不能独立验证每个 Loss 的原始测量过程。B3 明确为插值轨迹，B2 按题目说明为半合成。B4 为跨族汇总表；B5 为六个文献来源标签的整理表，缺少完整评测口径，均不当作同口径独立实验的证明。',
        f"B1 实际 {d['B1_trajectory_count']} 条模型轨迹，每条 {len(b1['N'])//d['B1_trajectory_count']} checkpoints，总 {len(b1['N'])} 行。N范围={d['N_range']} B；D范围={d['D_range']} B tokens；Loss范围={d['Loss_range']}。",'',
        '字段、实际路径、缺失、Inf、重复、B12 对照和 FLOPs 核验详见 outputs/scaling/phase1/data_diagnosis.md 与相应 CSV。','',
        '## 2. checkpoint 分组与单位','',
        'B1 的 run_id 是每条 checkpoint 唯一标识，不能作为独立模型 ID。trajectory_id=B1 + 精确规范化 N_params_B；各规模对应一个稳定轨迹，没有用近似数值合并模型。每个轨迹 N 固定、D 按 steps 严格递增，Loss 无局部反弹；没有重复行或重复 checkpoint。相同规模内未标注的重复实验无法由附件进一步识别。',
        'N=N_params_B，D=D_tokens_B，以 1B 为参考单位直接参与幂律。因此 A、B 数值依赖该单位约定；不能和使用原始参数/token 个数的系数直接比较。FLOPs 单独换算：6(N_B×1e9)(D_B×1e9)/1e21=0.006 N_B D_B。',
        f"B1 C 与 6ND 有 {d['B1_C_6ND_rounding_consistent_count']}/{len(b1['N'])} 条在实际导出精度容差内一致。",'',
        '## 3. 经典模型与估计方法','',
        'Lhat=E+A N^-alpha+B D^-beta，全部五参数严格为正。使用 scipy.optimize.least_squares 有界非线性最小二乘；优化的是 log(参数) 以保持正值，残差始终为原始 Loss 差，绝没有对 Loss 取 log 作普通线性回归。使用解析 Jacobian，原始 Loss 平方误差目标。',
        '主拟合每条 checkpoint 的原始权重为 1/n_g，各轨迹总权重相同，整体归一化至 1。objective=sum_i w_i (L_i-Lhat_i)^2，least_squares 内部 cost 的两倍。',
        f"每次拟合 {settings['multistart_count']} 个不同初值，初始 alpha/beta 在 {settings['initial_exponent_range']} 对数分散，初始幅度依实际 N/D/Loss 尺度构造；选成功、正值、有限且目标最小的解。主模型成功初值 {len(successful)}/{len(all_starts)}。参数边界明示于新配置 {settings['parameter_bounds']}，不是统计 CI；边界接触记录在多初值表。",'',
        md_table([dict(parameter=k,estimate=v) for k,v in params.items()],['parameter','estimate']),
        f"主目标={primary['objective']:.12g}；B1 表内 RMSE={fitted['RMSE']:.12g}；状态={primary['status']}。",'',
        '## 4. 多初值、可识别性与不确定性','',
        md_table(spread,['parameter','min','max','relative_range']),
        f"主拟合 log-parameter 加权 Jacobian 秩={primary['jacobian_rank']}/5，条件数={primary['jacobian_condition_number']:.9g}。条件数依赖参数化尺度，不等于外部科学可识别性的完整证据。",'',
        f"Bootstrap 单位是完整模型轨迹，每次抽取 {d['B1_trajectory_count']} 条轨迹、有放回且保留各自全部 checkpoint，重复抽中的轨迹按 multiplicity 增加权重。共 {boot_summary['attempted']} 次，每次 {settings['bootstrap_multistart_count']} 初值重新拟合。独立固定 RNG 种子 {settings['random_state']}，失败率 {boot_summary['solver_failure_rate']:.6%}；收敛但秩不足 {boot_summary['converged_nonidentifiable']} 次，参数 CI 使用 {boot_summary['identifiable']} 个可识别拟合。没有 checkpoint IID bootstrap。",'',
        md_table(boot_stats,['parameter','main_estimate','median','mean','std','CI95_low','CI95_high','min','max']),
        md_table(tails,['parameter','full_range_over_95CI_width','min_extreme_distinct_trajectories','max_extreme_distinct_trajectories']),
        f"存在稀疏轨迹抽样对应的尾部，极值的轨迹覆盖数见上表。全范围比 95% CI 更宽，但仍需结合前表的绝对数值判断。bootstrap 接触参数边界次数={touching}。不删去可识别的尾部拟合，{boot_summary['identifiable']} 个可识别结果用于 CI。",
        '以上为给定八条轨迹的 cluster-bootstrap percentile CI，不包括 tokenizer、评测集、源数据生成或跨模型族偏差。B1 各轨迹共用 checkpoint/token 网格且数据高度贴合加性幂律，窄 CI 不能证明真实训练损失的不确定性同样小。',
        f"B1 拟合均值的 bootstrap 区间平均宽度={boot_summary['prediction_mean_CI_width']:.12g}，最大宽度={boot_summary['prediction_max_CI_width']:.12g}。它们是拟合均值不确定性，不是未来实验 Loss 的预测区间。",'',
        md_table(pairs,['parameter_1','parameter_2','bootstrap_Pearson']),
        '相关矩阵、alpha-beta、E 与其余参数、极值和 1%/99% 尾部分位均已输出；参数相关不自动意味着秩亏，但强相关提示共同变化方向。仅有八个模型规模，bootstrap 来源覆盖有限，不能将 1176 行当成 1176 个独立实验来增强置信度。','',
        '## 5. 留一轨迹验证与简单基线','',
        '按完整轨迹执行 leave-one-model-out，八折；每次训练其余七条全部 checkpoint，完整留出一条。所有训练/留出行号及模型参数已持久化；不存在随机 checkpoint KFold。主模型和仅 N、仅 D 对照使用相同留出分组和估计目标。','',
        md_table(main_cv,['heldout_trajectory','N_heldout','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias']),
        md_table(comparison,['model','folds','RMSE','MAE','R2','Pearson','Spearman']),
        '仅 N 模型在同一规模轨迹内预测常数，Pearson/Spearman 为未定义，不能伪造零相关系数；这也是 N-only 无法解释训练进度的直接限制。','',
        '## 6. balanced/unweighted 对照','',
        md_table(weighting,['weighting','E','A','B','alpha','beta','objective','RMSE','max_abs_prediction_difference_from_balanced']),
        '本次八条轨迹均有 147 个 checkpoint，balanced 和 unweighted 的归一化权重数学等价，出现相同解是预期结果，不是加权方案对一般不平衡数据无用。','',
        '## 7. terminal-only 对照及结构性不可识别','',
        md_table(terminal_rows,['dataset','n_fit','E','A','B','alpha','beta','jacobian_rank','terminal_RMSE','full_B1_RMSE_descriptive','E_effective_at_terminal_D']),
        md_table(terminal_spread,['parameter','min','max']),
        '全部终点 D 相同，只有 E_eff=E+B D_terminal^-beta 可识别。terminal-only Jacobian 最多秩 3，五参数存在结构性混淆，多初值可有近似相同终点预测但不同 E/B/beta。该解只作敏感性，不替换主模型；主模型包含训练轨迹才能区分数据项和不可约项。',
        '两种约束模型对 N、D 的理论导数均为负，但这种同向来自正参数约束；终点数据没有 D 变化，不能说终点经验验证了 beta 的方向。终点 N–Loss 趋势可检验，D 方向不可由这些终点单独验证。','',
        '## 8. 冻结与 B3 插值一致性','',
        '完成所有 B1 拟合与训练内诊断后写入 freeze_before_validation.json，后续验证只应用主模型参数，没有重拟合或偏置校准。B3 插值轨迹与 B1 共用模型规模和训练范围，因此不是独立真实实验。','',
        md_table(validations['B3'][0],['stratum','group_key','n','RMSE','MAE','Pearson','Spearman','mean_bias']),
        '插值表可检验数值/曲线一致性，但其较密集的 4000 个点不产生额外独立实验样本量。','',
        '## 9. B2 半合成辅助族外检验','',
        md_table(b2,['stratum','group_key','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias']),
        f"B2 总体 RMSE={b2[0]['RMSE']:.9g}、平均残差={b2[0]['mean_bias']:+.9g}，绝对 Loss 系统低估明显。各模型轨迹内 Spearman 在 {min(a['Spearman'] for a in b2[1:]):.6f}–{max(a['Spearman'] for a in b2[1:]):.6f}，整体 Spearman={b2[0]['Spearman']:.6f}。单轨迹下降排序一致不能弥补族外绝对水平失败。",
        '排序/下降趋势与绝对误差分别判断；B2 的半合成属性限制对真实跨族泛化的解释。没有借 B2 修正 E 或其他参数。','',
        '## 10. B4 跨族表一致性','',
        md_table(validations['B4'][0],['stratum','group_key','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias','median_residual']),
        f"B4 整体 Spearman={validations['B4'][0][0]['Spearman']:.6f}，但平均残差={validations['B4'][0][0]['mean_bias']:+.6f}，存在系统偏差。family 平均残差范围 {min(a['mean_bias'] for a in family):+.6f}–{max(a['mean_bias'] for a in family):+.6f}；表中 Pythia 层也有平均残差 {next(a['mean_bias'] for a in family if a['group_key']=='Pythia'):+.6f}。因此支持该表上的排序趋势迁移，不支持统一绝对 Loss 预测。",
        'B4 没有给出可核实的共同 evaluation dataset/tokenizer 口径。表上的整体排序与家族分层残差是数值诊断，不能单凭整体相关就断言 Pythia Loss 参数可直接迁移到全部族。不同 family 的偏置不能通过全局校准掩盖，本阶段未校准。','',
        '## 11. B5 文献分层验证','',
        md_table(source,['group_key','n','Pearson','Spearman','mean_bias','median_residual','RMSE']),
        f"B5 source 内 Spearman 范围 {min(a['Spearman'] for a in source):.6f}–{max(a['Spearman'] for a in source):.6f}，支持整理表中的来源内趋势一致性；没有证明不同来源的绝对 Loss 已统一。",
        'B5 的 source/family 均保留分层。没有 loss_type、dataset、tokenizer，无法确认跨文献绝对 Loss 可比性；核心证据是 source 内的趋势和排序，表中 RMSE/残差仅为附件数值的描述。每 source 点数也较少，相关不能视为充分实验验证。',
        '没有从文献标签推断未提供的测量协议，也没有按验证值选择参数。完整 family/source-family 层在 B5_literature_validation.csv。','',
        '## 12. 残差结构','',
        md_table(associations,['variable','Pearson','Spearman','n_checkpoints','independent_trajectory_count']),
        md_table(residual_summary,['stratum','group','n','mean_bias','RMSE','residual_std','lag1_residual_Pearson']),
        f"早期训练 P10 RMSE={early['RMSE']:.12g}，末期 P90 RMSE={late['RMSE']:.12g}。残差形状、相关和相邻 checkpoint 相关是结构性诊断，不做 IID 显著性 p 值。", 
        f"早期/末期 RMSE 比={early['RMSE']/late['RMSE']:.6f}，存在早期误差较大的异方差结构；总体残差与 log N/log D 的 Pearson 较小，未见强的全局线性均值偏移。即便早期误差较大，仍应同时报告其绝对数值，不能把相对差异夸大为大幅模型失效。",
        '即使绝对残差很小，也需区分表内近幂律拟合与真实跨数据源泛化。残差与 log D/progress 或小规模模型的结构不通过增加复杂模型消除，本阶段保持经典基线。','',
        '## 13. 是否可以进入下一阶段','',
        '当前经典基线的 B1 参数估计与轨迹分组预测稳定，可以作为后续受控扩展检验的冻结参考；这不等于绝对 Loss 已跨族/跨文献普适。B2 的系统误差、B4 的 family 偏差、B5 的未知 Loss 口径应单独保留。',
        '后续若人工批准质量扩展，需要验证是否超出这一基线的真实增益，而不能把家族/口径/数据来源的偏差强行归因于质量。只有八个轨迹的 cluster CI 无法涵盖这些误差。',
        f"冻结旧项目文件 {protected_n} 个的 SHA-256 全部一致。验收与独立公式重建在 independent_verification.csv；所有数值来源、bootstrap 抽样、32 初值拟合、CV 分组均可追溯。",
        '**本轮结束于问题二 PHASE 1，等待人工确认；未构造质量扩展、未读取 B6–B10 进行建模、未进入问题三。**','']
    (ROOT/'outputs/report/question2_phase1_classical_scaling.md').write_text('\n\n'.join(lines),encoding='utf-8')
    write_csv(OUT/'multistart_parameter_stability.csv',spread)
    write_csv(OUT/'parameter_pair_correlations.csv',pairs)
    write_csv(OUT/'parameter_tail_diagnostics.csv',tails)
    write_csv(OUT/'terminal_multistart_parameter_spread.csv',terminal_spread)


def refresh_saved(figures=False):
    """Regenerate only presentation from completed saved fits; never call fit/bootstrap."""
    def numeric(path):
        records=read_rows(path)
        for r in records:
            for key,value in r.items():
                if value=='True':r[key]=True
                elif value=='False':r[key]=False
                elif value=='':r[key]=None
                else:
                    try:r[key]=float(value)
                    except ValueError:pass
        return records
    if not (OUT/'execution_summary.json').exists():raise RuntimeError('Only completed results can be refreshed')
    frozen=json.loads((OUT/'frozen_project_before.json').read_text(encoding='utf-8'))
    freeze=json.loads((OUT/'freeze_before_validation.json').read_text(encoding='utf-8'))
    for path,digest in freeze['files'].items():
        if sha256(ROOT/path)!=digest:raise RuntimeError('Frozen phase1 parameters/config changed')
    files,trajectory_paths,_=input_paths(ROOT);b1=load_csv(files['B1'],'B1');tables={k:load_csv(v,k) for k,v in files.items() if k!='B1'}
    trajectories=[load_csv(path,'B3') for path in trajectory_paths]
    models=json.loads((OUT/'baseline_models.json').read_text(encoding='utf-8'));unweighted=json.loads((OUT/'unweighted_model.json').read_text(encoding='utf-8'))
    boot_summary=json.loads((OUT/'bootstrap_summary.json').read_text(encoding='utf-8'))
    settings=yaml.safe_load((ROOT/'configs/scaling_phase1.yaml').read_text(encoding='utf-8'))
    validations={}
    for label in ['B2','B3','B4','B5']:
        name=f'{label}_validation.csv' if label in ['B2','B3'] else ('B4_cross_family_validation.csv' if label=='B4' else 'B5_literature_validation.csv')
        pred=numeric(OUT/f'{label}_direct_predictions.csv');validations[label]=(numeric(OUT/name),np.array([r['predicted_loss'] for r in pred]))
    residuals=numeric(OUT/'B1_residuals.csv');bootstrap_rows=numeric(OUT/'parameter_bootstrap_draws.csv')
    if figures:plot_all(b1,models,numeric(OUT/'B1_group_cv_predictions.csv'),bootstrap_rows,tables,trajectories,validations,residuals,settings)
    protected_n=diagnostic.verify_protected(frozen)
    build_report(b1,models,numeric(OUT/'multistart_results.csv'),numeric(OUT/'B1_group_cv.csv'),numeric(OUT/'baseline_model_comparison.csv'),
        unweighted,numeric(OUT/'parameter_bootstrap.csv'),boot_summary,numeric(OUT/'terminal_checkpoint_sensitivity.csv'),validations,
        numeric(OUT/'residual_associations.csv'),numeric(OUT/'residual_structure_summary.csv'),settings,protected_n)
    print('Saved-result report/figures refreshed; no fitting, bootstrap or calibration.')


def main():
    """Complete B1-only fitting first, freeze parameters, then validate other sources."""
    if (OUT/'execution_summary.json').exists():raise RuntimeError('Phase 1 already completed and frozen; do not rerun fitting.')
    if not (OUT/'data_diagnosis.json').exists():raise RuntimeError('Run 10_diagnose_scaling_data.py first; no blind fit')
    for p in [OUT,FIG,ROOT/'outputs/report']:p.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'fitting.log',encoding='utf-8')],force=True)
    frozen=json.loads((OUT/'frozen_project_before.json').read_text(encoding='utf-8'));diagnostic.verify_protected(frozen)
    settings=yaml.safe_load((ROOT/'configs/scaling_phase1.yaml').read_text(encoding='utf-8'))
    if settings['multistart_count']<30 or settings['bootstrap_multistart_count']<30:raise ValueError('At least 30 starts required')
    files,trajectory_paths,_=input_paths(ROOT);b1=load_csv(files['B1'],'B1')
    # Only B1 is loaded numerically until all fits/parameter choices are frozen.
    models,starts,cv,comparison,unweighted=fit_and_cv(b1,settings)
    bootstrap_rows,boot_stats,boot_summary=cluster_bootstrap(b1,models['M0_N_D'],settings)
    terminal,terminal_rows=terminal_fit(b1,models['M0_N_D'],settings)
    residuals,associations,residual_summary=residual_diagnostics(b1,models['M0_N_D'])
    freeze_paths=[OUT/name for name in ['main_model.json','baseline_models.json','unweighted_model.json','group_cv_models.json','terminal_only_model.json']]+[ROOT/'configs/scaling_phase1.yaml']
    freeze={p.relative_to(ROOT).as_posix():sha256(p) for p in freeze_paths}
    write_json(OUT/'freeze_before_validation.json',dict(fit_datasets=['B1'],validation_datasets=['B2','B3','B4','B5'],
        parameters_used_for_validation=models['M0_N_D']['parameters'],files=freeze,validation_used_for_parameter_selection=False))
    tables={label:load_csv(path,label) for label,path in files.items() if label!='B1'}
    trajectories=[load_csv(path,'B3') for path in trajectory_paths];b3=combine_trajectories(trajectories);parameters=models['M0_N_D']['parameters']
    validations={}
    validations['B3']=validate(b3,parameters,'B3',{'trajectory':b3['group']},'interpolated_trajectory_consistency',True)
    b2=tables['B2'];validations['B2']=validate(b2,parameters,'B2',{'trajectory':b2['group']},'semi_synthetic_auxiliary',False)
    b4=tables['B4'];validations['B4']=validate(b4,parameters,'B4',{'family':[r['family'] for r in b4['records']]},'cross_family_table',False)
    b5=tables['B5'];validations['B5']=validate(b5,parameters,'B5',{'source':[r['source'] for r in b5['records']],
        'family':[r['family'] for r in b5['records']],'source_family':[r['source']+' / '+r['family'] for r in b5['records']]},'literature_compilation_evaluation_scope_unconfirmed',False)
    for label,(stats,_) in validations.items():logging.info('%s frozen prediction RMSE=%.7g Spearman=%s',label,stats[0]['RMSE'],stats[0]['Spearman'])
    predictions=read_rows(OUT/'B1_group_cv_predictions.csv')
    plot_all(b1,models,predictions,bootstrap_rows,tables,trajectories,validations,residuals,settings)
    protected_n=diagnostic.verify_protected(frozen)
    build_report(b1,models,starts,cv,comparison,unweighted,boot_stats,boot_summary,terminal_rows,validations,associations,residual_summary,settings,protected_n)
    for p,h in freeze.items():
        if sha256(ROOT/p)!=h:raise RuntimeError('Scaling parameters/config changed during validation')
    summary=dict(main_model='M0_N_D',main_fit_dataset='B1_only',parameters=parameters,
        objective=models['M0_N_D']['objective'],jacobian_rank=models['M0_N_D']['jacobian_rank'],
        bootstrap=boot_summary,group_cv=comparison,validation_overall={label:data[0][0] for label,data in validations.items()},
        frozen_project_files_unchanged=protected_n,quality_extension_started=False,phase3_started=False)
    write_json(OUT/'execution_summary.json',summary)
    logging.info('COMPLETE: question2 phase1; bootstrap=%d; frozen files=%d; stopped for human confirmation',len(bootstrap_rows),protected_n)


if __name__=='__main__':
    try:
        if sys.argv[1:] in [['--report-only'],['--figures-only']]:refresh_saved(sys.argv[1:]==['--figures-only'])
        else:main()
    except Exception:
        logging.exception('Scaling phase1 failed; frozen files must remain unchanged')
        raise

# ============================================================
# Source: src/scaling/quality_extension.py
# Original lines: 1-118
# Purpose: 最终链复用读入/reference工具；常数gamma MQ候选不是最终主模型
# MQ1/MQ2/gamma fitting: candidate / sensitivity model; not used as final main model. Final chain uses read_parameters, read_data, subset and group_reference.
# ============================================================
"""Quality-only extensions with immutable classical coefficients and strict intake."""
from __future__ import annotations
from collections import defaultdict
from decimal import Decimal
import numpy as np
from scipy.optimize import least_squares
from .io import csv_rows

NAMES = {'B6':'supplementary_NQ_experiment.csv',
         'B7':'supplementary_NQ_experiment_expanded.csv',
         'B8':'supplementary_NQ_experiment_large.csv'}
MODELS = ['M0', 'MQ1', 'MQ2', 'MQ1_offset', 'MQ2_offset']


def read_parameters(root):
    """Read full-precision primary Phase 1 row; never estimate classical terms."""
    path=root/'outputs/scaling/phase1/classical_scaling_parameters.csv'
    _, rows=csv_rows(path)
    primary=[r for r in rows if r['model']=='M0_N_D' and r['weighting']=='trajectory_balanced']
    if len(primary)!=1:raise ValueError('Ambiguous frozen classical parameter row')
    return {k:float(primary[0][k]) for k in ['E','A','B','alpha','beta']}, path


def read_data(root, label):
    """Canonical exact decimal NDQ keys; no imputation, deletion or Q reversal."""
    path=root/'real_attachments/B_scaling_laws'/NAMES[label]
    columns, rows=csv_rows(path)
    a=np.array([[float(r[c]) for c in ['N_params_B','D_tokens_B','Q_score','val_loss']] for r in rows])
    if not np.isfinite(a).all() or (a[:,:3]<=0).any() or (a[:,2]>1).any():
        raise ValueError(f'{label}: illegal N/D/Q/L')
    keys=[tuple(format(Decimal(r[c]).normalize(),'f') for c in ['N_params_B','D_tokens_B','Q_score']) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError(f'{label}: repeated candidate key; cannot take first/last')
    group=np.array(['N='+k[0]+';D='+k[1] for k in keys])
    return dict(label=label,path=path,columns=columns,records=rows,keys=keys,group=group,
                N=a[:,0],D=a[:,1],Q=a[:,2],L=a[:,3])


def subset(t, index):
    """Subset arrays for fitting, retaining source rows in the original dataset."""
    return {k:t[k][index] for k in ['N','D','Q','L','group']}


def predict_quality(t, p, model, gamma=0.0, offset=0.0):
    """Predict raw Loss with N/D in billions; no modification to frozen p."""
    N,D,Q=t['N'],t['D'],t['Q']
    base=p['E']+p['A']*N**(-p['alpha'])
    term=p['B']*D**(-p['beta'])
    if model=='M0':return base+term
    factor=-p['beta']*np.log(Q) if model.startswith('MQ1') else p['beta']*(1-Q)
    result=base+term*np.exp(gamma*factor)+(offset if model.endswith('_offset') else 0.0)
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite quality prediction')
    return result


def fit_quality(t, p, model, config, multiplicities=None):
    """Bounded raw-Loss least squares from every specified start.

    A bootstrap multiplicity repeats the complete ND block. All starts use only
    the supplied training rows. Classical coefficients are read-only arguments.
    """
    if model not in MODELS[1:]:raise ValueError(model)
    n=len(t['L']); weights=np.ones(n)
    if multiplicities is not None:
        weights=np.array([multiplicities.get(g,0) for g in t['group']],dtype=float)
    active=weights>0
    if not active.any():raise ValueError('Empty resampled training set')
    a=subset(t,active); sw=np.sqrt(weights[active]/weights.sum())
    base=p['E']+p['A']*a['N']**(-p['alpha'])
    term=p['B']*a['D']**(-p['beta'])
    factor=-p['beta']*np.log(a['Q']) if model.startswith('MQ1') else p['beta']*(1-a['Q'])
    has_offset=model.endswith('_offset')
    def fun(x):
        return sw*(base+term*np.exp(x[0]*factor)+(x[1] if has_offset else 0)-a['L'])
    def jac(x):
        j=sw*term*factor*np.exp(x[0]*factor)
        return np.column_stack([j,sw]) if has_offset else j[:,None]
    lo=[config['gamma_bounds'][0]];hi=[config['gamma_bounds'][1]]
    if has_offset:
        lo.append(config['offset_bounds'][0]);hi.append(config['offset_bounds'][1])
    runs=[]
    for i,initial in enumerate(config['gamma_starts'],1):
        x0=[initial,0.0] if has_offset else [initial]
        opt=least_squares(fun,x0,jac=jac,bounds=(lo,hi),method='trf',loss='linear',
                          ftol=config['optimizer_tolerance'],xtol=config['optimizer_tolerance'],
                          gtol=config['optimizer_tolerance'],max_nfev=config['max_nfev'])
        objective=float(np.dot(opt.fun,opt.fun))
        legal=bool(opt.success and np.isfinite(objective) and opt.x[0]>0 and np.isfinite(opt.x).all())
        runs.append(dict(start=i,initial_gamma=float(initial),initial_offset=0.0,
                         gamma=float(opt.x[0]),offset=float(opt.x[1]) if has_offset else 0.0,
                         objective=objective,successful=bool(opt.success),legal_solution=legal,
                         status=int(opt.status),nfev=int(opt.nfev),optimality=float(opt.optimality),
                         at_bound=bool(np.any(np.isclose(opt.x,lo,rtol=0,atol=1e-6)) or np.any(np.isclose(opt.x,hi,rtol=0,atol=1e-6))),
                         message=opt.message))
    valid=[r for r in runs if r['legal_solution']]
    if not valid:raise RuntimeError('No successful legal quality fit: '+model)
    best=min(valid,key=lambda r:r['objective'])
    return dict(model=model,**best,n_starts=len(runs),
                distinct_solution_gamma_range=max(r['gamma'] for r in valid)-min(r['gamma'] for r in valid),
                n_training_rows=int(active.sum()),n_training_ND_groups=len(set(a['group']))), runs


def group_reference(t, prediction):
    """Differences relative to highest observed Q in the full evaluation dataset.

    For B7 new Q points, the Q=1 reference belongs to the shared ND trajectory.
    No model is fitted on that reference. References themselves are marked and
    excluded from primary difference metrics to avoid trivial-zero inflation.
    """
    groups=defaultdict(list)
    for i,g in enumerate(t['group']):groups[g].append(i)
    obs=np.zeros(len(t['L']));pred=np.zeros(len(t['L']));refq=np.zeros(len(t['L']))
    refs=np.zeros(len(t['L']),dtype=bool)
    for idx in groups.values():
        ref=max(idx,key=lambda i:t['Q'][i])
        obs[idx]=t['L'][idx]-t['L'][ref]
        pred[idx]=prediction[idx]-prediction[ref]
        refq[idx]=t['Q'][ref];refs[ref]=True
    return obs,pred,refq,refs

# ============================================================
# Source: src/scaling/scale_quality.py
# Original lines: 1-133
# Purpose: S3原参数来源；S1/S2只为比较候选，最终去v且不重拟合
# S1/S2 are candidate / sensitivity models, not used as final main model. S3 C,u are frozen inputs to local Delta_Q; the near-zero v is removed without refitting.
# ============================================================
"""Three scale-modulated quality models; no estimation of classical coefficients."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

PARAMETERS={'S1':['gamma0','u','v'],'S2':['C','u','v','eta'],'S3':['C','u','v']}
POSITIVE={'gamma0','C','eta'}


def centers(t):
    """Median in log space over unique design levels, not response-weighted scales."""
    return dict(N0=float(np.exp(np.median(np.log(np.unique(t['N']))))),
                D0=float(np.exp(np.median(np.log(np.unique(t['D']))))))


def components(t,p,reference):
    """Reusable raw-Loss and dimensionless logarithmic design terms."""
    logN=np.log(t['N']/reference['N0']);logD=np.log(t['D']/reference['D0']);qlog=-np.log(t['Q'])
    term=p['B']*t['D']**(-p['beta'])
    l0=p['E']+p['A']*t['N']**(-p['alpha'])+term
    return logN,logD,qlog,term,l0


def physical(x,keys):
    """Log coordinates for strictly positive parameters; u/v include exact zero."""
    return {k:float(np.exp(a)) if k in POSITIVE else float(a) for k,a in zip(keys,x)}


def encoded(parameters,keys):
    return np.array([np.log(parameters[k]) if k in POSITIVE else parameters[k] for k in keys])


def response_jacobian(x,keys,model,parts,beta):
    """Analytic prediction/Jacobian in the optimizer's coordinates.

    The broad S1 bounds still keep exponents below double-precision overflow
    on the complete B6 support. No residual clipping or altered model is used.
    """
    a=physical(x,keys);n,d,z,term,l0=parts;scale=np.exp(-a['u']*n+a['v']*d)
    if model=='S1':
        gamma=a['gamma0']*scale;f=beta*gamma*z
        expf=np.exp(f);quality=term*np.expm1(f);j0=term*expf*f
        derivatives={'gamma0':j0,'u':-j0*n,'v':j0*d}
    elif model=='S2':
        f=a['eta']*z;expf=np.exp(f);quality=a['C']*scale*np.expm1(f)
        derivatives={'C':quality,'u':-quality*n,'v':quality*d,'eta':a['C']*scale*expf*f}
    elif model=='S3':
        quality=a['C']*scale*z;derivatives={'C':quality,'u':-quality*n,'v':quality*d}
    else:raise ValueError(model)
    pred=l0+quality
    if 'offset' in keys:pred+=a['offset'];derivatives['offset']=np.ones(len(z))
    jac=np.column_stack([derivatives[k] for k in keys])
    if not np.isfinite(pred).all() or not np.isfinite(jac).all():raise FloatingPointError('Nonfinite model evaluation')
    return pred,jac


def predict_scale(t,p,reference,model,parameters):
    """Frozen classical skeleton plus a nonnegative Q=1-anchored quality term."""
    keys=PARAMETERS[model]+(['offset'] if 'offset' in parameters else [])
    return response_jacobian(encoded(parameters,keys),keys,model,components(t,p,reference),p['beta'])[0]


def marginal_quality(t,p,reference,model,parameters):
    """Exact derivative dLoss/dQ, including S1's baseline D dependence."""
    n,d,z,term,_=components(t,p,reference)
    scale=np.exp(-parameters['u']*n+parameters['v']*d)
    if model=='S1':
        gamma=parameters['gamma0']*scale
        grad=-term*np.exp(p['beta']*gamma*z)*p['beta']*gamma/t['Q']
    elif model=='S2':
        grad=-parameters['C']*scale*parameters['eta']*np.exp(parameters['eta']*z)/t['Q']
    elif model=='S3':grad=-parameters['C']*scale/t['Q']
    else:raise ValueError(model)
    return grad


def initial_parameters(model,cfg,seed,has_offset=False,count=None):
    """At least 50 dispersed, deterministic starts independent of evaluation Loss."""
    rng=np.random.default_rng(seed);count=count or cfg['multistart_count'];starts=[]
    if model=='S1':
        for g in [.1,.25,.5,1,2,4,8]:starts.append(dict(gamma0=g,u=.1,v=.1))
    elif model=='S2':
        for c,eta in [(.1,.1),(.2,.5),(.5,1),(1,.2),(5,.05),(50,.003),(100,.002)]:
            starts.append(dict(C=c,u=.1,v=.05,eta=eta))
    else:
        for c in [.03,.1,.2,.5,1,3,8]:starts.append(dict(C=c,u=.1,v=.1))
    while len(starts)<count:
        a=dict(u=float(rng.uniform(0,.5)),v=float(rng.uniform(0,.5)))
        if model=='S1':a['gamma0']=float(np.exp(rng.uniform(np.log(.08),np.log(3))))
        else:a['C']=float(np.exp(rng.uniform(np.log(.03),np.log(10))))
        if model=='S2':a['eta']=float(np.exp(rng.uniform(np.log(.002),np.log(1.5))))
        starts.append(a)
    if has_offset:
        for a in starts:a['offset']=float(rng.uniform(-.1,.1))
    return starts[:count]


def fit_scale(t,p,reference,model,cfg,seed,multiplicities=None,has_offset=False,count=None):
    """Bounded nonlinear LS with every start audited; bootstrap repeats whole ND blocks."""
    keys=PARAMETERS[model]+(['offset'] if has_offset else [])
    weights=np.ones(len(t['L'])) if multiplicities is None else np.array([multiplicities.get(g,0) for g in t['group']],dtype=float)
    ix=weights>0
    a={k:t[k][ix] for k in ['N','D','Q','L','group']};weights=weights[ix];weights/=weights.sum();sw=np.sqrt(weights)
    parts=components(a,p,reference);bounds=cfg['parameter_bounds']
    lo=encoded({k:bounds[k][0] for k in keys},keys);hi=encoded({k:bounds[k][1] for k in keys},keys)
    # u/v have zero lower bounds, which encoded() preserves without taking log.
    def fun(x):return sw*(response_jacobian(x,keys,model,parts,p['beta'])[0]-a['L'])
    def jac(x):return sw[:,None]*response_jacobian(x,keys,model,parts,p['beta'])[1]
    runs=[];tol=cfg['optimizer_tolerance']
    for i,start in enumerate(initial_parameters(model,cfg,seed,has_offset,count),1):
        x0=np.clip(encoded(start,keys),lo,hi)
        opt=least_squares(fun,x0,jac=jac,bounds=(lo,hi),method='trf',loss='linear',x_scale='jac',
                          max_nfev=cfg['max_nfev'],ftol=tol,xtol=tol,gtol=tol)
        params=physical(opt.x,keys);obj=float(np.dot(opt.fun,opt.fun))
        sv=np.linalg.svd(opt.jac,compute_uv=False)
        rank=int((sv>sv[0]*cfg['jacobian_rank_relative_tolerance']).sum()) if len(sv) and sv[0]>0 else 0
        condition=float(sv[0]/sv[-1]) if len(sv) and sv[-1]>0 else None
        hit={k:bool(params[k]-bounds[k][0]<=cfg['boundary_absolute_tolerance'] or bounds[k][1]-params[k]<=cfg['boundary_absolute_tolerance']) for k in keys}
        legal=bool(opt.success and np.isfinite(obj) and all(np.isfinite(v) for v in params.values()))
        runs.append(dict(start=i,initial_parameters=start,parameters=params,objective=obj,
                         convergence=bool(opt.success),legal_solution=legal,status=int(opt.status),nfev=int(opt.nfev),
                         jacobian_rank=rank,jacobian_condition=condition,jacobian_singular_values=sv.tolist(),
                         boundary_hit=any(hit.values()),boundary_parameters=[k for k,v in hit.items() if v],
                         per_parameter_boundary=hit,message=opt.message))
    valid=[r for r in runs if r['legal_solution']]
    if not valid:raise RuntimeError('No successful legal fit: '+model)
    best=min(valid,key=lambda r:r['objective'])
    nearby=[r for r in valid if r['objective']<=best['objective']+1e-9]
    spread={k:max(r['parameters'][k] for r in nearby)-min(r['parameters'][k] for r in nearby) for k in keys}
    return dict(model=model,source_offset=has_offset,**best,n_starts=len(runs),n_successful_starts=len(valid),
                parameter_count=len(keys),near_optimal_parameter_spread=spread,
                all_start_boundary_rate=sum(r['boundary_hit'] for r in valid)/len(valid),
                n_training_rows=int(ix.sum()),n_training_ND_groups=len(set(a['group']))),runs

# ============================================================
# Source: src/scaling/scale_quality_analysis.py
# Original lines: 1-253
# Purpose: ND分组CV、Q层留出、整块bootstrap及B6/B7差分检验
# ============================================================
"""Grouped model comparison, audited multi-start bootstrap and sensitivity surfaces."""
from __future__ import annotations
import csv
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from .quality_extension import subset,group_reference
from .scale_quality import PARAMETERS,fit_scale,predict_scale,marginal_quality
from .classical import predict,metrics
from .io import write_csv,csv_rows


def evaluate(t,pred,model,stage,mask=None):
    """Report absolute and reference-excluded within-ND differences together."""
    if mask is None:mask=np.ones(len(t['L']),dtype=bool)
    obs,dp,refq,refs=group_reference(t,pred)
    active=mask&~refs
    if not np.all(refq[mask]==1):raise ValueError('This B6/B7 evaluation requires observed Q=1 references')
    return [dict(model=model,stage=stage,target='absolute_Loss',**metrics(t['L'][mask],pred[mask])),
            dict(model=model,stage=stage,target='within_ND_DeltaL',reference_zero_rows_excluded=True,
                 reference_rule='Q=1 from full source ND group; reference may overlap B6 for B7',
                 **metrics(obs[active],dp[active]))]


def group_cv(t,p,reference,cfg,out):
    """Independent 50-start train-fold fits; ND blocks cannot be split."""
    models=['M0']+cfg['models'];oof={m:np.full(len(t['L']),np.nan) for m in models}
    folds=[];fits=[];starts=[];foldmetrics=[];rowfold=np.zeros(len(t['L']),dtype=int)
    for fold,(train,test) in enumerate(GroupKFold(n_splits=cfg['group_cv_folds']).split(t['N'],groups=t['group']),1):
        tr,te=subset(t,train),subset(t,test);rowfold[test]=fold
        trgroups,tegroups=sorted(set(tr['group'])),sorted(set(te['group']))
        if set(trgroups)&set(tegroups):raise AssertionError('ND group leakage')
        folds.append(dict(fold=fold,train_rows=len(train),test_rows=len(test),train_groups=trgroups,test_groups=tegroups))
        for m in models:
            if m=='M0':yp=predict(p,te['N'],te['D'])
            else:
                f,rs=fit_scale(tr,p,reference,m,cfg,cfg['random_state']+fold)
                fits.append(dict(fold=fold,**f));starts.extend(dict(stage='GroupCV',model=m,fold=fold,**r) for r in rs)
                yp=predict_scale(te,p,reference,m,f['parameters'])
            oof[m][test]=yp;foldmetrics+=evaluate(te,yp,m,'fold_'+str(fold))
        logging.info('Group CV fold %d/5 completed, training groups=%d test groups=%d',fold,len(trgroups),len(tegroups))
    comparison=[]
    for m,yp in oof.items():
        if not np.isfinite(yp).all():raise AssertionError('Incomplete OOF predictions')
        comparison+=evaluate(t,yp,m,'B6_5fold_GroupCV_OOF')
    pointrows=[dict(row_index=i,fold=int(rowfold[i]),ND_group=t['group'][i],N_params_B=float(t['N'][i]),D_tokens_B=float(t['D'][i]),
                    Q_score=float(t['Q'][i]),actual_Loss=float(t['L'][i]),**{m+'_prediction':float(v[i]) for m,v in oof.items()}) for i in range(len(t['L']))]
    write_csv(out/'group_cv_predictions.csv',pointrows);write_csv(out/'group_cv_folds.csv',folds)
    write_csv(out/'group_cv_model_parameters.csv',fits);write_csv(out/'group_cv_fold_metrics.csv',foldmetrics)
    write_csv(out/'B6_group_cv.csv',comparison)
    return oof,comparison,starts


def q_level_cv(t,p,reference,cfg,out):
    """Hold out each entire Q layer; fitting always uses training observations only."""
    rows=[];starts=[];predpoints=[]
    levels=np.unique(t['Q'])
    for j,q in enumerate(levels):
        tr=subset(t,t['Q']!=q);te=subset(t,t['Q']==q)
        for m in cfg['models']:
            f,rs=fit_scale(tr,p,reference,m,cfg,cfg['random_state']+100+j)
            starts.extend(dict(stage='leave_one_Q_out',model=m,held_Q=float(q),**r) for r in rs)
            yp=predict_scale(te,p,reference,m,f['parameters'])
            rows.append(dict(model=m,held_Q=float(q),test_kind='endpoint_extrapolation' if q in [levels[0],levels[-1]] else 'interior_interpolation',
                             parameters=f['parameters'],boundary_parameters=f['boundary_parameters'],**metrics(te['L'],yp)))
            predpoints.extend(dict(model=m,held_Q=float(q),ND_group=g,actual_Loss=float(y),predicted_Loss=float(v)) for g,y,v in zip(te['group'],te['L'],yp))
    write_csv(out/'leave_one_Q_level_out.csv',rows);write_csv(out/'leave_one_Q_predictions.csv',predpoints)
    return rows,starts


_WORKER_DATA=None


def initialize_worker(t,p,reference,cfg):
    """Set small read-only worker arrays; no source/output writes in workers."""
    global _WORKER_DATA
    from threadpoolctl import threadpool_limits
    # Small 360x4 systems benefit from one BLAS thread per CPU worker.
    threadpool_limits(limits=1)
    _WORKER_DATA=(t,p,reference,cfg)


def bootstrap_worker(job):
    """Fit all three models on the same whole-ND bootstrap draw, 50 starts each."""
    rep,groups,counts=job;t,p,reference,cfg=_WORKER_DATA
    mult=dict(zip(groups,counts));rows=[];starts=[]
    for m in cfg['models']:
        try:
            f,rs=fit_scale(t,p,reference,m,cfg,cfg['random_state']+1000+rep,mult,
                           count=cfg['bootstrap_multistart_count'])
            rows.append(dict(replicate=rep,model=m,success=True,parameters=f['parameters'],objective=f['objective'],
                             jacobian_rank=f['jacobian_rank'],jacobian_condition=f['jacobian_condition'],
                             boundary_hit=f['boundary_hit'],boundary_parameters=f['boundary_parameters'],
                             n_starts=f['n_starts'],n_successful_starts=f['n_successful_starts'],error=''))
            starts.extend(dict(replicate=rep,model=m,**r) for r in rs)
        except (RuntimeError,ValueError,FloatingPointError) as ex:
            rows.append(dict(replicate=rep,model=m,success=False,parameters=None,objective=None,jacobian_rank=None,
                             jacobian_condition=None,boundary_hit=None,boundary_parameters=[],
                             n_starts=cfg['bootstrap_multistart_count'],n_successful_starts=0,error=str(ex)))
    return rep,rows,starts


def resampling_jobs(t,cfg):
    """Uniform with-replacement sampling over 45 exact ND blocks."""
    groups=np.unique(t['group']);G=len(groups);rng=np.random.default_rng(cfg['random_state'])
    return [(rep,groups.tolist(),rng.multinomial(G,np.full(G,1/G)).tolist()) for rep in range(1,cfg['bootstrap_repetitions']+1)]


def bootstrap_models(t,p,reference,cfg,out,resume=False):
    """Bounded CPU workers; main process writes an audit of every one of 300k starts."""
    jobs=resampling_jobs(t,cfg);rows=[];params=[];correlations=[]
    write_csv(out/'bootstrap_block_draws.csv',[dict(replicate=r,ND_groups=g,multiplicities=c,n_sampled_blocks=sum(c),n_distinct_blocks=sum(v>0 for v in c)) for r,g,c in jobs])
    cols=['replicate','model','start','initial_parameters','parameters','objective','convergence','legal_solution','status','nfev',
          'jacobian_rank','jacobian_condition','jacobian_singular_values','boundary_hit','boundary_parameters','per_parameter_boundary','message']
    completed=0
    if resume:
        import json
        _,saved=csv_rows(out/'bootstrap_replicates.csv')
        for r in saved:
            rows.append(dict(replicate=int(r['replicate']),model=r['model'],success=r['success']=='True',
                             parameters=json.loads(r['parameters']) if r['parameters'] else None,
                             objective=float(r['objective']) if r['objective'] else None,
                             jacobian_rank=int(r['jacobian_rank']) if r['jacobian_rank'] else None,
                             jacobian_condition=float(r['jacobian_condition']) if r['jacobian_condition'] else None,
                             boundary_hit=r['boundary_hit']=='True',boundary_parameters=json.loads(r['boundary_parameters']),
                             n_starts=int(r['n_starts']),n_successful_starts=int(r['n_successful_starts']),error=r['error']))
        completed=max(r['replicate'] for r in rows)
        if len(rows)!=completed*len(cfg['models']):raise ValueError('Incomplete bootstrap checkpoint')
        if any({r['model'] for r in rows if r['replicate']==i}!=set(cfg['models']) for i in range(1,completed+1)):
            raise ValueError('Checkpoint model/replicate gaps')
        expected=completed*len(cfg['models'])*cfg['bootstrap_multistart_count'];retained=0
        temp=out/'bootstrap_resume_verified_prefix.csv'
        with (out/'bootstrap_multistart_solutions.csv').open(encoding='utf-8-sig',newline='') as f,temp.open('w',encoding='utf-8-sig',newline='') as g:
            rd=csv.DictReader(f);wr=csv.DictWriter(g,fieldnames=cols);wr.writeheader()
            for r in rd:
                if retained==expected:break
                if None in r or any(v is None for v in r.values()) or int(r['replicate'])>completed:raise ValueError('Invalid checkpoint audit prefix')
                wr.writerow(r);retained+=1
        if retained!=expected:raise ValueError('Missing multistart audit in checkpoint')
        temp.replace(out/'bootstrap_multistart_solutions.csv')
        jobs=[job for job in jobs if job[0]>completed]
        logging.info('Resuming from %d completed draws, preserving %d start records; BLAS threads per worker=1',completed,retained)
        from .io import write_json
        write_json(out/'bootstrap_resume_runtime.json',dict(preserved_replicates=completed,preserved_multistart_records=retained,
                   change='BLAS threads per worker reduced to 1; model/bounds/seed/starts/resamples unchanged'))
    initialize_worker(t,p,reference,cfg)
    executor=None
    if cfg['bootstrap_workers']>1:
        executor=ProcessPoolExecutor(max_workers=cfg['bootstrap_workers'],initializer=initialize_worker,initargs=(t,p,reference,cfg))
        results=executor.map(bootstrap_worker,jobs,chunksize=1)
    else:results=map(bootstrap_worker,jobs)
    try:
        with (out/'bootstrap_multistart_solutions.csv').open('a' if resume else 'w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=cols)
            if not resume:writer.writeheader()
            import json
            for rep,rr,starts in results:
                rows.extend(rr)
                writer.writerows({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in starts)
                if rep%cfg['bootstrap_log_interval']==0:
                    f.flush();write_csv(out/'bootstrap_replicates.csv',rows)
                    logging.info('ND block bootstrap %d/%d, %d model fits, %d starts per model',rep,cfg['bootstrap_repetitions'],len(rows),cfg['bootstrap_multistart_count'])
    finally:
        if executor is not None:executor.shutdown(wait=True,cancel_futures=True)
    write_csv(out/'bootstrap_replicates.csv',rows)
    for m in cfg['models']:
        valid=[r for r in rows if r['model']==m and r['success']]
        if not valid:raise RuntimeError('All block-bootstrap fits failed: '+m)
        for key in PARAMETERS[m]:
            v=np.array([r['parameters'][key] for r in valid]);lo,hi=cfg['parameter_bounds'][key]
            hit=np.array([key in r['boundary_parameters'] for r in valid])
            params.append(dict(model=m,parameter=key,n_requested=cfg['bootstrap_repetitions'],n_success=len(v),
                               mean=float(v.mean()),median=float(np.median(v)),std=float(v.std(ddof=1)),
                               P025=float(np.quantile(v,.025)),P975=float(np.quantile(v,.975)),
                               failure_rate=1-len(valid)/cfg['bootstrap_repetitions'],parameter_boundary_rate=float(hit.mean()),
                               model_any_boundary_rate=sum(r['boundary_hit'] for r in valid)/len(valid),
                               lower_bound=lo,upper_bound=hi,bootstrap_unit='exact_ND_pair',fixed_classical_parameters=True))
    correlations=parameter_correlation_rows(rows,cfg)
    write_csv(out/'bootstrap_parameters.csv',params);write_csv(out/'parameter_correlations.csv',correlations)
    return params,rows,correlations


def parameter_correlation_rows(rows,cfg):
    """Do not interpret floating noise at a fixed active bound as covariance."""
    result=[]
    for m in cfg['models']:
        valid=[r for r in rows if r['model']==m and r['success']]
        a=np.array([[r['parameters'][key] for key in PARAMETERS[m]] for r in valid])
        usable={}
        for j,key in enumerate(PARAMETERS[m]):
            fixed_bound=all(key in r['boundary_parameters'] for r in valid) and np.ptp(a[:,j])<=cfg['boundary_absolute_tolerance']
            spread=np.std(a[:,j])>np.finfo(float).eps*max(1,abs(np.mean(a[:,j])))
            usable[key]=not fixed_bound and spread
        for j,key in enumerate(PARAMETERS[m]):
            for l,other in enumerate(PARAMETERS[m]):
                defined=usable[key] and usable[other]
                result.append(dict(model=m,parameter_1=key,parameter_2=other,
                                   Pearson=float(np.corrcoef(a[:,j],a[:,l])[0,1]) if defined else None,
                                   Spearman=float(spearmanr(a[:,j],a[:,l]).statistic) if defined else None,
                                   n=len(valid),defined=bool(defined),
                                   undefined_reason='' if defined else 'At least one parameter is numerically constant or fixed at an active bound'))
    return result


def sensitivity_surfaces(t,p,reference,fitted,out,domain_rows,cfg):
    """Analytic/finite-difference Q gradients and within-design N/D trend checks."""
    rows=[];checks=[];trends=[];domain=[]
    for m,f in fitted.items():
        params=f['parameters'];grad=marginal_quality(t,p,reference,m,params);yp=predict_scale(t,p,reference,m,params)
        qt={**t,'Q':np.ones(len(t['Q']))};refpred=predict_scale(qt,p,reference,m,params)
        step=1e-5
        plus={**t,'Q':np.minimum(t['Q']+step,1.)};minus={**t,'Q':np.maximum(t['Q']-step,t['Q'].min())}
        numeric=(predict_scale(plus,p,reference,m,params)-predict_scale(minus,p,reference,m,params))/(plus['Q']-minus['Q'])
        checks.append(dict(model=m,n=len(grad),analytic_violation_count=int((grad>=0).sum()),numerical_violation_count=int((numeric>=0).sum()),
                           analytic_violation_rate=float((grad>=0).mean()),numerical_violation_rate=float((numeric>=0).mean()),
                           max_abs_analytic_numeric_discrepancy=float(max(abs(grad-numeric))),
                           min_derivative=float(grad.min()),max_derivative=float(grad.max())))
        for i in range(len(t['L'])):
            gamma=params['gamma0']*(t['N'][i]/reference['N0'])**(-params['u'])*(t['D'][i]/reference['D0'])**params['v'] if m=='S1' else None
            rows.append(dict(model=m,N_params_B=float(t['N'][i]),D_tokens_B=float(t['D'][i]),Q_score=float(t['Q'][i]),
                             effective_gamma=gamma,dL_dQ=float(grad[i]),abs_quality_sensitivity=float(abs(grad[i])),
                             numerical_dL_dQ=float(numeric[i]),Delta_Loss_relative_Q1=float(yp[i]-refpred[i])))
        # Assess actual |dL/dQ|; do not mistake increasing gamma for increasing marginal Loss sensitivity.
        for axis in ['N','D']:
            other='D' if axis=='N' else 'N';g=defaultdict(list)
            for i in range(len(grad)):g[(float(t[other][i]),float(t['Q'][i]))].append(i)
            relative=[]
            for idx in g.values():
                idx=sorted(idx,key=lambda i:t[axis][i]);v=abs(grad[idx]);relative.extend((np.diff(v)/np.maximum(v[:-1],1e-300)).tolist())
            a=np.array(relative);tol=1e-8
            trends.append(dict(model=m,axis=axis,n_adjacent_pairs=len(a),decreasing_rate=float((a< -tol).mean()),
                               increasing_rate=float((a>tol).mean()),flat_rate=float((abs(a)<=tol).mean()),
                               min_relative_change=float(a.min()),max_relative_change=float(a.max()),comparison_numerical_tolerance=tol))
        design=[(float(n),float(d)) for n in np.unique(t['N']) for d in np.unique(t['D'])]
        # Fixed domain scores are used only as numeric Q coordinates, never in a mixture.
        coordinates=[dict(domain=r['domain'],Q=float(r['Q_domain'])) for r in domain_rows]
        low=min(x['Q'] for x in coordinates);high=max(x['Q'] for x in coordinates)
        coordinates += [dict(domain='Q_interval_low',Q=low),dict(domain='Q_interval_high',Q=high)]
        for coord in coordinates:
            tt=dict(N=np.array([n for n,d in design]),D=np.array([d for n,d in design]),Q=np.full(len(design),coord['Q']))
            derivative=marginal_quality(tt,p,reference,m,params);prediction=predict_scale(tt,p,reference,m,params)
            baseline=predict(p,tt['N'],tt['D'])
            for i,(n,d) in enumerate(design):
                domain.append(dict(model=m,domain_coordinate=coord['domain'],N_params_B=n,D_tokens_B=d,Q_coordinate=coord['Q'],
                                    dL_dQ=float(derivative[i]),abs_quality_sensitivity=float(abs(derivative[i])),
                                    Delta_Loss_relative_Q1=float(prediction[i]-baseline[i]),
                                    numeric_support_only=True,semantic_Q_equivalence_confirmed=False))
    write_csv(out/'quality_sensitivity_surface.csv',rows);write_csv(out/'monotonicity_constraint_checks.csv',checks)
    write_csv(out/'quality_sensitivity_scale_trends.csv',trends);write_csv(out/'question1_Q_interval_analysis.csv',domain)
    return rows,checks,trends,domain

# ============================================================
# Source: src/scaling/quality_analysis.py
# Original lines: 1-240
# Purpose: Phase2C复用B8压力检验工具；MQ候选不是最终模型
# MQ candidate functions are not final main model; Phase2C calls stress_test only.
# ============================================================
"""ND-group validation and reproducible block bootstrap for quality extensions."""
from __future__ import annotations
import csv
import logging
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold
from .quality_extension import MODELS, fit_quality, predict_quality, subset, group_reference
from .classical import metrics
from .io import write_csv, write_json


def evaluate(y, pred, **labels):
    """Metrics in raw Loss; bias is actual minus prediction."""
    return dict(**labels, **metrics(y,pred))


def cross_validate(t, p, cfg, out):
    """Each exact ND block appears wholly in one test fold; no full-fit warm starts."""
    n=len(t['L']); oof={m:np.full(n,np.nan) for m in MODELS}
    folds=[];fit_rows=[];starts=[];fold_metrics=[];fold_of_row=np.zeros(n,dtype=int)
    splitter=GroupKFold(n_splits=cfg['group_cv_folds'])
    for fold,(train,test) in enumerate(splitter.split(t['N'],groups=t['group']),1):
        tr,te=subset(t,train),subset(t,test)
        train_groups,test_groups=sorted(set(tr['group'])),sorted(set(te['group']))
        if set(train_groups)&set(test_groups):raise AssertionError('ND leakage')
        fold_of_row[test]=fold
        folds.append(dict(fold=fold,train_rows=len(train),test_rows=len(test),
                          train_groups=train_groups,test_groups=test_groups))
        for model in MODELS:
            if model=='M0':gamma=offset=0.0
            else:
                fit,runs=fit_quality(tr,p,model,cfg);gamma,offset=fit['gamma'],fit['offset']
                fit_rows.append(dict(stage='GroupCV',fold=fold,**fit))
                starts.extend(dict(stage='GroupCV',model=model,fold=fold,**r) for r in runs)
            yp=predict_quality(te,p,model,gamma,offset);oof[model][test]=yp
            fold_metrics.append(evaluate(te['L'],yp,model=model,fold=fold))
    if any(not np.isfinite(v).all() for v in oof.values()):raise AssertionError('Incomplete OOF predictions')
    predictions=[]
    for i in range(n):
        for m in MODELS:
            predictions.append(dict(row_index=i,ND_group=t['group'][i],fold=int(fold_of_row[i]),
                                    N_params_B=float(t['N'][i]),D_tokens_B=float(t['D'][i]),Q_score=float(t['Q'][i]),
                                    model=m,actual_Loss=float(t['L'][i]),predicted_Loss=float(oof[m][i])))
    write_csv(out/'group_cv_predictions.csv',predictions)
    write_csv(out/'group_cv_folds.csv',folds)
    write_csv(out/'group_cv_model_parameters.csv',fit_rows)
    write_csv(out/'group_cv_fold_metrics.csv',fold_metrics)
    comparison=[evaluate(t['L'],oof[m],dataset='B6',evaluation='5fold_ND_GroupCV',model=m,
                         role='source_offset_sensitivity' if m.endswith('_offset') else ('main_candidate' if m=='MQ1' else 'baseline_or_form_sensitivity')) for m in MODELS]
    write_csv(out/'quality_model_comparison.csv',comparison)
    return oof, comparison, starts


def q_level_cv(t,p,cfg,out):
    """All rows at a Q level are held out; endpoint tests are Q extrapolation."""
    rows=[]; starts=[];predictions=[]
    levels=np.unique(t['Q'])
    for q in levels:
        train=t['Q']!=q;test=~train
        tr,te=subset(t,train),subset(t,test)
        for m in MODELS:
            if m=='M0':gamma=offset=0.0
            else:
                fit,runs=fit_quality(tr,p,m,cfg);gamma,offset=fit['gamma'],fit['offset']
                starts.extend(dict(stage='leave_one_Q_out',held_Q=float(q),model=m,**r) for r in runs)
            yp=predict_quality(te,p,m,gamma,offset)
            rows.append(evaluate(te['L'],yp,held_Q=float(q),model=m,gamma=gamma,offset=offset,
                                 Q_test_kind='endpoint_extrapolation' if q in [levels[0],levels[-1]] else 'interior_interpolation'))
            predictions.extend(dict(model=m,held_Q=float(q),ND_group=g,actual_Loss=float(y),predicted_Loss=float(v)) for g,y,v in zip(te['group'],te['L'],yp))
    write_csv(out/'leave_one_Q_level_out.csv',rows)
    write_csv(out/'leave_one_Q_level_out_predictions.csv',predictions)
    return rows,starts


def layer_fit(t,p,cfg,out):
    """Separate one-gamma fits by N and D; offset fits diagnose confounding only."""
    stats=[];starts=[];tables={}
    for axis,file in [('N','gamma_by_scale.csv'),('D','gamma_by_data.csv')]:
        rows=[]
        for level in np.unique(t[axis]):
            a=subset(t,t[axis]==level)
            if len(np.unique(a['Q']))<2 or len(set(a['group']))<2:continue
            for m in ['MQ1','MQ1_offset']:
                fit,runs=fit_quality(a,p,m,cfg)
                yp=predict_quality(a,p,m,fit['gamma'],fit['offset'])
                rows.append(evaluate(a['L'],yp,axis=axis,level=float(level),model=m,
                                     gamma=fit['gamma'],eta=p['beta']*fit['gamma'],offset=fit['offset'],
                                     n_ND_groups=len(set(a['group'])),n_Q_levels=len(np.unique(a['Q']))))
                starts.extend(dict(stage='layer_fit',axis=axis,level=float(level),model=m,**r) for r in runs)
        write_csv(out/file,rows);tables[axis]=rows
        for m in ['MQ1','MQ1_offset']:
            r=[x for x in rows if x['model']==m]
            x=np.log([x['level'] for x in r]);y=np.array([x['gamma'] for x in r])
            stats.append(dict(axis=axis,model=m,n_layers=len(r),gamma_min=float(y.min()),gamma_max=float(y.max()),
                              gamma_mean=float(y.mean()),gamma_std=float(y.std(ddof=1)),
                              Pearson_gamma_logscale=float(pearsonr(x,y).statistic),
                              Spearman_gamma_logscale=float(spearmanr(x,y).statistic)))
    write_csv(out/'gamma_scale_dependence.csv',stats)
    return tables,stats,starts


def bootstrap(t,p,cfg,out,oof):
    """2000 whole ND-block resamples, paired across model forms and offset versions.

    Every draw uses all seven initializations. Rows carry block multiplicity;
    no IID-point bootstrap is used. Classical coefficients remain fixed, so
    these intervals are conditional on the frozen beta and source mechanism.
    """
    groups=np.unique(t['group']);G=len(groups);rng=np.random.default_rng(cfg['random_state'])
    rows=[];draws=[];summary=[];fitmodels=cfg['bootstrap_models'];paired=[]
    startcols=['replicate','model','start','initial_gamma','initial_offset','gamma','offset','objective','successful','legal_solution','status','nfev','optimality','at_bound','message']
    with (out/'bootstrap_multistart_solutions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=startcols);writer.writeheader()
        for rep in range(1,cfg['bootstrap_repetitions']+1):
            counts=rng.multinomial(G,np.full(G,1/G))
            multiplicities=dict(zip(groups,counts))
            draws.append(dict(replicate=rep,ND_groups=groups.tolist(),multiplicities=counts.tolist(),n_sampled_blocks=int(counts.sum()),n_distinct_blocks=int((counts>0).sum())))
            for model in fitmodels:
                try:
                    fit,runs=fit_quality(t,p,model,cfg,multiplicities)
                    writer.writerows(dict(replicate=rep,model=model,**r) for r in runs)
                    rows.append(dict(replicate=rep,model=model,success=True,gamma=fit['gamma'],
                                     eta=p['beta']*fit['gamma'] if model.startswith('MQ1') else None,
                                     offset=fit['offset'],objective=fit['objective'],at_bound=fit['at_bound'],
                                     n_distinct_blocks=int((counts>0).sum()),n_starts=fit['n_starts'],
                                     all_starts_converged=all(r['legal_solution'] for r in runs),error=''))
                except (RuntimeError,ValueError,FloatingPointError) as ex:
                    rows.append(dict(replicate=rep,model=model,success=False,gamma=None,eta=None,offset=None,objective=None,
                                     at_bound=None,n_distinct_blocks=int((counts>0).sum()),n_starts=len(cfg['gamma_starts']),
                                     all_starts_converged=False,error=str(ex)))
            ww=np.array([multiplicities[g] for g in t['group']],dtype=float);ww/=ww.sum()
            rmse={m:float(np.sqrt(np.sum(ww*(t['L']-v)**2))) for m,v in oof.items()}
            paired.append(dict(replicate=rep,**{m+'_RMSE':v for m,v in rmse.items()},
                               MQ1_RMSE_gain_over_M0=rmse['M0']-rmse['MQ1'],
                               MQ2_RMSE_gain_over_MQ1=rmse['MQ1']-rmse['MQ2']))
            if rep%cfg['bootstrap_log_interval']==0:
                f.flush();logging.info('ND block bootstrap %d/%d; %d model fits',rep,cfg['bootstrap_repetitions'],len(rows))
                write_csv(out/'quality_bootstrap_replicates.csv',rows)
    for model in fitmodels:
        valid=[r for r in rows if r['model']==model and r['success']]
        for parameter in ['gamma','eta','offset']:
            if parameter=='eta' and not model.startswith('MQ1'):continue
            if parameter=='offset' and not model.endswith('_offset'):continue
            vals=np.array([r[parameter] for r in valid],dtype=float)
            if len(vals)==0:raise RuntimeError('All bootstrap fits failed')
            summary.append(dict(model=model,parameter=parameter,n_requested=cfg['bootstrap_repetitions'],n_success=len(vals),
                                failure_rate=1-len(vals)/cfg['bootstrap_repetitions'],mean=float(vals.mean()),
                                median=float(np.median(vals)),std=float(vals.std(ddof=1)),P025=float(np.quantile(vals,.025)),
                                P975=float(np.quantile(vals,.975)),bootstrap_unit='exact_ND_pair',random_state=cfg['random_state'],
                                fixed_classical_parameters=True,at_bound_count=sum(r['at_bound'] for r in valid)))
    write_csv(out/'quality_bootstrap_replicates.csv',rows)
    write_csv(out/'quality_parameter_bootstrap.csv',summary)
    write_csv(out/'bootstrap_block_draws.csv',draws)
    write_csv(out/'group_cv_paired_bootstrap.csv',paired)
    gains=[]
    for k in ['MQ1_RMSE_gain_over_M0','MQ2_RMSE_gain_over_MQ1']:
        v=np.array([r[k] for r in paired])
        gains.append(dict(comparison=k,mean=float(v.mean()),P025=float(np.quantile(v,.025)),P975=float(np.quantile(v,.975)),
                          fraction_positive=float((v>0).mean()),method='ND-block resampling of fixed OOF predictions; not a nested refit CI'))
    write_csv(out/'group_cv_improvement_uncertainty.csv',gains)
    return summary,rows,gains


def difference_table(t, predictions, masks, outrows, pointrows, label_prefix=''):
    """Validate differences within full source ND groups, primary excludes references."""
    for m,yp in predictions.items():
        obs,pred,qref,refs=group_reference(t,yp)
        for name,mask in masks.items():
            for include_refs in [False,True]:
                ix=mask if include_refs else mask&~refs
                if ix.sum()<2:continue
                outrows.append(evaluate(obs[ix],pred[ix],dataset=t['label'],subset=name,model=m,
                                        evaluation=label_prefix or 'fixed_B6_parameters',include_reference_zero_rows=include_refs,
                                        q_reference_rule='highest Q in full source ND group; no reference used for refitting'))
            for i in np.flatnonzero(mask&~refs):
                pointrows.append(dict(dataset=t['label'],subset=name,model=m,evaluation=label_prefix or 'fixed_B6_parameters',
                                      row_index=int(i),ND_group=t['group'][i],N_params_B=float(t['N'][i]),D_tokens_B=float(t['D'][i]),
                                      Q_score=float(t['Q'][i]),Q_ref=float(qref[i]),Delta_L_obs=float(obs[i]),Delta_L_pred=float(pred[i])))


def monotonicity_details(t,out):
    """Save B6 local rebounds and magnitude; do not confuse noise with reversed trend."""
    groups=defaultdict(list)
    for i,g in enumerate(t['group']):groups[g].append(i)
    rows=[];viol=[]
    for g,idx in groups.items():
        idx=sorted(idx,key=lambda i:t['Q'][i]);loss=t['L'][idx];q=t['Q'][idx];diff=np.diff(loss)
        positives=diff[diff>1e-12]
        rows.append(dict(ND_group=g,n_Q=len(idx),spearman=float(spearmanr(q,loss).statistic),
                         strictly_decreasing=bool((diff< -1e-12).all()),violation_count=len(positives),
                         max_local_rebound=float(positives.max()) if len(positives) else 0.,
                         mean_local_rebound=float(positives.mean()) if len(positives) else 0.))
        for j in np.flatnonzero(diff>1e-12):
            viol.append(dict(ND_group=g,Q_lower=float(q[j]),Q_higher=float(q[j+1]),Loss_lower=float(loss[j]),
                             Loss_higher=float(loss[j+1]),rebound_magnitude=float(diff[j])))
    write_csv(out/'B6_monotonicity_details.csv',rows)
    write_csv(out/'B6_local_violations.csv',viol)
    return dict(n_groups=len(rows),negative_correlation_group_rate=sum(r['spearman']<0 for r in rows)/len(rows),
                strictly_monotone_group_rate=sum(r['strictly_decreasing'] for r in rows)/len(rows),
                local_violation_count=len(viol),mean_rebound=float(np.mean([r['rebound_magnitude'] for r in viol])),
                max_rebound=max(r['rebound_magnitude'] for r in viol))


def stress_test(t,b6,p,predictions,keys6,keys7,out):
    """B8 is solely an incompatibility pressure test; never used for fitting/selection."""
    inside=np.ones(len(t['L']),dtype=bool)
    for k in ['N','D','Q']:inside&=(t[k]>=b6[k].min())&(t[k]<=b6[k].max())
    nonoverlap=np.array([k not in keys6|keys7 for k in t['keys']])
    masks={'full':np.ones(len(t['L']),dtype=bool),'nonoverlap':nonoverlap,
           'within_B6_numeric_ranges':inside,'outside_B6_numeric_ranges':~inside,
           'nonoverlap_within_ranges':inside&nonoverlap,'nonoverlap_outside_ranges':~inside&nonoverlap}
    diagnostics=[];group_rows=[];performance=[]
    for name,mask in masks.items():
        valid_rhos=[];insufficient=0;local=[];groups=defaultdict(list)
        for i in np.flatnonzero(mask):groups[t['group'][i]].append(i)
        for g,idx in groups.items():
            q,l=t['Q'][idx],t['L'][idx]
            if len(np.unique(q))<2 or np.ptp(l)==0:insufficient+=1;continue
            rho=float(spearmanr(q,l).statistic);valid_rhos.append(rho)
            group_rows.append(dict(subset=name,ND_group=g,n_Q=len(idx),spearman_Q_Loss=rho))
            ix=sorted(idx,key=lambda i:t['Q'][i]);local.extend(np.diff(t['L'][ix]).tolist())
        rv=np.array(valid_rhos);lv=np.array(local)
        diagnostics.append(dict(subset=name,n=int(mask.sum()),n_ND_groups=len(groups),n_groups_with_correlation=len(rv),
                                n_groups_insufficient_Q=insufficient,positive_groups=int((rv>0).sum()),negative_groups=int((rv<0).sum()),
                                positive_group_rate=float((rv>0).mean()) if len(rv) else None,
                                spearman_min=float(rv.min()) if len(rv) else None,spearman_P25=float(np.quantile(rv,.25)) if len(rv) else None,
                                spearman_median=float(np.median(rv)) if len(rv) else None,spearman_mean=float(rv.mean()) if len(rv) else None,
                                spearman_P75=float(np.quantile(rv,.75)) if len(rv) else None,spearman_max=float(rv.max()) if len(rv) else None,
                                loss_below_E_count=int((t['L'][mask]<p['E']).sum()),loss_below_E_rate=float((t['L'][mask]<p['E']).mean()),
                                increasing_adjacent_pair_rate=float((lv>1e-12).mean()) if len(lv) else None,
                                role='incompatible_semisynthetic_stress_test'))
        for m,yp in predictions.items():
            performance.append(evaluate(t['L'][mask],yp[mask],dataset='B8',subset=name,model=m,
                                        role='incompatible_semisynthetic_stress_test'))
    write_csv(out/'B8_incompatibility_diagnostics.csv',diagnostics)
    write_csv(out/'B8_subset_group_correlations.csv',group_rows)
    write_csv(out/'B8_range_validation.csv',performance)
    return diagnostics,performance,masks

# ============================================================
# Source: src/scaling/quality_reporting.py
# Original lines: 1-204
# Purpose: Phase2C复用绘图和表格工具
# ============================================================
"""Figures and a numerically generated report with explicit evidence boundaries."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .quality_extension import predict_quality,group_reference,subset


def save(fig, folder, name, dpi):
    """Save a standalone scientific figure and release its memory."""
    fig.tight_layout();fig.savefig(folder/name,dpi=dpi);plt.close(fig)


def obs_pred(ax, y, yp, title, q=None):
    """Observed/predicted scatter with common limits and identity line."""
    s=ax.scatter(y,yp,c=q,s=17,alpha=.7,cmap='viridis' if q is not None else None)
    lo=min(float(np.min(y)),float(np.min(yp)));hi=max(float(np.max(y)),float(np.max(yp)))
    margin=(hi-lo)*.04 or .01
    ax.plot([lo-margin,hi+margin],[lo-margin,hi+margin],'k--',lw=1)
    ax.set(xlabel='Observed Loss',ylabel='Predicted Loss',title=title,
           xlim=(lo-margin,hi+margin),ylim=(lo-margin,hi+margin))
    return s


def make_figures(root,ts,p,fits,oof,comparison,layers,brows,masks7,masks8,predictions,diffpoints,cfg):
    """Re-use prior coverage diagnosis and add all fit/validation evidence plots."""
    folder=root/'outputs/figures/question2_phase2';folder.mkdir(parents=True,exist_ok=True)
    dpi=cfg['figure_dpi'];b6=ts['B6'];b7=ts['B7'];b8=ts['B8']
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    Ns=np.unique(b6['N']);Ds=np.unique(b6['D'])
    for ax,(n,d) in zip(axes.flat,[(Ns[0],Ds[0]),(Ns[0],Ds[-1]),(Ns[-1],Ds[0]),(Ns[-1],Ds[-1])]):
        ix=(b6['N']==n)&(b6['D']==d);a=subset(b6,ix);order=np.argsort(a['Q'])
        ax.plot(a['Q'][order],a['L'][order],'ko',label='B6 observations')
        fine=dict(N=np.full(100,n),D=np.full(100,d),Q=np.linspace(b6['Q'].min(),b6['Q'].max(),100))
        for model in ['M0','MQ1','MQ2']:
            f=fits.get(model,dict(gamma=0.,offset=0.))
            ax.plot(fine['Q'],predict_quality(fine,p,model,f['gamma'],f['offset']),label=model)
        ax.set(xlabel='Q_score',ylabel='Validation Loss',title=f'N={n:g} B, D={d:g} B');ax.legend(fontsize=8)
    save(fig,folder,'04_B6_fixed_ND_quality_curves.png',dpi)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,metric in zip(axes,['RMSE','MAE']):
        ax.bar([r['model'] for r in comparison],[r[metric] for r in comparison],color=['gray','tab:blue','tab:orange','lightblue','wheat'])
        ax.set(ylabel=metric+' (raw Loss)',title='B6: five-fold ND Group CV');ax.tick_params(axis='x',rotation=25)
    save(fig,folder,'05_B6_group_CV_comparison.png',dpi)
    fig,ax=plt.subplots(figsize=(6,5));s=obs_pred(ax,b6['L'],oof['MQ1'],'MQ1: B6 held-out ND groups',b6['Q']);fig.colorbar(s,ax=ax,label='Q_score')
    save(fig,folder,'06_MQ1_B6_observed_predicted_OOF.png',dpi)
    obs,pred,_,refs=group_reference(b6,oof['MQ1'])
    fig,ax=plt.subplots(figsize=(6,5));s=obs_pred(ax,obs[~refs],pred[~refs],'MQ1: B6 OOF within-ND differences',b6['Q'][~refs])
    ax.set(xlabel='Observed Delta Loss (vs Q=1)',ylabel='Predicted Delta Loss (vs Q=1)');fig.colorbar(s,ax=ax,label='Q_score')
    save(fig,folder,'07_within_ND_difference_OOF.png',dpi)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,model in zip(axes,['MQ1','MQ2']):
        for m in [model,model+'_offset']:
            v=[r['gamma'] for r in brows if r['model']==m and r['success']]
            ax.hist(v,bins=35,alpha=.55,label=m)
        ax.set(xlabel='gamma',ylabel='ND-block bootstrap count',title=model+': 2000 block resamples');ax.legend()
    save(fig,folder,'08_gamma_block_bootstrap.png',dpi)
    for axis,name in [('N','09_gamma_by_N.png'),('D','10_gamma_by_D.png')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for m in ['MQ1','MQ1_offset']:
            r=[x for x in layers[axis] if x['model']==m]
            ax.plot([x['level'] for x in r],[x['gamma'] for x in r],'o-',label=m)
        ax.set(xscale='log',xlabel=axis+(' (billion parameters)' if axis=='N' else ' (billion tokens)'),ylabel='Layer-specific gamma',title='B6: gamma by '+axis);ax.legend()
        save(fig,folder,name,dpi)
    ix=masks7['nonoverlap'];fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,m in zip(axes,['MQ1','MQ2']):
        s=obs_pred(ax,b7['L'][ix],predictions['B7'][m][ix],m+': B7 non-overlap (n='+str(int(ix.sum()))+')',b7['Q'][ix]);fig.colorbar(s,ax=ax,label='Q_score')
    save(fig,folder,'11_B7_nonoverlap_predictions.png',dpi)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,name in zip(axes,['within_B6_numeric_ranges','outside_B6_numeric_ranges']):
        ix=masks8[name];s=obs_pred(ax,b8['L'][ix],predictions['B8']['MQ1'][ix],name.replace('_',' ')+f' (n={ix.sum()})',b8['Q'][ix]);fig.colorbar(s,ax=ax,label='Q_score')
        ax.axvline(p['E'],color='tab:red',ls=':',label='Frozen E');ax.legend(fontsize=8)
    fig.suptitle('B8 incompatible semisynthetic pressure test (no fitting)',fontsize=12)
    save(fig,folder,'12_B8_inside_outside_pressure_test.png',dpi)
    from .io import csv_rows
    _,rows=csv_rows(root/'outputs/scaling/phase2/leave_one_Q_level_out.csv')
    fig,ax=plt.subplots(figsize=(7,4))
    for m in ['MQ1','MQ2']:
        r=[x for x in rows if x['model']==m]
        ax.plot([float(x['held_Q']) for x in r],[float(x['RMSE']) for x in r],'o-',label=m)
    ax.set(xlabel='Held-out Q level',ylabel='RMSE (raw Loss)',title='B6: leave-one-Q-level-out; endpoints extrapolate');ax.legend()
    save(fig,folder,'13_leave_one_Q_level_out.png',dpi)
    fig,ax=plt.subplots(figsize=(8,4));ix=b6['Q']==1
    for m in ['M0','MQ1_offset','MQ2_offset']:
        ax.hist(b6['L'][ix]-predictions['B6'][m][ix],bins=16,alpha=.5,label=m)
    ax.axvline(0,color='black',lw=1);ax.set(xlabel='Observed - predicted Loss at Q=1',ylabel='ND-group count',title='B6 source-offset anchor sensitivity');ax.legend()
    save(fig,folder,'14_source_offset_anchor_sensitivity.png',dpi)
    fig,ax=plt.subplots(figsize=(7,4));ix=(b8['N']==.07)&(b8['D']==50)&masks8['within_B6_numeric_ranges']
    if ix.sum()<2:
        g=b8['group'][np.flatnonzero(masks8['within_B6_numeric_ranges'])[0]];ix=(b8['group']==g)&masks8['within_B6_numeric_ranges']
    idx=np.flatnonzero(ix);idx=idx[np.argsort(b8['Q'][idx])]
    ax.plot(b8['Q'][idx],b8['L'][idx],'o-',label='B8 observations (unchanged)')
    ax.plot(b8['Q'][idx],predictions['B8']['MQ1'][idx],'o-',label='Fixed B6 MQ1 prediction')
    ax.set(xlabel='Q_score',ylabel='Loss',title=f'B8 inside numeric ranges: N={b8["N"][idx[0]]:g} B, D={b8["D"][idx[0]]:g} B');ax.legend()
    save(fig,folder,'15_B8_within_range_direction_reversal.png',dpi)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,m in zip(axes,['MQ1','MQ2']):
        obs,pred,_,refs=group_reference(b7,predictions['B7'][m]);ix=masks7['nonoverlap']&~refs
        s=obs_pred(ax,obs[ix],pred[ix],m+': B7 new-Q quality differences',b7['Q'][ix])
        ax.set(xlabel='Observed Delta Loss (vs Q=1)',ylabel='Predicted Delta Loss (vs Q=1)');fig.colorbar(s,ax=ax,label='New Q level')
    save(fig,folder,'16_B7_nonoverlap_quality_differences.png',dpi)


def table(rows, columns, names=None):
    """Compact markdown table with sufficient displayed precision."""
    names=names or columns
    text=['|'+'|'.join(names)+'|','|'+'|'.join(['---']*len(columns))+'|']
    for r in rows:
        vals=[]
        for c in columns:
            v=r.get(c)
            vals.append(f'{v:.8g}' if isinstance(v,float) else ('n.a.' if v is None else str(v)))
        text.append('|'+'|'.join(vals)+'|')
    return text


def make_report(root,s,bs,diffrows,anchors):
    """Report the final evidence chain, including form/offset limitations."""
    p=s['classical_parameters'];fits=s['fit_models'];g=fits['MQ1']['gamma'];eta=p['beta']*g
    ci=next(r for r in bs if r['model']=='MQ1' and r['parameter']=='gamma')
    out=root/'outputs/scaling/phase2'
    lines=['# 问题二 PHASE 2B：质量 Q 标度律与适用边界','',
           '状态：本轮实验完成，等待人工确认。经典五参数和问题一全部结果保持冻结。', '',
           ('结论：本轮不建议冻结 L(N,D,Q)。MQ1 明显改善绝对 Loss 的 ND 分组预测，但 B7 新增 Q 水平上的组内差分预测未通过，且统一 γ 存在明显 N/D 层间变化。' if not s['freeze_recommended'] else '结论：必要证据通过，可提交支持域内近似质量模型供人工确认。'),
           f"MQ1 在 B7_nonoverlap 的绝对 Loss R²={s['B7_primary_absolute']['R2']:.8g}，组内差分 R²={s['B7_primary_difference']['R2']:.8g}、平均低估量={s['B7_primary_difference']['mean_bias']:.8g}。应以组内差分作为质量效应量复核，不能用跨 N/D 的绝对预测相关代替。", '',
           '## 证据层级与来源','',
           'B6 为唯一 γ 拟合与模型选择数据。B7 的 90 条非重叠记录为主要扩展验证，360 条重叠记录和 450 条全量结果仅作描述。B8 单列为“大范围半合成不兼容压力测试/负面证据”。',
           'B6–B8 统一称“基于真实数据校准的半合成补充实验”，不得据此宣称真实直接训练的跨源普适定律。',
           '主键使用 Decimal 精确规范化的 (N,D,Q)，不以原始字符串格式区别 0.07 和 0.070。诊断和重叠表沿用已确认结果。',
           'B7_nonoverlap 排除 B6 主键；B8_nonoverlap 排除 B6∪B7 主键。B7 新增 Q 为 0.5、0.7，ND 工作点仍与 B6 共享，因此主要检验新 Q 水平而非新 ND 外部泛化。', '',
           '## 冻结基线与有效数据量假设','',
           '```text',f"E={p['E']:.17g}\nA={p['A']:.17g}\nB={p['B']:.17g}\nalpha={p['alpha']:.17g}\nbeta={p['beta']:.17g}",'```','',
           r'\(L_0=E+A N^{-\alpha}+B D^{-\beta}\)，N 与 D 使用 billions 单位，拟合原始 Loss。',
           r'主候选 MQ1：\(D_{eff}=D Q^\gamma\)，\(L=E+A N^{-\alpha}+B D^{-\beta}Q^{-\beta\gamma}\)。',
           r'函数形式敏感性 MQ2：\(D_{eff}=D\exp[\gamma(Q-1)]\)，数据项为 \(B D^{-\beta}\exp[\beta\gamma(1-Q)]\)。',
           '两者均 γ>0，无 offset 时在 Q=1 精确退化为 L0。不得从拟合成功推断 D_eff 是已被真实训练实验证明的因果有效 token 数。',
           'source offset 版本仅作为来源/函数形式混杂敏感性，不替换无 offset 主结果。', '',
           '## 拟合、交叉验证与不确定性协议','',
           'bounded nonlinear least squares；γ 初值 0.1、0.25、0.5、1、2、4、8，每次比较全部合法成功解的原始 Loss MSE。计算边界 γ∈[1e-8,50]，offset∈[-10,10]；边界不是数据拟合结论，须检查是否触边。',
           '每个原始观测等权。完整网格中每个 ND 组均有 8 个 Q，因此各 ND 总权也相等。Bootstrap 对 45 个 ND 块有放回采样，整块所有 Q 带同一重数。',
           '5-fold GroupKFold 按精确 ND 分组，同一 ND 的全部 Q 在同一折。各训练折独立使用七个预设初值，不使用全量拟合参数作初值。',
           'leave-one-Q-level-out 每次整层 Q 留出。边界 0.1/1 为 Q 外推敏感性，其余为插值。',
           '2000 次 ND block bootstrap，固定 seed=2026；四个候选版本采用同一批整组重采样。置信区间条件于冻结的经典参数、固定的 Q 网格和半合成来源；不包含 PHASE 1 β 不确定性或真实跨源误差。',
           'OOF RMSE 改善区间来自固定 OOF 预测的 ND 重采样，不是完整嵌套 CV 重拟合置信区间，未使用 IID 点级显著性检验。', '',
           '## B6 质量方向与局部噪声','',
           f"45 组负相关比例 {s['B6_monotonicity']['negative_correlation_group_rate']:.2%}；严格逐点下降比例 {s['B6_monotonicity']['strictly_monotone_group_rate']:.2%}。",
           f"共 {s['B6_monotonicity']['local_violation_count']} 个相邻反弹，平均幅度 {s['B6_monotonicity']['mean_rebound']:.8g}，最大 {s['B6_monotonicity']['max_rebound']:.8g}。局部反弹不等于总体质量方向错误，也不据此更改 γ 符号。每组相关与幅度见 B6_monotonicity_details.csv。", '',
           '## B6 主参数与 Bootstrap','',
           f'无 offset MQ1 γ={g:.12g}，95% CI=[{ci["P025"]:.12g},{ci["P975"]:.12g}]；η=βγ={eta:.12g}。', '']
    lines+=table([dict(model=m,gamma=f['gamma'],eta=p['beta']*f['gamma'] if m.startswith('MQ1') else None,
                       offset=f['offset'],fit_RMSE=np.sqrt(f['objective']),n_starts=f['n_starts'],gamma_spread=f['distinct_solution_gamma_range']) for m,f in fits.items()],
                 ['model','gamma','eta','offset','fit_RMSE','n_starts','gamma_spread'])
    lines+=['']+table(bs,['model','parameter','mean','median','std','P025','P975','failure_rate','at_bound_count'])
    lines+=['','## B6 按 ND 分组的预测性能','']
    lines+=table(s['B6_group_cv'],['model','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias'])
    lines += ['',f"MQ1 相对 M0 的 Group-CV RMSE 降低 {s['CV_RMSE_gain']:.2%}。仅按 B6 CV、无 offset 两种函数形式的数值赢家为 {s['B6_numeric_form_winner']}；MQ1 仍按用户指定保留为主候选，MQ2 为形式敏感性，不依据 B7/B8 更换主模型。",'']
    lines+=table(s['bootstrap_CV_gains'],['comparison','mean','P025','P975','fraction_positive'])
    lines+=['','## Source offset 与 Q=1 锚点','']
    lines+=table(s['source_offset'],['model','gamma_no_offset','gamma_with_offset','gamma_relative_change','delta_B6'])
    lines+=['']+table(anchors,['model','n','RMSE','MAE','mean_bias','median_residual'])
    lines+=['','offset 对 γ 的影响应结合数值变化和锚点表现解释。全 Q 拟合的 offset 不必等于 Q=1 平均偏差；若其改善全量损失却恶化 Q=1 锚点，它也可能吸收有效数据量函数的形状误差，不能将其全部解释成纯来源零点偏移。',
            '本轮保留无 offset 主结果，同时报告“质量效应与来源偏差存在部分混杂风险”；不能仅因均值锚点偏差小就省略敏感性。', '',
            '## 组内差分：质量效应关键复核','',
            r'B6/B7 均以每个 ND 的 Q=1 为参考。\(\Delta\hat L=B D^{-\beta}(Q^{-\beta\gamma}-1)\)，使用同一 γ，不单独拟合差分。E 和 N 项精确抵消。',
            '主差分统计排除参考点自身的零差分；含零参考点的结果保留为辅助表。B6 同时报告全量拟合和 held-ND OOF 差分。',
            'B7_nonoverlap 的 Q=1 实测参考来自全量 B7、与 B6 重叠。这是组内变化验证，参考并非新增独立观测；不把共享参考包装成独立外部实验。', '']
    selected=[r for r in diffrows if r['model'] in ['M0','MQ1','MQ2'] and not r['include_reference_zero_rows'] and (r['dataset']=='B6' and r['evaluation']=='B6_GroupCV_out_of_fold' or r['dataset']=='B7' and r['subset']=='nonoverlap')]
    lines+=table(selected,['dataset','subset','evaluation','model','n','RMSE','MAE','R2','Pearson','Spearman'])
    b7q2=next(r for r in diffrows if r['dataset']=='B7' and r['subset']=='nonoverlap' and r['model']=='MQ2' and not r['include_reference_zero_rows'])
    lines+=['',f"B7 非重叠差分 MQ1 R²={s['B7_primary_difference']['R2']:.8g}、MQ2 R²={b7q2['R2']:.8g}，均未优于该差分子集的均值预测。MQ2 的 B6 数值优势不能据此变成质量模块已通过扩展验证的结论。"]
    lines+=['','正 R² 表示差分预测优于均值差分，但不足以证明零误差或普遍因果规律。误差大小、形式差异与共享参考依赖需同时保留。', '',
            '## B7 扩展验证：固定全部 B6 参数','']
    lines+=table(s['B7_validation'],['subset','model','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias'])
    lines+=['','模型和参数在读入本轮 B7/B8 评价数据前已写入 parameters_before_B7_B8.json；未用验证结果调参。', '',
            '## 未见 Q 水平泛化','']
    lines+=table([r for r in s['leave_one_Q_level_out'] if r['model'] in ['MQ1','MQ2']],['held_Q','Q_test_kind','model','gamma','RMSE','MAE','mean_bias'])
    endpoints=[r for r in s['leave_one_Q_level_out'] if r['model']=='MQ1' and r['held_Q'] in [.1,1.]]
    interior=[r['RMSE'] for r in s['leave_one_Q_level_out'] if r['model']=='MQ1' and r['held_Q'] not in [.1,1.]]
    lines+=['',f"MQ1 内部 Q 层 RMSE 均值 {np.mean(interior):.8g}；端点结果分别为 "+'、'.join(f"Q={r['held_Q']:g}: {r['RMSE']:.8g}" for r in endpoints)+'。端点误差需独立审查，不能只汇总全层平均。', '',
            '## γ 对 N/D 的稳定性诊断','']
    lines+=table(s['gamma_scale_dependence'],['axis','model','n_layers','gamma_min','gamma_max','gamma_std','Pearson_gamma_logscale','Spearman_gamma_logscale'])
    lines+=['','按层估计的幅度差异与对数规模相关均为描述性证据，只有 9 个 N 层和 5 个 D 层，不能仅凭相关符号宣称显著因果尺度依赖。offset 对层参数的变化有助于显示混杂。若 γ 层间变化明显，只标记统一 γ 是支持域平均近似，本阶段不添加复杂交互。', '',
            '## B8：大范围半合成不兼容压力测试','']
    lines+=table(s['B8_incompatibility'],['subset','n','n_groups_with_correlation','positive_group_rate','spearman_min','spearman_median','spearman_max','loss_below_E_rate'])
    lines+=['']+table(s['B8_overlap_loss_differences'],['comparison','n','mean','median','std','min','max','mean_abs_difference'])
    lines+=['']+table([r for r in s['B8_pressure_performance'] if r['model'] in ['M0','MQ1','MQ2']],['subset','model','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias'])
    lines+=['','数值范围内/外按 B6 的 N、D、Q 各自闭区间判定，范围内不等于已完成数据生成语义标定。范围内仍正相关时，不兼容不能归因于普通数值外推。',
            '“B8 与由 B1+B6 建立的统一 N-D-Q 标度关系不兼容，故不纳入主质量参数估计，仅作为广义标度律适用边界的压力检验。”',
            '未翻转 Q、未允许负 γ、未修改经典参数、未删除 Loss<E 的记录，未将不兼容定性为数据错误或猜测 Q 定义反了。', '',
            '## 问题一 Q 的尺度兼容性','',
            f"领域 Q_domain 范围 {s['question1_Q_domain_range'][0]:.12g}–{s['question1_Q_domain_range'][1]:.12g}，数值尺度位于 B6 支持范围内 [0.1,1]。",
            'Q_domain 与 B6 Q_score 来自不同构造过程，未证明两种 Q 具有完全相同语义；本轮不重新读取原始质量 JSONL/XZ，也不把领域 Q 代入配比模型。', '',
            '## 冻结判断与停止点','']
    lines+=table([dict(condition=k,passes=v) for k,v in s['freeze_evidence_gates'].items()],['condition','passes'])
    lines+=['',('四项必要证据门槛均通过，可提交“仅在 B6/B7 支持域内有效”的 MQ1 近似模型供人工冻结确认。' if s['freeze_recommended'] else '必要证据门槛未全部通过，当前不建议冻结主模型。'),
            '该判断不抹去 MQ2 的函数形式差异、offset 混杂风险和层间 γ 变化，不表示真实跨源普适性已经验证。适用域 N∈[0.07,11.97] B、D∈[10,600] B、Q∈[0.1,1]，以及 B6/B7 半合成构造机制。',
            '当前未执行人工冻结，未引入 p/K，未使用 B9/B10，未推导 Q+0.1 参数替代关系、最终弹性或开始问题三。', '',
            '## 输出与复现','',
            '执行：`python -B scripts/13_fit_quality_scaling.py`；只读复核：`python -B scripts/13_verify_quality_scaling.py`。',
            '配置：configs/scaling_quality_phase2B.yaml。主要统计位于 outputs/scaling/phase2，图表位于 outputs/figures/question2_phase2。',
            f"保护本轮开始时的 {s['protected_existing_files']} 个既有文件，结束逐文件 SHA-256 核验。既有诊断报告保留，新报告为 question2_phase2B_quality_scaling.md。"]
    path=root/'outputs/report/question2_phase2B_quality_scaling.md'
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')

# ============================================================
# Source: src/scaling/scale_quality_reporting.py
# Original lines: 1-246
# Purpose: 最终CSV/图表/报告呈现，compact省略纯绘图和排版
# ============================================================
"""Scientific figures and stage report from saved scale-dependent experiment results."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .quality_extension import subset,group_reference
from .scale_quality import predict_scale
from .quality_reporting import save,obs_pred,table


def figures(root,s,ts,p,ref,oof,predictions,points,surfaces,domain,cfg):
    """Plots retain observed points, true parameter boundaries and negative evidence."""
    folder=root/'outputs/figures/question2_phase2C';folder.mkdir(parents=True,exist_ok=True)
    dpi=cfg['figure_dpi'];models=cfg['models'];b6,b7,b8=[ts[k] for k in ['B6','B7','B8']]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,target in zip(axes,['absolute_Loss','within_ND_DeltaL']):
        rows=[r for r in s['B6_CV'] if r['target']==target]
        ax.bar([r['model'] for r in rows],[r['RMSE'] for r in rows],color=['gray','tab:blue','tab:orange','tab:green'])
        ax.set(ylabel='OOF RMSE',title='B6 ND Group CV: '+target.replace('_',' '))
    save(fig,folder,'01_B6_group_CV_comparison.png',dpi)
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for j,m in enumerate(models):
        sc=obs_pred(axes[0,j],b6['L'],oof[m],m+': absolute Loss OOF',b6['Q']);fig.colorbar(sc,ax=axes[0,j],label='Q')
        obs,pp,_,refs=group_reference(b6,oof[m]);sc=obs_pred(axes[1,j],obs[~refs],pp[~refs],m+': within-ND Delta Loss OOF',b6['Q'][~refs])
        axes[1,j].set(xlabel='Observed Delta Loss',ylabel='Predicted Delta Loss');fig.colorbar(sc,ax=axes[1,j],label='Q')
    save(fig,folder,'02_B6_absolute_and_difference_OOF.png',dpi)
    new=np.array([k not in set(b6['keys']) for k in b7['keys']]);fig,axes=plt.subplots(2,3,figsize=(15,9))
    for j,m in enumerate(models):
        sc=obs_pred(axes[0,j],b7['L'][new],predictions['B7'][m][new],m+': B7 new-Q absolute',b7['Q'][new]);fig.colorbar(sc,ax=axes[0,j],label='New Q level')
        obs,pp,_,refs=group_reference(b7,predictions['B7'][m]);ix=new&~refs
        sc=obs_pred(axes[1,j],obs[ix],pp[ix],m+': B7 new-Q differences',b7['Q'][ix]);fig.colorbar(sc,ax=axes[1,j],label='New Q level')
        axes[1,j].set(xlabel='Observed Delta Loss',ylabel='Predicted Delta Loss')
    save(fig,folder,'03_B7_nonoverlap_absolute_and_difference.png',dpi)
    fig,axes=plt.subplots(2,2,figsize=(11,8));Ns=np.unique(b6['N']);Ds=np.unique(b6['D'])
    for ax,(n,d) in zip(axes.flat,[(Ns[0],Ds[0]),(Ns[0],Ds[-1]),(Ns[-1],Ds[0]),(Ns[-1],Ds[-1])]):
        ix=(b6['N']==n)&(b6['D']==d);a=subset(b6,ix);order=np.argsort(a['Q'])
        ax.plot(a['Q'][order],a['L'][order],'ko',label='B6 observations')
        tt=dict(N=np.full(100,n),D=np.full(100,d),Q=np.linspace(.1,1,100))
        for m in models:ax.plot(tt['Q'],predict_scale(tt,p,ref,m,s['fitted'][m]['parameters']),label=m)
        ax.set(xlabel='Q_score',ylabel='Loss',title=f'N={n:g} B, D={d:g} B');ax.legend(fontsize=8)
    save(fig,folder,'04_fixed_ND_quality_curves.png',dpi)
    fig,ax=plt.subplots(figsize=(8,4))
    for m in models:
        r=[r for r in s['Q_level_CV'] if r['model']==m]
        ax.plot([r['held_Q'] for r in r],[r['RMSE'] for r in r],'o-',label=m)
    ax.set(xlabel='Held-out Q level',ylabel='RMSE',title='B6 leave-one-Q-level-out (endpoints extrapolate)');ax.legend()
    save(fig,folder,'05_leave_one_Q_level_out.png',dpi)
    _uncertainty_figures(root,s,folder,dpi)
    _sensitivity_figures(surfaces,models,folder,dpi)
    inside=((b8['N']>=b6['N'].min())&(b8['N']<=b6['N'].max())&(b8['D']>=b6['D'].min())&(b8['D']<=b6['D'].max())&(b8['Q']>=b6['Q'].min())&(b8['Q']<=b6['Q'].max()))
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for j,m in enumerate(models):
        for ax,ix,title in [(axes[0,j],inside,'inside B6 numeric ranges'),(axes[1,j],~inside,'outside B6 numeric ranges')]:
            sc=obs_pred(ax,b8['L'][ix],predictions['B8'][m][ix],m+': '+title,b8['Q'][ix]);fig.colorbar(sc,ax=ax,label='Q_score');ax.axvline(p['E'],color='red',ls=':',lw=1,label='Frozen E');ax.legend(fontsize=8)
    fig.suptitle('B8 incompatible semisynthetic pressure test; no fitting',fontsize=12)
    save(fig,folder,'12_B8_pressure_inside_outside.png',dpi)
    _extra_figures(root,s,ts,p,ref,domain,cfg)


def _uncertainty_figures(root,s,folder,dpi):
    """Bootstrap histograms and correlation matrices, preserving boundary masses."""
    from .io import csv_rows
    import json
    _,rows=csv_rows(root/'outputs/scaling/phase2C/bootstrap_replicates.csv')
    for j,m in enumerate(['S1','S2','S3'],6):
        parameters=list(s['fitted'][m]['parameters']);fig,axes=plt.subplots(1,len(parameters),figsize=(4*len(parameters),3.6))
        for ax,key in zip(np.atleast_1d(axes),parameters):
            v=[json.loads(r['parameters'])[key] for r in rows if r['model']==m and r['success']=='True']
            bound=next(r for r in s['bootstrap_parameters'] if r['model']==m and r['parameter']==key)['parameter_boundary_rate']==1
            if bound:
                ax.bar(['Parameter bound'],[len(v)],color='tab:orange');ax.text(.5,.85,'100% boundary mass',transform=ax.transAxes,ha='center',fontsize=9)
            else:ax.hist(v,bins=35,color='tab:blue',alpha=.8)
            ax.set(xlabel=key,ylabel='ND-bootstrap count',title=m+' '+key)
        save(fig,folder,f'{j:02d}_{m}_parameter_bootstrap.png',dpi)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,m in zip(axes,['S1','S2','S3']):
        keys=list(s['fitted'][m]['parameters']);a=np.full((len(keys),len(keys)),np.nan)
        for r in s['parameter_correlations']:
            if r['model']==m and r['Pearson'] is not None:a[keys.index(r['parameter_1']),keys.index(r['parameter_2'])]=r['Pearson']
        cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#d9d9d9')
        im=ax.imshow(np.ma.masked_invalid(a),vmin=-1,vmax=1,cmap=cmap)
        for j in range(len(keys)):
            for k in range(len(keys)):
                if np.isnan(a[j,k]):ax.text(k,j,'n.a.',ha='center',va='center',fontsize=8)
        ax.set_xticks(range(len(keys)),keys);ax.set_yticks(range(len(keys)),keys);ax.set_title(m+' Pearson\nGray: undefined at bound/constant')
        fig.colorbar(im,ax=ax,shrink=.8)
    save(fig,folder,'09_bootstrap_parameter_correlations.png',dpi)


def _sensitivity_figures(rows,models,folder,dpi):
    """Discrete actual design cells; magnitude explicitly separated from gradient sign."""
    Ns=sorted(set(r['N_params_B'] for r in rows));Ds=sorted(set(r['D_tokens_B'] for r in rows))
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for i,q in enumerate([.1,.6]):
        for j,m in enumerate(models):
            ax=axes[i,j];a=np.full((len(Ns),len(Ds)),np.nan)
            for r in rows:
                if r['model']==m and r['Q_score']==q:a[Ns.index(r['N_params_B']),Ds.index(r['D_tokens_B'])]=r['abs_quality_sensitivity']
            im=ax.imshow(a,origin='lower',aspect='auto',cmap='viridis')
            ax.set_xticks(range(len(Ds)),[f'{d:g}' for d in Ds]);ax.set_yticks(range(len(Ns)),[f'{n:g}' for n in Ns])
            ax.set(xlabel='D (billion tokens)',ylabel='N (billion parameters)',title=f'{m}: |dLoss/dQ| at Q={q:g}')
            fig.colorbar(im,ax=ax,label='Loss per unit Q')
    save(fig,folder,'10_quality_sensitivity_ND_surfaces.png',dpi)
    fig,ax=plt.subplots(figsize=(8,5));a=np.full((len(Ns),len(Ds)),np.nan)
    for r in rows:
        if r['model']=='S1' and r['Q_score']==.6:a[Ns.index(r['N_params_B']),Ds.index(r['D_tokens_B'])]=r['effective_gamma']
    im=ax.imshow(a,origin='lower',aspect='auto',cmap='viridis');fig.colorbar(im,ax=ax,label='S1 gamma(N,D)')
    ax.set_xticks(range(len(Ds)),[f'{d:g}' for d in Ds]);ax.set_yticks(range(len(Ns)),[f'{n:g}' for n in Ns])
    ax.set(xlabel='D (billion tokens)',ylabel='N (billion parameters)',title='S1 effective gamma; distinct from marginal Loss sensitivity')
    save(fig,folder,'11_S1_effective_gamma_surface.png',dpi)


def _extra_figures(root,s,ts,p,ref,domain,cfg):
    """Numeric domain-Q interval and C/eta bound confounding; no source retuning."""
    from .scale_quality import marginal_quality
    folder=root/'outputs/figures/question2_phase2C';dpi=cfg['figure_dpi']
    low,high=s['question1_Q_domain_range'];q=np.linspace(low,high,100)
    tt=dict(N=np.full(100,ref['N0']),D=np.full(100,ref['D0']),Q=q)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    l0=p['E']+p['A']*ref['N0']**(-p['alpha'])+p['B']*ref['D0']**(-p['beta'])
    for m in ['S1','S2','S3']:
        a=s['fitted'][m]['parameters']
        axes[0].plot(q,-marginal_quality(tt,p,ref,m,a),label=m)
        axes[1].plot(q,predict_scale(tt,p,ref,m,a)-l0,label=m)
    axes[0].set(xlabel='Q coordinate',ylabel='|dLoss/dQ|',title='Frozen domain-Q interval: numeric support only')
    axes[1].set(xlabel='Q coordinate',ylabel='Extra Loss vs Q=1',title='At fixed reference N0,D0; no mixture model')
    for ax in axes:ax.legend()
    save(fig,folder,'13_question1_Q_numeric_interval.png',dpi)
    r=s['S2_bound_sensitivity'];fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([x['C_upper_bound'] for x in r],[x['C'] for x in r],'o-',label='C');axes[0].set(xscale='log',yscale='log',xlabel='Computational C upper bound',ylabel='Fitted C',title='S2 boundary sensitivity')
    axes[1].plot([x['C_upper_bound'] for x in r],[x['C_times_eta'] for x in r],'o-');axes[1].set(xscale='log',xlabel='Computational C upper bound',ylabel='Fitted C x eta',title='Stable product does not identify both factors')
    save(fig,folder,'14_S2_bound_identifiability.png',dpi)


def report(root,s,domain):
    """Generate conclusions from the fixed gate, never from preferred-looking plots."""
    out=root/'outputs/scaling/phase2C';p=s['classical_parameters'];ref=s['reference_scales']
    lines=['# 问题二 PHASE 2C：尺度调制的质量 Scaling Law','',
           '本阶段已完成，等待人工确认。未进入领域配比融合。', '',
           ('结论：三个候选均未通过 B7 非重叠组内差分门槛；不冻结统一 L(N,D,Q)，停止继续增加复杂函数。' if s['all_B7_DeltaL_failed'] else '结论：按预设门槛和 S3→S1→S2 优先级审查固定候选，最终状态见接受表。'),
           'B6 的方向和预测改善支持质量影响，但不足以证明某个候选的效应量可以迁移。不能以绝对 Loss 的高相关替代组内质量差分验证。', '',
           '## 输入、冻结与证据层级','',
           'B6 是唯一参数拟合来源；B7 的 90 个非重叠 Q 点是主要扩展复核；B8 为“大范围半合成不兼容压力测试/负面证据”，不参与任何拟合、模型或超参数选择。',
           'B6–B8 为“基于真实数据校准的半合成补充实验”。B7 完整包含 B6 的 360 条记录，新点只新增 Q=0.5、0.7，ND 组与 B6 相同；不可称为新的独立 ND 外部训练实验。',
           'B7 已在前一轮反复用于诊断，当前候选受到已知失效模式启发；本轮仍是扩展复核，不提供未见真实外部实验的无偏确认。',
           '读取 PHASE 1 主参数 CSV 的完整精度；五参数未重估。问题一仅仅读取冻结领域 Q 汇总，未解析原始质量 JSONL/XZ，也未重建 Q/K。', '',
           '```text', '\n'.join(f'{k}={v:.17g}' for k,v in p.items()),'```', '',
           '## 参考尺度与模型','',
           f"N0=exp(median(log(unique N_B6)))={ref['N0']:.17g} B；D0=exp(median(log(unique D_B6)))={ref['D0']:.17g} B（数值上 150 B）。",
           '这是唯一设计尺度在对数空间的中位数，不是 Loss 加权中心。中心固定为重参数化常数，各 CV/Bootstrap 不估计它们；没有使用 held-out Loss 求中心。',
           r'S1：\(\gamma(N,D)=\gamma_0(N/N_0)^{-u}(D/D_0)^v\)，\(L=E+A N^{-\alpha}+B[D Q^{\gamma(N,D)}]^{-\beta}\)。',
           r'S2：\(L=L_0+C(N/N_0)^{-u}(D/D_0)^v(Q^{-\eta}-1)\)。',
           r'S3：\(L=L_0+C(N/N_0)^{-u}(D/D_0)^v(-\ln Q)\)。',
           '三者无主模型 offset，Q=1 时精确退化为冻结 L0。γ0/C/η>0，u/v≥0。有效数据量是候选建模解释，未被真实直接训练实验证明。', '',
           '## 拟合与分组验证协议','',
           'bounded nonlinear least squares，直接拟合原始 Loss；严格正参数在 log 坐标中优化，u/v 使用原坐标以保留零边界。每次主拟合、CV、Q-level 留出和每个 Bootstrap 模型都执行 50 个独立初值，保存参数、目标值、收敛状态、Jacobian rank/condition 和触边字段。',
           '计算边界：γ0∈[1e-6,8]，C∈[1e-6,100]，u/v∈[0,1]，η∈[1e-5,5]；并非假装由附件估计出的统计阈值。S1 在 B6 支持域内的最大指数仍处于可计算范围，不截断或修改模型公式。',
           '5-fold GroupKFold 按精确 (N,D) 组进行。同组 8 个 Q 点同时进入训练或测试，训练折独立使用预设初值，不使用全量拟合初值。主差分评价排除 Q=1 自身零差分，避免抬高相关与 R²。',
           '2000 次 ND block bootstrap；每次有放回抽取 45 整组，组内所有 Q 带同一重数；seed=2026，三个模型共享相同抽样。全网格每组 8 点，因此逐观测等权也给各组相同总权。',
           'Bootstrap CI 条件于冻结经典参数、固定网格和半合成生成机制；不包含经典 β 不确定性、未见 Q 水平机制变化或真实跨源误差。计算仅使用 CPU。', '',
           '## 主拟合参数与识别诊断','']
    parameter_rows=[]
    for m,f in s['fitted'].items():
        parameter_rows.append(dict(model=m,**f['parameters'],fit_RMSE=float(np.sqrt(f['objective'])),
                                   jacobian_rank=f['jacobian_rank'],jacobian_condition=f['jacobian_condition'],
                                   boundary_parameters=','.join(f['boundary_parameters']) or 'none',
                                   n_starts=f['n_starts'],n_successful_starts=f['n_successful_starts']))
    lines+=table(parameter_rows,['model','gamma0','C','u','v','eta','fit_RMSE','jacobian_rank','jacobian_condition','boundary_parameters','n_successful_starts'])
    lines+=['','## 参数 Bootstrap 与相关','']
    lines+=table(s['bootstrap_parameters'],['model','parameter','mean','median','std','P025','P975','failure_rate','parameter_boundary_rate'])
    lines+=['','参数相关矩阵全部保存于 parameter_correlations.csv，同时包含 Pearson 和 Spearman。','',
            '全部重采样在同一边界、且变化仅为浮点求解误差的参数，其相关性标为未定义；不能用该噪声计算的 Pearson 解释参数耦合。C/η 的实用不可识别另由 B6 上界敏感性证明。',
            'S2 若 C 达到上界、η 同时变小，C×η 才可能受到数据稳定约束：当 η→0，C(Q^{-η}−1)≈Cη(−lnQ)，S2 接近 S3。C 与 η 的单独数值存在实用不可识别问题；触边 CI 的集中不等于参数已被识别。',
            'v 在非负边界意味着附件未识别出正的 D 增长效应。即使 S3 数值 Jacobian 满秩，也只能对边界参数作单侧推断，不能把 v 的浮点微小正值解释成实质增长。', '']
    pairs=[]
    for r in s['parameter_correlations']:
        ks=list(s['fitted'][r['model']]['parameters'])
        if ks.index(r['parameter_1'])<ks.index(r['parameter_2']):pairs.append(r)
    lines+=table(pairs,['model','parameter_1','parameter_2','Pearson','Spearman','defined'])+['']
    lines+=table(s['S2_bound_sensitivity'],['C_upper_bound','C','eta','C_times_eta','u','v','objective','jacobian_condition'])
    lines+=['','C 上界敏感性只在 B6 上改变计算约束，未使用 B7 选择边界，也未据此替换主结果。','',
            '## B6 absolute Loss 与 within-ND Delta-L OOF','']
    lines+=table(s['B6_CV'],['model','target','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias'])
    lines+=['','B6 组内差分使用同一模型/参数和同一 held-ND 折的 Q=1 预测参考，没有独立拟合差分。N/E 项与固定 offset 在差分中抵消。','',
            '## B7 非重叠 90 点：最终质量效应门槛','']
    lines+=table(s['B7_nonoverlap'],['model','target','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias'])
    lines+=['','B7 的 Q=1 实测参考来自该 ND 的全量 B7，并与 B6 重叠；差分是在新增 Q 点上的复核，不能包装成新增独立参考观测。450 条 full /360 条 overlap 指标另见 B7_validation_all_subsets.csv。',
            '预设关键门槛：B7 非重叠差分 R²>0 且 Pearson/Spearman>0。正相关但 R²≤0 表示方向有所复现，效应量预测仍未优于该子集的均值差分。不得改用绝对 Loss 的高 R² 声称通过。','',
            '## 完整 Q-level 留出','']
    lines+=table(s['Q_level_CV'],['model','held_Q','test_kind','RMSE','MAE','R2','mean_bias'])
    from .io import csv_rows,write_csv
    _,previous=csv_rows(root/'outputs/scaling/phase2/leave_one_Q_level_out.csv')
    benchmarks=[dict(model=r['model'],phase='frozen_Phase2B',held_Q=float(r['held_Q']),RMSE=float(r['RMSE']),MAE=float(r['MAE']),mean_bias=float(r['mean_bias'])) for r in previous if r['model'] in ['MQ1','MQ2'] and float(r['held_Q'])==.1]
    benchmarks += [dict(model=r['model'],phase='Phase2C',held_Q=r['held_Q'],RMSE=r['RMSE'],MAE=r['MAE'],mean_bias=r['mean_bias']) for r in s['Q_level_CV'] if r['held_Q']==.1]
    write_csv(out/'low_Q_extrapolation_comparison.csv',benchmarks)
    lines+=['','最低 Q 层与冻结上一轮模型的对照（均为整层留出，不重拟合上一轮结果）：','']
    lines+=table(benchmarks,['phase','model','held_Q','RMSE','MAE','mean_bias'])
    _report_tail(root,s,domain,lines)


def _report_tail(root,s,domain,lines):
    """Complete limitations and stopping decisions without inventing acceptance thresholds."""
    lines+=['','Q=0.1 和 Q=1 是端点留出外推，其他已观测 Q 层为插值；逐层结果不可仅用平均误差替代。各层没有读取被留出 Loss 作初值。','',
            '## 隐含质量敏感度与方向约束','',
            r'S1：\(\partial L/\partial Q=-B D^{-\beta}\exp[\beta\gamma(-\ln Q)]\beta\gamma/Q\)。',
            r'S2：\(\partial L/\partial Q=-C(N/N_0)^{-u}(D/D_0)^v\eta Q^{-\eta-1}\)。',
            r'S3：\(\partial L/\partial Q=-C(N/N_0)^{-u}(D/D_0)^v/Q\)。',
            '在 B6 全部 360 点核验解析导数和数值差分导数。Q 边界使用支持域内单侧差分；步长 1e-5 是数值检查精度设置，不是数据阈值。', '']
    lines+=table(s['monotonicity_checks'],['model','n','analytic_violation_rate','numerical_violation_rate','min_derivative','max_derivative','max_abs_analytic_numeric_discrepancy'])
    lines+=['','尺度趋势按固定另一尺度和 Q 的相邻设计点比较 |∂L/∂Q|，数值相对变化≤1e-8 视为平坦。此容差防止将边界 v≈1e-20 误称实质增长。','']
    lines+=table(s['sensitivity_scale_trends'],['model','axis','n_adjacent_pairs','decreasing_rate','increasing_rate','flat_rate','min_relative_change','max_relative_change'])
    lines+=['','S1 的 γ(N,D) 随 D 增加不保证实际 |∂L/∂Q| 同样增加：经典 D^{-β} 会抵消部分增长，效应也依赖 Q。S2/S3 的 v 若触零边界，只支持 D 敏感度基本平坦。当前不强行增加交互来满足预设方向。','',
            '## 问题一冻结 Q 区间：仅数值支持域分析','',
            f"Q_domain={s['question1_Q_domain_range'][0]:.12g}–{s['question1_Q_domain_range'][1]:.12g}，数值尺度位于 B6 [0.1,1] 支持范围内。",
            'question1_Q_interval_analysis.csv 在每个 B6 ND 工作点给出冻结领域 Q 坐标、两区间端点的质量敏感度和相对 Q=1 的额外 Loss。所有候选均保留，未通过者结果仅作诊断；没有把领域坐标变成配比预测。',
            '未证明问题一 CRITIC Q 与 B6 Q_score 具有完全相同语义，不能把数值落入支持域当作语义标定。','',
            '## 无 offset 主模型与最终来源敏感性','']
    lines+=table(s['offset_sensitivity'],['model','offset','B6_absolute_RMSE','Q1_RMSE','main_parameters','offset_parameters','relative_changes'])
    lines+=['','offset 版本只在全部主候选评估之后，用 B6 单独拟合 50 初值，未参与接受判断或修复 B7 差分。全 Q offset 可以吸收函数形状误差，不能将其全部解释为真实来源零点偏差。','',
            '## B8 不兼容压力检验','']
    lines+=table(s['B8_incompatibility'],['subset','n','n_groups_with_correlation','positive_group_rate','spearman_median','loss_below_E_count','loss_below_E_rate'])
    lines+=['']+table([r for r in s['B8_pressure_performance'] if r['subset'] in ['full','within_B6_numeric_ranges','outside_B6_numeric_ranges']],['subset','model','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias'])
    lines+=['','模型的预测 Q 方向全部为下降，B8 组内实际方向为上升，包括 B6 数值范围内的工作点。数据未翻转或删除，低于冻结 E 的记录全部保留。',
            '维持结论：B8 与 B1+B6/B7 的质量标度机制不兼容，仅作为适用边界的半合成压力检验，不判定数据错误，也不猜测 Q 定义反了。','',
            '## 接受门槛与最终选择','']
    lines+=table(s['model_acceptance'],['model','accepted','B6_absolute_CV_improves_M0','B6_DeltaL_CV_positive_R2','B7_absolute_improves_M0',
                                          'B7_DeltaL_positive_R2','B7_DeltaL_positive_correlations','bootstrap_without_failure_or_boundary_flags','strictly_decreasing_Q'])
    lines+=['','absolute 稳定性用相同验证子集上 RMSE 优于 M0 检查；并同时呈现偏差和相关，不使用人为追加的“理想误差阈值”。Bootstrap 无失败/触边标记是保守审核条件，边界单侧推断须单独审查。',
            '预设优先级 S3→S1→S2；不按 B7 最佳误差重新调参。对参数触边和实用不可识别，不能只看数值 Jacobian 满秩或漂亮的窄 CI。', '']
    if s['all_B7_DeltaL_failed']:
        lines+=['**最终不冻结 L(N,D,Q)。**三个候选在 B7 非重叠差分上均 R²≤0，未达到用户规定的必要门槛。',
                '“附件 B 当前不足以支持统一可迁移的 N-D-Q 函数形式。”此判断限定于本轮候选形式、冻结经典骨架和验证准则，不能推演成不存在任何质量效应或所有数学形式都不可能。',
                'S3 是最简单且 B6 内部表现较好的诊断对照，但没有通过扩展集复核，不应为了达到冻结目标继续堆高阶模型。']
    elif s['selected_model']:
        lines += [f"按预设规则通过的最简单候选为 {s['selected_model']}；仍需人工确认冻结，仅限半合成支持域。"]
    else:lines += ['没有模型同时通过全部条件；不冻结，停止进一步增加复杂函数。']
    lines+=['','## 复现、输出与冻结核验','',
            '运行 B6 计算：`python -B scripts/14_fit_scale_dependent_quality.py --compute`。',
            '生成固定参数验证与报告：`python -B scripts/14_fit_scale_dependent_quality.py --finalize`。',
            '中断续算：`--compute --reuse-basics --resume-bootstrap`，复用完整检查点，保持抽样序列和每次 50 初值不变。',
            '只读独立复核：`python -B scripts/14_verify_scale_dependent_quality.py`。',
            '表格输出 outputs/scaling/phase2C/，图 outputs/figures/question2_phase2C/。既有结果没有覆写，新代码使用 pathlib 从脚本位置识别根目录。',
            f"保护本阶段开始时的 {s['protected_files']} 个文件，包括官方附件和已确认全部结果；结束 SHA-256 核验。",
            '本阶段停止，等待人工确认。未引入配比 p 或冲突 K，未使用 B9/B10，未推导参数替代量或进入问题三。']
    (root/'outputs/report/question2_phase2C_scale_dependent_quality.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

# ============================================================
# Source: scripts/14_fit_scale_dependent_quality.py
# Original lines: 1-210
# Purpose: 最终阶段实际执行入口：14_fit_scale_dependent_quality.py
# ============================================================
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

# ============================================================
# Source: src/scaling/mixture_transfer.py
# Original lines: 1-283
# Purpose: 冻结ILR预测、逐尺度lambda与配方bootstrap；P1/P2曲线仅比较候选
# P1/P2 smooth lambda curves: candidate / sensitivity models, not final main model. Final lambda is piecewise linear in ln N; these curve labels differ from Question3 resource policies.
# ============================================================
"""Frozen-mixture scale transfer and modular fusion; no source model refitting.

All sizes are in billions. Saved replacement compositions and ILR coordinates
are reused directly: they must not undergo a second zero replacement.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import pearsonr, spearmanr, t

SCALES = [('1M', 'test_1M', .001, 'observed'),
          ('60M', 'test_60M', .06, 'observed'),
          ('1B', 'test_1B', 1., 'observed'),
          ('10B', 'estimated_10B', 10., 'estimated'),
          ('70B', 'estimated_70B', 70., 'estimated')]


def frozen_metadata(root):
    """Read exactly the human-selected L_mean first-order ElasticNet metadata."""
    base = Path(root) / 'outputs/mixture'
    policy = json.loads((base/'stage2/final_model_policy.json').read_text(encoding='utf-8'))
    if policy['selected_model'] != 'M1_ElasticNet':
        raise ValueError('Frozen main mixture model differs from the authorized model')
    rows = json.loads((base/'final_model_metadata.json').read_text(encoding='utf-8'))
    selected = [r for r in rows if r['model']=='M1_ElasticNet' and r['response']=='L_mean']
    if len(selected) != 1 or selected[0]['n_model_features'] != 16:
        raise ValueError('Ambiguous or non-first-order frozen mixture model')
    return selected[0]


def predict_ilr(x, meta):
    """Evaluate the frozen first-order model on saved ILR, without fitting."""
    a = np.asarray(x, float)
    return (float(meta['y_mean']) + float(meta['y_scale']) *
            (((a-np.array(meta['X_mean']))/np.array(meta['X_scale'])) @
             np.array(meta['coefficients']) + float(meta['intercept'])))


def predict_recipe(p, meta):
    """Apply frozen closure/zero convention for a new evaluation composition."""
    a = np.atleast_2d(np.asarray(p, float))
    if a.shape[1] != 17 or not np.isfinite(a).all() or (a<0).any() or (a.sum(1)<=0).any():
        raise ValueError('Expected finite nonnegative 17-domain recipes')
    a = a/a.sum(axis=1, keepdims=True)
    zero = a==0
    a = a*(1-zero.sum(axis=1)*meta['zero_delta'])[:, None]
    a[zero] = meta['zero_delta']
    return predict_ilr(np.log(a) @ np.asarray(meta['ilr_basis']).T, meta)


def inverse_ilr(z, meta):
    """Inverse frozen orthonormal basis, producing a strictly positive recipe."""
    v = np.atleast_2d(z) @ np.asarray(meta['ilr_basis'])
    p = np.exp(v-v.max(axis=1, keepdims=True))
    return p/p.sum(axis=1, keepdims=True)


def read_frozen_recipes(root, meta):
    """Join prediction to processed recipes by index; never read A4-A15 originals."""
    base = Path(root)/'outputs/mixture'
    pred = pd.read_csv(base/'test_predictions.csv')
    pred = pred[(pred.model=='M1_ElasticNet') & (pred.response=='L_mean')]
    est = pd.read_csv(base/'estimated_scale_predictions.csv')
    est = est[est.model=='M1_ElasticNet']
    train = pd.read_csv(base/'train_1M_closed_replaced_composition.csv').set_index('index')
    names = meta['domain_order']
    groups = []
    for label, dataset, size, evidence in SCALES:
        if evidence=='observed':
            g = pred[pred.dataset==dataset].copy()
            recipe = pd.read_csv(base/(dataset+'_closed_replaced_composition.csv')).set_index('index')
            ilr = pd.read_csv(base/(dataset+'_ILR_features.csv')).set_index('index')
            g = g.rename(columns={'predicted_loss':'M_p', 'true_loss':'Loss'})
            g['matched_train_index'] = pd.NA
            g['prediction_source'] = 'outputs/mixture/test_predictions.csv'
        else:
            g = est[est.dataset==dataset].copy()
            g = g.rename(columns={'uncalibrated_prediction':'M_p', 'estimated_loss':'Loss'})
            recipe = train.loc[g['matched_train_index']].copy()
            recipe.index = g['index'].to_numpy()
            ilr = None
            g['prediction_source'] = 'outputs/mixture/estimated_scale_predictions.csv'
        if g.empty or g['index'].duplicated().any() or recipe.index.duplicated().any():
            raise ValueError('Missing or duplicated recipe index: '+dataset)
        p = recipe.loc[g['index'], names].to_numpy(float)
        if not np.isfinite(p).all() or (p<=0).any() or not np.allclose(p.sum(1),1):
            raise ValueError('Invalid saved replacement composition')
        x = np.log(p) @ np.asarray(meta['ilr_basis']).T
        if ilr is not None and not np.allclose(x, ilr.loc[g['index']].to_numpy(float), atol=1e-12):
            raise ValueError('Saved composition and ILR disagree')
        if not np.allclose(predict_ilr(x, meta), g['M_p'], rtol=0, atol=1e-12):
            raise ValueError('Frozen predictor and saved M(p) disagree: '+dataset)
        if not np.isfinite(g[['Loss','M_p']].to_numpy(float)).all():
            raise ValueError('Missing prediction/Loss; no imputation authorized')
        for j, name in enumerate(names):
            g[name+'_saved_replaced'] = p[:,j]
        # Only compare recipe equality at saved floating precision, never index alone.
        g['recipe_signature'] = [json.dumps(np.round(v,12).tolist(),separators=(',',':')) for v in p]
        g['scale'] = label
        g['N_params_B'] = size
        g['evidence'] = evidence
        g['m_centered'] = g.M_p-g.M_p.mean()
        g['y_centered'] = g.Loss-g.Loss.mean()
        g['M_scale_mean'] = g.M_p.mean()
        g['Loss_scale_mean'] = g.Loss.mean()
        groups.append(g)
    return pd.concat(groups, ignore_index=True)


def transfer(g):
    """Centered OLS slope, conventional SE (n-2), and held-scale fit metrics.

    Despite no fitted intercept in centered coordinates, two means have been
    estimated from the pairs; the conventional original-coordinate OLS uses
    residual degrees of freedom n-2. Bootstrap is the primary uncertainty.
    """
    x = g.M_p.to_numpy(float); y = g.Loss.to_numpy(float)
    x = x-x.mean(); y = y-y.mean()
    sxx = float(x@x)
    if len(x)<3 or sxx<=0:
        raise ValueError('Insufficient recipe variation for transfer')
    slope = float(x@y/sxx)
    resid = y-slope*x
    se = float(np.sqrt((resid@resid)/(len(x)-2)/sxx))
    critical = t.ppf(.975,len(x)-2)
    return dict(lambda_s=slope, standard_error=se,
                OLS_CI_low=float(slope-critical*se), OLS_CI_high=float(slope+critical*se),
                R2=float(1-(resid@resid)/(y@y)), RMSE=float(np.sqrt(np.mean(resid**2))),
                Pearson=float(pearsonr(x,y).statistic), Spearman=float(spearmanr(x,y).statistic),
                n=len(x), residual_df=len(x)-2, M_scale_mean=float(g.M_p.mean()),
                Loss_scale_mean=float(g.Loss.mean()), intercept_after_centering=0.)


def bootstrap_transfers(data, cfg):
    """2000 paired recipe-index resamples, recentering each resampled scale.

    Entire recipe sets match for 1M/60M and 10B/70B, respectively. Use the same
    index draws within each pair to preserve cross-scale dependence. 1B recipes
    have no overlap and use independent draws. The frozen M(p) is never refit.
    """
    groups = {s:data[data.scale==s].sort_values('index').reset_index(drop=True) for s,_,_,_ in SCALES}
    sets = {s:set(g.recipe_signature) for s,g in groups.items()}
    cohorts = [('1M','60M'), ('1B',), ('10B','70B')]
    for cohort in cohorts:
        signatures = groups[cohort[0]].recipe_signature.tolist()
        for s in cohort[1:]:
            if groups[s].recipe_signature.tolist()!=signatures:
                raise ValueError('Expected paired recipes differ; revise resampling first')
    for i, c1 in enumerate(cohorts):
        for c2 in cohorts[i+1:]:
            if sets[c1[0]] & sets[c2[0]]:
                raise ValueError('Partial overlap requires an explicit joint resampling design')
    rng = np.random.default_rng(cfg['random_state'])
    B = cfg['bootstrap_repetitions']
    slopes = np.empty((B,5))
    draws = []
    for cohort in cohorts:
        g0 = groups[cohort[0]]; n = len(g0)
        ix = rng.integers(0,n,size=(B,n))
        draws.extend(dict(replicate=b+1, cohort='+'.join(cohort),
                          sampled_recipe_indices=g0['index'].to_numpy()[ix[b]].tolist()) for b in range(B))
        for s in cohort:
            g = groups[s]; x = g.M_p.to_numpy()[ix]; y = g.Loss.to_numpy()[ix]
            x -= x.mean(axis=1,keepdims=True); y -= y.mean(axis=1,keepdims=True)
            denom = np.sum(x*x,axis=1)
            if (denom<=0).any():
                raise ValueError('Bootstrap draw has no M(p) variation')
            j = [a[0] for a in SCALES].index(s)
            slopes[:,j] = np.sum(x*y,axis=1)/denom
    records=[]; summaries=[]
    for j,(s,_,N,evidence) in enumerate(SCALES):
        v = slopes[:,j]
        summaries.append(dict(scale=s,N_params_B=N,evidence=evidence,
                              bootstrap_mean=float(v.mean()),bootstrap_median=float(np.median(v)),
                              bootstrap_std=float(v.std(ddof=1)),CI_low=float(np.quantile(v,.025)),
                              CI_high=float(np.quantile(v,.975)),positive_fraction=float((v>0).mean()),
                              negative_fraction=float((v<0).mean()),n_bootstrap=B))
        records.extend(dict(replicate=b+1,scale=s,lambda_s=float(v[b])) for b in range(B))
    return slopes, summaries, records, draws


def p2_value(x, theta):
    """Log-parameterized positive amplitude/slope; ln Nc is dimensionless B scale."""
    a,b = np.exp(theta[:2]); return a*np.tanh(b*(theta[2]-x))


def p2_jac(x, theta):
    """Exact Jacobian in ln(amplitude), ln(slope), ln(Nc/B)."""
    a,b = np.exp(theta[:2]); z = b*(theta[2]-x); h = np.tanh(z); s = 1-h*h
    return np.c_[a*h,a*s*z,a*s*b]


def fit_curve(x,y,model,cfg,starts=None,nstarts=None,rng=None):
    """Equal-weight-by-scale P1/P2 only; no global (N,D,Q,p) optimization."""
    x,y = np.asarray(x,float), np.asarray(y,float)
    if model=='P1':
        design = np.c_[np.ones(len(x)),x]
        v = np.linalg.lstsq(design,y,rcond=None)[0]
        pred = design@v
        return dict(parameters=dict(a=float(v[0]),b=float(v[1])),theta=v.tolist(),
                    prediction=pred.tolist(),SSE=float(np.sum((pred-y)**2)),success=True,
                    boundary_hit=False,jacobian_rank=int(np.linalg.matrix_rank(design)),
                    jacobian_condition=float(np.linalg.cond(design)),n_starts=1)
    if len(x)<3:
        return dict(success=False,reason='fewer_scales_than_parameters')
    bounds = cfg['p2_bounds']
    lo = np.array([np.log(bounds['amplitude'][0]),np.log(bounds['slope'][0]),bounds['log_Nc_B'][0]])
    hi = np.array([np.log(bounds['amplitude'][1]),np.log(bounds['slope'][1]),bounds['log_Nc_B'][1]])
    count = nstarts or cfg['full_fit_starts']
    seeds = [np.array(t,float) for t in (starts or [])]
    p1 = np.linalg.lstsq(np.c_[np.ones(len(x)),x],y,rcond=None)[0]
    root = float(-p1[0]/p1[1]) if abs(p1[1])>1e-12 else float(np.median(x))
    seeds += [np.array([np.log(max(1.,np.max(abs(y)))),np.log(.15),root]),
              np.array([np.log(max(.5,np.max(abs(y)))),np.log(.4),float(np.median(x))])]
    if rng is None: rng=np.random.default_rng(cfg['random_state'])
    while len(seeds)<count:
        seeds.append(np.array([rng.uniform(-1,2),rng.uniform(-4,.5),rng.uniform(x.min()-3,x.max()+3)]))
    fits=[]
    for initial in seeds[:count]:
        initial = np.clip(initial,lo+1e-8,hi-1e-8)
        f = least_squares(lambda t:p2_value(x,t)-y,initial,jac=lambda t:p2_jac(x,t),bounds=(lo,hi),
                          max_nfev=cfg['max_nfev'],xtol=1e-10,ftol=1e-10,gtol=1e-10)
        if f.success and np.isfinite(f.fun).all(): fits.append(f)
    if not fits: return dict(success=False,reason='no_successful_start')
    f=min(fits,key=lambda v:float(v.fun@v.fun)); theta=f.x
    a,b=np.exp(theta[:2]); jac=p2_jac(x,theta)
    tol=cfg['boundary_tolerance']; boundary=bool(np.any((theta-lo)<tol)|np.any((hi-theta)<tol))
    return dict(parameters=dict(a=float(a),b=float(b),log_Nc_B=float(theta[2])),theta=theta.tolist(),
                prediction=p2_value(x,theta).tolist(),SSE=float(f.fun@f.fun),success=True,
                boundary_hit=boundary,jacobian_rank=int(np.linalg.matrix_rank(jac,tol=1e-10)),
                jacobian_condition=float(np.linalg.cond(jac)),n_starts=count,
                n_successful_starts=len(fits))


def curve_root(fit, model):
    """Return ln(Nc/B); a finite root is not proof of empirical identification."""
    if not fit['success']: return None
    p=fit['parameters']
    if model=='P1':
        if abs(p['b'])<1e-12:return None
        return float(-p['a']/p['b'])
    return float(p['log_Nc_B'])


def empirical_lambda(N, sizes, slopes):
    """Piecewise linear in ln N, refusing extrapolation beyond provided knots."""
    N=np.asarray(N,float); sizes=np.asarray(sizes,float); slopes=np.asarray(slopes,float)
    if not np.isfinite(N).all() or np.any(N<sizes.min()) or np.any(N>sizes.max()):
        raise ValueError('N outside supported empirical knots; extrapolation not authorized')
    return np.interp(np.log(N),np.log(sizes),slopes)


class ModularScaling:
    """Evaluate fixed modules. This class never fits or constructs joint test data."""
    def __init__(self, classical, quality, meta, p_ref, sizes, slopes):
        self.classical=dict(classical); self.quality=dict(quality); self.meta=meta
        self.p_ref=np.asarray(p_ref,float); self.M_ref=float(predict_recipe(self.p_ref,meta)[0])
        self.sizes=np.asarray(sizes,float); self.slopes=np.asarray(slopes,float)

    def components(self,N,D,Q,p):
        """Loss modules with N,D in billions, 0<Q<=1, and a closed recipe."""
        N,D,Q=np.broadcast_arrays(np.asarray(N,float),np.asarray(D,float),np.asarray(Q,float))
        if not np.isfinite(N+D+Q).all() or (N<=0).any() or (D<=0).any() or (Q<=0).any() or (Q>1).any():
            raise ValueError('Illegal N,D,Q for modular candidate')
        c=self.classical; q=self.quality
        base=c['E']+c['A']*N**(-c['alpha'])+c['B']*D**(-c['beta'])
        dq=q['C_Q']*N**(-q['u_Q'])*(-np.log(Q))
        dp=empirical_lambda(N,self.sizes,self.slopes)*(predict_recipe(p,self.meta)-self.M_ref)
        return dict(L0=base,Delta_Q=dq,Delta_p=dp,L_gen=base+dq+dp)

    def partials(self,N,D,Q,p):
        """Analytic derivatives; N-knot derivative has distinct left/right limits."""
        N,D,Q=float(N),float(D),float(Q)
        comp=self.components(N,D,Q,p); c=self.classical; q=self.quality
        i=int(np.clip(np.searchsorted(self.sizes,N,side='right')-1,0,len(self.sizes)-2))
        dlam=(self.slopes[i+1]-self.slopes[i])/np.log(self.sizes[i+1]/self.sizes[i])/N
        mdiff=float(predict_recipe(p,self.meta)[0]-self.M_ref)
        return dict(dL_dN=float(-c['alpha']*c['A']*N**(-c['alpha']-1)-q['u_Q']/N*comp['Delta_Q']+dlam*mdiff),
                    dL_dD=float(-c['beta']*c['B']*D**(-c['beta']-1)),
                    dL_dQ=float(-q['C_Q']*N**(-q['u_Q'])/Q))

# ============================================================
# Source: src/scaling/mixture_transfer_reporting.py
# Original lines: 1-205
# Purpose: 最终CSV/图表/报告呈现，compact省略纯绘图和排版
# ============================================================
"""Plain scientific figures and a provenance-aware Phase 3 Markdown report."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .mixture_transfer import SCALES,p2_value
from .io import write_csv


def markdown_table(rows,columns):
    """Render finite numeric values; absent statistics remain explicitly unavailable."""
    def cell(v):
        if v is None:return 'n.a.'
        if isinstance(v,float):return f'{v:.8g}' if np.isfinite(v) else 'n.a.'
        if isinstance(v,(dict,list)):return json.dumps(v,ensure_ascii=False)
        return str(v)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
                      ['| '+' | '.join(cell(r.get(c)) for c in columns)+' |' for r in rows])


def plot_results(out,figdir,rows,data,fits,comparison,critical,cfg):
    """Plot observed/estimated evidence separately and display conditional intervals."""
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figdir=Path(figdir);manifest=[]
    def save(fig,name,description):
        path=figdir/name;fig.savefig(path,dpi=cfg['figure_dpi'],bbox_inches='tight');plt.close(fig)
        manifest.append(dict(filename=name,description=description))
    N=np.array([r['N_params_B'] for r in rows]);lam=np.array([r['lambda_s'] for r in rows])
    lo=np.array([r['CI_low'] for r in rows]);hi=np.array([r['CI_high'] for r in rows])
    def knots(ax):
        for mask,color,marker,label in [(np.arange(5)<3,'#28649b','o','Observed mixture checks'),
                                       (np.arange(5)>=3,'#bb6a28','s','Estimated tables')]:
            ax.errorbar(np.log10(N[mask]),lam[mask],yerr=[lam[mask]-lo[mask],hi[mask]-lam[mask]],
                        fmt=marker,color=color,capsize=4,linestyle='none',label=label)
        ax.axhline(0,color='gray',lw=.8)
        ax.set_xticks(np.log10(N),[r['scale'] for r in rows]);ax.set_xlabel('Parameter scale N (log axis)')
        ax.set_ylabel('Centered mixture transfer coefficient lambda')
    fig,ax=plt.subplots(figsize=(8,4.8));knots(ax);ax.plot(np.log10(N),lam,color='gray',lw=1,label='Empirical log-linear interpolation')
    ax.set_title('Mixture transfer by scale (recipe bootstrap 95% CI)');ax.legend(fontsize=9)
    save(fig,'01_mixture_transfer_by_scale.png','Observed and estimated lambda with paired recipe Bootstrap intervals')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    grid=np.geomspace(N.min(),N.max(),250)
    for ax,(label,title) in zip(axes,[('A_real_only','Fit A: three observed scales'),('B_with_estimated','Fit B: observed + estimated scales')]):
        knots(ax)
        for m,color in [('P1','#39915b'),('P2','#8b4b9b')]:
            f=fits[(label,m)];p=f['parameters']
            y=p['a']+p['b']*np.log(grid) if m=='P1' else p2_value(np.log(grid),f['theta'])
            support=grid<=1 if label=='A_real_only' else np.ones(len(grid),bool)
            ax.plot(np.log10(grid[support]),y[support],color=color,label=m)
            if label=='A_real_only':
                ax.plot(np.log10(grid[~support]),y[~support],color=color,linestyle='--',alpha=.6)
        ax.set_title(title);ax.legend(fontsize=8)
    fig.suptitle('P1/P2 scale fits; dashed Fit A curves are unsupported extrapolation',fontsize=11)
    fig.tight_layout();save(fig,'02_lambda_scale_functions.png','P1/P2 Fits A/B; Fit A beyond 1B marked as extrapolation')
    reps=pd.read_csv(out/'mixture_transfer_bootstrap_replicates.csv')
    fig,axes=plt.subplots(1,5,figsize=(15,3.4))
    for ax,r in zip(axes,rows):
        vals=reps.loc[reps.scale==r['scale'],'lambda_s'];color='#28649b' if r['evidence']=='observed' else '#bb6a28'
        ax.hist(vals,bins=35,color=color,alpha=.8);ax.axvline(0,color='black',lw=.8)
        ax.axvline(r['lambda_s'],color='red',lw=1);ax.set_title(r['scale']+' ('+r['evidence']+')',fontsize=10)
        ax.set_xlabel('lambda');ax.set_ylabel('Bootstrap count')
    fig.suptitle('Recipe Bootstrap distributions (2000 resamples per scale)');fig.tight_layout()
    save(fig,'03_lambda_bootstrap_distributions.png','No source model refit; paired draws within shared recipe sets')
    fig,axes=plt.subplots(1,5,figsize=(16,3.8))
    for ax,r in zip(axes,rows):
        g=data[data.scale==r['scale']];x=g.m_centered.to_numpy()
        ax.scatter(x,g.y_centered,s=10,alpha=.55,color='#28649b' if r['evidence']=='observed' else '#bb6a28')
        gx=np.array([x.min(),x.max()]);ax.plot(gx,r['lambda_s']*gx,color='black',lw=1)
        ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5)
        ax.set_title(r['scale']+f"  R2={r['R2']:.3f}",fontsize=10)
        ax.set_xlabel('M(p) minus scale mean');ax.set_ylabel('Loss minus scale mean')
    fig.suptitle('Within-scale centered calibration (not an external validation)');fig.tight_layout()
    save(fig,'04_centered_recipe_calibration.png','All 702 saved recipe points, with no change to their Loss or prediction')
    roots=pd.read_csv(out/'lambda_scale_bootstrap_replicates.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,(fit,m) in zip(axes.flat,[(f,m) for f in ['A_real_only','B_with_estimated'] for m in ['P1','P2']]):
        g=roots[(roots['fit']==fit)&(roots.model==m)]
        valid=g.log_Nc_B.notna();a=g.loc[valid,'log_Nc_B'].to_numpy()/np.log(10)
        ax.hist(a,bins=40,color='#28649b' if fit=='A_real_only' else '#bb6a28',alpha=.8)
        ax.set_title(fit.replace('_',' ')+' '+m,fontsize=10)
        ax.set_xlabel('log10(Nc / B)');ax.set_ylabel('Bootstrap count')
    fig.suptitle('Candidate crossing uncertainty; large-scale evidence is estimated');fig.tight_layout()
    save(fig,'05_conditional_crossing_bootstrap.png','Crossing of fitted lambda functions, not an identified physical transition')
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for ax,(f,title) in zip(axes,[('A_real_only','Observed scales only'),('B_with_estimated','Including estimated scales')]):
        subset=[r for r in comparison if r['fit']==f];labels=[r['model'] for r in subset]
        ax.bar(np.arange(2)-.18,[r['RMSE'] for r in subset],width=.36,label='In-scale RMSE')
        vals=[r['LOSO_RMSE'] if r['LOSO_RMSE'] is not None else np.nan for r in subset]
        ax.bar(np.arange(2)+.18,vals,width=.36,label='Leave-scale-out RMSE')
        if f=='A_real_only':ax.text(1,.02,'P2 leave-scale-out\nunidentified (2 points, 3 params)',ha='center',fontsize=8)
        ax.set_xticks(range(2),labels);ax.set_ylabel('RMSE in lambda');ax.set_title(title);ax.legend(fontsize=8)
    fig.tight_layout();save(fig,'06_scale_fit_validation.png','Saturated P2 Fit A and unavailable two-point leave-scale-out fit disclosed')
    write_csv(out/'figure_manifest.csv',manifest)


def build_report(path,out,rows,comparison,critical,decision,config,component):
    """Report modular evidence, formulas, uncertainties, and the explicit stop point."""
    classical=config['classical'];q=config['quality'];reference=config['reference']
    inputs=pd.read_csv(out/'input_manifest.csv').to_dict('records')
    observed=[r for r in rows if r['evidence']=='observed'];estimated=[r for r in rows if r['evidence']=='estimated']
    printable_critical=[]
    for r in critical:
        view=dict(r)
        if not r['conditional_interval_stable']:
            for k in ['Nc_B_median','Nc_B_CI_low','Nc_B_CI_high']:view[k]=None
        printable_critical.append(view)
    cols=['scale','evidence','n','lambda_s','standard_error','OLS_CI_low','OLS_CI_high','CI_low','CI_high','R2','Pearson','Spearman']
    text=['# 问题二 PHASE 3：领域配比的尺度调制与模块化广义标度律','',
      '本阶段完成冻结模块的组合和逐尺度配比传递校准。最终候选使用分尺度经验系数及 ln N 上的分段线性插值。'
      '不冻结新的全局 N-D-Q 函数，不重新训练配比模型，不进行联合最小二乘，不使用 B9/B10。','',
      '## 1. 冻结输入与证据等级','',
      '第一问主模型为 L_mean 的 ILR + 一阶 ElasticNet。M(p) 是其 1M 尺度预测 Loss。'
      '只解析此前输出，未解析 A4–A15 原始附件。现成逐配方预测与冻结模型参数、保存 ILR 坐标逐行核对。'
      'L_trim、Ridge、增强 Qp/Kp 模型均未替代主模型。','',
      markdown_table(config['evidence'],['module','parameters','evidence','support']),'',
      'Class A 是骨架参数的直接数据支持，不表示 B4/B5 已统一评测口径。Class B 是 B6 半合成局部证据。'
      'Class C 是问题一真实配比实验。Class D 是 10B/70B 估算表的条件性证据。','',
      '## 2. 逐尺度中心化与传递系数','',
      r'$m_{si}=M(p_{si})-\overline{M}_s$, $y_{si}=L_{si}-\overline{L}_s$。', '',
      r'$\hat\lambda_s=\sum_i m_{si}y_{si}/\sum_i m_{si}^2$，中心化坐标不设截距。', '',
      'standard error 使用原坐标含截距 OLS 等价的 n−2 残差自由度，因为两侧均值从数据估计。'
      'OLS CI 为 t 区间，CI_low/CI_high 为主要报告的配方 Bootstrap 百分位区间。'
      'R² 是校准后的组内拟合 R²，不是未经目标尺度校准的外部预测性能。Pearson/Spearman 比较原始 M(p) 与 Loss，保留估算表负相关。','',
      markdown_table(rows,cols),'',
      f"真实三尺度的系数同为正：{all(r['CI_low']>0 for r in observed)}（三个 Bootstrap 区间均不跨 0）。"
      f"估算两尺度同为负：{all(r['CI_high']<0 for r in estimated)}（两个区间均不跨 0）。",'',
      '定量结果支持真实尺度内配比方向一致，并在估算表中出现反向。真实实验尚未观测到符号翻转。'
      '**配比反向及临界规模结论依赖估算表。**','',
      '## 3. 配方 Bootstrap','',
      '固定种子 2026，每尺度 2000 次按配方 index 有放回抽样。每次重新计算样本内两个均值，再估计无截距传递系数。'
      '1M/60M 的完整 256 个配方相同，10B/70B 的完整 63 个配方相同，分别共享抽样索引。'
      '1B 的 64 个配方独立，三类配方集合互不重叠。不能仅按同名 index 关联不同配方。'
      '因此 1B 与其他组的配方覆盖不同，传递系数变化不能完全归因于 N；评测口径、配方覆盖和估算生成方式也可能影响系数。'
      '此 Bootstrap 条件于冻结 M(p)，不包含第一问模型再训练不确定性和估算表生成误差。','',
      markdown_table(rows,['scale','bootstrap_mean','bootstrap_median','bootstrap_std','CI_low','CI_high','positive_fraction','negative_fraction']),'',
      '## 4. P1/P2 的真实数据拟合与估算敏感性','',
      r'P1：$\lambda(N)=a+b\ln(N/N_{ref})$，$N_{ref}=1$B。', '',
      r'P2：$\lambda(N)=a\tanh[b(\ln N_c-\ln N)]$，$a,b>0$。', '',
      'N 使用十亿参数为单位，所以 P2 的对数为 ln(N/B)。各尺度等权拟合，避免小标准误的尺度自动主导。'
      'Fit A 仅三项真实检验，Fit B 加两项估算表。P2 全拟合 16 初值，Bootstrap 每次 3 初值，正参数有界拟合。'
      'AICc 仅在 n>k+1 时定义。三点三参数的 P2 Fit A 是饱和拟合，残差自由度为 0；留出一尺度后只有两点，禁止解释成可识别的三参数模型。','',
      markdown_table(comparison,['fit','model','parameters','RMSE','LOSO_RMSE','AICc','residual_df','jacobian_condition','saturated']),'',
      'Fit B 中 P2 的拟合 RMSE 和留尺度 RMSE 均低于 P1；P1 参数更少且 AICc 更低，适合作为简约趋势对照。'
      '数值性能和简约准则未一致选择同一形式，P2 的较小误差也没有足够真实尺度验证支撑。'
      '因此不把任一光滑函数冻结为全局配比尺度律，保留分尺度经验系数。','',
      '## 5. 临界规模的条件性与辨识限制','',
      '对 P1 求 ln(Nc/B)=−a/b，P2 使用拟合 ln(Nc/B)。每次使用上述配方 Bootstrap 系数重新拟合曲线。'
      '表中 valid_fraction 报告有限根比例，boundary_fraction 报告触边比例。conditional_interval_stable 要求有限根比例≥95%、触边比例≤5%，'
      '且 95% CI 完整位于数据中相反符号节点的括区间内。Fit A 没有符号括区间，不能判为稳定临界规模。这个条件性标记'
      '不表示真实实验已确认物理临界规模。负向趋势与反向传递不是因果机制证明。','',
      markdown_table(printable_critical,['fit','model','valid_fraction','boundary_fraction','Nc_B_median','Nc_B_CI_low','Nc_B_CI_high','conditional_interval_stable']),'',
      '**不报告已被真实实验稳定识别的具体临界规模。**真实尺度均为正，没有过零括区间；'
      'Fit A 的 P1/P2 外推根还存在结构依赖。Fit B 和经验插值的过零位置仅作为依赖估算表的条件性诊断，不能用于宣称普适临界规模。','',
      '## 6. 最终配比项：经验系数及分段形式','',
      markdown_table(config['lambda_knots'],['N_params_B','lambda_s','CI_low','CI_high','evidence']),'',
      r'令 $x=\ln(N/B)$、$x_s=\ln(N_s/B)$。在 $N_s\le N\le N_{s+1}$ 内：', '',
      r'$\lambda_p(N)=\lambda_s+(\lambda_{s+1}-\lambda_s)(x-x_s)/(x_{s+1}-x_s)$。', '',
      '只在节点范围 0.001B–70B 内插值，接口拒绝范围外外推。0.001B–1B 的节点来自真实检验，超过 1B 的插值依赖 Class D 估算证据。'
      '在 ln N 而非原 N 上线性插值，保持与尺度趋势图一致。节点处可连续，但导数一般有左右两值。','',
      f"参考配方直接读取冻结的 A4_ILR_center，M(p_ref)={reference['M_ref']:.17g}。"
      '中心是在冻结零替换后的 ILR 空间取均值并反变换，17 维配方完整存于 reference_recipe.json。', '',
      r'$\Delta_p(N,p)=\lambda_p(N)[M(p)-M(p_{ref})]$，所以参考配方修正严格为 0。', '',
      '逐尺度中心化只校准斜率，不改变经典绝对基线。参考配方锚定不会证明各源的绝对 Loss 基线已对齐。','',
      '## 7. 模块化广义标度律候选','',
      r'$L_{gen}=E+AN^{-\alpha}+BD^{-\beta}+C_Q(N/N_0)^{-u_Q}(-\ln Q)+\lambda_p(N)[M(p)-M(p_{ref})]$。', '',
      f"N、D 以 B 为单位，N0=1B。完整冻结参数：{json.dumps(classical,ensure_ascii=False)}。",'',
      f"C_Q={q['C_Q']:.17g}，u_Q={q['u_Q']:.17g}，直接读取 PHASE 2C S3，未重新拟合。"
      '原 v≈0 且触边，按人工确认删除 D^v 项。仅称“B6 校准的局部质量修正项”。', '',
      r'$Q=1$ 且 $p=p_{ref}$ 时，$L_{gen}=L_0$。anchor_identity_checks.csv 是公式恒等核验，不是联合观测测试集。', '',
      '质量与配比相加采用模块独立性假设，尚未验证 Q×p、N×D×p 的联合交互。配比表没有共同 N/D/Q/p/L 联合观测。'
      '模型是在各自证据源独立标定后组合的候选，不提供统一外部预测精度。','',
      '## 8. 边际效应公式与 ILR 有限差分','',
      r'记 $G(p)=M(p)-M(p_{ref})$，$\Delta_Q=C_Q(N/N_0)^{-u_Q}(-\ln Q)$。', '',
      r'$\partial_N L=-\alpha A N^{-\alpha-1}-(u_Q/N)\Delta_Q+\lambda_p\prime(N)G(p)$。', '',
      r'$\partial_D L=-\beta B D^{-\beta-1}$。', '',
      r'$\partial_Q L=-C_Q(N/N_0)^{-u_Q}/Q$。', '',
      r'分段内部：$\lambda_p\prime(N)=(\lambda_{s+1}-\lambda_s)/[N\ln(N_{s+1}/N_s)]$。', '',
      r'P1 对照的导数为 $b/N$；P2 对照为 $-ab[1-\tanh^2(b(\ln N_c-\ln N))]/N$。', '',
      r'令 $t=ILR(p)$，$p_{\pm}=ILR^{-1}(t\pm h e_k)$。', '',
      r'$[\Delta_p(N,p_+)-\Delta_p(N,p_-)]/(2h)=\lambda_p(N)[M(p_+)-M(p_-)]/(2h)$。', '',
      r'冻结一阶模型下，若 $g_k=s_y\beta_k/s_{X,k}$，上述有限差分等于 $\lambda_p(N)g_k$。', '',
      '此关系针对严格正配方的 ILR 扰动。原始零配比先按冻结约定替换，不把对替换组成的有限差分解释为原始结构零处的导数。'
      '本轮仅推导公式及实现核验，不计算弹性、Q 提升 0.1 的参数等价。','',
      '## 9. 分模块验证及局限性','',
      markdown_table(component,['module','evidence','dataset','stage','target','n','RMSE','R2','Pearson','Spearman','limitation']),'',
      'L0：B1 的验证来自此前留轨迹 CV，B3 是插值轨迹一致性，B4/B5 保留不同评测口径限制。'
      'Delta_Q：B6 质量修正有效，但 B7 差分 R² 未通过；此失败在组合模型中仍然保留。'
      'Delta_p：问题一原预测检验与本轮中心化斜率校准分别报告，后者不能声称独立外部验证。'
      '10B/70B：估算且为训练配方子集，不是大模型真实联合实验。','',
      '**广义模型采用模块化证据融合，无法进行单一完整联合外部验证。**未制造联合测试集，未重拟合经典或质量参数。'
      '此前 B8 机制不兼容结论保持冻结，本轮未用 B8 校准配比项。','',
      '## 10. 输出与停止点','',
      '主要输出：mixture_transfer_by_scale.csv、mixture_transfer_bootstrap.csv、lambda_scale_model_comparison.csv、'
      'critical_scale_diagnosis.csv、lambda_empirical_knots.csv、generalized_scaling_candidate.json、component_validation.csv。'
      '复现：python -B scripts/15_calibrate_mixture_scaling.py；核验：python -B scripts/15_verify_mixture_scaling.py。','',
      '可以提交本阶段候选进行人工确认。尚未授权进入下一阶段。'
      '对于超过 70B 的 B9/B10 目标，本轮分段接口没有外推策略，下一阶段须先明确外推规则和 Class D 情景身份。'
      '不把本阶段称为完整联合模型验证通过。已停止，不执行弹性、参数等价或 B9/B10 外推。','',
      '## 11. 实际解析文件','',markdown_table(inputs,['path','bytes','sha256','source_role']),'']
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text('\n'.join(text),encoding='utf-8')

# ============================================================
# Source: scripts/15_calibrate_mixture_scaling.py
# Original lines: 1-342
# Purpose: 最终阶段实际执行入口：15_calibrate_mixture_scaling.py
# ============================================================
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

# ============================================================
# Source: src/scaling/marginal_analysis.py
# Original lines: 1-153
# Purpose: 最终广义标度律、边界策略、解析导数/弹性及Q+0.1精确求根
# ============================================================
"""Frozen modular model, boundary policies, derivatives and exact Q equivalence.

No parameter estimation occurs here. N and D are in billions. Physical quality
transfer is conditional on the frozen B6 local calibration, not a global law.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from .mixture_transfer import predict_recipe


class FrozenGeneralizedScaling:
    """Evaluate the human-confirmed modular form with explicit lambda boundaries."""
    def __init__(self,classical,quality,meta,M_ref,knots,config):
        self.c=dict(classical);self.q=dict(quality);self.meta=meta;self.M_ref=float(M_ref)
        self.Ns=np.array([k['N_params_B'] for k in knots],float)
        self.lambdas=np.array([k['lambda_s'] for k in knots],float)
        self.slopes=np.diff(self.lambdas)/np.diff(np.log(self.Ns))
        self.cfg=config;self.shrink=float(config['lambda_shrink_zero_B'])
        if not np.all(np.diff(self.Ns)>0) or self.shrink<=self.Ns[-1]:
            raise ValueError('Invalid frozen lambda knots or shrink endpoint')

    def delta_M(self,p):
        """Frozen 1M mixture prediction relative to the frozen training center."""
        return predict_recipe(p,self.meta)-self.M_ref

    def lambda_value(self,N,policy='S-A'):
        """Main clamp; sensitivity continuation or ln-scale shrink to zero."""
        x=np.asarray(N,float)
        if not np.isfinite(x).all() or (x<=0).any():raise ValueError('N must be positive')
        if policy not in ['S-A','S-B','S-C']:raise ValueError('Unknown lambda policy')
        value=np.interp(np.log(x),np.log(self.Ns),self.lambdas)
        high=x>self.Ns[-1]
        if policy=='S-B':
            value=np.where(high,self.lambdas[-1]+self.slopes[-1]*np.log(x/self.Ns[-1]),value)
        elif policy=='S-C':
            fraction=np.clip(np.log(x/self.Ns[-1])/np.log(self.shrink/self.Ns[-1]),0,1)
            value=np.where(high,self.lambdas[-1]*(1-fraction),value)
        return value

    def log_slope(self,N,policy='S-A',side='right'):
        """Return d lambda/d ln N; choose a one-sided derivative at each knot."""
        x=np.asarray(N,float)
        if not np.isfinite(x).all() or (x<=0).any() or side not in ['left','right']:
            raise ValueError('Illegal N or derivative side')
        j=np.searchsorted(self.Ns,x,side=side)-1
        value=np.zeros(x.shape,float)
        middle=(j>=0)&(j<len(self.Ns)-1)
        value=np.where(middle,self.slopes[np.clip(j,0,len(self.slopes)-1)],value)
        high=x>=self.Ns[-1] if side=='right' else x>self.Ns[-1]
        if policy=='S-B':value=np.where(high,self.slopes[-1],value)
        elif policy=='S-C':
            shrinking=high & (x<self.shrink if side=='right' else x<=self.shrink)
            value=np.where(shrinking,-self.lambdas[-1]/np.log(self.shrink/self.Ns[-1]),value)
        elif policy!='S-A':raise ValueError('Unknown lambda policy')
        return value

    def terms(self,N,D,Q,deltaM=0.,policy='S-A'):
        """Compute actual model terms; only reference scenarios use Q=1, deltaM=0."""
        N,D,Q,deltaM=np.broadcast_arrays(np.asarray(N,float),np.asarray(D,float),np.asarray(Q,float),np.asarray(deltaM,float))
        if not np.isfinite(N+D+Q+deltaM).all() or (N<=0).any() or (D<=0).any() or (Q<=0).any() or (Q>1).any():
            raise ValueError('Illegal N,D,Q,mixture score')
        c,q=self.c,self.q
        nt=c['A']*N**(-c['alpha']);dt=c['B']*D**(-c['beta'])
        dq=q['C_Q']*(N/q['N0_B'])**(-q['u_Q'])*(-np.log(Q))
        dp=self.lambda_value(N,policy)*deltaM;R=nt+dt+dq+dp
        return dict(N_term=nt,D_term=dt,Delta_Q=dq,Delta_p=dp,R=R,L0=c['E']+nt+dt,L_gen=c['E']+R)

    def derivatives(self,N,D,Q,deltaM=0.,policy='S-A',side='right'):
        """Exact interior or chosen one-sided derivatives; do not smooth knots."""
        t=self.terms(N,D,Q,deltaM,policy);N=np.asarray(N,float);D=np.asarray(D,float);Q=np.asarray(Q,float)
        c,q=self.c,self.q;s=self.log_slope(N,policy,side)
        gain=c['alpha']*t['N_term']+q['u_Q']*t['Delta_Q']-s*np.asarray(deltaM)
        return dict(dL_dN=-gain/N,dL_dD=-c['beta']*t['D_term']/D,
                    dL_dQ=-q['C_Q']*(N/q['N0_B'])**(-q['u_Q'])/Q,Gain_N=gain,s_i=s)

    def elasticity(self,N,D,Q,deltaM=0.,policy='S-A',side='right'):
        """Loss-reduction elasticities for R>0; traditional signs in an appendix."""
        t=self.terms(N,D,Q,deltaM,policy);d=self.derivatives(N,D,Q,deltaM,policy,side)
        valid=t['R']>0
        def ratio(a):return np.divide(a,t['R'],out=np.full(np.shape(t['R']),np.nan),where=valid)
        return dict(E_N=ratio(d['Gain_N']),E_D=ratio(self.c['beta']*t['D_term']),
                    E_Q=ratio(self.q['C_Q']*(np.asarray(N)/self.q['N0_B'])**(-self.q['u_Q'])),
                    traditional_N=np.asarray(N)/t['L_gen']*d['dL_dN'],
                    traditional_D=np.asarray(D)/t['L_gen']*d['dL_dD'],
                    traditional_Q=np.asarray(Q)/t['L_gen']*d['dL_dQ'],valid=valid)

    def N_dependent_loss(self,N,Q,deltaM,policy='S-A'):
        """Exact N-dependent part; E and B D^-beta cancel in Q equivalence."""
        N=np.asarray(N,float)
        return (self.c['A']*N**(-self.c['alpha'])+
                self.q['C_Q']*(N/self.q['N0_B'])**(-self.q['u_Q'])*(-np.log(Q))+
                self.lambda_value(N,policy)*deltaM)

    def equivalence(self,N,Q,deltaM=0.,policy='S-A'):
        """Find first exact N_eq>N, respecting nonmonotone lambda segments.

        Within a segment Gain_N=positive decreasing powers minus a constant,
        hence has at most one zero. Include each stationary point as well as
        each lambda knot in the root brackets: an endpoint-only sign check can
        miss a valid first root when F(N) decreases and then increases.
        """
        N,Q,deltaM=float(N),float(Q),float(deltaM);dQ=float(self.cfg['Q_gain'])
        if not 0<Q<=1 or N<=0:raise ValueError('Illegal equivalence workpoint')
        new=Q+dQ
        result=dict(Q_new=new,root_policy='first_root_above_N',N_max_B=max(self.cfg['root_max_multiple']*N,self.cfg['root_min_max_B']))
        if Q>.9 or new>1+1e-14:return dict(**result,status='Q_gain_out_of_range',N_eq_B=None)
        gain=float(self.derivatives(N,150.,Q,deltaM,policy,self.cfg['knot_derivative_main'])['Gain_N'])
        result['Gain_N']=gain
        if gain<=0:return dict(**result,status='no_local_equivalence',N_eq_B=None)
        local=float(self.q['C_Q']*(N/self.q['N0_B'])**(-self.q['u_Q'])/(Q*gain)*dQ)
        result['local_approx_gain']=local
        target=float(self.N_dependent_loss(N,new,deltaM,policy))
        def objective(x):return float(self.N_dependent_loss(x,Q,deltaM,policy)-target)
        upper=result['N_max_B'];points=[N]+[float(k) for k in self.Ns if N<k<upper]+[upper]
        if policy=='S-C' and N<self.shrink<upper:points.append(self.shrink)
        points=sorted(set(points));stationary=[]
        def gain_at(x,s):
            return (self.c['alpha']*self.c['A']*x**(-self.c['alpha'])+
                    self.q['u_Q']*self.q['C_Q']*(x/self.q['N0_B'])**(-self.q['u_Q'])*(-np.log(Q))-s*deltaM)
        for lo,hi in zip(points[:-1],points[1:]):
            s=float(self.log_slope(np.sqrt(lo*hi),policy));g0,g1=gain_at(lo,s),gain_at(hi,s)
            if g0*g1<0:
                stationary.append(float(brentq(lambda x:gain_at(x,s),lo,hi,xtol=self.cfg['root_xtol'],rtol=self.cfg['root_rtol'])))
        points=sorted(set(points+stationary));root=None;chosen=None
        for lo,hi in zip(points[:-1],points[1:]):
            flo,fhi=objective(lo),objective(hi)
            if flo==0 and lo>N:root=lo;chosen=(lo,lo);break
            if fhi==0:root=hi;chosen=(hi,hi);break
            if flo*fhi<0:
                root=float(brentq(objective,lo,hi,xtol=self.cfg['root_xtol'],rtol=self.cfg['root_rtol']))
                chosen=(lo,hi);break
        result.update(n_stationary_points=len(stationary),n_search_segments=len(points)-1,
                      objective_at_start=objective(N),objective_at_max=objective(upper))
        if root is None:return dict(**result,status='no_finite_equivalence_within_range',N_eq_B=None)
        change=root/N-1
        return dict(**result,status='solved',N_eq_B=root,equivalent_parameter_gain=change,
                    equivalent_parameter_gain_pct=100*change,local_approx_gain_pct=100*local,
                    local_minus_exact_pct_points=100*(local-change),
                    local_relative_error=(local-change)/change,
                    equation_residual=objective(root),root_bracket_low_B=chosen[0],root_bracket_high_B=chosen[1])


def finite_difference(model,N,D,Q,deltaM,variable,config,side=None):
    """Fourth-order central or one-sided derivative; no stencil crosses an N knot."""
    vals=dict(N=float(N),D=float(D),Q=float(Q));x=vals[variable]
    h=config['finite_difference_relative_step']*x
    def f(offset):
        v=dict(vals);v[variable]=x+offset*h
        return float(model.terms(deltaM=deltaM,**v)['L_gen'])
    if side=='right':return (-25*f(0)+48*f(1)-36*f(2)+16*f(3)-3*f(4))/(12*h)
    if side=='left':return (25*f(0)-48*f(-1)+36*f(-2)-16*f(-3)+3*f(-4))/(12*h)
    return (-f(2)+8*f(1)-8*f(-1)+f(-2))/(12*h)

# ============================================================
# Source: src/scaling/marginal_reporting.py
# Original lines: 1-245
# Purpose: 最终CSV/图表/报告呈现，compact省略纯绘图和排版
# ============================================================
"""Question 2 final scientific plots and report, with conditional evidence retained."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .mixture_transfer_reporting import markdown_table
from .io import write_csv


def plot_analysis(out,figdir,grid,eqs,domains,joints,metrics,sensitivity,cfg):
    """Standalone scientific figures; every plotted value is from computed output."""
    out,figdir=Path(out),Path(figdir);manifest=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name,description):
        fig.savefig(figdir/name,dpi=cfg['figure_dpi'],bbox_inches='tight');plt.close(fig)
        manifest.append(dict(filename=name,description=description))
    g=pd.DataFrame(grid);q=pd.DataFrame(eqs)
    main=g[(g.recipe=='p_ref')&(g.Q_label=='Q_0.7')&(g.D_label=='median')].sort_values('N_params_B')
    fig,ax=plt.subplots(figsize=(8,4.6))
    for factor,color in [('N','#28649b'),('D','#39915b'),('Q','#bb6a28')]:
        ax.plot(main.N_params_B,main['E_'+factor],marker='o',color=color,label='E_'+factor)
    ax.axvline(11.97,color='gray',ls='--',lw=1,label='B6/B7 upper N boundary')
    ax.set_xscale('log');ax.set_xlabel('N (B parameters, log scale)');ax.set_ylabel('Loss-reduction elasticity of R=L-E')
    ax.set_title('Reference recipe, Q=0.7, median D');ax.legend(fontsize=9)
    save(fig,'01_typical_elasticities.png','Main elasticities; outside B6/B7 range is extrapolative')
    fig,axes=plt.subplots(1,3,figsize=(12,4.6),sharey=True)
    for ax,recipe in zip(axes,['p_ref','P10','P90']):
        sub=g[(g.recipe==recipe)&(g.Q_label=='Q_0.7')]
        table=sub.pivot(index='D_tokens_B',columns='N_params_B',values='dominant_factor')
        values=np.array([[{'N':0,'D':1,'Q':2}[v] for v in row] for row in table.to_numpy()])
        from matplotlib.colors import ListedColormap
        im=ax.imshow(values,aspect='auto',cmap=ListedColormap(['#28649b','#39915b','#bb6a28']),vmin=-.5,vmax=2.5)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):ax.text(j,i,['N','D','Q'][values[i,j]],ha='center',va='center',color='white')
        ax.set_xticks(range(len(table.columns)),[f'{v:g}' for v in table.columns],rotation=45)
        ax.set_yticks(range(len(table.index)),[f'{v:g}' for v in table.index]);ax.set_xlabel('N (B)');ax.set_ylabel('D (B tokens)')
        ax.set_title(recipe)
    fig.suptitle('Dominant loss-reduction elasticity at Q=0.7 (right derivative at knots)');fig.tight_layout()
    save(fig,'02_dominant_factor_regions.png','Categorical comparison at real saved recipe baselines')
    sub=q[(q.recipe=='p_ref')&(q.D_label=='median')&(q.N_params_B.isin(cfg['N_primary_B']))&(q.Q_label!='Q_0.7')]
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,N in zip(axes,cfg['N_primary_B']):
        s=sub[sub.N_params_B==N].sort_values('Q');x=np.arange(len(s))
        ax.bar(x-.18,s.equivalent_parameter_gain_pct,width=.36,label='Exact first root')
        ax.bar(x+.18,s.local_approx_gain_pct,width=.36,label='Local approximation')
        ax.set_xticks(x,[f'{v:.3f}' for v in s.Q]);ax.set_xlabel('Initial Q');ax.set_ylabel('Equivalent parameter increase (%)');ax.set_title(f'N={N:g}B')
        ax.legend(fontsize=8)
    fig.suptitle('Q + 0.1 equivalence, reference recipe (12B slightly outside B6/B7 range)');fig.tight_layout()
    save(fig,'03_Q01_parameter_equivalence.png','Exact numerical equivalence versus local approximation, nine main workpoints')
    domain=pd.DataFrame(domains);scales=['1M','60M','1B','10B','70B']
    table=domain.pivot(index='domain',columns='scale',values='DeltaL_j')[scales]
    names=[v.removeprefix('train_the_pile_') for v in table.index]
    extent=float(abs(table.to_numpy()).max());fig,ax=plt.subplots(figsize=(8,7.5))
    im=ax.imshow(table.to_numpy(),cmap='RdBu_r',vmin=-extent,vmax=extent,aspect='auto')
    ax.set_xticks(range(5),['1M observed','60M observed','1B observed','10B estimated','70B estimated'],rotation=25,ha='right')
    ax.set_yticks(range(17),names);ax.set_title('Domain +1pp: mean Loss change across five frozen baselines')
    fig.colorbar(im,ax=ax,label='Delta Loss');fig.tight_layout()
    save(fig,'04_domain_effect_by_scale.png','Scaled effects from Question 1; estimated directions explicitly marked')
    pairs=pd.DataFrame(joints);globalmax=float(abs(pairs.scaled_I_jk).max())
    for scale in scales:
        matrix=pd.read_csv(out/f'joint_trend_matrix_{scale}.csv',index_col=0)
        fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(matrix.to_numpy(),cmap='RdBu_r',vmin=-globalmax,vmax=globalmax)
        names=[v.removeprefix('train_the_pile_') for v in matrix.index]
        ax.set_xticks(range(17),names,rotation=90,fontsize=7);ax.set_yticks(range(17),names,fontsize=7)
        ax.set_title(f'Joint perturbation deviation at {scale} ('+('estimated-supported' if scale in ['10B','70B'] else 'observed-supported')+')')
        fig.colorbar(im,ax=ax,label='Scaled I_jk (diagonal not applicable)');fig.tight_layout()
        save(fig,'05_joint_trend_'+scale+'.png','136 pair deviations mirrored; diagonal is not an observed pair')
    pred=pd.read_csv(out/'B10_reference_predictions.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));ax=axes[0]
    sc=ax.scatter(pred.val_loss,pred.L0_prediction,c=np.log10(pred.N_params_B),cmap='viridis',s=24)
    lo=min(pred.val_loss.min(),pred.L0_prediction.min());hi=max(pred.val_loss.max(),pred.L0_prediction.max())
    ax.plot([lo,hi],[lo,hi],color='gray',lw=1);ax.set_xlabel('B10 estimated Loss');ax.set_ylabel('Frozen L0 prediction')
    ax.set_title('Estimated backbone consistency only');fig.colorbar(sc,ax=ax,label='log10(N/B)')
    ax=axes[1];ax.scatter(pred.N_params_B,pred.residual_prediction_minus_estimated,s=24)
    ax.axhline(0,color='gray',lw=1);ax.set_xscale('log');ax.set_xlabel('N (B)');ax.set_ylabel('Prediction minus estimated Loss')
    ax.set_title('B10 residuals; Q=1 and p=p_ref are structural references');fig.tight_layout()
    save(fig,'06_B10_estimated_consistency.png','No Q/p observations inferred and no full-model validation claimed')
    ss=pd.DataFrame(sensitivity);fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharex=True)
    for ax,recipe in zip(axes,['P10','P90']):
        for policy,style in [('S-A','-'),('S-B','--'),('S-C',':')]:
            s=ss[(ss.recipe==recipe)&(ss.policy==policy)].sort_values('N_params_B')
            ax.plot(s.N_params_B,s.Delta_p,marker='o',ls=style,label=policy)
        ax.axhline(0,color='gray',lw=.7);ax.set_xscale('log');ax.set_xlabel('N (B)');ax.set_ylabel('Mixture correction Delta_p')
        ax.set_title(recipe+' frozen recipe');ax.legend()
    fig.suptitle('Beyond 70B policy sensitivity; S-A is the main boundary rule');fig.tight_layout()
    save(fig,'07_lambda_beyond70B_sensitivity.png','Hold-last, log continuation, and shrink-to-zero are compared without selection')
    checks=pd.read_csv(out/'marginal_derivative_checks.csv');fig,ax=plt.subplots(figsize=(6,4.4))
    for variable in ['N','D','Q']:
        s=checks[checks.variable==variable];ax.scatter(s.analytic,s.finite_difference,s=12,alpha=.55,label=variable)
    lo=checks.analytic.min();hi=checks.analytic.max();ax.plot([lo,hi],[lo,hi],color='gray',lw=1)
    ax.set_xlabel('Analytic derivative');ax.set_ylabel('Fourth-order finite difference');ax.set_title('Derivative verification, including one-sided N knots');ax.legend()
    save(fig,'08_derivative_verification.png','Formula checks are not synthetic joint observation data')
    write_csv(out/'figure_manifest.csv',manifest)


def build_final_report(path,out,final,points,grid,eqs,dominance,regions,domains,joints,metrics,sensitivity,schemas,cfg):
    """Seventeen requested topics, using actual rows and inherited module limitations."""
    c,q=final['classical'],final['quality'];out=Path(out)
    main=pd.read_csv(out/'Q01_parameter_equivalence.csv',float_precision='round_trip').to_dict('records')
    recipes=pd.read_csv(out/'Q01_parameter_equivalence_recipe_sensitivity.csv',float_precision='round_trip').to_dict('records')
    g=pd.DataFrame(grid);typ=g[(g.recipe=='p_ref')&(g.Q_label=='Q_0.7')&(g.D_label=='median')].to_dict('records')
    eq_summary=json.loads((out/'equivalence_summary.json').read_text(encoding='utf-8'))
    region=pd.DataFrame(regions);status=region.groupby('status')['n'].sum().to_dict()
    unavailable=pd.read_csv(out/'no_equivalence_records.csv',float_precision='round_trip')
    local_missing=unavailable[unavailable.status=='no_local_equivalence']
    missing_N=sorted(local_missing.N_params_B.unique().tolist())
    missing_indices=sorted(local_missing.recipe_index.astype(int).unique().tolist())
    recipe_indices={r['recipe']:r['index'] for r in points['recipes']}
    joins=pd.read_csv(out/'B9_B10_join.csv')
    join_counts=joins.join_status.value_counts().to_dict()
    d=pd.DataFrame(domains);j=pd.DataFrame(joints);sen=pd.DataFrame(sensitivity)
    jointcounts=[]
    for scale,s in j.groupby('scale',sort=False):
        jointcounts.append(dict(scale=scale,negative_mean=int((s.scaled_I_jk<0).sum()),positive_mean=int((s.scaled_I_jk>0).sum()),
            stable_negative_across_five_baselines=int((s.scaled_max<0).sum()),stable_positive_across_five_baselines=int((s.scaled_min>0).sum()),
            baseline_dependent=int(((s.scaled_min<=0)&(s.scaled_max>=0)).sum())))
    effects=d[d.scale=='1M'].sort_values('DeltaM_j')
    evidence=final['evidence'];component=pd.read_csv(out.parent/'phase3/component_validation.csv',float_precision='round_trip').to_dict('records')
    for r in component:
        for k,v in r.items():
            if isinstance(v,float) and np.isnan(v):r[k]=None
    content=['# 问题二最终报告：模块化广义标度律、边际收益与参数等价','',
      '问题二按人工确认冻结的模块化形式完成。所有骨架、质量和配比参数保持原值，仅分析、推导、数值求解及授权的外推一致性检查。'
      '主 λ 规则为 ln N 分段线性插值，低于 1M 保持首节点，高于 70B 保持末节点。没有进入问题三。','',
      '## 1. 经典标度律','',r'$L_0(N,D)=E+AN^{-\alpha}+BD^{-\beta}$。N、D 均以十亿为单位。','',
      markdown_table([dict(parameter=k,value=v) for k,v in c.items()],['parameter','value']),'',
      '读取 PHASE 1 的 trajectory_balanced 主行，程序使用完整精度。主拟合数据是 B1，未使用 B9/B10 回调参数。','',
      '## 2. 质量模块及验证边界','',
      r'$\Delta_Q=C_Q(N/N_0)^{-u_Q}(-\ln Q)$，$N_0=1$B。','',
      f"C_Q={q['C_Q']:.17g}，u_Q={q['u_Q']:.17g}，从冻结 PHASE 2C S3 读取；删除已确认 v≈0 的 D 项，未重拟合。",'',
      '仅称“B6 校准的局部质量修正项”。Q 对 B6 Loss 有增量解释能力，但 B7 非重叠的组内差分幅值没有稳定复现。'
      'B8 与此前机制不兼容，不宣称存在全局统一质量弹性。第一问领域 Q 仅提供数值工作点，不证明与 B6 的 Q 具有同一生成语义。','',
      f"B6/B7 数值支持范围：{points['quality_numeric_support']}。12B 略高于真实上限 11.97B，已标记轻微外推；70B、100B、1000B 为明显外推。",'',
      '## 3. 配比模块与 λ 的尺度变化','',
      'M(p) 是冻结 ILR + 一阶 ElasticNet 对配方的 1M L_mean 预测。参考配方为冻结训练 ILR 中心，未重新读取或拟合 A4–A15。','',
      markdown_table(final['lambda_knots'],['N_params_B','lambda_s','CI_low','CI_high','evidence']),'',
      '三个真实尺度同为正，两个估算尺度同为负。方向翻转及过零结论依赖估算表。'
      '超过 1B 的配比段已经包含 Class D 证据，不能把 70B 以内的所有配比结果都称为真实观测支持。','',
      '## 4. λ 的最终分段与边界规则','',
      r'$s_i=(\lambda_{i+1}-\lambda_i)/\ln(N_{i+1}/N_i)$；内部 $\lambda(N)=\lambda_i+s_i\ln(N/N_i)$，$\lambda\prime(N)=s_i/N$。','',
      'N<1M：λ=λ_1M，导数为 0。N>70B：主方案 λ=λ_70B，导数为 0。'
      '节点上 λ 连续，但 N 导数通常不连续，表中同时记录左右弹性；主口径使用右导数。','',
      '特别是 1B 节点本身的 λ 来自真实检验，但其右导数使用 1B→10B 段，依赖估算表。'
      '非参考配方在 1B 的参数弹性、优势因素和局部等价判断因此含 Class D 条件性证据。','',
      markdown_table(pd.read_csv(out/'lambda_piecewise_definition.csv').where(pd.notna(pd.read_csv(out/'lambda_piecewise_definition.csv')),None).to_dict('records'),
                     ['segment','N_low','N_high','lambda_low','lambda_high','s_i']),'',
      '## 5. 模块化广义标度律及退化条件','',
      r'$L_{gen}=E+AN^{-\alpha}+BD^{-\beta}+C_Q(N/N_0)^{-u_Q}(-\ln Q)+\lambda_p(N)[M(p)-M(p_{ref})]$。','',
      f"M(p_ref)={final['reference']['M_ref']:.17g}。17 维参考配方保存在冻结 PHASE 3 reference_recipe.json。",'',
      r'$Q=1$、$p=p_{ref}$ 时两个修正项均为 0，$L_{gen}=L_0$。该退化是结构恒等，不是观测到的大模型质量或配方。','',
      '## 6. N/D/Q 边际效应及有限差分验证','',
      r'令 $\Delta M=M(p)-M(p_{ref})$：','',
      r'$\partial_D L=-\beta B D^{-\beta-1}$；$\partial_Q L=-C_Q(N/N_0)^{-u_Q}/Q$。','',
      r'$\partial_N L=-\alpha A N^{-\alpha-1}-(u_Q/N)\Delta_Q+(s_i/N)\Delta M$。','',
      '边界保持区令 s_i=0。在折点处分别取左右 s_i。使用四阶中心差分检查内部，四阶单侧差分检查折点；不使用跨折点中心差分验证单侧导数。','',
      f"marginal_derivative_checks.csv 全部通过。最大相对误差={eq_summary['derivative_max_relative_error']:.8g}。"
      '判据为相对误差≤1e−5 或绝对误差≤1e−8。','',
      '## 7. 可约 Loss 的降低弹性','',
      r'$R=L_{gen}-E$，$\mathcal E_x=-\partial\ln R/\partial\ln x$。正值表示增加资源降低可约 Loss。','',
      r'$\mathcal E_D=\beta BD^{-\beta}/R$；$\mathcal E_Q=C_Q(N/N_0)^{-u_Q}/R$。','',
      r'$\mathcal E_N=[\alpha AN^{-\alpha}+u_Q\Delta_Q-s_i\Delta M]/R$。','',
      '仅 R>0 时计算对数弹性。传统 x/L·∂L/∂x 保留在 elasticity_grid.csv 的 traditional_N/D/Q 附表字段，正文使用上述降低弹性。'
      '弹性比较的是相同相对变化比例，不等于相同成本或相同绝对增量的收益。','',
      '## 8. 工作点与优势区域','',
      f"D 从 B6 的 45 个不同 N-D 块取 P25/median/P75：{points['D_quantiles']}B。"
      '这避免把重复 Q 层当作额外 D 设计；不是删去任何原始数据。三个值均在 B6 支持范围内。',
      f"Q 为七个领域 Q_domain 的最小、中位、最大值，另加 0.7：{points['domain_Q_points']}。",'',
      f"配方为 p_ref、冻结 M(p) P10 邻近配方 index={recipe_indices['P10']}、P90 邻近配方 index={recipe_indices['P90']}。"
      f"完整工作点 N×D×Q×p 共 {len(grid)} 条，不创造配方。",'',
      markdown_table(typ,['N_params_B','D_tokens_B','Q','E_N','E_D','E_Q','dominant_factor','extrapolative']),'',
      markdown_table([r for r in dominance if r['dominant_count']>0],['N_params_B','recipe','factor','dominant_count','n_workpoints','dominant_fraction']),'',
      '上述优势区是该冻结候选的结构结果。估算配比段及质量外推的证据等级不会因弹性值较大而提高。','',
      '## 9. Q 提升 0.1 的精确参数等价','',
      r'固定 D、p，求首个 $N_{eq}>N$：$L_{gen}(N_{eq},D,Q,p)=L_{gen}(N,D,Q+0.1,p)$。','',
      r'等价参数增加为 $N_{eq}/N-1$。E 和 $BD^{-\beta}$ 从方程两侧严格抵消，因此本模型的等价解不随固定 D 改变。','',
      '正式结果使用 brentq。搜索上限 max(100N,10000B)。每个分段内加入全部 λ 节点及 Gain_N=0 的驻点，'
      '检查所有单调子区间，返回第一根，避免在非单调情况下漏掉根。','',
      markdown_table(main,['N_params_B','D_tokens_B','Q','Q_new','N_eq_B','equivalent_parameter_gain_pct','local_approx_gain_pct',
                           'quality_support','equivalent_N_outside_quality_support']),'',
      '主表仅九条，12B 的原工作点及其等价 N 已超质量拟合上限。Q 数值取自真实附件输出，但上述等价结论仍条件于 B6 局部质量模块。','',
      '## 10. 等价存在条件与局部解析近似','',
      r'$Gain_N=-N\partial_NL=\alpha AN^{-\alpha}+u_Q\Delta_Q-s_i\Delta M$。','',
      'Gain_N≤0 时标记 no_local_equivalence，不强行求根。Gain_N>0 但搜索范围内无根时标记 no_finite_equivalence_within_range。'
      '后者仅表示指定有限范围内无解，不能断言任意无穷范围都无解。折点采用已声明的右侧局部判据。','',
      r'$\Delta N/N\approx[C_Q(N/N_0)^{-u_Q}/(Q\,Gain_N)]\Delta Q$，这里 $\Delta Q=0.1$。','',
      '0.1 不一定是严格小扰动；局部式只解释替代条件，不能替代数值解。','',
      f"216 个代表网格的状态：{eq_summary['status_counts']}；局部近似平均绝对相对误差="
      f"{eq_summary['approximation_absolute_relative_error_mean']:.6%}，最大={eq_summary['approximation_absolute_relative_error_max']:.6%}。",'',
      f"进一步复用全部 512 个真实训练配方在相同 N/Q 工作点诊断，状态计数：{status}。"
      'training_recipe_equivalence_diagnosis.csv 保留全部记录，no_equivalence_records.csv 保留无局部/无有限解情况。'
      '这仅是模型结构诊断，不是新的大模型观测或优化。','',
      f"本次无局部等价发生的 N={missing_N}B，涉及配方 index={missing_indices}。"
      '其中 1B 右侧的判断依赖估算支持的 1B→10B 斜率。'
      '这些工作点按指令没有求全局等价根，因此不能说它们在任意更大 N 都无有限解。其余完成搜索的工作点未检测到范围内无根。','',
      markdown_table(recipes,['recipe','recipe_index','N_params_B','Q','N_eq_B','equivalent_parameter_gain_pct','status']),'',
      '## 11. 十七领域的尺度化边际效应','',
      r'$\Delta L_j(N)=\lambda_p(N)\Delta M_j$，复用问题一每域增加 1 个百分点的冻结结果。','',
      '主表的 ΔM_j 为五个代表基准的等权平均，另保留全部基准案例、零激活及最小/最大范围。'
      '存在基准依赖的领域不能按平均值宣称普遍有利或不利。','',
      markdown_table(effects.to_dict('records'),['domain','DeltaM_j','source_baseline_direction']),'',
      '1M/60M/1B 的正 λ 保持原方向且逐步衰减；10B/70B 的负 λ 使对应方向反转，但仅是 estimated-supported 的条件性结果。'
      '完整 17×5 结果见 domain_effect_by_scale.csv。','',
      '## 12. 领域组合趋势','',
      r'$I_{jk}(N)=\lambda_p(N)I_{jk}$。每尺度 136 对、17×17 对称矩阵，主对角线不适用而非真实零偏离。','',
      markdown_table(jointcounts,['scale','negative_mean','positive_mean','stable_negative_across_five_baselines',
                                   'stable_positive_across_five_baselines','baseline_dependent']),'',
      'λ 变号时，组合偏离符号同步翻转，基准依赖程度保留。负向称条件性互补趋势，正向称条件性竞争趋势。'
      'ILR 与闭合操作本身也会形成组合偏离，不作为因果互补/竞争或真实交互的证据。','',
      '## 13. B9/B10 百亿/千亿参数级外推','',
      f"B9：{schemas['B9']['n']} 条元数据，N={schemas['B9']['N_min']}–{schemas['B9']['N_max']}B，"
      f"D={schemas['B9']['D_min']}–{schemas['B9']['D_max']}B。无 Loss，非正 D={schemas['B9']['nonpositive_D']} 条，全部保留。",'',
      f"B10：{schemas['B10']['n']} 条估算 Loss。B9.model_name 与 B10.family 精确一对一对应，128 条匹配，"
      f"{join_counts.get('left_only',0)} 条仅 B9，匹配 N/D 相同。缺失、重复及全部字段见 B9_B10_diagnosis.md。", '',
      '参考情景设置 Q=1、p=p_ref，故只预测 L0(N,D)。没有给真实大模型虚构观测 Q 或配方。'
      'B9 用于现实尺度锚点，B10 用于估算 Loss 对照，不能验证完整 N-D-Q-p 模型。','',
      markdown_table(metrics,['bucket','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias']),'',
      '分桶采用 [100,300)、[300,1000)、[1000,∞)B。统一称“百亿/千亿参数级估算一致性检验”。'
      '参数总量/激活参数、稠密/稀疏模型和不同评测口径未统一，因此高一致性也不等于真实同口径泛化性能。','',
      '## 14. 超 70B 的 λ 结构敏感性','',
      'S-A 主方案：保持 λ_70B。S-B：延续 10B→70B 的 ln N 斜率。'
      'S-C：70B 后在 ln N 上线性收缩，到 1000B 为 0，之后保持 0。未按结果好看程度更换主方案。','',
      markdown_table(sensitivity,['N_params_B','recipe','policy','lambda_p','DeltaM','Delta_p','role']),'',
      f"代表配方的三规则最大 Δp 跨方案差值为 {max(float(s.Delta_p.max()-s.Delta_p.min()) for _,s in sen.groupby(['N_params_B','recipe'])):.8g}。"
      '**超70B领域配比修正具有高外推不确定性。**一类规则保留偏移，一类持续增大，一类收缩至零；B9/B10 无法验证 p 项，不能据此选出真实机制。','',
      '## 15. 证据等级与分模块验证','',
      markdown_table(evidence,['module','parameters','evidence','support']),'',
      markdown_table(component,['module','dataset','stage','target','n','RMSE','R2','limitation']),'',
      'Class A：B1 主拟合经典五参数。Class B：B6 半合成局部质量校准。Class C：第一问真实配比实验的 M(p)。'
      'Class D：10B/70B 估算 λ 节点及其大尺度段。大尺度质量项更属于外推，不能借 Class D 配比证据补强。','',
      '## 16. 局限性','',
      '不存在完整 (N,D,Q,p,L) 联合直接观测。采用 modular calibration + component-wise validation，无法进行单一完整联合外部验证。'
      '相加结构未验证 Q×p 等联合交互。不同尺度配方覆盖和评测分布可能混杂传递系数变化。'
      '第一问领域 Q 与 B6 Q 的语义一致性没有联合观测支持。估算表不会变成真实实验。'
      '等价参数数值是冻结模型内的条件性替代，不是训练实测承诺，也未纳入质量改善成本或算力约束优化。','',
      '## 17. 结论与停止点','',
      '经典骨架、局部质量修正、尺度调制配比项可按明确锚点组合，所有拟合参数保留冻结值。'
      '质量提升与参数扩张的等价使用精确求根，局部近似只解释机制。配比及组合方向反转依赖估算表，超 70B 边界规则具有结构不确定性。'
      'B10 只支持经典骨架的估算一致性，不能补充缺失的完整联合验证。问题二完成后停止，不进入问题三。','',
      '复现：python -B scripts/16_analyze_generalized_scaling.py。独立核验：python -B scripts/16_verify_generalized_scaling.py。'
      '全部结果位于 outputs/scaling/phase4，图表位于 outputs/figures/question2_phase4。','']
    Path(path).write_text('\n'.join(content),encoding='utf-8')

# ============================================================
# Source: scripts/16_analyze_generalized_scaling.py
# Original lines: 1-432
# Purpose: 最终阶段实际执行入口：16_analyze_generalized_scaling.py
# ============================================================
"""Question 2 final phase; frozen analysis only, no refitting and no Question 3.

Run: python -B scripts/16_analyze_generalized_scaling.py
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
from scipy.stats import pearsonr,spearmanr
from src.scaling.io import sha256,write_csv,write_json,csv_rows
from src.scaling.quality_extension import read_parameters
from src.scaling.mixture_transfer import frozen_metadata,predict_recipe,predict_ilr
from src.scaling.marginal_analysis import FrozenGeneralizedScaling,finite_difference

OUT=ROOT/'outputs/scaling/phase4'
FIG=ROOT/'outputs/figures/question2_phase4'
REPORT=ROOT/'outputs/report/question2_final.md'
NEW={'configs/scaling_phase4.yaml','src/scaling/marginal_analysis.py',
     'src/scaling/marginal_reporting.py','scripts/16_analyze_generalized_scaling.py',
     'scripts/16_verify_generalized_scaling.py'}
INPUTS=[]


def read(relative):
    """Read a declared CSV with round-trip float precision, recording provenance."""
    INPUTS.append(relative)
    return pd.read_csv(ROOT/relative,float_precision='round_trip')


def protected_snapshot():
    """Protect all files from earlier phases, including raw attachment bytes."""
    path=OUT/'frozen_before_phase4.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    rows=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT).as_posix()
        if rel in NEW or p==REPORT or rel.startswith('outputs/scaling/phase4/') or rel.startswith('outputs/figures/question2_phase4/'):
            continue
        rows.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha256(p)))
    result=dict(files=rows,n_files=len(rows))
    write_json(path,result);return result


def verify_frozen(snapshot):
    """Check bytes; never parse original A attachments or rerun previous scripts."""
    bad=[]
    for r in snapshot['files']:
        p=ROOT/r['path']
        if not p.is_file() or p.stat().st_size!=r['bytes'] or sha256(p)!=r['sha256']:bad.append(r['path'])
    if bad:raise RuntimeError('Frozen earlier files changed: '+repr(bad))
    return dict(passed=True,n_files=len(snapshot['files']),changed_files=[])


def diagnose_large():
    """Inspect verified disk schemas before deciding admissible B9/B10 comparison."""
    base='real_attachments/B_scaling_laws/'
    paths=dict(B9=base+'supplementary_large_models.csv',B10=base+'supplementary_large_baseline.csv')
    schemas={};frames={};fields=[]
    for label,relative in paths.items():
        names,records=csv_rows(ROOT/relative)
        t=read(relative);frames[label]=t
        schemas[label]=dict(path=relative,columns=names,n=len(records),
            exact_duplicate_rows=int(t.duplicated().sum()),missing=t.isna().sum().to_dict())
        fields.extend(dict(dataset=label,field=c,missing=int(t[c].isna().sum()),
                           dtype=str(t[c].dtype)) for c in names)
    # These are the actual schemas established by the initial read above.
    expected={'B9':{'model_name','N_params_B','D_tokens_B'},
              'B10':{'family','N_params_B','D_tokens_B','val_loss'}}
    for label,names in expected.items():
        if not names.issubset(frames[label].columns):raise ValueError(label+' schema differs; stop dependent analysis')
    a,b=frames['B9'],frames['B10']
    for label,t,key in [('B9',a,'model_name'),('B10',b,'family')]:
        if t[key].isna().any() or t[key].duplicated().any():raise ValueError('Ambiguous large-model join key')
        schemas[label].update(N_min=float(t.N_params_B.min()),N_max=float(t.N_params_B.max()),
              D_min=float(t.D_tokens_B.min()),D_max=float(t.D_tokens_B.max()),
              nonpositive_N=int((t.N_params_B<=0).sum()),nonpositive_D=int((t.D_tokens_B<=0).sum()),
              name_duplicates=int(t[key].duplicated().sum()))
    j=a.merge(b,left_on='model_name',right_on='family',how='outer',suffixes=['_B9','_B10'],indicator=True,validate='one_to_one')
    both=j._merge=='both'
    if not np.array_equal(j.loc[both,'N_params_B_B9'],j.loc[both,'N_params_B_B10']) or not np.array_equal(j.loc[both,'D_tokens_B_B9'],j.loc[both,'D_tokens_B_B10']):
        raise ValueError('Matched large-model scales disagree')
    j['join_status']=j.pop('_merge').astype(str)
    j.to_csv(OUT/'B9_B10_join.csv',index=False,encoding='utf-8-sig')
    write_csv(OUT/'B9_B10_field_diagnosis.csv',fields)
    write_json(OUT/'B9_B10_schema.json',schemas)
    notes=['# B9/B10 实际数据诊断','',
        'B9 是模型元数据，无直接可比较 Loss。B10 的磁盘字段名为 val_loss，但按赛题口径是估算 Loss。','']
    for label,s in schemas.items():
        notes += [f"## {label}",'',f"文件：{s['path']}",f"实际字段：{s['columns']}",
                  f"行数：{s['n']}；N 范围：{s['N_min']}–{s['N_max']}B；D 范围：{s['D_min']}–{s['D_max']}B。",
                  f"缺失字段计数：{s['missing']}；完全重复：{s['exact_duplicate_rows']}；名称重复：{s['name_duplicates']}。",
                  f"非正 N：{s['nonpositive_N']}；非正 D：{s['nonpositive_D']}。",'']
    notes += ['## 对应关系','',f"按 B9.model_name = B10.family 精确一对一关联：匹配 {int(both.sum())}，"
              f"仅 B9 {int((j.join_status=='left_only').sum())}，仅 B10 {int((j.join_status=='right_only').sum())}。",
              '匹配行 N、D 完全一致。B9 中 D=0 的行完整保留，D=0 不可代入经典公式，不能视作实际零训练或擅自填补。',
              'B9 只提供所报告参数规模和训练数据规模的现实锚点。元数据未统一稠密/稀疏模型、总/激活参数及各家评测口径。',
              'B10 对照仅设置结构参考 Q=1、p=p_ref，计算 L0；这不是测得的大模型真实 Q/p，也不验证完整广义模型。','']
    (OUT/'B9_B10_diagnosis.md').write_text('\n'.join(notes),encoding='utf-8')
    return a,b,schemas


def frozen_model(cfg):
    """Load full-precision frozen parameters, never invoke fitting functions."""
    classical,path=read_parameters(ROOT);INPUTS.append(path.relative_to(ROOT).as_posix())
    relative='outputs/scaling/phase3/generalized_scaling_candidate.json';INPUTS.append(relative)
    prior=json.loads((ROOT/relative).read_text(encoding='utf-8'))
    if classical!=prior['classical']:raise ValueError('Frozen backbone mismatch')
    params=read('outputs/scaling/phase2C/model_parameters.csv')
    qp=json.loads(params[params.model=='S3'].iloc[0].parameters)
    if prior['quality']['C_Q']!=qp['C'] or prior['quality']['u_Q']!=qp['u']:
        raise ValueError('Frozen quality module mismatch')
    t=read('outputs/scaling/phase3/mixture_transfer_by_scale.csv').sort_values('N_params_B')
    knots=t[['N_params_B','lambda_s','evidence','CI_low','CI_high']].to_dict('records')
    if any(k['lambda_s']!=old['lambda_s'] for k,old in zip(knots,prior['lambda_knots'])):
        raise ValueError('Frozen lambda nodes changed')
    INPUTS.extend(['outputs/mixture/final_model_metadata.json','outputs/mixture/stage2/final_model_policy.json'])
    meta=frozen_metadata(ROOT)
    model=FrozenGeneralizedScaling(classical,prior['quality'],meta,prior['reference']['M_ref'],knots,cfg)
    return model,prior,knots


def workpoints(model,prior,cfg):
    """Derive observed D quantiles, domain-Q quantiles and actual saved recipes."""
    b6=read('outputs/scaling/phase2C/B6_predictions.csv')
    b7=read('outputs/scaling/phase2C/B7_predictions.csv')
    pairs=np.array(sorted(set(zip(b6.N_params_B,b6.D_tokens_B))))
    ds=np.quantile(pairs[:,1],cfg['D_quantiles'])
    domains=read('outputs/quality/domain_quality_summary_A1.csv')
    qs=[('domain_min',float(domains.Q_domain.min())),('domain_median',float(domains.Q_domain.median())),
        ('domain_max',float(domains.Q_domain.max())),('Q_0.7',float(cfg['Q_representative']))]
    bases=read('outputs/mixture/stage2/representative_baselines.csv')
    recipes=[]
    for label,name in [('p_ref','A4_ILR_center'),('P10','A4_predicted_low'),('P90','A4_predicted_high')]:
        g=bases[bases.baseline==name]
        if len(g)!=1:raise ValueError('Missing frozen recipe baseline')
        p=g[model.meta['domain_order']].to_numpy()[0]
        m=float(predict_recipe(p,model.meta)[0])
        if abs(m-float(g.iloc[0].frozen_EN_prediction))>1e-12:raise ValueError('Frozen baseline M(p) mismatch')
        # p_ref uses the already frozen reference and score, exactly.
        delta=0. if label=='p_ref' else m-model.M_ref
        recipes.append(dict(recipe=label,baseline=name,index=None if pd.isna(g.iloc[0].recipe_index) else int(g.iloc[0].recipe_index),
                            p=p.tolist(),M_p=m,DeltaM=delta))
    support=dict(N_min_B=float(min(b6.N_params_B.min(),b7.N_params_B.min())),
                 N_max_B=float(max(b6.N_params_B.max(),b7.N_params_B.max())),
                 D_min_B=float(min(b6.D_tokens_B.min(),b7.D_tokens_B.min())),
                 D_max_B=float(max(b6.D_tokens_B.max(),b7.D_tokens_B.max())),
                 Q_min=float(min(b6.Q_score.min(),b7.Q_score.min())),Q_max=float(max(b6.Q_score.max(),b7.Q_score.max())))
    result=dict(D_quantiles=dict(zip(['P25','median','P75'],map(float,ds))),
        D_definition='B6 observed 45 distinct N-D blocks; exclude repeated Q levels, not data records',
        domain_Q_points=dict(qs),domain_Q_values=domains[['domain','Q_domain']].to_dict('records'),
        recipes=recipes,quality_numeric_support=support,Q_semantic_alignment_not_jointly_observed=True,
        primary_12B_is_slightly_above_actual_11_97B=12>support['N_max_B'])
    write_json(OUT/'workpoint_provenance.json',result)
    return ds,qs,recipes,support,result


def lambda_definition(model):
    """Exact main-policy segments, including open positive-N clamp boundaries."""
    r=[dict(segment='lower_clamp',N_low=0.,N_high=float(model.Ns[0]),
            lambda_low=float(model.lambdas[0]),lambda_high=float(model.lambdas[0]),s_i=0.,domain='0<N<1M')]
    for i,s in enumerate(model.slopes):
        r.append(dict(segment=f'interior_{i+1}',N_low=float(model.Ns[i]),N_high=float(model.Ns[i+1]),
               lambda_low=float(model.lambdas[i]),lambda_high=float(model.lambdas[i+1]),s_i=float(s),
               domain='continuous at knots; one-sided derivatives'))
    r.append(dict(segment='upper_clamp',N_low=float(model.Ns[-1]),N_high=None,
                  lambda_low=float(model.lambdas[-1]),lambda_high=float(model.lambdas[-1]),s_i=0.,domain='N>70B'))
    write_csv(OUT/'lambda_piecewise_definition.csv',r)
    return r


def elasticity_grid(model,ds,qs,recipes,support,cfg):
    """All requested workpoints, derivative side checks, and exact Q+0.1 roots."""
    grid=[];checks=[];eqs=[]
    for N in cfg['N_workpoints_B']:
        knot=bool(N in model.Ns)
        for dlabel,D in zip(['P25','median','P75'],ds):
            for qlabel,Q in qs:
                for recipe in recipes:
                    dm=recipe['DeltaM'];side=cfg['knot_derivative_main']
                    terms=model.terms(N,D,Q,dm);der=model.derivatives(N,D,Q,dm,side=side)
                    elastic=model.elasticity(N,D,Q,dm,side=side)
                    left=model.elasticity(N,D,Q,dm,side='left');right=model.elasticity(N,D,Q,dm,side='right')
                    valid=bool(elastic['valid']);dominant=max(['N','D','Q'],key=lambda k:float(elastic['E_'+k])) if valid else None
                    record=dict(N_params_B=N,D_tokens_B=float(D),D_label=dlabel,Q=Q,Q_label=qlabel,
                        recipe=recipe['recipe'],recipe_index=recipe['index'],DeltaM=dm,
                        lambda_p=float(model.lambda_value(N)),**{k:float(v) for k,v in terms.items()},
                        **{k:float(v) for k,v in der.items()},
                        E_N=float(elastic['E_N']) if valid else None,E_D=float(elastic['E_D']) if valid else None,
                        E_Q=float(elastic['E_Q']) if valid else None,
                        E_N_left=float(left['E_N']) if valid else None,E_N_right=float(right['E_N']) if valid else None,
                        dominant_factor=dominant,elasticity_status='valid' if valid else 'nonpositive_reducible_loss',
                        N_is_lambda_knot=knot,N_derivative_kind='right_at_knot' if knot else 'interior',
                        extrapolative=bool(N<support['N_min_B'] or N>support['N_max_B']),
                        quality_evidence='Class B local' if support['N_min_B']<=N<=support['N_max_B'] else 'Class B extrapolation',
                        mixture_evidence='Class D estimated-supported' if N>1 else 'Class C observed-scale interpolation',
                        mixture_N_derivative_evidence=('not_applicable_reference_recipe' if dm==0 else
                            'Class D: 1B right side or larger N' if N>=1 else 'Class C observed-scale interpolation'))
                    for k in ['N','D','Q']:record['traditional_'+k]=float(elastic['traditional_'+k])
                    grid.append(record)
                    for variable in ['N','D','Q']:
                        sides=['left','right'] if variable=='N' and knot else [None]
                        for chosen in sides:
                            exact=float(model.derivatives(N,D,Q,dm,side=chosen or 'right')['dL_d'+variable])
                            numeric=finite_difference(model,N,D,Q,dm,variable,cfg,side=chosen)
                            absolute=abs(exact-numeric);relative=absolute/max(abs(exact),1e-12)
                            passed=relative<=cfg['derivative_relative_tolerance'] or absolute<=cfg['derivative_absolute_tolerance']
                            checks.append(dict(N_params_B=N,D_tokens_B=float(D),Q=Q,recipe=recipe['recipe'],variable=variable,
                                derivative_side=chosen or 'central',analytic=exact,finite_difference=numeric,
                                absolute_error=absolute,relative_error=relative,passed=passed))
                    equivalence=model.equivalence(N,Q,dm)
                    root=equivalence.get('N_eq_B')
                    eqs.append(dict(N_params_B=N,D_tokens_B=float(D),D_label=dlabel,Q=Q,Q_label=qlabel,
                        recipe=recipe['recipe'],recipe_index=recipe['index'],DeltaM=dm,
                        extrapolative=record['extrapolative'],quality_support='extrapolation' if record['extrapolative'] else 'supported_numeric_range',
                        equivalent_N_outside_quality_support=bool(root is not None and root>support['N_max_B']),
                        root_crosses_lambda_knots=bool(root is not None and np.any((model.Ns>N)&(model.Ns<root))),
                        **equivalence))
    write_csv(OUT/'elasticity_grid.csv',grid);write_csv(OUT/'marginal_derivative_checks.csv',checks)
    if not all(r['passed'] for r in checks):raise AssertionError('Derivative verification failed')
    write_csv(OUT/'Q01_parameter_equivalence_grid.csv',eqs)
    primary=[r for r in eqs if r['N_params_B'] in cfg['N_primary_B'] and r['D_label']=='median' and r['Q_label']!='Q_0.7']
    write_csv(OUT/'Q01_parameter_equivalence.csv',[r for r in primary if r['recipe']=='p_ref'])
    write_csv(OUT/'Q01_parameter_equivalence_recipe_sensitivity.csv',[r for r in primary if r['recipe']!='p_ref'])
    g=pd.DataFrame(grid);summaries=[]
    for (N,recipe),sub in g.groupby(['N_params_B','recipe']):
        for factor in ['N','D','Q']:
            summaries.append(dict(N_params_B=float(N),recipe=recipe,factor=factor,n_workpoints=len(sub),
                dominant_count=int((sub.dominant_factor==factor).sum()),
                dominant_fraction=float((sub.dominant_factor==factor).mean()),
                elasticity_min=float(sub['E_'+factor].min()),elasticity_max=float(sub['E_'+factor].max())))
    write_csv(OUT/'elasticity_dominance_summary.csv',summaries)
    solved=pd.DataFrame(eqs);solved=solved[solved.status=='solved'];v=solved.local_relative_error.to_numpy()
    write_json(OUT/'equivalence_summary.json',dict(status_counts=pd.Series([r['status'] for r in eqs]).value_counts().to_dict(),
        n_main_rows=sum(r['recipe']=='p_ref' for r in primary),
        approximation_relative_error_mean=float(v.mean()),approximation_absolute_relative_error_mean=float(np.mean(abs(v))),
        approximation_absolute_relative_error_max=float(np.max(abs(v))),
        D_independence='Exact under fixed additive D term; it cancels from both sides',
        max_equation_residual=float(np.max(abs(solved.equation_residual))),
        quality_derivative_max_relative_error=max(r['relative_error'] for r in checks if r['variable']=='Q'),
        derivative_max_relative_error=max(r['relative_error'] for r in checks)))
    return grid,checks,eqs,summaries


def training_recipe_regions(model,ds,qs,support,cfg):
    """Actual 512 saved recipes: diagnose local/finite equivalence beyond P10/P90."""
    t=read('outputs/mixture/train_1M_ILR_features.csv');M=predict_ilr(t.iloc[:,1:].to_numpy(),model.meta)
    records=[]
    for N in cfg['N_workpoints_B']:
        for label,Q in qs:
            for index,m in zip(t['index'],M):
                dm=float(m-model.M_ref);sol=model.equivalence(N,Q,dm)
                records.append(dict(recipe_index=int(index),N_params_B=N,D_tokens_B=float(ds[1]),Q=Q,Q_label=label,
                    M_p=float(m),DeltaM=dm,extrapolative=bool(N>support['N_max_B'] or N<support['N_min_B']),**sol))
    write_csv(OUT/'training_recipe_equivalence_diagnosis.csv',records)
    write_csv(OUT/'no_equivalence_records.csv',[r for r in records if r['status']!='solved'],columns=list(records[0]))
    summary=[]
    for N in cfg['N_workpoints_B']:
        for label,Q in qs:
            sub=[r for r in records if r['N_params_B']==N and r['Q_label']==label]
            for status in ['solved','no_local_equivalence','no_finite_equivalence_within_range']:
                summary.append(dict(N_params_B=N,Q=Q,Q_label=label,status=status,n=int(sum(r['status']==status for r in sub)),total=len(sub)))
    write_csv(OUT/'training_recipe_equivalence_status_summary.csv',summary)
    return summary


def domain_and_joint(model):
    """Scale frozen 1pp first-order effects and 136 pair deviations, no causal claim."""
    effects=read('outputs/mixture/domain_marginal_effects.csv')
    effects=effects[(effects.model=='M1_ElasticNet') & np.isclose(effects.step,.01,rtol=0,atol=1e-14)]
    joint=read('outputs/mixture/domain_joint_effects.csv')
    joint=joint[(joint.model=='M1_ElasticNet') & np.isclose(joint.step,.01,rtol=0,atol=1e-14)]
    if len(effects)!=17 or len(joint)!=136:raise ValueError('Frozen effect dimensions differ')
    out=[];pairs=[]
    for N,label,evidence in [( .001,'1M','observed-supported'),(.06,'60M','observed-supported'),(1.,'1B','observed-supported'),
                              (10.,'10B','estimated-supported'),(70.,'70B','estimated-supported')]:
        lam=float(model.lambda_value(N))
        for _,r in effects.iterrows():
            values=lam*np.array([r.min_effect,r.max_effect]);low,high=float(values.min()),float(values.max())
            out.append(dict(scale=label,N_params_B=N,evidence=evidence,domain=r.mixture_domain,
                step=.01,summary_scope='equal_mean_of_five_frozen_representative_baselines',
                n_baselines=int(r.n_baselines),DeltaM_j=float(r.mean_effect),lambda_p=lam,
                DeltaL_j=float(lam*r.mean_effect),min_effect=low,max_effect=high,
                direction='lower_loss' if high<0 else 'higher_loss' if low>0 else 'baseline_dependent',
                source_baseline_direction=r.baseline_direction))
        matrix=np.full((17,17),np.nan);names=model.meta['domain_order']
        for _,r in joint.iterrows():
            value=float(lam*r.mean_effect);i,j=names.index(r.domain_1),names.index(r.domain_2)
            matrix[i,j]=matrix[j,i]=value
            values=lam*np.array([r.min_effect,r.max_effect]);lo,hi=float(values.min()),float(values.max())
            pairs.append(dict(scale=label,N_params_B=N,evidence=evidence,domain_1=r.domain_1,domain_2=r.domain_2,
                step=.01,n_baselines=int(r.n_baselines),I_jk=float(r.mean_effect),lambda_p=lam,
                scaled_I_jk=value,scaled_min=lo,scaled_max=hi,
                trend='conditional_complementary' if hi<0 else 'conditional_competitive' if lo>0 else 'baseline_dependent',
                source_deviation_direction=r.baseline_deviation_direction,causal_interpretation=False))
        pd.DataFrame(matrix,index=names,columns=names).to_csv(OUT/f'joint_trend_matrix_{label}.csv',encoding='utf-8-sig',index_label='domain')
    write_csv(OUT/'domain_effect_by_scale.csv',out);write_csv(OUT/'joint_trend_by_scale.csv',pairs)
    # Preserve all five baseline cases as well as the equal-baseline summary.
    c=read('outputs/mixture/stage2/domain_marginal_effect_cases.csv')
    c=c[(c.model=='M1_ElasticNet') & np.isclose(c.step,.01,rtol=0,atol=1e-14)]
    d=read('outputs/mixture/stage2/domain_joint_effect_cases.csv')
    d=d[(d.model=='M1_ElasticNet') & np.isclose(d.step,.01,rtol=0,atol=1e-14)]
    cases=[];jc=[]
    for N in model.Ns:
        lam=float(model.lambda_value(N))
        cases.extend(dict(N_params_B=float(N),lambda_p=lam,scaled_effect=float(lam*r['effect']),**r) for r in c.to_dict('records'))
        jc.extend(dict(N_params_B=float(N),lambda_p=lam,scaled_deviation=float(lam*r['joint_perturbation_deviation']),**r) for r in d.to_dict('records'))
    write_csv(OUT/'domain_effect_cases_by_scale.csv',cases);write_csv(OUT/'joint_trend_cases_by_scale.csv',jc)
    return out,pairs


def large_validation(model,b9,b10,cfg):
    """B10 estimated consistency only at the structural reference Q=1,p=p_ref."""
    if not np.isfinite(b10[['N_params_B','D_tokens_B','val_loss']]).all().all() or (b10[['N_params_B','D_tokens_B']]<=0).any().any():
        raise ValueError('B10 invalid required scales/Loss; do not impute')
    truth=b10.val_loss.to_numpy();pred=model.terms(b10.N_params_B.to_numpy(),b10.D_tokens_B.to_numpy(),1.,0.)['L0']
    b10=b10.copy();b10['L0_prediction']=pred;b10['residual_prediction_minus_estimated']=pred-truth
    b10['Q_reference_scenario']=1.;b10['p_reference_scenario']='p_ref';b10['Delta_Q']=0.;b10['Delta_p']=0.
    b10['validation_kind']='estimated_Loss_consistency_of_classical_backbone_only'
    b10.to_csv(OUT/'B10_reference_predictions.csv',index=False,encoding='utf-8-sig')
    metrics=[]
    for name,mask in [('all',np.ones(len(b10),bool)),('100-300B',(b10.N_params_B>=100)&(b10.N_params_B<300)),
                      ('300-1000B',(b10.N_params_B>=300)&(b10.N_params_B<1000)),('1000B_and_above',b10.N_params_B>=1000)]:
        y=truth[mask];p=pred[mask]
        defined=len(y)>1 and np.ptp(y)>0 and np.ptp(p)>0
        metrics.append(dict(bucket=name,n=len(y),RMSE=float(np.sqrt(np.mean((p-y)**2))) if len(y) else None,
           MAE=float(np.mean(abs(p-y))) if len(y) else None,
           R2=float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)) if len(y)>1 and np.ptp(y)>0 else None,
           Pearson=float(pearsonr(y,p).statistic) if defined else None,Spearman=float(spearmanr(y,p).statistic) if defined else None,
           mean_bias=float(np.mean(p-y)) if len(y) else None,evidence='estimated_consistency_only',
           bucket_convention='[100,300),[300,1000),[1000,infinity)'))
    write_csv(OUT/'B10_extrapolation_metrics.csv',metrics)
    anchors=[]
    for N in cfg['N_workpoints_B']+cfg['N_lambda_sensitivity_B']:
        if N in [r['N_params_B'] for r in anchors]:continue
        exact=b9[b9.N_params_B==N]
        anchors.append(dict(N_params_B=N,within_B9_reported_N_range=bool(b9.N_params_B.min()<=N<=b9.N_params_B.max()),
              exact_reported_scale_model_count=len(exact),models_at_exact_scale=exact.model_name.tolist(),
              scope='metadata scale anchor only; total/active parameters not harmonized'))
    write_csv(OUT/'B9_scale_coverage.csv',anchors)
    return metrics


def lambda_sensitivity(model,recipes,cfg):
    """Compare authorized policies, without selecting by attractive prediction."""
    rows=[]
    for N in cfg['N_lambda_sensitivity_B']:
        for r in recipes:
            if r['recipe']=='p_ref':continue
            for policy in cfg['lambda_sensitivity']:
                lam=float(model.lambda_value(N,policy))
                rows.append(dict(N_params_B=N,recipe=r['recipe'],recipe_index=r['index'],policy=policy,
                    DeltaM=r['DeltaM'],lambda_p=lam,Delta_p=lam*r['DeltaM'],
                    lambda_log_slope=float(model.log_slope(N,policy)),
                    role='main' if policy=='S-A' else 'structural_sensitivity_only',
                    evidence='Class D extrapolation_not_validated_by_B9_B10'))
    write_csv(OUT/'lambda_beyond70B_sensitivity.csv',rows)
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
          handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'execution.log',encoding='utf-8')])
    snapshot=protected_snapshot();verify_frozen(snapshot)
    logging.info('Phase 4 starts: %d prior files frozen; analyze only',len(snapshot['files']))
    cfg=yaml.safe_load((ROOT/'configs/scaling_phase4.yaml').read_text(encoding='utf-8'));INPUTS.append('configs/scaling_phase4.yaml')
    b9,b10,schemas=diagnose_large();logging.info('Schemas diagnosed: B9=%d metadata, B10=%d estimated Loss',len(b9),len(b10))
    model,prior,knots=frozen_model(cfg)
    ds,qs,recipes,support,points=workpoints(model,prior,cfg);definition=lambda_definition(model)
    grid,checks,eqs,dominance=elasticity_grid(model,ds,qs,recipes,support,cfg)
    logging.info('Workpoint grid %d; derivative checks %d passed; equivalence statuses=%s',len(grid),len(checks),pd.Series([r['status'] for r in eqs]).value_counts().to_dict())
    regions=training_recipe_regions(model,ds,qs,support,cfg);logging.info('All 512 real training recipes checked for equivalence existence')
    domains,joints=domain_and_joint(model);metrics=large_validation(model,b9,b10,cfg)
    sensitivity=lambda_sensitivity(model,recipes,cfg)
    final=dict(classical=model.c,quality=model.q,reference=prior['reference'],lambda_knots=knots,
         lambda_main_policy='S-A',lambda_interpolation='linear_in_ln_N',below_first_knot='hold-first',above_last_knot='hold-last',
         evidence=prior['evidence'],status='human_confirmed_modular_form_with_conditional_evidence',
         joint_external_validation_available=False,no_parameters_refitted=True,N_D_units='billions',
         quality_module_interpretation='B6-calibrated local quality correction',quality_numeric_support=support,
         derivative_at_knots='left/right both recorded; right derivative is main convention',
         extrapolation_of_quality_not_empirically_validated=True)
    write_json(OUT/'generalized_scaling_final.json',final)
    INPUTS.append('outputs/scaling/phase3/component_validation.csv')
    write_csv(OUT/'input_manifest.csv',[dict(path=p,sha256=sha256(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in dict.fromkeys(INPUTS)])
    from src.scaling.marginal_reporting import plot_analysis,build_final_report
    plot_analysis(OUT,FIG,grid,eqs,domains,joints,metrics,sensitivity,cfg)
    build_final_report(REPORT,OUT,final,points,grid,eqs,dominance,regions,domains,joints,metrics,sensitivity,schemas,cfg)
    write_json(OUT/'frozen_integrity_check.json',verify_frozen(snapshot))
    write_json(OUT/'execution_summary.json',dict(completed=True,n_frozen_files=len(snapshot['files']),n_grid_points=len(grid),
        n_derivative_checks=len(checks),all_derivative_checks_passed=True,n_training_recipe_equivalence_points=512*len(cfg['N_workpoints_B'])*len(qs),
        model_refit=False,original_A_parsed=False,B9_role='metadata_anchor',B10_role='estimated_classical_consistency',
        joint_external_test_fabricated=False,question3_started=False))
    write_json(OUT/'implementation_manifest.json',[dict(path=p,sha256=sha256(ROOT/p)) for p in sorted(NEW) if (ROOT/p).exists()])
    logging.info('Question 2 Phase 4 completed normally; stopped before Question 3')


def refresh_report():
    """Refresh text from saved calculations only; never rerun analysis or plots."""
    snapshot=protected_snapshot();verify_frozen(snapshot)
    cfg=yaml.safe_load((ROOT/'configs/scaling_phase4.yaml').read_text(encoding='utf-8'))
    def records(name):return pd.read_csv(OUT/name,float_precision='round_trip').to_dict('records')
    final=json.loads((OUT/'generalized_scaling_final.json').read_text(encoding='utf-8'))
    points=json.loads((OUT/'workpoint_provenance.json').read_text(encoding='utf-8'))
    schemas=json.loads((OUT/'B9_B10_schema.json').read_text(encoding='utf-8'))
    from src.scaling.marginal_reporting import build_final_report
    build_final_report(REPORT,OUT,final,points,records('elasticity_grid.csv'),records('Q01_parameter_equivalence_grid.csv'),
        records('elasticity_dominance_summary.csv'),records('training_recipe_equivalence_status_summary.csv'),
        records('domain_effect_by_scale.csv'),records('joint_trend_by_scale.csv'),
        records('B10_extrapolation_metrics.csv'),records('lambda_beyond70B_sensitivity.csv'),schemas,cfg)
    write_json(OUT/'implementation_manifest.json',[dict(path=p,sha256=sha256(ROOT/p)) for p in sorted(NEW)])
    write_json(OUT/'frozen_integrity_check.json',verify_frozen(snapshot))
    print('Report refreshed from saved calculations; frozen files unchanged')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--refresh-report',action='store_true')
    args=parser.parse_args()
    try:
        if args.refresh_report:refresh_report()
        else:main()
    except Exception:
        logging.exception('Phase 4 stopped on error; no frozen parameters changed')
        raise

