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
