"""Read small official RegMix tables with strict index/schema checks."""
from __future__ import annotations
import csv
from decimal import Decimal
from pathlib import Path
import numpy as np

PAIRS={
    'train_1M':('train_mixture_1m.csv','train_pile_loss_1m.csv','observed'),
    'test_1M':('test_mixture_1m.csv','test_pile_loss_1m.csv','observed'),
    'test_60M':('test_mixture_60m.csv','test_pile_loss_60m.csv','observed'),
    'test_1B':('test_mixture_1B.csv','test_pile_loss_1B.csv','observed'),
    'estimated_10B':('est_mixture_10b.csv','est_pile_loss_10b.csv','estimated'),
    'estimated_70B':('est_mixture_70b.csv','est_pile_loss_70b.csv','estimated')}


def read_table(path: Path, expected_width: int):
    """Stop for malformed fields, duplicate indices or nonfinite numeric values."""
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);columns=reader.fieldnames
        if not columns or columns[0]!='index' or len(columns)!=expected_width+1 or len(set(columns))!=len(columns):
            raise ValueError(f'Unexpected schema: {path.name}')
        records=list(reader)
    if not records or any(None in r or any(v is None for v in r.values()) for r in records):raise ValueError(f'Malformed CSV: {path}')
    indices=[r['index'] for r in records]
    if any(not i for i in indices) or len(set(indices))!=len(indices):raise ValueError(f'Duplicate/empty index: {path}')
    values=np.array([[float(r[k]) for k in columns[1:]] for r in records])
    if not np.isfinite(values).all():raise ValueError(f'Nonfinite values: {path}')
    decimals=max(max(-Decimal(r[k]).as_tuple().exponent,0) for r in records for k in columns[1:])
    return {'path':path,'columns':columns[1:],'indices':indices,'values':values,'decimal_places':decimals,'records':records}


def load_datasets(root: Path):
    """Pair all A4–A15 by index, without joining indices across experiments."""
    base=root/'real_attachments/A_data_value/regmix_tables';datasets={};canonical_p=None;canonical_l=None
    for label,(mixture,loss,provenance) in PAIRS.items():
        p=read_table(base/mixture,17);l=read_table(base/loss,13)
        if set(p['indices'])!=set(l['indices']):raise ValueError(f'Non-bijective mixture/Loss indices: {label}')
        if canonical_p is None:canonical_p,canonical_l=p['columns'],l['columns']
        if p['columns']!=canonical_p or l['columns']!=canonical_l:raise ValueError(f'Changed domain column order: {label}')
        lookup={index:i for i,index in enumerate(l['indices'])};order=[lookup[index] for index in p['indices']]
        y=l['values'][order];x=p['values']
        if (x<0).any() or (x.sum(axis=1)<=0).any():raise ValueError(f'Negative or all-zero mixture: {label}')
        if (y<0).any():raise ValueError(f'Negative cross entropy: {label}')
        datasets[label]={'p_table':p,'loss_table':l,'P':x,'losses':y,'indices':p['indices'],
            'provenance':provenance,'L_mean':y.mean(axis=1),'L_trim':np.sort(y,axis=1)[:,1:-1].mean(axis=1),
            'L_median':np.median(y,axis=1)}
    return datasets,canonical_p,canonical_l


def signatures(values):
    """Composition keys after closure, rounded only for matching exported recipes."""
    closed=values/values.sum(axis=1,keepdims=True)
    return [tuple(row) for row in np.round(closed,12)]
