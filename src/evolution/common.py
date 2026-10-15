"""Paths, output, conservative identity normalization and small statistics."""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr


FILES={'C1':'leaderboard_cleaned.csv','C2':'leaderboard_enhanced.csv',
       'C3':'leaderboard_extended_timeseries.csv','C4':'epoch_all_ai_models.csv',
       'C5':'loss_benchmark_bridge.csv','C6':'loss_benchmark_bridge_expanded.csv',
       'C7':'model_architecture_metadata.csv'}


def csv(path,rows):
    """Write only new-stage tables with full float precision."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    (rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)).to_csv(path,index=False,encoding='utf-8-sig')


def dump(path,value):
    """Save auditable model/diagnostic JSON."""
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def normalized_name(value):
    """Only case, whitespace, known URL prefixes and trailing slash changes."""
    if pd.isna(value):return ''
    s=str(value).strip().lower()
    for prefix in ['https://huggingface.co/','http://huggingface.co/','huggingface.co/','models/']:
        if s.startswith(prefix):s=s[len(prefix):]
    return re.sub(r'\s+','',s).rstrip('/')


def corr(x,y,method='pearson'):
    """Undefined correlation stays NaN, never becomes a fabricated zero."""
    a=np.asarray(x,float);b=np.asarray(y,float);ok=np.isfinite(a)&np.isfinite(b);a=a[ok];b=b[ok]
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:return np.nan
    return float((pearsonr if method=='pearson' else spearmanr)(a,b).statistic)


def table(g):
    """Compact Markdown report table without another dependency."""
    def fmt(v):
        if pd.isna(v):return '—'
        return f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v).replace('|','/')
    return '\n'.join(['| '+' | '.join(g.columns)+' |','| '+' | '.join(['---']*len(g.columns))+' |']+
        ['| '+' | '.join(fmt(v) for v in r)+' |' for r in g.itertuples(index=False,name=None)])
