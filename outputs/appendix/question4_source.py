# 问题四：论文源程序逐文件汇编
# For appendix display only; NOT a standalone merged executable.
# Source bodies, imports, formulas, branches, parameters and entry guards are unchanged.
# Use the original scripts from the project root for the original workflow; DO NOT run this listing.
# This extraction task did not rerun any model or experiment.

# ============================================================
# Source: src/evolution/common.py
# Original lines: 1-49
# Purpose: 模型名规范、CSV和相关系数辅助
# ============================================================
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

# ============================================================
# Source: src/evolution/metadata.py
# Original lines: 1-157
# Purpose: C1/C2去重、S0/S1类型证据及C4元数据匹配
# ============================================================
"""Raw C-table diagnosis, per-model sample policy, and auditable joins."""
import json
import logging
import re
from difflib import get_close_matches
import numpy as np
import pandas as pd
from .common import FILES,csv,dump,normalized_name,table


def inspect_sources(root,cfg):
    """Read seven actual CSVs, compare cores, retain every raw diagnostic."""
    base=root/'real_attachments/C_efficiency_evolution';out=root/'outputs/evolution/phase1'
    data={key:pd.read_csv(base/name,float_precision='round_trip') for key,name in FILES.items()}
    summary=[];columns=[]
    for key,g in data.items():
        summary.append(dict(dataset=key,file=FILES[key],rows=len(g),columns=len(g.columns),
            exact_duplicate_rows=int(g.duplicated().sum()),duplicated_model_rows=int(g['Model' if 'Model' in g else 'model_name'].duplicated().sum())))
        for c in g:
            v=g[c];num=pd.to_numeric(v,errors='coerce');date=pd.to_datetime(v,errors='coerce') if 'date' in c.lower() else None
            columns.append(dict(dataset=key,field=c,missing=int(v.isna().sum()),missing_rate=float(v.isna().mean()),
                unique_nonmissing=int(v.nunique()),numeric_count=int(num.notna().sum()),numeric_min=num.min(),numeric_max=num.max(),
                date_min=str(date.min().date()) if date is not None and date.notna().any() else '',
                date_max=str(date.max().date()) if date is not None and date.notna().any() else ''))
    csv(out/'file_diagnosis.csv',summary);csv(out/'field_diagnosis.csv',columns)
    a,b=data['C1'],data['C2'];checks=[]
    checks.append(dict(check='row_count',equal=len(a)==len(b),C1_value=len(a),C2_value=len(b)))
    checks.append(dict(check='Model_set',equal=set(a.Model)==set(b.Model),C1_value=a.Model.nunique(),C2_value=b.Model.nunique()))
    for c in a:
        equal=a[c].equals(b[c]);numeric=pd.api.types.is_numeric_dtype(a[c])
        difference=float((a[c]-b[c]).abs().max()) if numeric else np.nan
        numerical_consistency=bool(np.allclose(a[c],b[c],rtol=0,atol=1e-12,equal_nan=True)) if numeric else equal
        checks.append(dict(check=c,equal=equal,numerically_consistent=numerical_consistency,
            maximum_absolute_difference=difference,C1_value=len(a),C2_value=len(b),
            mismatched_rows=int((~(a[c].eq(b[c])|(a[c].isna()&b[c].isna()))).sum())))
    csv(out/'C1_C2_consistency.csv',checks)
    if not all(r.get('numerically_consistent',r['equal']) for r in checks):raise ValueError('C1/C2 substantive core difference: inspect consistency table before computing ability')
    csv(out/'license_value_counts.csv',b['Hub License'].fillna('<missing>').value_counts().rename_axis('Hub License').reset_index(name='n'))
    csv(out/'type_value_counts.csv',b.Type.fillna('<missing>').value_counts().rename_axis('Type').reset_index(name='n'))
    csv(out/'C4_open_weight_values.csv',data['C4']['Open model weights?'].fillna('<missing>').value_counts().rename_axis('Open model weights?').reset_index(name='n'))
    # Parquet is an additional ORIGINAL attachment: architecture/raw-score
    # audit only. Hub availability alone is not asserted to be open weights.
    parquet=base/'data/train-00000-of-00001.parquet'
    raw=pd.read_parquet(parquet);data['raw_parquet']=raw
    text=['# Question 4 data diagnosis',table(pd.DataFrame(summary)),
        'C2与C1核心逐行数值一致（比较容差1e-12，仅用于CSV舍入审计），但若干浮点并非逐位相等，最大差异见一致性表。C2新增'+', '.join(c for c in b if c not in a)+'。这些列标为Epoch AI增强元数据，但没有Epoch模型ID或显式匹配模型名；不能据此静默恢复唯一C4记录。',
        '全部字段及缺失/数值/日期范围见field_diagnosis.csv；重复模型名不等同于重复评测。原始parquet用于架构与原始得分溯源，不读取任何旧问题四拟合结果。',
        '许可证只有标识符，没有逐模型授权条款。遵照本轮要求，不能仅据标识符确认研究/复现授权，统一license_unknown；没有自动做法律解释。',
        'C7 training_data_TB与token数不同，禁止转换为D_tokens。C4的Dataset字段也不是统一token单位，只在notes明确标注token时提取D_tokens。']
    (out/'data_diagnosis.md').write_text('\n\n'.join(text),encoding='utf8')
    return data


def canonical_models(data,root,cfg):
    """One model per name; latest dated cohort, tied scores median, never best."""
    raw=data['C2'].copy();raw['record_row']=np.arange(len(raw));raw['date_parsed']=pd.to_datetime(raw['Submission Date'],errors='coerce')
    rows=[];audit=[];metrics=cfg['benchmarks'];avg=cfg['average_column']
    for name,g in raw.groupby('Model',sort=False,dropna=False):
        latest=g.date_parsed.max();selected=g[g.date_parsed==latest] if pd.notna(latest) else g
        r={'Model':name,'record_count':len(g),'selected_record_count':len(selected),
           'source_rows':json.dumps(selected.record_row.tolist()),'Submission Date':latest.date().isoformat() if pd.notna(latest) else np.nan}
        for c in raw.columns:
            if c in ['Model','record_row','date_parsed','Submission Date']:continue
            if c in metrics+[avg]:
                # Numeric evaluation replicates, never a maximum selection.
                r[c]=pd.to_numeric(selected[c],errors='coerce').median()
            else:
                values=selected[c].dropna().unique();r[c]=values[0] if len(values)==1 else np.nan
                if len(values)>1:audit.append(dict(Model=name,field=c,issue='conflicting_tied_metadata',values=json.dumps(values.tolist()),action='retain missing'))
        for c in metrics:
            if len(g)>1:audit.append(dict(Model=name,field=c,issue='repeated_evaluation',values=json.dumps(g[c].tolist()),
                action='latest dated cohort; componentwise median of ties',selected_rows=r['source_rows']))
        rows.append(r)
    models=pd.DataFrame(rows);models['name_key']=models.Model.map(normalized_name)
    csv(root/'outputs/evolution/phase1/duplicate_model_resolution.csv',audit)
    for i,r in models.iterrows():
        t=r.Type
        if pd.notna(t):
            label=cfg['type_mapping'].get(str(t),'other_unknown');source='official_type'
        elif re.search(r'(?i)(?:^|[-_/])(chat|instruct|sft|rlhf)(?:$|[-_/])',r.Model):label='chat_finetuned';source='name_fallback'
        elif re.search(r'(?i)(?:^|[-_/])base(?:$|[-_/])',r.Model):label='pretrained';source='name_fallback'
        else:label='other_unknown';source='unknown'
        models.loc[i,'model_type']=label;models.loc[i,'classification_source']=source
    csv(root/'outputs/evolution/phase1/model_type_classification.csv',models[['Model','Type','model_type','classification_source']])
    return models


def epoch_join(models,data,root,cfg):
    """Prefer provided metadata; accept only auditable conservative C4 matches."""
    out=root/'outputs/evolution/phase1';epoch=data['C4'].copy();epoch['epoch_row']=np.arange(len(epoch));epoch['name_key']=epoch.Model.map(normalized_name)
    matches=[];proposals=[]
    for r in models.to_dict('records'):
        key=r['name_key'];base=key.split('/')[-1];owner=key.split('/')[0] if '/' in key else ''
        g=epoch[epoch.name_key==key];method='exact_normalized';confidence='exact'
        if len(g)==0:
            g=epoch[epoch.name_key==base]
            if len(g)==1:
                e=g.iloc[0];date_ok=pd.notna(r['Epoch_AI_Publication_Date']) and str(r['Epoch_AI_Publication_Date'])==str(e['Publication date'])
                org_ok=pd.notna(r['Epoch_AI_Organization']) and normalized_name(r['Epoch_AI_Organization'])==normalized_name(e.Organization)
                owner_ok=owner and (normalized_name(e['Hugging Face developer id'])==owner or normalized_name(e.Organization)==owner)
                if (date_ok and org_ok) or owner_ok:method='unique_full_basename_with_metadata_corroboration';confidence='high-confidence'
                else:g=epoch.iloc[:0];method='unconfirmed_basename';confidence='unmatched'
        if len(g)>1:
            candidates=g
            if pd.notna(r['Epoch_AI_Publication_Date']):g=g[g['Publication date'].astype(str)==str(r['Epoch_AI_Publication_Date'])]
            if pd.notna(r['Epoch_AI_Organization']):g=g[g.Organization.map(normalized_name)==normalized_name(r['Epoch_AI_Organization'])]
            if len(g)!=1:
                proposals.append(dict(Model=r['Model'],status='ambiguous',candidates=json.dumps(candidates.Model.tolist()),accepted=False));g=epoch.iloc[:0];confidence='ambiguous'
        row=dict(Model=r['Model'],Epoch_match=np.nan,match_method=method if len(g)==1 else 'unmatched',confidence=confidence if len(g)==1 else 'unmatched',
            leaderboard_N_B=r['#Params (B)'],Epoch_N_B=np.nan,Release_Date=r['Epoch_AI_Publication_Date'],
            release_date_source='C2_Epoch_AI_Publication_Date' if pd.notna(r['Epoch_AI_Publication_Date']) else 'missing',
            training_compute=np.nan,training_dataset_raw=np.nan,D_tokens=np.nan,D_unit_status='unknown',
            open_weights=r['Epoch_AI_Open_Weights'],open_weights_source='C2_Epoch_AI_Open_Weights' if pd.notna(r['Epoch_AI_Open_Weights']) else 'unknown',
            Organization=r['Epoch_AI_Organization'],parameter_relative_error=np.nan,parameter_conflict=False,open_metadata_conflict=False)
        if len(g)==1:
            e=g.iloc[0];en=pd.to_numeric(e.Parameters,errors='coerce')/1e9
            relative=abs(r['#Params (B)']-en)/abs(en) if pd.notna(en) and en>0 and pd.notna(r['#Params (B)']) else np.nan
            ds=pd.to_numeric(e['Training dataset size (total)'],errors='coerce');notes=str(e['Dataset size notes'])
            token_confirmed=bool(re.search(r'(?i)\btokens?\b',notes))
            row.update(Epoch_match=e.Model,epoch_row=int(e.epoch_row),Epoch_N_B=en,
                Release_Date=e['Publication date'],release_date_source='C4_Publication date',Organization=e.Organization,
                training_compute=pd.to_numeric(e['Training compute (FLOP)'],errors='coerce'),
                training_dataset_raw=e['Training dataset size (total)'],Dataset_size_notes=e['Dataset size notes'],
                D_tokens=ds if token_confirmed else np.nan,D_unit_status='tokens_confirmed_in_notes' if token_confirmed else 'unknown',
                parameter_relative_error=relative,parameter_conflict=bool(pd.notna(relative) and relative>cfg['parameter_conflict_relative_threshold']),
                C4_open_weights=e['Open model weights?'],Parameters_notes=e['Parameters notes'],
                C4_Model_accessibility=e['Model accessibility'])
            supplied=r['Epoch_AI_Open_Weights'];observed=e['Open model weights?']
            contradiction=pd.notna(supplied) and pd.notna(observed) and str(supplied).lower()!=str(observed).lower()
            row['open_metadata_conflict']=bool(contradiction)
            if contradiction:row.update(open_weights=np.nan,open_weights_source='conflicting_C2_C4')
            elif pd.notna(observed):row.update(open_weights=observed,open_weights_source='C4_Open model weights?')
        else:
            candidates=epoch[epoch.name_key==base]
            if len(candidates):proposals.append(dict(Model=r['Model'],status='unconfirmed_basename_candidate',candidates=json.dumps(candidates.Model.tolist()),accepted=False))
        matches.append(row)
    joined=pd.DataFrame(matches);csv(out/'leaderboard_epoch_matches.csv',joined)
    csv(out/'epoch_match_candidates.csv',pd.DataFrame(proposals,columns=['Model','status','candidates','accepted']))
    rows=[]
    for field in ['Parameters','Training compute (FLOP)','Training dataset size (total)','Publication date','Open model weights?','Organization']:
        rows.append(dict(scope='all_C4',field=field,n=len(epoch),available=int(epoch[field].notna().sum()),missing_rate=float(epoch[field].isna().mean())))
    numeric=epoch[['Parameters','Training compute (FLOP)','Training dataset size (total)']].apply(pd.to_numeric,errors='coerce')
    rows.append(dict(scope='all_C4',field='N_compute_dataset_numeric_joint',n=len(epoch),available=int((numeric.notna()&(numeric>0)).all(axis=1).sum())))
    csv(out/'C4_scale_metadata_summary.csv',rows)
    models=models.merge(joined.drop(columns='leaderboard_N_B'),on='Model',validate='one_to_one')
    metrics=cfg['benchmarks'];models['benchmark_complete']=models[metrics].notna().all(axis=1)
    models['benchmark_valid']=np.isfinite(models[metrics].to_numpy(float)).all(axis=1)&models[metrics].ge(0).all(axis=1)&models[metrics].le(100).all(axis=1)
    models['S0']=models.benchmark_complete&models.benchmark_valid
    models['S1']=models.S0&models.open_weights.fillna('').astype(str).str.lower().isin(['yes','true'])&~models.open_metadata_conflict
    # No explicit per-model licensing grant text in the supplied C CSVs.
    models['license_status']='license_unknown';models['S2']=False
    models['license']=models['Hub License'];models['N_B']=models['#Params (B)'];models['Release Date']=models.Release_Date
    csv(out/'open_source_filtering.csv',models[['Model','S0','S1','S2','open_weights','open_weights_source','open_metadata_conflict','license','license_status']])
    sub=pd.to_datetime(models['Submission Date'],errors='coerce');rel=pd.to_datetime(models['Release Date'],errors='coerce')
    models['submission_release_days']=(sub-rel).dt.days
    csv(out/'time_axis_audit.csv',models[['Model','Submission Date','Release Date','release_date_source','submission_release_days']])
    return models

# ============================================================
# Source: src/evolution/ability.py
# Original lines: 1-183
# Purpose: 固定六Benchmark Winsor/median/IQR、A_z、PCA/Average敏感性
# ============================================================
"""One fixed six-task ability definition, descriptive chronology, scale table."""
import json
import logging
import math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from .common import csv,dump,corr,normalized_name,table


def transform(frame,parameters):
    """Apply the fixed S1 task transform to any table, without refitting."""
    result=pd.DataFrame(index=frame.index)
    for r in parameters['tasks']:
        x=pd.to_numeric(frame[r['metric']],errors='coerce').clip(r['P01'],r['P99'])
        result[r['metric']]=(x-r['median'])/r['IQR'] if r['IQR']>0 else np.where(x.notna(),0,np.nan)
    return result


def profile(frame,metrics,scope):
    """Real values, error counts, sample standard deviations and quantiles."""
    rows=[]
    for m in metrics:
        x=pd.to_numeric(frame[m],errors='coerce');v=x.dropna();quantiles=v.quantile([.01,.05,.5,.95,.99])
        rows.append(dict(scope=scope,metric=m,n=len(v),missing=int(x.isna().sum()),missing_rate=float(x.isna().mean()),
            mean=v.mean(),std=v.std(ddof=1),min=v.min(),P01=quantiles.loc[.01],P05=quantiles.loc[.05],
            median=quantiles.loc[.5],P95=quantiles.loc[.95],P99=quantiles.loc[.99],max=v.max(),
            below_zero=int((v<0).sum()),above_100=int((v>100).sum()),unique_values=int(v.nunique()),constant=bool(v.nunique()<=1)))
    return rows


def comparisons(models,out,cfg):
    """Model ordering, monthly frontier ordering, and top-decile overlap."""
    rows=[];pairs=[('A_z','A_pca'),('A_z',cfg['average_column']),('A_pca',cfg['average_column'])]
    for scope in ['S0','S1']:
        for a,b in pairs:
            g=models[models[scope]&models[[a,b]].notna().all(axis=1)].sort_values('Model')
            if g.empty:continue
            k=max(1,math.ceil(.1*len(g)));left=set(g.sort_values(a,ascending=False,kind='stable').head(k).Model)
            right=set(g.sort_values(b,ascending=False,kind='stable').head(k).Model)
            rows.append(dict(scope=scope,comparison='model_index',index_1=a,index_2=b,n=len(g),
                Pearson=corr(g[a],g[b]),Spearman=corr(g[a],g[b],'spearman'),top10_overlap=len(left&right)/k,
                top_k=k,tie_policy='model-name lexical tie order'))
            h=g.dropna(subset=['Submission Date']).copy();h['month']=pd.to_datetime(h['Submission Date']).dt.to_period('M').astype(str)
            monthly=h.groupby('month')[[a,b]].quantile(.95)
            rows.append(dict(scope=scope,comparison='monthly_P95_order',index_1=a,index_2=b,n=len(monthly),
                Pearson=corr(monthly[a],monthly[b]),Spearman=corr(monthly[a],monthly[b],'spearman')))
            for month,m in h.groupby('month'):
                k=max(1,math.ceil(.1*len(m)));l=set(m.sort_values(a,ascending=False,kind='stable').head(k).Model);r=set(m.sort_values(b,ascending=False,kind='stable').head(k).Model)
                rows.append(dict(scope=scope,comparison='within_month_top10',month=month,index_1=a,index_2=b,n=len(m),top_k=k,top10_overlap=len(l&r)/k))
    csv(out/'ability_index_comparison.csv',rows)


def chronology_audit(data,out,cfg):
    """C3 has YEAR only: never invent a month or pool historic incompatible scores."""
    c3=data['C3'];c2=data['C2'];mapped={'IFEval':'IFEval','BBH':'BBH','MATH Lvl 5':'MATH_Lvl5','GPQA':'GPQA','MUSR':'MUSR','MMLU-PRO':'MMLU_PRO'}
    groups={k:g for k,g in c2.groupby('Model',sort=False)};rows=[]
    for i,r in c3.iterrows():
        g=groups.get(r.Model);same=False;error=np.nan
        if g is not None:
            x=np.array([r[mapped[c]] for c in cfg['benchmarks']],float);diff=np.abs(g[cfg['benchmarks']].to_numpy()-x)
            errors=diff.max(axis=1);error=float(errors.min());same=bool(error<1e-9)
        rows.append(dict(C3_row=i,Model=r.Model,Year=r.Year,Source=r.Source,matched_C2_model=g is not None,
            exact_six_score_evaluation=same,maximum_task_difference_best_record=error,
            comparable_main_scale=bool(same and r.Source=='Open LLM Leaderboard'),
            time_precision='year_only',full_date_available=False))
    csv(out/'C3_time_series_audit.csv',rows)
    csv(out/'C3_year_source_summary.csv',c3.groupby(['Year','Source']).size().rename('records').reset_index())


def construct_ability(root,cfg,models,data):
    """Fit only the new six-task transform on S1; main A is equal-weighted."""
    out=root/'outputs/evolution/phase1';metrics=cfg['benchmarks'];main=models[models[cfg['main_sample']]]
    if len(main)<7:raise ValueError('Insufficient independently confirmed open-weight sample to define six-task scaling')
    parameters={'fit_scope':'S1 unique models, six tasks complete and valid; whole observed analysis period',
        'fit_model_count':len(main),'metrics':metrics,'weights':[1/6]*6,'tasks':[],'monthly_refit':False}
    for m in metrics:
        x=main[m];L,U=x.quantile(cfg['winsor_quantiles']).to_numpy();w=x.clip(L,U)
        q25,med,q75=w.quantile([.25,.5,.75]);iqr=float(q75-q25)
        parameters['tasks'].append(dict(metric=m,P01=float(L),P99=float(U),median=float(med),IQR=iqr,
            constant_transform=iqr==0,P25=float(q25),P75=float(q75)))
    Z=transform(models,parameters);models['A_z']=Z.mean(axis=1,skipna=False);models['A_equal']=models.A_z
    models.loc[~models.S0,['A_z','A_equal']]=np.nan
    pca=PCA(n_components=6,svd_solver='full').fit(Z.loc[models.S1].to_numpy())
    loading=pca.components_[0].copy()
    if loading.sum()<0:loading=-loading
    accepted=bool((loading>0).all() and pca.explained_variance_ratio_[0]>=cfg['pca_min_explained_variance'])
    parameters['pca']=dict(accepted=accepted,loadings=loading.tolist(),mean=pca.mean_.tolist(),
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),acceptance_min_variance=cfg['pca_min_explained_variance'],
        main_index=False)
    models['A_pca']=(Z-pca.mean_)@loading if accepted else np.nan
    lo,hi=models.loc[models.S0,'A_z'].quantile(cfg['display_quantiles']);parameters['display_mapping']=dict(scope='all S0 models, fixed across months',P01=float(lo),P99=float(hi),display_only=True)
    models['A_100']=100*((models.A_z-lo)/(hi-lo)).clip(0,1) if hi>lo else 50.
    for m in metrics:models['z_'+m]=Z[m]
    dump(out/'ability_transform_parameters.json',parameters);csv(out/'ability_transform_parameters.csv',parameters['tasks'])
    csv(out/'PCA_loadings_sensitivity.csv',[dict(metric=m,loading=float(v),accepted=accepted,PC1_variance_ratio=float(pca.explained_variance_ratio_[0])) for m,v in zip(metrics,loading)])
    prof=profile(data['C2'],metrics,'C2_all_evaluations')+profile(models[models.S0],metrics,'S0_unique_models')+profile(main,metrics,'S1_unique_models')
    csv(out/'benchmark_profile.csv',prof)
    correlations=[]
    for scope in ['S0','S1']:
        for method in ['pearson','spearman']:
            matrix=models.loc[models[scope],metrics].corr(method=method)
            for a in metrics:
                for b in metrics:correlations.append(dict(scope=scope,method=method,metric_1=a,metric_2=b,correlation=matrix.loc[a,b]))
    csv(out/'benchmark_correlations.csv',correlations);comparisons(models,out,cfg)
    monthly=[]
    for scope in ['S0','S1','S2']:
        g=models[models[scope]&models['Submission Date'].notna()].copy();g['month']=pd.to_datetime(g['Submission Date']).dt.to_period('M').astype(str)
        for label in ['all','pretrained','chat_finetuned','other_unknown']:
            h=g if label=='all' else g[g.model_type==label]
            for month,v in h.groupby('month'):
                x=v.A_z;quant=x.quantile([.5,.75,.9,.95]);k=min(5,len(v))
                monthly.append(dict(sample=scope,model_type=label,month=month,n=len(v),median_A=quant.loc[.5],P75_A=quant.loc[.75],
                    P90_A=quant.loc[.9],P95_A=quant.loc[.95],max_A=x.max(),top5_mean_A=x.nlargest(k).mean(),actual_top_k=k))
    csv(out/'monthly_ability_summary.csv',monthly);chronology_audit(data,out,cfg)
    # Actual architecture attachment and original parquet, no inferred context.
    c7=data['C7'].set_index(data['C7'].model_name.map(normalized_name));raw=data['raw_parquet']
    arch=[]
    for r in models.to_dict('records'):
        rows=json.loads(r['source_rows']);p=raw.iloc[rows]
        entry=dict(Model=r['Model'])
        for field in ['Architecture','MoE','Weight type','Available on the hub','Base Model']:
            values=p[field].dropna().astype(str).unique();entry[field]=values[0] if len(values)==1 else np.nan
        entry['architecture_source']='C_original_parquet, canonical selected evaluation rows'
        if r['name_key'] in c7.index:
            v=c7.loc[r['name_key']]
            if isinstance(v,pd.DataFrame):raise ValueError('Ambiguous C7 exact normalized name')
            entry.update({field:v[field] for field in ['n_layers','n_heads','d_model','vocab_size','max_position_embeddings','training_data_TB']})
            entry['context_length']=v.max_position_embeddings;entry['context_source']='C7_exact_normalized_match'
        arch.append(entry)
    models=models.merge(pd.DataFrame(arch),on='Model',validate='one_to_one')
    bridge=data['C6'].copy();bridge['name_key']=bridge.Model.map(normalized_name)
    if bridge.name_key.duplicated().any():raise ValueError('C6 duplicated model identity; do not silently choose a row')
    aux=bridge[['name_key','N_params_B','D_tokens_B','Loss_Comparability']].rename(columns={'N_params_B':'C6_N_B','D_tokens_B':'C6_D_tokens_B'})
    collisions=models.name_key.duplicated(keep=False)
    csv(out/'normalized_model_name_collisions.csv',models.loc[collisions,['Model','name_key']])
    models['bridge_match_key']=models.name_key.mask(collisions)
    aux=aux.rename(columns={'name_key':'bridge_match_key'})
    models=models.merge(aux,on='bridge_match_key',how='left',validate='many_to_one')
    models['D_tokens_source']=np.where(models.D_tokens.notna(),'C4 Dataset size notes confirm tokens','missing')
    observed_aux=models.D_tokens.isna()&models.C6_D_tokens_B.notna()
    models.loc[observed_aux,'D_tokens']=models.loc[observed_aux,'C6_D_tokens_B']*1e9
    models.loc[observed_aux,'D_tokens_source']='C6 explicit D_tokens_B (source/comparability retained)'
    models['N_source']='C2 #Params (B), unique consistent value among selected evaluation replicas; conflicts missing'
    models['compute_source']=np.where(models.training_compute.notna(),'matched C4 Training compute (FLOP)','missing')
    models['ability_source']='independent six-task S1 robust transform; equal weight'
    models['license_source']='C2 Hub License identifier; grant not independently confirmed'
    models['scale_joint_available']=models[['N_B','D_tokens','training_compute']].notna().all(axis=1)&models[['N_B','D_tokens','training_compute']].gt(0).all(axis=1)
    models['contribution_sample_candidate']=models.S1&models.scale_joint_available&models['Submission Date'].notna()&~models.parameter_conflict
    models['contribution_sample_pretrained']=models.contribution_sample_candidate&(models.model_type=='pretrained')
    models['C8_capability_imbalance']=np.nan;models['C8_bottom20_task_score']=np.nan
    # Restore independently computed C8 diagnostics if only the metadata
    # checkpoint is being rerun; never reparse C8 merely to regenerate a join.
    cached=out/'C8_matched_model_analysis.csv'
    if cached.exists():
        old=pd.read_csv(cached,float_precision='round_trip');old=old[old.unique_model_match]
        attach=old[['Model','capability_imbalance','bottom20_task_score','difficulty_adjusted_imbalance','difficulty_adjusted_bottom20']].rename(
            columns={'capability_imbalance':'C8_capability_imbalance','bottom20_task_score':'C8_bottom20_task_score'})
        models=models.drop(columns=['C8_capability_imbalance','C8_bottom20_task_score']).merge(attach,on='Model',how='left',validate='one_to_one')
        models['C8_source']=np.where(models.C8_capability_imbalance.notna(),'latest parsable C8 JSON; exact normalized unique identity','unmatched/ambiguous')
    csv(out/'evolution_model_table.csv',models)
    time=models.submission_release_days.dropna()
    summary=dict(raw_C1_C2_records=len(data['C2']),unique_models=len(models),duplicate_extra_records=len(data['C2'])-len(models),
        S0=int(models.S0.sum()),S1=int(models.S1.sum()),S2=int(models.S2.sum()),
        benchmark_complete_records=int(data['C2'][metrics].notna().all(axis=1).sum()),
        model_type_counts=models.model_type.value_counts().to_dict(),S1_type_counts=models[models.S1].model_type.value_counts().to_dict(),
        C4_matched=int(models.Epoch_match.notna().sum()),C4_parameter_conflicts=int(models.parameter_conflict.sum()),
        time_lag_n=len(time),time_lag_median=float(time.median()),time_lag_P10=float(time.quantile(.1)),time_lag_P90=float(time.quantile(.9)),
        negative_submission_release_lag=int((time<0).sum()),joint_N_D_compute=int(models.scale_joint_available.sum()),
        contribution_candidates=int(models.contribution_sample_candidate.sum()),contribution_pretrained=int(models.contribution_sample_pretrained.sum()),
        pca_accepted=accepted,PC1_explained_variance=float(pca.explained_variance_ratio_[0]),
        C3_year_min=int(data['C3'].Year.min()),C3_year_max=int(data['C3'].Year.max()),C3_has_full_dates=False,
        historical_C3_not_pooled=True,no_scale_interpolation=True)
    dump(out/'ability_execution_summary.json',summary)
    text=['# Ability sample definition',f"原始评测{len(data['C2'])}条；每模型一行{len(models)}个。主标定S1={summary['S1']}；S0={summary['S0']}，S2={summary['S2']}。",
        '重复模型按最新Submission Date选评测集合；同日多条逐任务取中位数，不选最高分，不修改官方CSV。模型类型元数据冲突保留unknown。',
        '六维完整且0–100有效是S0。S1要求C2明确Epoch开权重Yes或保守C4匹配Yes，出现C2/C4冲突则不纳入。原始parquet的Available on hub不等同于研究许可，也不单独作为open weights证据。',
        'S2需要明确授权条款；附件只有许可证标识，因此license_unknown，当前严格敏感性样本为空，不将其伪造成已确认开放许可。',
        'A_z=A_equal=sum((clip(x,P01,P99)-median)/IQR)/6；全部参数一次由S1计算，跨月固定。退化IQR为0的任务置中性0并记录；缺任务不填补，不与完整指数混合。A_100只作展示，基于所有S0的固定P01/P99映射。',
        '本阶段用完整已观测时期定义固定口径；后续如做时间外验证，应只在相应训练期拟合新的分析版本，不能将此全期标定冒称无前视验证。',
        'N取排行榜实际评测参数量，C4 N另列；超过25%相对差异标记conflict，不覆盖。D优先C4明确token证据，否则可用C6明确D_tokens_B并保留来源；不推导compute=6ND，不把C7 TB换成token。']
    (out/'ability_sample_definition.md').write_text('\n\n'.join(text),encoding='utf8')
    logging.info('Ability complete: %s',summary)

# ============================================================
# Source: src/evolution/detailed.py
# Original lines: 1-209
# Purpose: C8 JSON全遍历、损坏文件记录与偏科/弱尾
# ============================================================
"""Exhaustive C8 parsing and true subtask-level, version-aware diagnostics."""
import json
import logging
import re
from difflib import get_close_matches
import numpy as np
import pandas as pd
from .common import csv,dump,corr,normalized_name


def value(r,metric):
    """Nonfinite/out-of-range raw evaluation scores stay missing."""
    try:v=float(r.get(metric,np.nan))
    except (ValueError,TypeError):return np.nan
    return v if np.isfinite(v) and 0<=v<=1 else np.nan


def normalize_score(raw,baseline=0):
    """Official chance correction and below-chance floor, expressed in percent."""
    return 100*max(0,(raw-baseline)/(1-baseline)) if np.isfinite(raw) else np.nan


def extract(r,directory,filename):
    """Parse leaf task scores, retaining metrics, versions and baseline evidence."""
    results=r.get('results',{});configs=r.get('configs',{});versions=r.get('versions',{});leaves=[];six={}
    model=r.get('model_name')
    if not model:
        arg=r.get('config',{}).get('model_args','');match=re.search(r'(?:^|,)pretrained=([^,]+)',arg)
        model=match.group(1) if match else directory
    def add(family,task,metric,baseline,proof):
        raw=value(results.get(task,{}),metric)
        if np.isfinite(raw):leaves.append(dict(directory=directory,Model=model,source_file=filename,benchmark=family,
            subtask=task+'::'+metric,task=task,metric=metric,raw_score=raw,random_baseline=baseline,
            task_score=normalize_score(raw,baseline),baseline_evidence=proof,task_version=str(versions.get(task,'unknown'))))
    # IFEval has two strict components; loose variants/stderr are not extra tasks.
    for metric in ['prompt_level_strict_acc,none','inst_level_strict_acc,none']:
        add('IFEval','leaderboard_ifeval',metric,0,'official strict generative metric')
    for task in results:
        if task.startswith('leaderboard_bbh_'):
            choices=configs.get(task,{}).get('doc_to_choice')
            if isinstance(choices,list) and len(choices)>1:add('BBH',task,'acc_norm,none',1/len(choices),'C8 config doc_to_choice list')
        elif task.startswith('leaderboard_math_') and task!='leaderboard_math_hard':
            add('MATH Lvl 5',task,'exact_match,none',0,'generative exact match; version preserved')
        elif task.startswith('leaderboard_gpqa_'):
            add('GPQA',task,'acc_norm,none',.25,'official four-choice GPQA; C8 configurations')
        elif task.startswith('leaderboard_musr_'):
            choices={'leaderboard_musr_murder_mysteries':2,'leaderboard_musr_object_placements':5,'leaderboard_musr_team_allocation':3}
            if task in choices:add('MUSR',task,'acc_norm,none',1/choices[task],'official normalization task choice counts')
    add('MMLU-PRO','leaderboard_mmlu_pro','acc,none',.1,'official MMLU-PRO chance baseline')
    for family in ['IFEval','BBH','MUSR']:
        vals=[v['task_score'] for v in leaves if v['benchmark']==family]
        six[family]=float(np.mean(vals)) if vals else np.nan
    # Group GPQA and MATH can be sample-weighted; keep the group estimate,
    # and save equal subtask means separately rather than conflating them.
    six['GPQA']=normalize_score(value(results.get('leaderboard_gpqa',{}),'acc_norm,none'),.25)
    six['MATH Lvl 5']=normalize_score(value(results.get('leaderboard_math_hard',{}),'exact_match,none'))
    six['MMLU-PRO']=normalize_score(value(results.get('leaderboard_mmlu_pro',{}),'acc,none'),.1)
    return model,six,leaves


def analyze_details(root,cfg):
    """Read every JSON; choose latest parsable file per directory, never sample."""
    out=root/'outputs/evolution/phase1';base=root/'real_attachments/C_efficiency_evolution/detailed_results'
    audits=[];aggregations=[];leaves=[];no_usable=[]
    directories=sorted(p for p in base.iterdir() if p.is_dir())
    for number,d in enumerate(directories,1):
        usable=[]
        for p in sorted(d.rglob('*.json')):
            row=dict(directory=d.name,source_file=p.relative_to(root).as_posix(),parse_success=False,selected=False,parse_error='')
            try:
                r=json.loads(p.read_text(encoding='utf-8-sig'))
                row['parse_success']=True
                if not isinstance(r,dict) or not isinstance(r.get('results'),dict):raise ValueError('JSON parsed but lacks evaluation results mapping')
                match=re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?)',p.name)
                timestamp=pd.to_datetime(match.group(1)[:10]+'T'+match.group(1)[11:].replace('-',':'),utc=True,errors='coerce') if match else pd.NaT
                source='filename_run_timestamp'
                if pd.isna(timestamp):
                    rawdate=r.get('date');timestamp=pd.to_datetime(rawdate,unit='s',utc=True,errors='coerce') if isinstance(rawdate,(int,float)) else pd.to_datetime(rawdate,utc=True,errors='coerce')
                    source='JSON_date'
                row.update(run_time=str(timestamp),time_source=source,model_name=r.get('model_name'),
                    task_version_signature=json.dumps(r.get('versions',{}),sort_keys=True))
                model,six,sub=extract(r,d.name,row['source_file']);row['six_complete']=all(np.isfinite(list(six.values())))
                usable.append((timestamp.value if pd.notna(timestamp) else -1,p.name,len(audits),model,six,sub))
            except (OSError,UnicodeError,json.JSONDecodeError,ValueError,TypeError) as exc:row['parse_error']=str(exc)[:500]
            audits.append(row)
        if not usable:no_usable.append(d.name);continue
        chosen=max(usable,key=lambda v:(v[0],v[1]));_,_,i,model,six,sub=chosen;audits[i]['selected']=True
        leaves.extend(sub)
        vals=np.array([x['task_score'] for x in sub],float)
        agg=dict(directory=d.name,Model=model,source_file=audits[i]['source_file'],run_time=audits[i]['run_time'],
            six_complete=audits[i]['six_complete'],subtask_count=len(vals),
            mean_task_score=float(np.mean(list(six.values()))) if audits[i]['six_complete'] else np.nan,
            subtask_mean=vals.mean() if len(vals) else np.nan,subtask_std=vals.std(ddof=1) if len(vals)>1 else np.nan,
            weakest_subtask_P10=np.quantile(vals,.1) if len(vals) else np.nan,
            strongest_weakest_span=np.ptp(vals) if len(vals) else np.nan,
            capability_imbalance=float(np.quantile(vals,.9)-np.quantile(vals,.1)) if len(vals) else np.nan,
            bottom20_task_score=np.sort(vals)[:max(1,int(np.ceil(.2*len(vals))))].mean() if len(vals) else np.nan,
            MMLU_category_detail_available=False)
        for family,v in six.items():
            agg[family]=v;familyvals=np.array([x['task_score'] for x in sub if x['benchmark']==family])
            agg[family+'_subtask_mean']=familyvals.mean() if len(familyvals) else np.nan
            agg[family+'_subtask_count']=len(familyvals)
        aggregations.append(agg)
        if number%300==0:logging.info('C8 traversed %d/%d directories',number,len(directories))
    audit=pd.DataFrame(audits);scores=pd.DataFrame(leaves);agg=pd.DataFrame(aggregations)
    # Auxiliary within-subtask robust scaling controls task difficulty;
    # it does NOT redefine the six-task ability index or its parameters.
    scores['subtask_z']=np.nan;task_parameters=[]
    for (task,version),g in scores.groupby(['subtask','task_version']):
        x=g.task_score;lo,hi=x.quantile([.01,.99]);v=x.clip(lo,hi);med=v.median();iqr=v.quantile(.75)-v.quantile(.25)
        scores.loc[g.index,'subtask_z']=(v-med)/iqr if iqr>0 else 0
        task_parameters.append(dict(subtask=task,version=version,n=len(g),P01=lo,P99=hi,median=med,IQR=iqr))
    for directory,g in scores.groupby('directory'):
        z=g.subtask_z.to_numpy();index=agg.index[agg.directory==directory]
        agg.loc[index,'difficulty_adjusted_imbalance']=np.quantile(z,.9)-np.quantile(z,.1)
        agg.loc[index,'difficulty_adjusted_bottom20']=np.sort(z)[:max(1,int(np.ceil(.2*len(z))))].mean()
    csv(out/'C8_parse_audit.csv',audit);csv(out/'C8_subtask_scores.csv',scores);csv(out/'C8_subtask_transform_parameters.csv',task_parameters)
    models=pd.read_csv(out/'evolution_model_table.csv',float_precision='round_trip');keys={k:g for k,g in models.groupby('name_key')}
    names=list(keys);matches=[];matched=[]
    for r in agg.to_dict('records'):
        key=normalized_name(r['Model']);g=keys.get(key);status='unmatched';accepted=False;candidate=[]
        if g is not None:
            if len(g)==1:status='exact';accepted=True;model=g.iloc[0]
            else:status='ambiguous';candidate=g.Model.tolist()
        else:
            close=get_close_matches(key,names,n=3,cutoff=.9)
            candidate=close;status='high-confidence' if len(close)==1 else 'ambiguous' if len(close)>1 else 'unmatched'
        matches.append(dict(directory=r['directory'],C8_Model=r['Model'],status=status,accepted=accepted,
            match_method='exact normalized model_name/config identity' if accepted else 'fuzzy candidate only; never auto accepted',
            candidates=json.dumps(candidate),Model=model.Model if accepted else np.nan))
        if accepted:
            item=dict(directory=r['directory'],Model=model.Model,A_z=model.A_z,S1=model.S1,
                capability_imbalance=r['capability_imbalance'],bottom20_task_score=r['bottom20_task_score'],
                difficulty_adjusted_imbalance=r.get('difficulty_adjusted_imbalance',np.nan),
                difficulty_adjusted_bottom20=r.get('difficulty_adjusted_bottom20',np.nan),six_complete=r['six_complete'])
            for family in cfg['benchmarks']:item['C8_'+family]=r[family];item['C2_'+family]=model[family]
            matched.append(item)
    matches=pd.DataFrame(matches);matched=pd.DataFrame(matched)
    # A model appearing in several directories is not silently picked.
    duplicates=matched.Model.duplicated(keep=False)
    matched['unique_model_match']=~duplicates
    valid=matched[matched.unique_model_match&matched.six_complete&matched.A_z.notna()].copy()
    analyses=[]
    for scope in ['S0','S1']:
        h=valid if scope=='S0' else valid[valid.S1]
        if h.empty:continue
        aq=h.A_z.quantile(.75);tq=h.bottom20_task_score.quantile(.25);adjusted=h.difficulty_adjusted_bottom20.quantile(.25)
        flag=(h.A_z>=aq)&(h.bottom20_task_score<=tq);aflag=(h.A_z>=aq)&(h.difficulty_adjusted_bottom20<=adjusted)
        analyses.append(dict(sample=scope,n=len(h),A_P75=aq,bottom20_P25=tq,high_average_weak_tail_count=int(flag.sum()),
            high_average_weak_tail_rate=float(flag.mean()),rate_within_high_ability=float(flag.sum()/(h.A_z>=aq).sum()),
            difficulty_adjusted_weak_tail_count=int(aflag.sum()),difficulty_adjusted_weak_tail_rate=float(aflag.mean()),
            bottom20_zero_rate=float(h.bottom20_task_score.eq(0).mean()),
            Pearson_A_imbalance=corr(h.A_z,h.capability_imbalance),Spearman_A_imbalance=corr(h.A_z,h.capability_imbalance,'spearman'),
            Pearson_A_adjusted_imbalance=corr(h.A_z,h.difficulty_adjusted_imbalance)))
        matched.loc[h.index,'high_average_weak_tail_'+scope]=flag
    comparison=[]
    for family in cfg['benchmarks']:
        h=matched[matched.unique_model_match].dropna(subset=['C8_'+family,'C2_'+family]);diff=h['C8_'+family]-h['C2_'+family]
        comparison.append(dict(benchmark=family,n=len(h),MAE=diff.abs().mean(),systematic_bias=diff.mean(),
            Pearson=corr(h['C8_'+family],h['C2_'+family]),Spearman=corr(h['C8_'+family],h['C2_'+family],'spearman'),
            exact_within_1e_8=int((diff.abs()<1e-8).sum()),different_time_version_not_auto_corrected=True))
    csv(out/'C8_model_task_aggregation.csv',agg);csv(out/'C8_leaderboard_matches.csv',matches)
    csv(out/'C8_matched_model_analysis.csv',matched);csv(out/'C8_six_task_comparison.csv',comparison);csv(out/'C8_weak_tail_diagnosis.csv',analyses)
    attach=matched[matched.unique_model_match][['Model','capability_imbalance','bottom20_task_score','difficulty_adjusted_imbalance','difficulty_adjusted_bottom20']].rename(
        columns={'capability_imbalance':'C8_capability_imbalance','bottom20_task_score':'C8_bottom20_task_score'})
    models=models.drop(columns=['C8_capability_imbalance','C8_bottom20_task_score','difficulty_adjusted_imbalance','difficulty_adjusted_bottom20'],errors='ignore').merge(attach,on='Model',how='left',validate='one_to_one')
    models['C8_source']=np.where(models.C8_capability_imbalance.notna(),'latest parsable C8 JSON; exact normalized unique identity','unmatched/ambiguous')
    csv(out/'evolution_model_table.csv',models)
    summary=dict(directories=len(directories),JSON_files=len(audit),JSON_parse_success=int(audit.parse_success.sum()),
        JSON_parse_failures=int((~audit.parse_success).sum()),directories_no_usable_JSON=len(no_usable),
        no_usable_directories=no_usable,selected_models=len(agg),six_complete_models=int(agg.six_complete.sum()),
        exact_matches=int(matches.accepted.sum()),match_status_counts=matches.status.value_counts().to_dict(),
        valid_six_complete_matched_models=len(valid),leaf_score_records=len(scores),subtask_keys=int(scores.subtask.nunique()),
        fuzzy_matches_automatically_accepted=0,MMLU_category_detail_available=False,
        raw_vs_normalized_scores_separated=True,normalization_source=cfg['C8_normalization_reference'])
    dump(out/'C8_execution_summary.json',summary);logging.info('C8 complete: %s',summary)


def refresh_weak_tail(root,cfg):
    """Compare same-run C8 ability too, avoiding a snapshot-version confound.

    Reuses ONLY the new Phase 1 parsed task table, never old Q4 outputs.
    No need to reparse all JSONs to evaluate the fixed six-task transform.
    """
    from .ability import transform
    out=root/'outputs/evolution/phase1';matched=pd.read_csv(out/'C8_matched_model_analysis.csv',float_precision='round_trip')
    params=json.loads((out/'ability_transform_parameters.json').read_text(encoding='utf8'))
    tasks=pd.DataFrame({m:matched['C8_'+m] for m in cfg['benchmarks']})
    matched['A_z_C8']=transform(tasks,params).mean(axis=1,skipna=False)
    valid=matched[matched.unique_model_match&matched.six_complete&matched.A_z.notna()].copy();analyses=[]
    for scope in ['S0','S1']:
        for basis in ['A_z','A_z_C8']:
            h=valid if scope=='S0' else valid[valid.S1]
            if h.empty:continue
            aq=h[basis].quantile(.75);tq=h.bottom20_task_score.quantile(.25);adjusted=h.difficulty_adjusted_bottom20.quantile(.25)
            flag=(h[basis]>=aq)&(h.bottom20_task_score<=tq);aflag=(h[basis]>=aq)&(h.difficulty_adjusted_bottom20<=adjusted)
            analyses.append(dict(sample=scope,ability_basis='C2_snapshot' if basis=='A_z' else 'C8_same_run',n=len(h),A_P75=aq,bottom20_P25=tq,
                high_average_weak_tail_count=int(flag.sum()),high_average_weak_tail_rate=float(flag.mean()),
                rate_within_high_ability=float(flag.sum()/(h[basis]>=aq).sum()),
                difficulty_adjusted_weak_tail_count=int(aflag.sum()),difficulty_adjusted_weak_tail_rate=float(aflag.mean()),
                bottom20_zero_rate=float(h.bottom20_task_score.eq(0).mean()),Pearson_A_imbalance=corr(h[basis],h.capability_imbalance),
                Spearman_A_imbalance=corr(h[basis],h.capability_imbalance,'spearman'),
                Pearson_A_adjusted_imbalance=corr(h[basis],h.difficulty_adjusted_imbalance)))
            matched.loc[h.index,'high_average_weak_tail_'+scope+'_'+basis]=flag
    csv(out/'C8_weak_tail_diagnosis.csv',analyses);csv(out/'C8_matched_model_analysis.csv',matched)
    model=pd.read_csv(out/'evolution_model_table.csv',float_precision='round_trip')
    model=model.drop(columns='C8_A_z_same_run',errors='ignore').merge(matched[matched.unique_model_match][['Model','A_z_C8']].rename(
        columns={'A_z_C8':'C8_A_z_same_run'}),on='Model',how='left',validate='one_to_one')
    csv(out/'evolution_model_table.csv',model)

# ============================================================
# Source: src/evolution/bridge.py
# Original lines: 1-152
# Purpose: B0/B1/B2、LOMO/bootstrap和桥接失败诊断
# B0/B1/B2 are failed bridge candidates. Preserve bridge_valid_for_conversion=False; do not convert Loss to Benchmark.
# ============================================================
"""High-only small-sample bridge, LOMO validation and bootstrap uncertainty."""
import json
import logging
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from .ability import transform
from .common import csv,dump,corr


def fit_candidate(kind,x,y):
    """Three simple response models; no medium rows or previous fitted bridge."""
    x=np.asarray(x,float);y=np.asarray(y,float)
    if kind=='B2':
        model=IsotonicRegression(increasing=False,out_of_bounds='clip').fit(x,y)
        return dict(kind=kind,x=model.X_thresholds_.tolist(),y=model.y_thresholds_.tolist())
    degree=1 if kind=='B0' else 2
    coef=np.linalg.lstsq(np.column_stack([x**k for k in range(degree+1)]),y,rcond=None)[0]
    return dict(kind=kind,coefficients=coef.tolist())


def predict(candidate,x):
    """Evaluate persisted coefficients/thresholds without a refit."""
    x=np.asarray(x,float)
    if candidate['kind']=='B2':return np.interp(x,candidate['x'],candidate['y'])
    return sum(v*x**k for k,v in enumerate(candidate['coefficients']))


def metrics(observed,predicted):
    """Undefined R²/correlations remain NaN for a constant response/prediction."""
    observed=np.asarray(observed,float);predicted=np.asarray(predicted,float);error=predicted-observed
    total=np.sum((observed-observed.mean())**2)
    return dict(n=len(observed),RMSE=float(np.sqrt(np.mean(error**2))),MAE=float(np.mean(abs(error))),
        R2=float(1-error@error/total) if total>0 else np.nan,
        Pearson=corr(observed,predicted),Spearman=corr(observed,predicted,'spearman'),bias=float(error.mean()))


def decreasing(candidate,lo,hi):
    """Diagnose fitted direction; never change an empirical slope's sign."""
    if candidate['kind']=='B2':return bool(np.all(np.diff(candidate['y'])<=1e-12))
    c=candidate['coefficients'];derivative=[c[1]+(2*c[2]*x if len(c)==3 else 0) for x in [lo,hi]]
    return bool(max(derivative)<0)


def predict_bridge(loss,model_path):
    """Reusable calibrated-range predictor; extrapolation is explicitly flagged.

    Bootstrap bands are saved separately; do not apply this bridge to future
    resource/benchmark forecasts without carrying that uncertainty.
    """
    metadata=json.loads(model_path.read_text(encoding='utf8'));x=np.atleast_1d(loss).astype(float)
    lo,hi=metadata['calibrated_loss_range']
    return pd.DataFrame(dict(Val_Loss=x,A_pred=predict(metadata['selected_model'],x),
        outside_high_calibration_range=(x<lo)|(x>hi),
        validated_for_loss_to_ability_conversion=metadata['bridge_valid_for_conversion'],
        evidence=metadata['bridge_status']))


def analyze_bridge(root,cfg):
    """High-only fitting; medium external validation; 2000 paired bootstraps."""
    out=root/'outputs/evolution/phase1';base=root/'real_attachments/C_efficiency_evolution'
    data=pd.read_csv(base/'loss_benchmark_bridge_expanded.csv',float_precision='round_trip')
    params=json.loads((out/'ability_transform_parameters.json').read_text(encoding='utf8'))
    mapping={'IFEval':'LB_IFEval','BBH':'LB_BBH','MATH Lvl 5':'LB_MATH','GPQA':'LB_GPQA','MUSR':'LB_MUSR','MMLU-PRO':'LB_MMLU_PRO'}
    tasks=pd.DataFrame({m:pd.to_numeric(data[field],errors='coerce') for m,field in mapping.items()})
    valid=tasks.notna().all(axis=1)&tasks.ge(0).all(axis=1)&tasks.le(100).all(axis=1)
    data['A_bridge']=transform(tasks,params).mean(axis=1,skipna=False);data.loc[~valid,'A_bridge']=np.nan
    data['comparability']=data.Loss_Comparability.str.extract(r'^(High|Medium)',expand=False).fillna('other')
    data['model_family_diagnostic']=np.where(data.Model.str.contains('pythia',case=False),'Pythia',data.Model.str.split('/').str[0])
    csv(out/'C6_bridge_transformed_records.csv',data)
    c5=pd.read_csv(base/'loss_benchmark_bridge.csv',float_precision='round_trip');comparison=[]
    for r in c5.to_dict('records'):
        matches=data[data.Model==r['Model']]
        for field in c5.columns:
            if field=='Model':continue
            same=False;gap=np.nan
            if len(matches)==1:
                v=matches[field].iloc[0]
                if isinstance(r[field],(int,float)) and isinstance(v,(int,float)):
                    gap=abs(r[field]-v);same=bool(np.isclose(r[field],v,rtol=0,atol=1e-12,equal_nan=True))
                else:same=(pd.isna(r[field]) and pd.isna(v)) or str(r[field])==str(v)
            comparison.append(dict(Model=r['Model'],field=field,C6_matches=len(matches),consistent=same,absolute_difference=gap))
    csv(out/'C5_C6_consistency.csv',comparison)
    valid_rows=data.A_bridge.notna()&pd.to_numeric(data.Val_Loss,errors='coerce').gt(0)
    high=data[(data.comparability=='High')&valid_rows].copy();medium=data[(data.comparability=='Medium')&valid_rows].copy()
    if len(high)<4 or high.Model.duplicated().any():raise ValueError('Insufficient/ambiguous independent High models; bridge cannot proceed')
    x=high.Val_Loss.to_numpy();y=high.A_bridge.to_numpy();lo,hi=float(x.min()),float(x.max());grid=np.linspace(lo,hi,cfg['bridge_grid_points'])
    # No family column exists in C6. Inferred High family is solely Pythia,
    # hence true family-holdout validation is impossible, not substituted.
    kinds=['B0','B1','B2'];fitted={};cv={};records=[];scores=[];folds=[]
    for kind in kinds:
        full=fit_candidate(kind,x,y);fitted[kind]=full;oof=np.empty(len(x));directions=[]
        for i in range(len(x)):
            mask=np.arange(len(x))!=i;fold=fit_candidate(kind,x[mask],y[mask]);oof[i]=float(predict(fold,[x[i]])[0])
            direction=decreasing(fold,float(x[mask].min()),float(x[mask].max()));directions.append(direction)
            folds.append(dict(candidate=kind,held_out_Model=high.Model.iloc[i],training_n=int(mask.sum()),
                decreasing_on_fold_training_range=direction,held_out_outside_training_range=bool(x[i]<x[mask].min() or x[i]>x[mask].max()),
                formula=json.dumps(fold),observed_A=float(y[i]),predicted_A=float(oof[i])))
        cv[kind]=oof;valid_direction=decreasing(full,lo,hi)
        scores.append(dict(candidate=kind,evaluation='High_LOMO_CV',**metrics(y,oof),
            decreasing_full=valid_direction,decreasing_folds=sum(directions),folds=len(x),
            CV_method='Leave-One-Model-Out; only one High family; no out-of-family test'))
        external=predict(full,medium.Val_Loss)
        scores.append(dict(candidate=kind,evaluation='Medium_external',**metrics(medium.A_bridge,external),
            outside_High_loss_range=int(((medium.Val_Loss<lo)|(medium.Val_Loss>hi)).sum()),decreasing_full=valid_direction))
        for i,r in enumerate(high.to_dict('records')):
            records.append(dict(candidate=kind,evaluation='High_LOMO_CV',Model=r['Model'],Val_Loss=r['Val_Loss'],
                observed_A=r['A_bridge'],predicted_A=float(oof[i]),comparability='High'))
        for i,r in enumerate(medium.to_dict('records')):
            records.append(dict(candidate=kind,evaluation='Medium_external',Model=r['Model'],Val_Loss=r['Val_Loss'],
                observed_A=r['A_bridge'],predicted_A=float(external[i]),comparability='Medium',
                outside_High_loss_range=bool(r['Val_Loss']<lo or r['Val_Loss']>hi)))
    scores=pd.DataFrame(scores);eligible=scores[(scores.evaluation=='High_LOMO_CV')&scores.decreasing_full].copy()
    selected=str(eligible.sort_values(['RMSE','candidate']).candidate.iloc[0])
    rng=np.random.default_rng(cfg['random_state']);B=cfg['bridge_bootstrap_repetitions'];band=[];arrays={};bootstrap_parameters=[]
    for kind in kinds:
        predictions=[];directions=[];attempts=0
        while len(predictions)<B:
            attempts+=1;ix=rng.integers(0,len(x),len(x));needed=3 if kind=='B1' else 2
            if np.unique(x[ix]).size<needed:continue
            fit=fit_candidate(kind,x[ix],y[ix]);predictions.append(predict(fit,grid));directions.append(decreasing(fit,lo,hi))
            bootstrap_parameters.append(dict(candidate=kind,replicate=len(predictions),decreasing=bool(directions[-1]),
                fitted_parameters=json.dumps(fit),source_model_indices=json.dumps(ix.tolist())))
        curves=np.array(predictions);residual=y-cv[kind];centered=residual-residual.mean()
        # Conditional residual-noise band, using OOF rather than in-sample
        # residuals; seven same-family rows do NOT justify a universal band.
        noise=rng.choice(centered,size=curves.shape);predictive=curves+noise
        ci=np.quantile(curves,[.025,.975],axis=0);pi=np.quantile(predictive,[.025,.975],axis=0);mean=predict(fitted[kind],grid)
        for i,l in enumerate(grid):
            band.append(dict(candidate=kind,selected=kind==selected,Val_Loss=float(l),A_pred=float(mean[i]),
                mean_CI_low=float(ci[0,i]),mean_CI_high=float(ci[1,i]),prediction_band_low=float(pi[0,i]),prediction_band_high=float(pi[1,i]),
                bootstrap_repetitions=B,bootstrap_decreasing_fraction=float(np.mean(directions)),resampling_attempts=attempts,
                outside_High_calibration_range=False,uncertainty_scope='within-Pythia paired bootstrap plus centered High LOMO residual noise'))
        arrays[kind+'_bootstrap_curve']=curves
    np.savez_compressed(out/'bridge_bootstrap_curves.npz',Val_Loss=grid,**arrays)
    csv(out/'loss_benchmark_bridge.csv',band);csv(out/'bridge_bootstrap_parameters.csv',bootstrap_parameters)
    csv(out/'bridge_validation_metrics.csv',scores);csv(out/'bridge_validation_predictions.csv',records);csv(out/'bridge_CV_fold_diagnostics.csv',folds)
    selected_fit=fitted[selected]
    cv_row=scores[(scores.candidate==selected)&(scores.evaluation=='High_LOMO_CV')].iloc[0]
    flat=bool(np.ptp(predict(selected_fit,grid))<1e-12)
    supported=bool(cv_row.R2>0 and cv_row.Pearson>0 and not flat)
    summary=dict(raw_C6_count=len(data),High_count=len(high),Medium_count=len(medium),
        other_count=int((data.comparability=='other').sum()),selected_candidate=selected,selection_rule='minimum High LOMO RMSE among decreasing fits; Medium never used to select',
        candidate_models=fitted,selected_model=selected_fit,calibrated_loss_range=[lo,hi],
        fixed_ability_parameters_source='outputs/evolution/phase1/ability_transform_parameters.json',
        High_families=high.model_family_diagnostic.unique().tolist(),CV='LOMO; Group CV impossible for sole High family',
        bootstrap_repetitions=B,Medium_outside_High_loss_range=int(((medium.Val_Loss<lo)|(medium.Val_Loss>hi)).sum()),
        selected_fit_is_constant=flat,bridge_valid_for_conversion=supported,
        bridge_status='DIAGNOSTIC_ONLY_FLAT_ISOTONIC' if flat else 'DIAGNOSTIC_ONLY_CV_NOT_VALIDATED' if not supported else 'WITHIN_FAMILY_LIMITED_VALIDATION',
        interpretation='numerical candidate ranking only; if CV fails or fit is constant, no useful predictive bridge is validated; no universal cross-family evidence',
        future_forecast_performed=False,contribution_decomposition_performed=False)
    dump(out/'bridge_model.json',summary);logging.info('Bridge selected %s; validation: %s',selected,scores.to_dict('records'))

# ============================================================
# Source: src/evolution/reporting.py
# Original lines: 1-212
# Purpose: 最终CSV/图表/报告呈现，compact省略纯绘图和排版
# ============================================================
"""Data readiness, twelve scientific figures, and a bounded Phase 1 report."""
import hashlib
import json
import logging
import re
import numpy as np
import pandas as pd
from .common import FILES,csv,dump,table


def read(out,name):return pd.read_csv(out/name,float_precision='round_trip')


def scale_readiness(root,cfg,models):
    """Inspect observed scale fields/units and sample support, without attribution."""
    out=root/'outputs/evolution/phase1';base=root/'real_attachments/C_efficiency_evolution'
    e=pd.read_csv(base/FILES['C4'],float_precision='round_trip');rows=[]
    for i,r in e.iterrows():
        notes=str(r['Parameters notes']);match=re.search(r'(?i)(\d+(?:\.\d+)?)\s*([kmbt])\b',notes)
        implied=float(match.group(1))*{'k':1e3,'m':1e6,'b':1e9,'t':1e12}[match.group(2).lower()] if match else np.nan
        N=pd.to_numeric(r.Parameters,errors='coerce');D=pd.to_numeric(r['Training dataset size (total)'],errors='coerce');F=pd.to_numeric(r['Training compute (FLOP)'],errors='coerce')
        ratio=abs(N-implied)/implied if pd.notna(N) and pd.notna(implied) and implied>0 else np.nan
        rows.append(dict(C4_row=i,Model=r.Model,Domain=r.Domain,N_parameters_raw=N,training_compute_raw=F,
            training_dataset_raw=r['Training dataset size (total)'],dataset_numeric=D,
            dataset_token_unit_confirmed=bool(re.search(r'(?i)\btokens?\b',str(r['Dataset size notes']))),
            parameter_notes_first_explicit_count=implied,parameter_notes_relative_difference=ratio,
            parameter_notes_possible_conflict=bool(pd.notna(ratio) and ratio>cfg['parameter_conflict_relative_threshold']),
            parameter_notes=r['Parameters notes'],Dataset_size_notes=r['Dataset size notes'],
            caution='notes may describe total versus active parameters; diagnostic only, no correction'))
    csv(out/'C4_scale_unit_audit.csv',rows)
    valid_date=models['Submission Date'].notna();positiveN=models.N_B.gt(0);sample=[]
    for name,mask in [('S0',models.S0),('S1',models.S1),('S2',models.S2),
        ('S1_N_submission',models.S1&positiveN&valid_date),('all_joint_N_D_compute',models.scale_joint_available),
        ('S1_joint_N_D_compute_submission_no_N_conflict',models.contribution_sample_candidate),
        ('S1_pretrained_joint',models.contribution_sample_pretrained),
        ('S1_joint_with_nonnegative_release_lag',models.contribution_sample_candidate&models.submission_release_days.ge(0))]:
        sample.append(dict(sample=name,n=int(mask.sum()),definition='no interpolated N/D/compute; unit/source/conflict checks retained'))
    csv(out/'contribution_sample_sizes.csv',sample)
    g=models[models.contribution_sample_candidate]
    correlation=np.log(g[['N_B','D_tokens','training_compute']]).corr()
    correlations=[]
    for a in correlation:
        for b in correlation:correlations.append(dict(variable_1=a,variable_2=b,log_correlation=correlation.loc[a,b],n=len(g)))
    csv(out/'scale_input_correlations.csv',correlations)
    counts={r['sample']:r['n'] for r in sample}
    readiness=dict(sample_counts=counts,contribution_analysis_started=False,
        sufficient_for_descriptive_ability_analysis=True,full_N_D_compute_decomposition_strongly_supported=False,
        universal_loss_bridge_validated=False,strict_license_sample_available=counts['S2']>0,
        recommended_phase2_scope='conditional limited-sample/N-only sensitivity possible after human confirmation; no causal or precise contribution-ratio claim yet',
        unresolved=['small complete-scale sample and selection bias','single-family High bridge fails predictive validation',
                    'license grant not established from attachments','submission cohorts are not necessarily contemporaneous evaluation frontiers',
                    'release/submission discrepancies and conservative model identity joins'])
    dump(out/'phase2_readiness.json',readiness);return readiness


def figures(root,cfg,models):
    """Twelve plots, including failed bridge evidence; no future extrapolation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'outputs/evolution/phase1';dest=root/'outputs/figures/question4_phase1';dest.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    manifest=[]
    def save(fig,name,title):
        fig.tight_layout();path=dest/name;fig.savefig(path,dpi=170);plt.close(fig)
        manifest.append(dict(file=path.relative_to(root).as_posix(),title=title))
    raw=pd.read_csv(root/'real_attachments/C_efficiency_evolution'/FILES['C2']);metrics=cfg['benchmarks']
    fig,ax=plt.subplots(figsize=(9,4));ax.boxplot([raw[m] for m in metrics],tick_labels=['IFEval','BBH','MATH-5','GPQA','MUSR','MMLU-PRO'],showfliers=False)
    ax.set(ylabel='Leaderboard score (0-100)',title='Six benchmark distributions: all original evaluation records',ylim=(0,100))
    save(fig,'01_benchmark_distributions.png','Six benchmark distributions')
    fig,ax=plt.subplots(figsize=(7,6));c=models.loc[models.S1,metrics].corr();im=ax.imshow(c,vmin=-1,vmax=1,cmap='RdBu_r')
    labels=['IFEval','BBH','MATH-5','GPQA','MUSR','MMLU-PRO'];ax.set_xticks(range(6),labels,rotation=35,ha='right');ax.set_yticks(range(6),labels)
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{c.iloc[i,j]:.2f}',ha='center',va='center',color='white' if abs(c.iloc[i,j])>.65 else 'black')
    fig.colorbar(im,ax=ax,label='Pearson r');ax.set_title('S1 task correlations');save(fig,'02_benchmark_correlations.png','S1 task Pearson heatmap')
    for number,col,title in [(3,cfg['average_column'],'Equal robust index versus official Average'),(4,'A_pca','Equal robust index versus PCA sensitivity')]:
        fig,ax=plt.subplots(figsize=(6,4));g=models[models.S0];ax.scatter(g.A_z,g[col],s=9,alpha=.15,color='#777777',label='S0')
        h=g[g.S1];ax.scatter(h.A_z,h[col],s=12,alpha=.45,color='#2878a8',label='S1');ax.set(xlabel='A_z (fixed robust equal-weight index)',ylabel='A_pca' if col=='A_pca' else 'Official Average',title=title);ax.legend()
        save(fig,f'{number:02d}_ability_index_comparison.png',title)
    match=read(out,'C8_matched_model_analysis.csv');g=match[match.unique_model_match&match.six_complete]
    fig,ax=plt.subplots(figsize=(7,4));ax.scatter(g.A_z_C8,g.bottom20_task_score,s=10,alpha=.3,color='#777777',label='S0 exact matches')
    h=g[g.S1];ax.scatter(h.A_z_C8,h.bottom20_task_score,s=15,alpha=.6,color='#2878a8',label='S1 exact matches')
    ax.set(xlabel='Same-run C8 ability, using fixed S1 transform',ylabel='Bottom 20% leaf-task mean (0-100)',title='C8 ability versus weak tail; chance-floor effects retained');ax.legend()
    save(fig,'05_C8_ability_vs_tail.png','C8 same-run ability and bottom20 tail')
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].hist(g.capability_imbalance.dropna(),bins=30,color='#2878a8',alpha=.8);axes[0].set(xlabel='Leaf score P90-P10 (percentage points)',ylabel='Models',title='Chance-normalized imbalance')
    axes[1].hist(g.difficulty_adjusted_imbalance.dropna(),bins=30,color='#c07328',alpha=.8);axes[1].set(xlabel='Within-task robust-z P90-P10',ylabel='Models',title='Task-difficulty sensitivity')
    save(fig,'06_C8_imbalance_distribution.png','C8 imbalance and difficulty-adjusted sensitivity')
    monthly=read(out,'monthly_ability_summary.csv');g=monthly[(monthly['sample']=='S1')&(monthly.model_type=='all')].sort_values('month')
    fig,ax=plt.subplots(figsize=(9,4))
    for field,label in [('median_A','P50'),('P90_A','P90'),('P95_A','P95')]:ax.plot(g.month,g[field],'o-',label=label)
    ax.set(ylabel='A_z',title='S1 submission cohorts: descriptive snapshot ability');ax.tick_params(axis='x',rotation=40);ax.legend();ax.grid(alpha=.2)
    save(fig,'07_monthly_ability_quantiles.png','Monthly S1 P50/P90/P95')
    fig,ax=plt.subplots(figsize=(9,4))
    for kind in ['pretrained','chat_finetuned']:
        h=monthly[(monthly['sample']=='S1')&(monthly.model_type==kind)].sort_values('month');ax.plot(h.month,h.P95_A,'o-',label=kind)
    ax.set(ylabel='Monthly A_z P95',title='S1 pretrained/chat-finetuned submission cohorts');ax.tick_params(axis='x',rotation=40);ax.legend();ax.grid(alpha=.2)
    save(fig,'08_model_type_ability_trends.png','Pretrained versus chat-finetuned monthly description')
    for number,field,label in [(9,'N_B','Leaderboard parameters (B)'),(10,'training_compute','Matched training compute (FLOPs)')]:
        fig,ax=plt.subplots(figsize=(7,4));g=models[models.S1&models[field].gt(0)&models.A_z.notna()]
        for kind,color in [('pretrained','#2878a8'),('chat_finetuned','#c07328'),('other_unknown','#777777')]:
            h=g[g.model_type==kind];ax.scatter(h[field],h.A_z,s=20,alpha=.6,label=f'{kind} (n={len(h)})',color=color)
        ax.set_xscale('log');ax.set(xlabel=label,ylabel='A_z',title='Observed metadata versus ability; no causal fit');ax.legend(fontsize=8)
        save(fig,f'{number:02d}_scale_vs_ability.png',label+' versus A_z')
    bridge=read(out,'C6_bridge_transformed_records.csv');bands=read(out,'loss_benchmark_bridge.csv');selected=json.loads((out/'bridge_model.json').read_text(encoding='utf8'))
    fig,ax=plt.subplots(figsize=(8,4))
    for kind,color in [('High','#2878a8'),('Medium','#c07328')]:
        h=bridge[bridge.comparability==kind];ax.scatter(h.Val_Loss,h.A_bridge,s=24,alpha=.7,label=f'{kind} (n={len(h)})',color=color)
    b=bands[bands.selected];ax.plot(b.Val_Loss,b.A_pred,color='#333333',label='High isotonic fit: constant, diagnostic only')
    ax.fill_between(b.Val_Loss.to_numpy(),b.mean_CI_low.to_numpy(),b.mean_CI_high.to_numpy(),color='#777777',alpha=.2,label='High paired-bootstrap mean band')
    ax.set(xlabel='Val_Loss (source-dependent comparability)',ylabel='A_bridge (same fixed ability scale)',title='Loss-ability bridge: no validated universal conversion');ax.legend(fontsize=8)
    save(fig,'11_loss_vs_ability_bridge.png','Bridge stratified by comparability')
    pred=read(out,'bridge_validation_predictions.csv');g=pred[pred.candidate==selected['selected_candidate']];fig,axes=plt.subplots(1,2,figsize=(10,4))
    scores=read(out,'bridge_validation_metrics.csv')
    for ax,kind in zip(axes,['High_LOMO_CV','Medium_external']):
        h=g[g.evaluation==kind];ax.scatter(h.observed_A,h.predicted_A,s=22,alpha=.7,color='#2878a8');lo=min(h.observed_A.min(),h.predicted_A.min());hi=max(h.observed_A.max(),h.predicted_A.max())
        ax.plot([lo,hi],[lo,hi],ls='--',color='#777777');r=scores[(scores.candidate==selected['selected_candidate'])&(scores.evaluation==kind)].iloc[0]
        ax.set(xlabel='Observed A_bridge',ylabel='Predicted A_bridge',title=f'{kind}\nRMSE={r.RMSE:.4f}, R2={r.R2:.3f}')
    save(fig,'12_bridge_observed_vs_predicted.png','Failed predictive validation remains visible')
    csv(out/'figure_manifest.csv',manifest)


def stage_report(root,cfg,models,readiness):
    """Answer every Phase 1 reporting question with real output links."""
    out=root/'outputs/evolution/phase1';ability=json.loads((out/'ability_execution_summary.json').read_text(encoding='utf8'))
    c8=json.loads((out/'C8_execution_summary.json').read_text(encoding='utf8'));bridge=json.loads((out/'bridge_model.json').read_text(encoding='utf8'))
    profile=read(out,'benchmark_profile.csv').query('scope == "C2_all_evaluations"')
    compare=read(out,'ability_index_comparison.csv');tails=read(out,'C8_weak_tail_diagnosis.csv');validation=read(out,'bridge_validation_metrics.csv');c3=read(out,'C3_time_series_audit.csv')
    counts=read(out,'contribution_sample_sizes.csv');consistent=read(out,'C1_C2_consistency.csv');parameters=json.loads((out/'ability_transform_parameters.json').read_text(encoding='utf8'))
    text=['# 问题四 PHASE 1：数据口径、综合能力与Loss–Benchmark桥接',
        '本阶段从C附件独立计算，没有读取旧问题四权重、指数、贡献率或预测，没有重新拟合前三问。所有“趋势”仅为已有样本描述；没有预测未来12/24个月、分解贡献比例、建立最终动力学或做因果解释。',
        '## 1 实际附件和数据诊断',table(read(out,'file_diagnosis.csv')),
        f"C1/C2有{ability['raw_C1_C2_records']}条六维完整评测、{ability['unique_models']}个不同Model字符串，重复额外记录{ability['duplicate_extra_records']}条。模型集合、参数、日期和类别一致。六维及Average若干浮点存在CSV重写舍入差异，最大{consistent.maximum_absolute_difference.max():.3g}；不是逐位完全相同，但1e-12审计容差内一致，不改动数据。主工作表采用C2。",
        'C2新增Epoch_AI_Publication_Date、Epoch_AI_Organization、Epoch_AI_Open_Weights，附件将其标为Epoch AI增强元数据，但没有明确C4模型ID；保守匹配须复核。原始同目录parquet额外提供评测行的架构/精度/权重类型、原始得分等，用于溯源，未读历史问题四草稿。',
        '全部字段缺失率、日期、参数及Benchmark范围见field_diagnosis.csv；原始数据只读。C4混合领域且存在缺失、非统一数据单位及部分参数notes冲突，不按模型名称猜值。',
        '## 2 样本层级、模型类型和时间轴',
        f"S0={ability['S0']}，S1={ability['S1']}，S2={ability['S2']}（每模型一行）。S0为六维完整、有限且0–100有效；S1仅接受C2明确Epoch开权重Yes或经保守C4匹配的Yes，并排除来源冲突。Hub可用不等同于开放权重或研究授权，未单独用作S1证据。",
        '许可证只有标识符而无模型授权条款，因此按指令标license_unknown；S2空，不能声称已经完成严格许可证敏感性验证。需要后续独立许可核验后才可使用S2，不自行做法律判断。',
        '重复模型按最新Submission Date评测集合；同日六任务和Average取中位数，不挑最高分。参数、类型、许可证等关键元数据若同日冲突则保留NaN，不平均参数量。名称规范化仅大小写、空白、常见URL前缀和尾斜杠；碰撞另存，不静默合并。',
        f"每模型类型：{ability['model_type_counts']}；S1内：{ability['S1_type_counts']}。优先Type；continuously pretrained归pretrained，领域微调归chat_finetuned，base merges单列other_unknown。仅Type缺失可名称辅助。全部分类来源见model_type_classification.csv。",
        f"排行榜主时间轴Submission Date。C4使用Publication date；二者不能互换。有效日期差{ability['time_lag_n']}个，中位数{ability['time_lag_median']:.6g}天，P10/P90={ability['time_lag_P10']:.6g}/{ability['time_lag_P90']:.6g}天，负差{ability['negative_submission_release_lag']}个，全部保留审计而非自动纠正。",
        'Submission月份描述的是当前快照下提交队列的得分，未证明每个分数是在该月测得；尤其历史记录可能已回溯重评。不能把这些曲线直接视为当时可获得能力的无偏实时前沿。发布日口径后续作为敏感性，负日期差/元数据身份不明需单列。',
        '## 3 六维Benchmark与固定综合能力',table(profile[['metric','n','missing','mean','std','min','P01','median','P99','max']]),
        f"六维完整率{ability['benchmark_complete_records']/ability['raw_C1_C2_records']:.2%}；没有负值、超过100的值或常数任务，未删异常。Pearson/Spearman矩阵见benchmark_correlations.csv。",
        r'主指标$A_z=A_{equal}=\frac16\sum_j[\operatorname{clip}(x_j,P01_j,P99_j)-median_j]/IQR_j$。六个任务越大越好；固定参数由S1全观测时期一次计算，没有每月重新标准化。缺任务不填补、不与完整六维指标混合。A_z允许负数，它表示相对标定样本的标准化能力，不是负的准确率。',
        'A_100只展示：以全部S0的A_z固定P01/P99线性映射并截到0–100；所有回归响应保留A_z。全期标定不是时间外验证，后续训练/验证分割必须说明是否重新建立仅训练期标定版本。',
        f"PCA第一成分六载荷同号、解释方差{ability['PC1_explained_variance']:.2%}，满足本阶段预先配置的≥50%条件。A_pca仅敏感性，主指标保持等权。",
        table(compare[(compare.scope=='S1')&(compare.comparison!='within_month_top10')][['comparison','index_1','index_2','n','Pearson','Spearman','top10_overlap']]),
        '等权与PCA高度接近；但与Average不能笼统称“完全一致”：S1头部和月度排序差异明显。原始Average受任务原始离散尺度影响，IQR标准化改变各任务每分的相对作用，尤其低离散任务更有区分作用。保留这些差异而不是为贴近Average调权。',
        '## 4 C8完整解析与细粒度短板',
        f"遍历{c8['directories']}目录、{c8['JSON_files']}JSON：可解析{c8['JSON_parse_success']}，损坏{c8['JSON_parse_failures']}；{c8['directories_no_usable_JSON']}目录无可用JSON。按实际文件运行时间选最新可解析，最终{c8['selected_models']}模型，六维完整{c8['six_complete_models']}。所有文件、版本、时间及选择情况保存在C8_parse_audit.csv，没有抽样。",
        f"真正细粒度数据为{c8['leaf_score_records']}个叶任务指标记录、{c8['subtask_keys']}种键。BBH/MATH/GPQA/MUSR保留子任务；IFEval使用两个strict组成项。C8 JSON的MMLU-PRO只有总体结果，没有类别级得分，未伪造分类短板；详表保存实际粒度。",
        '随机基线归一化在子任务层执行并保留baseline证据，低于猜测基线取0；IFEval/MATH生成任务基线0，GPQA为四选一，MMLU-PRO基线0.1，BBH用配置选项数，MUSR依2/5/3选项数。GPQA/MATH组聚合与等权子任务均值分开保存，避免样本加权口径混用。[官方归一化说明](https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/normalization)。',
        f"C8精确规范化接受{c8['exact_matches']}个匹配；状态{c8['match_status_counts']}。模糊候选不自动接受，多重规范化身份保留ambiguous。六维完整且唯一身份可分析{c8['valid_six_complete_matched_models']}个。",
        table(read(out,'C8_six_task_comparison.csv')[['benchmark','n','MAE','systematic_bias','Pearson','Spearman']]),
        'MATH差异明显大于其他任务，不自动修正。官方2025年2月说明曾用Math-Verify重新评价历史模型；因此版本/重评差异是有依据的可能解释，但这些统计不能逐条证明差异原因。[官方重评说明](https://huggingface.co/blog/math_verify_leaderboard)。',
        r'capability_imbalance为实际叶任务0–100分数的P90−P10；另保留标准差、P10、最大最小跨度、bottom20均值。主操作定义：$A\ge P75$且$bottom20\le P25$，阈值由各匹配样本实际计算。',
        table(tails[['sample','ability_basis','n','high_average_weak_tail_count','high_average_weak_tail_rate','rate_within_high_ability','difficulty_adjusted_weak_tail_rate']]),
        '同时用C8同一运行六维得分经相同固定参数得到A_z_C8，避免把新排行榜A与旧C8短板混用。另按子任务及版本做稳健标准化控制任务难度，作为短板敏感性。任务数量不均（BBH叶任务多）和猜测基线截零均影响汇总；小比例案例说明某些局部差异存在，不能声称普遍严重掩盖，也不能从不均衡与A相关性推出因果。',
        '## 5 C3时序和C4尺度元数据',
        f"C3实际年份{ability['C3_year_min']}–{ability['C3_year_max']}；只有Year，没有月/完整日期，不制造2019年1月等时间点。原始来源包含历史论文/报告和排行榜。{int(c3.comparable_main_scale.sum())}条与C2同名记录在六任务上匹配；历史补充不直接纳入共同尺度A或月度序列。逐记录审计见C3_time_series_audit.csv。",
        f"保守C4匹配{ability['C4_matched']}个排行榜模型。已有C2元数据优先保留；C4全名或唯一完整basename须元数据/owner佐证才接受，未接受模糊猜测。参数相对差>25%标conflict，共{ability['C4_parameter_conflicts']}个，阈值为诊断约定，未自动覆盖排行榜参数。",
        table(read(out,'C4_scale_metadata_summary.csv')[['scope','field','n','available']]),
        'C4全库的三字段齐全不等于排行榜分析可用：训练数据总量混合tokens/图像等。只在notes确认token单位时记D_tokens；否则可用C6明确D_tokens_B作为有来源的辅助数据，comparability仍保留。C7 training_data_TB没有转换为token。逐字段来源、原值、日期、架构/上下文和C8指标保留在evolution_model_table.csv。',
        table(counts[['sample','n']]),
        '模型变量来源不混同：N_B取实际排行榜值，Epoch_N_B及C6_N_B并列；compute仅来自匹配C4，不以6ND填补。观察到的N/D/compute相关性见scale_input_correlations.csv，只诊断共线性，没有贡献分解。',
        '## 6 High主拟合、Medium外验的Loss桥接',
        f"C6共有{bridge['raw_C6_count']}个模型，High={bridge['High_count']}、Medium={bridge['Medium_count']}；C5仅作一致性对照，没有合并进主拟合。High全部Pythia。C6不存在实际family列，因此采用Leave-One-Model-Out；由于只有一个High族，跨族Group CV不可实施，不能将LOMO称为跨族验证。",
        '六个LB字段应用相同S1变换和1/6权重得到A_bridge，未在C6重拟合指数。候选仅B0线性、B1二次、B2单调递减isotonic。主拟合只用High；Medium只外验，不参与选模。',
        f"B0：A={bridge['candidate_models']['B0']['coefficients'][0]:.9g}+({bridge['candidate_models']['B0']['coefficients'][1]:.9g})Loss，斜率为正，理论方向未获支持。B1系数={bridge['candidate_models']['B1']['coefficients']}，仅曲率敏感性，在校准区间未保持单调下降。",
        f"数值候选中选择{bridge['selected_candidate']}；但它为常数A={bridge['selected_model']['y'][0]:.9g}，不是有辨识能力的Loss–能力桥接。状态={bridge['bridge_status']}，bridge_valid_for_conversion={bridge['bridge_valid_for_conversion']}。**本阶段没有通过验证的可用预测桥接，不能据此把前三问Loss可靠转换成Benchmark提升。**",
        table(validation[['candidate','evaluation','n','RMSE','MAE','R2','Pearson','Spearman','decreasing_full']]),
        f"所有High CV R²为负；常数单调模型的LOMO预测可与留出响应负相关，这是留出均值变化的结果，不能把Spearman=-1误读为理想Loss方向。Medium中的{bridge['Medium_outside_High_loss_range']}个Loss超出High范围；常数外验相关性未定义，保留NaN而不写0。",
        f"每个候选执行{bridge['bootstrap_repetitions']}次High模型配对bootstrap，保存参数、抽样索引、均值95%带及加中心化LOMO残差的条件预测带。只在High Loss区间{bridge['calibrated_loss_range']}生成曲线。7个同族模型的bootstrap不能证明跨族/未来预测有效，更不能用窄带掩盖桥接失败。",
        '可复用函数evolution.bridge.predict_bridge仅返回诊断值、校准范围标记和有效性状态。后续若得到可靠桥接，必须传播其不确定性；当前Benchmark历史模型仍应独立基于Leaderboard，不能由失败桥接主导。',
        '## 7 PHASE 2准备程度与停止点',
        f"已有{readiness['sample_counts']['S1_N_submission']}个S1模型可做N与Submission时间条件描述；完整N/D/compute、日期和无参数冲突的候选仅{readiness['sample_counts']['S1_joint_N_D_compute_submission_no_N_conflict']}个，其中pretrained {readiness['sample_counts']['S1_pretrained_joint']}。允许在人工确认后开展有限样本、分类型、非因果敏感性分析；尚不足以声称强证据的完整规模/非规模技术贡献率分解。",
        '主要待确认：S2许可核验缺失，规模全信息样本稀少且有选择偏差，快照分数与提交时间不等于实时前沿，发布日异常需要分层，High桥接失效。这些限制已保留，不通过插值、合并不可比Loss或模型名猜值补齐。',
        '本阶段到此停止。没有预测、没有贡献比例、没有因果结论、没有最终动力学，也没有论文全文整合；等待人工确认。',
        '## 附录：输出与复现',
        '`python -X utf8 -B scripts/20_build_evolution_ability_bridge.py --stage all`；分阶段ability/C8/bridge/report。`python -X utf8 -B scripts/20_verify_evolution_phase1.py`独立核验。目录自动创建，随机种子固定，pathlib定位根目录，前三问文件只读。',
    ]
    for p in sorted(out.glob('*.csv')):text.append(f'- [{p.name}](../evolution/phase1/{p.name})')
    for p in ['ability_transform_parameters.json','bridge_model.json','phase2_readiness.json','verification_summary.json']:
        text.append(f'- [{p}](../evolution/phase1/{p})')
    for p in read(out,'figure_manifest.csv').file:text.append(f'![{p.split("/")[-1]}](../{p.removeprefix("outputs/")})')
    (root/'outputs/report/question4_phase1_data_ability_bridge.md').write_text('\n\n'.join(text)+'\n',encoding='utf8')


def finalize(root,cfg):
    """Refresh only this run's cached C8 analysis; write final figures and report."""
    out=root/'outputs/evolution/phase1'
    from .detailed import refresh_weak_tail
    refresh_weak_tail(root,cfg);models=read(out,'evolution_model_table.csv')
    readiness=scale_readiness(root,cfg,models);figures(root,cfg,models)
    base=root/'real_attachments/C_efficiency_evolution';sources=[base/p for p in FILES.values()]+[base/'data/train-00000-of-00001.parquet']
    sources+=sorted((base/'detailed_results').rglob('*.json'))
    rows=[]
    for p in sources:
        with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        rows.append(dict(file=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest))
    csv(out/'input_manifest.csv',rows)
    implementation=[root/'configs/evolution_phase1.yaml',root/'scripts/20_build_evolution_ability_bridge.py',root/'scripts/20_verify_evolution_phase1.py']+sorted((root/'src/evolution').glob('*.py'))
    rows=[]
    for p in implementation:
        with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        rows.append(dict(file=p.relative_to(root).as_posix(),sha256=digest))
    csv(out/'implementation_manifest.csv',rows)
    stage_report(root,cfg,models,readiness)
    summary=dict(phase='Question 4 PHASE 1',old_question4_results_reused=False,previous_questions_refitted=False,
        future_prediction_performed=False,contribution_decomposition_performed=False,causal_claims=False,waiting_for_human_confirmation=True,
        figures=12,main_ability='A_z=A_equal',strict_license_status='unconfirmed',readiness=readiness)
    dump(out/'execution_summary.json',summary);logging.info('Question 4 PHASE 1 report complete; awaiting human confirmation')

# ============================================================
# Source: scripts/20_build_evolution_ability_bridge.py
# Original lines: 1-42
# Purpose: 最终阶段实际执行入口：20_build_evolution_ability_bridge.py
# ============================================================
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

# ============================================================
# Source: src/evolution/contribution_regression.py
# Original lines: 1-211
# Purpose: OLS/HC3/Huber、LOOCV、Cook/VIF、双因素Shapley与5000模型bootstrap
# Huber, quadratic time and month FE are sensitivity/candidate comparisons. Shapley signed components are not future growth weights.
# ============================================================
"""Small, auditable regression and paired-bootstrap contribution calculations."""
import json
import logging
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import HuberT
from .common import corr


def design(data,terms):
    """Use an explicit intercept and the requested predictors in fixed order."""
    return np.column_stack([np.ones(len(data)),data[terms].to_numpy(float)])


def accuracy(y,pred):
    """Undefined correlations/R² remain missing, rather than become zero."""
    y=np.asarray(y,float);pred=np.asarray(pred,float);ok=np.isfinite(y)&np.isfinite(pred)
    y=y[ok];pred=pred[ok];err=pred-y;ss=np.sum((y-y.mean())**2)
    return dict(n=len(y),RMSE=float(np.sqrt(np.mean(err**2))),MAE=float(np.mean(abs(err))),
                R2=float(1-err@err/ss) if ss>0 else np.nan,
                Pearson=corr(y,pred),Spearman=corr(y,pred,'spearman'))


def fit_regression(data,terms,label,scope,axis='submission',method='OLS'):
    """Fit OLS or HuberT RLM without importing incompatible statsmodels.api."""
    X=design(data,terms);y=data.A_z.to_numpy(float);names=['intercept']+terms
    if len(data)<=X.shape[1]+1 or np.linalg.matrix_rank(X)<X.shape[1]:
        raise ValueError(f'{label}: insufficient full-rank sample')
    if method=='OLS':
        fit=OLS(y,X).fit();leverage=np.einsum('ij,ji->i',X,np.linalg.pinv(X))
        robust=fit.get_robustcov_results(cov_type='HC3',use_t=True) if np.min(1-leverage)>1e-9 else None
        variants={'OLS':fit,'HC3':robust}
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');fit=RLM(y,X,M=HuberT()).fit(maxiter=200,cov='H3')
        variants={'Huber_H3':fit}
        for w in caught:logging.warning('%s: %s',label,str(w.message))
    coefficients=[]
    for inference,v in variants.items():
        if v is None:
            ci=np.full((len(names),2),np.nan);values=np.asarray(fit.params);SE=np.full(len(names),np.nan);pv=SE.copy()
            status='UNIDENTIFIABLE_LEVERAGE_ONE'
        else:
            ci=np.asarray(v.conf_int());values=np.asarray(v.params);SE=np.asarray(v.bse);pv=np.asarray(v.pvalues);status='AVAILABLE'
        for j,name in enumerate(names):
            coefficients.append(dict(model=label,scope=scope,time_axis=axis,method=method,inference=inference,
                term=name,n=len(data),coef=values[j],SE=SE[j],CI_low=ci[j,0],CI_high=ci[j,1],p_value=pv[j],inference_status=status))
    fitted=np.asarray(fit.fittedvalues);evaluation=dict(model=label,scope=scope,time_axis=axis,method=method,
        evaluation='in_sample',**accuracy(y,fitted),adjusted_R2=float(fit.rsquared_adj) if method=='OLS' else np.nan,
        AIC=float(fit.aic) if method=='OLS' else np.nan,BIC=float(fit.bic) if method=='OLS' else np.nan,
        AICc=float(fit.aic+2*X.shape[1]*(X.shape[1]+1)/(len(data)-X.shape[1]-1)) if method=='OLS' else np.nan)
    predictions=[];oof=[];fold_rows=[]
    for i in range(len(data)):
        mask=np.arange(len(data))!=i;Xt=X[mask];yt=y[mask];rank=np.linalg.matrix_rank(Xt)
        identifiable=rank==X.shape[1]
        if identifiable:
            if method=='OLS':beta=np.linalg.lstsq(Xt,yt,rcond=None)[0]
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore');beta=np.asarray(RLM(yt,Xt,M=HuberT()).fit(maxiter=200).params)
            estimate=float(X[i]@beta)
        else:beta=np.full(X.shape[1],np.nan);estimate=np.nan
        oof.append(estimate)
        fold_rows.append(dict(model=label,scope=scope,time_axis=axis,method=method,held_out_Model=data.Model.iloc[i],
            full_rank=identifiable,training_n=len(yt),training_rank=rank,parameters=json.dumps(beta.tolist())))
        predictions.append(dict(model=label,scope=scope,time_axis=axis,method=method,Model=data.Model.iloc[i],
            observed_A=y[i],fitted_A=fitted[i],LOOCV_predicted_A=estimate,residual=y[i]-fitted[i]))
    count=int(np.isfinite(oof).sum());cv=dict(model=label,scope=scope,time_axis=axis,method=method,evaluation='LOOCV',
        **accuracy(y,oof),eligible_n=len(data),unidentifiable_folds=len(data)-count,
        full_LOOCV_available=count==len(data))
    # A partial subset is not substituted for a full CV score in model selection.
    cv['partial_RMSE']=cv['RMSE']
    if count<len(data):
        for k in ['RMSE','MAE','R2','Pearson','Spearman']:cv[k]=np.nan
    return dict(label=label,scope=scope,axis=axis,method=method,terms=terms,names=names,n=len(data),
        data=data.copy(),X=X,y=y,fit=fit,beta=np.asarray(fit.params),
        coefficients=coefficients,evaluations=[evaluation,cv],predictions=predictions,folds=fold_rows)


def diagnostics(model):
    """VIF, standardized condition number, Koenker BP, residuals and Cook's D."""
    X=model['X'];y=model['y'];beta=model['beta'];n,p=X.shape;res=y-X@beta
    hat=np.einsum('ij,ji->i',X,np.linalg.pinv(X));mse=res@res/(n-p)
    cook=np.full(n,np.nan);identifiable=(1-hat)>1e-9
    if mse>0:cook[identifiable]=(res[identifiable]**2/(p*mse))*hat[identifiable]/(1-hat[identifiable])**2
    else:cook[identifiable]=0
    vif=[]
    for j in range(1,p):
        target=X[:,j];other=np.delete(X,j,axis=1);error=target-other@np.linalg.lstsq(other,target,rcond=None)[0]
        total=np.sum((target-target.mean())**2);r2=1-error@error/total if total>0 else 1
        v=1/(1-r2) if r2<1-1e-14 else np.nan
        vif.append(dict(model=model['label'],scope=model['scope'],term=model['names'][j],VIF=v,
            exactly_collinear=r2>=1-1e-14))
    standardized=(X[:,1:]-X[:,1:].mean(axis=0))/X[:,1:].std(axis=0,ddof=0)
    raw_condition=np.linalg.cond(X);scaled_condition=np.linalg.cond(np.column_stack([np.ones(n),standardized]))
    aux=OLS(res**2,X).fit();LM=n*aux.rsquared;df=p-1
    BPp=stats.chi2.sf(LM,df);fvalue=aux.fvalue;fp=aux.f_pvalue
    JB=stats.jarque_bera(res)
    summary=dict(model=model['label'],scope=model['scope'],n=n,p=p,
        condition_number_raw=raw_condition,condition_number_standardized=scaled_condition,
        Koenker_BP_LM=LM,Koenker_BP_p=BPp,Koenker_BP_F=fvalue,Koenker_BP_F_p=fp,
        residual_mean=res.mean(),residual_std=res.std(ddof=1),Jarque_Bera=JB.statistic,JB_p=JB.pvalue,
        Cook_threshold=4/n,Cook_exceed_count=int((cook>4/n).sum()),Cook_unidentifiable_count=int((~identifiable).sum()),max_Cook=float(np.nanmax(cook)),
        max_Cook_Model=model['data'].Model.iloc[int(np.nanargmax(cook))])
    influence=[dict(model=model['label'],Model=row.Model,leverage=hat[i],Cooks_distance=cook[i],
        Cook_warning=bool(cook[i]>4/n),residual=res[i]) for i,row in enumerate(model['data'].itertuples())]
    return vif,summary,influence


def shapley(beta,terms,early,late):
    """Two-factor endpoint Shapley with grouped scale predictors; preserve signs."""
    def f(state,t):return beta[0]+sum(beta[k+1]*(t if term=='t_month' else state[term]) for k,term in enumerate(terms))
    s0={term:early[term] for term in terms if term!='t_month'};s1={term:late[term] for term in terms if term!='t_month'}
    t0=early['t_month'];t1=late['t_month'];f00=f(s0,t0);f10=f(s1,t0);f01=f(s0,t1);f11=f(s1,t1)
    ds=.5*(f10-f00+f11-f01);dt=.5*(f01-f00+f11-f10);total=f11-f00;err=abs(total-ds-dt)
    if err>=1e-10:raise ArithmeticError('Shapley components do not sum to predicted change')
    physical=total>0 and ds>=0 and dt>=0;den=abs(ds)+abs(dt)
    return dict(A_early_pred=f00,A_late_pred=f11,Delta_pred=total,Delta_scale=ds,Delta_tech=dt,
        identity_error=err,nonnegative_growth_shares_valid=physical,cancellation_effect=bool(ds*dt<0),
        share_scale=ds/total if physical else np.nan,share_tech=dt/total if physical else np.nan,
        signed_ratio_scale=ds/total if abs(total)>1e-12 else np.nan,
        signed_ratio_tech=dt/total if abs(total)>1e-12 else np.nan,
        absolute_share_scale=abs(ds)/den if den>1e-12 else np.nan,
        absolute_share_tech=abs(dt)/den if den>1e-12 else np.nan)


def endpoints(data,terms,bounds,fraction):
    """Fixed original time thirds/quarters; endpoint medians use resampled models."""
    lo,hi=bounds
    e=data[data.t_month.between(lo,lo+fraction*(hi-lo))]
    l=data[data.t_month.between(hi-fraction*(hi-lo),hi)]
    if e.empty or l.empty:return None,None,len(e),len(l)
    fields=set(terms+['t_month','lnN','lnD','lnC','A_z','N_B','D_tokens','training_compute'])
    states=[]
    for h in [e,l]:
        state={k:h[k].median() for k in fields}
        for raw,log in [('D_tokens','lnD'),('training_compute','lnC')]:
            state[log]=np.log(state[raw]) if np.isfinite(state[raw]) and state[raw]>0 else np.nan
        states.append(state)
    return states[0],states[1],len(e),len(l)


def contribution(model,bounds,fraction,series):
    """Main fitted contribution plus actual endpoint-state diagnostics."""
    e,l,ne,nl=endpoints(model['data'],model['terms'],bounds,fraction)
    base=dict(series=series,model=model['label'],scope=model['scope'],time_axis=model['axis'],
        window_fraction=fraction,n=model['n'],early_n=ne,late_n=nl,time_bound_min=bounds[0],time_bound_max=bounds[1])
    if e is None:return dict(**base,status='EMPTY_ENDPOINT_WINDOW')
    base.update(status='OK',**shapley(model['beta'],model['terms'],e,l))
    for prefix,s in [('early',e),('late',l)]:
        for k,v in s.items():base[prefix+'_'+k]=v
    base['observed_median_A_change']=l['A_z']-e['A_z']
    return base


def bootstrap(model,bounds,fractions,series,B,seed):
    """Exactly B model draws; unavailable endpoints/rank retained and audited."""
    rng=np.random.default_rng(seed);d=model['data'];X=model['X'];y=model['y'];n,p=X.shape
    allrows=[];coefs=[]
    # Avoid repeated pandas sampling/grouping in 5000 small bootstrap draws.
    fields=['lnN','lnD','lnC','t_month','A_z','N_B','D_tokens','training_compute'];values=d[fields].to_numpy(float);fi={k:i for i,k in enumerate(fields)}
    for rep in range(1,B+1):
        ix=rng.integers(0,n,n);Xt=X[ix];yt=y[ix];full=np.linalg.matrix_rank(Xt)==p
        b=np.linalg.lstsq(Xt,yt,rcond=None)[0] if full else np.full(p,np.nan)
        coefs.append(dict(model=model['label'],series=series,time_axis=model['axis'],replicate=rep,
            full_rank=full,source_model_indices=json.dumps(ix.tolist()),**dict(zip(model['names'],b))))
        v=values[ix];time=v[:,fi['t_month']]
        for frac in fractions:
            lo,hi=bounds;em=(time>=lo)&(time<=lo+frac*(hi-lo));lm=(time>=hi-frac*(hi-lo))&(time<=hi)
            row=dict(series=series,model=model['label'],scope=model['scope'],time_axis=model['axis'],
                window_fraction=frac,replicate=rep,early_n=int(em.sum()),late_n=int(lm.sum()),full_rank=full)
            if not full:row['status']='RANK_DEFICIENT'
            elif not em.any() or not lm.any():row['status']='EMPTY_ENDPOINT_WINDOW'
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    ev=np.nanmedian(v[em],axis=0);lv=np.nanmedian(v[lm],axis=0)
                e=dict(zip(fields,ev));l=dict(zip(fields,lv))
                for state in [e,l]:
                    for raw,log in [('D_tokens','lnD'),('training_compute','lnC')]:
                        state[log]=np.log(state[raw]) if np.isfinite(state[raw]) and state[raw]>0 else np.nan
                row.update(status='OK',**shapley(b,model['terms'],e,l))
            allrows.append(row)
        if rep%1000==0:logging.info('%s bootstrap %d/%d',model['label'],rep,B)
    return allrows,coefs


def bootstrap_summary(rows):
    """Conditional valid percentiles, signed ratios and nonnegative-growth rates."""
    frame=pd.DataFrame(rows);result=[]
    quantities=['Delta_pred','Delta_scale','Delta_tech','signed_ratio_scale','signed_ratio_tech',
        'share_scale','share_tech','absolute_share_scale','absolute_share_tech']
    for keys,g in frame.groupby(['series','model','scope','time_axis','window_fraction'],dropna=False):
        valid=g[g.status=='OK'];base=dict(zip(['series','model','scope','time_axis','window_fraction'],keys))
        for q in quantities:
            x=valid[q].dropna() if q in valid else pd.Series(dtype=float)
            result.append(dict(**base,quantity=q,draws=len(g),valid_decomposition_draws=len(valid),
                valid_quantity_draws=len(x),invalid_window_or_rank_draws=len(g)-len(valid),
                CI_low=x.quantile(.025) if len(x) else np.nan,CI_high=x.quantile(.975) if len(x) else np.nan,
                median=x.median() if len(x) else np.nan,
                nonnegative_growth_fraction=float(valid.nonnegative_growth_shares_valid.mean()) if len(valid) else np.nan,
                positive_scale_fraction=float(valid.Delta_scale.gt(0).mean()) if len(valid) else np.nan,
                positive_tech_fraction=float(valid.Delta_tech.gt(0).mean()) if len(valid) else np.nan,
                cancellation_fraction=float(valid.cancellation_effect.mean()) if len(valid) else np.nan,
                ratio_CI_outside_zero_one=bool(len(x) and q.startswith('signed_ratio') and (x.quantile(.025)<0 or x.quantile(.975)>1)),
                interval_scope='percentile; conditional on identifiable sampled endpoints; physical shares additionally conditional on nonnegative growth'))
    return result

# ============================================================
# Source: src/evolution/contribution_analysis.py
# Original lines: 1-284
# Purpose: T0-T3、M1/M2-C/M3、组织/时间轴对照、技术残差和分解执行
# ============================================================
"""Question 4 Phase 2 samples and descriptive scale/time contribution analysis."""
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from .common import csv,table
from .contribution_regression import (fit_regression,diagnostics,contribution,bootstrap,bootstrap_summary,design)


def write_json(path,data):
    """Convert nonfinite diagnostic values to JSON null."""
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
        if isinstance(x,(np.bool_,)):return bool(x)
        if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
        if isinstance(x,(np.integer,)):return int(x)
        return x
    Path(path).write_text(json.dumps(clean(data),ensure_ascii=False,indent=2),encoding='utf8')


def prepare(root,cfg):
    """Use the frozen per-model table, independently audit original record selection."""
    out=root/'outputs/evolution/phase2';p1=root/'outputs/evolution/phase1'
    d=pd.read_csv(p1/'evolution_model_table.csv',float_precision='round_trip')
    raw=pd.read_csv(root/'real_attachments/C_efficiency_evolution/leaderboard_enhanced.csv',float_precision='round_trip')
    if not d.Model.is_unique:raise ValueError('Frozen Phase 1 model table is not unique')
    dates=pd.to_datetime(raw['Submission Date'],errors='coerce');audits=[]
    for row in d.to_dict('records'):
        g=raw[raw.Model==row['Model']];latest=dates.loc[g.index].max()
        chosen=g.loc[dates.loc[g.index].eq(latest)] if pd.notna(latest) else g
        ids=json.loads(row['source_rows']);consistent=sorted(chosen.index.tolist())==sorted(ids)
        if not consistent:raise ValueError('Frozen dedup selection differs from latest-date rule: '+row['Model'])
        audits.append(dict(Model=row['Model'],original_records=len(g),latest_submission=str(latest),
            retained_same_date_evaluations=len(ids),retained_source_rows=json.dumps(ids),
            discarded_older_source_rows=json.dumps(sorted(set(g.index)-set(ids))),
            tie_rule='componentwise benchmark median; frozen Phase 1 deterministic rule; metadata requires agreement',
            output_model_rows=1,latest_date_verified=consistent,winner_selection_used=False))
    csv(out/'dedup_audit.csv',audits)
    for col in ['A_z','N_B','D_tokens','training_compute']:d[col]=pd.to_numeric(d[col],errors='coerce')
    d['submission_date']=pd.to_datetime(d['Submission Date'],errors='coerce')
    d['release_date']=pd.to_datetime(d['Release Date'],errors='coerce')
    ok=np.isfinite(d.A_z)&np.isfinite(d.N_B)&d.N_B.gt(0)&d.submission_date.notna()
    primary=d.S1&d.model_type.eq('pretrained')&ok
    if primary.sum()<10:raise ValueError('Insufficient main pretrained sample')
    origin=d.loc[primary,'submission_date'].min();end=d.loc[primary,'submission_date'].max()
    release_origin=d.loc[primary,'release_date'].min()
    d['t_submission']=(d.submission_date-origin).dt.total_seconds()/86400/cfg['days_per_month']
    d['t_release']=(d.release_date-release_origin).dt.total_seconds()/86400/cfg['days_per_month']
    for rawcol,logcol in [('N_B','lnN'),('D_tokens','lnD'),('training_compute','lnC')]:
        d[logcol]=np.log(d[rawcol].where(np.isfinite(d[rawcol])&d[rawcol].gt(0)))
    counts=[]
    for sample in ['S0','S1']:
        for kind in ['pretrained','chat_finetuned','other_unknown']:
            basic=d[sample]&d.model_type.eq(kind)&ok
            for tier,extras in [('T0',[]),('T1',['lnC']),('T2',['lnD']),('T3',['lnC','lnD'])]:
                mask=basic.copy()
                for field in extras:mask&=d[field].notna()
                name=f'{sample}_{kind}_{tier}';d[name]=mask
                counts.append(dict(sample=sample,model_type=kind,tier=tier,n=int(mask.sum()),
                    parameter_conflict_n=int((mask&d.parameter_conflict).sum()),
                    release_date_n=int((mask&d.release_date.notna()).sum())))
    csv(out/'contribution_sample_tiers.csv',counts);csv(out/'analysis_model_table.csv',d)
    metadata=dict(primary_origin=str(origin.date()),primary_end=str(end.date()),
        release_origin=str(release_origin.date()),days_per_month=cfg['days_per_month'],
        primary_bounds=[0,float((end-origin).total_seconds()/86400/cfg['days_per_month'])],
        N_units='billion parameters; natural log',D_units='tokens; natural log',C_units='observed C4 FLOP; natural log',
        dedup='frozen latest-date cohort; same-date deterministic componentwise medians',
        no_frozen_ability_or_scale_imputation=True)
    write_json(out/'sample_definition.json',metadata)
    logging.info('Final tiers: %s',counts[:12]);return d,metadata


def model_data(d,scope,axis):
    """Take precisely one declared sample; origin changes do not refit ability."""
    g=d[d[scope]].copy()
    if axis=='release':g=g[g.release_date.notna()&g.Epoch_match.notna()].copy()
    g['t_month']=g['t_release' if axis=='release' else 't_submission']
    return g.reset_index(drop=True)


def fit_models(root,cfg,d,metadata):
    """OLS main, prescribed robustness, low-dimensional restricted fits."""
    out=root/'outputs/evolution/phase2';models={};coef=[];evaluation=[];prediction=[];folds=[];vifs=[];diag=[];influence=[];definitions=[]
    def add(label,scope,terms,axis='submission',method='OLS',data=None,do_diagnostics=True):
        g=model_data(d,scope,axis) if data is None else data.reset_index(drop=True)
        m=fit_regression(g,terms,label,scope,axis,method);models[label]=m
        coef.extend(m['coefficients']);evaluation.extend(m['evaluations']);prediction.extend(m['predictions']);folds.extend(m['folds'])
        if method=='OLS' and do_diagnostics:
            v,s,i=diagnostics(m);vifs.extend(v);diag.append(s);influence.extend(i)
        definitions.append(dict(model=label,scope=scope,terms=terms,time_axis=axis,method=method,Models=g.Model.tolist(),
            beta=m['beta'].tolist(),n=len(g)))
        return m
    main_scope='S1_pretrained_T0';T3='S1_pretrained_T3';terms=['lnN','t_month']
    main=add('M1_linear',main_scope,terms)
    huber=add('M1_Huber',main_scope,terms,method='Huber')
    quad=main['data'].copy();quad['t_squared']=quad.t_month**2
    add('M1_quadratic',main_scope,['lnN','t_month','t_squared'],data=quad)
    month=main['data'].copy();dummy=pd.get_dummies(month.submission_date.dt.to_period('M').astype(str),prefix='month',drop_first=True,dtype=float)
    month=pd.concat([month,dummy],axis=1)
    if month.submission_date.dt.to_period('M').nunique()>=3 and len(month)>len(dummy.columns)+5:
        add('M1_month_FE',main_scope,['lnN']+dummy.columns.tolist(),data=month)
    for number in [1,2]:
        inf=pd.DataFrame([r for r in influence if r['model']=='M1_linear']).sort_values(['Cooks_distance','Model'],ascending=[False,True])
        removed=inf.head(number).Model.tolist();g=main['data'][~main['data'].Model.isin(removed)]
        add(f'M1_drop_Cook_top{number}',main_scope,terms,data=g)
    add('M1_S0_sensitivity','S0_pretrained_T0',terms)
    add('M1_chat','S1_chat_finetuned_T0',terms)
    add('M1_same_T3',T3,terms)
    add('M2_C',T3,['lnC','t_month'])
    add('M3_ND',T3,['lnN','lnD','t_month'])
    for label,terms_here in [('M1_same_T3',['lnN','t_month']),('M2_C',['lnC','t_month']),('M3_ND',['lnN','lnD','t_month'])]:
        g=models[label]['data'];filtered=g[~g.parameter_conflict]
        if len(filtered)<len(g):add(label+'_no_N_conflict',T3,terms_here,data=filtered)
    for label,scope,terms_here in [('M1',main_scope,terms),('M2_C',T3,['lnC','t_month']),('M3_ND',T3,['lnN','lnD','t_month'])]:
        matched=model_data(d,scope,'release')
        if len(matched)>len(terms_here)+4:
            add(label+'_release',scope,terms_here,axis='release')
            sub=model_data(d,scope,'submission');sub=sub[sub.Model.isin(matched.Model)]
            if len(sub)!=models['M1_linear' if label=='M1' else label]['n']:
                add(label+'_submission_matched',scope,terms_here,data=sub)
    org=model_data(d,main_scope,'submission');org['org_cluster']=org.Organization.fillna('').str.strip().str.casefold()
    known=org[org.org_cluster.ne('')];group_counts=known.org_cluster.value_counts();org_audit=[]
    if len(group_counts)>=5 and len(known)>len(terms)+5:
        org_model=add('M1_organization_known',main_scope,terms,data=known)
        robust=org_model['fit'].get_robustcov_results(cov_type='cluster',groups=known.org_cluster.to_numpy(),use_correction=True,df_correction=True,use_t=True)
        ci=np.asarray(robust.conf_int())
        for j,name in enumerate(org_model['names']):
            coef.append(dict(model='M1_organization_known',scope=main_scope,time_axis='submission',method='OLS',inference='organization_clustered',
                term=name,n=len(known),coef=robust.params[j],SE=robust.bse[j],CI_low=ci[j,0],CI_high=ci[j,1],p_value=robust.pvalues[j],clusters=len(group_counts)))
    if len(group_counts):
        largest=group_counts.index[0];remove=known[known.org_cluster==largest].Model.tolist()
        add('M1_drop_largest_organization',main_scope,terms,data=org[~org.Model.isin(remove)])
        org_audit.append(dict(model='M1_drop_largest_organization',organization=largest,removed_n=len(remove),
            removed_Models=json.dumps(remove),known_org_models=len(known),distinct_clusters=len(group_counts),
            restriction='exact consortium strings; overlapping consortium memberships not independent clusters'))
    csv(out/'organization_sensitivity.csv',org_audit)
    for file,rows in [('model_coefficients.csv',coef),('model_performance.csv',evaluation),('model_predictions.csv',prediction),
        ('LOOCV_coefficients.csv',folds),('VIF_diagnostics.csv',vifs),('regression_diagnostics.csv',diag),('influence_diagnostics.csv',influence)]:csv(out/file,rows)
    write_json(out/'model_definitions.json',definitions)
    corrrows=[];g=models['M3_ND']['data']
    for a in ['lnN','lnD','lnC','t_month']:
        for b in ['lnN','lnD','lnC','t_month']:
            corrrows.append(dict(scope=T3,variable_1=a,variable_2=b,Pearson=g[a].corr(g[b]),Spearman=g[a].corr(g[b],method='spearman')))
    csv(out/'scale_correlations.csv',corrrows)
    comparison=[]
    for name in ['M1_linear','M1_chat','M1_S0_sensitivity','M1_same_T3','M1_drop_largest_organization']:
        m=models[name];comparison.append(dict(model=name,n=m['n'],beta_N=m['beta'][1],tau=m['beta'][2],
            sample=m['scope'],meaning='descriptive conditional time trend; not a causal post-training effect'))
    csv(out/'model_type_comparison.csv',comparison)
    trend=make_technology_tables(root,models,d)
    coefdf=pd.DataFrame(coef);perf=pd.DataFrame(evaluation)
    linear=perf[(perf.model=='M1_linear')&(perf.evaluation=='LOOCV')].iloc[0]
    nonlinear_better=[]
    for label in ['M1_quadratic','M1_month_FE']:
        ins=perf[(perf.model==label)&(perf.evaluation=='in_sample')];cv=perf[(perf.model==label)&(perf.evaluation=='LOOCV')]
        base=perf[(perf.model=='M1_linear')&(perf.evaluation=='in_sample')].iloc[0]
        if len(ins) and len(cv) and cv.iloc[0]['full_LOOCV_available'] and cv.iloc[0].RMSE<.95*linear.RMSE and ins.iloc[0].AICc<base.AICc and ins.iloc[0].BIC<base.BIC:
            nonlinear_better.append(label)
    if nonlinear_better:raise ValueError('Nonlinear has material joint improvement; review main model before decomposition')
    write_json(out/'model_selection.json',dict(selected='M1_linear',reason='quadratic lacks joint LOOCV/AICc/BIC improvement; month FE may have unidentifiable singleton-month folds',
        comparison_rule='alternative requires >5% full-LOOCV RMSE reduction and both smaller AICc/BIC',
        AICc_parameter_convention='p mean coefficients including intercept; matches statsmodels OLS AIC convention',
        environment_note='statsmodels.api and statsmodels.stats.diagnostic fail with installed pandas decorator API; direct OLS/RLM imports and independently implemented Koenker BP/Cook/VIF used; no installed package modified'))
    return models


def make_technology_tables(root,models,d):
    """Residualized ability and monthly descriptive frontiers, with actual counts."""
    out=root/'outputs/evolution/phase2';records=[];monthly=[];frontiers=[];trend=[]
    for label in ['M1_linear','M1_chat','M1_S0_sensitivity']:
        m=models[label];g=m['data'].copy();a,b,tau=m['beta']
        g['TechResidual']=g.A_z-(a+b*g.lnN)
        for r in g.to_dict('records'):
            records.append(dict(model=label,Model=r['Model'],scope=m['scope'],A_z=r['A_z'],lnN=r['lnN'],
                t_month=r['t_month'],Submission_Date=str(r['submission_date'].date()),TechResidual=r['TechResidual'],
                non_scale_time_component=tau*r['t_month'],regression_residual=r['TechResidual']-tau*r['t_month']))
        for month,h in g.groupby(g.submission_date.dt.to_period('M').astype(str)):
            vals=h.TechResidual;top=h.sort_values(['A_z','Model'],ascending=[False,True]).head(3)
            monthly.append(dict(model=label,month=month,n=len(h),median_TechResidual=vals.median(),P75=vals.quantile(.75),
                P90=vals.quantile(.9),P95=vals.quantile(.95),median_t=h.t_month.median()))
            if label=='M1_linear':
                frontiers.append(dict(month=month,n=len(h),P90_A=h.A_z.quantile(.9),P95_A=h.A_z.quantile(.95),
                    top3_mean_A=top.A_z.mean(),top3_actual_n=len(top),median_lnN=h.lnN.median(),
                    top3_median_lnN=top.lnN.median(),top3_Models=json.dumps(top.Model.tolist()),median_t=h.t_month.median()))
        f=fit_regression(g.assign(A_z=g.TechResidual),['t_month'],label+'_TechResidual_time',m['scope'])
        trend.extend(f['coefficients'])
    csv(out/'non_scale_technology_index.csv',records);csv(out/'monthly_technology_frontier.csv',monthly)
    csv(out/'monthly_ability_frontier.csv',frontiers);csv(out/'technology_residual_trend.csv',trend)
    # The generated residual trend reuses estimated coefficients: its naive SE
    # is descriptive only; inferential evidence comes from the joint model HC3/bootstrap.
    associations=[]
    for label in ['M1_linear','M1_chat']:
        m=models[label];g=m['data'].copy();g['TechResidual']=g.A_z-(m['beta'][0]+m['beta'][1]*g.lnN)
        g['TechResidual_C8_same_run']=g.C8_A_z_same_run-(m['beta'][0]+m['beta'][1]*g.lnN)
        for family,fields in [('C8',['C8_capability_imbalance','C8_bottom20_task_score','difficulty_adjusted_imbalance','difficulty_adjusted_bottom20']),
            ('architecture',['context_length','n_layers','d_model','n_heads'])]:
            for field in fields:
                h=g.dropna(subset=['TechResidual',field]);x=h.TechResidual;y=h[field]
                valid=len(h)>=3 and x.nunique()>1 and y.nunique()>1
                p=stats.pearsonr(x,y) if valid else None;s=stats.spearmanr(x,y) if valid else None
                associations.append(dict(model=label,analysis=family,variable=field,n=len(h),Pearson=p.statistic if p else np.nan,
                    Pearson_p=p.pvalue if p else np.nan,Spearman=s.statistic if s else np.nan,Spearman_p=s.pvalue if s else np.nan,
                    restriction='exploratory correlation; no causal claim; raw C8 may differ in evaluation version'))
        for field in ['C8_capability_imbalance','difficulty_adjusted_imbalance']:
            h=g.dropna(subset=['TechResidual_C8_same_run',field]);x=h.TechResidual_C8_same_run;y=h[field]
            valid=len(h)>=3 and x.nunique()>1 and y.nunique()>1
            p=stats.pearsonr(x,y) if valid else None;s=stats.spearmanr(x,y) if valid else None
            associations.append(dict(model=label,analysis='C8_same_run_auxiliary',variable=field,n=len(h),
                Pearson=p.statistic if p else np.nan,Pearson_p=p.pvalue if p else np.nan,
                Spearman=s.statistic if s else np.nan,Spearman_p=s.pvalue if s else np.nan,
                restriction='same-run sensitivity with joint-model size coefficient fixed; not a refitted ability index'))
    frame=pd.DataFrame(associations)
    for _,g in frame.groupby(['model','analysis']):
        for pcol in ['Pearson_p','Spearman_p']:
            v=g[pcol].dropna().sort_values();q=np.minimum.accumulate((v.to_numpy()*len(v)/np.arange(1,len(v)+1))[::-1])[::-1]
            frame.loc[v.index,pcol+'_BH_q']=np.clip(q,0,1)
    csv(out/'technology_auxiliary_correlations.csv',frame)
    return records


def run_decomposition(root,cfg,models,metadata):
    """Separate R1/R2/R3; paired bootstrap refits coefficients and endpoint states."""
    out=root/'outputs/evolution/phase2';bounds=metadata['primary_bounds'];fractions=cfg['window_fractions']
    selected=[('R1','M1_linear'),('R2','M2_C'),('R3','M3_ND'),('chat_sensitivity','M1_chat'),('S0_sensitivity','M1_S0_sensitivity'),
        ('R1_release','M1_release'),('R2_release','M2_C_release'),('R3_release','M3_ND_release')]
    selected.extend((label,label) for label in models if label.endswith('_submission_matched'))
    decompositions=[];boot=[];coefs=[]
    for index,(series,label) in enumerate(selected):
        if label not in models:continue
        m=models[label];b=bounds if m['axis']=='submission' else [float(m['data'].t_month.min()),float(m['data'].t_month.max())]
        for frac in fractions:decompositions.append(contribution(m,b,frac,series))
        rows,params=bootstrap(m,b,fractions,series,cfg['bootstrap_repetitions'],cfg['random_state']+index*101)
        boot.extend(rows);coefs.extend(params)
    for label in ['M1_drop_Cook_top1','M1_drop_Cook_top2','M1_Huber','M1_same_T3','M2_C_no_N_conflict','M3_ND_no_N_conflict','M1_drop_largest_organization']:
        if label in models:
            for frac in fractions:decompositions.append(contribution(models[label],bounds,frac,'diagnostic_'+label))
    csv(out/'contribution_decomposition.csv',decompositions);csv(out/'contribution_bootstrap.csv',boot)
    csv(out/'coefficient_bootstrap.csv',coefs);summaries=bootstrap_summary(boot);csv(out/'contribution_bootstrap_summary.csv',summaries)
    params=pd.DataFrame(coefs);coef_summary=[]
    for (label,series,axis),g in params.groupby(['model','series','time_axis']):
        m=models[label]
        for name in m['names']:
            x=g.loc[g.full_rank,name].dropna()
            coef_summary.append(dict(model=label,series=series,time_axis=axis,term=name,draws=len(g),valid_draws=len(x),
                point_coef=m['beta'][m['names'].index(name)],CI_low=x.quantile(.025),CI_high=x.quantile(.975),
                positive_fraction=x.gt(0).mean(),negative_fraction=x.lt(0).mean()))
    csv(out/'coefficient_bootstrap_summary.csv',coef_summary)
    df=pd.DataFrame(decompositions);time_rows=[]
    for s,r,base in [('R1','R1_release','M1'),('R2','R2_release','M2_C'),('R3','R3_release','M3_ND')]:
        for frac in fractions:
            sub=df[(df.series==s)&np.isclose(df.window_fraction,frac)];rel=df[(df.series==r)&np.isclose(df.window_fraction,frac)]
            matched=df[(df.model==base+'_submission_matched')&np.isclose(df.window_fraction,frac)]
            if len(matched):sub=matched
            if len(sub)!=1 or len(rel)!=1:continue
            a=sub.iloc[0];b=rel.iloc[0];gap=abs(a.absolute_share_scale-b.absolute_share_scale)
            reversal=bool(a.Delta_tech*b.Delta_tech<0 or a.Delta_scale*b.Delta_scale<0)
            time_rows.append(dict(series=s,window_fraction=frac,submission_n=a.n,release_n=b.n,
                submission_Delta_scale=a.Delta_scale,release_Delta_scale=b.Delta_scale,
                submission_Delta_tech=a.Delta_tech,release_Delta_tech=b.Delta_tech,
                submission_signed_ratio_scale=a.signed_ratio_scale,release_signed_ratio_scale=b.signed_ratio_scale,
                submission_signed_ratio_tech=a.signed_ratio_tech,release_signed_ratio_tech=b.signed_ratio_tech,
                submission_share_scale=a.share_scale,release_share_scale=b.share_scale,
                submission_share_tech=a.share_tech,release_share_tech=b.share_tech,
                submission_absolute_share_scale=a.absolute_share_scale,release_absolute_share_scale=b.absolute_share_scale,
                submission_absolute_share_tech=a.absolute_share_tech,release_absolute_share_tech=b.absolute_share_tech,
                absolute_share_difference=gap,contribution_sign_reversal=reversal,
                time_axis_sensitive=bool(reversal or gap>cfg['time_axis_absolute_share_difference_warning']),
                definition='same models; submission windows on main calendar; release windows on observed release range; chronological orders differ'))
    csv(out/'time_axis_sensitivity.csv',time_rows)
    boot_status=pd.DataFrame(boot).groupby(['model','time_axis','window_fraction','status']).size().reset_index(name='draws')
    csv(out/'bootstrap_status_audit.csv',boot_status)
    logging.info('Contribution bootstrap complete: %d rows; models=%d',len(boot),len(selected));return df


def input_manifest(root):
    """Hash precisely consumed frozen tables and the original dedup/C7 sources."""
    paths=[root/'outputs/evolution/phase1'/f for f in ['evolution_model_table.csv','ability_transform_parameters.json','bridge_model.json','bridge_validation_metrics.csv','C8_execution_summary.json']]
    paths+=[root/'real_attachments/C_efficiency_evolution'/f for f in ['leaderboard_enhanced.csv','model_architecture_metadata.csv']]
    return [dict(file=p.relative_to(root).as_posix(),size=p.stat().st_size,
        sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in paths]

# ============================================================
# Source: src/evolution/contribution_reporting.py
# Original lines: 1-222
# Purpose: 最终CSV/图表/报告呈现，compact省略纯绘图和排版
# ============================================================
"""Plots and a factual Chinese report for the frozen-input contribution study."""
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import csv,table
from .contribution_analysis import input_manifest,write_json


def read(out,name):
    """Preserve full round-trip precision in numerical outputs."""
    return pd.read_csv(out/name,float_precision='round_trip')


def make_figures(root,models):
    """Produce descriptive and diagnostic figures; never project future time."""
    out=root/'outputs/evolution/phase2';figdir=root/'outputs/figures/question4_phase2';figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':160})
    manifest=[]
    def save(fig,name,source):
        fig.tight_layout();p=figdir/name;fig.savefig(p,bbox_inches='tight');plt.close(fig)
        manifest.append(dict(file=p.relative_to(root).as_posix(),source=source,bytes=p.stat().st_size))
    m=models['M1_linear'];g=m['data'];b=np.asarray(m['beta'])
    fig,ax=plt.subplots(figsize=(8,5));sc=ax.scatter(g.lnN,g.A_z,c=g.t_month,cmap='viridis',s=35,alpha=.8)
    fig.colorbar(sc,ax=ax,label='Months since main submission origin');ax.set(xlabel='ln(parameters in billions)',ylabel='Frozen A_z',title=f'S1 pretrained: ability and parameter scale (n={len(g)})')
    save(fig,'01_ability_vs_parameter_scale.png','analysis_model_table.csv')
    residual=g.A_z-(b[0]+b[1]*g.lnN);grid=np.linspace(g.t_month.min(),g.t_month.max(),100)
    fig,ax=plt.subplots(figsize=(8,5));ax.scatter(g.t_month,residual,s=35,alpha=.7,label='Individual models')
    ax.plot(grid,b[2]*grid,color='tab:red',label='Joint-model time component');ax.set(xlabel='Submission time (months)',ylabel='TechResidual',title='Ability after removing fitted parameter-scale component');ax.legend()
    save(fig,'02_technology_residual_vs_time.png','non_scale_technology_index.csv')
    fig,ax=plt.subplots(figsize=(8,5))
    for label,color in [('M1_linear','tab:blue'),('M1_chat','tab:orange')]:
        h=models[label]['data'];beta=np.asarray(models[label]['beta']);x=np.linspace(h.t_month.min(),h.t_month.max(),100)
        y=beta[0]+beta[1]*h.lnN.median()+beta[2]*x
        ax.plot(x,y,color=color,label=f'{label}, tau={beta[2]:.4f}/month')
    ax.set(xlabel='Submission time (months)',ylabel='Fitted A_z at each group median lnN',title='Separate pretrained and chat trends; descriptive comparison');ax.legend()
    save(fig,'03_pretrained_chat_time_trends.png','model_type_comparison.csv')
    preds=read(out,'model_predictions.csv');perf=read(out,'model_performance.csv')
    for label,num in [('M1_linear','04'),('M2_C','05')]:
        h=preds[(preds.model==label)&preds.method.eq('OLS')];cv=perf[(perf.model==label)&perf.evaluation.eq('LOOCV')].iloc[0]
        fig,ax=plt.subplots(figsize=(6,5));ax.scatter(h.observed_A,h.fitted_A,alpha=.7,label='In-sample fit')
        ax.scatter(h.observed_A,h.LOOCV_predicted_A,marker='x',color='tab:orange',label='LOOCV')
        vals=np.r_[h.observed_A,h.fitted_A,h.LOOCV_predicted_A];lo,hi=np.nanmin(vals),np.nanmax(vals);ax.plot([lo,hi],[lo,hi],'--',color='gray')
        ax.set(xlabel='Observed frozen A_z',ylabel='Predicted A_z',title=f'{label}: n={len(h)}, LOOCV RMSE={cv.RMSE:.3f}, R2={cv.R2:.3f}');ax.legend()
        save(fig,num+'_observed_vs_predicted.png','model_predictions.csv')
    dec=read(out,'contribution_decomposition.csv');point=dec[dec.series.isin(['R1','R2','R3'])&np.isclose(dec.window_fraction,1/3)]
    fig,axes=plt.subplots(1,3,figsize=(13,4.8))
    for ax,(_,r) in zip(axes,point.iterrows()):
        values=[r.A_early_pred,r.Delta_scale,r.Delta_tech,r.A_late_pred];bottom=[0,r.A_early_pred,r.A_early_pred+r.Delta_scale,0]
        colors=['gray','tab:blue','tab:orange','gray'];ax.bar(range(4),values,bottom=bottom,color=colors,alpha=.8)
        for j,v in enumerate(values):ax.annotate(f'{v:+.3f}',(j,bottom[j]+v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
        ax.axhline(0,color='gray',lw=.7);ax.set_xticks(range(4),['Early','Scale','Time','Late']);ax.set(title=r.series+' (signed components)',ylabel='A_z / change in A_z')
    fig.suptitle('Early to late: predicted endpoint-state change, not observed causal attribution',y=1.03)
    save(fig,'06_endpoint_shapley_waterfall.png','contribution_decomposition.csv')
    boot=read(out,'contribution_bootstrap.csv');fig,axes=plt.subplots(2,3,figsize=(13,8.5))
    for j,s in enumerate(['R1','R2','R3']):
        ax=axes[0,j]
        h=boot[(boot.series==s)&np.isclose(boot.window_fraction,1/3)&boot.status.eq('OK')]
        vals=np.r_[h.signed_ratio_scale.dropna(),h.signed_ratio_tech.dropna()];lo,hi=np.quantile(vals,[.01,.99])
        if hi<=lo:lo-=.1;hi+=.1
        for field,label,color in [('signed_ratio_scale','Scale','tab:blue'),('signed_ratio_tech','Time','tab:orange')]:
            v=h[field].dropna();ax.hist(v,bins=np.linspace(lo,hi,45),alpha=.5,label=label,color=color)
        ax.axvline(0,color='gray',lw=.8);ax.axvline(1,color='gray',lw=.8,ls='--')
        ax.set(xlabel='Signed component / predicted change',ylabel='Bootstrap draws',title=s+'; plot P01-P99 only');ax.legend()
        bottom=axes[1,j]
        for field,label,color in [('absolute_share_scale','Scale','tab:blue'),('absolute_share_tech','Time','tab:orange')]:
            bottom.hist(h[field].dropna(),bins=np.linspace(0,1,35),alpha=.5,label=label,color=color)
        bottom.set(xlabel='Absolute component / sum of absolute components',ylabel='Bootstrap draws',title=s+' absolute shares; opposing signs allowed');bottom.legend()
    save(fig,'07_bootstrap_contribution_ratios.png','contribution_bootstrap.csv; full tails retained in CSV')
    frontier=read(out,'monthly_ability_frontier.csv');x=np.arange(len(frontier));fig,ax=plt.subplots(figsize=(9,5))
    for col,label in [('P90_A','P90'),('P95_A','P95'),('top3_mean_A','Top3 mean (up to 3)')]:ax.plot(x,frontier[col],'-o',label=label)
    ax.set_xticks(x,frontier.month,rotation=45,ha='right');ax.set(ylabel='Frozen A_z',title='S1 pretrained submission cohorts: descriptive ability frontier')
    for i,r in frontier.iterrows():ax.annotate(f'n={r.n}',(i,r.P95_A),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    ax.legend(loc='upper right');save(fig,'08_monthly_ability_frontier.png','monthly_ability_frontier.csv')
    tech=read(out,'monthly_technology_frontier.csv');h=tech[tech.model=='M1_linear'];x=np.arange(len(h));fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(x,h.median_TechResidual,'-o',label='P50');ax.plot(x,h.P90,'-o',label='P90')
    ax.set_xticks(x,h.month,rotation=45,ha='right');ax.set(ylabel='TechResidual',title='Monthly parameter-adjusted ability; sparse months remain visible');ax.legend()
    save(fig,'09_monthly_technology_index.png','monthly_technology_frontier.csv')
    time=read(out,'time_axis_sensitivity.csv');h=time[np.isclose(time.window_fraction,1/3)];x=np.arange(len(h));fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(x-.18,h.submission_absolute_share_scale,width=.36,label='Submission');ax.bar(x+.18,h.release_absolute_share_scale,width=.36,label='Release')
    ax.set_xticks(x,h.series);ax.set(ylim=(0,1.15),ylabel='Absolute scale share (opposing signs allowed)',title='Same-model time-axis sensitivity; endpoint chronology differs');ax.legend()
    for j,r in enumerate(h.itertuples()):ax.text(j,1.04,'sensitive' if r.time_axis_sensitive else 'within threshold',ha='center',fontsize=9)
    save(fig,'10_submission_release_sensitivity.png','time_axis_sensitivity.csv')
    h=preds[(preds.model=='M1_linear')&preds.method.eq('OLS')];fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    axes[0].scatter(h.fitted_A,h.residual,alpha=.7);axes[0].axhline(0,color='gray',ls='--');axes[0].set(xlabel='Fitted A_z',ylabel='Residual',title='M1 residual pattern')
    stats.probplot(h.residual,dist='norm',plot=axes[1]);axes[1].set_title('M1 residual normal Q-Q (diagnostic)')
    save(fig,'11_M1_residual_diagnostics.png','model_predictions.csv')
    influence=read(out,'influence_diagnostics.csv');h=influence[influence.model.eq('M1_linear')].sort_values(['Cooks_distance','Model'],ascending=[False,True]);x=np.arange(len(h));fig,ax=plt.subplots(figsize=(9,5))
    ax.stem(x,h.Cooks_distance,basefmt=' ');ax.axhline(4/len(h),ls='--',color='tab:red',label='4/n diagnostic threshold')
    for j,r in enumerate(h.head(2).itertuples()):ax.annotate(r.Model.split('/')[-1],(j,r.Cooks_distance),xytext=(8,2-j*15),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Models ordered by Cook distance',ylabel="Cook's distance",title='Influence diagnosis; main fit retains all models');ax.legend()
    save(fig,'12_M1_influence_diagnostics.png','influence_diagnostics.csv');csv(out/'figure_manifest.csv',manifest)
    return manifest


def finalize(root,cfg,models,d,metadata):
    """Report both positive associations and failed/stable sensitivity evidence."""
    out=root/'outputs/evolution/phase2';report=root/'outputs/report/question4_phase2_contribution_decomposition.md'
    figures=make_figures(root,models)
    coeff=read(out,'model_coefficients.csv');perf=read(out,'model_performance.csv');dec=read(out,'contribution_decomposition.csv')
    tiers=read(out,'contribution_sample_tiers.csv');boot=read(out,'contribution_bootstrap_summary.csv');cb=read(out,'coefficient_bootstrap_summary.csv')
    aux=read(out,'technology_auxiliary_correlations.csv');vif=read(out,'VIF_diagnostics.csv');diag=read(out,'regression_diagnostics.csv')
    time=read(out,'time_axis_sensitivity.csv');influence=read(out,'influence_diagnostics.csv');folds=read(out,'LOOCV_coefficients.csv')
    bridge=json.loads((root/'outputs/evolution/phase1/bridge_model.json').read_text(encoding='utf8'))
    if bridge['bridge_valid_for_conversion']:raise ValueError('Frozen negative bridge evidence changed unexpectedly')
    bridge_metrics=pd.read_csv(root/'outputs/evolution/phase1/bridge_validation_metrics.csv')
    bridge_text='\n\n'.join(['# Loss–Benchmark负面证据（PHASE 1冻结）',
        f"High仅{bridge['High_count']}条、均为Pythia；主拟合只用High，Medium={bridge['Medium_count']}仅作外验。",
        table(bridge_metrics[['candidate','evaluation','n','RMSE','MAE','R2','Pearson','Spearman']]),
        f"单调模型退化常数A={bridge['selected_model']['y'][0]:.12g}；三个候选High CV R²均为负，Medium外验失败。",
        '`bridge_valid_for_conversion=False`保持冻结。问题一至三Loss不批量转换为A_z，不参与本阶段贡献计算。',
        '本阶段不重新开发桥接、不做未来预测。后续Benchmark前沿需独立基于可比排行榜记录；前三问只提供资源/算力情景。'])
    (out/'bridge_negative_evidence.md').write_text(bridge_text,encoding='utf8')
    primary=dec[dec.series.isin(['R1','R2','R3'])&np.isclose(dec.window_fraction,1/3)]
    maincoef=coeff[coeff.model.isin(['M1_linear','M2_C','M3_ND'])&coeff.inference.eq('HC3')]
    mainperf=perf[perf.model.isin(['M1_linear','M2_C','M3_ND'])]
    meaningful=['Delta_scale','Delta_tech','signed_ratio_scale','signed_ratio_tech','absolute_share_scale','absolute_share_tech']
    mainboot=boot[boot.series.isin(['R1','R2','R3'])&np.isclose(boot.window_fraction,1/3)&boot.quantity.isin(meaningful)]
    comparison=coeff[coeff.model.isin(['M1_linear','M1_Huber','M1_chat','M1_S0_sensitivity','M1_drop_Cook_top1','M1_drop_Cook_top2','M1_drop_largest_organization','M1_organization_known'])&coeff.term.eq('t_month')&coeff.inference.isin(['HC3','Huber_H3','organization_clustered'])]
    tc=maincoef[(maincoef.model=='M1_linear')&(maincoef.term=='t_month')].iloc[0]
    org=coeff[(coeff.inference=='organization_clustered')&(coeff.term=='t_month')].iloc[0]
    S0=coeff[(coeff.model=='M1_S0_sensitivity')&(coeff.inference=='HC3')&(coeff.term=='t_month')].iloc[0]
    deleted=coeff[(coeff.model=='M1_drop_largest_organization')&(coeff.inference=='HC3')&(coeff.term=='t_month')].iloc[0]
    tech_boot=cb[(cb.model=='M1_linear')&(cb.term=='t_month')].iloc[0]
    conclusion=(f"M1的控制参数时间项tau={tc.coef:.6g}/月，HC3 p={tc.p_value:.6g}，bootstrap 95%区间[{tech_boot.CI_low:.6g},{tech_boot.CI_high:.6g}]。"
        f"但组织聚类p={org.p_value:.6g}，去除最大组织p={deleted.p_value:.6g}，S0敏感性p={S0.p_value:.6g}。"
        '因此能支持特定S1样本内的条件时间关联；证据具有样本选择和组织依赖，不能宣称普遍稳定或严格因果的非规模技术进步。')
    readiness=dict(main_M1_models=models['M1_linear']['n'],restricted_M2_M3_models=models['M2_C']['n'],
        conditional_time_association_HC3_positive_and_significant=bool(tc.coef>0 and tc.p_value<.05),
        robust_general_non_scale_advance_established=False,
        organization_cluster_time_p=org.p_value,S0_time_p=S0.p_value,
        phase3_started=False,future_prediction_performed=False,loss_bridge_used=False,
        phase3_readiness='conditional descriptive benchmark frontier sensitivity only after human confirmation; no reliable 12/24 month extrapolation established',
        blockers=['ten submission cohorts, several singleton months','snapshot evaluation versus actual historical frontier',
            'sample/organization dependence of time trend','small complete-scale sample with only two late-window models',
            'Release changes ordering and endpoints','frozen Loss bridge unusable'])
    write_json(out/'phase3_readiness.json',readiness)
    selection=json.loads((out/'model_selection.json').read_text(encoding='utf8'))
    monthly=read(out,'monthly_ability_frontier.csv');status=read(out,'bootstrap_status_audit.csv')
    signatures=[]
    for label in ['M1_linear','M2_C','M3_ND']:
        h=folds[(folds.model==label)&folds.method.eq('OLS')&folds.full_rank]
        names=models[label]['names'];arr=np.array([json.loads(v) for v in h.parameters])
        for j,name in enumerate(names):signatures.append(dict(model=label,term=name,folds=len(h),positive_fraction=float((arr[:,j]>0).mean()),
            negative_fraction=float((arr[:,j]<0).mean()),min_coef=arr[:,j].min(),max_coef=arr[:,j].max()))
    csv(out/'LOOCV_sign_stability.csv',signatures)
    sections=['# 问题四 PHASE 2：规模扩张与非规模技术进步贡献分解',
        '本阶段读取已人工冻结的PHASE 1能力、模型身份、分类和尺度元数据。A_z和主S1定义不变；没有重拟合前三问、没有重构能力指数、没有Loss数值桥接、没有未来12/24个月预测。',
        '## 1 样本、去重与时间口径',
        '主分析为confirmed open-weight S1中的pretrained：后训练/对齐混杂较少，但仍有模型族、训练数据、架构和评测选择等混杂。chat_finetuned单独作为敏感性，other_unknown不进入主贡献比例。S0仅检查开放权重确认/匹配缺失的样本选择；S2授权证据仍未建立。',
        table(tiers[tiers['sample']=='S1'][['model_type','tier','n','parameter_conflict_n','release_date_n']]),
        '每个Model一行，完全沿用冻结的最新Submission Date集合及同日确定性规则。逐模型重新对照C2核验；未择优选最高A。dedup_audit.csv保留所有选中与旧记录行号，同日重复评测采用PHASE 1已冻结的逐Benchmark中位数。',
        f"主时间起点{metadata['primary_origin']}、末日{metadata['primary_end']}；t_month=实际经过天数/{metadata['days_per_month']}，连续月数，不用整数年份。Release以受限样本的实际发布日计算，另作敏感性。",
        '按用户本阶段T0–T3定义统计，T3为21，并非预设20。保留1条PHASE 1参数冲突记录；主结果不自动删除，另列无冲突20模型敏感性。缺N/D/compute不插值，compute只使用公开C4字段，不由6ND或前三问反算。',
        '## 2 模型、估计及线性时间选择',
        r'$M1:A=a+\beta_N\ln N+\tau t+\epsilon$；$M2\text{-}C:A=a+\beta_C\ln C_{train}+\tau_Ct+\epsilon$；$M3:A=a+\beta_N\ln N+\beta_D\ln D+\tau_{ND}t+\epsilon$。M2/M3在同一T3样本上拟合，不同时加入lnC、lnN和lnD。',
        '时间项统一称“控制规模后的非规模技术进步代理项”，不是可识别的因果技术效应。lnN是参数规模，lnC是整体训练规模代理，lnN+lnD是结构敏感性；R1/R2/R3不能合并成一个贡献数字。尤其M1只控制参数量，时间项仍可能吸收未观察的训练数据量/训练算力扩张，不能称为排除全部规模因素后的纯技术效应。',
        table(maincoef[['model','term','n','coef','SE','CI_low','CI_high','p_value']]),
        'HC3主诊断使用t参考分布；OLS常规SE/CI/p亦完整保存。HuberT为稳健敏感性。实现依据[statsmodels稳健协方差文档](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html)及[Huber示例](https://www.statsmodels.org/stable/examples/notebooks/generated/robust_models_0.html)。',
        table(mainperf[['model','evaluation','n','R2','adjusted_R2','RMSE','MAE','Pearson','Spearman']]),
        table(perf[perf.model.isin(['M1_linear','M1_quadratic','M1_month_FE'])][['model','evaluation','n','RMSE','AICc','BIC','unidentifiable_folds']]),
        selection['reason']+'。月固定效应有4个单样本月份，留出该模型即无法估计该月效应；没有把最小范数外推或局部62折成绩冒充完整LOOCV。HC3因杠杆值1而退化，SE/CI/p均明确记NaN及UNIDENTIFIABLE_LEVERAGE_ONE，不能使用库产生的inf标准误及p=1进行推断。',
        '## 3 稳健性、共线性与影响点',
        table(diag[diag.model.isin(['M1_linear','M2_C','M3_ND'])][['model','condition_number_raw','condition_number_standardized','Koenker_BP_p','Jarque_Bera','JB_p','Cook_exceed_count','max_Cook_Model']]),
        table(vif[vif.model.isin(['M1_linear','M2_C','M3_ND'])][['model','term','VIF']]),
        '报告原始与中心化尺度一致后的condition number，避免把lnD/lnC单位导致的大截距误判为强共线性。M3本次VIF未达到配置警戒值，但小样本、bootstrap和留一系数符号不稳仍会限制贡献解释。Cook>4/n仅诊断；主模型保留全部记录，删除最高1/2条只是规定敏感性。',
        table(influence[influence.model=='M1_linear'].sort_values('Cooks_distance',ascending=False).head(5)[['Model','Cooks_distance','leverage','residual']]),
        table(comparison[['model','inference','n','coef','CI_low','CI_high','p_value']]),
        table(cb[cb.model.isin(['M1_linear','M2_C','M3_ND'])][['model','term','valid_draws','CI_low','CI_high','positive_fraction']]),
        table(pd.DataFrame(signatures)),
        '组织按附件实际完整组织/合作团体字符串聚类，不按模型名猜实验室；多组织合作团体可能共享成员，聚类独立性仍有限。没有加入大量组织虚拟变量。S1主样本65条具有组织记录，另1条未知保留在主拟合。',
        '## 4 Early/Late状态与Shapley贡献',
        '主窗口为主Submission日历区间前1/3与后1/3；敏感性为前25%与后25%。中段参与拟合；N状态取median lnN，D/C按要求先取原始token/FLOP的中位数再取对数，t取中位数，不按模型个数三等分。原始N/D/C中位数也保存在端点审计。此为典型状态预测变化，不等于观察到的中位能力变化或最大能力前沿变化。',
        r'$\Delta_{scale}=\tfrac12[f(S_1,t_0)-f(S_0,t_0)+f(S_1,t_1)-f(S_0,t_1)]$；$\Delta_{tech}=\tfrac12[f(S_0,t_1)-f(S_0,t_0)+f(S_1,t_1)-f(S_1,t_0)]$。所有可识别计算验证$|\Delta_{pred}-\Delta_{scale}-\Delta_{tech}|<10^{-10}$。加性线性模型没有交互，Shapley退化为相应系数乘状态差；公式仍完整执行。',
        table(primary[['series','n','early_n','late_n','Delta_scale','Delta_tech','Delta_pred','share_scale','share_tech','absolute_share_scale','absolute_share_tech','cancellation_effect']]),
        table(dec[dec.series.isin(['R1','R2','R3'])&np.isclose(dec.window_fraction,.25)][['series','early_n','late_n','Delta_scale','Delta_tech','Delta_pred','share_scale','absolute_share_scale']]),
        '只有预测增长>0且两项均非负时share_scale/share_tech才是普通0–100%增长比例。若一项负值，普通share留NaN，保留signed contribution、signed_ratio和absolute share，并标明抵消效应。signed_ratio的区间可以越出0/1；它不是合法非负贡献百分比。没有截断负贡献。',
        'T3晚期仅2个模型，都是2023年发布、2025年提交的Pythia。数据上“晚提交”不等于“新技术”；两个端点构成对时间口径和典型状态尤其敏感。R2/R3均为restricted-sample evidence，不是全体开源模型的比例。',
        '## 5 模型级bootstrap及贡献区间',
        f"每套方案执行{cfg['bootstrap_repetitions']}次模型级有放回抽样，每次重新拟合、重新计算抽中模型的端点中位数并重新Shapley；固定原始日历边界。主R1/R2/R3、chat、S0和三套Release敏感性均执行，固定随机种子。",
        table(mainboot[['series','quantity','draws','valid_decomposition_draws','valid_quantity_draws','CI_low','CI_high','ratio_CI_outside_zero_one']]),
        table(status[status.model.isin(['M1_linear','M2_C','M3_ND'])]),
        '当重抽样导致窗口为空或设计矩阵降秩，保留该次抽样及状态，不补模型、不悄悄重抽直到成功。贡献CI以可识别draw计算，明确是条件区间；普通非负share的CI还额外条件于非负增长。完整signed_ratio区间用于暴露抵消及分母接近零的不稳定性。模型级bootstrap不能处理族内依赖，组织敏感性另行保留。',
        '## 6 模型类型、S0与时间口径敏感性',
        table(read(out,'model_type_comparison.csv')),
        'pretrained与chat的tau差只描述分组趋势强弱，不等于某种对齐方法的因果效应。S0扩大样本改变tau，说明confirmed-open匹配/元数据选择不可忽略。',
        table(time),
        f"time_axis_sensitive当贡献符号反转，或absolute scale share差>{cfg['time_axis_absolute_share_difference_warning']:.0%}时标记；该阈值是透明诊断约定，不是统计显著性检验。Release只纳入PHASE 1接受的C4身份且有发布日期的模型；Submission对照使用完全相同的模型，模型数及端点变化完整列出。实际时间范围和排序改变，因而窗口模型组成也改变。",
        '## 7 技术残差、月度前沿、C8及架构线索',
        r'$TechResidual=A-[\hat a+\hat\beta_N\ln N]$。按月保存P50/P75/P90/P95并做时间描述；辅助残差~time回归使用生成响应，标准误不作为独立证据，主要推断仍来自联合M1的HC3/bootstrap。',
        table(monthly[['month','n','P90_A','P95_A','top3_mean_A','top3_actual_n','median_lnN','top3_median_lnN']]),
        '不足3模型的月份top3_mean只取实际可用模型并明确实际n。月度P90/P95明显起伏，未呈稳定单调提升，不能将个体回归的正时间项直接等同于前沿持续改善。多个月份n=1，无法从10个队列前沿点建立稳定因果贡献或长期预测。当前快照可能回溯重评，Submission cohort不是已证实的当时能力前沿。',
        table(aux[['model','analysis','variable','n','Pearson','Pearson_p_BH_q','Spearman','Spearman_p_BH_q']]),
        '原始C8偏科同时受任务难度、任务数量及测评版本影响，补充difficulty-adjusted与same-run对照。相关弱也不能证明偏科没有作用；若相关强，技术残差不能排除任务结构影响。C7只用单变量架构相关；未建立多变量架构回归，BH q只作探索性多重检验提示。',
        '## 8 结论、负面桥接与停止点',conclusion,
        'C6桥接负面证据单独保存bridge_negative_evidence.md：High仅7条同族，CV R²均负，单调退化常数，Medium外验失败。固定bridge_valid_for_conversion=False，前三问Loss不转换为A_z、不参与贡献比例。',
        'PHASE 3仅具备在人工确认后开展受限、描述性Benchmark前沿敏感性分析的输入准备；未建立可靠12/24个月外推条件。需要明确历史快照/真实评测时点、稀疏月份、族/组织依赖和可比数据支持。到此停止，不拟合最终前沿、不预测未来、不做因果结论。',
        '## 附录：复现与证据文件',
        '`python -X utf8 -B scripts/21_build_evolution_contribution.py --stage all`；支持samples/models/bootstrap/report分阶段。`python -X utf8 -B scripts/21_verify_evolution_contribution.py`独立复核。新配置、代码和输出均独立，冻结文件按SHA256核验。']
    for p in sorted(out.glob('*.csv')):sections.append(f'[{p.name}](../evolution/phase2/{p.name})')
    sections.append('[phase3_readiness.json](../evolution/phase2/phase3_readiness.json)')
    sections.append('[bridge_negative_evidence.md](../evolution/phase2/bridge_negative_evidence.md)')
    sections.append('[verification_summary.json](../evolution/phase2/verification_summary.json)')
    for r in figures:sections.append(f"![{Path(r['file']).name}](../figures/question4_phase2/{Path(r['file']).name})")
    report.write_text('\n\n'.join(sections),encoding='utf8')
    csv(out/'input_manifest.csv',input_manifest(root))
    code=[root/'configs/evolution_phase2.yaml',root/'scripts/21_build_evolution_contribution.py',root/'scripts/21_verify_evolution_contribution.py']
    code+=sorted((root/'src/evolution').glob('contribution_*.py'))
    csv(out/'implementation_manifest.csv',[dict(file=p.relative_to(root).as_posix(),size=p.stat().st_size,
        sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in code if p.is_file()])
    write_json(out/'execution_summary.json',dict(phase='Question 4 PHASE 2',main_model='M1_linear',
        bootstrap_replicates_per_model=cfg['bootstrap_repetitions'],figures=len(figures),
        frozen_ability_refitted=False,previous_questions_modified=False,bridge_refitted=False,loss_conversion_performed=False,
        future_prediction_performed=False,final_frontier_fitted=False,causal_conclusion=False,waiting_for_human_confirmation=True,
        readiness=readiness))
    logging.info('PHASE 2 report and %d figures complete; awaiting human confirmation',len(figures))

# ============================================================
# Source: scripts/21_build_evolution_contribution.py
# Original lines: 1-61
# Purpose: 最终阶段实际执行入口：21_build_evolution_contribution.py
# ============================================================
"""Question 4 PHASE 2: contribution analysis only; no ability/bridge refitting."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/'src'))
import argparse
import json
import logging
import pandas as pd
import yaml
from evolution.contribution_analysis import prepare,fit_models,run_decomposition,model_data
from evolution.contribution_regression import design


def load_models(root):
    """Recover this phase's cached estimates, retaining frozen input model IDs."""
    out=root/'outputs/evolution/phase2';d=pd.read_csv(out/'analysis_model_table.csv',float_precision='round_trip')
    for col in ['submission_date','release_date']:d[col]=pd.to_datetime(d[col],errors='coerce')
    definitions=json.loads((out/'model_definitions.json').read_text(encoding='utf8'));models={}
    for item in definitions:
        g=d.set_index('Model',drop=False).loc[item['Models']].reset_index(drop=True)
        g['t_month']=g['t_release' if item['time_axis']=='release' else 't_submission']
        if 't_squared' in item['terms']:g['t_squared']=g.t_month**2
        if any(t.startswith('month_') for t in item['terms']):
            dummy=pd.get_dummies(g.submission_date.dt.to_period('M').astype(str),prefix='month',drop_first=True,dtype=float)
            g=pd.concat([g,dummy],axis=1)
        terms=item['terms'];models[item['model']]=dict(label=item['model'],scope=item['scope'],axis=item['time_axis'],
            method=item['method'],terms=terms,names=['intercept']+terms,n=len(g),data=g,
            X=design(g,terms),y=g.A_z.to_numpy(float),beta=item['beta'])
    metadata=json.loads((out/'sample_definition.json').read_text(encoding='utf8'))
    return models,d,metadata


def main():
    """Checkpoint samples, models, bootstrap and report without touching Phase 1."""
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['samples','models','bootstrap','report','all'],default='all');args=p.parse_args()
    out=PROJECT_ROOT/'outputs/evolution/phase2';out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(out/'execution.log',encoding='utf8')])
    cfg=yaml.safe_load((PROJECT_ROOT/'configs/evolution_phase2.yaml').read_text(encoding='utf8'))
    if cfg['future_prediction_enabled'] or cfg['loss_conversion_enabled']:raise ValueError('Outside authorized Phase 2 scope')
    if not (out/'frozen_before_question4_phase2.json').exists():raise ValueError('Freeze snapshot required before stage execution')
    if args.stage in ['samples','models','all']:
        d,metadata=prepare(PROJECT_ROOT,cfg)
        if args.stage in ['models','all']:models=fit_models(PROJECT_ROOT,cfg,d,metadata)
    if args.stage in ['bootstrap','all']:
        if args.stage=='bootstrap':models,d,metadata=load_models(PROJECT_ROOT)
        run_decomposition(PROJECT_ROOT,cfg,models,metadata)
    if args.stage in ['report','all']:
        if args.stage=='report':models,d,metadata=load_models(PROJECT_ROOT)
        from evolution.contribution_reporting import finalize
        finalize(PROJECT_ROOT,cfg,models,d,metadata)
    logging.info('Question 4 PHASE 2 %s complete; no future forecasting or Loss conversion',args.stage)


if __name__=='__main__':
    try:main()
    except Exception:
        logging.exception('Question 4 PHASE 2 stopped on an error; frozen inputs remain read-only')
        raise

# ============================================================
# Source: src/evolution/projection_data.py
# Original lines: 1-193
# Purpose: 实际执行Phase3：锚点、C4去重滚动前沿、增长bootstrap
# ============================================================
"""Frozen ability anchors, dated compute-frontier growth, exhaustive bootstrap."""
import json
import logging
import warnings
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from statsmodels.regression.quantile_regression import QuantReg
from .common import csv
from .contribution_analysis import write_json


SCOPES={'F0_pretrained':'pretrained','F1_chat':'chat_finetuned','F2_overall':None}


def load_frozen(root):
    """Read only approved Phase 1/2 observations; no older Q4 drafts."""
    d=pd.read_csv(root/'outputs/evolution/phase1/evolution_model_table.csv',float_precision='round_trip')
    d['date']=pd.to_datetime(d['Submission Date'],errors='coerce');d['release']=pd.to_datetime(d['Release Date'],errors='coerce')
    d=d[d.S1&d.A_z.notna()&np.isfinite(d.A_z)&d.date.notna()].copy()
    if not d.Model.is_unique:raise ValueError('Frozen S1 is not per-model unique')
    return d.reset_index(drop=True)


def select_scope(d,scope):
    """Keep pretrained, chat and overall distinct."""
    kind=SCOPES[scope];return d if kind is None else d[d.model_type.eq(kind)]


def frontier_stats(values):
    """P95 main; P90 and top3 sensitivity; maximum only auxiliary."""
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if not len(x):return dict(P95=np.nan,P90=np.nan,Top3=np.nan,maximum=np.nan)
    return dict(P95=float(np.quantile(x,.95)),P90=float(np.quantile(x,.9)),Top3=float(np.sort(x)[-min(3,len(x)):].mean()),maximum=float(x.max()))


def build_anchors(root,cfg,d):
    """Extend 90→120→180 solely by sample count; never optimize frontier level."""
    out=root/'outputs/evolution/phase3';base=d.date.max();rows=[];attempts=[];boot=[];members=[]
    for k,scope in enumerate(SCOPES):
        g=select_scope(d,scope)
        for days in cfg['anchor_windows_days']:
            h=g[g.date.gt(base-pd.Timedelta(days=days))&g.date.le(base)]
            attempts.append(dict(model_scope=scope,window_days=days,n=len(h),minimum_met=len(h)>=cfg['anchor_minimum_models']))
            if len(h)>=cfg['anchor_minimum_models']:break
        if h.empty:raise ValueError('No usable anchor models: '+scope)
        x=h.A_z.to_numpy();rng=np.random.default_rng(cfg['random_state']+k*37)
        for i in range(cfg['bootstrap_repetitions']):
            ix=rng.integers(0,len(x),len(x));vals=frontier_stats(x[ix])
            boot.append(dict(model_scope=scope,replicate=i+1,source_model_indices=json.dumps(ix.tolist()),**vals))
        vals=frontier_stats(x)
        for definition in ['P95','Top3','P90']:
            rows.append(dict(model_scope=scope,frontier_definition=definition,base_date=str(base.date()),window_days=days,
                window_start=str((base-pd.Timedelta(days=days)).date()),window_end=str(base.date()),left_endpoint_open=True,
                n=len(h),minimum_models=cfg['anchor_minimum_models'],anchor_below_minimum=len(h)<cfg['anchor_minimum_models'],
                A_z_anchor=vals[definition],top3_actual_n=min(3,len(h)),latest_model_date=str(h.date.max().date())))
        for r in h.to_dict('records'):members.append(dict(model_scope=scope,Model=r['Model'],date=str(r['date'].date()),A_z=r['A_z'],
            C8_capability_imbalance=r['C8_capability_imbalance'],C8_bottom20_task_score=r['C8_bottom20_task_score']))
    csv(out/'frontier_anchor.csv',rows);csv(out/'anchor_window_audit.csv',attempts);csv(out/'anchor_bootstrap.csv',boot);csv(out/'anchor_models.csv',members)
    # Known high-imbalance records are excluded; missing C8 is retained as unknown.
    pre=select_scope(d,'F0_pretrained');known=pre.C8_capability_imbalance.dropna();threshold=known.quantile(.9)
    chosen=[r for r in rows if r['model_scope']=='F0_pretrained' and r['frontier_definition']=='P95'][0]
    h=pre[pre.date.gt(base-pd.Timedelta(days=chosen['window_days']))];filtered=h[h.C8_capability_imbalance.isna()|h.C8_capability_imbalance.le(threshold)]
    audit=[]
    for label,g in [('all_anchor_models',h),('exclude_known_top10_imbalance',filtered),('C8_complete_only',h[h.C8_capability_imbalance.notna()])]:
        audit.append(dict(sample=label,n=len(g),known_C8_n=int(g.C8_capability_imbalance.notna().sum()),
            unknown_C8_n=int(g.C8_capability_imbalance.isna().sum()),imbalance_P90=threshold,**frontier_stats(g.A_z)))
    csv(out/'C8_anchor_sensitivity.csv',audit)
    return base,pd.DataFrame(rows)


def benchmark_series(root,cfg,d,base):
    """Historical 90-day pooled monthly frontiers; no max used in main curves."""
    out=root/'outputs/evolution/phase3';rows=[]
    for scope in SCOPES:
        g=select_scope(d,scope);origin=g.date.min()
        for month in pd.period_range(origin,base,freq='M'):
            date=min(month.end_time.normalize(),base);h=g[g.date.gt(date-pd.Timedelta(days=cfg['benchmark_frontier_window_days']))&g.date.le(date)]
            rows.append(dict(model_scope=scope,date=str(date.date()),month=str(month),n=len(h),
                t_month=float((date-d.date.min()).days/cfg['days_per_month']),window_days=cfg['benchmark_frontier_window_days'],
                sparse_window=len(h)<cfg['anchor_minimum_models'],**frontier_stats(h.A_z)))
    csv(out/'historical_benchmark_frontier.csv',rows);return pd.DataFrame(rows)


def compute_candidates(root,d,base):
    """Use actual C4 values, one row per accepted Epoch project; no name guessing."""
    out=root/'outputs/evolution/phase3';c4=pd.read_csv(root/'real_attachments/C_efficiency_evolution/epoch_all_ai_models.csv',float_precision='round_trip')
    g=d[d.model_type.eq('pretrained')&d.training_compute.gt(0)&np.isfinite(d.training_compute)&d.Epoch_match.notna()]
    rows=[]
    for key,h in g.groupby('epoch_row',sort=True):
        r=c4.iloc[int(key)];value=pd.to_numeric(r['Training compute (FLOP)'],errors='coerce');date=pd.to_datetime(r['Publication date'],errors='coerce')
        weights=str(r['Open model weights?']).strip().casefold();confirmed=weights in ['yes','true']
        valid=np.isfinite(value) and value>0 and pd.notna(date) and date<=base and confirmed
        rows.append(dict(epoch_row=int(key),C4_Model=r['Model'],matched_Models=json.dumps(sorted(h.Model.tolist())),
            matched_count=len(h),compute_FLOP=value,lnC=np.log(value) if value>0 else np.nan,date=str(date.date()) if pd.notna(date) else None,
            C4_open_weights=r['Open model weights?'],C4_Domain=r['Domain'],C4_Task=r['Task'],Organization=r['Organization'],
            compute_notes=r['Training compute notes'],compute_method=r['Training compute estimation method'],valid=bool(valid),
            scope='C4 exact accepted mappings to frozen S1 pretrained; no broad-name inference',
            exclusion='not confirmed C4 open/date/compute/as-of' if not valid else ''))
    csv(out/'compute_model_audit.csv',rows);result=pd.DataFrame(rows);result['date']=pd.to_datetime(result.date,errors='coerce')
    return result[result.valid].reset_index(drop=True)


def select_compute_window(data,end,cfg):
    """Original unique-model counts choose 36/48/60 months, not best slope."""
    attempts=[]
    for months in cfg['compute_recent_windows_months']:
        start=end-pd.DateOffset(months=months);n=int(data.date.between(start,end).sum())
        attempts.append(dict(window_months=months,n=n,minimum_met=n>=cfg['compute_minimum_models']))
        if n>=cfg['compute_minimum_models']:break
    if n<cfg['compute_minimum_models']:return None,None,attempts
    context=data[data.date.between(start-pd.DateOffset(months=cfg['compute_frontier_window_months']),end)].reset_index(drop=True)
    return context,start,attempts


def compute_grid(context,start,end,cfg):
    """Six calendar-month rolling windows at month ends, final date exactly T0."""
    dates=[min(m.end_time.normalize(),end) for m in pd.period_range(start,end,freq='M')]
    times=np.array([(x-start).days/cfg['days_per_month'] for x in dates]);raw=context.date.to_numpy(dtype='datetime64[ns]')
    masks=np.array([(raw>=np.datetime64(x-pd.DateOffset(months=cfg['compute_frontier_window_months'])))&(raw<=np.datetime64(x)) for x in dates])
    return dates,times,masks


def rolling_compute(log_values,masks,indices=None):
    """Recompute empirical log-frontiers after model-level resampling."""
    x=np.asarray(log_values,float);ix=np.arange(len(x)) if indices is None else indices;rows=[]
    for mask in masks:
        v=x[ix][mask[ix]]
        rows.append([np.quantile(v,.9),np.quantile(v,.95),np.sort(v)[-min(3,len(v)):].mean(),len(v)] if len(v) else [np.nan,np.nan,np.nan,0])
    return np.asarray(rows,float)


def slope(x,y,kind='TheilSen'):
    """Median pairwise slope, preserving negative or zero empirical growth."""
    ok=np.isfinite(y)&np.isfinite(x);a=x[ok];b=y[ok]
    if len(a)<3 or np.ptp(a)==0:return np.nan
    if kind=='OLS':return float(np.linalg.lstsq(np.column_stack([np.ones(len(a)),a]),b,rcond=None)[0][1])
    i,j=np.triu_indices(len(a),1);return float(np.median((b[j]-b[i])/(a[j]-a[i])))


def estimate_growth(root,cfg,data,base):
    """5000 model bootstraps of rolling-frontier slopes; sensitivity never refits A."""
    out=root/'outputs/evolution/phase3';context,start,attempts=select_compute_window(data,base,cfg)
    csv(out/'compute_window_audit.csv',attempts)
    if context is None:raise ValueError('Fewer than 20 strict mapped compute models even in 60 months; broader verified sample requires diagnosis')
    dates,t,masks=compute_grid(context,start,base,cfg);v=rolling_compute(context.lnC,masks);rows=[]
    for date,time,values in zip(dates,t,v):rows.append(dict(date=str(date.date()),t_month=time,P90_lnC=values[0],P95_lnC=values[1],Top3_lnC=values[2],n=int(values[3])))
    csv(out/'compute_frontier_history.csv',rows);csv(out/'compute_bootstrap_models.csv',context)
    B=cfg['bootstrap_repetitions'];rng=np.random.default_rng(cfg['random_state']+101);boots=[]
    for rep in range(1,B+1):
        ix=rng.integers(0,len(context),len(context));r=rolling_compute(context.lnC,masks,ix)
        boots.append(dict(replicate=rep,source_model_indices=json.dumps(ix.tolist()),
            TheilSen_P90=slope(t,r[:,0]),TheilSen_P95=slope(t,r[:,1]),TheilSen_Top3=slope(t,r[:,2]),OLS_P90=slope(t,r[:,0],'OLS'),
            nonempty_windows=int(np.isfinite(r[:,0]).sum())))
        if rep%1000==0:logging.info('C4 compute growth bootstrap %d/%d',rep,B)
    boots=pd.DataFrame(boots);csv(out/'compute_growth_bootstrap.csv',boots)
    estimates=[]
    for definition,col in [('P90',0),('P95',1),('Top3',2)]:
        for method in ['TheilSen','OLS']:
            key=method+'_'+definition;point=slope(t,v[:,col],method)
            dist=boots[key].dropna() if key in boots else pd.Series(dtype=float)
            estimates.append(dict(method=method,frontier_definition=definition,response='6-month rolling log compute frontier',
                monthly_log_growth=point,annualized_multiplier=np.exp(point*12),CI_low=dist.quantile(.025) if len(dist) else np.nan,
                CI_high=dist.quantile(.975) if len(dist) else np.nan,bootstrap_draws=len(dist),window_months=attempts[-1]['window_months'],
                recent_unique_models=attempts[-1]['n'],context_models=len(context),nonempty_months=int(np.isfinite(v[:,col]).sum()),
                base_date=str(base.date()),latest_compute_release=str(context.date.max().date()),no_monotonic_envelope=True))
    recent=context[context.date.ge(start)];tx=(recent.date-start).dt.days.to_numpy()/cfg['days_per_month'];X=np.column_stack([np.ones(len(tx)),tx])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always');q=QuantReg(recent.lnC.to_numpy(),X).fit(q=.9,max_iter=5000)
    estimates.append(dict(method='QuantReg_tau0.9',frontier_definition='conditional_P90',response='individual_model_log_compute',monthly_log_growth=q.params[1],
        annualized_multiplier=np.exp(q.params[1]*12),CI_low=np.nan,CI_high=np.nan,bootstrap_draws=0,window_months=attempts[-1]['window_months'],
        recent_unique_models=len(recent),context_models=len(context),nonempty_months=len(tx),base_date=str(base.date()),
        latest_compute_release=str(context.date.max().date()),solver_warnings='; '.join(str(w.message) for w in captured)))
    csv(out/'compute_growth_estimation.csv',estimates)
    metadata=dict(base_date=str(base.date()),compute_recent_start=str(start.date()),window_months=attempts[-1]['window_months'],
        recent_unique_models=attempts[-1]['n'],context_models=len(context),source_unique_models=len(data),
        main_growth=slope(t,v[:,0]),main_frontier='P90',method='TheilSen',last_frontier=float(v[np.flatnonzero(np.isfinite(v[:,0]))[-1],0]),
        last_frontier_date=str(dates[np.flatnonzero(np.isfinite(v[:,0]))[-1]].date()),
        latest_compute_release=str(context.date.max().date()),compute_staleness_days=int((base-context.date.max()).days),
        observed_lnC_max=float(context.lnC.max()),observed_lnC_IQR=float(context.lnC.quantile(.75)-context.lnC.quantile(.25)),
        model_bootstrap_accounts_overlapping_windows=True,universe_selection='strict accepted S1 pretrained→C4; not representative of all world compute')
    write_json(out/'compute_growth_model.json',metadata);return metadata


def c3_direction(root):
    """Year/source counts and parameter medians only; never concatenate ability scales."""
    out=root/'outputs/evolution/phase3';d=pd.read_csv(root/'real_attachments/C_efficiency_evolution/leaderboard_extended_timeseries.csv')
    rows=[]
    for (year,source),g in d.groupby(['Year','Source']):
        x=pd.to_numeric(g['Params_B'],errors='coerce')
        rows.append(dict(Year=year,Source=source,n=len(g),unique_models=g.Model.nunique(),median_N_B=x.median(),P90_N_B=x.quantile(.9),
            ability_used_in_projection=False,historical_scores_not_pooled=True))
    csv(out/'C3_long_run_direction_audit.csv',rows)

# ============================================================
# Source: src/evolution/projection_models.py
# Original lines: 1-249
# Purpose: 冻结成对系数的情景Monte Carlo、回测、补偿、风险标记
# Executed scenario projection, not a reliable long-term forecast. Frozen full-period calibration/model lookahead is disclosed. No Loss bridge conversion.
# ============================================================
"""Scenario increments and conditional hindcasts using frozen M2 coefficient pairs."""
import json
import logging
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from .common import csv
from .contribution_analysis import write_json
from .projection_data import SCOPES,select_scope,frontier_stats,select_compute_window,compute_grid,rolling_compute,slope


def read_csv(path):
    """Preserve frozen floating-point values."""
    return pd.read_csv(path,float_precision='round_trip')


def frozen_m2(root):
    """Load, never fit: beta and tau must remain from the same bootstrap row."""
    specs=json.loads((root/'outputs/evolution/phase2/model_definitions.json').read_text(encoding='utf8'))
    b=read_csv(root/'outputs/evolution/phase2/coefficient_bootstrap.csv');result={}
    for name in ['M2_C','M2_C_release']:
        spec=next(x for x in specs if x['model']==name);boot=b[b.model.eq(name)].reset_index(drop=True)
        if len(boot)!=5000 or not boot.full_rank.all() or not np.isfinite(boot[['lnC','t_month']]).all().all():raise ValueError('Invalid frozen bootstrap: '+name)
        result[name]=(spec,boot)
    return result


def linear_fit(x,y,target):
    """New Benchmark-only OLS and conventional one-step prediction interval."""
    X=np.column_stack([np.ones(len(x)),x]);b=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@b
    x0=np.array([1.,target]);prediction=float(x0@b);s2=float(res@res/max(1,len(x)-2))
    radius=float(student_t.ppf(.975,max(1,len(x)-2))*np.sqrt(s2*(1+x0@np.linalg.pinv(X.T@X)@x0)))
    return prediction,prediction-radius,prediction+radius,b


def build_hindcast_and_trends(root,cfg,d):
    """Train monthly trend only on prior months; audit frozen calibration lookahead."""
    out=root/'outputs/evolution/phase3';hist=read_csv(out/'historical_benchmark_frontier.csv');hist['date']=pd.to_datetime(hist.date)
    models=read_csv(out/'compute_model_audit.csv');models=models[models.valid].copy();models['date']=pd.to_datetime(models.date)
    frozen=frozen_m2(root);spec,coeff=frozen['M2_C'];fit_max=d[d.Model.isin(spec['Models'])].date.max();base=d.date.max()
    predictions=[];trend_specs=[];slope_boot=[];B=cfg['bootstrap_repetitions']
    for k,scope in enumerate(SCOPES):
        g=select_scope(d,scope).reset_index(drop=True);s=hist[hist.model_scope.eq(scope)&hist.P95.notna()].reset_index(drop=True)
        dates=s.date.to_list();raw=g.date.to_numpy(dtype='datetime64[ns]');mask=np.array([(raw>np.datetime64(x-pd.Timedelta(days=90)))&(raw<=np.datetime64(x)) for x in dates]);x=g.A_z.to_numpy()
        point=slope(s.t_month.to_numpy(),s.P95.to_numpy(),'OLS')
        trend_specs.append(dict(model_scope=scope,monthly_slope=point,n_months=len(s),history_months=(base-g.date.min()).days/cfg['days_per_month'],
            first_date=str(g.date.min().date()),last_date=str(base.date()),method='Benchmark-only linear P95 of 90-day pooled models',
            damped_k_identifiable=False,half_slope_is_assumption=True,used_M2=False))
        rng=np.random.default_rng(cfg['random_state']+201+k)
        for rep in range(1,B+1):
            ix=rng.integers(0,len(x),len(x));front=[frontier_stats(x[ix][m[ix]])['P95'] for m in mask]
            slope_boot.append(dict(model_scope=scope,replicate=rep,monthly_slope=slope(s.t_month.to_numpy(),np.asarray(front),'OLS'),source_model_indices=json.dumps(ix.tolist())))
        for ntrain in range(cfg['hindcast_minimum_training_months'],len(s)):
            train=s.iloc[:ntrain];future=s.iloc[ntrain];origin=train.iloc[-1]['date'];target=future['date'];elapsed=(target-origin).days/cfg['days_per_month']
            pred,lo,hi,b=linear_fit(train.t_month.to_numpy(),train.P95.to_numpy(),future.t_month)
            common=dict(model_scope=scope,train_months=ntrain,origin=str(origin.date()),target_date=str(target.date()),elapsed_months=elapsed,
                observed=future.P95,ability_calibration_uses_full_period=True,strict_temporal_out_of_sample=False,
                dependence_note='overlapping 90-day windows; same fixed full-period ability transform')
            predictions.append(dict(**common,method='Benchmark_linear',predicted=pred,low=lo,high=hi,covered=lo<=future.P95<=hi,
                frozen_coefficient_uses_future_models=False,compute_models=np.nan))
            if scope!='F0_pretrained':continue
            past=models[models.date.le(origin)];context,start,attempts=select_compute_window(past,origin,cfg)
            if context is None:
                predictions.append(dict(**common,method='mechanism_C0_T2_frozen',predicted=np.nan,low=np.nan,high=np.nan,covered=None,
                    frozen_coefficient_uses_future_models=fit_max>origin,compute_models=attempts[-1]['n'],failure='fewer than20 unique dated compute projects'))
                continue
            _,tx,masks=compute_grid(context,start,origin,cfg);v=rolling_compute(context.lnC,masks);growth=slope(tx,v[:,0])
            h=g[g.date.gt(origin-pd.Timedelta(days=90))&g.date.le(origin)];ar=h.A_z.to_numpy();anchor=frontier_stats(ar)['P95']
            rng_h=np.random.default_rng(cfg['random_state']+301+ntrain)
            anchor_b=np.quantile(ar[rng_h.integers(0,len(ar),(B,len(ar)))],.95,axis=1)
            # Historical reference uses observed-trend C0/T2, not future slowdown assumptions.
            pred=anchor+spec['beta'][1]*growth*elapsed+spec['beta'][2]*elapsed
            draws=anchor_b+(coeff.lnC.to_numpy()*growth+coeff.t_month.to_numpy())*elapsed;lo,hi=np.quantile(draws,[.025,.975])
            predictions.append(dict(**common,method='mechanism_C0_T2_frozen',predicted=pred,low=lo,high=hi,covered=lo<=future.P95<=hi,
                frozen_coefficient_uses_future_models=fit_max>origin,compute_models=attempts[-1]['n'],compute_growth_asof=growth,
                interval_note='conditional on as-of compute slope; no slope uncertainty in this hindcast interval'))
        logging.info('Benchmark trend bootstrap and hindcast completed: %s',scope)
    p=pd.DataFrame(predictions);p['error']=p.predicted-p.observed;p['absolute_error']=p.error.abs()
    summaries=[]
    for (scope,method),g in p.groupby(['model_scope','method']):
        valid=g[g.predicted.notna()]
        summaries.append(dict(model_scope=scope,method=method,n=len(valid),MAE=valid.absolute_error.mean(),RMSE=np.sqrt((valid.error**2).mean()),
            coverage=valid.covered.mean(),nominal_interval=.95,frozen_future_coefficient_origins=int(valid.frozen_coefficient_uses_future_models.sum()),
            strict_temporal_out_of_sample=False,ability_calibration_uses_full_period=True,independent_calibration_not_available=True))
    csv(out/'hindcast_predictions.csv',p);csv(out/'hindcast_metrics.csv',summaries);csv(out/'benchmark_trend_models.csv',trend_specs);csv(out/'benchmark_trend_bootstrap.csv',slope_boot)
    return p


def quantiles(x):
    """Parameter/data percentiles, never label as complete-world probability."""
    q=np.quantile(x,[.025,.1,.5,.9,.975]);return dict(zip(['A_z_P2.5','A_z_P10','A_z_median','A_z_P90','A_z_P97.5'],map(float,q)))


def level(value,thresholds):
    """Transparent ordered warning thresholds; not a fitted risk probability."""
    return int(sum(value>x for x in thresholds))


def build_projections(root,cfg,d):
    """Joint Monte Carlo of frozen paired M2, compute growth, and current anchors."""
    out=root/'outputs/evolution/phase3';base=d.date.max();frozen=frozen_m2(root);spec,coeff=frozen['M2_C'];release_spec,release_coeff=frozen['M2_C_release']
    anchors=read_csv(out/'frontier_anchor.csv');anchor_boot=read_csv(out/'anchor_bootstrap.csv');growth_boot=read_csv(out/'compute_growth_bootstrap.csv')
    growth=json.loads((out/'compute_growth_model.json').read_text(encoding='utf8'));transform=json.loads((root/'outputs/evolution/phase1/ability_transform_parameters.json').read_text(encoding='utf8'))
    tr=transform['display_mapping'];lo,hi=tr['P01'],tr['P99'];upper=float(np.mean([(x['P99']-x['median'])/x['IQR'] for x in transform['tasks']]))
    lower=float(np.mean([(x['P01']-x['median'])/x['IQR'] for x in transform['tasks']]))
    metrics=read_csv(out/'hindcast_metrics.csv');trends=read_csv(out/'benchmark_trend_models.csv');trend_boot=read_csv(out/'benchmark_trend_bootstrap.csv')
    c8=read_csv(out/'C8_anchor_sensitivity.csv');structure_shift=abs(c8.loc[c8['sample'].eq('exclude_known_top10_imbalance'),'P95'].iloc[0]-c8.loc[c8['sample'].eq('all_anchor_models'),'P95'].iloc[0])
    n=cfg['monte_carlo_repetitions'];rng=np.random.default_rng(cfg['random_state']+401)
    ci=rng.integers(0,len(coeff),n);ri=rng.integers(0,len(release_coeff),n);gi=rng.integers(0,len(growth_boot),n)
    if not np.isfinite(growth_boot.TheilSen_P90).all():raise ValueError('Invalid growth bootstrap; do not silently drop invalid draws')
    joint=pd.DataFrame(dict(draw_id=np.arange(1,n+1),coefficient_replicate=coeff.replicate.to_numpy()[ci],release_coefficient_replicate=release_coeff.replicate.to_numpy()[ri],
        growth_replicate=growth_boot.replicate.to_numpy()[gi],beta_C=coeff.lnC.to_numpy()[ci],tau_C=coeff.t_month.to_numpy()[ci],
        beta_C_release=release_coeff.lnC.to_numpy()[ri],tau_C_release=release_coeff.t_month.to_numpy()[ri],g_C=growth_boot.TheilSen_P90.to_numpy()[gi]))
    for scope in SCOPES:
        ab=anchor_boot[anchor_boot.model_scope.eq(scope)].reset_index(drop=True);ix=rng.integers(0,len(ab),n)
        joint[scope+'_anchor_replicate']=ab.replicate.to_numpy()[ix]
        for definition in ['P95','Top3','P90']:joint[scope+'_anchor_'+definition]=ab[definition].to_numpy()[ix]
        tb=trend_boot[trend_boot.model_scope.eq(scope)].reset_index(drop=True);ix=rng.integers(0,len(tb),n)
        joint[scope+'_trend_replicate']=tb.replicate.to_numpy()[ix];joint[scope+'_slope']=tb.monthly_slope.to_numpy()[ix]
    csv(out/'mc_joint_draws.csv',joint)
    history={scope:float(trends.loc[trends.model_scope.eq(scope),'history_months'].iloc[0]) for scope in SCOPES}
    iqrs={scope:float(select_scope(d,scope).A_z.quantile(.75)-select_scope(d,scope).A_z.quantile(.25)) for scope in SCOPES}
    obs_max={scope:float(select_scope(d,scope).A_z.max()) for scope in SCOPES}
    obs_p99={scope:float(select_scope(d,scope).A_z.quantile(.99)) for scope in SCOPES}
    rmses={scope:float(metrics[metrics.model_scope.eq(scope)].RMSE.max()) for scope in SCOPES}
    rows=[];mc=[];buffers=[]
    def anchor(scope,definition):return anchors[anchors.model_scope.eq(scope)&anchors.frontier_definition.eq(definition)].iloc[0]
    def summarize(scope,definition,h,c,t,kind,point,draws,scale,tech):
        a=anchor(scope,definition);q=quantiles(draws);buffer=1.96*rmses[scope]*np.sqrt(h)+(structure_shift if scope=='F0_pretrained' else 0.)
        r=dict(model_scope=scope,frontier_definition=definition,model_kind=kind,base_date=str(base.date()),horizon_months=h,
            forecast_date=str((base+pd.DateOffset(months=h)).date()),compute_scenario=c,technology_scenario=t,A_z_point=point,**q,
            A_z_anchor=a.A_z_anchor,anchor_n=int(a.n),anchor_window_days=int(a.window_days),anchor_below_minimum=bool(a.anchor_below_minimum),
            A_100_unclipped=100*(q['A_z_median']-lo)/(hi-lo),A_100_clipped_for_plot=float(np.clip(100*(q['A_z_median']-lo)/(hi-lo),0,100)),
            A_100=100*(q['A_z_median']-lo)/(hi-lo),Delta_scale=scale,Delta_non_scale=tech,
            expanded_risk_low=q['A_z_P2.5']-buffer,expanded_risk_high=q['A_z_P97.5']+buffer,hindcast_structure_buffer=buffer,
            interval_kind='parameter_data_percentiles; expanded_risk_* is uncalibrated error stress envelope',
            history_months=history[scope],horizon_history_ratio=h/history[scope],observed_A_z_max=obs_max[scope],observed_A_z_P99=obs_p99[scope],observed_A_z_IQR=iqrs[scope],
            operational_fixed_transform_lower=lower,operational_fixed_transform_upper=upper,
            exceeds_fixed_transform_upper_bound=bool(point>upper),MC_fraction_above_fixed_transform_upper=float((draws>upper).mean()),
            extreme_ability_extrapolation=bool(q['A_z_median']>obs_max[scope]+2*iqrs[scope]),
            mean_to_frontier_transport_is_assumption=kind=='frozen_M2_increment',scenario_factors_are_assumptions=True,
            independent_parameter_growth_anchor_blocks_assumed=True,loss_bridge_used=False)
        if scale>=0 and tech>=0 and scale+tech>0:r.update(future_scale_share=scale/(scale+tech),future_non_scale_share=tech/(scale+tech))
        else:r.update(future_scale_share=np.nan,future_non_scale_share=np.nan)
        return r
    for definition in ['P95','Top3','P90']:
        for h in cfg['horizons_months']:
            for c,cf in cfg['compute_scenario_factors'].items():
                for t,tf in cfg['technology_scenario_factors'].items():
                    scale=spec['beta'][1]*growth['main_growth']*cf*h;tech=spec['beta'][2]*tf*h
                    point=anchor('F0_pretrained',definition).A_z_anchor+scale+tech
                    delta_scale=joint.beta_C.to_numpy()*joint.g_C.to_numpy()*cf*h;delta_tech=joint.tau_C.to_numpy()*tf*h
                    draws=joint['F0_pretrained_anchor_'+definition].to_numpy()+delta_scale+delta_tech
                    rows.append(summarize('F0_pretrained',definition,h,c,t,'frozen_M2_increment',point,draws,scale,tech))
                    if definition=='P95':mc.append(pd.DataFrame(dict(draw_id=joint.draw_id,model_scope='F0_pretrained',frontier_definition=definition,
                        horizon_months=h,compute_scenario=c,technology_scenario=t,A_z=draws,Delta_scale=delta_scale,Delta_non_scale=delta_tech)))
    for scope in SCOPES:
        b=float(trends.loc[trends.model_scope.eq(scope),'monthly_slope'].iloc[0])
        for h in cfg['horizons_months']:
            for t,factor in cfg['direct_scenario_factors'].items():
                point=anchor(scope,'P95').A_z_anchor+b*factor*h;draws=joint[scope+'_anchor_P95'].to_numpy()+joint[scope+'_slope'].to_numpy()*factor*h
                rows.append(summarize(scope,'P95',h,'benchmark_only',t,'Benchmark_only',point,draws,np.nan,np.nan))
    result=pd.DataFrame(rows);csv(out/'frontier_projection_mc.csv',pd.concat(mc,ignore_index=True))
    # Counterfactual differences share the exact same MC draws; technology cancels.
    slowdown=[];compensation=[];tau_p5,tau_p95=coeff.t_month.quantile([.05,.95]);tau_stats=[]
    for c,cf in cfg['compute_scenario_factors'].items():
        req=.5*spec['beta'][2]+spec['beta'][1]*(1-cf)*growth['main_growth']
        rq=.5*joint.tau_C.to_numpy()+joint.beta_C.to_numpy()*(1-cf)*joint.g_C.to_numpy()
        label='below historical P5' if req<tau_p5 else ('above historical P95' if req>tau_p95 else 'within historical P5-P95 range')
        qr=np.quantile(rq,[.025,.5,.975]);compensation.append(dict(compute_scenario=c,compute_factor=cf,tau_required_point=req,tau_required_P2_5=qr[0],tau_required_median=qr[1],tau_required_P97_5=qr[2],
            historical_tau_P5=tau_p5,historical_tau_P95=tau_p95,range_comparison=label,relative_to_tau_point=req/spec['beta'][2],
            target='C0+T1',horizon_invariant=True,not_probability_of_innovation=True))
        if c=='C0':continue
        for h in cfg['horizons_months']:
            values=joint.beta_C.to_numpy()*joint.g_C.to_numpy()*(1-cf)*h;q=np.quantile(values,[.025,.1,.5,.9,.975])
            for t in cfg['technology_scenario_factors']:
                slowdown.append(dict(horizon_months=h,compute_scenario=c,technology_scenario=t,point_loss=spec['beta'][1]*growth['main_growth']*(1-cf)*h,
                    loss_P2_5=q[0],loss_P10=q[1],loss_median=q[2],loss_P90=q[3],loss_P97_5=q[4],comparison='C0 minus scenario, same tech and paired draws'))
    csv(out/'slowdown_ability_loss.csv',slowdown);csv(out/'technology_compensation.csv',compensation)
    tau_stats=[dict(axis=axis,negative_tau_count=int((boot.t_month<0).sum()),bootstrap_n=len(boot),beta_point=sp['beta'][1],
        beta_P2_5=boot.lnC.quantile(.025),beta_P97_5=boot.lnC.quantile(.975),tau_point=sp['beta'][2],tau_P2_5=boot.t_month.quantile(.025),tau_P97_5=boot.t_month.quantile(.975)) for axis,(sp,boot) in frozen.items()]
    csv(out/'frozen_coefficient_uncertainty.csv',tau_stats)
    # Same anchor isolates coefficient-axis sensitivity; release-pooled anchor is separate.
    pre=select_scope(d,'F0_pretrained');rel=pre[pre.release.notna()&pre.release.le(base)];release_audit=[]
    for days in cfg['anchor_windows_days']:
        rg=rel[rel.release.gt(base-pd.Timedelta(days=days))];release_audit.append(dict(window_days=days,n=len(rg),minimum_met=len(rg)>=20))
        if len(rg)>=20:break
    release_anchor=frontier_stats(rg.A_z)['P95'];rng_r=np.random.default_rng(cfg['random_state']+402)
    release_indices=rng_r.integers(0,len(rg),(n,len(rg))) if len(rg) else np.empty((n,0),int)
    release_anchor_draws=np.quantile(rg.A_z.to_numpy()[release_indices],.95,axis=1) if len(rg) else np.full(n,np.nan)
    csv(out/'release_anchor_bootstrap.csv',[dict(draw_id=i+1,P95=release_anchor_draws[i],source_model_indices=json.dumps(ix.tolist())) for i,ix in enumerate(release_indices)])
    csv(out/'release_anchor_window_audit.csv',release_audit);csv(out/'release_anchor_models.csv',rg[['Model','release','A_z']])
    time=[]
    for h in cfg['horizons_months']:
        main=result[result.model_kind.eq('frozen_M2_increment')&result.frontier_definition.eq('P95')&result.compute_scenario.eq('C1')&result.technology_scenario.eq('T1')&result.horizon_months.eq(h)].iloc[0]
        inc=(joint.beta_C_release.to_numpy()*joint.g_C.to_numpy()*.5+joint.tau_C_release.to_numpy()*.5)*h
        for label,a,ab in [('same_submission_anchor',main.A_z_anchor,joint.F0_pretrained_anchor_P95.to_numpy()),('release_pooled_anchor',release_anchor,release_anchor_draws)]:
            point=a+(release_spec['beta'][1]*growth['main_growth']*.5+release_spec['beta'][2]*.5)*h
            q=quantiles(ab+inc);difference=point-main.A_z_point
            raw_width=main['A_z_P97.5']-main['A_z_P2.5'];expanded_width=main.expanded_risk_high-main.expanded_risk_low
            time.append(dict(horizon_months=h,comparison=label,submission_point=main.A_z_point,release_point=point,difference=difference,**q,
                submission_parameter_interval_width=raw_width,submission_expanded_interval_width=expanded_width,
                high_time_axis_sensitivity=bool(abs(difference)>.5*raw_width),high_time_axis_sensitivity_expanded=bool(abs(difference)>.5*expanded_width),
                release_anchor_n=len(rg) if label=='release_pooled_anchor' else int(main.anchor_n),release_anchor_window_days=days if label=='release_pooled_anchor' else int(main.anchor_window_days),
                same_T0_and_future_calendar_dates=True,frozen_phase2_time_axis_sensitive=True))
    csv(out/'time_axis_projection_sensitivity.csv',time)
    definitions=[]
    for h in cfg['horizons_months']:
        subset=result[result.model_kind.eq('frozen_M2_increment')&result.compute_scenario.eq('C1')&result.technology_scenario.eq('T1')&result.horizon_months.eq(h)]
        main=subset[subset.frontier_definition.eq('P95')].iloc[0]
        for _,r in subset.iterrows():definitions.append(dict(horizon_months=h,frontier_definition=r.frontier_definition,A_z_point=r.A_z_point,A_z_median=r.A_z_median,
            difference_from_P95=r.A_z_point-main.A_z_point,warning_threshold=.1*iqrs['F0_pretrained'],frontier_definition_sensitive=abs(r.A_z_point-main.A_z_point)>.1*iqrs['F0_pretrained']))
    csv(out/'frontier_definition_sensitivity.csv',definitions)
    risks=[]
    for _,r in result.iterrows():
        mechanism=r.model_kind=='frozen_M2_increment';cf=cfg['compute_scenario_factors'][r.compute_scenario] if mechanism else np.nan
        future_lnC=growth['last_frontier']+growth['main_growth']*cf*r.horizon_months if mechanism else np.nan
        distance=max(0,future_lnC-growth['observed_lnC_max'])/growth['observed_lnC_IQR'] if mechanism else 0.
        uq=(r['A_z_P97.5']-r['A_z_P2.5'])/max(abs(r.A_z_point-r.A_z_anchor),iqrs[r.model_scope])
        def_sensitive=any(x['frontier_definition_sensitive'] for x in definitions if x['horizon_months']==r.horizon_months) if mechanism else False
        projection_time_flag=any(x['high_time_axis_sensitivity'] for x in time if x['horizon_months']==r.horizon_months) if mechanism else False
        coef_delta=(coeff.lnC.to_numpy()*growth['main_growth']*cf+coeff.t_month.to_numpy()*cfg['technology_scenario_factors'][r.technology_scenario])*r.horizon_months if mechanism else None
        coef_ratio=float(np.ptp(np.quantile(coef_delta,[.025,.975]))/max(abs(r.A_z_point-r.A_z_anchor),iqrs[r.model_scope])) if mechanism else 0.
        rule=cfg['risk_rule'];dimensions=dict(history_severity=level(r.horizon_history_ratio,rule['history_ratio']),compute_severity=level(distance,rule['compute_excess_IQR']),
            frozen_coefficient_width_severity=level(coef_ratio,rule['uncertainty_width_over_signal_or_IQR']) if mechanism else 0,
            coefficient_data_width_severity=level(uq,rule['uncertainty_width_over_signal_or_IQR']),time_axis_severity=rule['frozen_time_axis_sensitive_severity'] if mechanism else 0,
            frontier_definition_severity=rule['frontier_definition_sensitive_severity'] if def_sensitive else 0,
            anchor_count_severity=rule['anchor_below_minimum_severity'] if r.anchor_below_minimum else 0,
            fixed_transform_bound_severity=rule['exceeds_operational_ability_bound_severity'] if r.exceeds_fixed_transform_upper_bound else 0)
        severity=max(dimensions.values());risk=['LOW','MEDIUM','HIGH','VERY_HIGH'][severity]
        risks.append(dict(model_scope=r.model_scope,frontier_definition=r.frontier_definition,model_kind=r.model_kind,horizon_months=r.horizon_months,
            compute_scenario=r.compute_scenario,technology_scenario=r.technology_scenario,risk_level=risk,**dimensions,
            horizon_history_ratio=r.horizon_history_ratio,compute_excess_IQR=distance,uncertainty_width_over_signal_or_IQR=uq,coefficient_only_width_ratio=coef_ratio,
            frozen_beta_bootstrap_width=float(coeff.lnC.quantile(.975)-coeff.lnC.quantile(.025)) if mechanism else np.nan,
            frozen_tau_bootstrap_width=float(coeff.t_month.quantile(.975)-coeff.t_month.quantile(.025)) if mechanism else np.nan,
            projected_lnC=future_lnC,high_time_axis_sensitivity=projection_time_flag,frozen_time_axis_sensitive=mechanism,
            frontier_definition_sensitive=def_sensitive,extreme_ability_extrapolation=r.extreme_ability_extrapolation,
            exceeds_fixed_transform_upper_bound=r.exceeds_fixed_transform_upper_bound,extrapolation_flag=True,rule='maximum dimension severity from configs/evolution_phase3.yaml'))
    risk_table=pd.DataFrame(risks);result['risk_level']=risk_table.risk_level;result['extrapolation_flag']=True
    csv(out/'future_frontier_scenarios.csv',result);csv(out/'forecast_risk_flags.csv',risk_table)
    csv(out/'projection_interval_expansion.csv',[dict(model_scope=s,one_step_RMSE=rmses[s],structure_shift=structure_shift if s=='F0_pretrained' else 0.,
        rule=cfg['hindcast_buffer_rule'],n_hindcast=int(metrics[metrics.model_scope.eq(s)].n.sum()),not_calibrated_95_probability_interval=True) for s in SCOPES])
    write_json(out/'projection_execution_summary.json',dict(base_date=str(base.date()),S1_count=len(d),main_scenarios=24,mechanism_rows=72,
        monte_carlo_repetitions=n,stored_main_mc_rows=n*24,compute_bootstrap_repetitions=len(growth_boot),frozen_M2_bootstrap_repetitions=len(coeff),
        beta_tau_pairing_preserved=True,negative_tau_retained=True,negative_growth_retained=True,operational_A_z_upper=upper,operational_A_z_lower=lower,
        scale_model_refitted=False,ability_transform_refitted=False,loss_bridge_used=False,
        uncertainty_coverage='conditional parameter/data percentiles; independent blocks assumption; overlapping windows; no full future-world distribution',
        risk_expansion='1.96 * max available conditional one-step RMSE within scope * sqrt(h) + pretrained C8 anchor shift; uncalibrated stress envelope'))
    logging.info('Completed %d scenario rows; %d main Monte Carlo rows; frozen models unchanged',len(result),n*24)

# ============================================================
# Source: src/evolution/projection_reporting.py
# Original lines: 1-270
# Purpose: 最终CSV/图表/报告呈现，compact省略纯绘图和排版
# ============================================================
"""Readable final evidence report and scientific plots, keeping scenario warnings visible."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .common import csv,table
from .projection_models import read_csv,frozen_m2


def build_figures(root,cfg,d):
    """At least ten modest static figures; raw A_z remains unclipped everywhere."""
    out=root/'outputs/evolution/phase3';folder=root/'outputs/figures/question4_phase3';folder.mkdir(parents=True,exist_ok=True)
    s=read_csv(out/'future_frontier_scenarios.csv');a=read_csv(out/'frontier_anchor.csv');hist=read_csv(out/'historical_benchmark_frontier.csv');hist['date']=pd.to_datetime(hist.date)
    g=read_csv(out/'compute_growth_estimation.csv');ch=read_csv(out/'compute_frontier_history.csv');ch['date']=pd.to_datetime(ch.date)
    draws=read_csv(out/'mc_joint_draws.csv');base=pd.Timestamp(s.base_date.iloc[0]);manifest=[]
    main=s[s.model_kind.eq('frozen_M2_increment')&s.frontier_definition.eq('P95')&s.compute_scenario.eq('C1')&s.technology_scenario.eq('T1')].sort_values('horizon_months')
    pre=s[s.model_kind.eq('frozen_M2_increment')&s.frontier_definition.eq('P95')];names={'F0_pretrained':'Pretrained','F1_chat':'Chat / finetuned','F2_overall':'All S1'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120})
    def save(fig,name,note):
        fig.tight_layout();path=folder/name;fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig);manifest.append(dict(file=str(path.relative_to(root)).replace('\\','/'),purpose=note))
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    actual=hist[hist.model_scope.eq('F0_pretrained')];anchor=main.A_z_anchor.iloc[0];h=np.arange(25);dates=[base+pd.DateOffset(months=int(x)) for x in h]
    y=draws.F0_pretrained_anchor_P95.to_numpy()[:,None]+(draws.beta_C.to_numpy()*draws.g_C.to_numpy()*.5+draws.tau_C.to_numpy()*.5)[:,None]*h
    low,med,high=np.quantile(y,[.025,.5,.975],axis=0);point=anchor+(main.Delta_scale.iloc[0]+main.Delta_non_scale.iloc[0])/12*h
    for ax in axes:
        ax.plot(actual.date,actual.P95,'o-',color='#333333',label='Observed 90-day P95')
        ax.scatter([base],[anchor],marker='D',color='#157a6e',label='Current 180-day P95 (n=18)',zorder=5)
        ax.plot(dates,point,'--',color='#157a6e',label='C1 x T1 plug-in scenario')
        ax.plot(dates,med,color='#3573b9',label='Monte Carlo median')
        ax.fill_between(dates,low,high,color='#3573b9',alpha=.2,label='Parameter/data P2.5-P97.5')
        ax.axvline(base,color='#777777',lw=.8);ax.set_ylabel('Frozen ability index A_z (unclipped)');ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));ax.grid(alpha=.15)
    axes[0].axhline(main.operational_fixed_transform_upper.iloc[0],color='#b54b4b',ls=':',label='Fixed transform attainable upper bound')
    axes[0].set_title('Pretrained frontier: conditional projection');axes[0].legend(fontsize=8,loc='upper left')
    expansion=read_csv(out/'projection_interval_expansion.csv');e=expansion[expansion.model_scope.eq('F0_pretrained')].iloc[0];buffer=1.96*e.one_step_RMSE*np.sqrt(h)+e.structure_shift
    buffer[0]=0;axes[1].fill_between(dates,low-buffer,high+buffer,color='#d78731',alpha=.17,label='Uncalibrated hindcast/structure stress envelope')
    axes[1].set_title('Risk expansion: not a 95% probability interval');axes[1].legend(fontsize=8,loc='upper left')
    save(fig,'01_frontier_projection_fan.png','Frozen historical 90-day series, current count-selected180-day anchor, raw parameter fan and separate risk envelope')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,horizon in zip(axes,[12,24]):
        z=pre[pre.technology_scenario.eq('T1')&pre.horizon_months.eq(horizon)].sort_values('compute_scenario');pos=np.arange(4)
        ax.errorbar(pos,z.A_z_median,yerr=[z.A_z_median-z['A_z_P2.5'],z['A_z_P97.5']-z.A_z_median],fmt='o',capsize=4,label='MC median and parameter interval')
        ax.scatter(pos,z.A_z_point,marker='D',color='#157a6e',label='Plug-in scenario');ax.set_xticks(pos,z.compute_scenario);ax.set_ylabel('A_z');ax.set_title(f'{horizon} months, T1 fixed');ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);save(fig,'02_compute_slowdown_effect.png','Compute slowdown paired scenarios; unclipped A_z')
    fig,ax=plt.subplots(figsize=(7.5,4));pos=np.arange(2);ax.bar(pos,main.Delta_scale,label='Compute-scale increment');ax.bar(pos,main.Delta_non_scale,bottom=main.Delta_scale,label='Non-scale time-proxy increment');ax.set_xticks(pos,['12 months','24 months']);ax.set_ylabel('Signed delta A_z');ax.set_title('C1 x T1 future increment decomposition (scenario)');ax.legend();save(fig,'03_future_increment_decomposition.png','Signed point increments; not historical absolute shares')
    for horizon in [12,24]:
        mat=pre[pre.horizon_months.eq(horizon)].pivot(index='compute_scenario',columns='technology_scenario',values='A_z_point').reindex(index=['C0','C1','C2','C3'],columns=['T0','T1','T2'])
        fig,ax=plt.subplots(figsize=(6.5,4.5));im=ax.imshow(mat,cmap='viridis',aspect='auto');ax.set_xticks(range(3),mat.columns);ax.set_yticks(range(4),mat.index)
        for i in range(4):
            for j in range(3):ax.text(j,i,f'{mat.iloc[i,j]:.3f}',ha='center',va='center',color='white' if mat.iloc[i,j]<np.nanmedian(mat) else 'black')
        ax.set_xlabel('Technology factor: 0 / 0.5 / 1');ax.set_ylabel('Compute factor: 1 / 0.5 / 0.25 / 0');ax.set_title(f'{horizon}-month P95 plug-in scenarios (A_z)');fig.colorbar(im,ax=ax,label='Unclipped A_z');save(fig,f'0{4 if horizon==12 else 5}_scenario_matrix_{horizon}m.png','24 assumptions across two horizons; full numeric intervals in CSV')
    growth=json.loads((out/'compute_growth_model.json').read_text(encoding='utf8'));fig,ax=plt.subplots(figsize=(10,4.5));audit=read_csv(out/'compute_model_audit.csv');audit=audit[audit.valid];ax.scatter(pd.to_datetime(audit.date),audit.lnC,alpha=.35,label='Unique C4 training projects')
    for col in ['P90_lnC','P95_lnC','Top3_lnC']:ax.plot(ch.date,ch[col],label='6-month '+col)
    last=growth['last_frontier'];origin=growth['last_frontier_date'];future=[base+pd.DateOffset(months=x) for x in range(25)]
    for c,factor in cfg['compute_scenario_factors'].items():ax.plot(future,last+growth['main_growth']*factor*np.arange(25),ls='--',label=f'{c}: {factor:g} x historical growth')
    ax.axvline(base,color='grey',ls=':');ax.set_ylabel('Natural log training compute (FLOP)');ax.set_title('C4 compute frontier: empirical rolling windows and scenario growth');ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.15);save(fig,'06_compute_frontier_history_projection.png','Only dated unique C4 accepted open-weight pretrained mappings, no future compute data')
    hp=read_csv(out/'hindcast_predictions.csv');fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,scope in zip(axes,names):
        z=hp[hp.model_scope.eq(scope)];first=z[z.method.eq('Benchmark_linear')];ax.plot(pd.to_datetime(first.target_date),first.observed,'o-',color='#333333',label='Observed')
        for method,gp in z.groupby('method'):ax.plot(pd.to_datetime(gp.target_date),gp.predicted,'o--',label='Mechanism (frozen)' if 'mechanism' in method else 'Benchmark linear')
        ax.set_title(names[scope]);ax.set_ylabel('A_z');ax.set_xticks(pd.to_datetime(first.target_date));ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'));ax.tick_params(axis='x',labelsize=8);ax.grid(alpha=.15)
    axes[0].legend(fontsize=8);fig.suptitle('Conditional rolling-origin hindcast: fixed full-period calibration');save(fig,'07_hindcast_observed_predicted.png','Four forecasts per scope, not strictly temporal OOS; frozen M2 future-model warning')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for scope in names:
        z=hist[hist.model_scope.eq(scope)];axes[0].plot(z.date,z.P95,'o-',label=names[scope]);zz=s[s.model_scope.eq(scope)&s.model_kind.eq('Benchmark_only')&s.technology_scenario.eq('half_slope')].sort_values('horizon_months')
        aa=a[a.model_scope.eq(scope)&a.frontier_definition.eq('P95')].A_z_anchor.iloc[0];axes[1].plot([0,12,24],[aa,*zz.A_z_point],marker='o',label=names[scope]+' (half slope)')
    axes[1].plot([0,12,24],[anchor,*main.A_z_point],marker='D',ls='--',label='Pretrained mechanism C1 x T1');axes[0].set_title('Separate observed 90-day P95 frontiers');axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2));axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'));axes[1].set_title('Scenario comparison; different mechanisms');axes[1].set_xlabel('Months after '+str(base.date()))
    for ax in axes:ax.set_ylabel('A_z');ax.legend(fontsize=8);ax.grid(alpha=.15)
    save(fig,'08_pretrained_chat_overall_comparison.png','Chat/all Benchmark-only; no compute mechanism or RLHF attribution')
    time=read_csv(out/'time_axis_projection_sensitivity.csv');fig,ax=plt.subplots(figsize=(8,4.5));ax.plot(main.horizon_months,main.A_z_point,'o-',label='Submission frozen M2')
    for comparison,z in time.groupby('comparison'):ax.plot(z.horizon_months,z.release_point,'o--',label='Release M2: '+comparison.replace('_',' '))
    ax.set_xlabel('Forecast months');ax.set_ylabel('A_z point scenario');ax.set_title('Time-axis sensitivity, same T0 and C1 x T1');ax.legend(fontsize=8);ax.grid(alpha=.15);save(fig,'09_submission_release_sensitivity.png','Same anchor coefficient sensitivity and separately pooled Release anchor')
    comp=read_csv(out/'technology_compensation.csv');fig,ax=plt.subplots(figsize=(8,4.5));ax.axhspan(comp.historical_tau_P5.iloc[0],comp.historical_tau_P95.iloc[0],color='#3573b9',alpha=.18,label='Frozen historical tau bootstrap P5-P95')
    ax.plot(range(4),comp.tau_required_point,'o-',label='Required tau for C0+T1 target');ax.axhline(comp.tau_required_point.iloc[0]*2,color='grey',ls='--',label='Frozen tau point (T2)');ax.set_xticks(range(4),comp.compute_scenario);ax.set_ylabel('Required A_z increment per month');ax.set_title('Conditional non-scale compensation; not innovation probability');ax.legend(fontsize=8);save(fig,'10_non_scale_compensation.png','Range comparison only, horizon-invariant required rate')
    fig,ax=plt.subplots(figsize=(8,4));c8=read_csv(out/'C8_anchor_sensitivity.csv');ax.bar(['All pretrained anchor','Exclude known top10% imbalance','C8 known only'],c8.P95);ax.set_ylabel('Current P95 A_z');ax.set_title('C8 structure sensitivity: 12 / 18 anchor models are unknown');ax.tick_params(axis='x',labelsize=8);save(fig,'11_C8_anchor_structure_sensitivity.png','Known high imbalance removal and missing-C8 selection caveat')
    csv(out/'figure_manifest.csv',manifest)


def build_report_and_figures(root,cfg,d):
    """Assemble the final report from real current/frozen outputs, never invented results."""
    build_figures(root,cfg,d)
    build_final_report(root,cfg,d)


def build_final_report(root,cfg,d):
    """Final 4.1–4.13 report with frozen evidence separated from extrapolations."""
    out=root/'outputs/evolution/phase3';p1=root/'outputs/evolution/phase1';p2=root/'outputs/evolution/phase2'
    s=read_csv(out/'future_frontier_scenarios.csv');a=read_csv(out/'frontier_anchor.csv');g=read_csv(out/'compute_growth_estimation.csv');gm=json.loads((out/'compute_growth_model.json').read_text(encoding='utf8'))
    main=s[s.model_kind.eq('frozen_M2_increment')&s.frontier_definition.eq('P95')&s.compute_scenario.eq('C1')&s.technology_scenario.eq('T1')].sort_values('horizon_months')
    c8=read_csv(out/'C8_anchor_sensitivity.csv');weak=read_csv(p1/'C8_weak_tail_diagnosis.csv');co=read_csv(p2/'model_coefficients.csv');perf=read_csv(p2/'model_performance.csv');decomp=read_csv(p2/'contribution_decomposition.csv')
    bridge=read_csv(p1/'bridge_validation_metrics.csv');bm=json.loads((p1/'bridge_model.json').read_text(encoding='utf8'));spec,boot=frozen_m2(root)['M2_C'];trans=json.loads((p1/'ability_transform_parameters.json').read_text(encoding='utf8'))
    growth=gm['main_growth'];comp=read_csv(out/'technology_compensation.csv');loss=read_csv(out/'slowdown_ability_loss.csv');hind=read_csv(out/'hindcast_metrics.csv');time=read_csv(out/'time_axis_projection_sensitivity.csv');defs=read_csv(out/'frontier_definition_sensitivity.csv');risk=read_csv(out/'forecast_risk_flags.csv')
    types=d.groupby('model_type').size().rename('n').reset_index();core=decomp[decomp.series.isin(['R1','R2','R3'])&decomp.window_fraction.sub(1/3).abs().lt(1e-8)&decomp.time_axis.eq('submission')]
    coefficient_table=co[co.model.isin(['M1_linear','M2_C','M3_ND'])&co.inference.eq('HC3')]
    scenarios=pd.DataFrame([dict(compute_scenario=c,factor=f,monthly_log_growth=f*growth,annualized_multiplier=np.exp(12*f*growth),assumption=True) for c,f in cfg['compute_scenario_factors'].items()])
    csv(out/'compute_scenario_assumptions.csv',scenarios)
    visible=['horizon_months','forecast_date','A_z_point','A_z_median','A_z_P2.5','A_z_P10','A_z_P90','A_z_P97.5','expanded_risk_low','expanded_risk_high','risk_level']
    contribution=['horizon_months','Delta_scale','Delta_non_scale','future_scale_share','future_non_scale_share','A_100_unclipped','A_100_clipped_for_plot']
    benchmark=s[s.model_kind.eq('Benchmark_only')][['model_scope','horizon_months','technology_scenario','A_z_point','A_z_median','A_z_P2.5','A_z_P97.5','risk_level']]
    rm=main.iloc[0];r24=main.iloc[1];bounds=(rm.operational_fixed_transform_lower,rm.operational_fixed_transform_upper)
    text=f'''# 问题四最终报告：算力增长放缓下的情景化能力前沿投影

基于附件快照，T0={rm.base_date}；未来日期为{rm.forecast_date}、{r24.forecast_date}。这是相对于附件历史起点的12/24个月情景投影，不是相对于运行日的新预测。问题一至三、问题四PHASE1/2的结果保持冻结。PHASE3只新建算力增长、前沿统计、Benchmark-only对照、情景传播及诊断。

**定位：scenario-based frontier projection。** 以下点情景、条件参数/数据区间、扩大风险包络分别输出；不宣称可靠长期点预测、不把情景设定赋予发生概率。

## 4.1 数据与开源模型口径〔Level A〕

冻结C1/C2各4576条原记录、4497个去重模型，S0=4497、S1=422、S2=0。S1表示附件明确确认开放权重，不能据此证明严格开源许可证授权。当前有效S1分类：

{table(types)}

F0仅pretrained进入M2-C机制投影；F1 chat_finetuned仅Benchmark趋势；F2包含S1全部类型，仅描述实际开源权重能力前沿。提交日期既不一定是模型发布日期，也不一定是当时真实评测日；快照去重产生存留与选择效应。因此“历史前沿”严格说是当前附件按提交月份回溯的样本前沿。冻结详细证据见[PHASE1报告](question4_phase1_data_ability_bridge.md)与[PHASE2报告](question4_phase2_contribution_decomposition.md)。

本阶段真实读取冻结evolution_model_table.csv、ability_transform_parameters.json、model_definitions.json、coefficient_bootstrap.csv及各诊断表；原始C4 epoch_all_ai_models.csv仅用于新的算力趋势，原始C3 leaderboard_extended_timeseries.csv仅用于年份/参数方向检查。input_manifest.csv逐文件保留路径与SHA256。没有读取旧版问题四预测作为结论依据。

## 4.2 综合能力指标〔Level A〕

主指标为六个官方Benchmark的等权稳健标准化：$A_z=\\frac16\\sum_k(\\operatorname{{clip}}(x_k,P01_k,P99_k)-m_k)/IQR_k$，指标顺序{', '.join(trans['metrics'])}。全部使用冻结S1全期阈值，不月度重拟合。PCA只是冻结敏感性，不替换主指标。A100显示使用原映射P01={trans['display_mapping']['P01']:.9f}、P99={trans['display_mapping']['P99']:.9f}，未来值不重新归一化。

标准化分数本身没有天然上界，但**当前冻结的逐任务Winsor公式有操作性可实现范围 [{bounds[0]:.6f},{bounds[1]:.6f}]**。本阶段不修改公式、不给外推偷偷截断。超界原始A_z和A100仍保存并标记；它们只能表示增量模型的外推信号，不能作为该冻结Benchmark指标可实现的未来分数。A_100_unclipped与A_100_clipped_for_plot同时输出；正文用未截断A_z。这里没有新增logistic饱和上限。

## 4.3 C8细粒度能力检验〔Level A〕

冻结C8同轮与快照弱尾诊断：

{table(weak[['sample','ability_basis','n','high_average_weak_tail_count','high_average_weak_tail_rate']])}

不能把版本差异解释成模型退化。当前pretrained锚点C8敏感性如下；阈值是有C8的S1 pretrained样本imbalance P90，而不是主观筛选。未知C8保留并单列。

{table(c8)}

排除1个已知最高10%偏科模型后，P95下降{float(c8.P95.iloc[0]-c8.P95.iloc[1]):.6f}；{int(c8.unknown_C8_n.iloc[0])}/{int(c8.n.iloc[0])}模型C8未知。差异明显，不能说前沿不受偏科影响。将绝对锚点差作为结构压力包络的一项，未把缺失C8当作“无偏科”。

## 4.4 规模与非规模进步分解〔Level B〕

冻结回归HC3诊断，不重新拟合：

{table(coefficient_table[['model','term','n','coef','CI_low','CI_high','p_value']])}

M1控制参数规模，M2控制训练算力，M3控制N、D；时间项是非规模进步代理的条件关联，不是因果技术效应或RLHF效率。M2只有21条完整记录，tau的HC3区间跨0，Bootstrap中{int((boot.t_month<0).sum())}/5000个tau为负，本轮原样保留。组织依赖、样本选择和时间口径限制已在冻结PHASE2中验证，不因本轮情景计算而消失。

历史主窗口R1/R2/R3：

{table(core[['series','n','early_n','late_n','Delta_scale','Delta_tech','Delta_pred','absolute_share_scale','absolute_share_tech']])}

这些absolute share只描述过去抵消项的绝对贡献，**没有被乘以未来增长**。未来分解严格使用beta_C×未来log算力增量和tau×情景时间增量。冻结M2的LOOCV亦只支持有限的关联泛化：

{table(perf[perf.model.isin(['M1_linear','M2_C','M3_ND'])&perf.evaluation.eq('LOOCV')][['model','n','RMSE','MAE','R2']])}

## 4.5 Loss-Benchmark桥接负结果〔Level A/B：不成立〕

C6 High仅{bm['High_count']}个、全部{', '.join(bm['High_families'])}；Medium={bm['Medium_count']}。B0/B1/B2均未取得可靠CV泛化：

{table(bridge[['candidate','evaluation','n','RMSE','MAE','R2']])}

B2是常数保序回归诊断，bridge_valid_for_conversion={bm['bridge_valid_for_conversion']}。不能把前三问Loss可靠转换为第四问Benchmark；本轮没有使用任何Loss桥接数值。前三问与本问通过scale/compute资源变量及机制解释联系。

## 4.6 能力前沿定义〔Level A〕

按样本数量依次90→120→180天，仅不足20才扩大，不追求更高前沿；达到180天仍不足就保留并警示。主定义P95，P90与Top3敏感性；maximum只存在辅助附录统计，未作为主锚点。

{table(a[['model_scope','frontier_definition','base_date','window_days','n','A_z_anchor','anchor_below_minimum']])}

pretrained只有18条，必须承认锚点稀疏。历史Benchmark-only与hindcast固定90天滚动前沿，主机制当前锚点按规则扩大180天，二者不是同一窗口。Top3指现有最多3条模型的均值；Bootstrap重复条目属于重采样经验分布，不伪称新的不同模型。

## 4.7 算力增长趋势〔Level C所需历史输入〕

C4由冻结S1 pretrained可靠映射：38条匹配记录属于37个不同训练项目，按epoch_row去重。最近36个月有{gm['recent_unique_models']}个原始去重项目达到20要求；向前多6个月的滚动起始上下文共{gm['context_models']}个，Bootstrap重采样上下文模型，重算所有滚动窗口，避免把重叠窗口当独立观测。

{table(g[['method','frontier_definition','monthly_log_growth','annualized_multiplier','CI_low','CI_high','bootstrap_draws','recent_unique_models']])}

主Theil-Sen是两点斜率中位数，主响应6个月滚动P90 logCompute；OLS与P95/Top3敏感性同为滚动前沿，QuantReg tau=.9是个体模型条件分位数，估计对象不同。方法参照[SciPy Theil-Sen定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html)和[statsmodels QuantReg](https://www.statsmodels.org/v0.14.4/generated/statsmodels.regression.quantile_regression.QuantReg.fit.html)。主5000模型Bootstrap月度斜率区间保留全部有效结果，不强制增长非负、不施加单调包络。OLS-P95/Top3及QuantReg只有点敏感性，没有伪造Bootstrap区间。

最新有算力项目日期{gm['latest_compute_release']}，滞后T0 {gm['compute_staleness_days']}天；这不是全球算力普查。样本来自被Leaderboard匹配到的开放pretrained项目，算力估计及发布时间本身也有误差，5000次重采样不能消除选择偏差。C3按年份/来源的模型数与N统计保留在C3_long_run_direction_audit.csv，26条历史补充记录没有拼入A_z。

## 4.8 算力放缓情景〔Level C〕

{table(scenarios)}

情景系数1、.5、.25、0是题设假设，不是数据估计；C1为主，C0只是无放缓参照。T0/T1/T2对应0、.5、1倍冻结非规模时间代理，T1为中性主情景，不称最可能。主增量公式：

$$A(h)=A_{{anchor}}+\\beta_C g_C f_C h+\\tau_C f_T h.$$

beta_C={spec['beta'][1]:.12f}，tau_C={spec['beta'][2]:.12f}，h=12/24。把横截面条件均值斜率迁移到P95前沿是附加不变性假设，没有重新拟合分位能力模型。24主情景加P90/Top3敏感性共72条机制结果；另18条Benchmark-only结果，全表90条。Monte Carlo 10000次：同一冻结系数Bootstrap行抽beta/tau，算力与锚点Bootstrap各独立抽取；三块可能共享模型，独立块假设仍可能低估协方差，未假装完整联合世界概率模型。240000条主情景draw保留在frontier_projection_mc.csv，mc_joint_draws.csv保留每块replicate可复算。

## 4.9 未来12个月前沿〔Level C〕

{table(main[main.horizon_months.eq(12)][visible])}

点情景是插入原估计所得，MC中位数来自重采样传播，不必相等。以下P2.5/P97.5只是参数/数据分位数。主P95方案全部12个月情景：

{table(s[s.model_kind.eq('frozen_M2_increment')&s.frontier_definition.eq('P95')&s.horizon_months.eq(12)][['compute_scenario','technology_scenario','A_z_point','A_z_median','A_z_P2.5','A_z_P97.5','risk_level']])}

## 4.10 未来24个月前沿〔Level D〕

{table(main[main.horizon_months.eq(24)][visible])}

24个月点情景{r24.A_z_point:.6f}与MC中位数{r24.A_z_median:.6f}均超出冻结操作性上界{bounds[1]:.6f}，不可当作可实现Benchmark分数。继续保留未截断数值作为压力外推，以VERY_HIGH标识。相对观测max/P99、超过max+2IQR、超过冻结可实现范围分别诊断，不能相互替代。

F1/F2只做flat、50%、full slope延续，不套用M2：

{table(benchmark)}

F0 Benchmark-only斜率为负而机制为正，来源是稀疏提交组和滚动窗口组成变化，两套预测并不相互验证。chat增长也不能解释为RLHF技术进步。约10个月不足识别指数阻尼k，故没有拟合damped，50% slope是显式假设。不同类型窗口/分位样本组成不同，不能据数值大小直接推出模型类型优劣。

## 4.11 非规模进步补偿效应〔Level C；24个月应用为D〕

未来主情景的有符号增量与显示映射：

{table(main[contribution])}

仅两项非负且总增量正时报告份额；Bootstrap中的负项仍保留，没有强制归一化。对应同一T1，算力放缓损失相对C0：

{table(loss[loss.technology_scenario.eq('T1')][['horizon_months','compute_scenario','point_loss','loss_P2_5','loss_median','loss_P97_5']])}

同一joint draw下算力差beta*g*(1-f_C)*h，技术项相消。条件补偿：$\\tau_{{required}}=.5\\tau_C+\\beta_C(1-f_C)g_C$，在线性固定参数假设下不随h变化。

{table(comp)}

所需速率位于历史Bootstrap P5–P95范围只表示参数比较没有超出该区间，**不代表未来技术成功补偿的概率，也不证明必然可补偿**；历史tau区间很宽、部分为负。

## 4.12 不确定性与外推风险

逐月训练1..6预测7，依次到1..9预测10：

{table(hind)}

Benchmark线性只拟合当时过去的6–9个前沿点；机制不违背冻结规则，因此没有逐折重拟合M2，前两折冻结M2使用后来提交模型，全部折能力标定用全期S1。**这些是条件回溯诊断，不是严格时间外验证**。90日窗口重叠，四折覆盖率无从校准长期区间；机制hindcast使用C0×T2历史延续，且区间条件于当时算力斜率，遗漏该斜率不确定性。不能把50%覆盖当成真实预测概率。

回测误差大，正式表保留MC分位数，另增加expanded_risk_low/high：两侧各增加$1.96\\,RMSE_{{1step}}\\sqrt h+\\Delta_{{C8-anchor}}$。RMSE取同类型可用线性/机制回测中的最大值，C8项只加pretrained。sqrt(h)仅为步误差累积压力假设，不是用4折估计出的真实扩散规律；1.96也不赋予其95%覆盖含义。风险包络非常宽，恰表明数据无法提供精确长期区间；不把它隐藏成“可靠95%CI”。

主12/24外推长度/实际历史跨度分别{rm.horizon_history_ratio:.4f}/{r24.horizon_history_ratio:.4f}。固定10000次参数传播缺少未来算法、评测污染、数据质量、算力计量与组织依赖的完整不确定性。均值到前沿迁移、C8覆盖不全、算力发布时间滞后同时限制解释。

Submission/Release对照（共用T0与未来日期，另列固定Submission锚点以隔离系数效应）：

{table(time[['horizon_months','comparison','submission_point','release_point','difference','release_anchor_n','high_time_axis_sensitivity']])}

high_time_axis_sensitivity按差值>主参数区间宽度50%判定；另记录相对扩大包络结果。未达到阈值不撤销冻结PHASE2 time_axis_sensitive=True：宽区间可能掩盖轴间差异，不能据此宣称时间口径稳健。Release池180天仅10条，也未达20。

前沿定义对照：

{table(defs)}

明显差异阈值=.1×pretrained历史A_z IQR，为显式诊断规则，不是统计检验。P90/Top3均触发当前敏感性标记。所有结果风险来自configs/evolution_phase3.yaml：history ratio>.5/1/2；算力外推超历史max的IQR距离>0/1/2；参数区间宽度/max(|点增量|,A_z IQR)>.5/1/2，对应逐级0–3；冻结时间敏感、前沿定义敏感、锚点<20各至少HIGH，超冻结可实现上界VERY_HIGH；总等级取各项最大。无人工个别打分，不是事件发生概率。全部具体维度保留forecast_risk_flags.csv。

11张图位于outputs/figures/question4_phase3，manifest逐图注明用途。所有正文A_z图保留未截断情景；第1图将参数扇形与扩大压力包络分面显示，防止混淆。核心图：

![历史前沿、条件参数扇形和单独的压力包络](../figures/question4_phase3/01_frontier_projection_fan.png)

![逐月条件回溯诊断](../figures/question4_phase3/07_hindcast_observed_predicted.png)

![pretrained、chat、overall分别比较](../figures/question4_phase3/08_pretrained_chat_overall_comparison.png)

![非规模进步的条件补偿](../figures/question4_phase3/10_non_scale_compensation.png)

## 4.13 问题四结论与证据等级

1. **Level A：** 真实S1支持类型分层和稀疏前沿统计。pretrained P95锚点{rm.A_z_anchor:.6f}，180天18条；chat/overall样本较多，但它们不提供pretrained算力机制证据。
2. **Level B：** 冻结回归显示训练规模与能力的条件正关联；时间代理对组织、样本与时间轴敏感，不能声称纯技术因果贡献，历史绝对份额不能用于未来权重。
3. **Level C：** 当前C4子样本的g={growth:.6f}与冻结M2给出C1×T1的12月点情景{rm.A_z_point:.6f}，MC参数区间[{rm['A_z_P2.5']:.6f},{rm['A_z_P97.5']:.6f}]；减半算力增长相对C0降低未来增量，但只是条件场景。
4. **Level D：** 24月点情景{r24.A_z_point:.6f}已超计分公式可实现范围，且历史跨度比{r24.horizon_history_ratio:.3f}；VERY_HIGH风险，不据此宣称真实未来Benchmark达到该值。
5. **Level C/D：** 补偿所需tau可与历史宽Bootstrap范围比较，但无法给未来技术创新的成功概率；C8、直接前沿负趋势与回测大误差使总体结论更保守。
6. **Level A/B负结果：** Loss桥接无可靠CV泛化，前三问Loss没有被转换成第四问能力；问题四全部预测建立于真实冻结Leaderboard能力、资源关联与显式情景假设。

复现：在项目根目录运行 `python -X utf8 -B scripts/22_build_evolution_projection.py --stage all`，随后 `python -X utf8 -B scripts/22_verify_evolution_projection.py`。只重建PHASE3新文件，不执行任何冻结PHASE1/2脚本。所有历史冻结文件SHA256完整核对见phase3/verification_summary.json。本阶段结束，问题四停止；不改前三问。
'''
    report=root/'outputs/report/question4_final.md';report.parent.mkdir(parents=True,exist_ok=True);report.write_text(text,encoding='utf8')

# ============================================================
# Source: scripts/22_build_evolution_projection.py
# Original lines: 1-36
# Purpose: 最终阶段实际执行入口：22_build_evolution_projection.py
# ============================================================
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

