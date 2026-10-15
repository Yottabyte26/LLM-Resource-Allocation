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
