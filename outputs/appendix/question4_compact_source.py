# ============================================================
# Source: src/evolution/common.py
# Original lines: 2-13
# Purpose: 模型名规范、CSV和相关系数辅助
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: src/evolution/common.py
# Original lines: 27-33
# Purpose: 模型名规范、CSV和相关系数辅助
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def normalized_name(value):
    """Only case, whitespace, known URL prefixes and trailing slash changes."""
    if pd.isna(value):return ''
    s=str(value).strip().lower()
    for prefix in ['https://huggingface.co/','http://huggingface.co/','huggingface.co/','models/']:
        if s.startswith(prefix):s=s[len(prefix):]
    return re.sub(r'\s+','',s).rstrip('/')

# ============================================================
# Source: src/evolution/metadata.py
# Original lines: 2-8
# Purpose: C1/C2去重、S0/S1类型证据及C4元数据匹配
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
import json
import logging
import re
from difflib import get_close_matches
import numpy as np
import pandas as pd
from .common import FILES,csv,dump,normalized_name,table

# ============================================================
# Source: src/evolution/metadata.py
# Original lines: 54-157
# Purpose: C1/C2去重、S0/S1类型证据及C4元数据匹配
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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
# Original lines: 2-17
# Purpose: 固定六Benchmark Winsor/median/IQR、A_z、PCA/Average敏感性
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: src/evolution/ability.py
# Original lines: 71-183
# Purpose: 固定六Benchmark Winsor/median/IQR、A_z、PCA/Average敏感性
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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
# Original lines: 2-176
# Purpose: C8 JSON全遍历、损坏文件记录与偏科/弱尾
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: src/evolution/bridge.py
# Original lines: 2-42
# Purpose: B0/B1/B2、LOMO/bootstrap和桥接失败诊断
# B0/B1/B2 are failed bridge candidates. Preserve bridge_valid_for_conversion=False; do not convert Loss to Benchmark.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: src/evolution/bridge.py
# Original lines: 59-152
# Purpose: B0/B1/B2、LOMO/bootstrap和桥接失败诊断
# B0/B1/B2 are failed bridge candidates. Preserve bridge_valid_for_conversion=False; do not convert Loss to Benchmark.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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
# Source: src/evolution/contribution_regression.py
# Original lines: 2-189
# Purpose: OLS/HC3/Huber、LOOCV、Cook/VIF、双因素Shapley与5000模型bootstrap
# Huber, quadratic time and month FE are sensitivity/candidate comparisons. Shapley signed components are not future growth weights.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: src/evolution/contribution_analysis.py
# Original lines: 2-10
# Purpose: T0-T3、M1/M2-C/M3、组织/时间轴对照、技术残差和分解执行
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from .common import csv,table
from .contribution_regression import (fit_regression,diagnostics,contribution,bootstrap,bootstrap_summary,design)

# ============================================================
# Source: src/evolution/contribution_analysis.py
# Original lines: 25-276
# Purpose: T0-T3、M1/M2-C/M3、组织/时间轴对照、技术残差和分解执行
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: src/evolution/projection_data.py
# Original lines: 2-193
# Purpose: 实际执行Phase3：锚点、C4去重滚动前沿、增长bootstrap
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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
# Original lines: 2-9
# Purpose: 冻结成对系数的情景Monte Carlo、回测、补偿、风险标记
# Executed scenario projection, not a reliable long-term forecast. Frozen full-period calibration/model lookahead is disclosed. No Loss bridge conversion.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
import json
import logging
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from .common import csv
from .contribution_analysis import write_json
from .projection_data import SCOPES,select_scope,frontier_stats,select_compute_window,compute_grid,rolling_compute,slope

# ============================================================
# Source: src/evolution/projection_models.py
# Original lines: 17-249
# Purpose: 冻结成对系数的情景Monte Carlo、回测、补偿、风险标记
# Executed scenario projection, not a reliable long-term forecast. Frozen full-period calibration/model lookahead is disclosed. No Loss bridge conversion.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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
# Source: src/evolution/metadata.py
# Original lines: 11-14
# Purpose: C1/C2去重、S0/S1类型证据及C4元数据匹配
# Original call-site excerpt, not a standalone function.
# ============================================================
def inspect_sources(root,cfg):
    """Read seven actual CSVs, compare cores, retain every raw diagnostic."""
    base=root/'real_attachments/C_efficiency_evolution';out=root/'outputs/evolution/phase1'
    data={key:pd.read_csv(base/name,float_precision='round_trip') for key,name in FILES.items()}

