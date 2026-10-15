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
