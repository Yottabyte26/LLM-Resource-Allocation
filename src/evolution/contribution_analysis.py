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
