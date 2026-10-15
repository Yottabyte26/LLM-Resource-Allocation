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
