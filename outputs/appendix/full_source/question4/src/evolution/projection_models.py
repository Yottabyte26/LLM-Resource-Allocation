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
