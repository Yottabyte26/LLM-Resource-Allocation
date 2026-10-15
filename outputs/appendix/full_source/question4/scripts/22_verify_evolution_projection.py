"""Independent numerical reconstruction and exhaustive frozen-file integrity checks."""
import hashlib
import json
import logging
import sys
from pathlib import Path
sys.dont_write_bytecode=True
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/'src'))
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy.stats import theilslopes


def digest(path):
    """Bounded-memory SHA256 for raw and frozen files."""
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main():
    """Do not invoke the production numerical routines to verify their formulas."""
    root=PROJECT_ROOT;out=root/'outputs/evolution/phase3';checks=[]
    def check(name,ok,evidence=''):
        checks.append(dict(check=name,passed=bool(ok),evidence=str(evidence)))
        if not ok:raise AssertionError(name+': '+str(evidence))
    def read(path):return pd.read_csv(path,float_precision='round_trip')
    cfg=yaml.safe_load((root/'configs/evolution_phase3.yaml').read_text(encoding='utf8'))
    snap=json.loads((out/'frozen_before_question4_phase3.json').read_text(encoding='utf8'));integrity=[]
    for relative,expected in snap.items():
        p=root/relative;ok=p.is_file() and p.stat().st_size==expected['size'] and digest(p)==expected['sha256'];integrity.append(dict(file=relative,unchanged=ok))
    check('ALL_FROZEN_FILES_UNCHANGED',all(x['unchanged'] for x in integrity),len(integrity))
    pd.DataFrame(integrity).to_csv(out/'frozen_file_integrity.csv',index=False,encoding='utf-8-sig')
    d=read(root/'outputs/evolution/phase1/evolution_model_table.csv');d=d[d.S1];d['date']=pd.to_datetime(d['Submission Date']);d=d[d.date.notna()&np.isfinite(d.A_z)]
    base=d.date.max();check('T0_LAST_VALID_S1',str(base.date())==read(out/'frontier_anchor.csv').base_date.iloc[0])
    anchors=read(out/'frontier_anchor.csv');members=read(out/'anchor_models.csv');audit=read(out/'anchor_window_audit.csv');ab=read(out/'anchor_bootstrap.csv')
    kinds={'F0_pretrained':'pretrained','F1_chat':'chat_finetuned','F2_overall':None}
    for scope,kind in kinds.items():
        source=d if kind is None else d[d.model_type.eq(kind)]
        for days in cfg['anchor_windows_days']:
            pool=source[source.date.gt(base-pd.Timedelta(days=days))&source.date.le(base)]
            if len(pool)>=20:break
        aa=anchors[anchors.model_scope.eq(scope)];mm=members[members.model_scope.eq(scope)];v=mm.A_z.to_numpy()
        check(scope+'_COUNT_ONLY_WINDOW',aa.window_days.eq(days).all() and aa.n.eq(len(pool)).all() and set(mm.Model)==set(pool.Model))
        for definition in ['P95','P90','Top3']:
            expected=np.sort(v)[-3:].mean() if definition=='Top3' else np.quantile(v,.95 if definition=='P95' else .9)
            check(scope+'_'+definition+'_REAL_ANCHOR',np.isclose(aa[aa.frontier_definition.eq(definition)].A_z_anchor.iloc[0],expected,atol=1e-14))
        boots=ab[ab.model_scope.eq(scope)];check(scope+'_ANCHOR_BOOTSTRAP_5000',len(boots)==5000)
        for _,r in boots.iloc[[0,1234,4999]].iterrows():
            ix=json.loads(r.source_model_indices);check(scope+'_ANCHOR_RESAMPLE_'+str(r.replicate),np.isclose(r.P95,np.quantile(v[ix],.95),atol=1e-14))
    projects=read(out/'compute_model_audit.csv');valid=projects[projects.valid];check('COMPUTE_UNIQUE_TRAINING_PROJECTS',valid.epoch_row.is_unique,len(valid))
    c4=read(root/'real_attachments/C_efficiency_evolution/epoch_all_ai_models.csv')
    check('C4_ORIGINAL_COMPUTE_AND_DATE',all(np.isclose(r.compute_FLOP,float(c4.iloc[int(r.epoch_row)]['Training compute (FLOP)']),rtol=1e-14) and r.date==str(pd.Timestamp(c4.iloc[int(r.epoch_row)]['Publication date']).date()) for _,r in valid.iterrows()))
    gm=json.loads((out/'compute_growth_model.json').read_text(encoding='utf8'));cp=read(out/'compute_bootstrap_models.csv');cp['date']=pd.to_datetime(cp.date);ch=read(out/'compute_frontier_history.csv');ch['date']=pd.to_datetime(ch.date)
    for _,r in ch.iterrows():
        pool=cp[cp.date.between(r.date-pd.DateOffset(months=6),r.date)]
        check('COMPUTE_WINDOW_'+str(r.date.date()),(len(pool)==int(r.n)) and (not len(pool) or np.isclose(np.quantile(pool.lnC,.9),r.P90_lnC,atol=1e-12)))
    non=ch.P90_lnC.notna();gs=theilslopes(ch.loc[non,'P90_lnC'],ch.loc[non,'t_month']).slope
    check('SCIPY_THEILSEN_MAIN',np.isclose(gs,gm['main_growth'],atol=1e-14),gs)
    gb=read(out/'compute_growth_bootstrap.csv');check('COMPUTE_BOOTSTRAP_5000_FINITE',len(gb)==5000 and np.isfinite(gb.TheilSen_P90).all())
    for _,r in gb.iloc[[0,1789,4999]].iterrows():
        sample=cp.iloc[json.loads(r.source_model_indices)];ys=[];tx=[]
        for _,win in ch.iterrows():
            h=sample[sample.date.between(win.date-pd.DateOffset(months=6),win.date)]
            if len(h):ys.append(np.quantile(h.lnC,.9));tx.append(win.t_month)
        check('COMPUTE_RESAMPLE_'+str(r.replicate),np.isclose(theilslopes(ys,tx).slope,r.TheilSen_P90,atol=1e-12))
        b=np.linalg.lstsq(np.column_stack([np.ones(len(tx)),tx]),ys,rcond=None)[0][1]
        check('OLS_BOOTSTRAP_'+str(r.replicate),np.isclose(b,r.OLS_P90,atol=1e-12))
    joint=read(out/'mc_joint_draws.csv');coeff=read(root/'outputs/evolution/phase2/coefficient_bootstrap.csv');specs=json.loads((root/'outputs/evolution/phase2/model_definitions.json').read_text(encoding='utf8'));spec=next(x for x in specs if x['model']=='M2_C')
    for model,bcol,tcol,repcol in [('M2_C','beta_C','tau_C','coefficient_replicate'),('M2_C_release','beta_C_release','tau_C_release','release_coefficient_replicate')]:
        bs=coeff[coeff.model.eq(model)].set_index('replicate');mapped=bs.loc[joint[repcol].to_list()]
        check(model+'_EXACT_PAIRED_ROW',np.array_equal(mapped.lnC.to_numpy(),joint[bcol].to_numpy()) and np.array_equal(mapped.t_month.to_numpy(),joint[tcol].to_numpy()))
    check('NEGATIVE_TAU_NOT_TRUNCATED',(joint.tau_C<0).any() and (joint.tau_C_release<0).any())
    check('MC_AT_LEAST_10000',len(joint)>=10000,len(joint))
    check('MC_EXACT_GROWTH_ROWS',np.array_equal(gb.set_index('replicate').loc[joint.growth_replicate,'TheilSen_P90'].to_numpy(),joint.g_C.to_numpy()))
    mc=read(out/'frontier_projection_mc.csv');s=read(out/'future_frontier_scenarios.csv');check('MAIN_24_SCENARIOS',len(mc.groupby(['horizon_months','compute_scenario','technology_scenario']))==24,len(mc))
    mapping=json.loads((root/'outputs/evolution/phase1/ability_transform_parameters.json').read_text(encoding='utf8'));m=mapping['display_mapping'];expansion=read(out/'projection_interval_expansion.csv')
    for _,r in s.iterrows():
        scope=r.model_scope;h=r.horizon_months
        if r.model_kind=='frozen_M2_increment':
            cf=cfg['compute_scenario_factors'][r.compute_scenario];tf=cfg['technology_scenario_factors'][r.technology_scenario]
            expected=joint[scope+'_anchor_'+r.frontier_definition].to_numpy()+joint.beta_C.to_numpy()*joint.g_C.to_numpy()*cf*h+joint.tau_C.to_numpy()*tf*h
            point=r.A_z_anchor+spec['beta'][1]*gs*cf*h+spec['beta'][2]*tf*h
            check('SCENARIO_POINT_'+str(r.name),np.isclose(r.A_z_point,point,atol=1e-13))
            if r.frontier_definition=='P95':
                z=mc[mc.horizon_months.eq(h)&mc.compute_scenario.eq(r.compute_scenario)&mc.technology_scenario.eq(r.technology_scenario)].sort_values('draw_id')
                check('MAIN_MC_DRAW_'+str(r.name),len(z)==len(joint) and np.allclose(z.A_z,expected,atol=1e-13,rtol=1e-13))
        else:
            factor=cfg['direct_scenario_factors'][r.technology_scenario];expected=joint[scope+'_anchor_P95'].to_numpy()+joint[scope+'_slope'].to_numpy()*factor*h
            check('NO_CHAT_OR_OVERALL_M2_'+str(r.name),scope=='F0_pretrained' or r.compute_scenario=='benchmark_only')
        q=np.quantile(expected,[.025,.1,.5,.9,.975]);actual=r[['A_z_P2.5','A_z_P10','A_z_median','A_z_P90','A_z_P97.5']].to_numpy(dtype=float)
        check('SCENARIO_PERCENTILES_'+str(r.name),np.allclose(q,actual,rtol=1e-13,atol=1e-13))
        check('FORECAST_CALENDAR_DATE_'+str(r.name),r.forecast_date==str((base+pd.DateOffset(months=int(h))).date()))
        display=100*(r.A_z_median-m['P01'])/(m['P99']-m['P01'])
        check('FIXED_DISPLAY_MAPPING_'+str(r.name),np.isclose(display,r.A_100_unclipped,atol=1e-12) and np.isclose(np.clip(display,0,100),r.A_100_clipped_for_plot))
        e=expansion[expansion.model_scope.eq(scope)].iloc[0];buffer=1.96*e.one_step_RMSE*np.sqrt(h)+e.structure_shift
        check('EXPLICIT_RISK_EXPANSION_'+str(r.name),np.isclose(r.expanded_risk_low,q[0]-buffer) and np.isclose(r.expanded_risk_high,q[-1]+buffer))
    hp=read(out/'hindcast_predictions.csv');hm=read(out/'hindcast_metrics.csv')
    for _,r in hm.iterrows():
        z=hp[hp.model_scope.eq(r.model_scope)&hp.method.eq(r.method)&hp.predicted.notna()];error=z.predicted-z.observed
        check('HINDCAST_METRICS_'+r.model_scope+r.method,len(z)==r.n and np.isclose(error.abs().mean(),r.MAE) and np.isclose(np.sqrt((error**2).mean()),r.RMSE) and np.isclose(z.covered.mean(),r.coverage))
    check('LOOKAHEAD_DISCLOSED',not hp.strict_temporal_out_of_sample.any() and hp.ability_calibration_uses_full_period.all() and hp.frozen_coefficient_uses_future_models.any())
    losses=read(out/'slowdown_ability_loss.csv')
    check('SLOWDOWN_SIGNED_IDENTITY',all(np.isclose(r.point_loss,spec['beta'][1]*gs*(1-cfg['compute_scenario_factors'][r.compute_scenario])*r.horizon_months) for _,r in losses.iterrows()))
    compensation=read(out/'technology_compensation.csv');check('CONDITIONAL_COMPENSATION_IDENTITY',all(np.isclose(r.tau_required_point,.5*spec['beta'][2]+spec['beta'][1]*gs*(1-r.compute_factor)) for _,r in compensation.iterrows()))
    flags=read(out/'forecast_risk_flags.csv');severity=flags[[c for c in flags if c.endswith('_severity')]].max(axis=1)
    check('RISK_MAX_RULE',np.array_equal(np.array(['LOW','MEDIUM','HIGH','VERY_HIGH'])[severity.to_numpy(dtype=int)],flags.risk_level.to_numpy()))
    check('SCENARIO_RISK_JOIN',np.array_equal(s.risk_level.to_numpy(),flags.risk_level.to_numpy()))
    figures=read(out/'figure_manifest.csv');check('TEN_OR_MORE_FIGURES',len(figures)>=10,len(figures))
    for relative in figures.file:
        with Image.open(root/relative) as im:im.verify()
    check('ALL_FIGURES_PARSE',True)
    report=(root/'outputs/report/question4_final.md').read_text(encoding='utf8');check('REPORT_13_SECTIONS',all(('## 4.'+str(i)+' ') in report for i in range(1,14)))
    check('BRIDGE_FAILURE_PRESERVED','bridge_valid_for_conversion=False' in report)
    inputs=['outputs/evolution/phase1/evolution_model_table.csv','outputs/evolution/phase1/ability_transform_parameters.json','outputs/evolution/phase1/bridge_validation_metrics.csv','outputs/evolution/phase1/bridge_model.json',
        'outputs/evolution/phase1/C8_weak_tail_diagnosis.csv','outputs/evolution/phase2/model_definitions.json','outputs/evolution/phase2/coefficient_bootstrap.csv','outputs/evolution/phase2/model_coefficients.csv','outputs/evolution/phase2/model_performance.csv','outputs/evolution/phase2/contribution_decomposition.csv',
        'real_attachments/C_efficiency_evolution/epoch_all_ai_models.csv','real_attachments/C_efficiency_evolution/leaderboard_extended_timeseries.csv']
    pd.DataFrame([dict(file=p,size=(root/p).stat().st_size,sha256=digest(root/p),read_only=True) for p in inputs]).to_csv(out/'input_manifest.csv',index=False,encoding='utf-8-sig')
    code=['configs/evolution_phase3.yaml','scripts/22_build_evolution_projection.py','scripts/22_verify_evolution_projection.py','src/evolution/projection_data.py','src/evolution/projection_models.py','src/evolution/projection_reporting.py']
    pd.DataFrame([dict(file=p,size=(root/p).stat().st_size,sha256=digest(root/p)) for p in code]).to_csv(out/'implementation_manifest.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(out/'independent_verification.csv',index=False,encoding='utf-8-sig')
    result=dict(verified=True,checks=len(checks),frozen_files_checked=len(integrity),frozen_files_modified=0,figures=len(figures),scenario_rows=len(s),main_MC_draws=len(mc),
        strict_temporal_OOS_claimed=False,previous_models_refitted=False,loss_bridge_used=False,question4_complete=True)
    (out/'verification_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
