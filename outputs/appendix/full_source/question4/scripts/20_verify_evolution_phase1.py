"""Independent data/formula/sample/band/provenance checks for Question 4 PHASE 1."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
PROJECT_ROOT=Path(__file__).resolve().parents[1]
import hashlib
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr


def main():
    """Verify new outputs and every frozen file, without running prior models."""
    root=PROJECT_ROOT;out=root/'outputs/evolution/phase1';base=root/'real_attachments/C_efficiency_evolution';checks=[]
    def check(name,passed,detail=''):checks.append(dict(check=name,passed=bool(passed),detail=str(detail)))
    def read(name):return pd.read_csv(out/name,float_precision='round_trip')
    raw=pd.read_csv(base/'leaderboard_enhanced.csv',float_precision='round_trip');models=read('evolution_model_table.csv')
    params=json.loads((out/'ability_transform_parameters.json').read_text(encoding='utf8'));metrics=params['metrics'];primary=models[models.S1]
    check('one_row_per_original_Model',models.Model.is_unique and set(models.Model)==set(raw.Model))
    check('S1_subset_S0',(~models.S1|models.S0).all());check('S2_requires_license_evidence',not models.S2.any() and models.license_status.eq('license_unknown').all())
    check('open_weights_Yes_required',models.loc[models.S1,'open_weights'].str.lower().isin(['yes','true']).all())
    check('equal_weights_six',len(params['weights'])==6 and np.allclose(params['weights'],np.ones(6)/6,atol=0,rtol=0))
    columns=[]
    for p in params['tasks']:
        x=primary[p['metric']];L,U=x.quantile([.01,.99]);v=x.clip(L,U);median=v.median();iqr=v.quantile(.75)-v.quantile(.25)
        check('task_parameters_'+p['metric'],np.allclose([L,U,median,iqr],[p['P01'],p['P99'],p['median'],p['IQR']],atol=1e-12,rtol=0))
        z=(models[p['metric']].clip(L,U)-median)/iqr if iqr>0 else np.zeros(len(models))
        columns.append(z);check('all_task_z_'+p['metric'],np.allclose(z,models['z_'+p['metric']],atol=1e-12,rtol=0,equal_nan=True))
    Z=np.column_stack(columns);ability=Z.mean(axis=1)
    check('ability_all_models',np.allclose(ability,models.A_z,atol=1e-12,rtol=0,equal_nan=True))
    display=params['display_mapping'];A100=np.clip((ability-display['P01'])/(display['P99']-display['P01']),0,1)*100
    check('fixed_display_mapping',np.allclose(A100,models.A_100,atol=1e-11,rtol=0,equal_nan=True))
    check('PCA_eligibility',params['pca']['accepted']==bool(min(params['pca']['loadings'])>0 and params['pca']['explained_variance_ratio'][0]>=.5))
    if params['pca']['accepted']:
        scores=(Z-np.array(params['pca']['mean']))@np.array(params['pca']['loadings'])
        check('PCA_fixed_projection',np.allclose(scores,models.A_pca,atol=1e-12,rtol=0,equal_nan=True))
    rawdates=pd.to_datetime(raw['Submission Date'],errors='coerce')
    for r in models.to_dict('records'):
        ix=json.loads(r['source_rows']);g=raw.iloc[ix];date=rawdates[raw.Model==r['Model']].max()
        check('source_Model_'+r['Model'],g.Model.eq(r['Model']).all())
        check('latest_cohort_'+r['Model'],pd.isna(date) or pd.to_datetime(g['Submission Date']).eq(date).all())
        unique=g['#Params (B)'].dropna().unique();expected=unique[0] if len(unique)==1 else np.nan
        check('N_source_no_interpolation_'+r['Model'],pd.isna(expected) and pd.isna(r['N_B']) or expected==r['N_B'])
        check('benchmark_replicate_medians_'+r['Model'],np.allclose(g[metrics].median(),[r[m] for m in metrics],atol=1e-12,rtol=0,equal_nan=True))
    epoch=pd.read_csv(base/'epoch_all_ai_models.csv',float_precision='round_trip')
    for r in models[models.Epoch_match.notna()].to_dict('records'):
        e=epoch.iloc[int(r['epoch_row'])];compute=pd.to_numeric(e['Training compute (FLOP)'],errors='coerce')
        check('compute_observed_source_'+r['Model'],pd.isna(compute) and pd.isna(r['training_compute']) or compute==r['training_compute'])
        if r['D_tokens_source'].startswith('C4'):
            d=pd.to_numeric(e['Training dataset size (total)'],errors='coerce')
            check('D_tokens_confirmed_'+r['Model'],bool(re.search(r'(?i)\btokens?\b',str(e['Dataset size notes']))) and d==r['D_tokens'])
    profile=read('benchmark_profile.csv')
    for m in metrics:
        v=raw[m];p=profile[(profile.scope=='C2_all_evaluations')&(profile.metric==m)].iloc[0]
        check('raw_profile_'+m,np.allclose([len(v),v.mean(),v.std(ddof=1),v.min(),v.median(),v.max()],
            [p.n,p['mean'],p['std'],p['min'],p['median'],p['max']],atol=1e-12,rtol=0))
    audit=read('C8_parse_audit.csv');all_json=sorted((base/'detailed_results').rglob('*.json'))
    check('C8_every_JSON_audited',len(audit)==len(all_json) and set(audit.source_file)=={p.relative_to(root).as_posix() for p in all_json})
    check('C8_selection_parseable',audit.loc[audit.selected,'parse_success'].all() and audit[audit.selected].directory.is_unique)
    for name,g in audit.groupby('directory'):
        valid=g[g.parse_success&g.parse_error.isna()]
        if len(valid):
            newest=pd.to_datetime(valid.run_time,utc=True,format='mixed').max();chosen=g[g.selected]
            check('C8_latest_'+name,len(chosen)==1 and pd.to_datetime(chosen.run_time.iloc[0],utc=True)==newest)
        else:check('C8_no_usable_'+name,not g.selected.any())
    leaves=read('C8_subtask_scores.csv');expected=100*np.maximum(0,(leaves.raw_score-leaves.random_baseline)/(1-leaves.random_baseline))
    check('C8_leaf_chance_normalization',np.allclose(leaves.task_score,expected,atol=1e-12,rtol=0))
    check('C8_real_finite_leaf_scores',leaves.raw_score.between(0,1).all() and np.isfinite(leaves.task_score).all())
    agg=read('C8_model_task_aggregation.csv').set_index('directory')
    for name,g in leaves.groupby('directory'):
        row=agg.loc[name];x=g.task_score.to_numpy();bottom=np.sort(x)[:max(1,int(np.ceil(.2*len(x))))].mean()
        check('C8_imbalance_'+name,abs(row.capability_imbalance-(np.quantile(x,.9)-np.quantile(x,.1)))<1e-10)
        check('C8_tail_'+name,abs(row.bottom20_task_score-bottom)<1e-10)
    matched=read('C8_matched_model_analysis.csv');tails=read('C8_weak_tail_diagnosis.csv')
    for r in tails.to_dict('records'):
        h=matched[matched.unique_model_match&matched.six_complete&matched.A_z.notna()]
        if r['sample']=='S1':h=h[h.S1]
        basis='A_z' if r['ability_basis']=='C2_snapshot' else 'A_z_C8'
        flag=(h[basis]>=h[basis].quantile(.75))&(h.bottom20_task_score<=h.bottom20_task_score.quantile(.25))
        check('weak_tail_'+r['sample']+'_'+basis,int(flag.sum())==r['high_average_weak_tail_count'] and abs(flag.mean()-r['high_average_weak_tail_rate'])<1e-15)
    cb=read('C6_bridge_transformed_records.csv');mapping=['LB_IFEval','LB_BBH','LB_MATH','LB_GPQA','LB_MUSR','LB_MMLU_PRO'];cz=[]
    for field,p in zip(mapping,params['tasks']):cz.append((cb[field].clip(p['P01'],p['P99'])-p['median'])/p['IQR'] if p['IQR']>0 else np.zeros(len(cb)))
    check('C6_same_fixed_ability_transform',np.allclose(np.column_stack(cz).mean(axis=1),cb.A_bridge,atol=1e-12,rtol=0))
    bridge=json.loads((out/'bridge_model.json').read_text(encoding='utf8'));high=cb[cb.comparability=='High'];medium=cb[cb.comparability=='Medium']
    check('bridge_strata_counts',len(high)==bridge['High_count'] and len(medium)==bridge['Medium_count'])
    X=np.column_stack([np.ones(len(high)),high.Val_Loss]);coef=np.linalg.lstsq(X,high.A_bridge,rcond=None)[0]
    check('B0_High_only_fit',np.allclose(coef,bridge['candidate_models']['B0']['coefficients'],atol=1e-12,rtol=0))
    pred=read('bridge_validation_predictions.csv');validation=read('bridge_validation_metrics.csv')
    for (candidate,kind),g in pred.groupby(['candidate','evaluation']):
        errors=g.predicted_A-g.observed_A;v=validation[(validation.candidate==candidate)&(validation.evaluation==kind)].iloc[0]
        total=np.sum((g.observed_A-g.observed_A.mean())**2);R2=1-np.sum(errors**2)/total
        check('validation_'+candidate+'_'+kind,np.allclose([np.sqrt(np.mean(errors**2)),abs(errors).mean(),R2],[v.RMSE,v.MAE,v.R2],atol=1e-12,rtol=0))
    bands=read('loss_benchmark_bridge.csv');bootstrap=read('bridge_bootstrap_parameters.csv')
    check('bootstrap_count_per_candidate',bootstrap.groupby('candidate').size().eq(2000).all())
    check('bootstrap_only_High_indices',all(min(json.loads(x))>=0 and max(json.loads(x))<len(high) for x in bootstrap.source_model_indices))
    check('ordered_uncertainty_bands',(bands.mean_CI_low<=bands.mean_CI_high).all() and (bands.prediction_band_low<=bands.prediction_band_high).all())
    check('bands_High_range_only',bands.Val_Loss.between(*bridge['calibrated_loss_range']).all())
    if bridge['selected_fit_is_constant']:
        check('failed_bridge_not_recommended',not bridge['bridge_valid_for_conversion'] and bridge['bridge_status']=='DIAGNOSTIC_ONLY_FLAT_ISOTONIC')
    c3=pd.read_csv(base/'leaderboard_extended_timeseries.csv');caudit=read('C3_time_series_audit.csv')
    check('C3_all_rows_year_only',len(caudit)==len(c3) and not caudit.full_date_available.any())
    check('C3_historic_not_pooled',not caudit[caudit.Source!='Open LLM Leaderboard'].comparable_main_scale.any())
    monthly=read('monthly_ability_summary.csv')
    for r in monthly.to_dict('records'):
        mask=models[r['sample']]&pd.to_datetime(models['Submission Date'],errors='coerce').dt.to_period('M').astype(str).eq(r['month'])
        if r['model_type']!='all':mask&=models.model_type.eq(r['model_type'])
        x=models.loc[mask,'A_z'];check('month_'+str(tuple(r[k] for k in ['sample','model_type','month'])),
            len(x)==r['n'] and np.allclose([x.median(),x.quantile(.9),x.quantile(.95),x.max()],[r['median_A'],r['P90_A'],r['P95_A'],r['max_A']],atol=1e-12,rtol=0))
    report=(root/'outputs/report/question4_phase1_data_ability_bridge.md').read_text(encoding='utf8')
    for target in re.findall(r'\]\(([^)]+)\)',report):
        if target.startswith('http') or target.endswith('verification_summary.json'):continue
        check('report_link_'+target,(root/'outputs/report'/target).resolve().is_file())
    manifest=read('figure_manifest.csv');check('twelve_figures',len(manifest)==12 and all((root/p).stat().st_size>10000 for p in manifest.file))
    for manifest_name in ['input_manifest.csv','implementation_manifest.csv']:
        for r in read(manifest_name).to_dict('records'):
            with (root/r['file']).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            check('manifest_'+r['file'],digest==r['sha256'])
    for p in out.glob('*.csv'):check('CSV_parse_'+p.name,p.stat().st_size>0 and len(pd.read_csv(p).columns)>0)
    frozen=json.loads((out/'frozen_before_question4_phase1.json').read_text(encoding='utf8'));changes=[]
    for relative,item in frozen.items():
        p=root/relative
        if not p.is_file():changes.append(relative);continue
        with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if p.stat().st_size!=item['size'] or digest!=item['sha256']:changes.append(relative)
    check('all_earlier_files_frozen',not changes,changes)
    summary=json.loads((out/'execution_summary.json').read_text(encoding='utf8'))
    check('scope_no_prediction_or_decomposition',not summary['future_prediction_performed'] and not summary['contribution_decomposition_performed'] and not summary['old_question4_results_reused'])
    frame=pd.DataFrame(checks);frame.to_csv(out/'independent_verification.csv',index=False,encoding='utf-8-sig')
    result=dict(checks=len(frame),passed=int(frame.passed.sum()),failed=int((~frame.passed).sum()),frozen_files_checked=len(frozen),frozen_files_changed=changes,
        future_forecast_performed=False,contribution_decomposition_performed=False)
    (out/'verification_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))
    if not frame.passed.all():print(frame[~frame.passed].to_string(index=False));raise ValueError('Question 4 PHASE 1 verification failed')


if __name__=='__main__':main()
