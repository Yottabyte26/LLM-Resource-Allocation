"""Independent numerical checks for part 3 final; never refit frozen models."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import yaml
from scipy.stats import spearmanr,trim_mean

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mixture.io import load_datasets,signatures
from mixture.augmented import candidate,metadata
from quality.io import sha256_file
from quality.vector_diagnostics import write_rows
OUT=ROOT/'outputs/mixture';STAGE=OUT/'stage2'


def rows(path):
    """Require well-formed nonempty result CSVs."""
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);data=list(reader)
    if not data or any(None in r or any(v is None for v in r.values()) for r in data):raise ValueError(f'Malformed CSV {path}')
    return data


def metric(y,p):
    """Direct metric formulas, independent of production scores helper."""
    e=y-p
    return dict(RMSE=float(np.sqrt(np.mean(e*e))),MAE=float(np.mean(abs(e))),
        R2=float(1-np.sum(e*e)/np.sum((y-y.mean())**2)),Pearson=float(np.corrcoef(y,p)[0,1]),
        Spearman=float(spearmanr(y,p).statistic))


def predict(P,meta,augmented=False):
    """Rebuild ILR using sequential geometric means, then saved scaling/coefficients."""
    p=np.asarray(P)/np.sum(P,axis=1,keepdims=True);z=p==0
    rp=p*(1-z.sum(axis=1)*meta['zero_delta'])[:,None];rp[z]=meta['zero_delta'];lp=np.log(rp)
    ilr=np.column_stack([np.sqrt(k/(k+1))*(lp[:,:k].mean(axis=1)-lp[:,k]) for k in range(1,17)])
    if augmented:
        x=np.column_stack([ilr,p@meta['q_values']])
        if meta['k_values'] is not None:x=np.column_stack([x,p@meta['k_values']])
    else:x=ilr
    return meta['y_mean']+meta['y_scale']*((x-np.array(meta['X_mean']))/np.array(meta['X_scale'])@np.array(meta['coefficients'])+meta['intercept'])


def modified(p,targets,step):
    """Independent loop implementation of proportional-donor perturbation."""
    out=np.zeros(17);donor=sum(p[k] for k in range(17) if k not in targets)
    for k in range(17):out[k]=p[k]+step if k in targets else p[k]*(1-len(targets)*step/donor)
    if min(out)<0 or abs(sum(out)-1)>1e-12:raise AssertionError('Simplex perturbation incorrect')
    # Positive ratios among donors stay fixed, zero donor shares stay zero.
    pos=[k for k in range(17) if k not in targets and p[k]>0]
    if not np.allclose(out[pos]/p[pos],out[pos[0]]/p[pos[0]],atol=1e-14):raise AssertionError('Donor ratios changed')
    return out


def main():
    """Check saved equations, training isolation, MC and all frozen-file hashes."""
    checks=[]
    def check(name,condition,detail):
        if not condition:raise AssertionError(f'{name}: {detail}')
        checks.append(dict(check=name,passed=True,detail=detail))
    ds,names,loss_names=load_datasets(ROOT);train=ds['train_1M'];P=train['P'];y=train['L_mean'];ids=train['indices'];idx={v:i for i,v in enumerate(ids)}
    protected=json.loads((STAGE/'protected_before.json').read_text(encoding='utf-8'))
    check('all prior frozen files unchanged',all((ROOT/p).exists() and sha256_file(ROOT/p)==digest for p,digest in protected.items()),f'{len(protected)} files')
    frozen=json.loads((STAGE/'freeze_before_extrapolation.json').read_text(encoding='utf-8'))
    check('parameters stayed frozen during extrapolation',all(sha256_file(ROOT/p)==h for p,h in frozen['model_files'].items()),'No calibration; all recorded model hashes unchanged')
    metas=json.loads((OUT/'final_model_metadata.json').read_text(encoding='utf-8'))
    base_metas={r['model']:r for r in metas if r.get('response')=='L_mean'}
    if not base_metas:
        # metadata is a list; official existing entries use a response field in stage 1.
        raise ValueError('Inspect frozen metadata response schema before reconstructing')
    for label in ['M1_ElasticNet','M1_Ridge']:
        model=joblib.load(OUT/'models'/f'L_mean_{label}.joblib')
        check(label+' saved independent predictor',np.allclose(predict(P,base_metas[label]),model.predict(P),atol=1e-12,rtol=0),'512 actual A4 recipes')
    validation=rows(STAGE/'model_comparison_validation.csv');pred_rows=rows(STAGE/'model_comparison_test_predictions.csv')
    for r in validation:
        d=ds[r['dataset']];pr=[a for a in pred_rows if a['model']==r['model'] and a['dataset']==r['dataset']]
        actual_metrics=metric(d['L_mean'],np.array([float(a['predicted_loss']) for a in pr]))
        check(r['model']+' '+r['dataset']+' test metrics',
              all(abs(actual_metrics[k]-float(r[k]))<1e-10 for k in ['RMSE','MAE','R2','Pearson','Spearman']),
              f"{len(pr)} direct uncalibrated predictions")
    for r in rows(OUT/'scale_transfer_diagnostics.csv'):
        meta=base_metas[r['model']];d=ds[r['dataset']];e=d['L_mean']-predict(d['P'],meta)
        correct=abs(e.mean()-float(r['mean_bias']))<1e-12 and abs(np.median(e)-float(r['median_bias']))<1e-12 and abs(np.sqrt(np.mean((e-e.mean())**2))-float(r['centered_RMSE_diagnostic_only']))<1e-12
        check(r['model']+' '+r['dataset']+' residual diagnosis',correct,'e=y-p; centered metric diagnostic only')
    baseline_rows=rows(STAGE/'representative_baselines.csv');baseline={r['baseline']:np.array([float(r[n]) for n in names]) for r in baseline_rows}
    check('five distinct representative baselines',len(baseline)==5,'center, median, P10/P50/P90 predicted recipes')
    marginal=rows(STAGE/'domain_marginal_effect_cases.csv')
    for r in marginal:
        p=baseline[r['baseline']];j=names.index(r['mixture_domain']);xp=modified(p,[j],float(r['step']))
        expected=np.diff(predict(np.stack([p,xp]),base_metas[r['model']]))[0]
        if abs(expected-float(r['effect']))>1e-12:raise AssertionError('Single perturbation reconstruction failed')
    check('all single perturbations independent reconstruction',len(marginal)==510,'2 models x 3 steps x 5 baselines x 17 domains; donor ratios and simplex checked')
    joint=rows(STAGE/'domain_joint_effect_cases.csv')
    for r in joint:
        p=baseline[r['baseline']];j=names.index(r['domain_1']);k=names.index(r['domain_2']);step=float(r['step'])
        pred=predict(np.stack([p,modified(p,[j],step),modified(p,[k],step),modified(p,[j,k],step)]),base_metas[r['model']])
        expected=pred[3]-pred[1]-pred[2]+pred[0]
        if abs(expected-float(r['joint_perturbation_deviation']))>1e-12:raise AssertionError('Joint deviation reconstruction failed')
    check('all joint deviations independent reconstruction',len(joint)==4080,'2 x 3 x 5 x 136 pairs; not causal interaction')
    aggregated=rows(OUT/'domain_marginal_effects.csv')
    for r in aggregated:
        vals=np.array([float(a['effect']) for a in marginal if a['model']==r['model'] and a['step']==r['step'] and a['mixture_domain']==r['mixture_domain']])
        for key,value in [('mean_effect',vals.mean()),('median_effect',np.median(vals)),('std_effect',vals.std(ddof=1)),('min_effect',vals.min()),('max_effect',vals.max())]:
            if abs(value-float(r[key]))>1e-12:raise AssertionError('Marginal summary incorrect')
    check('all marginal summaries',len(aggregated)==102,'Actual five-baseline mean/median/std/min/max')
    outer={r['index']:int(r['outer_holdout_fold']) for r in rows(OUT/'outer_fold_assignments.csv')}
    for label in ['M3_Qp','M4_Qp_Kp','M3_alternative_mapping']:
        meta=json.loads((STAGE/'models'/f'{label}.json').read_text(encoding='utf-8'));model=joblib.load(STAGE/'models'/f'{label}.joblib')
        check(label+' all-training boundary',meta['fitting_recipe_indices']==ids and meta['n_fit']==512 and meta['fitting_dataset']=='A4/A5_only','All fits restricted to original train recipes')
        check(label+' independent saved predictor',np.allclose(predict(P,meta,True),model.predict(P),rtol=0,atol=1e-12),'No load-time refitting')
        raw=meta['y_scale']*np.array(meta['coefficients'])/np.array(meta['X_scale'])
        check(label+' original-unit gamma',abs(raw[16]-meta['gamma_original_loss_per_Qp'])<1e-12,'gamma=y_scale*beta/X_scale')
        outer_metas=json.loads((STAGE/'models'/f'{label}_outer.json').read_text(encoding='utf-8'))
        oof=np.empty(512)
        for f,m in enumerate(outer_metas,1):
            hold=[i for i,index in enumerate(ids) if outer[index]==f];training=[i for i,index in enumerate(ids) if outer[index]!=f]
            check(label+f' outer{f} training isolation',m['fitting_recipe_indices']==[ids[i] for i in training],'Identical frozen M1 outer holdouts; no heldout fit IDs')
            tr=P[training]/P[training].sum(axis=1,keepdims=True);delta=.5*tr[tr>0].min()
            check(label+f' outer{f} zero parameter',abs(delta-m['zero_delta'])<1e-15,'Fit only current outer training fold')
            # Independently audit both feature scaling and target scaling.
            rp=tr*(1-(tr==0).sum(axis=1)*delta)[:,None];rp[tr==0]=delta
            lp=np.log(rp);x=np.column_stack([np.sqrt(k/(k+1))*(lp[:,:k].mean(axis=1)-lp[:,k]) for k in range(1,17)])
            x=np.column_stack([x,tr@m['q_values']])
            if m['k_values'] is not None:x=np.column_stack([x,tr@m['k_values']])
            check(label+f' outer{f} feature and target scalers',np.allclose(x.mean(axis=0),m['X_mean'],atol=1e-12) and np.allclose(x.std(axis=0),m['X_scale'],atol=1e-12) and abs(y[training].mean()-m['y_mean'])<1e-12,'No fold leakage in X/y scales')
            oof[hold]=predict(P[hold],m,True)
        actual=rows(STAGE/f'{label}_OOF_predictions.csv')
        check(label+' OOF reconstruction',np.allclose(oof,[float(r['predicted_loss']) for r in actual],atol=1e-12,rtol=0),'All 512 heldout predictions independently reconstructed')
    mapping=rows(OUT/'domain_quality_mapping.csv')
    quality={r['domain']:r for r in rows(ROOT/'outputs/quality/domain_quality_summary_A1.csv')}
    q=np.array([float(r['Q_domain']) for r in mapping])
    check('mapping provenance retained',sum(r['mapping_type']=='inferred' for r in mapping)==11 and all(r['A16_quality_domain']=='(none)' and r['researcher_assumption']=='True' for r in mapping if r['mapping_type']=='inferred'),'No official inferred mapping invented')
    check('locked domain quality values',all(float(r['Q_domain'])==float(quality[r['quality_domain']]['Q_domain']) for r in mapping),'No Q or CI recomputation')
    kp=rows(STAGE/'domain_conflict_trimmed_summary_A1.csv');kv={r['domain']:float(r['K_domain']) for r in kp};k=np.array([kv[r['quality_domain']] for r in mapping])
    for label,d in ds.items():
        desc=[r for r in rows(OUT/'recipe_quality_conflict_descriptors.csv') if r['dataset']==label];closed=d['P']/d['P'].sum(axis=1,keepdims=True)
        check(label+' descriptors',np.allclose(closed@q,[float(r['Q_p']) for r in desc],atol=1e-14,rtol=0) and np.allclose(closed@k,[float(r['K_p']) for r in desc],atol=1e-14,rtol=0),'Original closed shares, not pseudo-mass replaced shares')
    existing_k={domain:[] for domain in kv}
    with (ROOT/'outputs/conflict/A1_sample_conflict.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):existing_k[r['domain']].append(float(r['K']))
    for domain,v in existing_k.items():
        values=np.sort(v);trim=int(len(v)*.1);expected=values[trim:len(v)-trim].mean()
        check(domain+' existing K trim',abs(expected-kv[domain])<1e-14,f'n={len(v)}; no K refit')
    mc=rows(OUT/'quality_mapping_monte_carlo_results.csv');draws=rows(STAGE/'mapping_MC_draws.csv');a=yaml.safe_load((ROOT/'configs/mixture_analysis.yaml').read_text(encoding='utf-8'))
    rng=np.random.default_rng(2026);draw_matrix=np.tile(q,(500,1))
    for j,r in enumerate(mapping):
        if r['mapping_type']=='inferred':draw_matrix[:,j]=rng.uniform(float(r['Q_CI_low']),float(r['Q_CI_high']),size=500)
    check('500 reproducible MC draws',len(mc)==len(draws)==500 and np.array_equal(draw_matrix,np.array([[float(r[n]) for n in names] for r in draws])),'seed 2026; independent uniform draws within inferred proxy CIs only')
    settings=yaml.safe_load((ROOT/'configs/mixture_modeling.yaml').read_text(encoding='utf-8'))
    m3meta=json.loads((STAGE/'models/M3_Qp.json').read_text(encoding='utf-8'));m3outer=json.loads((STAGE/'models/M3_Qp_outer.json').read_text(encoding='utf-8'))
    # Refit only the permitted new auxiliary M3, independently spot-check three draws.
    for rep in [0,249,499]:
        q_mc=draw_matrix[rep];model,_=candidate(q_mc,None,settings);model.set_params(**m3meta['parameters']).fit(P,y)
        g=model.transformer_.scale_[0]*model.regressor_.named_steps['regression'].coef_[16]/model.regressor_.named_steps['scale'].scale_[16]
        oof=np.empty(len(y))
        for f,m in enumerate(m3outer,1):
            tr=[i for i,index in enumerate(ids) if outer[index]!=f];va=[i for i,index in enumerate(ids) if outer[index]==f]
            fold_model,_=candidate(q_mc,None,settings);fold_model.set_params(**m['parameters']).fit(P[tr],y[tr]);oof[va]=fold_model.predict(P[va])
        check(f'MC repetition {rep+1} gamma and OOF',abs(g-float(mc[rep]['gamma']))<1e-12 and abs(metric(y,oof)['RMSE']-float(mc[rep]['CV_RMSE']))<1e-12,'Independent selected-parameter train-only refit; frozen M1 untouched')
        for label in ['test_1M','test_60M','test_1B']:
            metrics=metric(ds[label]['L_mean'],model.predict(ds[label]['P']))
            check(f'MC repetition {rep+1} '+label,abs(metrics['RMSE']-float(mc[rep][label+'_RMSE']))<1e-12 and abs(metrics['Spearman']-float(mc[rep][label+'_Spearman']))<1e-12,'Fixed predictor application only')
    single=rows(OUT/'train_domain_vs_validation_loss_effects.csv')
    check('13 validation-domain matrix coverage',len(single)==442 and len({r['training_domain'] for r in single})==17 and len({r['validation_loss'] for r in single})==13,'26 first-order auxiliary fits; 17x13 per model')
    xp=np.stack([modified(baseline[r['baseline']],[j],.01) for r in baseline_rows for j in range(17)])
    x=np.stack([baseline[r['baseline']] for r in baseline_rows])
    for loss in loss_names:
        for label in ['M1_ElasticNet','M1_Ridge']:
            model=joblib.load(STAGE/'single_loss_models'/f"{loss.removeprefix('metric/the_pile_').removesuffix('_val_loss')}_{label}.joblib")
            effect=model.predict(xp).reshape(5,17)-model.predict(x)[:,None]
            vals=[next(float(r['mean_effect']) for r in single if r['model']==label and r['validation_loss']==loss and r['training_domain']==n) for n in names]
            check(loss+' '+label+' perturbation matrix',np.allclose(effect.mean(axis=0),vals,rtol=0,atol=1e-12),'All 17 mean effects from five real/derived A4 baselines')
    estpred=rows(OUT/'estimated_scale_predictions.csv')
    lookup={key:i for i,key in enumerate(signatures(P))}
    for label in ['estimated_10B','estimated_70B']:
        d=ds[label];positions=[lookup[key] for key in signatures(d['P'])]
        check(label+' recipe subset',len(positions)==63,'Composition-vector matches, not cross-table index matching')
        for model in ['M1_ElasticNet','M1_Ridge','M3_Qp']:
            pr=[r for r in estpred if r['dataset']==label and r['model']==model]
            check(label+' '+model+' matched training indices',all(r['matched_train_index']==ids[positions[i]] for i,r in enumerate(pr)),'Estimated Loss is not independent observed holdout')
            if model=='M3_Qp':
                meta=json.loads((STAGE/'models/M3_Qp.json').read_text(encoding='utf-8'));pred=predict(d['P'],meta,True)
            else:pred=predict(d['P'],base_metas[model])
            check(label+' '+model+' estimated predictions',np.allclose(pred,[float(r['uncalibrated_prediction']) for r in pr],rtol=0,atol=1e-12),
                  'Independent predictor reconstruction and actual matched table rows')
            score=metric(d['L_mean'],pred)
            exported=next(r for r in rows(OUT/'estimated_scale_consistency.csv') if r['model']==model and r['dataset']==label)
            check(label+' '+model+' estimated metrics',all(abs(score[k]-float(exported[k]))<1e-12 for k in score),
                  'Negative rank correlation retained with its actual sign')
        source=next(r for r in rows(OUT/'matched_recipe_rank_consistency.csv') if r['comparison']=='observed_train_1M_vs_'+label)
        check(label+' matched observed/estimated ranks',abs(spearmanr(y[positions],d['L_mean']).statistic-float(source['Spearman']))<1e-12,
              'Real 1M loss also opposes estimated macro ranking; no sorting or alignment correction')
    policy=json.loads((STAGE/'final_model_policy.json').read_text(encoding='utf-8'))
    check('main model and K rule',policy['selected_model'] in ['M1_ElasticNet','M3_Qp'] and not policy['Kp_retained'] and policy['no_test_calibration'],'No quadratic main model; no conflict penalty')
    check('all old files still unchanged after verification',all(sha256_file(ROOT/p)==h for p,h in protected.items()),f'{len(protected)} files')
    write_rows(STAGE/'independent_verification.csv',checks)
    print(json.dumps(dict(passed_checks=len(checks),protected_files=len(protected),selected_model=policy['selected_model']),ensure_ascii=True))


if __name__=='__main__':main()
