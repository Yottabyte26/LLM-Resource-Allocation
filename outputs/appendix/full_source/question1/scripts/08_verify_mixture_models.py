"""Independent reconstruction of saved ILR predictors, metrics and CV split audits."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import sys
import joblib
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mixture.io import load_datasets,signatures
from quality.io import sha256_file
from quality.vector_diagnostics import write_rows
OUT=ROOT/'outputs/mixture'


def rows(name):
    """Parse derived numerical tables only."""
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def replaced(P,delta):
    """Independent positive-mass replacement, preserving nonzero ratios."""
    p=P/P.sum(axis=1,keepdims=True);zeros=p==0;out=p.copy()
    for i in range(len(p)):
        out[i,~zeros[i]]=p[i,~zeros[i]]*(1-int(zeros[i].sum())*delta)
        out[i,zeros[i]]=delta
    return out


def features(P,meta):
    """Compute ILR by the geometric-mean formula, without sklearn pipeline."""
    logp=np.log(replaced(P,meta['zero_delta']))
    x=np.column_stack([np.sqrt(k/(k+1))*(logp[:,:k].mean(axis=1)-logp[:,k]) for k in range(1,17)])
    powers=np.array(meta['feature_powers'],dtype=int)
    return np.prod(x[:,None,:]**powers[None,:,:],axis=2)


def predict(P,meta):
    """Reconstruct candidate from persisted numerical parameters only."""
    if meta['model']=='M0':return np.full(len(P),meta['constant'])
    f=features(P,meta);standard=(f-np.array(meta['X_mean']))/np.array(meta['X_scale'])
    return meta['y_mean']+meta['y_scale']*(standard@np.array(meta['coefficients'])+meta['intercept'])


def metrics(y,p):
    """Independent metric formulas; constant correlations remain undefined."""
    residual=p-y;variance=np.sum((y-y.mean())**2);defined=np.ptp(p)>0 and np.ptp(y)>0
    return {'RMSE':float(np.sqrt(np.mean(residual**2))),'MAE':float(np.mean(abs(residual))),
        'R2':float(1-np.sum(residual**2)/variance),
        'Pearson':float(np.corrcoef(y,p)[0,1]) if defined else None,
        'Spearman':float(spearmanr(y,p).statistic) if defined else None,'prediction_bias_mean':float(residual.mean())}


def main():
    checks=[]
    def check(name,condition,detail):
        if not condition:raise AssertionError(f'{name}: {detail}')
        checks.append({'check':name,'passed':True,'detail':detail})
    datasets,names,loss_names=load_datasets(ROOT);train=datasets['train_1M'];P=train['P'];ids=train['indices'];lookup={v:i for i,v in enumerate(ids)}
    response=rows('loss_response_summary.csv')
    for label,d in datasets.items():
        values=[r for r in response if r['dataset']==label]
        check(label+' response count',len(values)==len(d['P']),str(len(values)))
        for i,r in enumerate(values):
            losses=[float(r[n]) for n in loss_names]
            if r['index']!=d['indices'][i] or not np.array_equal(np.array(losses),d['losses'][i]):raise AssertionError('Response/raw aligned loss mismatch')
            if abs(math.fsum(losses)/13-float(r['L_mean']))>3e-15 or abs(math.fsum(sorted(losses)[1:-1])/11-float(r['L_trim']))>3e-15:
                raise AssertionError('Response aggregation incorrect')
    check('all 13-domain response aggregates',True,f'{len(response)} actual recipes; no missing subset means')
    transform=json.loads((OUT/'composition_transform_A4.json').read_text(encoding='utf-8'));h=np.array(transform['basis'])
    closed=P/P.sum(axis=1,keepdims=True);delta=.5*closed[closed>0].min()
    check('A4-only zero parameter',abs(delta-transform['zero_delta'])<1e-16,'closed A4 minimum positive; estimated/test data excluded')
    check('ILR orthonormal basis',np.allclose(h@h.T,np.eye(16),atol=2e-15) and np.allclose(h@np.ones(17),0,atol=2e-15),'HHt=I, H1=0, shape16x17')
    for label in ['train_1M','test_1M','test_60M','test_1B']:
        d=datasets[label];rp=replaced(d['P'],delta);ilog=np.log(rp)
        expected=np.column_stack([np.sqrt(k/(k+1))*(ilog[:,:k].mean(axis=1)-ilog[:,k]) for k in range(1,17)])
        saved=rows(label+'_ILR_features.csv');actual=np.array([[float(r[f'ilr_{j+1:02d}']) for j in range(16)] for r in saved])
        check(label+' ILR coordinates',np.allclose(expected,actual,atol=4e-14,rtol=0),f'n={len(actual)},16 dimensions, geometric-mean formula')
        check(label+' replacement simplex',np.all(rp>0) and np.allclose(rp.sum(axis=1),1,atol=2e-15), 'fixed training parameter, positive closure')
    metadata=json.loads((OUT/'final_model_metadata.json').read_text(encoding='utf-8'));test_rows=rows('test_predictions.csv');validation=rows('model_validation.csv')
    for meta in metadata:
        model,target=meta['model'],meta['response'];y=train[target]
        check(target+'/'+model+' training IDs',meta['fitting_recipe_indices']==ids and meta['n_fit']==len(P),'only A4/A5 indices; no test or estimated labels')
        estimator=joblib.load(OUT/'models'/f'{target}_{model}.joblib')
        for label in ['test_1M','test_60M','test_1B']:
            d=datasets[label];prediction=predict(d['P'],meta)
            values=[r for r in test_rows if r['response']==target and r['model']==model and r['dataset']==label]
            saved=np.array([float(r['predicted_loss']) for r in values])
            check(target+'/'+model+'/'+label+' predictions',len(values)==len(prediction) and np.allclose(prediction,saved,atol=2e-12,rtol=0)
                and np.allclose(estimator.predict(d['P']),saved,atol=2e-12,rtol=0),'independent formula plus persisted joblib; no model fit')
            actual_metrics=metrics(d[target],saved);reported=next(r for r in validation if r['response']==target and r['model']==model and r['dataset']==label)
            for k,v in actual_metrics.items():
                if (v is None and reported[k]!='') or (v is not None and abs(v-float(reported[k]))>1e-11):raise AssertionError(f'Metric mismatch {label}/{k}')
            if reported['test_used_for_fitting']!='False':raise AssertionError('Test fitting flag')
    check('all direct test metrics',True,f'{len(validation)} model/target/scale views reconstructed')
    folds={r['index']:int(r['outer_holdout_fold']) for r in rows('outer_fold_assignments.csv')};oof=rows('nested_OOF_predictions.csv')
    summary=rows('nested_CV_summary.csv')
    for meta in metadata:
        model,target=meta['model'],meta['response'];prediction=np.empty(len(P));y=train[target]
        for fold in range(1,6):
            fit=np.array([i for i,index in enumerate(ids) if folds[index]!=fold]);hold=np.array([i for i,index in enumerate(ids) if folds[index]==fold])
            outer_meta=json.loads((OUT/'models/outer'/f'{target}_{model}_fold{fold}.json').read_text())
            check(target+'/'+model+f'/fold{fold} fit scope',outer_meta['fitting_recipe_indices']==[ids[i] for i in fit],f'train={len(fit)},hold={len(hold)}; disjoint')
            if model!='M0':
                p=P[fit]/P[fit].sum(axis=1,keepdims=True);f=features(P[fit],outer_meta)
                check(target+'/'+model+f'/fold{fold} training-only transforms',abs(.5*p[p>0].min()-outer_meta['zero_delta'])<1e-16
                    and np.allclose(f.mean(axis=0),outer_meta['X_mean'],atol=2e-13,rtol=0)
                    and abs(y[fit].mean()-outer_meta['y_mean'])<2e-15,
                    'current fold zero replacement, X mean and y mean use training subset only')
            elif abs(outer_meta['constant']-y[fit].mean())>2e-15:raise AssertionError('Baseline mean leaked')
            prediction[hold]=predict(P[hold],outer_meta)
        values={r['index']:r for r in oof if r['response']==target and r['model']==model}
        saved=np.array([float(values[index]['predicted_loss']) for index in ids])
        check(target+'/'+model+' nested OOF',len(values)==len(P) and np.allclose(prediction,saved,atol=2e-12,rtol=0),'all held predictions reconstructed from outer-fold fit parameters')
        reported=next(r for r in summary if r['response']==target and r['model']==model)
        if abs(metrics(y,saved)['RMSE']-float(reported['RMSE']))>1e-14:raise AssertionError('Nested RMSE mismatch')
    inner=rows('inner_fold_assignments.csv')
    check('inner CV leakage audit',all(folds[r['index']]!=int(r['outer_fold']) for r in inner),'no outer held-out recipe is in inner CV fitting/selection population')
    selection=json.loads((OUT/'model_selection.json').read_text())
    numerical_best=min([r for r in summary if r['response']=='L_mean'],key=lambda r:float(r['RMSE']))['model']
    check('train-only model selection',selection['recommended_model']==numerical_best and not selection['test_used_for_selection'] and not selection['Q_or_K_used'] and not selection['estimated_loss_used'],'nested OOF RMSE only')
    scale=rows('cross_scale_recipe_rank_stability.csv')
    for r in scale:
        a,b=datasets[r['dataset_1']],datasets[r['dataset_2']];ma={key:i for i,key in enumerate(signatures(a['P']))};mb={key:i for i,key in enumerate(signatures(b['P']))};keys=sorted(set(ma)&set(mb))
        check(r['response']+'/'+r['dataset_1']+'/'+r['dataset_2']+' shared compositions',len(keys)==int(r['shared_recipe_n']), 'match recipe vectors, not reused indices')
        if not keys:
            if r['actual_Pearson'] or r['actual_Spearman']:raise AssertionError('Invented correlation for absent pairs')
        else:
            ya=np.array([a[r['response']][ma[k]] for k in keys]);yb=np.array([b[r['response']][mb[k]] for k in keys])
            if abs(spearmanr(ya,yb).statistic-float(r['actual_Spearman']))>1e-14:raise AssertionError('Scale ranks mismatch')
    check('no large-scale prediction',set(r['dataset'] for r in test_rows)=={'test_1M','test_60M','test_1B'},'10B/70B diagnostic responses only, no final extrapolation')
    locked=rows('locked_inputs_integrity.csv')
    check('frozen file hashes',all(sha256_file(ROOT/r['path'])==r['sha256_before']==r['sha256_after'] for r in locked),f'{len(locked)} originals/prior outputs/code unchanged')
    check('original CSV manifest',all(sha256_file(ROOT/r['path'])==r['sha256'] for r in rows('input_manifest.csv')),'all twelve official CSV files unchanged')
    write_rows(OUT/'mixture_verification.csv',checks)
    print(json.dumps({'checks_passed':len(checks),'frozen_files':len(locked),'numeric_best':numerical_best},ensure_ascii=False))


if __name__=='__main__':main()
