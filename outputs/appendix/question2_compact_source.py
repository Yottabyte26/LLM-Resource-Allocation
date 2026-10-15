# ============================================================
# Source: src/scaling/io.py
# Original lines: 2-11
# Purpose: 真实附件读取、索引/字段兼容与输入单位
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
import csv
from collections import Counter
from decimal import Decimal
import hashlib
from pathlib import Path
import json
import numpy as np

PRIMARY_COLUMNS=['N_params_B','D_tokens_B','val_loss']

# ============================================================
# Source: src/scaling/io.py
# Original lines: 54-65
# Purpose: 真实附件读取、索引/字段兼容与输入单位
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def load_csv(path,label):
    """N/D in billions, raw positive cross-entropy Loss; no transformations or imputation."""
    columns,records=csv_rows(path)
    if any(c not in columns for c in PRIMARY_COLUMNS):raise ValueError(f'Missing N/D/Loss: {path}')
    values=np.array([[float(r[c]) for c in PRIMARY_COLUMNS] for r in records])
    if not np.isfinite(values).all() or (values<=0).any():
        raise ValueError(f'Nonfinite/nonpositive required N/D/Loss; inspect rather than impute: {path}')
    N,D,L=values.T
    # run_id is checkpoint-specific in B1, but a repeated trajectory ID in B2.
    group=np.array([f'{label}:N={size_key(r["N_params_B"])}' for r in records])
    return dict(path=path,label=label,columns=columns,records=records,N=N,D=D,L=L,group=group,
        group_rule='dataset/source file + exact canonical N_params_B; run_id not assumed to mean model ID')

# ============================================================
# Source: src/scaling/io.py
# Original lines: 80-87
# Purpose: 真实附件读取、索引/字段兼容与输入单位
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def combine_trajectories(trajectories):
    """B3 rows keep source file provenance; filenames are trajectory grouping units."""
    keys=['N','D','L','group'];combined={k:[] for k in keys};records=[];source=[]
    for t in trajectories:
        if len(set(t['N']))!=1:raise ValueError('B3 source file contains multiple model sizes')
        for k in ['N','D','L']:combined[k].extend(t[k])
        combined['group'].extend([t['path'].stem]*len(t['N']));records+=t['records'];source.extend([t['path'].name]*len(t['N']))
    return dict(label='B3',records=records,source=source,**{k:np.array(v) for k,v in combined.items()})

# ============================================================
# Source: src/scaling/classical.py
# Original lines: 2-99
# Purpose: 经典五参数非线性拟合、解析Jacobian、误差指标
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import pearsonr,spearmanr

PARAMETERS={'M0_N_D':['E','A','B','alpha','beta'], 'M1_N_only':['E','A','alpha'], 'M2_D_only':['E','B','beta']}


def weights(groups,balanced=True,multiplicities=None):
    """Each selected trajectory occurrence has equal total mass; normalize to one.

    Cluster-bootstrap multiplicity represents resampled whole trajectories,
    never randomly selected checkpoint rows.
    """
    groups=np.asarray(groups);unique,count=np.unique(groups,return_counts=True);counts=dict(zip(unique,count))
    if multiplicities is None:multiplicities={g:1 for g in unique}
    w=np.array([multiplicities.get(g,0)/(counts[g] if balanced else 1) for g in groups],dtype=float)
    if not np.isfinite(w).all() or w.sum()<=0:raise ValueError('Invalid trajectory weights')
    return w/w.sum()


def predict(parameters,N,D,model='M0_N_D'):
    """N and D measured in units of 1B. This does NOT log-transform Loss."""
    result=np.full(np.shape(N),parameters['E'],dtype=float)
    if 'A' in PARAMETERS[model]:result+=parameters['A']*np.exp(-parameters['alpha']*np.log(N))
    if 'B' in PARAMETERS[model]:result+=parameters['B']*np.exp(-parameters['beta']*np.log(D))
    if not np.isfinite(result).all() or (result<=0).any():raise FloatingPointError('Invalid predicted Loss')
    return result


def prediction_jacobian(logtheta,lnN,lnD,model):
    """Analytic derivatives with respect to LOG PARAMETERS, not log Loss."""
    keys=PARAMETERS[model];p=dict(zip(keys,np.exp(logtheta)));n=len(lnN)
    pred=np.full(n,p['E']);parts={'E':np.full(n,p['E'])}
    if 'A' in p:
        a=p['A']*np.exp(-p['alpha']*lnN);pred+=a;parts['A']=a;parts['alpha']=-p['alpha']*lnN*a
    if 'B' in p:
        b=p['B']*np.exp(-p['beta']*lnD);pred+=b;parts['B']=b;parts['beta']=-p['beta']*lnD*b
    return pred,np.column_stack([parts[k] for k in keys])


def initial_values(N,D,L,model,settings,rng,count,warm=None):
    """Reproducible dispersed exponent starts and data-scale positive amplitudes."""
    starts=[];keys=PARAMETERS[model];bounds=settings['parameter_bounds']
    if warm is not None:starts.append(np.array([warm[k] for k in keys]))
    exponent_lo,exponent_hi=settings['initial_exponent_range']
    while len(starts)<count:
        a,b=np.exp(rng.uniform(np.log(exponent_lo),np.log(exponent_hi),size=2))
        E=float(rng.uniform(.05,.95)*L.min());excess=max(float(np.median(L))-E,1e-5)
        fraction=float(rng.uniform(.1,.9)) if model=='M0_N_D' else (1.0 if model=='M1_N_only' else 0.0)
        p=dict(E=E,A=excess*fraction*float(np.median(N))**a,B=excess*(1-fraction)*float(np.median(D))**b,alpha=a,beta=b)
        starts.append(np.array([p[k] for k in keys]))
    lo=np.array([bounds[k][0] for k in keys]);hi=np.array([bounds[k][1] for k in keys])
    return [np.log(np.clip(x,lo*1.01,hi*.99)) for x in starts]


def fit(N,D,L,w,model,settings,seed,warm=None,count=None,save_starts=True):
    """Weighted least squares in ORIGINAL Loss with strict positive parameter bounds.

    Independent multi-start solves; only legal successful finite solutions are
    eligible. No validation data appears in this function or its selection.
    """
    keys=PARAMETERS[model];w=np.asarray(w);sel=w>0
    N,D,L,w=np.asarray(N)[sel],np.asarray(D)[sel],np.asarray(L)[sel],w[sel];w/=w.sum()
    lnN=np.log(N);lnD=np.log(D);sw=np.sqrt(w)
    bounds=settings['parameter_bounds'];lo=np.log([bounds[k][0] for k in keys]);hi=np.log([bounds[k][1] for k in keys])
    def fun(x):return sw*(prediction_jacobian(x,lnN,lnD,model)[0]-L)
    def jac(x):return sw[:,None]*prediction_jacobian(x,lnN,lnD,model)[1]
    starts=initial_values(N,D,L,model,settings,np.random.default_rng(seed),count or settings['multistart_count'],warm)
    runs=[];best=None;tol=settings['optimizer_tolerance']
    for number,x0 in enumerate(starts,1):
        solved=least_squares(fun,x0,jac=jac,bounds=(lo,hi),method='trf',x_scale='jac',loss='linear',
            max_nfev=settings['max_nfev'],ftol=tol,xtol=tol,gtol=tol)
        p=dict(zip(keys,map(float,np.exp(solved.x))));objective=float(np.dot(solved.fun,solved.fun))
        legal=bool(solved.success and np.isfinite(objective) and all(np.isfinite(v) and v>0 for v in p.values()))
        if save_starts:
            runs.append(dict(start=number,initial_parameters=dict(zip(keys,map(float,np.exp(x0)))),**p,
                objective=objective,successful=bool(solved.success),legal_solution=legal,status=int(solved.status),
                nfev=int(solved.nfev),optimality=float(solved.optimality),message=solved.message,
                active_bounds=solved.active_mask.tolist()))
        if legal and (best is None or objective<best['objective']):
            singular=np.linalg.svd(solved.jac,compute_uv=False);rank=int(np.linalg.matrix_rank(solved.jac))
            best=dict(model=model,parameters=p,objective=objective,successful=True,status=int(solved.status),
                nfev=int(solved.nfev),optimality=float(solved.optimality),chosen_start=number,
                active_bounds=solved.active_mask.tolist(),jacobian_singular_values=singular.tolist(),jacobian_rank=rank,
                jacobian_condition_number=float(singular[0]/singular[-1]) if singular[-1]>0 else None,
                log_parameters=solved.x.tolist(),n_positive_weight_rows=len(L),n_starts=len(starts))
    if best is None:raise RuntimeError('All nonlinear starts failed; do not substitute invented parameters')
    return best,runs


def metrics(y,p):
    """Observed-minus-predicted bias; constant ranks and zero-variance R2 undefined."""
    y=np.asarray(y);p=np.asarray(p);e=y-p;defined=len(y)>1 and np.ptp(y)>0 and np.ptp(p)>0
    return dict(n=len(y),RMSE=float(np.sqrt(np.mean(e**2))),MAE=float(np.mean(abs(e))),
        R2=float(1-np.sum(e**2)/np.sum((y-y.mean())**2)) if len(y)>1 and np.ptp(y)>0 else None,
        Pearson=float(pearsonr(y,p).statistic) if defined else None,Spearman=float(spearmanr(y,p).statistic) if defined else None,
        mean_bias=float(e.mean()),median_residual=float(np.median(e)),residual_std=float(e.std(ddof=1)) if len(y)>1 else None)

# ============================================================
# Source: src/scaling/bootstrap.py
# Original lines: 2-34
# Purpose: 整条trajectory重采样拟合
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
from collections import Counter
import numpy as np
from .classical import weights,fit,predict,metrics

_DATA=None


def initialize(data):
    """Workers receive ONLY B1 arrays, its primary parameters and fixed fit settings."""
    global _DATA
    _DATA=data


def repetition(task):
    """Fit each resampled trajectory multiplicity with 32 positive multi-starts."""
    number,drawn,seed=task;d=_DATA;groups=d['group'];counts=Counter(drawn)
    selected=[g for g in np.unique(groups) if counts[g]>0]
    prefix=dict(repetition=number,drawn_trajectories=list(drawn),multiplicities=dict(counts),
        distinct_trajectories=len(selected),seed=seed,fit_dataset='B1_only',bootstrap_unit='trajectory')
    try:
        w=weights(groups,True,counts)
        best,_=fit(d['N'],d['D'],d['L'],w,'M0_N_D',d['settings'],seed,
            warm=d['primary_parameters'],count=d['settings']['bootstrap_multistart_count'],save_starts=False)
        p=predict(best['parameters'],d['N'],d['D'])
        identified=best['jacobian_rank']==5 and len(selected)>=3
        result=dict(**prefix,**best['parameters'],objective=best['objective'],converged=True,
            identifiable=identified,eligible_parameter_CI=identified,status=best['status'],jacobian_rank=best['jacobian_rank'],
            condition_number=best['jacobian_condition_number'],active_bounds=best['active_bounds'],
            n_starts=best['n_starts'],prediction_RMSE_on_full_B1=metrics(d['L'],p)['RMSE'],error=None)
        return result,p
    except Exception as exc:
        return dict(**prefix,converged=False,identifiable=False,eligible_parameter_CI=False,error=repr(exc)),None

# ============================================================
# Source: scripts/11_fit_classical_scaling.py
# Original lines: 6-31
# Purpose: 最终阶段实际执行入口：11_fit_classical_scaling.py
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
import csv
import json
import logging
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import time
import numpy as np
import yaml
from scipy.stats import pearsonr,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FormatStrFormatter

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from scaling.io import input_paths,load_csv,combine_trajectories,write_csv,write_json,sha256,csv_rows
from scaling.classical import fit,weights,predict,metrics,PARAMETERS
from scaling.bootstrap import initialize,repetition
from importlib import import_module
diagnostic=import_module('10_diagnose_scaling_data')
OUT=ROOT/'outputs/scaling/phase1';FIG=ROOT/'outputs/figures/question2_phase1'

# ============================================================
# Source: scripts/11_fit_classical_scaling.py
# Original lines: 48-154
# Purpose: 最终阶段实际执行入口：11_fit_classical_scaling.py
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def fit_and_cv(b1,settings):
    """32 starts per fit; leave out every checkpoint of one exact-size trajectory."""
    N,D,L,G=[b1[k] for k in ['N','D','L','group']];unique=np.unique(G);models={};starts=[];cv=[];predictions=[];cv_models=[]
    for model in PARAMETERS:
        balanced=weights(G)
        best,runs=fit(N,D,L,balanced,model,settings,settings['random_state'])
        models[model]=best
        starts += [dict(context='full_B1_balanced',model=model,**r) for r in runs]
        logging.info('%s full B1 objective %.9g params=%s',model,best['objective'],best['parameters'])
        for fold,group in enumerate(unique,1):
            va=G==group;tr=~va
            # Even initialization is computed exclusively on the current training
            # trajectories; do not warm-start CV from the all-B1 fitted model.
            fold_best,runs=fit(N[tr],D[tr],L[tr],weights(G[tr]),model,settings,settings['random_state']+fold)
            pred=predict(fold_best['parameters'],N[va],D[va],model)
            cv.append(dict(model=model,fold=fold,heldout_trajectory=group,N_heldout=float(N[va][0]),
                n_training_trajectories=len(unique)-1,n_training_rows=int(tr.sum()),
                **metrics(L[va],pred),fit_dataset='other_B1_trajectories_only',**fold_best['parameters']))
            cv_models.append(dict(model=model,fold=fold,heldout_trajectory=group,
                training_trajectories=unique[unique!=group].tolist(),training_source_rows=(np.where(tr)[0]+1).tolist(),
                heldout_source_rows=(np.where(va)[0]+1).tolist(),fit=fold_best))
            predictions += [dict(model=model,fold=fold,source_row=int(i+1),trajectory_id=group,
                N_params_B=float(N[i]),D_tokens_B=float(D[i]),true_loss=float(L[i]),predicted_loss=float(p)) for i,p in zip(np.where(va)[0],pred)]
            starts += [dict(context='group_cv',model=model,heldout_trajectory=group,**r) for r in runs]
            logging.info('%s leave-trajectory-out %d/%d RMSE %.8g',model,fold,len(unique),cv[-1]['RMSE'])
    # Same raw record weights because all B1 trajectory checkpoint counts are equal.
    unweighted,runs=fit(N,D,L,weights(G,False),'M0_N_D',settings,settings['random_state'])
    starts += [dict(context='full_B1_unweighted',model='M0_N_D',**r) for r in runs]
    write_json(OUT/'main_model.json',models['M0_N_D']);write_json(OUT/'baseline_models.json',models)
    write_json(OUT/'group_cv_models.json',cv_models);write_json(OUT/'unweighted_model.json',unweighted)
    write_csv(OUT/'multistart_results.csv',starts)
    write_csv(OUT/'classical_scaling_parameters.csv',[selected_row(models['M0_N_D'],'trajectory_balanced',len(unique)),selected_row(unweighted,'unweighted',len(unique))])
    write_csv(OUT/'B1_group_cv.csv',cv);write_csv(OUT/'B1_group_cv_predictions.csv',predictions)
    comparison=[]
    for model in PARAMETERS:
        actual=sorted([r for r in predictions if r['model']==model],key=lambda r:r['source_row'])
        p=np.array([r['predicted_loss'] for r in actual])
        comparison.append(dict(model=model,validation='leave_one_trajectory_out',folds=len(unique),**metrics(L,p)))
    write_csv(OUT/'baseline_model_comparison.csv',comparison)
    sensitivity=[]
    for label,best in [('trajectory_balanced',models['M0_N_D']),('unweighted',unweighted)]:
        pred=predict(best['parameters'],N,D)
        sensitivity.append(dict(weighting=label,**best['parameters'],objective=best['objective'],**metrics(L,pred),
            max_abs_prediction_difference_from_balanced=float(abs(pred-predict(models['M0_N_D']['parameters'],N,D)).max()),
            trajectory_checkpoint_counts=dict(Counter(G))))
    write_csv(OUT/'weighting_sensitivity.csv',sensitivity)
    return models,starts,cv,comparison,unweighted


def cluster_bootstrap(b1,primary,settings):
    """2,000 whole-trajectory draws, each with >=30 starts; resumable checkpoints."""
    rng=np.random.default_rng(settings['random_state']);groups=np.unique(b1['group']);n=settings['bootstrap_repetitions']
    tasks=[(i+1,rng.choice(groups,size=len(groups),replace=True).tolist(),settings['random_state']+10000+i) for i in range(n)]
    draw_rows=[dict(repetition=i,drawn_trajectories=draws,seed=seed) for i,draws,seed in tasks]
    write_csv(OUT/'bootstrap_cluster_draws.csv',draw_rows)
    # Resume only under unchanged inputs/config/model/draw plan.
    signature={k:sha256(OUT/k) for k in ['main_model.json','bootstrap_cluster_draws.csv','input_manifest.csv']}
    signature['config']=sha256(ROOT/'configs/scaling_phase1.yaml')
    checkpoint=OUT/'bootstrap_checkpoint.json';rows=[];pred=np.full((n,len(b1['L'])),np.nan,dtype=float)
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text(encoding='utf-8'))
        if saved['signature']!=signature:raise RuntimeError('Bootstrap input/config changed; cannot silently resume')
        rows=saved['rows']
        with (OUT/'bootstrap_predictions_checkpoint.npy').open('rb') as f:loaded=np.load(f)
        if loaded.shape!=pred.shape:raise RuntimeError('Bootstrap checkpoint prediction shape mismatch')
        pred[:len(rows)]=loaded[:len(rows)]
        logging.info('Bootstrap resume %d/%d',len(rows),n)
    data={k:b1[k] for k in ['N','D','L','group']};data.update(settings=settings,primary_parameters=primary['parameters'])
    if len(rows)<n:
        started=time.monotonic()
        with ProcessPoolExecutor(max_workers=settings['bootstrap_workers'],initializer=initialize,initargs=(data,)) as pool:
            for row,p in pool.map(repetition,tasks[len(rows):],chunksize=5):
                i=len(rows)
                if p is not None:pred[i]=p
                rows.append(row)
                if len(rows)%settings['bootstrap_checkpoint_interval']==0 or len(rows)==n:
                    with (OUT/'bootstrap_predictions_checkpoint.npy').open('wb') as f:np.save(f,pred)
                    write_json(checkpoint,dict(signature=signature,rows=rows))
                    logging.info('Cluster bootstrap %d/%d, failures=%d, elapsed=%.1fs',len(rows),n,sum(not r['converged'] for r in rows),time.monotonic()-started)
    write_csv(OUT/'parameter_bootstrap_draws.csv',rows)
    valid=[r for r in rows if r['eligible_parameter_CI']];failed=sum(not r['converged'] for r in rows);unidentified=sum(r['converged'] and not r['identifiable'] for r in rows)
    if not valid:raise RuntimeError('No identifiable bootstrap samples; parameter uncertainty cannot be estimated')
    statistics=[]
    for key in PARAMETERS['M0_N_D']:
        x=np.array([r[key] for r in valid])
        statistics.append(dict(parameter=key,main_estimate=primary['parameters'][key],median=float(np.median(x)),mean=float(x.mean()),std=float(x.std(ddof=1)),
            CI95_low=float(np.quantile(x,.025)),CI95_high=float(np.quantile(x,.975)),min=float(x.min()),max=float(x.max()),
            P01=float(np.quantile(x,.01)),P99=float(np.quantile(x,.99)),successful_identifiable_repetitions=len(valid),
            attempted_repetitions=n,solver_failure_count=failed,solver_failure_rate=failed/n,
            converged_nonidentifiable_count=unidentified,eligible_CI_exclusion_rate=(failed+unidentified)/n,
            bootstrap_unit='whole_trajectory',random_state=settings['random_state'],n_multistart_per_fit=settings['bootstrap_multistart_count']))
    write_csv(OUT/'parameter_bootstrap.csv',statistics)
    values=np.array([[r[k] for k in PARAMETERS['M0_N_D']] for r in valid]);correlation=np.corrcoef(values,rowvar=False)
    write_csv(OUT/'parameter_correlation.csv',[dict(parameter=key,**dict(zip(PARAMETERS['M0_N_D'],row))) for key,row in zip(PARAMETERS['M0_N_D'],correlation)])
    indices=[i for i,r in enumerate(rows) if r['eligible_parameter_CI']];vp=pred[indices]
    prediction_ci=np.quantile(vp,[.025,.5,.975],axis=0)
    write_csv(OUT/'bootstrap_B1_prediction_intervals.csv',[dict(source_row=i+1,trajectory_id=b1['group'][i],
        N_params_B=float(b1['N'][i]),D_tokens_B=float(b1['D'][i]),pred_CI95_low=float(prediction_ci[0,i]),pred_median=float(prediction_ci[1,i]),
        pred_CI95_high=float(prediction_ci[2,i]),CI_width=float(prediction_ci[2,i]-prediction_ci[0,i]),
        interval_kind='fitted_mean_uncertainty_not_future_loss_prediction_interval') for i in range(len(b1['L']))])
    result=dict(attempted=n,converged=n-failed,identifiable=len(valid),solver_failure_rate=failed/n,
        converged_nonidentifiable=unidentified,parameter_correlation=correlation.tolist(),
        parameter_CI_basis='converged_full_Jacobian_rank_5_whole_trajectory_resamples',
        prediction_mean_CI_width=float((prediction_ci[2]-prediction_ci[0]).mean()),
        prediction_max_CI_width=float((prediction_ci[2]-prediction_ci[0]).max()))
    write_json(OUT/'bootstrap_summary.json',result)
    return rows,statistics,result

# ============================================================
# Source: scripts/11_fit_classical_scaling.py
# Original lines: 178-195
# Purpose: 最终阶段实际执行入口：11_fit_classical_scaling.py
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def validate(t,parameters,label,group_keys,provenance,absolute_comparability):
    """Direct predictions of frozen B1 parameters; source-layer errors remain descriptive."""
    pred=predict(parameters,t['N'],t['D']);rows=[]
    rows.append(dict(dataset=label,stratum='overall',group_key='all',provenance=provenance,
        absolute_loss_comparability=absolute_comparability,absolute_error_role='descriptive_only' if not absolute_comparability else 'file_scale_direct_validation',
        no_calibration=True,**metrics(t['L'],pred)))
    for group_name,groups in group_keys.items():
        for group in sorted(set(groups)):
            sel=np.array(groups)==group
            rows.append(dict(dataset=label,stratum=group_name,group_key=group,provenance=provenance,
                absolute_loss_comparability=absolute_comparability,absolute_error_role='descriptive_only' if not absolute_comparability else 'file_scale_direct_validation',
                no_calibration=True,**metrics(t['L'][sel],pred[sel])))
    write_csv(OUT/f'{label}_validation.csv' if label in ['B2','B3'] else OUT/('B4_cross_family_validation.csv' if label=='B4' else 'B5_literature_validation.csv'),rows)
    write_csv(OUT/f'{label}_direct_predictions.csv',[dict(source_row=i+1,dataset=label,
        source=t.get('source',[r.get('source','') for r in t['records']])[i],family=t['records'][i].get('family',''),
        trajectory_id=t['group'][i],N_params_B=float(t['N'][i]),D_tokens_B=float(t['D'][i]),true_loss=float(t['L'][i]),
        predicted_loss=float(pred[i]),residual=float(t['L'][i]-pred[i]),provenance=provenance) for i in range(len(pred))])
    return rows,pred

# ============================================================
# Source: src/scaling/quality_extension.py
# Original lines: 2-40
# Purpose: 最终链复用读入/reference工具；常数gamma MQ候选不是最终主模型
# MQ1/MQ2/gamma fitting: candidate / sensitivity model; not used as final main model. Final chain uses read_parameters, read_data, subset and group_reference.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
from collections import defaultdict
from decimal import Decimal
import numpy as np
from scipy.optimize import least_squares
from .io import csv_rows

NAMES = {'B6':'supplementary_NQ_experiment.csv',
         'B7':'supplementary_NQ_experiment_expanded.csv',
         'B8':'supplementary_NQ_experiment_large.csv'}
MODELS = ['M0', 'MQ1', 'MQ2', 'MQ1_offset', 'MQ2_offset']


def read_parameters(root):
    """Read full-precision primary Phase 1 row; never estimate classical terms."""
    path=root/'outputs/scaling/phase1/classical_scaling_parameters.csv'
    _, rows=csv_rows(path)
    primary=[r for r in rows if r['model']=='M0_N_D' and r['weighting']=='trajectory_balanced']
    if len(primary)!=1:raise ValueError('Ambiguous frozen classical parameter row')
    return {k:float(primary[0][k]) for k in ['E','A','B','alpha','beta']}, path


def read_data(root, label):
    """Canonical exact decimal NDQ keys; no imputation, deletion or Q reversal."""
    path=root/'real_attachments/B_scaling_laws'/NAMES[label]
    columns, rows=csv_rows(path)
    a=np.array([[float(r[c]) for c in ['N_params_B','D_tokens_B','Q_score','val_loss']] for r in rows])
    if not np.isfinite(a).all() or (a[:,:3]<=0).any() or (a[:,2]>1).any():
        raise ValueError(f'{label}: illegal N/D/Q/L')
    keys=[tuple(format(Decimal(r[c]).normalize(),'f') for c in ['N_params_B','D_tokens_B','Q_score']) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError(f'{label}: repeated candidate key; cannot take first/last')
    group=np.array(['N='+k[0]+';D='+k[1] for k in keys])
    return dict(label=label,path=path,columns=columns,records=rows,keys=keys,group=group,
                N=a[:,0],D=a[:,1],Q=a[:,2],L=a[:,3])


def subset(t, index):
    """Subset arrays for fitting, retaining source rows in the original dataset."""
    return {k:t[k][index] for k in ['N','D','Q','L','group']}

# ============================================================
# Source: src/scaling/quality_extension.py
# Original lines: 102-118
# Purpose: 最终链复用读入/reference工具；常数gamma MQ候选不是最终主模型
# MQ1/MQ2/gamma fitting: candidate / sensitivity model; not used as final main model. Final chain uses read_parameters, read_data, subset and group_reference.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def group_reference(t, prediction):
    """Differences relative to highest observed Q in the full evaluation dataset.

    For B7 new Q points, the Q=1 reference belongs to the shared ND trajectory.
    No model is fitted on that reference. References themselves are marked and
    excluded from primary difference metrics to avoid trivial-zero inflation.
    """
    groups=defaultdict(list)
    for i,g in enumerate(t['group']):groups[g].append(i)
    obs=np.zeros(len(t['L']));pred=np.zeros(len(t['L']));refq=np.zeros(len(t['L']))
    refs=np.zeros(len(t['L']),dtype=bool)
    for idx in groups.values():
        ref=max(idx,key=lambda i:t['Q'][i])
        obs[idx]=t['L'][idx]-t['L'][ref]
        pred[idx]=prediction[idx]-prediction[ref]
        refq[idx]=t['Q'][ref];refs[ref]=True
    return obs,pred,refq,refs

# ============================================================
# Source: src/scaling/quality_analysis.py
# Original lines: 2-11
# Purpose: Phase2C复用B8压力检验工具；MQ候选不是最终模型
# MQ candidate functions are not final main model; Phase2C calls stress_test only.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
import csv
import logging
from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold
from .quality_extension import MODELS, fit_quality, predict_quality, subset, group_reference
from .classical import metrics
from .io import write_csv, write_json

# ============================================================
# Source: src/scaling/quality_analysis.py
# Original lines: 206-240
# Purpose: Phase2C复用B8压力检验工具；MQ候选不是最终模型
# MQ candidate functions are not final main model; Phase2C calls stress_test only.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def stress_test(t,b6,p,predictions,keys6,keys7,out):
    """B8 is solely an incompatibility pressure test; never used for fitting/selection."""
    inside=np.ones(len(t['L']),dtype=bool)
    for k in ['N','D','Q']:inside&=(t[k]>=b6[k].min())&(t[k]<=b6[k].max())
    nonoverlap=np.array([k not in keys6|keys7 for k in t['keys']])
    masks={'full':np.ones(len(t['L']),dtype=bool),'nonoverlap':nonoverlap,
           'within_B6_numeric_ranges':inside,'outside_B6_numeric_ranges':~inside,
           'nonoverlap_within_ranges':inside&nonoverlap,'nonoverlap_outside_ranges':~inside&nonoverlap}
    diagnostics=[];group_rows=[];performance=[]
    for name,mask in masks.items():
        valid_rhos=[];insufficient=0;local=[];groups=defaultdict(list)
        for i in np.flatnonzero(mask):groups[t['group'][i]].append(i)
        for g,idx in groups.items():
            q,l=t['Q'][idx],t['L'][idx]
            if len(np.unique(q))<2 or np.ptp(l)==0:insufficient+=1;continue
            rho=float(spearmanr(q,l).statistic);valid_rhos.append(rho)
            group_rows.append(dict(subset=name,ND_group=g,n_Q=len(idx),spearman_Q_Loss=rho))
            ix=sorted(idx,key=lambda i:t['Q'][i]);local.extend(np.diff(t['L'][ix]).tolist())
        rv=np.array(valid_rhos);lv=np.array(local)
        diagnostics.append(dict(subset=name,n=int(mask.sum()),n_ND_groups=len(groups),n_groups_with_correlation=len(rv),
                                n_groups_insufficient_Q=insufficient,positive_groups=int((rv>0).sum()),negative_groups=int((rv<0).sum()),
                                positive_group_rate=float((rv>0).mean()) if len(rv) else None,
                                spearman_min=float(rv.min()) if len(rv) else None,spearman_P25=float(np.quantile(rv,.25)) if len(rv) else None,
                                spearman_median=float(np.median(rv)) if len(rv) else None,spearman_mean=float(rv.mean()) if len(rv) else None,
                                spearman_P75=float(np.quantile(rv,.75)) if len(rv) else None,spearman_max=float(rv.max()) if len(rv) else None,
                                loss_below_E_count=int((t['L'][mask]<p['E']).sum()),loss_below_E_rate=float((t['L'][mask]<p['E']).mean()),
                                increasing_adjacent_pair_rate=float((lv>1e-12).mean()) if len(lv) else None,
                                role='incompatible_semisynthetic_stress_test'))
        for m,yp in predictions.items():
            performance.append(evaluate(t['L'][mask],yp[mask],dataset='B8',subset=name,model=m,
                                        role='incompatible_semisynthetic_stress_test'))
    write_csv(out/'B8_incompatibility_diagnostics.csv',diagnostics)
    write_csv(out/'B8_subset_group_correlations.csv',group_rows)
    write_csv(out/'B8_range_validation.csv',performance)
    return diagnostics,performance,masks

# ============================================================
# Source: src/scaling/scale_quality.py
# Original lines: 2-60
# Purpose: S3原参数来源；S1/S2只为比较候选，最终去v且不重拟合
# S1/S2 are candidate / sensitivity models, not used as final main model. S3 C,u are frozen inputs to local Delta_Q; the near-zero v is removed without refitting.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

PARAMETERS={'S1':['gamma0','u','v'],'S2':['C','u','v','eta'],'S3':['C','u','v']}
POSITIVE={'gamma0','C','eta'}


def centers(t):
    """Median in log space over unique design levels, not response-weighted scales."""
    return dict(N0=float(np.exp(np.median(np.log(np.unique(t['N']))))),
                D0=float(np.exp(np.median(np.log(np.unique(t['D']))))))


def components(t,p,reference):
    """Reusable raw-Loss and dimensionless logarithmic design terms."""
    logN=np.log(t['N']/reference['N0']);logD=np.log(t['D']/reference['D0']);qlog=-np.log(t['Q'])
    term=p['B']*t['D']**(-p['beta'])
    l0=p['E']+p['A']*t['N']**(-p['alpha'])+term
    return logN,logD,qlog,term,l0


def physical(x,keys):
    """Log coordinates for strictly positive parameters; u/v include exact zero."""
    return {k:float(np.exp(a)) if k in POSITIVE else float(a) for k,a in zip(keys,x)}


def encoded(parameters,keys):
    return np.array([np.log(parameters[k]) if k in POSITIVE else parameters[k] for k in keys])


def response_jacobian(x,keys,model,parts,beta):
    """Analytic prediction/Jacobian in the optimizer's coordinates.

    The broad S1 bounds still keep exponents below double-precision overflow
    on the complete B6 support. No residual clipping or altered model is used.
    """
    a=physical(x,keys);n,d,z,term,l0=parts;scale=np.exp(-a['u']*n+a['v']*d)
    if model=='S1':
        gamma=a['gamma0']*scale;f=beta*gamma*z
        expf=np.exp(f);quality=term*np.expm1(f);j0=term*expf*f
        derivatives={'gamma0':j0,'u':-j0*n,'v':j0*d}
    elif model=='S2':
        f=a['eta']*z;expf=np.exp(f);quality=a['C']*scale*np.expm1(f)
        derivatives={'C':quality,'u':-quality*n,'v':quality*d,'eta':a['C']*scale*expf*f}
    elif model=='S3':
        quality=a['C']*scale*z;derivatives={'C':quality,'u':-quality*n,'v':quality*d}
    else:raise ValueError(model)
    pred=l0+quality
    if 'offset' in keys:pred+=a['offset'];derivatives['offset']=np.ones(len(z))
    jac=np.column_stack([derivatives[k] for k in keys])
    if not np.isfinite(pred).all() or not np.isfinite(jac).all():raise FloatingPointError('Nonfinite model evaluation')
    return pred,jac


def predict_scale(t,p,reference,model,parameters):
    """Frozen classical skeleton plus a nonnegative Q=1-anchored quality term."""
    keys=PARAMETERS[model]+(['offset'] if 'offset' in parameters else [])
    return response_jacobian(encoded(parameters,keys),keys,model,components(t,p,reference),p['beta'])[0]

# ============================================================
# Source: src/scaling/scale_quality.py
# Original lines: 77-133
# Purpose: S3原参数来源；S1/S2只为比较候选，最终去v且不重拟合
# S1/S2 are candidate / sensitivity models, not used as final main model. S3 C,u are frozen inputs to local Delta_Q; the near-zero v is removed without refitting.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def initial_parameters(model,cfg,seed,has_offset=False,count=None):
    """At least 50 dispersed, deterministic starts independent of evaluation Loss."""
    rng=np.random.default_rng(seed);count=count or cfg['multistart_count'];starts=[]
    if model=='S1':
        for g in [.1,.25,.5,1,2,4,8]:starts.append(dict(gamma0=g,u=.1,v=.1))
    elif model=='S2':
        for c,eta in [(.1,.1),(.2,.5),(.5,1),(1,.2),(5,.05),(50,.003),(100,.002)]:
            starts.append(dict(C=c,u=.1,v=.05,eta=eta))
    else:
        for c in [.03,.1,.2,.5,1,3,8]:starts.append(dict(C=c,u=.1,v=.1))
    while len(starts)<count:
        a=dict(u=float(rng.uniform(0,.5)),v=float(rng.uniform(0,.5)))
        if model=='S1':a['gamma0']=float(np.exp(rng.uniform(np.log(.08),np.log(3))))
        else:a['C']=float(np.exp(rng.uniform(np.log(.03),np.log(10))))
        if model=='S2':a['eta']=float(np.exp(rng.uniform(np.log(.002),np.log(1.5))))
        starts.append(a)
    if has_offset:
        for a in starts:a['offset']=float(rng.uniform(-.1,.1))
    return starts[:count]


def fit_scale(t,p,reference,model,cfg,seed,multiplicities=None,has_offset=False,count=None):
    """Bounded nonlinear LS with every start audited; bootstrap repeats whole ND blocks."""
    keys=PARAMETERS[model]+(['offset'] if has_offset else [])
    weights=np.ones(len(t['L'])) if multiplicities is None else np.array([multiplicities.get(g,0) for g in t['group']],dtype=float)
    ix=weights>0
    a={k:t[k][ix] for k in ['N','D','Q','L','group']};weights=weights[ix];weights/=weights.sum();sw=np.sqrt(weights)
    parts=components(a,p,reference);bounds=cfg['parameter_bounds']
    lo=encoded({k:bounds[k][0] for k in keys},keys);hi=encoded({k:bounds[k][1] for k in keys},keys)
    # u/v have zero lower bounds, which encoded() preserves without taking log.
    def fun(x):return sw*(response_jacobian(x,keys,model,parts,p['beta'])[0]-a['L'])
    def jac(x):return sw[:,None]*response_jacobian(x,keys,model,parts,p['beta'])[1]
    runs=[];tol=cfg['optimizer_tolerance']
    for i,start in enumerate(initial_parameters(model,cfg,seed,has_offset,count),1):
        x0=np.clip(encoded(start,keys),lo,hi)
        opt=least_squares(fun,x0,jac=jac,bounds=(lo,hi),method='trf',loss='linear',x_scale='jac',
                          max_nfev=cfg['max_nfev'],ftol=tol,xtol=tol,gtol=tol)
        params=physical(opt.x,keys);obj=float(np.dot(opt.fun,opt.fun))
        sv=np.linalg.svd(opt.jac,compute_uv=False)
        rank=int((sv>sv[0]*cfg['jacobian_rank_relative_tolerance']).sum()) if len(sv) and sv[0]>0 else 0
        condition=float(sv[0]/sv[-1]) if len(sv) and sv[-1]>0 else None
        hit={k:bool(params[k]-bounds[k][0]<=cfg['boundary_absolute_tolerance'] or bounds[k][1]-params[k]<=cfg['boundary_absolute_tolerance']) for k in keys}
        legal=bool(opt.success and np.isfinite(obj) and all(np.isfinite(v) for v in params.values()))
        runs.append(dict(start=i,initial_parameters=start,parameters=params,objective=obj,
                         convergence=bool(opt.success),legal_solution=legal,status=int(opt.status),nfev=int(opt.nfev),
                         jacobian_rank=rank,jacobian_condition=condition,jacobian_singular_values=sv.tolist(),
                         boundary_hit=any(hit.values()),boundary_parameters=[k for k,v in hit.items() if v],
                         per_parameter_boundary=hit,message=opt.message))
    valid=[r for r in runs if r['legal_solution']]
    if not valid:raise RuntimeError('No successful legal fit: '+model)
    best=min(valid,key=lambda r:r['objective'])
    nearby=[r for r in valid if r['objective']<=best['objective']+1e-9]
    spread={k:max(r['parameters'][k] for r in nearby)-min(r['parameters'][k] for r in nearby) for k in keys}
    return dict(model=model,source_offset=has_offset,**best,n_starts=len(runs),n_successful_starts=len(valid),
                parameter_count=len(keys),near_optimal_parameter_spread=spread,
                all_start_boundary_rate=sum(r['boundary_hit'] for r in valid)/len(valid),
                n_training_rows=int(ix.sum()),n_training_ND_groups=len(set(a['group']))),runs

# ============================================================
# Source: src/scaling/scale_quality_analysis.py
# Original lines: 2-183
# Purpose: ND分组CV、Q层留出、整块bootstrap及B6/B7差分检验
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
import csv
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from .quality_extension import subset,group_reference
from .scale_quality import PARAMETERS,fit_scale,predict_scale,marginal_quality
from .classical import predict,metrics
from .io import write_csv,csv_rows


def evaluate(t,pred,model,stage,mask=None):
    """Report absolute and reference-excluded within-ND differences together."""
    if mask is None:mask=np.ones(len(t['L']),dtype=bool)
    obs,dp,refq,refs=group_reference(t,pred)
    active=mask&~refs
    if not np.all(refq[mask]==1):raise ValueError('This B6/B7 evaluation requires observed Q=1 references')
    return [dict(model=model,stage=stage,target='absolute_Loss',**metrics(t['L'][mask],pred[mask])),
            dict(model=model,stage=stage,target='within_ND_DeltaL',reference_zero_rows_excluded=True,
                 reference_rule='Q=1 from full source ND group; reference may overlap B6 for B7',
                 **metrics(obs[active],dp[active]))]


def group_cv(t,p,reference,cfg,out):
    """Independent 50-start train-fold fits; ND blocks cannot be split."""
    models=['M0']+cfg['models'];oof={m:np.full(len(t['L']),np.nan) for m in models}
    folds=[];fits=[];starts=[];foldmetrics=[];rowfold=np.zeros(len(t['L']),dtype=int)
    for fold,(train,test) in enumerate(GroupKFold(n_splits=cfg['group_cv_folds']).split(t['N'],groups=t['group']),1):
        tr,te=subset(t,train),subset(t,test);rowfold[test]=fold
        trgroups,tegroups=sorted(set(tr['group'])),sorted(set(te['group']))
        if set(trgroups)&set(tegroups):raise AssertionError('ND group leakage')
        folds.append(dict(fold=fold,train_rows=len(train),test_rows=len(test),train_groups=trgroups,test_groups=tegroups))
        for m in models:
            if m=='M0':yp=predict(p,te['N'],te['D'])
            else:
                f,rs=fit_scale(tr,p,reference,m,cfg,cfg['random_state']+fold)
                fits.append(dict(fold=fold,**f));starts.extend(dict(stage='GroupCV',model=m,fold=fold,**r) for r in rs)
                yp=predict_scale(te,p,reference,m,f['parameters'])
            oof[m][test]=yp;foldmetrics+=evaluate(te,yp,m,'fold_'+str(fold))
        logging.info('Group CV fold %d/5 completed, training groups=%d test groups=%d',fold,len(trgroups),len(tegroups))
    comparison=[]
    for m,yp in oof.items():
        if not np.isfinite(yp).all():raise AssertionError('Incomplete OOF predictions')
        comparison+=evaluate(t,yp,m,'B6_5fold_GroupCV_OOF')
    pointrows=[dict(row_index=i,fold=int(rowfold[i]),ND_group=t['group'][i],N_params_B=float(t['N'][i]),D_tokens_B=float(t['D'][i]),
                    Q_score=float(t['Q'][i]),actual_Loss=float(t['L'][i]),**{m+'_prediction':float(v[i]) for m,v in oof.items()}) for i in range(len(t['L']))]
    write_csv(out/'group_cv_predictions.csv',pointrows);write_csv(out/'group_cv_folds.csv',folds)
    write_csv(out/'group_cv_model_parameters.csv',fits);write_csv(out/'group_cv_fold_metrics.csv',foldmetrics)
    write_csv(out/'B6_group_cv.csv',comparison)
    return oof,comparison,starts


def q_level_cv(t,p,reference,cfg,out):
    """Hold out each entire Q layer; fitting always uses training observations only."""
    rows=[];starts=[];predpoints=[]
    levels=np.unique(t['Q'])
    for j,q in enumerate(levels):
        tr=subset(t,t['Q']!=q);te=subset(t,t['Q']==q)
        for m in cfg['models']:
            f,rs=fit_scale(tr,p,reference,m,cfg,cfg['random_state']+100+j)
            starts.extend(dict(stage='leave_one_Q_out',model=m,held_Q=float(q),**r) for r in rs)
            yp=predict_scale(te,p,reference,m,f['parameters'])
            rows.append(dict(model=m,held_Q=float(q),test_kind='endpoint_extrapolation' if q in [levels[0],levels[-1]] else 'interior_interpolation',
                             parameters=f['parameters'],boundary_parameters=f['boundary_parameters'],**metrics(te['L'],yp)))
            predpoints.extend(dict(model=m,held_Q=float(q),ND_group=g,actual_Loss=float(y),predicted_Loss=float(v)) for g,y,v in zip(te['group'],te['L'],yp))
    write_csv(out/'leave_one_Q_level_out.csv',rows);write_csv(out/'leave_one_Q_predictions.csv',predpoints)
    return rows,starts


_WORKER_DATA=None


def initialize_worker(t,p,reference,cfg):
    """Set small read-only worker arrays; no source/output writes in workers."""
    global _WORKER_DATA
    from threadpoolctl import threadpool_limits
    # Small 360x4 systems benefit from one BLAS thread per CPU worker.
    threadpool_limits(limits=1)
    _WORKER_DATA=(t,p,reference,cfg)


def bootstrap_worker(job):
    """Fit all three models on the same whole-ND bootstrap draw, 50 starts each."""
    rep,groups,counts=job;t,p,reference,cfg=_WORKER_DATA
    mult=dict(zip(groups,counts));rows=[];starts=[]
    for m in cfg['models']:
        try:
            f,rs=fit_scale(t,p,reference,m,cfg,cfg['random_state']+1000+rep,mult,
                           count=cfg['bootstrap_multistart_count'])
            rows.append(dict(replicate=rep,model=m,success=True,parameters=f['parameters'],objective=f['objective'],
                             jacobian_rank=f['jacobian_rank'],jacobian_condition=f['jacobian_condition'],
                             boundary_hit=f['boundary_hit'],boundary_parameters=f['boundary_parameters'],
                             n_starts=f['n_starts'],n_successful_starts=f['n_successful_starts'],error=''))
            starts.extend(dict(replicate=rep,model=m,**r) for r in rs)
        except (RuntimeError,ValueError,FloatingPointError) as ex:
            rows.append(dict(replicate=rep,model=m,success=False,parameters=None,objective=None,jacobian_rank=None,
                             jacobian_condition=None,boundary_hit=None,boundary_parameters=[],
                             n_starts=cfg['bootstrap_multistart_count'],n_successful_starts=0,error=str(ex)))
    return rep,rows,starts


def resampling_jobs(t,cfg):
    """Uniform with-replacement sampling over 45 exact ND blocks."""
    groups=np.unique(t['group']);G=len(groups);rng=np.random.default_rng(cfg['random_state'])
    return [(rep,groups.tolist(),rng.multinomial(G,np.full(G,1/G)).tolist()) for rep in range(1,cfg['bootstrap_repetitions']+1)]


def bootstrap_models(t,p,reference,cfg,out,resume=False):
    """Bounded CPU workers; main process writes an audit of every one of 300k starts."""
    jobs=resampling_jobs(t,cfg);rows=[];params=[];correlations=[]
    write_csv(out/'bootstrap_block_draws.csv',[dict(replicate=r,ND_groups=g,multiplicities=c,n_sampled_blocks=sum(c),n_distinct_blocks=sum(v>0 for v in c)) for r,g,c in jobs])
    cols=['replicate','model','start','initial_parameters','parameters','objective','convergence','legal_solution','status','nfev',
          'jacobian_rank','jacobian_condition','jacobian_singular_values','boundary_hit','boundary_parameters','per_parameter_boundary','message']
    completed=0
    if resume:
        import json
        _,saved=csv_rows(out/'bootstrap_replicates.csv')
        for r in saved:
            rows.append(dict(replicate=int(r['replicate']),model=r['model'],success=r['success']=='True',
                             parameters=json.loads(r['parameters']) if r['parameters'] else None,
                             objective=float(r['objective']) if r['objective'] else None,
                             jacobian_rank=int(r['jacobian_rank']) if r['jacobian_rank'] else None,
                             jacobian_condition=float(r['jacobian_condition']) if r['jacobian_condition'] else None,
                             boundary_hit=r['boundary_hit']=='True',boundary_parameters=json.loads(r['boundary_parameters']),
                             n_starts=int(r['n_starts']),n_successful_starts=int(r['n_successful_starts']),error=r['error']))
        completed=max(r['replicate'] for r in rows)
        if len(rows)!=completed*len(cfg['models']):raise ValueError('Incomplete bootstrap checkpoint')
        if any({r['model'] for r in rows if r['replicate']==i}!=set(cfg['models']) for i in range(1,completed+1)):
            raise ValueError('Checkpoint model/replicate gaps')
        expected=completed*len(cfg['models'])*cfg['bootstrap_multistart_count'];retained=0
        temp=out/'bootstrap_resume_verified_prefix.csv'
        with (out/'bootstrap_multistart_solutions.csv').open(encoding='utf-8-sig',newline='') as f,temp.open('w',encoding='utf-8-sig',newline='') as g:
            rd=csv.DictReader(f);wr=csv.DictWriter(g,fieldnames=cols);wr.writeheader()
            for r in rd:
                if retained==expected:break
                if None in r or any(v is None for v in r.values()) or int(r['replicate'])>completed:raise ValueError('Invalid checkpoint audit prefix')
                wr.writerow(r);retained+=1
        if retained!=expected:raise ValueError('Missing multistart audit in checkpoint')
        temp.replace(out/'bootstrap_multistart_solutions.csv')
        jobs=[job for job in jobs if job[0]>completed]
        logging.info('Resuming from %d completed draws, preserving %d start records; BLAS threads per worker=1',completed,retained)
        from .io import write_json
        write_json(out/'bootstrap_resume_runtime.json',dict(preserved_replicates=completed,preserved_multistart_records=retained,
                   change='BLAS threads per worker reduced to 1; model/bounds/seed/starts/resamples unchanged'))
    initialize_worker(t,p,reference,cfg)
    executor=None
    if cfg['bootstrap_workers']>1:
        executor=ProcessPoolExecutor(max_workers=cfg['bootstrap_workers'],initializer=initialize_worker,initargs=(t,p,reference,cfg))
        results=executor.map(bootstrap_worker,jobs,chunksize=1)
    else:results=map(bootstrap_worker,jobs)
    try:
        with (out/'bootstrap_multistart_solutions.csv').open('a' if resume else 'w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=cols)
            if not resume:writer.writeheader()
            import json
            for rep,rr,starts in results:
                rows.extend(rr)
                writer.writerows({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in starts)
                if rep%cfg['bootstrap_log_interval']==0:
                    f.flush();write_csv(out/'bootstrap_replicates.csv',rows)
                    logging.info('ND block bootstrap %d/%d, %d model fits, %d starts per model',rep,cfg['bootstrap_repetitions'],len(rows),cfg['bootstrap_multistart_count'])
    finally:
        if executor is not None:executor.shutdown(wait=True,cancel_futures=True)
    write_csv(out/'bootstrap_replicates.csv',rows)
    for m in cfg['models']:
        valid=[r for r in rows if r['model']==m and r['success']]
        if not valid:raise RuntimeError('All block-bootstrap fits failed: '+m)
        for key in PARAMETERS[m]:
            v=np.array([r['parameters'][key] for r in valid]);lo,hi=cfg['parameter_bounds'][key]
            hit=np.array([key in r['boundary_parameters'] for r in valid])
            params.append(dict(model=m,parameter=key,n_requested=cfg['bootstrap_repetitions'],n_success=len(v),
                               mean=float(v.mean()),median=float(np.median(v)),std=float(v.std(ddof=1)),
                               P025=float(np.quantile(v,.025)),P975=float(np.quantile(v,.975)),
                               failure_rate=1-len(valid)/cfg['bootstrap_repetitions'],parameter_boundary_rate=float(hit.mean()),
                               model_any_boundary_rate=sum(r['boundary_hit'] for r in valid)/len(valid),
                               lower_bound=lo,upper_bound=hi,bootstrap_unit='exact_ND_pair',fixed_classical_parameters=True))
    correlations=parameter_correlation_rows(rows,cfg)
    write_csv(out/'bootstrap_parameters.csv',params);write_csv(out/'parameter_correlations.csv',correlations)
    return params,rows,correlations

# ============================================================
# Source: scripts/14_fit_scale_dependent_quality.py
# Original lines: 8-27
# Purpose: 最终阶段实际执行入口：14_fit_scale_dependent_quality.py
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import json
import logging
import copy
import numpy as np
import yaml
from src.scaling.io import csv_rows,write_csv,write_json,sha256
from src.scaling.quality_extension import read_parameters,read_data,group_reference
from src.scaling.classical import predict
from src.scaling.scale_quality import centers,fit_scale,predict_scale,PARAMETERS
from src.scaling.scale_quality_analysis import (group_cv,q_level_cv,bootstrap_models,evaluate,sensitivity_surfaces)

OUT=ROOT/'outputs/scaling/phase2C'
CONFIG=ROOT/'configs/scaling_quality_phase2C.yaml'

# ============================================================
# Source: scripts/14_fit_scale_dependent_quality.py
# Original lines: 41-60
# Purpose: 最终阶段实际执行入口：14_fit_scale_dependent_quality.py
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def basics(cfg,b6,p,ref):
    """Fit/CV only B6; every fit evaluates 50 independent starts."""
    fitted={};starts=[]
    for m in cfg['models']:
        f,rs=fit_scale(b6,p,ref,m,cfg,cfg['random_state']);fitted[m]=f
        starts.extend(dict(stage='full_B6',model=m,**r) for r in rs)
        logging.info('B6 %s params=%s RMSE=%.9g Jacobian rank=%d boundary=%s',m,f['parameters'],np.sqrt(f['objective']),f['jacobian_rank'],f['boundary_parameters'])
    oof,cv,rs=group_cv(b6,p,ref,cfg,OUT);starts+=rs
    qcv,rs=q_level_cv(b6,p,ref,cfg,OUT);starts+=rs
    write_csv(OUT/'multistart_solutions.csv',starts)
    write_csv(OUT/'model_parameters.csv',[dict(model=m,parameters=f['parameters'],objective=f['objective'],
               fit_RMSE=float(np.sqrt(f['objective'])),jacobian_rank=f['jacobian_rank'],jacobian_condition=f['jacobian_condition'],
               boundary_hit=f['boundary_hit'],boundary_parameters=f['boundary_parameters'],
               all_start_boundary_rate=f['all_start_boundary_rate'],n_starts=f['n_starts'],n_successful_starts=f['n_successful_starts'],
               near_optimal_parameter_spread=f['near_optimal_parameter_spread']) for m,f in fitted.items()])
    data=dict(classical_parameters=p,reference_scales=ref,fitted=fitted,CV=cv,Q_level_CV=qcv,
              OOF_predictions={m:v.tolist() for m,v in oof.items()},config_sha256=sha256(CONFIG),
              B6_sha256=sha256(b6['path']),fitting_source='B6_only',B7_B8_used_for_selection=False)
    write_json(OUT/'B6_compute_basics.json',data)
    return data

# ============================================================
# Source: src/scaling/mixture_transfer.py
# Original lines: 6-51
# Purpose: 冻结ILR预测、逐尺度lambda与配方bootstrap；P1/P2曲线仅比较候选
# P1/P2 smooth lambda curves: candidate / sensitivity models, not final main model. Final lambda is piecewise linear in ln N; these curve labels differ from Question3 resource policies.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import pearsonr, spearmanr, t

SCALES = [('1M', 'test_1M', .001, 'observed'),
          ('60M', 'test_60M', .06, 'observed'),
          ('1B', 'test_1B', 1., 'observed'),
          ('10B', 'estimated_10B', 10., 'estimated'),
          ('70B', 'estimated_70B', 70., 'estimated')]


def frozen_metadata(root):
    """Read exactly the human-selected L_mean first-order ElasticNet metadata."""
    base = Path(root) / 'outputs/mixture'
    policy = json.loads((base/'stage2/final_model_policy.json').read_text(encoding='utf-8'))
    if policy['selected_model'] != 'M1_ElasticNet':
        raise ValueError('Frozen main mixture model differs from the authorized model')
    rows = json.loads((base/'final_model_metadata.json').read_text(encoding='utf-8'))
    selected = [r for r in rows if r['model']=='M1_ElasticNet' and r['response']=='L_mean']
    if len(selected) != 1 or selected[0]['n_model_features'] != 16:
        raise ValueError('Ambiguous or non-first-order frozen mixture model')
    return selected[0]


def predict_ilr(x, meta):
    """Evaluate the frozen first-order model on saved ILR, without fitting."""
    a = np.asarray(x, float)
    return (float(meta['y_mean']) + float(meta['y_scale']) *
            (((a-np.array(meta['X_mean']))/np.array(meta['X_scale'])) @
             np.array(meta['coefficients']) + float(meta['intercept'])))


def predict_recipe(p, meta):
    """Apply frozen closure/zero convention for a new evaluation composition."""
    a = np.atleast_2d(np.asarray(p, float))
    if a.shape[1] != 17 or not np.isfinite(a).all() or (a<0).any() or (a.sum(1)<=0).any():
        raise ValueError('Expected finite nonnegative 17-domain recipes')
    a = a/a.sum(axis=1, keepdims=True)
    zero = a==0
    a = a*(1-zero.sum(axis=1)*meta['zero_delta'])[:, None]
    a[zero] = meta['zero_delta']
    return predict_ilr(np.log(a) @ np.asarray(meta['ilr_basis']).T, meta)

# ============================================================
# Source: src/scaling/mixture_transfer.py
# Original lines: 61-182
# Purpose: 冻结ILR预测、逐尺度lambda与配方bootstrap；P1/P2曲线仅比较候选
# P1/P2 smooth lambda curves: candidate / sensitivity models, not final main model. Final lambda is piecewise linear in ln N; these curve labels differ from Question3 resource policies.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def read_frozen_recipes(root, meta):
    """Join prediction to processed recipes by index; never read A4-A15 originals."""
    base = Path(root)/'outputs/mixture'
    pred = pd.read_csv(base/'test_predictions.csv')
    pred = pred[(pred.model=='M1_ElasticNet') & (pred.response=='L_mean')]
    est = pd.read_csv(base/'estimated_scale_predictions.csv')
    est = est[est.model=='M1_ElasticNet']
    train = pd.read_csv(base/'train_1M_closed_replaced_composition.csv').set_index('index')
    names = meta['domain_order']
    groups = []
    for label, dataset, size, evidence in SCALES:
        if evidence=='observed':
            g = pred[pred.dataset==dataset].copy()
            recipe = pd.read_csv(base/(dataset+'_closed_replaced_composition.csv')).set_index('index')
            ilr = pd.read_csv(base/(dataset+'_ILR_features.csv')).set_index('index')
            g = g.rename(columns={'predicted_loss':'M_p', 'true_loss':'Loss'})
            g['matched_train_index'] = pd.NA
            g['prediction_source'] = 'outputs/mixture/test_predictions.csv'
        else:
            g = est[est.dataset==dataset].copy()
            g = g.rename(columns={'uncalibrated_prediction':'M_p', 'estimated_loss':'Loss'})
            recipe = train.loc[g['matched_train_index']].copy()
            recipe.index = g['index'].to_numpy()
            ilr = None
            g['prediction_source'] = 'outputs/mixture/estimated_scale_predictions.csv'
        if g.empty or g['index'].duplicated().any() or recipe.index.duplicated().any():
            raise ValueError('Missing or duplicated recipe index: '+dataset)
        p = recipe.loc[g['index'], names].to_numpy(float)
        if not np.isfinite(p).all() or (p<=0).any() or not np.allclose(p.sum(1),1):
            raise ValueError('Invalid saved replacement composition')
        x = np.log(p) @ np.asarray(meta['ilr_basis']).T
        if ilr is not None and not np.allclose(x, ilr.loc[g['index']].to_numpy(float), atol=1e-12):
            raise ValueError('Saved composition and ILR disagree')
        if not np.allclose(predict_ilr(x, meta), g['M_p'], rtol=0, atol=1e-12):
            raise ValueError('Frozen predictor and saved M(p) disagree: '+dataset)
        if not np.isfinite(g[['Loss','M_p']].to_numpy(float)).all():
            raise ValueError('Missing prediction/Loss; no imputation authorized')
        for j, name in enumerate(names):
            g[name+'_saved_replaced'] = p[:,j]
        # Only compare recipe equality at saved floating precision, never index alone.
        g['recipe_signature'] = [json.dumps(np.round(v,12).tolist(),separators=(',',':')) for v in p]
        g['scale'] = label
        g['N_params_B'] = size
        g['evidence'] = evidence
        g['m_centered'] = g.M_p-g.M_p.mean()
        g['y_centered'] = g.Loss-g.Loss.mean()
        g['M_scale_mean'] = g.M_p.mean()
        g['Loss_scale_mean'] = g.Loss.mean()
        groups.append(g)
    return pd.concat(groups, ignore_index=True)


def transfer(g):
    """Centered OLS slope, conventional SE (n-2), and held-scale fit metrics.

    Despite no fitted intercept in centered coordinates, two means have been
    estimated from the pairs; the conventional original-coordinate OLS uses
    residual degrees of freedom n-2. Bootstrap is the primary uncertainty.
    """
    x = g.M_p.to_numpy(float); y = g.Loss.to_numpy(float)
    x = x-x.mean(); y = y-y.mean()
    sxx = float(x@x)
    if len(x)<3 or sxx<=0:
        raise ValueError('Insufficient recipe variation for transfer')
    slope = float(x@y/sxx)
    resid = y-slope*x
    se = float(np.sqrt((resid@resid)/(len(x)-2)/sxx))
    critical = t.ppf(.975,len(x)-2)
    return dict(lambda_s=slope, standard_error=se,
                OLS_CI_low=float(slope-critical*se), OLS_CI_high=float(slope+critical*se),
                R2=float(1-(resid@resid)/(y@y)), RMSE=float(np.sqrt(np.mean(resid**2))),
                Pearson=float(pearsonr(x,y).statistic), Spearman=float(spearmanr(x,y).statistic),
                n=len(x), residual_df=len(x)-2, M_scale_mean=float(g.M_p.mean()),
                Loss_scale_mean=float(g.Loss.mean()), intercept_after_centering=0.)


def bootstrap_transfers(data, cfg):
    """2000 paired recipe-index resamples, recentering each resampled scale.

    Entire recipe sets match for 1M/60M and 10B/70B, respectively. Use the same
    index draws within each pair to preserve cross-scale dependence. 1B recipes
    have no overlap and use independent draws. The frozen M(p) is never refit.
    """
    groups = {s:data[data.scale==s].sort_values('index').reset_index(drop=True) for s,_,_,_ in SCALES}
    sets = {s:set(g.recipe_signature) for s,g in groups.items()}
    cohorts = [('1M','60M'), ('1B',), ('10B','70B')]
    for cohort in cohorts:
        signatures = groups[cohort[0]].recipe_signature.tolist()
        for s in cohort[1:]:
            if groups[s].recipe_signature.tolist()!=signatures:
                raise ValueError('Expected paired recipes differ; revise resampling first')
    for i, c1 in enumerate(cohorts):
        for c2 in cohorts[i+1:]:
            if sets[c1[0]] & sets[c2[0]]:
                raise ValueError('Partial overlap requires an explicit joint resampling design')
    rng = np.random.default_rng(cfg['random_state'])
    B = cfg['bootstrap_repetitions']
    slopes = np.empty((B,5))
    draws = []
    for cohort in cohorts:
        g0 = groups[cohort[0]]; n = len(g0)
        ix = rng.integers(0,n,size=(B,n))
        draws.extend(dict(replicate=b+1, cohort='+'.join(cohort),
                          sampled_recipe_indices=g0['index'].to_numpy()[ix[b]].tolist()) for b in range(B))
        for s in cohort:
            g = groups[s]; x = g.M_p.to_numpy()[ix]; y = g.Loss.to_numpy()[ix]
            x -= x.mean(axis=1,keepdims=True); y -= y.mean(axis=1,keepdims=True)
            denom = np.sum(x*x,axis=1)
            if (denom<=0).any():
                raise ValueError('Bootstrap draw has no M(p) variation')
            j = [a[0] for a in SCALES].index(s)
            slopes[:,j] = np.sum(x*y,axis=1)/denom
    records=[]; summaries=[]
    for j,(s,_,N,evidence) in enumerate(SCALES):
        v = slopes[:,j]
        summaries.append(dict(scale=s,N_params_B=N,evidence=evidence,
                              bootstrap_mean=float(v.mean()),bootstrap_median=float(np.median(v)),
                              bootstrap_std=float(v.std(ddof=1)),CI_low=float(np.quantile(v,.025)),
                              CI_high=float(np.quantile(v,.975)),positive_fraction=float((v>0).mean()),
                              negative_fraction=float((v<0).mean()),n_bootstrap=B))
        records.extend(dict(replicate=b+1,scale=s,lambda_s=float(v[b])) for b in range(B))
    return slopes, summaries, records, draws

# ============================================================
# Source: src/scaling/mixture_transfer.py
# Original lines: 248-253
# Purpose: 冻结ILR预测、逐尺度lambda与配方bootstrap；P1/P2曲线仅比较候选
# P1/P2 smooth lambda curves: candidate / sensitivity models, not final main model. Final lambda is piecewise linear in ln N; these curve labels differ from Question3 resource policies.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def empirical_lambda(N, sizes, slopes):
    """Piecewise linear in ln N, refusing extrapolation beyond provided knots."""
    N=np.asarray(N,float); sizes=np.asarray(sizes,float); slopes=np.asarray(slopes,float)
    if not np.isfinite(N).all() or np.any(N<sizes.min()) or np.any(N>sizes.max()):
        raise ValueError('N outside supported empirical knots; extrapolation not authorized')
    return np.interp(np.log(N),np.log(sizes),slopes)

# ============================================================
# Source: src/scaling/marginal_analysis.py
# Original lines: 6-153
# Purpose: 最终广义标度律、边界策略、解析导数/弹性及Q+0.1精确求根
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from .mixture_transfer import predict_recipe


class FrozenGeneralizedScaling:
    """Evaluate the human-confirmed modular form with explicit lambda boundaries."""
    def __init__(self,classical,quality,meta,M_ref,knots,config):
        self.c=dict(classical);self.q=dict(quality);self.meta=meta;self.M_ref=float(M_ref)
        self.Ns=np.array([k['N_params_B'] for k in knots],float)
        self.lambdas=np.array([k['lambda_s'] for k in knots],float)
        self.slopes=np.diff(self.lambdas)/np.diff(np.log(self.Ns))
        self.cfg=config;self.shrink=float(config['lambda_shrink_zero_B'])
        if not np.all(np.diff(self.Ns)>0) or self.shrink<=self.Ns[-1]:
            raise ValueError('Invalid frozen lambda knots or shrink endpoint')

    def delta_M(self,p):
        """Frozen 1M mixture prediction relative to the frozen training center."""
        return predict_recipe(p,self.meta)-self.M_ref

    def lambda_value(self,N,policy='S-A'):
        """Main clamp; sensitivity continuation or ln-scale shrink to zero."""
        x=np.asarray(N,float)
        if not np.isfinite(x).all() or (x<=0).any():raise ValueError('N must be positive')
        if policy not in ['S-A','S-B','S-C']:raise ValueError('Unknown lambda policy')
        value=np.interp(np.log(x),np.log(self.Ns),self.lambdas)
        high=x>self.Ns[-1]
        if policy=='S-B':
            value=np.where(high,self.lambdas[-1]+self.slopes[-1]*np.log(x/self.Ns[-1]),value)
        elif policy=='S-C':
            fraction=np.clip(np.log(x/self.Ns[-1])/np.log(self.shrink/self.Ns[-1]),0,1)
            value=np.where(high,self.lambdas[-1]*(1-fraction),value)
        return value

    def log_slope(self,N,policy='S-A',side='right'):
        """Return d lambda/d ln N; choose a one-sided derivative at each knot."""
        x=np.asarray(N,float)
        if not np.isfinite(x).all() or (x<=0).any() or side not in ['left','right']:
            raise ValueError('Illegal N or derivative side')
        j=np.searchsorted(self.Ns,x,side=side)-1
        value=np.zeros(x.shape,float)
        middle=(j>=0)&(j<len(self.Ns)-1)
        value=np.where(middle,self.slopes[np.clip(j,0,len(self.slopes)-1)],value)
        high=x>=self.Ns[-1] if side=='right' else x>self.Ns[-1]
        if policy=='S-B':value=np.where(high,self.slopes[-1],value)
        elif policy=='S-C':
            shrinking=high & (x<self.shrink if side=='right' else x<=self.shrink)
            value=np.where(shrinking,-self.lambdas[-1]/np.log(self.shrink/self.Ns[-1]),value)
        elif policy!='S-A':raise ValueError('Unknown lambda policy')
        return value

    def terms(self,N,D,Q,deltaM=0.,policy='S-A'):
        """Compute actual model terms; only reference scenarios use Q=1, deltaM=0."""
        N,D,Q,deltaM=np.broadcast_arrays(np.asarray(N,float),np.asarray(D,float),np.asarray(Q,float),np.asarray(deltaM,float))
        if not np.isfinite(N+D+Q+deltaM).all() or (N<=0).any() or (D<=0).any() or (Q<=0).any() or (Q>1).any():
            raise ValueError('Illegal N,D,Q,mixture score')
        c,q=self.c,self.q
        nt=c['A']*N**(-c['alpha']);dt=c['B']*D**(-c['beta'])
        dq=q['C_Q']*(N/q['N0_B'])**(-q['u_Q'])*(-np.log(Q))
        dp=self.lambda_value(N,policy)*deltaM;R=nt+dt+dq+dp
        return dict(N_term=nt,D_term=dt,Delta_Q=dq,Delta_p=dp,R=R,L0=c['E']+nt+dt,L_gen=c['E']+R)

    def derivatives(self,N,D,Q,deltaM=0.,policy='S-A',side='right'):
        """Exact interior or chosen one-sided derivatives; do not smooth knots."""
        t=self.terms(N,D,Q,deltaM,policy);N=np.asarray(N,float);D=np.asarray(D,float);Q=np.asarray(Q,float)
        c,q=self.c,self.q;s=self.log_slope(N,policy,side)
        gain=c['alpha']*t['N_term']+q['u_Q']*t['Delta_Q']-s*np.asarray(deltaM)
        return dict(dL_dN=-gain/N,dL_dD=-c['beta']*t['D_term']/D,
                    dL_dQ=-q['C_Q']*(N/q['N0_B'])**(-q['u_Q'])/Q,Gain_N=gain,s_i=s)

    def elasticity(self,N,D,Q,deltaM=0.,policy='S-A',side='right'):
        """Loss-reduction elasticities for R>0; traditional signs in an appendix."""
        t=self.terms(N,D,Q,deltaM,policy);d=self.derivatives(N,D,Q,deltaM,policy,side)
        valid=t['R']>0
        def ratio(a):return np.divide(a,t['R'],out=np.full(np.shape(t['R']),np.nan),where=valid)
        return dict(E_N=ratio(d['Gain_N']),E_D=ratio(self.c['beta']*t['D_term']),
                    E_Q=ratio(self.q['C_Q']*(np.asarray(N)/self.q['N0_B'])**(-self.q['u_Q'])),
                    traditional_N=np.asarray(N)/t['L_gen']*d['dL_dN'],
                    traditional_D=np.asarray(D)/t['L_gen']*d['dL_dD'],
                    traditional_Q=np.asarray(Q)/t['L_gen']*d['dL_dQ'],valid=valid)

    def N_dependent_loss(self,N,Q,deltaM,policy='S-A'):
        """Exact N-dependent part; E and B D^-beta cancel in Q equivalence."""
        N=np.asarray(N,float)
        return (self.c['A']*N**(-self.c['alpha'])+
                self.q['C_Q']*(N/self.q['N0_B'])**(-self.q['u_Q'])*(-np.log(Q))+
                self.lambda_value(N,policy)*deltaM)

    def equivalence(self,N,Q,deltaM=0.,policy='S-A'):
        """Find first exact N_eq>N, respecting nonmonotone lambda segments.

        Within a segment Gain_N=positive decreasing powers minus a constant,
        hence has at most one zero. Include each stationary point as well as
        each lambda knot in the root brackets: an endpoint-only sign check can
        miss a valid first root when F(N) decreases and then increases.
        """
        N,Q,deltaM=float(N),float(Q),float(deltaM);dQ=float(self.cfg['Q_gain'])
        if not 0<Q<=1 or N<=0:raise ValueError('Illegal equivalence workpoint')
        new=Q+dQ
        result=dict(Q_new=new,root_policy='first_root_above_N',N_max_B=max(self.cfg['root_max_multiple']*N,self.cfg['root_min_max_B']))
        if Q>.9 or new>1+1e-14:return dict(**result,status='Q_gain_out_of_range',N_eq_B=None)
        gain=float(self.derivatives(N,150.,Q,deltaM,policy,self.cfg['knot_derivative_main'])['Gain_N'])
        result['Gain_N']=gain
        if gain<=0:return dict(**result,status='no_local_equivalence',N_eq_B=None)
        local=float(self.q['C_Q']*(N/self.q['N0_B'])**(-self.q['u_Q'])/(Q*gain)*dQ)
        result['local_approx_gain']=local
        target=float(self.N_dependent_loss(N,new,deltaM,policy))
        def objective(x):return float(self.N_dependent_loss(x,Q,deltaM,policy)-target)
        upper=result['N_max_B'];points=[N]+[float(k) for k in self.Ns if N<k<upper]+[upper]
        if policy=='S-C' and N<self.shrink<upper:points.append(self.shrink)
        points=sorted(set(points));stationary=[]
        def gain_at(x,s):
            return (self.c['alpha']*self.c['A']*x**(-self.c['alpha'])+
                    self.q['u_Q']*self.q['C_Q']*(x/self.q['N0_B'])**(-self.q['u_Q'])*(-np.log(Q))-s*deltaM)
        for lo,hi in zip(points[:-1],points[1:]):
            s=float(self.log_slope(np.sqrt(lo*hi),policy));g0,g1=gain_at(lo,s),gain_at(hi,s)
            if g0*g1<0:
                stationary.append(float(brentq(lambda x:gain_at(x,s),lo,hi,xtol=self.cfg['root_xtol'],rtol=self.cfg['root_rtol'])))
        points=sorted(set(points+stationary));root=None;chosen=None
        for lo,hi in zip(points[:-1],points[1:]):
            flo,fhi=objective(lo),objective(hi)
            if flo==0 and lo>N:root=lo;chosen=(lo,lo);break
            if fhi==0:root=hi;chosen=(hi,hi);break
            if flo*fhi<0:
                root=float(brentq(objective,lo,hi,xtol=self.cfg['root_xtol'],rtol=self.cfg['root_rtol']))
                chosen=(lo,hi);break
        result.update(n_stationary_points=len(stationary),n_search_segments=len(points)-1,
                      objective_at_start=objective(N),objective_at_max=objective(upper))
        if root is None:return dict(**result,status='no_finite_equivalence_within_range',N_eq_B=None)
        change=root/N-1
        return dict(**result,status='solved',N_eq_B=root,equivalent_parameter_gain=change,
                    equivalent_parameter_gain_pct=100*change,local_approx_gain_pct=100*local,
                    local_minus_exact_pct_points=100*(local-change),
                    local_relative_error=(local-change)/change,
                    equation_residual=objective(root),root_bracket_low_B=chosen[0],root_bracket_high_B=chosen[1])


def finite_difference(model,N,D,Q,deltaM,variable,config,side=None):
    """Fourth-order central or one-sided derivative; no stencil crosses an N knot."""
    vals=dict(N=float(N),D=float(D),Q=float(Q));x=vals[variable]
    h=config['finite_difference_relative_step']*x
    def f(offset):
        v=dict(vals);v[variable]=x+offset*h
        return float(model.terms(deltaM=deltaM,**v)['L_gen'])
    if side=='right':return (-25*f(0)+48*f(1)-36*f(2)+16*f(3)-3*f(4))/(12*h)
    if side=='left':return (25*f(0)-48*f(-1)+36*f(-2)-16*f(-3)+3*f(-4))/(12*h)
    return (-f(2)+8*f(1)-8*f(-1)+f(-2))/(12*h)

# ============================================================
# Source: scripts/16_analyze_generalized_scaling.py
# Original lines: 5-29
# Purpose: 最终阶段实际执行入口：16_analyze_generalized_scaling.py
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import argparse
import logging
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr,spearmanr
from src.scaling.io import sha256,write_csv,write_json,csv_rows
from src.scaling.quality_extension import read_parameters
from src.scaling.mixture_transfer import frozen_metadata,predict_recipe,predict_ilr
from src.scaling.marginal_analysis import FrozenGeneralizedScaling,finite_difference

OUT=ROOT/'outputs/scaling/phase4'
FIG=ROOT/'outputs/figures/question2_phase4'
REPORT=ROOT/'outputs/report/question2_final.md'
NEW={'configs/scaling_phase4.yaml','src/scaling/marginal_analysis.py',
     'src/scaling/marginal_reporting.py','scripts/16_analyze_generalized_scaling.py',
     'scripts/16_verify_generalized_scaling.py'}
INPUTS=[]

# ============================================================
# Source: scripts/16_analyze_generalized_scaling.py
# Original lines: 111-128
# Purpose: 最终阶段实际执行入口：16_analyze_generalized_scaling.py
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def frozen_model(cfg):
    """Load full-precision frozen parameters, never invoke fitting functions."""
    classical,path=read_parameters(ROOT);INPUTS.append(path.relative_to(ROOT).as_posix())
    relative='outputs/scaling/phase3/generalized_scaling_candidate.json';INPUTS.append(relative)
    prior=json.loads((ROOT/relative).read_text(encoding='utf-8'))
    if classical!=prior['classical']:raise ValueError('Frozen backbone mismatch')
    params=read('outputs/scaling/phase2C/model_parameters.csv')
    qp=json.loads(params[params.model=='S3'].iloc[0].parameters)
    if prior['quality']['C_Q']!=qp['C'] or prior['quality']['u_Q']!=qp['u']:
        raise ValueError('Frozen quality module mismatch')
    t=read('outputs/scaling/phase3/mixture_transfer_by_scale.csv').sort_values('N_params_B')
    knots=t[['N_params_B','lambda_s','evidence','CI_low','CI_high']].to_dict('records')
    if any(k['lambda_s']!=old['lambda_s'] for k,old in zip(knots,prior['lambda_knots'])):
        raise ValueError('Frozen lambda nodes changed')
    INPUTS.extend(['outputs/mixture/final_model_metadata.json','outputs/mixture/stage2/final_model_policy.json'])
    meta=frozen_metadata(ROOT)
    model=FrozenGeneralizedScaling(classical,prior['quality'],meta,prior['reference']['M_ref'],knots,cfg)
    return model,prior,knots

# ============================================================
# Source: scripts/16_analyze_generalized_scaling.py
# Original lines: 320-366
# Purpose: 最终阶段实际执行入口：16_analyze_generalized_scaling.py
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
def large_validation(model,b9,b10,cfg):
    """B10 estimated consistency only at the structural reference Q=1,p=p_ref."""
    if not np.isfinite(b10[['N_params_B','D_tokens_B','val_loss']]).all().all() or (b10[['N_params_B','D_tokens_B']]<=0).any().any():
        raise ValueError('B10 invalid required scales/Loss; do not impute')
    truth=b10.val_loss.to_numpy();pred=model.terms(b10.N_params_B.to_numpy(),b10.D_tokens_B.to_numpy(),1.,0.)['L0']
    b10=b10.copy();b10['L0_prediction']=pred;b10['residual_prediction_minus_estimated']=pred-truth
    b10['Q_reference_scenario']=1.;b10['p_reference_scenario']='p_ref';b10['Delta_Q']=0.;b10['Delta_p']=0.
    b10['validation_kind']='estimated_Loss_consistency_of_classical_backbone_only'
    b10.to_csv(OUT/'B10_reference_predictions.csv',index=False,encoding='utf-8-sig')
    metrics=[]
    for name,mask in [('all',np.ones(len(b10),bool)),('100-300B',(b10.N_params_B>=100)&(b10.N_params_B<300)),
                      ('300-1000B',(b10.N_params_B>=300)&(b10.N_params_B<1000)),('1000B_and_above',b10.N_params_B>=1000)]:
        y=truth[mask];p=pred[mask]
        defined=len(y)>1 and np.ptp(y)>0 and np.ptp(p)>0
        metrics.append(dict(bucket=name,n=len(y),RMSE=float(np.sqrt(np.mean((p-y)**2))) if len(y) else None,
           MAE=float(np.mean(abs(p-y))) if len(y) else None,
           R2=float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)) if len(y)>1 and np.ptp(y)>0 else None,
           Pearson=float(pearsonr(y,p).statistic) if defined else None,Spearman=float(spearmanr(y,p).statistic) if defined else None,
           mean_bias=float(np.mean(p-y)) if len(y) else None,evidence='estimated_consistency_only',
           bucket_convention='[100,300),[300,1000),[1000,infinity)'))
    write_csv(OUT/'B10_extrapolation_metrics.csv',metrics)
    anchors=[]
    for N in cfg['N_workpoints_B']+cfg['N_lambda_sensitivity_B']:
        if N in [r['N_params_B'] for r in anchors]:continue
        exact=b9[b9.N_params_B==N]
        anchors.append(dict(N_params_B=N,within_B9_reported_N_range=bool(b9.N_params_B.min()<=N<=b9.N_params_B.max()),
              exact_reported_scale_model_count=len(exact),models_at_exact_scale=exact.model_name.tolist(),
              scope='metadata scale anchor only; total/active parameters not harmonized'))
    write_csv(OUT/'B9_scale_coverage.csv',anchors)
    return metrics


def lambda_sensitivity(model,recipes,cfg):
    """Compare authorized policies, without selecting by attractive prediction."""
    rows=[]
    for N in cfg['N_lambda_sensitivity_B']:
        for r in recipes:
            if r['recipe']=='p_ref':continue
            for policy in cfg['lambda_sensitivity']:
                lam=float(model.lambda_value(N,policy))
                rows.append(dict(N_params_B=N,recipe=r['recipe'],recipe_index=r['index'],policy=policy,
                    DeltaM=r['DeltaM'],lambda_p=lam,Delta_p=lam*r['DeltaM'],
                    lambda_log_slope=float(model.log_slope(N,policy)),
                    role='main' if policy=='S-A' else 'structural_sensitivity_only',
                    evidence='Class D extrapolation_not_validated_by_B9_B10'))
    write_csv(OUT/'lambda_beyond70B_sensitivity.csv',rows)
    return rows

# ============================================================
# Source: scripts/15_calibrate_mixture_scaling.py
# Original lines: 193-201
# Purpose: 最终阶段实际执行入口：15_calibrate_mixture_scaling.py
# Original call-site excerpt, not a standalone function.
# ============================================================
    logging.info('Phase 3 starts; %d prior files protected; no original A4-A15 parsing',len(snapshot['files']))
    classical,_=read_parameters(ROOT)
    qrows=pd.read_csv(ROOT/'outputs/scaling/phase2C/model_parameters.csv')
    qrow=qrows[qrows.model=='S3']
    if len(qrow)!=1:raise ValueError('Ambiguous frozen S3')
    qp=json.loads(qrow.iloc[0]['parameters'])
    quality=dict(C_Q=qp['C'],u_Q=qp['u'],N0_B=1.,v_removed_without_refit=True,
                 description='B6-calibrated local quality correction')
    meta=frozen_metadata(ROOT);data=read_frozen_recipes(ROOT,meta)

# ============================================================
# Source: scripts/15_calibrate_mixture_scaling.py
# Original lines: 208-238
# Purpose: 最终阶段实际执行入口：15_calibrate_mixture_scaling.py
# Original call-site excerpt, not a standalone function.
# ============================================================
    data.to_csv(OUT/'mixture_recipe_points.csv',index=False,encoding='utf-8-sig')
    rows=[]
    for s,dataset,N,e in SCALES:
        r=dict(scale=s,dataset=dataset,N_params_B=N,evidence=e,**transfer(data[data.scale==s]))
        rows.append(r);logging.info('%s lambda=%.12g Pearson=%.6g n=%d',s,r['lambda_s'],r['Pearson'],r['n'])
    slopes,summary,replicates,draws=bootstrap_transfers(data,cfg)
    for row,uncertainty in zip(rows,summary):row.update(uncertainty)
    write_csv(OUT/'mixture_transfer_by_scale.csv',rows)
    write_csv(OUT/'mixture_transfer_bootstrap.csv',summary)
    write_csv(OUT/'mixture_transfer_bootstrap_replicates.csv',replicates)
    write_csv(OUT/'mixture_transfer_bootstrap_draws.csv',draws)
    point=np.array([r['lambda_s'] for r in rows]);sizes=np.array([s[2] for s in SCALES])
    fits,comparison,critical,decision=scale_curves(slopes,point,cfg)
    ref=pd.read_csv(ROOT/'outputs/mixture/stage2/representative_baselines.csv')
    reference=ref[ref.baseline=='A4_ILR_center']
    if len(reference)!=1:raise ValueError('No unique frozen ILR center')
    p_ref=reference[meta['domain_order']].to_numpy(float)[0]
    M_ref=float(predict_recipe(p_ref,meta)[0])
    if abs(M_ref-float(reference.iloc[0].frozen_EN_prediction))>1e-12:
        raise ValueError('Reference prediction changed')
    train_ilr=pd.read_csv(ROOT/'outputs/mixture/train_1M_ILR_features.csv').iloc[:,1:].to_numpy(float)
    if not np.allclose(np.log(p_ref)@np.asarray(meta['ilr_basis']).T,train_ilr.mean(axis=0),atol=1e-12):
        raise ValueError('Reference is not frozen training ILR center')
    reference_config=dict(name='A4_ILR_center',domain_order=meta['domain_order'],p_ref=p_ref.tolist(),M_ref=M_ref)
    write_json(OUT/'reference_recipe.json',reference_config)
    knots=[dict(N_params_B=N,lambda_s=float(point[j]),evidence=e,
                CI_low=rows[j]['CI_low'],CI_high=rows[j]['CI_high']) for j,(s,d,N,e) in enumerate(SCALES)]
    write_csv(OUT/'lambda_empirical_knots.csv',knots)
    evidence=[dict(module='L0',parameters=['E','A','B','alpha','beta'],evidence='Class A',support='B1 observations; frozen B3/B4/B5 validations with original comparability limits'),
              dict(module='Delta_Q',parameters=['C_Q','u_Q'],evidence='Class B',support='B6 semisynthetic local calibration; B7 difference validation failed'),
              dict(module='M(p)',parameters=['frozen ILR first-order ElasticNet'],evidence='Class C',support='Question 1 training and true mixture checks at 1M/60M/1B'),

# ============================================================
# Source: scripts/14_fit_scale_dependent_quality.py
# Original lines: 77-134
# Purpose: 最终阶段实际执行入口：14_fit_scale_dependent_quality.py
# Original call-site excerpt, not a standalone function.
# ============================================================
               evaluation_sources_not_used_for_fitting=['B7','B8'],config_sha256=sha256(CONFIG),
               B6_sha256=sha256(b6['path']),bootstrap_repetitions=cfg['bootstrap_repetitions'],
               final_selection_priority=cfg['selection_priority'],freeze_status='candidate_only_pending_validation_and_human_confirmation'))
    verify_prior();logging.info('B6 compute step complete: 2000 draws x 3 models x 50 starts')


def finalize(cfg):
    """Evaluate fixed candidates without tuning, then offset sensitivity and reporting."""
    nprotected=verify_prior();data=json.loads((OUT/'B6_computation_complete.json').read_text(encoding='utf-8'))
    if data['config_sha256']!=sha256(CONFIG):raise ValueError('Finalize configuration differs from computed parameters')
    p,parameter_path=read_parameters(ROOT);ref=data['reference_scales'];fits=data['fitted']
    if p!=data['classical_parameters']:raise ValueError('Frozen classical coefficients changed')
    ts={label:read_data(ROOT,label) for label in ['B6','B7','B8']};b6,b7,b8=[ts[k] for k in ['B6','B7','B8']]
    if sha256(b6['path'])!=data['B6_sha256']:raise ValueError('B6 changed after computation')
    oof={m:np.array(v) for m,v in data['OOF_predictions'].items()};predictions={}
    for label,t in ts.items():
        predictions[label]={'M0':predict(p,t['N'],t['D'])}
        for m in cfg['models']:predictions[label][m]=predict_scale(t,p,ref,m,fits[m]['parameters'])
        records=[dict(dataset=label,source_row=i+2,**r,**{m+'_prediction':float(v[i]) for m,v in predictions[label].items()}) for i,r in enumerate(t['records'])]
        write_csv(OUT/(label+'_predictions.csv'),records)
    keys6=set(b6['keys']);keys7=set(b7['keys']);over=np.array([k in keys6 for k in b7['keys']])
    masks7=dict(full=np.ones(len(over),dtype=bool),overlap=over,nonoverlap=~over)
    if int((~over).sum())!=90:raise ValueError('Unexpected B7 nonoverlap size')
    validation=[]
    for name,mask in masks7.items():
        for m,yp in predictions['B7'].items():
            validation.extend(dict(subset=name,**r) for r in evaluate(b7,yp,m,'B7_fixed_B6_parameters',mask))
    write_csv(OUT/'B7_validation_all_subsets.csv',validation)
    write_csv(OUT/'B7_nonoverlap_validation.csv',[r for r in validation if r['subset']=='nonoverlap'])
    comparison=data['CV']+[r for r in validation if r['subset']=='nonoverlap']
    write_csv(OUT/'model_comparison.csv',comparison)
    within=[r for r in comparison if r['target']=='within_ND_DeltaL']
    for m,yp in predictions['B6'].items():within.append(evaluate(b6,yp,m,'B6_in_sample')[1])
    write_csv(OUT/'within_ND_validation.csv',within)
    dp=[]
    for label,source,preds,masks in [('B6','OOF',oof,dict(full=np.ones(len(b6['L']),dtype=bool))),
                                     ('B7','fixed_B6_parameters',predictions['B7'],masks7)]:
        t=ts[label]
        for m,yp in preds.items():
            obs,pp,qref,refs=group_reference(t,yp)
            for name,mask in masks.items():
                for i in np.flatnonzero(mask&~refs):
                    dp.append(dict(dataset=label,evaluation=source,subset=name,model=m,row_index=int(i),ND_group=t['group'][i],
                                   Q_score=float(t['Q'][i]),Q_ref=float(qref[i]),Delta_L_obs=float(obs[i]),Delta_L_pred=float(pp[i])))
    write_csv(OUT/'within_ND_prediction_points.csv',dp)
    # Main candidate acceptance is determined before any offset or B8 pressure result.
    bp=data['bootstrap_parameters'];acceptance=[]
    cv0=next(r for r in data['CV'] if r['model']=='M0' and r['target']=='absolute_Loss')
    v0=next(r for r in validation if r['model']=='M0' and r['subset']=='nonoverlap' and r['target']=='absolute_Loss')
    _,domains=csv_rows(ROOT/'outputs/quality/domain_quality_summary_A1.csv')
    surfaces,mono,trends,domain=sensitivity_surfaces(b6,p,ref,fits,OUT,domains,cfg)
    for m in cfg['models']:
        acv=next(r for r in data['CV'] if r['model']==m and r['target']=='absolute_Loss')
        dcv=next(r for r in data['CV'] if r['model']==m and r['target']=='within_ND_DeltaL')
        av=next(r for r in validation if r['model']==m and r['subset']=='nonoverlap' and r['target']=='absolute_Loss')
        dv=next(r for r in validation if r['model']==m and r['subset']=='nonoverlap' and r['target']=='within_ND_DeltaL')
        br=[r for r in bp if r['model']==m];mon=next(r for r in mono if r['model']==m)
        gates=dict(B6_absolute_CV_improves_M0=acv['RMSE']<cv0['RMSE'],B6_DeltaL_CV_positive_R2=dcv['R2']>0,

# ============================================================
# Source: scripts/14_fit_scale_dependent_quality.py
# Original lines: 165-168
# Purpose: 最终阶段实际执行入口：14_fit_scale_dependent_quality.py
# Original call-site excerpt, not a standalone function.
# ============================================================
    write_csv(OUT/'S2_bound_identifiability_sensitivity.csv',boundrows)
    from src.scaling.quality_analysis import stress_test
    stress,pressure,masks8=stress_test(b8,b6,p,predictions['B8'],keys6,keys7,OUT)
    write_csv(OUT/'B8_stress_test.csv',pressure)

