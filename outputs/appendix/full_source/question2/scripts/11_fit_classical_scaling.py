"""Classical scaling law: B1-only multistart fits, grouped CV and cluster bootstrap.

B2/B3/B4/B5 are never fitted or used for selection. Frozen question 1 is never
imported. N and D remain in billions. No Q, K, mixtures, or effective tokens.
"""
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


def read_rows(path):
    """Read derived result rows without changing any training data."""
    return csv_rows(path)[1]


def selected_row(best,weighting,n_trajectories):
    """Export objective=sum normalized weight*(raw Loss residual)^2, no half factor."""
    return dict(model=best['model'],weighting=weighting,**best['parameters'],objective=best['objective'],
        convergence_status=best['status'],converged=best['successful'],n_starts=best['n_starts'],
        n_trajectories=n_trajectories,n_checkpoints=best['n_positive_weight_rows'],chosen_start=best['chosen_start'],
        jacobian_rank=best['jacobian_rank'],jacobian_condition_number=best['jacobian_condition_number'],
        units='N_B,D_B',fit_dataset='B1_only')


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


def terminal_fit(b1,primary,settings):
    """Only final D per model; report rank deficiency and invariant intercept."""
    select=np.array([np.where(b1['group']==g)[0][np.argmax(b1['D'][b1['group']==g])] for g in np.unique(b1['group'])])
    N,D,L=b1['N'][select],b1['D'][select],b1['L'][select]
    best,runs=fit(N,D,L,weights(b1['group'][select]),'M0_N_D',settings,settings['random_state']+500,warm=primary['parameters'])
    write_json(OUT/'terminal_only_model.json',best)
    write_csv(OUT/'terminal_only_multistart.csv',runs)
    rows=[]
    for label,fit_result in [('full_checkpoint_primary',primary),('terminal_only_sensitivity',best)]:
        p=fit_result['parameters'];terminal=predict(p,N,D);full=predict(p,b1['N'],b1['D'])
        rows.append(dict(dataset=label,n_fit=fit_result['n_positive_weight_rows'],**p,objective=fit_result['objective'],
            jacobian_rank=fit_result['jacobian_rank'],parameters_identifiable=fit_result['jacobian_rank']==5,
            terminal_D_unique_count=len(set(D)),E_effective_at_terminal_D=float(p['E']+p['B']*D[0]**(-p['beta'])),
            terminal_RMSE=metrics(L,terminal)['RMSE'],terminal_Pearson=metrics(L,terminal)['Pearson'],
            full_B1_RMSE_descriptive=metrics(b1['L'],full)['RMSE'],
            N_trend='decreasing_by_positive_constraints',D_trend='decreasing_by_positive_constraints',
            D_direction_empirically_testable=len(set(D))>1))
    write_csv(OUT/'terminal_checkpoint_sensitivity.csv',rows)
    return best,rows


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


def residual_diagnostics(b1,primary):
    """Group/early-vs-late structure and serial correlations without IID p-values."""
    p=predict(primary['parameters'],b1['N'],b1['D']);e=b1['L']-p
    progress=np.array([float(r['steps']) for r in b1['records']]);progress/=max(progress)
    residual_rows=[dict(source_row=i+1,trajectory_id=b1['group'][i],N_params_B=float(b1['N'][i]),D_tokens_B=float(b1['D'][i]),
        steps=float(b1['records'][i]['steps']),training_progress=float(progress[i]),true_loss=float(b1['L'][i]),predicted_loss=float(p[i]),residual=float(e[i])) for i in range(len(e))]
    write_csv(OUT/'B1_residuals.csv',residual_rows)
    associations=[]
    for label,x in [('log_N',np.log(b1['N'])),('log_D',np.log(b1['D'])),('predicted_Loss',p),('training_progress',progress)]:
        associations.append(dict(variable=label,Pearson=float(pearsonr(e,x).statistic),Spearman=float(spearmanr(e,x).statistic),
            n_checkpoints=len(e),independent_trajectory_count=len(set(b1['group'])),IID_p_value_reported=False))
    write_csv(OUT/'residual_associations.csv',associations)
    summaries=[]
    for g in np.unique(b1['group']):
        sel=np.where(b1['group']==g)[0];sel=sel[np.argsort(b1['D'][sel])];res=e[sel]
        summaries.append(dict(stratum='trajectory',group=g,**metrics(b1['L'][sel],p[sel]),
            lag1_residual_Pearson=float(pearsonr(res[:-1],res[1:]).statistic)))
    # The quantile-derived progress strata are diagnostics, not deleted training rows.
    lo,hi=np.quantile(progress,[.1,.9])
    for label,sel in [('early_P10',progress<=lo),('middle_P10_P90',(progress>lo)&(progress<hi)),('late_P90',progress>=hi)]:
        summaries.append(dict(stratum='training_progress',group=label,**metrics(b1['L'][sel],p[sel]),
            lag1_residual_Pearson=None))
    write_csv(OUT/'residual_structure_summary.csv',summaries)
    return residual_rows,associations,summaries


def plot_all(b1,models,cv_predictions,bootstrap_rows,tables,trajectories,validations,residuals,settings):
    """Paper-style standalone plots; no interpolation plot treated as independent evidence."""
    FIG.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    manifest=[]
    def save(fig,name):
        fig.tight_layout();path=FIG/name;fig.savefig(path,dpi=settings['figure_dpi'],bbox_inches='tight');plt.close(fig)
        manifest.append(dict(path=path.relative_to(ROOT).as_posix()))
    pred=predict(models['M0_N_D']['parameters'],b1['N'],b1['D']);res=b1['L']-pred
    fig,ax=plt.subplots(figsize=(8,5));ax.scatter(b1['N'],b1['D'],s=8,alpha=.5)
    ax.set(xscale='log',yscale='log',xlabel='N (billions of parameters)',ylabel='D (billions of trained tokens)',title='B1 coverage: eight model trajectories');save(fig,'01_B1_N_D_coverage.png')
    groups=np.unique(b1['group']);fig,axes=plt.subplots(2,4,figsize=(15,7))
    for ax,g in zip(axes.flat,groups):
        sel=np.where(b1['group']==g)[0];sel=sel[np.argsort(b1['D'][sel])]
        ax.plot(b1['D'][sel],b1['L'][sel],'.',ms=3,label='Provided B1 Loss');ax.plot(b1['D'][sel],pred[sel],'-',lw=1.3,label='Frozen classical fit')
        ax.set(xscale='log',title=f"N={b1['N'][sel[0]]:.6g}B",xlabel='D (B tokens)',ylabel='Validation Loss')
    axes.flat[0].legend(fontsize=8);save(fig,'02_B1_Loss_D_trajectories.png')
    def observed_plot(y,p,title,name,groups=None):
        fig,ax=plt.subplots(figsize=(7,5))
        if groups is None:ax.scatter(y,p,s=10,alpha=.6)
        else:
            for group in sorted(set(groups)):
                sel=np.array(groups)==group;ax.scatter(y[sel],p[sel],s=25,alpha=.8,label=group)
            ax.legend(fontsize=7,ncol=2)
        lo,hi=min(y.min(),p.min()),max(y.max(),p.max());ax.plot([lo,hi],[lo,hi],'--',color='gray',lw=1)
        ax.set(xlabel='Provided validation Loss',ylabel='B1 model predicted Loss',title=title);save(fig,name)
    observed_plot(b1['L'],pred,'B1 fitted checkpoints (not IID)','03_B1_observed_predicted.png')
    ordered=sorted([r for r in cv_predictions if r['model']=='M0_N_D'],key=lambda r:int(r['source_row']));oof=np.array([float(r['predicted_loss']) for r in ordered])
    observed_plot(b1['L'],oof,'B1 leave-one-trajectory-out predictions','04_group_CV_observed_predicted.png')
    for name,x,title in [('05',np.log(b1['N']),'log N (N measured in B)'),('06',np.log(b1['D']),'log D (D measured in B)'),('07',pred,'Predicted Loss'),('08',np.array([r['training_progress'] for r in residuals]),'Training progress (step / final step)')]:
        fig,ax=plt.subplots(figsize=(8,5));ax.scatter(x,res,s=9,alpha=.5);ax.axhline(0,color='gray',lw=.8)
        ax.set(xlabel=title,ylabel='Residual (provided - predicted)',title='B1 residual structure; no IID significance test');save(fig,f'{name}_residual_structure.png')
    fig,ax=plt.subplots(figsize=(7,4));ax.hist(res,bins=40);ax.set(xlabel='Residual (provided - predicted)',ylabel='Checkpoint count',title='B1 residual histogram');save(fig,'09_residual_histogram.png')
    fig,axes=plt.subplots(2,3,figsize=(13,7));valid=[r for r in bootstrap_rows if r['eligible_parameter_CI']]
    for ax,key in zip(axes.flat,PARAMETERS['M0_N_D']):
        ax.hist([r[key] for r in valid],bins=35);ax.axvline(models['M0_N_D']['parameters'][key],color='gray',linestyle='--');ax.set(xlabel=key,ylabel='Whole-trajectory bootstrap draws')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))
        ax.tick_params(axis='x',labelsize=8)
    axes.flat[-1].axis('off');fig.suptitle('B1 cluster bootstrap parameter distributions (conditional on this table)');save(fig,'10_parameter_bootstrap.png')
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for ax,t in zip(axes.flat,trajectories):
        pp=predict(models['M0_N_D']['parameters'],t['N'],t['D']);order=np.argsort(t['D'])
        ax.plot(t['D'][order],t['L'][order],'.',ms=2,label='B3 interpolated table');ax.plot(t['D'][order],pp[order],'-',lw=1.3,label='B1 frozen model')
        ax.set(xscale='log',title=f"N={t['N'][0]:.6g}B",xlabel='D (B tokens)',ylabel='Validation Loss')
    axes.flat[0].legend(fontsize=8);save(fig,'11_B3_interpolated_trajectory_validation.png')
    observed_plot(tables['B2']['L'],validations['B2'][1],'B2 semi-synthetic auxiliary consistency','12_B2_auxiliary_validation.png')
    observed_plot(tables['B4']['L'],validations['B4'][1],'B4 cross-family table consistency; evaluation scope unspecified',
        '13_B4_cross_family_validation.png',[r['family'] for r in tables['B4']['records']])
    b5=tables['B5'];sources=sorted({r['source'] for r in b5['records']});fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,source in zip(axes.flat,sources):
        sel=np.array([r['source']==source for r in b5['records']]);yy=b5['L'][sel];pp=validations['B5'][1][sel]
        ax.scatter(yy,pp,s=28);lo,hi=min(yy.min(),pp.min()),max(yy.max(),pp.max());ax.plot([lo,hi],[lo,hi],'--',color='gray',lw=1)
        ax.set(title=source,xlabel='Reported val_loss (scope unspecified)',ylabel='B1 model prediction')
    fig.suptitle('B5 literature source layers: absolute Loss comparability unconfirmed');save(fig,'14_B5_literature_comparison.png')
    write_csv(OUT/'figure_manifest.csv',manifest)


def md_table(rows,columns):
    """Precise estimates/intervals retain meaningful digits even for narrow CIs."""
    def fmt(x):
        if x is None:return '未定义/不适用'
        if isinstance(x,(float,np.floating)):return f'{x:.10g}'
        return str(x).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join('---' for c in columns)+' |']+[
        '| '+' | '.join(fmt(r.get(c)) for c in columns)+' |' for r in rows])


def build_report(b1,models,starts,cv,comparison,unweighted,boot_stats,boot_summary,terminal_rows,validations,associations,residual_summary,settings,protected_n):
    """Full requested report, separating conditional fit from external scientific evidence."""
    primary=models['M0_N_D'];params=primary['parameters'];d=json.loads((OUT/'data_diagnosis.json').read_text(encoding='utf-8'))
    all_starts=[r for r in starts if r['context']=='full_B1_balanced' and r['model']=='M0_N_D'];successful=[r for r in all_starts if r['legal_solution']]
    spread=[dict(parameter=k,min=float(min(r[k] for r in successful)),max=float(max(r[k] for r in successful)),
        relative_range=float((max(r[k] for r in successful)-min(r[k] for r in successful))/params[k])) for k in params]
    main_cv=[r for r in cv if r['model']=='M0_N_D'];weighting=read_rows(OUT/'weighting_sensitivity.csv')
    corr=np.array(boot_summary['parameter_correlation']);key_order=PARAMETERS['M0_N_D'];pairs=[]
    for i in range(5):
        for j in range(i+1,5):pairs.append(dict(parameter_1=key_order[i],parameter_2=key_order[j],bootstrap_Pearson=float(corr[i,j])))
    source=[r for r in validations['B5'][0] if r['stratum']=='source'];family=[r for r in validations['B4'][0] if r['stratum']=='family']
    b3=[r for r in validations['B3'][0] if r['stratum']=='trajectory'];b2=validations['B2'][0]
    early=next(r for r in residual_summary if r['group']=='early_P10');late=next(r for r in residual_summary if r['group']=='late_P90')
    fitted=metrics(b1['L'],predict(params,b1['N'],b1['D']))
    boot_draws=read_rows(OUT/'parameter_bootstrap_draws.csv')
    tails=[]
    for stat in boot_stats:
        key=stat['parameter'];eligible=[r for r in boot_draws if r['eligible_parameter_CI']=='True']
        lo=min(eligible,key=lambda r:float(r[key]));hi=max(eligible,key=lambda r:float(r[key]))
        tails.append(dict(parameter=key,min=stat['min'],max=stat['max'],
            full_range_over_95CI_width=(stat['max']-stat['min'])/(stat['CI95_high']-stat['CI95_low']),
            min_extreme_distinct_trajectories=int(lo['distinct_trajectories']),max_extreme_distinct_trajectories=int(hi['distinct_trajectories'])))
    touching=sum(any(v!=0 for v in json.loads(r['active_bounds'])) for r in boot_draws if r['converged']=='True')
    terminal_starts=[r for r in read_rows(OUT/'terminal_only_multistart.csv') if r['legal_solution']=='True']
    terminal_spread=[dict(parameter=k,min=min(float(a[k]) for a in terminal_starts),max=max(float(a[k]) for a in terminal_starts)) for k in params]
    lines=['# 问题二 PHASE 1：经典 Scaling Law 基线拟合与跨数据源验证','',
        '## 1. 数据范围与可信度','',
        '本阶段只拟合 L(N,D)=E+A N^(-alpha)+B D^(-beta)。仅 B1 参与参数拟合、LOTO CV、bootstrap 和敏感性。B2/B3/B4/B5 仅应用冻结参数，无截距/尺度校准。没有引入 Q、K、p 或 D_eff，没有使用 B6–B10，没有进入问题三。',
        'B1 为附件中的 Pythia existing training log，B11/B12 提供元数据与 checkpoint 来源辅助；这些索引本身不能独立验证每个 Loss 的原始测量过程。B3 明确为插值轨迹，B2 按题目说明为半合成。B4 为跨族汇总表；B5 为六个文献来源标签的整理表，缺少完整评测口径，均不当作同口径独立实验的证明。',
        f"B1 实际 {d['B1_trajectory_count']} 条模型轨迹，每条 {len(b1['N'])//d['B1_trajectory_count']} checkpoints，总 {len(b1['N'])} 行。N范围={d['N_range']} B；D范围={d['D_range']} B tokens；Loss范围={d['Loss_range']}。",'',
        '字段、实际路径、缺失、Inf、重复、B12 对照和 FLOPs 核验详见 outputs/scaling/phase1/data_diagnosis.md 与相应 CSV。','',
        '## 2. checkpoint 分组与单位','',
        'B1 的 run_id 是每条 checkpoint 唯一标识，不能作为独立模型 ID。trajectory_id=B1 + 精确规范化 N_params_B；各规模对应一个稳定轨迹，没有用近似数值合并模型。每个轨迹 N 固定、D 按 steps 严格递增，Loss 无局部反弹；没有重复行或重复 checkpoint。相同规模内未标注的重复实验无法由附件进一步识别。',
        'N=N_params_B，D=D_tokens_B，以 1B 为参考单位直接参与幂律。因此 A、B 数值依赖该单位约定；不能和使用原始参数/token 个数的系数直接比较。FLOPs 单独换算：6(N_B×1e9)(D_B×1e9)/1e21=0.006 N_B D_B。',
        f"B1 C 与 6ND 有 {d['B1_C_6ND_rounding_consistent_count']}/{len(b1['N'])} 条在实际导出精度容差内一致。",'',
        '## 3. 经典模型与估计方法','',
        'Lhat=E+A N^-alpha+B D^-beta，全部五参数严格为正。使用 scipy.optimize.least_squares 有界非线性最小二乘；优化的是 log(参数) 以保持正值，残差始终为原始 Loss 差，绝没有对 Loss 取 log 作普通线性回归。使用解析 Jacobian，原始 Loss 平方误差目标。',
        '主拟合每条 checkpoint 的原始权重为 1/n_g，各轨迹总权重相同，整体归一化至 1。objective=sum_i w_i (L_i-Lhat_i)^2，least_squares 内部 cost 的两倍。',
        f"每次拟合 {settings['multistart_count']} 个不同初值，初始 alpha/beta 在 {settings['initial_exponent_range']} 对数分散，初始幅度依实际 N/D/Loss 尺度构造；选成功、正值、有限且目标最小的解。主模型成功初值 {len(successful)}/{len(all_starts)}。参数边界明示于新配置 {settings['parameter_bounds']}，不是统计 CI；边界接触记录在多初值表。",'',
        md_table([dict(parameter=k,estimate=v) for k,v in params.items()],['parameter','estimate']),
        f"主目标={primary['objective']:.12g}；B1 表内 RMSE={fitted['RMSE']:.12g}；状态={primary['status']}。",'',
        '## 4. 多初值、可识别性与不确定性','',
        md_table(spread,['parameter','min','max','relative_range']),
        f"主拟合 log-parameter 加权 Jacobian 秩={primary['jacobian_rank']}/5，条件数={primary['jacobian_condition_number']:.9g}。条件数依赖参数化尺度，不等于外部科学可识别性的完整证据。",'',
        f"Bootstrap 单位是完整模型轨迹，每次抽取 {d['B1_trajectory_count']} 条轨迹、有放回且保留各自全部 checkpoint，重复抽中的轨迹按 multiplicity 增加权重。共 {boot_summary['attempted']} 次，每次 {settings['bootstrap_multistart_count']} 初值重新拟合。独立固定 RNG 种子 {settings['random_state']}，失败率 {boot_summary['solver_failure_rate']:.6%}；收敛但秩不足 {boot_summary['converged_nonidentifiable']} 次，参数 CI 使用 {boot_summary['identifiable']} 个可识别拟合。没有 checkpoint IID bootstrap。",'',
        md_table(boot_stats,['parameter','main_estimate','median','mean','std','CI95_low','CI95_high','min','max']),
        md_table(tails,['parameter','full_range_over_95CI_width','min_extreme_distinct_trajectories','max_extreme_distinct_trajectories']),
        f"存在稀疏轨迹抽样对应的尾部，极值的轨迹覆盖数见上表。全范围比 95% CI 更宽，但仍需结合前表的绝对数值判断。bootstrap 接触参数边界次数={touching}。不删去可识别的尾部拟合，{boot_summary['identifiable']} 个可识别结果用于 CI。",
        '以上为给定八条轨迹的 cluster-bootstrap percentile CI，不包括 tokenizer、评测集、源数据生成或跨模型族偏差。B1 各轨迹共用 checkpoint/token 网格且数据高度贴合加性幂律，窄 CI 不能证明真实训练损失的不确定性同样小。',
        f"B1 拟合均值的 bootstrap 区间平均宽度={boot_summary['prediction_mean_CI_width']:.12g}，最大宽度={boot_summary['prediction_max_CI_width']:.12g}。它们是拟合均值不确定性，不是未来实验 Loss 的预测区间。",'',
        md_table(pairs,['parameter_1','parameter_2','bootstrap_Pearson']),
        '相关矩阵、alpha-beta、E 与其余参数、极值和 1%/99% 尾部分位均已输出；参数相关不自动意味着秩亏，但强相关提示共同变化方向。仅有八个模型规模，bootstrap 来源覆盖有限，不能将 1176 行当成 1176 个独立实验来增强置信度。','',
        '## 5. 留一轨迹验证与简单基线','',
        '按完整轨迹执行 leave-one-model-out，八折；每次训练其余七条全部 checkpoint，完整留出一条。所有训练/留出行号及模型参数已持久化；不存在随机 checkpoint KFold。主模型和仅 N、仅 D 对照使用相同留出分组和估计目标。','',
        md_table(main_cv,['heldout_trajectory','N_heldout','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias']),
        md_table(comparison,['model','folds','RMSE','MAE','R2','Pearson','Spearman']),
        '仅 N 模型在同一规模轨迹内预测常数，Pearson/Spearman 为未定义，不能伪造零相关系数；这也是 N-only 无法解释训练进度的直接限制。','',
        '## 6. balanced/unweighted 对照','',
        md_table(weighting,['weighting','E','A','B','alpha','beta','objective','RMSE','max_abs_prediction_difference_from_balanced']),
        '本次八条轨迹均有 147 个 checkpoint，balanced 和 unweighted 的归一化权重数学等价，出现相同解是预期结果，不是加权方案对一般不平衡数据无用。','',
        '## 7. terminal-only 对照及结构性不可识别','',
        md_table(terminal_rows,['dataset','n_fit','E','A','B','alpha','beta','jacobian_rank','terminal_RMSE','full_B1_RMSE_descriptive','E_effective_at_terminal_D']),
        md_table(terminal_spread,['parameter','min','max']),
        '全部终点 D 相同，只有 E_eff=E+B D_terminal^-beta 可识别。terminal-only Jacobian 最多秩 3，五参数存在结构性混淆，多初值可有近似相同终点预测但不同 E/B/beta。该解只作敏感性，不替换主模型；主模型包含训练轨迹才能区分数据项和不可约项。',
        '两种约束模型对 N、D 的理论导数均为负，但这种同向来自正参数约束；终点数据没有 D 变化，不能说终点经验验证了 beta 的方向。终点 N–Loss 趋势可检验，D 方向不可由这些终点单独验证。','',
        '## 8. 冻结与 B3 插值一致性','',
        '完成所有 B1 拟合与训练内诊断后写入 freeze_before_validation.json，后续验证只应用主模型参数，没有重拟合或偏置校准。B3 插值轨迹与 B1 共用模型规模和训练范围，因此不是独立真实实验。','',
        md_table(validations['B3'][0],['stratum','group_key','n','RMSE','MAE','Pearson','Spearman','mean_bias']),
        '插值表可检验数值/曲线一致性，但其较密集的 4000 个点不产生额外独立实验样本量。','',
        '## 9. B2 半合成辅助族外检验','',
        md_table(b2,['stratum','group_key','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias']),
        f"B2 总体 RMSE={b2[0]['RMSE']:.9g}、平均残差={b2[0]['mean_bias']:+.9g}，绝对 Loss 系统低估明显。各模型轨迹内 Spearman 在 {min(a['Spearman'] for a in b2[1:]):.6f}–{max(a['Spearman'] for a in b2[1:]):.6f}，整体 Spearman={b2[0]['Spearman']:.6f}。单轨迹下降排序一致不能弥补族外绝对水平失败。",
        '排序/下降趋势与绝对误差分别判断；B2 的半合成属性限制对真实跨族泛化的解释。没有借 B2 修正 E 或其他参数。','',
        '## 10. B4 跨族表一致性','',
        md_table(validations['B4'][0],['stratum','group_key','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias','median_residual']),
        f"B4 整体 Spearman={validations['B4'][0][0]['Spearman']:.6f}，但平均残差={validations['B4'][0][0]['mean_bias']:+.6f}，存在系统偏差。family 平均残差范围 {min(a['mean_bias'] for a in family):+.6f}–{max(a['mean_bias'] for a in family):+.6f}；表中 Pythia 层也有平均残差 {next(a['mean_bias'] for a in family if a['group_key']=='Pythia'):+.6f}。因此支持该表上的排序趋势迁移，不支持统一绝对 Loss 预测。",
        'B4 没有给出可核实的共同 evaluation dataset/tokenizer 口径。表上的整体排序与家族分层残差是数值诊断，不能单凭整体相关就断言 Pythia Loss 参数可直接迁移到全部族。不同 family 的偏置不能通过全局校准掩盖，本阶段未校准。','',
        '## 11. B5 文献分层验证','',
        md_table(source,['group_key','n','Pearson','Spearman','mean_bias','median_residual','RMSE']),
        f"B5 source 内 Spearman 范围 {min(a['Spearman'] for a in source):.6f}–{max(a['Spearman'] for a in source):.6f}，支持整理表中的来源内趋势一致性；没有证明不同来源的绝对 Loss 已统一。",
        'B5 的 source/family 均保留分层。没有 loss_type、dataset、tokenizer，无法确认跨文献绝对 Loss 可比性；核心证据是 source 内的趋势和排序，表中 RMSE/残差仅为附件数值的描述。每 source 点数也较少，相关不能视为充分实验验证。',
        '没有从文献标签推断未提供的测量协议，也没有按验证值选择参数。完整 family/source-family 层在 B5_literature_validation.csv。','',
        '## 12. 残差结构','',
        md_table(associations,['variable','Pearson','Spearman','n_checkpoints','independent_trajectory_count']),
        md_table(residual_summary,['stratum','group','n','mean_bias','RMSE','residual_std','lag1_residual_Pearson']),
        f"早期训练 P10 RMSE={early['RMSE']:.12g}，末期 P90 RMSE={late['RMSE']:.12g}。残差形状、相关和相邻 checkpoint 相关是结构性诊断，不做 IID 显著性 p 值。", 
        f"早期/末期 RMSE 比={early['RMSE']/late['RMSE']:.6f}，存在早期误差较大的异方差结构；总体残差与 log N/log D 的 Pearson 较小，未见强的全局线性均值偏移。即便早期误差较大，仍应同时报告其绝对数值，不能把相对差异夸大为大幅模型失效。",
        '即使绝对残差很小，也需区分表内近幂律拟合与真实跨数据源泛化。残差与 log D/progress 或小规模模型的结构不通过增加复杂模型消除，本阶段保持经典基线。','',
        '## 13. 是否可以进入下一阶段','',
        '当前经典基线的 B1 参数估计与轨迹分组预测稳定，可以作为后续受控扩展检验的冻结参考；这不等于绝对 Loss 已跨族/跨文献普适。B2 的系统误差、B4 的 family 偏差、B5 的未知 Loss 口径应单独保留。',
        '后续若人工批准质量扩展，需要验证是否超出这一基线的真实增益，而不能把家族/口径/数据来源的偏差强行归因于质量。只有八个轨迹的 cluster CI 无法涵盖这些误差。',
        f"冻结旧项目文件 {protected_n} 个的 SHA-256 全部一致。验收与独立公式重建在 independent_verification.csv；所有数值来源、bootstrap 抽样、32 初值拟合、CV 分组均可追溯。",
        '**本轮结束于问题二 PHASE 1，等待人工确认；未构造质量扩展、未读取 B6–B10 进行建模、未进入问题三。**','']
    (ROOT/'outputs/report/question2_phase1_classical_scaling.md').write_text('\n\n'.join(lines),encoding='utf-8')
    write_csv(OUT/'multistart_parameter_stability.csv',spread)
    write_csv(OUT/'parameter_pair_correlations.csv',pairs)
    write_csv(OUT/'parameter_tail_diagnostics.csv',tails)
    write_csv(OUT/'terminal_multistart_parameter_spread.csv',terminal_spread)


def refresh_saved(figures=False):
    """Regenerate only presentation from completed saved fits; never call fit/bootstrap."""
    def numeric(path):
        records=read_rows(path)
        for r in records:
            for key,value in r.items():
                if value=='True':r[key]=True
                elif value=='False':r[key]=False
                elif value=='':r[key]=None
                else:
                    try:r[key]=float(value)
                    except ValueError:pass
        return records
    if not (OUT/'execution_summary.json').exists():raise RuntimeError('Only completed results can be refreshed')
    frozen=json.loads((OUT/'frozen_project_before.json').read_text(encoding='utf-8'))
    freeze=json.loads((OUT/'freeze_before_validation.json').read_text(encoding='utf-8'))
    for path,digest in freeze['files'].items():
        if sha256(ROOT/path)!=digest:raise RuntimeError('Frozen phase1 parameters/config changed')
    files,trajectory_paths,_=input_paths(ROOT);b1=load_csv(files['B1'],'B1');tables={k:load_csv(v,k) for k,v in files.items() if k!='B1'}
    trajectories=[load_csv(path,'B3') for path in trajectory_paths]
    models=json.loads((OUT/'baseline_models.json').read_text(encoding='utf-8'));unweighted=json.loads((OUT/'unweighted_model.json').read_text(encoding='utf-8'))
    boot_summary=json.loads((OUT/'bootstrap_summary.json').read_text(encoding='utf-8'))
    settings=yaml.safe_load((ROOT/'configs/scaling_phase1.yaml').read_text(encoding='utf-8'))
    validations={}
    for label in ['B2','B3','B4','B5']:
        name=f'{label}_validation.csv' if label in ['B2','B3'] else ('B4_cross_family_validation.csv' if label=='B4' else 'B5_literature_validation.csv')
        pred=numeric(OUT/f'{label}_direct_predictions.csv');validations[label]=(numeric(OUT/name),np.array([r['predicted_loss'] for r in pred]))
    residuals=numeric(OUT/'B1_residuals.csv');bootstrap_rows=numeric(OUT/'parameter_bootstrap_draws.csv')
    if figures:plot_all(b1,models,numeric(OUT/'B1_group_cv_predictions.csv'),bootstrap_rows,tables,trajectories,validations,residuals,settings)
    protected_n=diagnostic.verify_protected(frozen)
    build_report(b1,models,numeric(OUT/'multistart_results.csv'),numeric(OUT/'B1_group_cv.csv'),numeric(OUT/'baseline_model_comparison.csv'),
        unweighted,numeric(OUT/'parameter_bootstrap.csv'),boot_summary,numeric(OUT/'terminal_checkpoint_sensitivity.csv'),validations,
        numeric(OUT/'residual_associations.csv'),numeric(OUT/'residual_structure_summary.csv'),settings,protected_n)
    print('Saved-result report/figures refreshed; no fitting, bootstrap or calibration.')


def main():
    """Complete B1-only fitting first, freeze parameters, then validate other sources."""
    if (OUT/'execution_summary.json').exists():raise RuntimeError('Phase 1 already completed and frozen; do not rerun fitting.')
    if not (OUT/'data_diagnosis.json').exists():raise RuntimeError('Run 10_diagnose_scaling_data.py first; no blind fit')
    for p in [OUT,FIG,ROOT/'outputs/report']:p.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'fitting.log',encoding='utf-8')],force=True)
    frozen=json.loads((OUT/'frozen_project_before.json').read_text(encoding='utf-8'));diagnostic.verify_protected(frozen)
    settings=yaml.safe_load((ROOT/'configs/scaling_phase1.yaml').read_text(encoding='utf-8'))
    if settings['multistart_count']<30 or settings['bootstrap_multistart_count']<30:raise ValueError('At least 30 starts required')
    files,trajectory_paths,_=input_paths(ROOT);b1=load_csv(files['B1'],'B1')
    # Only B1 is loaded numerically until all fits/parameter choices are frozen.
    models,starts,cv,comparison,unweighted=fit_and_cv(b1,settings)
    bootstrap_rows,boot_stats,boot_summary=cluster_bootstrap(b1,models['M0_N_D'],settings)
    terminal,terminal_rows=terminal_fit(b1,models['M0_N_D'],settings)
    residuals,associations,residual_summary=residual_diagnostics(b1,models['M0_N_D'])
    freeze_paths=[OUT/name for name in ['main_model.json','baseline_models.json','unweighted_model.json','group_cv_models.json','terminal_only_model.json']]+[ROOT/'configs/scaling_phase1.yaml']
    freeze={p.relative_to(ROOT).as_posix():sha256(p) for p in freeze_paths}
    write_json(OUT/'freeze_before_validation.json',dict(fit_datasets=['B1'],validation_datasets=['B2','B3','B4','B5'],
        parameters_used_for_validation=models['M0_N_D']['parameters'],files=freeze,validation_used_for_parameter_selection=False))
    tables={label:load_csv(path,label) for label,path in files.items() if label!='B1'}
    trajectories=[load_csv(path,'B3') for path in trajectory_paths];b3=combine_trajectories(trajectories);parameters=models['M0_N_D']['parameters']
    validations={}
    validations['B3']=validate(b3,parameters,'B3',{'trajectory':b3['group']},'interpolated_trajectory_consistency',True)
    b2=tables['B2'];validations['B2']=validate(b2,parameters,'B2',{'trajectory':b2['group']},'semi_synthetic_auxiliary',False)
    b4=tables['B4'];validations['B4']=validate(b4,parameters,'B4',{'family':[r['family'] for r in b4['records']]},'cross_family_table',False)
    b5=tables['B5'];validations['B5']=validate(b5,parameters,'B5',{'source':[r['source'] for r in b5['records']],
        'family':[r['family'] for r in b5['records']],'source_family':[r['source']+' / '+r['family'] for r in b5['records']]},'literature_compilation_evaluation_scope_unconfirmed',False)
    for label,(stats,_) in validations.items():logging.info('%s frozen prediction RMSE=%.7g Spearman=%s',label,stats[0]['RMSE'],stats[0]['Spearman'])
    predictions=read_rows(OUT/'B1_group_cv_predictions.csv')
    plot_all(b1,models,predictions,bootstrap_rows,tables,trajectories,validations,residuals,settings)
    protected_n=diagnostic.verify_protected(frozen)
    build_report(b1,models,starts,cv,comparison,unweighted,boot_stats,boot_summary,terminal_rows,validations,associations,residual_summary,settings,protected_n)
    for p,h in freeze.items():
        if sha256(ROOT/p)!=h:raise RuntimeError('Scaling parameters/config changed during validation')
    summary=dict(main_model='M0_N_D',main_fit_dataset='B1_only',parameters=parameters,
        objective=models['M0_N_D']['objective'],jacobian_rank=models['M0_N_D']['jacobian_rank'],
        bootstrap=boot_summary,group_cv=comparison,validation_overall={label:data[0][0] for label,data in validations.items()},
        frozen_project_files_unchanged=protected_n,quality_extension_started=False,phase3_started=False)
    write_json(OUT/'execution_summary.json',summary)
    logging.info('COMPLETE: question2 phase1; bootstrap=%d; frozen files=%d; stopped for human confirmation',len(bootstrap_rows),protected_n)


if __name__=='__main__':
    try:
        if sys.argv[1:] in [['--report-only'],['--figures-only']]:refresh_saved(sys.argv[1:]==['--figures-only'])
        else:main()
    except Exception:
        logging.exception('Scaling phase1 failed; frozen files must remain unchanged')
        raise
