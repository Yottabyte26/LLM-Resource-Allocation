"""Question 1 part 3 final: frozen-model effects and train-only Qp/Kp validation.

Never rerun prior stages, alter attachments, or tune against test/estimated Loss.
All new model fitting uses A4/A5. Long Monte Carlo runs have a separate checkpoint.
"""
from __future__ import annotations
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
import sys
import warnings
import joblib
import numpy as np
import yaml
from scipy.stats import spearmanr,rankdata,trim_mean
from sklearn.exceptions import ConvergenceWarning
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mixture.io import load_datasets,signatures
from mixture.composition import ILRTransformer
from mixture.modeling import scores,group_folds,tune as tune_base
from mixture.effects import baselines,analyze,perturb,direction,summary
from mixture.augmented import tune,candidate,metadata,checked
from quality.io import sha256_file
from quality.vector_diagnostics import write_rows

OUT=ROOT/'outputs/mixture'
STAGE=OUT/'stage2'
FIG=ROOT/'outputs/figures/question1_part3/stage2'


def read_csv(path):
    """Strict nonempty CSV parse; no silent missing data substitutions."""
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);rows=list(reader)
    if not rows or any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f'Empty/malformed CSV: {path}')
    return rows


def dump(path,data):
    """Write finite JSON and automatically create its directory."""
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')


def short(name):
    """Display only; saved schema retains exact official column names."""
    return name.removeprefix('train_the_pile_').removeprefix('metric/the_pile_').removesuffix('_val_loss')


def capture_protected():
    """Snapshot every existing original/prior result and frozen source before work.

    A subsequent run reuses the original manifest, so it cannot legitimize an
    accidental change to an old file. New outputs are not part of this snapshot.
    """
    manifest=STAGE/'protected_before.json'
    if manifest.exists():
        snapshot=json.loads(manifest.read_text(encoding='utf-8'))
        verify_protected(snapshot)
        return snapshot
    paths=[]
    for folder in ['real_attachments','outputs','configs','src','scripts']:
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.is_relative_to(STAGE):continue
            if p.name in ['mixture_analysis.yaml','effects.py','augmented.py'] or p.name.startswith('09_'):continue
            paths.append(p)
    snapshot={p.relative_to(ROOT).as_posix():sha256_file(p) for p in sorted(paths)}
    dump(manifest,snapshot)
    return snapshot


def verify_protected(snapshot):
    """Stop before publishing if any already locked file changed."""
    rows=[]
    for relative,digest in snapshot.items():
        path=ROOT/relative
        current=sha256_file(path) if path.exists() else 'MISSING'
        rows.append(dict(path=relative,before_sha256=digest,after_sha256=current,unchanged=current==digest))
        if current!=digest: raise RuntimeError(f'Locked input/result changed: {relative}')
    write_rows(STAGE/'locked_files_integrity.csv',rows)
    return len(rows)


def frozen_oof(train):
    """Align already frozen M1 OOF predictions by index, without refitting M1."""
    records=read_csv(OUT/'nested_OOF_predictions.csv')
    lookup={r['index']:r for r in records if r['response']=='L_mean' and r['model']=='M1_ElasticNet'}
    if len(lookup)!=len(train['indices']):raise ValueError('Frozen OOF index mismatch')
    if any(abs(float(lookup[i]['true_loss'])-train['L_mean'][k])>1e-12 for k,i in enumerate(train['indices'])):
        raise ValueError('Frozen OOF response mismatch')
    return np.array([float(lookup[i]['predicted_loss']) for i in train['indices']])


def scale_diagnostics(datasets,models):
    """Residual is y-p; centered RMSE is explanatory only and never a calibration."""
    rows=[]
    for model_name,model in models.items():
        for label in ['test_1M','test_60M','test_1B']:
            d=datasets[label];pred=model.predict(d['P']);e=d['L_mean']-pred
            rows.append(dict(model=model_name,dataset=label,residual_convention='observed_minus_prediction',
                mean_bias=float(e.mean()),median_bias=float(np.median(e)),residual_std=float(e.std(ddof=1)),
                centered_RMSE_diagnostic_only=float(np.sqrt(np.mean((e-e.mean())**2))),
                calibration_applied=False,**scores(d['L_mean'],pred)))
    write_rows(OUT/'scale_transfer_diagnostics.csv',rows)
    return rows


def mapping(names,settings):
    """Retain A16 original '(none)' while labeling researcher proxy assumptions."""
    guide=read_csv(ROOT/'real_attachments/A_data_value/domain_mapping_guide.csv')
    guide={r['mixture_domain']:r for r in guide}
    quality={r['domain']:r for r in read_csv(ROOT/'outputs/quality/domain_quality_summary_A1.csv')}
    rows=[]
    for name in names:
        domain=short(name);original=guide[domain];inferred=original['mapping_type']=='inferred'
        if inferred:
            if original['quality_domain']!='(none)':raise ValueError('Changed A16 inferred schema; inspect before proceeding')
            proxy=settings['inferred_mapping'][domain];qd=proxy['quality_domain'];rationale=proxy['rationale']
        else:qd=original['quality_domain'];rationale=original['note']
        q=quality[qd]
        rows.append(dict(mixture_domain=name,mixture_domain_short=domain,quality_domain=qd,
            mapping_type=original['mapping_type'],A16_quality_domain=original['quality_domain'],A16_note=original['note'],
            researcher_assumption=inferred,association_status='assumed_proxy' if inferred else original['mapping_type'],
            mapping_rationale=rationale,Q_domain=float(q['Q_domain']),Q_CI_low=float(q['ci95_low']),Q_CI_high=float(q['ci95_high']),
            Q_source='outputs/quality/domain_quality_summary_A1.csv',quality_n=int(q['n'])))
    if len(rows)!=17:raise ValueError('Mapping must contain exactly 17 domains')
    write_rows(OUT/'domain_quality_mapping.csv',rows)
    return rows,quality


def original_k_domains():
    """Aggregate existing K only; no K recomputation, normalization or penalty."""
    values=defaultdict(list)
    with (ROOT/'outputs/conflict/A1_sample_conflict.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):values[r['domain']].append(float(r['K']))
    if sum(map(len,values.values()))!=51230:raise ValueError('Unexpected locked A1 K record count')
    rows=[dict(domain=d,n=len(v),K_domain=float(trim_mean(v,.1)),trim_fraction_per_tail=.1,
               source='outputs/conflict/A1_sample_conflict.csv') for d,v in sorted(values.items())]
    write_rows(STAGE/'domain_conflict_trimmed_summary_A1.csv',rows)
    return {r['domain']:r['K_domain'] for r in rows}


def descriptors(datasets,q,k):
    """Export fixed Qp/Kp derived from original closed shares; K is descriptive."""
    rows=[]
    for label,d in datasets.items():
        closed=ILRTransformer.closure(d['P']);qp=closed@q;kp=closed@k
        for i,index in enumerate(d['indices']):
            rows.append(dict(dataset=label,index=index,loss_provenance=d['provenance'],Q_p=float(qp[i]),K_p=float(kp[i]),
                composition_used='closed_original_shares',mapping_scheme='A_assumed_proxy'))
    write_rows(OUT/'recipe_quality_conflict_descriptors.csv',rows)


def nested_augmented(label,q,k,train,names,settings,outer,store=True):
    """Matched outer folds, inner fold preprocessing, plus final train-only selection."""
    P,y=train['P'],train['L_mean'];ids=train['indices'];oof=np.empty(len(y));outer_models=[];metas=[];grid_rows=[];fold_rows=[]
    for fold,(tr,va) in enumerate(outer,1):
        inner=group_folds(P[tr],settings['inner_folds'],settings['random_state']+fold)
        model,params,grid=tune(q,k,P[tr],y[tr],settings,inner)
        oof[va]=model.predict(P[va]);outer_models.append((tr,va,params,model))
        metas.append(metadata(model,params,[ids[i] for i in tr],names,label+f'_outer{fold}'))
        fold_rows.append(dict(model=label,outer_fold=fold,**scores(y[va],oof[va])))
        grid_rows += [dict(model=label,outer_fold=fold,**r) for r in grid]
        logging.info('%s outer fold %d/5 RMSE %.6f',label,fold,fold_rows[-1]['RMSE'])
    folds=group_folds(P,settings['final_selection_folds'],settings['random_state']+100)
    model,params,grid=tune(q,k,P,y,settings,folds)
    final_meta=metadata(model,params,ids,names,label)
    grid_rows += [dict(model=label,outer_fold='full_training',**r) for r in grid]
    if store:
        joblib.dump(model,STAGE/'models'/f'{label}.joblib')
        dump(STAGE/'models'/f'{label}.json',final_meta)
        dump(STAGE/'models'/f'{label}_outer.json',metas)
        write_rows(STAGE/f'{label}_hyperparameter_search.csv',grid_rows)
        write_rows(STAGE/f'{label}_OOF_predictions.csv',[dict(index=ids[i],true_loss=float(y[i]),predicted_loss=float(oof[i])) for i in range(len(y))])
    return dict(model=model,params=params,meta=final_meta,oof=oof,outer=outer_models,fold_rows=fold_rows,
                cv=dict(model=label,**scores(y,oof)))


def single_loss_models(train,bases,names,loss_names,settings):
    """26 lightweight first-order models; 5-fold training selection, no test selection."""
    folds=group_folds(train['P'],settings['final_selection_folds'],settings['random_state']+100)
    records=[];model_rows=[]
    x=np.stack([p for _,_,p in bases]);xp=np.stack([perturb(p,[j],.01) for _,_,p in bases for j in range(17)])
    for col,loss_name in enumerate(loss_names):
        y=train['losses'][:,col]
        for label in ['M1_ElasticNet','M1_Ridge']:
            model,params,grid=tune_base(label,train['P'],y,settings,folds)
            pipe=model.regressor_;reg=pipe.named_steps['regression']
            if getattr(reg,'n_iter_',None) is not None and reg.n_iter_>=reg.max_iter:
                raise RuntimeError('Single-loss model did not converge')
            pred=model.predict(x);effects=model.predict(xp).reshape(len(bases),17)-pred[:,None]
            for j,name in enumerate(names):
                records.append(dict(model=label,training_domain=name,validation_loss=loss_name,step=.01,
                    n_baselines=len(bases),**summary(effects[:,j]),baseline_direction=direction(effects[:,j])))
            model_rows.append(dict(validation_loss=loss_name,model=label,parameters=params,
                best_training_5fold_selection_RMSE=min(r['mean_CV_RMSE'] for r in grid),n_fit=len(y),
                zero_delta=pipe.named_steps['ilr'].zero_delta_,selection_dataset='A4/A5_only',
                performance_note='selection_CV_not_nested_unbiased_performance'))
            joblib.dump(model,STAGE/'single_loss_models'/f'{short(loss_name)}_{label}.joblib')
        logging.info('Single-loss diagnostic %d/13: %s',col+1,short(loss_name))
    for r in records:
        other=next(s for s in records if s['model']!=r['model'] and s['training_domain']==r['training_domain'] and s['validation_loss']==r['validation_loss'])
        r['models_mean_direction_agree']=bool(np.sign(r['mean_effect'])==np.sign(other['mean_effect']))
        r['stable_across_models_and_baselines']=r['baseline_direction']==other['baseline_direction'] and r['baseline_direction']!='uncertain'
    write_rows(OUT/'train_domain_vs_validation_loss_effects.csv',records)
    write_rows(STAGE/'single_loss_model_selection.csv',model_rows)
    return records


def monte_carlo(m3,mapping_rows,train,datasets,names,settings,analysis_settings,baseline_oof):
    """500 conditional refits, reusing A4/A5-selected hyperparameters per outer fold.

    Independent uniform draws for each inferred domain. Sampling ranges are
    proxy bootstrap CIs, not credible bounds for semantic mapping error.
    All fold scalers/zero parameters remain learned from their training folds.
    """
    checkpoint=STAGE/'mapping_MC_checkpoint.json'
    seed=analysis_settings['random_state'];n=analysis_settings['monte_carlo_repetitions']
    rng=np.random.default_rng(seed);base_q=np.array([r['Q_domain'] for r in mapping_rows]);inferred=np.array([r['researcher_assumption'] for r in mapping_rows])
    draws=np.tile(base_q,(n,1))
    for j,r in enumerate(mapping_rows):
        if inferred[j]:draws[:,j]=rng.uniform(r['Q_CI_low'],r['Q_CI_high'],size=n)
    write_rows(STAGE/'mapping_MC_draws.csv',[dict(repetition=i+1,**dict(zip(names,row))) for i,row in enumerate(draws)])
    rows=[];prediction_sum={label:np.zeros(len(d['P'])) for label,d in datasets.items()};prediction_square={label:np.zeros(len(d['P'])) for label,d in datasets.items()}
    if checkpoint.exists():
        saved=json.loads(checkpoint.read_text(encoding='utf-8'))
        if saved['draw_hash']!=sha256_file(STAGE/'mapping_MC_draws.csv'):raise ValueError('MC configuration changed; cannot resume')
        rows=saved['rows'];prediction_sum={k:np.array(v) for k,v in saved['prediction_sum'].items()};prediction_square={k:np.array(v) for k,v in saved['prediction_square'].items()}
        logging.info('Resuming conditional MC at %d/%d',len(rows),n)
    P,y=train['P'],train['L_mean'];base_cv=scores(y,baseline_oof)
    for i in range(len(rows),n):
        q=draws[i];oof=np.empty(len(y))
        for tr,va,params,_ in m3['outer']:
            model,_=candidate(q,None,settings);model.set_params(**params).fit(P[tr],y[tr]);checked(model)
            oof[va]=model.predict(P[va])
        final,_=candidate(q,None,settings);final.set_params(**m3['params']).fit(P,y);checked(final)
        meta=metadata(final,m3['params'],train['indices'],names,'M3_MC')
        row=dict(repetition=i+1,gamma=meta['gamma_original_loss_per_Qp'],CV_RMSE=scores(y,oof)['RMSE'],
                 CV_MAE=scores(y,oof)['MAE'],CV_R2=scores(y,oof)['R2'],CV_Spearman=scores(y,oof)['Spearman'])
        row['CV_RMSE_difference_from_M1']=row['CV_RMSE']-base_cv['RMSE']
        for label,d in datasets.items():
            pred=final.predict(d['P']);prediction_sum[label]+=pred;prediction_square[label]+=pred**2
            if label!='train_1M':
                s=scores(d['L_mean'],pred)
                row[label+'_RMSE']=s['RMSE'];row[label+'_Spearman']=s['Spearman']
        rows.append(row)
        if (i+1)%25==0 or i+1==n:
            dump(checkpoint,dict(draw_hash=sha256_file(STAGE/'mapping_MC_draws.csv'),rows=rows,
                prediction_sum={k:v.tolist() for k,v in prediction_sum.items()},prediction_square={k:v.tolist() for k,v in prediction_square.items()}))
            logging.info('Mapping conditional MC %d/%d',i+1,n)
    write_rows(OUT/'quality_mapping_monte_carlo_results.csv',rows)
    summary_rows=[]
    for key in rows[0]:
        if key=='repetition':continue
        values=np.array([r[key] for r in rows])
        summary_rows.append(dict(metric=key,repetitions=n,mean=float(values.mean()),std=float(values.std(ddof=1)),
            min=float(values.min()),P025=float(np.quantile(values,.025)),median=float(np.median(values)),P975=float(np.quantile(values,.975)),max=float(values.max())))
    write_rows(OUT/'quality_mapping_monte_carlo_summary.csv',summary_rows)
    prediction_rows=[]
    for label,d in datasets.items():
        mean=prediction_sum[label]/n;std=np.sqrt(np.maximum(0,prediction_square[label]/n-mean**2))
        prediction_rows += [dict(dataset=label,index=index,mean_prediction=float(mean[i]),std_prediction=float(std[i])) for i,index in enumerate(d['indices'])]
    write_rows(STAGE/'mapping_MC_prediction_summary.csv',prediction_rows)
    return rows,summary_rows


def extrapolation(datasets,models):
    """Estimated recipe subset consistency; no calibration or new external evidence."""
    train=datasets['train_1M'];lookup={key:i for i,key in enumerate(signatures(train['P']))}
    rows=[];pred_rows=[];matches={};links=[]
    for label in ['estimated_10B','estimated_70B']:
        d=datasets[label];positions=np.array([lookup[key] for key in signatures(d['P'])]);matches[label]=positions
        for name,model in models.items():
            pred=model.predict(d['P']);m=scores(d['L_mean'],pred)
            rows.append(dict(dataset=label,model=name,validation_kind='estimated_training_recipe_subset_consistency',
                calibration_applied=False,**m))
            pred_rows += [dict(dataset=label,model=name,index=index,matched_train_index=train['indices'][positions[i]],
                estimated_loss=float(d['L_mean'][i]),uncalibrated_prediction=float(pred[i]),
                estimated_rank=float(rankdata(d['L_mean'])[i]),predicted_rank=float(rankdata(pred)[i])) for i,index in enumerate(d['indices'])]
            links.append(dict(comparison=f'{name}_prediction_vs_{label}',n=len(pred),Spearman=m['Spearman']))
        y1=train['L_mean'][positions];ye=d['L_mean'];slope=np.dot(y1,ye)/np.dot(y1,y1)
        pair_ranks=[]
        for col in range(13):
            actual=train['losses'][positions,col];estimated=d['losses'][:,col]
            ratio=np.dot(actual,estimated)/np.dot(actual,actual)
            pair_ranks.append(dict(dataset=label,validation_loss=d['loss_table']['columns'][col],
                through_origin_diagnostic_ratio=float(ratio),max_abs_deviation=float(abs(estimated-ratio*actual).max()),
                observed_vs_estimated_Spearman=float(spearmanr(actual,estimated).statistic)))
        write_rows(STAGE/f'{label}_generation_relationship.csv',pair_ranks)
        links.append(dict(comparison=f'observed_train_1M_vs_{label}',n=len(ye),Spearman=float(spearmanr(y1,ye).statistic),
            diagnostic_ratio_estimated_over_1M=float(slope),diagnostic_max_abs_scaling_deviation=float(abs(ye-slope*y1).max())))
    a=datasets['estimated_10B'];b=datasets['estimated_70B'];blookup={key:i for i,key in enumerate(signatures(b['P']))}
    order=[blookup[key] for key in signatures(a['P'])]
    links.append(dict(comparison='estimated_10B_vs_estimated_70B_same_recipes',n=len(order),Spearman=float(spearmanr(a['L_mean'],b['L_mean'][order]).statistic)))
    write_rows(OUT/'estimated_scale_consistency.csv',rows)
    write_rows(OUT/'estimated_scale_predictions.csv',pred_rows)
    write_rows(OUT/'matched_recipe_rank_consistency.csv',links)
    return rows,links


def figures(names,loss_names,effects,stability,single,cv,validation,mc):
    """Standalone scientific plots; effects are signed changes in uncalibrated Loss."""
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,path):
        fig.tight_layout();fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,9),sharey=True)
    for ax,model in zip(axes,['M1_ElasticNet','M1_Ridge']):
        rows=[next(r for r in effects if r['model']==model and r['step']==.01 and r['mixture_domain']==n) for n in names]
        means=np.array([r['mean_effect'] for r in rows]);lo=np.array([r['min_effect'] for r in rows]);hi=np.array([r['max_effect'] for r in rows])
        ax.barh(np.arange(17),means,color=['#4777a8' if m<0 else '#b77755' for m in means])
        ax.errorbar(means,np.arange(17),xerr=np.stack([means-lo,hi-means]),fmt='none',ecolor='#555555',capsize=3)
        ax.axvline(0,color='#777777',lw=.8);ax.set(yticks=np.arange(17),yticklabels=[short(n) for n in names],
            title=model,xlabel='Predicted Loss change (+1 pp); bars = mean, whiskers = baseline range')
    fig.suptitle('Five A4 baselines; associations, not causal effects');save(fig,FIG/'marginal_effects.png')
    matrices=[]
    for model in ['M1_ElasticNet','M1_Ridge']:
        matrix=np.array([[next(r['mean_effect'] for r in single if r['model']==model and r['training_domain']==n and r['validation_loss']==loss) for loss in loss_names] for n in names]);matrices.append(matrix)
    vmax=max(abs(x).max() for x in matrices)
    fig,axes=plt.subplots(1,2,figsize=(18,11),sharey=True)
    for ax,matrix,label in zip(axes,matrices,['ElasticNet','Ridge']):
        im=ax.imshow(matrix,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
        ax.set(xticks=range(13),xticklabels=[short(n) for n in loss_names],yticks=range(17),yticklabels=[short(n) for n in names],
            title=f'{label}: +1 percentage point',xlabel='Validation Loss domain',ylabel='Training mixture domain')
        ax.tick_params(axis='x',rotation=65)
    fig.subplots_adjust(left=.12,right=.88,bottom=.22,wspace=.16,top=.92)
    color_axis=fig.add_axes([.91,.3,.016,.52])
    fig.colorbar(im,cax=color_axis,label='Mean predicted Loss change')
    path=ROOT/'outputs/figures/question1_part3/train_domain_validation_effect_heatmap.png'
    fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].bar([r['model'] for r in cv],[r['RMSE'] for r in cv],color='#4777a8');axes[0].set(ylabel='Nested OOF RMSE',title='A4/A5 only: matched outer folds')
    for model in ['M1_ElasticNet','M3_Qp','M4_Qp_Kp']:
        rows=[r for r in validation if r['model']==model]
        axes[1].plot(range(3),[r['Spearman'] for r in rows],'o-',label=model)
    axes[1].set(xticks=range(3),xticklabels=['1M','60M','1B'],ylabel='Spearman',title='Uncalibrated observed-table ranking');axes[1].legend()
    save(fig,FIG/'quality_conflict_model_comparison.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].hist([r['gamma'] for r in mc],bins=25,color='#4777a8');axes[0].set(xlabel='Gamma in original Loss / Qp units',ylabel='MC draws',title='500 inferred-proxy CI draws')
    axes[1].hist([r['CV_RMSE_difference_from_M1'] for r in mc],bins=25,color='#4777a8');axes[1].axvline(0,color='#777777');axes[1].set(xlabel='Nested OOF RMSE difference from frozen M1',title='Fixed training-selected hyperparameters')
    save(fig,FIG/'mapping_monte_carlo.png')


def table(rows,columns):
    """Markdown table with finite actual results, identifiers preserved."""
    def fmt(x):
        if x is None:return '未定义'
        if isinstance(x,(float,np.floating)):return f'{x:.6f}'
        return str(x).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |']+[
        '| '+' | '.join(fmt(r.get(c,'')) for c in columns)+' |' for r in rows])


def report(result):
    """Answer all 15 requested modeling questions with quantitative limitations."""
    r=result;stable=r['stability'];marginal=r['marginal'];joint=r['joint_stability']
    main=[dict(domain=short(n),**{k:next(a[k] for a in marginal if a['model']=='M1_ElasticNet' and a['step']==.01 and a['mixture_domain']==n) for k in ['mean_effect','min_effect','max_effect']},
        strict_direction=next(a['direction'] for a in stable if a['mixture_domain']==n)) for n in r['names']]
    stable_j=[a for a in joint if a['stable_deviation']]
    stable_j=sorted(stable_j,key=lambda a:abs(a['mean_deviation']),reverse=True)
    gamma=r['m3']['meta']['gamma_original_loss_per_Qp'];eta=r['m4']['meta']['eta_original_loss_per_Kp']
    def test(model,dataset,key):
        return next(a[key] for a in r['validation'] if a['model']==model and a['dataset']==dataset)
    def link(name):
        return next(a['Spearman'] for a in r['rank_links'] if a['comparison']==name)
    fold_rows=read_csv(STAGE/'model_comparison_outer_fold_scores.csv')
    fold_improved=sum(float(next(a['RMSE'] for a in fold_rows if a['model']=='M3_Qp' and a['outer_fold']==str(f)))<
                      float(next(a['RMSE'] for a in fold_rows if a['model']=='M1_ElasticNet' and a['outer_fold']==str(f))) for f in range(1,6))
    lines=['# 问题一第3小问最终报告','',
        f"最终主模型：**{r['selected_model']}**。第一小问 Q、第二小问 K 和上一轮 ILR/权重/模型文件均保持锁定。",'',
        '## 1. 输入、训练边界与冻结核验','',
        'A4/A5 为唯一训练数据；A6–A11 只作原值检验；A12–A15 的 10B/70B Loss 为估算值，且配方是训练集子集。没有用检验 Loss 调参数、截距或尺度。',
        f"实际配方数：{', '.join(k+'='+str(len(v['P'])) for k,v in r['datasets'].items())}。受保护文件 {r['protected_n']} 个，SHA-256 全部一致。",'',
        '已有真实同配方 1M/60M 的 256 条 L_mean：Spearman=0.9657980945296404。这是排序跨尺度稳定性的核心观测证据，不能把不同配方的 1B 表按 index 强行对应。','',
        '## 2. 为什么配比不能直接 OLS；ILR、零配比与主模型','',
        '17 配比和为 1，原始配比加截距的 OLS 存在精确共线性；删去一域虽可计算，但解释依赖参考域且忽略相对比例几何。闭合后采用 16 维正交 Helmert ILR，ILR(p)=H log(p)。',
        '零项按训练折最小正闭合配比的一半进行乘法替换，非零项同比收缩，保持正比值。主模型沿用上一轮固定参数：ElasticNet alpha=0.03、l1_ratio=0.1；Ridge alpha=10 作数值对照。新模型中的 ILR 零参数、X/y 标准化均在每个 CV 训练折内拟合。',
        '这是数值正则化，不代表结构零对应真实正质量。ILR 中心由冻结零替换后的几何中心求得，分量中位数基准由原闭合配比求得。',
        'ElasticNet 已按人工确认锁定为主模型，提供收缩与有限稀疏性；Ridge 上轮 CV RMSE 略低，故不声称 ElasticNet 数值最优。上轮二阶模型没有获得嵌套 CV 收益，本轮没有恢复二阶主模型。','',
        '## 3. 17 域边际关联与稳定性','',
        '基准为 A4 ILR 中心、闭合分量中位数、冻结 ElasticNet 预测 P10/P50/P90 的真实配方，共五个。基准不使用检验 Loss 选择。对目标域增加绝对 0.005/0.01/0.02（0.5/1/2 个百分点），其余域按原比例缩减。',
        'Delta L_j=f(p perturb j)-f(p)。以下为 ElasticNet 增加 1 个百分点的五基准均值与范围。strict_direction 只有两个模型、三个步长、五个基准共 30 个效应全部同向才成立；uncertain 不给总体有利/不利结论。数值零容差 1e-10 仅防浮点误判，不是显著性阈值。','',table(main,['domain','mean_effect','min_effect','max_effect','strict_direction']),'',
        '这些是相对于按比例减小其余领域的模型关联，不是因果作用或孤立单域贡献。均值是五个代表基准的等权摘要，不是全体 A4 配方的平均处理效应。零激活效应包含从替换质量到实际正配比的变化，不能当作在 0 处的导数。','',
        '## 4. 双域联合扰动与组合偏离','',
        '同时给 j,k 各增加相同步长，其余领域按比例缩减。I_jk=Delta L_jk-Delta L_j-Delta L_k，仅称组合偏离指数。即使一阶 ILR 模型也因闭合和对数变换产生非加性，不能据此证明真实交互或因果互补。',
        f"136 对中严格同向组合偏离 {len(stable_j)} 对：负向 {sum(a['direction']=='lower_loss' for a in stable_j)}，正向 {sum(a['direction']=='higher_loss' for a in stable_j)}。完整结果在 domain_joint_effects.csv、stage2/joint_effect_stability.csv。以下仅列绝对平均偏离最大的同向对：",'',
        table([dict(domain_1=short(a['domain_1']),domain_2=short(a['domain_2']),**{k:a[k] for k in ['direction','mean_deviation','min_deviation','max_deviation']}) for a in stable_j[:12]],['domain_1','domain_2','direction','mean_deviation','min_deviation','max_deviation']),'',
        '负向同向对可描述为扰动情景下的互补趋势，正向为竞争趋势。小偏离未做抽样显著性检验，不把数值同号等同于稳定的真实交互证据。','',
        '## 5. 13 个验证域的异质性','',
        '对 A5 的每个验证 Loss 分别在 A4/A5 内训练一阶 ElasticNet/Ridge；五折选择超参数，得到 26 个辅助诊断模型，没有用检验表选择。选择 CV 的最小 RMSE 不是无偏嵌套泛化评估。',
        f"17×13 共 221 个训练域/验证域单元中，两模型均值方向不一致 {r['single_disagree']} 个；ElasticNet 在不同验证域出现正负混合方向的训练领域 {r['single_mixed_domains']} 个。因此宏观 L_mean 会掩盖部分领域差异。矩阵和两模型热图已输出。",'',
        '## 6. 17 域质量映射与 Qp','',
        'Qp=sum(p_j Q_j)，使用原配比闭合值，不使用零替换后的虚拟质量。Q_j 读取第一小问已锁定的 A1 各领域 10% 双侧截尾均值及 bootstrap 95% CI。Qp 是配比 p 的确定函数，是外部质量评价构造的质量加权配比特征，不是独立实验变量。',
        '**A16 只有 3 个 direct、3 个 near_direct 的确定关联，11 个 inferred 的 quality_domain 实际为 (none)。Scheme A 对这 11 个添加有明确理由的研究者代理假设，原 A16 字段、类型、note 都保留。绝不能把代理 Q_j 写成该语料实测质量。**','',
        table(r['mapping'],['mixture_domain_short','A16_quality_domain','quality_domain','mapping_type','Q_domain','Q_CI_low','Q_CI_high']),'',
        '## 7. M3 的嵌套 CV 与直接检验','',
        'M3=一阶 ILR ElasticNet + Qp。沿用原模型标准化与同一 ElasticNet 网格；五个外折严格复用上一轮划分，各外训练内再四折选参，最后仅全 A4/A5 五折选最终超参数。M1 使用冻结的原 OOF 预测，没有重调。',
        f"M3 最终参数：{r['m3']['params']}。gamma 原 Loss/Qp 单位={gamma:.9f}；还原公式 gamma=s_y beta_Qp/s_Qp。该系数受确定性配比特征与正则化共线性影响，不具有因果解释。",'',
        table(r['cv'],['model','RMSE','MAE','R2','Pearson','Spearman']),'',
        table(r['validation'],['model','dataset','RMSE','MAE','Pearson','Spearman']),'',
        f"最终判据：{r['policy']['reason']}。Qp {'进入最终模型' if r['selected_model']=='M3_Qp' else '保留为辅助解释特征，不进入最终主模型'}。全部预测未用 test bias 校准。",
        f"M3 相对 M1 的嵌套 CV RMSE 降幅 {(r['cv'][0]['RMSE']-r['cv'][1]['RMSE'])/r['cv'][0]['RMSE']*100:.4f}%。改善外折 {fold_improved}/5；1M test MAE 差 {test('M3_Qp','test_1M','MAE')-test('M1_ElasticNet','test_1M','MAE'):+.9f}。综合改善幅度和外折变化，本轮不足以证明质量特征具有稳定的实用增益。保守规则要求各外折方向一致，属于操作性稳定标准，不是统计显著性检验。",'',
        '## 8. 映射不确定性：CI 扰动与替代语义代理','',
        'Scheme B：seed=2026，对每个 inferred 域独立 Uniform(Q_CI_low,Q_CI_high) 抽样 500 次，direct/near_direct 固定。每次重算 Qp、用 Scheme A 已由训练 CV 选定的各外折和全训练超参数重拟合 M3，重新计算 OOF 与检验预测。',
        '这是条件于已选择超参数的 Monte Carlo，不重复选择超参数，不包含模型选择不确定性。独立抽样假设允许同代理域的不同 inferred 域有所区别，未建模来源相关性。',
        f"500 次 gamma>0 比例 {r['mc_gamma_positive_rate']:.6f}；gamma<0 比例 {r['mc_gamma_negative_rate']:.6f}；OOF RMSE 优于冻结 M1 的比例 {r['mc_cv_improve_rate']:.6f}。",'',
        table([a for a in r['mc_summary'] if a['metric'] in ['gamma','CV_RMSE_difference_from_M1','test_1M_RMSE','test_1M_Spearman','test_60M_Spearman','test_1B_Spearman']],['metric','mean','std','P025','median','P975']),'',
        'CI 仅刻画已有代理质量域截尾均值的抽样波动，不包含“语义关联选错”的偏差，不能由窄 CI 下稳定推出映射可靠。另设未经检验 Loss 选择的替代语义代理组合，在同一 A4/A5 嵌套 CV 内分析：','',
        table(r['alternative'],['scheme','gamma_original_loss_per_Qp','CV_RMSE','CV_MAE','CV_R2','CV_Spearman']),'',
        f"替代代理下 gamma 相对 Scheme A 比值 {r['alternative'][1]['gamma_original_loss_per_Qp']/gamma:.6f}；替代方案相对 M1 的 CV RMSE 差为 {r['alternative'][1]['CV_RMSE']-r['cv'][0]['RMSE']:+.9f}。因此 CI 内稳定与代理语义变化稳定必须分开，不能由前者推断后者具有相同增益幅度。",
        '替代映射列出在 stage2/alternative_domain_quality_mapping.csv。各方案不按 test 结果择优替换主映射。直接映射也只代表来源域名对应，不证明 A1 文本质量均值准确代表 A4 中全部同名训练语料。','',
        '## 9. Kp 仅作辅助检查','',
        '读取已有 A1_sample_conflict.csv 的 K，分别按原领域作两端各 10% 截尾均值，不重新计算 K。通过相同显式代理映射形成 K_j 和 Kp=sum(p_j K_j)。M4=M3+eta Kp，仍在 A4/A5 内嵌套 CV。',
        f"M4 eta 原 Loss/Kp 单位={eta:.9f}，与 M3 的 OOF RMSE 差为 {r['m4']['cv']['RMSE']-r['m3']['cv']['RMSE']:+.9f}。Kp {'未获得 CV 改善，舍弃' if r['m4']['cv']['RMSE']>=r['m3']['cv']['RMSE'] else '仅保留辅助敏感性结果；CV 获得改善，但泛化检验未稳定复现'}，不进入本轮最终主模型。K 是评价冲突，不是低质量程度，未作为惩罚项。",
        f"M4 的 1M test RMSE={test('M4_Qp_Kp','test_1M','RMSE'):.6f}，M3={test('M3_Qp','test_1M','RMSE'):.6f}；1B Spearman M4={test('M4_Qp_Kp','test_1B','Spearman'):.6f}，M3={test('M3_Qp','test_1B','Spearman'):.6f}。eta 符号由训练拟合自由决定，不能解释为质量惩罚。M4 gamma={r['m4']['meta']['gamma_original_loss_per_Qp']:.6f}，加入 Kp 后系数幅度变化，提示确定性配比描述符间的依赖与正则化分配影响系数，不能解释为独立因果贡献。", '',
        '## 10. 尺度迁移诊断，不做偏置校准','',
        '残差 e=y-yhat；centered RMSE=sqrt(mean((e-mean(e))^2)) 仅解释偏移贡献。它没有提供可在未知目标尺度应用的偏置，不能称为校准后的预测性能。', '',
        table(r['scale'],['model','dataset','mean_bias','median_bias','residual_std','RMSE','centered_RMSE_diagnostic_only','Pearson','Spearman']),'',
        '绝对误差跨尺度显著增大，去除样本均值残差后的诊断误差远小于原 RMSE，表明整体 Loss 水平偏移是主要来源；正排序相关仍显示有限的关联迁移。60M 与 1B 的配方集合不同，不能把其相关大小直接解读为规模单调变化。','',
        '## 11. 10B/70B 外推一致性与证据边界','',
        '所有超参数选择在查看估算性能之前完成并冻结；按闭合配比向量匹配训练子集，不按跨表 index 假定相同配方。10B、70B 各 63 个配方均属于 A4，因此只称估算训练配方子集的外推一致性检验。', '',
        table(r['estimated'],['dataset','model','RMSE','MAE','Pearson','Spearman']),'',
        table(r['rank_links'],['comparison','n','Spearman','diagnostic_ratio_estimated_over_1M','diagnostic_max_abs_scaling_deviation']),'',
        f"独立核验后的实际结论：主模型与 10B/70B 估算宏观 Loss 的 Spearman 分别为 {link('M1_ElasticNet_prediction_vs_estimated_10B'):.6f}/{link('M1_ElasticNet_prediction_vs_estimated_70B'):.6f}，排序发生反向关联，**不支持主模型在这两张估算表上的排序迁移**。M3 同样没有修复反向关系。10B 与 70B 彼此 Spearman={link('estimated_10B_vs_estimated_70B_same_recipes'):.6f}，仅说明两张估算表之间排序相近，不能代替“1M 模型预测与大规模估算表一致”。",
        f"同配方 1M 实测宏观 Loss 与 10B/70B 估算宏观 Loss 的 Spearman 分别为 {link('observed_train_1M_vs_estimated_10B'):.6f}/{link('observed_train_1M_vs_estimated_70B'):.6f}，说明反向关系并非只由模型拟合误差造成。配方已按向量确认一致；但这些是估算值，不能据此断言真实大模型发生相同反转。",
        '逐验证域关系与宏观均值关系不同，估算值也不是原 1M 宏观 Loss 的简单正比例缩放。生成关系诊断在 stage2/estimated_*_generation_relationship.csv；其中比例系数仅是对已有表的诊断数，没有用于拟合、校准或构造新 Loss。现有附件不足以识别估算生成机制，不能把诊断比例写成已知 Scaling Law。','',
        '## 12. 最终选择、局限性与复用文件','',
        f"主模型采用 {r['selected_model']}；Ridge 保留数值对照。原 Q 与 K 分别表示综合质量与评价冲突。参数与选择理由记录在 stage2/final_model_policy.json，冻结阶段清单记录在 stage2/freeze_before_extrapolation.json。", 
        '局限性：512 个小规模训练配方、较多零配比、对零激活与基准敏感；17 域与 13 验证域不齐；没有因果实验识别；新增代理映射缺直接质量观测；Qp/Kp 均由 p 确定且相互相关；嵌套 CV 差异未证明显著；估算表不是外部观测，且配方属于训练子集。',
        '复用：domain_marginal_effects.csv、domain_effect_stability.csv、domain_joint_effects.csv、train_domain_vs_validation_loss_effects.csv、domain_quality_mapping.csv、recipe_quality_conflict_descriptors.csv、stage2/model_comparison_nested_CV.csv、stage2/model_comparison_validation.csv、estimated_scale_consistency.csv。',
        '**本轮完成后停止，未开始问题二，未重新拟合前两小问结果。**','']
    (ROOT/'outputs/report/question1_part3_final.md').write_text('\n\n'.join(lines),encoding='utf-8')


def refresh_report_only(refresh_figures=False):
    """Refresh prose from completed saved results; no fitting or parameter changes."""
    datasets,names,loss_names=load_datasets(ROOT)
    summary=json.loads((STAGE/'execution_summary.json').read_text(encoding='utf-8'))
    policy=json.loads((STAGE/'final_model_policy.json').read_text(encoding='utf-8'))
    def numeric_rows(path):
        records=read_csv(path)
        for row in records:
            for key,value in row.items():
                if value=='True':row[key]=True
                elif value=='False':row[key]=False
                elif value=='':row[key]=None
                else:
                    try:row[key]=float(value)
                    except ValueError:pass
        return records
    cv=summary['cv'];m3=dict(meta=json.loads((STAGE/'models/M3_Qp.json').read_text(encoding='utf-8')),cv=cv[1])
    m3['params']=m3['meta']['parameters'];m4=dict(meta=json.loads((STAGE/'models/M4_Qp_Kp.json').read_text(encoding='utf-8')),cv=cv[2])
    mc=numeric_rows(OUT/'quality_mapping_monte_carlo_results.csv')
    single=numeric_rows(OUT/'train_domain_vs_validation_loss_effects.csv');en=[r for r in single if r['model']=='M1_ElasticNet']
    result=dict(names=names,datasets=datasets,marginal=numeric_rows(OUT/'domain_marginal_effects.csv'),
        stability=numeric_rows(OUT/'domain_effect_stability.csv'),joint_stability=numeric_rows(STAGE/'joint_effect_stability.csv'),
        mapping=numeric_rows(OUT/'domain_quality_mapping.csv'),m3=m3,m4=m4,cv=cv,validation=summary['validation'],
        scale=numeric_rows(OUT/'scale_transfer_diagnostics.csv'),alternative=numeric_rows(OUT/'quality_proxy_mapping_sensitivity.csv'),
        mc_summary=numeric_rows(OUT/'quality_mapping_monte_carlo_summary.csv'),
        mc_gamma_positive_rate=float(np.mean([r['gamma']>0 for r in mc])),mc_gamma_negative_rate=float(np.mean([r['gamma']<0 for r in mc])),
        mc_cv_improve_rate=float(np.mean([r['CV_RMSE_difference_from_M1']<0 for r in mc])),
        single_disagree=sum(not r['models_mean_direction_agree'] for r in en),
        single_mixed_domains=sum(direction([r['mean_effect'] for r in en if r['training_domain']==n])=='uncertain' for n in names),
        estimated=summary['estimated'],rank_links=summary['rank_links'],protected_n=summary['protected_files_unchanged'],
        selected_model=summary['selected_model'],policy=policy)
    report(result)
    if refresh_figures:
        figures(names,loss_names,result['marginal'],result['stability'],single,cv,result['validation'],mc)
    verify_protected(json.loads((STAGE/'protected_before.json').read_text(encoding='utf-8')))
    print('Report refreshed from existing results; no fitting.')


def main():
    """Run the authorized final round sequentially, fail loudly on inconsistent inputs."""
    if (STAGE/'execution_summary.json').exists():
        raise RuntimeError('This round is complete and frozen. Use --report-only or --figures-only; do not refit completed models.')
    for p in [STAGE,STAGE/'models',STAGE/'single_loss_models',FIG]:p.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(STAGE/'analysis.log',encoding='utf-8')],force=True)
    settings=yaml.safe_load((ROOT/'configs/mixture_modeling.yaml').read_text(encoding='utf-8'))
    a=yaml.safe_load((ROOT/'configs/mixture_analysis.yaml').read_text(encoding='utf-8'))
    protected=capture_protected();logging.info('Protected %d original/locked files',len(protected))
    datasets,names,loss_names=load_datasets(ROOT);train=datasets['train_1M'];y=train['L_mean'];P=train['P']
    models={label:joblib.load(OUT/'models'/f'L_mean_{label}.joblib') for label in ['M1_ElasticNet','M1_Ridge']}
    # Merely applying already frozen predictors; no re-fitting/re-scaling.
    scale=scale_diagnostics(datasets,models)
    bases=baselines(P,train['indices'],models['M1_ElasticNet'])
    write_rows(STAGE/'representative_baselines.csv',[dict(baseline=label,recipe_index=index,
        frozen_EN_prediction=float(models['M1_ElasticNet'].predict(p[None,:])[0]),**dict(zip(names,p))) for label,index,p in bases])
    singles,marginal,stability,joint_cases,joints,joint_stability=analyze(models,bases,names,a['perturbation_steps'],a['numerical_sign_tolerance'])
    write_rows(STAGE/'domain_marginal_effect_cases.csv',singles);write_rows(OUT/'domain_marginal_effects.csv',marginal)
    write_rows(OUT/'domain_effect_stability.csv',stability);write_rows(STAGE/'domain_joint_effect_cases.csv',joint_cases)
    write_rows(OUT/'domain_joint_effects.csv',joints);write_rows(STAGE/'joint_effect_stability.csv',joint_stability)
    logging.info('Frozen effects: %d stable domains, %d stable joint deviations',sum(r['stable_direction'] for r in stability),sum(r['stable_deviation'] for r in joint_stability))
    single=single_loss_models(train,bases,names,loss_names,settings)
    mapping_rows,quality=mapping(names,a);q=np.array([r['Q_domain'] for r in mapping_rows])
    outer=group_folds(P,settings['outer_folds'],settings['random_state'])
    assignments={r['index']:int(r['outer_holdout_fold']) for r in read_csv(OUT/'outer_fold_assignments.csv')}
    for fold,(_,va) in enumerate(outer,1):
        if any(assignments[train['indices'][i]]!=fold for i in va):raise ValueError('Outer folds differ from frozen M1')
    base_oof=frozen_oof(train);base_cv=dict(model='M1_ElasticNet',**scores(y,base_oof))
    logging.info('Starting M3 nested CV on A4/A5 only')
    m3=nested_augmented('M3_Qp',q,None,train,names,settings,outer)
    # Conditional inferred-quality CI sensitivity precedes any Kp model.
    observed={label:d for label,d in datasets.items() if d['provenance']=='observed'}
    mc,mc_summary=monte_carlo(m3,mapping_rows,train,observed,names,settings,a,base_oof)
    alternative_mapping=[]
    q_alt=q.copy()
    for j,r in enumerate(mapping_rows):
        domain=short(names[j]);qd=a['alternative_inferred_mapping'][domain] if r['researcher_assumption'] else r['quality_domain']
        q_alt[j]=float(quality[qd]['Q_domain']);alternative_mapping.append(dict(mixture_domain=names[j],mapping_type=r['mapping_type'],
            quality_domain=qd,Q_domain=q_alt[j],researcher_assumption=r['researcher_assumption']))
    write_rows(STAGE/'alternative_domain_quality_mapping.csv',alternative_mapping)
    alt=nested_augmented('M3_alternative_mapping',q_alt,None,train,names,settings,outer)
    alternative=[dict(scheme='A_assumed_proxy',gamma_original_loss_per_Qp=m3['meta']['gamma_original_loss_per_Qp'],
        **{'CV_'+k:m3['cv'][k] for k in ['RMSE','MAE','R2','Spearman']}),
        dict(scheme='alternative_assumed_proxy',gamma_original_loss_per_Qp=alt['meta']['gamma_original_loss_per_Qp'],
        **{'CV_'+k:alt['cv'][k] for k in ['RMSE','MAE','R2','Spearman']})]
    write_rows(OUT/'quality_proxy_mapping_sensitivity.csv',alternative)
    # Now and only now aggregate fixed K and test M4.
    kd=original_k_domains();k=np.array([kd[r['quality_domain']] for r in mapping_rows])
    descriptors(datasets,q,k)
    write_rows(STAGE/'domain_quality_conflict_mapping.csv',[dict(**r,K_domain=float(k[j])) for j,r in enumerate(mapping_rows)])
    m4=nested_augmented('M4_Qp_Kp',q,k,train,names,settings,outer)
    cv=[base_cv,m3['cv'],m4['cv']];write_rows(STAGE/'model_comparison_nested_CV.csv',cv)
    folds=[]
    for fold,(_,va) in enumerate(outer,1):folds.append(dict(model='M1_ElasticNet',outer_fold=fold,**scores(y[va],base_oof[va])))
    folds+=m3['fold_rows']+m4['fold_rows'];write_rows(STAGE/'model_comparison_outer_fold_scores.csv',folds)
    validation=[];predictions=[]
    test_models={**models,'M3_Qp':m3['model'],'M4_Qp_Kp':m4['model']}
    for name,model in test_models.items():
        for label in ['test_1M','test_60M','test_1B']:
            d=datasets[label];pred=model.predict(d['P']);validation.append(dict(model=name,dataset=label,**scores(d['L_mean'],pred)))
            predictions += [dict(model=name,dataset=label,index=index,true_loss=float(d['L_mean'][i]),predicted_loss=float(pred[i])) for i,index in enumerate(d['indices'])]
    write_rows(STAGE/'model_comparison_validation.csv',validation);write_rows(STAGE/'model_comparison_test_predictions.csv',predictions)
    # No test-based hyperparameter change: only the user's predeclared retain/reject rule.
    b1=next(r for r in validation if r['model']=='M1_ElasticNet' and r['dataset']=='test_1M')
    q1=next(r for r in validation if r['model']=='M3_Qp' and r['dataset']=='test_1M')
    fold_improvements=[m3['fold_rows'][i]['RMSE']<folds[i]['RMSE'] for i in range(5)]
    rank_improvements=[next(r['Spearman'] for r in validation if r['model']=='M3_Qp' and r['dataset']==d)>
                       next(r['Spearman'] for r in validation if r['model']=='M1_ElasticNet' and r['dataset']==d) for d in ['test_1M','test_60M','test_1B']]
    retain=bool(m3['cv']['RMSE']<base_cv['RMSE'] and q1['RMSE']<b1['RMSE'] and all(rank_improvements) and all(fold_improvements))
    selected='M3_Qp' if retain else 'M1_ElasticNet'
    policy=dict(selected_model=selected,Qp_retained=retain,Kp_retained=False,
        reason=f"M3 nested RMSE difference={m3['cv']['RMSE']-base_cv['RMSE']:+.9f}; 1M test RMSE difference={q1['RMSE']-b1['RMSE']:+.9f}; "
               f"outer RMSE improved {sum(fold_improvements)}/5; three observed scales ranking improved {sum(rank_improvements)}/3; "
               'conservative retention requires consistent outer-fold and observed-scale improvement',
        model_selection_uses_no_estimated_loss=True,hyperparameters_selected_on='A4/A5_only',
        no_test_calibration=True,Qp_mapping_assumptions_explicit=True)
    dump(STAGE/'final_model_policy.json',policy)
    protected_n=verify_protected(protected)
    # Declare selected predictor and all newly trained params frozen BEFORE estimated diagnostics.
    freeze={str(p.relative_to(ROOT)):sha256_file(p) for p in sorted((STAGE/'models').glob('*')) if p.is_file()}
    freeze.update({str(p.relative_to(ROOT)):sha256_file(p) for p in sorted((OUT/'models').glob('L_mean_M1*'))})
    dump(STAGE/'freeze_before_extrapolation.json',dict(selected_model=selected,model_files=freeze,estimated_loss_used_for_selection=False))
    estimated,rank_links=extrapolation(datasets,{'M1_ElasticNet':models['M1_ElasticNet'],'M1_Ridge':models['M1_Ridge'],'M3_Qp':m3['model']})
    # Count heterogeneity based on the actual sensitivity matrices.
    en=[r for r in single if r['model']=='M1_ElasticNet']
    mixed=sum(direction([r['mean_effect'] for r in en if r['training_domain']==n])=='uncertain' for n in names)
    result=dict(names=names,datasets=datasets,marginal=marginal,stability=stability,joint_stability=joint_stability,
        mapping=mapping_rows,m3=m3,m4=m4,cv=cv,validation=validation,scale=scale,alternative=alternative,
        mc_summary=mc_summary,mc_gamma_positive_rate=float(np.mean([r['gamma']>0 for r in mc])),
        mc_gamma_negative_rate=float(np.mean([r['gamma']<0 for r in mc])),mc_cv_improve_rate=float(np.mean([r['CV_RMSE_difference_from_M1']<0 for r in mc])),
        single_disagree=sum(not r['models_mean_direction_agree'] for r in en),single_mixed_domains=mixed,
        estimated=estimated,rank_links=rank_links,protected_n=protected_n,selected_model=selected,policy=policy)
    figures(names,loss_names,marginal,stability,single,cv,validation,mc)
    report(result)
    protected_n=verify_protected(protected)
    for relative,digest in freeze.items():
        if sha256_file(ROOT/relative)!=digest:raise RuntimeError('Model changed after extrapolation freeze')
    dump(STAGE/'execution_summary.json',dict(selected_model=selected,protected_files_unchanged=protected_n,
        stable_domains=[short(r['mixture_domain']) for r in stability if r['stable_direction']],
        stable_joint_pairs=sum(r['stable_deviation'] for r in joint_stability),cv=cv,validation=validation,
        mapping_gamma=gamma if (gamma:=m3['meta']['gamma_original_loss_per_Qp']) else 0,
        MC_repetitions=len(mc),estimated=estimated,rank_links=rank_links))
    logging.info('COMPLETE: question1 part3 final; selected=%s; protected=%d; MC=%d; stopped before question2',selected,protected_n,len(mc))


if __name__=='__main__':
    try:
        if sys.argv[1:] in [['--report-only'],['--figures-only']]:
            refresh_report_only(sys.argv[1:]==['--figures-only'])
            sys.exit(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            main()
        write_rows(STAGE/'convergence_warnings.csv',[dict(category=w.category.__name__,message=str(w.message)) for w in caught],['category','message'])
    except Exception:
        logging.exception('Analysis failed; prior frozen files must remain unchanged')
        raise
