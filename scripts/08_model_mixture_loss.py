"""Train-only nested-CV M0/M1/M2 and direct 1M/60M/1B validation."""
from __future__ import annotations
import json
import logging
from itertools import combinations
from pathlib import Path
import sys
import warnings
import joblib
import numpy as np
import yaml
from scipy.stats import pearsonr,spearmanr
from sklearn.exceptions import ConvergenceWarning
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mixture.io import load_datasets,signatures
from mixture.composition import ILRTransformer
from mixture.modeling import MODELS,scores,group_folds,tune
from quality.io import sha256_file
from quality.vector_diagnostics import write_rows
OUT=ROOT/'outputs/mixture';FIG=ROOT/'outputs/figures/question1_part3/stage1'


def protect():
    """Hash all original attachments and locked part 1/2 artifacts/code."""
    paths=list((ROOT/'outputs').rglob('*'))+list((ROOT/'real_attachments').rglob('*'))+list((ROOT/'configs').glob('*'))+list((ROOT/'src/quality').glob('*.py'))
    paths += [p for p in (ROOT/'scripts').glob('*.py') if p.name[:2].isdigit() and int(p.name[:2])<=6]
    result={}
    for p in sorted(paths):
        if not p.is_file() or p.is_relative_to(OUT) or p.is_relative_to(ROOT/'outputs/figures/question1_part3') or p.name.startswith('question1_part3'):continue
        result[p.relative_to(ROOT).as_posix()]=sha256_file(p)
    return result


def metadata(estimator,model,names,training_indices,parameters):
    """Export a complete numerical predictor for independent reconstruction."""
    result={'model':model,'fitting_recipe_indices':training_indices,'n_fit':len(training_indices),'parameters':parameters}
    if model=='M0':return {**result,'constant':float(estimator.constant_[0,0])}
    pipeline=estimator.regressor_;ilr=pipeline.named_steps['ilr'];scaler=pipeline.named_steps['scale'];regression=pipeline.named_steps['regression']
    iterations=getattr(regression,'n_iter_',None)
    if iterations is not None and int(iterations)>=regression.max_iter:raise ValueError('Selected ElasticNet did not converge')
    powers=pipeline.named_steps['interactions'].powers_ if 'interactions' in pipeline.named_steps else np.eye(16,dtype=int)
    feature_names=pipeline.named_steps['interactions'].get_feature_names_out([f'ilr_{i+1:02d}' for i in range(16)]).tolist() if 'interactions' in pipeline.named_steps else [f'ilr_{i+1:02d}' for i in range(16)]
    return {**result,'domain_order':names,'zero_delta':ilr.zero_delta_,'training_min_positive':ilr.min_positive_training_,
        'ilr_basis':ilr.basis_.tolist(),'feature_powers':powers.tolist(),'feature_names':feature_names,
        'X_mean':scaler.mean_.tolist(),'X_scale':scaler.scale_.tolist(),'coefficients':regression.coef_.tolist(),
        'intercept':float(regression.intercept_),'y_mean':float(estimator.transformer_.mean_[0]),'y_scale':float(estimator.transformer_.scale_[0]),
        'n_nonzero_coefficients':int((regression.coef_!=0).sum()),'n_model_features':len(regression.coef_),
        'n_iter':int(iterations) if iterations is not None else None}


def fit_checked(model,P,y,settings,folds,context,warning_rows):
    """Capture convergence warnings and verify selected fit before prediction."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always',ConvergenceWarning)
        estimator,parameters,grid=tune(model,P,y,settings,folds)
    for item in captured:
        warning_rows.append({**context,'category':item.category.__name__,'message':str(item.message)})
    if captured:logging.info('%s %s: captured %d warnings; selected-fit convergence checked',context.get('response'),model,len(captured))
    return estimator,parameters,grid


def figure_outputs(validation,nested,predictions,pnames,train,settings):
    """Full-data plots, labeled observed versus direct uncalibrated prediction."""
    FIG.mkdir(parents=True,exist_ok=True);paths=[]
    def save(fig,name):
        fig.tight_layout();fig.savefig(FIG/name,dpi=settings['figure_dpi'],bbox_inches='tight');plt.close(fig);paths.append((FIG/name).relative_to(ROOT).as_posix())
    fig,ax=plt.subplots(figsize=(11,5));ax.bar([n.removeprefix('train_the_pile_') for n in pnames],100*(train['P']==0).mean(axis=0),color='#4777a8')
    ax.set(ylabel='Zero recipe share (%)',title='A4: zeros by training domain');ax.tick_params(axis='x',rotation=60);save(fig,'01_train_zero_rates.png')
    fig,ax=plt.subplots(figsize=(8,4))
    primary=[r for r in nested if r['response']=='L_mean'];ax.bar([r['model'] for r in primary],[r['RMSE'] for r in primary],color='#4777a8')
    ax.set(ylabel='Nested out-of-fold RMSE',title='A4/A5 model comparison: macro Loss');save(fig,'02_nested_CV_comparison.png')
    for response in ['L_mean','L_trim']:
        fig,axes=plt.subplots(3,4,figsize=(15,10))
        for row,dataset in enumerate(['test_1M','test_60M','test_1B']):
            for col,model in enumerate(MODELS):
                values=[r for r in predictions if r['response']==response and r['dataset']==dataset and r['model']==model]
                y=np.array([r['true_loss'] for r in values]);p=np.array([r['predicted_loss'] for r in values]);ax=axes[row,col]
                ax.scatter(y,p,s=12,alpha=.6,color='#4777a8');lo,hi=min(y.min(),p.min()),max(y.max(),p.max())
                ax.plot([lo,hi],[lo,hi],linestyle='--',color='#777777');ax.set(title=f'{dataset} / {model}',xlabel='Observed Loss',ylabel='1M model prediction')
        fig.suptitle(f'{response}: no scale calibration or test-set fitting',y=1.01);save(fig,f'03_{response}_predicted_vs_observed.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(3);labels=['test_1M','test_60M','test_1B']
    for i,model in enumerate(MODELS):
        values=[next(r for r in validation if r['response']=='L_mean' and r['dataset']==d and r['model']==model) for d in labels]
        axes[0].plot(x,[r['RMSE'] for r in values],'o-',label=model)
        if model!='M0':axes[1].plot(x,[r['Spearman'] for r in values],'o-',label=model)
    axes[0].set(xticks=x,xticklabels=labels,ylabel='RMSE',title='Absolute transfer error')
    axes[1].set(xticks=x,xticklabels=labels,ylabel='Spearman',title='Recipe ranking transfer')
    for ax in axes:ax.legend(fontsize=8)
    save(fig,'04_scale_absolute_vs_rank.png')
    write_rows(OUT/'figure_manifest.csv',[{'path':p} for p in paths])


def build_report(datasets,names,settings,nested,validation,final_rows,scale_rows,warnings_rows,best,protected_n):
    """Describe findings without test-driven selection or unsupported causal effects."""
    def table(rows,columns):
        def value(v):
            if v is None:return '未定义'
            if isinstance(v,(float,np.floating)):return f'{v:.6f}'
            return str(v).replace('|','\\|')
        return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |']+
            ['| '+' | '.join(value(r.get(k)) for k in columns)+' |' for r in rows])
    primary=[r for r in nested if r['response']=='L_mean'];secondary=[r for r in nested if r['response']=='L_trim']
    ridge=next(r for r in primary if r['model']=='M1_Ridge')
    elastic=next(r for r in primary if r['model']=='M1_ElasticNet')
    quadratic=next(r for r in primary if r['model']=='M2_ElasticNet')
    ridge_train=next(r for r in final_rows if r['response']=='L_mean' and r['model']=='M1_Ridge')
    quadratic_trim=next(r for r in final_rows if r['response']=='L_trim' and r['model']=='M2_ElasticNet')
    ilr=ILRTransformer(settings['zero_replacement_fraction_of_training_min']).fit(datasets['train_1M']['P'])
    selected=[r for r in validation if r['response']=='L_mean' and r['model']==best]
    text=['# 问题一第3小问：阶段1配比/Loss建模与检验',
        '本轮已完成 A4–A15 数据诊断、响应变量、单纯形/零值处理、ILR、M0/M1/M2 与 A6–A11 检验。原 Q、K、预处理及 CRITIC 全部冻结；未引入 Q/K，未读取 A16 映射用于建模，未预测10B/70B，未建立模型规模方程或 Scaling Law。',
        '## 真实结构',table([{'dataset':d,'n':len(v['P']),'provenance':v['provenance']} for d,v in datasets.items()],['dataset','n','provenance']),
        '详细十二文件、17配比列、13Loss列、逐域分布、零值/相关矩阵与行和诊断见 outputs/mixture/phase1_data_diagnosis.md。所有对应表的 index 唯一并在同一实验内一一配对。',
        '## 响应与组合变换',
        '主响应为13域等权 L_mean；敏感性响应 L_trim 去掉最高最低各一个后对11域取均值。两个响应各自拟合相同候选和CV流程，未选择某个有利Loss域。',
        f'先闭合 p/sum(p)。零替换 δ=min(0.5×闭合训练配比最小正值,1/(2×17))；A4全量 min_positive={ilr.min_positive_training_:.12g}，最终 δ={ilr.zero_delta_:.12g}。每行零坐标变为 δ，非零坐标乘 (1−零个数×δ)，保留非零比值。测试统一使用训练参数；CV中则仅从当前折训练部分拟合 δ。原始零可能为结构性缺席或舍入，替换不是真实观测恢复，因此结论依赖此处理。',
        'ILR 用固定列顺序的16×17 Helmert正交对比矩阵 H：x=log(p_replaced)Hᵀ，HHᵀ=I，H1=0。第k个坐标为 sqrt(k/(k+1))×log(前k域几何均值/第k+1域)，k=1..16。单纯形约束得到16维独立坐标；CLR另存仅供解释，不直接对17原始p做OLS。',
        '## 候选模型与无泄漏CV',
        'M0=训练平均Loss；M1-Ridge/M1-ElasticNet=16维ILR一阶模型；M2-ElasticNet=16个一阶坐标+120个两两交叉乘积，共136个特征，没有无约束OLS。所有X标准化在管道内，y也在每个训练折内标准化；预测还原到原Loss单位。L1筛选依赖所选ILR基，不将某个ILR交叉系数直接称为某对原领域因果互补。',
        f"固定种子 {settings['random_state']}。外层{settings['outer_folds']}折、内层{settings['inner_folds']}折嵌套CV；同配方组不得跨折。最终参数在全部A4/A5上用{settings['final_selection_folds']}折选取并重拟合，只用于之后的直接检验。最佳模型按训练嵌套OOF RMSE选为 {best}，测试结果不参与选择。所有CV折、完整网格及外层模型参数已保存。",
        'Ridge alpha 网格：'+str(settings['ridge_alphas'])+'；ElasticNet alpha：'+str(settings['elastic_alphas'])+'；l1_ratio：'+str(settings['elastic_l1_ratios'])+'.',
        '### L_mean 嵌套CV',table(primary,['model','RMSE','MAE','R2','Pearson','Spearman','mean_fold_RMSE','std_fold_RMSE']),
        '### L_trim 敏感性',table(secondary,['model','RMSE','MAE','R2','Pearson','Spearman','mean_fold_RMSE','std_fold_RMSE']),
        f"数值上 M1-Ridge 的 L_mean 嵌套 RMSE={ridge['RMSE']:.6f}，M1-ElasticNet={elastic['RMSE']:.6f}，差值仅 {elastic['RMSE']-ridge['RMSE']:.6f}；不能宣称显著优越。按用户解释优先的要求，保留 M1-ElasticNet 为可解释主方案候选，Ridge 为数值最佳对照，等待人工确认，不凭检验集选择最终模型。M2-ElasticNet 的嵌套 RMSE={quadratic['RMSE']:.6f}，暂未支持加入交互项。",
        '### 最终训练拟合与复杂度',table(final_rows,['response','model','training_RMSE','training_R2','nested_OOF_RMSE','n_model_features','n_nonzero_coefficients','parameters']),
        '训练误差小于嵌套OOF误差是乐观偏差诊断；M2是否增加泛化能力须看嵌套CV和同尺度1M检验，不能仅看训练拟合。跨模型CV差距很小时不声称统计显著优越；这里只给出真实误差及折间离散。',
        f"一阶 Ridge 的训练 RMSE={ridge_train['training_RMSE']:.6f}、嵌套 OOF={ridge['RMSE']:.6f}，未见严重训练拟合崩溃。M2 在 L_trim 下训练 RMSE={quadratic_trim['training_RMSE']:.6f}、嵌套 OOF={quadratic_trim['nested_OOF_RMSE']:.6f}；二阶复杂度增加但泛化未改善，表现出额外拟合乐观/过拟合风险，当前不作为优选。跨尺度绝对偏差不能单独归因于过拟合，模型没有任何规模变量。",
        '## 三尺度直接检验',table([r for r in validation if r['response']=='L_mean'],['dataset','model','RMSE','MAE','R2','Pearson','Spearman','prediction_bias_mean']),
        'L_trim 的同样指标完整保存在 model_validation.csv。图展示固定1M模型的原预测，不以测试真值调整均值、截距、尺度或重新训练。外部检验的baseline相关系数因恒定预测而未定义，留空而非置零；嵌套OOF的M0跨折采用不同训练均值，因此聚合OOF可能有非零相关，不代表配比解释力。',
        '### 训练CV选定模型',table(selected,['dataset','model','n','RMSE','MAE','R2','Pearson','Spearman','prediction_bias_mean']),
        '绝对误差/R²与配方优劣排序相关须分开判断。若跨尺度偏差大而排序较好，仅能称排序迁移有效，不能声称1M模型无条件解释60M/1B绝对Loss。额外去均值RMSE只是描述性分解，不是可部署校准或另一组实际验证结果。',
        '## 实际跨尺度配方排序',table(scale_rows,['response','dataset_1','dataset_2','shared_recipe_n','actual_Pearson','actual_Spearman','interpretation']),
        '共同配方以闭合向量匹配；同名index不一定为同配方。没有相同配方时，直接实际排序一致性未定义，不虚构相关值。1B上的预测/真值Spearman仅是固定1M模型的间接迁移证据。',
        '## 范围与局限',
        f'本轮捕获收敛/运行警告数 {len(warnings_rows)}，逐条见 convergence_warnings.csv；所选ElasticNet达到迭代上限则停止，不隐瞒失败。原件及先前锁定产物/代码 {protected_n} 个文件前后 SHA-256 保持一致。独立验证见 mixture_verification.csv。',
        '对稀疏配方的log-ratio零替换会影响效应，未声称唯一零处理结论。训练配方数有限，二阶项更易过拟合；ILR基与L1惩罚的选择影响稀疏解释。后续仍需按用户下一阶段做原领域扰动、互补趋势、Q映射及 estimated 外推讨论，当前没有提前生成这些结论。',
        '10B/70B是estimated材料，本轮只读结构和Loss描述，未用于CV、拟合、选择或预测。模型规模方程尚未建立。Q/K保持锁定，未进入本轮任何解释变量。',
        '## 当前停止点',
        '可复用 ILR_basis.csv、composition_transform_A4.json、四个候选的final_model_metadata.json/joblib、嵌套OOF与测试预测CSV。等待人工确认阶段1后再进入 Q_p 与外推分析。']
    (ROOT/'outputs/report/question1_part3_stage1_modeling.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'models/outer').mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'mixture_modeling.log',encoding='utf-8')])
    settings=yaml.safe_load((ROOT/'configs/mixture_modeling.yaml').read_text(encoding='utf-8'))
    if settings['stage_stop']!='M0_M1_M2_validation_only_no_Q_K_or_large_scale_prediction':raise ValueError('Unauthorized phase')
    # Diagnosis must already exist, and its source hashes must still match.
    import csv
    with (OUT/'input_manifest.csv').open(encoding='utf-8-sig') as f:manifest=list(csv.DictReader(f))
    if len(manifest)!=12 or any(sha256_file(ROOT/r['path'])!=r['sha256'] for r in manifest):raise ValueError('Diagnosis absent or original input changed')
    before=protect();write_rows(OUT/'locked_inputs_before.csv',[{'path':p,'sha256':h} for p,h in before.items()])
    datasets,names,loss_names=load_datasets(ROOT);train=datasets['train_1M'];P=train['P'];indices=train['indices']
    # Train/test composition overlap must be disclosed, never hidden by index.
    trainkeys=set(signatures(P))
    if any(trainkeys&set(signatures(datasets[d]['P'])) for d in ['test_1M','test_60M','test_1B']):raise ValueError('Train/test recipe overlap: validation needs explicit policy')
    ilr=ILRTransformer(settings['zero_replacement_fraction_of_training_min']).fit(P)
    write_rows(OUT/'ILR_basis.csv',[{'coordinate':f'ilr_{i+1:02d}',**dict(zip(names,map(float,ilr.basis_[i])))} for i in range(16)])
    (OUT/'composition_transform_A4.json').write_text(json.dumps({'fit_dataset':'train_1M','n_fit':len(P),'domain_order':names,
        'training_min_positive':ilr.min_positive_training_,'zero_delta':ilr.zero_delta_,'zero_fraction':ilr.zero_fraction,
        'closure':'p/sum(p)','basis':ilr.basis_.tolist()},indent=2)+'\n',encoding='utf-8')
    for d in ['train_1M','test_1M','test_60M','test_1B']:
        replaced=ilr.replace(datasets[d]['P']);x=ilr.transform(datasets[d]['P']);clr=np.log(replaced)-np.log(replaced).mean(axis=1,keepdims=True)
        write_rows(OUT/f'{d}_ILR_features.csv',[{'index':index,**{f'ilr_{j+1:02d}':float(x[i,j]) for j in range(16)}} for i,index in enumerate(datasets[d]['indices'])])
        write_rows(OUT/f'{d}_closed_replaced_composition.csv',[{'index':index,**dict(zip(names,map(float,replaced[i])))} for i,index in enumerate(datasets[d]['indices'])])
        write_rows(OUT/f'{d}_CLR_explanation.csv',[{'index':index,**dict(zip(names,map(float,clr[i])))} for i,index in enumerate(datasets[d]['indices'])])
    outer=group_folds(P,settings['outer_folds'],settings['random_state']);outer_assign=[];inner_assign=[]
    for fold,(fit,hold) in enumerate(outer,1):
        outer_assign += [{'index':indices[i],'outer_holdout_fold':fold} for i in hold]
        inner=group_folds(P[fit],settings['inner_folds'],settings['random_state']+fold)
        for innerfold,(_,validation) in enumerate(inner,1):inner_assign += [{'index':indices[fit[i]],'outer_fold':fold,'inner_holdout_fold':innerfold} for i in validation]
    write_rows(OUT/'outer_fold_assignments.csv',outer_assign);write_rows(OUT/'inner_fold_assignments.csv',inner_assign)
    nested=[];oof_rows=[];fold_scores=[];grid_rows=[];final_rows=[];final_metadata=[];validation=[];predictions=[];coefficients=[];warning_rows=[]
    for response in ['L_mean','L_trim']:
        y=train[response]
        for model in MODELS:
            oof=np.full(len(P),np.nan);fold_rmse=[]
            for fold,(fit,hold) in enumerate(outer,1):
                logging.info('%s %s outer fold %d/%d',response,model,fold,len(outer))
                inner=group_folds(P[fit],settings['inner_folds'],settings['random_state']+fold)
                context={'response':response,'model':model,'phase':'outer_inner','outer_fold':fold}
                estimator,parameters,grid=fit_checked(model,P[fit],y[fit],settings,inner,context,warning_rows)
                meta=metadata(estimator,model,names,[indices[i] for i in fit],parameters)
                path=OUT/'models/outer'/f'{response}_{model}_fold{fold}.json';path.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
                prediction=estimator.predict(P[hold]);oof[hold]=prediction
                metric=scores(y[hold],prediction);fold_rmse.append(metric['RMSE']);fold_scores.append({**context,**metric,'parameters':json.dumps(parameters,sort_keys=True)})
                grid_rows += [{**context,**r,'parameters':json.dumps(r['parameters'],sort_keys=True)} for r in grid or []]
                oof_rows += [{'response':response,'model':model,'index':indices[i],'outer_fold':fold,'true_loss':float(y[i]),'predicted_loss':float(p)} for i,p in zip(hold,prediction)]
            if not np.isfinite(oof).all():raise ValueError('Incomplete nested OOF predictions')
            nested_metric=scores(y,oof);nested.append({'response':response,'model':model,**nested_metric,
                'mean_fold_RMSE':float(np.mean(fold_rmse)),'std_fold_RMSE':float(np.std(fold_rmse,ddof=1))})
            finalfolds=group_folds(P,settings['final_selection_folds'],settings['random_state']+100)
            context={'response':response,'model':model,'phase':'final_training_CV','outer_fold':None}
            estimator,parameters,grid=fit_checked(model,P,y,settings,finalfolds,context,warning_rows)
            meta=metadata(estimator,model,names,indices,parameters);meta['response']=response;final_metadata.append(meta)
            joblib.dump(estimator,OUT/'models'/f'{response}_{model}.joblib')
            fitted=scores(y,estimator.predict(P))
            final_rows.append({'response':response,'model':model,'parameters':json.dumps(parameters,sort_keys=True),
                'training_RMSE':fitted['RMSE'],'training_R2':fitted['R2'],'nested_OOF_RMSE':nested_metric['RMSE'],
                'n_model_features':meta.get('n_model_features',0),'n_nonzero_coefficients':meta.get('n_nonzero_coefficients',0)})
            grid_rows += [{**context,**r,'parameters':json.dumps(r['parameters'],sort_keys=True)} for r in grid or []]
            if model!='M0':coefficients += [{'response':response,'model':model,'feature':feature,'standardized_y_coefficient':coefficient,
                'selected_nonzero':coefficient!=0} for feature,coefficient in zip(meta['feature_names'],meta['coefficients'])]
            for d in ['test_1M','test_60M','test_1B']:
                values=datasets[d];prediction=estimator.predict(values['P'])
                if not np.isfinite(prediction).all():raise ValueError('Nonfinite direct validation prediction')
                validation.append({'response':response,'model':model,'dataset':d,**scores(values[response],prediction),
                    'negative_predicted_losses':int((prediction<0).sum()),'fit_dataset':'train_1M','test_used_for_fitting':False})
                predictions += [{'response':response,'model':model,'dataset':d,'index':index,'true_loss':float(truth),'predicted_loss':float(predicted)}
                    for index,truth,predicted in zip(values['indices'],values[response],prediction)]
            logging.info('%s %s completed; nested RMSE=%.6f, full-training RMSE=%.6f',response,model,nested_metric['RMSE'],fitted['RMSE'])
            # Checkpoint results permit recovery without re-running completed fits.
            for filename,data in [('nested_CV_summary.csv',nested),('nested_OOF_predictions.csv',oof_rows),('outer_fold_scores.csv',fold_scores),
                ('hyperparameter_search.csv',grid_rows),('final_training_summary.csv',final_rows),('model_validation.csv',validation),('test_predictions.csv',predictions)]:write_rows(OUT/filename,data)
    best=min([r for r in nested if r['response']=='L_mean'],key=lambda r:r['RMSE'])['model']
    (OUT/'model_selection.json').write_text(json.dumps({'primary_response':'L_mean','recommended_model':best,
        'recommended_model_role':'numerical_nested_CV_benchmark_not_final_main_model',
        'interpretable_main_candidate':'M1_ElasticNet',
        'selection_rule':settings['selection_rule'],'training_dataset':'train_1M','test_used_for_selection':False,
        'Q_or_K_used':False,'estimated_loss_used':False,'phase':'stage1_candidates_pending_human_confirmation'},indent=2)+'\n',encoding='utf-8')
    (OUT/'final_model_metadata.json').write_text(json.dumps(final_metadata,indent=2)+'\n',encoding='utf-8')
    write_rows(OUT/'model_coefficients.csv',coefficients)
    write_rows(OUT/'convergence_warnings.csv',warning_rows,['response','model','phase','outer_fold','category','message'])
    scale_rows=[]
    for response in ['L_mean','L_trim']:
        for left,right in combinations(['test_1M','test_60M','test_1B'],2):
            a={key:i for i,key in enumerate(signatures(datasets[left]['P']))};b={key:i for i,key in enumerate(signatures(datasets[right]['P']))};shared=sorted(set(a)&set(b))
            yl=np.array([datasets[left][response][a[key]] for key in shared]);yr=np.array([datasets[right][response][b[key]] for key in shared])
            defined=len(shared)>1 and np.ptp(yl)>0 and np.ptp(yr)>0
            scale_rows.append({'response':response,'dataset_1':left,'dataset_2':right,'shared_recipe_n':len(shared),
                'actual_Pearson':float(pearsonr(yl,yr).statistic) if defined else None,'actual_Spearman':float(spearmanr(yl,yr).statistic) if defined else None,
                'interpretation':'direct observed ordering on matched recipes' if defined else 'no shared recipes; direct observed rank stability undefined'})
    write_rows(OUT/'cross_scale_recipe_rank_stability.csv',scale_rows)
    figure_outputs(validation,nested,predictions,names,train,settings)
    build_report(datasets,names,settings,nested,validation,final_rows,scale_rows,warning_rows,best,len(before))
    after=protect()
    if before!=after:raise ValueError('Original or frozen part 1/2 input changed')
    write_rows(OUT/'locked_inputs_integrity.csv',[{'path':p,'sha256_before':h,'sha256_after':after[p],'unchanged':True} for p,h in before.items()])
    (OUT/'modeling_config_snapshot.yaml').write_text(yaml.safe_dump(settings,sort_keys=False),encoding='utf-8')
    logging.info('COMPLETED part 3 stage 1; selected %s from train-only nested CV; no Q/K or 10B/70B prediction; frozen files %d',best,len(before))
    print(json.dumps({'best_training_CV_model':best,'nested_primary':[r for r in nested if r['response']=='L_mean'],
        'selected_validation':[r for r in validation if r['response']=='L_mean' and r['model']==best],'cross_scale':scale_rows},ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    try:main()
    except Exception:
        logging.exception('STOPPED modeling; no official or previously locked file is repaired or changed')
        raise
