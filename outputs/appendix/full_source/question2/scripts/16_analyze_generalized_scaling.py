"""Question 2 final phase; frozen analysis only, no refitting and no Question 3.

Run: python -B scripts/16_analyze_generalized_scaling.py
"""
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


def read(relative):
    """Read a declared CSV with round-trip float precision, recording provenance."""
    INPUTS.append(relative)
    return pd.read_csv(ROOT/relative,float_precision='round_trip')


def protected_snapshot():
    """Protect all files from earlier phases, including raw attachment bytes."""
    path=OUT/'frozen_before_phase4.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    rows=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT).as_posix()
        if rel in NEW or p==REPORT or rel.startswith('outputs/scaling/phase4/') or rel.startswith('outputs/figures/question2_phase4/'):
            continue
        rows.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha256(p)))
    result=dict(files=rows,n_files=len(rows))
    write_json(path,result);return result


def verify_frozen(snapshot):
    """Check bytes; never parse original A attachments or rerun previous scripts."""
    bad=[]
    for r in snapshot['files']:
        p=ROOT/r['path']
        if not p.is_file() or p.stat().st_size!=r['bytes'] or sha256(p)!=r['sha256']:bad.append(r['path'])
    if bad:raise RuntimeError('Frozen earlier files changed: '+repr(bad))
    return dict(passed=True,n_files=len(snapshot['files']),changed_files=[])


def diagnose_large():
    """Inspect verified disk schemas before deciding admissible B9/B10 comparison."""
    base='real_attachments/B_scaling_laws/'
    paths=dict(B9=base+'supplementary_large_models.csv',B10=base+'supplementary_large_baseline.csv')
    schemas={};frames={};fields=[]
    for label,relative in paths.items():
        names,records=csv_rows(ROOT/relative)
        t=read(relative);frames[label]=t
        schemas[label]=dict(path=relative,columns=names,n=len(records),
            exact_duplicate_rows=int(t.duplicated().sum()),missing=t.isna().sum().to_dict())
        fields.extend(dict(dataset=label,field=c,missing=int(t[c].isna().sum()),
                           dtype=str(t[c].dtype)) for c in names)
    # These are the actual schemas established by the initial read above.
    expected={'B9':{'model_name','N_params_B','D_tokens_B'},
              'B10':{'family','N_params_B','D_tokens_B','val_loss'}}
    for label,names in expected.items():
        if not names.issubset(frames[label].columns):raise ValueError(label+' schema differs; stop dependent analysis')
    a,b=frames['B9'],frames['B10']
    for label,t,key in [('B9',a,'model_name'),('B10',b,'family')]:
        if t[key].isna().any() or t[key].duplicated().any():raise ValueError('Ambiguous large-model join key')
        schemas[label].update(N_min=float(t.N_params_B.min()),N_max=float(t.N_params_B.max()),
              D_min=float(t.D_tokens_B.min()),D_max=float(t.D_tokens_B.max()),
              nonpositive_N=int((t.N_params_B<=0).sum()),nonpositive_D=int((t.D_tokens_B<=0).sum()),
              name_duplicates=int(t[key].duplicated().sum()))
    j=a.merge(b,left_on='model_name',right_on='family',how='outer',suffixes=['_B9','_B10'],indicator=True,validate='one_to_one')
    both=j._merge=='both'
    if not np.array_equal(j.loc[both,'N_params_B_B9'],j.loc[both,'N_params_B_B10']) or not np.array_equal(j.loc[both,'D_tokens_B_B9'],j.loc[both,'D_tokens_B_B10']):
        raise ValueError('Matched large-model scales disagree')
    j['join_status']=j.pop('_merge').astype(str)
    j.to_csv(OUT/'B9_B10_join.csv',index=False,encoding='utf-8-sig')
    write_csv(OUT/'B9_B10_field_diagnosis.csv',fields)
    write_json(OUT/'B9_B10_schema.json',schemas)
    notes=['# B9/B10 实际数据诊断','',
        'B9 是模型元数据，无直接可比较 Loss。B10 的磁盘字段名为 val_loss，但按赛题口径是估算 Loss。','']
    for label,s in schemas.items():
        notes += [f"## {label}",'',f"文件：{s['path']}",f"实际字段：{s['columns']}",
                  f"行数：{s['n']}；N 范围：{s['N_min']}–{s['N_max']}B；D 范围：{s['D_min']}–{s['D_max']}B。",
                  f"缺失字段计数：{s['missing']}；完全重复：{s['exact_duplicate_rows']}；名称重复：{s['name_duplicates']}。",
                  f"非正 N：{s['nonpositive_N']}；非正 D：{s['nonpositive_D']}。",'']
    notes += ['## 对应关系','',f"按 B9.model_name = B10.family 精确一对一关联：匹配 {int(both.sum())}，"
              f"仅 B9 {int((j.join_status=='left_only').sum())}，仅 B10 {int((j.join_status=='right_only').sum())}。",
              '匹配行 N、D 完全一致。B9 中 D=0 的行完整保留，D=0 不可代入经典公式，不能视作实际零训练或擅自填补。',
              'B9 只提供所报告参数规模和训练数据规模的现实锚点。元数据未统一稠密/稀疏模型、总/激活参数及各家评测口径。',
              'B10 对照仅设置结构参考 Q=1、p=p_ref，计算 L0；这不是测得的大模型真实 Q/p，也不验证完整广义模型。','']
    (OUT/'B9_B10_diagnosis.md').write_text('\n'.join(notes),encoding='utf-8')
    return a,b,schemas


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


def workpoints(model,prior,cfg):
    """Derive observed D quantiles, domain-Q quantiles and actual saved recipes."""
    b6=read('outputs/scaling/phase2C/B6_predictions.csv')
    b7=read('outputs/scaling/phase2C/B7_predictions.csv')
    pairs=np.array(sorted(set(zip(b6.N_params_B,b6.D_tokens_B))))
    ds=np.quantile(pairs[:,1],cfg['D_quantiles'])
    domains=read('outputs/quality/domain_quality_summary_A1.csv')
    qs=[('domain_min',float(domains.Q_domain.min())),('domain_median',float(domains.Q_domain.median())),
        ('domain_max',float(domains.Q_domain.max())),('Q_0.7',float(cfg['Q_representative']))]
    bases=read('outputs/mixture/stage2/representative_baselines.csv')
    recipes=[]
    for label,name in [('p_ref','A4_ILR_center'),('P10','A4_predicted_low'),('P90','A4_predicted_high')]:
        g=bases[bases.baseline==name]
        if len(g)!=1:raise ValueError('Missing frozen recipe baseline')
        p=g[model.meta['domain_order']].to_numpy()[0]
        m=float(predict_recipe(p,model.meta)[0])
        if abs(m-float(g.iloc[0].frozen_EN_prediction))>1e-12:raise ValueError('Frozen baseline M(p) mismatch')
        # p_ref uses the already frozen reference and score, exactly.
        delta=0. if label=='p_ref' else m-model.M_ref
        recipes.append(dict(recipe=label,baseline=name,index=None if pd.isna(g.iloc[0].recipe_index) else int(g.iloc[0].recipe_index),
                            p=p.tolist(),M_p=m,DeltaM=delta))
    support=dict(N_min_B=float(min(b6.N_params_B.min(),b7.N_params_B.min())),
                 N_max_B=float(max(b6.N_params_B.max(),b7.N_params_B.max())),
                 D_min_B=float(min(b6.D_tokens_B.min(),b7.D_tokens_B.min())),
                 D_max_B=float(max(b6.D_tokens_B.max(),b7.D_tokens_B.max())),
                 Q_min=float(min(b6.Q_score.min(),b7.Q_score.min())),Q_max=float(max(b6.Q_score.max(),b7.Q_score.max())))
    result=dict(D_quantiles=dict(zip(['P25','median','P75'],map(float,ds))),
        D_definition='B6 observed 45 distinct N-D blocks; exclude repeated Q levels, not data records',
        domain_Q_points=dict(qs),domain_Q_values=domains[['domain','Q_domain']].to_dict('records'),
        recipes=recipes,quality_numeric_support=support,Q_semantic_alignment_not_jointly_observed=True,
        primary_12B_is_slightly_above_actual_11_97B=12>support['N_max_B'])
    write_json(OUT/'workpoint_provenance.json',result)
    return ds,qs,recipes,support,result


def lambda_definition(model):
    """Exact main-policy segments, including open positive-N clamp boundaries."""
    r=[dict(segment='lower_clamp',N_low=0.,N_high=float(model.Ns[0]),
            lambda_low=float(model.lambdas[0]),lambda_high=float(model.lambdas[0]),s_i=0.,domain='0<N<1M')]
    for i,s in enumerate(model.slopes):
        r.append(dict(segment=f'interior_{i+1}',N_low=float(model.Ns[i]),N_high=float(model.Ns[i+1]),
               lambda_low=float(model.lambdas[i]),lambda_high=float(model.lambdas[i+1]),s_i=float(s),
               domain='continuous at knots; one-sided derivatives'))
    r.append(dict(segment='upper_clamp',N_low=float(model.Ns[-1]),N_high=None,
                  lambda_low=float(model.lambdas[-1]),lambda_high=float(model.lambdas[-1]),s_i=0.,domain='N>70B'))
    write_csv(OUT/'lambda_piecewise_definition.csv',r)
    return r


def elasticity_grid(model,ds,qs,recipes,support,cfg):
    """All requested workpoints, derivative side checks, and exact Q+0.1 roots."""
    grid=[];checks=[];eqs=[]
    for N in cfg['N_workpoints_B']:
        knot=bool(N in model.Ns)
        for dlabel,D in zip(['P25','median','P75'],ds):
            for qlabel,Q in qs:
                for recipe in recipes:
                    dm=recipe['DeltaM'];side=cfg['knot_derivative_main']
                    terms=model.terms(N,D,Q,dm);der=model.derivatives(N,D,Q,dm,side=side)
                    elastic=model.elasticity(N,D,Q,dm,side=side)
                    left=model.elasticity(N,D,Q,dm,side='left');right=model.elasticity(N,D,Q,dm,side='right')
                    valid=bool(elastic['valid']);dominant=max(['N','D','Q'],key=lambda k:float(elastic['E_'+k])) if valid else None
                    record=dict(N_params_B=N,D_tokens_B=float(D),D_label=dlabel,Q=Q,Q_label=qlabel,
                        recipe=recipe['recipe'],recipe_index=recipe['index'],DeltaM=dm,
                        lambda_p=float(model.lambda_value(N)),**{k:float(v) for k,v in terms.items()},
                        **{k:float(v) for k,v in der.items()},
                        E_N=float(elastic['E_N']) if valid else None,E_D=float(elastic['E_D']) if valid else None,
                        E_Q=float(elastic['E_Q']) if valid else None,
                        E_N_left=float(left['E_N']) if valid else None,E_N_right=float(right['E_N']) if valid else None,
                        dominant_factor=dominant,elasticity_status='valid' if valid else 'nonpositive_reducible_loss',
                        N_is_lambda_knot=knot,N_derivative_kind='right_at_knot' if knot else 'interior',
                        extrapolative=bool(N<support['N_min_B'] or N>support['N_max_B']),
                        quality_evidence='Class B local' if support['N_min_B']<=N<=support['N_max_B'] else 'Class B extrapolation',
                        mixture_evidence='Class D estimated-supported' if N>1 else 'Class C observed-scale interpolation',
                        mixture_N_derivative_evidence=('not_applicable_reference_recipe' if dm==0 else
                            'Class D: 1B right side or larger N' if N>=1 else 'Class C observed-scale interpolation'))
                    for k in ['N','D','Q']:record['traditional_'+k]=float(elastic['traditional_'+k])
                    grid.append(record)
                    for variable in ['N','D','Q']:
                        sides=['left','right'] if variable=='N' and knot else [None]
                        for chosen in sides:
                            exact=float(model.derivatives(N,D,Q,dm,side=chosen or 'right')['dL_d'+variable])
                            numeric=finite_difference(model,N,D,Q,dm,variable,cfg,side=chosen)
                            absolute=abs(exact-numeric);relative=absolute/max(abs(exact),1e-12)
                            passed=relative<=cfg['derivative_relative_tolerance'] or absolute<=cfg['derivative_absolute_tolerance']
                            checks.append(dict(N_params_B=N,D_tokens_B=float(D),Q=Q,recipe=recipe['recipe'],variable=variable,
                                derivative_side=chosen or 'central',analytic=exact,finite_difference=numeric,
                                absolute_error=absolute,relative_error=relative,passed=passed))
                    equivalence=model.equivalence(N,Q,dm)
                    root=equivalence.get('N_eq_B')
                    eqs.append(dict(N_params_B=N,D_tokens_B=float(D),D_label=dlabel,Q=Q,Q_label=qlabel,
                        recipe=recipe['recipe'],recipe_index=recipe['index'],DeltaM=dm,
                        extrapolative=record['extrapolative'],quality_support='extrapolation' if record['extrapolative'] else 'supported_numeric_range',
                        equivalent_N_outside_quality_support=bool(root is not None and root>support['N_max_B']),
                        root_crosses_lambda_knots=bool(root is not None and np.any((model.Ns>N)&(model.Ns<root))),
                        **equivalence))
    write_csv(OUT/'elasticity_grid.csv',grid);write_csv(OUT/'marginal_derivative_checks.csv',checks)
    if not all(r['passed'] for r in checks):raise AssertionError('Derivative verification failed')
    write_csv(OUT/'Q01_parameter_equivalence_grid.csv',eqs)
    primary=[r for r in eqs if r['N_params_B'] in cfg['N_primary_B'] and r['D_label']=='median' and r['Q_label']!='Q_0.7']
    write_csv(OUT/'Q01_parameter_equivalence.csv',[r for r in primary if r['recipe']=='p_ref'])
    write_csv(OUT/'Q01_parameter_equivalence_recipe_sensitivity.csv',[r for r in primary if r['recipe']!='p_ref'])
    g=pd.DataFrame(grid);summaries=[]
    for (N,recipe),sub in g.groupby(['N_params_B','recipe']):
        for factor in ['N','D','Q']:
            summaries.append(dict(N_params_B=float(N),recipe=recipe,factor=factor,n_workpoints=len(sub),
                dominant_count=int((sub.dominant_factor==factor).sum()),
                dominant_fraction=float((sub.dominant_factor==factor).mean()),
                elasticity_min=float(sub['E_'+factor].min()),elasticity_max=float(sub['E_'+factor].max())))
    write_csv(OUT/'elasticity_dominance_summary.csv',summaries)
    solved=pd.DataFrame(eqs);solved=solved[solved.status=='solved'];v=solved.local_relative_error.to_numpy()
    write_json(OUT/'equivalence_summary.json',dict(status_counts=pd.Series([r['status'] for r in eqs]).value_counts().to_dict(),
        n_main_rows=sum(r['recipe']=='p_ref' for r in primary),
        approximation_relative_error_mean=float(v.mean()),approximation_absolute_relative_error_mean=float(np.mean(abs(v))),
        approximation_absolute_relative_error_max=float(np.max(abs(v))),
        D_independence='Exact under fixed additive D term; it cancels from both sides',
        max_equation_residual=float(np.max(abs(solved.equation_residual))),
        quality_derivative_max_relative_error=max(r['relative_error'] for r in checks if r['variable']=='Q'),
        derivative_max_relative_error=max(r['relative_error'] for r in checks)))
    return grid,checks,eqs,summaries


def training_recipe_regions(model,ds,qs,support,cfg):
    """Actual 512 saved recipes: diagnose local/finite equivalence beyond P10/P90."""
    t=read('outputs/mixture/train_1M_ILR_features.csv');M=predict_ilr(t.iloc[:,1:].to_numpy(),model.meta)
    records=[]
    for N in cfg['N_workpoints_B']:
        for label,Q in qs:
            for index,m in zip(t['index'],M):
                dm=float(m-model.M_ref);sol=model.equivalence(N,Q,dm)
                records.append(dict(recipe_index=int(index),N_params_B=N,D_tokens_B=float(ds[1]),Q=Q,Q_label=label,
                    M_p=float(m),DeltaM=dm,extrapolative=bool(N>support['N_max_B'] or N<support['N_min_B']),**sol))
    write_csv(OUT/'training_recipe_equivalence_diagnosis.csv',records)
    write_csv(OUT/'no_equivalence_records.csv',[r for r in records if r['status']!='solved'],columns=list(records[0]))
    summary=[]
    for N in cfg['N_workpoints_B']:
        for label,Q in qs:
            sub=[r for r in records if r['N_params_B']==N and r['Q_label']==label]
            for status in ['solved','no_local_equivalence','no_finite_equivalence_within_range']:
                summary.append(dict(N_params_B=N,Q=Q,Q_label=label,status=status,n=int(sum(r['status']==status for r in sub)),total=len(sub)))
    write_csv(OUT/'training_recipe_equivalence_status_summary.csv',summary)
    return summary


def domain_and_joint(model):
    """Scale frozen 1pp first-order effects and 136 pair deviations, no causal claim."""
    effects=read('outputs/mixture/domain_marginal_effects.csv')
    effects=effects[(effects.model=='M1_ElasticNet') & np.isclose(effects.step,.01,rtol=0,atol=1e-14)]
    joint=read('outputs/mixture/domain_joint_effects.csv')
    joint=joint[(joint.model=='M1_ElasticNet') & np.isclose(joint.step,.01,rtol=0,atol=1e-14)]
    if len(effects)!=17 or len(joint)!=136:raise ValueError('Frozen effect dimensions differ')
    out=[];pairs=[]
    for N,label,evidence in [( .001,'1M','observed-supported'),(.06,'60M','observed-supported'),(1.,'1B','observed-supported'),
                              (10.,'10B','estimated-supported'),(70.,'70B','estimated-supported')]:
        lam=float(model.lambda_value(N))
        for _,r in effects.iterrows():
            values=lam*np.array([r.min_effect,r.max_effect]);low,high=float(values.min()),float(values.max())
            out.append(dict(scale=label,N_params_B=N,evidence=evidence,domain=r.mixture_domain,
                step=.01,summary_scope='equal_mean_of_five_frozen_representative_baselines',
                n_baselines=int(r.n_baselines),DeltaM_j=float(r.mean_effect),lambda_p=lam,
                DeltaL_j=float(lam*r.mean_effect),min_effect=low,max_effect=high,
                direction='lower_loss' if high<0 else 'higher_loss' if low>0 else 'baseline_dependent',
                source_baseline_direction=r.baseline_direction))
        matrix=np.full((17,17),np.nan);names=model.meta['domain_order']
        for _,r in joint.iterrows():
            value=float(lam*r.mean_effect);i,j=names.index(r.domain_1),names.index(r.domain_2)
            matrix[i,j]=matrix[j,i]=value
            values=lam*np.array([r.min_effect,r.max_effect]);lo,hi=float(values.min()),float(values.max())
            pairs.append(dict(scale=label,N_params_B=N,evidence=evidence,domain_1=r.domain_1,domain_2=r.domain_2,
                step=.01,n_baselines=int(r.n_baselines),I_jk=float(r.mean_effect),lambda_p=lam,
                scaled_I_jk=value,scaled_min=lo,scaled_max=hi,
                trend='conditional_complementary' if hi<0 else 'conditional_competitive' if lo>0 else 'baseline_dependent',
                source_deviation_direction=r.baseline_deviation_direction,causal_interpretation=False))
        pd.DataFrame(matrix,index=names,columns=names).to_csv(OUT/f'joint_trend_matrix_{label}.csv',encoding='utf-8-sig',index_label='domain')
    write_csv(OUT/'domain_effect_by_scale.csv',out);write_csv(OUT/'joint_trend_by_scale.csv',pairs)
    # Preserve all five baseline cases as well as the equal-baseline summary.
    c=read('outputs/mixture/stage2/domain_marginal_effect_cases.csv')
    c=c[(c.model=='M1_ElasticNet') & np.isclose(c.step,.01,rtol=0,atol=1e-14)]
    d=read('outputs/mixture/stage2/domain_joint_effect_cases.csv')
    d=d[(d.model=='M1_ElasticNet') & np.isclose(d.step,.01,rtol=0,atol=1e-14)]
    cases=[];jc=[]
    for N in model.Ns:
        lam=float(model.lambda_value(N))
        cases.extend(dict(N_params_B=float(N),lambda_p=lam,scaled_effect=float(lam*r['effect']),**r) for r in c.to_dict('records'))
        jc.extend(dict(N_params_B=float(N),lambda_p=lam,scaled_deviation=float(lam*r['joint_perturbation_deviation']),**r) for r in d.to_dict('records'))
    write_csv(OUT/'domain_effect_cases_by_scale.csv',cases);write_csv(OUT/'joint_trend_cases_by_scale.csv',jc)
    return out,pairs


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


def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
          handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'execution.log',encoding='utf-8')])
    snapshot=protected_snapshot();verify_frozen(snapshot)
    logging.info('Phase 4 starts: %d prior files frozen; analyze only',len(snapshot['files']))
    cfg=yaml.safe_load((ROOT/'configs/scaling_phase4.yaml').read_text(encoding='utf-8'));INPUTS.append('configs/scaling_phase4.yaml')
    b9,b10,schemas=diagnose_large();logging.info('Schemas diagnosed: B9=%d metadata, B10=%d estimated Loss',len(b9),len(b10))
    model,prior,knots=frozen_model(cfg)
    ds,qs,recipes,support,points=workpoints(model,prior,cfg);definition=lambda_definition(model)
    grid,checks,eqs,dominance=elasticity_grid(model,ds,qs,recipes,support,cfg)
    logging.info('Workpoint grid %d; derivative checks %d passed; equivalence statuses=%s',len(grid),len(checks),pd.Series([r['status'] for r in eqs]).value_counts().to_dict())
    regions=training_recipe_regions(model,ds,qs,support,cfg);logging.info('All 512 real training recipes checked for equivalence existence')
    domains,joints=domain_and_joint(model);metrics=large_validation(model,b9,b10,cfg)
    sensitivity=lambda_sensitivity(model,recipes,cfg)
    final=dict(classical=model.c,quality=model.q,reference=prior['reference'],lambda_knots=knots,
         lambda_main_policy='S-A',lambda_interpolation='linear_in_ln_N',below_first_knot='hold-first',above_last_knot='hold-last',
         evidence=prior['evidence'],status='human_confirmed_modular_form_with_conditional_evidence',
         joint_external_validation_available=False,no_parameters_refitted=True,N_D_units='billions',
         quality_module_interpretation='B6-calibrated local quality correction',quality_numeric_support=support,
         derivative_at_knots='left/right both recorded; right derivative is main convention',
         extrapolation_of_quality_not_empirically_validated=True)
    write_json(OUT/'generalized_scaling_final.json',final)
    INPUTS.append('outputs/scaling/phase3/component_validation.csv')
    write_csv(OUT/'input_manifest.csv',[dict(path=p,sha256=sha256(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in dict.fromkeys(INPUTS)])
    from src.scaling.marginal_reporting import plot_analysis,build_final_report
    plot_analysis(OUT,FIG,grid,eqs,domains,joints,metrics,sensitivity,cfg)
    build_final_report(REPORT,OUT,final,points,grid,eqs,dominance,regions,domains,joints,metrics,sensitivity,schemas,cfg)
    write_json(OUT/'frozen_integrity_check.json',verify_frozen(snapshot))
    write_json(OUT/'execution_summary.json',dict(completed=True,n_frozen_files=len(snapshot['files']),n_grid_points=len(grid),
        n_derivative_checks=len(checks),all_derivative_checks_passed=True,n_training_recipe_equivalence_points=512*len(cfg['N_workpoints_B'])*len(qs),
        model_refit=False,original_A_parsed=False,B9_role='metadata_anchor',B10_role='estimated_classical_consistency',
        joint_external_test_fabricated=False,question3_started=False))
    write_json(OUT/'implementation_manifest.json',[dict(path=p,sha256=sha256(ROOT/p)) for p in sorted(NEW) if (ROOT/p).exists()])
    logging.info('Question 2 Phase 4 completed normally; stopped before Question 3')


def refresh_report():
    """Refresh text from saved calculations only; never rerun analysis or plots."""
    snapshot=protected_snapshot();verify_frozen(snapshot)
    cfg=yaml.safe_load((ROOT/'configs/scaling_phase4.yaml').read_text(encoding='utf-8'))
    def records(name):return pd.read_csv(OUT/name,float_precision='round_trip').to_dict('records')
    final=json.loads((OUT/'generalized_scaling_final.json').read_text(encoding='utf-8'))
    points=json.loads((OUT/'workpoint_provenance.json').read_text(encoding='utf-8'))
    schemas=json.loads((OUT/'B9_B10_schema.json').read_text(encoding='utf-8'))
    from src.scaling.marginal_reporting import build_final_report
    build_final_report(REPORT,OUT,final,points,records('elasticity_grid.csv'),records('Q01_parameter_equivalence_grid.csv'),
        records('elasticity_dominance_summary.csv'),records('training_recipe_equivalence_status_summary.csv'),
        records('domain_effect_by_scale.csv'),records('joint_trend_by_scale.csv'),
        records('B10_extrapolation_metrics.csv'),records('lambda_beyond70B_sensitivity.csv'),schemas,cfg)
    write_json(OUT/'implementation_manifest.json',[dict(path=p,sha256=sha256(ROOT/p)) for p in sorted(NEW)])
    write_json(OUT/'frozen_integrity_check.json',verify_frozen(snapshot))
    print('Report refreshed from saved calculations; frozen files unchanged')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--refresh-report',action='store_true')
    args=parser.parse_args()
    try:
        if args.refresh_report:refresh_report()
        else:main()
    except Exception:
        logging.exception('Phase 4 stopped on error; no frozen parameters changed')
        raise
