"""Data readiness, twelve scientific figures, and a bounded Phase 1 report."""
import hashlib
import json
import logging
import re
import numpy as np
import pandas as pd
from .common import FILES,csv,dump,table


def read(out,name):return pd.read_csv(out/name,float_precision='round_trip')


def scale_readiness(root,cfg,models):
    """Inspect observed scale fields/units and sample support, without attribution."""
    out=root/'outputs/evolution/phase1';base=root/'real_attachments/C_efficiency_evolution'
    e=pd.read_csv(base/FILES['C4'],float_precision='round_trip');rows=[]
    for i,r in e.iterrows():
        notes=str(r['Parameters notes']);match=re.search(r'(?i)(\d+(?:\.\d+)?)\s*([kmbt])\b',notes)
        implied=float(match.group(1))*{'k':1e3,'m':1e6,'b':1e9,'t':1e12}[match.group(2).lower()] if match else np.nan
        N=pd.to_numeric(r.Parameters,errors='coerce');D=pd.to_numeric(r['Training dataset size (total)'],errors='coerce');F=pd.to_numeric(r['Training compute (FLOP)'],errors='coerce')
        ratio=abs(N-implied)/implied if pd.notna(N) and pd.notna(implied) and implied>0 else np.nan
        rows.append(dict(C4_row=i,Model=r.Model,Domain=r.Domain,N_parameters_raw=N,training_compute_raw=F,
            training_dataset_raw=r['Training dataset size (total)'],dataset_numeric=D,
            dataset_token_unit_confirmed=bool(re.search(r'(?i)\btokens?\b',str(r['Dataset size notes']))),
            parameter_notes_first_explicit_count=implied,parameter_notes_relative_difference=ratio,
            parameter_notes_possible_conflict=bool(pd.notna(ratio) and ratio>cfg['parameter_conflict_relative_threshold']),
            parameter_notes=r['Parameters notes'],Dataset_size_notes=r['Dataset size notes'],
            caution='notes may describe total versus active parameters; diagnostic only, no correction'))
    csv(out/'C4_scale_unit_audit.csv',rows)
    valid_date=models['Submission Date'].notna();positiveN=models.N_B.gt(0);sample=[]
    for name,mask in [('S0',models.S0),('S1',models.S1),('S2',models.S2),
        ('S1_N_submission',models.S1&positiveN&valid_date),('all_joint_N_D_compute',models.scale_joint_available),
        ('S1_joint_N_D_compute_submission_no_N_conflict',models.contribution_sample_candidate),
        ('S1_pretrained_joint',models.contribution_sample_pretrained),
        ('S1_joint_with_nonnegative_release_lag',models.contribution_sample_candidate&models.submission_release_days.ge(0))]:
        sample.append(dict(sample=name,n=int(mask.sum()),definition='no interpolated N/D/compute; unit/source/conflict checks retained'))
    csv(out/'contribution_sample_sizes.csv',sample)
    g=models[models.contribution_sample_candidate]
    correlation=np.log(g[['N_B','D_tokens','training_compute']]).corr()
    correlations=[]
    for a in correlation:
        for b in correlation:correlations.append(dict(variable_1=a,variable_2=b,log_correlation=correlation.loc[a,b],n=len(g)))
    csv(out/'scale_input_correlations.csv',correlations)
    counts={r['sample']:r['n'] for r in sample}
    readiness=dict(sample_counts=counts,contribution_analysis_started=False,
        sufficient_for_descriptive_ability_analysis=True,full_N_D_compute_decomposition_strongly_supported=False,
        universal_loss_bridge_validated=False,strict_license_sample_available=counts['S2']>0,
        recommended_phase2_scope='conditional limited-sample/N-only sensitivity possible after human confirmation; no causal or precise contribution-ratio claim yet',
        unresolved=['small complete-scale sample and selection bias','single-family High bridge fails predictive validation',
                    'license grant not established from attachments','submission cohorts are not necessarily contemporaneous evaluation frontiers',
                    'release/submission discrepancies and conservative model identity joins'])
    dump(out/'phase2_readiness.json',readiness);return readiness


def figures(root,cfg,models):
    """Twelve plots, including failed bridge evidence; no future extrapolation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'outputs/evolution/phase1';dest=root/'outputs/figures/question4_phase1';dest.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    manifest=[]
    def save(fig,name,title):
        fig.tight_layout();path=dest/name;fig.savefig(path,dpi=170);plt.close(fig)
        manifest.append(dict(file=path.relative_to(root).as_posix(),title=title))
    raw=pd.read_csv(root/'real_attachments/C_efficiency_evolution'/FILES['C2']);metrics=cfg['benchmarks']
    fig,ax=plt.subplots(figsize=(9,4));ax.boxplot([raw[m] for m in metrics],tick_labels=['IFEval','BBH','MATH-5','GPQA','MUSR','MMLU-PRO'],showfliers=False)
    ax.set(ylabel='Leaderboard score (0-100)',title='Six benchmark distributions: all original evaluation records',ylim=(0,100))
    save(fig,'01_benchmark_distributions.png','Six benchmark distributions')
    fig,ax=plt.subplots(figsize=(7,6));c=models.loc[models.S1,metrics].corr();im=ax.imshow(c,vmin=-1,vmax=1,cmap='RdBu_r')
    labels=['IFEval','BBH','MATH-5','GPQA','MUSR','MMLU-PRO'];ax.set_xticks(range(6),labels,rotation=35,ha='right');ax.set_yticks(range(6),labels)
    for i in range(6):
        for j in range(6):ax.text(j,i,f'{c.iloc[i,j]:.2f}',ha='center',va='center',color='white' if abs(c.iloc[i,j])>.65 else 'black')
    fig.colorbar(im,ax=ax,label='Pearson r');ax.set_title('S1 task correlations');save(fig,'02_benchmark_correlations.png','S1 task Pearson heatmap')
    for number,col,title in [(3,cfg['average_column'],'Equal robust index versus official Average'),(4,'A_pca','Equal robust index versus PCA sensitivity')]:
        fig,ax=plt.subplots(figsize=(6,4));g=models[models.S0];ax.scatter(g.A_z,g[col],s=9,alpha=.15,color='#777777',label='S0')
        h=g[g.S1];ax.scatter(h.A_z,h[col],s=12,alpha=.45,color='#2878a8',label='S1');ax.set(xlabel='A_z (fixed robust equal-weight index)',ylabel='A_pca' if col=='A_pca' else 'Official Average',title=title);ax.legend()
        save(fig,f'{number:02d}_ability_index_comparison.png',title)
    match=read(out,'C8_matched_model_analysis.csv');g=match[match.unique_model_match&match.six_complete]
    fig,ax=plt.subplots(figsize=(7,4));ax.scatter(g.A_z_C8,g.bottom20_task_score,s=10,alpha=.3,color='#777777',label='S0 exact matches')
    h=g[g.S1];ax.scatter(h.A_z_C8,h.bottom20_task_score,s=15,alpha=.6,color='#2878a8',label='S1 exact matches')
    ax.set(xlabel='Same-run C8 ability, using fixed S1 transform',ylabel='Bottom 20% leaf-task mean (0-100)',title='C8 ability versus weak tail; chance-floor effects retained');ax.legend()
    save(fig,'05_C8_ability_vs_tail.png','C8 same-run ability and bottom20 tail')
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].hist(g.capability_imbalance.dropna(),bins=30,color='#2878a8',alpha=.8);axes[0].set(xlabel='Leaf score P90-P10 (percentage points)',ylabel='Models',title='Chance-normalized imbalance')
    axes[1].hist(g.difficulty_adjusted_imbalance.dropna(),bins=30,color='#c07328',alpha=.8);axes[1].set(xlabel='Within-task robust-z P90-P10',ylabel='Models',title='Task-difficulty sensitivity')
    save(fig,'06_C8_imbalance_distribution.png','C8 imbalance and difficulty-adjusted sensitivity')
    monthly=read(out,'monthly_ability_summary.csv');g=monthly[(monthly['sample']=='S1')&(monthly.model_type=='all')].sort_values('month')
    fig,ax=plt.subplots(figsize=(9,4))
    for field,label in [('median_A','P50'),('P90_A','P90'),('P95_A','P95')]:ax.plot(g.month,g[field],'o-',label=label)
    ax.set(ylabel='A_z',title='S1 submission cohorts: descriptive snapshot ability');ax.tick_params(axis='x',rotation=40);ax.legend();ax.grid(alpha=.2)
    save(fig,'07_monthly_ability_quantiles.png','Monthly S1 P50/P90/P95')
    fig,ax=plt.subplots(figsize=(9,4))
    for kind in ['pretrained','chat_finetuned']:
        h=monthly[(monthly['sample']=='S1')&(monthly.model_type==kind)].sort_values('month');ax.plot(h.month,h.P95_A,'o-',label=kind)
    ax.set(ylabel='Monthly A_z P95',title='S1 pretrained/chat-finetuned submission cohorts');ax.tick_params(axis='x',rotation=40);ax.legend();ax.grid(alpha=.2)
    save(fig,'08_model_type_ability_trends.png','Pretrained versus chat-finetuned monthly description')
    for number,field,label in [(9,'N_B','Leaderboard parameters (B)'),(10,'training_compute','Matched training compute (FLOPs)')]:
        fig,ax=plt.subplots(figsize=(7,4));g=models[models.S1&models[field].gt(0)&models.A_z.notna()]
        for kind,color in [('pretrained','#2878a8'),('chat_finetuned','#c07328'),('other_unknown','#777777')]:
            h=g[g.model_type==kind];ax.scatter(h[field],h.A_z,s=20,alpha=.6,label=f'{kind} (n={len(h)})',color=color)
        ax.set_xscale('log');ax.set(xlabel=label,ylabel='A_z',title='Observed metadata versus ability; no causal fit');ax.legend(fontsize=8)
        save(fig,f'{number:02d}_scale_vs_ability.png',label+' versus A_z')
    bridge=read(out,'C6_bridge_transformed_records.csv');bands=read(out,'loss_benchmark_bridge.csv');selected=json.loads((out/'bridge_model.json').read_text(encoding='utf8'))
    fig,ax=plt.subplots(figsize=(8,4))
    for kind,color in [('High','#2878a8'),('Medium','#c07328')]:
        h=bridge[bridge.comparability==kind];ax.scatter(h.Val_Loss,h.A_bridge,s=24,alpha=.7,label=f'{kind} (n={len(h)})',color=color)
    b=bands[bands.selected];ax.plot(b.Val_Loss,b.A_pred,color='#333333',label='High isotonic fit: constant, diagnostic only')
    ax.fill_between(b.Val_Loss.to_numpy(),b.mean_CI_low.to_numpy(),b.mean_CI_high.to_numpy(),color='#777777',alpha=.2,label='High paired-bootstrap mean band')
    ax.set(xlabel='Val_Loss (source-dependent comparability)',ylabel='A_bridge (same fixed ability scale)',title='Loss-ability bridge: no validated universal conversion');ax.legend(fontsize=8)
    save(fig,'11_loss_vs_ability_bridge.png','Bridge stratified by comparability')
    pred=read(out,'bridge_validation_predictions.csv');g=pred[pred.candidate==selected['selected_candidate']];fig,axes=plt.subplots(1,2,figsize=(10,4))
    scores=read(out,'bridge_validation_metrics.csv')
    for ax,kind in zip(axes,['High_LOMO_CV','Medium_external']):
        h=g[g.evaluation==kind];ax.scatter(h.observed_A,h.predicted_A,s=22,alpha=.7,color='#2878a8');lo=min(h.observed_A.min(),h.predicted_A.min());hi=max(h.observed_A.max(),h.predicted_A.max())
        ax.plot([lo,hi],[lo,hi],ls='--',color='#777777');r=scores[(scores.candidate==selected['selected_candidate'])&(scores.evaluation==kind)].iloc[0]
        ax.set(xlabel='Observed A_bridge',ylabel='Predicted A_bridge',title=f'{kind}\nRMSE={r.RMSE:.4f}, R2={r.R2:.3f}')
    save(fig,'12_bridge_observed_vs_predicted.png','Failed predictive validation remains visible')
    csv(out/'figure_manifest.csv',manifest)


def stage_report(root,cfg,models,readiness):
    """Answer every Phase 1 reporting question with real output links."""
    out=root/'outputs/evolution/phase1';ability=json.loads((out/'ability_execution_summary.json').read_text(encoding='utf8'))
    c8=json.loads((out/'C8_execution_summary.json').read_text(encoding='utf8'));bridge=json.loads((out/'bridge_model.json').read_text(encoding='utf8'))
    profile=read(out,'benchmark_profile.csv').query('scope == "C2_all_evaluations"')
    compare=read(out,'ability_index_comparison.csv');tails=read(out,'C8_weak_tail_diagnosis.csv');validation=read(out,'bridge_validation_metrics.csv');c3=read(out,'C3_time_series_audit.csv')
    counts=read(out,'contribution_sample_sizes.csv');consistent=read(out,'C1_C2_consistency.csv');parameters=json.loads((out/'ability_transform_parameters.json').read_text(encoding='utf8'))
    text=['# 问题四 PHASE 1：数据口径、综合能力与Loss–Benchmark桥接',
        '本阶段从C附件独立计算，没有读取旧问题四权重、指数、贡献率或预测，没有重新拟合前三问。所有“趋势”仅为已有样本描述；没有预测未来12/24个月、分解贡献比例、建立最终动力学或做因果解释。',
        '## 1 实际附件和数据诊断',table(read(out,'file_diagnosis.csv')),
        f"C1/C2有{ability['raw_C1_C2_records']}条六维完整评测、{ability['unique_models']}个不同Model字符串，重复额外记录{ability['duplicate_extra_records']}条。模型集合、参数、日期和类别一致。六维及Average若干浮点存在CSV重写舍入差异，最大{consistent.maximum_absolute_difference.max():.3g}；不是逐位完全相同，但1e-12审计容差内一致，不改动数据。主工作表采用C2。",
        'C2新增Epoch_AI_Publication_Date、Epoch_AI_Organization、Epoch_AI_Open_Weights，附件将其标为Epoch AI增强元数据，但没有明确C4模型ID；保守匹配须复核。原始同目录parquet额外提供评测行的架构/精度/权重类型、原始得分等，用于溯源，未读历史问题四草稿。',
        '全部字段缺失率、日期、参数及Benchmark范围见field_diagnosis.csv；原始数据只读。C4混合领域且存在缺失、非统一数据单位及部分参数notes冲突，不按模型名称猜值。',
        '## 2 样本层级、模型类型和时间轴',
        f"S0={ability['S0']}，S1={ability['S1']}，S2={ability['S2']}（每模型一行）。S0为六维完整、有限且0–100有效；S1仅接受C2明确Epoch开权重Yes或经保守C4匹配的Yes，并排除来源冲突。Hub可用不等同于开放权重或研究授权，未单独用作S1证据。",
        '许可证只有标识符而无模型授权条款，因此按指令标license_unknown；S2空，不能声称已经完成严格许可证敏感性验证。需要后续独立许可核验后才可使用S2，不自行做法律判断。',
        '重复模型按最新Submission Date评测集合；同日六任务和Average取中位数，不挑最高分。参数、类型、许可证等关键元数据若同日冲突则保留NaN，不平均参数量。名称规范化仅大小写、空白、常见URL前缀和尾斜杠；碰撞另存，不静默合并。',
        f"每模型类型：{ability['model_type_counts']}；S1内：{ability['S1_type_counts']}。优先Type；continuously pretrained归pretrained，领域微调归chat_finetuned，base merges单列other_unknown。仅Type缺失可名称辅助。全部分类来源见model_type_classification.csv。",
        f"排行榜主时间轴Submission Date。C4使用Publication date；二者不能互换。有效日期差{ability['time_lag_n']}个，中位数{ability['time_lag_median']:.6g}天，P10/P90={ability['time_lag_P10']:.6g}/{ability['time_lag_P90']:.6g}天，负差{ability['negative_submission_release_lag']}个，全部保留审计而非自动纠正。",
        'Submission月份描述的是当前快照下提交队列的得分，未证明每个分数是在该月测得；尤其历史记录可能已回溯重评。不能把这些曲线直接视为当时可获得能力的无偏实时前沿。发布日口径后续作为敏感性，负日期差/元数据身份不明需单列。',
        '## 3 六维Benchmark与固定综合能力',table(profile[['metric','n','missing','mean','std','min','P01','median','P99','max']]),
        f"六维完整率{ability['benchmark_complete_records']/ability['raw_C1_C2_records']:.2%}；没有负值、超过100的值或常数任务，未删异常。Pearson/Spearman矩阵见benchmark_correlations.csv。",
        r'主指标$A_z=A_{equal}=\frac16\sum_j[\operatorname{clip}(x_j,P01_j,P99_j)-median_j]/IQR_j$。六个任务越大越好；固定参数由S1全观测时期一次计算，没有每月重新标准化。缺任务不填补、不与完整六维指标混合。A_z允许负数，它表示相对标定样本的标准化能力，不是负的准确率。',
        'A_100只展示：以全部S0的A_z固定P01/P99线性映射并截到0–100；所有回归响应保留A_z。全期标定不是时间外验证，后续训练/验证分割必须说明是否重新建立仅训练期标定版本。',
        f"PCA第一成分六载荷同号、解释方差{ability['PC1_explained_variance']:.2%}，满足本阶段预先配置的≥50%条件。A_pca仅敏感性，主指标保持等权。",
        table(compare[(compare.scope=='S1')&(compare.comparison!='within_month_top10')][['comparison','index_1','index_2','n','Pearson','Spearman','top10_overlap']]),
        '等权与PCA高度接近；但与Average不能笼统称“完全一致”：S1头部和月度排序差异明显。原始Average受任务原始离散尺度影响，IQR标准化改变各任务每分的相对作用，尤其低离散任务更有区分作用。保留这些差异而不是为贴近Average调权。',
        '## 4 C8完整解析与细粒度短板',
        f"遍历{c8['directories']}目录、{c8['JSON_files']}JSON：可解析{c8['JSON_parse_success']}，损坏{c8['JSON_parse_failures']}；{c8['directories_no_usable_JSON']}目录无可用JSON。按实际文件运行时间选最新可解析，最终{c8['selected_models']}模型，六维完整{c8['six_complete_models']}。所有文件、版本、时间及选择情况保存在C8_parse_audit.csv，没有抽样。",
        f"真正细粒度数据为{c8['leaf_score_records']}个叶任务指标记录、{c8['subtask_keys']}种键。BBH/MATH/GPQA/MUSR保留子任务；IFEval使用两个strict组成项。C8 JSON的MMLU-PRO只有总体结果，没有类别级得分，未伪造分类短板；详表保存实际粒度。",
        '随机基线归一化在子任务层执行并保留baseline证据，低于猜测基线取0；IFEval/MATH生成任务基线0，GPQA为四选一，MMLU-PRO基线0.1，BBH用配置选项数，MUSR依2/5/3选项数。GPQA/MATH组聚合与等权子任务均值分开保存，避免样本加权口径混用。[官方归一化说明](https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/normalization)。',
        f"C8精确规范化接受{c8['exact_matches']}个匹配；状态{c8['match_status_counts']}。模糊候选不自动接受，多重规范化身份保留ambiguous。六维完整且唯一身份可分析{c8['valid_six_complete_matched_models']}个。",
        table(read(out,'C8_six_task_comparison.csv')[['benchmark','n','MAE','systematic_bias','Pearson','Spearman']]),
        'MATH差异明显大于其他任务，不自动修正。官方2025年2月说明曾用Math-Verify重新评价历史模型；因此版本/重评差异是有依据的可能解释，但这些统计不能逐条证明差异原因。[官方重评说明](https://huggingface.co/blog/math_verify_leaderboard)。',
        r'capability_imbalance为实际叶任务0–100分数的P90−P10；另保留标准差、P10、最大最小跨度、bottom20均值。主操作定义：$A\ge P75$且$bottom20\le P25$，阈值由各匹配样本实际计算。',
        table(tails[['sample','ability_basis','n','high_average_weak_tail_count','high_average_weak_tail_rate','rate_within_high_ability','difficulty_adjusted_weak_tail_rate']]),
        '同时用C8同一运行六维得分经相同固定参数得到A_z_C8，避免把新排行榜A与旧C8短板混用。另按子任务及版本做稳健标准化控制任务难度，作为短板敏感性。任务数量不均（BBH叶任务多）和猜测基线截零均影响汇总；小比例案例说明某些局部差异存在，不能声称普遍严重掩盖，也不能从不均衡与A相关性推出因果。',
        '## 5 C3时序和C4尺度元数据',
        f"C3实际年份{ability['C3_year_min']}–{ability['C3_year_max']}；只有Year，没有月/完整日期，不制造2019年1月等时间点。原始来源包含历史论文/报告和排行榜。{int(c3.comparable_main_scale.sum())}条与C2同名记录在六任务上匹配；历史补充不直接纳入共同尺度A或月度序列。逐记录审计见C3_time_series_audit.csv。",
        f"保守C4匹配{ability['C4_matched']}个排行榜模型。已有C2元数据优先保留；C4全名或唯一完整basename须元数据/owner佐证才接受，未接受模糊猜测。参数相对差>25%标conflict，共{ability['C4_parameter_conflicts']}个，阈值为诊断约定，未自动覆盖排行榜参数。",
        table(read(out,'C4_scale_metadata_summary.csv')[['scope','field','n','available']]),
        'C4全库的三字段齐全不等于排行榜分析可用：训练数据总量混合tokens/图像等。只在notes确认token单位时记D_tokens；否则可用C6明确D_tokens_B作为有来源的辅助数据，comparability仍保留。C7 training_data_TB没有转换为token。逐字段来源、原值、日期、架构/上下文和C8指标保留在evolution_model_table.csv。',
        table(counts[['sample','n']]),
        '模型变量来源不混同：N_B取实际排行榜值，Epoch_N_B及C6_N_B并列；compute仅来自匹配C4，不以6ND填补。观察到的N/D/compute相关性见scale_input_correlations.csv，只诊断共线性，没有贡献分解。',
        '## 6 High主拟合、Medium外验的Loss桥接',
        f"C6共有{bridge['raw_C6_count']}个模型，High={bridge['High_count']}、Medium={bridge['Medium_count']}；C5仅作一致性对照，没有合并进主拟合。High全部Pythia。C6不存在实际family列，因此采用Leave-One-Model-Out；由于只有一个High族，跨族Group CV不可实施，不能将LOMO称为跨族验证。",
        '六个LB字段应用相同S1变换和1/6权重得到A_bridge，未在C6重拟合指数。候选仅B0线性、B1二次、B2单调递减isotonic。主拟合只用High；Medium只外验，不参与选模。',
        f"B0：A={bridge['candidate_models']['B0']['coefficients'][0]:.9g}+({bridge['candidate_models']['B0']['coefficients'][1]:.9g})Loss，斜率为正，理论方向未获支持。B1系数={bridge['candidate_models']['B1']['coefficients']}，仅曲率敏感性，在校准区间未保持单调下降。",
        f"数值候选中选择{bridge['selected_candidate']}；但它为常数A={bridge['selected_model']['y'][0]:.9g}，不是有辨识能力的Loss–能力桥接。状态={bridge['bridge_status']}，bridge_valid_for_conversion={bridge['bridge_valid_for_conversion']}。**本阶段没有通过验证的可用预测桥接，不能据此把前三问Loss可靠转换成Benchmark提升。**",
        table(validation[['candidate','evaluation','n','RMSE','MAE','R2','Pearson','Spearman','decreasing_full']]),
        f"所有High CV R²为负；常数单调模型的LOMO预测可与留出响应负相关，这是留出均值变化的结果，不能把Spearman=-1误读为理想Loss方向。Medium中的{bridge['Medium_outside_High_loss_range']}个Loss超出High范围；常数外验相关性未定义，保留NaN而不写0。",
        f"每个候选执行{bridge['bootstrap_repetitions']}次High模型配对bootstrap，保存参数、抽样索引、均值95%带及加中心化LOMO残差的条件预测带。只在High Loss区间{bridge['calibrated_loss_range']}生成曲线。7个同族模型的bootstrap不能证明跨族/未来预测有效，更不能用窄带掩盖桥接失败。",
        '可复用函数evolution.bridge.predict_bridge仅返回诊断值、校准范围标记和有效性状态。后续若得到可靠桥接，必须传播其不确定性；当前Benchmark历史模型仍应独立基于Leaderboard，不能由失败桥接主导。',
        '## 7 PHASE 2准备程度与停止点',
        f"已有{readiness['sample_counts']['S1_N_submission']}个S1模型可做N与Submission时间条件描述；完整N/D/compute、日期和无参数冲突的候选仅{readiness['sample_counts']['S1_joint_N_D_compute_submission_no_N_conflict']}个，其中pretrained {readiness['sample_counts']['S1_pretrained_joint']}。允许在人工确认后开展有限样本、分类型、非因果敏感性分析；尚不足以声称强证据的完整规模/非规模技术贡献率分解。",
        '主要待确认：S2许可核验缺失，规模全信息样本稀少且有选择偏差，快照分数与提交时间不等于实时前沿，发布日异常需要分层，High桥接失效。这些限制已保留，不通过插值、合并不可比Loss或模型名猜值补齐。',
        '本阶段到此停止。没有预测、没有贡献比例、没有因果结论、没有最终动力学，也没有论文全文整合；等待人工确认。',
        '## 附录：输出与复现',
        '`python -X utf8 -B scripts/20_build_evolution_ability_bridge.py --stage all`；分阶段ability/C8/bridge/report。`python -X utf8 -B scripts/20_verify_evolution_phase1.py`独立核验。目录自动创建，随机种子固定，pathlib定位根目录，前三问文件只读。',
    ]
    for p in sorted(out.glob('*.csv')):text.append(f'- [{p.name}](../evolution/phase1/{p.name})')
    for p in ['ability_transform_parameters.json','bridge_model.json','phase2_readiness.json','verification_summary.json']:
        text.append(f'- [{p}](../evolution/phase1/{p})')
    for p in read(out,'figure_manifest.csv').file:text.append(f'![{p.split("/")[-1]}](../{p.removeprefix("outputs/")})')
    (root/'outputs/report/question4_phase1_data_ability_bridge.md').write_text('\n\n'.join(text)+'\n',encoding='utf8')


def finalize(root,cfg):
    """Refresh only this run's cached C8 analysis; write final figures and report."""
    out=root/'outputs/evolution/phase1'
    from .detailed import refresh_weak_tail
    refresh_weak_tail(root,cfg);models=read(out,'evolution_model_table.csv')
    readiness=scale_readiness(root,cfg,models);figures(root,cfg,models)
    base=root/'real_attachments/C_efficiency_evolution';sources=[base/p for p in FILES.values()]+[base/'data/train-00000-of-00001.parquet']
    sources+=sorted((base/'detailed_results').rglob('*.json'))
    rows=[]
    for p in sources:
        with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        rows.append(dict(file=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest))
    csv(out/'input_manifest.csv',rows)
    implementation=[root/'configs/evolution_phase1.yaml',root/'scripts/20_build_evolution_ability_bridge.py',root/'scripts/20_verify_evolution_phase1.py']+sorted((root/'src/evolution').glob('*.py'))
    rows=[]
    for p in implementation:
        with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        rows.append(dict(file=p.relative_to(root).as_posix(),sha256=digest))
    csv(out/'implementation_manifest.csv',rows)
    stage_report(root,cfg,models,readiness)
    summary=dict(phase='Question 4 PHASE 1',old_question4_results_reused=False,previous_questions_refitted=False,
        future_prediction_performed=False,contribution_decomposition_performed=False,causal_claims=False,waiting_for_human_confirmation=True,
        figures=12,main_ability='A_z=A_equal',strict_license_status='unconfirmed',readiness=readiness)
    dump(out/'execution_summary.json',summary);logging.info('Question 4 PHASE 1 report complete; awaiting human confirmation')
