"""Plots and a factual Chinese report for the frozen-input contribution study."""
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import csv,table
from .contribution_analysis import input_manifest,write_json


def read(out,name):
    """Preserve full round-trip precision in numerical outputs."""
    return pd.read_csv(out/name,float_precision='round_trip')


def make_figures(root,models):
    """Produce descriptive and diagnostic figures; never project future time."""
    out=root/'outputs/evolution/phase2';figdir=root/'outputs/figures/question4_phase2';figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':160})
    manifest=[]
    def save(fig,name,source):
        fig.tight_layout();p=figdir/name;fig.savefig(p,bbox_inches='tight');plt.close(fig)
        manifest.append(dict(file=p.relative_to(root).as_posix(),source=source,bytes=p.stat().st_size))
    m=models['M1_linear'];g=m['data'];b=np.asarray(m['beta'])
    fig,ax=plt.subplots(figsize=(8,5));sc=ax.scatter(g.lnN,g.A_z,c=g.t_month,cmap='viridis',s=35,alpha=.8)
    fig.colorbar(sc,ax=ax,label='Months since main submission origin');ax.set(xlabel='ln(parameters in billions)',ylabel='Frozen A_z',title=f'S1 pretrained: ability and parameter scale (n={len(g)})')
    save(fig,'01_ability_vs_parameter_scale.png','analysis_model_table.csv')
    residual=g.A_z-(b[0]+b[1]*g.lnN);grid=np.linspace(g.t_month.min(),g.t_month.max(),100)
    fig,ax=plt.subplots(figsize=(8,5));ax.scatter(g.t_month,residual,s=35,alpha=.7,label='Individual models')
    ax.plot(grid,b[2]*grid,color='tab:red',label='Joint-model time component');ax.set(xlabel='Submission time (months)',ylabel='TechResidual',title='Ability after removing fitted parameter-scale component');ax.legend()
    save(fig,'02_technology_residual_vs_time.png','non_scale_technology_index.csv')
    fig,ax=plt.subplots(figsize=(8,5))
    for label,color in [('M1_linear','tab:blue'),('M1_chat','tab:orange')]:
        h=models[label]['data'];beta=np.asarray(models[label]['beta']);x=np.linspace(h.t_month.min(),h.t_month.max(),100)
        y=beta[0]+beta[1]*h.lnN.median()+beta[2]*x
        ax.plot(x,y,color=color,label=f'{label}, tau={beta[2]:.4f}/month')
    ax.set(xlabel='Submission time (months)',ylabel='Fitted A_z at each group median lnN',title='Separate pretrained and chat trends; descriptive comparison');ax.legend()
    save(fig,'03_pretrained_chat_time_trends.png','model_type_comparison.csv')
    preds=read(out,'model_predictions.csv');perf=read(out,'model_performance.csv')
    for label,num in [('M1_linear','04'),('M2_C','05')]:
        h=preds[(preds.model==label)&preds.method.eq('OLS')];cv=perf[(perf.model==label)&perf.evaluation.eq('LOOCV')].iloc[0]
        fig,ax=plt.subplots(figsize=(6,5));ax.scatter(h.observed_A,h.fitted_A,alpha=.7,label='In-sample fit')
        ax.scatter(h.observed_A,h.LOOCV_predicted_A,marker='x',color='tab:orange',label='LOOCV')
        vals=np.r_[h.observed_A,h.fitted_A,h.LOOCV_predicted_A];lo,hi=np.nanmin(vals),np.nanmax(vals);ax.plot([lo,hi],[lo,hi],'--',color='gray')
        ax.set(xlabel='Observed frozen A_z',ylabel='Predicted A_z',title=f'{label}: n={len(h)}, LOOCV RMSE={cv.RMSE:.3f}, R2={cv.R2:.3f}');ax.legend()
        save(fig,num+'_observed_vs_predicted.png','model_predictions.csv')
    dec=read(out,'contribution_decomposition.csv');point=dec[dec.series.isin(['R1','R2','R3'])&np.isclose(dec.window_fraction,1/3)]
    fig,axes=plt.subplots(1,3,figsize=(13,4.8))
    for ax,(_,r) in zip(axes,point.iterrows()):
        values=[r.A_early_pred,r.Delta_scale,r.Delta_tech,r.A_late_pred];bottom=[0,r.A_early_pred,r.A_early_pred+r.Delta_scale,0]
        colors=['gray','tab:blue','tab:orange','gray'];ax.bar(range(4),values,bottom=bottom,color=colors,alpha=.8)
        for j,v in enumerate(values):ax.annotate(f'{v:+.3f}',(j,bottom[j]+v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
        ax.axhline(0,color='gray',lw=.7);ax.set_xticks(range(4),['Early','Scale','Time','Late']);ax.set(title=r.series+' (signed components)',ylabel='A_z / change in A_z')
    fig.suptitle('Early to late: predicted endpoint-state change, not observed causal attribution',y=1.03)
    save(fig,'06_endpoint_shapley_waterfall.png','contribution_decomposition.csv')
    boot=read(out,'contribution_bootstrap.csv');fig,axes=plt.subplots(2,3,figsize=(13,8.5))
    for j,s in enumerate(['R1','R2','R3']):
        ax=axes[0,j]
        h=boot[(boot.series==s)&np.isclose(boot.window_fraction,1/3)&boot.status.eq('OK')]
        vals=np.r_[h.signed_ratio_scale.dropna(),h.signed_ratio_tech.dropna()];lo,hi=np.quantile(vals,[.01,.99])
        if hi<=lo:lo-=.1;hi+=.1
        for field,label,color in [('signed_ratio_scale','Scale','tab:blue'),('signed_ratio_tech','Time','tab:orange')]:
            v=h[field].dropna();ax.hist(v,bins=np.linspace(lo,hi,45),alpha=.5,label=label,color=color)
        ax.axvline(0,color='gray',lw=.8);ax.axvline(1,color='gray',lw=.8,ls='--')
        ax.set(xlabel='Signed component / predicted change',ylabel='Bootstrap draws',title=s+'; plot P01-P99 only');ax.legend()
        bottom=axes[1,j]
        for field,label,color in [('absolute_share_scale','Scale','tab:blue'),('absolute_share_tech','Time','tab:orange')]:
            bottom.hist(h[field].dropna(),bins=np.linspace(0,1,35),alpha=.5,label=label,color=color)
        bottom.set(xlabel='Absolute component / sum of absolute components',ylabel='Bootstrap draws',title=s+' absolute shares; opposing signs allowed');bottom.legend()
    save(fig,'07_bootstrap_contribution_ratios.png','contribution_bootstrap.csv; full tails retained in CSV')
    frontier=read(out,'monthly_ability_frontier.csv');x=np.arange(len(frontier));fig,ax=plt.subplots(figsize=(9,5))
    for col,label in [('P90_A','P90'),('P95_A','P95'),('top3_mean_A','Top3 mean (up to 3)')]:ax.plot(x,frontier[col],'-o',label=label)
    ax.set_xticks(x,frontier.month,rotation=45,ha='right');ax.set(ylabel='Frozen A_z',title='S1 pretrained submission cohorts: descriptive ability frontier')
    for i,r in frontier.iterrows():ax.annotate(f'n={r.n}',(i,r.P95_A),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    ax.legend(loc='upper right');save(fig,'08_monthly_ability_frontier.png','monthly_ability_frontier.csv')
    tech=read(out,'monthly_technology_frontier.csv');h=tech[tech.model=='M1_linear'];x=np.arange(len(h));fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(x,h.median_TechResidual,'-o',label='P50');ax.plot(x,h.P90,'-o',label='P90')
    ax.set_xticks(x,h.month,rotation=45,ha='right');ax.set(ylabel='TechResidual',title='Monthly parameter-adjusted ability; sparse months remain visible');ax.legend()
    save(fig,'09_monthly_technology_index.png','monthly_technology_frontier.csv')
    time=read(out,'time_axis_sensitivity.csv');h=time[np.isclose(time.window_fraction,1/3)];x=np.arange(len(h));fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(x-.18,h.submission_absolute_share_scale,width=.36,label='Submission');ax.bar(x+.18,h.release_absolute_share_scale,width=.36,label='Release')
    ax.set_xticks(x,h.series);ax.set(ylim=(0,1.15),ylabel='Absolute scale share (opposing signs allowed)',title='Same-model time-axis sensitivity; endpoint chronology differs');ax.legend()
    for j,r in enumerate(h.itertuples()):ax.text(j,1.04,'sensitive' if r.time_axis_sensitive else 'within threshold',ha='center',fontsize=9)
    save(fig,'10_submission_release_sensitivity.png','time_axis_sensitivity.csv')
    h=preds[(preds.model=='M1_linear')&preds.method.eq('OLS')];fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    axes[0].scatter(h.fitted_A,h.residual,alpha=.7);axes[0].axhline(0,color='gray',ls='--');axes[0].set(xlabel='Fitted A_z',ylabel='Residual',title='M1 residual pattern')
    stats.probplot(h.residual,dist='norm',plot=axes[1]);axes[1].set_title('M1 residual normal Q-Q (diagnostic)')
    save(fig,'11_M1_residual_diagnostics.png','model_predictions.csv')
    influence=read(out,'influence_diagnostics.csv');h=influence[influence.model.eq('M1_linear')].sort_values(['Cooks_distance','Model'],ascending=[False,True]);x=np.arange(len(h));fig,ax=plt.subplots(figsize=(9,5))
    ax.stem(x,h.Cooks_distance,basefmt=' ');ax.axhline(4/len(h),ls='--',color='tab:red',label='4/n diagnostic threshold')
    for j,r in enumerate(h.head(2).itertuples()):ax.annotate(r.Model.split('/')[-1],(j,r.Cooks_distance),xytext=(8,2-j*15),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Models ordered by Cook distance',ylabel="Cook's distance",title='Influence diagnosis; main fit retains all models');ax.legend()
    save(fig,'12_M1_influence_diagnostics.png','influence_diagnostics.csv');csv(out/'figure_manifest.csv',manifest)
    return manifest


def finalize(root,cfg,models,d,metadata):
    """Report both positive associations and failed/stable sensitivity evidence."""
    out=root/'outputs/evolution/phase2';report=root/'outputs/report/question4_phase2_contribution_decomposition.md'
    figures=make_figures(root,models)
    coeff=read(out,'model_coefficients.csv');perf=read(out,'model_performance.csv');dec=read(out,'contribution_decomposition.csv')
    tiers=read(out,'contribution_sample_tiers.csv');boot=read(out,'contribution_bootstrap_summary.csv');cb=read(out,'coefficient_bootstrap_summary.csv')
    aux=read(out,'technology_auxiliary_correlations.csv');vif=read(out,'VIF_diagnostics.csv');diag=read(out,'regression_diagnostics.csv')
    time=read(out,'time_axis_sensitivity.csv');influence=read(out,'influence_diagnostics.csv');folds=read(out,'LOOCV_coefficients.csv')
    bridge=json.loads((root/'outputs/evolution/phase1/bridge_model.json').read_text(encoding='utf8'))
    if bridge['bridge_valid_for_conversion']:raise ValueError('Frozen negative bridge evidence changed unexpectedly')
    bridge_metrics=pd.read_csv(root/'outputs/evolution/phase1/bridge_validation_metrics.csv')
    bridge_text='\n\n'.join(['# Loss–Benchmark负面证据（PHASE 1冻结）',
        f"High仅{bridge['High_count']}条、均为Pythia；主拟合只用High，Medium={bridge['Medium_count']}仅作外验。",
        table(bridge_metrics[['candidate','evaluation','n','RMSE','MAE','R2','Pearson','Spearman']]),
        f"单调模型退化常数A={bridge['selected_model']['y'][0]:.12g}；三个候选High CV R²均为负，Medium外验失败。",
        '`bridge_valid_for_conversion=False`保持冻结。问题一至三Loss不批量转换为A_z，不参与本阶段贡献计算。',
        '本阶段不重新开发桥接、不做未来预测。后续Benchmark前沿需独立基于可比排行榜记录；前三问只提供资源/算力情景。'])
    (out/'bridge_negative_evidence.md').write_text(bridge_text,encoding='utf8')
    primary=dec[dec.series.isin(['R1','R2','R3'])&np.isclose(dec.window_fraction,1/3)]
    maincoef=coeff[coeff.model.isin(['M1_linear','M2_C','M3_ND'])&coeff.inference.eq('HC3')]
    mainperf=perf[perf.model.isin(['M1_linear','M2_C','M3_ND'])]
    meaningful=['Delta_scale','Delta_tech','signed_ratio_scale','signed_ratio_tech','absolute_share_scale','absolute_share_tech']
    mainboot=boot[boot.series.isin(['R1','R2','R3'])&np.isclose(boot.window_fraction,1/3)&boot.quantity.isin(meaningful)]
    comparison=coeff[coeff.model.isin(['M1_linear','M1_Huber','M1_chat','M1_S0_sensitivity','M1_drop_Cook_top1','M1_drop_Cook_top2','M1_drop_largest_organization','M1_organization_known'])&coeff.term.eq('t_month')&coeff.inference.isin(['HC3','Huber_H3','organization_clustered'])]
    tc=maincoef[(maincoef.model=='M1_linear')&(maincoef.term=='t_month')].iloc[0]
    org=coeff[(coeff.inference=='organization_clustered')&(coeff.term=='t_month')].iloc[0]
    S0=coeff[(coeff.model=='M1_S0_sensitivity')&(coeff.inference=='HC3')&(coeff.term=='t_month')].iloc[0]
    deleted=coeff[(coeff.model=='M1_drop_largest_organization')&(coeff.inference=='HC3')&(coeff.term=='t_month')].iloc[0]
    tech_boot=cb[(cb.model=='M1_linear')&(cb.term=='t_month')].iloc[0]
    conclusion=(f"M1的控制参数时间项tau={tc.coef:.6g}/月，HC3 p={tc.p_value:.6g}，bootstrap 95%区间[{tech_boot.CI_low:.6g},{tech_boot.CI_high:.6g}]。"
        f"但组织聚类p={org.p_value:.6g}，去除最大组织p={deleted.p_value:.6g}，S0敏感性p={S0.p_value:.6g}。"
        '因此能支持特定S1样本内的条件时间关联；证据具有样本选择和组织依赖，不能宣称普遍稳定或严格因果的非规模技术进步。')
    readiness=dict(main_M1_models=models['M1_linear']['n'],restricted_M2_M3_models=models['M2_C']['n'],
        conditional_time_association_HC3_positive_and_significant=bool(tc.coef>0 and tc.p_value<.05),
        robust_general_non_scale_advance_established=False,
        organization_cluster_time_p=org.p_value,S0_time_p=S0.p_value,
        phase3_started=False,future_prediction_performed=False,loss_bridge_used=False,
        phase3_readiness='conditional descriptive benchmark frontier sensitivity only after human confirmation; no reliable 12/24 month extrapolation established',
        blockers=['ten submission cohorts, several singleton months','snapshot evaluation versus actual historical frontier',
            'sample/organization dependence of time trend','small complete-scale sample with only two late-window models',
            'Release changes ordering and endpoints','frozen Loss bridge unusable'])
    write_json(out/'phase3_readiness.json',readiness)
    selection=json.loads((out/'model_selection.json').read_text(encoding='utf8'))
    monthly=read(out,'monthly_ability_frontier.csv');status=read(out,'bootstrap_status_audit.csv')
    signatures=[]
    for label in ['M1_linear','M2_C','M3_ND']:
        h=folds[(folds.model==label)&folds.method.eq('OLS')&folds.full_rank]
        names=models[label]['names'];arr=np.array([json.loads(v) for v in h.parameters])
        for j,name in enumerate(names):signatures.append(dict(model=label,term=name,folds=len(h),positive_fraction=float((arr[:,j]>0).mean()),
            negative_fraction=float((arr[:,j]<0).mean()),min_coef=arr[:,j].min(),max_coef=arr[:,j].max()))
    csv(out/'LOOCV_sign_stability.csv',signatures)
    sections=['# 问题四 PHASE 2：规模扩张与非规模技术进步贡献分解',
        '本阶段读取已人工冻结的PHASE 1能力、模型身份、分类和尺度元数据。A_z和主S1定义不变；没有重拟合前三问、没有重构能力指数、没有Loss数值桥接、没有未来12/24个月预测。',
        '## 1 样本、去重与时间口径',
        '主分析为confirmed open-weight S1中的pretrained：后训练/对齐混杂较少，但仍有模型族、训练数据、架构和评测选择等混杂。chat_finetuned单独作为敏感性，other_unknown不进入主贡献比例。S0仅检查开放权重确认/匹配缺失的样本选择；S2授权证据仍未建立。',
        table(tiers[tiers['sample']=='S1'][['model_type','tier','n','parameter_conflict_n','release_date_n']]),
        '每个Model一行，完全沿用冻结的最新Submission Date集合及同日确定性规则。逐模型重新对照C2核验；未择优选最高A。dedup_audit.csv保留所有选中与旧记录行号，同日重复评测采用PHASE 1已冻结的逐Benchmark中位数。',
        f"主时间起点{metadata['primary_origin']}、末日{metadata['primary_end']}；t_month=实际经过天数/{metadata['days_per_month']}，连续月数，不用整数年份。Release以受限样本的实际发布日计算，另作敏感性。",
        '按用户本阶段T0–T3定义统计，T3为21，并非预设20。保留1条PHASE 1参数冲突记录；主结果不自动删除，另列无冲突20模型敏感性。缺N/D/compute不插值，compute只使用公开C4字段，不由6ND或前三问反算。',
        '## 2 模型、估计及线性时间选择',
        r'$M1:A=a+\beta_N\ln N+\tau t+\epsilon$；$M2\text{-}C:A=a+\beta_C\ln C_{train}+\tau_Ct+\epsilon$；$M3:A=a+\beta_N\ln N+\beta_D\ln D+\tau_{ND}t+\epsilon$。M2/M3在同一T3样本上拟合，不同时加入lnC、lnN和lnD。',
        '时间项统一称“控制规模后的非规模技术进步代理项”，不是可识别的因果技术效应。lnN是参数规模，lnC是整体训练规模代理，lnN+lnD是结构敏感性；R1/R2/R3不能合并成一个贡献数字。尤其M1只控制参数量，时间项仍可能吸收未观察的训练数据量/训练算力扩张，不能称为排除全部规模因素后的纯技术效应。',
        table(maincoef[['model','term','n','coef','SE','CI_low','CI_high','p_value']]),
        'HC3主诊断使用t参考分布；OLS常规SE/CI/p亦完整保存。HuberT为稳健敏感性。实现依据[statsmodels稳健协方差文档](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html)及[Huber示例](https://www.statsmodels.org/stable/examples/notebooks/generated/robust_models_0.html)。',
        table(mainperf[['model','evaluation','n','R2','adjusted_R2','RMSE','MAE','Pearson','Spearman']]),
        table(perf[perf.model.isin(['M1_linear','M1_quadratic','M1_month_FE'])][['model','evaluation','n','RMSE','AICc','BIC','unidentifiable_folds']]),
        selection['reason']+'。月固定效应有4个单样本月份，留出该模型即无法估计该月效应；没有把最小范数外推或局部62折成绩冒充完整LOOCV。HC3因杠杆值1而退化，SE/CI/p均明确记NaN及UNIDENTIFIABLE_LEVERAGE_ONE，不能使用库产生的inf标准误及p=1进行推断。',
        '## 3 稳健性、共线性与影响点',
        table(diag[diag.model.isin(['M1_linear','M2_C','M3_ND'])][['model','condition_number_raw','condition_number_standardized','Koenker_BP_p','Jarque_Bera','JB_p','Cook_exceed_count','max_Cook_Model']]),
        table(vif[vif.model.isin(['M1_linear','M2_C','M3_ND'])][['model','term','VIF']]),
        '报告原始与中心化尺度一致后的condition number，避免把lnD/lnC单位导致的大截距误判为强共线性。M3本次VIF未达到配置警戒值，但小样本、bootstrap和留一系数符号不稳仍会限制贡献解释。Cook>4/n仅诊断；主模型保留全部记录，删除最高1/2条只是规定敏感性。',
        table(influence[influence.model=='M1_linear'].sort_values('Cooks_distance',ascending=False).head(5)[['Model','Cooks_distance','leverage','residual']]),
        table(comparison[['model','inference','n','coef','CI_low','CI_high','p_value']]),
        table(cb[cb.model.isin(['M1_linear','M2_C','M3_ND'])][['model','term','valid_draws','CI_low','CI_high','positive_fraction']]),
        table(pd.DataFrame(signatures)),
        '组织按附件实际完整组织/合作团体字符串聚类，不按模型名猜实验室；多组织合作团体可能共享成员，聚类独立性仍有限。没有加入大量组织虚拟变量。S1主样本65条具有组织记录，另1条未知保留在主拟合。',
        '## 4 Early/Late状态与Shapley贡献',
        '主窗口为主Submission日历区间前1/3与后1/3；敏感性为前25%与后25%。中段参与拟合；N状态取median lnN，D/C按要求先取原始token/FLOP的中位数再取对数，t取中位数，不按模型个数三等分。原始N/D/C中位数也保存在端点审计。此为典型状态预测变化，不等于观察到的中位能力变化或最大能力前沿变化。',
        r'$\Delta_{scale}=\tfrac12[f(S_1,t_0)-f(S_0,t_0)+f(S_1,t_1)-f(S_0,t_1)]$；$\Delta_{tech}=\tfrac12[f(S_0,t_1)-f(S_0,t_0)+f(S_1,t_1)-f(S_1,t_0)]$。所有可识别计算验证$|\Delta_{pred}-\Delta_{scale}-\Delta_{tech}|<10^{-10}$。加性线性模型没有交互，Shapley退化为相应系数乘状态差；公式仍完整执行。',
        table(primary[['series','n','early_n','late_n','Delta_scale','Delta_tech','Delta_pred','share_scale','share_tech','absolute_share_scale','absolute_share_tech','cancellation_effect']]),
        table(dec[dec.series.isin(['R1','R2','R3'])&np.isclose(dec.window_fraction,.25)][['series','early_n','late_n','Delta_scale','Delta_tech','Delta_pred','share_scale','absolute_share_scale']]),
        '只有预测增长>0且两项均非负时share_scale/share_tech才是普通0–100%增长比例。若一项负值，普通share留NaN，保留signed contribution、signed_ratio和absolute share，并标明抵消效应。signed_ratio的区间可以越出0/1；它不是合法非负贡献百分比。没有截断负贡献。',
        'T3晚期仅2个模型，都是2023年发布、2025年提交的Pythia。数据上“晚提交”不等于“新技术”；两个端点构成对时间口径和典型状态尤其敏感。R2/R3均为restricted-sample evidence，不是全体开源模型的比例。',
        '## 5 模型级bootstrap及贡献区间',
        f"每套方案执行{cfg['bootstrap_repetitions']}次模型级有放回抽样，每次重新拟合、重新计算抽中模型的端点中位数并重新Shapley；固定原始日历边界。主R1/R2/R3、chat、S0和三套Release敏感性均执行，固定随机种子。",
        table(mainboot[['series','quantity','draws','valid_decomposition_draws','valid_quantity_draws','CI_low','CI_high','ratio_CI_outside_zero_one']]),
        table(status[status.model.isin(['M1_linear','M2_C','M3_ND'])]),
        '当重抽样导致窗口为空或设计矩阵降秩，保留该次抽样及状态，不补模型、不悄悄重抽直到成功。贡献CI以可识别draw计算，明确是条件区间；普通非负share的CI还额外条件于非负增长。完整signed_ratio区间用于暴露抵消及分母接近零的不稳定性。模型级bootstrap不能处理族内依赖，组织敏感性另行保留。',
        '## 6 模型类型、S0与时间口径敏感性',
        table(read(out,'model_type_comparison.csv')),
        'pretrained与chat的tau差只描述分组趋势强弱，不等于某种对齐方法的因果效应。S0扩大样本改变tau，说明confirmed-open匹配/元数据选择不可忽略。',
        table(time),
        f"time_axis_sensitive当贡献符号反转，或absolute scale share差>{cfg['time_axis_absolute_share_difference_warning']:.0%}时标记；该阈值是透明诊断约定，不是统计显著性检验。Release只纳入PHASE 1接受的C4身份且有发布日期的模型；Submission对照使用完全相同的模型，模型数及端点变化完整列出。实际时间范围和排序改变，因而窗口模型组成也改变。",
        '## 7 技术残差、月度前沿、C8及架构线索',
        r'$TechResidual=A-[\hat a+\hat\beta_N\ln N]$。按月保存P50/P75/P90/P95并做时间描述；辅助残差~time回归使用生成响应，标准误不作为独立证据，主要推断仍来自联合M1的HC3/bootstrap。',
        table(monthly[['month','n','P90_A','P95_A','top3_mean_A','top3_actual_n','median_lnN','top3_median_lnN']]),
        '不足3模型的月份top3_mean只取实际可用模型并明确实际n。月度P90/P95明显起伏，未呈稳定单调提升，不能将个体回归的正时间项直接等同于前沿持续改善。多个月份n=1，无法从10个队列前沿点建立稳定因果贡献或长期预测。当前快照可能回溯重评，Submission cohort不是已证实的当时能力前沿。',
        table(aux[['model','analysis','variable','n','Pearson','Pearson_p_BH_q','Spearman','Spearman_p_BH_q']]),
        '原始C8偏科同时受任务难度、任务数量及测评版本影响，补充difficulty-adjusted与same-run对照。相关弱也不能证明偏科没有作用；若相关强，技术残差不能排除任务结构影响。C7只用单变量架构相关；未建立多变量架构回归，BH q只作探索性多重检验提示。',
        '## 8 结论、负面桥接与停止点',conclusion,
        'C6桥接负面证据单独保存bridge_negative_evidence.md：High仅7条同族，CV R²均负，单调退化常数，Medium外验失败。固定bridge_valid_for_conversion=False，前三问Loss不转换为A_z、不参与贡献比例。',
        'PHASE 3仅具备在人工确认后开展受限、描述性Benchmark前沿敏感性分析的输入准备；未建立可靠12/24个月外推条件。需要明确历史快照/真实评测时点、稀疏月份、族/组织依赖和可比数据支持。到此停止，不拟合最终前沿、不预测未来、不做因果结论。',
        '## 附录：复现与证据文件',
        '`python -X utf8 -B scripts/21_build_evolution_contribution.py --stage all`；支持samples/models/bootstrap/report分阶段。`python -X utf8 -B scripts/21_verify_evolution_contribution.py`独立复核。新配置、代码和输出均独立，冻结文件按SHA256核验。']
    for p in sorted(out.glob('*.csv')):sections.append(f'[{p.name}](../evolution/phase2/{p.name})')
    sections.append('[phase3_readiness.json](../evolution/phase2/phase3_readiness.json)')
    sections.append('[bridge_negative_evidence.md](../evolution/phase2/bridge_negative_evidence.md)')
    sections.append('[verification_summary.json](../evolution/phase2/verification_summary.json)')
    for r in figures:sections.append(f"![{Path(r['file']).name}](../figures/question4_phase2/{Path(r['file']).name})")
    report.write_text('\n\n'.join(sections),encoding='utf8')
    csv(out/'input_manifest.csv',input_manifest(root))
    code=[root/'configs/evolution_phase2.yaml',root/'scripts/21_build_evolution_contribution.py',root/'scripts/21_verify_evolution_contribution.py']
    code+=sorted((root/'src/evolution').glob('contribution_*.py'))
    csv(out/'implementation_manifest.csv',[dict(file=p.relative_to(root).as_posix(),size=p.stat().st_size,
        sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in code if p.is_file()])
    write_json(out/'execution_summary.json',dict(phase='Question 4 PHASE 2',main_model='M1_linear',
        bootstrap_replicates_per_model=cfg['bootstrap_repetitions'],figures=len(figures),
        frozen_ability_refitted=False,previous_questions_modified=False,bridge_refitted=False,loss_conversion_performed=False,
        future_prediction_performed=False,final_frontier_fitted=False,causal_conclusion=False,waiting_for_human_confirmation=True,
        readiness=readiness))
    logging.info('PHASE 2 report and %d figures complete; awaiting human confirmation',len(figures))
