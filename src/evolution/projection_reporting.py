"""Readable final evidence report and scientific plots, keeping scenario warnings visible."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .common import csv,table
from .projection_models import read_csv,frozen_m2


def build_figures(root,cfg,d):
    """At least ten modest static figures; raw A_z remains unclipped everywhere."""
    out=root/'outputs/evolution/phase3';folder=root/'outputs/figures/question4_phase3';folder.mkdir(parents=True,exist_ok=True)
    s=read_csv(out/'future_frontier_scenarios.csv');a=read_csv(out/'frontier_anchor.csv');hist=read_csv(out/'historical_benchmark_frontier.csv');hist['date']=pd.to_datetime(hist.date)
    g=read_csv(out/'compute_growth_estimation.csv');ch=read_csv(out/'compute_frontier_history.csv');ch['date']=pd.to_datetime(ch.date)
    draws=read_csv(out/'mc_joint_draws.csv');base=pd.Timestamp(s.base_date.iloc[0]);manifest=[]
    main=s[s.model_kind.eq('frozen_M2_increment')&s.frontier_definition.eq('P95')&s.compute_scenario.eq('C1')&s.technology_scenario.eq('T1')].sort_values('horizon_months')
    pre=s[s.model_kind.eq('frozen_M2_increment')&s.frontier_definition.eq('P95')];names={'F0_pretrained':'Pretrained','F1_chat':'Chat / finetuned','F2_overall':'All S1'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120})
    def save(fig,name,note):
        fig.tight_layout();path=folder/name;fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig);manifest.append(dict(file=str(path.relative_to(root)).replace('\\','/'),purpose=note))
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    actual=hist[hist.model_scope.eq('F0_pretrained')];anchor=main.A_z_anchor.iloc[0];h=np.arange(25);dates=[base+pd.DateOffset(months=int(x)) for x in h]
    y=draws.F0_pretrained_anchor_P95.to_numpy()[:,None]+(draws.beta_C.to_numpy()*draws.g_C.to_numpy()*.5+draws.tau_C.to_numpy()*.5)[:,None]*h
    low,med,high=np.quantile(y,[.025,.5,.975],axis=0);point=anchor+(main.Delta_scale.iloc[0]+main.Delta_non_scale.iloc[0])/12*h
    for ax in axes:
        ax.plot(actual.date,actual.P95,'o-',color='#333333',label='Observed 90-day P95')
        ax.scatter([base],[anchor],marker='D',color='#157a6e',label='Current 180-day P95 (n=18)',zorder=5)
        ax.plot(dates,point,'--',color='#157a6e',label='C1 x T1 plug-in scenario')
        ax.plot(dates,med,color='#3573b9',label='Monte Carlo median')
        ax.fill_between(dates,low,high,color='#3573b9',alpha=.2,label='Parameter/data P2.5-P97.5')
        ax.axvline(base,color='#777777',lw=.8);ax.set_ylabel('Frozen ability index A_z (unclipped)');ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));ax.grid(alpha=.15)
    axes[0].axhline(main.operational_fixed_transform_upper.iloc[0],color='#b54b4b',ls=':',label='Fixed transform attainable upper bound')
    axes[0].set_title('Pretrained frontier: conditional projection');axes[0].legend(fontsize=8,loc='upper left')
    expansion=read_csv(out/'projection_interval_expansion.csv');e=expansion[expansion.model_scope.eq('F0_pretrained')].iloc[0];buffer=1.96*e.one_step_RMSE*np.sqrt(h)+e.structure_shift
    buffer[0]=0;axes[1].fill_between(dates,low-buffer,high+buffer,color='#d78731',alpha=.17,label='Uncalibrated hindcast/structure stress envelope')
    axes[1].set_title('Risk expansion: not a 95% probability interval');axes[1].legend(fontsize=8,loc='upper left')
    save(fig,'01_frontier_projection_fan.png','Frozen historical 90-day series, current count-selected180-day anchor, raw parameter fan and separate risk envelope')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,horizon in zip(axes,[12,24]):
        z=pre[pre.technology_scenario.eq('T1')&pre.horizon_months.eq(horizon)].sort_values('compute_scenario');pos=np.arange(4)
        ax.errorbar(pos,z.A_z_median,yerr=[z.A_z_median-z['A_z_P2.5'],z['A_z_P97.5']-z.A_z_median],fmt='o',capsize=4,label='MC median and parameter interval')
        ax.scatter(pos,z.A_z_point,marker='D',color='#157a6e',label='Plug-in scenario');ax.set_xticks(pos,z.compute_scenario);ax.set_ylabel('A_z');ax.set_title(f'{horizon} months, T1 fixed');ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);save(fig,'02_compute_slowdown_effect.png','Compute slowdown paired scenarios; unclipped A_z')
    fig,ax=plt.subplots(figsize=(7.5,4));pos=np.arange(2);ax.bar(pos,main.Delta_scale,label='Compute-scale increment');ax.bar(pos,main.Delta_non_scale,bottom=main.Delta_scale,label='Non-scale time-proxy increment');ax.set_xticks(pos,['12 months','24 months']);ax.set_ylabel('Signed delta A_z');ax.set_title('C1 x T1 future increment decomposition (scenario)');ax.legend();save(fig,'03_future_increment_decomposition.png','Signed point increments; not historical absolute shares')
    for horizon in [12,24]:
        mat=pre[pre.horizon_months.eq(horizon)].pivot(index='compute_scenario',columns='technology_scenario',values='A_z_point').reindex(index=['C0','C1','C2','C3'],columns=['T0','T1','T2'])
        fig,ax=plt.subplots(figsize=(6.5,4.5));im=ax.imshow(mat,cmap='viridis',aspect='auto');ax.set_xticks(range(3),mat.columns);ax.set_yticks(range(4),mat.index)
        for i in range(4):
            for j in range(3):ax.text(j,i,f'{mat.iloc[i,j]:.3f}',ha='center',va='center',color='white' if mat.iloc[i,j]<np.nanmedian(mat) else 'black')
        ax.set_xlabel('Technology factor: 0 / 0.5 / 1');ax.set_ylabel('Compute factor: 1 / 0.5 / 0.25 / 0');ax.set_title(f'{horizon}-month P95 plug-in scenarios (A_z)');fig.colorbar(im,ax=ax,label='Unclipped A_z');save(fig,f'0{4 if horizon==12 else 5}_scenario_matrix_{horizon}m.png','24 assumptions across two horizons; full numeric intervals in CSV')
    growth=json.loads((out/'compute_growth_model.json').read_text(encoding='utf8'));fig,ax=plt.subplots(figsize=(10,4.5));audit=read_csv(out/'compute_model_audit.csv');audit=audit[audit.valid];ax.scatter(pd.to_datetime(audit.date),audit.lnC,alpha=.35,label='Unique C4 training projects')
    for col in ['P90_lnC','P95_lnC','Top3_lnC']:ax.plot(ch.date,ch[col],label='6-month '+col)
    last=growth['last_frontier'];origin=growth['last_frontier_date'];future=[base+pd.DateOffset(months=x) for x in range(25)]
    for c,factor in cfg['compute_scenario_factors'].items():ax.plot(future,last+growth['main_growth']*factor*np.arange(25),ls='--',label=f'{c}: {factor:g} x historical growth')
    ax.axvline(base,color='grey',ls=':');ax.set_ylabel('Natural log training compute (FLOP)');ax.set_title('C4 compute frontier: empirical rolling windows and scenario growth');ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.15);save(fig,'06_compute_frontier_history_projection.png','Only dated unique C4 accepted open-weight pretrained mappings, no future compute data')
    hp=read_csv(out/'hindcast_predictions.csv');fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,scope in zip(axes,names):
        z=hp[hp.model_scope.eq(scope)];first=z[z.method.eq('Benchmark_linear')];ax.plot(pd.to_datetime(first.target_date),first.observed,'o-',color='#333333',label='Observed')
        for method,gp in z.groupby('method'):ax.plot(pd.to_datetime(gp.target_date),gp.predicted,'o--',label='Mechanism (frozen)' if 'mechanism' in method else 'Benchmark linear')
        ax.set_title(names[scope]);ax.set_ylabel('A_z');ax.set_xticks(pd.to_datetime(first.target_date));ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'));ax.tick_params(axis='x',labelsize=8);ax.grid(alpha=.15)
    axes[0].legend(fontsize=8);fig.suptitle('Conditional rolling-origin hindcast: fixed full-period calibration');save(fig,'07_hindcast_observed_predicted.png','Four forecasts per scope, not strictly temporal OOS; frozen M2 future-model warning')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for scope in names:
        z=hist[hist.model_scope.eq(scope)];axes[0].plot(z.date,z.P95,'o-',label=names[scope]);zz=s[s.model_scope.eq(scope)&s.model_kind.eq('Benchmark_only')&s.technology_scenario.eq('half_slope')].sort_values('horizon_months')
        aa=a[a.model_scope.eq(scope)&a.frontier_definition.eq('P95')].A_z_anchor.iloc[0];axes[1].plot([0,12,24],[aa,*zz.A_z_point],marker='o',label=names[scope]+' (half slope)')
    axes[1].plot([0,12,24],[anchor,*main.A_z_point],marker='D',ls='--',label='Pretrained mechanism C1 x T1');axes[0].set_title('Separate observed 90-day P95 frontiers');axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2));axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'));axes[1].set_title('Scenario comparison; different mechanisms');axes[1].set_xlabel('Months after '+str(base.date()))
    for ax in axes:ax.set_ylabel('A_z');ax.legend(fontsize=8);ax.grid(alpha=.15)
    save(fig,'08_pretrained_chat_overall_comparison.png','Chat/all Benchmark-only; no compute mechanism or RLHF attribution')
    time=read_csv(out/'time_axis_projection_sensitivity.csv');fig,ax=plt.subplots(figsize=(8,4.5));ax.plot(main.horizon_months,main.A_z_point,'o-',label='Submission frozen M2')
    for comparison,z in time.groupby('comparison'):ax.plot(z.horizon_months,z.release_point,'o--',label='Release M2: '+comparison.replace('_',' '))
    ax.set_xlabel('Forecast months');ax.set_ylabel('A_z point scenario');ax.set_title('Time-axis sensitivity, same T0 and C1 x T1');ax.legend(fontsize=8);ax.grid(alpha=.15);save(fig,'09_submission_release_sensitivity.png','Same anchor coefficient sensitivity and separately pooled Release anchor')
    comp=read_csv(out/'technology_compensation.csv');fig,ax=plt.subplots(figsize=(8,4.5));ax.axhspan(comp.historical_tau_P5.iloc[0],comp.historical_tau_P95.iloc[0],color='#3573b9',alpha=.18,label='Frozen historical tau bootstrap P5-P95')
    ax.plot(range(4),comp.tau_required_point,'o-',label='Required tau for C0+T1 target');ax.axhline(comp.tau_required_point.iloc[0]*2,color='grey',ls='--',label='Frozen tau point (T2)');ax.set_xticks(range(4),comp.compute_scenario);ax.set_ylabel('Required A_z increment per month');ax.set_title('Conditional non-scale compensation; not innovation probability');ax.legend(fontsize=8);save(fig,'10_non_scale_compensation.png','Range comparison only, horizon-invariant required rate')
    fig,ax=plt.subplots(figsize=(8,4));c8=read_csv(out/'C8_anchor_sensitivity.csv');ax.bar(['All pretrained anchor','Exclude known top10% imbalance','C8 known only'],c8.P95);ax.set_ylabel('Current P95 A_z');ax.set_title('C8 structure sensitivity: 12 / 18 anchor models are unknown');ax.tick_params(axis='x',labelsize=8);save(fig,'11_C8_anchor_structure_sensitivity.png','Known high imbalance removal and missing-C8 selection caveat')
    csv(out/'figure_manifest.csv',manifest)


def build_report_and_figures(root,cfg,d):
    """Assemble the final report from real current/frozen outputs, never invented results."""
    build_figures(root,cfg,d)
    build_final_report(root,cfg,d)


def build_final_report(root,cfg,d):
    """Final 4.1–4.13 report with frozen evidence separated from extrapolations."""
    out=root/'outputs/evolution/phase3';p1=root/'outputs/evolution/phase1';p2=root/'outputs/evolution/phase2'
    s=read_csv(out/'future_frontier_scenarios.csv');a=read_csv(out/'frontier_anchor.csv');g=read_csv(out/'compute_growth_estimation.csv');gm=json.loads((out/'compute_growth_model.json').read_text(encoding='utf8'))
    main=s[s.model_kind.eq('frozen_M2_increment')&s.frontier_definition.eq('P95')&s.compute_scenario.eq('C1')&s.technology_scenario.eq('T1')].sort_values('horizon_months')
    c8=read_csv(out/'C8_anchor_sensitivity.csv');weak=read_csv(p1/'C8_weak_tail_diagnosis.csv');co=read_csv(p2/'model_coefficients.csv');perf=read_csv(p2/'model_performance.csv');decomp=read_csv(p2/'contribution_decomposition.csv')
    bridge=read_csv(p1/'bridge_validation_metrics.csv');bm=json.loads((p1/'bridge_model.json').read_text(encoding='utf8'));spec,boot=frozen_m2(root)['M2_C'];trans=json.loads((p1/'ability_transform_parameters.json').read_text(encoding='utf8'))
    growth=gm['main_growth'];comp=read_csv(out/'technology_compensation.csv');loss=read_csv(out/'slowdown_ability_loss.csv');hind=read_csv(out/'hindcast_metrics.csv');time=read_csv(out/'time_axis_projection_sensitivity.csv');defs=read_csv(out/'frontier_definition_sensitivity.csv');risk=read_csv(out/'forecast_risk_flags.csv')
    types=d.groupby('model_type').size().rename('n').reset_index();core=decomp[decomp.series.isin(['R1','R2','R3'])&decomp.window_fraction.sub(1/3).abs().lt(1e-8)&decomp.time_axis.eq('submission')]
    coefficient_table=co[co.model.isin(['M1_linear','M2_C','M3_ND'])&co.inference.eq('HC3')]
    scenarios=pd.DataFrame([dict(compute_scenario=c,factor=f,monthly_log_growth=f*growth,annualized_multiplier=np.exp(12*f*growth),assumption=True) for c,f in cfg['compute_scenario_factors'].items()])
    csv(out/'compute_scenario_assumptions.csv',scenarios)
    visible=['horizon_months','forecast_date','A_z_point','A_z_median','A_z_P2.5','A_z_P10','A_z_P90','A_z_P97.5','expanded_risk_low','expanded_risk_high','risk_level']
    contribution=['horizon_months','Delta_scale','Delta_non_scale','future_scale_share','future_non_scale_share','A_100_unclipped','A_100_clipped_for_plot']
    benchmark=s[s.model_kind.eq('Benchmark_only')][['model_scope','horizon_months','technology_scenario','A_z_point','A_z_median','A_z_P2.5','A_z_P97.5','risk_level']]
    rm=main.iloc[0];r24=main.iloc[1];bounds=(rm.operational_fixed_transform_lower,rm.operational_fixed_transform_upper)
    text=f'''# 问题四最终报告：算力增长放缓下的情景化能力前沿投影

基于附件快照，T0={rm.base_date}；未来日期为{rm.forecast_date}、{r24.forecast_date}。这是相对于附件历史起点的12/24个月情景投影，不是相对于运行日的新预测。问题一至三、问题四PHASE1/2的结果保持冻结。PHASE3只新建算力增长、前沿统计、Benchmark-only对照、情景传播及诊断。

**定位：scenario-based frontier projection。** 以下点情景、条件参数/数据区间、扩大风险包络分别输出；不宣称可靠长期点预测、不把情景设定赋予发生概率。

## 4.1 数据与开源模型口径〔Level A〕

冻结C1/C2各4576条原记录、4497个去重模型，S0=4497、S1=422、S2=0。S1表示附件明确确认开放权重，不能据此证明严格开源许可证授权。当前有效S1分类：

{table(types)}

F0仅pretrained进入M2-C机制投影；F1 chat_finetuned仅Benchmark趋势；F2包含S1全部类型，仅描述实际开源权重能力前沿。提交日期既不一定是模型发布日期，也不一定是当时真实评测日；快照去重产生存留与选择效应。因此“历史前沿”严格说是当前附件按提交月份回溯的样本前沿。冻结详细证据见[PHASE1报告](question4_phase1_data_ability_bridge.md)与[PHASE2报告](question4_phase2_contribution_decomposition.md)。

本阶段真实读取冻结evolution_model_table.csv、ability_transform_parameters.json、model_definitions.json、coefficient_bootstrap.csv及各诊断表；原始C4 epoch_all_ai_models.csv仅用于新的算力趋势，原始C3 leaderboard_extended_timeseries.csv仅用于年份/参数方向检查。input_manifest.csv逐文件保留路径与SHA256。没有读取旧版问题四预测作为结论依据。

## 4.2 综合能力指标〔Level A〕

主指标为六个官方Benchmark的等权稳健标准化：$A_z=\\frac16\\sum_k(\\operatorname{{clip}}(x_k,P01_k,P99_k)-m_k)/IQR_k$，指标顺序{', '.join(trans['metrics'])}。全部使用冻结S1全期阈值，不月度重拟合。PCA只是冻结敏感性，不替换主指标。A100显示使用原映射P01={trans['display_mapping']['P01']:.9f}、P99={trans['display_mapping']['P99']:.9f}，未来值不重新归一化。

标准化分数本身没有天然上界，但**当前冻结的逐任务Winsor公式有操作性可实现范围 [{bounds[0]:.6f},{bounds[1]:.6f}]**。本阶段不修改公式、不给外推偷偷截断。超界原始A_z和A100仍保存并标记；它们只能表示增量模型的外推信号，不能作为该冻结Benchmark指标可实现的未来分数。A_100_unclipped与A_100_clipped_for_plot同时输出；正文用未截断A_z。这里没有新增logistic饱和上限。

## 4.3 C8细粒度能力检验〔Level A〕

冻结C8同轮与快照弱尾诊断：

{table(weak[['sample','ability_basis','n','high_average_weak_tail_count','high_average_weak_tail_rate']])}

不能把版本差异解释成模型退化。当前pretrained锚点C8敏感性如下；阈值是有C8的S1 pretrained样本imbalance P90，而不是主观筛选。未知C8保留并单列。

{table(c8)}

排除1个已知最高10%偏科模型后，P95下降{float(c8.P95.iloc[0]-c8.P95.iloc[1]):.6f}；{int(c8.unknown_C8_n.iloc[0])}/{int(c8.n.iloc[0])}模型C8未知。差异明显，不能说前沿不受偏科影响。将绝对锚点差作为结构压力包络的一项，未把缺失C8当作“无偏科”。

## 4.4 规模与非规模进步分解〔Level B〕

冻结回归HC3诊断，不重新拟合：

{table(coefficient_table[['model','term','n','coef','CI_low','CI_high','p_value']])}

M1控制参数规模，M2控制训练算力，M3控制N、D；时间项是非规模进步代理的条件关联，不是因果技术效应或RLHF效率。M2只有21条完整记录，tau的HC3区间跨0，Bootstrap中{int((boot.t_month<0).sum())}/5000个tau为负，本轮原样保留。组织依赖、样本选择和时间口径限制已在冻结PHASE2中验证，不因本轮情景计算而消失。

历史主窗口R1/R2/R3：

{table(core[['series','n','early_n','late_n','Delta_scale','Delta_tech','Delta_pred','absolute_share_scale','absolute_share_tech']])}

这些absolute share只描述过去抵消项的绝对贡献，**没有被乘以未来增长**。未来分解严格使用beta_C×未来log算力增量和tau×情景时间增量。冻结M2的LOOCV亦只支持有限的关联泛化：

{table(perf[perf.model.isin(['M1_linear','M2_C','M3_ND'])&perf.evaluation.eq('LOOCV')][['model','n','RMSE','MAE','R2']])}

## 4.5 Loss-Benchmark桥接负结果〔Level A/B：不成立〕

C6 High仅{bm['High_count']}个、全部{', '.join(bm['High_families'])}；Medium={bm['Medium_count']}。B0/B1/B2均未取得可靠CV泛化：

{table(bridge[['candidate','evaluation','n','RMSE','MAE','R2']])}

B2是常数保序回归诊断，bridge_valid_for_conversion={bm['bridge_valid_for_conversion']}。不能把前三问Loss可靠转换为第四问Benchmark；本轮没有使用任何Loss桥接数值。前三问与本问通过scale/compute资源变量及机制解释联系。

## 4.6 能力前沿定义〔Level A〕

按样本数量依次90→120→180天，仅不足20才扩大，不追求更高前沿；达到180天仍不足就保留并警示。主定义P95，P90与Top3敏感性；maximum只存在辅助附录统计，未作为主锚点。

{table(a[['model_scope','frontier_definition','base_date','window_days','n','A_z_anchor','anchor_below_minimum']])}

pretrained只有18条，必须承认锚点稀疏。历史Benchmark-only与hindcast固定90天滚动前沿，主机制当前锚点按规则扩大180天，二者不是同一窗口。Top3指现有最多3条模型的均值；Bootstrap重复条目属于重采样经验分布，不伪称新的不同模型。

## 4.7 算力增长趋势〔Level C所需历史输入〕

C4由冻结S1 pretrained可靠映射：38条匹配记录属于37个不同训练项目，按epoch_row去重。最近36个月有{gm['recent_unique_models']}个原始去重项目达到20要求；向前多6个月的滚动起始上下文共{gm['context_models']}个，Bootstrap重采样上下文模型，重算所有滚动窗口，避免把重叠窗口当独立观测。

{table(g[['method','frontier_definition','monthly_log_growth','annualized_multiplier','CI_low','CI_high','bootstrap_draws','recent_unique_models']])}

主Theil-Sen是两点斜率中位数，主响应6个月滚动P90 logCompute；OLS与P95/Top3敏感性同为滚动前沿，QuantReg tau=.9是个体模型条件分位数，估计对象不同。方法参照[SciPy Theil-Sen定义](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html)和[statsmodels QuantReg](https://www.statsmodels.org/v0.14.4/generated/statsmodels.regression.quantile_regression.QuantReg.fit.html)。主5000模型Bootstrap月度斜率区间保留全部有效结果，不强制增长非负、不施加单调包络。OLS-P95/Top3及QuantReg只有点敏感性，没有伪造Bootstrap区间。

最新有算力项目日期{gm['latest_compute_release']}，滞后T0 {gm['compute_staleness_days']}天；这不是全球算力普查。样本来自被Leaderboard匹配到的开放pretrained项目，算力估计及发布时间本身也有误差，5000次重采样不能消除选择偏差。C3按年份/来源的模型数与N统计保留在C3_long_run_direction_audit.csv，26条历史补充记录没有拼入A_z。

## 4.8 算力放缓情景〔Level C〕

{table(scenarios)}

情景系数1、.5、.25、0是题设假设，不是数据估计；C1为主，C0只是无放缓参照。T0/T1/T2对应0、.5、1倍冻结非规模时间代理，T1为中性主情景，不称最可能。主增量公式：

$$A(h)=A_{{anchor}}+\\beta_C g_C f_C h+\\tau_C f_T h.$$

beta_C={spec['beta'][1]:.12f}，tau_C={spec['beta'][2]:.12f}，h=12/24。把横截面条件均值斜率迁移到P95前沿是附加不变性假设，没有重新拟合分位能力模型。24主情景加P90/Top3敏感性共72条机制结果；另18条Benchmark-only结果，全表90条。Monte Carlo 10000次：同一冻结系数Bootstrap行抽beta/tau，算力与锚点Bootstrap各独立抽取；三块可能共享模型，独立块假设仍可能低估协方差，未假装完整联合世界概率模型。240000条主情景draw保留在frontier_projection_mc.csv，mc_joint_draws.csv保留每块replicate可复算。

## 4.9 未来12个月前沿〔Level C〕

{table(main[main.horizon_months.eq(12)][visible])}

点情景是插入原估计所得，MC中位数来自重采样传播，不必相等。以下P2.5/P97.5只是参数/数据分位数。主P95方案全部12个月情景：

{table(s[s.model_kind.eq('frozen_M2_increment')&s.frontier_definition.eq('P95')&s.horizon_months.eq(12)][['compute_scenario','technology_scenario','A_z_point','A_z_median','A_z_P2.5','A_z_P97.5','risk_level']])}

## 4.10 未来24个月前沿〔Level D〕

{table(main[main.horizon_months.eq(24)][visible])}

24个月点情景{r24.A_z_point:.6f}与MC中位数{r24.A_z_median:.6f}均超出冻结操作性上界{bounds[1]:.6f}，不可当作可实现Benchmark分数。继续保留未截断数值作为压力外推，以VERY_HIGH标识。相对观测max/P99、超过max+2IQR、超过冻结可实现范围分别诊断，不能相互替代。

F1/F2只做flat、50%、full slope延续，不套用M2：

{table(benchmark)}

F0 Benchmark-only斜率为负而机制为正，来源是稀疏提交组和滚动窗口组成变化，两套预测并不相互验证。chat增长也不能解释为RLHF技术进步。约10个月不足识别指数阻尼k，故没有拟合damped，50% slope是显式假设。不同类型窗口/分位样本组成不同，不能据数值大小直接推出模型类型优劣。

## 4.11 非规模进步补偿效应〔Level C；24个月应用为D〕

未来主情景的有符号增量与显示映射：

{table(main[contribution])}

仅两项非负且总增量正时报告份额；Bootstrap中的负项仍保留，没有强制归一化。对应同一T1，算力放缓损失相对C0：

{table(loss[loss.technology_scenario.eq('T1')][['horizon_months','compute_scenario','point_loss','loss_P2_5','loss_median','loss_P97_5']])}

同一joint draw下算力差beta*g*(1-f_C)*h，技术项相消。条件补偿：$\\tau_{{required}}=.5\\tau_C+\\beta_C(1-f_C)g_C$，在线性固定参数假设下不随h变化。

{table(comp)}

所需速率位于历史Bootstrap P5–P95范围只表示参数比较没有超出该区间，**不代表未来技术成功补偿的概率，也不证明必然可补偿**；历史tau区间很宽、部分为负。

## 4.12 不确定性与外推风险

逐月训练1..6预测7，依次到1..9预测10：

{table(hind)}

Benchmark线性只拟合当时过去的6–9个前沿点；机制不违背冻结规则，因此没有逐折重拟合M2，前两折冻结M2使用后来提交模型，全部折能力标定用全期S1。**这些是条件回溯诊断，不是严格时间外验证**。90日窗口重叠，四折覆盖率无从校准长期区间；机制hindcast使用C0×T2历史延续，且区间条件于当时算力斜率，遗漏该斜率不确定性。不能把50%覆盖当成真实预测概率。

回测误差大，正式表保留MC分位数，另增加expanded_risk_low/high：两侧各增加$1.96\\,RMSE_{{1step}}\\sqrt h+\\Delta_{{C8-anchor}}$。RMSE取同类型可用线性/机制回测中的最大值，C8项只加pretrained。sqrt(h)仅为步误差累积压力假设，不是用4折估计出的真实扩散规律；1.96也不赋予其95%覆盖含义。风险包络非常宽，恰表明数据无法提供精确长期区间；不把它隐藏成“可靠95%CI”。

主12/24外推长度/实际历史跨度分别{rm.horizon_history_ratio:.4f}/{r24.horizon_history_ratio:.4f}。固定10000次参数传播缺少未来算法、评测污染、数据质量、算力计量与组织依赖的完整不确定性。均值到前沿迁移、C8覆盖不全、算力发布时间滞后同时限制解释。

Submission/Release对照（共用T0与未来日期，另列固定Submission锚点以隔离系数效应）：

{table(time[['horizon_months','comparison','submission_point','release_point','difference','release_anchor_n','high_time_axis_sensitivity']])}

high_time_axis_sensitivity按差值>主参数区间宽度50%判定；另记录相对扩大包络结果。未达到阈值不撤销冻结PHASE2 time_axis_sensitive=True：宽区间可能掩盖轴间差异，不能据此宣称时间口径稳健。Release池180天仅10条，也未达20。

前沿定义对照：

{table(defs)}

明显差异阈值=.1×pretrained历史A_z IQR，为显式诊断规则，不是统计检验。P90/Top3均触发当前敏感性标记。所有结果风险来自configs/evolution_phase3.yaml：history ratio>.5/1/2；算力外推超历史max的IQR距离>0/1/2；参数区间宽度/max(|点增量|,A_z IQR)>.5/1/2，对应逐级0–3；冻结时间敏感、前沿定义敏感、锚点<20各至少HIGH，超冻结可实现上界VERY_HIGH；总等级取各项最大。无人工个别打分，不是事件发生概率。全部具体维度保留forecast_risk_flags.csv。

11张图位于outputs/figures/question4_phase3，manifest逐图注明用途。所有正文A_z图保留未截断情景；第1图将参数扇形与扩大压力包络分面显示，防止混淆。核心图：

![历史前沿、条件参数扇形和单独的压力包络](../figures/question4_phase3/01_frontier_projection_fan.png)

![逐月条件回溯诊断](../figures/question4_phase3/07_hindcast_observed_predicted.png)

![pretrained、chat、overall分别比较](../figures/question4_phase3/08_pretrained_chat_overall_comparison.png)

![非规模进步的条件补偿](../figures/question4_phase3/10_non_scale_compensation.png)

## 4.13 问题四结论与证据等级

1. **Level A：** 真实S1支持类型分层和稀疏前沿统计。pretrained P95锚点{rm.A_z_anchor:.6f}，180天18条；chat/overall样本较多，但它们不提供pretrained算力机制证据。
2. **Level B：** 冻结回归显示训练规模与能力的条件正关联；时间代理对组织、样本与时间轴敏感，不能声称纯技术因果贡献，历史绝对份额不能用于未来权重。
3. **Level C：** 当前C4子样本的g={growth:.6f}与冻结M2给出C1×T1的12月点情景{rm.A_z_point:.6f}，MC参数区间[{rm['A_z_P2.5']:.6f},{rm['A_z_P97.5']:.6f}]；减半算力增长相对C0降低未来增量，但只是条件场景。
4. **Level D：** 24月点情景{r24.A_z_point:.6f}已超计分公式可实现范围，且历史跨度比{r24.horizon_history_ratio:.3f}；VERY_HIGH风险，不据此宣称真实未来Benchmark达到该值。
5. **Level C/D：** 补偿所需tau可与历史宽Bootstrap范围比较，但无法给未来技术创新的成功概率；C8、直接前沿负趋势与回测大误差使总体结论更保守。
6. **Level A/B负结果：** Loss桥接无可靠CV泛化，前三问Loss没有被转换成第四问能力；问题四全部预测建立于真实冻结Leaderboard能力、资源关联与显式情景假设。

复现：在项目根目录运行 `python -X utf8 -B scripts/22_build_evolution_projection.py --stage all`，随后 `python -X utf8 -B scripts/22_verify_evolution_projection.py`。只重建PHASE3新文件，不执行任何冻结PHASE1/2脚本。所有历史冻结文件SHA256完整核对见phase3/verification_summary.json。本阶段结束，问题四停止；不改前三问。
'''
    report=root/'outputs/report/question4_final.md';report.parent.mkdir(parents=True,exist_ok=True);report.write_text(text,encoding='utf8')
