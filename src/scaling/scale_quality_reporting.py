"""Scientific figures and stage report from saved scale-dependent experiment results."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .quality_extension import subset,group_reference
from .scale_quality import predict_scale
from .quality_reporting import save,obs_pred,table


def figures(root,s,ts,p,ref,oof,predictions,points,surfaces,domain,cfg):
    """Plots retain observed points, true parameter boundaries and negative evidence."""
    folder=root/'outputs/figures/question2_phase2C';folder.mkdir(parents=True,exist_ok=True)
    dpi=cfg['figure_dpi'];models=cfg['models'];b6,b7,b8=[ts[k] for k in ['B6','B7','B8']]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,target in zip(axes,['absolute_Loss','within_ND_DeltaL']):
        rows=[r for r in s['B6_CV'] if r['target']==target]
        ax.bar([r['model'] for r in rows],[r['RMSE'] for r in rows],color=['gray','tab:blue','tab:orange','tab:green'])
        ax.set(ylabel='OOF RMSE',title='B6 ND Group CV: '+target.replace('_',' '))
    save(fig,folder,'01_B6_group_CV_comparison.png',dpi)
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for j,m in enumerate(models):
        sc=obs_pred(axes[0,j],b6['L'],oof[m],m+': absolute Loss OOF',b6['Q']);fig.colorbar(sc,ax=axes[0,j],label='Q')
        obs,pp,_,refs=group_reference(b6,oof[m]);sc=obs_pred(axes[1,j],obs[~refs],pp[~refs],m+': within-ND Delta Loss OOF',b6['Q'][~refs])
        axes[1,j].set(xlabel='Observed Delta Loss',ylabel='Predicted Delta Loss');fig.colorbar(sc,ax=axes[1,j],label='Q')
    save(fig,folder,'02_B6_absolute_and_difference_OOF.png',dpi)
    new=np.array([k not in set(b6['keys']) for k in b7['keys']]);fig,axes=plt.subplots(2,3,figsize=(15,9))
    for j,m in enumerate(models):
        sc=obs_pred(axes[0,j],b7['L'][new],predictions['B7'][m][new],m+': B7 new-Q absolute',b7['Q'][new]);fig.colorbar(sc,ax=axes[0,j],label='New Q level')
        obs,pp,_,refs=group_reference(b7,predictions['B7'][m]);ix=new&~refs
        sc=obs_pred(axes[1,j],obs[ix],pp[ix],m+': B7 new-Q differences',b7['Q'][ix]);fig.colorbar(sc,ax=axes[1,j],label='New Q level')
        axes[1,j].set(xlabel='Observed Delta Loss',ylabel='Predicted Delta Loss')
    save(fig,folder,'03_B7_nonoverlap_absolute_and_difference.png',dpi)
    fig,axes=plt.subplots(2,2,figsize=(11,8));Ns=np.unique(b6['N']);Ds=np.unique(b6['D'])
    for ax,(n,d) in zip(axes.flat,[(Ns[0],Ds[0]),(Ns[0],Ds[-1]),(Ns[-1],Ds[0]),(Ns[-1],Ds[-1])]):
        ix=(b6['N']==n)&(b6['D']==d);a=subset(b6,ix);order=np.argsort(a['Q'])
        ax.plot(a['Q'][order],a['L'][order],'ko',label='B6 observations')
        tt=dict(N=np.full(100,n),D=np.full(100,d),Q=np.linspace(.1,1,100))
        for m in models:ax.plot(tt['Q'],predict_scale(tt,p,ref,m,s['fitted'][m]['parameters']),label=m)
        ax.set(xlabel='Q_score',ylabel='Loss',title=f'N={n:g} B, D={d:g} B');ax.legend(fontsize=8)
    save(fig,folder,'04_fixed_ND_quality_curves.png',dpi)
    fig,ax=plt.subplots(figsize=(8,4))
    for m in models:
        r=[r for r in s['Q_level_CV'] if r['model']==m]
        ax.plot([r['held_Q'] for r in r],[r['RMSE'] for r in r],'o-',label=m)
    ax.set(xlabel='Held-out Q level',ylabel='RMSE',title='B6 leave-one-Q-level-out (endpoints extrapolate)');ax.legend()
    save(fig,folder,'05_leave_one_Q_level_out.png',dpi)
    _uncertainty_figures(root,s,folder,dpi)
    _sensitivity_figures(surfaces,models,folder,dpi)
    inside=((b8['N']>=b6['N'].min())&(b8['N']<=b6['N'].max())&(b8['D']>=b6['D'].min())&(b8['D']<=b6['D'].max())&(b8['Q']>=b6['Q'].min())&(b8['Q']<=b6['Q'].max()))
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for j,m in enumerate(models):
        for ax,ix,title in [(axes[0,j],inside,'inside B6 numeric ranges'),(axes[1,j],~inside,'outside B6 numeric ranges')]:
            sc=obs_pred(ax,b8['L'][ix],predictions['B8'][m][ix],m+': '+title,b8['Q'][ix]);fig.colorbar(sc,ax=ax,label='Q_score');ax.axvline(p['E'],color='red',ls=':',lw=1,label='Frozen E');ax.legend(fontsize=8)
    fig.suptitle('B8 incompatible semisynthetic pressure test; no fitting',fontsize=12)
    save(fig,folder,'12_B8_pressure_inside_outside.png',dpi)
    _extra_figures(root,s,ts,p,ref,domain,cfg)


def _uncertainty_figures(root,s,folder,dpi):
    """Bootstrap histograms and correlation matrices, preserving boundary masses."""
    from .io import csv_rows
    import json
    _,rows=csv_rows(root/'outputs/scaling/phase2C/bootstrap_replicates.csv')
    for j,m in enumerate(['S1','S2','S3'],6):
        parameters=list(s['fitted'][m]['parameters']);fig,axes=plt.subplots(1,len(parameters),figsize=(4*len(parameters),3.6))
        for ax,key in zip(np.atleast_1d(axes),parameters):
            v=[json.loads(r['parameters'])[key] for r in rows if r['model']==m and r['success']=='True']
            bound=next(r for r in s['bootstrap_parameters'] if r['model']==m and r['parameter']==key)['parameter_boundary_rate']==1
            if bound:
                ax.bar(['Parameter bound'],[len(v)],color='tab:orange');ax.text(.5,.85,'100% boundary mass',transform=ax.transAxes,ha='center',fontsize=9)
            else:ax.hist(v,bins=35,color='tab:blue',alpha=.8)
            ax.set(xlabel=key,ylabel='ND-bootstrap count',title=m+' '+key)
        save(fig,folder,f'{j:02d}_{m}_parameter_bootstrap.png',dpi)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,m in zip(axes,['S1','S2','S3']):
        keys=list(s['fitted'][m]['parameters']);a=np.full((len(keys),len(keys)),np.nan)
        for r in s['parameter_correlations']:
            if r['model']==m and r['Pearson'] is not None:a[keys.index(r['parameter_1']),keys.index(r['parameter_2'])]=r['Pearson']
        cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#d9d9d9')
        im=ax.imshow(np.ma.masked_invalid(a),vmin=-1,vmax=1,cmap=cmap)
        for j in range(len(keys)):
            for k in range(len(keys)):
                if np.isnan(a[j,k]):ax.text(k,j,'n.a.',ha='center',va='center',fontsize=8)
        ax.set_xticks(range(len(keys)),keys);ax.set_yticks(range(len(keys)),keys);ax.set_title(m+' Pearson\nGray: undefined at bound/constant')
        fig.colorbar(im,ax=ax,shrink=.8)
    save(fig,folder,'09_bootstrap_parameter_correlations.png',dpi)


def _sensitivity_figures(rows,models,folder,dpi):
    """Discrete actual design cells; magnitude explicitly separated from gradient sign."""
    Ns=sorted(set(r['N_params_B'] for r in rows));Ds=sorted(set(r['D_tokens_B'] for r in rows))
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for i,q in enumerate([.1,.6]):
        for j,m in enumerate(models):
            ax=axes[i,j];a=np.full((len(Ns),len(Ds)),np.nan)
            for r in rows:
                if r['model']==m and r['Q_score']==q:a[Ns.index(r['N_params_B']),Ds.index(r['D_tokens_B'])]=r['abs_quality_sensitivity']
            im=ax.imshow(a,origin='lower',aspect='auto',cmap='viridis')
            ax.set_xticks(range(len(Ds)),[f'{d:g}' for d in Ds]);ax.set_yticks(range(len(Ns)),[f'{n:g}' for n in Ns])
            ax.set(xlabel='D (billion tokens)',ylabel='N (billion parameters)',title=f'{m}: |dLoss/dQ| at Q={q:g}')
            fig.colorbar(im,ax=ax,label='Loss per unit Q')
    save(fig,folder,'10_quality_sensitivity_ND_surfaces.png',dpi)
    fig,ax=plt.subplots(figsize=(8,5));a=np.full((len(Ns),len(Ds)),np.nan)
    for r in rows:
        if r['model']=='S1' and r['Q_score']==.6:a[Ns.index(r['N_params_B']),Ds.index(r['D_tokens_B'])]=r['effective_gamma']
    im=ax.imshow(a,origin='lower',aspect='auto',cmap='viridis');fig.colorbar(im,ax=ax,label='S1 gamma(N,D)')
    ax.set_xticks(range(len(Ds)),[f'{d:g}' for d in Ds]);ax.set_yticks(range(len(Ns)),[f'{n:g}' for n in Ns])
    ax.set(xlabel='D (billion tokens)',ylabel='N (billion parameters)',title='S1 effective gamma; distinct from marginal Loss sensitivity')
    save(fig,folder,'11_S1_effective_gamma_surface.png',dpi)


def _extra_figures(root,s,ts,p,ref,domain,cfg):
    """Numeric domain-Q interval and C/eta bound confounding; no source retuning."""
    from .scale_quality import marginal_quality
    folder=root/'outputs/figures/question2_phase2C';dpi=cfg['figure_dpi']
    low,high=s['question1_Q_domain_range'];q=np.linspace(low,high,100)
    tt=dict(N=np.full(100,ref['N0']),D=np.full(100,ref['D0']),Q=q)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    l0=p['E']+p['A']*ref['N0']**(-p['alpha'])+p['B']*ref['D0']**(-p['beta'])
    for m in ['S1','S2','S3']:
        a=s['fitted'][m]['parameters']
        axes[0].plot(q,-marginal_quality(tt,p,ref,m,a),label=m)
        axes[1].plot(q,predict_scale(tt,p,ref,m,a)-l0,label=m)
    axes[0].set(xlabel='Q coordinate',ylabel='|dLoss/dQ|',title='Frozen domain-Q interval: numeric support only')
    axes[1].set(xlabel='Q coordinate',ylabel='Extra Loss vs Q=1',title='At fixed reference N0,D0; no mixture model')
    for ax in axes:ax.legend()
    save(fig,folder,'13_question1_Q_numeric_interval.png',dpi)
    r=s['S2_bound_sensitivity'];fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([x['C_upper_bound'] for x in r],[x['C'] for x in r],'o-',label='C');axes[0].set(xscale='log',yscale='log',xlabel='Computational C upper bound',ylabel='Fitted C',title='S2 boundary sensitivity')
    axes[1].plot([x['C_upper_bound'] for x in r],[x['C_times_eta'] for x in r],'o-');axes[1].set(xscale='log',xlabel='Computational C upper bound',ylabel='Fitted C x eta',title='Stable product does not identify both factors')
    save(fig,folder,'14_S2_bound_identifiability.png',dpi)


def report(root,s,domain):
    """Generate conclusions from the fixed gate, never from preferred-looking plots."""
    out=root/'outputs/scaling/phase2C';p=s['classical_parameters'];ref=s['reference_scales']
    lines=['# 问题二 PHASE 2C：尺度调制的质量 Scaling Law','',
           '本阶段已完成，等待人工确认。未进入领域配比融合。', '',
           ('结论：三个候选均未通过 B7 非重叠组内差分门槛；不冻结统一 L(N,D,Q)，停止继续增加复杂函数。' if s['all_B7_DeltaL_failed'] else '结论：按预设门槛和 S3→S1→S2 优先级审查固定候选，最终状态见接受表。'),
           'B6 的方向和预测改善支持质量影响，但不足以证明某个候选的效应量可以迁移。不能以绝对 Loss 的高相关替代组内质量差分验证。', '',
           '## 输入、冻结与证据层级','',
           'B6 是唯一参数拟合来源；B7 的 90 个非重叠 Q 点是主要扩展复核；B8 为“大范围半合成不兼容压力测试/负面证据”，不参与任何拟合、模型或超参数选择。',
           'B6–B8 为“基于真实数据校准的半合成补充实验”。B7 完整包含 B6 的 360 条记录，新点只新增 Q=0.5、0.7，ND 组与 B6 相同；不可称为新的独立 ND 外部训练实验。',
           'B7 已在前一轮反复用于诊断，当前候选受到已知失效模式启发；本轮仍是扩展复核，不提供未见真实外部实验的无偏确认。',
           '读取 PHASE 1 主参数 CSV 的完整精度；五参数未重估。问题一仅仅读取冻结领域 Q 汇总，未解析原始质量 JSONL/XZ，也未重建 Q/K。', '',
           '```text', '\n'.join(f'{k}={v:.17g}' for k,v in p.items()),'```', '',
           '## 参考尺度与模型','',
           f"N0=exp(median(log(unique N_B6)))={ref['N0']:.17g} B；D0=exp(median(log(unique D_B6)))={ref['D0']:.17g} B（数值上 150 B）。",
           '这是唯一设计尺度在对数空间的中位数，不是 Loss 加权中心。中心固定为重参数化常数，各 CV/Bootstrap 不估计它们；没有使用 held-out Loss 求中心。',
           r'S1：\(\gamma(N,D)=\gamma_0(N/N_0)^{-u}(D/D_0)^v\)，\(L=E+A N^{-\alpha}+B[D Q^{\gamma(N,D)}]^{-\beta}\)。',
           r'S2：\(L=L_0+C(N/N_0)^{-u}(D/D_0)^v(Q^{-\eta}-1)\)。',
           r'S3：\(L=L_0+C(N/N_0)^{-u}(D/D_0)^v(-\ln Q)\)。',
           '三者无主模型 offset，Q=1 时精确退化为冻结 L0。γ0/C/η>0，u/v≥0。有效数据量是候选建模解释，未被真实直接训练实验证明。', '',
           '## 拟合与分组验证协议','',
           'bounded nonlinear least squares，直接拟合原始 Loss；严格正参数在 log 坐标中优化，u/v 使用原坐标以保留零边界。每次主拟合、CV、Q-level 留出和每个 Bootstrap 模型都执行 50 个独立初值，保存参数、目标值、收敛状态、Jacobian rank/condition 和触边字段。',
           '计算边界：γ0∈[1e-6,8]，C∈[1e-6,100]，u/v∈[0,1]，η∈[1e-5,5]；并非假装由附件估计出的统计阈值。S1 在 B6 支持域内的最大指数仍处于可计算范围，不截断或修改模型公式。',
           '5-fold GroupKFold 按精确 (N,D) 组进行。同组 8 个 Q 点同时进入训练或测试，训练折独立使用预设初值，不使用全量拟合初值。主差分评价排除 Q=1 自身零差分，避免抬高相关与 R²。',
           '2000 次 ND block bootstrap；每次有放回抽取 45 整组，组内所有 Q 带同一重数；seed=2026，三个模型共享相同抽样。全网格每组 8 点，因此逐观测等权也给各组相同总权。',
           'Bootstrap CI 条件于冻结经典参数、固定网格和半合成生成机制；不包含经典 β 不确定性、未见 Q 水平机制变化或真实跨源误差。计算仅使用 CPU。', '',
           '## 主拟合参数与识别诊断','']
    parameter_rows=[]
    for m,f in s['fitted'].items():
        parameter_rows.append(dict(model=m,**f['parameters'],fit_RMSE=float(np.sqrt(f['objective'])),
                                   jacobian_rank=f['jacobian_rank'],jacobian_condition=f['jacobian_condition'],
                                   boundary_parameters=','.join(f['boundary_parameters']) or 'none',
                                   n_starts=f['n_starts'],n_successful_starts=f['n_successful_starts']))
    lines+=table(parameter_rows,['model','gamma0','C','u','v','eta','fit_RMSE','jacobian_rank','jacobian_condition','boundary_parameters','n_successful_starts'])
    lines+=['','## 参数 Bootstrap 与相关','']
    lines+=table(s['bootstrap_parameters'],['model','parameter','mean','median','std','P025','P975','failure_rate','parameter_boundary_rate'])
    lines+=['','参数相关矩阵全部保存于 parameter_correlations.csv，同时包含 Pearson 和 Spearman。','',
            '全部重采样在同一边界、且变化仅为浮点求解误差的参数，其相关性标为未定义；不能用该噪声计算的 Pearson 解释参数耦合。C/η 的实用不可识别另由 B6 上界敏感性证明。',
            'S2 若 C 达到上界、η 同时变小，C×η 才可能受到数据稳定约束：当 η→0，C(Q^{-η}−1)≈Cη(−lnQ)，S2 接近 S3。C 与 η 的单独数值存在实用不可识别问题；触边 CI 的集中不等于参数已被识别。',
            'v 在非负边界意味着附件未识别出正的 D 增长效应。即使 S3 数值 Jacobian 满秩，也只能对边界参数作单侧推断，不能把 v 的浮点微小正值解释成实质增长。', '']
    pairs=[]
    for r in s['parameter_correlations']:
        ks=list(s['fitted'][r['model']]['parameters'])
        if ks.index(r['parameter_1'])<ks.index(r['parameter_2']):pairs.append(r)
    lines+=table(pairs,['model','parameter_1','parameter_2','Pearson','Spearman','defined'])+['']
    lines+=table(s['S2_bound_sensitivity'],['C_upper_bound','C','eta','C_times_eta','u','v','objective','jacobian_condition'])
    lines+=['','C 上界敏感性只在 B6 上改变计算约束，未使用 B7 选择边界，也未据此替换主结果。','',
            '## B6 absolute Loss 与 within-ND Delta-L OOF','']
    lines+=table(s['B6_CV'],['model','target','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias'])
    lines+=['','B6 组内差分使用同一模型/参数和同一 held-ND 折的 Q=1 预测参考，没有独立拟合差分。N/E 项与固定 offset 在差分中抵消。','',
            '## B7 非重叠 90 点：最终质量效应门槛','']
    lines+=table(s['B7_nonoverlap'],['model','target','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias'])
    lines+=['','B7 的 Q=1 实测参考来自该 ND 的全量 B7，并与 B6 重叠；差分是在新增 Q 点上的复核，不能包装成新增独立参考观测。450 条 full /360 条 overlap 指标另见 B7_validation_all_subsets.csv。',
            '预设关键门槛：B7 非重叠差分 R²>0 且 Pearson/Spearman>0。正相关但 R²≤0 表示方向有所复现，效应量预测仍未优于该子集的均值差分。不得改用绝对 Loss 的高 R² 声称通过。','',
            '## 完整 Q-level 留出','']
    lines+=table(s['Q_level_CV'],['model','held_Q','test_kind','RMSE','MAE','R2','mean_bias'])
    from .io import csv_rows,write_csv
    _,previous=csv_rows(root/'outputs/scaling/phase2/leave_one_Q_level_out.csv')
    benchmarks=[dict(model=r['model'],phase='frozen_Phase2B',held_Q=float(r['held_Q']),RMSE=float(r['RMSE']),MAE=float(r['MAE']),mean_bias=float(r['mean_bias'])) for r in previous if r['model'] in ['MQ1','MQ2'] and float(r['held_Q'])==.1]
    benchmarks += [dict(model=r['model'],phase='Phase2C',held_Q=r['held_Q'],RMSE=r['RMSE'],MAE=r['MAE'],mean_bias=r['mean_bias']) for r in s['Q_level_CV'] if r['held_Q']==.1]
    write_csv(out/'low_Q_extrapolation_comparison.csv',benchmarks)
    lines+=['','最低 Q 层与冻结上一轮模型的对照（均为整层留出，不重拟合上一轮结果）：','']
    lines+=table(benchmarks,['phase','model','held_Q','RMSE','MAE','mean_bias'])
    _report_tail(root,s,domain,lines)


def _report_tail(root,s,domain,lines):
    """Complete limitations and stopping decisions without inventing acceptance thresholds."""
    lines+=['','Q=0.1 和 Q=1 是端点留出外推，其他已观测 Q 层为插值；逐层结果不可仅用平均误差替代。各层没有读取被留出 Loss 作初值。','',
            '## 隐含质量敏感度与方向约束','',
            r'S1：\(\partial L/\partial Q=-B D^{-\beta}\exp[\beta\gamma(-\ln Q)]\beta\gamma/Q\)。',
            r'S2：\(\partial L/\partial Q=-C(N/N_0)^{-u}(D/D_0)^v\eta Q^{-\eta-1}\)。',
            r'S3：\(\partial L/\partial Q=-C(N/N_0)^{-u}(D/D_0)^v/Q\)。',
            '在 B6 全部 360 点核验解析导数和数值差分导数。Q 边界使用支持域内单侧差分；步长 1e-5 是数值检查精度设置，不是数据阈值。', '']
    lines+=table(s['monotonicity_checks'],['model','n','analytic_violation_rate','numerical_violation_rate','min_derivative','max_derivative','max_abs_analytic_numeric_discrepancy'])
    lines+=['','尺度趋势按固定另一尺度和 Q 的相邻设计点比较 |∂L/∂Q|，数值相对变化≤1e-8 视为平坦。此容差防止将边界 v≈1e-20 误称实质增长。','']
    lines+=table(s['sensitivity_scale_trends'],['model','axis','n_adjacent_pairs','decreasing_rate','increasing_rate','flat_rate','min_relative_change','max_relative_change'])
    lines+=['','S1 的 γ(N,D) 随 D 增加不保证实际 |∂L/∂Q| 同样增加：经典 D^{-β} 会抵消部分增长，效应也依赖 Q。S2/S3 的 v 若触零边界，只支持 D 敏感度基本平坦。当前不强行增加交互来满足预设方向。','',
            '## 问题一冻结 Q 区间：仅数值支持域分析','',
            f"Q_domain={s['question1_Q_domain_range'][0]:.12g}–{s['question1_Q_domain_range'][1]:.12g}，数值尺度位于 B6 [0.1,1] 支持范围内。",
            'question1_Q_interval_analysis.csv 在每个 B6 ND 工作点给出冻结领域 Q 坐标、两区间端点的质量敏感度和相对 Q=1 的额外 Loss。所有候选均保留，未通过者结果仅作诊断；没有把领域坐标变成配比预测。',
            '未证明问题一 CRITIC Q 与 B6 Q_score 具有完全相同语义，不能把数值落入支持域当作语义标定。','',
            '## 无 offset 主模型与最终来源敏感性','']
    lines+=table(s['offset_sensitivity'],['model','offset','B6_absolute_RMSE','Q1_RMSE','main_parameters','offset_parameters','relative_changes'])
    lines+=['','offset 版本只在全部主候选评估之后，用 B6 单独拟合 50 初值，未参与接受判断或修复 B7 差分。全 Q offset 可以吸收函数形状误差，不能将其全部解释为真实来源零点偏差。','',
            '## B8 不兼容压力检验','']
    lines+=table(s['B8_incompatibility'],['subset','n','n_groups_with_correlation','positive_group_rate','spearman_median','loss_below_E_count','loss_below_E_rate'])
    lines+=['']+table([r for r in s['B8_pressure_performance'] if r['subset'] in ['full','within_B6_numeric_ranges','outside_B6_numeric_ranges']],['subset','model','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias'])
    lines+=['','模型的预测 Q 方向全部为下降，B8 组内实际方向为上升，包括 B6 数值范围内的工作点。数据未翻转或删除，低于冻结 E 的记录全部保留。',
            '维持结论：B8 与 B1+B6/B7 的质量标度机制不兼容，仅作为适用边界的半合成压力检验，不判定数据错误，也不猜测 Q 定义反了。','',
            '## 接受门槛与最终选择','']
    lines+=table(s['model_acceptance'],['model','accepted','B6_absolute_CV_improves_M0','B6_DeltaL_CV_positive_R2','B7_absolute_improves_M0',
                                          'B7_DeltaL_positive_R2','B7_DeltaL_positive_correlations','bootstrap_without_failure_or_boundary_flags','strictly_decreasing_Q'])
    lines+=['','absolute 稳定性用相同验证子集上 RMSE 优于 M0 检查；并同时呈现偏差和相关，不使用人为追加的“理想误差阈值”。Bootstrap 无失败/触边标记是保守审核条件，边界单侧推断须单独审查。',
            '预设优先级 S3→S1→S2；不按 B7 最佳误差重新调参。对参数触边和实用不可识别，不能只看数值 Jacobian 满秩或漂亮的窄 CI。', '']
    if s['all_B7_DeltaL_failed']:
        lines+=['**最终不冻结 L(N,D,Q)。**三个候选在 B7 非重叠差分上均 R²≤0，未达到用户规定的必要门槛。',
                '“附件 B 当前不足以支持统一可迁移的 N-D-Q 函数形式。”此判断限定于本轮候选形式、冻结经典骨架和验证准则，不能推演成不存在任何质量效应或所有数学形式都不可能。',
                'S3 是最简单且 B6 内部表现较好的诊断对照，但没有通过扩展集复核，不应为了达到冻结目标继续堆高阶模型。']
    elif s['selected_model']:
        lines += [f"按预设规则通过的最简单候选为 {s['selected_model']}；仍需人工确认冻结，仅限半合成支持域。"]
    else:lines += ['没有模型同时通过全部条件；不冻结，停止进一步增加复杂函数。']
    lines+=['','## 复现、输出与冻结核验','',
            '运行 B6 计算：`python -B scripts/14_fit_scale_dependent_quality.py --compute`。',
            '生成固定参数验证与报告：`python -B scripts/14_fit_scale_dependent_quality.py --finalize`。',
            '中断续算：`--compute --reuse-basics --resume-bootstrap`，复用完整检查点，保持抽样序列和每次 50 初值不变。',
            '只读独立复核：`python -B scripts/14_verify_scale_dependent_quality.py`。',
            '表格输出 outputs/scaling/phase2C/，图 outputs/figures/question2_phase2C/。既有结果没有覆写，新代码使用 pathlib 从脚本位置识别根目录。',
            f"保护本阶段开始时的 {s['protected_files']} 个文件，包括官方附件和已确认全部结果；结束 SHA-256 核验。",
            '本阶段停止，等待人工确认。未引入配比 p 或冲突 K，未使用 B9/B10，未推导参数替代量或进入问题三。']
    (root/'outputs/report/question2_phase2C_scale_dependent_quality.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
