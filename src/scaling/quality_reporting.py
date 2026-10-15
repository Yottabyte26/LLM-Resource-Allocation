"""Figures and a numerically generated report with explicit evidence boundaries."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .quality_extension import predict_quality,group_reference,subset


def save(fig, folder, name, dpi):
    """Save a standalone scientific figure and release its memory."""
    fig.tight_layout();fig.savefig(folder/name,dpi=dpi);plt.close(fig)


def obs_pred(ax, y, yp, title, q=None):
    """Observed/predicted scatter with common limits and identity line."""
    s=ax.scatter(y,yp,c=q,s=17,alpha=.7,cmap='viridis' if q is not None else None)
    lo=min(float(np.min(y)),float(np.min(yp)));hi=max(float(np.max(y)),float(np.max(yp)))
    margin=(hi-lo)*.04 or .01
    ax.plot([lo-margin,hi+margin],[lo-margin,hi+margin],'k--',lw=1)
    ax.set(xlabel='Observed Loss',ylabel='Predicted Loss',title=title,
           xlim=(lo-margin,hi+margin),ylim=(lo-margin,hi+margin))
    return s


def make_figures(root,ts,p,fits,oof,comparison,layers,brows,masks7,masks8,predictions,diffpoints,cfg):
    """Re-use prior coverage diagnosis and add all fit/validation evidence plots."""
    folder=root/'outputs/figures/question2_phase2';folder.mkdir(parents=True,exist_ok=True)
    dpi=cfg['figure_dpi'];b6=ts['B6'];b7=ts['B7'];b8=ts['B8']
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    Ns=np.unique(b6['N']);Ds=np.unique(b6['D'])
    for ax,(n,d) in zip(axes.flat,[(Ns[0],Ds[0]),(Ns[0],Ds[-1]),(Ns[-1],Ds[0]),(Ns[-1],Ds[-1])]):
        ix=(b6['N']==n)&(b6['D']==d);a=subset(b6,ix);order=np.argsort(a['Q'])
        ax.plot(a['Q'][order],a['L'][order],'ko',label='B6 observations')
        fine=dict(N=np.full(100,n),D=np.full(100,d),Q=np.linspace(b6['Q'].min(),b6['Q'].max(),100))
        for model in ['M0','MQ1','MQ2']:
            f=fits.get(model,dict(gamma=0.,offset=0.))
            ax.plot(fine['Q'],predict_quality(fine,p,model,f['gamma'],f['offset']),label=model)
        ax.set(xlabel='Q_score',ylabel='Validation Loss',title=f'N={n:g} B, D={d:g} B');ax.legend(fontsize=8)
    save(fig,folder,'04_B6_fixed_ND_quality_curves.png',dpi)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,metric in zip(axes,['RMSE','MAE']):
        ax.bar([r['model'] for r in comparison],[r[metric] for r in comparison],color=['gray','tab:blue','tab:orange','lightblue','wheat'])
        ax.set(ylabel=metric+' (raw Loss)',title='B6: five-fold ND Group CV');ax.tick_params(axis='x',rotation=25)
    save(fig,folder,'05_B6_group_CV_comparison.png',dpi)
    fig,ax=plt.subplots(figsize=(6,5));s=obs_pred(ax,b6['L'],oof['MQ1'],'MQ1: B6 held-out ND groups',b6['Q']);fig.colorbar(s,ax=ax,label='Q_score')
    save(fig,folder,'06_MQ1_B6_observed_predicted_OOF.png',dpi)
    obs,pred,_,refs=group_reference(b6,oof['MQ1'])
    fig,ax=plt.subplots(figsize=(6,5));s=obs_pred(ax,obs[~refs],pred[~refs],'MQ1: B6 OOF within-ND differences',b6['Q'][~refs])
    ax.set(xlabel='Observed Delta Loss (vs Q=1)',ylabel='Predicted Delta Loss (vs Q=1)');fig.colorbar(s,ax=ax,label='Q_score')
    save(fig,folder,'07_within_ND_difference_OOF.png',dpi)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,model in zip(axes,['MQ1','MQ2']):
        for m in [model,model+'_offset']:
            v=[r['gamma'] for r in brows if r['model']==m and r['success']]
            ax.hist(v,bins=35,alpha=.55,label=m)
        ax.set(xlabel='gamma',ylabel='ND-block bootstrap count',title=model+': 2000 block resamples');ax.legend()
    save(fig,folder,'08_gamma_block_bootstrap.png',dpi)
    for axis,name in [('N','09_gamma_by_N.png'),('D','10_gamma_by_D.png')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for m in ['MQ1','MQ1_offset']:
            r=[x for x in layers[axis] if x['model']==m]
            ax.plot([x['level'] for x in r],[x['gamma'] for x in r],'o-',label=m)
        ax.set(xscale='log',xlabel=axis+(' (billion parameters)' if axis=='N' else ' (billion tokens)'),ylabel='Layer-specific gamma',title='B6: gamma by '+axis);ax.legend()
        save(fig,folder,name,dpi)
    ix=masks7['nonoverlap'];fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,m in zip(axes,['MQ1','MQ2']):
        s=obs_pred(ax,b7['L'][ix],predictions['B7'][m][ix],m+': B7 non-overlap (n='+str(int(ix.sum()))+')',b7['Q'][ix]);fig.colorbar(s,ax=ax,label='Q_score')
    save(fig,folder,'11_B7_nonoverlap_predictions.png',dpi)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,name in zip(axes,['within_B6_numeric_ranges','outside_B6_numeric_ranges']):
        ix=masks8[name];s=obs_pred(ax,b8['L'][ix],predictions['B8']['MQ1'][ix],name.replace('_',' ')+f' (n={ix.sum()})',b8['Q'][ix]);fig.colorbar(s,ax=ax,label='Q_score')
        ax.axvline(p['E'],color='tab:red',ls=':',label='Frozen E');ax.legend(fontsize=8)
    fig.suptitle('B8 incompatible semisynthetic pressure test (no fitting)',fontsize=12)
    save(fig,folder,'12_B8_inside_outside_pressure_test.png',dpi)
    from .io import csv_rows
    _,rows=csv_rows(root/'outputs/scaling/phase2/leave_one_Q_level_out.csv')
    fig,ax=plt.subplots(figsize=(7,4))
    for m in ['MQ1','MQ2']:
        r=[x for x in rows if x['model']==m]
        ax.plot([float(x['held_Q']) for x in r],[float(x['RMSE']) for x in r],'o-',label=m)
    ax.set(xlabel='Held-out Q level',ylabel='RMSE (raw Loss)',title='B6: leave-one-Q-level-out; endpoints extrapolate');ax.legend()
    save(fig,folder,'13_leave_one_Q_level_out.png',dpi)
    fig,ax=plt.subplots(figsize=(8,4));ix=b6['Q']==1
    for m in ['M0','MQ1_offset','MQ2_offset']:
        ax.hist(b6['L'][ix]-predictions['B6'][m][ix],bins=16,alpha=.5,label=m)
    ax.axvline(0,color='black',lw=1);ax.set(xlabel='Observed - predicted Loss at Q=1',ylabel='ND-group count',title='B6 source-offset anchor sensitivity');ax.legend()
    save(fig,folder,'14_source_offset_anchor_sensitivity.png',dpi)
    fig,ax=plt.subplots(figsize=(7,4));ix=(b8['N']==.07)&(b8['D']==50)&masks8['within_B6_numeric_ranges']
    if ix.sum()<2:
        g=b8['group'][np.flatnonzero(masks8['within_B6_numeric_ranges'])[0]];ix=(b8['group']==g)&masks8['within_B6_numeric_ranges']
    idx=np.flatnonzero(ix);idx=idx[np.argsort(b8['Q'][idx])]
    ax.plot(b8['Q'][idx],b8['L'][idx],'o-',label='B8 observations (unchanged)')
    ax.plot(b8['Q'][idx],predictions['B8']['MQ1'][idx],'o-',label='Fixed B6 MQ1 prediction')
    ax.set(xlabel='Q_score',ylabel='Loss',title=f'B8 inside numeric ranges: N={b8["N"][idx[0]]:g} B, D={b8["D"][idx[0]]:g} B');ax.legend()
    save(fig,folder,'15_B8_within_range_direction_reversal.png',dpi)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,m in zip(axes,['MQ1','MQ2']):
        obs,pred,_,refs=group_reference(b7,predictions['B7'][m]);ix=masks7['nonoverlap']&~refs
        s=obs_pred(ax,obs[ix],pred[ix],m+': B7 new-Q quality differences',b7['Q'][ix])
        ax.set(xlabel='Observed Delta Loss (vs Q=1)',ylabel='Predicted Delta Loss (vs Q=1)');fig.colorbar(s,ax=ax,label='New Q level')
    save(fig,folder,'16_B7_nonoverlap_quality_differences.png',dpi)


def table(rows, columns, names=None):
    """Compact markdown table with sufficient displayed precision."""
    names=names or columns
    text=['|'+'|'.join(names)+'|','|'+'|'.join(['---']*len(columns))+'|']
    for r in rows:
        vals=[]
        for c in columns:
            v=r.get(c)
            vals.append(f'{v:.8g}' if isinstance(v,float) else ('n.a.' if v is None else str(v)))
        text.append('|'+'|'.join(vals)+'|')
    return text


def make_report(root,s,bs,diffrows,anchors):
    """Report the final evidence chain, including form/offset limitations."""
    p=s['classical_parameters'];fits=s['fit_models'];g=fits['MQ1']['gamma'];eta=p['beta']*g
    ci=next(r for r in bs if r['model']=='MQ1' and r['parameter']=='gamma')
    out=root/'outputs/scaling/phase2'
    lines=['# 问题二 PHASE 2B：质量 Q 标度律与适用边界','',
           '状态：本轮实验完成，等待人工确认。经典五参数和问题一全部结果保持冻结。', '',
           ('结论：本轮不建议冻结 L(N,D,Q)。MQ1 明显改善绝对 Loss 的 ND 分组预测，但 B7 新增 Q 水平上的组内差分预测未通过，且统一 γ 存在明显 N/D 层间变化。' if not s['freeze_recommended'] else '结论：必要证据通过，可提交支持域内近似质量模型供人工确认。'),
           f"MQ1 在 B7_nonoverlap 的绝对 Loss R²={s['B7_primary_absolute']['R2']:.8g}，组内差分 R²={s['B7_primary_difference']['R2']:.8g}、平均低估量={s['B7_primary_difference']['mean_bias']:.8g}。应以组内差分作为质量效应量复核，不能用跨 N/D 的绝对预测相关代替。", '',
           '## 证据层级与来源','',
           'B6 为唯一 γ 拟合与模型选择数据。B7 的 90 条非重叠记录为主要扩展验证，360 条重叠记录和 450 条全量结果仅作描述。B8 单列为“大范围半合成不兼容压力测试/负面证据”。',
           'B6–B8 统一称“基于真实数据校准的半合成补充实验”，不得据此宣称真实直接训练的跨源普适定律。',
           '主键使用 Decimal 精确规范化的 (N,D,Q)，不以原始字符串格式区别 0.07 和 0.070。诊断和重叠表沿用已确认结果。',
           'B7_nonoverlap 排除 B6 主键；B8_nonoverlap 排除 B6∪B7 主键。B7 新增 Q 为 0.5、0.7，ND 工作点仍与 B6 共享，因此主要检验新 Q 水平而非新 ND 外部泛化。', '',
           '## 冻结基线与有效数据量假设','',
           '```text',f"E={p['E']:.17g}\nA={p['A']:.17g}\nB={p['B']:.17g}\nalpha={p['alpha']:.17g}\nbeta={p['beta']:.17g}",'```','',
           r'\(L_0=E+A N^{-\alpha}+B D^{-\beta}\)，N 与 D 使用 billions 单位，拟合原始 Loss。',
           r'主候选 MQ1：\(D_{eff}=D Q^\gamma\)，\(L=E+A N^{-\alpha}+B D^{-\beta}Q^{-\beta\gamma}\)。',
           r'函数形式敏感性 MQ2：\(D_{eff}=D\exp[\gamma(Q-1)]\)，数据项为 \(B D^{-\beta}\exp[\beta\gamma(1-Q)]\)。',
           '两者均 γ>0，无 offset 时在 Q=1 精确退化为 L0。不得从拟合成功推断 D_eff 是已被真实训练实验证明的因果有效 token 数。',
           'source offset 版本仅作为来源/函数形式混杂敏感性，不替换无 offset 主结果。', '',
           '## 拟合、交叉验证与不确定性协议','',
           'bounded nonlinear least squares；γ 初值 0.1、0.25、0.5、1、2、4、8，每次比较全部合法成功解的原始 Loss MSE。计算边界 γ∈[1e-8,50]，offset∈[-10,10]；边界不是数据拟合结论，须检查是否触边。',
           '每个原始观测等权。完整网格中每个 ND 组均有 8 个 Q，因此各 ND 总权也相等。Bootstrap 对 45 个 ND 块有放回采样，整块所有 Q 带同一重数。',
           '5-fold GroupKFold 按精确 ND 分组，同一 ND 的全部 Q 在同一折。各训练折独立使用七个预设初值，不使用全量拟合参数作初值。',
           'leave-one-Q-level-out 每次整层 Q 留出。边界 0.1/1 为 Q 外推敏感性，其余为插值。',
           '2000 次 ND block bootstrap，固定 seed=2026；四个候选版本采用同一批整组重采样。置信区间条件于冻结的经典参数、固定的 Q 网格和半合成来源；不包含 PHASE 1 β 不确定性或真实跨源误差。',
           'OOF RMSE 改善区间来自固定 OOF 预测的 ND 重采样，不是完整嵌套 CV 重拟合置信区间，未使用 IID 点级显著性检验。', '',
           '## B6 质量方向与局部噪声','',
           f"45 组负相关比例 {s['B6_monotonicity']['negative_correlation_group_rate']:.2%}；严格逐点下降比例 {s['B6_monotonicity']['strictly_monotone_group_rate']:.2%}。",
           f"共 {s['B6_monotonicity']['local_violation_count']} 个相邻反弹，平均幅度 {s['B6_monotonicity']['mean_rebound']:.8g}，最大 {s['B6_monotonicity']['max_rebound']:.8g}。局部反弹不等于总体质量方向错误，也不据此更改 γ 符号。每组相关与幅度见 B6_monotonicity_details.csv。", '',
           '## B6 主参数与 Bootstrap','',
           f'无 offset MQ1 γ={g:.12g}，95% CI=[{ci["P025"]:.12g},{ci["P975"]:.12g}]；η=βγ={eta:.12g}。', '']
    lines+=table([dict(model=m,gamma=f['gamma'],eta=p['beta']*f['gamma'] if m.startswith('MQ1') else None,
                       offset=f['offset'],fit_RMSE=np.sqrt(f['objective']),n_starts=f['n_starts'],gamma_spread=f['distinct_solution_gamma_range']) for m,f in fits.items()],
                 ['model','gamma','eta','offset','fit_RMSE','n_starts','gamma_spread'])
    lines+=['']+table(bs,['model','parameter','mean','median','std','P025','P975','failure_rate','at_bound_count'])
    lines+=['','## B6 按 ND 分组的预测性能','']
    lines+=table(s['B6_group_cv'],['model','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias'])
    lines += ['',f"MQ1 相对 M0 的 Group-CV RMSE 降低 {s['CV_RMSE_gain']:.2%}。仅按 B6 CV、无 offset 两种函数形式的数值赢家为 {s['B6_numeric_form_winner']}；MQ1 仍按用户指定保留为主候选，MQ2 为形式敏感性，不依据 B7/B8 更换主模型。",'']
    lines+=table(s['bootstrap_CV_gains'],['comparison','mean','P025','P975','fraction_positive'])
    lines+=['','## Source offset 与 Q=1 锚点','']
    lines+=table(s['source_offset'],['model','gamma_no_offset','gamma_with_offset','gamma_relative_change','delta_B6'])
    lines+=['']+table(anchors,['model','n','RMSE','MAE','mean_bias','median_residual'])
    lines+=['','offset 对 γ 的影响应结合数值变化和锚点表现解释。全 Q 拟合的 offset 不必等于 Q=1 平均偏差；若其改善全量损失却恶化 Q=1 锚点，它也可能吸收有效数据量函数的形状误差，不能将其全部解释成纯来源零点偏移。',
            '本轮保留无 offset 主结果，同时报告“质量效应与来源偏差存在部分混杂风险”；不能仅因均值锚点偏差小就省略敏感性。', '',
            '## 组内差分：质量效应关键复核','',
            r'B6/B7 均以每个 ND 的 Q=1 为参考。\(\Delta\hat L=B D^{-\beta}(Q^{-\beta\gamma}-1)\)，使用同一 γ，不单独拟合差分。E 和 N 项精确抵消。',
            '主差分统计排除参考点自身的零差分；含零参考点的结果保留为辅助表。B6 同时报告全量拟合和 held-ND OOF 差分。',
            'B7_nonoverlap 的 Q=1 实测参考来自全量 B7、与 B6 重叠。这是组内变化验证，参考并非新增独立观测；不把共享参考包装成独立外部实验。', '']
    selected=[r for r in diffrows if r['model'] in ['M0','MQ1','MQ2'] and not r['include_reference_zero_rows'] and (r['dataset']=='B6' and r['evaluation']=='B6_GroupCV_out_of_fold' or r['dataset']=='B7' and r['subset']=='nonoverlap')]
    lines+=table(selected,['dataset','subset','evaluation','model','n','RMSE','MAE','R2','Pearson','Spearman'])
    b7q2=next(r for r in diffrows if r['dataset']=='B7' and r['subset']=='nonoverlap' and r['model']=='MQ2' and not r['include_reference_zero_rows'])
    lines+=['',f"B7 非重叠差分 MQ1 R²={s['B7_primary_difference']['R2']:.8g}、MQ2 R²={b7q2['R2']:.8g}，均未优于该差分子集的均值预测。MQ2 的 B6 数值优势不能据此变成质量模块已通过扩展验证的结论。"]
    lines+=['','正 R² 表示差分预测优于均值差分，但不足以证明零误差或普遍因果规律。误差大小、形式差异与共享参考依赖需同时保留。', '',
            '## B7 扩展验证：固定全部 B6 参数','']
    lines+=table(s['B7_validation'],['subset','model','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias'])
    lines+=['','模型和参数在读入本轮 B7/B8 评价数据前已写入 parameters_before_B7_B8.json；未用验证结果调参。', '',
            '## 未见 Q 水平泛化','']
    lines+=table([r for r in s['leave_one_Q_level_out'] if r['model'] in ['MQ1','MQ2']],['held_Q','Q_test_kind','model','gamma','RMSE','MAE','mean_bias'])
    endpoints=[r for r in s['leave_one_Q_level_out'] if r['model']=='MQ1' and r['held_Q'] in [.1,1.]]
    interior=[r['RMSE'] for r in s['leave_one_Q_level_out'] if r['model']=='MQ1' and r['held_Q'] not in [.1,1.]]
    lines+=['',f"MQ1 内部 Q 层 RMSE 均值 {np.mean(interior):.8g}；端点结果分别为 "+'、'.join(f"Q={r['held_Q']:g}: {r['RMSE']:.8g}" for r in endpoints)+'。端点误差需独立审查，不能只汇总全层平均。', '',
            '## γ 对 N/D 的稳定性诊断','']
    lines+=table(s['gamma_scale_dependence'],['axis','model','n_layers','gamma_min','gamma_max','gamma_std','Pearson_gamma_logscale','Spearman_gamma_logscale'])
    lines+=['','按层估计的幅度差异与对数规模相关均为描述性证据，只有 9 个 N 层和 5 个 D 层，不能仅凭相关符号宣称显著因果尺度依赖。offset 对层参数的变化有助于显示混杂。若 γ 层间变化明显，只标记统一 γ 是支持域平均近似，本阶段不添加复杂交互。', '',
            '## B8：大范围半合成不兼容压力测试','']
    lines+=table(s['B8_incompatibility'],['subset','n','n_groups_with_correlation','positive_group_rate','spearman_min','spearman_median','spearman_max','loss_below_E_rate'])
    lines+=['']+table(s['B8_overlap_loss_differences'],['comparison','n','mean','median','std','min','max','mean_abs_difference'])
    lines+=['']+table([r for r in s['B8_pressure_performance'] if r['model'] in ['M0','MQ1','MQ2']],['subset','model','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias'])
    lines+=['','数值范围内/外按 B6 的 N、D、Q 各自闭区间判定，范围内不等于已完成数据生成语义标定。范围内仍正相关时，不兼容不能归因于普通数值外推。',
            '“B8 与由 B1+B6 建立的统一 N-D-Q 标度关系不兼容，故不纳入主质量参数估计，仅作为广义标度律适用边界的压力检验。”',
            '未翻转 Q、未允许负 γ、未修改经典参数、未删除 Loss<E 的记录，未将不兼容定性为数据错误或猜测 Q 定义反了。', '',
            '## 问题一 Q 的尺度兼容性','',
            f"领域 Q_domain 范围 {s['question1_Q_domain_range'][0]:.12g}–{s['question1_Q_domain_range'][1]:.12g}，数值尺度位于 B6 支持范围内 [0.1,1]。",
            'Q_domain 与 B6 Q_score 来自不同构造过程，未证明两种 Q 具有完全相同语义；本轮不重新读取原始质量 JSONL/XZ，也不把领域 Q 代入配比模型。', '',
            '## 冻结判断与停止点','']
    lines+=table([dict(condition=k,passes=v) for k,v in s['freeze_evidence_gates'].items()],['condition','passes'])
    lines+=['',('四项必要证据门槛均通过，可提交“仅在 B6/B7 支持域内有效”的 MQ1 近似模型供人工冻结确认。' if s['freeze_recommended'] else '必要证据门槛未全部通过，当前不建议冻结主模型。'),
            '该判断不抹去 MQ2 的函数形式差异、offset 混杂风险和层间 γ 变化，不表示真实跨源普适性已经验证。适用域 N∈[0.07,11.97] B、D∈[10,600] B、Q∈[0.1,1]，以及 B6/B7 半合成构造机制。',
            '当前未执行人工冻结，未引入 p/K，未使用 B9/B10，未推导 Q+0.1 参数替代关系、最终弹性或开始问题三。', '',
            '## 输出与复现','',
            '执行：`python -B scripts/13_fit_quality_scaling.py`；只读复核：`python -B scripts/13_verify_quality_scaling.py`。',
            '配置：configs/scaling_quality_phase2B.yaml。主要统计位于 outputs/scaling/phase2，图表位于 outputs/figures/question2_phase2。',
            f"保护本轮开始时的 {s['protected_existing_files']} 个既有文件，结束逐文件 SHA-256 核验。既有诊断报告保留，新报告为 question2_phase2B_quality_scaling.md。"]
    path=root/'outputs/report/question2_phase2B_quality_scaling.md'
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
