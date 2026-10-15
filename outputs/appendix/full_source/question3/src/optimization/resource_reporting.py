"""Question 3 optimized results, readable evidence limits and scientific figures."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .resource_solver import load,csv,ResourceProblem


def mdtable(frame,columns=None):
    """Readable Markdown with bounded display precision; CSV retains full precision."""
    f=frame if columns is None else frame[columns]
    lines=['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']
    for r in f.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.7g}' if isinstance(v,float) and np.isfinite(v) else '—' if pd.isna(v) else str(v).replace('|','/') for v in r)+' |')
    return '\n'.join(lines)


def figures(root,out,path,official,compare,hard,soft,shadow):
    """Twelve figures, consistently distinguishing main/extrapolation and infeasibility."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageOps,ImageDraw
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figdir=root/'outputs/figures/question3_phase2';figdir.mkdir(parents=True,exist_ok=True)
    colors={'exponential':'#4477AA','power':'#CC6677','logarithmic':'#228833'};families=list(colors);manifest=[]
    def finish(fig,name,caption):
        fig.tight_layout();target=figdir/name;fig.savefig(target,dpi=180);plt.close(fig)
        manifest.append(dict(figure=target.relative_to(root).as_posix(),caption=caption))
    feasible=path[path.feasible==True].copy();main=feasible[feasible.domain=='main']
    for i,(metric,label) in enumerate([('N_star','N* (billion parameters)'),('D_star','D* (billion tokens)'),
                                     ('Q_star','Q*'),('L_star','Optimal validation loss')],1):
        fig,ax=plt.subplots(figsize=(8,4))
        for family in families:
            for domain,style in [('main','-'),('extended','--')]:
                f=feasible[(feasible.cost_function==family)&(feasible.L_ctx==2048)&(feasible.domain==domain)].sort_values('budget')
                ax.plot(f.budget,f[metric],color=colors[family],ls=style,label=family+' '+domain)
        ax.set(xscale='log',xlabel='Compute budget (FLOPs)',ylabel=label,title='P0 allocations, context 2048')
        if metric in ('N_star','D_star'):ax.set_yscale('log')
        ax.legend(fontsize=8,ncol=2);finish(fig,f'{i:02d}_{metric}_vs_budget.png',f'P0 {metric}：主域实线、扩展域虚线；Lctx=2048')
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,family in zip(axes,families):
        f=main[(main.cost_function==family)&(main.L_ctx==2048)].sort_values('budget')
        for m,color in [('s_train','#4477AA'),('s_quality','#CC6677'),('s_attn','#228833')]:ax.plot(f.budget,f[m],label=m,color=color)
        ax.plot(f.budget,1-f.s_train-f.s_quality-f.s_attn,ls=':',color='gray',label='unused fraction')
        ax.set(xscale='log',xlabel='Budget (FLOPs)',title=family,ylim=(-.02,1.02))
    axes[0].set_ylabel('Cost / allocated budget');axes[-1].legend(fontsize=8)
    fig.suptitle('P0 main-domain cost shares, context 2048');finish(fig,'05_cost_shares.png','预算分母下三类成本与未使用份额')
    fig,ax=plt.subplots(figsize=(8,4))
    for family in families:
        f=shadow[(shadow.domain=='main')&(shadow.cost_function==family)&(shadow.L_ctx==2048)].sort_values('budget')
        f=f[f.shadow_price>1e-30];ax.plot(f.budget,f.shadow_price,color=colors[family],label=family)
    ax.set(xscale='log',yscale='log',xlabel='Budget (FLOPs)',ylabel='Shadow price (-dLoss*/dC)',title='P0 shadow prices; zero plateau omitted from log axis')
    ax.legend();finish(fig,'06_shadow_price.png','有限差分影子价格；饱和后的零值不画在对数轴上')
    fig,axes=plt.subplots(1,5,figsize=(15,3.5),sharey=True)
    for ax,l in zip(axes,[2048,4096,8192,32768,131072]):
        for family in families:
            f=main[(main.cost_function==family)&(main.L_ctx==l)].sort_values('budget')
            ax.plot(f.budget,f.Q_star,label=family,color=colors[family])
        ax.set(xscale='log',xlabel='Budget (FLOPs)',title=f'Context {l}',ylim=(.70,1.015));ax.tick_params(axis='x',labelsize=8)
    axes[0].set_ylabel('Q*');axes[-1].legend(fontsize=7)
    finish(fig,'07_quality_cost_function_comparison.png','五上下文下三种官方成本的质量投资路径')
    for index,metric,label in [(8,'N_star','N* (billion parameters)'),(9,'D_star','D* (billion tokens)'),(10,'L_star','Loss*')]:
        fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
        for ax,family in zip(axes,families):
            for l in [2048,4096,8192,32768,131072]:
                f=main[(main.cost_function==family)&(main.L_ctx==l)].sort_values('budget')
                ax.plot(f.budget,f[metric],label=str(l))
            ax.set(xscale='log',xlabel='Budget (FLOPs)',title=family)
            if metric in ('N_star','D_star'):ax.set_yscale('log')
        axes[0].set_ylabel(label);axes[-1].legend(title='Context',fontsize=8)
        finish(fig,f'{index:02d}_five_context_{metric}.png','主域五上下文对照：'+metric)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,C in zip(axes,[1e19,1e22,1e24]):
        for family in families:
            f=compare[(compare.domain=='main')&(compare.cost_function==family)&(compare.budget==C)&(compare.status=='SOLVED')].sort_values('L_ctx')
            ax.plot(f.L_ctx.astype(str),f.Loss_improvement,'o-',color=colors[family],label=family)
        ax.set(xlabel='Context',title=f'Budget {C:.0e}');ax.tick_params(axis='x',labelrotation=40)
    axes[0].set_ylabel('P0 Loss* - P1 Loss*');axes[-1].legend(fontsize=8)
    fig.suptitle('P1 modeled improvement; large-N transfer is conditional/estimated')
    finish(fig,'11_P0_vs_P1_improvement.png','P1相对P0的模型Loss改善，非新观测实验收益')
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    labels=['active_D_lower','active_Q_lower','active_Q_upper','active_D_upper','active_N_upper','budget_binding']
    for ax,family in zip(axes,families):
        h=hard[(hard.domain=='main')&(hard.cost_function==family)&(hard.L_ctx==2048)]
        for i,label in enumerate(labels):
            g=h[h.constraint==label];ax.scatter(g.budget_critical,np.full(len(g),i),s=40)
            for r in g.itertuples():ax.annotate(r.transition,(r.budget_critical,i),xytext=(3,5),textcoords='offset points',fontsize=7)
        s=soft[(soft.domain=='main')&(soft.cost_function==family)&(soft.L_ctx==2048)&(soft.significant==True)]
        for v in s.break_budget.dropna().unique():ax.axvline(v,ls=':',color='gray',alpha=.3)
        ax.set(xscale='log',yticks=range(len(labels)),yticklabels=labels,title=family)
    axes[-1].set_xlabel('Budget (FLOPs); dots: Type I, dotted lines: significant Type II curve fits')
    finish(fig,'12_structural_transitions.png','2048上下文硬转移与显著软断点示意；软线不是额外独立物理相变')
    csv(out/'figure_manifest.csv',manifest)
    sheet=Image.new('RGB',(1600,2040),'white')
    for i,r in enumerate(manifest):
        im=Image.open(root/r['figure']).convert('RGB');thumb=ImageOps.contain(im,(785,310))
        x=(i%2)*800;y=(i//2)*340;sheet.paste(thumb,(x,y));ImageDraw.Draw(sheet).text((x+5,y+315),Path(r['figure']).name,fill='black')
    sheet.save(out/'figure_contact_sheet.png')
    return manifest


def run_report(root):
    """Write every requested result from saved numerical tables, with scope limits."""
    root=Path(root);out=root/'outputs/optimization/phase2';setup,frozen,cfg,model,candidates,ext=load(root)
    from .transitions import complete_feasibility_boundary_events
    complete_feasibility_boundary_events(root)
    path=pd.concat([pd.read_csv(out/f'P0_{d}_budget_path.csv',float_precision='round_trip') for d in ('main','extended')],ignore_index=True)
    read=lambda name:pd.read_csv(out/name,float_precision='round_trip')
    official=read('official_budget_results.csv');comparison=read('P0_vs_P1_official_comparison.csv')
    hard=read('typeI_active_set_transitions.csv');soft=read('typeII_soft_transitions.csv');shadow=read('shadow_price_budget_elasticity.csv')
    saturation=read('domain_saturation_budgets.csv');peaks=read('quality_cost_share_peaks.csv');q0=read('Q0_sensitivity_results.csv')
    optimizer=read('optimizer_cross_checks.csv');full3d=read('D_elimination_3D_check.csv');enumeration=read('P1_candidate_reduction_check.csv')
    supports=[]
    for event_type,events in [('Type I',hard.to_dict('records')),
                             ('Type II',soft[soft.significant==True].to_dict('records'))]:
        for event in events:
            C=event['budget_critical'] if event_type=='Type I' else event['break_budget']
            g=shadow[(shadow.domain==event['domain'])&(shadow.cost_function==event['cost_function'])&(shadow.L_ctx==event['L_ctx'])].sort_values('budget')
            before=g[g.budget<C];after=g[g.budget>C]
            if before.empty or after.empty:continue
            left,right=before.iloc[-1],after.iloc[0]
            supports.append(dict(domain=event['domain'],cost_function=event['cost_function'],L_ctx=event['L_ctx'],
                event_type=event_type,event=event.get('constraint',event.get('metric')),transition_budget=C,
                neighbor_budget_left=left.budget,neighbor_budget_right=right.budget,
                mu_left=left.shadow_price,mu_right=right.shadow_price,
                elasticity_left=left.budget_elasticity,elasticity_right=right.budget_elasticity,
                elasticity_change=right.budget_elasticity-left.budget_elasticity,
                purpose='finite-grid corroboration only, not an independent transition criterion'))
    csv(out/'transition_shadow_support.csv',supports)
    main=official[(official.domain=='main')&(official['mode']=='P0')].sort_values(['budget','cost_function','L_ctx'])
    expanded=official[(official.domain=='extended')&(official['mode']=='P0')].sort_values(['budget','cost_function','L_ctx'])
    contextcompare=[]
    for (family,C),g in main.groupby(['cost_function','budget']):
        left=g[g.L_ctx==8192].iloc[0];right=g[g.L_ctx==32768].iloc[0]
        contextcompare.append(dict(cost_function=family,budget=C,N_8192=left.get('N_star',np.nan),N_32768=right.get('N_star',np.nan),
            D_8192=left.get('D_star',np.nan),D_32768=right.get('D_star',np.nan),Q_8192=left.get('Q_star',np.nan),Q_32768=right.get('Q_star',np.nan),
            Loss_change=right.get('L_star',np.nan)-left.get('L_star',np.nan),ratio_8192=setup['official']['eta']*8192/6,ratio_32768=setup['official']['eta']*32768/6))
    csv(out/'context_8192_vs_32768.csv',contextcompare)
    senscompare=[]
    for r in q0.to_dict('records'):
        match=main[(main.cost_function==r['cost_function'])&(main.L_ctx==r['L_ctx'])&(main.budget==r['budget'])].iloc[0]
        row={k:r[k] for k in ('Q0','budget','cost_function','L_ctx','status')}
        if r['feasible'] and match.feasible:
            row.update(N_change=r['N_star']-match.N_star,D_change=r['D_star']-match.D_star,Q_change=r['Q_star']-match.Q_star,
                Loss_change=r['L_star']-match.L_star,s_train_change=r['s_train']-match.s_train,s_quality_change=r['s_quality']-match.s_quality,
                s_attn_change=r['s_attn']-match.s_attn,
                active_structure_changed=any(bool(r[k])!=bool(match[k]) for k in ['active_N_lower','active_N_upper','active_D_lower','active_D_upper','active_Q_lower','active_Q_upper','budget_binding']))
        senscompare.append(row)
    csv(out/'Q0_sensitivity_comparison.csv',senscompare)
    sensframe=pd.DataFrame(senscompare)
    figs=figures(root,out,path,official,comparison,hard,soft,shadow)
    tablecols=['budget','cost_function','L_ctx','status','N_star','D_star','Q_star','L_star']
    costs=['budget','cost_function','L_ctx','C_train','C_quality','C_attn','unused_budget','budget_binding','evidence_domain_saturated','extrapolative_Q']
    mainsoft=soft[(soft.domain=='main')&(soft.significant==True)];mainhard=hard[hard.domain=='main']
    means=comparison[comparison.domain=='main'].groupby('budget').Loss_improvement.agg(['min','max','mean']).reset_index()
    summary=dict(main_official_P0_rows=len(main),main_official_P0_feasible=int(main.feasible.sum()),
        grid_base_count=setup['config']['budget_log_grid_count'],grid_with_official_count=int(path.budget.nunique()),
        main_typeI_rows=len(mainhard),main_typeII_significant_metric_fits=len(mainsoft),
        main_saturation_budget_min=float(saturation[saturation.domain=='main'].exact_saturation_budget.min()),
        main_saturation_budget_max=float(saturation[saturation.domain=='main'].exact_saturation_budget.max()),
        main_quality_share_reversals=int(peaks[peaks.domain=='main'].quality_share_reversal.sum()),
        main_P1_loss_improvement_min=float(comparison[comparison.domain=='main'].Loss_improvement.min()),
        main_P1_loss_improvement_max=float(comparison[comparison.domain=='main'].Loss_improvement.max()),
        high_budget_main_saturated_count=int(main[main.budget==1e24].evidence_domain_saturated.sum()),
        high_budget_extended_N_or_D_extrapolative_count=int(((expanded[expanded.budget==1e24].extrapolative_N==True)|(expanded[expanded.budget==1e24].extrapolative_D==True)).sum()),
        Q0_active_structure_changed_count=int(sensframe.active_structure_changed.fillna(False).sum()),
        Q0_Loss_change_min=float(sensframe.Loss_change.min()),Q0_Loss_change_max=float(sensframe.Loss_change.max()),
        maximum_optimizer_loss_gap=float(optimizer.DE_minus_final_loss.max()),
        full3D_checks_passed=int(full3d.passed.sum()),enumeration_checks_passed=int(enumeration.passed.sum()),P2_performed=False,Question4_started=False)
    (out/'execution_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    text='# 问题三 PHASE 2 正式资源优化与结构性转移\n\n'
    text+='冻结问题一/二与PHASE 1保持只读。P0为正文主方案，三类官方成本平行运行；P1为经验配方增强，主域和扩展域分开。没有P2 simplex优化或问题四。\n\n'
    text+='## 1 解析消去D与单位\n\n固定N/Q/p/Lctx时，∂L/∂D_B=-beta B D_B^(-beta-1)<0；成本线性为D_B c_D。'
    text+='因此D*=min(D_max,C_budget/c_D)，c_D=N_B(6+etaLctx)10^18+10^9[g(Q)-g(Q0)]。D*<D_min则该N/Q不可行。'
    text+='这使连续优化降为logN/Q二维，没有强设预算等号；D达到上界时允许预算松弛。'
    text+='目标仍用B单位，成本显式转换实际参数/token。新15组Decimal单位核验通过，未重拟合参数。\n\n'
    text+='冻结目标：L_gen=E+A N_B^(-alpha)+B D_B^(-beta)+C_Qloss(N_B/N0_B)^(-u_Q)(-lnQ)+lambda_p(N_B)[M(p)-M_ref]。'
    text+='C_Qloss对应冻结JSON的quality.C_Q字段，只是符号区分，不改系数；它与质量算力成本C_quality不同。\n\n'
    text+='## 2 求解与独立最优性核验\n\n每个光滑分段采用121×81网格及可行边界/预算D上界折线、至少10个不同初值SLSQP，显式评估节点两侧和节点本身。'
    text+='官方以及Q0敏感性可行场景全部使用全域differential_evolution独立复核，再细化其候选。'
    text+=f'最大DE细化与最终Loss差={summary["maximum_optimizer_loss_gap"]:.7g}；完整三维核验{summary["full3D_checks_passed"]}/{len(full3d)}通过，'
    text+=f'832配方全枚举核验{summary["enumeration_checks_passed"]}/{len(enumeration)}通过。'
    text+='全枚举是在每个随机场景的31×23资源点上对832配方逐一评价，验证点态及整个资源网格最优一致，非逐配方重复连续求解。'
    text+='主/扩展路径Loss随预算不恶化的检查见monotonicity_checks.csv；这提供强数值复核，有限多优化器不构成任意非凸问题的形式化全局证明。\n\n'
    text+='## 3 官方低中高预算 主域P0\n\n'+mdtable(main,tablecols)+'\n\n'
    text+='成本/未使用预算均为FLOPs：\n\n'+mdtable(main,costs)+'\n\n'
    text+='最低预算×131072上下文最低成本约2.27247e19 FLOPs，故1e19证据支持域内不可行；不可行行保留状态、没有伪造决策。'
    text+='Q=1超过问题一实际样本质量上界，虽在B6半合成数值范围内，不能称直接验证。\n\n'
    text+='## 4 主域饱和与扩展情景\n\n'
    text+='P0各资源边际Loss均负。主域饱和预算可解析为D_max[(6+etaLctx)N_max+Δg(1)/10^9]10^18；'
    text+='达到此预算后P0取三个上界且剩余预算不再改变冻结目标。不是求解失败，也不能覆盖此结果而偷偷扩大边界。\n\n'
    text+=mdtable(saturation[saturation.domain=='main'],['cost_function','L_ctx','exact_saturation_budget','first_saturated_grid_budget'])+'\n\n'
    text+='以下是独立扩展域P0情景，所有超出主N/D/Q实测范围的点分别标记，scenario_extrapolation始终为True：\n\n'+mdtable(expanded,tablecols)+'\n\n'
    text+=f'1e24主域P0全部{summary["high_budget_main_saturated_count"]}个成本×上下文场景饱和；扩展域对应{summary["high_budget_extended_N_or_D_extrapolative_count"]}个结果在N或D上超出主域。'
    text+='B9元数据、B10估算Loss及总/激活参数差异不能为这些联合最优点提供直接验证。\n\n'
    text+='## 5 三种质量成本与投资节奏\n\n'
    text+='三类公式与参数均保持题给：1e7exp(6Q)、5e9Q^4、2e9ln(1+10Q)，均扣除g(Q0)、乘实际D。'
    text+='报告不按结果是否漂亮选择主成本；具体Q脱离Q0+0.01、达到0.9/0.95/0.99/1的局部二分预算见quality_threshold_budgets.csv。'
    text+='“明显离开”以绝对0.01定义（研究者阈值）；满分判定容差1e-7。所有转移预算是冻结模型条件结果。\n\n'
    representative=read('quality_threshold_budgets.csv');representative=representative[(representative.domain=='main')&(representative.L_ctx==2048)]
    text+=mdtable(representative,['cost_function','threshold','budget_threshold'])+'\n\n'
    text+='## 6 上下文与30k临界值\n\n'+mdtable(read('context_cost_ratio.csv'))+'\n\n'
    text+='C_attn/C_train=etaLctx/6与预算/决策/配方无关。8192→32768时比例从0.27307增至1.09227；跨临界表示成本代理的注意力超过基础训练，不是由预算驱动的两个份额交叉。'
    text+='不能要求N/D/Q全部同向变化：它们存在替代、边界和质量成本效应。官方预算对照：\n\n'+mdtable(pd.DataFrame(contextcompare))+'\n\n'
    text+='## 7 Type I硬转移\n\n活跃集采用1e-6归一化容差，先从44点路径发现变化，再局部log预算二分至括区宽≤1e-5。'
    text+='可行性阈值另由解析最低成本给出。并发N上界与预算松弛可对应同一转移，不能数作两个独立经济机制。'
    text+='代表2048上下文的主域事件：\n\n'+mdtable(mainhard[mainhard.L_ctx==2048],['cost_function','constraint','transition','budget_critical','bracket_low','bracket_high'])+'\n\n'
    text+='完整五上下文及扩展域见typeI_active_set_transitions.csv。数值活跃容差下的精确括区，不表示真实系统可支持同样精度。\n\n'
    text+='## 8 Type II软转移及影子价格\n\n对log10预算拟合单线性与连续单断点hinge，两侧至少5点，BIC参数数目分别2/4；只有DeltaBIC≥10才标显著。'
    text+='300次长度3循环块残差扰动提供断点区间；这是确定性模型曲线拟合/网格与近似误差的区间，不是现实转移预算的统计置信区间。'
    text+=f'主域有{len(mainsoft)}条指标曲线满足门槛；不能当作{len(mainsoft)}个独立物理转移。'
    text+='训练与注意力份额固定成比例，其BIC与断点重复，不是独立证据；单断点也只是对含多次边界变化曲线的近似。\n\n'
    text+=mdtable(mainsoft[mainsoft.L_ctx==2048],['cost_function','metric','delta_BIC','break_budget','break_interval_low','break_interval_high'])+'\n\n'
    text+='mu=-dL*/dC以及-E的剩余Loss预算弹性由相邻预算有限差分计算，见shadow_price_budget_elasticity.csv。'
    text+='Type I/II附近左右预算点的mu与弹性变化另列transition_shadow_support.csv，用于逐个事件核对；'
    text+='节点附近有限网格导数只作佐证；饱和后的影子价格应为零。不能单凭mu或弹性定义转移。\n\n'
    text+='## 9 质量成本份额反转\n\n'+mdtable(peaks[peaks.domain=='main'],['cost_function','L_ctx','quality_share_reversal','C_peak_Q','peak_s_quality','active_Q_upper_at_peak','nearest_active_transition'])+'\n\n'
    text+='峰值在预算网格邻域用一维搜索细化；是否先升后降按两侧分别至少变化1e-4判断（数值判据）。'
    text+='Q上界后质量处理总成本仍可随D增长，所以质量份额峰值不必与Q达到1完全同时。\n\n'
    text+='## 10 P1精确配方降维及证据边界\n\n成本与p无关，固定N时lambda>0取最小M、lambda<0取最大M、零点全部等价。'
    text+='因此经验全集合的点态下包络等于两个极值配方的包络；再对N/Q优化是数学精确降维。'
    text+=mdtable(ext,['role','candidate_id','M_p','DeltaM','source_members'])+'\n\n'
    text+='N区间按冻结节点与条件过零点分割；过零是冻结估算段的结果，不是新观测的尺度反转。'
    text+='P1官方预算Loss改善范围：\n\n'+mdtable(means)+'\n\n'
    text+='完整N/D/Q变化及选择配方见P0_vs_P1_official_comparison.csv。N>1B的配比效应明确标mixture_transfer_uses_estimated_support=True、Class D；'
    text+='它只是增强情景，不能当高可信真实配方建议。冻结模型输入配方经过已有零替换，不再预处理；物理部署需回溯原始来源。\n\n'
    text+='## 11 Q0敏感性\n\n仅针对三官方预算×三个指定上下文×三成本运行三候选，共81行；未重跑完整连续路径或改变主Q0。'
    text+=mdtable(sensframe.groupby('Q0').agg(Loss_change_min=('Loss_change','min'),Loss_change_max=('Loss_change','max'),
                       N_change_min=('N_change','min'),N_change_max=('N_change','max'),
                       active_structure_changed=('active_structure_changed','sum')).reset_index())+'\n\n'
    text+='基线改变可能影响低预算质量与Loss以及中预算配置，不能说所有数值不变；'
    text+='最低可行预算不变（CQ(Q0)=0），高预算主域饱和判定需按新的Δg(1)重新计算，但官方1e24依然可覆盖上界成本。'
    text+='敏感性只验证这些官方点，不据此宣称所有连续转移预算对Q0不敏感。\n\n'
    text+='## 12 结论范围与停止点\n\n以上是已冻结模块、成本代理、跨体系Q0映射及N/D矩形支持域下的最优配置；'
    text+='质量局部校准为半合成、估算lambda大尺度段及扩展域不可升级为直接证据。主域高预算饱和是支持范围受限。'
    text+='具备进入下一阶段P2理论敏感性及第三问收尾的技术条件，需人工确认本阶段结果；本轮没有执行P2或开始问题四。\n\n'
    text+='## 图表\n\n'+'\n'.join('- ['+r['caption']+'](../../'+r['figure']+')' for r in figs)+'\n'
    (root/'outputs/report/question3_phase2_optimization.md').write_text(text,encoding='utf-8')
    code=[root/'scripts/18_optimize_compute_resources.py',root/'scripts/18_verify_compute_optimization.py',root/'configs/optimization_phase2.yaml',
          *[root/'src/optimization'/p for p in ['resource_solver.py','transitions.py','resource_reporting.py']]]
    csv(out/'implementation_manifest.csv',[dict(path=p.relative_to(root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in code if p.exists()])
    inputs=[root/'outputs/optimization/phase1/optimization_setup.json',root/setup['frozen_objective_source'],
            root/'outputs/optimization/phase1/empirical_recipe_candidates.csv',root/'outputs/optimization/phase1/compute_budget_scenarios.csv',
            root/'configs/scaling_phase4.yaml',root/'configs/optimization_phase2.yaml']
    csv(out/'input_manifest.csv',[dict(path=p.relative_to(root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs])
