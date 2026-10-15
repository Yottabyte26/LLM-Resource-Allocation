"""Final Question 3 synthesis from frozen Phase 2 and new P2 diagnostics."""
from __future__ import annotations
import json
import hashlib
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .resource_solver import csv, load


def read(base, name):
    """Read saved full-precision output without changing it."""
    return pd.read_csv(base/name,float_precision='round_trip')


def mdtable(frame):
    """Small Markdown table with honest missing/infeasible values."""
    def cell(x):
        if pd.isna(x):return '—'
        if isinstance(x,(float,np.floating)):return f'{x:.6g}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def structural_summary(base, out, cfg):
    """Classify existing hinge breaks; never detect or refit a breakpoint."""
    hard=read(base,'typeI_active_set_transitions.csv');soft=read(base,'typeII_soft_transitions.csv')
    classes={'s_train':'train_attention_cost','s_attn':'train_attention_cost',
             's_quality':'quality_cost','ln_N':'N','ln_D':'D','Q':'Q'}
    rows=[];timeline=[]
    for r in hard.to_dict('records'):
        timeline.append(dict(domain=r['domain'],cost_function=r['cost_function'],L_ctx=r['L_ctx'],
            event=r['constraint']+':'+r['transition'],budget=r['budget_critical'],interval_low=r['bracket_low'],
            interval_high=r['bracket_high'],evidence_type='Type I active-set change',
            classification='constraint event',source_file='typeI_active_set_transitions.csv',
            interval_meaning='numerical root bracket; not statistical uncertainty'))
    for keys,g in soft.groupby(['domain','cost_function','L_ctx']):
        h=hard[(hard.domain==keys[0])&(hard.cost_function==keys[1])&(hard.L_ctx==keys[2])]
        significant=g[g.significant==True]
        for r in g.to_dict('records'):
            if pd.isna(r['break_budget']):continue
            distances=np.abs(np.log10(h.budget_critical.to_numpy()/r['break_budget']))
            near=int(np.argmin(distances));a=bool(distances[near]<=cfg['primary_transition_log10_distance'])
            neighbors=significant[np.abs(significant.break_log10C-r['break_log10C'])<=cfg['nearby_break_log10_distance']]
            categories=set(classes[m] for m in neighbors.metric)
            b=len(categories)>=3
            primary=bool(r['significant'] and (a or b))
            label='primary structural transition' if primary else 'secondary curve break'
            r.update(resource_category=classes[r['metric']],nearest_TypeI=h.constraint.iloc[near],
                nearest_TypeI_log10_distance=float(distances[near]),criterion_A=a,criterion_B=b,
                nearby_independent_categories=len(categories),classification=label)
            rows.append(r)
            if r['significant']:
                timeline.append(dict(domain=keys[0],cost_function=keys[1],L_ctx=keys[2],event=r['metric']+' curve break',
                    budget=r['break_budget'],interval_low=r['break_interval_low'],interval_high=r['break_interval_high'],
                    evidence_type='Type II saved hinge-fit diagnostic',classification=label,
                    delta_BIC=r['delta_BIC'],source_file='typeII_soft_transitions.csv',
                    interval_meaning='saved block-residual perturbation interval; not physical phase-transition CI'))
    classified=pd.DataFrame(rows);ordered=pd.DataFrame(timeline).sort_values(['domain','cost_function','L_ctx','budget','event'])
    csv(out/'typeII_final_classification.csv',classified);csv(out/'structural_transition_timeline.csv',ordered)
    peaks=read(base,'quality_cost_share_peaks.csv');thresholds=read(base,'quality_threshold_budgets.csv')
    qs=thresholds[thresholds.threshold.astype(str)=='1.0'][['domain','cost_function','L_ctx','budget_threshold']].rename(columns={'budget_threshold':'C_Qsat'})
    peaks=peaks.merge(qs,on=['domain','cost_function','L_ctx'],validate='one_to_one')
    peaks['relative_gap']=np.abs(np.log10(peaks.C_peak_Q/peaks.C_Qsat))
    csv(out/'quality_share_peak_vs_saturation.csv',peaks)
    transitions=[]
    saturation=read(base,'domain_saturation_budgets.csv')
    for keys,g in hard.groupby(['domain','cost_function','L_ctx']):
        r=dict(domain=keys[0],cost_function=keys[1],L_ctx=keys[2])
        for c,name in [('active_Q_lower','C_Q_leave'),('active_Q_upper','C_Q1'),('active_D_lower','C_D_leave'),('active_D_upper','C_Dmax')]:
            v=g[g.constraint==c];r[name]=float(v.budget_critical.iloc[0]) if len(v) else np.nan
        sat=saturation[(saturation.domain==keys[0])&(saturation.cost_function==keys[1])&(saturation.L_ctx==keys[2])]
        r['C_saturation']=float(sat.exact_saturation_budget.iloc[0])
        p=peaks[(peaks.domain==keys[0])&(peaks.cost_function==keys[1])&(peaks.L_ctx==keys[2])].iloc[0]
        r.update(C_peak_Q=p.C_peak_Q,max_s_Q=p.peak_s_quality,peak_Qsat_log10_gap=p.relative_gap)
        events=[(r[k],k) for k in ['C_Q_leave','C_Q1','C_D_leave','C_Dmax','C_saturation'] if np.isfinite(r[k])]
        r['investment_event_order']=' → '.join(name for _,name in sorted(events))
        transitions.append(r)
    thresholds=pd.DataFrame(transitions);csv(out/'cost_function_transition_summary.csv',thresholds)
    return classified,ordered,peaks,thresholds


def allocation_summary(base,out,combined,setup):
    """Context changes, evidence-aware tiers, and budget-stratum summaries."""
    main=combined[(combined.domain=='main')&(combined['mode']=='P0')].copy()
    ctx=[]
    for r in main.to_dict('records'):
        ref=main[(main.budget==r['budget'])&(main.cost_function==r['cost_function'])&(main.L_ctx==8192)].iloc[0]
        r.update(context_category='short' if r['L_ctx']<=8192 else 'near_critical' if r['L_ctx']==32768 else 'ultralong',
                 ratio_attention_training=setup['official']['eta']*r['L_ctx']/6,
                 L_critical=6/setup['official']['eta'],
                 N_ratio_vs_8192=r['N_star']/ref.N_star if r['feasible'] else np.nan,
                 D_ratio_vs_8192=r['D_star']/ref.D_star if r['feasible'] else np.nan,
                 Loss_change_vs_8192=r['L_star']-ref.L_star if r['feasible'] else np.nan)
        ctx.append(r)
    context=pd.DataFrame(ctx);csv(out/'context_final_summary.csv',context)
    high=combined[(combined.domain=='main')&(combined.budget==1e24)&combined['mode'].isin(['P0','P1'])]
    csv(out/'high_budget_main_saturation.csv',high)
    tiers=[]
    for r in combined.to_dict('records'):
        tier='Conservative' if r['domain']=='main' and r['mode']=='P0' else 'Empirical-enhanced' if r['domain']=='main' and r['mode']=='P1' else 'Exploratory'
        credible=r['mixture_evidence_level']=='real-scale-supported' and r['mode']=='P1' and r['domain']=='main'
        interpretation='reference allocation; frozen-model conditional' if tier=='Conservative' else \
            'empirical candidate enhancement with real-node mixture transfer' if credible else \
            'conditional estimated-scale mixture enhancement' if tier=='Empirical-enhanced' else \
            'unbounded sensitivity; no finite allocation to recommend' if r['mode']=='P2_B' else 'scenario extrapolation only'
        tiers.append(dict(tier=tier,**{k:r.get(k) for k in ['budget','cost_function','L_ctx','domain','mode','status','N_star','D_star','Q_star','L_star',
            'selected_candidate','mixture_evidence_level','Loss_improvement_vs_P0','unused_budget_fraction']},
            engineering_eligible=bool(r['status']=='SOLVED' and tier!='Exploratory'),
            mixture_gain_real_node_supported=credible,interpretation=interpretation,
            quality_warning='Q=1 exceeds observed A1 sample quality range; quality module has no direct joint N-D-Q validation' if r.get('Q_star',0)>.9511217995313428 else 'frozen Q0 proxy/baseline and modular-model assumptions apply'))
    csv(out/'recommendation_tiers.csv',tiers)
    budgets=[]
    for (domain,budget,mode),g in combined.groupby(['domain','budget','mode']):
        valid=g[g.status=='SOLVED'];budgets.append(dict(domain=domain,budget=budget,mode=mode,
            scenarios=len(g),solved=len(valid),Loss_min=valid.L_star.min(),Loss_max=valid.L_star.max(),
            improvement_min=valid.Loss_improvement_vs_P0.min(),improvement_max=valid.Loss_improvement_vs_P0.max(),
            N_min=valid.N_star.min(),N_max=valid.N_star.max(),D_min=valid.D_star.min(),D_max=valid.D_star.max(),
            Q_min=valid.Q_star.min(),Q_max=valid.Q_star.max()))
    csv(out/'budget_level_comparison_summary.csv',budgets)
    return main,context,high,pd.DataFrame(budgets)


def robustness_summary(base,out,combined,peaks):
    """Explicit scopes prevent baseline conclusions being generalized to P2."""
    q0=read(base,'Q0_sensitivity_comparison.csv')
    feasible=q0[q0.status=='SOLVED']
    q0bound=float(feasible.Loss_change.abs().max())
    records=[
        ('A','低预算质量保持Q0','robust','P0/P1主域，三个成本、全部可行官方低预算上下文及已有Q0敏感性',
         'P2理论外推允许不同投资路径；不将P0低预算规律推广到P2-B无界情景'),
        ('B','中预算Q=1','moderately_sensitive','P0/P1，C=1e22，三成本五上下文；Q0已有敏感性',
         'Q=1本身超出A1实际质量范围，依赖冻结质量模块；P2_A结果单列核对'),
        ('C','高预算主域饱和','robust','P0/P1，C=1e24，三成本五上下文及Q0敏感性',
         '依赖主域上界；扩展域结果不同，P2极端外推也可改变最优N'),
        ('D','质量成本份额先升后降且峰值靠近Q饱和','robust','冻结P0主/扩展域30条路径；主域15个峰值',
         '未重做Q0/P1/P2完整路径，不声称在所有配比方案下已验证'),
        ('E','长上下文压缩N/D并提高Loss','moderately_sensitive','P0/P1未饱和的可行预算情景',
         '低预算D在下界可不变；高预算主域饱和N/D/Loss相同，消耗与剩余预算改变'),
        ('F','经验配比优化改善Loss','highly_sensitive','P1经验可行集内条件性模型收益；低预算真实节点更可信',
         '大尺度lambda来自估计节点且变号，P2_B无界揭示配比模型外推风险'),
        ('G','具体转移预算保持稳定','highly_sensitive','数值根括区间精度高；不是模型参数不确定性区间',
         '三类成本和上下文显著移动阈值；早期投资顺序亦可改变'),
    ]
    sources=['quality cost function','Q0','L_ctx','P0/P1/P2','main/extended domain','large-scale lambda evidence']
    evidence=['三成本全部冻结对照','三候选基线敏感性，最大绝对Loss变化='+f'{q0bound:.6g}',
              '五场景与解析成本比；不拟合上下文性能收益','832候选P1/P2-A全局证书/P2-B无界',
              '主域与扩展域官方配置独立标记','>1B为估计迁移；>70B为lambda外推']
    levels={
        'A':['robust','robust','robust','highly_sensitive','robust','moderately_sensitive'],
        'B':['robust','robust','robust','highly_sensitive','moderately_sensitive','moderately_sensitive'],
        'C':['robust','robust','robust','highly_sensitive','highly_sensitive','moderately_sensitive'],
        'D':['robust','moderately_sensitive','robust','highly_sensitive','robust','moderately_sensitive'],
        'E':['robust','moderately_sensitive','moderately_sensitive','highly_sensitive','highly_sensitive','moderately_sensitive'],
        'F':['moderately_sensitive','moderately_sensitive','moderately_sensitive','highly_sensitive','highly_sensitive','highly_sensitive'],
        'G':['highly_sensitive','moderately_sensitive','highly_sensitive','highly_sensitive','highly_sensitive','highly_sensitive']}
    details=[];summaries=[]
    for key,claim,label,scope,limitation in records:
        summaries.append(dict(conclusion_id=key,conclusion=claim,robustness=label,verified_scope=scope,limitation=limitation))
        for j,source in enumerate(sources):
            details.append(dict(conclusion_id=key,conclusion=claim,sensitivity_source=source,
                robustness=levels[key][j],overall_robustness=label,verified_scope=scope,
                evidence=evidence[j],limitation=limitation,
                scope_note='robust means within explicitly verified scope; an untested full path is not evidence of invariance'))
    csv(out/'robustness_summary.csv',details);csv(out/'robustness_core_conclusions.csv',summaries)
    return pd.DataFrame(summaries)


def final_figures(root,out,combined,hard,classified,recipes,names):
    """Reuse frozen figures; add only three figures answering new questions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=root/'outputs/figures/question3_phase3';figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    manifest=[]
    for name,purpose in [('01_N_star_vs_budget.png','N versus budget'),('02_D_star_vs_budget.png','D versus budget'),
        ('03_Q_star_vs_budget.png','Q versus budget'),('05_cost_shares.png','three cost shares'),
        ('08_five_context_N_star.png','context N'),('09_five_context_D_star.png','context D'),
        ('10_five_context_L_star.png','context Loss')]:
        path=root/'outputs/figures/question3_phase2'/name
        if not path.is_file():raise FileNotFoundError(path)
        manifest.append(dict(path=path.relative_to(root).as_posix(),purpose=purpose,reused=True))
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,(budget,label) in zip(axes,[(1e19,'Low'),(1e22,'Middle'),(1e24,'High')]):
        for mode,color in [('P0','#555555'),('P1','#2878a8'),('P2_A','#c07328')]:
            g=combined[(combined.domain=='main')&(combined.budget==budget)&(combined.cost_function=='exponential')&
                (combined['mode']==mode)&(combined.status=='SOLVED')].sort_values('L_ctx')
            ax.plot(np.log2(g.L_ctx),g.L_star,'o-',label=mode,color=color)
        ax.set_xticks(np.log2([2048,4096,8192,32768,131072]),['2k','4k','8k','32k','128k'])
        ax.set_title(f'{label} budget: {budget:.0e} FLOPs');ax.set_xlabel('Context length');ax.grid(alpha=.2)
    axes[0].set_ylabel('Frozen-model optimized Loss');axes[-1].legend()
    fig.suptitle('Main-domain allocations; exponential cost; P2-A is theoretical extrapolation')
    fig.tight_layout();path=figdir/'01_P0_P1_P2_comparison.png';fig.savefig(path,dpi=180);plt.close(fig)
    manifest.append(dict(path=path.relative_to(root).as_posix(),purpose='official P0/P1/P2-A comparison; P2-B has no finite optimum',reused=False))
    fig,axes=plt.subplots(1,2,figsize=(12,6),sharex=True)
    labels=[n.removeprefix('train_the_pile_') for n in names]
    for ax,r in zip(axes,recipes.to_dict('records')):
        p=np.array([r[n] for n in names]);order=np.argsort(p)
        ax.barh(np.array(labels)[order],p[order],color='#2878a8')
        ax.set_title(f"P2-A {r['role']}: M={r['M_p']:.6f}");ax.set_xlabel('Domain fraction')
        ax.axvline(.8,color='#c07328',ls=':',label='concentration diagnostic > 0.8');ax.legend(fontsize=8)
    fig.suptitle('Frozen epsilon-constrained simplex extrema; neither is an engineering recipe')
    fig.tight_layout();path=figdir/'02_simplex_extreme_recipes.png';fig.savefig(path,dpi=180);plt.close(fig)
    manifest.append(dict(path=path.relative_to(root).as_posix(),purpose='full 17-domain theoretical simplex extrema',reused=False))
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    core={'active_Q_lower':('Q leaves floor','#2878a8','o'),'active_Q_upper':('Q reaches 1','#2c9473','s'),
          'active_D_upper':('D reaches cap','#c07328','^'),'budget_binding':('Main saturation','#735ca3','D')}
    for ax,family in zip(axes,['exponential','power','logarithmic']):
        for i,lctx in enumerate([2048,4096,8192,32768,131072]):
            h=hard[(hard.domain=='main')&(hard.cost_function==family)&(hard.L_ctx==lctx)]
            for c,(label,color,mark) in core.items():
                g=h[h.constraint==c]
                if len(g):ax.scatter(np.log10(g.budget_critical.iloc[0]),i,c=color,marker=mark,label=label if i==0 else None,s=40)
            g=classified[(classified.domain=='main')&(classified.cost_function==family)&(classified.L_ctx==lctx)&
                (classified.classification=='primary structural transition')&(classified.significant==True)&(classified.metric=='Q')]
            if len(g):ax.scatter(g.break_log10C,[i+.15]*len(g),marker='x',color='#333333',label='Primary Q curve break' if i==0 else None)
        ax.set_title(family);ax.set_xlabel('log10 allocated FLOPs');ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(5),['2048','4096','8192','32768','131072']);axes[0].set_ylabel('Context length')
    handles,labels=axes[-1].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=9,loc='lower center',ncol=5)
    fig.suptitle('Core constraint events and selected saved curve breaks; breaks are not independent physical transitions')
    fig.tight_layout(rect=(0,.1,1,1));path=figdir/'03_final_structural_timeline.png';fig.savefig(path,dpi=180);plt.close(fig)
    manifest.append(dict(path=path.relative_to(root).as_posix(),purpose='selected interpretable transition timeline',reused=False))
    csv(out/'final_figure_manifest.csv',manifest)


def final_report(root,out,combined,main,context,high,classified,peaks,thresholds,robust,diagnostics,summary):
    """Nine requested sections and exactly five core tables; appendices linked."""
    recipes=read(out,'P2_A_extreme_recipes.csv');p1=read(out,'P1_final_recipes.csv')
    search=read(out,'P2_global_search.csv');witness=read(out,'P2_B_boundary_limit_witness.csv')
    representative=main[main.L_ctx.isin([8192,32768])][['budget','cost_function','L_ctx','status','N_star','D_star','Q_star','L_star']]
    t2=thresholds[(thresholds.domain=='main')&(thresholds.L_ctx==2048)][['cost_function','C_Q_leave','C_Q1','C_D_leave','C_Dmax','C_saturation','C_peak_Q','max_s_Q']]
    t3=context[context.cost_function=='exponential'][['budget','L_ctx','ratio_attention_training','status','N_star','D_star','L_star','s_train','s_quality','s_attn']]
    t4=combined[(combined.domain=='main')&(combined.L_ctx==8192)&combined['mode'].isin(['P0','P1','P2_A'])].pivot(
        index=['budget','cost_function'],columns='mode',values='L_star').reset_index()
    t4['P2_B']='无有限最优解';t4['evidence']='P2_A为配比外推；P1按实际最优N判定lambda证据'
    t5=robust[['conclusion_id','conclusion','robustness','verified_scope']]
    mains=classified[classified.domain=='main'];mp=peaks[peaks.domain=='main'];mh=high[high['mode']=='P0']
    paragraphs=['# 问题三最终报告：算力约束下的资源配置\n',
      '问题一、问题二和问题三PHASE 1/2全部冻结。本报告读取已保存P0/P1、阈值、断点及敏感性结果；仅新增P2理论分析和收尾汇总。这里的配置是冻结模型下计算出的优化解，不是新增训练实验的观测值。',
      '## 3.1 优化模型与成本构成',
      r'目标：$L_{gen}=E+A N_B^{-\alpha}+B D_B^{-\beta}+C_{Q,loss}(N_B/N_{0,B})^{-u_Q}(-\ln Q)+\lambda_p(N_B)[M(p)-M(p_{ref})]$。系数和lambda插值全部沿用问题二。',
      r'约束：$C_{train}+C_{quality}+C_{attn}\le C$；$C_{train}=6N_BD_B10^{18}$，$C_{attn}=\eta L_{ctx}N_BD_B10^{18}$，$C_{quality}=D_B10^9[g(Q)-g(Q_0)]_+$。p不进入成本。三个官方g平行保留，不评选哪个“正确”。',
      'Q0='+f"{summary['Q0_main']:.16g}"+'；主域N_B∈[0.070542,11.965825]，D_B∈[10,299.893]，Q∈[Q0,1]。它是证据支持的资源箱体，不能等同于箱内每个N-D-Q组合都做过实测验证。尤其Q=1高于A1实际样本质量上限，质量模块缺少完整联合N-D-Q直接验证。',
      '## 3.2 解析降维与求解方法',
      r'由于$\partial L/\partial D_B<0$，$D_B^*=\min[D_{max},C/((6+\eta L_{ctx})N_B10^{18}+10^9\Delta g)]$。不强制花完预算。P0/P1直接采用PHASE 2结果；P2-A用相同二维资源求解器，并做独立DE复核，成本式未改变。',
      r'冻结一阶ILR模型可精确展开为$M(p)=a+\sum_j b_j\ln p_j$，$\sum b_j=0$。P2-A下界epsilon='+f"{summary['epsilon']:.17g}"+'，是冻结零替换尺度，未新增更大下界。minM使正系数坐标取下界，剩余负系数按−b水位分配；maxM反之。这给出解析全局证书。',
      f"三类数值核验：DE在softmax/log-ratio空间搜索后梯度细化；32起点SLSQP；{summary['random_samples_unbiased']:,}个广泛Dirichlet点加{summary['targeted_samples']:,}个靠近证书的定向Dirichlet点。随机点只给出可行值：采样最小值≥真实最小值、采样最大值≤真实最大值，不能把采样极值称为严格全局边界。DE/SLSQP与证书的误差详见P2_global_search.csv。",
      'P2-B的重要结论：冻结规则只替换精确0，不截断任意小的正数。沿p_j=t→0+、其余分量为固定正比例，M=b_j ln t+O(1)。有正负系数，所以inf M=−∞、sup M=+∞，均不取到。精确t=0经零替换反而返回有限值，边界处不连续。因而不能合法报告P2-B有限p_minM/p_maxM，也不能靠新floor修补。所有可行官方资源场景lambda≠0时Loss下确界为−∞；lambda=0时配比项为0，所有配比等价。此为模型外推失效证据，不是可实现性能提升。',
      '## 3.3 低中高预算最优配置',
      '**表1：P0主域，代表上下文8192/32768；N、D单位均为B，Loss无量纲。**',mdtable(representative),
      '低预算1e19：全部可行情景Q=Q0，P0的D=10B；131072上下文最低成本约2.27247e19，因此三类成本均不可行。中预算1e22：P0/P1全部Q=1。高预算1e24：P0/P1主域N/D/Q都达到上界。',
      f"主域P0高预算未用预算比例为{mh.unused_budget_fraction.min():.4%}–{mh.unused_budget_fraction.max():.4%}。其含义是已验证资源支持域饱和，现有模型无法可靠区分更大空间内配置；不能说真实训练最优只需要当前用量。扩展域单独标为scenario extrapolation。",
      '## 3.4 结构性转移',
      '**表2：Lctx=2048，三个成本函数的代表转移阈值；FLOPs。**',mdtable(t2),
      '完整时间轴含所有既有Type I活跃集事件及Type II断点，没有重新检测。活跃集根括区间是数值定位误差；Type II沿用分块残差扰动区间，不是物理相变或参数不确定性的置信区间。',
      f"主域90条曲线中{int(mains.significant.sum())}条满足ΔBIC≥10；按最终规则，{int(((mains.classification=='primary structural transition')&mains.significant).sum())}条是primary curve diagnostics。它们仍不是同样数量的独立结构事件。A规则为距任一Type I事件≤0.5 log10预算；B规则为半径0.5 log10邻域内至少三种独立资源/成本类别。训练和注意力份额严格成比例，合并为同一类，不能重复计数。正文仅保留代表断点与硬约束事件。",
      f"主域15个质量成本份额路径均先升后降。峰值相对Q=1阈值的|log10(Cpeak/CQsat)|最大为{mp.relative_gap.max():.6g}，系统性贴近质量饱和。每个成本×上下文的Cpeak、峰值份额和两种数值阈值材料均可追溯。这里Qsat使用PHASE 2已有target Q=1阈值，其容差与Type I active_Q_upper容差不同。",
      '## 3.5 上下文长度临界分析',
      r'$C_{attn}/C_{train}=\eta L_{ctx}/6$，$L_{crit}=6/\eta=30000$。短上下文2048/4096/8192、临界附近32768、超长131072。此临界只表示两项成本相等，不表示性能最优上下文，也没有建模增加上下文可能带来的性能收益。',
      '**表3：指数成本P0主域的五上下文对照；成本份额分母是分配预算。**',mdtable(t3),
      '跨8192→32768时，注意力/训练成本比从0.273067变为1.09227。未饱和中预算N和D下降、Loss上升；低预算D卡下界不变而N下降。高预算主域尺寸已经饱和，不再压缩N/D或改变Loss，但注意力消耗增加、未用预算下降。',
      '## 3.6 质量成本函数敏感性',
      '三种成本主要改变质量投资时机、峰值与阈值，也会改变局部早期阶段顺序。Lctx=2048时，指数/幂成本先退出D下界再提高质量；对数成本先提高质量并达到Q=1后才退出D下界。因此不能写“各成本完全保持相同阶段顺序”。相同官方低预算P0结果一致，中预算配置和Loss略有差异，高预算主域均饱和。完整五上下文顺序见cost_function_transition_summary.csv。',
      '## 3.7 领域配比优化：P0/P1/P2',
      '**表4：Lctx=8192，主域各官方预算下优化Loss；P2-A仅为理论配比外推。**',mdtable(t4),
    ]
    for r in diagnostics.to_dict('records'):
        paragraphs.append(f"{r['mode']} {r['role']}：max share={r['max_share']:.4%}，H={r['entropy']:.6g}，Hnorm={r['normalized_entropy']:.6g}，Neff={r['effective_domain_count']:.6g}，HHI={r['HHI']:.6g}；距参考L1/L2/JS={r['L1_to_ref']:.6g}/{r['L2_to_ref']:.6g}/{r['JS_distance_to_ref']:.6g}。最近经验L1/JS={r['minimum_empirical_L1']:.6g}/{r['minimum_empirical_JS_distance']:.6g}；超经验坐标范围领域数={r['out_of_empirical_range_domains']}；unsupported_mixture_extrapolation={r['unsupported_mixture_extrapolation']}。")
    paragraphs.extend([
      '两个P2-A端点都没有触发maxshare>0.8或Neff<2的集中阈值，更未触发maxshare>0.95的近退化阈值；不能预设纯simplex必然单域退化。但最优配方均偏离经验配方集，仍属unsupported mixture extrapolation。此标记依据坐标越界或最近距离超过832候选leave-one-out最近邻距离P95；是数据驱动的稀疏性诊断，不是统计置信区域。',
      f"P1的minM/maxM来自832个冻结实测配方。它们的完整17域、来源成员、模型值及集中度见附录；maxM配方本身最大单域{float(diagnostics[(diagnostics['mode']=='P1')&(diagnostics.role=='maxM')].max_share.iloc[0]):.2%}，Neff<2，是高度集中的经验配方，经验可行不代表分散。lambda>0选择minM；lambda<0选择maxM；条件性变号来自估计的大尺度迁移节点，不能理解为新的大模型训练观测。",
      '推荐层级：Conservative=主域P0；Empirical-enhanced=主域P1，其中60M–1B真实节点区间内增益证据较强，>1B估计节点依赖，>70B为lambda外推；Exploratory=扩展域和/或P2。P2不进入工程推荐。建议以P0作为主结论，P1作为按证据层级注明的增强方案。分类按各自最优N判断，不按预算自动划分。',
      '固定同一N/D/Q/Lctx的成本不变量核验已输出，P2收益仅来自冻结配比项及由此引起的资源重新分配，成本公式完全一致。',
      '## 3.8 鲁棒性与证据边界',
      '**表5：核心结论的鲁棒性及已核验范围。**',mdtable(t5),
      '六种敏感性来源在robustness_summary.csv逐项列出：成本函数、Q0、上下文、P0/P1/P2、主/扩展域、lambda证据。robust均限制在明示的已验证范围；Q0只沿用已有官方场景敏感性，未重新检测转移；未计算P1/P2完整预算路径，不能声称其质量份额峰值规律已经验证。模型箱体内仍有插值/模块组合假设，不能称全部结果“直接实测支持”。',
      '## 3.9 问题三结论',
      '1. 解析消去D和按lambda符号选配方，使资源优化降为二维；P0/P1历史结果保持冻结，P2-A另有全局配方证书。\n2. 官方低预算优先保留质量基线，中预算质量达到模型上界，高预算主域饱和；对应证据和适用范围必须写清。\n3. 质量投资份额先升后降，15个主域峰值均靠近质量饱和；成本形状改变时机，部分改变早期阶段顺序。\n4. 30000为成本临界，上下文增加使未饱和配置承受更高注意力负担；不是性能最优点。\n5. 主推荐P0，P1作为证据分层的经验增强；P2-A虽未单域退化仍远离实测配方，P2-B无界，均不得工程部署。\n6. 精确转移阈值、配比迁移收益和大资源外推高度依赖假设；1e24未用预算反映证据域饱和。',
      '## 附录与复现',
      '新脚本：`python -X utf8 -B scripts/19_finalize_compute_optimization.py --stage simplex|official|report`；完整运行可用`--stage all`。脚本使用pathlib自动定位项目根。只有新phase3输出会更新；现有阶段文件不修改。',
    ])
    for name in ['P2_global_search.csv','P2_A_extreme_recipes.csv','P2_B_boundary_limit_witness.csv','P2_B_exact_zero_vertices.csv',
        'P1_final_recipes.csv','recipe_concentration_and_support.csv','empirical_distance_calibration.csv','P2_empirical_coordinate_support.csv',
        'P0_P1_P2_official_comparison.csv','recommendation_tiers.csv','structural_transition_timeline.csv','typeII_final_classification.csv',
        'quality_share_peak_vs_saturation.csv','cost_function_transition_summary.csv','context_final_summary.csv','robustness_summary.csv',
        'high_budget_main_saturation.csv','fixed_resource_cost_invariance.csv','final_figure_manifest.csv',
        'input_manifest.csv','implementation_manifest.csv','independent_verification.csv','verification_summary.json']:
        paragraphs.append(f'- [{name}](../optimization/phase3/{name})')
    paragraphs.append('附录图：复用PHASE 2的N/D/Q路径、成本份额及上下文图，新图仅补P0/P1/P2-A比较、17域simplex配方、核心事件时间轴。')
    for path in ['../figures/question3_phase2/01_N_star_vs_budget.png','../figures/question3_phase2/02_D_star_vs_budget.png',
        '../figures/question3_phase2/03_Q_star_vs_budget.png','../figures/question3_phase2/05_cost_shares.png',
        '../figures/question3_phase2/10_five_context_L_star.png','../figures/question3_phase3/03_final_structural_timeline.png',
        '../figures/question3_phase3/01_P0_P1_P2_comparison.png','../figures/question3_phase3/02_simplex_extreme_recipes.png']:
        paragraphs.append(f'![{Path(path).stem}]({path})')
    paragraphs.append('独立公式/输出及冻结文件SHA256核验结果以附录verification_summary.json为准。运行`python -X utf8 -B scripts/19_verify_final_optimization.py`可独立复核；不需要重跑PHASE 2。')
    paragraphs.append('问题三至此结束。没有开始问题四，没有改动冻结附件、预处理、质量权重、Q/K、ILR/ElasticNet、Scaling Law、lambda或PHASE 2优化结果。')
    report=root/'outputs/report/question3_final.md';report.write_text('\n\n'.join(paragraphs)+'\n',encoding='utf8')


def finalize(root,cfg):
    """Build final tables, three new figures, and the integrated report."""
    root=Path(root);out=root/'outputs/optimization/phase3';base=root/'outputs/optimization/phase2'
    source_paths=[root/'outputs/optimization/phase1/optimization_setup.json',
        root/'outputs/optimization/phase1/empirical_recipe_candidates.csv',
        root/'outputs/mixture/final_model_metadata.json',root/'outputs/mixture/stage2/final_model_policy.json',
        root/'outputs/scaling/phase4/generalized_scaling_final.json',root/'configs/scaling_phase4.yaml',
        root/'configs/optimization_phase2.yaml',root/'configs/optimization_phase3.yaml']
    source_paths += [base/n for n in ['official_budget_results.csv','typeI_active_set_transitions.csv',
        'typeII_soft_transitions.csv','quality_cost_share_peaks.csv','quality_threshold_budgets.csv',
        'domain_saturation_budgets.csv','Q0_sensitivity_comparison.csv']]
    def manifest(paths):
        rows=[]
        for p in paths:
            with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            rows.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest))
        return rows
    csv(out/'input_manifest.csv',manifest(source_paths))
    csv(out/'implementation_manifest.csv',manifest([root/'configs/optimization_phase3.yaml',
        root/'src/optimization/simplex_sensitivity.py',root/'src/optimization/final_reporting.py',
        root/'scripts/19_finalize_compute_optimization.py',root/'scripts/19_verify_final_optimization.py']))
    setup,frozen,solver_cfg,model,candidates,empirical=load(root)
    from .simplex_sensitivity import recipe_summary,support_calibration,evidence_level
    names=frozen['reference']['domain_order'];pref=np.asarray(frozen['reference']['p_ref'])
    cfg=support_calibration(candidates,names,cfg,out)
    diagnostics=[];recipes=read(out,'P2_A_extreme_recipes.csv')
    for mode,frame in [('P2_A',recipes),('P1',empirical)]:
        for r in frame.to_dict('records'):
            diagnostics.append(dict(candidate_id=r['candidate_id'],mode=mode,role=r['role'],
                **recipe_summary(np.array([r[n] for n in names]),pref,candidates,names,cfg)))
    diagnostics=pd.DataFrame(diagnostics);csv(out/'recipe_concentration_and_support.csv',diagnostics)
    combined=read(out,'P0_P1_P2_official_comparison.csv')
    combined['mixture_evidence_level']=combined.apply(evidence_level,axis=1)
    csv(out/'P0_P1_P2_official_comparison.csv',combined)
    classified,timeline,peaks,thresholds=structural_summary(base,out,cfg)
    main,context,high,budgets=allocation_summary(base,out,combined,setup)
    robust=robustness_summary(base,out,combined,peaks)
    final_figures(root,out,combined,read(base,'typeI_active_set_transitions.csv'),classified,recipes,names)
    summary=json.loads((out/'simplex_summary.json').read_text(encoding='utf8'))
    summary.update(Q0_main=solver_cfg['Q0_main'],official_comparison_rows=len(combined),
        simplex_numerical_extremes_not_concentrated=bool(~diagnostics[diagnostics['mode']=='P2_A'].highly_concentrated.any()),
        P2_A_unsupported_mixture_extrapolation=bool(diagnostics[diagnostics['mode']=='P2_A'].unsupported_mixture_extrapolation.all()),
        main_quality_peak_max_log10_gap=float(peaks[peaks.domain=='main'].relative_gap.max()),
        main_typeII_significant=int(classified[classified.domain=='main'].significant.sum()),
        main_typeII_primary=int(((classified.domain=='main')&classified.significant&
            (classified.classification=='primary structural transition')).sum()),question4_started=False,
        P0_P1_frozen_results_reoptimized=False,earlier_parameters_refitted=False)
    (out/'execution_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    final_report(root,out,combined,main,context,high,classified,peaks,thresholds,robust,diagnostics,summary)
    logging.info('Final report and three supplementary figures complete')
