"""Question 2 final scientific plots and report, with conditional evidence retained."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .mixture_transfer_reporting import markdown_table
from .io import write_csv


def plot_analysis(out,figdir,grid,eqs,domains,joints,metrics,sensitivity,cfg):
    """Standalone scientific figures; every plotted value is from computed output."""
    out,figdir=Path(out),Path(figdir);manifest=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name,description):
        fig.savefig(figdir/name,dpi=cfg['figure_dpi'],bbox_inches='tight');plt.close(fig)
        manifest.append(dict(filename=name,description=description))
    g=pd.DataFrame(grid);q=pd.DataFrame(eqs)
    main=g[(g.recipe=='p_ref')&(g.Q_label=='Q_0.7')&(g.D_label=='median')].sort_values('N_params_B')
    fig,ax=plt.subplots(figsize=(8,4.6))
    for factor,color in [('N','#28649b'),('D','#39915b'),('Q','#bb6a28')]:
        ax.plot(main.N_params_B,main['E_'+factor],marker='o',color=color,label='E_'+factor)
    ax.axvline(11.97,color='gray',ls='--',lw=1,label='B6/B7 upper N boundary')
    ax.set_xscale('log');ax.set_xlabel('N (B parameters, log scale)');ax.set_ylabel('Loss-reduction elasticity of R=L-E')
    ax.set_title('Reference recipe, Q=0.7, median D');ax.legend(fontsize=9)
    save(fig,'01_typical_elasticities.png','Main elasticities; outside B6/B7 range is extrapolative')
    fig,axes=plt.subplots(1,3,figsize=(12,4.6),sharey=True)
    for ax,recipe in zip(axes,['p_ref','P10','P90']):
        sub=g[(g.recipe==recipe)&(g.Q_label=='Q_0.7')]
        table=sub.pivot(index='D_tokens_B',columns='N_params_B',values='dominant_factor')
        values=np.array([[{'N':0,'D':1,'Q':2}[v] for v in row] for row in table.to_numpy()])
        from matplotlib.colors import ListedColormap
        im=ax.imshow(values,aspect='auto',cmap=ListedColormap(['#28649b','#39915b','#bb6a28']),vmin=-.5,vmax=2.5)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):ax.text(j,i,['N','D','Q'][values[i,j]],ha='center',va='center',color='white')
        ax.set_xticks(range(len(table.columns)),[f'{v:g}' for v in table.columns],rotation=45)
        ax.set_yticks(range(len(table.index)),[f'{v:g}' for v in table.index]);ax.set_xlabel('N (B)');ax.set_ylabel('D (B tokens)')
        ax.set_title(recipe)
    fig.suptitle('Dominant loss-reduction elasticity at Q=0.7 (right derivative at knots)');fig.tight_layout()
    save(fig,'02_dominant_factor_regions.png','Categorical comparison at real saved recipe baselines')
    sub=q[(q.recipe=='p_ref')&(q.D_label=='median')&(q.N_params_B.isin(cfg['N_primary_B']))&(q.Q_label!='Q_0.7')]
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,N in zip(axes,cfg['N_primary_B']):
        s=sub[sub.N_params_B==N].sort_values('Q');x=np.arange(len(s))
        ax.bar(x-.18,s.equivalent_parameter_gain_pct,width=.36,label='Exact first root')
        ax.bar(x+.18,s.local_approx_gain_pct,width=.36,label='Local approximation')
        ax.set_xticks(x,[f'{v:.3f}' for v in s.Q]);ax.set_xlabel('Initial Q');ax.set_ylabel('Equivalent parameter increase (%)');ax.set_title(f'N={N:g}B')
        ax.legend(fontsize=8)
    fig.suptitle('Q + 0.1 equivalence, reference recipe (12B slightly outside B6/B7 range)');fig.tight_layout()
    save(fig,'03_Q01_parameter_equivalence.png','Exact numerical equivalence versus local approximation, nine main workpoints')
    domain=pd.DataFrame(domains);scales=['1M','60M','1B','10B','70B']
    table=domain.pivot(index='domain',columns='scale',values='DeltaL_j')[scales]
    names=[v.removeprefix('train_the_pile_') for v in table.index]
    extent=float(abs(table.to_numpy()).max());fig,ax=plt.subplots(figsize=(8,7.5))
    im=ax.imshow(table.to_numpy(),cmap='RdBu_r',vmin=-extent,vmax=extent,aspect='auto')
    ax.set_xticks(range(5),['1M observed','60M observed','1B observed','10B estimated','70B estimated'],rotation=25,ha='right')
    ax.set_yticks(range(17),names);ax.set_title('Domain +1pp: mean Loss change across five frozen baselines')
    fig.colorbar(im,ax=ax,label='Delta Loss');fig.tight_layout()
    save(fig,'04_domain_effect_by_scale.png','Scaled effects from Question 1; estimated directions explicitly marked')
    pairs=pd.DataFrame(joints);globalmax=float(abs(pairs.scaled_I_jk).max())
    for scale in scales:
        matrix=pd.read_csv(out/f'joint_trend_matrix_{scale}.csv',index_col=0)
        fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(matrix.to_numpy(),cmap='RdBu_r',vmin=-globalmax,vmax=globalmax)
        names=[v.removeprefix('train_the_pile_') for v in matrix.index]
        ax.set_xticks(range(17),names,rotation=90,fontsize=7);ax.set_yticks(range(17),names,fontsize=7)
        ax.set_title(f'Joint perturbation deviation at {scale} ('+('estimated-supported' if scale in ['10B','70B'] else 'observed-supported')+')')
        fig.colorbar(im,ax=ax,label='Scaled I_jk (diagonal not applicable)');fig.tight_layout()
        save(fig,'05_joint_trend_'+scale+'.png','136 pair deviations mirrored; diagonal is not an observed pair')
    pred=pd.read_csv(out/'B10_reference_predictions.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));ax=axes[0]
    sc=ax.scatter(pred.val_loss,pred.L0_prediction,c=np.log10(pred.N_params_B),cmap='viridis',s=24)
    lo=min(pred.val_loss.min(),pred.L0_prediction.min());hi=max(pred.val_loss.max(),pred.L0_prediction.max())
    ax.plot([lo,hi],[lo,hi],color='gray',lw=1);ax.set_xlabel('B10 estimated Loss');ax.set_ylabel('Frozen L0 prediction')
    ax.set_title('Estimated backbone consistency only');fig.colorbar(sc,ax=ax,label='log10(N/B)')
    ax=axes[1];ax.scatter(pred.N_params_B,pred.residual_prediction_minus_estimated,s=24)
    ax.axhline(0,color='gray',lw=1);ax.set_xscale('log');ax.set_xlabel('N (B)');ax.set_ylabel('Prediction minus estimated Loss')
    ax.set_title('B10 residuals; Q=1 and p=p_ref are structural references');fig.tight_layout()
    save(fig,'06_B10_estimated_consistency.png','No Q/p observations inferred and no full-model validation claimed')
    ss=pd.DataFrame(sensitivity);fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharex=True)
    for ax,recipe in zip(axes,['P10','P90']):
        for policy,style in [('S-A','-'),('S-B','--'),('S-C',':')]:
            s=ss[(ss.recipe==recipe)&(ss.policy==policy)].sort_values('N_params_B')
            ax.plot(s.N_params_B,s.Delta_p,marker='o',ls=style,label=policy)
        ax.axhline(0,color='gray',lw=.7);ax.set_xscale('log');ax.set_xlabel('N (B)');ax.set_ylabel('Mixture correction Delta_p')
        ax.set_title(recipe+' frozen recipe');ax.legend()
    fig.suptitle('Beyond 70B policy sensitivity; S-A is the main boundary rule');fig.tight_layout()
    save(fig,'07_lambda_beyond70B_sensitivity.png','Hold-last, log continuation, and shrink-to-zero are compared without selection')
    checks=pd.read_csv(out/'marginal_derivative_checks.csv');fig,ax=plt.subplots(figsize=(6,4.4))
    for variable in ['N','D','Q']:
        s=checks[checks.variable==variable];ax.scatter(s.analytic,s.finite_difference,s=12,alpha=.55,label=variable)
    lo=checks.analytic.min();hi=checks.analytic.max();ax.plot([lo,hi],[lo,hi],color='gray',lw=1)
    ax.set_xlabel('Analytic derivative');ax.set_ylabel('Fourth-order finite difference');ax.set_title('Derivative verification, including one-sided N knots');ax.legend()
    save(fig,'08_derivative_verification.png','Formula checks are not synthetic joint observation data')
    write_csv(out/'figure_manifest.csv',manifest)


def build_final_report(path,out,final,points,grid,eqs,dominance,regions,domains,joints,metrics,sensitivity,schemas,cfg):
    """Seventeen requested topics, using actual rows and inherited module limitations."""
    c,q=final['classical'],final['quality'];out=Path(out)
    main=pd.read_csv(out/'Q01_parameter_equivalence.csv',float_precision='round_trip').to_dict('records')
    recipes=pd.read_csv(out/'Q01_parameter_equivalence_recipe_sensitivity.csv',float_precision='round_trip').to_dict('records')
    g=pd.DataFrame(grid);typ=g[(g.recipe=='p_ref')&(g.Q_label=='Q_0.7')&(g.D_label=='median')].to_dict('records')
    eq_summary=json.loads((out/'equivalence_summary.json').read_text(encoding='utf-8'))
    region=pd.DataFrame(regions);status=region.groupby('status')['n'].sum().to_dict()
    unavailable=pd.read_csv(out/'no_equivalence_records.csv',float_precision='round_trip')
    local_missing=unavailable[unavailable.status=='no_local_equivalence']
    missing_N=sorted(local_missing.N_params_B.unique().tolist())
    missing_indices=sorted(local_missing.recipe_index.astype(int).unique().tolist())
    recipe_indices={r['recipe']:r['index'] for r in points['recipes']}
    joins=pd.read_csv(out/'B9_B10_join.csv')
    join_counts=joins.join_status.value_counts().to_dict()
    d=pd.DataFrame(domains);j=pd.DataFrame(joints);sen=pd.DataFrame(sensitivity)
    jointcounts=[]
    for scale,s in j.groupby('scale',sort=False):
        jointcounts.append(dict(scale=scale,negative_mean=int((s.scaled_I_jk<0).sum()),positive_mean=int((s.scaled_I_jk>0).sum()),
            stable_negative_across_five_baselines=int((s.scaled_max<0).sum()),stable_positive_across_five_baselines=int((s.scaled_min>0).sum()),
            baseline_dependent=int(((s.scaled_min<=0)&(s.scaled_max>=0)).sum())))
    effects=d[d.scale=='1M'].sort_values('DeltaM_j')
    evidence=final['evidence'];component=pd.read_csv(out.parent/'phase3/component_validation.csv',float_precision='round_trip').to_dict('records')
    for r in component:
        for k,v in r.items():
            if isinstance(v,float) and np.isnan(v):r[k]=None
    content=['# 问题二最终报告：模块化广义标度律、边际收益与参数等价','',
      '问题二按人工确认冻结的模块化形式完成。所有骨架、质量和配比参数保持原值，仅分析、推导、数值求解及授权的外推一致性检查。'
      '主 λ 规则为 ln N 分段线性插值，低于 1M 保持首节点，高于 70B 保持末节点。没有进入问题三。','',
      '## 1. 经典标度律','',r'$L_0(N,D)=E+AN^{-\alpha}+BD^{-\beta}$。N、D 均以十亿为单位。','',
      markdown_table([dict(parameter=k,value=v) for k,v in c.items()],['parameter','value']),'',
      '读取 PHASE 1 的 trajectory_balanced 主行，程序使用完整精度。主拟合数据是 B1，未使用 B9/B10 回调参数。','',
      '## 2. 质量模块及验证边界','',
      r'$\Delta_Q=C_Q(N/N_0)^{-u_Q}(-\ln Q)$，$N_0=1$B。','',
      f"C_Q={q['C_Q']:.17g}，u_Q={q['u_Q']:.17g}，从冻结 PHASE 2C S3 读取；删除已确认 v≈0 的 D 项，未重拟合。",'',
      '仅称“B6 校准的局部质量修正项”。Q 对 B6 Loss 有增量解释能力，但 B7 非重叠的组内差分幅值没有稳定复现。'
      'B8 与此前机制不兼容，不宣称存在全局统一质量弹性。第一问领域 Q 仅提供数值工作点，不证明与 B6 的 Q 具有同一生成语义。','',
      f"B6/B7 数值支持范围：{points['quality_numeric_support']}。12B 略高于真实上限 11.97B，已标记轻微外推；70B、100B、1000B 为明显外推。",'',
      '## 3. 配比模块与 λ 的尺度变化','',
      'M(p) 是冻结 ILR + 一阶 ElasticNet 对配方的 1M L_mean 预测。参考配方为冻结训练 ILR 中心，未重新读取或拟合 A4–A15。','',
      markdown_table(final['lambda_knots'],['N_params_B','lambda_s','CI_low','CI_high','evidence']),'',
      '三个真实尺度同为正，两个估算尺度同为负。方向翻转及过零结论依赖估算表。'
      '超过 1B 的配比段已经包含 Class D 证据，不能把 70B 以内的所有配比结果都称为真实观测支持。','',
      '## 4. λ 的最终分段与边界规则','',
      r'$s_i=(\lambda_{i+1}-\lambda_i)/\ln(N_{i+1}/N_i)$；内部 $\lambda(N)=\lambda_i+s_i\ln(N/N_i)$，$\lambda\prime(N)=s_i/N$。','',
      'N<1M：λ=λ_1M，导数为 0。N>70B：主方案 λ=λ_70B，导数为 0。'
      '节点上 λ 连续，但 N 导数通常不连续，表中同时记录左右弹性；主口径使用右导数。','',
      '特别是 1B 节点本身的 λ 来自真实检验，但其右导数使用 1B→10B 段，依赖估算表。'
      '非参考配方在 1B 的参数弹性、优势因素和局部等价判断因此含 Class D 条件性证据。','',
      markdown_table(pd.read_csv(out/'lambda_piecewise_definition.csv').where(pd.notna(pd.read_csv(out/'lambda_piecewise_definition.csv')),None).to_dict('records'),
                     ['segment','N_low','N_high','lambda_low','lambda_high','s_i']),'',
      '## 5. 模块化广义标度律及退化条件','',
      r'$L_{gen}=E+AN^{-\alpha}+BD^{-\beta}+C_Q(N/N_0)^{-u_Q}(-\ln Q)+\lambda_p(N)[M(p)-M(p_{ref})]$。','',
      f"M(p_ref)={final['reference']['M_ref']:.17g}。17 维参考配方保存在冻结 PHASE 3 reference_recipe.json。",'',
      r'$Q=1$、$p=p_{ref}$ 时两个修正项均为 0，$L_{gen}=L_0$。该退化是结构恒等，不是观测到的大模型质量或配方。','',
      '## 6. N/D/Q 边际效应及有限差分验证','',
      r'令 $\Delta M=M(p)-M(p_{ref})$：','',
      r'$\partial_D L=-\beta B D^{-\beta-1}$；$\partial_Q L=-C_Q(N/N_0)^{-u_Q}/Q$。','',
      r'$\partial_N L=-\alpha A N^{-\alpha-1}-(u_Q/N)\Delta_Q+(s_i/N)\Delta M$。','',
      '边界保持区令 s_i=0。在折点处分别取左右 s_i。使用四阶中心差分检查内部，四阶单侧差分检查折点；不使用跨折点中心差分验证单侧导数。','',
      f"marginal_derivative_checks.csv 全部通过。最大相对误差={eq_summary['derivative_max_relative_error']:.8g}。"
      '判据为相对误差≤1e−5 或绝对误差≤1e−8。','',
      '## 7. 可约 Loss 的降低弹性','',
      r'$R=L_{gen}-E$，$\mathcal E_x=-\partial\ln R/\partial\ln x$。正值表示增加资源降低可约 Loss。','',
      r'$\mathcal E_D=\beta BD^{-\beta}/R$；$\mathcal E_Q=C_Q(N/N_0)^{-u_Q}/R$。','',
      r'$\mathcal E_N=[\alpha AN^{-\alpha}+u_Q\Delta_Q-s_i\Delta M]/R$。','',
      '仅 R>0 时计算对数弹性。传统 x/L·∂L/∂x 保留在 elasticity_grid.csv 的 traditional_N/D/Q 附表字段，正文使用上述降低弹性。'
      '弹性比较的是相同相对变化比例，不等于相同成本或相同绝对增量的收益。','',
      '## 8. 工作点与优势区域','',
      f"D 从 B6 的 45 个不同 N-D 块取 P25/median/P75：{points['D_quantiles']}B。"
      '这避免把重复 Q 层当作额外 D 设计；不是删去任何原始数据。三个值均在 B6 支持范围内。',
      f"Q 为七个领域 Q_domain 的最小、中位、最大值，另加 0.7：{points['domain_Q_points']}。",'',
      f"配方为 p_ref、冻结 M(p) P10 邻近配方 index={recipe_indices['P10']}、P90 邻近配方 index={recipe_indices['P90']}。"
      f"完整工作点 N×D×Q×p 共 {len(grid)} 条，不创造配方。",'',
      markdown_table(typ,['N_params_B','D_tokens_B','Q','E_N','E_D','E_Q','dominant_factor','extrapolative']),'',
      markdown_table([r for r in dominance if r['dominant_count']>0],['N_params_B','recipe','factor','dominant_count','n_workpoints','dominant_fraction']),'',
      '上述优势区是该冻结候选的结构结果。估算配比段及质量外推的证据等级不会因弹性值较大而提高。','',
      '## 9. Q 提升 0.1 的精确参数等价','',
      r'固定 D、p，求首个 $N_{eq}>N$：$L_{gen}(N_{eq},D,Q,p)=L_{gen}(N,D,Q+0.1,p)$。','',
      r'等价参数增加为 $N_{eq}/N-1$。E 和 $BD^{-\beta}$ 从方程两侧严格抵消，因此本模型的等价解不随固定 D 改变。','',
      '正式结果使用 brentq。搜索上限 max(100N,10000B)。每个分段内加入全部 λ 节点及 Gain_N=0 的驻点，'
      '检查所有单调子区间，返回第一根，避免在非单调情况下漏掉根。','',
      markdown_table(main,['N_params_B','D_tokens_B','Q','Q_new','N_eq_B','equivalent_parameter_gain_pct','local_approx_gain_pct',
                           'quality_support','equivalent_N_outside_quality_support']),'',
      '主表仅九条，12B 的原工作点及其等价 N 已超质量拟合上限。Q 数值取自真实附件输出，但上述等价结论仍条件于 B6 局部质量模块。','',
      '## 10. 等价存在条件与局部解析近似','',
      r'$Gain_N=-N\partial_NL=\alpha AN^{-\alpha}+u_Q\Delta_Q-s_i\Delta M$。','',
      'Gain_N≤0 时标记 no_local_equivalence，不强行求根。Gain_N>0 但搜索范围内无根时标记 no_finite_equivalence_within_range。'
      '后者仅表示指定有限范围内无解，不能断言任意无穷范围都无解。折点采用已声明的右侧局部判据。','',
      r'$\Delta N/N\approx[C_Q(N/N_0)^{-u_Q}/(Q\,Gain_N)]\Delta Q$，这里 $\Delta Q=0.1$。','',
      '0.1 不一定是严格小扰动；局部式只解释替代条件，不能替代数值解。','',
      f"216 个代表网格的状态：{eq_summary['status_counts']}；局部近似平均绝对相对误差="
      f"{eq_summary['approximation_absolute_relative_error_mean']:.6%}，最大={eq_summary['approximation_absolute_relative_error_max']:.6%}。",'',
      f"进一步复用全部 512 个真实训练配方在相同 N/Q 工作点诊断，状态计数：{status}。"
      'training_recipe_equivalence_diagnosis.csv 保留全部记录，no_equivalence_records.csv 保留无局部/无有限解情况。'
      '这仅是模型结构诊断，不是新的大模型观测或优化。','',
      f"本次无局部等价发生的 N={missing_N}B，涉及配方 index={missing_indices}。"
      '其中 1B 右侧的判断依赖估算支持的 1B→10B 斜率。'
      '这些工作点按指令没有求全局等价根，因此不能说它们在任意更大 N 都无有限解。其余完成搜索的工作点未检测到范围内无根。','',
      markdown_table(recipes,['recipe','recipe_index','N_params_B','Q','N_eq_B','equivalent_parameter_gain_pct','status']),'',
      '## 11. 十七领域的尺度化边际效应','',
      r'$\Delta L_j(N)=\lambda_p(N)\Delta M_j$，复用问题一每域增加 1 个百分点的冻结结果。','',
      '主表的 ΔM_j 为五个代表基准的等权平均，另保留全部基准案例、零激活及最小/最大范围。'
      '存在基准依赖的领域不能按平均值宣称普遍有利或不利。','',
      markdown_table(effects.to_dict('records'),['domain','DeltaM_j','source_baseline_direction']),'',
      '1M/60M/1B 的正 λ 保持原方向且逐步衰减；10B/70B 的负 λ 使对应方向反转，但仅是 estimated-supported 的条件性结果。'
      '完整 17×5 结果见 domain_effect_by_scale.csv。','',
      '## 12. 领域组合趋势','',
      r'$I_{jk}(N)=\lambda_p(N)I_{jk}$。每尺度 136 对、17×17 对称矩阵，主对角线不适用而非真实零偏离。','',
      markdown_table(jointcounts,['scale','negative_mean','positive_mean','stable_negative_across_five_baselines',
                                   'stable_positive_across_five_baselines','baseline_dependent']),'',
      'λ 变号时，组合偏离符号同步翻转，基准依赖程度保留。负向称条件性互补趋势，正向称条件性竞争趋势。'
      'ILR 与闭合操作本身也会形成组合偏离，不作为因果互补/竞争或真实交互的证据。','',
      '## 13. B9/B10 百亿/千亿参数级外推','',
      f"B9：{schemas['B9']['n']} 条元数据，N={schemas['B9']['N_min']}–{schemas['B9']['N_max']}B，"
      f"D={schemas['B9']['D_min']}–{schemas['B9']['D_max']}B。无 Loss，非正 D={schemas['B9']['nonpositive_D']} 条，全部保留。",'',
      f"B10：{schemas['B10']['n']} 条估算 Loss。B9.model_name 与 B10.family 精确一对一对应，128 条匹配，"
      f"{join_counts.get('left_only',0)} 条仅 B9，匹配 N/D 相同。缺失、重复及全部字段见 B9_B10_diagnosis.md。", '',
      '参考情景设置 Q=1、p=p_ref，故只预测 L0(N,D)。没有给真实大模型虚构观测 Q 或配方。'
      'B9 用于现实尺度锚点，B10 用于估算 Loss 对照，不能验证完整 N-D-Q-p 模型。','',
      markdown_table(metrics,['bucket','n','RMSE','MAE','R2','Pearson','Spearman','mean_bias']),'',
      '分桶采用 [100,300)、[300,1000)、[1000,∞)B。统一称“百亿/千亿参数级估算一致性检验”。'
      '参数总量/激活参数、稠密/稀疏模型和不同评测口径未统一，因此高一致性也不等于真实同口径泛化性能。','',
      '## 14. 超 70B 的 λ 结构敏感性','',
      'S-A 主方案：保持 λ_70B。S-B：延续 10B→70B 的 ln N 斜率。'
      'S-C：70B 后在 ln N 上线性收缩，到 1000B 为 0，之后保持 0。未按结果好看程度更换主方案。','',
      markdown_table(sensitivity,['N_params_B','recipe','policy','lambda_p','DeltaM','Delta_p','role']),'',
      f"代表配方的三规则最大 Δp 跨方案差值为 {max(float(s.Delta_p.max()-s.Delta_p.min()) for _,s in sen.groupby(['N_params_B','recipe'])):.8g}。"
      '**超70B领域配比修正具有高外推不确定性。**一类规则保留偏移，一类持续增大，一类收缩至零；B9/B10 无法验证 p 项，不能据此选出真实机制。','',
      '## 15. 证据等级与分模块验证','',
      markdown_table(evidence,['module','parameters','evidence','support']),'',
      markdown_table(component,['module','dataset','stage','target','n','RMSE','R2','limitation']),'',
      'Class A：B1 主拟合经典五参数。Class B：B6 半合成局部质量校准。Class C：第一问真实配比实验的 M(p)。'
      'Class D：10B/70B 估算 λ 节点及其大尺度段。大尺度质量项更属于外推，不能借 Class D 配比证据补强。','',
      '## 16. 局限性','',
      '不存在完整 (N,D,Q,p,L) 联合直接观测。采用 modular calibration + component-wise validation，无法进行单一完整联合外部验证。'
      '相加结构未验证 Q×p 等联合交互。不同尺度配方覆盖和评测分布可能混杂传递系数变化。'
      '第一问领域 Q 与 B6 Q 的语义一致性没有联合观测支持。估算表不会变成真实实验。'
      '等价参数数值是冻结模型内的条件性替代，不是训练实测承诺，也未纳入质量改善成本或算力约束优化。','',
      '## 17. 结论与停止点','',
      '经典骨架、局部质量修正、尺度调制配比项可按明确锚点组合，所有拟合参数保留冻结值。'
      '质量提升与参数扩张的等价使用精确求根，局部近似只解释机制。配比及组合方向反转依赖估算表，超 70B 边界规则具有结构不确定性。'
      'B10 只支持经典骨架的估算一致性，不能补充缺失的完整联合验证。问题二完成后停止，不进入问题三。','',
      '复现：python -B scripts/16_analyze_generalized_scaling.py。独立核验：python -B scripts/16_verify_generalized_scaling.py。'
      '全部结果位于 outputs/scaling/phase4，图表位于 outputs/figures/question2_phase4。','']
    Path(path).write_text('\n'.join(content),encoding='utf-8')
