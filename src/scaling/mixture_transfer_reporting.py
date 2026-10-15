"""Plain scientific figures and a provenance-aware Phase 3 Markdown report."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .mixture_transfer import SCALES,p2_value
from .io import write_csv


def markdown_table(rows,columns):
    """Render finite numeric values; absent statistics remain explicitly unavailable."""
    def cell(v):
        if v is None:return 'n.a.'
        if isinstance(v,float):return f'{v:.8g}' if np.isfinite(v) else 'n.a.'
        if isinstance(v,(dict,list)):return json.dumps(v,ensure_ascii=False)
        return str(v)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
                      ['| '+' | '.join(cell(r.get(c)) for c in columns)+' |' for r in rows])


def plot_results(out,figdir,rows,data,fits,comparison,critical,cfg):
    """Plot observed/estimated evidence separately and display conditional intervals."""
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figdir=Path(figdir);manifest=[]
    def save(fig,name,description):
        path=figdir/name;fig.savefig(path,dpi=cfg['figure_dpi'],bbox_inches='tight');plt.close(fig)
        manifest.append(dict(filename=name,description=description))
    N=np.array([r['N_params_B'] for r in rows]);lam=np.array([r['lambda_s'] for r in rows])
    lo=np.array([r['CI_low'] for r in rows]);hi=np.array([r['CI_high'] for r in rows])
    def knots(ax):
        for mask,color,marker,label in [(np.arange(5)<3,'#28649b','o','Observed mixture checks'),
                                       (np.arange(5)>=3,'#bb6a28','s','Estimated tables')]:
            ax.errorbar(np.log10(N[mask]),lam[mask],yerr=[lam[mask]-lo[mask],hi[mask]-lam[mask]],
                        fmt=marker,color=color,capsize=4,linestyle='none',label=label)
        ax.axhline(0,color='gray',lw=.8)
        ax.set_xticks(np.log10(N),[r['scale'] for r in rows]);ax.set_xlabel('Parameter scale N (log axis)')
        ax.set_ylabel('Centered mixture transfer coefficient lambda')
    fig,ax=plt.subplots(figsize=(8,4.8));knots(ax);ax.plot(np.log10(N),lam,color='gray',lw=1,label='Empirical log-linear interpolation')
    ax.set_title('Mixture transfer by scale (recipe bootstrap 95% CI)');ax.legend(fontsize=9)
    save(fig,'01_mixture_transfer_by_scale.png','Observed and estimated lambda with paired recipe Bootstrap intervals')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    grid=np.geomspace(N.min(),N.max(),250)
    for ax,(label,title) in zip(axes,[('A_real_only','Fit A: three observed scales'),('B_with_estimated','Fit B: observed + estimated scales')]):
        knots(ax)
        for m,color in [('P1','#39915b'),('P2','#8b4b9b')]:
            f=fits[(label,m)];p=f['parameters']
            y=p['a']+p['b']*np.log(grid) if m=='P1' else p2_value(np.log(grid),f['theta'])
            support=grid<=1 if label=='A_real_only' else np.ones(len(grid),bool)
            ax.plot(np.log10(grid[support]),y[support],color=color,label=m)
            if label=='A_real_only':
                ax.plot(np.log10(grid[~support]),y[~support],color=color,linestyle='--',alpha=.6)
        ax.set_title(title);ax.legend(fontsize=8)
    fig.suptitle('P1/P2 scale fits; dashed Fit A curves are unsupported extrapolation',fontsize=11)
    fig.tight_layout();save(fig,'02_lambda_scale_functions.png','P1/P2 Fits A/B; Fit A beyond 1B marked as extrapolation')
    reps=pd.read_csv(out/'mixture_transfer_bootstrap_replicates.csv')
    fig,axes=plt.subplots(1,5,figsize=(15,3.4))
    for ax,r in zip(axes,rows):
        vals=reps.loc[reps.scale==r['scale'],'lambda_s'];color='#28649b' if r['evidence']=='observed' else '#bb6a28'
        ax.hist(vals,bins=35,color=color,alpha=.8);ax.axvline(0,color='black',lw=.8)
        ax.axvline(r['lambda_s'],color='red',lw=1);ax.set_title(r['scale']+' ('+r['evidence']+')',fontsize=10)
        ax.set_xlabel('lambda');ax.set_ylabel('Bootstrap count')
    fig.suptitle('Recipe Bootstrap distributions (2000 resamples per scale)');fig.tight_layout()
    save(fig,'03_lambda_bootstrap_distributions.png','No source model refit; paired draws within shared recipe sets')
    fig,axes=plt.subplots(1,5,figsize=(16,3.8))
    for ax,r in zip(axes,rows):
        g=data[data.scale==r['scale']];x=g.m_centered.to_numpy()
        ax.scatter(x,g.y_centered,s=10,alpha=.55,color='#28649b' if r['evidence']=='observed' else '#bb6a28')
        gx=np.array([x.min(),x.max()]);ax.plot(gx,r['lambda_s']*gx,color='black',lw=1)
        ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5)
        ax.set_title(r['scale']+f"  R2={r['R2']:.3f}",fontsize=10)
        ax.set_xlabel('M(p) minus scale mean');ax.set_ylabel('Loss minus scale mean')
    fig.suptitle('Within-scale centered calibration (not an external validation)');fig.tight_layout()
    save(fig,'04_centered_recipe_calibration.png','All 702 saved recipe points, with no change to their Loss or prediction')
    roots=pd.read_csv(out/'lambda_scale_bootstrap_replicates.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,(fit,m) in zip(axes.flat,[(f,m) for f in ['A_real_only','B_with_estimated'] for m in ['P1','P2']]):
        g=roots[(roots['fit']==fit)&(roots.model==m)]
        valid=g.log_Nc_B.notna();a=g.loc[valid,'log_Nc_B'].to_numpy()/np.log(10)
        ax.hist(a,bins=40,color='#28649b' if fit=='A_real_only' else '#bb6a28',alpha=.8)
        ax.set_title(fit.replace('_',' ')+' '+m,fontsize=10)
        ax.set_xlabel('log10(Nc / B)');ax.set_ylabel('Bootstrap count')
    fig.suptitle('Candidate crossing uncertainty; large-scale evidence is estimated');fig.tight_layout()
    save(fig,'05_conditional_crossing_bootstrap.png','Crossing of fitted lambda functions, not an identified physical transition')
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for ax,(f,title) in zip(axes,[('A_real_only','Observed scales only'),('B_with_estimated','Including estimated scales')]):
        subset=[r for r in comparison if r['fit']==f];labels=[r['model'] for r in subset]
        ax.bar(np.arange(2)-.18,[r['RMSE'] for r in subset],width=.36,label='In-scale RMSE')
        vals=[r['LOSO_RMSE'] if r['LOSO_RMSE'] is not None else np.nan for r in subset]
        ax.bar(np.arange(2)+.18,vals,width=.36,label='Leave-scale-out RMSE')
        if f=='A_real_only':ax.text(1,.02,'P2 leave-scale-out\nunidentified (2 points, 3 params)',ha='center',fontsize=8)
        ax.set_xticks(range(2),labels);ax.set_ylabel('RMSE in lambda');ax.set_title(title);ax.legend(fontsize=8)
    fig.tight_layout();save(fig,'06_scale_fit_validation.png','Saturated P2 Fit A and unavailable two-point leave-scale-out fit disclosed')
    write_csv(out/'figure_manifest.csv',manifest)


def build_report(path,out,rows,comparison,critical,decision,config,component):
    """Report modular evidence, formulas, uncertainties, and the explicit stop point."""
    classical=config['classical'];q=config['quality'];reference=config['reference']
    inputs=pd.read_csv(out/'input_manifest.csv').to_dict('records')
    observed=[r for r in rows if r['evidence']=='observed'];estimated=[r for r in rows if r['evidence']=='estimated']
    printable_critical=[]
    for r in critical:
        view=dict(r)
        if not r['conditional_interval_stable']:
            for k in ['Nc_B_median','Nc_B_CI_low','Nc_B_CI_high']:view[k]=None
        printable_critical.append(view)
    cols=['scale','evidence','n','lambda_s','standard_error','OLS_CI_low','OLS_CI_high','CI_low','CI_high','R2','Pearson','Spearman']
    text=['# 问题二 PHASE 3：领域配比的尺度调制与模块化广义标度律','',
      '本阶段完成冻结模块的组合和逐尺度配比传递校准。最终候选使用分尺度经验系数及 ln N 上的分段线性插值。'
      '不冻结新的全局 N-D-Q 函数，不重新训练配比模型，不进行联合最小二乘，不使用 B9/B10。','',
      '## 1. 冻结输入与证据等级','',
      '第一问主模型为 L_mean 的 ILR + 一阶 ElasticNet。M(p) 是其 1M 尺度预测 Loss。'
      '只解析此前输出，未解析 A4–A15 原始附件。现成逐配方预测与冻结模型参数、保存 ILR 坐标逐行核对。'
      'L_trim、Ridge、增强 Qp/Kp 模型均未替代主模型。','',
      markdown_table(config['evidence'],['module','parameters','evidence','support']),'',
      'Class A 是骨架参数的直接数据支持，不表示 B4/B5 已统一评测口径。Class B 是 B6 半合成局部证据。'
      'Class C 是问题一真实配比实验。Class D 是 10B/70B 估算表的条件性证据。','',
      '## 2. 逐尺度中心化与传递系数','',
      r'$m_{si}=M(p_{si})-\overline{M}_s$, $y_{si}=L_{si}-\overline{L}_s$。', '',
      r'$\hat\lambda_s=\sum_i m_{si}y_{si}/\sum_i m_{si}^2$，中心化坐标不设截距。', '',
      'standard error 使用原坐标含截距 OLS 等价的 n−2 残差自由度，因为两侧均值从数据估计。'
      'OLS CI 为 t 区间，CI_low/CI_high 为主要报告的配方 Bootstrap 百分位区间。'
      'R² 是校准后的组内拟合 R²，不是未经目标尺度校准的外部预测性能。Pearson/Spearman 比较原始 M(p) 与 Loss，保留估算表负相关。','',
      markdown_table(rows,cols),'',
      f"真实三尺度的系数同为正：{all(r['CI_low']>0 for r in observed)}（三个 Bootstrap 区间均不跨 0）。"
      f"估算两尺度同为负：{all(r['CI_high']<0 for r in estimated)}（两个区间均不跨 0）。",'',
      '定量结果支持真实尺度内配比方向一致，并在估算表中出现反向。真实实验尚未观测到符号翻转。'
      '**配比反向及临界规模结论依赖估算表。**','',
      '## 3. 配方 Bootstrap','',
      '固定种子 2026，每尺度 2000 次按配方 index 有放回抽样。每次重新计算样本内两个均值，再估计无截距传递系数。'
      '1M/60M 的完整 256 个配方相同，10B/70B 的完整 63 个配方相同，分别共享抽样索引。'
      '1B 的 64 个配方独立，三类配方集合互不重叠。不能仅按同名 index 关联不同配方。'
      '因此 1B 与其他组的配方覆盖不同，传递系数变化不能完全归因于 N；评测口径、配方覆盖和估算生成方式也可能影响系数。'
      '此 Bootstrap 条件于冻结 M(p)，不包含第一问模型再训练不确定性和估算表生成误差。','',
      markdown_table(rows,['scale','bootstrap_mean','bootstrap_median','bootstrap_std','CI_low','CI_high','positive_fraction','negative_fraction']),'',
      '## 4. P1/P2 的真实数据拟合与估算敏感性','',
      r'P1：$\lambda(N)=a+b\ln(N/N_{ref})$，$N_{ref}=1$B。', '',
      r'P2：$\lambda(N)=a\tanh[b(\ln N_c-\ln N)]$，$a,b>0$。', '',
      'N 使用十亿参数为单位，所以 P2 的对数为 ln(N/B)。各尺度等权拟合，避免小标准误的尺度自动主导。'
      'Fit A 仅三项真实检验，Fit B 加两项估算表。P2 全拟合 16 初值，Bootstrap 每次 3 初值，正参数有界拟合。'
      'AICc 仅在 n>k+1 时定义。三点三参数的 P2 Fit A 是饱和拟合，残差自由度为 0；留出一尺度后只有两点，禁止解释成可识别的三参数模型。','',
      markdown_table(comparison,['fit','model','parameters','RMSE','LOSO_RMSE','AICc','residual_df','jacobian_condition','saturated']),'',
      'Fit B 中 P2 的拟合 RMSE 和留尺度 RMSE 均低于 P1；P1 参数更少且 AICc 更低，适合作为简约趋势对照。'
      '数值性能和简约准则未一致选择同一形式，P2 的较小误差也没有足够真实尺度验证支撑。'
      '因此不把任一光滑函数冻结为全局配比尺度律，保留分尺度经验系数。','',
      '## 5. 临界规模的条件性与辨识限制','',
      '对 P1 求 ln(Nc/B)=−a/b，P2 使用拟合 ln(Nc/B)。每次使用上述配方 Bootstrap 系数重新拟合曲线。'
      '表中 valid_fraction 报告有限根比例，boundary_fraction 报告触边比例。conditional_interval_stable 要求有限根比例≥95%、触边比例≤5%，'
      '且 95% CI 完整位于数据中相反符号节点的括区间内。Fit A 没有符号括区间，不能判为稳定临界规模。这个条件性标记'
      '不表示真实实验已确认物理临界规模。负向趋势与反向传递不是因果机制证明。','',
      markdown_table(printable_critical,['fit','model','valid_fraction','boundary_fraction','Nc_B_median','Nc_B_CI_low','Nc_B_CI_high','conditional_interval_stable']),'',
      '**不报告已被真实实验稳定识别的具体临界规模。**真实尺度均为正，没有过零括区间；'
      'Fit A 的 P1/P2 外推根还存在结构依赖。Fit B 和经验插值的过零位置仅作为依赖估算表的条件性诊断，不能用于宣称普适临界规模。','',
      '## 6. 最终配比项：经验系数及分段形式','',
      markdown_table(config['lambda_knots'],['N_params_B','lambda_s','CI_low','CI_high','evidence']),'',
      r'令 $x=\ln(N/B)$、$x_s=\ln(N_s/B)$。在 $N_s\le N\le N_{s+1}$ 内：', '',
      r'$\lambda_p(N)=\lambda_s+(\lambda_{s+1}-\lambda_s)(x-x_s)/(x_{s+1}-x_s)$。', '',
      '只在节点范围 0.001B–70B 内插值，接口拒绝范围外外推。0.001B–1B 的节点来自真实检验，超过 1B 的插值依赖 Class D 估算证据。'
      '在 ln N 而非原 N 上线性插值，保持与尺度趋势图一致。节点处可连续，但导数一般有左右两值。','',
      f"参考配方直接读取冻结的 A4_ILR_center，M(p_ref)={reference['M_ref']:.17g}。"
      '中心是在冻结零替换后的 ILR 空间取均值并反变换，17 维配方完整存于 reference_recipe.json。', '',
      r'$\Delta_p(N,p)=\lambda_p(N)[M(p)-M(p_{ref})]$，所以参考配方修正严格为 0。', '',
      '逐尺度中心化只校准斜率，不改变经典绝对基线。参考配方锚定不会证明各源的绝对 Loss 基线已对齐。','',
      '## 7. 模块化广义标度律候选','',
      r'$L_{gen}=E+AN^{-\alpha}+BD^{-\beta}+C_Q(N/N_0)^{-u_Q}(-\ln Q)+\lambda_p(N)[M(p)-M(p_{ref})]$。', '',
      f"N、D 以 B 为单位，N0=1B。完整冻结参数：{json.dumps(classical,ensure_ascii=False)}。",'',
      f"C_Q={q['C_Q']:.17g}，u_Q={q['u_Q']:.17g}，直接读取 PHASE 2C S3，未重新拟合。"
      '原 v≈0 且触边，按人工确认删除 D^v 项。仅称“B6 校准的局部质量修正项”。', '',
      r'$Q=1$ 且 $p=p_{ref}$ 时，$L_{gen}=L_0$。anchor_identity_checks.csv 是公式恒等核验，不是联合观测测试集。', '',
      '质量与配比相加采用模块独立性假设，尚未验证 Q×p、N×D×p 的联合交互。配比表没有共同 N/D/Q/p/L 联合观测。'
      '模型是在各自证据源独立标定后组合的候选，不提供统一外部预测精度。','',
      '## 8. 边际效应公式与 ILR 有限差分','',
      r'记 $G(p)=M(p)-M(p_{ref})$，$\Delta_Q=C_Q(N/N_0)^{-u_Q}(-\ln Q)$。', '',
      r'$\partial_N L=-\alpha A N^{-\alpha-1}-(u_Q/N)\Delta_Q+\lambda_p\prime(N)G(p)$。', '',
      r'$\partial_D L=-\beta B D^{-\beta-1}$。', '',
      r'$\partial_Q L=-C_Q(N/N_0)^{-u_Q}/Q$。', '',
      r'分段内部：$\lambda_p\prime(N)=(\lambda_{s+1}-\lambda_s)/[N\ln(N_{s+1}/N_s)]$。', '',
      r'P1 对照的导数为 $b/N$；P2 对照为 $-ab[1-\tanh^2(b(\ln N_c-\ln N))]/N$。', '',
      r'令 $t=ILR(p)$，$p_{\pm}=ILR^{-1}(t\pm h e_k)$。', '',
      r'$[\Delta_p(N,p_+)-\Delta_p(N,p_-)]/(2h)=\lambda_p(N)[M(p_+)-M(p_-)]/(2h)$。', '',
      r'冻结一阶模型下，若 $g_k=s_y\beta_k/s_{X,k}$，上述有限差分等于 $\lambda_p(N)g_k$。', '',
      '此关系针对严格正配方的 ILR 扰动。原始零配比先按冻结约定替换，不把对替换组成的有限差分解释为原始结构零处的导数。'
      '本轮仅推导公式及实现核验，不计算弹性、Q 提升 0.1 的参数等价。','',
      '## 9. 分模块验证及局限性','',
      markdown_table(component,['module','evidence','dataset','stage','target','n','RMSE','R2','Pearson','Spearman','limitation']),'',
      'L0：B1 的验证来自此前留轨迹 CV，B3 是插值轨迹一致性，B4/B5 保留不同评测口径限制。'
      'Delta_Q：B6 质量修正有效，但 B7 差分 R² 未通过；此失败在组合模型中仍然保留。'
      'Delta_p：问题一原预测检验与本轮中心化斜率校准分别报告，后者不能声称独立外部验证。'
      '10B/70B：估算且为训练配方子集，不是大模型真实联合实验。','',
      '**广义模型采用模块化证据融合，无法进行单一完整联合外部验证。**未制造联合测试集，未重拟合经典或质量参数。'
      '此前 B8 机制不兼容结论保持冻结，本轮未用 B8 校准配比项。','',
      '## 10. 输出与停止点','',
      '主要输出：mixture_transfer_by_scale.csv、mixture_transfer_bootstrap.csv、lambda_scale_model_comparison.csv、'
      'critical_scale_diagnosis.csv、lambda_empirical_knots.csv、generalized_scaling_candidate.json、component_validation.csv。'
      '复现：python -B scripts/15_calibrate_mixture_scaling.py；核验：python -B scripts/15_verify_mixture_scaling.py。','',
      '可以提交本阶段候选进行人工确认。尚未授权进入下一阶段。'
      '对于超过 70B 的 B9/B10 目标，本轮分段接口没有外推策略，下一阶段须先明确外推规则和 Class D 情景身份。'
      '不把本阶段称为完整联合模型验证通过。已停止，不执行弹性、参数等价或 B9/B10 外推。','',
      '## 11. 实际解析文件','',markdown_table(inputs,['path','bytes','sha256','source_role']),'']
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text('\n'.join(text),encoding='utf-8')
