"""Phase 5B: candidate validation on frozen Q/K/Z, never activate a new Q."""
from __future__ import annotations
import csv
import json
import logging
from pathlib import Path
import sys
import numpy as np
import yaml
from scipy.stats import pearsonr,spearmanr,rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quality.robust_reaggregation import reaggregate
from quality.io import sha256_file
from quality.vector_diagnostics import write_rows

OUT=ROOT/'outputs/conflict/phase5b'
FIG=ROOT/'outputs/figures/question1_part2/phase5b'
csv.field_size_limit(2**31-1)
FILES={'A1':'A1_sample_conflict.csv','A2':'A2_arxiv_conflict.csv','A3':'A3_github_conflict.csv'}


def rows(path):
    """Parse small metadata tables and preselected text material."""
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def frozen_hashes():
    """Protect every existing output, official source and fixed configuration."""
    files=list((ROOT/'real_attachments').rglob('*'))+list((ROOT/'outputs').rglob('*'))+list((ROOT/'configs').glob('*'))
    result={}
    for p in sorted(files):
        if not p.is_file() or p.is_relative_to(OUT) or p.is_relative_to(FIG) or p.name.startswith('question1_part2_phase5b'):continue
        result[p.relative_to(ROOT).as_posix()]=sha256_file(p)
    return result


def read_input(dataset,names,n):
    """Read saved Q/K only; never calculate/re-fit either frozen quantity."""
    z=np.empty((n,22));numbers=np.empty((n,4));metadata=[]
    with (ROOT/'outputs/conflict'/FILES[dataset]).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        required=set(names+['Q','K','Q_equal','Delta_Q','id','domain'])
        if not required<=set(reader.fieldnames or []):raise ValueError('Missing frozen columns')
        for i,r in enumerate(reader):
            if i>=n:raise ValueError('Unexpected extra input row')
            z[i]=[float(r[k]) for k in names];numbers[i]=[float(r[k]) for k in ['Q','K','Q_equal','Delta_Q']]
            metadata.append({k:r[k] for k in ['id','dataset','domain','sub_path','source_file','line_number','imputed_count']})
        if len(metadata)!=n or len({r['id'] for r in metadata})!=n:raise ValueError('Missing rows or duplicate id')
    return {'Z':z,'Q':numbers[:,0],'K':numbers[:,1],'Q_equal':numbers[:,2],'Delta_Q':numbers[:,3],
            'meta':metadata,'domains':np.array([r['domain'] for r in metadata])}


def corr(a,b,method='pearson'):
    """Report undefined correlations as empty, never as zero."""
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:return None
    return float((pearsonr if method=='pearson' else spearmanr)(a,b).statistic)


def summarize(data,result,mask,delta,thresholds,settings,meta):
    """Describe movement and ranks in a stated view, using fixed A1 thresholds."""
    q,qr,change=[x[mask] for x in [data['Q'],result['Q_robust'],result['change']]]
    k=data['K'][mask];high=k>thresholds['P90'];low=k<=thresholds['P50'];extreme=k>thresholds['P95']
    tol=settings['direction_equality_tolerance'];absolute=np.abs(change)
    ranks=np.abs(rankdata(q)-rankdata(qr))
    def mean_if(x,m):return float(x[m].mean()) if m.any() else None
    return {**meta,'delta':delta,'epsilon':settings['epsilon'],'n':len(q),'n_high':int(high.sum()),
        'pearson_Q_Qrobust':corr(q,qr),'spearman_Q_Qrobust':corr(q,qr,'spearman'),
        'pearson_high_Q_Qrobust':corr(q[high],qr[high]),'spearman_high_Q_Qrobust':corr(q[high],qr[high],'spearman'),
        'mean_abs_change_all':float(absolute.mean()),'mean_abs_change_low':mean_if(absolute,low),
        'max_abs_change_low':float(absolute[low].max()) if low.any() else None,
        'max_abs_change_ordinary':float(absolute[~high].max()) if (~high).any() else None,
        'mean_abs_change_high':mean_if(absolute,high),'median_abs_change_high':float(np.median(absolute[high])) if high.any() else None,
        'P90_abs_change_high':float(np.quantile(absolute[high],.9)) if high.any() else None,
        'P95_abs_change_high':float(np.quantile(absolute[high],.95)) if high.any() else None,
        'mean_abs_change_P95_conflict':mean_if(absolute,extreme),'mean_signed_change_high':mean_if(change,high),
        'max_abs_change_all':float(absolute.max()),'P95_abs_rank_change_all':float(np.quantile(ranks,.95)),
        'max_abs_rank_change_all':float(ranks.max()),
        'up_count_all':int((change>tol).sum()),'down_count_all':int((change<-tol).sum()),'unchanged_count_all':int((absolute<=tol).sum()),
        'up_rate_all':float((change>tol).mean()),'down_rate_all':float((change<-tol).mean()),'unchanged_rate_all':float((absolute<=tol).mean()),
        'up_count_high':int((change[high]>tol).sum()),'down_count_high':int((change[high]<-tol).sum()),
        'up_rate_high':float((change[high]>tol).mean()) if high.any() else None,
        'down_rate_high':float((change[high]<-tol).mean()) if high.any() else None,
        'unchanged_rate_high':float((absolute[high]<=tol).mean()) if high.any() else None,
        'Q_mean_high':mean_if(q,high),'Qrobust_mean_high':mean_if(qr,high),
        'A1_fixed_P90':thresholds['P90'],'direction_equality_tolerance':tol}


def weight_diagnostics(data,result,w,names,delta,high,settings):
    """Separate raw downweight coefficients from normalized local weights."""
    coefficients=result['coefficients_high'];local=result['local_weights_high'];ratio=local/w
    residual=np.abs(data['Z'][high]-data['Q'][high,None]);extremes=residual==residual.max(axis=1,keepdims=True)
    per_metric=[]
    for j,name in enumerate(names):
        per_metric.append({'delta':delta,'metric':name,'original_weight':float(w[j]),'mean_a_high':float(coefficients[:,j].mean()),
            'mean_local_weight_high':float(local[:,j].mean()),'mean_normalized_weight_ratio':float(ratio[:,j].mean()),
            'normalized_downweight_rate':float((ratio[:,j]<1-1e-12).mean()),
            'significant_downweight_rate':float((ratio[:,j]<=settings['significant_weight_ratio']).mean())})
    summary={'delta':delta,'n_high':int(high.sum()),'mean_local_weight_L1_change':float(np.abs(local-w).sum(axis=1).mean()),
        'original_effective_metric_count':float(1/np.square(w).sum()),
        'mean_robust_effective_metric_count_high':float((1/np.square(local).sum(axis=1)).mean()),
        'extreme_residual_entries_n':int(extremes.sum()),
        'extreme_residual_normalized_weight_down_rate':float((ratio[extremes]<1-1e-12).mean()),
        'extreme_residual_significant_downweight_rate':float((ratio[extremes]<=settings['significant_weight_ratio']).mean()),
        'max_local_weight_high':float(local.max()),'local_weights_sum_max_error':float(np.abs(local.sum(axis=1)-1).max())}
    return summary,per_metric


def choose_delta(sensitivity):
    """Recommend the largest tested delta if it is empirically conservative.

    This chooses a validation candidate only, not final score activation.
    No unprovided numerical acceptability cutoff is introduced.
    """
    candidate=max(sensitivity,key=lambda r:r['delta'])
    minimum_mean=min(r['mean_abs_change_high'] for r in sensitivity)
    minimum_max=min(r['max_abs_change_all'] for r in sensitivity)
    if candidate['mean_abs_change_high']>minimum_mean+1e-14 or candidate['max_abs_change_all']>minimum_max+1e-14:
        raise ValueError('Largest delta is not conservative on this dataset; manual parameter judgment needed')
    if candidate['max_abs_change_ordinary']!=0:raise ValueError('Ordinary samples changed')
    return float(candidate['delta'])


def select_cases(a1,result,w,names,high,settings):
    """Choose extremes from existing text material, plus exact-unchanged controls."""
    text=rows(ROOT/'outputs/conflict/A1_high_conflict_text_samples.csv')
    by_id={r['id']:i for i,r in enumerate(a1['meta'])};known={by_id[r['id']]:r for r in text}
    tol=settings['direction_equality_tolerance'];change=result['change']
    groups={
        'up':sorted([i for i in known if high[i] and change[i]>tol],key=lambda i:(-change[i],a1['meta'][i]['id'])),
        'down':sorted([i for i in known if high[i] and change[i]<-tol],key=lambda i:(change[i],a1['meta'][i]['id'])),
        'unchanged':sorted([i for i in known if abs(change[i])<=tol],key=lambda i:(abs(change[i]),a1['meta'][i]['id'])),
        'high_conflict_smallest_change':sorted([i for i in known if high[i]],key=lambda i:(abs(change[i]),a1['meta'][i]['id']))}
    high_index=np.full(len(high),-1,dtype=int);high_index[high]=np.arange(high.sum())
    selections=[];case_rows=[];local_rows=[]
    for group,indices in groups.items():
        count=settings['case_high_conflict_smallest_change_count'] if group=='high_conflict_smallest_change' else settings['case_count_per_direction']
        indices_selected=indices[:count]
        selections.append({'group':group,'available_n':len(indices),'requested_n':count,'selected_n':len(indices_selected),
            'definition':'high-conflict smallest absolute changes; not claimed near-zero' if group=='high_conflict_smallest_change' else 'saved text material; unchanged uses stated numeric tolerance'})
        for i in indices_selected:
            local=result['local_weights_high'][high_index[i]] if high[i] else w
            coefficient=result['coefficients_high'][high_index[i]] if high[i] else np.ones(22)
            ratios=local/w;significant=[names[j] for j in range(22) if ratios[j]<=settings['significant_weight_ratio']]
            reductions=sorted(range(22),key=lambda j:(ratios[j],j))[:3]
            case_rows.append({**a1['meta'][i],'case_group':group,'high_conflict':bool(high[i]),'Q':float(a1['Q'][i]),'K':float(a1['K'][i]),
                'Q_robust':float(result['Q_robust'][i]),'change':float(change[i]),'abs_change':float(abs(change[i])),
                'significantly_downweighted_metrics':';'.join(significant),'significant_definition':f"normalized local/original weight <= {settings['significant_weight_ratio']}",
                'most_downweighted_metrics':';'.join(names[j] for j in reductions) if high[i] else '',
                'local_weights_json':json.dumps(dict(zip(names,map(float,local))),ensure_ascii=False),
                'a_coefficients_json':json.dumps(dict(zip(names,map(float,coefficient))),ensure_ascii=False),
                'review_status':'待人工确认重聚合解释是否改善',**dict(zip(names,map(float,a1['Z'][i]))),
                'content':known[i]['content']})
            for j,name in enumerate(names):
                local_rows.append({'id':a1['meta'][i]['id'],'case_group':group,'metric':name,'z':float(a1['Z'][i,j]),
                    'residual_from_original_Q':float(a1['Z'][i,j]-a1['Q'][i]),'original_weight':float(w[j]),
                    'a_coefficient':float(coefficient[j]),'local_weight':float(local[j]),'normalized_weight_ratio':float(ratios[j]),
                    'significant_downweight':bool(ratios[j]<=settings['significant_weight_ratio'])})
    write_rows(OUT/'typical_cases.csv',case_rows);write_rows(OUT/'typical_case_local_weights.csv',local_rows)
    write_rows(OUT/'case_selection_summary.csv',selections)
    return case_rows,selections


def export_candidate(dataset,data,result,delta,high):
    """Export candidate separately with original Q/K and an explicit inactive flag."""
    columns=list(data['meta'][0])+['Q','K','Q_robust_candidate','change','high_conflict','delta','final_policy']
    with (OUT/f'{dataset}_robust_candidate.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for i,meta in enumerate(data['meta']):
            writer.writerow({**meta,'Q':float(data['Q'][i]),'K':float(data['K'][i]),'Q_robust_candidate':float(result['Q_robust'][i]),
                'change':float(result['change'][i]),'high_conflict':bool(high[i]),'delta':delta,'final_policy':'B_Q_plus_K; candidate_not_activated'})


def table(values,columns):
    """Render actual computed results; no fixed final statistics."""
    def cell(v):
        if v is None:return '未定义'
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'
        return str(v).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |']+
        ['| '+' | '.join(cell(r.get(k)) for k in columns)+' |' for r in values])


def make_figures(data,result,sensitivity,domain_rows,extension,settings):
    """Draw candidate movement and ranks; all statistics use full observations."""
    FIG.mkdir(parents=True,exist_ok=True);files=[]
    def save(fig,name):
        fig.tight_layout();fig.savefig(FIG/name,dpi=settings['figure_dpi'],bbox_inches='tight');plt.close(fig);files.append(name)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4));x=[r['delta'] for r in sensitivity]
    axes[0].plot(x,[r['mean_abs_change_high'] for r in sensitivity],'o-',label='High K mean |change|')
    axes[0].plot(x,[r['mean_abs_change_P95_conflict'] for r in sensitivity],'s-',label='Extreme K mean |change|')
    axes[0].plot(x,[r['max_abs_change_all'] for r in sensitivity],'^-',label='Maximum |change|')
    axes[0].set(xlabel='delta',ylabel='Absolute score change',title='A1 movement across delta');axes[0].legend(fontsize=8)
    axes[1].plot(x,[r['spearman_Q_Qrobust'] for r in sensitivity],'o-',label='All records')
    axes[1].plot(x,[r['spearman_high_Q_Qrobust'] for r in sensitivity],'s-',label='High K only')
    axes[1].set(xlabel='delta',ylabel='Spearman(Q, candidate)',title='Global vs high-conflict rank stability');axes[1].legend()
    save(fig,'01_delta_sensitivity.png')
    high=data['high'];rng=np.random.default_rng(settings['random_state'])
    indexes=np.flatnonzero(high);indexes=np.sort(rng.choice(indexes,min(len(indexes),settings['scatter_max_points']),replace=False))
    fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].scatter(data['Q'][indexes],result['Q_robust'][indexes],s=6,alpha=.3)
    axes[0].plot([0,1],[0,1],color='#777777',linestyle='--');axes[0].set(xlabel='Original Q',ylabel='Q_robust candidate',title='A1 high-conflict scores')
    axes[1].hist(result['change'][high],bins=60,color='#4777a8');axes[1].axvline(0,color='#777777',linestyle='--')
    axes[1].set(xlabel='Candidate minus original Q',ylabel='Count',title='A1 high-conflict adjustment directions');save(fig,'02_A1_candidates.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4));labels=[r['domain'] for r in domain_rows]
    axes[0].bar(labels,[r['mean_abs_change_high'] for r in domain_rows],color='#4777a8')
    axes[0].set(ylabel='High K mean absolute change',title='A1 domain adjustment sizes')
    axes[1].bar(labels,[r['up_rate_high'] for r in domain_rows],label='Up',color='#4777a8')
    axes[1].bar(labels,[r['down_rate_high'] for r in domain_rows],bottom=[r['up_rate_high'] for r in domain_rows],label='Down',color='#c07842')
    axes[1].set(ylabel='High K fraction',title='A1 domain directions');axes[1].legend()
    for ax in axes:ax.tick_params(axis='x',rotation=30)
    save(fig,'03_domain_adjustments.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(dataset,domain) in zip(axes,[('A2','arxiv'),('A3','github')]):
        mask=high&(data['domains']==domain)
        ax.hist(result['change'][mask],bins=np.linspace(-.16,.16,65),density=True,histtype='step',label=f'A1-{domain}')
        ext=extension[dataset];ax.hist(ext['result']['change'][ext['data']['high']],bins=np.linspace(-.16,.16,65),density=True,histtype='step',label=f'{dataset} full')
        ax.axvline(0,linestyle='--',color='#777777');ax.set(xlabel='Candidate minus Q',ylabel='Density',title=f'{domain}: high-conflict changes');ax.legend()
    save(fig,'04_extension_adjustments.png')
    write_rows(OUT/'figure_manifest.csv',[{'path':p.relative_to(ROOT).as_posix()} for p in [FIG/n for n in files]])


def build_report(sensitivity,selected,weight_rows,domains,extension_rows,cases,selections,equal_rows,overlaps,protected_n):
    """Report evidence and keep the candidate inactive absent verified improvement."""
    recommended=next(r for r in sensitivity if r['delta']==selected)
    material=[r for r in cases if r['case_group'] in ['up','down','unchanged']]
    a2=next(r for r in extension_rows if r['dataset']=='A2' and r['view']=='full')
    a3=next(r for r in extension_rows if r['dataset']=='A3' and r['view']=='full')
    equal_a1=next(r for r in equal_rows if r['dataset']=='A1')
    # Actual inspected cases; short cases were viewed in full, long cases only
    # at the opening. These observations are not independent human labels.
    observation_map={
        'BkiUfI_xK1yAgXBVGnXn':('开头片段','开头为瑞典语书籍简介。低分 DSIR/词数/句数被降低权重后上调；可能降低长度效用的影响，但不能据片段确认真实质量提高。'),
        'BkiUbvjxK6wB9mn49q52':('开头片段','开头为含公式/引用的介子跃迁论文；词数、数值字符及 DSIR 低分被减权。专业用途可解释评分分歧，但不足以证明上调幅度正确。'),
        'BkiUdmg5qYVBXTJ3BsEF':('开头片段','开头为天文学 LaTeX 论文，词数/独特词/大写比例低分被减权；原K仅略高于阈值，却产生较大上调，提醒分段激活可能形成边界跳变。'),
        'BkiUdPY5qWTD6fkJfvzP':('完整短文本','全文为23字符的重复大写串；降低部分高分指标后下调，方向与缺乏自然语言内容的观察相符，但不等于已获独立准确性标签。'),
        'BkiUainxK1Thg9qFa8nv':('完整短文本','全文为重复沙发关键词行。部分高分格式/长度指标减权后下调，解释方向合理；只能作为案例线索。'),
        'BkiUbRzxK3xgpdbchQki':('完整短文本','全文是短 PHP 配置语句。下调不一定改善代码用途评价，说明不能把偏离较远的指标自动视为错误。'),
        'BkiUdwo5qhLBXeFpceh2':('完整短文本','全文为 Android XML 颜色选择配置，调整很小；两侧指标减权接近平衡，不足以验证代码实际质量。')}
    observed=[];seen=set()
    for r in cases:
        if r['id'] not in observation_map or r['id'] in seen:continue
        seen.add(r['id']);scope,observation=observation_map[r['id']]
        observed.append({k:r[k] for k in ['id','domain','Q','K','Q_robust','change']}|
            {'annotation_status':'AI辅助复核，非最终人工标注','inspection_scope':scope,
             'observation':observation,'content_excerpt':r['content'][:1200]})
    write_rows(OUT/'typical_case_AI_observations.csv',observed)
    notes=['# PHASE 5B：高冲突样本稳健重聚合验证',
        '第一小问 Q/Z/全局权重、第一阶段 K/P90/P95 全部固定。输入直接来自三份已有冲突 CSV；没有调用预处理、CRITIC、Q/K 拟合或指数惩罚。输出中的 Q_robust 仅为候选，不覆盖原 Q。',
        '## 算法与统计口径',
        '高冲突使用阈值文件的完整精度 A1 P90，不使用显示四舍五入的 0.333206。普通样本 Q_robust=原 Q（精确复制）。高冲突：r=z−原Q，a=min(1,δ/(|r|+1e−8))，局部权重=w×a/Σ(w×a)，候选评分=Σ局部权重×z。只做一次重聚合，中心固定为原 Q；不迭代。',
        'a 只是残差启发式系数，尚未校准为真实可靠性。K 继续表示内部指标分歧，不作为额外低质量程度。方向统计以 |变化|≤1e−10 为数值近似不变；这个容差不是实际意义上的最小重要差异。低冲突为 K≤原P50，普通为 K≤原P90，极高为 K>原P95。',
        '## 五个 δ 的 A1 全量结果',
        table(sensitivity,['delta','pearson_Q_Qrobust','spearman_Q_Qrobust','spearman_high_Q_Qrobust','mean_abs_change_low','mean_abs_change_high','mean_abs_change_P95_conflict','P95_abs_change_high','max_abs_change_all','up_rate_high','down_rate_high']),
        '全体/高冲突的变化方向计数、全体排名变化、最大调整、低/普通组是否精确不变均保存于 delta_sensitivity_A1.csv。全局稳定性部分源自只调整原 P90 以上样本，不能代替高冲突组内稳定性。',
        f'## 保守候选 δ={selected:g}',
        f"按照用户较大 δ 优先原则，推荐 δ={selected:g} 作为验证候选；它同时取得网格内最小高冲突平均绝对变化和最小最大变化。未按最大效果选择，未增加拍脑袋的接受门槛。高冲突平均 |变化|={recommended['mean_abs_change_high']:.6f}，P95 |变化|={recommended['P95_abs_change_high']:.6f}，最大 |变化|={recommended['max_abs_change_all']:.6f}。最大调整不能称为对每个样本都很小，需逐例检查。",
        '## 极端指标降权是否实际发生',
        table(weight_rows,['delta','mean_local_weight_L1_change','original_effective_metric_count','mean_robust_effective_metric_count_high','extreme_residual_normalized_weight_down_rate','extreme_residual_significant_downweight_rate','max_local_weight_high']),
        '显著降权的操作定义为归一化局部权重/原权重≤0.5，见配置；这是供查看的诊断界线，不是优化参数。a<1 不保证归一化后的权重比原来小，因此单独记录局部实际权重。最大绝对残差并列时逐个计数。有效指标数为 1/Σw²；它下降表明局部权重可能更集中，不能把结果自动称为全面更稳健。权重表仅证明极端指标权重变化，不证明真实质量误差变小。',
        '## 典型核验材料',
        table(selections,['group','available_n','requested_n','selected_n','definition']),
        '上调/下调各选已有文本中变化最大的真实案例；近似不变按数值容差筛选。普通样本精确不变是算法规定，不构成高冲突重聚合有效性证据。另保存高冲突组中调整最小的十条，明确标为 smallest_change，不声称其变化接近零。案例分组可复用同一文本。',
        table(material,['case_group','id','domain','Q','K','Q_robust','change','most_downweighted_metrics']),
        'typical_cases.csv 保留完整原22维Z、Q/K、候选变化、降权指标、全部局部权重/系数以及已有 content；不重新读取任何原始 JSONL。typical_case_local_weights.csv 给出逐指标审计明细。当前仅有第一阶段的人工确认/AI片段线索，尚无对本轮重聚合解释改善的独立人工标注。',
        '下表与 typical_case_AI_observations.csv 是明确标记的 AI 辅助观察；短文本检查全文，长文仅看开头片段。不是最终人工标注，不能作为外部实验标签。',
        table(observed,['id','domain','Q','Q_robust','change','inspection_scope','observation']),
        '## A2/A3 固定候选验证',
        'δ 只从 A1 选择，A2/A3 固定同一 δ 与 epsilon、全局权重和 A1 P90。full 保留跨集重叠；nonoverlap 用既有索引独立对照，不删除数据。',
        table(extension_rows,['dataset','domain','view','delta','n','n_high','pearson_Q_Qrobust','spearman_Q_Qrobust','spearman_high_Q_Qrobust','mean_abs_change_high','mean_abs_change_P95_conflict','max_abs_change_all','up_rate_high','down_rate_high']),
        table(overlaps,['dataset','shared_n','nonoverlap_n','max_shared_candidate_abs_difference']),
        f"实际复现具有领域差异：A2-arxiv 全体 Spearman={a2['spearman_Q_Qrobust']:.6f}、高冲突平均 |变化|={a2['mean_abs_change_high']:.6f}、上调率={a2['up_rate_high']:.6f}；A3-github 分别为 {a3['spearman_Q_Qrobust']:.6f}、{a3['mean_abs_change_high']:.6f}、{a3['up_rate_high']:.6f}。arxiv 的单向较大上调也已存在于 A1 域内，并非仅扩展集造成；不能用混合领域 A1 全体的高相关概括每个领域的排名稳定性。",
        '## 辅助赋权敏感性',
        '仅对选定 δ 作等权辅助对照：中心为原冲突表已保存 Q_equal，采用同一高冲突掩码和 δ 做一次等权局部重聚合。不会替代主权重、原 Q 或阈值。比较主候选与等权候选的距离是否比原有 Delta_Q 小；该距离也不是真实误差。',
        table(equal_rows,['dataset','n_high','mean_original_Delta_Q_high','mean_candidate_Delta_Q_high','candidate_over_original_ratio','candidate_difference_reduced_rate_high']),
        f"在这个辅助对照下，A1 高冲突组差异均值由 {equal_a1['mean_original_Delta_Q_high']:.6f} 增至 {equal_a1['mean_candidate_Delta_Q_high']:.6f}，比值={equal_a1['candidate_over_original_ratio']:.6f}。只对 {equal_a1['candidate_difference_reduced_rate_high']:.6f} 的样本减小了该距离。因此不能声称此算法已经降低对赋权方法的敏感性。此结论限于已声明的等权候选对照。",
        '## 当前最终建议：B，保留 (Q,K)',
        '本轮可以确认普通样本不变、候选整体排名稳定及极端残差降权；这些是算法/内部统计性质。没有充分证据证明典型文本解释或外部评价准确性得到改善。局部重聚合会放大剩余指标的相对权重，最大调整和高冲突组排名需要单独审查，不能将局部系数视为已验证可靠性。',
        '结合 arxiv 较大的单向上调、域内排名变化、辅助赋权敏感性未下降以及尚无独立人工解释改善证据，本阶段建议采用 B：(Q,K)。δ 候选及 Q_robust 单独保留供人工复核，不激活分段 Q_final。部分短噪声案例下调方向可解释，但不能以少数案例覆盖域级不稳定性；不会反向修改第一小问。',
        f'原始附件、第一小问和冲突第一阶段的 {protected_n} 个已有文件均进行前后 SHA-256 核验。独立检查见 verification.csv。未测试指数惩罚或 lambda；不开始问题一第3小问。']
    (ROOT/'outputs/report/question1_part2_phase5b_robust_validation.md').write_text('\n\n'.join(notes)+'\n',encoding='utf-8')


def main():
    """Compare candidates on A1, apply conservative candidate and stop."""
    OUT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'robust_validation.log',encoding='utf-8')])
    settings=yaml.safe_load((ROOT/'configs/robust_reaggregation.yaml').read_text(encoding='utf-8'))
    logging.info('Snapshot all frozen outputs and official source hashes')
    before=frozen_hashes();write_rows(OUT/'locked_inputs_before.csv',[{'path':p,'sha256':h} for p,h in before.items()])
    names=json.loads((ROOT/'outputs/preprocess/preprocessing_model_A1.json').read_text(encoding='utf-8'))['metric_order']
    if len(names)!=22 or len(set(names))!=22:raise ValueError('Locked metric order is not 22-dimensional')
    wr=rows(ROOT/'outputs/quality/critic_weights_A1.csv');wm={r['metric']:float(r['critic_weight']) for r in wr}
    if len(wr)!=22 or set(wm)!=set(names):raise ValueError('Locked weight metrics mismatch')
    w=np.array([wm[n] for n in names]);thresholds={r['quantile']:float(r['K_threshold']) for r in rows(ROOT/'outputs/conflict/conflict_thresholds_A1.csv')}
    sizes={r['dataset']:int(r['n']) for r in rows(ROOT/'outputs/conflict/dataset_conflict_summary.csv')}
    a1=read_input('A1',names,sizes['A1']);high=a1['K']>thresholds['P90'];a1['high']=high
    sensitivity=[];weight_summaries=[];metric_diagnostics=[]
    for delta in settings['deltas']:
        logging.info('A1 candidate delta=%s; only %d high-conflict records adjusted',delta,int(high.sum()))
        result=reaggregate(a1['Z'],a1['Q'],high,w,delta,settings['epsilon'])
        sensitivity.append(summarize(a1,result,np.ones(len(high),dtype=bool),delta,thresholds,settings,{'dataset':'A1','domain':'all','view':'full'}))
        summary,metrics=weight_diagnostics(a1,result,w,names,delta,high,settings);weight_summaries.append(summary);metric_diagnostics+=metrics
    selected=choose_delta(sensitivity)
    logging.info('Conservative A1 validation candidate selected: delta=%s; not activated',selected)
    result=reaggregate(a1['Z'],a1['Q'],high,w,selected,settings['epsilon'])
    write_rows(OUT/'delta_sensitivity_A1.csv',sensitivity)
    write_rows(OUT/'local_weight_diagnostics.csv',weight_summaries);write_rows(OUT/'local_weight_by_metric.csv',metric_diagnostics)
    export_candidate('A1',a1,result,selected,high)
    domain_rows=[summarize(a1,result,a1['domains']==domain,selected,thresholds,settings,{'dataset':'A1','domain':domain,'view':'full'}) for domain in sorted(set(a1['domains']))]
    write_rows(OUT/'domain_adjustment_summary_A1.csv',domain_rows)
    cases,selections=select_cases(a1,result,w,names,high,settings)
    extension_rows=[];overlaps=[];extension={};equal_rows=[]
    def equal_check(dataset,data,result):
        mask=data['high'];er=reaggregate(data['Z'],data['Q_equal'],mask,np.full(22,1/22),selected,settings['epsilon'])
        distance=np.abs(result['Q_robust'][mask]-er['Q_robust'][mask]);original=data['Delta_Q'][mask]
        equal_rows.append({'dataset':dataset,'n_high':int(mask.sum()),'mean_original_Delta_Q_high':float(original.mean()),
            'mean_candidate_Delta_Q_high':float(distance.mean()),'candidate_over_original_ratio':float(distance.mean()/original.mean()),
            'candidate_difference_reduced_rate_high':float((distance<original).mean())})
    equal_check('A1',a1,result)
    for dataset,domain in [('A2','arxiv'),('A3','github')]:
        logging.info('Apply fixed delta=%s to %s, no retuning',selected,dataset)
        data=read_input(dataset,names,sizes[dataset]);data['high']=data['K']>thresholds['P90']
        er=reaggregate(data['Z'],data['Q'],data['high'],w,selected,settings['epsilon'])
        extension[dataset]={'data':data,'result':er};export_candidate(dataset,data,er,selected,data['high']);equal_check(dataset,data,er)
        nonoverlap_ids={r['id'] for r in rows(ROOT/f'outputs/preprocess/{dataset}_{domain}_nonoverlap_ids.csv')}
        nonoverlap=np.array([r['id'] in nonoverlap_ids for r in data['meta']])
        for view,mask in [('full',np.ones(len(data['Q']),dtype=bool)),('nonoverlap',nonoverlap)]:
            extension_rows.append(summarize(data,er,mask,selected,thresholds,settings,{'dataset':dataset,'domain':domain,'view':view}))
        reference={r['id']:i for i,r in enumerate(a1['meta']) if r['domain']==domain}
        shared=[(reference[r['id']],i) for i,r in enumerate(data['meta']) if r['id'] in reference]
        if len(shared)!=len(reference) or nonoverlap.sum()+len(shared)!=len(data['Q']):raise ValueError('Overlap view differs from frozen index')
        maxdifference=max(abs(result['Q_robust'][i]-er['Q_robust'][j]) for i,j in shared)
        if maxdifference>1e-14:raise ValueError('Shared candidate scores differ')
        overlaps.append({'dataset':dataset,'shared_n':len(shared),'nonoverlap_n':int(nonoverlap.sum()),'max_shared_candidate_abs_difference':maxdifference})
    write_rows(OUT/'extension_validation.csv',extension_rows);write_rows(OUT/'shared_record_consistency.csv',overlaps)
    write_rows(OUT/'equal_weight_sensitivity_diagnostic.csv',equal_rows)
    policy={'recommended_delta_for_validation':selected,'epsilon':settings['epsilon'],'selection_dataset':'A1',
        'delta_selection_rule':'largest tested; empirically smallest high-conflict mean and maximum absolute movement',
        'fixed_high_K_threshold':thresholds['P90'],'final_recommendation':'B_Q_plus_K','candidate_activated':False,
        'reason':'No independently verified improvement in text explanation or external quality accuracy',
        'diagnostic_limitations':['arxiv rank stability substantially lower than pooled A1',
                                  'A1 high-conflict candidate/equal-candidate distance increased'],
        'model_definition':{'quality':'original Q','conflict':'original K'},'exponential_penalty_tested':False}
    (OUT/'model_recommendation.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    make_figures(a1,result,sensitivity,domain_rows,extension,settings)
    build_report(sensitivity,selected,weight_summaries,domain_rows,extension_rows,cases,selections,equal_rows,overlaps,len(before))
    after=frozen_hashes()
    if before!=after:raise ValueError('Frozen input changed; stop without repair')
    write_rows(OUT/'locked_inputs_integrity.csv',[{'path':p,'sha256_before':h,'sha256_after':after[p],'unchanged':True} for p,h in before.items()])
    (OUT/'config_snapshot.yaml').write_text(yaml.safe_dump(settings,sort_keys=False,allow_unicode=True),encoding='utf-8')
    logging.info('COMPLETED PHASE 5B; frozen files unchanged=%d; recommendation B; no exponent penalty or part 3',len(before))
    print(json.dumps({'recommended_delta':selected,'A1':next(r for r in sensitivity if r['delta']==selected),'weight_diagnostic':weight_summaries[-1],'equal_sensitivity':equal_rows},ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    try:main()
    except Exception:
        logging.exception('STOPPED PHASE 5B; no frozen input is repaired/refitted')
        raise
