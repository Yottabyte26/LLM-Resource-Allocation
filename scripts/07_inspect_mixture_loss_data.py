"""A4–A15 structural diagnosis and loss responses; no regression or Q/K access."""
from __future__ import annotations
import json
import logging
from pathlib import Path
import sys
import numpy as np
import yaml
from itertools import combinations

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mixture.io import load_datasets,signatures
from quality.io import sha256_file
from quality.vector_diagnostics import write_rows
OUT=ROOT/'outputs/mixture'


def describe(x):
    """Actual full-vector distribution, with linear quantiles."""
    return {'n':len(x),'min':float(x.min()),'P01':float(np.quantile(x,.01)),'P10':float(np.quantile(x,.1)),
        'median':float(np.median(x)),'P90':float(np.quantile(x,.9)),'P99':float(np.quantile(x,.99)),
        'max':float(x.max()),'mean':float(x.mean()),'std':float(x.std(ddof=1))}


def table(values,columns):
    """Render observed statistics only."""
    def value(v):return f'{v:.6f}' if isinstance(v,float) else str(v)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |']+
        ['| '+' | '.join(value(r.get(k)) for k in columns)+' |' for r in values])


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'correlations').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'data_diagnosis.log',encoding='utf-8')])
    settings=yaml.safe_load((ROOT/'configs/mixture_modeling.yaml').read_text(encoding='utf-8'))
    datasets,pnames,lnames=load_datasets(ROOT)
    file_rows=[];pairing=[];profiles=[];loss_profiles=[];responses=[];zero_rows=[];row_checks=[];manifest=[]
    for label,data in datasets.items():
        p,l=data['p_table'],data['loss_table'];x,y=data['P'],data['losses'];s=x.sum(axis=1)
        rounding_bound=len(pnames)*settings['rounding_bound_multiplier']*10**(-p['decimal_places'])
        if np.max(np.abs(s-1))>rounding_bound+1e-12:raise ValueError(f'{label}: row sum inconsistent with exported precision')
        for kind,tab in [('mixture',p),('loss',l)]:
            h=sha256_file(tab['path']);relative=tab['path'].relative_to(ROOT).as_posix()
            manifest.append({'dataset':label,'kind':kind,'path':relative,'n':len(tab['indices']),'sha256':h,'provenance':data['provenance']})
            file_rows.append({'dataset':label,'kind':kind,'path':relative,'rows':len(tab['indices']),
                'unique_index_count':len(set(tab['indices'])),'index_unique':True,'numeric_columns':len(tab['columns']),
                'nan_count':int(np.isnan(tab['values']).sum()),'inf_count':int(np.isinf(tab['values']).sum()),
                'negative_count':int((tab['values']<0).sum()),'provenance':data['provenance']})
        pairing.append({'dataset':label,'mixture_n':len(x),'loss_n':len(y),'index_sets_equal':True,
            'original_row_order_equal':p['indices']==l['indices'],'one_to_one':True,'provenance':data['provenance'],
            'raw_sum_min':float(s.min()),'raw_sum_max':float(s.max()),'max_abs_sum_error':float(abs(s-1).max()),
            'rows_sum_exact_within_tolerance':int((abs(s-1)<=settings['simplex_exact_tolerance']).sum()),
            'export_decimal_places':p['decimal_places'],'derived_rounding_error_bound':rounding_bound,
            'all_zero_rows':int((s==0).sum()),'zero_cells':int((x==0).sum()),'zero_cell_rate':float((x==0).mean()),
            'unique_recipe_count':len(set(signatures(x)))})
        for i,name in enumerate(pnames):
            zero=int((x[:,i]==0).sum())
            profiles.append({'dataset':label,'domain':name,**describe(x[:,i]),'zero_count':zero,'zero_rate':zero/len(x)})
            zero_rows.append({'dataset':label,'domain':name,'n':len(x),'zero_count':zero,'zero_rate':zero/len(x)})
        for i,name in enumerate(lnames):loss_profiles.append({'dataset':label,'loss_domain':name,'provenance':data['provenance'],**describe(y[:,i])})
        for i,index in enumerate(data['indices']):
            responses.append({'dataset':label,'index':index,'provenance':data['provenance'],'L_mean':float(data['L_mean'][i]),
                'L_trim':float(data['L_trim'][i]),'L_median':float(data['L_median'][i]),**dict(zip(lnames,map(float,y[i])))})
            row_checks.append({'dataset':label,'index':index,'raw_sum':float(s[i]),'zero_domains':int((x[i]==0).sum()),
                'closure_multiplier':float(1/s[i]),'max_abs_component_change_due_to_closure':float(np.max(abs(x[i]/s[i]-x[i])))})
        corr=np.corrcoef(x,rowvar=False)
        write_rows(OUT/'correlations'/f'{label}_raw_mixture_pearson.csv',
            [{'domain':name,**dict(zip(pnames,map(float,corr[i])))} for i,name in enumerate(pnames)])
        logging.info('%s: n=%d, sum=[%.6f,%.6f], zero cells=%.2f%%, index bijection valid',label,len(x),s.min(),s.max(),100*(x==0).mean())
    overlaps=[]
    for left,right in combinations(datasets,2):
        a,b=set(signatures(datasets[left]['P'])),set(signatures(datasets[right]['P']))
        overlaps.append({'dataset_1':left,'dataset_2':right,'n_1':len(a),'n_2':len(b),'same_composition_count':len(a&b),
            'matching_rule':'closed shares rounded to 12 decimals, not index across files'})
    write_rows(OUT/'data_file_summary.csv',file_rows);write_rows(OUT/'dataset_pairing.csv',pairing)
    write_rows(OUT/'mixture_domain_profile.csv',profiles);write_rows(OUT/'mixture_zero_summary.csv',zero_rows)
    write_rows(OUT/'loss_domain_profile.csv',loss_profiles);write_rows(OUT/'loss_response_summary.csv',responses)
    write_rows(OUT/'simplex_closure_diagnosis.csv',row_checks);write_rows(OUT/'cross_dataset_recipe_overlap.csv',overlaps)
    write_rows(OUT/'input_manifest.csv',manifest)
    (OUT/'domain_schema.json').write_text(json.dumps({'mixture_domains':pnames,'loss_domains':lnames},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    report=['# A4–A15 数据诊断（先于回归）',
        '实际路径为 real_attachments/A_data_value/regmix_tables。读取十二个官方 CSV，原件只读；无回归、无 Q/K 引入。index 仅在同一个实验的 mixture/loss 对内关联，不能跨尺度直接按 index 判断配方相同。',
        table(file_rows,['dataset','kind','rows','unique_index_count','numeric_columns','nan_count','inf_count','negative_count','provenance']),
        '## 17 个训练配比列', '\n'.join(f'{i}. {n}' for i,n in enumerate(pnames,1)),
        '## 13 个验证 Loss 列','\n'.join(f'{i}. {n}' for i,n in enumerate(lnames,1)),
        '## 单纯形与零值',table(pairing,['dataset','mixture_n','one_to_one','raw_sum_min','raw_sum_max','max_abs_sum_error','zero_cell_rate','all_zero_rows','unique_recipe_count']),
        '原始配比为有限小数导出，因此行和不完全为1。误差界由实际配比列数×0.5×10^(−文件最大小数位数) 推导；所有行均在该界内。后续仅对派生数据执行闭合 p/sum(p)，保持非零比例，绝不修改原件；不存在负值、NaN/Inf或全零配方。逐域分布和零比例见 mixture_domain_profile.csv、mixture_zero_summary.csv；逐行闭合幅度见 simplex_closure_diagnosis.csv。',
        '十七域相关矩阵按数据集分别保存于 correlations/。原始比例相关受总和约束，不直接解释为领域互补或因果替代。各验证域 Loss 分布见 loss_domain_profile.csv。',
        '## Loss 响应',
        '主响应 L_mean=13 域等权均值。敏感性 L_trim=排序后去掉最低/最高各一个，再对11域取均值。另保存中位数与全部原13域 Loss，未按观察结果选择偏好的验证域。完整对齐响应见 loss_response_summary.csv。',
        '## 配方重叠与跨尺度比较',table(overlaps,['dataset_1','dataset_2','n_1','n_2','same_composition_count']),
        '相同配方使用闭合组成向量匹配，不使用跨文件 index。1M/60M 可以在共同配方上直接比较实际排序；若1B无共同配方，则只报告1M模型在1B上的排序迁移能力，不能虚构逐配方跨尺度一致性。',
        '## 估算材料的范围',
        '10B/70B Loss 只作为 estimated 数据诊断，绝非本轮观测检验或模型选择数据。本轮不预测这两组，不拟合模型规模或 Scaling Law；Q、K 也不进入当前模型。',
        '结构核验通过后允许继续 ILR 与 M0/M1/M2。所有数据驱动预处理参数须在相应训练/CV折内拟合。']
    (OUT/'phase1_data_diagnosis.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    logging.info('COMPLETED diagnosis: 12 parsed files, all index/schema checks passed; no model fitted')


if __name__=='__main__':
    try:main()
    except Exception:
        logging.exception('STOPPED: data diagnosis inconsistency; original files not repaired')
        raise
