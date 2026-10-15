"""Audit official inputs and frozen outputs; no fitting and no optimization."""
from __future__ import annotations
import hashlib
import json
import logging
import re
import zipfile
import xml.etree.ElementTree as ET
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import trim_mean
from .costs import cost_components, quality_increment_per_token, marginal_quality_per_token

NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}


def omml_text(node):
    """Preserve superscripts/subscripts/fractions when reading original OOXML."""
    tag = node.tag.split('}')[-1]
    if tag == 't':
        return node.text or ''
    if tag in ('sSup', 'sSub'):
        base = omml_text(node.find('m:e', NS))
        part = omml_text(node.find('m:sup' if tag == 'sSup' else 'm:sub', NS))
        return base + ('^' if tag == 'sSup' else '_') + '{' + part + '}'
    if tag == 'f':
        return '(' + omml_text(node.find('m:num', NS)) + ')/(' + omml_text(node.find('m:den', NS)) + ')'
    return ''.join(omml_text(child) for child in node)


def numeric_expression(expr):
    """Parse only decimal coefficients times base-ten powers; never eval text."""
    expr = ''.join(expr.split()).replace('−', '-').replace('×', '*')
    match = re.fullmatch(r'(?:(\d+(?:\.\d+)?)\*)?10\^\{(-?\d+)\}', expr)
    if match:
        return float(match[1] or 1)*10.0**int(match[2])
    if re.fullmatch(r'\d+(?:\.\d+)?', expr):
        return float(expr)
    raise ValueError('Unsupported official numeric expression: ' + expr)


def save_csv(path, rows):
    """Write full floating precision UTF-8 CSV; paths are new Phase 1 outputs."""
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def save_json(path, data):
    """Reject nonfinite JSON numbers and keep full Python floating precision."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(frame, columns=None):
    """Make simple Markdown tables without an optional tabulate dependency."""
    a = frame if columns is None else frame[columns]
    lines = ['| ' + ' | '.join(a.columns) + ' |', '| ' + ' | '.join(['---']*len(a.columns)) + ' |']
    for row in a.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(format(x, '.10g') if isinstance(x, float) else str(x).replace('|', '/') for x in row) + ' |')
    return '\n'.join(lines)


def audit_official(root, out):
    """Extract the actual Word formulas, fail on any unsupported structural change."""
    docx = next(root.glob('*.docx'))
    with zipfile.ZipFile(docx) as archive:
        xml = archive.read('word/document.xml')
    document = ET.fromstring(xml)
    paras = [omml_text(p) for p in document.findall('.//w:p', NS)]
    (out/'official_word_extraction.txt').write_text('\n'.join(f'{i}: {p}' for i,p in enumerate(paras)), encoding='utf-8')
    selected = {}
    for key, phrase in [('budget', '为便于对照'), ('train', '基础训练开销：'),
                        ('attention', '长文本注意力开销：'), ('increment', 'C_{Q}=D'),
                        ('exponential', '指数型：'), ('power', '幂函数型：'), ('logarithmic', '对数渐进型：')]:
        found = [(i,p) for i,p in enumerate(paras) if phrase in p
                 and (key != 'increment' or p.strip().startswith(phrase))]
        if len(found) != 1:
            raise ValueError(f'Official formula ambiguous: {key}: {found}')
        selected[key] = found[0]
    compact = lambda p: ''.join(p.split())
    train_text = compact(selected['train'][1])
    if 'C_{train}=6ND' not in train_text or 'C_{attn}=ηNDL_{ctx}' not in compact(selected['attention'][1]):
        raise ValueError('Unexpected official training/attention formula')
    if 'C_{Q}=D[g(Q)-g(Q_{0})]_{+}' not in compact(selected['increment'][1]):
        raise ValueError('Unexpected official incremental quality formula')
    eta_expr = re.search(r'η=([^，]+)', selected['attention'][1])[1]
    budgets = [numeric_expression('10^{'+e+'}') for e in re.findall(r'10\^\{(\d+)\}', selected['budget'][1])]
    if len(budgets) != 3:
        raise ValueError('Could not recover three official budgets from OMML')
    official = dict(training_coefficient=6.0, eta=numeric_expression(eta_expr),
                    budgets_FLOPs=budgets, budget_status='official_recommendation_not_mandatory', quality_costs={})
    expected = {'exponential': 'g(Q)=γe^{λQ}', 'power': 'g(Q)=γQ^{λ}', 'logarithmic': 'g(Q)=γln(1+λQ)'}
    for family in expected:
        idx, text = selected[family]
        if expected[family] not in compact(text):
            raise ValueError('Appendix B family differs from implemented formula: '+family)
        gamma_expr = re.search(r'γ=([^，]+)', text)[1]
        lam_expr = re.search(r'λ=([^；。]+)', text)[1]
        official['quality_costs'][family] = dict(gamma=numeric_expression(gamma_expr),
              **{'lambda': numeric_expression(lam_expr)}, source_paragraph_index_zero_based=idx,
              gamma_unit='FLOPs/token (dimensionally implied by official total budget)')
    save_json(out/'official_cost_parameters.json', official)
    evidence = []
    for key, (i,text) in selected.items():
        evidence.append(f'### {key}\n\n来源：`{docx.name}`，`word/document.xml` 第 {i} 号段落（零起始）。\n\n```text\n{text}\n```')
    (out/'problem3_formula_audit.md').write_text(
        '# 问题三官方公式核验\n\n直接解析原始 OMML，上下标保留；未修改 Word。\n\n'
        '## A 官方明确给出\n\n'+'\n\n'.join(evidence)+
        '\n\n## B 解析推导\n\nC_train=C_attn 在 N,D>0 时给出 L_crit=6/eta。'
        'CQ 增量在 Q=Q0 为零；三种 g 在 Q∈[Q0,1] 均递增且端点有限。'
        'gamma 必须解释为每 token 的 FLOPs 代理，lambda_Q 是无量纲形状参数；它与冻结 lambda_p(N) 不同。'
        '\n\n## C 研究者设定\n\nQ0 的聚合/跨域代理映射、主支持域交集、外推域、对数预算网格、经验配方候选集及未来转移判据属于建模选择。'
        '架构最大窗口是可行场景依据，不是每次训练真实序列长度；跨架构与 N 的自由组合也是简化代理假设。'
        '\n\n## D 旧草稿不继承\n\n未读取旧优化答案。Q0=0.8、质量成本上限 M、旧 N*/D*/Q*、旧转移预算均不继承。'
        'eta=2e-4 和三档预算此次由官方 XML 独立恢复，因此可采用；不是沿用旧稿。'
        '对数函数不能替换为 -ln(1-Q)；三类函数没有题给总成本上限 M。\n', encoding='utf-8')
    import pypdf
    logging.getLogger('pypdf').setLevel(logging.CRITICAL)
    pdf = next(root.glob('*.pdf'))
    reader = pypdf.PdfReader(pdf)
    # Only relevant complete pages; malformed CMaps in source font raise warnings,
    # but relevant C7/schema/unit text is extractable and checked below.
    pages = {str(i+1): reader.pages[i].extract_text() for i in (3,4,10,11,12)}
    text = '\n'.join(pages.values())
    if 'max_position_embeddings' not in text or 'N_params_B' not in text or 'D_tokens_B' not in text:
        raise ValueError('Official PDF relevant schema/unit extraction failed')
    (out/'official_data_description_excerpt.txt').write_text('\n\n'.join('PAGE '+k+'\n'+v for k,v in pages.items()), encoding='utf-8')
    return official, [docx, pdf]


def diagnose_context(root, out, official):
    """Read C7 actual columns first and select only observed context values."""
    path = root/'real_attachments/C_efficiency_evolution/model_architecture_metadata.csv'
    c7 = pd.read_csv(path, float_precision='round_trip')
    required = {'model_name','n_layers','n_heads','d_model','max_position_embeddings'}
    if not required.issubset(c7.columns):
        raise ValueError('C7 missing required actual fields: '+str(required-set(c7.columns)))
    vals = pd.to_numeric(c7.max_position_embeddings, errors='coerce')
    valid = vals[np.isfinite(vals) & (vals>0) & (vals==np.floor(vals))]
    if valid.empty:
        raise ValueError('No finite positive integer context windows')
    qlevels = [0,.01,.05,.10,.25,.50,.75,.90,.95,.99,1]
    names = ['min','P01','P05','P10','P25','median','P75','P90','P95','P99','max']
    stats = dict(zip(names, valid.quantile(qlevels).to_list()))
    stats.update(rows=len(c7), valid_count=len(valid), invalid_count=len(c7)-len(valid),
                 unique_value_count=valid.nunique(), mean=valid.mean(), std=valid.std(ddof=1))
    save_csv(out/'C7_context_summary.csv', [stats])
    freq = valid.value_counts().sort_index().rename_axis('L_ctx').reset_index(name='support_count')
    freq['L_ctx'] = freq.L_ctx.astype(int)
    freq['support_fraction'] = freq.support_count/len(valid)
    save_csv(out/'C7_context_frequency.csv', freq)
    save_csv(out/'C7_field_diagnosis.csv', [{'field': key, 'dtype': str(c7[key].dtype),
        'missing_count': int(c7[key].isna().sum()), 'unique_value_count': c7[key].nunique()} for key in c7])
    save_csv(out/'C7_architecture_inventory.csv', c7)
    scenarios = freq.copy()
    scenarios['scenario_type'] = ['short' if x==valid.min() else 'medium_median' if x==valid.median()
                                  else 'long' if x==valid.max() else 'observed_frequent' for x in scenarios.L_ctx]
    scenarios['evidence_source'] = path.relative_to(root).as_posix()
    scenarios['attention_to_training_ratio'] = official['eta']*scenarios.L_ctx/official['training_coefficient']
    save_csv(out/'context_scenarios.csv', scenarios)
    crit = official['training_coefficient']/official['eta']
    save_csv(out/'context_critical_analysis.csv', [dict(eta_scheme='official_main', eta=official['eta'],
        L_ctx_crit=crit, within_C7_observed_range=bool(valid.min()<=crit<=valid.max()),
        exact_C7_value=bool((valid==crit).any()), models_above_critical=int((valid>crit).sum()),
        models_equal_critical=int((valid==crit).sum()), models_below_critical=int((valid<crit).sum()),
        valid_models=len(valid), fraction_above=float((valid>crit).mean()), evidence='Official Word Question 3')])
    (out/'C7_diagnosis.md').write_text('# C7 真实结构诊断\n\n'
        f'文件：`{path.relative_to(root).as_posix()}`。实际 {len(c7)} 行 × {len(c7.columns)} 列。\n\n'
        f'字段：{", ".join(c7.columns)}。模型名重复 {c7.model_name.duplicated().sum()}；完全重复 {c7.duplicated().sum()}。'
        f'总缺失单元格 {c7.isna().sum().sum()}；有效上下文 {len(valid)}。\n\n'
        'hidden_size 对应磁盘 d_model。没有独立参数量 N 字段，不从模型名猜参数量；没有明确 training tokens 字段。'
        'training_data_TB 字段含有数值，但单位语义不能由列名可靠判定为 token，不能转换成 D_tokens_B。'
        '参数量和训练数据上下界来自冻结问题二，不从该列反推。\n\n'+table(pd.DataFrame([stats]))+
        '\n\n分位数采用线性插值，分位数可能不是实际窗口；场景只取实际观测离散值。\n\n'+table(scenarios)+
        f'\n\neta={official["eta"]} 是题给简化代理，不需要也没有用 C7 重新估计。单位为每参数、每训练 token、每窗口位置的 FLOPs 系数。'
        f'临界窗口 {crit:g} 在观测最小最大值之间，但不是观测离散值；超临界模型 {(valid>crit).sum()}。'
        'C7 频数描述本附件，不代表模型市场总体分布；最大支持窗口不证明所有训练序列都达到该长度。\n', encoding='utf-8')
    return c7, scenarios, stats, [path]


def frozen_inputs(root, out):
    """Load all coefficients at saved precision; do not fit or mutate prior results."""
    path = root/'outputs/scaling/phase4/generalized_scaling_final.json'
    frozen = json.loads(path.read_text(encoding='utf-8'))
    from scaling.mixture_transfer import frozen_metadata, predict_ilr
    meta = frozen_metadata(root)
    pref = np.array(frozen['reference']['p_ref'])
    if len(pref)!=17 or not np.isfinite(pref).all() or (pref<=0).any() or not np.isclose(pref.sum(),1,atol=1e-14,rtol=0):
        raise ValueError('Frozen reference recipe invalid')
    pred = float(predict_ilr(np.log(pref)[None,:]@np.asarray(meta['ilr_basis']).T, meta)[0])
    if not np.isclose(pred,frozen['reference']['M_ref'],atol=1e-12,rtol=0):
        raise ValueError('Frozen M_ref disagrees with frozen ILR model')
    rows=[]
    for module, key, evidence, support in [('L0','classical','Class A','B1; B3 interpolation; B4/B5 comparability-limited'),
            ('Delta_Q','quality','Class B','B6 semisynthetic local; B7 difference validation failed')]:
        for name,value in frozen[key].items():
            rows.append(dict(module=module, parameter=name, value=json.dumps(value,ensure_ascii=False),
                             evidence_class=evidence, source=path.relative_to(root).as_posix(), support=support))
    rows.extend([dict(module='M(p)',parameter='frozen_model_metadata',value=json.dumps(meta,ensure_ascii=False),
                     evidence_class='Class C',source='outputs/mixture/final_model_metadata.json',support='ILR first-order ElasticNet; no fit'),
                 dict(module='M(p)',parameter='reference',value=json.dumps(frozen['reference'],ensure_ascii=False),evidence_class='Class C',
                      source=path.relative_to(root).as_posix(),support='fixed p_ref and M_ref')])
    for knot in frozen['lambda_knots']:
        rows.append(dict(module='lambda_p',parameter='knot_'+str(knot['N_params_B']),value=json.dumps(knot),
                         evidence_class='Class C' if knot['evidence']=='observed' else 'Class D',
                         source=path.relative_to(root).as_posix(),support=knot['evidence']))
    save_csv(out/'frozen_objective_summary.csv',rows)
    return frozen,meta,[path,root/'outputs/mixture/final_model_metadata.json', root/'outputs/mixture/stage2/final_model_policy.json']


def baseline_quality(root,out,frozen):
    """Compare robust frozen-score aggregations; expose rather than hide proxy mass."""
    base=root/'outputs/quality/domain_quality_summary_A1.csv'
    domain=pd.read_csv(base,float_precision='round_trip')
    samples=root/'outputs/quality/A1_sample_quality.csv'
    values=pd.read_csv(samples,usecols=['Q'],float_precision='round_trip').Q.to_numpy()
    if len(values)!=int(domain.n.sum()) or not np.isfinite(values).all():
        raise ValueError('Frozen A1 Q records and domain count disagree')
    map_path=root/'outputs/mixture/stage2/domain_quality_conflict_mapping.csv'
    alt_path=root/'outputs/mixture/stage2/alternative_domain_quality_mapping.csv'
    names=frozen['reference']['domain_order']; p=np.array(frozen['reference']['p_ref'])
    mapping=pd.read_csv(map_path,float_precision='round_trip').set_index('mixture_domain').loc[names].reset_index()
    alt=pd.read_csv(alt_path,float_precision='round_trip').set_index('mixture_domain').loc[names]
    actual=domain.set_index('domain').Q_domain
    if not np.allclose(mapping.Q_domain,[actual[x] for x in mapping.quality_domain],rtol=0,atol=1e-15):
        raise ValueError('Frozen mapping/domain qualities disagree')
    proxy=mapping.researcher_assumption.to_numpy(bool)
    proxy_mass=float(p[proxy].sum()); q0=float(p@mapping.Q_domain.to_numpy())
    q_alt=float(p@alt.Q_domain.to_numpy())
    direct_mass=float(p[~proxy].sum()); known=float(p[~proxy]@mapping.Q_domain.to_numpy()[~proxy])
    lower=known+proxy_mass*float(actual.min()); upper=known+proxy_mass*float(actual.max())
    mapping['p_ref']=p; mapping['quality_contribution']=p*mapping.Q_domain
    save_csv(out/'baseline_quality_mapping_details.csv',mapping)
    grouped=mapping.groupby('quality_domain').p_ref.sum().reindex(actual.index,fill_value=0)
    save_csv(out/'baseline_quality_domain_weights.csv',pd.DataFrame(dict(
        quality_domain=actual.index,p_ref_mapped_weight=grouped.to_numpy(),Q_domain=actual.to_numpy(),
        quality_contribution=grouped.to_numpy()*actual.to_numpy())))
    rows=[dict(candidate='Q0_A',Q0=float(trim_mean(values,.1)),definition='A1全部样本Q的双侧各10%截尾均值',
          source=samples.relative_to(root).as_posix(),mapping_needed=False,uncertainty='按域抽样结构；非真实训练语料token权重',recommended=False),
          dict(candidate='Q0_B',Q0=q0,definition='冻结p_ref点乘冻结17→7代理映射Q_domain',
          source=map_path.relative_to(root).as_posix(),mapping_needed=True,
          uncertainty=f'依赖代理映射的配比权重{proxy_mass:.10%}; 替代映射={q_alt:.17g}; 未识别域范围包络=[{lower:.17g},{upper:.17g}]，不是统计CI',recommended=True),
          dict(candidate='Q0_C',Q0=float(actual.median()),definition='七领域Q_domain中位数（等域敏感性）',
          source=base.relative_to(root).as_posix(),mapping_needed=False,uncertainty='忽略p_ref权重；仅敏感性',recommended=False),
          dict(candidate='Q0_B_alternative',Q0=q_alt,definition='冻结替代映射下p_ref加权质量',source=alt_path.relative_to(root).as_posix(),
          mapping_needed=True,uncertainty='代理映射结构敏感性，非独立观测',recommended=False),
          dict(candidate='Q0_B_direct_only',Q0=known/direct_mass,definition='仅直接/近直接映射，按覆盖质量质量归一化',
          source=map_path.relative_to(root).as_posix(),mapping_needed=True,
          uncertainty=f'覆盖配比权重{direct_mass:.10%}，不是完整p_ref基线',recommended=False)]
    save_csv(out/'baseline_quality_candidates.csv',rows)
    info=dict(primary='Q0_B',Q0=q0,proxy_mass=proxy_mass,alternative=q_alt,
              proxy_range_envelope=[lower,upper],envelope_is_confidence_interval=False,
              A1_sample_count=len(values),A1_sample_Q_range=[float(values.min()),float(values.max())],
              A1_domain_Q_range=[float(actual.min()),float(actual.max())],
              choice_status='recommended_conditional_on_frozen_proxy_mapping_not_direct_measurement')
    save_json(out/'baseline_quality_provenance.json',info)
    return info,pd.DataFrame(rows),[base,samples,map_path,alt_path]


def decision_domains(root,out,frozen,qinfo):
    """Compare sources before setting a conservative joint numeric support box."""
    file_path=root/'outputs/scaling/phase1/data_file_summary.csv'
    source=pd.read_csv(file_path,float_precision='round_trip')
    rows=[]
    for dataset,g in source.groupby('dataset',sort=False):
        rows.append(dict(source=dataset,N_min_B=float(g.N_min.min()),N_max_B=float(g.N_max.max()),
                         D_min_B=float(g.D_min.min()),D_max_B=float(g.D_max.max()),
                         main_evidence=dataset=='B1',notes='同源插值非新增证据' if dataset=='B3' else
                         '半合成；不扩主域' if dataset=='B2' else '跨族/文献绝对Loss口径未确认；只列范围' if dataset in ('B4','B5') else '真实主拟合轨迹'))
    b1=source[source.dataset=='B1'].iloc[0]; qs=frozen['quality_numeric_support']
    nlo=max(float(b1.N_min),qs['N_min_B']); nhi=min(float(b1.N_max),qs['N_max_B'])
    dlo=max(float(b1.D_min),qs['D_min_B']); dhi=min(float(b1.D_max),qs['D_max_B'])
    schema_path=root/'outputs/scaling/phase4/B9_B10_schema.json'
    schema=json.loads(schema_path.read_text(encoding='utf-8'))
    for dataset in ('B9','B10'):
        s=schema[dataset];rows.append(dict(source=dataset,N_min_B=s['N_min'],N_max_B=s['N_max'],D_min_B=s['D_min'],
            D_max_B=s['D_max'],main_evidence=False,notes='元数据；D=0不是有效下界，不填补；总/激活参数未统一' if dataset=='B9' else '估算Loss，非直接观测'))
    rows.append(dict(source='B6_frozen_numeric_support',N_min_B=qs['N_min_B'],N_max_B=qs['N_max_B'],
        D_min_B=qs['D_min_B'],D_max_B=qs['D_max_B'],main_evidence=False,notes='质量模块局部半合成数值支持，非真实联合观测'))
    save_csv(out/'evidence_support_comparison.csv',rows)
    bounds=[]
    def add(scheme,var,lo,hi,unit,extrap,src,notes):
        bounds.append(dict(scheme=scheme,variable=var,lower=lo,upper=hi,unit=unit,extrapolation=extrap,evidence_source=src,notes=notes))
    add('main','N',nlo,nhi,'billion parameters',False,'B1 ∩ B6 numeric support','连续矩形是插值可行域，不是所有点都观测；P1在N>1B使用Class D配比转移')
    add('main','D',dlo,dhi,'billion tokens',False,'B1 ∩ B6 numeric support','D下界提高至质量模块数值下界；非改动Q2参数')
    add('classical_only_reference','N',float(b1.N_min),float(b1.N_max),'billion parameters',False,'B1','只用于对照，不是第二套主域')
    add('classical_only_reference','D',float(b1.D_min),float(b1.D_max),'billion tokens',False,'B1','D<10时质量模块外推，不能称联合支持')
    add('extrapolation','N',nlo,max(schema['B9']['N_max'],schema['B10']['N_max']),'billion parameters',True,'B1 + B9/B10 envelope','跨缺口连续包络；没有密集大模型观测验证')
    add('extrapolation','D',dlo,max(schema['B9']['D_max'],schema['B10']['D_max']),'billion tokens',True,'B1 + positive B9/B10 upper extent','保留主下界；D=0未知不使用')
    add('main','Q',qinfo['Q0'],1.,'dimensionless',True,'Q1 baseline + official Q≤1','整个区间不是全外推；上端超出Q1实际质量；具体子区间另表')
    add('extrapolation','Q',qinfo['Q0'],1.,'dimensionless',True,'same frozen Q0','Q0和质量模块均不重新标定')
    save_csv(out/'decision_bounds.csv',bounds)
    knots=sorted(set([qinfo['Q0'],1.,*qinfo['A1_sample_Q_range'],*qinfo['A1_domain_Q_range'],qs['Q_min'],qs['Q_max']]))
    support=[]
    for a,b in zip(knots[:-1],knots[1:]):
        if b<=qinfo['Q0'] or a>=1:
            continue
        a=max(a,qinfo['Q0']);b=min(b,1.);m=(a+b)/2
        support.append(dict(Q_lower=a,Q_upper=b,within_A1_sample_range=qinfo['A1_sample_Q_range'][0]<=m<=qinfo['A1_sample_Q_range'][1],
            within_A1_domain_range=qinfo['A1_domain_Q_range'][0]<=m<=qinfo['A1_domain_Q_range'][1],
            within_B6_numeric_range=qs['Q_min']<=m<=qs['Q_max'],
            direct_joint_quality_validation=False,notes='数值区间覆盖不等于直接实验质量或联合N-D-Q-p验证'))
    save_csv(out/'quality_support_intervals.csv',support)
    return dict(N_min_B=nlo,N_max_B=nhi,D_min_B=dlo,D_max_B=dhi,Q_min=qinfo['Q0'],Q_max=1.),pd.DataFrame(bounds),[file_path,schema_path]


def mixture_sets(root,out,frozen,meta,cfg):
    """Use already saved model-input compositions, without repeating preprocessing."""
    from scaling.mixture_transfer import predict_ilr
    names=frozen['reference']['domain_order'];frames=[];paths=[]
    for label in ('train_1M','test_1M','test_60M','test_1B'):
        path=root/f'outputs/mixture/{label}_closed_replaced_composition.csv';paths.append(path)
        f=pd.read_csv(path,float_precision='round_trip')
        p=f[names].to_numpy()
        if (p<=0).any() or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-14,rtol=0):
            raise ValueError('Frozen recipe composition invalid: '+label)
        f['source_dataset']=label
        f['source_path']=path.relative_to(root).as_posix()
        f['signature']=[json.dumps(np.round(row,cfg['mixture_empirical_signature_decimal_places']).tolist(),separators=(',',':')) for row in p]
        f['M_p']=predict_ilr(np.log(p)@np.asarray(meta['ilr_basis']).T,meta)
        frames.append(f)
    combined=pd.concat(frames,ignore_index=True)
    save_csv(out/'empirical_recipe_membership.csv',combined)
    records=[]
    for i,(signature,g) in enumerate(combined.groupby('signature',sort=False)):
        if np.ptp(g.M_p.to_numpy())>1e-10:
            raise ValueError('Recipe deduplication tolerance merges distinct predictions')
        row={'candidate_id':f'P1_{i:04d}','source_members':json.dumps(g[['source_dataset','index']].to_dict('records')),
             'source_record_count':len(g),'composition_space':'frozen_closed_zero_replaced',
             'M_p':float(g.M_p.iloc[0]),'signature_decimal_places':cfg['mixture_empirical_signature_decimal_places']}
        row.update({name:float(g[name].iloc[0]) for name in names})
        records.append(row)
    save_csv(out/'empirical_recipe_candidates.csv',records)
    counts=combined.groupby('source_dataset').size().to_dict()
    (out/'mixture_feasible_sets.md').write_text('# 领域配比可行集\n\n'
        f'P0：固定冻结 p_ref=A4_ILR_center，M_ref={frozen["reference"]["M_ref"]:.17g}；17分量见 frozen_objective_summary.csv。\n\n'
        f'P1：读取已冻结训练/检验的模型输入配方 {counts}；共 {len(combined)} 条来源记录，按17维配比{cfg["mixture_empirical_signature_decimal_places"]}位签名合并为 {len(records)} 个候选。'
        '配方ID不跨表当作配比相同证据；membership表保留全部来源，无修改旧数据。'
        '仅读取保存的 closed_replaced_composition 和冻结模型；不读取 A4–A15 原始文件，不再拟合/重复闭合与零替换。\n\n'
        '重要语义：这些是已冻结的闭合、零替换后配比，来源于真实训练/检验配方，但不是原始结构零比例。'
        'P1在冻结模型输入空间选择，不能把替换质量解释为测得非零训练比例；物理配方部署前须回溯对应原始记录。'
        'M(p) 直接由保存的 ILR+一阶ElasticNet 系数评价。候选集中不包含新邻域采样，不包含估算10B/70B额外配方。'
        '检验配方曾参与冻结阶段验证，因此这些候选的后续优化属于探索性选择，不可作为新的独立盲测。\n\n'
        'P2：Δ17={p≥0,Σp=1}，仅数学敏感性；边界需执行冻结零替换约定，不能直接取ln0。'
        '单域100%和极端角点没有充分支持，不作为实际建议。\n\n'
        '主分析P0+P1；P1先在每个有限候选p上求三维N-D-Q连续问题，再做外层选择。'
        '本阶段只建立集合，不调用优化器。Q0统一沿用参考基线；p改变而Q0不变属于可比较情景假设，'
        '不表示所有经验配方都测得同一原始质量；Q与p能独立改变、加工同一D的假设须保留。\n',encoding='utf-8')
    return dict(source_counts={k:int(v) for k,v in counts.items()},source_records=len(combined),unique_candidates=len(records)),paths


def budget_scenarios(out,official,cfg,bounds,contexts):
    """List budgets and analytic minimum-cost feasibility, never solve allocations."""
    budgets=official['budgets_FLOPs'];e=cfg['budget_extension_decades_per_side']
    grid=np.geomspace(min(budgets)/10**e,max(budgets)*10**e,cfg['budget_log_grid_count'])
    rows=[dict(scenario_id=label,C_budget_FLOPs=value,scenario_type='official_recommended',
               evidence_source='Official Word Question 3 OMML',researcher_assumption=False)
          for label,value in zip(('low','medium','high'),budgets)]
    rows.extend(dict(scenario_id=f'log_grid_{i:02d}',C_budget_FLOPs=float(x),scenario_type='research_log_grid',
                evidence_source='official extrema ± configured log decades',researcher_assumption=True) for i,x in enumerate(grid))
    save_csv(out/'compute_budget_scenarios.csv',rows)
    checks=[]
    for row in rows:
        for l in contexts.L_ctx:
            lower=(official['training_coefficient']+official['eta']*l)*bounds['N_min_B']*bounds['D_min_B']*1e18
            checks.append(dict(scenario_id=row['scenario_id'],C_budget_FLOPs=row['C_budget_FLOPs'],L_ctx=int(l),
                minimum_total_cost_FLOPs=lower,main_box_feasible=row['C_budget_FLOPs']>=lower,
                basis='monotone costs: Q=Q0 has zero incremental quality cost; no objective optimization'))
    save_csv(out/'budget_feasibility_screen.csv',checks)
    return pd.DataFrame(rows)


def numeric_units(out,official,qinfo,contexts):
    """Twelve analytic arithmetic cases, independently evaluated using Decimal."""
    rows=[];q0=qinfo['Q0'];ctx=contexts.L_ctx.to_list()
    for i in range(12):
        family=list(official['quality_costs'])[i%3]
        n=[.1,1.,7.,12.][i//3];d=[10.,100.,300.,600.][i//3]
        l=int(ctx[i%len(ctx)]);q=[q0,(q0+1)/2,1.][i%3]
        tr,qu,att=cost_components(n,d,q,q0,l,official,family)
        pars=official['quality_costs'][family]
        with localcontext() as dec:
            dec.prec=50
            nd,dd=Decimal(str(n))*Decimal('1e9'),Decimal(str(d))*Decimal('1e9')
            qd,q0d=Decimal(str(q)),Decimal(str(q0));gamma=Decimal(str(pars['gamma']));lam=Decimal(str(pars['lambda']))
            manual_t=Decimal(str(official['training_coefficient']))*nd*dd
            manual_a=Decimal(str(official['eta']))*nd*dd*Decimal(l)
            if family=='exponential':
                h=(lam*qd).exp()-(lam*q0d).exp()
            elif family=='power':
                h=qd**int(lam)-q0d**int(lam)
            else:
                h=(1+lam*qd).ln()-(1+lam*q0d).ln()
            manual_q=dd*gamma*max(Decimal(0),h)
        passed=all(np.isclose(float(a),float(b),rtol=2e-14,atol=1e-5) for a,b in zip((tr,qu,att),(manual_t,manual_q,manual_a)))
        if not passed:
            raise ValueError(f'Unit arithmetic mismatch: case {i}')
        rows.append(dict(case_id=i+1,family=family,N_B=n,D_B=d,L_ctx=l,Q=q,Q0=q0,
            actual_N=format(nd,'f'),actual_D=format(dd,'f'),program_train_FLOPs=tr,manual_decimal_train=str(manual_t),
            program_quality_FLOPs=float(qu),manual_decimal_quality=str(manual_q),program_attention_FLOPs=att,
            manual_decimal_attention=str(manual_a),passed=passed))
    save_csv(out/'unit_numeric_checks.csv',rows)
    md='# 算力单位核验\n\nN_B=N/10^9，D_B=D/10^9。目标函数读取问题二参数时仍用 N_B、D_B；成本函数输入B单位后显式换算。\n\n'
    md+='- C_train=6 N_B D_B ×10^18 FLOPs。\n- C_attn=eta N_B D_B L_ctx ×10^18 FLOPs。\n- CQ=D_B ×10^9 [g(Q)-g(Q0)]+ FLOPs。\n\n'
    md+='N 是参数个数，D 是训练token数，L_ctx 是窗口位置数；6为训练运算近似系数，eta为注意力代理系数。'
    md+='Q、Q0、p及lambda_Q无量纲；g和gamma必须按FLOPs/token解释，才能与题目FLOPs预算相加。'
    md+='附录未另列gamma单位；此单位是由题给CQ=DΔg和预算单位推得（B类），不是新增数值校准。'
    md+='不把gamma当货币，不乘N、不乘第二个D、不乘自设总成本上限M。'
    md+='成本CQ与目标函数系数C_Q虽然名字相近，单位与角色完全不同。\n\n'
    md+='## 独立算式与数值核验\n\n12例为公式算术校验，不是实验观测或最优解；包含超出主域的单位压力样例。'
    md+='“手工式”路径使用实际N=10^9 N_B与D=10^9 D_B，以及50位Decimal直接指数/幂/对数减法；'
    md+='程序路径使用B单位简式和数值稳定expm1/log1p。双方独立计算后比较，容差2e-14。\n\n'
    for row in rows:
        md+=f'例{row["case_id"]}：N={row["actual_N"]}，D={row["actual_D"]}，L={row["L_ctx"]}，Q={row["Q"]:.10g}，{row["family"]}。'
        md+=f'训练6×{row["actual_N"]}×{row["actual_D"]}={row["program_train_FLOPs"]:.12g}；'
        md+=f'注意力{official["eta"]}×{row["actual_N"]}×{row["actual_D"]}×{row["L_ctx"]}={row["program_attention_FLOPs"]:.12g}；'
        md+=f'质量D×[g(Q)-g(Q0)]+={row["program_quality_FLOPs"]:.12g} FLOPs；独立核验通过。\n\n'
    (out/'unit_audit.md').write_text(md,encoding='utf-8')
    return rows


def quality_cost_report(out,official,qinfo):
    """Document the unmodified official families and their analytic derivatives."""
    q0=qinfo['Q0'];rows=[]
    shapes={'exponential':('γ exp(λ_Q Q)','γ λ_Q exp(λ_Q Q)','凸'),
            'power':('γ Q^λ_Q','γ λ_Q Q^(λ_Q-1)','凸'),
            'logarithmic':('γ ln(1+λ_Q Q)','γ λ_Q/(1+λ_Q Q)','凹，边际成本递减')}
    for family,pars in official['quality_costs'].items():
        g,derivative,shape=shapes[family]
        rows.append(dict(family=family,g_formula=g,gamma=pars['gamma'],lambda_Q=pars['lambda'],
              gamma_unit='FLOPs/token',marginal_formula='D × '+derivative,
              cost_at_Q0_per_token=float(quality_increment_per_token(q0,q0,family,pars['gamma'],pars['lambda'])),
              cost_at_Q1_per_token=float(quality_increment_per_token(1.,q0,family,pars['gamma'],pars['lambda'])),
              endpoint_finite=True,shape=shape))
    save_csv(out/'quality_cost_function_parameters.csv',rows)
    (out/'quality_cost_functions.md').write_text('# 官方质量成本函数\n\n'
        '统一 CQ(D,Q;Q0,θ)=D[gθ(Q)-gθ(Q0)]+，D为实际token数，θ=(gamma,lambda_Q)。'
        f'推荐Q0={q0:.17g}；三个官方参数全部从Word OMML读取。\n\n'+table(pd.DataFrame(rows))+
        '\n\n在Q>Q0时 dCQ/dQ=D g′(Q)>0；Q<Q0时为零；Q=Q0处有左导数0与右导数Dg′(Q0)，'
        '主可行域采用右导数。对D的导数为[g(Q)-g(Q0)]+，非负；三类均显式乘同一训练D。'
        'Q=Q0增量为零；Q→1时三类均有限，不排除Q=1。'
        '对数渐进型是ln(1+10Q)，不是-ln(1-Q)，也不是越趋完美越发散的模型。\n\n'
        '指数/幂的质量边际成本递增，对数型边际成本递减，因此后续可能产生不同活跃集，不能先套同一凸性或内点结论。'
        '本阶段三类均保留，未依据旧最优结果选主成本。曲线以CQ/D和(dCQ/dQ)/D显示，避免主观选D。'
        '没有额外最大值M；Q=1有限上界是由给定函数和Q0解析导出的，不是新的调参上限。\n',encoding='utf-8')
    return pd.DataFrame(rows)


def math_reports(out,official,bounds):
    """Formulate P0/P1 and future diagnostics only; retain budget inequality."""
    (out/'optimization_formulation.md').write_text(r'''# 优化数学形式

所有目标参数、Q/K/M(p)/lambda_p均冻结。N_B、D_B为十亿单位，物理N=10^9N_B、D=10^9D_B。

L_gen=E+A N_B^(-alpha)+B D_B^(-beta)+C_Q (N_B/N0_B)^(-u_Q)(-lnQ)+lambda_p(N_B)[M(p)-M_ref]。

P0：min_{N_B,D_B,Q} L_gen(N_B,D_B,Q,p_ref)。约束

(6+eta L_ctx)N_B D_B 10^18 + D_B 10^9 [g(Q)-g(Q0)]+ ≤ C_budget，
N_min≤N_B≤N_max，D_min≤D_B≤D_max，Q0≤Q≤1。

L_ctx为外生C7离散场景，成本函数类型与预算也为外生场景。P1对每个已保存经验候选p建立同样三维问题，再外层比较候选；本阶段不求解。P2只作为后续敏感性。

## 单调性与预算松弛

P0的DeltaM=0：
dL/dD_B=-beta B D_B^(-beta-1)<0；
dL/dQ=-C_Q (N_B/N0_B)^(-u_Q)/Q<0；
dL/dN_B=-alpha A N_B^(-alpha-1)-u_Q DeltaQ/N_B<0。

成本对N_B的导数=(6+eta L_ctx)D_B 10^18>0；对D_B为(6+eta L_ctx)N_B 10^18+10^9Δg>0；
对Q在Q≥Q0取右导数D_B10^9 g′(Q)>0。
因此可行域内若有任一变量尚未触达上界且预算严格松弛，就可以小幅提高该变量降低目标。
P0最优预算必须绑定，或全部N/D/Q达到各自上界。全部上界仍成本小于预算时允许剩余预算，不能硬编码等号。

P1中记DeltaM=M(p)-M_ref，s=d lambda_p/d lnN_B：
dL/dN_B=[-alpha A N_B^(-alpha)-u_Q DeltaQ+s DeltaM]/N_B。
该导数未保证为负，必须保留冻结分段lambda并检查左右导数；在节点主约定为右导数。
不过D/Q在固定p下仍严格递减：若任意一个未达到上界，预算松弛仍可被排除。
若D和Q已触上界，N局部不单调、位于节点或达到上界，可存在预算松弛。
质量模块外推不自动改变公式的单调符号，但降低上述数学结论的经验解释可信度。

## 后续数值求解要求

对每种成本/上下文/预算/域方案先检查可行性，不以调整边界或Q0掩盖不可行；记录不等式松弛。
分段lambda、对数型凹质量成本、有限候选p可能导致非凸/多分支，后续需比较边界和分段解，而非假定一个KKT内点就是全局解。
在质量模块局部证据外，最优解只解释为冻结模型情景结果。P1在N>1B时保留Class D标记。
''',encoding='utf-8')
    (out/'structural_shift_definition.md').write_text(r'''# 结构性转移定义 仅建立判据

没有计算最优轨迹或转移预算。以下均为PHASE 2待应用定义。

1. 活跃集 A(C)：包含达到N/D/Q上下界的变量、绑定预算、以及P1选中配方的身份。
归一化距离与预算松弛均≤1e-6时暂标活跃（研究者数值容差）；后续用1e-5/1e-7检验容差稳定性。
连续网格上身份/约束变更为候选，须局部加密与独立求解确认，不将网格单次翻转直接称转移。

2. 成本份额 s_train=C_train/C_budget，s_Q=CQ/C_budget，s_attn=C_attn/C_budget；预算松弛时三者和≤1而非必为1。
同时记录未使用份额1-C_total/C_budget，并可另列相对于实际消耗的份额，不能混用分母。
同一L_ctx与eta下C_attn/C_train=eta L_ctx/6恒定，因此这两类份额彼此固定比例，不应虚构独立的注意力/训练相互转移。
“注意力主导”是否成立由外生上下文决定，不随预算单独变化；质量份额、上下界和配方选择仍可能改变。

3. 最优决策弹性：e_N=dlnN*/dlnC，e_D=dlnD*/dlnC，e_Q=dQ*/dlnC。
注意e_Q不是dlnQ/dlnC。对连续预算用中心差分或局部分段回归；lambda节点另列左右斜率。
斜率/份额趋势的分段改变为候选，要求至少两类证据（如活跃集+弹性）、网格加密稳定和目标值/可行性核验。

后续识别方法：对数预算网格逐场景求解、候选附近加密；比较连续与一断点分段趋势的残差/BIC，
并对预算网格、数值容差、成本函数、Q0代理映射、主/外推域做稳定性对照。
这些是研究者拟定的方法；当前没有拟合断点，没有指定转移预算或最优N*/D*/Q*。
''',encoding='utf-8')


def figures(root,out,contexts,c7,official,qinfo,budgets,bounds):
    """Six scientific figures from observed context and audited formulas only."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    figdir=root/'outputs/figures/question3_phase1';figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    manifest=[]
    def finish(fig,name,caption):
        path=figdir/name;fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)
        manifest.append(dict(figure=path.relative_to(root).as_posix(),caption=caption))
    fig,ax=plt.subplots(figsize=(7,4))
    sortedctx=np.sort(c7.max_position_embeddings.to_numpy())
    ax.step(sortedctx,np.arange(1,len(sortedctx)+1)/len(sortedctx),where='post')
    ax.set(xscale='log',xlabel='Maximum supported context (positions)',ylabel='Empirical cumulative fraction',title=f'C7 context distribution (n={len(c7)})')
    ax.axvline(6/official['eta'],ls='--',color='gray',label='Attention = training');ax.legend()
    finish(fig,'01_C7_context_distribution.png','C7实测上下文经验分布')
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(contexts.L_ctx.astype(str),contexts.support_count,color='#4477AA')
    for i,n in enumerate(contexts.support_count):ax.text(i,n+.2,str(n),ha='center')
    ax.set(xlabel='Observed maximum context (positions)',ylabel='Model count',title='C7 observed context frequencies')
    ax.set_ylim(0,contexts.support_count.max()*1.18)
    finish(fig,'02_C7_context_frequency.png','C7实际离散窗口频数')
    q=np.linspace(qinfo['Q0'],1,401)
    for derivative,name,title,ylabel in [(False,'03_quality_cost_functions.png','Official incremental quality costs','C_quality / D (FLOPs per token)'),
             (True,'04_marginal_quality_costs.png','Official marginal quality costs','(dC_quality/dQ) / D (FLOPs per token per Q)')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for family,pars in official['quality_costs'].items():
            func=marginal_quality_per_token if derivative else quality_increment_per_token
            ax.plot(q,func(q,qinfo['Q0'],family,pars['gamma'],pars['lambda']),label=family)
        ax.set(xlabel='Target quality Q',ylabel=ylabel,title=title);ax.legend()
        finish(fig,name,title+'，不是最优结果')
    fig,ax=plt.subplots(figsize=(7,3));g=budgets[budgets.scenario_type=='research_log_grid'];o=budgets[budgets.scenario_type=='official_recommended']
    ax.scatter(g.C_budget_FLOPs,np.zeros(len(g)),s=15,label='41-point research log grid')
    ax.scatter(o.C_budget_FLOPs,np.ones(len(o)),marker='D',label='Official recommended budgets')
    for row in o.itertuples():ax.annotate(f'{row.scenario_id}: {row.C_budget_FLOPs:.0e}',(row.C_budget_FLOPs,1),xytext=(0,12),textcoords='offset points',ha='center')
    ax.set(xscale='log',xlabel='Compute budget (FLOPs)',yticks=[],ylim=(-.5,1.6),title='Budget scenarios only; no allocation solved')
    ax.legend(loc='center left',bbox_to_anchor=(0.01,.48),fontsize=8)
    finish(fig,'05_budget_scenarios.png','官方三档与研究预算网格')
    fig,ax=plt.subplots(figsize=(8,5))
    for ds,file,color,marker in [('B1','B1_residuals.csv','#4477AA','.'),('B4','B4_direct_predictions.csv','#CC6677','x'),('B5','B5_direct_predictions.csv','#228833','+')]:
        f=pd.read_csv(root/'outputs/scaling/phase1'/file,float_precision='round_trip')
        ax.scatter(f.N_params_B,f.D_tokens_B,s=8 if ds=='B1' else 24,c=color,marker=marker,alpha=.5,label=ds+' frozen data')
    join=pd.read_csv(root/'outputs/scaling/phase4/B9_B10_join.csv',float_precision='round_trip')
    valid=join[(join.N_params_B_B9>0)&(join.D_tokens_B_B9>0)]
    ax.scatter(valid.N_params_B_B9,valid.D_tokens_B_B9,s=10,c='#AA3377',alpha=.45,label='B9 metadata (D>0)')
    a=bounds;ax.add_patch(Rectangle((a['N_min_B'],a['D_min_B']),a['N_max_B']-a['N_min_B'],a['D_max_B']-a['D_min_B'],fill=False,edgecolor='black',lw=2,label='Main joint numeric box'))
    ax.set(xscale='log',yscale='log',xlabel='N (billion parameters)',ylabel='D (billion tokens)',title='Observed supports and proposed main domain')
    ax.legend(fontsize=8);finish(fig,'06_N_D_evidence_support.png','数据范围对照；B4/B5可比性有限，B9非Loss验证')
    save_csv(out/'figure_manifest.csv',manifest)
    # Contact sheet for visual QA, not a substitute for individual figures.
    from PIL import Image,ImageOps,ImageDraw
    sheet=Image.new('RGB',(1400,1110),'white')
    for i,row in enumerate(manifest):
        im=Image.open(root/row['figure']).convert('RGB');thumb=ImageOps.contain(im,(690,350))
        x=(i%2)*700;y=(i//2)*370;sheet.paste(thumb,(x,y));ImageDraw.Draw(sheet).text((x+5,y+350),Path(row['figure']).name,fill='black')
    sheet.save(out/'figure_contact_sheet.png')
    return manifest,[root/'outputs/scaling/phase1'/s for s in ('B1_residuals.csv','B4_direct_predictions.csv','B5_direct_predictions.csv')]+[root/'outputs/scaling/phase4/B9_B10_join.csv']


def final_report(root,out,official,ctxstats,contexts,qinfo,qtable,bounds,boundtable,frozen,mixture,costtable,figures):
    """Answer all fourteen setup questions with evidence and explicit assumptions."""
    report=root/'outputs/report/question3_phase1_setup.md';report.parent.mkdir(parents=True,exist_ok=True)
    text='# 问题三 PHASE 1 算力约束优化输入标定\n\n'
    text+='本阶段完成官方公式核验、实际C7诊断、成本与支持域整理、数学形式及未来转移判据。未拟合任何模型，未运行资源优化或结构转移检测。'
    text+='问题一/二冻结文件完整性与独立算术核验见本阶段验证表。\n\n'
    text+='## 1 官方三类成本及预算\n\nC_train=6ND；CQ=D[g(Q)-g(Q0)]+；C_attn=etaNDL_ctx。'
    text+=f'eta={official["eta"]}，来源官方问题三原始OMML；训练/质量/注意力使用同一D。'
    text+=f'预算建议{official["budgets_FLOPs"]} FLOPs，非强制唯一取值。附加41点研究对数网格从最低/10到最高×10，仅列场景。\n\n'
    text+='## 2 C7上下文及注意力临界值\n\n'+table(pd.DataFrame([ctxstats]))+'\n\n'+table(contexts)+'\n\n'
    fields=pd.read_csv(out/'C7_field_diagnosis.csv')
    exceeding=int(contexts.loc[contexts.L_ctx>6/official['eta'],'support_count'].sum())
    text+=f'C7实际{ctxstats["rows"]}行{len(fields)}列，完整字段与缺失/重复见C7_diagnosis.md；没有独立N或明确tokens字段，training_data_TB未换算为D。'
    text+=f'eta使用题给值，无另估区间；L_crit=6/eta={6/official["eta"]:g}，在观测范围内但非观测离散值；超临界{exceeding}/{ctxstats["valid_count"]}。'
    text+='上下文取五个全部实际窗口作为外生场景。架构最大窗口、训练真实序列与自由N/D组合存在差异；题给成本为简化代理。\n\n'
    text+='## 3 冻结目标与证据等级\n\n'
    text+='L_gen=E+A N_B^(-alpha)+B D_B^(-beta)+C_Q(N_B/1B)^(-u_Q)(-lnQ)+lambda_p(N_B)[M(p)-M_ref]。'
    text+='这里N_B/1B表示相对N0_B=1的数值比例，不能再把以B计的N乘10^9代入目标系数。\n\n'
    text+='Class A经典骨架、Class B局部质量、Class C冻结ILR+一阶ElasticNet配方模型、Class D大尺度配方转移。所有完整参数直接读取最终JSON，见frozen_objective_summary.csv。'
    text+='没有联合外部N-D-Q-p验证；B6是局部半合成质量校准，B7差分扩展失败，不改冻结参数也不夸大证据。\n\n'
    text+='## 4 基线质量\n\n'+table(qtable,['candidate','Q0','definition','mapping_needed','recommended'])+'\n\n'
    text+=f'建议Q0_B={qinfo["Q0"]:.17g}以保持与p_ref一致，但仅在冻结代理映射假设下。'
    text+=f'依赖代理映射的配比权重为{qinfo["proxy_mass"]:.6%}，替代映射Q0={qinfo["alternative"]:.17g}；未识别域包络{qinfo["proxy_range_envelope"]}不是置信区间。'
    text+='七质量域聚合权重见baseline_quality_domain_weights.csv；c4无直接对应的17域配比，当前主映射聚合权重为0，不补造配比。'
    text+='主/替代映射的结构敏感性不可被七域bootstrap CI替代；不继承Q0=.8，不重算样本Q。\n\n'
    text+='## 5 三种官方质量成本与边际成本\n\n'+table(costtable,['family','g_formula','gamma','lambda_Q','marginal_formula','endpoint_finite','shape'])+'\n\n'
    text+='统一乘实际D与正部差分，Q=Q0增量0，Q>Q0边际正，Q=1均有限。对数型边际递减，不能改成发散模型。三种全部保留待后续比较，未提前择优。\n\n'
    text+='## 6 N/D/Q支持域\n\n'+table(boundtable,['scheme','variable','lower','upper','unit','extrapolation'])+'\n\n'
    text+='主域不是未经论证直接指定B1：比较B1–B5范围与已冻结验证局限后，经典骨架采用B1高可信包络，再与B6质量数值支持求交集，'
    text+='因此D下界10B而非0.134B。B3同源插值不能扩证据，B4/B5跨评测Loss口径未确认、仅列扩展覆盖，不扩主域。'
    text+='该连续矩形仍是插值建模选择，真实B1只含8个固定N轨迹。质量模块没有直接实验支持，表中“非外推”仅指N/D各模块数值范围。'
    text+='外推域将N上界扩至10000B、D上界36000B，保留主下界；B9 D=0是未知而不是有效可行下界。总/激活参数、稠密/稀疏等尚未统一，不能把B10估算Loss当验证。\n\n'
    text+=f'Q∈[Q0,1]；A1样本质量范围{qinfo["A1_sample_Q_range"]}，七领域质量范围{qinfo["A1_domain_Q_range"]}；'
    text+=f'B6数值Q范围[{frozen["quality_numeric_support"]["Q_min"]},{frozen["quality_numeric_support"]["Q_max"]}]。'
    text+='质量范围覆盖不等于直接验证；Q超过A1样本最大值或领域最大值分别标记，Q≈1不能宣称测得验证。\n\n'
    text+='## 7 p可行集\n\nP0固定p_ref；P1读取真实训练/检验配方已冻结闭合零替换模型输入，'
    text+=f'{mixture["source_records"]}条来源记录对应{mixture["unique_candidates"]}个候选，完整membership保存。'
    text+='不新建邻域，不做17维联立寻优；P1有限候选外层选择有据可查，但模型输入的微小零替换质量不是原始训练比例。'
    text+='P2理论simplex只作敏感性，极端角点不能作建议。固定Q0与可独立改善Q的假设对P1也需保留。N>1B使用Class D，不因N在B1范围而升级配比证据。\n\n'
    text+='## 8 单位\n\nC_train=6N_BD_B10^18；C_attn=etaN_BD_BL_ctx10^18；CQ=D_B10^9Δg，均为FLOPs。'
    text+='gamma按官方总成本维度推定FLOPs/token，lambda_Q无量纲；没有货币系数或额外M。12个独立Decimal手工算式与程序核验通过，见unit_audit.md及unit_numeric_checks.csv。\n\n'
    text+='## 9 数学形式与预算绑定\n\nP0 min_{N_B,D_B,Q}L_gen(p_ref)，受三个成本之和≤预算及上述box约束；'
    text+='P1逐配方同样求解再外层选最优。约束保持不等式。P0各资源边际Loss都负，内域预算绑定或全部资源到上界；'
    text+='P1的N导数可能因分段lambda与配方项改变符号，须左右检查，D/Q仍严格递减。边界饱和可出现预算松弛。'
    text+='尤其低预算配长上下文时主域可能不可行：预算可行性只按主域成本解析下界筛查，未求Loss最优。详见budget_feasibility_screen.csv。\n\n'
    text+='## 10 结构性转移定义\n\n采用活跃集、以预算为分母的三项成本份额、最优N/D对预算log弹性与Q对ln预算导数。'
    text+='候选变化须后续局部加密、左右导数、数值与情景稳定性验证，不在本阶段识别断点。'
    text+='同一上下文下C_attn/C_train=etaL_ctx/6恒定，不能制造由预算驱动的注意力/训练比例交叉。\n\n'
    text+='## 11 真实输入、解析推导与假设\n\n真实输入：官方Word/PDF、C7、问题一冻结质量/配方结果、B1以及B9公开元数据。'
    text+='B6局部校准为半合成，B10及10B/70B配方转移为估算；ILR配方函数是冻结回归模型。'
    text+='推导：临界窗口、边际质量成本、FLOPs换算、预算最低可行成本及单调性。'
    text+='假设/敏感性：17→7代理映射、Q0选取、矩形支持域及其B6交集、跨架构上下文自由组合、p/Q独立改变、零替换配方空间、41点预算网格与未来识别容差。'
    text+='旧优化稿未作为数据输入。\n\n'
    text+='## 12 进入PHASE 2的条件\n\n已具备按已审计公式进行条件化数值求解的输入与数学定义，等待人工确认Q0代理映射及主/外推域。'
    text+='无公式/单位缺失；条件化模型不等于完整联合真实验证。后续不得强行求不可行预算场景，不得把估算大模型配比反转或Q=1端点当直接证据。'
    text+='当前停止于PHASE 1，未求低中高最优解、未检测结构转移、未开始问题四。\n\n'
    text+='## 图表\n\n'+'\n'.join('- ['+f['caption']+'](../.. /'+f['figure']+')' for f in figures).replace('../.. /','../../')+'\n'
    report.write_text(text,encoding='utf-8')


def run(root):
    """Run independent setup steps in order and record exact input provenance."""
    root=Path(root);out=root/'outputs/optimization/phase1';out.mkdir(parents=True,exist_ok=True)
    cfg_path=root/'configs/optimization_phase1.yaml';cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    if cfg['perform_optimization'] or cfg['perform_structural_shift_detection']:
        raise ValueError('Phase 1 cannot run optimization or shift detection')
    np.random.seed(cfg['random_state'])
    logging.info('Audit official Word OMML and relevant data-description pages')
    official,paths=audit_official(root,out)
    logging.info('Diagnose actual C7 schema and observed contexts')
    c7,contexts,ctxstats,p=diagnose_context(root,out,official);paths+=p
    logging.info('Read frozen objective full precision; no refit')
    frozen,meta,p=frozen_inputs(root,out);paths+=p
    qinfo,qtable,p=baseline_quality(root,out,frozen);paths+=p
    bounds,boundtable,p=decision_domains(root,out,frozen,qinfo);paths+=p
    mixture,p=mixture_sets(root,out,frozen,meta,cfg);paths+=p
    budgets=budget_scenarios(out,official,cfg,bounds,contexts)
    unitchecks=numeric_units(out,official,qinfo,contexts)
    costtable=quality_cost_report(out,official,qinfo)
    math_reports(out,official,bounds)
    logging.info('Generate six figures from diagnostics and official formulas')
    figs,p=figures(root,out,contexts,c7,official,qinfo,budgets,bounds);paths+=p
    final_report(root,out,official,ctxstats,contexts,qinfo,qtable,bounds,boundtable,frozen,mixture,costtable,figs)
    # This is a new setup artifact, not a new fitted generalized scaling model.
    save_json(out/'optimization_setup.json',dict(official=official,baseline=qinfo,main_bounds=bounds,
            mixture=mixture,frozen_objective_source='outputs/scaling/phase4/generalized_scaling_final.json',
            official_contexts=contexts.L_ctx.tolist(),config=cfg,optimization_performed=False,
            shift_detection_performed=False,unit_checks_passed=len(unitchecks)))
    manifest=[]
    for path in sorted(set(paths+[cfg_path])):
        manifest.append(dict(path=path.relative_to(root).as_posix(),size_bytes=path.stat().st_size,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save_csv(out/'input_manifest.csv',manifest)
    code_paths=[root/'scripts/17_setup_compute_optimization.py',root/'scripts/17_verify_compute_setup.py',
                root/'configs/optimization_phase1.yaml',*sorted((root/'src/optimization').glob('*.py'))]
    save_csv(out/'implementation_manifest.csv',[dict(path=p.relative_to(root).as_posix(),
             sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in code_paths if p.exists()])
    logging.info('PHASE 1 setup complete: Q0=%.12f, empirical candidates=%s; no optimization',qinfo['Q0'],mixture['unique_candidates'])
