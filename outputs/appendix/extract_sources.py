"""Read-only extraction tooling, NOT a modeling/experiment entry point.

Writes only under outputs/appendix; never imports project modeling modules.
All appendix code bodies are original source slices, not reimplementations.
"""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import re
import shutil
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/appendix'


QUESTIONS={
    1:dict(title='问题一',entries=['03_build_quality_features.py','04a_diagnose_critic_weights.py','04_build_quality_score.py','05_analyze_quality_conflict.py','08_model_mixture_loss.py','09_analyze_mixture_effects.py'],
        auxiliary=['02_profile_quality_data.py','02c_diagnose_records.py','07_inspect_mixture_loss_data.py','03_verify_quality_features.py','04b_verify_quality_score.py','04b_record_text_review.py','05_verify_quality_conflict.py','08_verify_mixture_models.py','09_verify_mixture_analysis.py'],
        outputs=['outputs/preprocess/preprocessing_model_A1.json','outputs/quality/critic_weights_A1.csv','outputs/quality/A1_sample_quality.csv','outputs/quality/domain_quality_summary_A1.csv','outputs/conflict/A1_sample_conflict.csv','outputs/mixture/final_model_metadata.json','outputs/mixture/stage2/final_model_policy.json','outputs/mixture/domain_marginal_effects.csv'],
        reports=['outputs/report/question1_part1_final.md','outputs/report/question1_part2_findings.md','outputs/report/question1_part2_phase5b_robust_validation.md','outputs/report/question1_part3_final.md'],
        model='A1冻结预处理→Pearson-CRITIC Q；K为加权绝对偏离；17域Helmert ILR＋一阶ElasticNet；Ridge为数值对照。'),
    2:dict(title='问题二',entries=['11_fit_classical_scaling.py','14_fit_scale_dependent_quality.py','15_calibrate_mixture_scaling.py','16_analyze_generalized_scaling.py'],
        auxiliary=['11_verify_classical_scaling.py','14_verify_scale_dependent_quality.py','15_verify_mixture_scaling.py','16_verify_generalized_scaling.py'],
        outputs=['outputs/scaling/phase1/classical_scaling_parameters.csv','outputs/scaling/phase1/B1_group_cv.csv','outputs/scaling/phase2C/model_parameters.csv','outputs/scaling/phase3/lambda_empirical_knots.csv','outputs/scaling/phase4/generalized_scaling_final.json','outputs/scaling/phase4/marginal_derivative_checks.csv'],
        reports=['outputs/report/question2_phase1_classical_scaling.md','outputs/report/question2_phase2C_scale_dependent_quality.md','outputs/report/question2_phase3_modular_mixture_scaling.md','outputs/report/question2_final.md'],
        model='经典L0＋B6局部Delta_Q＋lnN分段线性lambda_p乘冻结M(p)差值；主边界S-A。'),
    3:dict(title='问题三',entries=['17_setup_compute_optimization.py','18_optimize_compute_resources.py','19_finalize_compute_optimization.py'],
        auxiliary=['17_verify_compute_setup.py','18_verify_compute_optimization.py','19_verify_final_optimization.py'],
        outputs=['outputs/optimization/phase1/optimization_setup.json','outputs/optimization/phase1/empirical_recipe_candidates.csv','outputs/optimization/phase2/official_budget_results.csv','outputs/optimization/phase2/P0_main_budget_path.csv','outputs/optimization/phase3/P0_P1_P2_official_comparison.csv','outputs/optimization/phase3/P2_global_search.csv','outputs/optimization/phase3/P2_B_boundary_limit_witness.csv'],
        reports=['outputs/report/question3_phase1_setup.md','outputs/report/question3_phase2_optimization.md','outputs/report/question3_final.md'],
        model='解析消去D_B；P0主方案、P1经验增强；P2-A为理论外推，P2-B无有限最优解。'),
    4:dict(title='问题四',entries=['20_build_evolution_ability_bridge.py','21_build_evolution_contribution.py','22_build_evolution_projection.py'],
        auxiliary=['20_verify_evolution_phase1.py','21_verify_evolution_contribution.py','22_verify_evolution_projection.py'],
        outputs=['outputs/evolution/phase1/evolution_model_table.csv','outputs/evolution/phase1/ability_transform_parameters.json','outputs/evolution/phase1/bridge_model.json','outputs/evolution/phase2/model_definitions.json','outputs/evolution/phase2/contribution_decomposition.csv','outputs/evolution/phase3/future_frontier_scenarios.csv','outputs/evolution/phase3/verification_summary.json'],
        reports=['outputs/report/question4_phase1_data_ability_bridge.md','outputs/report/question4_phase2_contribution_decomposition.md','outputs/report/question4_final.md'],
        model='固定六任务A_z；M1/M2-C/M3＋双因素Shapley；桥接失败负结果；已执行Phase3情景投影及风险诊断。')}

PURPOSES={
 'io.py':'真实附件读取、索引/字段兼容与输入单位', 'vector_transform.py':'8个列表字段真实标量化与qurater分量参数',
 'vector_diagnostics.py':'共用JSON/CSV和数值辅助；softmax不是主分类评分', 'profiling.py':'历史描述统计，非A1主预处理参数',
 'record_diagnostics.py':'预处理前重复/ID冲突及缺失诊断', 'preprocess.py':'只拟合A1的填补/截尾参数，原样映射A2/A3',
 'normalize.py':'正负向Min-Max及适中型梯形效用', 'critic.py':'Pearson-CRITIC ddof=1及Spearman敏感性',
 'aggregate.py':'10%截尾均值、模型记录bootstrap、分布比较', 'conflict.py':'K=加权绝对偏离、冲突阈值和驱动组合',
 'composition.py':'训练折内乘法零替换与Helmert ILR', 'modeling.py':'Ridge/ElasticNet候选、训练内调参与分组CV',
 'effects.py':'真实领域配比1个百分点扰动及联合边际效应', 'augmented.py':'Qp/Kp候选对照；最终没有替代一阶ElasticNet',
 'classical.py':'经典五参数非线性拟合、解析Jacobian、误差指标', 'bootstrap.py':'整条trajectory重采样拟合',
 'quality_extension.py':'最终链复用读入/reference工具；常数gamma MQ候选不是最终主模型',
 'quality_analysis.py':'Phase2C复用B8压力检验工具；MQ候选不是最终模型', 'quality_reporting.py':'Phase2C复用绘图和表格工具',
 'scale_quality.py':'S3原参数来源；S1/S2只为比较候选，最终去v且不重拟合',
 'scale_quality_analysis.py':'ND分组CV、Q层留出、整块bootstrap及B6/B7差分检验',
 'mixture_transfer.py':'冻结ILR预测、逐尺度lambda与配方bootstrap；P1/P2曲线仅比较候选',
 'marginal_analysis.py':'最终广义标度律、边界策略、解析导数/弹性及Q+0.1精确求根',
 'costs.py':'6ND、注意力成本与exp/power/log质量增量成本', 'setup.py':'官方成本、Q0、决策边界与832经验候选的来源',
 'resource_solver.py':'D_B解析消元、可行性、网格/SLSQP/DE及P0/P1求解', 'transitions.py':'连续预算扫描路径上的Type I/II转移检验与区间',
 'simplex_sensitivity.py':'P2-A配方证书与数值交叉检验；P2-B边界无界性，仅理论敏感性',
 'final_reporting.py':'冻结P0/P1复用、证据分层与问题三最终表/报告',
 'metadata.py':'C1/C2去重、S0/S1类型证据及C4元数据匹配', 'ability.py':'固定六Benchmark Winsor/median/IQR、A_z、PCA/Average敏感性',
 'detailed.py':'C8 JSON全遍历、损坏文件记录与偏科/弱尾', 'bridge.py':'B0/B1/B2、LOMO/bootstrap和桥接失败诊断',
 'contribution_regression.py':'OLS/HC3/Huber、LOOCV、Cook/VIF、双因素Shapley与5000模型bootstrap',
 'contribution_analysis.py':'T0-T3、M1/M2-C/M3、组织/时间轴对照、技术残差和分解执行',
 'projection_data.py':'实际执行Phase3：锚点、C4去重滚动前沿、增长bootstrap',
 'projection_models.py':'冻结成对系数的情景Monte Carlo、回测、补偿、风险标记',
 'common.py':'模型名规范、CSV和相关系数辅助'}

NOTES={
 'src/quality/vector_transform.py':'softmax_expected_rating is auxiliary, not the final argmax rating model.',
 'src/mixture/modeling.py':'M0 and second-order M2 are candidate / sensitivity models; not used as final main model. M1_Ridge is the numerical comparator; M1_ElasticNet is selected.',
 'src/mixture/augmented.py':'candidate / sensitivity model; Qp/Kp augmented models are not used as final main model.',
 'scripts/09_analyze_mixture_effects.py':'M3_Qp / M4_Qp_Kp are candidate / sensitivity models, not final main model. The saved final_model_policy selects M1_ElasticNet.',
 'src/scaling/quality_extension.py':'MQ1/MQ2/gamma fitting: candidate / sensitivity model; not used as final main model. Final chain uses read_parameters, read_data, subset and group_reference.',
 'src/scaling/quality_analysis.py':'MQ candidate functions are not final main model; Phase2C calls stress_test only.',
 'src/scaling/scale_quality.py':'S1/S2 are candidate / sensitivity models, not used as final main model. S3 C,u are frozen inputs to local Delta_Q; the near-zero v is removed without refitting.',
 'src/scaling/mixture_transfer.py':'P1/P2 smooth lambda curves: candidate / sensitivity models, not final main model. Final lambda is piecewise linear in ln N; these curve labels differ from Question3 resource policies.',
 'src/optimization/simplex_sensitivity.py':'P2-A/P2-B are theory / sensitivity only. Final engineering hierarchy: P0 main, P1 empirical enhancement; P2-B has no finite attained optimum.',
 'src/evolution/bridge.py':'B0/B1/B2 are failed bridge candidates. Preserve bridge_valid_for_conversion=False; do not convert Loss to Benchmark.',
 'src/evolution/contribution_regression.py':'Huber, quadratic time and month FE are sensitivity/candidate comparisons. Shapley signed components are not future growth weights.',
 'src/evolution/projection_models.py':'Executed scenario projection, not a reliable long-term forecast. Frozen full-period calibration/model lookahead is disclosed. No Loss bridge conversion.'}

# Exact function/class selections for the compact appendix. No replacement bodies.
COMPACT={
 1:{
 'src/quality/io.py':['infer_source_domain','resolve_metric_value','discover_quality_files','iter_jsonl'],
 'src/quality/vector_transform.py':['parse_vector','scalarize_record','fit_qurater_parameters','transform_qurater'],
 'src/quality/preprocess.py':['impute_matrix','fit_preprocessing_model','apply_preprocessing'],
 'src/quality/normalize.py':['moderate_utility_transform','minmax_transform'],
 'src/quality/critic.py':['critic_weights'],
 'src/quality/aggregate.py':['finite_scores','score_profile','bootstrap_trimmed_mean','domain_summary','compare_distributions'],
 'scripts/03_build_quality_features.py':['read_dataset'],
 'scripts/04_build_quality_score.py':['load_weights','score_dataset'],
 'src/quality/conflict.py':['conflict_indices','fit_conflict_thresholds'],
 'src/mixture/io.py':['read_table','load_datasets','signatures'],
 'src/mixture/composition.py':['ILRTransformer'],
 'src/mixture/modeling.py':['scores','group_folds','candidate','tune'],
 'scripts/08_model_mixture_loss.py':['metadata','fit_checked'],
 'src/mixture/effects.py':['perturb','direction','summary','baselines','analyze'],
 'scripts/09_analyze_mixture_effects.py':['scale_diagnostics','extrapolation']},
 2:{
 'src/scaling/io.py':['load_csv','combine_trajectories'],
 'src/scaling/classical.py':['weights','predict','prediction_jacobian','initial_values','fit','metrics'],
 'src/scaling/bootstrap.py':['initialize','repetition'],
 'scripts/11_fit_classical_scaling.py':['fit_and_cv','cluster_bootstrap','validate'],
 'src/scaling/quality_extension.py':['read_parameters','read_data','subset','group_reference'],
 'src/scaling/quality_analysis.py':['stress_test'],
 'src/scaling/scale_quality.py':['centers','components','physical','encoded','response_jacobian','predict_scale','initial_parameters','fit_scale'],
 'src/scaling/scale_quality_analysis.py':['evaluate','group_cv','q_level_cv','initialize_worker','bootstrap_worker','resampling_jobs','bootstrap_models'],
 'scripts/14_fit_scale_dependent_quality.py':['basics'],
 'src/scaling/mixture_transfer.py':['frozen_metadata','predict_ilr','predict_recipe','read_frozen_recipes','transfer','bootstrap_transfers','empirical_lambda'],
 'src/scaling/marginal_analysis.py':['FrozenGeneralizedScaling','finite_difference'],
 'scripts/16_analyze_generalized_scaling.py':['frozen_model','large_validation','lambda_sensitivity']},
 3:{
 'src/optimization/costs.py':['_validate_quality','quality_increment_per_token','marginal_quality_per_token','cost_components'],
 'src/scaling/marginal_analysis.py':['FrozenGeneralizedScaling'],
 'src/optimization/setup.py':['decision_domains','mixture_sets','budget_scenarios'],
 'src/optimization/resource_solver.py':['load','lambda_zero','ResourceProblem','three_dimensional_checks','run_solve'],
 'src/optimization/simplex_sensitivity.py':['log_contrast','exact_extreme','simplex_search','evidence_level','official_comparison'],
 'src/optimization/transitions.py':['bisect_status','soft_fit','run_diagnose']},
 4:{
 'src/evolution/common.py':['normalized_name'],
 'src/evolution/metadata.py':['canonical_models','epoch_join'],
 'src/evolution/ability.py':['transform','construct_ability'],
 'src/evolution/detailed.py':['value','normalize_score','extract','analyze_details'],
 'src/evolution/bridge.py':['fit_candidate','predict','metrics','decreasing','analyze_bridge'],
 'src/evolution/contribution_regression.py':['design','accuracy','fit_regression','diagnostics','shapley','endpoints','contribution','bootstrap'],
 'src/evolution/contribution_analysis.py':['prepare','model_data','fit_models','make_technology_tables','run_decomposition'],
 'src/evolution/projection_data.py':['load_frozen','select_scope','frontier_stats','build_anchors','benchmark_series','compute_candidates','select_compute_window','compute_grid','rolling_compute','slope','estimate_growth','c3_direction'],
 'src/evolution/projection_models.py':['frozen_m2','linear_fit','build_hindcast_and_trends','quantiles','level','build_projections']}}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def text(path):return (ROOT/path).read_text(encoding='utf-8-sig')


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def closure(roots,edges):
    ordered=[];seen=set()
    def visit(p):
        if p in seen:return
        seen.add(p)
        for e in edges:
            if e['caller']==p:visit(e['source'])
        ordered.append(p)
    for p in roots:visit(p)
    return ordered


def purpose(p):
    name=Path(p).name
    if '_verify_' in name:return '独立公式/输出/冻结哈希验证；只归档，不在本轮执行'
    if name.startswith('02_'):return '历史pooled描述统计；03使用其缺失记录核对，不复用pooled阈值'
    if name.startswith('02c_'):return '03所需重复/重叠/天花板诊断输入的实际生成入口'
    if name.startswith('07_'):return '08所需A4–A15真实数据诊断与input_manifest生成入口'
    if name.startswith('04b_record'):return '既有文本核验材料归档，不当作独立人工实验'
    if p.startswith('scripts/'):return '最终阶段实际执行入口：'+name
    if 'report' in name or 'plot' in name:return PURPOSES.get(name,'最终CSV/图表/报告呈现，compact省略纯绘图和排版')
    return PURPOSES.get(name,'调用链共用依赖或包初始化')


def stage_outputs(p,q):
    if 'conflict' in p or Path(p).name.startswith('05_'):return 'outputs/conflict/'
    if 'profiling' in p or Path(p).name.startswith('02_'):return 'outputs/profile/ （03只复用缺失记录清单）'
    if 'record_diagnostics' in p or Path(p).name.startswith('02c_'):return 'outputs/inspection/ （03前置诊断证据）'
    if p.startswith('src/quality/') or re.match(r'scripts/0[234]',p):return 'outputs/preprocess/; outputs/quality/'
    if 'mixture/' in p and q==1 or Path(p).name.startswith(('07_','08_','09_')):return 'outputs/mixture/; outputs/mixture/stage2/'
    if q==2:
        if 'classical' in p or 'bootstrap.py' in p or Path(p).name.startswith(('10_','11_')):return 'outputs/scaling/phase1/'
        if 'scale_quality' in p or 'quality_' in p or Path(p).name.startswith('14_'):return 'outputs/scaling/phase2C/'
        if 'mixture_transfer' in p or Path(p).name.startswith('15_'):return 'outputs/scaling/phase3/'
        return 'outputs/scaling/phase4/'
    if q==3:return 'outputs/optimization/phase1/; phase2/; phase3/ （依来源模块）'
    if 'projection' in p or Path(p).name.startswith('22_'):return 'outputs/evolution/phase3/'
    if 'contribution' in p or Path(p).name.startswith('21_'):return 'outputs/evolution/phase2/'
    return 'outputs/evolution/phase1/'


def role(p):
    if p.startswith('scripts/'):
        if '_verify_' in p:return 'validation'
        return 'execution_entry'
    if 'report' in p or 'plot' in p:return 'output_presentation'
    if p.endswith('/augmented.py'):return 'candidate_comparison'
    return 'model_or_shared_dependency'


def latex_escape(s):
    return s.replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')


def header(p,start,end,note=''):
    lines=['# ============================================================','# Source: '+p,f'# Original lines: {start}-{end}','# Purpose: '+purpose(p)]
    if p in NOTES:lines+=['# '+NOTES[p]]
    if note:lines+=['# '+note]
    return '\n'.join(lines+['# ============================================================'])+'\n'


def chunks(code,start,p):
    """Split at existing syntax boundaries where possible; no source alteration."""
    lines=code.splitlines(keepends=True);tree=ast.parse(text(p));boundaries={n.end_lineno for n in ast.walk(tree) if isinstance(n,ast.stmt)}
    result=[];i=0
    while i<len(lines):
        cap=min(len(lines),i+100);choices=[b-start+1 for b in boundaries if i+45<=b-start+1<=cap]
        end=max(choices) if choices else cap
        if len(lines)-end<15:end=len(lines)
        result.append((start+i,start+end-1,''.join(lines[i:end])));i=end
    return result


def compile_tex(qblocks,filename,compact=False):
    lines=[r'% Appendix fragment, not a standalone document.',r'% Main document: XeLaTeX + ctex; load listings and xcolor.',r'% All source bodies are unchanged project code; NOT a merged executable script.',r'\appendix',r'\section{Python源程序}',
        r'\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,columns=fullflexible,keepspaces=true,numbers=left,numberstyle=\tiny,showstringspaces=false,frame=single,tabsize=4}']
    for q,blocks in qblocks.items():
        lines += [r'\subsection{'+QUESTIONS[q]['title']+'源程序}',latex_escape(QUESTIONS[q]['model']),r'以下按真实文件调用链摘录；源码位置和原行号用于追溯。候选分支不代表最终采用。']
        for module,p,start,end,code in blocks:
            for i,(a,b,part) in enumerate(chunks(code,start,p),1):
                title=latex_escape(module+'；'+p+f'：{a}–{b}')
                lines += [r'\subsubsection{'+title+'}']
                if p in NOTES:lines+=[latex_escape(NOTES[p])]
                lines += [r'\begin{lstlisting}[language=Python,firstnumber='+str(a)+']',part.rstrip('\r\n'),r'\end{lstlisting}']
    content='\n\n'.join(lines)+'\n';(OUT/filename).write_text(content,encoding='utf8')
    # Explicit estimate only, not a claim of rendered pages.
    wrapped=0;listings=0
    for blocks in qblocks.values():
        for module,p,start,end,code in blocks:
            for _,_,part in chunks(code,start,p):
                listings+=1;wrapped+=sum(max(1,math.ceil(len(line.expandtabs(4))/100)) for line in part.splitlines())
    return dict(estimated_pages_low=math.ceil((wrapped+4*listings)/68),estimated_pages_high=math.ceil((wrapped+7*listings)/55),
        code_visual_lines_at_100_columns=wrapped,listings=listings,assumptions='A4, text area about160x235mm, scriptsize mono, 55-68 effective lines/page; UTF8/long-line wrap depends on actual thesis class; NOT rendered')


def compact_ranges(p,names):
    """Copy complete selected definitions plus original imports/constants used for reading."""
    tree=ast.parse(text(p));ranges=[];available={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for name in names:
        if name not in available:raise ValueError('Missing real compact symbol: '+p+':'+name)
        n=available[name];start=min([n.lineno]+[d.lineno for d in n.decorator_list]);ranges.append((start,n.end_lineno,name))
    # Keep real top-level imports and small definitions of constants; no fabricated dependencies.
    for n in tree.body:
        if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign,ast.AnnAssign)):
            if isinstance(n,(ast.Assign,ast.AnnAssign)) and n.end_lineno-n.lineno>8:continue
            ranges.append((n.lineno,n.end_lineno,'original imports / constants'))
    # Merge adjacent excerpts without losing symbol provenance.
    merged=[]
    for a,b,name in sorted(ranges):
        if merged and a<=merged[-1][1]+3:
            x,y,label=merged[-1];merged[-1]=(x,max(y,b),label if name in label.split('; ') else label+'; '+name)
        else:merged.append((a,b,name))
    return merged


def main():
    inventory=json.loads((OUT/'source_inventory.json').read_text(encoding='utf8'));sources={x['file']:x for x in inventory};edges=json.loads((OUT/'import_graph.json').read_text(encoding='utf8'))
    before=json.loads((OUT/'original_files_before_extraction.json').read_text(encoding='utf8'))
    phase3=json.loads((ROOT/'outputs/evolution/phase3/verification_summary.json').read_text(encoding='utf8'))
    if not phase3.get('verified') or not phase3.get('question4_complete'):raise RuntimeError('Q4 Phase3 not finalized; must not include forecast drafts')
    manifests=[];traces=[];copies=[];fullblocks={};compactblocks={};summary={};plan={}
    for q,info in QUESTIONS.items():
        roots=['scripts/'+p for p in info['entries']];main_files=closure(roots,edges)
        aux_files=closure(['scripts/'+p for p in info['auxiliary']],edges)
        all_files=list(dict.fromkeys(main_files+aux_files))
        for p in list(all_files):
            if p.startswith('src/'):
                init=p.rsplit('/',1)[0]+'/__init__.py'
                if init in sources and init not in all_files:all_files.append(init)
        for p in info['outputs']+info['reports']:
            if not (ROOT/p).is_file():raise ValueError('Proposed final output evidence does not exist: '+p)
        plan[q]=dict(main_files=main_files,full_source_files=all_files,entries=roots,compact_symbols=COMPACT[q])
        fullblocks[q]=[];compactblocks[q]=[];assembled=['# '+info['title']+'：论文源程序逐文件汇编',
            '# For appendix display only; NOT a standalone merged executable.',
            '# Source bodies, imports, formulas, branches, parameters and entry guards are unchanged.',
            '# Use the original scripts from the project root for the original workflow; DO NOT run this listing.',
            '# This extraction task did not rerun any model or experiment.','']
        target_line=len(assembled)+1
        for p in all_files:
            dest=OUT/'full_source'/f'question{q}'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,dest)
            copies.append(dict(question=q,source_file=p,backup_file=dest.relative_to(ROOT).as_posix(),original_sha256=sha(ROOT/p),copied_sha256=sha(dest)))
            core=p in main_files;fullonly=not core
            manifests.append(dict(question=q,source_file=p,role=role(p),final_or_auxiliary='auxiliary' if fullonly or 'report' in p or 'plot' in p or p.endswith('/augmented.py') else 'final_call_chain',
                used_for_output=stage_outputs(p,q),include_in_appendix=core,exclude_reason='仅full_source归档：前置诊断/独立核验/包初始化，不进入论文主源码列表' if fullonly else '',
                copied_to_full_source=True,purpose=purpose(p),final_model=info['model'],candidate_note=NOTES.get(p,''),
                evidence='; '.join(info['reports']),source_sha256=sha(ROOT/p)))
            if not core:continue
            code=text(p);n=len(code.splitlines());h=header(p,1,n);combined=h+code.rstrip('\r\n')+'\n\n'
            hlines=len(h.splitlines());traces.append(dict(question=q,target_file=f'question{q}_source.py',target_start=target_line+hlines,target_end=target_line+hlines+n-1,source_file=p,source_start=1,source_end=n,symbol='complete original file',kind='full_listing',source_body_sha256=hashlib.sha256(('\n'.join(code.splitlines())+'\n').encode()).hexdigest()))
            assembled.extend(combined.splitlines());target_line=len(assembled)+1
            fullblocks[q].append((purpose(p),p,1,n,code))
        complete='\n'.join(assembled)+'\n';(OUT/f'question{q}_source.py').write_text(complete,encoding='utf8')
        compact_line=1;compact_py=[]
        for p,names in COMPACT[q].items():
            if p not in main_files:raise ValueError('Compact source not in verified final chain: '+p)
            original=text(p).splitlines(keepends=True)
            for a,b,symbol in compact_ranges(p,names):
                code=''.join(original[a-1:b]);h=header(p,a,b,'Unchanged excerpt; omitted routines remain in full_source.')
                compact_py.extend((h+code.rstrip('\r\n')+'\n\n').splitlines());start=compact_line+len(h.splitlines());end=start+b-a
                traces.append(dict(question=q,target_file=f'question{q}_compact_source.py',target_start=start,target_end=end,source_file=p,source_start=a,source_end=b,symbol=symbol,kind='compact_excerpt',source_body_sha256=hashlib.sha256(('\n'.join(code.splitlines())+'\n').encode()).hexdigest()))
                compact_line=len(compact_py)+1;compactblocks[q].append((purpose(p)+' / '+symbol,p,a,b,code))
        (OUT/f'question{q}_compact_source.py').write_text('\n'.join(compact_py)+'\n',encoding='utf8')
        # Comparison/training blocks embedded in entry functions are preserved as original excerpts.
        summary[q]=dict(main_file_count=len(main_files),full_source_file_count=len(all_files),source_lines=len(complete.splitlines()),
            compact_lines=len(compact_py),compact_original_lines=sum(b-a+1 for _,_,a,b,_ in compactblocks[q]))
    extra_compact_ranges={
        1:[('scripts/03_build_quality_features.py',215,271,'A1 fit/reload and A2/A3 mapping plus non-overlap'),('scripts/04_build_quality_score.py',205,242,'A1 domain bootstrap and full/non-overlap external score comparisons'),('scripts/08_model_mixture_loss.py',180,224,'nested CV, selected full fit and 1M/60M/1B validation'),('scripts/09_analyze_mixture_effects.py',543,562,'frozen final_model_policy retain/reject rule')],
        2:[('scripts/15_calibrate_mixture_scaling.py',193,201,'frozen S3 C/u to local quality module'),('scripts/15_calibrate_mixture_scaling.py',208,238,'five-scale transfer, bootstrap, reference and empirical knots'),('scripts/14_fit_scale_dependent_quality.py',77,134,'B6-only fits and grouped CV, B7 non-overlap validation'),('scripts/14_fit_scale_dependent_quality.py',165,168,'B8 stress validation call')],
        3:[('src/optimization/setup.py',164,169,'official Lcrit=training_coefficient/eta')],4:[('src/evolution/metadata.py',11,14,'actual C1-C7 CSV read from frozen official attachments')]}
    for q,items in extra_compact_ranges.items():
        dest=OUT/f'question{q}_compact_source.py';current=dest.read_text(encoding='utf8')
        for p,a,b,symbol in items:
            original=text(p).splitlines(keepends=True);code=''.join(original[a-1:b]);h=header(p,a,b,'Original call-site excerpt, not a standalone function.')
            start=len(current.splitlines())+1+len(h.splitlines());end=start+b-a
            traces.append(dict(question=q,target_file=dest.name,target_start=start,target_end=end,source_file=p,source_start=a,source_end=b,symbol=symbol,kind='compact_call_site',source_body_sha256=hashlib.sha256(('\n'.join(code.splitlines())+'\n').encode()).hexdigest()))
            current+=h+code.rstrip('\r\n')+'\n\n';compactblocks[q].append((symbol,p,a,b,code))
        dest.write_text(current,encoding='utf8');summary[q]['compact_lines']=len(current.splitlines());summary[q]['compact_original_lines']=sum(b-a+1 for _,_,a,b,_ in compactblocks[q])
    # Every near-stage Python file gets an explicit inclusion/exclusion decision.
    for q,info in QUESTIONS.items():
        existing={r['source_file'] for r in manifests if r['question']==q}
        stages=range(1,10) if q==1 else range(10,17) if q==2 else range(17,20) if q==3 else range(20,23)
        excluded=[p for p in sources if p.startswith('scripts/') and re.match(r'scripts/(\d+)',p) and int(re.match(r'scripts/(\d+)',p)[1]) in stages and p not in existing]
        if q==1:excluded+=['src/quality/robust_reaggregation.py']
        for p in excluded:
            reason='只作早期结构/描述性诊断，不在最终数值生成依赖链'
            if '06_' in p or 'robust_reaggregation' in p:reason='确实执行过，但最终选择(Q,K)，未激活Q_robust/Q_final；默认不纳入主附录或最终调用链备份'
            if q==2 and ('13_' in p):reason='常数gamma MQ1/MQ2候选未冻结为最终模型；最终局部Delta_Q来自14阶段S3'
            if q==2 and ('12_' in p):reason='质量扩展前的诊断阶段；无最终参数拟合，最终链由14重新严格读入B6/B7/B8'
            manifests.append(dict(question=q,source_file=p,role='excluded_historical_or_diagnostic',final_or_auxiliary='not_final',used_for_output='历史诊断/候选产物；不作为最终主结果',include_in_appendix=False,exclude_reason=reason,copied_to_full_source=False,purpose=purpose(p),final_model=info['model'],candidate_note='',evidence='; '.join(info['reports']),source_sha256=sha(ROOT/p)))
    write_csv(OUT/'source_manifest.csv',manifests);write_csv(OUT/'source_traceability.csv',traces);write_csv(OUT/'full_source_hashes.csv',copies)
    (OUT/'extraction_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8')
    full_pages=compile_tex(fullblocks,'python_appendix.tex');compact_pages=compile_tex(compactblocks,'python_appendix_compact.tex',True)
    result=dict(questions=summary,full_tex_estimate=full_pages,compact_tex_estimate=compact_pages,original_files=len(before),models_rerun=False)
    (OUT/'extraction_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    build_index(manifests,plan,result,traces,copies,before)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def build_index(manifests,plan,result,traces,copies,before):
    """Populate documentation and run static provenance/immutability checks only."""
    # Source verification is parsing/hashing only. Never import or call model code.
    if any(r['original_sha256']!=r['copied_sha256'] for r in copies):
        raise ValueError('Raw source backup differs from original bytes')
    function_rows=[]
    for r in traces:
        original=text(r['source_file']).splitlines()
        target=(OUT/r['target_file']).read_text(encoding='utf8').splitlines()
        expected=original[r['source_start']-1:r['source_end']]
        actual=target[r['target_start']-1:r['target_end']]
        if actual!=expected:raise ValueError('Original-line trace mismatch: '+str(r))
        if r['kind']=='full_listing':
            tree=ast.parse(text(r['source_file']),filename=r['source_file'])
            for n in ast.walk(tree):
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                    function_rows.append(dict(question=r['question'],source_file=r['source_file'],symbol=n.name,
                        symbol_type=type(n).__name__,source_start=n.lineno,source_end=n.end_lineno,
                        target_file=r['target_file'],target_start=r['target_start']+n.lineno-1,
                        target_end=r['target_start']+n.end_lineno-1))
    write_csv(OUT/'source_function_traceability.csv',function_rows)
    edges=json.loads((OUT/'import_graph.json').read_text(encoding='utf8'))
    for q,p in plan.items():
        files=set(p['full_source_files'])
        for e in edges:
            if e['caller'] in files and e['source'] not in files:
                raise ValueError('Incomplete raw backup import closure: '+str(e))
    for fname in ('python_appendix.tex','python_appendix_compact.tex'):
        t=(OUT/fname).read_text(encoding='utf8')
        if t.count(r'\begin{lstlisting}')!=t.count(r'\end{lstlisting}'):
            raise ValueError('Unbalanced listings in '+fname)
    historical=json.loads((OUT/'recorded_source_hash_checks.json').read_text(encoding='utf8'))
    if not all(r['match'] for r in historical):raise ValueError('Historical source version ambiguity')
    policy=json.loads((ROOT/'outputs/mixture/stage2/final_model_policy.json').read_text(encoding='utf8'))
    bridge=json.loads((ROOT/'outputs/evolution/phase1/bridge_model.json').read_text(encoding='utf8'))
    final=json.loads((ROOT/'outputs/scaling/phase4/generalized_scaling_final.json').read_text(encoding='utf8'))
    phase3=json.loads((ROOT/'outputs/evolution/phase3/verification_summary.json').read_text(encoding='utf8'))
    if policy['selected_model']!='M1_ElasticNet' or policy['Qp_retained'] or policy['Kp_retained']:
        raise ValueError('Final mixture policy disagrees with requested frozen model')
    if bridge['bridge_valid_for_conversion'] is not False:raise ValueError('Bridge failure was not retained')
    if final['quality']['N0_B']!=1 or not final['quality']['v_removed_without_refit']:
        raise ValueError('Frozen local quality formula differs')
    if 'src/quality/robust_reaggregation.py' in plan[1]['main_files']:
        raise ValueError('Rejected robust score included in main source')
    formula_checks=[
        ('Q1 CRITIC ddof=1', 'src/quality/critic.py','values.std(axis=0, ddof=1)'),
        ('Q1 CRITIC information','src/quality/critic.py','information = sigma * conflict'),
        ('Q1 score weighted mean','scripts/04_build_quality_score.py','values @ weights'),
        ('Q1 K weighted absolute deviation','src/quality/conflict.py','k=np.abs(deviation)@w'),
        ('Q1 fold-fitted Helmert ILR','src/mixture/composition.py','helmert'),
        ('Q2 local quality','src/scaling/marginal_analysis.py',"dq=q['C_Q']*(N/q['N0_B'])**(-q['u_Q'])*(-np.log(Q))"),
        ('Q2 ln N piecewise interpolation','src/scaling/marginal_analysis.py','np.interp(np.log(x),np.log(self.Ns),self.lambdas)'),
        ('Q3 reduced D_B','src/optimization/resource_solver.py','d=np.minimum(self.dmax,self.b/c)'),
        ('Q3 D_min constraint','src/optimization/resource_solver.py','self.b/self.dlo'),
        ('Q4 signed two-factor Shapley','src/evolution/contribution_regression.py','ds=.5*(f10-f00+f11-f01);dt=.5*(f01-f00+f11-f10)')]
    for label,p,token in formula_checks:
        if token not in text(p):raise ValueError('Static formula check failed: '+label)
    write_csv(OUT/'static_formula_checks.csv',[dict(check=label,source_file=p,original_code=token,passed=True) for label,p,token in formula_checks])
    # Actual reported execution evidence and configurations, not guessed filenames.
    evidence=[];config_by_question={}
    for q,info in QUESTIONS.items():
        bodies='\n'.join(text(p) for p in plan[q]['full_source_files'])
        configs=[p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'configs').glob('*.yaml')) if p.name in bodies]
        config_by_question[q]=configs
        paths=list(dict.fromkeys(info['reports']+info['outputs']+configs))
        for p in paths:
            evidence.append(dict(question=q,evidence_file=p,evidence_type='config' if p.startswith('configs/') else 'final_report' if p.endswith('.md') else 'frozen_result',sha256=sha(ROOT/p),size=(ROOT/p).stat().st_size))
    write_csv(OUT/'execution_evidence.csv',evidence)
    md=['# 四问最终源程序映射','',
        '本索引依据最终报告、冻结结果/参数、历史实现哈希及实际 import 关系建立。文件级调用链包含真实运行入口所导入的候选比较和呈现函数；被导入不等于其中所有分支都被最终模型采用。',
        '完整整理版不删除任何原文件正文；full_source 保持逐字节原样。独立核验/前置诊断在完整备份中保存，不当作主模型。','']
    readme=['# 论文附录 Python 源程序','',
        '只提取既有代码。本次没有读取质量附件记录、拟合、训练、优化、运行实验或绘图，也没有改变论文正文、源代码、配置和冻结结果。',
        'questionX_source.py 是按真实文件调用链排序的论文展示汇编，**不是独立可执行的合并脚本**。原 import、相对 import、多个 __future__ 和 main 入口全部保留；请勿直接执行该汇编。full_source 也只备份 Python，不打包附件、参数、配置或依赖环境。此处入口说明仅作溯源，不表示允许重跑冻结实验。',
        'questionX_compact_source.py 和 compact TeX 是原文件函数/调用片段，省略位置由原行号标注。它们同样用于论文展示。source_traceability.csv 和 source_function_traceability.csv 给出一一对应；full_source_hashes.csv 给出原始与备份 SHA256。','',
        '使用 LaTeX 时选择 full 或 compact 中的一份，在正文末尾使用 \\input{python_appendix_compact.tex}（按论文工程路径调整）。这是附录片段，不是独立文档；主文档需 XeLaTeX/ctex、listings、xcolor。正文已含 \\appendix 时合并时保留一次即可。长文件在原语法边界附近切为通常不超过 100 行的列表，行号仍是原文件行号。','',
        '本次只做源程序静态核验，未编译完整论文；页数为纸张/字号/折行假设下的估计，最终排版取决于论文模板。','']
    for q,info in QUESTIONS.items():
        records=[r for r in manifests if r['question']==q]
        core=[r for r in records if r['include_in_appendix']]
        aux=[r for r in records if r['copied_to_full_source'] and not r['include_in_appendix']]
        excluded=[r for r in records if not r['copied_to_full_source']]
        md += ['# '+info['title'],'','## 最终实际使用文件','',info['model'],'',
            '实际执行入口：'+', '.join('`scripts/'+p+'`' for p in info['entries'])+'。',
            '结果来源：'+', '.join('`'+p+'`' for p in info['outputs'])+'。',
            '报告证据：'+', '.join('`'+p+'`' for p in info['reports'])+'。','',
            '| 原始文件 | 用途 | 主模型/验证/辅助角色 | 对应最终输出 |',
            '|---|---|---|---|']
        for r in core+aux:
            label=r['role']+('；full_source only' if not r['include_in_appendix'] else '；主调用链')
            note='；'+r['candidate_note'] if r['candidate_note'] else ''
            md.append('| `'+r['source_file']+'` | '+r['purpose']+note+' | '+label+' | '+r['used_for_output']+' |')
        md += ['','## 未纳入的相近文件','']
        md += ['- `'+r['source_file']+'`：'+r['exclude_reason'] for r in excluded] or ['无其他相近执行入口被排除。']
        md += ['','## compact 摘录范围','']
        for p,names in COMPACT[q].items():md.append('- `'+p+'`：'+', '.join('`'+n+'`' for n in names))
        md+=['- 入口中的冻结选择/拟合/CV/外部验证调用片段另按原行号摘录，详见 source_traceability.csv。','']
        readme += ['## '+info['title'],'',info['model'],'',
            '入口顺序：'+ ' → '.join('`'+p+'`' for p in plan[q]['entries'])+'。',
            '读取的现有配置：'+', '.join('`'+p+'`' for p in config_by_question[q])+'。','',
            '原始代码文件：','']
        readme += ['- `'+r['source_file']+'`：'+r['purpose'] for r in core]
        readme+=['','compact 包含：','']
        readme += ['- `'+p+'`：'+', '.join(names) for p,names in COMPACT[q].items()]
        readme+=['','仅 full_source 保存（前置诊断、独立核验、初始化及其依赖）：','']
        readme += ['- `'+r['source_file']+'`：'+r['purpose'] for r in aux]
        readme+=['','未作为最终主模型纳入：','']
        readme += ['- `'+r['source_file']+'`：'+r['exclude_reason'] for r in excluded]
        readme+=['- 同一文件中的候选比较保持原样，并加文件块注释；具体见 source_manifest.md 和 candidate_note 列。',
            '- 完整目录级输出映射与逐个冻结结果路径见 source_manifest.md / execution_evidence.csv。','']
    readme+=['## 行数和页数','', '| 问题 | 完整调用链文件 | 原样备份文件 | 整理版行数 | compact 展示行数 | compact 原代码行数 |', '|---|---:|---:|---:|---:|---:|']
    for q,s in result['questions'].items():
        readme.append(f"| 问题{q} | {s['main_file_count']} | {s['full_source_file_count']} | {s['source_lines']} | {s['compact_lines']} | {s['compact_original_lines']} |")
    for label,key in [('完整','full_tex_estimate'),('compact','compact_tex_estimate')]:
        est=result[key];readme+=['',f"{label} TeX 估计 {est['estimated_pages_low']}–{est['estimated_pages_high']} 页；{est['listings']} 个列表。假设：{est['assumptions']}。"]
    readme+=['','compact 保留原函数完整正文及必要调用片段，因此部分候选分支、检查和日志仍随核心函数保留；未为缩页数改变实现。若投稿篇幅更紧，可按 source_manifest 的原函数范围继续人工取舍；本次不改论文正文。']
    (OUT/'source_manifest.md').write_text('\n'.join(md)+'\n',encoding='utf8')
    (OUT/'README.md').write_text('\n'.join(readme)+'\n',encoding='utf8')
    # Compare every pre-existing file, including original data, outputs and caches.
    changes=[]
    for i,(p,old) in enumerate(before.items(),1):
        file=ROOT/p
        exists=file.is_file();now=sha(file) if exists else ''
        size=file.stat().st_size if exists else None
        if now!=old['sha256'] or size!=old['size']:changes.append(dict(file=p,exists=exists,size=size,sha256=now))
        if i%800==0:print(f'Existing-file integrity checked {i}/{len(before)}',flush=True)
    unknown=[p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and not p.is_relative_to(OUT) and p.relative_to(ROOT).as_posix() not in before]
    # Word may create a temporary owner/lock file while the user views the DOCX.
    # The extractor never writes outside OUT; record this activity, do not delete it.
    office_locks=[p for p in unknown if Path(p).name.startswith('~$') and Path(p).suffix.lower()=='.docx']
    unexpected=[p for p in unknown if p not in office_locks]
    integrity=dict(original_files_checked=len(before),original_files_modified=len(changes),
        unexpected_files_outside_appendix=unexpected,observed_new_office_lock_files=office_locks,
        changes=changes,all_original_files_unchanged=not changes,all_unchanged=not changes and not unexpected)
    (OUT/'frozen_integrity_check.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if changes or unexpected:raise ValueError('Pre-existing project integrity changed; see frozen_integrity_check.json')
    checks=[
        ('原文件存在、原始备份字节一致',f'{len(copies)} 个按问题归档副本全部 SHA256 匹配'),
        ('逐行追溯',f'{len(traces)} 个原文件/函数/调用片段逐行匹配；{len(function_rows)} 个函数/类节点原位置可查'),
        ('最终源版本证据',f'{len(historical)} 条历史 implementation_manifest 源码哈希全部匹配；其余调用链由实际 import、最终报告、输出参数及冻结文件哈希支持'),
        ('问题一最终指标','Q=Z@w；K=abs(Z-Q)@w。最终保留(Q,K)，不采用Q*、Q_robust或分组Huber作为主评分；保留ILR＋一阶ElasticNet，Ridge只是数值对照'),
        ('问题二公式','L0+ C_Q*(N_B/1)^(-u_Q)*(-lnQ) + lambda_p(N_B)*(M(p)-M_ref)。S3的C/u原参数读入，v去掉且不重拟合；B6局部拟合不宣称统一N-D-Q法则；S-A为主边界'),
        ('问题二证据分层','1M/60M/1B观测配方与10B/70B估算Loss明确区分；后两者不作为真实实验或调参/模型选择证据；同脚本S1/S2/MQ及平滑lambda候选有明确注释'),
        ('问题三D解析消元','实现 d=min(Dmax_B,(C/1e18)/(N_B*(6+eta*Lctx)+[g(Q)-g(Q0)]_+/1e9))，与用户分母N_B*(6+eta*Lctx)*1e18+1e9*[g(Q)-g(Q0)]_+等价，输出是D_B而非原始token数；保留D_min约束'),
        ('问题三证据层级','P0正文主方案；P1经验候选增强；P2-A理论外推/敏感性，P2-B正配方边界极限无界，不能报告有限最优解；解析消元、实际优化器和连续预算识别均有原源码'),
        ('问题四回归与分解','M1含lnN/t，M2-C含lnC/t，M3含lnN/lnD/t；OLS+HC3及敏感性原样，双因素Shapley保留符号，Delta_scale+Delta_tech=预测增量，负项不能当作非负增长占比'),
        ('问题四桥接','冻结 bridge_valid_for_conversion=False 保留；禁止把失败桥接用于Loss到Benchmark数值转换'),
        ('问题四Phase3实际完成',f"已有独立核验verified={phase3['verified']}，question4_complete={phase3['question4_complete']}，{phase3['checks']}项检查、{phase3['scenario_rows']}情景行；因此纳入真实锚点/C4增长/MC/回测/风险代码；本次没有重运行"),
        ('问题四预测口径','冻结六任务Winsor/median/IQR使实际A_z为有操作性上界的变换；24个月情景越界应标风险，不称无限能力/可靠长期预测；保留全期变换及冻结M2的前视限制说明'),
        ('冻结完整性',f'{len(before)} 个任务前已存在文件逐一SHA256与大小核对，无修改；本提取器只写appendix。额外观察到Word临时锁文件：'+('; '.join(office_locks) if office_locks else '无')+'；不删除、不修改，不作为源码或冻结结果'),
        ('版本模糊性','没有无法确认最终使用入口的代码；文件包含未采用候选分支是实际比较过程的保留，不等于独立最终模型'),
        ('论文公式一致性检查范围','已核对用户列明公式、最终Phase报告和冻结模型。未获得最终整篇论文稿，不能据此声称所有正文公式逐字核对；未发现所列主公式实质冲突。D*单位及A_z操作性上界需保持明确')]
    audit=['# 源程序提取静态核验','',
        '执行范围：源码 AST/原行切片、文件复制与哈希、既有报告/JSON/清单读取。没有加载项目模型模块，没有重新训练/拟合/优化/预测/绘图，也没有重新读取 A1/A2/A3 质量记录。原数据仅作字节哈希完整性核对。',
        '逐文件原始正文保持完整。备份为逐字节复制；汇编只统一文本行结束并增加来源/用途/候选说明。compact 为原片段，不能当作完整可执行脚本。没有为凑篇幅或贴合文字改写函数、参数、公式、变量或模型。','']
    for title,detail in checks:audit+=['## '+title,'',detail,'']
    audit+=['## 证据文件','',
        '- source_manifest.csv / .md：问题→源文件→输出→用途→候选/排除规则。',
        '- extraction_plan.json / import_graph.json：实际项目 import 闭包与入口顺序。',
        '- source_traceability.csv / source_function_traceability.csv：完整/精简展示片段及原行号。',
        '- full_source_hashes.csv：每问备份与原文件 SHA256。',
        '- recorded_source_hash_checks.json：51条历史实现清单版本对照。',
        '- execution_evidence.csv：报告/冻结结果/配置真实路径、大小与SHA256。',
        '- static_formula_checks.csv：静态公式标记；不执行数值模型。',
        '- original_files_before_extraction.json / frozen_integrity_check.json：任务前后原文件完整性。',
        '- extraction_summary.json：真实行数和明确未渲染的页数估计。','',
        'LaTeX只核验列表成对、代码片段和原文件一致性，没有编译论文或生成新图表；页数不是渲染测量值。纯绘图/报告及独立审计从compact省略；关键拟合、CV、bootstrap、优化约束及预测/分解保留。',
        '附录完成后停止，不修改论文正文。']
    (OUT/'source_extraction_audit.md').write_text('\n'.join(audit)+'\n',encoding='utf8')


if __name__=='__main__':main()
