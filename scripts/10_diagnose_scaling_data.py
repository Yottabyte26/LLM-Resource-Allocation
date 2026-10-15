"""Question 2 phase 1 first step: B1-B5 diagnosis only, no fitting."""
from __future__ import annotations
import logging
from pathlib import Path
import sys
from collections import Counter
from decimal import Decimal
import json
import re
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from scaling.io import input_paths,load_csv,diagnosis,csv_rows,write_csv,write_json,sha256,size_key
OUT=ROOT/'outputs/scaling/phase1'


def protected_files():
    """Freeze every preexisting project file, including original attachments and Q1.

    Only the specifically new scaling paths are excluded. Original B6-B10 bytes
    may be hashed for integrity, but their CSV values are never ingested.
    """
    manifest=OUT/'frozen_project_before.json'
    if manifest.exists():return json.loads(manifest.read_text(encoding='utf-8'))
    paths=[]
    excluded=[ROOT/'outputs/scaling',ROOT/'outputs/figures/question2_phase1',ROOT/'src/scaling']
    for p in ROOT.rglob('*'):
        if not p.is_file() or '.git' in p.parts or any(p.is_relative_to(e) for e in excluded):continue
        if p.name=='scaling_phase1.yaml' or p.name.startswith('10_diagnose_scaling') or p.name.startswith('11_'):continue
        if p==ROOT/'outputs/report/question2_phase1_classical_scaling.md':continue
        paths.append(p)
    result={p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(paths)}
    write_json(manifest,result)
    return result


def verify_protected(before):
    """All frozen files stay byte-identical; fail if a file is missing or changed."""
    rows=[]
    for p,h in before.items():
        actual=sha256(ROOT/p) if (ROOT/p).exists() else 'MISSING'
        rows.append(dict(path=p,before_sha256=h,after_sha256=actual,unchanged=h==actual))
        if actual!=h:raise RuntimeError(f'Frozen project file changed: {p}')
    write_csv(OUT/'frozen_project_integrity.csv',rows)
    return len(rows)


def md_table(rows,columns):
    """Readable diagnosis tables with missing applicability distinguished from zero."""
    def value(x):
        if x is None:return '不适用/未提供'
        if isinstance(x,float):return f'{x:.6g}'
        return str(x).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join('---' for c in columns)+' |']+[
        '| '+' | '.join(value(r.get(c)) for c in columns)+' |' for r in rows])


def main():
    """Inspect actual tables and write provenance/grouping before any nonlinear fit."""
    OUT.mkdir(parents=True,exist_ok=True)
    (ROOT/'outputs/figures/question2_phase1').mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(OUT/'diagnosis.log',encoding='utf-8')])
    frozen=protected_files();verify_protected(frozen)
    files,trajectory_paths,aux=input_paths(ROOT)
    tables={label:load_csv(path,label) for label,path in files.items()}
    trajectories=[load_csv(path,'B3') for path in trajectory_paths]
    summaries=[];fields=[];manifest=[]
    for t in list(tables.values())+trajectories:
        s,f=diagnosis(t,ROOT);summaries.append(s);fields+=f
        manifest.append(dict(dataset=t['label'],path=t['path'].relative_to(ROOT).as_posix(),sha256=sha256(t['path']),
            usage='primary_fit' if t['label']=='B1' else 'validation_only'))
    auxiliary=[]
    for path in aux:
        columns,records=csv_rows(path)
        auxiliary.append(dict(path=path.relative_to(ROOT).as_posix(),rows=len(records),columns=len(columns),field_names=columns))
        manifest.append(dict(dataset='B11' if 'metadata' in path.name else 'B12',path=path.relative_to(ROOT).as_posix(),sha256=sha256(path),usage='grouping_provenance_only'))
    _,checkpoint_rows=csv_rows(aux[1]);_,family_metadata=csv_rows(aux[0])
    nominal={}
    for r in checkpoint_rows:
        label=r['model_size'];match=re.fullmatch(r'([\d.]+)([mb])',label.lower())
        if not match:raise ValueError(f'Unexpected B12 nominal model_size {label}')
        nominal[label]=float(match[1])/(1000 if match[2]=='m' else 1)
    b1=tables['B1'];group_rows=[];membership=[];flops=[];trend_events=[]
    run_counts=Counter(r['run_id'] for r in b1['records'])
    # Quantization bounds from actual exported column decimal precision.
    half={c:.5*10**(-max(max(-Decimal(r[c]).as_tuple().exponent,0) for r in b1['records'])) for c in ['N_params_B','D_tokens_B','C_FLOPs_1e21']}
    for group in np.unique(b1['group']):
        sel=np.where(b1['group']==group)[0];steps=np.array([float(b1['records'][i]['steps']) for i in sel]);order=np.argsort(steps);sel=sel[order];steps=steps[order]
        N,D,L=b1['N'][sel],b1['D'][sel],b1['L'][sel]
        if len(set(N))!=1:raise ValueError('N varies within inferred B1 trajectory')
        size=min(nominal,key=lambda key:abs(np.log(nominal[key]/N[0])))
        actual_steps=set(map(int,steps));b12_steps={int(r['step']) for r in checkpoint_rows if r['model_size']==size}
        metadata_repos={r['model_repo'] for r in family_metadata};repos={r['model_repo'] for r in checkpoint_rows if r['model_size']==size}
        info=dict(trajectory_id=group,N_params_B=float(N[0]),n_checkpoints=len(sel),N_fixed=True,
            unique_checkpoint_count=len(set(steps)),D_strictly_increasing=bool(np.all(np.diff(D)>0)),
            D_min=float(D.min()),D_max=float(D.max()),first_loss=float(L[0]),last_loss=float(L[-1]),
            loss_decrease_first_to_last=float(L[0]-L[-1]),loss_rebound_count=int(np.sum(np.diff(L)>0)),
            maximum_loss_rebound=float(max(0,np.diff(L).max())),matched_B12_nominal_size=size,
            nominal_match_rule='nearest log-scale B12 name; grouping is exact B1 N, not nominal value',
            B12_matching_checkpoint_steps=len(actual_steps&b12_steps),B12_unmatched_B1_steps=sorted(actual_steps-b12_steps),
            B11_repo_metadata_present=bool(repos&metadata_repos))
        group_rows.append(info)
        for position,i in enumerate(sel):
            r=b1['records'][i]
            membership.append(dict(source_row=i+1,run_id=r['run_id'],trajectory_id=group,N_params_B=float(N[position]),
                steps=int(steps[position]),D_tokens_B=float(D[position]),val_loss=float(L[position]),
                training_progress_steps=float(steps[position]/steps[-1]),checkpoint_weight_raw=1/len(sel),
                group_rule='B1 exact canonical N; run_id is unique checkpoint identifier'))
            observed=float(r['C_FLOPs_1e21']);expected=.006*N[position]*D[position]
            tolerance=half['C_FLOPs_1e21']+.006*(abs(D[position])*half['N_params_B']+abs(N[position])*half['D_tokens_B']+half['N_params_B']*half['D_tokens_B'])
            flops.append(dict(source_row=i+1,trajectory_id=group,C_reported_1e21=observed,C_6ND_1e21=float(expected),
                absolute_difference=float(abs(observed-expected)),relative_difference=float(abs(observed-expected)/expected),
                export_rounding_absolute_tolerance=float(tolerance),within_export_rounding=bool(abs(observed-expected)<=tolerance+1e-14)))
            if position and (D[position]<=D[position-1] or L[position]>L[position-1]):
                trend_events.append(dict(source_row=i+1,trajectory_id=group,D_change=float(D[position]-D[position-1]),Loss_change=float(L[position]-L[position-1])))
    # Never deduplicate or repair original rows: stop before dependent fitting if ambiguous.
    b1summary=next(r for r in summaries if r['dataset']=='B1')
    if b1summary['duplicate_checkpoint_groups'] or b1summary['exact_duplicate_groups']:
        raise ValueError('B1 duplicate checkpoints/rows require explicit review; no automatic deletion')
    if not all(r['D_strictly_increasing'] for r in group_rows):raise ValueError('B1 D trajectory not strictly increasing')
    write_csv(OUT/'data_file_summary.csv',summaries);write_csv(OUT/'field_diagnosis.csv',fields)
    write_csv(OUT/'auxiliary_schema_summary.csv',auxiliary);write_csv(OUT/'input_manifest.csv',manifest)
    write_csv(OUT/'B1_trajectory_summary.csv',group_rows);write_csv(OUT/'B1_trajectory_membership.csv',membership)
    write_csv(OUT/'B1_flops_consistency.csv',flops);write_csv(OUT/'B1_trajectory_anomalies.csv',trend_events,['source_row','trajectory_id','D_change','Loss_change'])
    _,literature=csv_rows(files['B5'])
    literature_sources=[dict(source=s,n=sum(r['source']==s for r in literature),families=sorted({r['family'] for r in literature if r['source']==s}),
        evaluation_dataset_provided=False,tokenizer_provided=False,loss_type_provided=False,
        absolute_loss_comparability='unconfirmed; source-layer trend primary, pooled errors diagnostic only') for s in sorted({r['source'] for r in literature})]
    write_csv(OUT/'B5_source_diagnosis.csv',literature_sources)
    state=dict(B1_trajectory_count=len(group_rows),B1_records=len(b1['N']),N_range=[float(b1['N'].min()),float(b1['N'].max())],
        D_range=[float(b1['D'].min()),float(b1['D'].max())],Loss_range=[float(b1['L'].min()),float(b1['L'].max())],
        B1_unique_run_ids=len(run_counts),B1_run_id_is_trajectory=False,group_rule=b1['group_rule'],
        terminal_D_values=sorted(set(r['D_max'] for r in group_rows)),B1_loss_rebounds=sum(r['loss_rebound_count'] for r in group_rows),
        B1_C_6ND_rounding_consistent_count=sum(r['within_export_rounding'] for r in flops),
        frozen_file_count=len(frozen),B5_absolute_loss_comparability_confirmed=False,
        data_sources_used_for_fitting=['B1'],disallowed_B6_B10_ingested=False)
    write_json(OUT/'data_diagnosis.json',state)
    lines=['# 问题二 PHASE 1 数据诊断（尚未拟合）','',
        'N_params_B、D_tokens_B 直接按 1B 单位输入模型，不乘 1e9。FLOPs 核验采用 6ND/1e21=0.006 N_B D_B。原始数据不填补、不删除、不覆盖。','',
        md_table(summaries,['dataset','file_path','rows','columns','missing_or_nan','inf_count','exact_duplicate_excess_rows','duplicate_checkpoint_excess_rows','N_min','N_max','D_min','D_max','Loss_min','Loss_max']),'',
        '字段完整列表见 data_file_summary.csv；逐字段 NaN/空缺与 Inf 见 field_diagnosis.csv。B2 的可选元数据空缺不等于 N/D/Loss 缺失，未做任何填补。无 checkpoint 字段的 B4/B5 记“不适用”，不虚报重复 checkpoint 为零。','',
        f"B1 实际 {len(group_rows)} 个规模、{len(b1['N'])} 条 checkpoint。run_id 有 {len(run_counts)} 个唯一值，是 checkpoint 标识，不是模型轨迹 ID。轨迹按来源 B1 + 精确规范化 N_params_B 分组；本文件每个规模只有一条候选轨迹，无法识别未标注的同规模重复运行。",'',
        md_table(group_rows,['trajectory_id','N_params_B','n_checkpoints','D_strictly_increasing','loss_rebound_count','D_max','matched_B12_nominal_size','B12_matching_checkpoint_steps']),'',
        'B12 的名义规模用于辅助对照，参数计数不一定等于命名规模；多出的名义模型不补造 B1 数据。run_id 不同不能使同一轨迹 checkpoint 成为独立观测。Group CV 留出整条轨迹，bootstrap 以轨迹整体重采样。','',
        f"B1 的 C 与 6ND：{state['B1_C_6ND_rounding_consistent_count']}/{len(flops)} 条在实际 CSV 导出精度容差内一致。相对误差在极小 C 处可能因四舍五入很大，不能只看相对值判异常。",'',
        'B3 各文件含 interpolated 字段，只称插值轨迹一致性检验；B2 按本轮给定的半合成来源定义，仅作辅助敏感性。B4 跨模型族表未提供完整评测数据集/tokenizer 口径，数值误差只反映给定表上的一致性。','',
        md_table(literature_sources,['source','n','families','absolute_loss_comparability']),'',
        'B5 有文献 source 和 family，没有 loss_type、dataset、tokenizer。跨文献绝对 Loss 可比性无法由附件确认，所以 source 内排序是核心文献证据，绝对误差只作分层描述，不当作统一口径精度。','',
        f"末端 D 的实际取值：{state['terminal_D_values']}。若所有模型的 terminal D 相同，则 E 与 B D_terminal^-beta 混淆，终点数据不能分别识别 E/B/beta；仍输出约束拟合对照，但不将 beta 的数值当作终点实证结论。",'',
        f"{len(frozen)} 个既有文件的哈希已建立并核验。只读取 B1-B5、B11/B12 作为数值/元数据输入，B6-B10 未用于拟合。",'']
    (OUT/'data_diagnosis.md').write_text('\n\n'.join(lines),encoding='utf-8')
    verify_protected(frozen)
    logging.info('DIAGNOSIS COMPLETE: B1 %d trajectories / %d checkpoints; no model fitted',len(group_rows),len(b1['N']))
    print(json.dumps(state,ensure_ascii=True))


if __name__=='__main__':main()
