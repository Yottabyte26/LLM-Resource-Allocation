"""Exhaustive C8 parsing and true subtask-level, version-aware diagnostics."""
import json
import logging
import re
from difflib import get_close_matches
import numpy as np
import pandas as pd
from .common import csv,dump,corr,normalized_name


def value(r,metric):
    """Nonfinite/out-of-range raw evaluation scores stay missing."""
    try:v=float(r.get(metric,np.nan))
    except (ValueError,TypeError):return np.nan
    return v if np.isfinite(v) and 0<=v<=1 else np.nan


def normalize_score(raw,baseline=0):
    """Official chance correction and below-chance floor, expressed in percent."""
    return 100*max(0,(raw-baseline)/(1-baseline)) if np.isfinite(raw) else np.nan


def extract(r,directory,filename):
    """Parse leaf task scores, retaining metrics, versions and baseline evidence."""
    results=r.get('results',{});configs=r.get('configs',{});versions=r.get('versions',{});leaves=[];six={}
    model=r.get('model_name')
    if not model:
        arg=r.get('config',{}).get('model_args','');match=re.search(r'(?:^|,)pretrained=([^,]+)',arg)
        model=match.group(1) if match else directory
    def add(family,task,metric,baseline,proof):
        raw=value(results.get(task,{}),metric)
        if np.isfinite(raw):leaves.append(dict(directory=directory,Model=model,source_file=filename,benchmark=family,
            subtask=task+'::'+metric,task=task,metric=metric,raw_score=raw,random_baseline=baseline,
            task_score=normalize_score(raw,baseline),baseline_evidence=proof,task_version=str(versions.get(task,'unknown'))))
    # IFEval has two strict components; loose variants/stderr are not extra tasks.
    for metric in ['prompt_level_strict_acc,none','inst_level_strict_acc,none']:
        add('IFEval','leaderboard_ifeval',metric,0,'official strict generative metric')
    for task in results:
        if task.startswith('leaderboard_bbh_'):
            choices=configs.get(task,{}).get('doc_to_choice')
            if isinstance(choices,list) and len(choices)>1:add('BBH',task,'acc_norm,none',1/len(choices),'C8 config doc_to_choice list')
        elif task.startswith('leaderboard_math_') and task!='leaderboard_math_hard':
            add('MATH Lvl 5',task,'exact_match,none',0,'generative exact match; version preserved')
        elif task.startswith('leaderboard_gpqa_'):
            add('GPQA',task,'acc_norm,none',.25,'official four-choice GPQA; C8 configurations')
        elif task.startswith('leaderboard_musr_'):
            choices={'leaderboard_musr_murder_mysteries':2,'leaderboard_musr_object_placements':5,'leaderboard_musr_team_allocation':3}
            if task in choices:add('MUSR',task,'acc_norm,none',1/choices[task],'official normalization task choice counts')
    add('MMLU-PRO','leaderboard_mmlu_pro','acc,none',.1,'official MMLU-PRO chance baseline')
    for family in ['IFEval','BBH','MUSR']:
        vals=[v['task_score'] for v in leaves if v['benchmark']==family]
        six[family]=float(np.mean(vals)) if vals else np.nan
    # Group GPQA and MATH can be sample-weighted; keep the group estimate,
    # and save equal subtask means separately rather than conflating them.
    six['GPQA']=normalize_score(value(results.get('leaderboard_gpqa',{}),'acc_norm,none'),.25)
    six['MATH Lvl 5']=normalize_score(value(results.get('leaderboard_math_hard',{}),'exact_match,none'))
    six['MMLU-PRO']=normalize_score(value(results.get('leaderboard_mmlu_pro',{}),'acc,none'),.1)
    return model,six,leaves


def analyze_details(root,cfg):
    """Read every JSON; choose latest parsable file per directory, never sample."""
    out=root/'outputs/evolution/phase1';base=root/'real_attachments/C_efficiency_evolution/detailed_results'
    audits=[];aggregations=[];leaves=[];no_usable=[]
    directories=sorted(p for p in base.iterdir() if p.is_dir())
    for number,d in enumerate(directories,1):
        usable=[]
        for p in sorted(d.rglob('*.json')):
            row=dict(directory=d.name,source_file=p.relative_to(root).as_posix(),parse_success=False,selected=False,parse_error='')
            try:
                r=json.loads(p.read_text(encoding='utf-8-sig'))
                row['parse_success']=True
                if not isinstance(r,dict) or not isinstance(r.get('results'),dict):raise ValueError('JSON parsed but lacks evaluation results mapping')
                match=re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?)',p.name)
                timestamp=pd.to_datetime(match.group(1)[:10]+'T'+match.group(1)[11:].replace('-',':'),utc=True,errors='coerce') if match else pd.NaT
                source='filename_run_timestamp'
                if pd.isna(timestamp):
                    rawdate=r.get('date');timestamp=pd.to_datetime(rawdate,unit='s',utc=True,errors='coerce') if isinstance(rawdate,(int,float)) else pd.to_datetime(rawdate,utc=True,errors='coerce')
                    source='JSON_date'
                row.update(run_time=str(timestamp),time_source=source,model_name=r.get('model_name'),
                    task_version_signature=json.dumps(r.get('versions',{}),sort_keys=True))
                model,six,sub=extract(r,d.name,row['source_file']);row['six_complete']=all(np.isfinite(list(six.values())))
                usable.append((timestamp.value if pd.notna(timestamp) else -1,p.name,len(audits),model,six,sub))
            except (OSError,UnicodeError,json.JSONDecodeError,ValueError,TypeError) as exc:row['parse_error']=str(exc)[:500]
            audits.append(row)
        if not usable:no_usable.append(d.name);continue
        chosen=max(usable,key=lambda v:(v[0],v[1]));_,_,i,model,six,sub=chosen;audits[i]['selected']=True
        leaves.extend(sub)
        vals=np.array([x['task_score'] for x in sub],float)
        agg=dict(directory=d.name,Model=model,source_file=audits[i]['source_file'],run_time=audits[i]['run_time'],
            six_complete=audits[i]['six_complete'],subtask_count=len(vals),
            mean_task_score=float(np.mean(list(six.values()))) if audits[i]['six_complete'] else np.nan,
            subtask_mean=vals.mean() if len(vals) else np.nan,subtask_std=vals.std(ddof=1) if len(vals)>1 else np.nan,
            weakest_subtask_P10=np.quantile(vals,.1) if len(vals) else np.nan,
            strongest_weakest_span=np.ptp(vals) if len(vals) else np.nan,
            capability_imbalance=float(np.quantile(vals,.9)-np.quantile(vals,.1)) if len(vals) else np.nan,
            bottom20_task_score=np.sort(vals)[:max(1,int(np.ceil(.2*len(vals))))].mean() if len(vals) else np.nan,
            MMLU_category_detail_available=False)
        for family,v in six.items():
            agg[family]=v;familyvals=np.array([x['task_score'] for x in sub if x['benchmark']==family])
            agg[family+'_subtask_mean']=familyvals.mean() if len(familyvals) else np.nan
            agg[family+'_subtask_count']=len(familyvals)
        aggregations.append(agg)
        if number%300==0:logging.info('C8 traversed %d/%d directories',number,len(directories))
    audit=pd.DataFrame(audits);scores=pd.DataFrame(leaves);agg=pd.DataFrame(aggregations)
    # Auxiliary within-subtask robust scaling controls task difficulty;
    # it does NOT redefine the six-task ability index or its parameters.
    scores['subtask_z']=np.nan;task_parameters=[]
    for (task,version),g in scores.groupby(['subtask','task_version']):
        x=g.task_score;lo,hi=x.quantile([.01,.99]);v=x.clip(lo,hi);med=v.median();iqr=v.quantile(.75)-v.quantile(.25)
        scores.loc[g.index,'subtask_z']=(v-med)/iqr if iqr>0 else 0
        task_parameters.append(dict(subtask=task,version=version,n=len(g),P01=lo,P99=hi,median=med,IQR=iqr))
    for directory,g in scores.groupby('directory'):
        z=g.subtask_z.to_numpy();index=agg.index[agg.directory==directory]
        agg.loc[index,'difficulty_adjusted_imbalance']=np.quantile(z,.9)-np.quantile(z,.1)
        agg.loc[index,'difficulty_adjusted_bottom20']=np.sort(z)[:max(1,int(np.ceil(.2*len(z))))].mean()
    csv(out/'C8_parse_audit.csv',audit);csv(out/'C8_subtask_scores.csv',scores);csv(out/'C8_subtask_transform_parameters.csv',task_parameters)
    models=pd.read_csv(out/'evolution_model_table.csv',float_precision='round_trip');keys={k:g for k,g in models.groupby('name_key')}
    names=list(keys);matches=[];matched=[]
    for r in agg.to_dict('records'):
        key=normalized_name(r['Model']);g=keys.get(key);status='unmatched';accepted=False;candidate=[]
        if g is not None:
            if len(g)==1:status='exact';accepted=True;model=g.iloc[0]
            else:status='ambiguous';candidate=g.Model.tolist()
        else:
            close=get_close_matches(key,names,n=3,cutoff=.9)
            candidate=close;status='high-confidence' if len(close)==1 else 'ambiguous' if len(close)>1 else 'unmatched'
        matches.append(dict(directory=r['directory'],C8_Model=r['Model'],status=status,accepted=accepted,
            match_method='exact normalized model_name/config identity' if accepted else 'fuzzy candidate only; never auto accepted',
            candidates=json.dumps(candidate),Model=model.Model if accepted else np.nan))
        if accepted:
            item=dict(directory=r['directory'],Model=model.Model,A_z=model.A_z,S1=model.S1,
                capability_imbalance=r['capability_imbalance'],bottom20_task_score=r['bottom20_task_score'],
                difficulty_adjusted_imbalance=r.get('difficulty_adjusted_imbalance',np.nan),
                difficulty_adjusted_bottom20=r.get('difficulty_adjusted_bottom20',np.nan),six_complete=r['six_complete'])
            for family in cfg['benchmarks']:item['C8_'+family]=r[family];item['C2_'+family]=model[family]
            matched.append(item)
    matches=pd.DataFrame(matches);matched=pd.DataFrame(matched)
    # A model appearing in several directories is not silently picked.
    duplicates=matched.Model.duplicated(keep=False)
    matched['unique_model_match']=~duplicates
    valid=matched[matched.unique_model_match&matched.six_complete&matched.A_z.notna()].copy()
    analyses=[]
    for scope in ['S0','S1']:
        h=valid if scope=='S0' else valid[valid.S1]
        if h.empty:continue
        aq=h.A_z.quantile(.75);tq=h.bottom20_task_score.quantile(.25);adjusted=h.difficulty_adjusted_bottom20.quantile(.25)
        flag=(h.A_z>=aq)&(h.bottom20_task_score<=tq);aflag=(h.A_z>=aq)&(h.difficulty_adjusted_bottom20<=adjusted)
        analyses.append(dict(sample=scope,n=len(h),A_P75=aq,bottom20_P25=tq,high_average_weak_tail_count=int(flag.sum()),
            high_average_weak_tail_rate=float(flag.mean()),rate_within_high_ability=float(flag.sum()/(h.A_z>=aq).sum()),
            difficulty_adjusted_weak_tail_count=int(aflag.sum()),difficulty_adjusted_weak_tail_rate=float(aflag.mean()),
            bottom20_zero_rate=float(h.bottom20_task_score.eq(0).mean()),
            Pearson_A_imbalance=corr(h.A_z,h.capability_imbalance),Spearman_A_imbalance=corr(h.A_z,h.capability_imbalance,'spearman'),
            Pearson_A_adjusted_imbalance=corr(h.A_z,h.difficulty_adjusted_imbalance)))
        matched.loc[h.index,'high_average_weak_tail_'+scope]=flag
    comparison=[]
    for family in cfg['benchmarks']:
        h=matched[matched.unique_model_match].dropna(subset=['C8_'+family,'C2_'+family]);diff=h['C8_'+family]-h['C2_'+family]
        comparison.append(dict(benchmark=family,n=len(h),MAE=diff.abs().mean(),systematic_bias=diff.mean(),
            Pearson=corr(h['C8_'+family],h['C2_'+family]),Spearman=corr(h['C8_'+family],h['C2_'+family],'spearman'),
            exact_within_1e_8=int((diff.abs()<1e-8).sum()),different_time_version_not_auto_corrected=True))
    csv(out/'C8_model_task_aggregation.csv',agg);csv(out/'C8_leaderboard_matches.csv',matches)
    csv(out/'C8_matched_model_analysis.csv',matched);csv(out/'C8_six_task_comparison.csv',comparison);csv(out/'C8_weak_tail_diagnosis.csv',analyses)
    attach=matched[matched.unique_model_match][['Model','capability_imbalance','bottom20_task_score','difficulty_adjusted_imbalance','difficulty_adjusted_bottom20']].rename(
        columns={'capability_imbalance':'C8_capability_imbalance','bottom20_task_score':'C8_bottom20_task_score'})
    models=models.drop(columns=['C8_capability_imbalance','C8_bottom20_task_score','difficulty_adjusted_imbalance','difficulty_adjusted_bottom20'],errors='ignore').merge(attach,on='Model',how='left',validate='one_to_one')
    models['C8_source']=np.where(models.C8_capability_imbalance.notna(),'latest parsable C8 JSON; exact normalized unique identity','unmatched/ambiguous')
    csv(out/'evolution_model_table.csv',models)
    summary=dict(directories=len(directories),JSON_files=len(audit),JSON_parse_success=int(audit.parse_success.sum()),
        JSON_parse_failures=int((~audit.parse_success).sum()),directories_no_usable_JSON=len(no_usable),
        no_usable_directories=no_usable,selected_models=len(agg),six_complete_models=int(agg.six_complete.sum()),
        exact_matches=int(matches.accepted.sum()),match_status_counts=matches.status.value_counts().to_dict(),
        valid_six_complete_matched_models=len(valid),leaf_score_records=len(scores),subtask_keys=int(scores.subtask.nunique()),
        fuzzy_matches_automatically_accepted=0,MMLU_category_detail_available=False,
        raw_vs_normalized_scores_separated=True,normalization_source=cfg['C8_normalization_reference'])
    dump(out/'C8_execution_summary.json',summary);logging.info('C8 complete: %s',summary)


def refresh_weak_tail(root,cfg):
    """Compare same-run C8 ability too, avoiding a snapshot-version confound.

    Reuses ONLY the new Phase 1 parsed task table, never old Q4 outputs.
    No need to reparse all JSONs to evaluate the fixed six-task transform.
    """
    from .ability import transform
    out=root/'outputs/evolution/phase1';matched=pd.read_csv(out/'C8_matched_model_analysis.csv',float_precision='round_trip')
    params=json.loads((out/'ability_transform_parameters.json').read_text(encoding='utf8'))
    tasks=pd.DataFrame({m:matched['C8_'+m] for m in cfg['benchmarks']})
    matched['A_z_C8']=transform(tasks,params).mean(axis=1,skipna=False)
    valid=matched[matched.unique_model_match&matched.six_complete&matched.A_z.notna()].copy();analyses=[]
    for scope in ['S0','S1']:
        for basis in ['A_z','A_z_C8']:
            h=valid if scope=='S0' else valid[valid.S1]
            if h.empty:continue
            aq=h[basis].quantile(.75);tq=h.bottom20_task_score.quantile(.25);adjusted=h.difficulty_adjusted_bottom20.quantile(.25)
            flag=(h[basis]>=aq)&(h.bottom20_task_score<=tq);aflag=(h[basis]>=aq)&(h.difficulty_adjusted_bottom20<=adjusted)
            analyses.append(dict(sample=scope,ability_basis='C2_snapshot' if basis=='A_z' else 'C8_same_run',n=len(h),A_P75=aq,bottom20_P25=tq,
                high_average_weak_tail_count=int(flag.sum()),high_average_weak_tail_rate=float(flag.mean()),
                rate_within_high_ability=float(flag.sum()/(h[basis]>=aq).sum()),
                difficulty_adjusted_weak_tail_count=int(aflag.sum()),difficulty_adjusted_weak_tail_rate=float(aflag.mean()),
                bottom20_zero_rate=float(h.bottom20_task_score.eq(0).mean()),Pearson_A_imbalance=corr(h[basis],h.capability_imbalance),
                Spearman_A_imbalance=corr(h[basis],h.capability_imbalance,'spearman'),
                Pearson_A_adjusted_imbalance=corr(h[basis],h.difficulty_adjusted_imbalance)))
            matched.loc[h.index,'high_average_weak_tail_'+scope+'_'+basis]=flag
    csv(out/'C8_weak_tail_diagnosis.csv',analyses);csv(out/'C8_matched_model_analysis.csv',matched)
    model=pd.read_csv(out/'evolution_model_table.csv',float_precision='round_trip')
    model=model.drop(columns='C8_A_z_same_run',errors='ignore').merge(matched[matched.unique_model_match][['Model','A_z_C8']].rename(
        columns={'A_z_C8':'C8_A_z_same_run'}),on='Model',how='left',validate='one_to_one')
    csv(out/'evolution_model_table.csv',model)
