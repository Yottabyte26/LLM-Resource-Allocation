"""Independent scalar/streaming checks of PHASE 5B; never recalculate Q or K."""
from __future__ import annotations
import csv
import json
import math
from itertools import zip_longest
from pathlib import Path
import sys
import numpy as np
import yaml
from scipy.stats import pearsonr,spearmanr,rankdata

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quality.io import sha256_file
from quality.vector_diagnostics import write_rows
csv.field_size_limit(2**31-1)
OUT=ROOT/'outputs/conflict/phase5b'


def rows(path):
    """Read small derived tables or the already-selected text material."""
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def scalar(z,q,w,delta,epsilon):
    """Independent math.fsum implementation of the user's one-step formula."""
    a=[min(1.,delta/(abs(v-q)+epsilon)) for v in z]
    raw=[wi*ai for wi,ai in zip(w,a)];denominator=math.fsum(raw)
    local=[v/denominator for v in raw]
    return math.fsum(wi*zi for wi,zi in zip(local,z)),a,local


def main():
    """Audit every candidate, five A1 settings, selected cases and frozen hashes."""
    settings=yaml.safe_load((ROOT/'configs/robust_reaggregation.yaml').read_text(encoding='utf-8'))
    policy=json.loads((OUT/'model_recommendation.json').read_text(encoding='utf-8'));selected=policy['recommended_delta_for_validation']
    names=json.loads((ROOT/'outputs/preprocess/preprocessing_model_A1.json').read_text(encoding='utf-8'))['metric_order']
    wmap={r['metric']:float(r['critic_weight']) for r in rows(ROOT/'outputs/quality/critic_weights_A1.csv')};w=[wmap[n] for n in names]
    thresholds={r['quantile']:float(r['K_threshold']) for r in rows(ROOT/'outputs/conflict/conflict_thresholds_A1.csv')}
    check_rows=[]
    def check(name,condition,detail):
        if not condition:raise AssertionError(f'{name}: {detail}')
        check_rows.append({'check':name,'passed':True,'detail':detail})
    all_data={};a1z=[];a1meta=[]
    files={'A1':'A1_sample_conflict.csv','A2':'A2_arxiv_conflict.csv','A3':'A3_github_conflict.csv'}
    for dataset,filename in files.items():
        numbers=[];domains=[];maxerror=0.;ordinary_n=0;equal_distances=[];reduced_equal_count=0
        with (ROOT/'outputs/conflict'/filename).open(encoding='utf-8-sig',newline='') as f,(OUT/f'{dataset}_robust_candidate.csv').open(encoding='utf-8-sig',newline='') as g:
            for original,candidate in zip_longest(csv.DictReader(f),csv.DictReader(g)):
                if original is None or candidate is None:raise AssertionError('Record count differs')
                if any(original[k]!=candidate[k] for k in ['id','dataset','domain','sub_path','source_file','line_number','imputed_count','Q','K']):
                    raise AssertionError('Original record identity or Q/K changed')
                z=[float(original[n]) for n in names];q=float(original['Q']);k=float(original['K']);qr=float(candidate['Q_robust_candidate']);high=k>thresholds['P90']
                expected=scalar(z,q,w,selected,settings['epsilon'])[0] if high else q
                maxerror=max(maxerror,abs(qr-expected))
                if not high:
                    ordinary_n+=1
                    if qr!=q or candidate['change']!='0.0':raise AssertionError('Ordinary Q changed')
                if candidate['high_conflict']!=str(high) or float(candidate['delta'])!=selected or candidate['final_policy']!='B_Q_plus_K; candidate_not_activated':
                    raise AssertionError('Scope, parameter or candidate activation mismatch')
                if abs(float(candidate['change'])-(qr-q))>1e-15:raise AssertionError('Score change mismatch')
                if not 0<=qr<=1:raise AssertionError('Nonconvex/nonfinite candidate')
                if high:
                    equal_candidate=scalar(z,float(original['Q_equal']),[1/22]*22,selected,settings['epsilon'])[0]
                    distance=abs(qr-equal_candidate);equal_distances.append(distance)
                    reduced_equal_count+=distance<float(original['Delta_Q'])
                numbers.append([q,k,qr,float(original['Q_equal']),float(original['Delta_Q'])]);domains.append(original['domain'])
                if dataset=='A1':a1z.append(z);a1meta.append(original['id'])
        x=np.array(numbers);all_data[dataset]={'x':x,'domains':np.array(domains),
            'mean_equal_candidate_distance':math.fsum(equal_distances)/len(equal_distances),
            'reduced_equal_rate':reduced_equal_count/len(equal_distances)}
        check(dataset+' every record scalar formula',maxerror<2e-15,f'n={len(x)},max_abs_error={maxerror}')
        check(dataset+' ordinary unchanged',True,f'{ordinary_n} rows exact Q copy; original id/Q/K/metadata equality')
    a=all_data['A1']['x'];z=np.array(a1z);q,k=a[:,0],a[:,1];high=k>thresholds['P90'];extreme=k>thresholds['P95'];tol=settings['direction_equality_tolerance']
    sensitivity=rows(OUT/'delta_sensitivity_A1.csv')
    check('five user deltas',set(float(r['delta']) for r in sensitivity)==set(settings['deltas']) and len(sensitivity)==5,'No additional deltas or extension retuning')
    for r in sensitivity:
        delta=float(r['delta']);candidate=q.copy()
        for i in np.flatnonzero(high):candidate[i]=scalar(z[i],q[i],w,delta,settings['epsilon'])[0]
        change=candidate-q
        check(f'A1 delta={delta} full Pearson/Spearman',abs(pearsonr(q,candidate).statistic-float(r['pearson_Q_Qrobust']))<1e-13
            and abs(spearmanr(q,candidate).statistic-float(r['spearman_Q_Qrobust']))<1e-13,'independent scalar candidates and scipy ranks')
        check(f'A1 delta={delta} movement and directions',abs(abs(change[high]).mean()-float(r['mean_abs_change_high']))<1e-14
            and abs(abs(change[extreme]).mean()-float(r['mean_abs_change_P95_conflict']))<1e-14
            and abs(abs(change).max()-float(r['max_abs_change_all']))<1e-14
            and int((change>tol).sum())==int(r['up_count_high']) and int((change<-tol).sum())==int(r['down_count_high']),
            'high/extreme mean changes, max change and signs match')
    extension=rows(OUT/'extension_validation.csv')
    for r in extension:
        d=all_data[r['dataset']];mask=np.ones(len(d['x']),dtype=bool)
        if r['view']=='nonoverlap':
            # Frozen nonoverlap indices identify physical row numbers; candidates keep row order.
            indexes=rows(ROOT/f"outputs/preprocess/{r['dataset']}_{r['domain']}_nonoverlap_ids.csv")
            mask[:]=False
            for index in indexes:mask[int(index['line_number'])-1]=True
        x=d['x'][mask];h=x[:,1]>thresholds['P90'];absolute=np.abs(x[:,2]-x[:,0])
        check(r['dataset']+'/'+r['view']+' extension ranks and movement',len(x)==int(r['n']) and int(h.sum())==int(r['n_high'])
            and abs(spearmanr(x[:,0],x[:,2]).statistic-float(r['spearman_Q_Qrobust']))<1e-13
            and abs(absolute[h].mean()-float(r['mean_abs_change_high']))<1e-14,'fixed A1 delta, exact saved nonoverlap row index')
    for r in rows(OUT/'equal_weight_sensitivity_diagnostic.csv'):
        d=all_data[r['dataset']];x=d['x'];check(r['dataset']+' original equal sensitivity',abs(x[x[:,1]>thresholds['P90'],4].mean()-float(r['mean_original_Delta_Q_high']))<1e-15,'uses saved first-stage Delta_Q')
        check(r['dataset']+' robust equal sensitivity',abs(d['mean_equal_candidate_distance']-float(r['mean_candidate_Delta_Q_high']))<2e-15
            and abs(d['reduced_equal_rate']-float(r['candidate_difference_reduced_rate_high']))<1e-15,
            'independent scalar equal candidate around saved Q_equal, same high mask and delta')
    # Full independent audit of all selected local weights and preserved texts.
    text={r['id']:r for r in rows(ROOT/'outputs/conflict/A1_high_conflict_text_samples.csv')};idindex={v:i for i,v in enumerate(a1meta)}
    cases=rows(OUT/'typical_cases.csv')
    for r in cases:
        i=idindex[r['id']];zz=[float(r[n]) for n in names]
        expected,aa,ll=scalar(zz,q[i],w,selected,settings['epsilon']) if high[i] else (q[i],[1.]*22,w)
        saved_a=json.loads(r['a_coefficients_json']);saved_w=json.loads(r['local_weights_json'])
        check('case '+r['case_group']+'/'+r['id'],np.array_equal(np.array(zz),z[i]) and abs(expected-float(r['Q_robust']))<2e-15
            and max(abs(saved_w[n]-v) for n,v in zip(names,ll))<1e-15
            and max(abs(saved_a[n]-v) for n,v in zip(names,aa))<1e-15
            and r['content']==text[r['id']]['content'],'original 22Z and content; scalar score and all local weights/coefficients')
    for r in rows(OUT/'case_selection_summary.csv'):
        count=sum(t['case_group']==r['group'] for t in cases)
        check('case count '+r['group'],count==int(r['selected_n'])==int(r['requested_n']),f'{count} actual selected records')
    check('policy B inactive candidate',policy['final_recommendation']=='B_Q_plus_K' and not policy['candidate_activated'] and not policy['exponential_penalty_tested'],'Q/K remain final; no exponential penalty')
    locked=rows(OUT/'locked_inputs_integrity.csv')
    check('frozen file hashes',all(sha256_file(ROOT/r['path'])==r['sha256_before']==r['sha256_after'] for r in locked),f'{len(locked)} frozen files unchanged')
    figures=rows(OUT/'figure_manifest.csv')
    check('four figure outputs',len(figures)==4 and all((ROOT/r['path']).stat().st_size>1000 for r in figures),'actual PNGs present')
    write_rows(OUT/'verification.csv',check_rows)
    print(json.dumps({'checks_passed':len(check_rows),'protected_files':len(locked),'selected_case_rows':len(cases)},ensure_ascii=False))


if __name__=='__main__':main()
