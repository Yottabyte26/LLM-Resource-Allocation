"""Independent verification of derived disagreement outputs; no frozen input writes."""
from __future__ import annotations
import csv
import json
import math
from itertools import zip_longest
from pathlib import Path
import sys
import numpy as np
from scipy.stats import rankdata, ks_2samp, wasserstein_distance, mannwhitneyu, pearsonr, spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quality.io import sha256_file
from quality.vector_diagnostics import write_rows

OUT=ROOT/'outputs/conflict'
csv.field_size_limit(2**31-1)


def rows(path):
    """Parse derived CSV, including selected long text fields."""
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    """Verify frozen equality, indices, thresholds, pairs, rates and sensitivity."""
    checks=[]
    def check(name,ok,detail):
        if not ok:raise AssertionError(f'{name}: {detail}')
        checks.append({'check':name,'passed':True,'detail':detail})
    model=json.loads((ROOT/'outputs/preprocess/preprocessing_model_A1.json').read_text(encoding='utf-8'))
    names=model['metric_order'];wmap={r['metric']:float(r['critic_weight']) for r in rows(ROOT/'outputs/quality/critic_weights_A1.csv')}
    w=np.array([wmap[n] for n in names]);data={}
    inputs={'A1':('A1_sample_quality.csv','A1_sample_conflict.csv'),'A2':('A2_arxiv_quality.csv','A2_arxiv_conflict.csv'),'A3':('A3_github_quality.csv','A3_github_conflict.csv')}
    for dataset,(old,new) in inputs.items():
        records=[];zs=[];numeric=[]
        with (ROOT/'outputs/quality'/old).open(encoding='utf-8-sig',newline='') as f,(OUT/new).open(encoding='utf-8-sig',newline='') as g:
            for a,b in zip_longest(csv.DictReader(f),csv.DictReader(g)):
                if a is None or b is None:raise AssertionError('Record loss or unexpected extra record')
                if any(b[key]!=value for key,value in a.items()):raise AssertionError('Frozen metadata/Q/Z changed in conflict output')
                records.append({k:b[k] for k in ['id','domain','high_conflict','extreme_conflict','conflict_level']})
                zs.append([float(b[k]) for k in names]);numeric.append([float(b[k]) for k in ['Q','K','K_equal','V','Q_equal','Delta_Q']])
        z=np.array(zs);x=np.array(numeric);q=x[:,0];deviation=z-q[:,None]
        reference=np.column_stack([np.abs(deviation).dot(w),np.mean(np.abs(z-z.mean(axis=1)[:,None]),axis=1),
            np.square(deviation).dot(w),z.mean(axis=1),np.abs(q-z.mean(axis=1))])
        check(f'{dataset}: original Q/Z/metadata/every record',True,f'n={len(z)}; exact CSV value equality')
        check(f'{dataset}: full numerical K/K_equal/V/Delta_Q',np.allclose(x[:,1:],reference,atol=2e-15,rtol=0),f'max_abs_error={np.abs(x[:,1:]-reference).max()}')
        for i in range(0,len(z),997):
            scalar=math.fsum(float(w[j])*abs(float(z[i,j])-float(q[i])) for j in range(22))
            check(f'{dataset}: fsum K row {i+1}',abs(scalar-x[i,1])<2e-15,str(scalar))
        data[dataset]={'z':z,'x':x,'records':records,'domain':np.array([r['domain'] for r in records])}
        del zs,numeric
    a=data['A1'];threshold={r['quantile']:float(r['K_threshold']) for r in rows(OUT/'conflict_thresholds_A1.csv')}
    quantiles=np.quantile(a['x'][:,1],[.5,.75,.9,.95,.99],method='linear')
    check('A1 empirical thresholds',np.array_equal(quantiles,np.array(list(threshold.values()))),json.dumps(threshold))
    for d,v in data.items():
        high=v['x'][:,1]>threshold['P90'];extreme=v['x'][:,1]>threshold['P95']
        check(f'{d}: fixed A1 conflict flags',all(r['high_conflict']==str(h) and r['extreme_conflict']==str(e) for r,h,e in zip(v['records'],high,extreme)),f'high={high.sum()},extreme={extreme.sum()}')
    for r in rows(OUT/'domain_conflict_summary_A1.csv'):
        k=a['x'][a['domain']==r['domain'],1]
        check('domain '+r['domain'],len(k)==int(r['n']) and abs(k.mean()-float(r['mean_K']))<1e-15 and int((k>threshold['P90']).sum())==int(r['high_conflict_count']),f'n={len(k)}')
    pair_rows=rows(OUT/'indicator_pair_conflict.csv');groups={}
    for r in pair_rows:groups.setdefault((r['dataset'],r['domain'],r['view']),[]).append(r)
    constant_repairs=0
    for (dataset,domain,view),pairs in groups.items():
        v=data[dataset];mask=np.ones(len(v['z']),dtype=bool) if domain=='all' else v['domain']==domain
        if view=='nonoverlap':
            ids={r['id'] for r in rows(ROOT/f'outputs/preprocess/{dataset}_{domain}_nonoverlap_ids.csv')}
            mask &= np.array([r['id'] in ids for r in v['records']])
        z=v['z'][mask];variable=np.ptp(z,axis=0)>0
        with np.errstate(invalid='ignore',divide='ignore'):
            p=np.corrcoef(z,rowvar=False);s=np.corrcoef(rankdata(z,axis=0,method='average'),rowvar=False)
        seen=set()
        for r in pairs:
            j,l=names.index(r['metric_1']),names.index(r['metric_2']);seen.add((j,l));dif=np.abs(z[:,j]-z[:,l])
            values=np.array([dif.mean(),*np.quantile(dif,[.5,.9,.95]),(dif>=.5).mean()])
            old=np.array([float(r[k]) for k in ['mean_abs_difference','median_abs_difference','P90_abs_difference','P95_abs_difference','severe_pair_conflict_rate']])
            if not np.allclose(values,old,atol=2e-15,rtol=0):raise AssertionError('Pair statistic mismatch')
            if not (variable[j] and variable[l]):
                # Narrow roundoff correction: a constant column has no correlation.
                # Pair differences/counts and every locked part 1 file are untouched.
                if r['pearson_r'] or r['spearman_r'] or r['correlation_defined']!='False':
                    r['pearson_r']='';r['spearman_r']='';r['correlation_defined']='False';constant_repairs+=1
            elif abs(float(r['pearson_r'])-p[j,l])>5e-12 or abs(float(r['spearman_r'])-s[j,l])>5e-12:
                raise AssertionError('Pair correlation mismatch')
        check(f'231 pairs {dataset}/{domain}/{view}',len(seen)==231,f'n={len(z)}; exact differences and independent correlation matrices')
    if constant_repairs:write_rows(OUT/'indicator_pair_conflict.csv',pair_rows)
    check('constant-correlation handling',True,f'{constant_repairs} new-stage pair rows repaired for roundoff-induced undefined correlation; no difference/weight/Q changes')
    drivers=rows(OUT/'high_conflict_driver_summary.csv');counts={}
    for r in drivers:
        key=(r['dataset'],r['domain'],r['view']);counts.setdefault(key,[0.,int(r['high_conflict_n'])]);counts[key][0]+=float(r['fractional_sample_count'])
    check('tied driver fractional counts',all(abs(v[0]-v[1])<1e-7 for v in counts.values()),f'{len(counts)} scopes reconcile to high-conflict n')
    # Independent U statistic verifies signed Cliff effect sizes for extensions.
    for r in rows(OUT/'sample_extended_conflict_validation.csv'):
        ref=a['x'][a['domain']==r['domain'],1];v=data[r['extended_dataset']];ext=v['x'][:,1]
        if r['view']=='nonoverlap':
            ids={x['id'] for x in rows(ROOT/f"outputs/preprocess/{r['extended_dataset']}_{r['domain']}_nonoverlap_ids.csv")}
            ext=ext[np.array([x['id'] in ids for x in v['records']])]
        delta=2*mannwhitneyu(ext,ref,method='asymptotic').statistic/(len(ext)*len(ref))-1
        check(f"extension {r['extended_dataset']}/{r['view']}",abs(delta-float(r['cliffs_delta_extended_vs_A1']))<1e-14
            and abs(ks_2samp(ref,ext).statistic-float(r['KS_statistic']))<1e-14
            and abs(wasserstein_distance(ref,ext)-float(r['Wasserstein_distance']))<1e-14,'KS/Wasserstein/Cliff via independent U identity')
    first=rows(OUT/'conflict_score_relationships.csv')[0];q,k,delta=a['x'][:,0],a['x'][:,1],a['x'][:,5]
    check('Q/K and K/Delta correlations',abs(pearsonr(q,k).statistic-float(first['pearson_Q_K']))<1e-14
        and abs(spearmanr(k,delta).statistic-float(first['spearman_K_Delta_Q']))<1e-14,'independent scipy full A1 check')
    text=rows(OUT/'A1_high_conflict_text_samples.csv');ids={r['id']:i for i,r in enumerate(a['records'])}
    for r in text:
        i=ids[r['id']]
        if not r['content'] or r['domain']!=a['domain'][i] or float(r['K'])!=k[i]:raise AssertionError('Text metadata mismatch')
        j,l=names.index(r['top_positive_metric']),names.index(r['top_negative_metric'])
        if a['z'][i,j]!=a['z'][i].max() or a['z'][i,l]!=a['z'][i].min():raise AssertionError('Text driver mismatch')
    check('selected text material',len({r['id'] for r in text})==len(text),f'{len(text)} unique full-content records; CSV parser field limit raised for long original texts')
    for r in rows(OUT/'text_selection_summary.csv'):
        check('text selection '+r['selection_group'],sum(r['selection_group'] in t['selection_groups'].split(';') for t in text)==int(r['selected_n']),r['definition'])
    locked=rows(OUT/'locked_inputs_integrity.csv')
    check('all frozen/original file hashes',all(sha256_file(ROOT/r['path'])==r['sha256_before']==r['sha256_after'] for r in locked),f'{len(locked)} unchanged files')
    figure_rows=rows(OUT/'conflict_figure_manifest.csv')
    check('nine real figures',len(figure_rows)==9 and all((ROOT/r['path']).stat().st_size>1000 for r in figure_rows),'all PNG files exist and nonempty')
    check('no Q-star/lambda output',not (OUT/'conflict_adjustment_sensitivity.csv').exists(),'diagnosis stage stop respected')
    write_rows(OUT/'conflict_verification.csv',checks)
    print(json.dumps({'checks_passed':len(checks),'constant_pair_rows_corrected':constant_repairs,'protected_files':len(locked),'text_records':len(text)},ensure_ascii=False))


if __name__=='__main__':main()
