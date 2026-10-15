"""Independent checks of Phase 2B numbers, group splits, bootstrap and provenance.

This verifier reads raw B6/B7/B8 CSVs and saved predictions. It uses scalar
minimization and direct formulas instead of the pipeline's least-squares fit,
and never modifies existing results or original attachments.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import csv
import json
import hashlib
from collections import defaultdict
from decimal import Decimal
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata
from PIL import Image

OUT=ROOT/'outputs/scaling/phase2'
CHECKS=[]


def read(name):
    """Reject malformed CSVs rather than silently dropping rows."""
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f);rows=list(rd)
    if not rd.fieldnames or any(None in r or any(v is None for v in r.values()) for r in rows):raise ValueError(name)
    return rows


def check(name,condition):
    """Fail visibly on numerical or scope deviations."""
    CHECKS.append(dict(check=name,passed=bool(condition)))
    if not condition:raise AssertionError(name)


def close(name,a,b,tol=1e-11):
    check(name,np.allclose(a,b,atol=tol,rtol=tol))


def scores(y,yp):
    """Direct independent metrics with average ranks for ties."""
    y=np.asarray(y,dtype=float);yp=np.asarray(yp,dtype=float);r=y-yp
    return dict(RMSE=float(np.sqrt(np.mean(r*r))),MAE=float(np.mean(abs(r))),
                R2=float(1-np.sum(r*r)/np.sum((y-y.mean())**2)),
                Pearson=float(np.corrcoef(y,yp)[0,1]) if np.ptp(yp) else None,
                Spearman=float(np.corrcoef(rankdata(y),rankdata(yp))[0,1]) if np.ptp(yp) else None,
                mean_bias=float(np.mean(r)))


def formula(n,d,q,p,model,gamma=0,offset=0):
    """Algebraic D_eff form, independently of the fitted module's expression."""
    deff=d if model=='M0' else d*q**gamma if model.startswith('MQ1') else d*np.exp(gamma*(q-1))
    return p['E']+p['A']*n**(-p['alpha'])+p['B']*deff**(-p['beta'])+(offset if model.endswith('_offset') else 0.)


def main():
    """Cross-check consequential results and all prior frozen file hashes."""
    summary=json.loads((OUT/'phase2B_execution_summary.json').read_text(encoding='utf-8'))
    candidate=json.loads((OUT/'parameters_before_B7_B8.json').read_text(encoding='utf-8'))
    p=summary['classical_parameters'];fits=summary['fit_models'];models=['M0','MQ1','MQ2','MQ1_offset','MQ2_offset']
    records={};arrays={};keys={}
    for label,name in [('B6','supplementary_NQ_experiment.csv'),('B7','supplementary_NQ_experiment_expanded.csv'),('B8','supplementary_NQ_experiment_large.csv')]:
        with (ROOT/'real_attachments/B_scaling_laws'/name).open(encoding='utf-8-sig') as f:rs=list(csv.DictReader(f))
        records[label]=rs
        arrays[label]=np.array([[float(r[c]) for c in ['N_params_B','D_tokens_B','Q_score','val_loss']] for r in rs])
        keys[label]=[tuple(Decimal(r[c]) for c in ['N_params_B','D_tokens_B','Q_score']) for r in rs]
    with (ROOT/'outputs/scaling/phase1/classical_scaling_parameters.csv').open(encoding='utf-8-sig') as f:pr=list(csv.DictReader(f))
    primary=next(r for r in pr if r['weighting']=='trajectory_balanced' and r['model']=='M0_N_D')
    for k in p:check('Frozen coefficient exact '+k,p[k]==float(primary[k]))
    close('Snapshot classical parameters',list(p.values()),[candidate['classical_parameters'][k] for k in p])
    check('No validation source in fitting',candidate['fit_dataset']=='B6_only' and candidate['selection_dataset']=='B6_5fold_ND_GroupCV_only')
    for m in fits:
        close('Saved pre-validation gamma '+m,fits[m]['gamma'],candidate['fitted_quality_models'][m]['gamma'])
        check('Positive gamma '+m,fits[m]['gamma']>0)
    n,d,q,y=arrays['B6'].T
    for model in ['MQ1','MQ2']:
        opt=minimize_scalar(lambda g:np.mean((formula(n,d,q,p,model,g)-y)**2),bounds=(1e-8,50),method='bounded',options={'xatol':1e-12})
        close('Independent scalar fit gamma '+model,opt.x,fits[model]['gamma'],tol=2e-6)
        close('Independent objective '+model,opt.fun,fits[model]['objective'])
        close('Q1 exact anchor identity '+model,formula(n,d,np.ones(len(q)),p,model,fits[model]['gamma']),formula(n,d,np.ones(len(q)),p,'M0'))
    predictions={}
    for label in records:
        rs=read(label+'_fixed_quality_predictions.csv');check(label+' no dropped source rows',len(rs)==len(records[label]))
        a=arrays[label];n,d,q,y=a.T;predictions[label]={}
        for model in models:
            f=fits.get(model,dict(gamma=0.,offset=0.))
            yp=formula(n,d,q,p,model,f['gamma'],f['offset'])
            saved=np.array([float(r[model+'_predicted_Loss']) for r in rs]);predictions[label][model]=saved
            close(label+' fixed formula '+model,yp,saved)
        for i,(raw,saved) in enumerate(zip(records[label],rs)):
            check(label+' source retained row '+str(i),all(saved[k]==v for k,v in raw.items()))
    oofrows=read('group_cv_predictions.csv');folds=read('group_cv_folds.csv');fitcv=read('group_cv_model_parameters.csv')
    oof={m:np.full(len(arrays['B6']),np.nan) for m in models}
    group_folds=defaultdict(set)
    for row in folds:
        tr=set(json.loads(row['train_groups']));te=set(json.loads(row['test_groups']))
        check('CV no ND leakage fold '+row['fold'],not tr&te and len(tr|te)==45)
    for r in oofrows:
        idx=int(r['row_index']);fold=r['fold'];m=r['model'];group_folds[r['ND_group']].add(fold)
        f=next((v for v in fitcv if v['model']==m and v['fold']==fold),dict(gamma=0.,offset=0.))
        n,d,q,y=arrays['B6'][idx]
        yp=formula(n,d,q,p,m,float(f['gamma']),float(f['offset']))
        close('OOF formula row '+str(idx)+' '+m,yp,float(r['predicted_Loss']))
        oof[m][idx]=float(r['predicted_Loss'])
    check('Every ND wholly held out',all(len(v)==1 for v in group_folds.values()) and len(group_folds)==45)
    y=arrays['B6'][:,3]
    for r in read('quality_model_comparison.csv'):
        vals=scores(y,oof[r['model']])
        for k,v in vals.items():
            if v is not None:close('CV metric '+r['model']+' '+k,float(r[k]),v)
    key6=set(keys['B6']);key7=set(keys['B7'])
    b7over=np.array([k in key6 for k in keys['B7']]);masks7={'full':np.ones(len(b7over),dtype=bool),'overlap':b7over,'nonoverlap':~b7over}
    check('B7 overlap count',int(b7over.sum())==360);check('B7 new count',int((~b7over).sum())==90)
    for r in read('B7_validation.csv'):
        ix=masks7[r['subset']];v=scores(arrays['B7'][ix,3],predictions['B7'][r['model']][ix])
        for k,a in v.items():
            if a is not None:close('B7 metric '+r['subset']+' '+r['model']+' '+k,float(r[k]),a)
    points=read('within_ND_quality_difference_points.csv')
    for r in points:
        label=r['dataset'];m=r['model'];i=int(r['row_index']);source=arrays[label]
        nd=source[i,:2];members=np.flatnonzero((source[:,:2]==nd).all(axis=1));ref=members[np.argmax(source[members,2])]
        check('Delta reference Q1 '+label+' '+str(i),source[ref,2]==1)
        obs=source[i,3]-source[ref,3]
        v=oof[m] if r['evaluation']=='B6_GroupCV_out_of_fold' else predictions[label][m]
        close('Delta actual '+label+' '+str(i)+' '+m,obs,float(r['Delta_L_obs']))
        close('Delta predicted '+label+' '+str(i)+' '+m,v[i]-v[ref],float(r['Delta_L_pred']))
    for r in read('within_ND_quality_difference_validation.csv'):
        if r['include_reference_zero_rows']=='True':continue
        ix=[v for v in points if v['dataset']==r['dataset'] and v['model']==r['model'] and v['subset']==r['subset'] and v['evaluation']==r['evaluation']]
        check('Delta metric count '+r['dataset']+' '+r['model']+' '+r['evaluation'],len(ix)==int(r['n']))
        a=scores([float(v['Delta_L_obs']) for v in ix],[float(v['Delta_L_pred']) for v in ix])
        for k,v in a.items():
            if v is not None:close('Delta metric '+r['dataset']+' '+r['model']+' '+r['evaluation']+' '+k,float(r[k]),v)
    draws=read('bootstrap_block_draws.csv');reps=read('quality_bootstrap_replicates.csv');bst=read('quality_parameter_bootstrap.csv')
    check('2000 ND bootstrap draws',len(draws)==2000)
    for r in draws:
        c=np.array(json.loads(r['multiplicities']));groups=json.loads(r['ND_groups'])
        check('Whole ND draw '+r['replicate'],len(c)==45 and c.sum()==45 and len(groups)==45 and np.all(c>=0))
    for m in fits:check('2000 fits '+m,sum(r['model']==m for r in reps)==2000)
    for r in bst:
        v=np.array([float(x[r['parameter']]) for x in reps if x['model']==r['model'] and x['success']=='True'])
        values=[v.mean(),np.median(v),v.std(ddof=1),np.quantile(v,.025),np.quantile(v,.975)]
        close('Bootstrap summaries '+r['model']+' '+r['parameter'],values,[float(r[k]) for k in ['mean','median','std','P025','P975']])
    for r in reps:
        if r['success']=='True' and r['model'].startswith('MQ1'):close('Eta identity rep '+r['replicate']+' '+r['model'],float(r['eta']),p['beta']*float(r['gamma']))
    starts=read('bootstrap_multistart_solutions.csv');check('All seven bootstrap starts',len(starts)==2000*4*7)
    check('Seven prescribed initial values',set(float(r['initial_gamma']) for r in starts)=={.1,.25,.5,1.,2.,4.,8.})
    startgroups=defaultdict(list)
    for r in starts:startgroups[(r['replicate'],r['model'])].append(r)
    for r in reps:
        sol=startgroups[(r['replicate'],r['model'])]
        check('Seven independent solves '+r['replicate']+' '+r['model'],len(sol)==7)
        legal=[x for x in sol if x['legal_solution']=='True']
        if r['success']=='True':
            best=min(legal,key=lambda x:float(x['objective']))
            close('Best multistart objective '+r['replicate']+' '+r['model'],float(r['objective']),float(best['objective']))
            close('Best multistart gamma '+r['replicate']+' '+r['model'],float(r['gamma']),float(best['gamma']))
    # B8 numerical-range subset is evaluated without dropping low-Loss observations.
    b8=arrays['B8'];b6=arrays['B6'];inside=((b8[:,:3]>=b6[:,:3].min(axis=0))&(b8[:,:3]<=b6[:,:3].max(axis=0))).all(axis=1)
    membership=read('B8_range_membership.csv')
    close('B8 numeric ranges membership',inside.astype(int),[int(r['within_B6_numeric_ranges']=='True') for r in membership])
    check('B8 low Loss retained',sum(b8[:,3]<p['E'])==1095 and len(membership)==1704)
    masks8={'full':np.ones(len(b8),dtype=bool),'nonoverlap':np.array([k not in key6|key7 for k in keys['B8']]),
            'within_B6_numeric_ranges':inside,'outside_B6_numeric_ranges':~inside}
    masks8['nonoverlap_within_ranges']=inside&masks8['nonoverlap'];masks8['nonoverlap_outside_ranges']=~inside&masks8['nonoverlap']
    for r in read('B8_range_validation.csv'):
        ix=masks8[r['subset']];v=scores(b8[ix,3],predictions['B8'][r['model']][ix])
        check('B8 subset count '+r['subset'],int(ix.sum())==int(r['n']))
        for k,a in v.items():
            if a is not None:close('B8 metric '+r['subset']+' '+r['model']+' '+k,float(r[k]),a)
    for r in read('B8_incompatibility_diagnostics.csv'):
        ix=masks8[r['subset']];groups=defaultdict(list)
        for row in b8[ix]:groups[tuple(row[:2])].append(row[2:])
        valid=[]
        for vals in groups.values():
            a=np.array(vals)
            if len(a)>=2 and np.ptp(a[:,1]):valid.append(np.corrcoef(rankdata(a[:,0]),rankdata(a[:,1]))[0,1])
        close('B8 positive groups '+r['subset'],np.mean(np.array(valid)>0),float(r['positive_group_rate']))
    for path in (ROOT/'outputs/figures/question2_phase2').glob('*.png'):
        with Image.open(path) as im:im.verify()
        check('Figure decodes '+path.name,True)
    snapshot=json.loads((OUT/'frozen_before_phase2B.json').read_text(encoding='utf-8'))
    for r in snapshot['files']:
        pth=ROOT/r['path'];h=hashlib.sha256()
        with pth.open('rb') as f:
            for b in iter(lambda:f.read(1048576),b''):h.update(b)
        check('Frozen '+r['path'],pth.stat().st_size==r['bytes'] and h.hexdigest()==r['sha256'])
    check('No premature human freeze',summary['actual_human_freeze'] is False)
    with (OUT/'phase2B_independent_verification.csv').open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=['check','passed']);wr.writeheader();wr.writerows(CHECKS)
    print('All independent checks passed:',len(CHECKS),'Protected files:',len(snapshot['files']))


if __name__=='__main__':main()
