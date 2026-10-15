"""Independent read-only numerical/provenance checks of Phase 2C artifacts."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import csv,json,hashlib
from collections import defaultdict
from decimal import Decimal
import numpy as np
from scipy.stats import rankdata
from PIL import Image

OUT=ROOT/'outputs/scaling/phase2C';CHECKS=[]


def read(name):
    """Strict rectangular CSV, never silently trim bad records."""
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f);rows=list(rd)
    if not rd.fieldnames or any(None in r or any(v is None for v in r.values()) for r in rows):raise ValueError(name)
    return rows


def check(name,condition):
    CHECKS.append(dict(check=name,passed=bool(condition)))
    if not condition:raise AssertionError(name)


def close(name,a,b,tol=1e-10):
    check(name,np.allclose(a,b,atol=tol,rtol=tol))


def formula(a,p,ref,model,params):
    """Direct algebra, independently of the implementation's expm1 prediction."""
    n,d,q=a[:,:3].T;l0=p['E']+p['A']*n**(-p['alpha'])+p['B']*d**(-p['beta'])
    if model=='M0':return l0
    scale=(n/ref['N0'])**(-params['u'])*(d/ref['D0'])**params['v']
    if model=='S1':
        effective=d*q**(params['gamma0']*scale)
        return p['E']+p['A']*n**(-p['alpha'])+p['B']*effective**(-p['beta'])
    if model=='S2':return l0+params['C']*scale*(q**(-params['eta'])-1)
    return l0+params['C']*scale*(-np.log(q))


def scores(y,yp):
    y=np.array(y,dtype=float);yp=np.array(yp,dtype=float);r=y-yp
    return dict(RMSE=np.sqrt(np.mean(r*r)),MAE=np.mean(abs(r)),R2=1-np.sum(r*r)/np.sum((y-y.mean())**2),
                Pearson=np.corrcoef(y,yp)[0,1] if np.ptp(yp)>0 else None,
                Spearman=np.corrcoef(rankdata(y),rankdata(yp))[0,1] if np.ptp(yp)>0 else None,
                mean_bias=np.mean(r))


def deltas(a,yp):
    """Subtract observed Q=1 by exact ND; exclude reference rows from metrics."""
    groups=defaultdict(list)
    for i,r in enumerate(a):groups[tuple(r[:2])].append(i)
    obs=np.zeros(len(a));pred=np.zeros(len(a));active=a[:,2]!=1
    for idx in groups.values():
        refs=[i for i in idx if a[i,2]==1]
        if len(refs)!=1:raise ValueError('No unique Q1 reference')
        j=refs[0];obs[idx]=a[idx,3]-a[j,3];pred[idx]=yp[idx]-yp[j]
    return obs,pred,active


def main():
    """Verify numerical effect estimates without refitting any frozen model."""
    s=json.loads((OUT/'execution_summary.json').read_text(encoding='utf-8'));p=s['classical_parameters'];ref=s['reference_scales']
    data={};raw={};keys={}
    for label,name in [('B6','supplementary_NQ_experiment.csv'),('B7','supplementary_NQ_experiment_expanded.csv'),('B8','supplementary_NQ_experiment_large.csv')]:
        with (ROOT/'real_attachments/B_scaling_laws'/name).open(encoding='utf-8-sig') as f:rs=list(csv.DictReader(f))
        raw[label]=rs;data[label]=np.array([[float(r[c]) for c in ['N_params_B','D_tokens_B','Q_score','val_loss']] for r in rs])
        keys[label]=[tuple(Decimal(r[c]) for c in ['N_params_B','D_tokens_B','Q_score']) for r in rs]
    with (ROOT/'outputs/scaling/phase1/classical_scaling_parameters.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    frozen=next(r for r in rows if r['model']=='M0_N_D' and r['weighting']=='trajectory_balanced')
    for k in p:check('Frozen classical coefficient '+k,p[k]==float(frozen[k]))
    for j,k in enumerate(['N0','D0']):close('Reference median '+k,ref[k],np.exp(np.median(np.log(np.unique(data['B6'][:,j])))))
    pred={}
    for label,a in data.items():
        saved=read(label+'_predictions.csv');check(label+' all source rows retained',len(saved)==len(raw[label]));pred[label]={}
        for i,(r,old) in enumerate(zip(saved,raw[label])):check(label+' unchanged raw observation '+str(i),all(r[k]==v for k,v in old.items()))
        for m in ['M0','S1','S2','S3']:
            params={} if m=='M0' else s['fitted'][m]['parameters'];yp=formula(a,p,ref,m,params)
            v=np.array([float(r[m+'_prediction']) for r in saved]);close(label+' direct formula '+m,yp,v);pred[label][m]=v
    for m,f in s['fitted'].items():
        a=data['B6'].copy();a[:,2]=1
        close(m+' Q1 anchor identity',formula(a,p,ref,m,f['parameters']),formula(a,p,ref,'M0',{}))
        check(m+' 50 full-data starts',f['n_starts']==50 and f['n_successful_starts']==50)
    oofrows=read('group_cv_predictions.csv');cvfits=read('group_cv_model_parameters.csv');folds=read('group_cv_folds.csv')
    oof={m:np.array([float(r[m+'_prediction']) for r in oofrows]) for m in ['M0','S1','S2','S3']}
    gf=defaultdict(set)
    for row in folds:
        tr=set(json.loads(row['train_groups']));te=set(json.loads(row['test_groups']))
        check('ND leakage absent fold '+row['fold'],not tr&te and len(tr|te)==45)
    for r in oofrows:
        i=int(r['row_index']);fold=r['fold'];gf[r['ND_group']].add(fold)
        for m in ['S1','S2','S3']:
            f=next(x for x in cvfits if x['model']==m and x['fold']==fold)
            close('OOF formula '+m+' '+str(i),formula(data['B6'][i:i+1],p,ref,m,json.loads(f['parameters']))[0],float(r[m+'_prediction']))
            check('OOF 50 independent starts '+m+' '+str(i),int(f['n_starts'])==50)
    check('ND groups appear in one test fold',len(gf)==45 and all(len(v)==1 for v in gf.values()))
    nonover=np.array([k not in set(keys['B6']) for k in keys['B7']]);check('90 new B7 points',nonover.sum()==90)
    for r in s['B6_CV']+s['B7_nonoverlap']:
        label='B6' if r['stage'].startswith('B6') else 'B7';a=data[label];v=oof[r['model']] if label=='B6' else pred[label][r['model']]
        mask=np.ones(len(a),dtype=bool) if label=='B6' else nonover
        if r['target']=='absolute_Loss':y=a[:,3]
        else:y,v,active=deltas(a,v);mask&=active
        z=scores(y[mask],v[mask]);check('Metric population '+label+' '+r['model']+' '+r['target'],mask.sum()==r['n'])
        for k,val in z.items():
            if val is not None:close('Independent metric '+label+' '+r['model']+' '+r['target']+' '+k,val,r[k])
    _remaining_checks(s,data,raw,keys,p,ref,pred)


def _remaining_checks(s,data,raw,keys,p,ref,pred):
    """Audit 300k solve records in a stream and validate CI/gradient populations."""
    draws=read('bootstrap_block_draws.csv');reps=read('bootstrap_replicates.csv');summaries=read('bootstrap_parameters.csv')
    check('2000 ND bootstrap draws',len(draws)==2000);check('6000 model bootstrap fits',len(reps)==6000)
    rng=np.random.default_rng(2026)
    for r in draws:
        c=np.array(json.loads(r['multiplicities']));expected=rng.multinomial(45,np.full(45,1/45))
        check('Reproducible complete block draw '+r['replicate'],len(c)==45 and c.sum()==45 and np.array_equal(c,expected))
    for r in summaries:
        vals=np.array([json.loads(x['parameters'])[r['parameter']] for x in reps if x['model']==r['model'] and x['success']=='True'])
        computed=[vals.mean(),np.median(vals),vals.std(ddof=1),np.quantile(vals,.025),np.quantile(vals,.975)]
        close('Bootstrap summary '+r['model']+' '+r['parameter'],computed,[float(r[k]) for k in ['mean','median','std','P025','P975']])
    for r in read('parameter_correlations.csv'):
        valid=[x for x in reps if x['model']==r['model'] and x['success']=='True']
        a=np.array([json.loads(x['parameters'])[r['parameter_1']] for x in valid]);b=np.array([json.loads(x['parameters'])[r['parameter_2']] for x in valid])
        fixed_a=all(r['parameter_1'] in json.loads(x['boundary_parameters']) for x in valid) and np.ptp(a)<=1e-6
        fixed_b=all(r['parameter_2'] in json.loads(x['boundary_parameters']) for x in valid) and np.ptp(b)<=1e-6
        name=r['model']+' '+r['parameter_1']+' '+r['parameter_2']
        if fixed_a or fixed_b:check('Boundary covariance not fabricated '+name,r['defined']=='False' and r['Pearson']=='' and r['Spearman']=='')
        elif r['defined']=='True':
            close('Parameter Pearson '+name,np.corrcoef(a,b)[0,1],float(r['Pearson']))
            close('Parameter Spearman '+name,np.corrcoef(rankdata(a),rankdata(b))[0,1],float(r['Spearman']))
    counts=defaultdict(int);best={};nsolves=0
    with (OUT/'bootstrap_multistart_solutions.csv').open(encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f)
        for r in rd:
            if None in r or any(v is None for v in r.values()):raise ValueError('Truncated bootstrap start CSV')
            key=(int(r['replicate']),r['model']);counts[key]+=1;nsolves+=1
            if r['legal_solution']=='True':
                if key not in best or float(r['objective'])<float(best[key]['objective']):best[key]=r
    check('300000 recorded bootstrap starts',nsolves==2000*3*50)
    for r in reps:
        key=(int(r['replicate']),r['model']);check('Exactly 50 solves '+str(key),counts[key]==50)
        if r['success']=='True':
            close('Minimum objective selected '+str(key),float(r['objective']),float(best[key]['objective']))
            a=json.loads(r['parameters']);b=json.loads(best[key]['parameters'])
            close('Best parameters selected '+str(key),list(a.values()),[b[k] for k in a])
    # Independently differentiate direct formulas; S1 D effect includes D^-beta.
    surfaces=read('quality_sensitivity_surface.csv')
    for r in surfaces:
        m=r['model'];n,d,q=[float(r[k]) for k in ['N_params_B','D_tokens_B','Q_score']];a=s['fitted'][m]['parameters']
        scale=(n/ref['N0'])**(-a['u'])*(d/ref['D0'])**a['v']
        if m=='S1':
            g=a['gamma0']*scale;grad=-p['B']*d**(-p['beta'])*p['beta']*g*q**(-p['beta']*g-1)
        elif m=='S2':grad=-a['C']*scale*a['eta']*q**(-a['eta']-1)
        else:grad=-a['C']*scale/q
        close('Gradient '+m+f' {n:g} {d:g} {q:g}',grad,float(r['dL_dQ']))
        check('Strict Q monotonicity '+m+f' {n:g} {d:g} {q:g}',grad<0 and float(r['numerical_dL_dQ'])<0)
    check('360-point surfaces for each model',len(surfaces)==3*360)
    # Analytic optimizer Jacobians are checked by central differences in log coordinates.
    from src.scaling.scale_quality import PARAMETERS,encoded,components,response_jacobian
    t=dict(N=data['B6'][:,0],D=data['B6'][:,1],Q=data['B6'][:,2]);parts=components(t,p,ref)
    for m,f in s['fitted'].items():
        ks=PARAMETERS[m];x=encoded(f['parameters'],ks);_,J=response_jacobian(x,ks,m,parts,p['beta'])
        for j,k in enumerate(ks):
            xp=x.copy();xm=x.copy();xp[j]+=1e-5;xm[j]-=1e-5
            def params(v):return {h:np.exp(a) if h in ['gamma0','C','eta'] else a for h,a in zip(ks,v)}
            numeric=(formula(data['B6'],p,ref,m,params(xp))-formula(data['B6'],p,ref,m,params(xm)))/2e-5
            close('Independent Jacobian '+m+' '+k,J[:,j],numeric,tol=1e-7)
    # All B8 records remain; evaluate both within-range and out-of-range failures.
    a=data['B8'];b=data['B6'];inside=((a[:,:3]>=b[:,:3].min(axis=0))&(a[:,:3]<=b[:,:3].max(axis=0))).all(axis=1)
    nonover=np.array([k not in set(keys['B6'])|set(keys['B7']) for k in keys['B8']])
    masks=dict(full=np.ones(len(a),dtype=bool),nonoverlap=nonover,within_B6_numeric_ranges=inside,outside_B6_numeric_ranges=~inside,
               nonoverlap_within_ranges=nonover&inside,nonoverlap_outside_ranges=nonover&~inside)
    for r in read('B8_stress_test.csv'):
        mask=masks[r['subset']];v=scores(a[mask,3],pred['B8'][r['model']][mask])
        check('B8 pressure population '+r['subset']+' '+r['model'],mask.sum()==int(r['n']))
        for k,z in v.items():
            if z is not None:close('B8 error '+r['subset']+' '+r['model']+' '+k,z,float(r[k]))
    check('All low-Loss B8 observations retained',len(raw['B8'])==1704 and int((a[:,3]<p['E']).sum())==1095)
    for path in (ROOT/'outputs/figures/question2_phase2C').glob('*.png'):
        with Image.open(path) as im:im.verify()
        check('Figure decodes '+path.name,True)
    snapshot=json.loads((OUT/'frozen_before_phase2C.json').read_text(encoding='utf-8'))
    for r in snapshot['files']:
        path=ROOT/r['path'];h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1048576),b''):h.update(block)
        check('Frozen prior '+r['path'],path.stat().st_size==r['bytes'] and h.hexdigest()==r['sha256'])
    check('Human freeze not performed',s['human_freeze'] is False)
    check('No model accepted if all Delta-L gates fail',not s['all_B7_DeltaL_failed'] or s['selected_model'] is None)
    with (OUT/'independent_verification.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['check','passed']);w.writeheader();w.writerows(CHECKS)
    print('All independent checks passed:',len(CHECKS),'Prior files preserved:',len(snapshot['files']))


if __name__=='__main__':main()
