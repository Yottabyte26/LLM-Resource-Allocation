"""Quality-only extensions with immutable classical coefficients and strict intake."""
from __future__ import annotations
from collections import defaultdict
from decimal import Decimal
import numpy as np
from scipy.optimize import least_squares
from .io import csv_rows

NAMES = {'B6':'supplementary_NQ_experiment.csv',
         'B7':'supplementary_NQ_experiment_expanded.csv',
         'B8':'supplementary_NQ_experiment_large.csv'}
MODELS = ['M0', 'MQ1', 'MQ2', 'MQ1_offset', 'MQ2_offset']


def read_parameters(root):
    """Read full-precision primary Phase 1 row; never estimate classical terms."""
    path=root/'outputs/scaling/phase1/classical_scaling_parameters.csv'
    _, rows=csv_rows(path)
    primary=[r for r in rows if r['model']=='M0_N_D' and r['weighting']=='trajectory_balanced']
    if len(primary)!=1:raise ValueError('Ambiguous frozen classical parameter row')
    return {k:float(primary[0][k]) for k in ['E','A','B','alpha','beta']}, path


def read_data(root, label):
    """Canonical exact decimal NDQ keys; no imputation, deletion or Q reversal."""
    path=root/'real_attachments/B_scaling_laws'/NAMES[label]
    columns, rows=csv_rows(path)
    a=np.array([[float(r[c]) for c in ['N_params_B','D_tokens_B','Q_score','val_loss']] for r in rows])
    if not np.isfinite(a).all() or (a[:,:3]<=0).any() or (a[:,2]>1).any():
        raise ValueError(f'{label}: illegal N/D/Q/L')
    keys=[tuple(format(Decimal(r[c]).normalize(),'f') for c in ['N_params_B','D_tokens_B','Q_score']) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError(f'{label}: repeated candidate key; cannot take first/last')
    group=np.array(['N='+k[0]+';D='+k[1] for k in keys])
    return dict(label=label,path=path,columns=columns,records=rows,keys=keys,group=group,
                N=a[:,0],D=a[:,1],Q=a[:,2],L=a[:,3])


def subset(t, index):
    """Subset arrays for fitting, retaining source rows in the original dataset."""
    return {k:t[k][index] for k in ['N','D','Q','L','group']}


def predict_quality(t, p, model, gamma=0.0, offset=0.0):
    """Predict raw Loss with N/D in billions; no modification to frozen p."""
    N,D,Q=t['N'],t['D'],t['Q']
    base=p['E']+p['A']*N**(-p['alpha'])
    term=p['B']*D**(-p['beta'])
    if model=='M0':return base+term
    factor=-p['beta']*np.log(Q) if model.startswith('MQ1') else p['beta']*(1-Q)
    result=base+term*np.exp(gamma*factor)+(offset if model.endswith('_offset') else 0.0)
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite quality prediction')
    return result


def fit_quality(t, p, model, config, multiplicities=None):
    """Bounded raw-Loss least squares from every specified start.

    A bootstrap multiplicity repeats the complete ND block. All starts use only
    the supplied training rows. Classical coefficients are read-only arguments.
    """
    if model not in MODELS[1:]:raise ValueError(model)
    n=len(t['L']); weights=np.ones(n)
    if multiplicities is not None:
        weights=np.array([multiplicities.get(g,0) for g in t['group']],dtype=float)
    active=weights>0
    if not active.any():raise ValueError('Empty resampled training set')
    a=subset(t,active); sw=np.sqrt(weights[active]/weights.sum())
    base=p['E']+p['A']*a['N']**(-p['alpha'])
    term=p['B']*a['D']**(-p['beta'])
    factor=-p['beta']*np.log(a['Q']) if model.startswith('MQ1') else p['beta']*(1-a['Q'])
    has_offset=model.endswith('_offset')
    def fun(x):
        return sw*(base+term*np.exp(x[0]*factor)+(x[1] if has_offset else 0)-a['L'])
    def jac(x):
        j=sw*term*factor*np.exp(x[0]*factor)
        return np.column_stack([j,sw]) if has_offset else j[:,None]
    lo=[config['gamma_bounds'][0]];hi=[config['gamma_bounds'][1]]
    if has_offset:
        lo.append(config['offset_bounds'][0]);hi.append(config['offset_bounds'][1])
    runs=[]
    for i,initial in enumerate(config['gamma_starts'],1):
        x0=[initial,0.0] if has_offset else [initial]
        opt=least_squares(fun,x0,jac=jac,bounds=(lo,hi),method='trf',loss='linear',
                          ftol=config['optimizer_tolerance'],xtol=config['optimizer_tolerance'],
                          gtol=config['optimizer_tolerance'],max_nfev=config['max_nfev'])
        objective=float(np.dot(opt.fun,opt.fun))
        legal=bool(opt.success and np.isfinite(objective) and opt.x[0]>0 and np.isfinite(opt.x).all())
        runs.append(dict(start=i,initial_gamma=float(initial),initial_offset=0.0,
                         gamma=float(opt.x[0]),offset=float(opt.x[1]) if has_offset else 0.0,
                         objective=objective,successful=bool(opt.success),legal_solution=legal,
                         status=int(opt.status),nfev=int(opt.nfev),optimality=float(opt.optimality),
                         at_bound=bool(np.any(np.isclose(opt.x,lo,rtol=0,atol=1e-6)) or np.any(np.isclose(opt.x,hi,rtol=0,atol=1e-6))),
                         message=opt.message))
    valid=[r for r in runs if r['legal_solution']]
    if not valid:raise RuntimeError('No successful legal quality fit: '+model)
    best=min(valid,key=lambda r:r['objective'])
    return dict(model=model,**best,n_starts=len(runs),
                distinct_solution_gamma_range=max(r['gamma'] for r in valid)-min(r['gamma'] for r in valid),
                n_training_rows=int(active.sum()),n_training_ND_groups=len(set(a['group']))), runs


def group_reference(t, prediction):
    """Differences relative to highest observed Q in the full evaluation dataset.

    For B7 new Q points, the Q=1 reference belongs to the shared ND trajectory.
    No model is fitted on that reference. References themselves are marked and
    excluded from primary difference metrics to avoid trivial-zero inflation.
    """
    groups=defaultdict(list)
    for i,g in enumerate(t['group']):groups[g].append(i)
    obs=np.zeros(len(t['L']));pred=np.zeros(len(t['L']));refq=np.zeros(len(t['L']))
    refs=np.zeros(len(t['L']),dtype=bool)
    for idx in groups.values():
        ref=max(idx,key=lambda i:t['Q'][i])
        obs[idx]=t['L'][idx]-t['L'][ref]
        pred[idx]=prediction[idx]-prediction[ref]
        refq[idx]=t['Q'][ref];refs[ref]=True
    return obs,pred,refq,refs
