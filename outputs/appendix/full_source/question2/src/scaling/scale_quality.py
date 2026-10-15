"""Three scale-modulated quality models; no estimation of classical coefficients."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares

PARAMETERS={'S1':['gamma0','u','v'],'S2':['C','u','v','eta'],'S3':['C','u','v']}
POSITIVE={'gamma0','C','eta'}


def centers(t):
    """Median in log space over unique design levels, not response-weighted scales."""
    return dict(N0=float(np.exp(np.median(np.log(np.unique(t['N']))))),
                D0=float(np.exp(np.median(np.log(np.unique(t['D']))))))


def components(t,p,reference):
    """Reusable raw-Loss and dimensionless logarithmic design terms."""
    logN=np.log(t['N']/reference['N0']);logD=np.log(t['D']/reference['D0']);qlog=-np.log(t['Q'])
    term=p['B']*t['D']**(-p['beta'])
    l0=p['E']+p['A']*t['N']**(-p['alpha'])+term
    return logN,logD,qlog,term,l0


def physical(x,keys):
    """Log coordinates for strictly positive parameters; u/v include exact zero."""
    return {k:float(np.exp(a)) if k in POSITIVE else float(a) for k,a in zip(keys,x)}


def encoded(parameters,keys):
    return np.array([np.log(parameters[k]) if k in POSITIVE else parameters[k] for k in keys])


def response_jacobian(x,keys,model,parts,beta):
    """Analytic prediction/Jacobian in the optimizer's coordinates.

    The broad S1 bounds still keep exponents below double-precision overflow
    on the complete B6 support. No residual clipping or altered model is used.
    """
    a=physical(x,keys);n,d,z,term,l0=parts;scale=np.exp(-a['u']*n+a['v']*d)
    if model=='S1':
        gamma=a['gamma0']*scale;f=beta*gamma*z
        expf=np.exp(f);quality=term*np.expm1(f);j0=term*expf*f
        derivatives={'gamma0':j0,'u':-j0*n,'v':j0*d}
    elif model=='S2':
        f=a['eta']*z;expf=np.exp(f);quality=a['C']*scale*np.expm1(f)
        derivatives={'C':quality,'u':-quality*n,'v':quality*d,'eta':a['C']*scale*expf*f}
    elif model=='S3':
        quality=a['C']*scale*z;derivatives={'C':quality,'u':-quality*n,'v':quality*d}
    else:raise ValueError(model)
    pred=l0+quality
    if 'offset' in keys:pred+=a['offset'];derivatives['offset']=np.ones(len(z))
    jac=np.column_stack([derivatives[k] for k in keys])
    if not np.isfinite(pred).all() or not np.isfinite(jac).all():raise FloatingPointError('Nonfinite model evaluation')
    return pred,jac


def predict_scale(t,p,reference,model,parameters):
    """Frozen classical skeleton plus a nonnegative Q=1-anchored quality term."""
    keys=PARAMETERS[model]+(['offset'] if 'offset' in parameters else [])
    return response_jacobian(encoded(parameters,keys),keys,model,components(t,p,reference),p['beta'])[0]


def marginal_quality(t,p,reference,model,parameters):
    """Exact derivative dLoss/dQ, including S1's baseline D dependence."""
    n,d,z,term,_=components(t,p,reference)
    scale=np.exp(-parameters['u']*n+parameters['v']*d)
    if model=='S1':
        gamma=parameters['gamma0']*scale
        grad=-term*np.exp(p['beta']*gamma*z)*p['beta']*gamma/t['Q']
    elif model=='S2':
        grad=-parameters['C']*scale*parameters['eta']*np.exp(parameters['eta']*z)/t['Q']
    elif model=='S3':grad=-parameters['C']*scale/t['Q']
    else:raise ValueError(model)
    return grad


def initial_parameters(model,cfg,seed,has_offset=False,count=None):
    """At least 50 dispersed, deterministic starts independent of evaluation Loss."""
    rng=np.random.default_rng(seed);count=count or cfg['multistart_count'];starts=[]
    if model=='S1':
        for g in [.1,.25,.5,1,2,4,8]:starts.append(dict(gamma0=g,u=.1,v=.1))
    elif model=='S2':
        for c,eta in [(.1,.1),(.2,.5),(.5,1),(1,.2),(5,.05),(50,.003),(100,.002)]:
            starts.append(dict(C=c,u=.1,v=.05,eta=eta))
    else:
        for c in [.03,.1,.2,.5,1,3,8]:starts.append(dict(C=c,u=.1,v=.1))
    while len(starts)<count:
        a=dict(u=float(rng.uniform(0,.5)),v=float(rng.uniform(0,.5)))
        if model=='S1':a['gamma0']=float(np.exp(rng.uniform(np.log(.08),np.log(3))))
        else:a['C']=float(np.exp(rng.uniform(np.log(.03),np.log(10))))
        if model=='S2':a['eta']=float(np.exp(rng.uniform(np.log(.002),np.log(1.5))))
        starts.append(a)
    if has_offset:
        for a in starts:a['offset']=float(rng.uniform(-.1,.1))
    return starts[:count]


def fit_scale(t,p,reference,model,cfg,seed,multiplicities=None,has_offset=False,count=None):
    """Bounded nonlinear LS with every start audited; bootstrap repeats whole ND blocks."""
    keys=PARAMETERS[model]+(['offset'] if has_offset else [])
    weights=np.ones(len(t['L'])) if multiplicities is None else np.array([multiplicities.get(g,0) for g in t['group']],dtype=float)
    ix=weights>0
    a={k:t[k][ix] for k in ['N','D','Q','L','group']};weights=weights[ix];weights/=weights.sum();sw=np.sqrt(weights)
    parts=components(a,p,reference);bounds=cfg['parameter_bounds']
    lo=encoded({k:bounds[k][0] for k in keys},keys);hi=encoded({k:bounds[k][1] for k in keys},keys)
    # u/v have zero lower bounds, which encoded() preserves without taking log.
    def fun(x):return sw*(response_jacobian(x,keys,model,parts,p['beta'])[0]-a['L'])
    def jac(x):return sw[:,None]*response_jacobian(x,keys,model,parts,p['beta'])[1]
    runs=[];tol=cfg['optimizer_tolerance']
    for i,start in enumerate(initial_parameters(model,cfg,seed,has_offset,count),1):
        x0=np.clip(encoded(start,keys),lo,hi)
        opt=least_squares(fun,x0,jac=jac,bounds=(lo,hi),method='trf',loss='linear',x_scale='jac',
                          max_nfev=cfg['max_nfev'],ftol=tol,xtol=tol,gtol=tol)
        params=physical(opt.x,keys);obj=float(np.dot(opt.fun,opt.fun))
        sv=np.linalg.svd(opt.jac,compute_uv=False)
        rank=int((sv>sv[0]*cfg['jacobian_rank_relative_tolerance']).sum()) if len(sv) and sv[0]>0 else 0
        condition=float(sv[0]/sv[-1]) if len(sv) and sv[-1]>0 else None
        hit={k:bool(params[k]-bounds[k][0]<=cfg['boundary_absolute_tolerance'] or bounds[k][1]-params[k]<=cfg['boundary_absolute_tolerance']) for k in keys}
        legal=bool(opt.success and np.isfinite(obj) and all(np.isfinite(v) for v in params.values()))
        runs.append(dict(start=i,initial_parameters=start,parameters=params,objective=obj,
                         convergence=bool(opt.success),legal_solution=legal,status=int(opt.status),nfev=int(opt.nfev),
                         jacobian_rank=rank,jacobian_condition=condition,jacobian_singular_values=sv.tolist(),
                         boundary_hit=any(hit.values()),boundary_parameters=[k for k,v in hit.items() if v],
                         per_parameter_boundary=hit,message=opt.message))
    valid=[r for r in runs if r['legal_solution']]
    if not valid:raise RuntimeError('No successful legal fit: '+model)
    best=min(valid,key=lambda r:r['objective'])
    nearby=[r for r in valid if r['objective']<=best['objective']+1e-9]
    spread={k:max(r['parameters'][k] for r in nearby)-min(r['parameters'][k] for r in nearby) for k in keys}
    return dict(model=model,source_offset=has_offset,**best,n_starts=len(runs),n_successful_starts=len(valid),
                parameter_count=len(keys),near_optimal_parameter_spread=spread,
                all_start_boundary_rate=sum(r['boundary_hit'] for r in valid)/len(valid),
                n_training_rows=int(ix.sum()),n_training_ND_groups=len(set(a['group']))),runs
