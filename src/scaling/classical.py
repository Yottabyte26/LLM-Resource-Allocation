"""Positive nonlinear additive scaling laws, balanced objectives and analytic Jacobians."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import pearsonr,spearmanr

PARAMETERS={'M0_N_D':['E','A','B','alpha','beta'], 'M1_N_only':['E','A','alpha'], 'M2_D_only':['E','B','beta']}


def weights(groups,balanced=True,multiplicities=None):
    """Each selected trajectory occurrence has equal total mass; normalize to one.

    Cluster-bootstrap multiplicity represents resampled whole trajectories,
    never randomly selected checkpoint rows.
    """
    groups=np.asarray(groups);unique,count=np.unique(groups,return_counts=True);counts=dict(zip(unique,count))
    if multiplicities is None:multiplicities={g:1 for g in unique}
    w=np.array([multiplicities.get(g,0)/(counts[g] if balanced else 1) for g in groups],dtype=float)
    if not np.isfinite(w).all() or w.sum()<=0:raise ValueError('Invalid trajectory weights')
    return w/w.sum()


def predict(parameters,N,D,model='M0_N_D'):
    """N and D measured in units of 1B. This does NOT log-transform Loss."""
    result=np.full(np.shape(N),parameters['E'],dtype=float)
    if 'A' in PARAMETERS[model]:result+=parameters['A']*np.exp(-parameters['alpha']*np.log(N))
    if 'B' in PARAMETERS[model]:result+=parameters['B']*np.exp(-parameters['beta']*np.log(D))
    if not np.isfinite(result).all() or (result<=0).any():raise FloatingPointError('Invalid predicted Loss')
    return result


def prediction_jacobian(logtheta,lnN,lnD,model):
    """Analytic derivatives with respect to LOG PARAMETERS, not log Loss."""
    keys=PARAMETERS[model];p=dict(zip(keys,np.exp(logtheta)));n=len(lnN)
    pred=np.full(n,p['E']);parts={'E':np.full(n,p['E'])}
    if 'A' in p:
        a=p['A']*np.exp(-p['alpha']*lnN);pred+=a;parts['A']=a;parts['alpha']=-p['alpha']*lnN*a
    if 'B' in p:
        b=p['B']*np.exp(-p['beta']*lnD);pred+=b;parts['B']=b;parts['beta']=-p['beta']*lnD*b
    return pred,np.column_stack([parts[k] for k in keys])


def initial_values(N,D,L,model,settings,rng,count,warm=None):
    """Reproducible dispersed exponent starts and data-scale positive amplitudes."""
    starts=[];keys=PARAMETERS[model];bounds=settings['parameter_bounds']
    if warm is not None:starts.append(np.array([warm[k] for k in keys]))
    exponent_lo,exponent_hi=settings['initial_exponent_range']
    while len(starts)<count:
        a,b=np.exp(rng.uniform(np.log(exponent_lo),np.log(exponent_hi),size=2))
        E=float(rng.uniform(.05,.95)*L.min());excess=max(float(np.median(L))-E,1e-5)
        fraction=float(rng.uniform(.1,.9)) if model=='M0_N_D' else (1.0 if model=='M1_N_only' else 0.0)
        p=dict(E=E,A=excess*fraction*float(np.median(N))**a,B=excess*(1-fraction)*float(np.median(D))**b,alpha=a,beta=b)
        starts.append(np.array([p[k] for k in keys]))
    lo=np.array([bounds[k][0] for k in keys]);hi=np.array([bounds[k][1] for k in keys])
    return [np.log(np.clip(x,lo*1.01,hi*.99)) for x in starts]


def fit(N,D,L,w,model,settings,seed,warm=None,count=None,save_starts=True):
    """Weighted least squares in ORIGINAL Loss with strict positive parameter bounds.

    Independent multi-start solves; only legal successful finite solutions are
    eligible. No validation data appears in this function or its selection.
    """
    keys=PARAMETERS[model];w=np.asarray(w);sel=w>0
    N,D,L,w=np.asarray(N)[sel],np.asarray(D)[sel],np.asarray(L)[sel],w[sel];w/=w.sum()
    lnN=np.log(N);lnD=np.log(D);sw=np.sqrt(w)
    bounds=settings['parameter_bounds'];lo=np.log([bounds[k][0] for k in keys]);hi=np.log([bounds[k][1] for k in keys])
    def fun(x):return sw*(prediction_jacobian(x,lnN,lnD,model)[0]-L)
    def jac(x):return sw[:,None]*prediction_jacobian(x,lnN,lnD,model)[1]
    starts=initial_values(N,D,L,model,settings,np.random.default_rng(seed),count or settings['multistart_count'],warm)
    runs=[];best=None;tol=settings['optimizer_tolerance']
    for number,x0 in enumerate(starts,1):
        solved=least_squares(fun,x0,jac=jac,bounds=(lo,hi),method='trf',x_scale='jac',loss='linear',
            max_nfev=settings['max_nfev'],ftol=tol,xtol=tol,gtol=tol)
        p=dict(zip(keys,map(float,np.exp(solved.x))));objective=float(np.dot(solved.fun,solved.fun))
        legal=bool(solved.success and np.isfinite(objective) and all(np.isfinite(v) and v>0 for v in p.values()))
        if save_starts:
            runs.append(dict(start=number,initial_parameters=dict(zip(keys,map(float,np.exp(x0)))),**p,
                objective=objective,successful=bool(solved.success),legal_solution=legal,status=int(solved.status),
                nfev=int(solved.nfev),optimality=float(solved.optimality),message=solved.message,
                active_bounds=solved.active_mask.tolist()))
        if legal and (best is None or objective<best['objective']):
            singular=np.linalg.svd(solved.jac,compute_uv=False);rank=int(np.linalg.matrix_rank(solved.jac))
            best=dict(model=model,parameters=p,objective=objective,successful=True,status=int(solved.status),
                nfev=int(solved.nfev),optimality=float(solved.optimality),chosen_start=number,
                active_bounds=solved.active_mask.tolist(),jacobian_singular_values=singular.tolist(),jacobian_rank=rank,
                jacobian_condition_number=float(singular[0]/singular[-1]) if singular[-1]>0 else None,
                log_parameters=solved.x.tolist(),n_positive_weight_rows=len(L),n_starts=len(starts))
    if best is None:raise RuntimeError('All nonlinear starts failed; do not substitute invented parameters')
    return best,runs


def metrics(y,p):
    """Observed-minus-predicted bias; constant ranks and zero-variance R2 undefined."""
    y=np.asarray(y);p=np.asarray(p);e=y-p;defined=len(y)>1 and np.ptp(y)>0 and np.ptp(p)>0
    return dict(n=len(y),RMSE=float(np.sqrt(np.mean(e**2))),MAE=float(np.mean(abs(e))),
        R2=float(1-np.sum(e**2)/np.sum((y-y.mean())**2)) if len(y)>1 and np.ptp(y)>0 else None,
        Pearson=float(pearsonr(y,p).statistic) if defined else None,Spearman=float(spearmanr(y,p).statistic) if defined else None,
        mean_bias=float(e.mean()),median_residual=float(np.median(e)),residual_std=float(e.std(ddof=1)) if len(y)>1 else None)
