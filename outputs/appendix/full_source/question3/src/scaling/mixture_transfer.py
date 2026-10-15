"""Frozen-mixture scale transfer and modular fusion; no source model refitting.

All sizes are in billions. Saved replacement compositions and ILR coordinates
are reused directly: they must not undergo a second zero replacement.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import pearsonr, spearmanr, t

SCALES = [('1M', 'test_1M', .001, 'observed'),
          ('60M', 'test_60M', .06, 'observed'),
          ('1B', 'test_1B', 1., 'observed'),
          ('10B', 'estimated_10B', 10., 'estimated'),
          ('70B', 'estimated_70B', 70., 'estimated')]


def frozen_metadata(root):
    """Read exactly the human-selected L_mean first-order ElasticNet metadata."""
    base = Path(root) / 'outputs/mixture'
    policy = json.loads((base/'stage2/final_model_policy.json').read_text(encoding='utf-8'))
    if policy['selected_model'] != 'M1_ElasticNet':
        raise ValueError('Frozen main mixture model differs from the authorized model')
    rows = json.loads((base/'final_model_metadata.json').read_text(encoding='utf-8'))
    selected = [r for r in rows if r['model']=='M1_ElasticNet' and r['response']=='L_mean']
    if len(selected) != 1 or selected[0]['n_model_features'] != 16:
        raise ValueError('Ambiguous or non-first-order frozen mixture model')
    return selected[0]


def predict_ilr(x, meta):
    """Evaluate the frozen first-order model on saved ILR, without fitting."""
    a = np.asarray(x, float)
    return (float(meta['y_mean']) + float(meta['y_scale']) *
            (((a-np.array(meta['X_mean']))/np.array(meta['X_scale'])) @
             np.array(meta['coefficients']) + float(meta['intercept'])))


def predict_recipe(p, meta):
    """Apply frozen closure/zero convention for a new evaluation composition."""
    a = np.atleast_2d(np.asarray(p, float))
    if a.shape[1] != 17 or not np.isfinite(a).all() or (a<0).any() or (a.sum(1)<=0).any():
        raise ValueError('Expected finite nonnegative 17-domain recipes')
    a = a/a.sum(axis=1, keepdims=True)
    zero = a==0
    a = a*(1-zero.sum(axis=1)*meta['zero_delta'])[:, None]
    a[zero] = meta['zero_delta']
    return predict_ilr(np.log(a) @ np.asarray(meta['ilr_basis']).T, meta)


def inverse_ilr(z, meta):
    """Inverse frozen orthonormal basis, producing a strictly positive recipe."""
    v = np.atleast_2d(z) @ np.asarray(meta['ilr_basis'])
    p = np.exp(v-v.max(axis=1, keepdims=True))
    return p/p.sum(axis=1, keepdims=True)


def read_frozen_recipes(root, meta):
    """Join prediction to processed recipes by index; never read A4-A15 originals."""
    base = Path(root)/'outputs/mixture'
    pred = pd.read_csv(base/'test_predictions.csv')
    pred = pred[(pred.model=='M1_ElasticNet') & (pred.response=='L_mean')]
    est = pd.read_csv(base/'estimated_scale_predictions.csv')
    est = est[est.model=='M1_ElasticNet']
    train = pd.read_csv(base/'train_1M_closed_replaced_composition.csv').set_index('index')
    names = meta['domain_order']
    groups = []
    for label, dataset, size, evidence in SCALES:
        if evidence=='observed':
            g = pred[pred.dataset==dataset].copy()
            recipe = pd.read_csv(base/(dataset+'_closed_replaced_composition.csv')).set_index('index')
            ilr = pd.read_csv(base/(dataset+'_ILR_features.csv')).set_index('index')
            g = g.rename(columns={'predicted_loss':'M_p', 'true_loss':'Loss'})
            g['matched_train_index'] = pd.NA
            g['prediction_source'] = 'outputs/mixture/test_predictions.csv'
        else:
            g = est[est.dataset==dataset].copy()
            g = g.rename(columns={'uncalibrated_prediction':'M_p', 'estimated_loss':'Loss'})
            recipe = train.loc[g['matched_train_index']].copy()
            recipe.index = g['index'].to_numpy()
            ilr = None
            g['prediction_source'] = 'outputs/mixture/estimated_scale_predictions.csv'
        if g.empty or g['index'].duplicated().any() or recipe.index.duplicated().any():
            raise ValueError('Missing or duplicated recipe index: '+dataset)
        p = recipe.loc[g['index'], names].to_numpy(float)
        if not np.isfinite(p).all() or (p<=0).any() or not np.allclose(p.sum(1),1):
            raise ValueError('Invalid saved replacement composition')
        x = np.log(p) @ np.asarray(meta['ilr_basis']).T
        if ilr is not None and not np.allclose(x, ilr.loc[g['index']].to_numpy(float), atol=1e-12):
            raise ValueError('Saved composition and ILR disagree')
        if not np.allclose(predict_ilr(x, meta), g['M_p'], rtol=0, atol=1e-12):
            raise ValueError('Frozen predictor and saved M(p) disagree: '+dataset)
        if not np.isfinite(g[['Loss','M_p']].to_numpy(float)).all():
            raise ValueError('Missing prediction/Loss; no imputation authorized')
        for j, name in enumerate(names):
            g[name+'_saved_replaced'] = p[:,j]
        # Only compare recipe equality at saved floating precision, never index alone.
        g['recipe_signature'] = [json.dumps(np.round(v,12).tolist(),separators=(',',':')) for v in p]
        g['scale'] = label
        g['N_params_B'] = size
        g['evidence'] = evidence
        g['m_centered'] = g.M_p-g.M_p.mean()
        g['y_centered'] = g.Loss-g.Loss.mean()
        g['M_scale_mean'] = g.M_p.mean()
        g['Loss_scale_mean'] = g.Loss.mean()
        groups.append(g)
    return pd.concat(groups, ignore_index=True)


def transfer(g):
    """Centered OLS slope, conventional SE (n-2), and held-scale fit metrics.

    Despite no fitted intercept in centered coordinates, two means have been
    estimated from the pairs; the conventional original-coordinate OLS uses
    residual degrees of freedom n-2. Bootstrap is the primary uncertainty.
    """
    x = g.M_p.to_numpy(float); y = g.Loss.to_numpy(float)
    x = x-x.mean(); y = y-y.mean()
    sxx = float(x@x)
    if len(x)<3 or sxx<=0:
        raise ValueError('Insufficient recipe variation for transfer')
    slope = float(x@y/sxx)
    resid = y-slope*x
    se = float(np.sqrt((resid@resid)/(len(x)-2)/sxx))
    critical = t.ppf(.975,len(x)-2)
    return dict(lambda_s=slope, standard_error=se,
                OLS_CI_low=float(slope-critical*se), OLS_CI_high=float(slope+critical*se),
                R2=float(1-(resid@resid)/(y@y)), RMSE=float(np.sqrt(np.mean(resid**2))),
                Pearson=float(pearsonr(x,y).statistic), Spearman=float(spearmanr(x,y).statistic),
                n=len(x), residual_df=len(x)-2, M_scale_mean=float(g.M_p.mean()),
                Loss_scale_mean=float(g.Loss.mean()), intercept_after_centering=0.)


def bootstrap_transfers(data, cfg):
    """2000 paired recipe-index resamples, recentering each resampled scale.

    Entire recipe sets match for 1M/60M and 10B/70B, respectively. Use the same
    index draws within each pair to preserve cross-scale dependence. 1B recipes
    have no overlap and use independent draws. The frozen M(p) is never refit.
    """
    groups = {s:data[data.scale==s].sort_values('index').reset_index(drop=True) for s,_,_,_ in SCALES}
    sets = {s:set(g.recipe_signature) for s,g in groups.items()}
    cohorts = [('1M','60M'), ('1B',), ('10B','70B')]
    for cohort in cohorts:
        signatures = groups[cohort[0]].recipe_signature.tolist()
        for s in cohort[1:]:
            if groups[s].recipe_signature.tolist()!=signatures:
                raise ValueError('Expected paired recipes differ; revise resampling first')
    for i, c1 in enumerate(cohorts):
        for c2 in cohorts[i+1:]:
            if sets[c1[0]] & sets[c2[0]]:
                raise ValueError('Partial overlap requires an explicit joint resampling design')
    rng = np.random.default_rng(cfg['random_state'])
    B = cfg['bootstrap_repetitions']
    slopes = np.empty((B,5))
    draws = []
    for cohort in cohorts:
        g0 = groups[cohort[0]]; n = len(g0)
        ix = rng.integers(0,n,size=(B,n))
        draws.extend(dict(replicate=b+1, cohort='+'.join(cohort),
                          sampled_recipe_indices=g0['index'].to_numpy()[ix[b]].tolist()) for b in range(B))
        for s in cohort:
            g = groups[s]; x = g.M_p.to_numpy()[ix]; y = g.Loss.to_numpy()[ix]
            x -= x.mean(axis=1,keepdims=True); y -= y.mean(axis=1,keepdims=True)
            denom = np.sum(x*x,axis=1)
            if (denom<=0).any():
                raise ValueError('Bootstrap draw has no M(p) variation')
            j = [a[0] for a in SCALES].index(s)
            slopes[:,j] = np.sum(x*y,axis=1)/denom
    records=[]; summaries=[]
    for j,(s,_,N,evidence) in enumerate(SCALES):
        v = slopes[:,j]
        summaries.append(dict(scale=s,N_params_B=N,evidence=evidence,
                              bootstrap_mean=float(v.mean()),bootstrap_median=float(np.median(v)),
                              bootstrap_std=float(v.std(ddof=1)),CI_low=float(np.quantile(v,.025)),
                              CI_high=float(np.quantile(v,.975)),positive_fraction=float((v>0).mean()),
                              negative_fraction=float((v<0).mean()),n_bootstrap=B))
        records.extend(dict(replicate=b+1,scale=s,lambda_s=float(v[b])) for b in range(B))
    return slopes, summaries, records, draws


def p2_value(x, theta):
    """Log-parameterized positive amplitude/slope; ln Nc is dimensionless B scale."""
    a,b = np.exp(theta[:2]); return a*np.tanh(b*(theta[2]-x))


def p2_jac(x, theta):
    """Exact Jacobian in ln(amplitude), ln(slope), ln(Nc/B)."""
    a,b = np.exp(theta[:2]); z = b*(theta[2]-x); h = np.tanh(z); s = 1-h*h
    return np.c_[a*h,a*s*z,a*s*b]


def fit_curve(x,y,model,cfg,starts=None,nstarts=None,rng=None):
    """Equal-weight-by-scale P1/P2 only; no global (N,D,Q,p) optimization."""
    x,y = np.asarray(x,float), np.asarray(y,float)
    if model=='P1':
        design = np.c_[np.ones(len(x)),x]
        v = np.linalg.lstsq(design,y,rcond=None)[0]
        pred = design@v
        return dict(parameters=dict(a=float(v[0]),b=float(v[1])),theta=v.tolist(),
                    prediction=pred.tolist(),SSE=float(np.sum((pred-y)**2)),success=True,
                    boundary_hit=False,jacobian_rank=int(np.linalg.matrix_rank(design)),
                    jacobian_condition=float(np.linalg.cond(design)),n_starts=1)
    if len(x)<3:
        return dict(success=False,reason='fewer_scales_than_parameters')
    bounds = cfg['p2_bounds']
    lo = np.array([np.log(bounds['amplitude'][0]),np.log(bounds['slope'][0]),bounds['log_Nc_B'][0]])
    hi = np.array([np.log(bounds['amplitude'][1]),np.log(bounds['slope'][1]),bounds['log_Nc_B'][1]])
    count = nstarts or cfg['full_fit_starts']
    seeds = [np.array(t,float) for t in (starts or [])]
    p1 = np.linalg.lstsq(np.c_[np.ones(len(x)),x],y,rcond=None)[0]
    root = float(-p1[0]/p1[1]) if abs(p1[1])>1e-12 else float(np.median(x))
    seeds += [np.array([np.log(max(1.,np.max(abs(y)))),np.log(.15),root]),
              np.array([np.log(max(.5,np.max(abs(y)))),np.log(.4),float(np.median(x))])]
    if rng is None: rng=np.random.default_rng(cfg['random_state'])
    while len(seeds)<count:
        seeds.append(np.array([rng.uniform(-1,2),rng.uniform(-4,.5),rng.uniform(x.min()-3,x.max()+3)]))
    fits=[]
    for initial in seeds[:count]:
        initial = np.clip(initial,lo+1e-8,hi-1e-8)
        f = least_squares(lambda t:p2_value(x,t)-y,initial,jac=lambda t:p2_jac(x,t),bounds=(lo,hi),
                          max_nfev=cfg['max_nfev'],xtol=1e-10,ftol=1e-10,gtol=1e-10)
        if f.success and np.isfinite(f.fun).all(): fits.append(f)
    if not fits: return dict(success=False,reason='no_successful_start')
    f=min(fits,key=lambda v:float(v.fun@v.fun)); theta=f.x
    a,b=np.exp(theta[:2]); jac=p2_jac(x,theta)
    tol=cfg['boundary_tolerance']; boundary=bool(np.any((theta-lo)<tol)|np.any((hi-theta)<tol))
    return dict(parameters=dict(a=float(a),b=float(b),log_Nc_B=float(theta[2])),theta=theta.tolist(),
                prediction=p2_value(x,theta).tolist(),SSE=float(f.fun@f.fun),success=True,
                boundary_hit=boundary,jacobian_rank=int(np.linalg.matrix_rank(jac,tol=1e-10)),
                jacobian_condition=float(np.linalg.cond(jac)),n_starts=count,
                n_successful_starts=len(fits))


def curve_root(fit, model):
    """Return ln(Nc/B); a finite root is not proof of empirical identification."""
    if not fit['success']: return None
    p=fit['parameters']
    if model=='P1':
        if abs(p['b'])<1e-12:return None
        return float(-p['a']/p['b'])
    return float(p['log_Nc_B'])


def empirical_lambda(N, sizes, slopes):
    """Piecewise linear in ln N, refusing extrapolation beyond provided knots."""
    N=np.asarray(N,float); sizes=np.asarray(sizes,float); slopes=np.asarray(slopes,float)
    if not np.isfinite(N).all() or np.any(N<sizes.min()) or np.any(N>sizes.max()):
        raise ValueError('N outside supported empirical knots; extrapolation not authorized')
    return np.interp(np.log(N),np.log(sizes),slopes)


class ModularScaling:
    """Evaluate fixed modules. This class never fits or constructs joint test data."""
    def __init__(self, classical, quality, meta, p_ref, sizes, slopes):
        self.classical=dict(classical); self.quality=dict(quality); self.meta=meta
        self.p_ref=np.asarray(p_ref,float); self.M_ref=float(predict_recipe(self.p_ref,meta)[0])
        self.sizes=np.asarray(sizes,float); self.slopes=np.asarray(slopes,float)

    def components(self,N,D,Q,p):
        """Loss modules with N,D in billions, 0<Q<=1, and a closed recipe."""
        N,D,Q=np.broadcast_arrays(np.asarray(N,float),np.asarray(D,float),np.asarray(Q,float))
        if not np.isfinite(N+D+Q).all() or (N<=0).any() or (D<=0).any() or (Q<=0).any() or (Q>1).any():
            raise ValueError('Illegal N,D,Q for modular candidate')
        c=self.classical; q=self.quality
        base=c['E']+c['A']*N**(-c['alpha'])+c['B']*D**(-c['beta'])
        dq=q['C_Q']*N**(-q['u_Q'])*(-np.log(Q))
        dp=empirical_lambda(N,self.sizes,self.slopes)*(predict_recipe(p,self.meta)-self.M_ref)
        return dict(L0=base,Delta_Q=dq,Delta_p=dp,L_gen=base+dq+dp)

    def partials(self,N,D,Q,p):
        """Analytic derivatives; N-knot derivative has distinct left/right limits."""
        N,D,Q=float(N),float(D),float(Q)
        comp=self.components(N,D,Q,p); c=self.classical; q=self.quality
        i=int(np.clip(np.searchsorted(self.sizes,N,side='right')-1,0,len(self.sizes)-2))
        dlam=(self.slopes[i+1]-self.slopes[i])/np.log(self.sizes[i+1]/self.sizes[i])/N
        mdiff=float(predict_recipe(p,self.meta)[0]-self.M_ref)
        return dict(dL_dN=float(-c['alpha']*c['A']*N**(-c['alpha']-1)-q['u_Q']/N*comp['Delta_Q']+dlam*mdiff),
                    dL_dD=float(-c['beta']*c['B']*D**(-c['beta']-1)),
                    dL_dQ=float(-q['C_Q']*N**(-q['u_Q'])/Q))
