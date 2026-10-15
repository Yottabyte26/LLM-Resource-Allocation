# 问题三：论文源程序逐文件汇编
# For appendix display only; NOT a standalone merged executable.
# Source bodies, imports, formulas, branches, parameters and entry guards are unchanged.
# Use the original scripts from the project root for the original workflow; DO NOT run this listing.
# This extraction task did not rerun any model or experiment.

# ============================================================
# Source: src/optimization/costs.py
# Original lines: 1-53
# Purpose: 6ND、注意力成本与exp/power/log质量增量成本
# ============================================================
"""Official cost proxies evaluated in FLOPs; model inputs N/D are billions."""
from __future__ import annotations
import numpy as np


def _validate_quality(q, q0):
    """Reject invalid or nonfinite qualities rather than silently clipping them."""
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or np.any((q <= 0) | (q > 1)) or not 0 < q0 <= 1:
        raise ValueError('Q and Q0 must be finite in (0, 1]')
    return q


def quality_increment_per_token(q, q0, family, gamma, lam):
    """Compute [g(Q)-g(Q0)]+, retaining the exact Appendix B families."""
    q = _validate_quality(q, q0)
    if gamma <= 0 or lam <= 0:
        raise ValueError('Positive official cost parameters required')
    if family == 'exponential':
        diff = gamma * np.exp(lam*q0) * np.expm1(lam*(q-q0))
    elif family == 'power':
        diff = gamma * (q**lam-q0**lam)
    elif family == 'logarithmic':
        diff = gamma * np.log1p(lam*(q-q0)/(1+lam*q0))
    else:
        raise ValueError(f'Unknown cost family: {family}')
    return np.maximum(diff, 0)


def marginal_quality_per_token(q, q0, family, gamma, lam):
    """Right derivative at Q0; below Q0 the positive-part cost is zero."""
    q = _validate_quality(q, q0)
    if family == 'exponential':
        derivative = gamma*lam*np.exp(lam*q)
    elif family == 'power':
        derivative = gamma*lam*q**(lam-1)
    elif family == 'logarithmic':
        derivative = gamma*lam/(1+lam*q)
    else:
        raise ValueError(f'Unknown cost family: {family}')
    return np.where(q >= q0, derivative, 0)


def cost_components(n_b, d_b, q, q0, l_ctx, official, family):
    """Return training, incremental quality and attention FLOPs, without fitting."""
    scalars = np.asarray([n_b, d_b, l_ctx], float)
    if not np.isfinite(scalars).all() or (scalars <= 0).any():
        raise ValueError('Finite positive N_B, D_B and L_ctx required')
    pars = official['quality_costs'][family]
    train = official['training_coefficient']*n_b*d_b*1e18
    attention = official['eta']*n_b*d_b*l_ctx*1e18
    quality = d_b*1e9*quality_increment_per_token(q, q0, family, pars['gamma'], pars['lambda'])
    return train, quality, attention

# ============================================================
# Source: src/scaling/mixture_transfer.py
# Original lines: 1-283
# Purpose: 冻结ILR预测、逐尺度lambda与配方bootstrap；P1/P2曲线仅比较候选
# P1/P2 smooth lambda curves: candidate / sensitivity models, not final main model. Final lambda is piecewise linear in ln N; these curve labels differ from Question3 resource policies.
# ============================================================
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

# ============================================================
# Source: src/optimization/setup.py
# Original lines: 1-672
# Purpose: 官方成本、Q0、决策边界与832经验候选的来源
# ============================================================
"""Audit official inputs and frozen outputs; no fitting and no optimization."""
from __future__ import annotations
import hashlib
import json
import logging
import re
import zipfile
import xml.etree.ElementTree as ET
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import trim_mean
from .costs import cost_components, quality_increment_per_token, marginal_quality_per_token

NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}


def omml_text(node):
    """Preserve superscripts/subscripts/fractions when reading original OOXML."""
    tag = node.tag.split('}')[-1]
    if tag == 't':
        return node.text or ''
    if tag in ('sSup', 'sSub'):
        base = omml_text(node.find('m:e', NS))
        part = omml_text(node.find('m:sup' if tag == 'sSup' else 'm:sub', NS))
        return base + ('^' if tag == 'sSup' else '_') + '{' + part + '}'
    if tag == 'f':
        return '(' + omml_text(node.find('m:num', NS)) + ')/(' + omml_text(node.find('m:den', NS)) + ')'
    return ''.join(omml_text(child) for child in node)


def numeric_expression(expr):
    """Parse only decimal coefficients times base-ten powers; never eval text."""
    expr = ''.join(expr.split()).replace('−', '-').replace('×', '*')
    match = re.fullmatch(r'(?:(\d+(?:\.\d+)?)\*)?10\^\{(-?\d+)\}', expr)
    if match:
        return float(match[1] or 1)*10.0**int(match[2])
    if re.fullmatch(r'\d+(?:\.\d+)?', expr):
        return float(expr)
    raise ValueError('Unsupported official numeric expression: ' + expr)


def save_csv(path, rows):
    """Write full floating precision UTF-8 CSV; paths are new Phase 1 outputs."""
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def save_json(path, data):
    """Reject nonfinite JSON numbers and keep full Python floating precision."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(frame, columns=None):
    """Make simple Markdown tables without an optional tabulate dependency."""
    a = frame if columns is None else frame[columns]
    lines = ['| ' + ' | '.join(a.columns) + ' |', '| ' + ' | '.join(['---']*len(a.columns)) + ' |']
    for row in a.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(format(x, '.10g') if isinstance(x, float) else str(x).replace('|', '/') for x in row) + ' |')
    return '\n'.join(lines)


def audit_official(root, out):
    """Extract the actual Word formulas, fail on any unsupported structural change."""
    docx = next(root.glob('*.docx'))
    with zipfile.ZipFile(docx) as archive:
        xml = archive.read('word/document.xml')
    document = ET.fromstring(xml)
    paras = [omml_text(p) for p in document.findall('.//w:p', NS)]
    (out/'official_word_extraction.txt').write_text('\n'.join(f'{i}: {p}' for i,p in enumerate(paras)), encoding='utf-8')
    selected = {}
    for key, phrase in [('budget', '为便于对照'), ('train', '基础训练开销：'),
                        ('attention', '长文本注意力开销：'), ('increment', 'C_{Q}=D'),
                        ('exponential', '指数型：'), ('power', '幂函数型：'), ('logarithmic', '对数渐进型：')]:
        found = [(i,p) for i,p in enumerate(paras) if phrase in p
                 and (key != 'increment' or p.strip().startswith(phrase))]
        if len(found) != 1:
            raise ValueError(f'Official formula ambiguous: {key}: {found}')
        selected[key] = found[0]
    compact = lambda p: ''.join(p.split())
    train_text = compact(selected['train'][1])
    if 'C_{train}=6ND' not in train_text or 'C_{attn}=ηNDL_{ctx}' not in compact(selected['attention'][1]):
        raise ValueError('Unexpected official training/attention formula')
    if 'C_{Q}=D[g(Q)-g(Q_{0})]_{+}' not in compact(selected['increment'][1]):
        raise ValueError('Unexpected official incremental quality formula')
    eta_expr = re.search(r'η=([^，]+)', selected['attention'][1])[1]
    budgets = [numeric_expression('10^{'+e+'}') for e in re.findall(r'10\^\{(\d+)\}', selected['budget'][1])]
    if len(budgets) != 3:
        raise ValueError('Could not recover three official budgets from OMML')
    official = dict(training_coefficient=6.0, eta=numeric_expression(eta_expr),
                    budgets_FLOPs=budgets, budget_status='official_recommendation_not_mandatory', quality_costs={})
    expected = {'exponential': 'g(Q)=γe^{λQ}', 'power': 'g(Q)=γQ^{λ}', 'logarithmic': 'g(Q)=γln(1+λQ)'}
    for family in expected:
        idx, text = selected[family]
        if expected[family] not in compact(text):
            raise ValueError('Appendix B family differs from implemented formula: '+family)
        gamma_expr = re.search(r'γ=([^，]+)', text)[1]
        lam_expr = re.search(r'λ=([^；。]+)', text)[1]
        official['quality_costs'][family] = dict(gamma=numeric_expression(gamma_expr),
              **{'lambda': numeric_expression(lam_expr)}, source_paragraph_index_zero_based=idx,
              gamma_unit='FLOPs/token (dimensionally implied by official total budget)')
    save_json(out/'official_cost_parameters.json', official)
    evidence = []
    for key, (i,text) in selected.items():
        evidence.append(f'### {key}\n\n来源：`{docx.name}`，`word/document.xml` 第 {i} 号段落（零起始）。\n\n```text\n{text}\n```')
    (out/'problem3_formula_audit.md').write_text(
        '# 问题三官方公式核验\n\n直接解析原始 OMML，上下标保留；未修改 Word。\n\n'
        '## A 官方明确给出\n\n'+'\n\n'.join(evidence)+
        '\n\n## B 解析推导\n\nC_train=C_attn 在 N,D>0 时给出 L_crit=6/eta。'
        'CQ 增量在 Q=Q0 为零；三种 g 在 Q∈[Q0,1] 均递增且端点有限。'
        'gamma 必须解释为每 token 的 FLOPs 代理，lambda_Q 是无量纲形状参数；它与冻结 lambda_p(N) 不同。'
        '\n\n## C 研究者设定\n\nQ0 的聚合/跨域代理映射、主支持域交集、外推域、对数预算网格、经验配方候选集及未来转移判据属于建模选择。'
        '架构最大窗口是可行场景依据，不是每次训练真实序列长度；跨架构与 N 的自由组合也是简化代理假设。'
        '\n\n## D 旧草稿不继承\n\n未读取旧优化答案。Q0=0.8、质量成本上限 M、旧 N*/D*/Q*、旧转移预算均不继承。'
        'eta=2e-4 和三档预算此次由官方 XML 独立恢复，因此可采用；不是沿用旧稿。'
        '对数函数不能替换为 -ln(1-Q)；三类函数没有题给总成本上限 M。\n', encoding='utf-8')
    import pypdf
    logging.getLogger('pypdf').setLevel(logging.CRITICAL)
    pdf = next(root.glob('*.pdf'))
    reader = pypdf.PdfReader(pdf)
    # Only relevant complete pages; malformed CMaps in source font raise warnings,
    # but relevant C7/schema/unit text is extractable and checked below.
    pages = {str(i+1): reader.pages[i].extract_text() for i in (3,4,10,11,12)}
    text = '\n'.join(pages.values())
    if 'max_position_embeddings' not in text or 'N_params_B' not in text or 'D_tokens_B' not in text:
        raise ValueError('Official PDF relevant schema/unit extraction failed')
    (out/'official_data_description_excerpt.txt').write_text('\n\n'.join('PAGE '+k+'\n'+v for k,v in pages.items()), encoding='utf-8')
    return official, [docx, pdf]


def diagnose_context(root, out, official):
    """Read C7 actual columns first and select only observed context values."""
    path = root/'real_attachments/C_efficiency_evolution/model_architecture_metadata.csv'
    c7 = pd.read_csv(path, float_precision='round_trip')
    required = {'model_name','n_layers','n_heads','d_model','max_position_embeddings'}
    if not required.issubset(c7.columns):
        raise ValueError('C7 missing required actual fields: '+str(required-set(c7.columns)))
    vals = pd.to_numeric(c7.max_position_embeddings, errors='coerce')
    valid = vals[np.isfinite(vals) & (vals>0) & (vals==np.floor(vals))]
    if valid.empty:
        raise ValueError('No finite positive integer context windows')
    qlevels = [0,.01,.05,.10,.25,.50,.75,.90,.95,.99,1]
    names = ['min','P01','P05','P10','P25','median','P75','P90','P95','P99','max']
    stats = dict(zip(names, valid.quantile(qlevels).to_list()))
    stats.update(rows=len(c7), valid_count=len(valid), invalid_count=len(c7)-len(valid),
                 unique_value_count=valid.nunique(), mean=valid.mean(), std=valid.std(ddof=1))
    save_csv(out/'C7_context_summary.csv', [stats])
    freq = valid.value_counts().sort_index().rename_axis('L_ctx').reset_index(name='support_count')
    freq['L_ctx'] = freq.L_ctx.astype(int)
    freq['support_fraction'] = freq.support_count/len(valid)
    save_csv(out/'C7_context_frequency.csv', freq)
    save_csv(out/'C7_field_diagnosis.csv', [{'field': key, 'dtype': str(c7[key].dtype),
        'missing_count': int(c7[key].isna().sum()), 'unique_value_count': c7[key].nunique()} for key in c7])
    save_csv(out/'C7_architecture_inventory.csv', c7)
    scenarios = freq.copy()
    scenarios['scenario_type'] = ['short' if x==valid.min() else 'medium_median' if x==valid.median()
                                  else 'long' if x==valid.max() else 'observed_frequent' for x in scenarios.L_ctx]
    scenarios['evidence_source'] = path.relative_to(root).as_posix()
    scenarios['attention_to_training_ratio'] = official['eta']*scenarios.L_ctx/official['training_coefficient']
    save_csv(out/'context_scenarios.csv', scenarios)
    crit = official['training_coefficient']/official['eta']
    save_csv(out/'context_critical_analysis.csv', [dict(eta_scheme='official_main', eta=official['eta'],
        L_ctx_crit=crit, within_C7_observed_range=bool(valid.min()<=crit<=valid.max()),
        exact_C7_value=bool((valid==crit).any()), models_above_critical=int((valid>crit).sum()),
        models_equal_critical=int((valid==crit).sum()), models_below_critical=int((valid<crit).sum()),
        valid_models=len(valid), fraction_above=float((valid>crit).mean()), evidence='Official Word Question 3')])
    (out/'C7_diagnosis.md').write_text('# C7 真实结构诊断\n\n'
        f'文件：`{path.relative_to(root).as_posix()}`。实际 {len(c7)} 行 × {len(c7.columns)} 列。\n\n'
        f'字段：{", ".join(c7.columns)}。模型名重复 {c7.model_name.duplicated().sum()}；完全重复 {c7.duplicated().sum()}。'
        f'总缺失单元格 {c7.isna().sum().sum()}；有效上下文 {len(valid)}。\n\n'
        'hidden_size 对应磁盘 d_model。没有独立参数量 N 字段，不从模型名猜参数量；没有明确 training tokens 字段。'
        'training_data_TB 字段含有数值，但单位语义不能由列名可靠判定为 token，不能转换成 D_tokens_B。'
        '参数量和训练数据上下界来自冻结问题二，不从该列反推。\n\n'+table(pd.DataFrame([stats]))+
        '\n\n分位数采用线性插值，分位数可能不是实际窗口；场景只取实际观测离散值。\n\n'+table(scenarios)+
        f'\n\neta={official["eta"]} 是题给简化代理，不需要也没有用 C7 重新估计。单位为每参数、每训练 token、每窗口位置的 FLOPs 系数。'
        f'临界窗口 {crit:g} 在观测最小最大值之间，但不是观测离散值；超临界模型 {(valid>crit).sum()}。'
        'C7 频数描述本附件，不代表模型市场总体分布；最大支持窗口不证明所有训练序列都达到该长度。\n', encoding='utf-8')
    return c7, scenarios, stats, [path]


def frozen_inputs(root, out):
    """Load all coefficients at saved precision; do not fit or mutate prior results."""
    path = root/'outputs/scaling/phase4/generalized_scaling_final.json'
    frozen = json.loads(path.read_text(encoding='utf-8'))
    from scaling.mixture_transfer import frozen_metadata, predict_ilr
    meta = frozen_metadata(root)
    pref = np.array(frozen['reference']['p_ref'])
    if len(pref)!=17 or not np.isfinite(pref).all() or (pref<=0).any() or not np.isclose(pref.sum(),1,atol=1e-14,rtol=0):
        raise ValueError('Frozen reference recipe invalid')
    pred = float(predict_ilr(np.log(pref)[None,:]@np.asarray(meta['ilr_basis']).T, meta)[0])
    if not np.isclose(pred,frozen['reference']['M_ref'],atol=1e-12,rtol=0):
        raise ValueError('Frozen M_ref disagrees with frozen ILR model')
    rows=[]
    for module, key, evidence, support in [('L0','classical','Class A','B1; B3 interpolation; B4/B5 comparability-limited'),
            ('Delta_Q','quality','Class B','B6 semisynthetic local; B7 difference validation failed')]:
        for name,value in frozen[key].items():
            rows.append(dict(module=module, parameter=name, value=json.dumps(value,ensure_ascii=False),
                             evidence_class=evidence, source=path.relative_to(root).as_posix(), support=support))
    rows.extend([dict(module='M(p)',parameter='frozen_model_metadata',value=json.dumps(meta,ensure_ascii=False),
                     evidence_class='Class C',source='outputs/mixture/final_model_metadata.json',support='ILR first-order ElasticNet; no fit'),
                 dict(module='M(p)',parameter='reference',value=json.dumps(frozen['reference'],ensure_ascii=False),evidence_class='Class C',
                      source=path.relative_to(root).as_posix(),support='fixed p_ref and M_ref')])
    for knot in frozen['lambda_knots']:
        rows.append(dict(module='lambda_p',parameter='knot_'+str(knot['N_params_B']),value=json.dumps(knot),
                         evidence_class='Class C' if knot['evidence']=='observed' else 'Class D',
                         source=path.relative_to(root).as_posix(),support=knot['evidence']))
    save_csv(out/'frozen_objective_summary.csv',rows)
    return frozen,meta,[path,root/'outputs/mixture/final_model_metadata.json', root/'outputs/mixture/stage2/final_model_policy.json']


def baseline_quality(root,out,frozen):
    """Compare robust frozen-score aggregations; expose rather than hide proxy mass."""
    base=root/'outputs/quality/domain_quality_summary_A1.csv'
    domain=pd.read_csv(base,float_precision='round_trip')
    samples=root/'outputs/quality/A1_sample_quality.csv'
    values=pd.read_csv(samples,usecols=['Q'],float_precision='round_trip').Q.to_numpy()
    if len(values)!=int(domain.n.sum()) or not np.isfinite(values).all():
        raise ValueError('Frozen A1 Q records and domain count disagree')
    map_path=root/'outputs/mixture/stage2/domain_quality_conflict_mapping.csv'
    alt_path=root/'outputs/mixture/stage2/alternative_domain_quality_mapping.csv'
    names=frozen['reference']['domain_order']; p=np.array(frozen['reference']['p_ref'])
    mapping=pd.read_csv(map_path,float_precision='round_trip').set_index('mixture_domain').loc[names].reset_index()
    alt=pd.read_csv(alt_path,float_precision='round_trip').set_index('mixture_domain').loc[names]
    actual=domain.set_index('domain').Q_domain
    if not np.allclose(mapping.Q_domain,[actual[x] for x in mapping.quality_domain],rtol=0,atol=1e-15):
        raise ValueError('Frozen mapping/domain qualities disagree')
    proxy=mapping.researcher_assumption.to_numpy(bool)
    proxy_mass=float(p[proxy].sum()); q0=float(p@mapping.Q_domain.to_numpy())
    q_alt=float(p@alt.Q_domain.to_numpy())
    direct_mass=float(p[~proxy].sum()); known=float(p[~proxy]@mapping.Q_domain.to_numpy()[~proxy])
    lower=known+proxy_mass*float(actual.min()); upper=known+proxy_mass*float(actual.max())
    mapping['p_ref']=p; mapping['quality_contribution']=p*mapping.Q_domain
    save_csv(out/'baseline_quality_mapping_details.csv',mapping)
    grouped=mapping.groupby('quality_domain').p_ref.sum().reindex(actual.index,fill_value=0)
    save_csv(out/'baseline_quality_domain_weights.csv',pd.DataFrame(dict(
        quality_domain=actual.index,p_ref_mapped_weight=grouped.to_numpy(),Q_domain=actual.to_numpy(),
        quality_contribution=grouped.to_numpy()*actual.to_numpy())))
    rows=[dict(candidate='Q0_A',Q0=float(trim_mean(values,.1)),definition='A1全部样本Q的双侧各10%截尾均值',
          source=samples.relative_to(root).as_posix(),mapping_needed=False,uncertainty='按域抽样结构；非真实训练语料token权重',recommended=False),
          dict(candidate='Q0_B',Q0=q0,definition='冻结p_ref点乘冻结17→7代理映射Q_domain',
          source=map_path.relative_to(root).as_posix(),mapping_needed=True,
          uncertainty=f'依赖代理映射的配比权重{proxy_mass:.10%}; 替代映射={q_alt:.17g}; 未识别域范围包络=[{lower:.17g},{upper:.17g}]，不是统计CI',recommended=True),
          dict(candidate='Q0_C',Q0=float(actual.median()),definition='七领域Q_domain中位数（等域敏感性）',
          source=base.relative_to(root).as_posix(),mapping_needed=False,uncertainty='忽略p_ref权重；仅敏感性',recommended=False),
          dict(candidate='Q0_B_alternative',Q0=q_alt,definition='冻结替代映射下p_ref加权质量',source=alt_path.relative_to(root).as_posix(),
          mapping_needed=True,uncertainty='代理映射结构敏感性，非独立观测',recommended=False),
          dict(candidate='Q0_B_direct_only',Q0=known/direct_mass,definition='仅直接/近直接映射，按覆盖质量质量归一化',
          source=map_path.relative_to(root).as_posix(),mapping_needed=True,
          uncertainty=f'覆盖配比权重{direct_mass:.10%}，不是完整p_ref基线',recommended=False)]
    save_csv(out/'baseline_quality_candidates.csv',rows)
    info=dict(primary='Q0_B',Q0=q0,proxy_mass=proxy_mass,alternative=q_alt,
              proxy_range_envelope=[lower,upper],envelope_is_confidence_interval=False,
              A1_sample_count=len(values),A1_sample_Q_range=[float(values.min()),float(values.max())],
              A1_domain_Q_range=[float(actual.min()),float(actual.max())],
              choice_status='recommended_conditional_on_frozen_proxy_mapping_not_direct_measurement')
    save_json(out/'baseline_quality_provenance.json',info)
    return info,pd.DataFrame(rows),[base,samples,map_path,alt_path]


def decision_domains(root,out,frozen,qinfo):
    """Compare sources before setting a conservative joint numeric support box."""
    file_path=root/'outputs/scaling/phase1/data_file_summary.csv'
    source=pd.read_csv(file_path,float_precision='round_trip')
    rows=[]
    for dataset,g in source.groupby('dataset',sort=False):
        rows.append(dict(source=dataset,N_min_B=float(g.N_min.min()),N_max_B=float(g.N_max.max()),
                         D_min_B=float(g.D_min.min()),D_max_B=float(g.D_max.max()),
                         main_evidence=dataset=='B1',notes='同源插值非新增证据' if dataset=='B3' else
                         '半合成；不扩主域' if dataset=='B2' else '跨族/文献绝对Loss口径未确认；只列范围' if dataset in ('B4','B5') else '真实主拟合轨迹'))
    b1=source[source.dataset=='B1'].iloc[0]; qs=frozen['quality_numeric_support']
    nlo=max(float(b1.N_min),qs['N_min_B']); nhi=min(float(b1.N_max),qs['N_max_B'])
    dlo=max(float(b1.D_min),qs['D_min_B']); dhi=min(float(b1.D_max),qs['D_max_B'])
    schema_path=root/'outputs/scaling/phase4/B9_B10_schema.json'
    schema=json.loads(schema_path.read_text(encoding='utf-8'))
    for dataset in ('B9','B10'):
        s=schema[dataset];rows.append(dict(source=dataset,N_min_B=s['N_min'],N_max_B=s['N_max'],D_min_B=s['D_min'],
            D_max_B=s['D_max'],main_evidence=False,notes='元数据；D=0不是有效下界，不填补；总/激活参数未统一' if dataset=='B9' else '估算Loss，非直接观测'))
    rows.append(dict(source='B6_frozen_numeric_support',N_min_B=qs['N_min_B'],N_max_B=qs['N_max_B'],
        D_min_B=qs['D_min_B'],D_max_B=qs['D_max_B'],main_evidence=False,notes='质量模块局部半合成数值支持，非真实联合观测'))
    save_csv(out/'evidence_support_comparison.csv',rows)
    bounds=[]
    def add(scheme,var,lo,hi,unit,extrap,src,notes):
        bounds.append(dict(scheme=scheme,variable=var,lower=lo,upper=hi,unit=unit,extrapolation=extrap,evidence_source=src,notes=notes))
    add('main','N',nlo,nhi,'billion parameters',False,'B1 ∩ B6 numeric support','连续矩形是插值可行域，不是所有点都观测；P1在N>1B使用Class D配比转移')
    add('main','D',dlo,dhi,'billion tokens',False,'B1 ∩ B6 numeric support','D下界提高至质量模块数值下界；非改动Q2参数')
    add('classical_only_reference','N',float(b1.N_min),float(b1.N_max),'billion parameters',False,'B1','只用于对照，不是第二套主域')
    add('classical_only_reference','D',float(b1.D_min),float(b1.D_max),'billion tokens',False,'B1','D<10时质量模块外推，不能称联合支持')
    add('extrapolation','N',nlo,max(schema['B9']['N_max'],schema['B10']['N_max']),'billion parameters',True,'B1 + B9/B10 envelope','跨缺口连续包络；没有密集大模型观测验证')
    add('extrapolation','D',dlo,max(schema['B9']['D_max'],schema['B10']['D_max']),'billion tokens',True,'B1 + positive B9/B10 upper extent','保留主下界；D=0未知不使用')
    add('main','Q',qinfo['Q0'],1.,'dimensionless',True,'Q1 baseline + official Q≤1','整个区间不是全外推；上端超出Q1实际质量；具体子区间另表')
    add('extrapolation','Q',qinfo['Q0'],1.,'dimensionless',True,'same frozen Q0','Q0和质量模块均不重新标定')
    save_csv(out/'decision_bounds.csv',bounds)
    knots=sorted(set([qinfo['Q0'],1.,*qinfo['A1_sample_Q_range'],*qinfo['A1_domain_Q_range'],qs['Q_min'],qs['Q_max']]))
    support=[]
    for a,b in zip(knots[:-1],knots[1:]):
        if b<=qinfo['Q0'] or a>=1:
            continue
        a=max(a,qinfo['Q0']);b=min(b,1.);m=(a+b)/2
        support.append(dict(Q_lower=a,Q_upper=b,within_A1_sample_range=qinfo['A1_sample_Q_range'][0]<=m<=qinfo['A1_sample_Q_range'][1],
            within_A1_domain_range=qinfo['A1_domain_Q_range'][0]<=m<=qinfo['A1_domain_Q_range'][1],
            within_B6_numeric_range=qs['Q_min']<=m<=qs['Q_max'],
            direct_joint_quality_validation=False,notes='数值区间覆盖不等于直接实验质量或联合N-D-Q-p验证'))
    save_csv(out/'quality_support_intervals.csv',support)
    return dict(N_min_B=nlo,N_max_B=nhi,D_min_B=dlo,D_max_B=dhi,Q_min=qinfo['Q0'],Q_max=1.),pd.DataFrame(bounds),[file_path,schema_path]


def mixture_sets(root,out,frozen,meta,cfg):
    """Use already saved model-input compositions, without repeating preprocessing."""
    from scaling.mixture_transfer import predict_ilr
    names=frozen['reference']['domain_order'];frames=[];paths=[]
    for label in ('train_1M','test_1M','test_60M','test_1B'):
        path=root/f'outputs/mixture/{label}_closed_replaced_composition.csv';paths.append(path)
        f=pd.read_csv(path,float_precision='round_trip')
        p=f[names].to_numpy()
        if (p<=0).any() or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-14,rtol=0):
            raise ValueError('Frozen recipe composition invalid: '+label)
        f['source_dataset']=label
        f['source_path']=path.relative_to(root).as_posix()
        f['signature']=[json.dumps(np.round(row,cfg['mixture_empirical_signature_decimal_places']).tolist(),separators=(',',':')) for row in p]
        f['M_p']=predict_ilr(np.log(p)@np.asarray(meta['ilr_basis']).T,meta)
        frames.append(f)
    combined=pd.concat(frames,ignore_index=True)
    save_csv(out/'empirical_recipe_membership.csv',combined)
    records=[]
    for i,(signature,g) in enumerate(combined.groupby('signature',sort=False)):
        if np.ptp(g.M_p.to_numpy())>1e-10:
            raise ValueError('Recipe deduplication tolerance merges distinct predictions')
        row={'candidate_id':f'P1_{i:04d}','source_members':json.dumps(g[['source_dataset','index']].to_dict('records')),
             'source_record_count':len(g),'composition_space':'frozen_closed_zero_replaced',
             'M_p':float(g.M_p.iloc[0]),'signature_decimal_places':cfg['mixture_empirical_signature_decimal_places']}
        row.update({name:float(g[name].iloc[0]) for name in names})
        records.append(row)
    save_csv(out/'empirical_recipe_candidates.csv',records)
    counts=combined.groupby('source_dataset').size().to_dict()
    (out/'mixture_feasible_sets.md').write_text('# 领域配比可行集\n\n'
        f'P0：固定冻结 p_ref=A4_ILR_center，M_ref={frozen["reference"]["M_ref"]:.17g}；17分量见 frozen_objective_summary.csv。\n\n'
        f'P1：读取已冻结训练/检验的模型输入配方 {counts}；共 {len(combined)} 条来源记录，按17维配比{cfg["mixture_empirical_signature_decimal_places"]}位签名合并为 {len(records)} 个候选。'
        '配方ID不跨表当作配比相同证据；membership表保留全部来源，无修改旧数据。'
        '仅读取保存的 closed_replaced_composition 和冻结模型；不读取 A4–A15 原始文件，不再拟合/重复闭合与零替换。\n\n'
        '重要语义：这些是已冻结的闭合、零替换后配比，来源于真实训练/检验配方，但不是原始结构零比例。'
        'P1在冻结模型输入空间选择，不能把替换质量解释为测得非零训练比例；物理配方部署前须回溯对应原始记录。'
        'M(p) 直接由保存的 ILR+一阶ElasticNet 系数评价。候选集中不包含新邻域采样，不包含估算10B/70B额外配方。'
        '检验配方曾参与冻结阶段验证，因此这些候选的后续优化属于探索性选择，不可作为新的独立盲测。\n\n'
        'P2：Δ17={p≥0,Σp=1}，仅数学敏感性；边界需执行冻结零替换约定，不能直接取ln0。'
        '单域100%和极端角点没有充分支持，不作为实际建议。\n\n'
        '主分析P0+P1；P1先在每个有限候选p上求三维N-D-Q连续问题，再做外层选择。'
        '本阶段只建立集合，不调用优化器。Q0统一沿用参考基线；p改变而Q0不变属于可比较情景假设，'
        '不表示所有经验配方都测得同一原始质量；Q与p能独立改变、加工同一D的假设须保留。\n',encoding='utf-8')
    return dict(source_counts={k:int(v) for k,v in counts.items()},source_records=len(combined),unique_candidates=len(records)),paths


def budget_scenarios(out,official,cfg,bounds,contexts):
    """List budgets and analytic minimum-cost feasibility, never solve allocations."""
    budgets=official['budgets_FLOPs'];e=cfg['budget_extension_decades_per_side']
    grid=np.geomspace(min(budgets)/10**e,max(budgets)*10**e,cfg['budget_log_grid_count'])
    rows=[dict(scenario_id=label,C_budget_FLOPs=value,scenario_type='official_recommended',
               evidence_source='Official Word Question 3 OMML',researcher_assumption=False)
          for label,value in zip(('low','medium','high'),budgets)]
    rows.extend(dict(scenario_id=f'log_grid_{i:02d}',C_budget_FLOPs=float(x),scenario_type='research_log_grid',
                evidence_source='official extrema ± configured log decades',researcher_assumption=True) for i,x in enumerate(grid))
    save_csv(out/'compute_budget_scenarios.csv',rows)
    checks=[]
    for row in rows:
        for l in contexts.L_ctx:
            lower=(official['training_coefficient']+official['eta']*l)*bounds['N_min_B']*bounds['D_min_B']*1e18
            checks.append(dict(scenario_id=row['scenario_id'],C_budget_FLOPs=row['C_budget_FLOPs'],L_ctx=int(l),
                minimum_total_cost_FLOPs=lower,main_box_feasible=row['C_budget_FLOPs']>=lower,
                basis='monotone costs: Q=Q0 has zero incremental quality cost; no objective optimization'))
    save_csv(out/'budget_feasibility_screen.csv',checks)
    return pd.DataFrame(rows)


def numeric_units(out,official,qinfo,contexts):
    """Twelve analytic arithmetic cases, independently evaluated using Decimal."""
    rows=[];q0=qinfo['Q0'];ctx=contexts.L_ctx.to_list()
    for i in range(12):
        family=list(official['quality_costs'])[i%3]
        n=[.1,1.,7.,12.][i//3];d=[10.,100.,300.,600.][i//3]
        l=int(ctx[i%len(ctx)]);q=[q0,(q0+1)/2,1.][i%3]
        tr,qu,att=cost_components(n,d,q,q0,l,official,family)
        pars=official['quality_costs'][family]
        with localcontext() as dec:
            dec.prec=50
            nd,dd=Decimal(str(n))*Decimal('1e9'),Decimal(str(d))*Decimal('1e9')
            qd,q0d=Decimal(str(q)),Decimal(str(q0));gamma=Decimal(str(pars['gamma']));lam=Decimal(str(pars['lambda']))
            manual_t=Decimal(str(official['training_coefficient']))*nd*dd
            manual_a=Decimal(str(official['eta']))*nd*dd*Decimal(l)
            if family=='exponential':
                h=(lam*qd).exp()-(lam*q0d).exp()
            elif family=='power':
                h=qd**int(lam)-q0d**int(lam)
            else:
                h=(1+lam*qd).ln()-(1+lam*q0d).ln()
            manual_q=dd*gamma*max(Decimal(0),h)
        passed=all(np.isclose(float(a),float(b),rtol=2e-14,atol=1e-5) for a,b in zip((tr,qu,att),(manual_t,manual_q,manual_a)))
        if not passed:
            raise ValueError(f'Unit arithmetic mismatch: case {i}')
        rows.append(dict(case_id=i+1,family=family,N_B=n,D_B=d,L_ctx=l,Q=q,Q0=q0,
            actual_N=format(nd,'f'),actual_D=format(dd,'f'),program_train_FLOPs=tr,manual_decimal_train=str(manual_t),
            program_quality_FLOPs=float(qu),manual_decimal_quality=str(manual_q),program_attention_FLOPs=att,
            manual_decimal_attention=str(manual_a),passed=passed))
    save_csv(out/'unit_numeric_checks.csv',rows)
    md='# 算力单位核验\n\nN_B=N/10^9，D_B=D/10^9。目标函数读取问题二参数时仍用 N_B、D_B；成本函数输入B单位后显式换算。\n\n'
    md+='- C_train=6 N_B D_B ×10^18 FLOPs。\n- C_attn=eta N_B D_B L_ctx ×10^18 FLOPs。\n- CQ=D_B ×10^9 [g(Q)-g(Q0)]+ FLOPs。\n\n'
    md+='N 是参数个数，D 是训练token数，L_ctx 是窗口位置数；6为训练运算近似系数，eta为注意力代理系数。'
    md+='Q、Q0、p及lambda_Q无量纲；g和gamma必须按FLOPs/token解释，才能与题目FLOPs预算相加。'
    md+='附录未另列gamma单位；此单位是由题给CQ=DΔg和预算单位推得（B类），不是新增数值校准。'
    md+='不把gamma当货币，不乘N、不乘第二个D、不乘自设总成本上限M。'
    md+='成本CQ与目标函数系数C_Q虽然名字相近，单位与角色完全不同。\n\n'
    md+='## 独立算式与数值核验\n\n12例为公式算术校验，不是实验观测或最优解；包含超出主域的单位压力样例。'
    md+='“手工式”路径使用实际N=10^9 N_B与D=10^9 D_B，以及50位Decimal直接指数/幂/对数减法；'
    md+='程序路径使用B单位简式和数值稳定expm1/log1p。双方独立计算后比较，容差2e-14。\n\n'
    for row in rows:
        md+=f'例{row["case_id"]}：N={row["actual_N"]}，D={row["actual_D"]}，L={row["L_ctx"]}，Q={row["Q"]:.10g}，{row["family"]}。'
        md+=f'训练6×{row["actual_N"]}×{row["actual_D"]}={row["program_train_FLOPs"]:.12g}；'
        md+=f'注意力{official["eta"]}×{row["actual_N"]}×{row["actual_D"]}×{row["L_ctx"]}={row["program_attention_FLOPs"]:.12g}；'
        md+=f'质量D×[g(Q)-g(Q0)]+={row["program_quality_FLOPs"]:.12g} FLOPs；独立核验通过。\n\n'
    (out/'unit_audit.md').write_text(md,encoding='utf-8')
    return rows


def quality_cost_report(out,official,qinfo):
    """Document the unmodified official families and their analytic derivatives."""
    q0=qinfo['Q0'];rows=[]
    shapes={'exponential':('γ exp(λ_Q Q)','γ λ_Q exp(λ_Q Q)','凸'),
            'power':('γ Q^λ_Q','γ λ_Q Q^(λ_Q-1)','凸'),
            'logarithmic':('γ ln(1+λ_Q Q)','γ λ_Q/(1+λ_Q Q)','凹，边际成本递减')}
    for family,pars in official['quality_costs'].items():
        g,derivative,shape=shapes[family]
        rows.append(dict(family=family,g_formula=g,gamma=pars['gamma'],lambda_Q=pars['lambda'],
              gamma_unit='FLOPs/token',marginal_formula='D × '+derivative,
              cost_at_Q0_per_token=float(quality_increment_per_token(q0,q0,family,pars['gamma'],pars['lambda'])),
              cost_at_Q1_per_token=float(quality_increment_per_token(1.,q0,family,pars['gamma'],pars['lambda'])),
              endpoint_finite=True,shape=shape))
    save_csv(out/'quality_cost_function_parameters.csv',rows)
    (out/'quality_cost_functions.md').write_text('# 官方质量成本函数\n\n'
        '统一 CQ(D,Q;Q0,θ)=D[gθ(Q)-gθ(Q0)]+，D为实际token数，θ=(gamma,lambda_Q)。'
        f'推荐Q0={q0:.17g}；三个官方参数全部从Word OMML读取。\n\n'+table(pd.DataFrame(rows))+
        '\n\n在Q>Q0时 dCQ/dQ=D g′(Q)>0；Q<Q0时为零；Q=Q0处有左导数0与右导数Dg′(Q0)，'
        '主可行域采用右导数。对D的导数为[g(Q)-g(Q0)]+，非负；三类均显式乘同一训练D。'
        'Q=Q0增量为零；Q→1时三类均有限，不排除Q=1。'
        '对数渐进型是ln(1+10Q)，不是-ln(1-Q)，也不是越趋完美越发散的模型。\n\n'
        '指数/幂的质量边际成本递增，对数型边际成本递减，因此后续可能产生不同活跃集，不能先套同一凸性或内点结论。'
        '本阶段三类均保留，未依据旧最优结果选主成本。曲线以CQ/D和(dCQ/dQ)/D显示，避免主观选D。'
        '没有额外最大值M；Q=1有限上界是由给定函数和Q0解析导出的，不是新的调参上限。\n',encoding='utf-8')
    return pd.DataFrame(rows)


def math_reports(out,official,bounds):
    """Formulate P0/P1 and future diagnostics only; retain budget inequality."""
    (out/'optimization_formulation.md').write_text(r'''# 优化数学形式

所有目标参数、Q/K/M(p)/lambda_p均冻结。N_B、D_B为十亿单位，物理N=10^9N_B、D=10^9D_B。

L_gen=E+A N_B^(-alpha)+B D_B^(-beta)+C_Q (N_B/N0_B)^(-u_Q)(-lnQ)+lambda_p(N_B)[M(p)-M_ref]。

P0：min_{N_B,D_B,Q} L_gen(N_B,D_B,Q,p_ref)。约束

(6+eta L_ctx)N_B D_B 10^18 + D_B 10^9 [g(Q)-g(Q0)]+ ≤ C_budget，
N_min≤N_B≤N_max，D_min≤D_B≤D_max，Q0≤Q≤1。

L_ctx为外生C7离散场景，成本函数类型与预算也为外生场景。P1对每个已保存经验候选p建立同样三维问题，再外层比较候选；本阶段不求解。P2只作为后续敏感性。

## 单调性与预算松弛

P0的DeltaM=0：
dL/dD_B=-beta B D_B^(-beta-1)<0；
dL/dQ=-C_Q (N_B/N0_B)^(-u_Q)/Q<0；
dL/dN_B=-alpha A N_B^(-alpha-1)-u_Q DeltaQ/N_B<0。

成本对N_B的导数=(6+eta L_ctx)D_B 10^18>0；对D_B为(6+eta L_ctx)N_B 10^18+10^9Δg>0；
对Q在Q≥Q0取右导数D_B10^9 g′(Q)>0。
因此可行域内若有任一变量尚未触达上界且预算严格松弛，就可以小幅提高该变量降低目标。
P0最优预算必须绑定，或全部N/D/Q达到各自上界。全部上界仍成本小于预算时允许剩余预算，不能硬编码等号。

P1中记DeltaM=M(p)-M_ref，s=d lambda_p/d lnN_B：
dL/dN_B=[-alpha A N_B^(-alpha)-u_Q DeltaQ+s DeltaM]/N_B。
该导数未保证为负，必须保留冻结分段lambda并检查左右导数；在节点主约定为右导数。
不过D/Q在固定p下仍严格递减：若任意一个未达到上界，预算松弛仍可被排除。
若D和Q已触上界，N局部不单调、位于节点或达到上界，可存在预算松弛。
质量模块外推不自动改变公式的单调符号，但降低上述数学结论的经验解释可信度。

## 后续数值求解要求

对每种成本/上下文/预算/域方案先检查可行性，不以调整边界或Q0掩盖不可行；记录不等式松弛。
分段lambda、对数型凹质量成本、有限候选p可能导致非凸/多分支，后续需比较边界和分段解，而非假定一个KKT内点就是全局解。
在质量模块局部证据外，最优解只解释为冻结模型情景结果。P1在N>1B时保留Class D标记。
''',encoding='utf-8')
    (out/'structural_shift_definition.md').write_text(r'''# 结构性转移定义 仅建立判据

没有计算最优轨迹或转移预算。以下均为PHASE 2待应用定义。

1. 活跃集 A(C)：包含达到N/D/Q上下界的变量、绑定预算、以及P1选中配方的身份。
归一化距离与预算松弛均≤1e-6时暂标活跃（研究者数值容差）；后续用1e-5/1e-7检验容差稳定性。
连续网格上身份/约束变更为候选，须局部加密与独立求解确认，不将网格单次翻转直接称转移。

2. 成本份额 s_train=C_train/C_budget，s_Q=CQ/C_budget，s_attn=C_attn/C_budget；预算松弛时三者和≤1而非必为1。
同时记录未使用份额1-C_total/C_budget，并可另列相对于实际消耗的份额，不能混用分母。
同一L_ctx与eta下C_attn/C_train=eta L_ctx/6恒定，因此这两类份额彼此固定比例，不应虚构独立的注意力/训练相互转移。
“注意力主导”是否成立由外生上下文决定，不随预算单独变化；质量份额、上下界和配方选择仍可能改变。

3. 最优决策弹性：e_N=dlnN*/dlnC，e_D=dlnD*/dlnC，e_Q=dQ*/dlnC。
注意e_Q不是dlnQ/dlnC。对连续预算用中心差分或局部分段回归；lambda节点另列左右斜率。
斜率/份额趋势的分段改变为候选，要求至少两类证据（如活跃集+弹性）、网格加密稳定和目标值/可行性核验。

后续识别方法：对数预算网格逐场景求解、候选附近加密；比较连续与一断点分段趋势的残差/BIC，
并对预算网格、数值容差、成本函数、Q0代理映射、主/外推域做稳定性对照。
这些是研究者拟定的方法；当前没有拟合断点，没有指定转移预算或最优N*/D*/Q*。
''',encoding='utf-8')


def figures(root,out,contexts,c7,official,qinfo,budgets,bounds):
    """Six scientific figures from observed context and audited formulas only."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    figdir=root/'outputs/figures/question3_phase1';figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    manifest=[]
    def finish(fig,name,caption):
        path=figdir/name;fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)
        manifest.append(dict(figure=path.relative_to(root).as_posix(),caption=caption))
    fig,ax=plt.subplots(figsize=(7,4))
    sortedctx=np.sort(c7.max_position_embeddings.to_numpy())
    ax.step(sortedctx,np.arange(1,len(sortedctx)+1)/len(sortedctx),where='post')
    ax.set(xscale='log',xlabel='Maximum supported context (positions)',ylabel='Empirical cumulative fraction',title=f'C7 context distribution (n={len(c7)})')
    ax.axvline(6/official['eta'],ls='--',color='gray',label='Attention = training');ax.legend()
    finish(fig,'01_C7_context_distribution.png','C7实测上下文经验分布')
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(contexts.L_ctx.astype(str),contexts.support_count,color='#4477AA')
    for i,n in enumerate(contexts.support_count):ax.text(i,n+.2,str(n),ha='center')
    ax.set(xlabel='Observed maximum context (positions)',ylabel='Model count',title='C7 observed context frequencies')
    ax.set_ylim(0,contexts.support_count.max()*1.18)
    finish(fig,'02_C7_context_frequency.png','C7实际离散窗口频数')
    q=np.linspace(qinfo['Q0'],1,401)
    for derivative,name,title,ylabel in [(False,'03_quality_cost_functions.png','Official incremental quality costs','C_quality / D (FLOPs per token)'),
             (True,'04_marginal_quality_costs.png','Official marginal quality costs','(dC_quality/dQ) / D (FLOPs per token per Q)')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for family,pars in official['quality_costs'].items():
            func=marginal_quality_per_token if derivative else quality_increment_per_token
            ax.plot(q,func(q,qinfo['Q0'],family,pars['gamma'],pars['lambda']),label=family)
        ax.set(xlabel='Target quality Q',ylabel=ylabel,title=title);ax.legend()
        finish(fig,name,title+'，不是最优结果')
    fig,ax=plt.subplots(figsize=(7,3));g=budgets[budgets.scenario_type=='research_log_grid'];o=budgets[budgets.scenario_type=='official_recommended']
    ax.scatter(g.C_budget_FLOPs,np.zeros(len(g)),s=15,label='41-point research log grid')
    ax.scatter(o.C_budget_FLOPs,np.ones(len(o)),marker='D',label='Official recommended budgets')
    for row in o.itertuples():ax.annotate(f'{row.scenario_id}: {row.C_budget_FLOPs:.0e}',(row.C_budget_FLOPs,1),xytext=(0,12),textcoords='offset points',ha='center')
    ax.set(xscale='log',xlabel='Compute budget (FLOPs)',yticks=[],ylim=(-.5,1.6),title='Budget scenarios only; no allocation solved')
    ax.legend(loc='center left',bbox_to_anchor=(0.01,.48),fontsize=8)
    finish(fig,'05_budget_scenarios.png','官方三档与研究预算网格')
    fig,ax=plt.subplots(figsize=(8,5))
    for ds,file,color,marker in [('B1','B1_residuals.csv','#4477AA','.'),('B4','B4_direct_predictions.csv','#CC6677','x'),('B5','B5_direct_predictions.csv','#228833','+')]:
        f=pd.read_csv(root/'outputs/scaling/phase1'/file,float_precision='round_trip')
        ax.scatter(f.N_params_B,f.D_tokens_B,s=8 if ds=='B1' else 24,c=color,marker=marker,alpha=.5,label=ds+' frozen data')
    join=pd.read_csv(root/'outputs/scaling/phase4/B9_B10_join.csv',float_precision='round_trip')
    valid=join[(join.N_params_B_B9>0)&(join.D_tokens_B_B9>0)]
    ax.scatter(valid.N_params_B_B9,valid.D_tokens_B_B9,s=10,c='#AA3377',alpha=.45,label='B9 metadata (D>0)')
    a=bounds;ax.add_patch(Rectangle((a['N_min_B'],a['D_min_B']),a['N_max_B']-a['N_min_B'],a['D_max_B']-a['D_min_B'],fill=False,edgecolor='black',lw=2,label='Main joint numeric box'))
    ax.set(xscale='log',yscale='log',xlabel='N (billion parameters)',ylabel='D (billion tokens)',title='Observed supports and proposed main domain')
    ax.legend(fontsize=8);finish(fig,'06_N_D_evidence_support.png','数据范围对照；B4/B5可比性有限，B9非Loss验证')
    save_csv(out/'figure_manifest.csv',manifest)
    # Contact sheet for visual QA, not a substitute for individual figures.
    from PIL import Image,ImageOps,ImageDraw
    sheet=Image.new('RGB',(1400,1110),'white')
    for i,row in enumerate(manifest):
        im=Image.open(root/row['figure']).convert('RGB');thumb=ImageOps.contain(im,(690,350))
        x=(i%2)*700;y=(i//2)*370;sheet.paste(thumb,(x,y));ImageDraw.Draw(sheet).text((x+5,y+350),Path(row['figure']).name,fill='black')
    sheet.save(out/'figure_contact_sheet.png')
    return manifest,[root/'outputs/scaling/phase1'/s for s in ('B1_residuals.csv','B4_direct_predictions.csv','B5_direct_predictions.csv')]+[root/'outputs/scaling/phase4/B9_B10_join.csv']


def final_report(root,out,official,ctxstats,contexts,qinfo,qtable,bounds,boundtable,frozen,mixture,costtable,figures):
    """Answer all fourteen setup questions with evidence and explicit assumptions."""
    report=root/'outputs/report/question3_phase1_setup.md';report.parent.mkdir(parents=True,exist_ok=True)
    text='# 问题三 PHASE 1 算力约束优化输入标定\n\n'
    text+='本阶段完成官方公式核验、实际C7诊断、成本与支持域整理、数学形式及未来转移判据。未拟合任何模型，未运行资源优化或结构转移检测。'
    text+='问题一/二冻结文件完整性与独立算术核验见本阶段验证表。\n\n'
    text+='## 1 官方三类成本及预算\n\nC_train=6ND；CQ=D[g(Q)-g(Q0)]+；C_attn=etaNDL_ctx。'
    text+=f'eta={official["eta"]}，来源官方问题三原始OMML；训练/质量/注意力使用同一D。'
    text+=f'预算建议{official["budgets_FLOPs"]} FLOPs，非强制唯一取值。附加41点研究对数网格从最低/10到最高×10，仅列场景。\n\n'
    text+='## 2 C7上下文及注意力临界值\n\n'+table(pd.DataFrame([ctxstats]))+'\n\n'+table(contexts)+'\n\n'
    fields=pd.read_csv(out/'C7_field_diagnosis.csv')
    exceeding=int(contexts.loc[contexts.L_ctx>6/official['eta'],'support_count'].sum())
    text+=f'C7实际{ctxstats["rows"]}行{len(fields)}列，完整字段与缺失/重复见C7_diagnosis.md；没有独立N或明确tokens字段，training_data_TB未换算为D。'
    text+=f'eta使用题给值，无另估区间；L_crit=6/eta={6/official["eta"]:g}，在观测范围内但非观测离散值；超临界{exceeding}/{ctxstats["valid_count"]}。'
    text+='上下文取五个全部实际窗口作为外生场景。架构最大窗口、训练真实序列与自由N/D组合存在差异；题给成本为简化代理。\n\n'
    text+='## 3 冻结目标与证据等级\n\n'
    text+='L_gen=E+A N_B^(-alpha)+B D_B^(-beta)+C_Q(N_B/1B)^(-u_Q)(-lnQ)+lambda_p(N_B)[M(p)-M_ref]。'
    text+='这里N_B/1B表示相对N0_B=1的数值比例，不能再把以B计的N乘10^9代入目标系数。\n\n'
    text+='Class A经典骨架、Class B局部质量、Class C冻结ILR+一阶ElasticNet配方模型、Class D大尺度配方转移。所有完整参数直接读取最终JSON，见frozen_objective_summary.csv。'
    text+='没有联合外部N-D-Q-p验证；B6是局部半合成质量校准，B7差分扩展失败，不改冻结参数也不夸大证据。\n\n'
    text+='## 4 基线质量\n\n'+table(qtable,['candidate','Q0','definition','mapping_needed','recommended'])+'\n\n'
    text+=f'建议Q0_B={qinfo["Q0"]:.17g}以保持与p_ref一致，但仅在冻结代理映射假设下。'
    text+=f'依赖代理映射的配比权重为{qinfo["proxy_mass"]:.6%}，替代映射Q0={qinfo["alternative"]:.17g}；未识别域包络{qinfo["proxy_range_envelope"]}不是置信区间。'
    text+='七质量域聚合权重见baseline_quality_domain_weights.csv；c4无直接对应的17域配比，当前主映射聚合权重为0，不补造配比。'
    text+='主/替代映射的结构敏感性不可被七域bootstrap CI替代；不继承Q0=.8，不重算样本Q。\n\n'
    text+='## 5 三种官方质量成本与边际成本\n\n'+table(costtable,['family','g_formula','gamma','lambda_Q','marginal_formula','endpoint_finite','shape'])+'\n\n'
    text+='统一乘实际D与正部差分，Q=Q0增量0，Q>Q0边际正，Q=1均有限。对数型边际递减，不能改成发散模型。三种全部保留待后续比较，未提前择优。\n\n'
    text+='## 6 N/D/Q支持域\n\n'+table(boundtable,['scheme','variable','lower','upper','unit','extrapolation'])+'\n\n'
    text+='主域不是未经论证直接指定B1：比较B1–B5范围与已冻结验证局限后，经典骨架采用B1高可信包络，再与B6质量数值支持求交集，'
    text+='因此D下界10B而非0.134B。B3同源插值不能扩证据，B4/B5跨评测Loss口径未确认、仅列扩展覆盖，不扩主域。'
    text+='该连续矩形仍是插值建模选择，真实B1只含8个固定N轨迹。质量模块没有直接实验支持，表中“非外推”仅指N/D各模块数值范围。'
    text+='外推域将N上界扩至10000B、D上界36000B，保留主下界；B9 D=0是未知而不是有效可行下界。总/激活参数、稠密/稀疏等尚未统一，不能把B10估算Loss当验证。\n\n'
    text+=f'Q∈[Q0,1]；A1样本质量范围{qinfo["A1_sample_Q_range"]}，七领域质量范围{qinfo["A1_domain_Q_range"]}；'
    text+=f'B6数值Q范围[{frozen["quality_numeric_support"]["Q_min"]},{frozen["quality_numeric_support"]["Q_max"]}]。'
    text+='质量范围覆盖不等于直接验证；Q超过A1样本最大值或领域最大值分别标记，Q≈1不能宣称测得验证。\n\n'
    text+='## 7 p可行集\n\nP0固定p_ref；P1读取真实训练/检验配方已冻结闭合零替换模型输入，'
    text+=f'{mixture["source_records"]}条来源记录对应{mixture["unique_candidates"]}个候选，完整membership保存。'
    text+='不新建邻域，不做17维联立寻优；P1有限候选外层选择有据可查，但模型输入的微小零替换质量不是原始训练比例。'
    text+='P2理论simplex只作敏感性，极端角点不能作建议。固定Q0与可独立改善Q的假设对P1也需保留。N>1B使用Class D，不因N在B1范围而升级配比证据。\n\n'
    text+='## 8 单位\n\nC_train=6N_BD_B10^18；C_attn=etaN_BD_BL_ctx10^18；CQ=D_B10^9Δg，均为FLOPs。'
    text+='gamma按官方总成本维度推定FLOPs/token，lambda_Q无量纲；没有货币系数或额外M。12个独立Decimal手工算式与程序核验通过，见unit_audit.md及unit_numeric_checks.csv。\n\n'
    text+='## 9 数学形式与预算绑定\n\nP0 min_{N_B,D_B,Q}L_gen(p_ref)，受三个成本之和≤预算及上述box约束；'
    text+='P1逐配方同样求解再外层选最优。约束保持不等式。P0各资源边际Loss都负，内域预算绑定或全部资源到上界；'
    text+='P1的N导数可能因分段lambda与配方项改变符号，须左右检查，D/Q仍严格递减。边界饱和可出现预算松弛。'
    text+='尤其低预算配长上下文时主域可能不可行：预算可行性只按主域成本解析下界筛查，未求Loss最优。详见budget_feasibility_screen.csv。\n\n'
    text+='## 10 结构性转移定义\n\n采用活跃集、以预算为分母的三项成本份额、最优N/D对预算log弹性与Q对ln预算导数。'
    text+='候选变化须后续局部加密、左右导数、数值与情景稳定性验证，不在本阶段识别断点。'
    text+='同一上下文下C_attn/C_train=etaL_ctx/6恒定，不能制造由预算驱动的注意力/训练比例交叉。\n\n'
    text+='## 11 真实输入、解析推导与假设\n\n真实输入：官方Word/PDF、C7、问题一冻结质量/配方结果、B1以及B9公开元数据。'
    text+='B6局部校准为半合成，B10及10B/70B配方转移为估算；ILR配方函数是冻结回归模型。'
    text+='推导：临界窗口、边际质量成本、FLOPs换算、预算最低可行成本及单调性。'
    text+='假设/敏感性：17→7代理映射、Q0选取、矩形支持域及其B6交集、跨架构上下文自由组合、p/Q独立改变、零替换配方空间、41点预算网格与未来识别容差。'
    text+='旧优化稿未作为数据输入。\n\n'
    text+='## 12 进入PHASE 2的条件\n\n已具备按已审计公式进行条件化数值求解的输入与数学定义，等待人工确认Q0代理映射及主/外推域。'
    text+='无公式/单位缺失；条件化模型不等于完整联合真实验证。后续不得强行求不可行预算场景，不得把估算大模型配比反转或Q=1端点当直接证据。'
    text+='当前停止于PHASE 1，未求低中高最优解、未检测结构转移、未开始问题四。\n\n'
    text+='## 图表\n\n'+'\n'.join('- ['+f['caption']+'](../.. /'+f['figure']+')' for f in figures).replace('../.. /','../../')+'\n'
    report.write_text(text,encoding='utf-8')


def run(root):
    """Run independent setup steps in order and record exact input provenance."""
    root=Path(root);out=root/'outputs/optimization/phase1';out.mkdir(parents=True,exist_ok=True)
    cfg_path=root/'configs/optimization_phase1.yaml';cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    if cfg['perform_optimization'] or cfg['perform_structural_shift_detection']:
        raise ValueError('Phase 1 cannot run optimization or shift detection')
    np.random.seed(cfg['random_state'])
    logging.info('Audit official Word OMML and relevant data-description pages')
    official,paths=audit_official(root,out)
    logging.info('Diagnose actual C7 schema and observed contexts')
    c7,contexts,ctxstats,p=diagnose_context(root,out,official);paths+=p
    logging.info('Read frozen objective full precision; no refit')
    frozen,meta,p=frozen_inputs(root,out);paths+=p
    qinfo,qtable,p=baseline_quality(root,out,frozen);paths+=p
    bounds,boundtable,p=decision_domains(root,out,frozen,qinfo);paths+=p
    mixture,p=mixture_sets(root,out,frozen,meta,cfg);paths+=p
    budgets=budget_scenarios(out,official,cfg,bounds,contexts)
    unitchecks=numeric_units(out,official,qinfo,contexts)
    costtable=quality_cost_report(out,official,qinfo)
    math_reports(out,official,bounds)
    logging.info('Generate six figures from diagnostics and official formulas')
    figs,p=figures(root,out,contexts,c7,official,qinfo,budgets,bounds);paths+=p
    final_report(root,out,official,ctxstats,contexts,qinfo,qtable,bounds,boundtable,frozen,mixture,costtable,figs)
    # This is a new setup artifact, not a new fitted generalized scaling model.
    save_json(out/'optimization_setup.json',dict(official=official,baseline=qinfo,main_bounds=bounds,
            mixture=mixture,frozen_objective_source='outputs/scaling/phase4/generalized_scaling_final.json',
            official_contexts=contexts.L_ctx.tolist(),config=cfg,optimization_performed=False,
            shift_detection_performed=False,unit_checks_passed=len(unitchecks)))
    manifest=[]
    for path in sorted(set(paths+[cfg_path])):
        manifest.append(dict(path=path.relative_to(root).as_posix(),size_bytes=path.stat().st_size,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    save_csv(out/'input_manifest.csv',manifest)
    code_paths=[root/'scripts/17_setup_compute_optimization.py',root/'scripts/17_verify_compute_setup.py',
                root/'configs/optimization_phase1.yaml',*sorted((root/'src/optimization').glob('*.py'))]
    save_csv(out/'implementation_manifest.csv',[dict(path=p.relative_to(root).as_posix(),
             sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in code_paths if p.exists()])
    logging.info('PHASE 1 setup complete: Q0=%.12f, empirical candidates=%s; no optimization',qinfo['Q0'],mixture['unique_candidates'])

# ============================================================
# Source: scripts/17_setup_compute_optimization.py
# Original lines: 1-74
# Purpose: 最终阶段实际执行入口：17_setup_compute_optimization.py
# ============================================================
"""Question 3 Phase 1 only: audit inputs and formulate, never optimize."""
from pathlib import Path
import argparse
import hashlib
import json
import logging
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
OUT = PROJECT_ROOT / 'outputs/optimization/phase1'
NEW_PREFIXES = ('outputs/optimization/phase1/', 'outputs/figures/question3_phase1/',
                'src/optimization/')
NEW_FILES = {'scripts/17_setup_compute_optimization.py',
             'scripts/17_verify_compute_setup.py', 'configs/optimization_phase1.yaml',
             'outputs/report/question3_phase1_setup.md'}


def file_hash(path):
    """Hash file bytes in blocks, without parsing or changing official data."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def protected_files():
    """Every existing project artifact outside this newly authorized phase."""
    for path in PROJECT_ROOT.rglob('*'):
        if not path.is_file():
            continue
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if rel in NEW_FILES or rel.startswith(NEW_PREFIXES):
            continue
        if any(x in path.parts for x in ('.git', '.codex', '.agents')):
            continue
        yield path, rel


def snapshot():
    """Write the before-state once; never silently replace the freeze baseline."""
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'frozen_before_phase1.json'
    if target.exists():
        return
    values = {rel: {'size': p.stat().st_size, 'sha256': file_hash(p)}
              for p, rel in protected_files()}
    target.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Protected files snapshotted: {len(values)}', flush=True)


def main():
    """Run setup only, with a persistent log and explicit failure reporting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-only', action='store_true')
    args = parser.parse_args()
    snapshot()
    if args.snapshot_only:
        return
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(OUT/'execution.log', encoding='utf-8')])
    try:
        from optimization.setup import run
        run(PROJECT_ROOT)
    except Exception:
        logging.exception('Phase 1 setup failed; no optimizer has been invoked')
        raise


if __name__ == '__main__':
    main()

# ============================================================
# Source: src/scaling/marginal_analysis.py
# Original lines: 1-153
# Purpose: 最终广义标度律、边界策略、解析导数/弹性及Q+0.1精确求根
# ============================================================
"""Frozen modular model, boundary policies, derivatives and exact Q equivalence.

No parameter estimation occurs here. N and D are in billions. Physical quality
transfer is conditional on the frozen B6 local calibration, not a global law.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from .mixture_transfer import predict_recipe


class FrozenGeneralizedScaling:
    """Evaluate the human-confirmed modular form with explicit lambda boundaries."""
    def __init__(self,classical,quality,meta,M_ref,knots,config):
        self.c=dict(classical);self.q=dict(quality);self.meta=meta;self.M_ref=float(M_ref)
        self.Ns=np.array([k['N_params_B'] for k in knots],float)
        self.lambdas=np.array([k['lambda_s'] for k in knots],float)
        self.slopes=np.diff(self.lambdas)/np.diff(np.log(self.Ns))
        self.cfg=config;self.shrink=float(config['lambda_shrink_zero_B'])
        if not np.all(np.diff(self.Ns)>0) or self.shrink<=self.Ns[-1]:
            raise ValueError('Invalid frozen lambda knots or shrink endpoint')

    def delta_M(self,p):
        """Frozen 1M mixture prediction relative to the frozen training center."""
        return predict_recipe(p,self.meta)-self.M_ref

    def lambda_value(self,N,policy='S-A'):
        """Main clamp; sensitivity continuation or ln-scale shrink to zero."""
        x=np.asarray(N,float)
        if not np.isfinite(x).all() or (x<=0).any():raise ValueError('N must be positive')
        if policy not in ['S-A','S-B','S-C']:raise ValueError('Unknown lambda policy')
        value=np.interp(np.log(x),np.log(self.Ns),self.lambdas)
        high=x>self.Ns[-1]
        if policy=='S-B':
            value=np.where(high,self.lambdas[-1]+self.slopes[-1]*np.log(x/self.Ns[-1]),value)
        elif policy=='S-C':
            fraction=np.clip(np.log(x/self.Ns[-1])/np.log(self.shrink/self.Ns[-1]),0,1)
            value=np.where(high,self.lambdas[-1]*(1-fraction),value)
        return value

    def log_slope(self,N,policy='S-A',side='right'):
        """Return d lambda/d ln N; choose a one-sided derivative at each knot."""
        x=np.asarray(N,float)
        if not np.isfinite(x).all() or (x<=0).any() or side not in ['left','right']:
            raise ValueError('Illegal N or derivative side')
        j=np.searchsorted(self.Ns,x,side=side)-1
        value=np.zeros(x.shape,float)
        middle=(j>=0)&(j<len(self.Ns)-1)
        value=np.where(middle,self.slopes[np.clip(j,0,len(self.slopes)-1)],value)
        high=x>=self.Ns[-1] if side=='right' else x>self.Ns[-1]
        if policy=='S-B':value=np.where(high,self.slopes[-1],value)
        elif policy=='S-C':
            shrinking=high & (x<self.shrink if side=='right' else x<=self.shrink)
            value=np.where(shrinking,-self.lambdas[-1]/np.log(self.shrink/self.Ns[-1]),value)
        elif policy!='S-A':raise ValueError('Unknown lambda policy')
        return value

    def terms(self,N,D,Q,deltaM=0.,policy='S-A'):
        """Compute actual model terms; only reference scenarios use Q=1, deltaM=0."""
        N,D,Q,deltaM=np.broadcast_arrays(np.asarray(N,float),np.asarray(D,float),np.asarray(Q,float),np.asarray(deltaM,float))
        if not np.isfinite(N+D+Q+deltaM).all() or (N<=0).any() or (D<=0).any() or (Q<=0).any() or (Q>1).any():
            raise ValueError('Illegal N,D,Q,mixture score')
        c,q=self.c,self.q
        nt=c['A']*N**(-c['alpha']);dt=c['B']*D**(-c['beta'])
        dq=q['C_Q']*(N/q['N0_B'])**(-q['u_Q'])*(-np.log(Q))
        dp=self.lambda_value(N,policy)*deltaM;R=nt+dt+dq+dp
        return dict(N_term=nt,D_term=dt,Delta_Q=dq,Delta_p=dp,R=R,L0=c['E']+nt+dt,L_gen=c['E']+R)

    def derivatives(self,N,D,Q,deltaM=0.,policy='S-A',side='right'):
        """Exact interior or chosen one-sided derivatives; do not smooth knots."""
        t=self.terms(N,D,Q,deltaM,policy);N=np.asarray(N,float);D=np.asarray(D,float);Q=np.asarray(Q,float)
        c,q=self.c,self.q;s=self.log_slope(N,policy,side)
        gain=c['alpha']*t['N_term']+q['u_Q']*t['Delta_Q']-s*np.asarray(deltaM)
        return dict(dL_dN=-gain/N,dL_dD=-c['beta']*t['D_term']/D,
                    dL_dQ=-q['C_Q']*(N/q['N0_B'])**(-q['u_Q'])/Q,Gain_N=gain,s_i=s)

    def elasticity(self,N,D,Q,deltaM=0.,policy='S-A',side='right'):
        """Loss-reduction elasticities for R>0; traditional signs in an appendix."""
        t=self.terms(N,D,Q,deltaM,policy);d=self.derivatives(N,D,Q,deltaM,policy,side)
        valid=t['R']>0
        def ratio(a):return np.divide(a,t['R'],out=np.full(np.shape(t['R']),np.nan),where=valid)
        return dict(E_N=ratio(d['Gain_N']),E_D=ratio(self.c['beta']*t['D_term']),
                    E_Q=ratio(self.q['C_Q']*(np.asarray(N)/self.q['N0_B'])**(-self.q['u_Q'])),
                    traditional_N=np.asarray(N)/t['L_gen']*d['dL_dN'],
                    traditional_D=np.asarray(D)/t['L_gen']*d['dL_dD'],
                    traditional_Q=np.asarray(Q)/t['L_gen']*d['dL_dQ'],valid=valid)

    def N_dependent_loss(self,N,Q,deltaM,policy='S-A'):
        """Exact N-dependent part; E and B D^-beta cancel in Q equivalence."""
        N=np.asarray(N,float)
        return (self.c['A']*N**(-self.c['alpha'])+
                self.q['C_Q']*(N/self.q['N0_B'])**(-self.q['u_Q'])*(-np.log(Q))+
                self.lambda_value(N,policy)*deltaM)

    def equivalence(self,N,Q,deltaM=0.,policy='S-A'):
        """Find first exact N_eq>N, respecting nonmonotone lambda segments.

        Within a segment Gain_N=positive decreasing powers minus a constant,
        hence has at most one zero. Include each stationary point as well as
        each lambda knot in the root brackets: an endpoint-only sign check can
        miss a valid first root when F(N) decreases and then increases.
        """
        N,Q,deltaM=float(N),float(Q),float(deltaM);dQ=float(self.cfg['Q_gain'])
        if not 0<Q<=1 or N<=0:raise ValueError('Illegal equivalence workpoint')
        new=Q+dQ
        result=dict(Q_new=new,root_policy='first_root_above_N',N_max_B=max(self.cfg['root_max_multiple']*N,self.cfg['root_min_max_B']))
        if Q>.9 or new>1+1e-14:return dict(**result,status='Q_gain_out_of_range',N_eq_B=None)
        gain=float(self.derivatives(N,150.,Q,deltaM,policy,self.cfg['knot_derivative_main'])['Gain_N'])
        result['Gain_N']=gain
        if gain<=0:return dict(**result,status='no_local_equivalence',N_eq_B=None)
        local=float(self.q['C_Q']*(N/self.q['N0_B'])**(-self.q['u_Q'])/(Q*gain)*dQ)
        result['local_approx_gain']=local
        target=float(self.N_dependent_loss(N,new,deltaM,policy))
        def objective(x):return float(self.N_dependent_loss(x,Q,deltaM,policy)-target)
        upper=result['N_max_B'];points=[N]+[float(k) for k in self.Ns if N<k<upper]+[upper]
        if policy=='S-C' and N<self.shrink<upper:points.append(self.shrink)
        points=sorted(set(points));stationary=[]
        def gain_at(x,s):
            return (self.c['alpha']*self.c['A']*x**(-self.c['alpha'])+
                    self.q['u_Q']*self.q['C_Q']*(x/self.q['N0_B'])**(-self.q['u_Q'])*(-np.log(Q))-s*deltaM)
        for lo,hi in zip(points[:-1],points[1:]):
            s=float(self.log_slope(np.sqrt(lo*hi),policy));g0,g1=gain_at(lo,s),gain_at(hi,s)
            if g0*g1<0:
                stationary.append(float(brentq(lambda x:gain_at(x,s),lo,hi,xtol=self.cfg['root_xtol'],rtol=self.cfg['root_rtol'])))
        points=sorted(set(points+stationary));root=None;chosen=None
        for lo,hi in zip(points[:-1],points[1:]):
            flo,fhi=objective(lo),objective(hi)
            if flo==0 and lo>N:root=lo;chosen=(lo,lo);break
            if fhi==0:root=hi;chosen=(hi,hi);break
            if flo*fhi<0:
                root=float(brentq(objective,lo,hi,xtol=self.cfg['root_xtol'],rtol=self.cfg['root_rtol']))
                chosen=(lo,hi);break
        result.update(n_stationary_points=len(stationary),n_search_segments=len(points)-1,
                      objective_at_start=objective(N),objective_at_max=objective(upper))
        if root is None:return dict(**result,status='no_finite_equivalence_within_range',N_eq_B=None)
        change=root/N-1
        return dict(**result,status='solved',N_eq_B=root,equivalent_parameter_gain=change,
                    equivalent_parameter_gain_pct=100*change,local_approx_gain_pct=100*local,
                    local_minus_exact_pct_points=100*(local-change),
                    local_relative_error=(local-change)/change,
                    equation_residual=objective(root),root_bracket_low_B=chosen[0],root_bracket_high_B=chosen[1])


def finite_difference(model,N,D,Q,deltaM,variable,config,side=None):
    """Fourth-order central or one-sided derivative; no stencil crosses an N knot."""
    vals=dict(N=float(N),D=float(D),Q=float(Q));x=vals[variable]
    h=config['finite_difference_relative_step']*x
    def f(offset):
        v=dict(vals);v[variable]=x+offset*h
        return float(model.terms(deltaM=deltaM,**v)['L_gen'])
    if side=='right':return (-25*f(0)+48*f(1)-36*f(2)+16*f(3)-3*f(4))/(12*h)
    if side=='left':return (25*f(0)-48*f(-1)+36*f(-2)-16*f(-3)+3*f(-4))/(12*h)
    return (-f(2)+8*f(1)-8*f(-1)+f(-2))/(12*h)

# ============================================================
# Source: src/optimization/resource_solver.py
# Original lines: 1-378
# Purpose: D_B解析消元、可行性、网格/SLSQP/DE及P0/P1求解
# ============================================================
"""D elimination, multistart resource solving, independent checks; frozen model."""
from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize, minimize_scalar, differential_evolution
from scaling.marginal_analysis import FrozenGeneralizedScaling
from .costs import quality_increment_per_token, marginal_quality_per_token


def csv(path,rows):
    """Save full floating precision, including explicit infeasible rows."""
    (rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)).to_csv(path,index=False,encoding='utf-8-sig')


def load(root):
    """Read only frozen coefficients and the authorized new solver settings."""
    root=Path(root)
    setup=json.loads((root/'outputs/optimization/phase1/optimization_setup.json').read_text(encoding='utf-8'))
    frozen=json.loads((root/setup['frozen_objective_source']).read_text(encoding='utf-8'))
    cfg=yaml.safe_load((root/'configs/optimization_phase2.yaml').read_text(encoding='utf-8'))
    phase4=yaml.safe_load((root/'configs/scaling_phase4.yaml').read_text(encoding='utf-8'))
    model=FrozenGeneralizedScaling(frozen['classical'],frozen['quality'],{},frozen['reference']['M_ref'],frozen['lambda_knots'],phase4)
    if frozen['lambda_main_policy']!='S-A' or cfg['P2_enabled']:raise ValueError('Unauthorized model/P2 policy')
    candidates=pd.read_csv(root/'outputs/optimization/phase1/empirical_recipe_candidates.csv',float_precision='round_trip')
    if len(candidates)!=setup['mixture']['unique_candidates']:raise ValueError('Changed frozen empirical set')
    ext=candidates.loc[[candidates.M_p.idxmin(),candidates.M_p.idxmax()]].copy()
    ext['role']=['minM','maxM'];ext['DeltaM']=ext.M_p-frozen['reference']['M_ref']
    return setup,frozen,cfg,model,candidates,ext


def lambda_zero(model):
    """Find analytic ln-N crossings of the frozen linear interpolation."""
    roots=[]
    for a,b,u,v in zip(model.Ns[:-1],model.Ns[1:],model.lambdas[:-1],model.lambdas[1:]):
        if u*v<0:roots.append(float(np.exp(np.log(a)-u*(np.log(b)-np.log(a))/(v-u))))
    return roots


class ResourceProblem:
    """Loss surface in log N,Q; D is its maximum feasible analytic value."""
    def __init__(self,setup,model,cfg,ext,budget,family,lctx,domain='main',mode='P0',q0=None):
        self.setup,self.model,self.cfg,self.ext=setup,model,cfg,ext
        self.C=float(budget);self.b=self.C/1e18;self.family=family;self.lctx=int(lctx)
        self.domain,self.mode=domain,mode;self.q0=float(cfg['Q0_main'] if q0 is None else q0)
        self.eta=setup['official']['eta'];self.a=6+self.eta*lctx
        bounds=setup['main_bounds'];self.nlo=bounds['N_min_B'];self.dlo=bounds['D_min_B']
        self.nmax=bounds['N_max_B'] if domain=='main' else 10000.
        self.dmax=bounds['D_max_B'] if domain=='main' else 36000.
        self.nhi=min(self.nmax,self.b/(self.a*self.dlo))
        self.pars=setup['official']['quality_costs'][family]
        self.feasible=self.nhi>=self.nlo*(1-1e-14)
        self.minM=float(ext.DeltaM.iloc[0]);self.maxM=float(ext.DeltaM.iloc[1])

    def h(self,q):
        """Quality increment cost per 1B tokens divided by 10^18 FLOPs."""
        return quality_increment_per_token(q,self.q0,self.family,self.pars['gamma'],self.pars['lambda'])/1e9

    def hp(self,q):
        """Scaled right marginal cost; lambda_Q differs from mixture lambda_p."""
        return marginal_quality_per_token(q,self.q0,self.family,self.pars['gamma'],self.pars['lambda'])/1e9

    def inverse_h(self,h):
        """Exact inverse of all three monotone official cost shapes, clipped at Q=1."""
        h=np.clip(np.asarray(h,float),0,float(self.h(1.)))
        g=self.pars['gamma']/1e9;l=self.pars['lambda'];q0=self.q0
        if self.family=='exponential':q=q0+np.log1p(h/(g*np.exp(l*q0)))/l
        elif self.family=='power':q=(q0**l+h/g)**(1/l)
        else:q=q0+np.expm1(h/g)*(1+l*q0)/l
        return np.minimum(1,np.maximum(q0,q))

    def qmax(self,n):return self.inverse_h(self.b/self.dlo-self.a*np.asarray(n))

    def delta(self,n):
        """Exact empirical envelope from cost independence and lambda sign."""
        n=np.asarray(n,float)
        return np.zeros_like(n) if self.mode=='P0' else np.where(self.model.lambda_value(n)>=0,self.minM,self.maxM)

    def evaluate(self,x):
        """Broadcast vector evaluation, including D-budget and its upper cap."""
        x=np.asarray(x,float);n=np.exp(x[...,0]);q=x[...,1]
        c=self.a*n+self.h(q);d=np.minimum(self.dmax,self.b/c)
        return self.model.terms(n,d,q,self.delta(n))['L_gen']

    def gradient(self,x,side='right'):
        """Analytic gradient on each smooth budget/D-cap/lambda branch."""
        n=np.exp(x[0]);q=x[1];c=self.a*n+float(self.h(q));d=min(self.dmax,self.b/c)
        terms=self.model.terms(n,d,q,self.delta(n));der=self.model.derivatives(n,d,q,self.delta(n),side=side)
        gx=float(der['dL_dN'])*n;gq=float(der['dL_dQ'])
        if self.b/c<=self.dmax:
            gx+=self.model.c['beta']*float(terms['D_term'])*self.a*n/c
            gq+=self.model.c['beta']*float(terms['D_term'])*float(self.hp(q))/c
        return np.array([gx,gq])

    def project(self,x):
        """Repair only floating boundary violations, not the scenario's physical bounds."""
        x=np.asarray(x,float).copy();x[0]=np.clip(x[0],np.log(self.nlo),np.log(max(self.nlo,self.nhi)))
        x[1]=np.clip(x[1],self.q0,float(self.qmax(np.exp(x[0]))))
        return x

    def constraint(self,x):
        return (self.b/self.dlo-self.a*np.exp(x[0])-float(self.h(x[1])))/max(self.b/self.dlo,1)

    def constraint_jac(self,x):
        return -np.array([self.a*np.exp(x[0]),float(self.hp(x[1]))])/max(self.b/self.dlo,1)

    def segments(self):
        """P1 split at every supported knot and the conditional zero crossing."""
        cuts=[self.nlo,self.nhi]
        if self.mode=='P1':cuts += [float(v) for v in [*self.model.Ns,*lambda_zero(self.model)] if self.nlo<v<self.nhi]
        return list(zip(np.log(sorted(set(cuts)))[:-1],np.log(sorted(set(cuts)))[1:]))

    def local(self,x,lo,hi):
        """Constrained SLSQP on one closed smooth lambda interval."""
        result=minimize(self.evaluate,self.project(x),jac=self.gradient,method='SLSQP',bounds=[(lo,hi),(self.q0,1.)],
            constraints=[dict(type='ineq',fun=self.constraint,jac=self.constraint_jac)],
            options=dict(ftol=self.cfg['local_ftol'],maxiter=self.cfg['local_maxiter']))
        y=self.project(result.x);y[0]=np.clip(y[0],lo,hi);y=self.project(y)
        return y,result

    def coarse(self,lo,hi):
        """Dense rectangular grid plus feasible/budget-cap boundary curves."""
        xs=np.linspace(lo,hi,self.cfg['grid_N_count']);qs=np.linspace(self.q0,1,self.cfg['grid_Q_count'])
        xx,qq=np.meshgrid(xs,qs,indexing='ij');points=np.c_[xx.ravel(),qq.ravel()]
        ok=(self.a*np.exp(points[:,0])+self.h(points[:,1]))<=self.b/self.dlo*(1+1e-14)
        points=points[ok]
        n=np.exp(xs)
        for target in (self.dlo,self.dmax):
            q=self.inverse_h(self.b/target-self.a*n)
            boundary=np.c_[xs,q];boundary=boundary[(self.a*n+self.h(q))<=self.b/self.dlo*(1+1e-14)]
            points=np.r_[points,boundary]
        if len(points)==0:points=np.array([self.project([lo,self.q0])])
        losses=self.evaluate(points)
        order=np.argsort(losses,kind='stable')
        # Distinct initial points; also explicitly retain corners in candidate pool.
        seeds=[]
        for idx in order:
            point=points[idx]
            if not any(np.linalg.norm(point-v)<1e-10 for v in seeds):seeds.append(point)
            if len(seeds)>=self.cfg['local_start_count']:break
        return np.array(seeds),points[order[0]],float(losses[order[0]])

    def solve(self,independent=False,seed=2026):
        """Take best legal candidate across grids, multistart, boundaries and optional DE."""
        if not self.feasible:return self.record(None,status='INFEASIBLE'),None
        if self.nhi<=self.nlo*(1+1e-13):
            point=np.array([np.log(self.nlo),self.q0])
            rec=self.record(point);rec.update(grid_min_loss=rec['L_star'],local_start_count=0,
                local_success_count=0,independent_DE=False,DE_raw_loss=np.nan,DE_refined_loss=np.nan,DE_minus_final_loss=np.nan)
            return rec,point
        pool=[];start_count=0;success_count=0;gridbest=np.inf
        for lo,hi in self.segments():
            starts,coarse,loss=self.coarse(lo,hi);pool.append(coarse);gridbest=min(gridbest,loss)
            for start in starts:
                x,res=self.local(start,lo,hi);pool.append(x);start_count+=1;success_count+=int(res.success)
            # Every knot, both one-sided neighborhoods and Q-boundaries are considered.
            for xn in (lo,hi,np.nextafter(lo,hi),np.nextafter(hi,lo)):
                qhi=float(self.qmax(np.exp(xn)))
                pool.extend([self.project([xn,self.q0]),self.project([xn,qhi])])
                if qhi-self.q0>1e-12:
                    r=minimize_scalar(lambda q:float(self.evaluate([xn,q])),bounds=(self.q0,qhi),method='bounded',options={'xatol':1e-12})
                    pool.append(self.project([xn,r.x]))
            # D_max kink curve must be searched explicitly, rather than smoothed.
            def kink(xn):return float(self.evaluate(self.project([xn,float(self.inverse_h(self.b/self.dmax-self.a*np.exp(xn)))])))
            r=minimize_scalar(kink,bounds=(lo,hi),method='bounded',options={'xatol':1e-11})
            pool.append(self.project([r.x,float(self.inverse_h(self.b/self.dmax-self.a*np.exp(r.x)))]))
        de_loss=np.nan;de_raw=np.nan
        if independent:
            # Feasible map t∈[0,1] -> Q0+t(Qmax(N)-Q0); no penalty or rejection.
            def mapx(v):
                qhi=float(self.qmax(np.exp(v[0])));return np.array([v[0],self.q0+v[1]*(qhi-self.q0)])
            if np.log(self.nhi)-np.log(self.nlo)>1e-13:
                de=differential_evolution(lambda v:float(self.evaluate(mapx(v))),[(np.log(self.nlo),np.log(self.nhi)),(0.,1.)],
                    seed=seed,maxiter=self.cfg['de_maxiter'],popsize=self.cfg['de_popsize'],tol=self.cfg['de_tol'],atol=1e-11,polish=False)
                dx=mapx(de.x);de_raw=float(self.evaluate(dx));de_loss=de_raw
                pool.append(dx)
                for lo,hi in self.segments():
                    if lo-1e-12<=dx[0]<=hi+1e-12:
                        polished,_=self.local(dx,lo,hi);pool.append(polished);de_loss=min(de_loss,float(self.evaluate(polished)))
        points=np.array([self.project(x) for x in pool]);losses=self.evaluate(points)
        best=points[np.argmin(losses)]
        rec=self.record(best);rec.update(grid_min_loss=gridbest,local_start_count=start_count,local_success_count=success_count,
             independent_DE=independent,DE_raw_loss=de_raw,DE_refined_loss=de_loss,
             DE_minus_final_loss=de_loss-rec['L_star'] if independent else np.nan)
        return rec,best

    def record(self,x,status='SOLVED'):
        """Complete allocation, cost shares, boundary flags and evidence provenance."""
        r=dict(budget=self.C,cost_function=self.family,L_ctx=self.lctx,domain=self.domain,mode=self.mode,Q0=self.q0,
               status=status,feasible=status=='SOLVED',scenario_extrapolation=self.domain!='main')
        if x is None:return r
        n=np.exp(x[0]);q=x[1];d=min(self.dmax,self.b/(self.a*n+float(self.h(q))))
        train=6*n*d*1e18;attn=self.eta*n*d*self.lctx*1e18;quality=d*float(self.h(q))*1e18;used=train+attn+quality
        dm=float(self.delta(n));lam=float(self.model.lambda_value(n));tol=self.cfg['active_tolerance']
        r.update(N_star=n,D_star=d,Q_star=q,L_star=float(self.model.terms(n,d,q,dm)['L_gen']),
                 C_train=train,C_quality=quality,C_attn=attn,C_used=used,unused_budget=max(0,self.C-used),
                 s_train=train/self.C,s_quality=quality/self.C,s_attn=attn/self.C,
                 lambda_value=lam,lambda_sign=int(np.sign(lam)),DeltaM=dm,
                 selected_candidate='p_ref' if self.mode=='P0' else str(self.ext.candidate_id.iloc[0 if lam>=0 else 1]),
                 mixture_transfer_uses_estimated_support=bool(self.mode=='P1' and n>1+1e-12),
                 mixture_evidence='reference_term_zero' if self.mode=='P0' else 'Class D' if n>1+1e-12 else 'Class C',
                 extrapolative_N=bool(n>self.setup['main_bounds']['N_max_B']*(1+tol)),
                 extrapolative_D=bool(d>self.setup['main_bounds']['D_max_B']*(1+tol)),
                 extrapolative_Q=bool(q>self.setup['baseline']['A1_sample_Q_range'][1]+tol),
                 Q_outside_A1_domain_range=bool(q>self.setup['baseline']['A1_domain_Q_range'][1]+tol),
                 quality_direct_joint_validation=False,
                 active_N_lower=abs(n/self.nlo-1)<=tol,active_N_upper=abs(n/self.nmax-1)<=tol,
                 active_D_lower=abs(d/self.dlo-1)<=tol,active_D_upper=abs(d/self.dmax-1)<=tol,
                 active_Q_lower=abs(q-self.q0)<=tol,active_Q_upper=abs(q-1)<=tol,
                 budget_binding=abs(used/self.C-1)<=tol)
        r['extrapolative']=bool(r['extrapolative_N'] or r['extrapolative_D'] or r['extrapolative_Q'])
        r['evidence_domain_saturated']=all(r[k] for k in ('active_N_upper','active_D_upper','active_Q_upper'))
        return r


def unit_checks(out,setup,cfg):
    """Fresh full-count versus B-unit arithmetic tests, distinct from Phase 1."""
    from decimal import Decimal,localcontext
    rows=[];q0=cfg['Q0_main'];families=list(setup['official']['quality_costs'])
    for i in range(15):
        n=[.070542,1.,11.965825,100.,10000.][i//3];d=[10.,50.,299.893,1000.,36000.][i//3]
        l=setup['official_contexts'][i//3];q=[q0,.9,1.][i%3];f=families[i%3];p=setup['official']['quality_costs'][f]
        tr=6*n*d*1e18;at=setup['official']['eta']*n*d*l*1e18
        qu=d*1e9*float(quality_increment_per_token(q,q0,f,p['gamma'],p['lambda']))
        with localcontext() as ctx:
            ctx.prec=50;N=Decimal(str(n))*Decimal('1e9');D=Decimal(str(d))*Decimal('1e9');Q=Decimal(str(q));Q0=Decimal(str(q0))
            gamma=Decimal(str(p['gamma']));lam=Decimal(str(p['lambda']))
            diff=(lam*Q).exp()-(lam*Q0).exp() if f=='exponential' else Q**int(lam)-Q0**int(lam) if f=='power' else (1+lam*Q).ln()-(1+lam*Q0).ln()
            a=Decimal(str(setup['official']['eta']))*N*D*Decimal(l);t=6*N*D;qual=D*gamma*diff
        passed=all(np.isclose(x,float(y),rtol=3e-14,atol=1e-4) for x,y in [(tr,t),(at,a),(qu,qual)])
        if not passed:raise ValueError('FLOPs unit check failed')
        rows.append(dict(case=i+1,N_B=n,D_B=d,Q=q,L_ctx=l,cost_function=f,train_FLOPs=tr,quality_FLOPs=qu,attention_FLOPs=at,
                         decimal_train=str(t),decimal_quality=str(qual),decimal_attention=str(a),passed=passed))
    csv(out/'unit_tests.csv',rows)


def candidate_checks(out,setup,model,cfg,ext,candidates,budgets):
    """Full 832-enumeration on random feasible resource grids, not just extrema lookup."""
    rng=np.random.default_rng(cfg['random_state']);all_dm=candidates.M_p.to_numpy()-model.M_ref;rows=[]
    for i in range(cfg['candidate_enumeration_check_count']):
        family=list(setup['official']['quality_costs'])[i%3];lctx=setup['official_contexts'][i%5]
        budget=float(rng.choice(budgets[budgets>=3e19]));domain='main' if i%2==0 else 'extended'
        problem=ResourceProblem(setup,model,cfg,ext,budget,family,lctx,domain,'P1')
        ns=np.geomspace(problem.nlo,problem.nhi,31);fractions=np.linspace(0,1,23)
        NN,TT=np.meshgrid(ns,fractions,indexing='ij');QQ=problem.q0+TT*(problem.qmax(NN)-problem.q0)
        DD=np.minimum(problem.dmax,problem.b/(problem.a*NN+problem.h(QQ)))
        base=model.terms(NN,DD,QQ,0)['L_gen'].ravel();lam=model.lambda_value(NN).ravel()
        full=base[:,None]+lam[:,None]*all_dm[None,:]
        direct=full.min(axis=1);envelope=base+lam*problem.delta(NN).ravel()
        gap=float(np.max(np.abs(direct-envelope)))
        rows.append(dict(case=i+1,budget=budget,L_ctx=lctx,cost_function=family,domain=domain,
            enumerated_candidates=len(candidates),resource_grid_points=len(base),maximum_pointwise_loss_gap=gap,
            full_grid_min=float(direct.min()),reduced_grid_min=float(envelope.min()),passed=gap<1e-12))
    csv(out/'P1_candidate_reduction_check.csv',rows)
    if not all(r['passed'] for r in rows):raise ValueError('Empirical reduction full-enumeration check failed')


def three_dimensional_checks(out,setup,model,cfg,ext,path):
    """Compare independent full 3-D multistart SLSQP with analytic D elimination."""
    feasible=path[path.feasible==True].copy()
    selected=feasible.sample(n=cfg['three_dimensional_check_count'],random_state=cfg['random_state'])
    rng=np.random.default_rng(cfg['random_state']+1);rows=[]
    for i,r in enumerate(selected.itertuples()):
        prob=ResourceProblem(setup,model,cfg,ext,r.budget,r.cost_function,r.L_ctx,r.domain,'P0')
        def loss(x):return float(model.terms(np.exp(x[0]),np.exp(x[1]),x[2],0)['L_gen'])
        def jac(x):
            n,d=np.exp(x[0]),np.exp(x[1]);v=model.derivatives(n,d,x[2],0)
            return np.array([float(v['dL_dN'])*n,float(v['dL_dD'])*d,float(v['dL_dQ'])])
        def con(x):return 1-np.exp(x[1])*(prob.a*np.exp(x[0])+float(prob.h(x[2])))/prob.b
        def cj(x):
            n,d=np.exp(x[0]),np.exp(x[1]);return -np.array([d*prob.a*n,d*(prob.a*n+float(prob.h(x[2]))),d*float(prob.hp(x[2]))])/prob.b
        bounds=[(np.log(prob.nlo),np.log(prob.nhi)),(np.log(prob.dlo),np.log(prob.dmax)),(prob.q0,1.)]
        starts=[[np.log(r.N_star),np.log(r.D_star),r.Q_star]]
        for _ in range(10):
            n=np.exp(rng.uniform(*bounds[0]));q=rng.uniform(prob.q0,float(prob.qmax(n)));dhi=min(prob.dmax,prob.b/(prob.a*n+float(prob.h(q))))
            starts.append([np.log(n),rng.uniform(np.log(prob.dlo),np.log(max(prob.dlo,dhi))),q])
        pool=[]
        for x in starts:
            res=minimize(loss,x,jac=jac,method='SLSQP',bounds=bounds,constraints=[dict(type='ineq',fun=con,jac=cj)],
                         options=dict(ftol=cfg['local_ftol'],maxiter=cfg['local_maxiter']))
            if con(res.x)>=-1e-8:pool.append(res.x)
        if not pool:raise ValueError('Independent full 3D validation found no legal solution')
        best=min(pool,key=loss);N,D=np.exp(best[0]),np.exp(best[1]);L=loss(best)
        rows.append(dict(case=i+1,budget=r.budget,cost_function=r.cost_function,L_ctx=r.L_ctx,domain=r.domain,
             reduced_N=r.N_star,reduced_D=r.D_star,reduced_Q=r.Q_star,reduced_L=r.L_star,
             full3D_N=N,full3D_D=D,full3D_Q=best[2],full3D_L=L,loss_gap=L-r.L_star,
             relative_N_gap=N/r.N_star-1,relative_D_gap=D/r.D_star-1,Q_gap=best[2]-r.Q_star,
             full3D_budget_violation=max(0,-con(best)),starts=len(starts),
             passed=abs(L-r.L_star)<cfg['loss_validation_tolerance'] and abs(N/r.N_star-1)<.003 and abs(D/r.D_star-1)<.003 and abs(best[2]-r.Q_star)<.001))
    csv(out/'D_elimination_3D_check.csv',rows)
    if not all(r['passed'] for r in rows):raise ValueError('Full 3D vs eliminated-D check failed; inspect saved diagnostic')


def run_solve(root):
    """Solve P0 paths, official P1 and limited baseline sensitivities with checkpoints."""
    root=Path(root);out=root/'outputs/optimization/phase2';out.mkdir(parents=True,exist_ok=True)
    setup,frozen,cfg,model,candidates,ext=load(root);csv(out/'P1_extreme_candidates.csv',ext)
    phase1budgets=pd.read_csv(root/'outputs/optimization/phase1/compute_budget_scenarios.csv',float_precision='round_trip')
    official=set(setup['official']['budgets_FLOPs']);budgets=np.unique(phase1budgets.C_budget_FLOPs.to_numpy())
    csv(out/'budget_grid.csv',[dict(budget=float(x),is_official=x in official) for x in budgets])
    csv(out/'minimum_feasible_budget.csv',[dict(L_ctx=l,N_min=setup['main_bounds']['N_min_B'],D_min=setup['main_bounds']['D_min_B'],
        Q0=cfg['Q0_main'],minimum_budget=(6+setup['official']['eta']*l)*setup['main_bounds']['N_min_B']*setup['main_bounds']['D_min_B']*1e18) for l in setup['official_contexts']])
    csv(out/'context_cost_ratio.csv',[dict(L_ctx=l,ratio_attn_train=setup['official']['eta']*l/6,
        L_ctx_crit=6/setup['official']['eta'],attention_dominant=l>6/setup['official']['eta']) for l in setup['official_contexts']])
    nodes=sorted(set([*model.Ns,*lambda_zero(model)]))
    csv(out/'lambda_node_diagnosis.csv',[dict(N_B=n,lambda_node=float(model.lambda_value(n)),
        lambda_left=float(model.lambda_value(np.nextafter(n,0))),lambda_right=float(model.lambda_value(np.nextafter(n,np.inf))),
        log_slope_left=float(model.log_slope(n,side='left')),log_slope_right=float(model.log_slope(n,side='right')),
        conditional_zero=n in lambda_zero(model),estimated_support_right=n>=1) for n in nodes])
    unit_checks(out,setup,cfg)
    # Reuse only this phase's checkpoints, keyed by all scenario settings.
    allpaths=[];seed=cfg['random_state']
    for domain in ('main','extended'):
        target=out/f'P0_{domain}_budget_path.csv'
        old=pd.read_csv(target,float_precision='round_trip').to_dict('records') if target.exists() else []
        known={(r['cost_function'],int(r['L_ctx']),r['budget']):r for r in old};rows=[]
        for family in setup['official']['quality_costs']:
            for l in setup['official_contexts']:
                for C in budgets:
                    key=(family,l,float(C))
                    if key in known:rec=known[key]
                    else:
                        prob=ResourceProblem(setup,model,cfg,ext,C,family,l,domain,'P0')
                        rec,_=prob.solve(independent=C in official,seed=seed);seed+=1
                    rows.append(rec)
                csv(target,rows)
                logging.info('P0 %s %s L=%s complete (%s budgets)',domain,family,l,len(budgets))
        allpaths+=rows
    path=pd.DataFrame(allpaths)
    p1file=out/'P1_official_budget_results.csv'
    old=pd.read_csv(p1file,float_precision='round_trip').to_dict('records') if p1file.exists() else []
    known={(r['domain'],r['cost_function'],int(r['L_ctx']),r['budget']):r for r in old};p1=[]
    for domain in ('main','extended'):
        for family in setup['official']['quality_costs']:
            for l in setup['official_contexts']:
                for C in sorted(official):
                    key=(domain,family,l,C)
                    if key in known:rec=known[key]
                    else:
                        rec,_=ResourceProblem(setup,model,cfg,ext,C,family,l,domain,'P1').solve(independent=True,seed=seed);seed+=1
                    p1.append(rec)
                csv(p1file,p1)
                logging.info('P1 official %s %s L=%s complete',domain,family,l)
    sensitivityfile=out/'Q0_sensitivity_results.csv';sens=[]
    old=pd.read_csv(sensitivityfile,float_precision='round_trip').to_dict('records') if sensitivityfile.exists() else []
    known={(r['Q0'],r['cost_function'],int(r['L_ctx']),r['budget']):r for r in old}
    for q0 in cfg['Q0_sensitivity']:
        for family in setup['official']['quality_costs']:
            for l in (2048,32768,131072):
                for C in sorted(official):
                    key=(q0,family,l,C)
                    if key in known:rec=known[key]
                    else:
                        rec,_=ResourceProblem(setup,model,cfg,ext,C,family,l,'main','P0',q0).solve(independent=True,seed=seed);seed+=1
                    sens.append(rec)
        csv(sensitivityfile,sens);logging.info('Q0 sensitivity %.10f complete',q0)
    official_p0=path[path.budget.isin(official)]
    officialall=pd.concat([official_p0,pd.DataFrame(p1)],ignore_index=True)
    csv(out/'official_budget_results.csv',officialall)
    optimizer= pd.concat([officialall,pd.DataFrame(sens)],ignore_index=True)
    csv(out/'optimizer_cross_checks.csv',optimizer[[c for c in ['domain','mode','Q0','budget','cost_function','L_ctx','status','L_star','DE_raw_loss','DE_refined_loss','DE_minus_final_loss','local_start_count','local_success_count'] if c in optimizer]])
    candidate_checks(out,setup,model,cfg,ext,candidates,budgets)
    three_dimensional_checks(out,setup,model,cfg,ext,path)
    compare=[]
    for keys,g in officialall.groupby(['domain','cost_function','L_ctx','budget']):
        p0=g[g['mode']=='P0'].iloc[0];p1r=g[g['mode']=='P1'].iloc[0]
        row=dict(zip(['domain','cost_function','L_ctx','budget'],keys));row.update(status=p0.status)
        if p0.feasible and p1r.feasible:
            row.update(P0_Loss=p0.L_star,P1_Loss=p1r.L_star,Loss_improvement=p0.L_star-p1r.L_star,
                 N_change=p1r.N_star-p0.N_star,D_change=p1r.D_star-p0.D_star,Q_change=p1r.Q_star-p0.Q_star,
                 selected_candidate=p1r.selected_candidate,lambda_value=p1r.lambda_value,
                 mixture_transfer_uses_estimated_support=p1r.mixture_transfer_uses_estimated_support,mixture_evidence=p1r.mixture_evidence)
        compare.append(row)
    csv(out/'P0_vs_P1_official_comparison.csv',compare)
    logging.info('Resource solve and independent checks complete; no P2 or Question 4')

# ============================================================
# Source: src/optimization/transitions.py
# Original lines: 1-176
# Purpose: 连续预算扫描路径上的Type I/II转移检验与区间
# ============================================================
"""Type I active-set and Type II curve diagnostics, always on solved P0 paths."""
from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from .resource_solver import load, ResourceProblem, csv

FLAGS=['active_N_lower','active_N_upper','active_D_lower','active_D_upper',
       'active_Q_lower','active_Q_upper','budget_binding']


def bisect_status(fun,lo,hi,initial,cfg):
    """Refine the first bracketed boolean crossing in log10 budget."""
    for i in range(cfg['transition_max_bisections']):
        if hi-lo<=cfg['transition_log10_bracket_tolerance']:break
        mid=(lo+hi)/2
        if bool(fun(10**mid))==initial:lo=mid
        else:hi=mid
    return lo,hi,i+1


def soft_fit(x,y,cfg,rng):
    """Continuous single-break hinge versus one line; block residual perturbation CI."""
    n=len(x);m=cfg['soft_transition_min_points_per_side']
    if n<2*m+1 or np.ptp(y)<1e-10:return dict(status='NO_VARIATION',delta_BIC=np.nan,significant=False)
    xx=x-x.mean();linear=np.c_[np.ones(n),xx]
    coef=np.linalg.lstsq(linear,y,rcond=None)[0];res=y-linear@coef
    sse0=float(res@res);locs=(x[m-1:n-m]+x[m:n-m+1])/2
    designs=np.array([np.c_[np.ones(n),xx,np.maximum(0,x-k)] for k in locs])
    inv=np.array([np.linalg.pinv(a) for a in designs])
    coefs=np.einsum('kjn,n->kj',inv,y);fits=np.einsum('knj,kj->kn',designs,coefs)
    sses=np.sum((fits-y)**2,axis=1);best=int(np.argmin(sses));sse1=float(sses[best])
    floor=max(1e-24,float(np.var(y))*n*1e-14)
    bic0=n*np.log(max(sse0,floor)/n)+2*np.log(n)
    bic1=n*np.log(max(sse1,floor)/n)+4*np.log(n) # intercept, two slopes, selected breakpoint
    delta=float(bic0-bic1);significant=delta>=cfg['soft_transition_delta_BIC']
    result=dict(status='FITTED',n=n,delta_BIC=delta,significant=significant,break_log10C=float(locs[best]),
        break_budget=float(10**locs[best]),slope_left=float(coefs[best,1]),slope_right=float(coefs[best,1]+coefs[best,2]),
        points_left=int((x<locs[best]).sum()),points_right=int((x>locs[best]).sum()),linear_SSE=sse0,piecewise_SSE=sse1)
    if significant:
        residual=y-fits[best];residual-=residual.mean();reps=cfg['soft_transition_bootstrap_repetitions'];block=cfg['soft_transition_bootstrap_block_length']
        perturb=[]
        for _ in range(reps):
            starts=rng.integers(0,n,size=int(np.ceil(n/block)))
            idx=np.concatenate([(s+np.arange(block))%n for s in starts])[:n]
            perturb.append(fits[best]+residual[idx])
        ys=np.array(perturb);boot_coef=np.einsum('kjn,rn->rkj',inv,ys)
        boot_fit=np.einsum('knj,rkj->rkn',designs,boot_coef)
        boot_sse=np.sum((boot_fit-ys[:,None,:])**2,axis=2);chosen=locs[np.argmin(boot_sse,axis=1)]
        lows,highs=np.quantile(chosen,[.025,.975])
        result.update(break_interval_low=10**float(lows),break_interval_high=10**float(highs),
            interval_method='circular_block_residual_perturbation_not_physical_sampling_CI',bootstrap_repetitions=reps,
            breakpoint_grid_resolution='midpoints between solved log-budget points')
    return result


def complete_feasibility_boundary_events(root):
    """Include active releases between the analytic feasibility boundary and first grid point."""
    root=Path(root);out=root/'outputs/optimization/phase2'
    setup,frozen,cfg,model,candidates,ext=load(root)
    hard=pd.read_csv(out/'typeI_active_set_transitions.csv',float_precision='round_trip')
    refined=pd.read_csv(out/'refined_budget_solutions.csv',float_precision='round_trip').to_dict('records')
    extra=[]
    for domain in ('main','extended'):
        path=pd.read_csv(out/f'P0_{domain}_budget_path.csv',float_precision='round_trip')
        for (family,lctx),frame in path[path.feasible==True].groupby(['cost_function','L_ctx']):
            first=frame.sort_values('budget').iloc[0]
            problem=ResourceProblem(setup,model,cfg,ext,first.budget,family,int(lctx),domain,'P0')
            minimum=problem.a*problem.nlo*problem.dlo*1e18
            if first.budget<=minimum*(1+1e-12):continue
            minimum_record,_=ResourceProblem(setup,model,cfg,ext,minimum,family,int(lctx),domain,'P0').solve()
            cache={minimum:minimum_record,float(first.budget):first.to_dict()}
            refined.append(minimum_record)
            def solve(C):
                if C not in cache:
                    cache[C],_=ResourceProblem(setup,model,cfg,ext,C,family,int(lctx),domain,'P0').solve()
                    refined.append(cache[C])
                return cache[C]
            for flag in FLAGS:
                if bool(minimum_record[flag])==bool(first[flag]):continue
                old=hard[(hard.domain==domain)&(hard.cost_function==family)&(hard.L_ctx==lctx)&(hard.constraint==flag)]
                if len(old) and old.budget_critical.min()<=first.budget:continue
                lo,hi,count=bisect_status(lambda C:solve(C)[flag],np.log10(minimum),np.log10(first.budget),bool(minimum_record[flag]),cfg)
                extra.append(dict(domain=domain,cost_function=family,L_ctx=int(lctx),constraint=flag,
                    transition=f'{bool(minimum_record[flag])}→{bool(first[flag])}',budget_critical=10**((lo+hi)/2),
                    bracket_low=10**lo,bracket_high=10**hi,method='feasibility_boundary_log_budget_bisection',
                    active_tolerance=cfg['active_tolerance'],bisections=count,log10_bracket_width=hi-lo))
    if extra:
        csv(out/'typeI_active_set_transitions.csv',pd.concat([hard,pd.DataFrame(extra)],ignore_index=True))
        csv(out/'refined_budget_solutions.csv',refined)
        p=out/'transition_execution_summary.json';summary=json.loads(p.read_text())
        summary['typeI_rows']=len(hard)+len(extra);summary['feasibility_boundary_release_rows']=len(extra)
        p.write_text(json.dumps(summary,indent=2),encoding='utf-8')


def run_diagnose(root):
    """Detect transitions only after solved paths exist; keep main/extrapolation apart."""
    root=Path(root);out=root/'outputs/optimization/phase2'
    setup,frozen,cfg,model,candidates,ext=load(root)
    path=pd.concat([pd.read_csv(out/f'P0_{d}_budget_path.csv',float_precision='round_trip') for d in ('main','extended')],ignore_index=True)
    typeI=[];typeII=[];thresholds=[];peaks=[];saturation=[];shadow=[];refined=[];monotonic=[]
    rng=np.random.default_rng(cfg['random_state']);families=list(setup['official']['quality_costs'])
    for (domain,family,lctx),frame in path.groupby(['domain','cost_function','L_ctx'],sort=False):
        f=frame[frame.feasible==True].sort_values('budget').copy();base=dict(domain=domain,cost_function=family,L_ctx=int(lctx))
        if f.empty:continue
        cache={float(r['budget']):r for r in f.to_dict('records')}
        def solve(C):
            C=float(C)
            if C not in cache:
                rec,_=ResourceProblem(setup,model,cfg,ext,C,family,int(lctx),domain,'P0').solve()
                cache[C]=rec;refined.append(rec)
            return cache[C]
        prob=ResourceProblem(setup,model,cfg,ext,float(f.budget.max()),family,int(lctx),domain,'P0')
        Cmin=prob.a*prob.nlo*prob.dlo*1e18
        Csat=prob.dmax*(prob.a*prob.nmax+float(prob.h(1.)))*1e18
        saturation.append(dict(**base,exact_saturation_budget=Csat,N_upper=prob.nmax,D_upper=prob.dmax,Q_upper=1.,
                               within_budget_grid=Csat<=frame.budget.max(),first_saturated_grid_budget=float(f.loc[f.evidence_domain_saturated==True,'budget'].min()) if (f.evidence_domain_saturated==True).any() else np.nan))
        typeI.append(dict(**base,constraint='feasibility',transition='INFEASIBLE→FEASIBLE',budget_critical=Cmin,
                          bracket_low=Cmin,bracket_high=Cmin,method='analytic_minimum_cost',active_tolerance=cfg['active_tolerance']))
        for flag in FLAGS:
            arr=f[flag].astype(bool).to_numpy();bud=f.budget.to_numpy()
            for i in np.flatnonzero(arr[1:]!=arr[:-1]):
                lo,hi,count=bisect_status(lambda C:solve(C)[flag],np.log10(bud[i]),np.log10(bud[i+1]),bool(arr[i]),cfg)
                C=10**((lo+hi)/2)
                typeI.append(dict(**base,constraint=flag,transition=f'{bool(arr[i])}→{bool(arr[i+1])}',budget_critical=C,
                    bracket_low=10**lo,bracket_high=10**hi,method='local_log_budget_bisection',bisections=count,
                    active_tolerance=cfg['active_tolerance'],log10_bracket_width=hi-lo))
        # First crossing, with domain-minimum lower bracket if already crossed at first grid point.
        targets=[('departure_Q0_plus_0.01',cfg['Q0_main']+cfg['quality_departure_absolute'])]+[(str(q),q) for q in cfg['quality_thresholds']]
        for name,target in targets:
            tolerance=1e-7 if target==1 else 1e-9
            reached=f.Q_star.to_numpy()>=target-tolerance;inds=np.flatnonzero(reached)
            if not len(inds):
                thresholds.append(dict(**base,threshold=name,target_Q=target,status='NOT_REACHED_IN_GRID',budget_threshold=np.nan));continue
            k=int(inds[0]);upper=float(f.budget.iloc[k]);lower=float(f.budget.iloc[k-1]) if k else Cmin
            lo,hi,_=bisect_status(lambda C:solve(C)['Q_star']>=target-tolerance,np.log10(lower),np.log10(upper),False,cfg)
            thresholds.append(dict(**base,threshold=name,target_Q=target,status='REACHED',budget_threshold=10**((lo+hi)/2),
                bracket_low=10**lo,bracket_high=10**hi,Q_tolerance=tolerance,
                Q_path_nonmonotonic=bool((np.diff(f.Q_star)<-1e-5).any())))
        # Shadow price finite differences on actual, unequal budget coordinates.
        Cs=f.budget.to_numpy();Ls=f.L_star.to_numpy();x=np.log10(Cs)
        mu=-np.gradient(Ls,Cs,edge_order=2);elasticity=-np.gradient(np.log(Ls-model.c['E']),np.log(Cs),edge_order=2)
        mu[np.abs(mu)<1e-30]=0
        for i,r in enumerate(f.to_dict('records')):
            shadow.append(dict(**base,budget=Cs[i],L_star=Ls[i],shadow_price=mu[i],budget_elasticity=elasticity[i],
                endpoint_one_sided=i in (0,len(Cs)-1),negative_mu_numerical=bool(mu[i]<-1e-24)))
        deltas=np.diff(Ls);monotonic.append(dict(**base,maximum_loss_increase=float(max(0,deltas.max())),
             passed=bool((deltas<=cfg['loss_validation_tolerance']).all()),n_comparisons=len(deltas)))
        for metric in ['s_train','s_quality','s_attn','ln_N','ln_D','Q']:
            values=np.log(f.N_star.to_numpy()) if metric=='ln_N' else np.log(f.D_star.to_numpy()) if metric=='ln_D' else f.Q_star.to_numpy() if metric=='Q' else f[metric].to_numpy()
            result=soft_fit(x,values,cfg,rng);typeII.append(dict(**base,metric=metric,**result))
        values=f.s_quality.to_numpy();k=int(np.argmax(values));peakC=Cs[k];peak=values[k]
        reversal=k>0 and k<len(Cs)-1 and peak-values[0]>1e-4 and peak-values[-1]>1e-4
        if reversal:
            result=minimize_scalar(lambda logC:-solve(10**logC)['s_quality'],bounds=(np.log10(Cs[k-1]),np.log10(Cs[k+1])),method='bounded',options={'xatol':1e-5})
            peakC=10**result.x;peak=solve(peakC)['s_quality']
        near=[r for r in typeI if r['domain']==domain and r['cost_function']==family and r['L_ctx']==lctx and r['constraint']!='feasibility']
        closest=min(near,key=lambda r:abs(np.log10(r['budget_critical']/peakC))) if near else None
        peaks.append(dict(**base,quality_share_reversal=bool(reversal),C_peak_Q=peakC,peak_s_quality=peak,
              start_s_quality=values[0],end_s_quality=values[-1],nearest_active_transition=closest['constraint'] if closest else '',
              nearest_transition_budget=closest['budget_critical'] if closest else np.nan,
              distance_log10_budget=abs(np.log10(closest['budget_critical']/peakC)) if closest else np.nan,
              active_Q_upper_at_peak=solve(peakC)['active_Q_upper']))
        logging.info('Transitions %s %s L=%s complete; refined %s budget points',domain,family,lctx,len(cache)-len(f))
        # Incremental checkpoint, useful if a desktop connection closes.
        for name,records in [('typeI_active_set_transitions.csv',typeI),('typeII_soft_transitions.csv',typeII),
             ('quality_threshold_budgets.csv',thresholds),('quality_cost_share_peaks.csv',peaks),('domain_saturation_budgets.csv',saturation),
             ('shadow_price_budget_elasticity.csv',shadow),('monotonicity_checks.csv',monotonic),('refined_budget_solutions.csv',refined)]:csv(out/name,records)
    if not all(r['passed'] for r in monotonic):raise ValueError('Loss increases with budget; must diagnose optimizer')
    summary=dict(typeI_rows=len(typeI),typeII_significant_metrics=sum(bool(r['significant']) for r in typeII),
       typeII_rows=len(typeII),quality_share_reversal_scenarios=sum(bool(r['quality_share_reversal']) for r in peaks),
       main_grid_points=path[path.domain=='main'].budget.nunique(),P2_performed=False)
    (out/'transition_execution_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

# ============================================================
# Source: src/optimization/resource_reporting.py
# Original lines: 1-248
# Purpose: 最终CSV/图表/报告呈现，compact省略纯绘图和排版
# ============================================================
"""Question 3 optimized results, readable evidence limits and scientific figures."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .resource_solver import load,csv,ResourceProblem


def mdtable(frame,columns=None):
    """Readable Markdown with bounded display precision; CSV retains full precision."""
    f=frame if columns is None else frame[columns]
    lines=['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']
    for r in f.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.7g}' if isinstance(v,float) and np.isfinite(v) else '—' if pd.isna(v) else str(v).replace('|','/') for v in r)+' |')
    return '\n'.join(lines)


def figures(root,out,path,official,compare,hard,soft,shadow):
    """Twelve figures, consistently distinguishing main/extrapolation and infeasibility."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageOps,ImageDraw
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figdir=root/'outputs/figures/question3_phase2';figdir.mkdir(parents=True,exist_ok=True)
    colors={'exponential':'#4477AA','power':'#CC6677','logarithmic':'#228833'};families=list(colors);manifest=[]
    def finish(fig,name,caption):
        fig.tight_layout();target=figdir/name;fig.savefig(target,dpi=180);plt.close(fig)
        manifest.append(dict(figure=target.relative_to(root).as_posix(),caption=caption))
    feasible=path[path.feasible==True].copy();main=feasible[feasible.domain=='main']
    for i,(metric,label) in enumerate([('N_star','N* (billion parameters)'),('D_star','D* (billion tokens)'),
                                     ('Q_star','Q*'),('L_star','Optimal validation loss')],1):
        fig,ax=plt.subplots(figsize=(8,4))
        for family in families:
            for domain,style in [('main','-'),('extended','--')]:
                f=feasible[(feasible.cost_function==family)&(feasible.L_ctx==2048)&(feasible.domain==domain)].sort_values('budget')
                ax.plot(f.budget,f[metric],color=colors[family],ls=style,label=family+' '+domain)
        ax.set(xscale='log',xlabel='Compute budget (FLOPs)',ylabel=label,title='P0 allocations, context 2048')
        if metric in ('N_star','D_star'):ax.set_yscale('log')
        ax.legend(fontsize=8,ncol=2);finish(fig,f'{i:02d}_{metric}_vs_budget.png',f'P0 {metric}：主域实线、扩展域虚线；Lctx=2048')
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,family in zip(axes,families):
        f=main[(main.cost_function==family)&(main.L_ctx==2048)].sort_values('budget')
        for m,color in [('s_train','#4477AA'),('s_quality','#CC6677'),('s_attn','#228833')]:ax.plot(f.budget,f[m],label=m,color=color)
        ax.plot(f.budget,1-f.s_train-f.s_quality-f.s_attn,ls=':',color='gray',label='unused fraction')
        ax.set(xscale='log',xlabel='Budget (FLOPs)',title=family,ylim=(-.02,1.02))
    axes[0].set_ylabel('Cost / allocated budget');axes[-1].legend(fontsize=8)
    fig.suptitle('P0 main-domain cost shares, context 2048');finish(fig,'05_cost_shares.png','预算分母下三类成本与未使用份额')
    fig,ax=plt.subplots(figsize=(8,4))
    for family in families:
        f=shadow[(shadow.domain=='main')&(shadow.cost_function==family)&(shadow.L_ctx==2048)].sort_values('budget')
        f=f[f.shadow_price>1e-30];ax.plot(f.budget,f.shadow_price,color=colors[family],label=family)
    ax.set(xscale='log',yscale='log',xlabel='Budget (FLOPs)',ylabel='Shadow price (-dLoss*/dC)',title='P0 shadow prices; zero plateau omitted from log axis')
    ax.legend();finish(fig,'06_shadow_price.png','有限差分影子价格；饱和后的零值不画在对数轴上')
    fig,axes=plt.subplots(1,5,figsize=(15,3.5),sharey=True)
    for ax,l in zip(axes,[2048,4096,8192,32768,131072]):
        for family in families:
            f=main[(main.cost_function==family)&(main.L_ctx==l)].sort_values('budget')
            ax.plot(f.budget,f.Q_star,label=family,color=colors[family])
        ax.set(xscale='log',xlabel='Budget (FLOPs)',title=f'Context {l}',ylim=(.70,1.015));ax.tick_params(axis='x',labelsize=8)
    axes[0].set_ylabel('Q*');axes[-1].legend(fontsize=7)
    finish(fig,'07_quality_cost_function_comparison.png','五上下文下三种官方成本的质量投资路径')
    for index,metric,label in [(8,'N_star','N* (billion parameters)'),(9,'D_star','D* (billion tokens)'),(10,'L_star','Loss*')]:
        fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
        for ax,family in zip(axes,families):
            for l in [2048,4096,8192,32768,131072]:
                f=main[(main.cost_function==family)&(main.L_ctx==l)].sort_values('budget')
                ax.plot(f.budget,f[metric],label=str(l))
            ax.set(xscale='log',xlabel='Budget (FLOPs)',title=family)
            if metric in ('N_star','D_star'):ax.set_yscale('log')
        axes[0].set_ylabel(label);axes[-1].legend(title='Context',fontsize=8)
        finish(fig,f'{index:02d}_five_context_{metric}.png','主域五上下文对照：'+metric)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,C in zip(axes,[1e19,1e22,1e24]):
        for family in families:
            f=compare[(compare.domain=='main')&(compare.cost_function==family)&(compare.budget==C)&(compare.status=='SOLVED')].sort_values('L_ctx')
            ax.plot(f.L_ctx.astype(str),f.Loss_improvement,'o-',color=colors[family],label=family)
        ax.set(xlabel='Context',title=f'Budget {C:.0e}');ax.tick_params(axis='x',labelrotation=40)
    axes[0].set_ylabel('P0 Loss* - P1 Loss*');axes[-1].legend(fontsize=8)
    fig.suptitle('P1 modeled improvement; large-N transfer is conditional/estimated')
    finish(fig,'11_P0_vs_P1_improvement.png','P1相对P0的模型Loss改善，非新观测实验收益')
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    labels=['active_D_lower','active_Q_lower','active_Q_upper','active_D_upper','active_N_upper','budget_binding']
    for ax,family in zip(axes,families):
        h=hard[(hard.domain=='main')&(hard.cost_function==family)&(hard.L_ctx==2048)]
        for i,label in enumerate(labels):
            g=h[h.constraint==label];ax.scatter(g.budget_critical,np.full(len(g),i),s=40)
            for r in g.itertuples():ax.annotate(r.transition,(r.budget_critical,i),xytext=(3,5),textcoords='offset points',fontsize=7)
        s=soft[(soft.domain=='main')&(soft.cost_function==family)&(soft.L_ctx==2048)&(soft.significant==True)]
        for v in s.break_budget.dropna().unique():ax.axvline(v,ls=':',color='gray',alpha=.3)
        ax.set(xscale='log',yticks=range(len(labels)),yticklabels=labels,title=family)
    axes[-1].set_xlabel('Budget (FLOPs); dots: Type I, dotted lines: significant Type II curve fits')
    finish(fig,'12_structural_transitions.png','2048上下文硬转移与显著软断点示意；软线不是额外独立物理相变')
    csv(out/'figure_manifest.csv',manifest)
    sheet=Image.new('RGB',(1600,2040),'white')
    for i,r in enumerate(manifest):
        im=Image.open(root/r['figure']).convert('RGB');thumb=ImageOps.contain(im,(785,310))
        x=(i%2)*800;y=(i//2)*340;sheet.paste(thumb,(x,y));ImageDraw.Draw(sheet).text((x+5,y+315),Path(r['figure']).name,fill='black')
    sheet.save(out/'figure_contact_sheet.png')
    return manifest


def run_report(root):
    """Write every requested result from saved numerical tables, with scope limits."""
    root=Path(root);out=root/'outputs/optimization/phase2';setup,frozen,cfg,model,candidates,ext=load(root)
    from .transitions import complete_feasibility_boundary_events
    complete_feasibility_boundary_events(root)
    path=pd.concat([pd.read_csv(out/f'P0_{d}_budget_path.csv',float_precision='round_trip') for d in ('main','extended')],ignore_index=True)
    read=lambda name:pd.read_csv(out/name,float_precision='round_trip')
    official=read('official_budget_results.csv');comparison=read('P0_vs_P1_official_comparison.csv')
    hard=read('typeI_active_set_transitions.csv');soft=read('typeII_soft_transitions.csv');shadow=read('shadow_price_budget_elasticity.csv')
    saturation=read('domain_saturation_budgets.csv');peaks=read('quality_cost_share_peaks.csv');q0=read('Q0_sensitivity_results.csv')
    optimizer=read('optimizer_cross_checks.csv');full3d=read('D_elimination_3D_check.csv');enumeration=read('P1_candidate_reduction_check.csv')
    supports=[]
    for event_type,events in [('Type I',hard.to_dict('records')),
                             ('Type II',soft[soft.significant==True].to_dict('records'))]:
        for event in events:
            C=event['budget_critical'] if event_type=='Type I' else event['break_budget']
            g=shadow[(shadow.domain==event['domain'])&(shadow.cost_function==event['cost_function'])&(shadow.L_ctx==event['L_ctx'])].sort_values('budget')
            before=g[g.budget<C];after=g[g.budget>C]
            if before.empty or after.empty:continue
            left,right=before.iloc[-1],after.iloc[0]
            supports.append(dict(domain=event['domain'],cost_function=event['cost_function'],L_ctx=event['L_ctx'],
                event_type=event_type,event=event.get('constraint',event.get('metric')),transition_budget=C,
                neighbor_budget_left=left.budget,neighbor_budget_right=right.budget,
                mu_left=left.shadow_price,mu_right=right.shadow_price,
                elasticity_left=left.budget_elasticity,elasticity_right=right.budget_elasticity,
                elasticity_change=right.budget_elasticity-left.budget_elasticity,
                purpose='finite-grid corroboration only, not an independent transition criterion'))
    csv(out/'transition_shadow_support.csv',supports)
    main=official[(official.domain=='main')&(official['mode']=='P0')].sort_values(['budget','cost_function','L_ctx'])
    expanded=official[(official.domain=='extended')&(official['mode']=='P0')].sort_values(['budget','cost_function','L_ctx'])
    contextcompare=[]
    for (family,C),g in main.groupby(['cost_function','budget']):
        left=g[g.L_ctx==8192].iloc[0];right=g[g.L_ctx==32768].iloc[0]
        contextcompare.append(dict(cost_function=family,budget=C,N_8192=left.get('N_star',np.nan),N_32768=right.get('N_star',np.nan),
            D_8192=left.get('D_star',np.nan),D_32768=right.get('D_star',np.nan),Q_8192=left.get('Q_star',np.nan),Q_32768=right.get('Q_star',np.nan),
            Loss_change=right.get('L_star',np.nan)-left.get('L_star',np.nan),ratio_8192=setup['official']['eta']*8192/6,ratio_32768=setup['official']['eta']*32768/6))
    csv(out/'context_8192_vs_32768.csv',contextcompare)
    senscompare=[]
    for r in q0.to_dict('records'):
        match=main[(main.cost_function==r['cost_function'])&(main.L_ctx==r['L_ctx'])&(main.budget==r['budget'])].iloc[0]
        row={k:r[k] for k in ('Q0','budget','cost_function','L_ctx','status')}
        if r['feasible'] and match.feasible:
            row.update(N_change=r['N_star']-match.N_star,D_change=r['D_star']-match.D_star,Q_change=r['Q_star']-match.Q_star,
                Loss_change=r['L_star']-match.L_star,s_train_change=r['s_train']-match.s_train,s_quality_change=r['s_quality']-match.s_quality,
                s_attn_change=r['s_attn']-match.s_attn,
                active_structure_changed=any(bool(r[k])!=bool(match[k]) for k in ['active_N_lower','active_N_upper','active_D_lower','active_D_upper','active_Q_lower','active_Q_upper','budget_binding']))
        senscompare.append(row)
    csv(out/'Q0_sensitivity_comparison.csv',senscompare)
    sensframe=pd.DataFrame(senscompare)
    figs=figures(root,out,path,official,comparison,hard,soft,shadow)
    tablecols=['budget','cost_function','L_ctx','status','N_star','D_star','Q_star','L_star']
    costs=['budget','cost_function','L_ctx','C_train','C_quality','C_attn','unused_budget','budget_binding','evidence_domain_saturated','extrapolative_Q']
    mainsoft=soft[(soft.domain=='main')&(soft.significant==True)];mainhard=hard[hard.domain=='main']
    means=comparison[comparison.domain=='main'].groupby('budget').Loss_improvement.agg(['min','max','mean']).reset_index()
    summary=dict(main_official_P0_rows=len(main),main_official_P0_feasible=int(main.feasible.sum()),
        grid_base_count=setup['config']['budget_log_grid_count'],grid_with_official_count=int(path.budget.nunique()),
        main_typeI_rows=len(mainhard),main_typeII_significant_metric_fits=len(mainsoft),
        main_saturation_budget_min=float(saturation[saturation.domain=='main'].exact_saturation_budget.min()),
        main_saturation_budget_max=float(saturation[saturation.domain=='main'].exact_saturation_budget.max()),
        main_quality_share_reversals=int(peaks[peaks.domain=='main'].quality_share_reversal.sum()),
        main_P1_loss_improvement_min=float(comparison[comparison.domain=='main'].Loss_improvement.min()),
        main_P1_loss_improvement_max=float(comparison[comparison.domain=='main'].Loss_improvement.max()),
        high_budget_main_saturated_count=int(main[main.budget==1e24].evidence_domain_saturated.sum()),
        high_budget_extended_N_or_D_extrapolative_count=int(((expanded[expanded.budget==1e24].extrapolative_N==True)|(expanded[expanded.budget==1e24].extrapolative_D==True)).sum()),
        Q0_active_structure_changed_count=int(sensframe.active_structure_changed.fillna(False).sum()),
        Q0_Loss_change_min=float(sensframe.Loss_change.min()),Q0_Loss_change_max=float(sensframe.Loss_change.max()),
        maximum_optimizer_loss_gap=float(optimizer.DE_minus_final_loss.max()),
        full3D_checks_passed=int(full3d.passed.sum()),enumeration_checks_passed=int(enumeration.passed.sum()),P2_performed=False,Question4_started=False)
    (out/'execution_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    text='# 问题三 PHASE 2 正式资源优化与结构性转移\n\n'
    text+='冻结问题一/二与PHASE 1保持只读。P0为正文主方案，三类官方成本平行运行；P1为经验配方增强，主域和扩展域分开。没有P2 simplex优化或问题四。\n\n'
    text+='## 1 解析消去D与单位\n\n固定N/Q/p/Lctx时，∂L/∂D_B=-beta B D_B^(-beta-1)<0；成本线性为D_B c_D。'
    text+='因此D*=min(D_max,C_budget/c_D)，c_D=N_B(6+etaLctx)10^18+10^9[g(Q)-g(Q0)]。D*<D_min则该N/Q不可行。'
    text+='这使连续优化降为logN/Q二维，没有强设预算等号；D达到上界时允许预算松弛。'
    text+='目标仍用B单位，成本显式转换实际参数/token。新15组Decimal单位核验通过，未重拟合参数。\n\n'
    text+='冻结目标：L_gen=E+A N_B^(-alpha)+B D_B^(-beta)+C_Qloss(N_B/N0_B)^(-u_Q)(-lnQ)+lambda_p(N_B)[M(p)-M_ref]。'
    text+='C_Qloss对应冻结JSON的quality.C_Q字段，只是符号区分，不改系数；它与质量算力成本C_quality不同。\n\n'
    text+='## 2 求解与独立最优性核验\n\n每个光滑分段采用121×81网格及可行边界/预算D上界折线、至少10个不同初值SLSQP，显式评估节点两侧和节点本身。'
    text+='官方以及Q0敏感性可行场景全部使用全域differential_evolution独立复核，再细化其候选。'
    text+=f'最大DE细化与最终Loss差={summary["maximum_optimizer_loss_gap"]:.7g}；完整三维核验{summary["full3D_checks_passed"]}/{len(full3d)}通过，'
    text+=f'832配方全枚举核验{summary["enumeration_checks_passed"]}/{len(enumeration)}通过。'
    text+='全枚举是在每个随机场景的31×23资源点上对832配方逐一评价，验证点态及整个资源网格最优一致，非逐配方重复连续求解。'
    text+='主/扩展路径Loss随预算不恶化的检查见monotonicity_checks.csv；这提供强数值复核，有限多优化器不构成任意非凸问题的形式化全局证明。\n\n'
    text+='## 3 官方低中高预算 主域P0\n\n'+mdtable(main,tablecols)+'\n\n'
    text+='成本/未使用预算均为FLOPs：\n\n'+mdtable(main,costs)+'\n\n'
    text+='最低预算×131072上下文最低成本约2.27247e19 FLOPs，故1e19证据支持域内不可行；不可行行保留状态、没有伪造决策。'
    text+='Q=1超过问题一实际样本质量上界，虽在B6半合成数值范围内，不能称直接验证。\n\n'
    text+='## 4 主域饱和与扩展情景\n\n'
    text+='P0各资源边际Loss均负。主域饱和预算可解析为D_max[(6+etaLctx)N_max+Δg(1)/10^9]10^18；'
    text+='达到此预算后P0取三个上界且剩余预算不再改变冻结目标。不是求解失败，也不能覆盖此结果而偷偷扩大边界。\n\n'
    text+=mdtable(saturation[saturation.domain=='main'],['cost_function','L_ctx','exact_saturation_budget','first_saturated_grid_budget'])+'\n\n'
    text+='以下是独立扩展域P0情景，所有超出主N/D/Q实测范围的点分别标记，scenario_extrapolation始终为True：\n\n'+mdtable(expanded,tablecols)+'\n\n'
    text+=f'1e24主域P0全部{summary["high_budget_main_saturated_count"]}个成本×上下文场景饱和；扩展域对应{summary["high_budget_extended_N_or_D_extrapolative_count"]}个结果在N或D上超出主域。'
    text+='B9元数据、B10估算Loss及总/激活参数差异不能为这些联合最优点提供直接验证。\n\n'
    text+='## 5 三种质量成本与投资节奏\n\n'
    text+='三类公式与参数均保持题给：1e7exp(6Q)、5e9Q^4、2e9ln(1+10Q)，均扣除g(Q0)、乘实际D。'
    text+='报告不按结果是否漂亮选择主成本；具体Q脱离Q0+0.01、达到0.9/0.95/0.99/1的局部二分预算见quality_threshold_budgets.csv。'
    text+='“明显离开”以绝对0.01定义（研究者阈值）；满分判定容差1e-7。所有转移预算是冻结模型条件结果。\n\n'
    representative=read('quality_threshold_budgets.csv');representative=representative[(representative.domain=='main')&(representative.L_ctx==2048)]
    text+=mdtable(representative,['cost_function','threshold','budget_threshold'])+'\n\n'
    text+='## 6 上下文与30k临界值\n\n'+mdtable(read('context_cost_ratio.csv'))+'\n\n'
    text+='C_attn/C_train=etaLctx/6与预算/决策/配方无关。8192→32768时比例从0.27307增至1.09227；跨临界表示成本代理的注意力超过基础训练，不是由预算驱动的两个份额交叉。'
    text+='不能要求N/D/Q全部同向变化：它们存在替代、边界和质量成本效应。官方预算对照：\n\n'+mdtable(pd.DataFrame(contextcompare))+'\n\n'
    text+='## 7 Type I硬转移\n\n活跃集采用1e-6归一化容差，先从44点路径发现变化，再局部log预算二分至括区宽≤1e-5。'
    text+='可行性阈值另由解析最低成本给出。并发N上界与预算松弛可对应同一转移，不能数作两个独立经济机制。'
    text+='代表2048上下文的主域事件：\n\n'+mdtable(mainhard[mainhard.L_ctx==2048],['cost_function','constraint','transition','budget_critical','bracket_low','bracket_high'])+'\n\n'
    text+='完整五上下文及扩展域见typeI_active_set_transitions.csv。数值活跃容差下的精确括区，不表示真实系统可支持同样精度。\n\n'
    text+='## 8 Type II软转移及影子价格\n\n对log10预算拟合单线性与连续单断点hinge，两侧至少5点，BIC参数数目分别2/4；只有DeltaBIC≥10才标显著。'
    text+='300次长度3循环块残差扰动提供断点区间；这是确定性模型曲线拟合/网格与近似误差的区间，不是现实转移预算的统计置信区间。'
    text+=f'主域有{len(mainsoft)}条指标曲线满足门槛；不能当作{len(mainsoft)}个独立物理转移。'
    text+='训练与注意力份额固定成比例，其BIC与断点重复，不是独立证据；单断点也只是对含多次边界变化曲线的近似。\n\n'
    text+=mdtable(mainsoft[mainsoft.L_ctx==2048],['cost_function','metric','delta_BIC','break_budget','break_interval_low','break_interval_high'])+'\n\n'
    text+='mu=-dL*/dC以及-E的剩余Loss预算弹性由相邻预算有限差分计算，见shadow_price_budget_elasticity.csv。'
    text+='Type I/II附近左右预算点的mu与弹性变化另列transition_shadow_support.csv，用于逐个事件核对；'
    text+='节点附近有限网格导数只作佐证；饱和后的影子价格应为零。不能单凭mu或弹性定义转移。\n\n'
    text+='## 9 质量成本份额反转\n\n'+mdtable(peaks[peaks.domain=='main'],['cost_function','L_ctx','quality_share_reversal','C_peak_Q','peak_s_quality','active_Q_upper_at_peak','nearest_active_transition'])+'\n\n'
    text+='峰值在预算网格邻域用一维搜索细化；是否先升后降按两侧分别至少变化1e-4判断（数值判据）。'
    text+='Q上界后质量处理总成本仍可随D增长，所以质量份额峰值不必与Q达到1完全同时。\n\n'
    text+='## 10 P1精确配方降维及证据边界\n\n成本与p无关，固定N时lambda>0取最小M、lambda<0取最大M、零点全部等价。'
    text+='因此经验全集合的点态下包络等于两个极值配方的包络；再对N/Q优化是数学精确降维。'
    text+=mdtable(ext,['role','candidate_id','M_p','DeltaM','source_members'])+'\n\n'
    text+='N区间按冻结节点与条件过零点分割；过零是冻结估算段的结果，不是新观测的尺度反转。'
    text+='P1官方预算Loss改善范围：\n\n'+mdtable(means)+'\n\n'
    text+='完整N/D/Q变化及选择配方见P0_vs_P1_official_comparison.csv。N>1B的配比效应明确标mixture_transfer_uses_estimated_support=True、Class D；'
    text+='它只是增强情景，不能当高可信真实配方建议。冻结模型输入配方经过已有零替换，不再预处理；物理部署需回溯原始来源。\n\n'
    text+='## 11 Q0敏感性\n\n仅针对三官方预算×三个指定上下文×三成本运行三候选，共81行；未重跑完整连续路径或改变主Q0。'
    text+=mdtable(sensframe.groupby('Q0').agg(Loss_change_min=('Loss_change','min'),Loss_change_max=('Loss_change','max'),
                       N_change_min=('N_change','min'),N_change_max=('N_change','max'),
                       active_structure_changed=('active_structure_changed','sum')).reset_index())+'\n\n'
    text+='基线改变可能影响低预算质量与Loss以及中预算配置，不能说所有数值不变；'
    text+='最低可行预算不变（CQ(Q0)=0），高预算主域饱和判定需按新的Δg(1)重新计算，但官方1e24依然可覆盖上界成本。'
    text+='敏感性只验证这些官方点，不据此宣称所有连续转移预算对Q0不敏感。\n\n'
    text+='## 12 结论范围与停止点\n\n以上是已冻结模块、成本代理、跨体系Q0映射及N/D矩形支持域下的最优配置；'
    text+='质量局部校准为半合成、估算lambda大尺度段及扩展域不可升级为直接证据。主域高预算饱和是支持范围受限。'
    text+='具备进入下一阶段P2理论敏感性及第三问收尾的技术条件，需人工确认本阶段结果；本轮没有执行P2或开始问题四。\n\n'
    text+='## 图表\n\n'+'\n'.join('- ['+r['caption']+'](../../'+r['figure']+')' for r in figs)+'\n'
    (root/'outputs/report/question3_phase2_optimization.md').write_text(text,encoding='utf-8')
    code=[root/'scripts/18_optimize_compute_resources.py',root/'scripts/18_verify_compute_optimization.py',root/'configs/optimization_phase2.yaml',
          *[root/'src/optimization'/p for p in ['resource_solver.py','transitions.py','resource_reporting.py']]]
    csv(out/'implementation_manifest.csv',[dict(path=p.relative_to(root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in code if p.exists()])
    inputs=[root/'outputs/optimization/phase1/optimization_setup.json',root/setup['frozen_objective_source'],
            root/'outputs/optimization/phase1/empirical_recipe_candidates.csv',root/'outputs/optimization/phase1/compute_budget_scenarios.csv',
            root/'configs/scaling_phase4.yaml',root/'configs/optimization_phase2.yaml']
    csv(out/'input_manifest.csv',[dict(path=p.relative_to(root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs])

# ============================================================
# Source: scripts/18_optimize_compute_resources.py
# Original lines: 1-65
# Purpose: 最终阶段实际执行入口：18_optimize_compute_resources.py
# ============================================================
"""Question 3 Phase 2: frozen resource optimization and transition diagnosis."""
from pathlib import Path
import argparse
import hashlib
import json
import logging
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'outputs/optimization/phase2'
NEW_FILES={'scripts/18_optimize_compute_resources.py','scripts/18_verify_compute_optimization.py',
           'configs/optimization_phase2.yaml','src/optimization/resource_solver.py',
           'src/optimization/transitions.py','src/optimization/resource_reporting.py',
           'outputs/report/question3_phase2_optimization.md'}
NEW_PREFIXES=('outputs/optimization/phase2/','outputs/figures/question3_phase2/')


def sha(path):
    """Hash protected data without parsing or rewriting it."""
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def snapshot():
    """Save a once-only byte baseline including all of frozen Phase 1."""
    OUT.mkdir(parents=True,exist_ok=True);target=OUT/'frozen_before_phase2.json'
    if target.exists():return
    records={}
    for p in ROOT.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ('.git','.codex','.agents')):continue
        rel=p.relative_to(ROOT).as_posix()
        if rel in NEW_FILES or rel.startswith(NEW_PREFIXES):continue
        records[rel]={'size':p.stat().st_size,'sha256':sha(p)}
    target.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Frozen files protected:',len(records),flush=True)


def main():
    """Run authorized Phase 2 only, with checkpointable outputs and logs."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-only',action='store_true')
    parser.add_argument('--stage',choices=['solve','diagnose','report','all'],default='all')
    args=parser.parse_args();snapshot()
    if args.snapshot_only:return
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(OUT/'execution.log',encoding='utf-8')])
    try:
        if args.stage in ('solve','all'):
            from optimization.resource_solver import run_solve
            run_solve(ROOT)
        if args.stage in ('diagnose','all'):
            from optimization.transitions import run_diagnose
            run_diagnose(ROOT)
        if args.stage in ('report','all'):
            from optimization.resource_reporting import run_report
            run_report(ROOT)
    except Exception:
        logging.exception('Phase 2 failed; protected prior files remain read-only');raise


if __name__=='__main__':main()

# ============================================================
# Source: src/optimization/simplex_sensitivity.py
# Original lines: 1-267
# Purpose: P2-A配方证书与数值交叉检验；P2-B边界无界性，仅理论敏感性
# P2-A/P2-B are theory / sensitivity only. Final engineering hierarchy: P0 main, P1 empirical enhancement; P2-B has no finite attained optimum.
# ============================================================
"""Frozen log-contrast simplex analysis; never fit or alter previous phases."""
from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq, differential_evolution, minimize
from scipy.special import softmax, xlogy
from scipy.spatial.distance import jensenshannon, pdist, squareform
from scaling.mixture_transfer import frozen_metadata, predict_recipe
from .resource_solver import csv, load, ResourceProblem


def log_contrast(meta):
    """Algebraically expand the frozen standardized linear ILR predictor."""
    coef = np.asarray(meta['coefficients']) / np.asarray(meta['X_scale'])
    b = float(meta['y_scale']) * coef @ np.asarray(meta['ilr_basis'])
    a = float(meta['y_mean']) + float(meta['y_scale']) * (
        float(meta['intercept']) - np.asarray(meta['X_mean']) @ coef)
    if abs(b.sum()) > 1e-12:
        raise ValueError('Frozen basis no longer gives a zero-sum log contrast')
    return a, b


def exact_extreme(b, epsilon, direction):
    """Global certificate: floor unfavorable signs, water-fill favorable signs.

    To minimize, any positive coefficient is at the floor: transferring its
    mass to a negative coefficient strictly decreases the objective. The
    remaining negative-log problem is convex. Maximization is symmetric.
    """
    favor = -b if direction == 'minM' else b
    active = favor > 0
    if not active.any():
        raise ValueError('Constant or unexpected frozen mixture model')
    p = np.full(len(b), epsilon)
    mass = 1 - (~active).sum() * epsilon
    t = brentq(lambda t: np.maximum(epsilon, t * favor[active]).sum() - mass,
               0, 2 / favor[active].min(), xtol=1e-14)
    p[active] = np.maximum(epsilon, t * favor[active])
    if abs(p.sum() - 1) > 1e-12:
        raise ValueError('Analytic simplex closure failed')
    return p


def recipe_summary(p, pref, candidates, names, cfg):
    """Describe concentration and distance in frozen positive model space."""
    a = candidates[names].to_numpy(float)
    entropy = float(-xlogy(p, p).sum())
    l1 = np.abs(a-p).sum(axis=1)
    js = jensenshannon(a, np.broadcast_to(p, a.shape), axis=1)
    i, k = int(l1.argmin()), int(js.argmin())
    outside = (p < a.min(axis=0)-1e-12) | (p > a.max(axis=0)+1e-12)
    top = np.argsort(-p)[:5]
    far = bool(l1[i]>cfg.get('empirical_neighbor_L1_P95',np.inf) or
               js[k]>cfg.get('empirical_neighbor_JS_P95',np.inf))
    return dict(max_share=float(p.max()), entropy=entropy,
        normalized_entropy=entropy/np.log(len(p)), effective_domain_count=float(np.exp(entropy)),
        HHI=float(p@p), L1_to_ref=float(np.abs(p-pref).sum()),
        L2_to_ref=float(np.linalg.norm(p-pref)), JS_distance_to_ref=float(jensenshannon(p, pref)),
        minimum_empirical_L1=float(l1[i]), minimum_empirical_JS_distance=float(js[k]),
        nearest_empirical_recipe=str(candidates.candidate_id.iloc[i]),
        nearest_empirical_recipe_by_JS=str(candidates.candidate_id.iloc[k]),
        out_of_empirical_range_domains=int(outside.sum()),
        outside_domains=json.dumps(np.asarray(names)[outside].tolist()),
        distance_exceeds_empirical_neighbor_P95=far,
        unsupported_mixture_extrapolation=bool(outside.any() or far),
        unsupported_rule='outside coordinate min/max OR nearest distance exceeds empirical leave-one-out nearest-neighbor P95; descriptive diagnostic, not a confidence region',
        highly_concentrated=bool(p.max()>cfg['high_concentration_share'] or
                                np.exp(entropy)<cfg['high_concentration_effective_domains']),
        near_degenerate=bool(p.max()>cfg['near_degenerate_share']),
        top5=json.dumps([dict(domain=names[j], share=float(p[j])) for j in top]))


def support_calibration(candidates,names,cfg,out):
    """Data-derived descriptive spacing benchmark; never refit M(p)."""
    a=candidates[names].to_numpy(float);result=dict(cfg);rows=[]
    for metric,key in [('cityblock','empirical_neighbor_L1_P95'),('jensenshannon','empirical_neighbor_JS_P95')]:
        d=squareform(pdist(a,metric));np.fill_diagonal(d,np.inf);nearest=d.min(axis=1)
        result[key]=float(np.quantile(nearest,.95))
        rows.append(dict(distance=metric,candidate_count=len(a),nearest_neighbor_P50=float(np.median(nearest)),
            nearest_neighbor_P95=result[key],nearest_neighbor_max=float(nearest.max()),
            purpose='descriptive empirical sparsity benchmark; not a statistical support boundary'))
    csv(out/'empirical_distance_calibration.csv',rows)
    return result


def simplex_search(root, cfg):
    """Three numerical methods, analytic certificates, and unboundedness witnesses."""
    root = Path(root); out = root/'outputs/optimization/phase3'
    setup, frozen, solver_cfg, model, candidates, empirical = load(root)
    meta = frozen_metadata(root); names = meta['domain_order']; epsilon = float(meta['zero_delta'])
    a, b = log_contrast(meta); pref = np.asarray(frozen['reference']['p_ref'])
    cfg=support_calibration(candidates,names,cfg,out)
    rng = np.random.default_rng(cfg['random_state'])
    samples = []
    for alpha in cfg['dirichlet_concentrations']:
        block = rng.dirichlet(np.full(17, alpha), cfg['dirichlet_samples']//4)
        # A numerical generator can underflow positive Dirichlet entries to 0.
        # P2-A's explicit floor keeps these values safe, without changing B.
        samples.append(block)
    raw_samples = np.concatenate(samples)
    A_samples = epsilon + (1-17*epsilon)*raw_samples
    exact = {role:exact_extreme(b, epsilon, role) for role in ('minM','maxM')}
    for p in exact.values():
        alpha = cfg['targeted_concentration'] * (p-epsilon)/(1-17*epsilon)
        alpha = np.maximum(alpha, cfg['targeted_alpha_floor'])
        block = rng.dirichlet(alpha, cfg['targeted_dirichlet_samples_per_extreme'])
        A_samples = np.r_[A_samples, epsilon+(1-17*epsilon)*block]
    values = a + np.log(A_samples) @ b
    if not np.allclose(values[:100], predict_recipe(A_samples[:100], meta), atol=1e-12):
        raise ValueError('Algebraic log contrast differs from frozen predictor')
    rows = []; recipes = []; diagnostics = []; slsqp_rows = []
    # Finite logit range only limits this numerical check; analytic/SLSQP
    # solutions use the exact simplex constraints, including the floor.
    bound = -np.log(np.finfo(float).eps)
    def mapping(v): return epsilon+(1-17*epsilon)*softmax(np.r_[v, 0.])
    for role, p in exact.items():
        sign = 1 if role=='minM' else -1
        optimum = float(a+np.log(p)@b)
        logging.info('P2-A %s certified M=%.12f; DE check starts', role, optimum)
        def de_objective(v): return sign*float(a+np.log(mapping(v))@b)
        def de_gradient(v):
            q=softmax(np.r_[v,0.]);x=epsilon+(1-17*epsilon)*q;g=b/x
            return (sign*(1-17*epsilon)*q*(g-q@g))[:16]
        # Broad +/-36 uniform logits mostly saturate softmax before searching.
        # A model-independent moderate initial population keeps coordinates
        # active; the full bounds still allow near-floor solutions.
        population=rng.normal(0,2,size=(cfg['simplex_DE_popsize']*16,16))
        de = differential_evolution(de_objective,
            [(-bound,bound)]*16, seed=cfg['random_state']+(role=='maxM'),
            maxiter=cfg['simplex_DE_maxiter'], popsize=cfg['simplex_DE_popsize'],
            tol=cfg['simplex_DE_tolerance'], atol=1e-12, polish=False,init=population)
        polish=minimize(de_objective,de.x,jac=de_gradient,method='L-BFGS-B',bounds=[(-bound,bound)]*16,
            options=dict(ftol=1e-15,gtol=1e-11,maxiter=2000,maxls=50))
        dp = mapping(polish.x if polish.fun<de.fun else de.x); dv = float(a+np.log(dp)@b)
        starts = [pref] + [A_samples[i] for i in rng.choice(len(A_samples), cfg['simplex_slsqp_starts']-1, replace=False)]
        best = None
        for k, start in enumerate(starts):
            r = minimize(lambda x: sign*float(a+np.log(x)@b), start,
                jac=lambda x: sign*b/x, method='SLSQP', bounds=[(epsilon,1)]*17,
                constraints=[dict(type='eq',fun=lambda x:x.sum()-1,jac=lambda x:np.ones(17))],
                options=dict(ftol=cfg['simplex_slsqp_ftol'],maxiter=cfg['simplex_slsqp_maxiter']))
            legal = abs(r.x.sum()-1)<1e-9 and r.x.min()>=epsilon-1e-12
            val = float(a+np.log(r.x)@b)
            slsqp_rows.append(dict(role=role,start=k+1,success=bool(r.success),feasible=legal,M_p=val,
                certificate_gap=sign*(val-optimum),iterations=r.nit))
            if legal and (best is None or sign*val<sign*best[0]): best=(val,r.x)
        if best is None: raise ValueError('No legal SLSQP point')
        idx = int(np.argmin(sign*values)); rv=float(values[idx])
        for method, val, count, numerical_p in [('analytic_global_certificate',optimum,1,p),
            ('differential_evolution_softmax_polished',dv,de.nfev+polish.nfev,dp),
            ('multistart_SLSQP',best[0],len(starts),best[1]),
            ('Dirichlet_random_search',rv,len(A_samples),A_samples[idx])]:
            gap = sign*(val-optimum)
            rows.append(dict(P2_mode='P2_A',role=role,method=method,M_p=val,certificate_M=optimum,
                gap_to_certificate=gap,epsilon=epsilon,evaluations_or_starts=count,
                maximum_share=float(numerical_p.max()), simplex_sum=float(numerical_p.sum()),
                minimum_share=float(numerical_p.min()),
                guarantee='analytic global optimum' if method.startswith('analytic') else
                'one-sided feasible witness, not a bound certificate' if method.startswith('Dirichlet') else 'numerical cross-check',
                status='FINITE_COMPACT_SIMPLEX'))
            tolerance=cfg['random_witness_gap_tolerance'] if method.startswith('Dirichlet') else cfg['optimizer_agreement_absolute']
            if gap < -1e-10 or gap>tolerance: raise ValueError(f'{role} {method} certificate gap {gap} exceeds {tolerance}')
        rec=dict(candidate_id='P2_A_'+role,role=role,M_p=optimum,DeltaM=optimum-frozen['reference']['M_ref'])
        rec.update(zip(names,p)); recipes.append(rec)
        diagnostics.append(dict(candidate_id=rec['candidate_id'],mode='P2_A',role=role,
            **recipe_summary(p,pref,candidates,names,cfg)))
        logging.info('%s DE/SLSQP/random certificate gaps: %.3g %.3g %.3g',role,sign*(dv-optimum),sign*(best[0]-optimum),sign*(rv-optimum))
    csv(out/'P2_global_search.csv',rows);csv(out/'P2_A_extreme_recipes.csv',recipes)
    csv(out/'simplex_multistart_checks.csv',slsqp_rows)
    # Exact zeros are replaced, but positive t is NOT floored in the frozen rule.
    witness=[]
    for role,j in [('minM',int(b.argmax())),('maxM',int(b.argmin()))]:
        for power in (3,6,12,24,48,96,192,300):
            t=10.**(-power); p=pref.copy();p[j]=0;p=p/p.sum()*(1-t);p[j]=t
            val=float(predict_recipe(p,meta)[0])
            witness.append(dict(mode='P2_B',role=role,t=t,log10_t=-power,vanishing_domain=names[j],
                contrast_coefficient=float(b[j]),M_p=val,zero_replacement_applied=False,status='POSITIVE_BOUNDARY_LIMIT',
                **dict(zip(names,p))))
        endpoint=pref.copy();endpoint[j]=0;endpoint/=endpoint.sum()
        witness.append(dict(mode='P2_B',role=role,t=0,log10_t=np.nan,vanishing_domain=names[j],
            contrast_coefficient=float(b[j]),M_p=float(predict_recipe(endpoint,meta)[0]),
            zero_replacement_applied=True,status='EXACT_ZERO_REGULARIZED_ENDPOINT',**dict(zip(names,endpoint))))
        rows.append(dict(P2_mode='P2_B',role=role,method='analytic_boundary_limit',M_p=np.nan,
            limit='-infinity' if role=='minM' else '+infinity',epsilon=0,status='UNBOUNDED_NO_ATTAINED_OPTIMUM',
            guarantee='nonzero mixed-sign log-contrast; positive entries are never floored'))
    vertices=np.eye(17); vm=predict_recipe(vertices,meta);vertex_records=[]
    for j in range(17):
        modelp=vertices[j]*(1-16*epsilon);modelp[modelp==0]=epsilon
        vertex_records.append(dict(candidate_id='P2_B_vertex_'+str(j+1),raw_domain=names[j],M_p=float(vm[j]),
            diagnostic_only=True,raw_max_share=1.,model_max_share=float(modelp.max()),**dict(zip(names,modelp))))
    csv(out/'P2_B_exact_zero_vertices.csv',vertex_records);csv(out/'P2_B_boundary_limit_witness.csv',witness)
    csv(out/'P2_global_search.csv',rows)
    for r in empirical.to_dict('records'):
        p=np.array([r[name] for name in names]);diagnostics.append(dict(candidate_id=r['candidate_id'],mode='P1',role=r['role'],
            **recipe_summary(p,pref,candidates,names,cfg)))
    csv(out/'recipe_concentration_and_support.csv',diagnostics)
    csv(out/'P1_final_recipes.csv',empirical)
    csv(out/'frozen_log_contrast.csv',[dict(domain=n,coefficient=float(v),intercept=a,zero_delta=epsilon) for n,v in zip(names,b)])
    coordinate=[]
    for r in recipes:
        for j,n in enumerate(names):
            lo,hi=float(candidates[n].min()),float(candidates[n].max());v=r[n]
            coordinate.append(dict(candidate_id=r['candidate_id'],domain=n,share=v,empirical_min=lo,empirical_max=hi,
                outside_empirical_range=bool(v<lo-1e-12 or v>hi+1e-12)))
    csv(out/'P2_empirical_coordinate_support.csv',coordinate)
    summary=dict(epsilon=epsilon,intercept=a,coefficient_sum=float(b.sum()),
        P2_A_minM=recipes[0]['M_p'],P2_A_maxM=recipes[1]['M_p'],
        random_samples_unbiased=len(raw_samples),targeted_samples=len(A_samples)-len(raw_samples),
        P2_B_status='UNBOUNDED_NO_ATTAINED_OPTIMUM',P2_B_reason='exact-zero replacement does not floor positive components',
        no_new_floor_or_refitting=True)
    (out/'simplex_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    return pd.DataFrame(recipes)


def evidence_level(r):
    """Classify transfer evidence with 1e-12 relative node-roundoff tolerance."""
    if r['mode']=='P0':return 'reference_term_zero'
    if pd.isna(r.get('N_star')):return 'no_finite_optimizer'
    n=r['N_star']
    return 'extrapolative-lambda' if n>70*(1+1e-12) else 'estimated-transfer-dependent' if n>1+1e-12 else 'real-scale-supported' if n>=.06*(1-1e-12) else 'outside_real_interpolation_interval'


def official_comparison(root, extremes):
    """Solve only P2-A; read frozen P0/P1 and mark literal P2-B unbounded."""
    root=Path(root);out=root/'outputs/optimization/phase3'
    setup,frozen,cfg,model,candidates,empirical=load(root)
    original=pd.read_csv(root/'outputs/optimization/phase2/official_budget_results.csv',float_precision='round_trip')
    results=[];cost_checks=[]
    for domain in ('main','extended'):
        for family in setup['official']['quality_costs']:
            for lctx in setup['official_contexts']:
                for budget in setup['official']['budgets_FLOPs']:
                    prob=ResourceProblem(setup,model,cfg,extremes,budget,family,lctx,domain,'P1')
                    r,_=prob.solve(independent=True,seed=cfg['random_state']+len(results))
                    r['mode']='P2_A';r['scenario_extrapolation']=True
                    r['unsupported_mixture_extrapolation']=True
                    results.append(r)
                    if r['feasible']:
                        for mode,ext in [('P0',empirical),('P1',empirical),('P2_A',extremes)]:
                            check=ResourceProblem(setup,model,cfg,ext,budget,family,lctx,domain,'P0' if mode=='P0' else 'P1')
                            q=r['Q_star'];n=r['N_star'];d=r['D_star']
                            costs=np.array([6*n*d*1e18,d*float(check.h(q))*1e18,check.eta*n*d*lctx*1e18])
                            gap=float(np.max(np.abs(costs-np.array([r['C_train'],r['C_quality'],r['C_attn']]))))
                            cost_checks.append(dict(domain=domain,cost_function=family,L_ctx=lctx,budget=budget,
                                mode=mode,N_B=n,D_B=d,Q=q,maximum_absolute_cost_gap=gap,passed=gap==0))
                    b=dict(budget=budget,cost_function=family,L_ctx=lctx,domain=domain,mode='P2_B',
                        Q0=cfg['Q0_main'],status='UNBOUNDED_NO_ATTAINED_OPTIMUM' if prob.feasible else 'INFEASIBLE',
                        feasible=prob.feasible,scenario_extrapolation=True,L_star=np.nan,
                        loss_infimum='-infinity' if prob.feasible else 'not feasible',
                        unsupported_mixture_extrapolation=True)
                    results.append(b)
                    logging.info('Official %s %s L=%s C=%.0e P2-A %s Loss=%s',domain,family,lctx,budget,r['status'],r.get('L_star'))
                    csv(out/'P2_official_budget_results.csv',results)
    combined=pd.concat([original,pd.DataFrame(results)],ignore_index=True)
    baseline=original[original['mode']=='P0'][['budget','cost_function','L_ctx','domain','L_star']].rename(columns={'L_star':'P0_Loss'})
    combined=combined.merge(baseline,on=['budget','cost_function','L_ctx','domain'],validate='many_to_one')
    combined['Loss_improvement_vs_P0']=combined.P0_Loss-combined.L_star
    combined['mixing_contribution']=combined.lambda_value*combined.DeltaM
    combined['unused_budget_fraction']=combined.unused_budget/combined.budget
    combined['mixture_evidence_level']=combined.apply(evidence_level,axis=1)
    combined['result_type']=np.where((combined.domain=='main')&combined['mode'].isin(['P0','P1']),
        'support-domain model-conditional optimization','scenario extrapolation')
    csv(out/'P0_P1_P2_official_comparison.csv',combined);csv(out/'fixed_resource_cost_invariance.csv',cost_checks)
    return combined

# ============================================================
# Source: src/optimization/final_reporting.py
# Original lines: 1-354
# Purpose: 冻结P0/P1复用、证据分层与问题三最终表/报告
# ============================================================
"""Final Question 3 synthesis from frozen Phase 2 and new P2 diagnostics."""
from __future__ import annotations
import json
import hashlib
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .resource_solver import csv, load


def read(base, name):
    """Read saved full-precision output without changing it."""
    return pd.read_csv(base/name,float_precision='round_trip')


def mdtable(frame):
    """Small Markdown table with honest missing/infeasible values."""
    def cell(x):
        if pd.isna(x):return '—'
        if isinstance(x,(float,np.floating)):return f'{x:.6g}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def structural_summary(base, out, cfg):
    """Classify existing hinge breaks; never detect or refit a breakpoint."""
    hard=read(base,'typeI_active_set_transitions.csv');soft=read(base,'typeII_soft_transitions.csv')
    classes={'s_train':'train_attention_cost','s_attn':'train_attention_cost',
             's_quality':'quality_cost','ln_N':'N','ln_D':'D','Q':'Q'}
    rows=[];timeline=[]
    for r in hard.to_dict('records'):
        timeline.append(dict(domain=r['domain'],cost_function=r['cost_function'],L_ctx=r['L_ctx'],
            event=r['constraint']+':'+r['transition'],budget=r['budget_critical'],interval_low=r['bracket_low'],
            interval_high=r['bracket_high'],evidence_type='Type I active-set change',
            classification='constraint event',source_file='typeI_active_set_transitions.csv',
            interval_meaning='numerical root bracket; not statistical uncertainty'))
    for keys,g in soft.groupby(['domain','cost_function','L_ctx']):
        h=hard[(hard.domain==keys[0])&(hard.cost_function==keys[1])&(hard.L_ctx==keys[2])]
        significant=g[g.significant==True]
        for r in g.to_dict('records'):
            if pd.isna(r['break_budget']):continue
            distances=np.abs(np.log10(h.budget_critical.to_numpy()/r['break_budget']))
            near=int(np.argmin(distances));a=bool(distances[near]<=cfg['primary_transition_log10_distance'])
            neighbors=significant[np.abs(significant.break_log10C-r['break_log10C'])<=cfg['nearby_break_log10_distance']]
            categories=set(classes[m] for m in neighbors.metric)
            b=len(categories)>=3
            primary=bool(r['significant'] and (a or b))
            label='primary structural transition' if primary else 'secondary curve break'
            r.update(resource_category=classes[r['metric']],nearest_TypeI=h.constraint.iloc[near],
                nearest_TypeI_log10_distance=float(distances[near]),criterion_A=a,criterion_B=b,
                nearby_independent_categories=len(categories),classification=label)
            rows.append(r)
            if r['significant']:
                timeline.append(dict(domain=keys[0],cost_function=keys[1],L_ctx=keys[2],event=r['metric']+' curve break',
                    budget=r['break_budget'],interval_low=r['break_interval_low'],interval_high=r['break_interval_high'],
                    evidence_type='Type II saved hinge-fit diagnostic',classification=label,
                    delta_BIC=r['delta_BIC'],source_file='typeII_soft_transitions.csv',
                    interval_meaning='saved block-residual perturbation interval; not physical phase-transition CI'))
    classified=pd.DataFrame(rows);ordered=pd.DataFrame(timeline).sort_values(['domain','cost_function','L_ctx','budget','event'])
    csv(out/'typeII_final_classification.csv',classified);csv(out/'structural_transition_timeline.csv',ordered)
    peaks=read(base,'quality_cost_share_peaks.csv');thresholds=read(base,'quality_threshold_budgets.csv')
    qs=thresholds[thresholds.threshold.astype(str)=='1.0'][['domain','cost_function','L_ctx','budget_threshold']].rename(columns={'budget_threshold':'C_Qsat'})
    peaks=peaks.merge(qs,on=['domain','cost_function','L_ctx'],validate='one_to_one')
    peaks['relative_gap']=np.abs(np.log10(peaks.C_peak_Q/peaks.C_Qsat))
    csv(out/'quality_share_peak_vs_saturation.csv',peaks)
    transitions=[]
    saturation=read(base,'domain_saturation_budgets.csv')
    for keys,g in hard.groupby(['domain','cost_function','L_ctx']):
        r=dict(domain=keys[0],cost_function=keys[1],L_ctx=keys[2])
        for c,name in [('active_Q_lower','C_Q_leave'),('active_Q_upper','C_Q1'),('active_D_lower','C_D_leave'),('active_D_upper','C_Dmax')]:
            v=g[g.constraint==c];r[name]=float(v.budget_critical.iloc[0]) if len(v) else np.nan
        sat=saturation[(saturation.domain==keys[0])&(saturation.cost_function==keys[1])&(saturation.L_ctx==keys[2])]
        r['C_saturation']=float(sat.exact_saturation_budget.iloc[0])
        p=peaks[(peaks.domain==keys[0])&(peaks.cost_function==keys[1])&(peaks.L_ctx==keys[2])].iloc[0]
        r.update(C_peak_Q=p.C_peak_Q,max_s_Q=p.peak_s_quality,peak_Qsat_log10_gap=p.relative_gap)
        events=[(r[k],k) for k in ['C_Q_leave','C_Q1','C_D_leave','C_Dmax','C_saturation'] if np.isfinite(r[k])]
        r['investment_event_order']=' → '.join(name for _,name in sorted(events))
        transitions.append(r)
    thresholds=pd.DataFrame(transitions);csv(out/'cost_function_transition_summary.csv',thresholds)
    return classified,ordered,peaks,thresholds


def allocation_summary(base,out,combined,setup):
    """Context changes, evidence-aware tiers, and budget-stratum summaries."""
    main=combined[(combined.domain=='main')&(combined['mode']=='P0')].copy()
    ctx=[]
    for r in main.to_dict('records'):
        ref=main[(main.budget==r['budget'])&(main.cost_function==r['cost_function'])&(main.L_ctx==8192)].iloc[0]
        r.update(context_category='short' if r['L_ctx']<=8192 else 'near_critical' if r['L_ctx']==32768 else 'ultralong',
                 ratio_attention_training=setup['official']['eta']*r['L_ctx']/6,
                 L_critical=6/setup['official']['eta'],
                 N_ratio_vs_8192=r['N_star']/ref.N_star if r['feasible'] else np.nan,
                 D_ratio_vs_8192=r['D_star']/ref.D_star if r['feasible'] else np.nan,
                 Loss_change_vs_8192=r['L_star']-ref.L_star if r['feasible'] else np.nan)
        ctx.append(r)
    context=pd.DataFrame(ctx);csv(out/'context_final_summary.csv',context)
    high=combined[(combined.domain=='main')&(combined.budget==1e24)&combined['mode'].isin(['P0','P1'])]
    csv(out/'high_budget_main_saturation.csv',high)
    tiers=[]
    for r in combined.to_dict('records'):
        tier='Conservative' if r['domain']=='main' and r['mode']=='P0' else 'Empirical-enhanced' if r['domain']=='main' and r['mode']=='P1' else 'Exploratory'
        credible=r['mixture_evidence_level']=='real-scale-supported' and r['mode']=='P1' and r['domain']=='main'
        interpretation='reference allocation; frozen-model conditional' if tier=='Conservative' else \
            'empirical candidate enhancement with real-node mixture transfer' if credible else \
            'conditional estimated-scale mixture enhancement' if tier=='Empirical-enhanced' else \
            'unbounded sensitivity; no finite allocation to recommend' if r['mode']=='P2_B' else 'scenario extrapolation only'
        tiers.append(dict(tier=tier,**{k:r.get(k) for k in ['budget','cost_function','L_ctx','domain','mode','status','N_star','D_star','Q_star','L_star',
            'selected_candidate','mixture_evidence_level','Loss_improvement_vs_P0','unused_budget_fraction']},
            engineering_eligible=bool(r['status']=='SOLVED' and tier!='Exploratory'),
            mixture_gain_real_node_supported=credible,interpretation=interpretation,
            quality_warning='Q=1 exceeds observed A1 sample quality range; quality module has no direct joint N-D-Q validation' if r.get('Q_star',0)>.9511217995313428 else 'frozen Q0 proxy/baseline and modular-model assumptions apply'))
    csv(out/'recommendation_tiers.csv',tiers)
    budgets=[]
    for (domain,budget,mode),g in combined.groupby(['domain','budget','mode']):
        valid=g[g.status=='SOLVED'];budgets.append(dict(domain=domain,budget=budget,mode=mode,
            scenarios=len(g),solved=len(valid),Loss_min=valid.L_star.min(),Loss_max=valid.L_star.max(),
            improvement_min=valid.Loss_improvement_vs_P0.min(),improvement_max=valid.Loss_improvement_vs_P0.max(),
            N_min=valid.N_star.min(),N_max=valid.N_star.max(),D_min=valid.D_star.min(),D_max=valid.D_star.max(),
            Q_min=valid.Q_star.min(),Q_max=valid.Q_star.max()))
    csv(out/'budget_level_comparison_summary.csv',budgets)
    return main,context,high,pd.DataFrame(budgets)


def robustness_summary(base,out,combined,peaks):
    """Explicit scopes prevent baseline conclusions being generalized to P2."""
    q0=read(base,'Q0_sensitivity_comparison.csv')
    feasible=q0[q0.status=='SOLVED']
    q0bound=float(feasible.Loss_change.abs().max())
    records=[
        ('A','低预算质量保持Q0','robust','P0/P1主域，三个成本、全部可行官方低预算上下文及已有Q0敏感性',
         'P2理论外推允许不同投资路径；不将P0低预算规律推广到P2-B无界情景'),
        ('B','中预算Q=1','moderately_sensitive','P0/P1，C=1e22，三成本五上下文；Q0已有敏感性',
         'Q=1本身超出A1实际质量范围，依赖冻结质量模块；P2_A结果单列核对'),
        ('C','高预算主域饱和','robust','P0/P1，C=1e24，三成本五上下文及Q0敏感性',
         '依赖主域上界；扩展域结果不同，P2极端外推也可改变最优N'),
        ('D','质量成本份额先升后降且峰值靠近Q饱和','robust','冻结P0主/扩展域30条路径；主域15个峰值',
         '未重做Q0/P1/P2完整路径，不声称在所有配比方案下已验证'),
        ('E','长上下文压缩N/D并提高Loss','moderately_sensitive','P0/P1未饱和的可行预算情景',
         '低预算D在下界可不变；高预算主域饱和N/D/Loss相同，消耗与剩余预算改变'),
        ('F','经验配比优化改善Loss','highly_sensitive','P1经验可行集内条件性模型收益；低预算真实节点更可信',
         '大尺度lambda来自估计节点且变号，P2_B无界揭示配比模型外推风险'),
        ('G','具体转移预算保持稳定','highly_sensitive','数值根括区间精度高；不是模型参数不确定性区间',
         '三类成本和上下文显著移动阈值；早期投资顺序亦可改变'),
    ]
    sources=['quality cost function','Q0','L_ctx','P0/P1/P2','main/extended domain','large-scale lambda evidence']
    evidence=['三成本全部冻结对照','三候选基线敏感性，最大绝对Loss变化='+f'{q0bound:.6g}',
              '五场景与解析成本比；不拟合上下文性能收益','832候选P1/P2-A全局证书/P2-B无界',
              '主域与扩展域官方配置独立标记','>1B为估计迁移；>70B为lambda外推']
    levels={
        'A':['robust','robust','robust','highly_sensitive','robust','moderately_sensitive'],
        'B':['robust','robust','robust','highly_sensitive','moderately_sensitive','moderately_sensitive'],
        'C':['robust','robust','robust','highly_sensitive','highly_sensitive','moderately_sensitive'],
        'D':['robust','moderately_sensitive','robust','highly_sensitive','robust','moderately_sensitive'],
        'E':['robust','moderately_sensitive','moderately_sensitive','highly_sensitive','highly_sensitive','moderately_sensitive'],
        'F':['moderately_sensitive','moderately_sensitive','moderately_sensitive','highly_sensitive','highly_sensitive','highly_sensitive'],
        'G':['highly_sensitive','moderately_sensitive','highly_sensitive','highly_sensitive','highly_sensitive','highly_sensitive']}
    details=[];summaries=[]
    for key,claim,label,scope,limitation in records:
        summaries.append(dict(conclusion_id=key,conclusion=claim,robustness=label,verified_scope=scope,limitation=limitation))
        for j,source in enumerate(sources):
            details.append(dict(conclusion_id=key,conclusion=claim,sensitivity_source=source,
                robustness=levels[key][j],overall_robustness=label,verified_scope=scope,
                evidence=evidence[j],limitation=limitation,
                scope_note='robust means within explicitly verified scope; an untested full path is not evidence of invariance'))
    csv(out/'robustness_summary.csv',details);csv(out/'robustness_core_conclusions.csv',summaries)
    return pd.DataFrame(summaries)


def final_figures(root,out,combined,hard,classified,recipes,names):
    """Reuse frozen figures; add only three figures answering new questions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=root/'outputs/figures/question3_phase3';figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    manifest=[]
    for name,purpose in [('01_N_star_vs_budget.png','N versus budget'),('02_D_star_vs_budget.png','D versus budget'),
        ('03_Q_star_vs_budget.png','Q versus budget'),('05_cost_shares.png','three cost shares'),
        ('08_five_context_N_star.png','context N'),('09_five_context_D_star.png','context D'),
        ('10_five_context_L_star.png','context Loss')]:
        path=root/'outputs/figures/question3_phase2'/name
        if not path.is_file():raise FileNotFoundError(path)
        manifest.append(dict(path=path.relative_to(root).as_posix(),purpose=purpose,reused=True))
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,(budget,label) in zip(axes,[(1e19,'Low'),(1e22,'Middle'),(1e24,'High')]):
        for mode,color in [('P0','#555555'),('P1','#2878a8'),('P2_A','#c07328')]:
            g=combined[(combined.domain=='main')&(combined.budget==budget)&(combined.cost_function=='exponential')&
                (combined['mode']==mode)&(combined.status=='SOLVED')].sort_values('L_ctx')
            ax.plot(np.log2(g.L_ctx),g.L_star,'o-',label=mode,color=color)
        ax.set_xticks(np.log2([2048,4096,8192,32768,131072]),['2k','4k','8k','32k','128k'])
        ax.set_title(f'{label} budget: {budget:.0e} FLOPs');ax.set_xlabel('Context length');ax.grid(alpha=.2)
    axes[0].set_ylabel('Frozen-model optimized Loss');axes[-1].legend()
    fig.suptitle('Main-domain allocations; exponential cost; P2-A is theoretical extrapolation')
    fig.tight_layout();path=figdir/'01_P0_P1_P2_comparison.png';fig.savefig(path,dpi=180);plt.close(fig)
    manifest.append(dict(path=path.relative_to(root).as_posix(),purpose='official P0/P1/P2-A comparison; P2-B has no finite optimum',reused=False))
    fig,axes=plt.subplots(1,2,figsize=(12,6),sharex=True)
    labels=[n.removeprefix('train_the_pile_') for n in names]
    for ax,r in zip(axes,recipes.to_dict('records')):
        p=np.array([r[n] for n in names]);order=np.argsort(p)
        ax.barh(np.array(labels)[order],p[order],color='#2878a8')
        ax.set_title(f"P2-A {r['role']}: M={r['M_p']:.6f}");ax.set_xlabel('Domain fraction')
        ax.axvline(.8,color='#c07328',ls=':',label='concentration diagnostic > 0.8');ax.legend(fontsize=8)
    fig.suptitle('Frozen epsilon-constrained simplex extrema; neither is an engineering recipe')
    fig.tight_layout();path=figdir/'02_simplex_extreme_recipes.png';fig.savefig(path,dpi=180);plt.close(fig)
    manifest.append(dict(path=path.relative_to(root).as_posix(),purpose='full 17-domain theoretical simplex extrema',reused=False))
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    core={'active_Q_lower':('Q leaves floor','#2878a8','o'),'active_Q_upper':('Q reaches 1','#2c9473','s'),
          'active_D_upper':('D reaches cap','#c07328','^'),'budget_binding':('Main saturation','#735ca3','D')}
    for ax,family in zip(axes,['exponential','power','logarithmic']):
        for i,lctx in enumerate([2048,4096,8192,32768,131072]):
            h=hard[(hard.domain=='main')&(hard.cost_function==family)&(hard.L_ctx==lctx)]
            for c,(label,color,mark) in core.items():
                g=h[h.constraint==c]
                if len(g):ax.scatter(np.log10(g.budget_critical.iloc[0]),i,c=color,marker=mark,label=label if i==0 else None,s=40)
            g=classified[(classified.domain=='main')&(classified.cost_function==family)&(classified.L_ctx==lctx)&
                (classified.classification=='primary structural transition')&(classified.significant==True)&(classified.metric=='Q')]
            if len(g):ax.scatter(g.break_log10C,[i+.15]*len(g),marker='x',color='#333333',label='Primary Q curve break' if i==0 else None)
        ax.set_title(family);ax.set_xlabel('log10 allocated FLOPs');ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(5),['2048','4096','8192','32768','131072']);axes[0].set_ylabel('Context length')
    handles,labels=axes[-1].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=9,loc='lower center',ncol=5)
    fig.suptitle('Core constraint events and selected saved curve breaks; breaks are not independent physical transitions')
    fig.tight_layout(rect=(0,.1,1,1));path=figdir/'03_final_structural_timeline.png';fig.savefig(path,dpi=180);plt.close(fig)
    manifest.append(dict(path=path.relative_to(root).as_posix(),purpose='selected interpretable transition timeline',reused=False))
    csv(out/'final_figure_manifest.csv',manifest)


def final_report(root,out,combined,main,context,high,classified,peaks,thresholds,robust,diagnostics,summary):
    """Nine requested sections and exactly five core tables; appendices linked."""
    recipes=read(out,'P2_A_extreme_recipes.csv');p1=read(out,'P1_final_recipes.csv')
    search=read(out,'P2_global_search.csv');witness=read(out,'P2_B_boundary_limit_witness.csv')
    representative=main[main.L_ctx.isin([8192,32768])][['budget','cost_function','L_ctx','status','N_star','D_star','Q_star','L_star']]
    t2=thresholds[(thresholds.domain=='main')&(thresholds.L_ctx==2048)][['cost_function','C_Q_leave','C_Q1','C_D_leave','C_Dmax','C_saturation','C_peak_Q','max_s_Q']]
    t3=context[context.cost_function=='exponential'][['budget','L_ctx','ratio_attention_training','status','N_star','D_star','L_star','s_train','s_quality','s_attn']]
    t4=combined[(combined.domain=='main')&(combined.L_ctx==8192)&combined['mode'].isin(['P0','P1','P2_A'])].pivot(
        index=['budget','cost_function'],columns='mode',values='L_star').reset_index()
    t4['P2_B']='无有限最优解';t4['evidence']='P2_A为配比外推；P1按实际最优N判定lambda证据'
    t5=robust[['conclusion_id','conclusion','robustness','verified_scope']]
    mains=classified[classified.domain=='main'];mp=peaks[peaks.domain=='main'];mh=high[high['mode']=='P0']
    paragraphs=['# 问题三最终报告：算力约束下的资源配置\n',
      '问题一、问题二和问题三PHASE 1/2全部冻结。本报告读取已保存P0/P1、阈值、断点及敏感性结果；仅新增P2理论分析和收尾汇总。这里的配置是冻结模型下计算出的优化解，不是新增训练实验的观测值。',
      '## 3.1 优化模型与成本构成',
      r'目标：$L_{gen}=E+A N_B^{-\alpha}+B D_B^{-\beta}+C_{Q,loss}(N_B/N_{0,B})^{-u_Q}(-\ln Q)+\lambda_p(N_B)[M(p)-M(p_{ref})]$。系数和lambda插值全部沿用问题二。',
      r'约束：$C_{train}+C_{quality}+C_{attn}\le C$；$C_{train}=6N_BD_B10^{18}$，$C_{attn}=\eta L_{ctx}N_BD_B10^{18}$，$C_{quality}=D_B10^9[g(Q)-g(Q_0)]_+$。p不进入成本。三个官方g平行保留，不评选哪个“正确”。',
      'Q0='+f"{summary['Q0_main']:.16g}"+'；主域N_B∈[0.070542,11.965825]，D_B∈[10,299.893]，Q∈[Q0,1]。它是证据支持的资源箱体，不能等同于箱内每个N-D-Q组合都做过实测验证。尤其Q=1高于A1实际样本质量上限，质量模块缺少完整联合N-D-Q直接验证。',
      '## 3.2 解析降维与求解方法',
      r'由于$\partial L/\partial D_B<0$，$D_B^*=\min[D_{max},C/((6+\eta L_{ctx})N_B10^{18}+10^9\Delta g)]$。不强制花完预算。P0/P1直接采用PHASE 2结果；P2-A用相同二维资源求解器，并做独立DE复核，成本式未改变。',
      r'冻结一阶ILR模型可精确展开为$M(p)=a+\sum_j b_j\ln p_j$，$\sum b_j=0$。P2-A下界epsilon='+f"{summary['epsilon']:.17g}"+'，是冻结零替换尺度，未新增更大下界。minM使正系数坐标取下界，剩余负系数按−b水位分配；maxM反之。这给出解析全局证书。',
      f"三类数值核验：DE在softmax/log-ratio空间搜索后梯度细化；32起点SLSQP；{summary['random_samples_unbiased']:,}个广泛Dirichlet点加{summary['targeted_samples']:,}个靠近证书的定向Dirichlet点。随机点只给出可行值：采样最小值≥真实最小值、采样最大值≤真实最大值，不能把采样极值称为严格全局边界。DE/SLSQP与证书的误差详见P2_global_search.csv。",
      'P2-B的重要结论：冻结规则只替换精确0，不截断任意小的正数。沿p_j=t→0+、其余分量为固定正比例，M=b_j ln t+O(1)。有正负系数，所以inf M=−∞、sup M=+∞，均不取到。精确t=0经零替换反而返回有限值，边界处不连续。因而不能合法报告P2-B有限p_minM/p_maxM，也不能靠新floor修补。所有可行官方资源场景lambda≠0时Loss下确界为−∞；lambda=0时配比项为0，所有配比等价。此为模型外推失效证据，不是可实现性能提升。',
      '## 3.3 低中高预算最优配置',
      '**表1：P0主域，代表上下文8192/32768；N、D单位均为B，Loss无量纲。**',mdtable(representative),
      '低预算1e19：全部可行情景Q=Q0，P0的D=10B；131072上下文最低成本约2.27247e19，因此三类成本均不可行。中预算1e22：P0/P1全部Q=1。高预算1e24：P0/P1主域N/D/Q都达到上界。',
      f"主域P0高预算未用预算比例为{mh.unused_budget_fraction.min():.4%}–{mh.unused_budget_fraction.max():.4%}。其含义是已验证资源支持域饱和，现有模型无法可靠区分更大空间内配置；不能说真实训练最优只需要当前用量。扩展域单独标为scenario extrapolation。",
      '## 3.4 结构性转移',
      '**表2：Lctx=2048，三个成本函数的代表转移阈值；FLOPs。**',mdtable(t2),
      '完整时间轴含所有既有Type I活跃集事件及Type II断点，没有重新检测。活跃集根括区间是数值定位误差；Type II沿用分块残差扰动区间，不是物理相变或参数不确定性的置信区间。',
      f"主域90条曲线中{int(mains.significant.sum())}条满足ΔBIC≥10；按最终规则，{int(((mains.classification=='primary structural transition')&mains.significant).sum())}条是primary curve diagnostics。它们仍不是同样数量的独立结构事件。A规则为距任一Type I事件≤0.5 log10预算；B规则为半径0.5 log10邻域内至少三种独立资源/成本类别。训练和注意力份额严格成比例，合并为同一类，不能重复计数。正文仅保留代表断点与硬约束事件。",
      f"主域15个质量成本份额路径均先升后降。峰值相对Q=1阈值的|log10(Cpeak/CQsat)|最大为{mp.relative_gap.max():.6g}，系统性贴近质量饱和。每个成本×上下文的Cpeak、峰值份额和两种数值阈值材料均可追溯。这里Qsat使用PHASE 2已有target Q=1阈值，其容差与Type I active_Q_upper容差不同。",
      '## 3.5 上下文长度临界分析',
      r'$C_{attn}/C_{train}=\eta L_{ctx}/6$，$L_{crit}=6/\eta=30000$。短上下文2048/4096/8192、临界附近32768、超长131072。此临界只表示两项成本相等，不表示性能最优上下文，也没有建模增加上下文可能带来的性能收益。',
      '**表3：指数成本P0主域的五上下文对照；成本份额分母是分配预算。**',mdtable(t3),
      '跨8192→32768时，注意力/训练成本比从0.273067变为1.09227。未饱和中预算N和D下降、Loss上升；低预算D卡下界不变而N下降。高预算主域尺寸已经饱和，不再压缩N/D或改变Loss，但注意力消耗增加、未用预算下降。',
      '## 3.6 质量成本函数敏感性',
      '三种成本主要改变质量投资时机、峰值与阈值，也会改变局部早期阶段顺序。Lctx=2048时，指数/幂成本先退出D下界再提高质量；对数成本先提高质量并达到Q=1后才退出D下界。因此不能写“各成本完全保持相同阶段顺序”。相同官方低预算P0结果一致，中预算配置和Loss略有差异，高预算主域均饱和。完整五上下文顺序见cost_function_transition_summary.csv。',
      '## 3.7 领域配比优化：P0/P1/P2',
      '**表4：Lctx=8192，主域各官方预算下优化Loss；P2-A仅为理论配比外推。**',mdtable(t4),
    ]
    for r in diagnostics.to_dict('records'):
        paragraphs.append(f"{r['mode']} {r['role']}：max share={r['max_share']:.4%}，H={r['entropy']:.6g}，Hnorm={r['normalized_entropy']:.6g}，Neff={r['effective_domain_count']:.6g}，HHI={r['HHI']:.6g}；距参考L1/L2/JS={r['L1_to_ref']:.6g}/{r['L2_to_ref']:.6g}/{r['JS_distance_to_ref']:.6g}。最近经验L1/JS={r['minimum_empirical_L1']:.6g}/{r['minimum_empirical_JS_distance']:.6g}；超经验坐标范围领域数={r['out_of_empirical_range_domains']}；unsupported_mixture_extrapolation={r['unsupported_mixture_extrapolation']}。")
    paragraphs.extend([
      '两个P2-A端点都没有触发maxshare>0.8或Neff<2的集中阈值，更未触发maxshare>0.95的近退化阈值；不能预设纯simplex必然单域退化。但最优配方均偏离经验配方集，仍属unsupported mixture extrapolation。此标记依据坐标越界或最近距离超过832候选leave-one-out最近邻距离P95；是数据驱动的稀疏性诊断，不是统计置信区域。',
      f"P1的minM/maxM来自832个冻结实测配方。它们的完整17域、来源成员、模型值及集中度见附录；maxM配方本身最大单域{float(diagnostics[(diagnostics['mode']=='P1')&(diagnostics.role=='maxM')].max_share.iloc[0]):.2%}，Neff<2，是高度集中的经验配方，经验可行不代表分散。lambda>0选择minM；lambda<0选择maxM；条件性变号来自估计的大尺度迁移节点，不能理解为新的大模型训练观测。",
      '推荐层级：Conservative=主域P0；Empirical-enhanced=主域P1，其中60M–1B真实节点区间内增益证据较强，>1B估计节点依赖，>70B为lambda外推；Exploratory=扩展域和/或P2。P2不进入工程推荐。建议以P0作为主结论，P1作为按证据层级注明的增强方案。分类按各自最优N判断，不按预算自动划分。',
      '固定同一N/D/Q/Lctx的成本不变量核验已输出，P2收益仅来自冻结配比项及由此引起的资源重新分配，成本公式完全一致。',
      '## 3.8 鲁棒性与证据边界',
      '**表5：核心结论的鲁棒性及已核验范围。**',mdtable(t5),
      '六种敏感性来源在robustness_summary.csv逐项列出：成本函数、Q0、上下文、P0/P1/P2、主/扩展域、lambda证据。robust均限制在明示的已验证范围；Q0只沿用已有官方场景敏感性，未重新检测转移；未计算P1/P2完整预算路径，不能声称其质量份额峰值规律已经验证。模型箱体内仍有插值/模块组合假设，不能称全部结果“直接实测支持”。',
      '## 3.9 问题三结论',
      '1. 解析消去D和按lambda符号选配方，使资源优化降为二维；P0/P1历史结果保持冻结，P2-A另有全局配方证书。\n2. 官方低预算优先保留质量基线，中预算质量达到模型上界，高预算主域饱和；对应证据和适用范围必须写清。\n3. 质量投资份额先升后降，15个主域峰值均靠近质量饱和；成本形状改变时机，部分改变早期阶段顺序。\n4. 30000为成本临界，上下文增加使未饱和配置承受更高注意力负担；不是性能最优点。\n5. 主推荐P0，P1作为证据分层的经验增强；P2-A虽未单域退化仍远离实测配方，P2-B无界，均不得工程部署。\n6. 精确转移阈值、配比迁移收益和大资源外推高度依赖假设；1e24未用预算反映证据域饱和。',
      '## 附录与复现',
      '新脚本：`python -X utf8 -B scripts/19_finalize_compute_optimization.py --stage simplex|official|report`；完整运行可用`--stage all`。脚本使用pathlib自动定位项目根。只有新phase3输出会更新；现有阶段文件不修改。',
    ])
    for name in ['P2_global_search.csv','P2_A_extreme_recipes.csv','P2_B_boundary_limit_witness.csv','P2_B_exact_zero_vertices.csv',
        'P1_final_recipes.csv','recipe_concentration_and_support.csv','empirical_distance_calibration.csv','P2_empirical_coordinate_support.csv',
        'P0_P1_P2_official_comparison.csv','recommendation_tiers.csv','structural_transition_timeline.csv','typeII_final_classification.csv',
        'quality_share_peak_vs_saturation.csv','cost_function_transition_summary.csv','context_final_summary.csv','robustness_summary.csv',
        'high_budget_main_saturation.csv','fixed_resource_cost_invariance.csv','final_figure_manifest.csv',
        'input_manifest.csv','implementation_manifest.csv','independent_verification.csv','verification_summary.json']:
        paragraphs.append(f'- [{name}](../optimization/phase3/{name})')
    paragraphs.append('附录图：复用PHASE 2的N/D/Q路径、成本份额及上下文图，新图仅补P0/P1/P2-A比较、17域simplex配方、核心事件时间轴。')
    for path in ['../figures/question3_phase2/01_N_star_vs_budget.png','../figures/question3_phase2/02_D_star_vs_budget.png',
        '../figures/question3_phase2/03_Q_star_vs_budget.png','../figures/question3_phase2/05_cost_shares.png',
        '../figures/question3_phase2/10_five_context_L_star.png','../figures/question3_phase3/03_final_structural_timeline.png',
        '../figures/question3_phase3/01_P0_P1_P2_comparison.png','../figures/question3_phase3/02_simplex_extreme_recipes.png']:
        paragraphs.append(f'![{Path(path).stem}]({path})')
    paragraphs.append('独立公式/输出及冻结文件SHA256核验结果以附录verification_summary.json为准。运行`python -X utf8 -B scripts/19_verify_final_optimization.py`可独立复核；不需要重跑PHASE 2。')
    paragraphs.append('问题三至此结束。没有开始问题四，没有改动冻结附件、预处理、质量权重、Q/K、ILR/ElasticNet、Scaling Law、lambda或PHASE 2优化结果。')
    report=root/'outputs/report/question3_final.md';report.write_text('\n\n'.join(paragraphs)+'\n',encoding='utf8')


def finalize(root,cfg):
    """Build final tables, three new figures, and the integrated report."""
    root=Path(root);out=root/'outputs/optimization/phase3';base=root/'outputs/optimization/phase2'
    source_paths=[root/'outputs/optimization/phase1/optimization_setup.json',
        root/'outputs/optimization/phase1/empirical_recipe_candidates.csv',
        root/'outputs/mixture/final_model_metadata.json',root/'outputs/mixture/stage2/final_model_policy.json',
        root/'outputs/scaling/phase4/generalized_scaling_final.json',root/'configs/scaling_phase4.yaml',
        root/'configs/optimization_phase2.yaml',root/'configs/optimization_phase3.yaml']
    source_paths += [base/n for n in ['official_budget_results.csv','typeI_active_set_transitions.csv',
        'typeII_soft_transitions.csv','quality_cost_share_peaks.csv','quality_threshold_budgets.csv',
        'domain_saturation_budgets.csv','Q0_sensitivity_comparison.csv']]
    def manifest(paths):
        rows=[]
        for p in paths:
            with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            rows.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=digest))
        return rows
    csv(out/'input_manifest.csv',manifest(source_paths))
    csv(out/'implementation_manifest.csv',manifest([root/'configs/optimization_phase3.yaml',
        root/'src/optimization/simplex_sensitivity.py',root/'src/optimization/final_reporting.py',
        root/'scripts/19_finalize_compute_optimization.py',root/'scripts/19_verify_final_optimization.py']))
    setup,frozen,solver_cfg,model,candidates,empirical=load(root)
    from .simplex_sensitivity import recipe_summary,support_calibration,evidence_level
    names=frozen['reference']['domain_order'];pref=np.asarray(frozen['reference']['p_ref'])
    cfg=support_calibration(candidates,names,cfg,out)
    diagnostics=[];recipes=read(out,'P2_A_extreme_recipes.csv')
    for mode,frame in [('P2_A',recipes),('P1',empirical)]:
        for r in frame.to_dict('records'):
            diagnostics.append(dict(candidate_id=r['candidate_id'],mode=mode,role=r['role'],
                **recipe_summary(np.array([r[n] for n in names]),pref,candidates,names,cfg)))
    diagnostics=pd.DataFrame(diagnostics);csv(out/'recipe_concentration_and_support.csv',diagnostics)
    combined=read(out,'P0_P1_P2_official_comparison.csv')
    combined['mixture_evidence_level']=combined.apply(evidence_level,axis=1)
    csv(out/'P0_P1_P2_official_comparison.csv',combined)
    classified,timeline,peaks,thresholds=structural_summary(base,out,cfg)
    main,context,high,budgets=allocation_summary(base,out,combined,setup)
    robust=robustness_summary(base,out,combined,peaks)
    final_figures(root,out,combined,read(base,'typeI_active_set_transitions.csv'),classified,recipes,names)
    summary=json.loads((out/'simplex_summary.json').read_text(encoding='utf8'))
    summary.update(Q0_main=solver_cfg['Q0_main'],official_comparison_rows=len(combined),
        simplex_numerical_extremes_not_concentrated=bool(~diagnostics[diagnostics['mode']=='P2_A'].highly_concentrated.any()),
        P2_A_unsupported_mixture_extrapolation=bool(diagnostics[diagnostics['mode']=='P2_A'].unsupported_mixture_extrapolation.all()),
        main_quality_peak_max_log10_gap=float(peaks[peaks.domain=='main'].relative_gap.max()),
        main_typeII_significant=int(classified[classified.domain=='main'].significant.sum()),
        main_typeII_primary=int(((classified.domain=='main')&classified.significant&
            (classified.classification=='primary structural transition')).sum()),question4_started=False,
        P0_P1_frozen_results_reoptimized=False,earlier_parameters_refitted=False)
    (out/'execution_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    final_report(root,out,combined,main,context,high,classified,peaks,thresholds,robust,diagnostics,summary)
    logging.info('Final report and three supplementary figures complete')

# ============================================================
# Source: scripts/19_finalize_compute_optimization.py
# Original lines: 1-32
# Purpose: 最终阶段实际执行入口：19_finalize_compute_optimization.py
# ============================================================
"""Question 3 PHASE 3 only: new diagnostics and final reporting; no refitting."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/'src'))
import argparse
import logging
import pandas as pd
import yaml
from optimization.simplex_sensitivity import simplex_search,official_comparison


def main():
    """Run separately checkpointed simplex, official comparisons, and reporting."""
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['simplex','official','report','all'],default='all')
    args=parser.parse_args();out=PROJECT_ROOT/'outputs/optimization/phase3';out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(out/'execution.log',encoding='utf8')])
    cfg=yaml.safe_load((PROJECT_ROOT/'configs/optimization_phase3.yaml').read_text(encoding='utf8'))
    if cfg['question4_enabled']:raise ValueError('Question 4 is outside scope')
    if args.stage in ('simplex','all'):simplex_search(PROJECT_ROOT,cfg)
    if args.stage in ('official','all'):
        extremes=pd.read_csv(out/'P2_A_extreme_recipes.csv',float_precision='round_trip')
        official_comparison(PROJECT_ROOT,extremes)
    if args.stage in ('report','all'):
        from optimization.final_reporting import finalize
        finalize(PROJECT_ROOT,cfg)
    logging.info('PHASE 3 stage %s completed; earlier phases unchanged; Question 4 not started',args.stage)


if __name__=='__main__':main()

