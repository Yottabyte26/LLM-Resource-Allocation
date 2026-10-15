"""Frozen-model compositional perturbations, distinct from causal effects."""
from __future__ import annotations
from itertools import combinations
import numpy as np
from .composition import ILRTransformer


def perturb(p, targets, step):
    """Add absolute share step to each target; shrink all other shares pro rata.

    A zero target may be activated. Its response includes the fitted zero-
    replacement convention, so it is not an infinitesimal derivative at zero.
    """
    p = np.asarray(p, dtype=float)
    if p.shape != (17,) or not np.isfinite(p).all() or np.any(p < 0) or abs(p.sum()-1) > 1e-12:
        raise ValueError('Expected a closed 17-domain composition')
    targets = list(targets)
    if len(set(targets)) != len(targets) or step <= 0:
        raise ValueError('Invalid perturbation targets or step')
    rest = np.ones(17, dtype=bool); rest[targets] = False
    donor = float(p[rest].sum()); mass = step*len(targets)
    if donor <= 0 or mass > donor:
        raise ValueError('Insufficient donor mass; perturbation unavailable')
    out = p.copy(); out[targets] += step; out[rest] *= (donor-mass)/donor
    if (out < 0).any() or abs(out.sum()-1) > 1e-12:
        raise ArithmeticError('Perturbation violated simplex')
    return out


def direction(values, tolerance=1e-10):
    """Strict numerical sign agreement, not statistical significance."""
    x=np.asarray(values)
    if np.all(x < -tolerance): return 'lower_loss'
    if np.all(x > tolerance): return 'higher_loss'
    if np.all(abs(x) <= tolerance): return 'numerically_zero'
    return 'uncertain'


def summary(values):
    """Equal-baseline summaries, not a population-average causal estimand."""
    x=np.asarray(values)
    return dict(mean_effect=float(x.mean()), median_effect=float(np.median(x)),
                std_effect=float(x.std(ddof=1)), min_effect=float(x.min()), max_effect=float(x.max()))


def baselines(P, indices, frozen_model):
    """A4 ILR center, closed component median, and predicted P10/P50/P90 recipes."""
    closed=ILRTransformer.closure(P)
    ilr=frozen_model.regressor_.named_steps['ilr']
    center=ilr.inverse_transform(ilr.transform(P).mean(axis=0,keepdims=True))[0]
    median=np.median(closed,axis=0)
    if median.sum()<=0: raise ValueError('Componentwise median cannot be closed')
    median/=median.sum()
    result=[('A4_ILR_center', '', center), ('A4_closed_component_median', '', median)]
    pred=frozen_model.predict(P)
    for label,q in [('low',.1),('middle',.5),('high',.9)]:
        i=int(np.argmin(abs(pred-np.quantile(pred,q))))
        result.append((f'A4_predicted_{label}',indices[i],closed[i].copy()))
    return result


def analyze(models, bases, names, steps, tolerance):
    """Evaluate every model, baseline and step for singles and joint deviations."""
    singles=[]; joint_cases=[]; marginal=[]; joints=[]; pairs=list(combinations(range(17),2))
    for model_name,model in models.items():
        for step in steps:
            for label,index,p0 in bases:
                x=np.stack([p0]+[perturb(p0,[j],step) for j in range(17)]+
                           [perturb(p0,[j,k],step) for j,k in pairs])
                pred=model.predict(x); effects=pred[1:18]-pred[0]
                for j,name in enumerate(names):
                    singles.append(dict(model=model_name,step=step,baseline=label,baseline_recipe_index=index,
                        mixture_domain=name,baseline_share=float(p0[j]),zero_activation=bool(p0[j]==0),
                        baseline_prediction=float(pred[0]),perturbed_prediction=float(pred[1+j]),effect=float(effects[j])))
                for idx,(j,k) in enumerate(pairs):
                    dj=float(pred[18+idx]-pred[0]); deviation=dj-effects[j]-effects[k]
                    joint_cases.append(dict(model=model_name,step=step,baseline=label,domain_1=names[j],domain_2=names[k],
                        joint_effect=dj,sum_single_effects=float(effects[j]+effects[k]),joint_perturbation_deviation=float(deviation)))
            for name in names:
                subset=[r for r in singles if r['model']==model_name and r['step']==step and r['mixture_domain']==name]
                marginal.append(dict(model=model_name,step=step,mixture_domain=name,n_baselines=len(bases),
                    zero_activation_baselines=sum(r['zero_activation'] for r in subset),
                    **summary([r['effect'] for r in subset]),baseline_direction=direction([r['effect'] for r in subset],tolerance)))
            for j,k in pairs:
                subset=[r for r in joint_cases if r['model']==model_name and r['step']==step and r['domain_1']==names[j] and r['domain_2']==names[k]]
                deviations=[r['joint_perturbation_deviation'] for r in subset]
                joints.append(dict(model=model_name,step=step,domain_1=names[j],domain_2=names[k],n_baselines=len(bases),
                    mean_joint_effect=float(np.mean([r['joint_effect'] for r in subset])),
                    mean_sum_single_effects=float(np.mean([r['sum_single_effects'] for r in subset])),
                    **summary(deviations),baseline_deviation_direction=direction(deviations,tolerance)))
    stability=[]
    for name in names:
        values=[r['effect'] for r in singles if r['mixture_domain']==name]
        means=[r['mean_effect'] for r in marginal if r['mixture_domain']==name]
        en=[r['effect'] for r in singles if r['mixture_domain']==name and r['model']=='M1_ElasticNet']
        ridge=[r['effect'] for r in singles if r['mixture_domain']==name and r['model']=='M1_Ridge']
        model_agreement=bool(np.all(np.sign(en)==np.sign(ridge)))
        d=direction(values,tolerance)
        stability.append(dict(mixture_domain=name,stable_direction=bool(model_agreement and d!='uncertain'),
            direction=d,models_casewise_agree=model_agreement,aggregate_direction=direction(means,tolerance),
            n_cases=len(values),lower_loss_cases=sum(v < -tolerance for v in values),higher_loss_cases=sum(v > tolerance for v in values),
            min_case_effect=float(min(values)),max_case_effect=float(max(values))))
    joint_stability=[]
    for j,k in pairs:
        values=[r['joint_perturbation_deviation'] for r in joint_cases if r['domain_1']==names[j] and r['domain_2']==names[k]]
        d=direction(values,tolerance)
        joint_stability.append(dict(domain_1=names[j],domain_2=names[k],stable_deviation=d!='uncertain',direction=d,
            mean_deviation=float(np.mean(values)),min_deviation=float(min(values)),max_deviation=float(max(values))))
    return singles,marginal,stability,joint_cases,joints,joint_stability
