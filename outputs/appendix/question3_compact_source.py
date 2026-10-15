# ============================================================
# Source: src/optimization/costs.py
# Original lines: 2-53
# Purpose: 6ND、注意力成本与exp/power/log质量增量成本
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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
# Source: src/scaling/marginal_analysis.py
# Original lines: 6-141
# Purpose: 最终广义标度律、边界策略、解析导数/弹性及Q+0.1精确求根
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: src/optimization/setup.py
# Original lines: 2-18
# Purpose: 官方成本、Q0、决策边界与832经验候选的来源
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: src/optimization/setup.py
# Original lines: 263-373
# Purpose: 官方成本、Q0、决策边界与832经验候选的来源
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: src/optimization/resource_solver.py
# Original lines: 2-11
# Purpose: D_B解析消元、可行性、网格/SLSQP/DE及P0/P1求解
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: src/optimization/resource_solver.py
# Original lines: 19-216
# Purpose: D_B解析消元、可行性、网格/SLSQP/DE及P0/P1求解
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: src/optimization/resource_solver.py
# Original lines: 261-378
# Purpose: D_B解析消元、可行性、网格/SLSQP/DE及P0/P1求解
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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
# Source: src/optimization/simplex_sensitivity.py
# Original lines: 2-44
# Purpose: P2-A配方证书与数值交叉检验；P2-B边界无界性，仅理论敏感性
# P2-A/P2-B are theory / sensitivity only. Final engineering hierarchy: P0 main, P1 empirical enhancement; P2-B has no finite attained optimum.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: src/optimization/simplex_sensitivity.py
# Original lines: 89-267
# Purpose: P2-A配方证书与数值交叉检验；P2-B边界无界性，仅理论敏感性
# P2-A/P2-B are theory / sensitivity only. Final engineering hierarchy: P0 main, P1 empirical enhancement; P2-B has no finite attained optimum.
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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
# Source: src/optimization/transitions.py
# Original lines: 2-57
# Purpose: 连续预算扫描路径上的Type I/II转移检验与区间
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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

# ============================================================
# Source: src/optimization/transitions.py
# Original lines: 99-176
# Purpose: 连续预算扫描路径上的Type I/II转移检验与区间
# Unchanged excerpt; omitted routines remain in full_source.
# ============================================================
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
# Source: src/optimization/setup.py
# Original lines: 164-169
# Purpose: 官方成本、Q0、决策边界与832经验候选的来源
# Original call-site excerpt, not a standalone function.
# ============================================================
    crit = official['training_coefficient']/official['eta']
    save_csv(out/'context_critical_analysis.csv', [dict(eta_scheme='official_main', eta=official['eta'],
        L_ctx_crit=crit, within_C7_observed_range=bool(valid.min()<=crit<=valid.max()),
        exact_C7_value=bool((valid==crit).any()), models_above_critical=int((valid>crit).sum()),
        models_equal_critical=int((valid==crit).sum()), models_below_critical=int((valid<crit).sum()),
        valid_models=len(valid), fraction_above=float((valid>crit).mean()), evidence='Official Word Question 3')])

