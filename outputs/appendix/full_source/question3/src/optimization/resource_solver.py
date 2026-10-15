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
