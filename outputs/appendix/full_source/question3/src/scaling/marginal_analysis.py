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
