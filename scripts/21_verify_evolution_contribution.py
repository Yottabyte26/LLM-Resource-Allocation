"""Independent equations, endpoints, resampling, inference and freeze verification."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
PROJECT_ROOT=Path(__file__).resolve().parents[1]
import hashlib
import json
import logging
import re
import warnings
import numpy as np
import pandas as pd
from scipy import stats


def main():
    """Recompute from frozen observations rather than trusting completion flags."""
    root=PROJECT_ROOT;out=root/'outputs/evolution/phase2';checks=[]
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    def read(f):return pd.read_csv(out/f,float_precision='round_trip')
    def check(name,passed,detail=''):checks.append(dict(check=name,passed=bool(passed),detail=str(detail)))
    def close(a,b,tol=1e-10):return np.allclose(a,b,atol=tol,rtol=1e-10,equal_nan=True)
    d=read('analysis_model_table.csv');p1=pd.read_csv(root/'outputs/evolution/phase1/evolution_model_table.csv',float_precision='round_trip')
    check('one_model_one_row',d.Model.is_unique and set(d.Model)==set(p1.Model))
    for k in ['A_z','A_100','N_B','D_tokens','training_compute']:
        check('frozen_values_'+k,np.array_equal(d.set_index('Model')[k].loc[p1.Model].to_numpy(),p1[k].to_numpy(),equal_nan=True))
    for k in ['S0','S1','S2','model_type','parameter_conflict','Epoch_match','Release Date','Submission Date']:
        check('frozen_identity_'+k,d[k].fillna('').tolist()==p1[k].fillna('').tolist())
    cfg=json.loads((out/'sample_definition.json').read_text(encoding='utf8'));origin=pd.to_datetime(cfg['primary_origin'])
    sub=pd.to_datetime(d['Submission Date'],errors='coerce');expected=(sub-origin).dt.total_seconds()/86400/cfg['days_per_month']
    check('continuous_months_actual_days',close(d.t_submission,expected,0))
    tiers=read('contribution_sample_tiers.csv');ok=d.A_z.notna()&np.isfinite(d.A_z)&d.N_B.gt(0)&np.isfinite(d.N_B)&sub.notna()
    for r in tiers.itertuples():
        mask=d[r.sample]&d.model_type.eq(r.model_type)&ok
        if r.tier in ['T1','T3']:mask&=np.isfinite(d.training_compute)&d.training_compute.gt(0)
        if r.tier in ['T2','T3']:mask&=np.isfinite(d.D_tokens)&d.D_tokens.gt(0)
        key='_'.join([r.sample,r.model_type,r.tier]);check('tier_'+key,int(mask.sum())==r.n and d[key].eq(mask).all())
    raw=pd.read_csv(root/'real_attachments/C_efficiency_evolution/leaderboard_enhanced.csv',float_precision='round_trip')
    dates=pd.to_datetime(raw['Submission Date'],errors='coerce');audit=read('dedup_audit.csv')
    for r in audit.itertuples():
        g=raw[raw.Model==r.Model];latest=dates.loc[g.index].max();selected=g.loc[dates.loc[g.index].eq(latest)] if pd.notna(latest) else g
        check('dedup_'+r.Model,sorted(json.loads(r.retained_source_rows))==sorted(selected.index.tolist()) and r.output_model_rows==1 and not r.winner_selection_used)
    definitions=json.loads((out/'model_definitions.json').read_text(encoding='utf8'));modelinfo={}
    coef=read('model_coefficients.csv');pred=read('model_predictions.csv');perf=read('model_performance.csv')
    check('no_infinite_inference',not np.isinf(coef.select_dtypes('number').to_numpy()).any())
    unavailable=coef[coef.inference_status.eq('UNIDENTIFIABLE_LEVERAGE_ONE')]
    check('unidentified_HC3_explicitly_missing',len(unavailable)>0 and unavailable[['SE','CI_low','CI_high','p_value']].isna().all().all())
    for info in definitions:
        label=info['model'];g=d.set_index('Model',drop=False).loc[info['Models']].reset_index(drop=True).copy()
        g['t_month']=g['t_release' if info['time_axis']=='release' else 't_submission']
        if 't_squared' in info['terms']:g['t_squared']=g.t_month**2
        if any(k.startswith('month_') for k in info['terms']):
            g=pd.concat([g,pd.get_dummies(pd.to_datetime(g['Submission Date']).dt.to_period('M').astype(str),prefix='month',drop_first=True,dtype=float)],axis=1)
        X=np.column_stack([np.ones(len(g)),g[info['terms']].to_numpy(float)]);y=g.A_z.to_numpy();beta=np.array(info['beta']);n,p=X.shape
        modelinfo[label]=(info,g,X,y,beta)
        if info['time_axis']=='release':check('release_C4_identity_'+label,g.Epoch_match.notna().all() and g['Release Date'].notna().all())
        if info['method']=='OLS':
            direct=np.linalg.lstsq(X,y,rcond=None)[0];check('OLS_coefficients_'+label,close(direct,beta))
            res=y-X@beta;pinv=np.linalg.pinv(X);hat=np.einsum('ij,ji->i',X,pinv)
            if np.min(1-hat)>1e-9:
                cov=(pinv*(res/(1-hat))**2)@pinv.T;SE=np.sqrt(np.diag(cov));critical=stats.t.ppf(.975,n-p)
                pv=2*stats.t.sf(abs(beta/SE),n-p)
                h=coef[(coef.model==label)&coef.inference.eq('HC3')].set_index('term').loc[['intercept']+info['terms']]
                check('manual_HC3_'+label,close(SE,h.SE,1e-8) and close(pv,h.p_value,1e-8) and close(beta-critical*SE,h.CI_low,1e-8) and close(beta+critical*SE,h.CI_high,1e-8))
            h=pred[pred.model==label].set_index('Model').loc[g.Model]
            check('in_sample_prediction_'+label,close(X@beta,h.fitted_A))
            for i in range(n):
                xx=np.delete(X,i,axis=0);yy=np.delete(y,i);full=np.linalg.matrix_rank(xx)==p
                expected=X[i]@np.linalg.lstsq(xx,yy,rcond=None)[0] if full else np.nan
                check('LOOCV_'+label+'_'+str(i),close(expected,h.LOOCV_predicted_A.iloc[i]))
        for evaluation,column in [('in_sample','fitted_A'),('LOOCV','LOOCV_predicted_A')]:
            h=pred[pred.model==label];v=perf[(perf.model==label)&perf.evaluation.eq(evaluation)].iloc[0];mask=h[column].notna()
            e=h.loc[mask,column]-h.loc[mask,'observed_A'];yobs=h.loc[mask,'observed_A'];total=np.sum((yobs-yobs.mean())**2)
            if evaluation=='LOOCV' and mask.sum()<len(h):check('partial_CV_not_substituted_'+label,pd.isna(v.RMSE) and not v.full_LOOCV_available)
            else:check('performance_'+label+'_'+evaluation,close([np.sqrt(np.mean(e**2)),abs(e).mean(),1-np.sum(e**2)/total],[v.RMSE,v.MAE,v.R2]))
    decomp=read('contribution_decomposition.csv');samples=json.loads((out/'sample_definition.json').read_text(encoding='utf8'))
    def state(data,terms,frac,bounds):
        lo,hi=bounds;t=data.t_month.to_numpy();em=(t>=lo)&(t<=lo+frac*(hi-lo));lm=(t>=hi-frac*(hi-lo))&(t<=hi)
        if not em.any() or not lm.any():return None,int(em.sum()),int(lm.sum())
        states=[]
        for mask in [em,lm]:
            h=data.iloc[np.flatnonzero(mask)]
            s={'lnN':h.lnN.median(),'lnD':np.log(h.D_tokens.median()) if h.D_tokens.notna().any() else np.nan,
                'lnC':np.log(h.training_compute.median()) if h.training_compute.notna().any() else np.nan,'t_month':h.t_month.median()}
            states.append(s)
        return states,int(em.sum()),int(lm.sum())
    for r in decomp.to_dict('records'):
        info,g,X,y,beta=modelinfo[r['model']];states,ne,nl=state(g,info['terms'],r['window_fraction'],[r['time_bound_min'],r['time_bound_max']])
        if states:
            early,late=states;ds=sum(beta[j+1]*(late[k]-early[k]) for j,k in enumerate(info['terms']) if k!='t_month')
            dt=beta[info['terms'].index('t_month')+1]*(late['t_month']-early['t_month'])
            check('point_ShAP_'+r['series']+'_'+str(r['window_fraction']),close([ds,dt,ds+dt],[r['Delta_scale'],r['Delta_tech'],r['Delta_pred']]) and ne==r['early_n'] and nl==r['late_n'])
            physical=ds+dt>0 and ds>=0 and dt>=0
            check('signed_shares_'+r['series']+'_'+str(r['window_fraction']),r['nonnegative_growth_shares_valid']==physical and
                (physical or pd.isna(r['share_scale']) and pd.isna(r['share_tech'])) and close(r['absolute_share_scale'],abs(ds)/(abs(ds)+abs(dt))))
    boot=read('contribution_bootstrap.csv');params=read('coefficient_bootstrap.csv')
    check('5000_draws_per_fitted_model',params.groupby('model').size().eq(5000).all())
    check('5000_draws_per_contribution_window',boot.groupby(['model','window_fraction']).size().eq(5000).all())
    boot_groups={key:g.set_index('replicate') for key,g in boot.groupby(['model','window_fraction'])}
    fractions=sorted(boot.window_fraction.unique())
    for label,rows in params.groupby('model',sort=False):
        info,g,X,y,beta=modelinfo[label];terms=info['terms'];names=['intercept']+terms;n,p=X.shape
        vals=g[['lnN','D_tokens','training_compute','t_month']].to_numpy(float)
        bounds=samples['primary_bounds'] if info['time_axis']=='submission' else [g.t_month.min(),g.t_month.max()]
        for r in rows.to_dict('records'):
            ix=np.array(json.loads(r['source_model_indices']),int);Xt=X[ix];yt=y[ix];full=np.linalg.matrix_rank(Xt)==p
            direct=np.linalg.lstsq(Xt,yt,rcond=None)[0] if full else np.full(p,np.nan)
            check('bootstrap_refit_'+label+'_'+str(r['replicate']),len(ix)==n and (ix>=0).all() and (ix<n).all() and full==r['full_rank'] and close(direct,[r[k] for k in names]))
            v=vals[ix]
            for frac in fractions:
                br=boot_groups[(label,frac)].loc[r['replicate']];lo,hi=bounds;em=(v[:,3]>=lo)&(v[:,3]<=lo+frac*(hi-lo));lm=(v[:,3]>=hi-frac*(hi-lo))&(v[:,3]<=hi)
                expected_status='RANK_DEFICIENT' if not full else 'EMPTY_ENDPOINT_WINDOW' if not em.any() or not lm.any() else 'OK'
                valid=br.status==expected_status and int(em.sum())==br.early_n and int(lm.sum())==br.late_n
                if expected_status=='OK':
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore',RuntimeWarning);e=np.nanmedian(v[em],axis=0);l=np.nanmedian(v[lm],axis=0)
                    changes={'lnN':l[0]-e[0],'lnD':np.log(l[1])-np.log(e[1]),'lnC':np.log(l[2])-np.log(e[2]),'t_month':l[3]-e[3]}
                    ds=sum(direct[j+1]*changes[k] for j,k in enumerate(terms) if k!='t_month');dt=direct[terms.index('t_month')+1]*changes['t_month']
                    valid=valid and close([ds,dt,ds+dt],[br.Delta_scale,br.Delta_tech,br.Delta_pred]) and abs(br.Delta_pred-br.Delta_scale-br.Delta_tech)<1e-10
                check('bootstrap_endpoints_'+label+'_'+str(r['replicate'])+'_'+str(frac),valid)
        logging.info('Verified all 5000 fits and endpoints for %s',label)
    ci=read('contribution_bootstrap_summary.csv')
    for r in ci.itertuples():
        h=boot[(boot.model==r.model)&np.isclose(boot.window_fraction,r.window_fraction)&boot.status.eq('OK')];v=h[r.quantity].dropna()
        check('bootstrap_CI_'+r.model+'_'+r.quantity+'_'+str(r.window_fraction),len(h)==r.valid_decomposition_draws and len(v)==r.valid_quantity_draws and close(v.quantile([.025,.975]).to_numpy(),[r.CI_low,r.CI_high]))
    indices=read('non_scale_technology_index.csv')
    for label,h in indices.groupby('model'):
        info,g,X,y,beta=modelinfo[label];h=h.set_index('Model').loc[g.Model]
        check('technology_residual_'+label,close(g.A_z-(beta[0]+beta[1]*g.lnN),h.TechResidual))
    monthly=read('monthly_technology_frontier.csv')
    for r in monthly.itertuples():
        h=indices[indices.model.eq(r.model)&pd.to_datetime(indices.Submission_Date).dt.to_period('M').astype(str).eq(r.month)];v=h.TechResidual
        check('monthly_tech_'+r.model+'_'+r.month,len(v)==r.n and close([v.median(),v.quantile(.75),v.quantile(.9),v.quantile(.95)],[r.median_TechResidual,r.P75,r.P90,r.P95]))
    front=read('monthly_ability_frontier.csv');g=modelinfo['M1_linear'][1]
    for r in front.itertuples():
        h=g[pd.to_datetime(g['Submission Date']).dt.to_period('M').astype(str).eq(r.month)];top=h.sort_values(['A_z','Model'],ascending=[False,True]).head(3)
        check('monthly_frontier_'+r.month,len(h)==r.n and len(top)==r.top3_actual_n and json.loads(r.top3_Models)==top.Model.tolist() and close([h.A_z.quantile(.9),h.A_z.quantile(.95),top.A_z.mean()],[r.P90_A,r.P95_A,r.top3_mean_A]))
    time=read('time_axis_sensitivity.csv')
    for r in time.itertuples():check('matched_time_axis_'+r.series+'_'+str(r.window_fraction),r.submission_n==r.release_n)
    aux=read('technology_auxiliary_correlations.csv')
    for r in aux.itertuples():
        info,g,X,y,beta=modelinfo[r.model];a=g.A_z if not r.analysis.startswith('C8_same_run') else g.C8_A_z_same_run
        residual=a-(beta[0]+beta[1]*g.lnN);h=pd.DataFrame({'x':residual,'y':g[r.variable]}).dropna()
        if len(h)>=3 and h.x.nunique()>1 and h.y.nunique()>1:
            check('auxiliary_'+r.model+'_'+r.analysis+'_'+r.variable,len(h)==r.n and close([stats.pearsonr(h.x,h.y).statistic,stats.spearmanr(h.x,h.y).statistic],[r.Pearson,r.Spearman]))
    report=(root/'outputs/report/question4_phase2_contribution_decomposition.md').read_text(encoding='utf8')
    check('report_no_broken_latex_controls',not any(c in report for c in ['\x08','\x0c','\x0b']))
    for target in re.findall(r'\]\(([^)]+)\)',report):
        if target.startswith('http') or target.endswith('verification_summary.json'):continue
        check('report_link_'+target,(root/'outputs/report'/target).resolve().is_file())
    figs=read('figure_manifest.csv');check('required_ten_plus_diagnostics_figures',len(figs)>=10 and all((root/p).stat().st_size>10000 for p in figs.file))
    for file in ['input_manifest.csv','implementation_manifest.csv']:
        for r in read(file).itertuples():
            with (root/r.file).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            check('manifest_'+r.file,digest==r.sha256)
    for p in out.glob('*.csv'):check('CSV_readable_'+p.name,p.stat().st_size>0 and len(pd.read_csv(p,nrows=5).columns)>0)
    frozen=json.loads((out/'frozen_before_question4_phase2.json').read_text(encoding='utf8'));changes=[]
    for relative,item in frozen.items():
        p=root/relative
        if not p.is_file():changes.append(relative);continue
        with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=item['sha256'] or p.stat().st_size!=item['size']:changes.append(relative)
    check('all_frozen_files_unchanged',not changes,changes)
    summary=json.loads((out/'execution_summary.json').read_text(encoding='utf8'))
    check('authorized_scope_only',not any(summary[k] for k in ['frozen_ability_refitted','bridge_refitted','loss_conversion_performed','future_prediction_performed','final_frontier_fitted','causal_conclusion']))
    frame=pd.DataFrame(checks);frame.to_csv(out/'independent_verification.csv',index=False,encoding='utf-8-sig')
    result=dict(checks=len(frame),passed=int(frame.passed.sum()),failed=int((~frame.passed).sum()),frozen_files_checked=len(frozen),frozen_files_changed=changes,
        contribution_model_bootstrap_draws=len(params),endpoint_bootstrap_records=len(boot),future_prediction_performed=False,Loss_conversion_performed=False)
    (out/'verification_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))
    if not frame.passed.all():print(frame[~frame.passed].head(50).to_string(index=False));raise ValueError('Question 4 PHASE 2 independent verification failed')


if __name__=='__main__':main()
