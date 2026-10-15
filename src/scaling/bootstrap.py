"""Whole-trajectory resampling; bounded CPU workers and no IID checkpoints."""
from __future__ import annotations
from collections import Counter
import numpy as np
from .classical import weights,fit,predict,metrics

_DATA=None


def initialize(data):
    """Workers receive ONLY B1 arrays, its primary parameters and fixed fit settings."""
    global _DATA
    _DATA=data


def repetition(task):
    """Fit each resampled trajectory multiplicity with 32 positive multi-starts."""
    number,drawn,seed=task;d=_DATA;groups=d['group'];counts=Counter(drawn)
    selected=[g for g in np.unique(groups) if counts[g]>0]
    prefix=dict(repetition=number,drawn_trajectories=list(drawn),multiplicities=dict(counts),
        distinct_trajectories=len(selected),seed=seed,fit_dataset='B1_only',bootstrap_unit='trajectory')
    try:
        w=weights(groups,True,counts)
        best,_=fit(d['N'],d['D'],d['L'],w,'M0_N_D',d['settings'],seed,
            warm=d['primary_parameters'],count=d['settings']['bootstrap_multistart_count'],save_starts=False)
        p=predict(best['parameters'],d['N'],d['D'])
        identified=best['jacobian_rank']==5 and len(selected)>=3
        result=dict(**prefix,**best['parameters'],objective=best['objective'],converged=True,
            identifiable=identified,eligible_parameter_CI=identified,status=best['status'],jacobian_rank=best['jacobian_rank'],
            condition_number=best['jacobian_condition_number'],active_bounds=best['active_bounds'],
            n_starts=best['n_starts'],prediction_RMSE_on_full_B1=metrics(d['L'],p)['RMSE'],error=None)
        return result,p
    except Exception as exc:
        return dict(**prefix,converged=False,identifiable=False,eligible_parameter_CI=False,error=repr(exc)),None
