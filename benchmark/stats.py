"""Case-matched statistics; hypothesis families must be explicitly declared."""
import numpy as np
import pandas as pd
from scipy import stats


def holm_adjust(p_values):
    p=np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any((p<0)|(p>1)):
        raise ValueError('Invalid p-values')
    order=np.argsort(p); adjusted=np.empty(len(p)); running=0.0
    for rank,i in enumerate(order):
        running=max(running, (len(p)-rank)*p[i]); adjusted[i]=min(1.0,running)
    return adjusted.tolist()


def bootstrap_ci(values, seed=42, n_boot=2000, confidence=.95):
    v=np.asarray(values,dtype=float)
    if v.ndim != 1 or len(v)==0 or not np.all(np.isfinite(v)) or n_boot<1 or not 0<confidence<1:
        raise ValueError('Invalid bootstrap inputs')
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(v,size=len(v),replace=True).mean() for _ in range(n_boot)])
    alpha=(1-confidence)/2
    return tuple(float(x) for x in np.quantile(means,[alpha,1-alpha]))


def paired_analysis(a,b,metric,tie_tolerance=0.0):
    for df in (a,b):
        if 'case_id' not in df or metric not in df or df['case_id'].isna().any() or df['case_id'].duplicated().any():
            raise ValueError('Missing or duplicate case IDs/metric')
    if set(a.case_id)!=set(b.case_id): raise ValueError('Case ID sets differ')
    pair=a[['case_id',metric]].merge(b[['case_id',metric]],on='case_id',validate='one_to_one',suffixes=('_a','_b'))
    d=pair[metric+'_a'].to_numpy(dtype=float)-pair[metric+'_b'].to_numpy(dtype=float)
    if len(d)==0 or not np.all(np.isfinite(d)): raise ValueError('Missing or non-finite paired metric')
    if tie_tolerance < 0: raise ValueError('Invalid tie tolerance')
    nonzero=d[d!=0]
    if len(nonzero):
        ranks=stats.rankdata(abs(nonzero)); effect=float((ranks[nonzero>0].sum()-ranks[nonzero<0].sum())/ranks.sum())
        p=float(stats.wilcoxon(d,alternative='two-sided').pvalue)
    else:
        effect=0.0; p=1.0
    return {'n':len(d), 'mean_difference':float(d.mean()),'p':p,'effect_size_rank_biserial':effect,
            'wins':int((d>tie_tolerance).sum()),'ties':int((abs(d)<=tie_tolerance).sum()),'losses':int((d < -tie_tolerance).sum())}


def analyze_family(tables, hypotheses):
    if not hypotheses: raise ValueError('Hypothesis family must be declared')
    results=[]
    for h in hypotheses:
        r=paired_analysis(tables[h['a']],tables[h['b']],h['metric'],h.get('tie_tolerance',0.0))
        direction=h.get('direction','higher')
        if direction not in ('higher','lower'): raise ValueError('Invalid metric direction')
        if direction == 'lower': r['wins'],r['losses']=r['losses'],r['wins']
        r.update(h); results.append(r)
    for r,p in zip(results,holm_adjust([r['p'] for r in results])): r['holm_p']=p
    return results

