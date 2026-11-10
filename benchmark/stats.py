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


def paired_analysis(a,b,metric,tie_tolerance=0.0,seed=42,n_boot=2000):
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
    ci_lo, ci_hi = bootstrap_ci(d, seed=seed, n_boot=n_boot, confidence=0.95)
    q25 = float(np.quantile(d, 0.25))
    q75 = float(np.quantile(d, 0.75))
    return {
        'n': len(d),
        'mean_difference': float(d.mean()),
        'diff_ci95_lo': ci_lo,
        'diff_ci95_hi': ci_hi,
        'median_difference': float(np.median(d)),
        'q25_difference': q25,
        'q75_difference': q75,
        'iqr_difference': float(q75 - q25),
        'p': p,
        'effect_size_rank_biserial': effect,
        'wins': int((d > tie_tolerance).sum()),
        'ties': int((abs(d) <= tie_tolerance).sum()),
        'losses': int((d < -tie_tolerance).sum()),
    }


def analyze_family(tables, hypotheses, allow_nan_filtering=False):
    if not hypotheses: raise ValueError('Hypothesis family must be declared')
    results=[]
    for h in hypotheses:
        direction=h.get('direction','higher')
        if direction not in ('higher','lower'): raise ValueError('Invalid metric direction')
        ta, tb = tables[h['a']], tables[h['b']]
        metric = h['metric']
        if allow_nan_filtering:
            for df in (ta, tb):
                if 'case_id' not in df or metric not in df or df['case_id'].isna().any() or df['case_id'].duplicated().any():
                    raise ValueError('Missing or duplicate case IDs/metric')
            if set(ta.case_id) != set(tb.case_id):
                raise ValueError('Case ID sets differ')
            merged = ta[['case_id', metric]].merge(tb[['case_id', metric]], on='case_id', validate='one_to_one', suffixes=('_a', '_b'))
            finite_ids = set(merged.dropna()['case_id'])
            ta_f = ta[ta['case_id'].isin(finite_ids)]
            tb_f = tb[tb['case_id'].isin(finite_ids)]
            n_excl = int(len(ta) - len(ta_f))
            if len(ta_f) == 0:
                r = {
                    'n': 0,
                    'mean_difference': np.nan,
                    'diff_ci95_lo': np.nan,
                    'diff_ci95_hi': np.nan,
                    'median_difference': np.nan,
                    'q25_difference': np.nan,
                    'q75_difference': np.nan,
                    'iqr_difference': np.nan,
                    'p': 1.0,
                    'effect_size_rank_biserial': np.nan,
                    'wins': 0,
                    'ties': 0,
                    'losses': 0,
                    'a_mean': np.nan,
                    'b_mean': np.nan,
                    'n_excluded_nan': n_excl,
                }
            else:
                r = paired_analysis(ta_f, tb_f, metric, h.get('tie_tolerance', 0.0))
                if direction == 'lower':
                    r['wins'], r['losses'] = r['losses'], r['wins']
                r['a_mean'] = float(ta_f[metric].mean())
                r['b_mean'] = float(tb_f[metric].mean())
                r['n_excluded_nan'] = n_excl
        else:
            r=paired_analysis(ta,tb,metric,h.get('tie_tolerance',0.0))
            if direction == 'lower': r['wins'],r['losses']=r['losses'],r['wins']
            r['a_mean'] = float(ta[metric].mean())
            r['b_mean'] = float(tb[metric].mean())
            r['n_excluded_nan'] = 0
        r.update(h); results.append(r)
    for r,p in zip(results,holm_adjust([r['p'] for r in results])): r['holm_p']=p
    return results


