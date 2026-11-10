import pandas as pd
import pytest
from benchmark.stats import paired_analysis, holm_adjust, bootstrap_ci

def frame(ids, vals): return pd.DataFrame({'case_id':ids, 'dice':vals})

def test_pairing_and_known_values():
    a=frame([1,2,3],[.5,.6,.7]); b=frame([3,1,2],[.5,.4,.6])
    r=paired_analysis(a,b,'dice')
    assert (r['wins'],r['ties'],r['losses'])==(2,1,0)
    assert r['mean_difference']==pytest.approx(.1)
    assert r['effect_size_rank_biserial']==pytest.approx(1)
    assert holm_adjust([.01,.04,.03])==pytest.approx([.03,.06,.06])
    assert bootstrap_ci([2,2,2], seed=42)==(2.,2.)

def test_bad_ids():
    with pytest.raises(ValueError): paired_analysis(frame([1,2],[1,2]),frame([2,3],[1,2]),'dice')
    with pytest.raises(ValueError): paired_analysis(frame([1,1],[1,2]),frame([1,2],[1,2]),'dice')

def test_lower_is_better_and_declared_family():
    from benchmark.stats import analyze_family
    a=frame([1,2,3],[1,2,3]); b=frame([1,2,3],[2,3,3])
    r=analyze_family({'ours':a,'baseline':b},[{'a':'ours','b':'baseline','metric':'dice','direction':'lower'}])[0]
    assert (r['wins'],r['ties'],r['losses'])==(2,1,0)
    assert 'diff_ci95_lo' in r and 'diff_ci95_hi' in r and 'median_difference' in r
    assert r['n_excluded_nan'] == 0
    with pytest.raises(ValueError): analyze_family({'ours':a,'baseline':b},[])


def test_allow_nan_filtering_tracks_exclusions():
    from benchmark.stats import analyze_family
    a = frame([1, 2, 3, 4], [0.8, float('nan'), 0.7, 0.9])
    b = frame([1, 2, 3, 4], [0.6, 0.5, 0.6, 0.7])
    with pytest.raises(ValueError):
        analyze_family({'a': a, 'b': b}, [{'a': 'a', 'b': 'b', 'metric': 'dice', 'direction': 'higher'}], allow_nan_filtering=False)
    res = analyze_family({'a': a, 'b': b}, [{'a': 'a', 'b': 'b', 'metric': 'dice', 'direction': 'higher'}], allow_nan_filtering=True)[0]
    assert res['n'] == 3
    assert res['n_excluded_nan'] == 1
    assert res['wins'] == 3


