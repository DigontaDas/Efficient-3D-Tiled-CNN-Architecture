import pytest
from benchmark.preflight import validate_splits, cache_key, validate_thresholds

def test_disjoint_and_locked_test():
    validate_splits({'train':[1,2], 'val':[3], 'test':[851,852]})
    with pytest.raises(ValueError): validate_splits({'train':[1,851], 'val':[3], 'test':[851]})
    with pytest.raises(ValueError): validate_splits({'train':[1], 'val':[3,3], 'test':[851]})
    with pytest.raises(ValueError): validate_splits({'train':[1], 'val':[3], 'test':[3]})

def test_cache_is_sensitive():
    x={'case_ids':[1,2], 'checkpoint':'abc','settings':{'arm':'A'}, 'metric_version':'1','commit':'abc'}
    assert cache_key(x)==cache_key(dict(x))
    for k,v in [('case_ids',[3,4]),('checkpoint','def'),('settings',{'arm':'B'}),('metric_version','2'),('commit','def')]:
        y=dict(x); y[k]=v; assert cache_key(x)!=cache_key(y)

def test_thresholds_validation_only():
    s={'train':[1], 'val':[2,3], 'test':[851]}
    good={'split':'val','case_ids':[2,3],'thresholds':{'rasnet':.5}}
    validate_thresholds(good,s,['rasnet'])
    for bad in [dict(good, split='test'),dict(good,case_ids=[2,851]),dict(good,thresholds={'rasnet':2})]:
        with pytest.raises(ValueError): validate_thresholds(bad,s,['rasnet'])
