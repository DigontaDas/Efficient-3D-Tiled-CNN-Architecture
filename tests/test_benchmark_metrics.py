import numpy as np
import pytest
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist
from benchmark.metrics import compute_metrics, surface_distances, check_geometry


def cube(lo, hi):
    a = np.zeros((9, 9, 9), bool); a[lo:hi, lo:hi, lo:hi] = True
    return a

def test_nested_surfaces():
    d = surface_distances(cube(2, 7), cube(1, 8), (1, 1, 1))
    assert np.allclose(d, 1)

def test_anisotropic_and_random_bruteforce():
    rng = np.random.default_rng(41)
    for _ in range(8):
        a, b = rng.random((5, 6, 7)) > .7, rng.random((5, 6, 7)) > .7
        spacing = (3., 1.5, .5)
        sa, sb = a & ~binary_erosion(a), b & ~binary_erosion(b)
        expected = cdist(np.argwhere(sa)*spacing, np.argwhere(sb)*spacing).min(axis=1)
        assert np.allclose(surface_distances(a,b,spacing), expected)

def test_shifted_masks():
    a = np.zeros((7,7,7), bool); a[2,2,2] = 1
    b = np.zeros_like(a); b[3,2,2] = 1
    m = compute_metrics(a,b,(2,1,1))
    assert m['hd95'] == 2 and m['asd'] == 2 and m['dice'] == 0

@pytest.mark.parametrize('pa,ga',[(0,0),(0,1),(1,0),(1,1)])
def test_empty_policy(pa, ga):
    a = np.zeros((4,4,4), bool); b = a.copy(); a[1,1,1]=pa; b[1,1,1]=ga
    m=compute_metrics(a,b,(1,1,1))
    assert (m['status'] == 'ok') == bool(pa and ga)
    if not (pa and ga): assert np.isnan(m['hd95']) and np.isnan(m['asd'])

def test_injected_failure(monkeypatch):
    import benchmark.metrics as bm
    def fail(*args, **kwargs): raise RuntimeError('injected')
    monkeypatch.setattr(bm, 'distance_transform_edt', fail)
    with pytest.raises(RuntimeError): compute_metrics(cube(2,7),cube(1,8),(1,1,1))

def test_geometry():
    x={'size':(3,4,5),'origin':(0,0,0),'spacing':(1,1,2),'direction':(1,0,0,0,1,0,0,0,1)}
    check_geometry(x,dict(x))
    for k,v in [('direction',(-1,0,0,0,1,0,0,0,1)),('origin',(1,0,0)),('spacing',(1,1,3)),('size',(4,4,5))]:
        y=dict(x); y[k]=v
        with pytest.raises(ValueError): check_geometry(x,y)
