"""Fail-closed split, provenance and cache validation."""
import hashlib
import json
import math


def validate_splits(splits):
    parts={}
    for name in ('train','val','test'):
        values=splits.get(name)
        if not isinstance(values,list) or not values or any(isinstance(x,bool) or not isinstance(x,int) for x in values):
            raise ValueError(f'Invalid or empty {name} IDs')
        if len(values)!=len(set(values)): raise ValueError(f'Duplicate {name} IDs')
        parts[name]=set(values)
    if parts['train'] & parts['val'] or parts['train'] & parts['test'] or parts['val'] & parts['test']:
        raise ValueError('Overlapping splits')
    if (parts['train'] | parts['val']) & set(range(851,1001)):
        raise ValueError('Locked test ID in train or validation')
    if not parts['test'] <= set(range(851,1001)):
        raise ValueError('ImageCAS test IDs must be in locked range')
    return parts


def cache_key(payload):
    required={'case_ids','checkpoint','settings','metric_version','commit'}
    if not required <= payload.keys(): raise ValueError('Incomplete cache key')
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_thresholds(data,splits,models):
    validate_splits(splits)
    ids=data.get('case_ids',[])
    if data.get('split')!='val' or len(ids)!=len(set(ids)) or set(ids)!=set(splits['val']):
        raise ValueError('Thresholds must use the entire validation split only')
    for model in models:
        t=data.get('thresholds',{}).get(model)
        if not isinstance(t,(int,float)) or isinstance(t,bool) or not math.isfinite(t) or not 0<=t<=1:
            raise ValueError(f'Invalid threshold for {model}')


def image_geometry(image):
    return {'size': image.GetSize(), 'origin': image.GetOrigin(),
            'spacing': image.GetSpacing(), 'direction': image.GetDirection()}


def validate_case_files(pairs):
    from pathlib import Path
    import SimpleITK as sitk
    from benchmark.metrics import check_geometry
    if not pairs: raise ValueError('Empty case list')
    for cid,(image,label) in pairs.items():
        for f in (image,label):
            if not Path(f).is_file(): raise FileNotFoundError(f'Case {cid}: {f}')
        check_geometry(image_geometry(sitk.ReadImage(str(image))), image_geometry(sitk.ReadImage(str(label))))
