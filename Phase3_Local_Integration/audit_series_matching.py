import pydicom, os

DICOM_ROOT = r'H:\CT_Scans_Thesis'

def select_best_ccta_series(dicom_dir):
    files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if not os.path.isdir(os.path.join(dicom_dir, f))]
    series_map = {}
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            if hasattr(ds, 'SeriesNumber'):
                se_num = int(ds.SeriesNumber)
                se_desc = str(getattr(ds, 'SeriesDescription', ''))
                thick = float(getattr(ds, 'SliceThickness', 999.0))
                z_pos = float(ds.ImagePositionPatient[2]) if hasattr(ds, 'ImagePositionPatient') else (float(ds.InstanceNumber) if hasattr(ds, 'InstanceNumber') else 0.0)
                if se_num not in series_map:
                    series_map[se_num] = {'desc': se_desc, 'thick': thick, 'files': []}
                series_map[se_num]['files'].append((z_pos, f))
        except Exception:
            pass

    EXCLUDE_KEYWORDS = ['scout', 'calcium', 'scoring', 'plain', 'report', 'film', 'pages', 'dose', 'prep', '0-90', 'auto state', 'saved state']
    
    candidates = []
    for se_num, info in series_map.items():
        desc = info['desc'].lower()
        count = len(info['files'])
        thick = info['thick']
        
        # Skip excluded series or partial reconstructions (< 50 slices)
        if any(kw in desc for kw in EXCLUDE_KEYWORDS) or count < 50:
            continue
            
        score = 0
        if 'ss-freeze' in desc or 'freeze' in desc:
            score += 100
        elif 'temporal' in desc:
            score += 80
        elif 'sseg' in desc and '0-90' not in desc:
            score += 70
        elif 'coronary' in desc:
            score += 50
            
        if '75%' in desc:
            score += 20
        elif '45%' in desc:
            score += 10
            
        if thick <= 0.75:
            score += 15
            
        candidates.append((score, count, se_num, info))
        
    if not candidates:
        return None, None
        
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_candidate = candidates[0]
    return best_candidate[2], best_candidate[3]

for cid in range(61, 91):
    case_folder = os.path.join(DICOM_ROOT, f'CT {cid}', 'A')
    if not os.path.exists(case_folder):
        print(f'CT{cid:02d}: NOT FOUND', flush=True)
        continue
    se_num, info = select_best_ccta_series(case_folder)
    if se_num is not None:
        desc = info['desc']
        cnt = len(info['files'])
        thk = info['thick']
        print(f"CT{cid:02d} -> Matched Se #{se_num:<4} | Desc: {desc:<35} | Slices: {cnt:>4} | Thick: {thk} mm", flush=True)
    else:
        print(f"CT{cid:02d} -> [!] No candidate found", flush=True)
