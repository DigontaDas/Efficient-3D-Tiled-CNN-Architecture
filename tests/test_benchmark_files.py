import numpy as np
import SimpleITK as sitk
import pytest
from benchmark.preflight import validate_case_files

def write(path, origin=(0,0,0)):
    x=sitk.GetImageFromArray(np.zeros((4,5,6),np.uint8)); x.SetOrigin(origin); sitk.WriteImage(x,str(path))

def test_preflight_checks_all_cases(tmp_path):
    write(tmp_path/'image.nii.gz');write(tmp_path/'label.nii.gz')
    pairs={851:(tmp_path/'image.nii.gz',tmp_path/'label.nii.gz')}
    validate_case_files(pairs)
    write(tmp_path/'label.nii.gz', (1,0,0))
    with pytest.raises(ValueError): validate_case_files(pairs)
    with pytest.raises(FileNotFoundError): validate_case_files({852:(tmp_path/'missing',tmp_path/'also_missing')})

