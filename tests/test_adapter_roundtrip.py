import numpy as np
import SimpleITK as sitk
import torch
from benchmark.adapters.monai_models import infer_probability

def test_nifti_roundtrip_and_window_padding(tmp_path):
    img=sitk.GetImageFromArray(np.ones((17,18,19),dtype=np.float32)*100)
    img.SetSpacing((.7,.8,1.2)); p=tmp_path/'synthetic.nii.gz'; sitk.WriteImage(img,str(p))
    class Predictor(torch.nn.Module):
        def forward(self,x): return torch.cat((torch.zeros_like(x),torch.ones_like(x)),dim=1)
    for name in ('rasnet','segresnet','vnet','3dunet'):
        net = Predictor() if name != '3dunet' else torch.nn.Identity()
        result=infer_probability(name,net,str(p),'cpu',roi_size=(32,32,32),sw_batch_size=1)
        assert result.shape == (17,18,19)
        assert np.isfinite(result).all() and np.all((result>=0)&(result<=1))
