import numpy as np
import pytest
import SimpleITK as sitk
import torch
from benchmark.adapters.monai_models import build_random_model, infer_probability
from benchmark.metrics import check_geometry
from benchmark.preflight import image_geometry

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


def test_cpu_end_to_end_dry_run_all_five_adapters_and_vnet_padding(tmp_path):
    """
    CPU synthetic end-to-end dry-run covering all 5 adapters:
    - Real architectures (untrained random weights via build_random_model) for RASNet, SegResNet, V-Net, and 3D U-Net.
    - Explicit nnUNetPredictor test double for nnU-Net V2 (verifying SimpleITK Z,Y,X axis/spacing contract without requiring GPU/plans).
    - Verifies coordinate inversion, 3D U-Net 128^3 resizing round-trip, and V-Net window padding when roi_size=(20,20,20) is not divisible by 16.
    """
    from benchmark.adapters.nnunet_adapter import infer_nnunet_probability, export_mask_like, build_nnunet_predictor

    arr = np.zeros((17, 18, 19), dtype=np.float32)
    arr[4:12, 5:14, 6:15] = 250.0
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.7, 0.8, 1.2))
    img.SetOrigin((10.5, -22.0, 4.25))
    img.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    img_path = tmp_path / "case_synth.nii.gz"
    sitk.WriteImage(img, str(img_path))

    # 1. Real architectures for the 4 MONAI models (including non-multiple-of-16 roi_size=(20,20,20) for V-Net padding)
    for name in ("rasnet", "segresnet", "vnet", "3dunet"):
        net = build_random_model(name)  # Intentionally do not call .eval() here to test adapter fail-safety
        roi = (20, 20, 20) if name == "vnet" else (32, 32, 32)
        prob = infer_probability(name, net, str(img_path), "cpu", roi_size=roi, sw_batch_size=1, tta=False)
        assert prob.shape == (17, 18, 19)
        assert prob.dtype == np.float32
        assert np.isfinite(prob).all() and np.all((prob >= 0.0) & (prob <= 1.0))

        out_mask_path = tmp_path / f"{name}_mask.nii.gz"
        export_mask_like((prob >= 0.5).astype(np.uint8), str(img_path), str(out_mask_path))
        check_geometry(image_geometry(img), image_geometry(sitk.ReadImage(str(out_mask_path))))

    # 2. Explicit test double for nnUNetPredictor verifying native Z,Y,X array & spacing contract
    class FakeNNUNetPredictor:
        def __init__(self):
            self.use_mirroring = False
            self.calls = []

        def predict_single_npy_array(self, input_image, image_properties, segmentation_previous_stage=None, output_file_truncated=None, save_or_return_probabilities=False):
            self.calls.append((input_image.shape, dict(image_properties), self.use_mirroring, save_or_return_probabilities))
            assert input_image.shape == (1, 17, 18, 19)
            assert np.allclose(image_properties["spacing"], (1.2, 0.8, 0.7))
            fg = np.full((17, 18, 19), 0.75 if self.use_mirroring else 0.25, dtype=np.float32)
            probs = np.stack([1.0 - fg, fg], axis=0)
            seg = (fg >= 0.5).astype(np.uint8)
            return (seg, probs) if save_or_return_probabilities else seg

    fake_pred = FakeNNUNetPredictor()
    prob_a = infer_nnunet_probability(fake_pred, str(img_path), tta=False)
    prob_b = infer_nnunet_probability(fake_pred, str(img_path), tta=True)
    assert prob_a.shape == (17, 18, 19) and np.allclose(prob_a, 0.25)
    assert prob_b.shape == (17, 18, 19) and np.allclose(prob_b, 0.75)

    # Verify build_nnunet_predictor fails loudly when trained model folder or plans/checkpoint are missing
    with pytest.raises(FileNotFoundError):
        build_nnunet_predictor(str(tmp_path / "nonexistent_nnunet_folder"), device="cpu")

