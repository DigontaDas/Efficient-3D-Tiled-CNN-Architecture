"""Native nnU-Net V2 adapter using nnUNetPredictor and SimpleITK Z,Y,X geometry."""
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from benchmark.metrics import check_geometry
from benchmark.preflight import image_geometry


def ensure_nnunet_trainer_registered():
    """Ensure nnUNetTrainer_200epochs is available in nnunetv2's trainer registry."""
    import torch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    import nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs as m

    if not hasattr(m, "nnUNetTrainer_200epochs"):
        class nnUNetTrainer_200epochs(nnUNetTrainer):
            def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                         device: torch.device = torch.device("cuda")):
                super().__init__(plans, configuration, fold, dataset_json, device)
                self.num_epochs = 200

        setattr(m, "nnUNetTrainer_200epochs", nnUNetTrainer_200epochs)


def build_nnunet_predictor(
    model_folder: str,
    fold: int = 0,
    checkpoint_name: str = "checkpoint_best.pth",
    device: str = "cuda",
    use_mirroring: bool = False,
):
    """Initialize native nnUNetPredictor from a verified trained-model folder."""
    folder = Path(model_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing nnU-Net trained model folder: {folder}")
    plans_path = folder / "plans.json"
    dataset_path = folder / "dataset.json"
    ckpt_path = folder / f"fold_{fold}" / checkpoint_name
    for req in (plans_path, dataset_path, ckpt_path):
        if not req.is_file():
            raise FileNotFoundError(f"Missing required nnU-Net artifact: {req}")
    if "best" not in checkpoint_name.lower():
        raise ValueError(f"Refusing non-best nnU-Net checkpoint: {checkpoint_name}")

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    ensure_nnunet_trainer_registered()
    dev = torch.device(device)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=bool(use_mirroring),
        perform_everything_on_device=(dev.type == "cuda"),
        device=dev,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(folder),
        use_folds=(int(fold),),
        checkpoint_name=checkpoint_name,
    )
    return predictor


def infer_nnunet_probability(predictor, image_path: str, tta: bool = False) -> np.ndarray:
    """
    Run native nnUNetPredictor on image_path and return foreground probability in (Z, Y, X) order.
    Uses SimpleITK array (Z, Y, X) and reversed spacing (sz, sy, sx) matching SimpleITKIO.
    """
    if not Path(image_path).is_file():
        raise FileNotFoundError(image_path)
    predictor.use_mirroring = bool(tta)
    img = sitk.ReadImage(str(image_path))
    arr_zyx = sitk.GetArrayFromImage(img).astype(np.float32, copy=False)
    input_image = arr_zyx[None, ...]  # (1, Z, Y, X)
    props = {"spacing": tuple(float(s) for s in reversed(img.GetSpacing()))}

    seg, probs = predictor.predict_single_npy_array(
        input_image,
        props,
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_or_return_probabilities=True,
    )
    fg_prob = np.asarray(probs[1], dtype=np.float32)
    if fg_prob.shape != arr_zyx.shape:
        raise RuntimeError(f"nnU-Net probability shape {fg_prob.shape} != image shape {arr_zyx.shape}")
    return fg_prob


def export_mask_like(mask_zyx: np.ndarray, reference_image_path: str, output_path: str) -> None:
    """Write a binary (Z, Y, X) mask with exact physical geometry copied from reference_image_path."""
    ref = sitk.ReadImage(str(reference_image_path))
    ref_shape = tuple(reversed(ref.GetSize()))
    mask_arr = np.asarray(mask_zyx, dtype=np.uint8)
    if mask_arr.shape != ref_shape:
        raise ValueError(f"Mask shape {mask_arr.shape} does not match reference Z,Y,X shape {ref_shape}")
    out_img = sitk.GetImageFromArray(mask_arr)
    out_img.CopyInformation(ref)
    check_geometry(image_geometry(ref), image_geometry(out_img))
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.name.endswith(".img.nii.gz"):
        tmp_p = out_p.with_name(out_p.name.replace(".img.nii.gz", "_tmp.nii.gz"))
        sitk.WriteImage(out_img, str(tmp_p))
        tmp_p.replace(out_p)
    else:
        sitk.WriteImage(out_img, str(out_p))
