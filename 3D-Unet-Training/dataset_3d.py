import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from scipy.ndimage import zoom

# Allow imports from Phase3_Local_Integration (for dataset_paths)
_PHASE3_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "Phase3_Local_Integration"
)
sys.path.insert(0, _PHASE3_DIR)
import dataset_paths  # provides find_image(case_id) / find_label(case_id)

_SPLITS_FILE = os.path.join(_PHASE3_DIR, "splits_final.json")

# ── HU clipping constants (match RASNet preprocessing) ───────────────────────
HU_MIN: float = -100.0
HU_MAX: float =  800.0


class MedicalImage3DDataset(Dataset):
    """
    Dataset class for 3D medical image segmentation.

    Loads NIfTI files via dataset_paths.find_image / find_label so that
    exactly the same file paths are used as in RASNet training / evaluation.
    Applies HU windowing [-100, 800] before spatial resampling to 128^3.
    """

    def __init__(
        self,
        case_ids: list[int],
        target_size: tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        self.case_ids   = case_ids
        self.target_size = target_size
        self.cache_dir = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\3D-Unet-Training\cache_128"
        os.makedirs(self.cache_dir, exist_ok=True)
        logging.info(f"Dataset initialised with {len(case_ids)} cases. Caching to {self.cache_dir}")

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        case_id = self.case_ids[idx]
        
        img_cache_path = os.path.join(self.cache_dir, f"{case_id}_img.npy")
        lbl_cache_path = os.path.join(self.cache_dir, f"{case_id}_lbl.npy")

        # If cache exists, load directly (takes ~1ms, bypassing CPU resize bottleneck)
        if os.path.exists(img_cache_path) and os.path.exists(lbl_cache_path):
            try:
                image = np.load(img_cache_path)
                label = np.load(lbl_cache_path)
                return torch.from_numpy(image), torch.from_numpy(label)
            except Exception as exc:
                logging.warning(f"Failed to load cache for case {case_id}: {exc}. Recomputing.")

        img_path = dataset_paths.find_image(case_id)
        lbl_path = dataset_paths.find_label(case_id)

        if not img_path or not lbl_path:
            logging.warning(f"Missing files for Case {case_id}. Returning zeros.")
            dummy = torch.zeros(1, *self.target_size)
            return dummy, dummy

        try:
            import nibabel as nib
            image = nib.load(img_path).get_fdata().astype(np.float32)
            label = nib.load(lbl_path).get_fdata().astype(np.float32)

            # ── Binarise label ────────────────────────────────────────────
            label = (label > 0.5).astype(np.float32)

            # ── HU windowing: clip then scale to [0, 1] ───────────────────
            image = np.clip(image, HU_MIN, HU_MAX)
            image = (image - HU_MIN) / (HU_MAX - HU_MIN)   # [0, 1]

            # ── Resize to target spatial dimensions ───────────────────────
            image = self._resize_volume(image, self.target_size)
            label = self._resize_volume(label, self.target_size)

            # ── Add channel dim  (C, D, H, W) ────────────────────────────
            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)

            # Save cache for next epoch
            np.save(img_cache_path, image)
            np.save(lbl_cache_path, label)

            return torch.from_numpy(image), torch.from_numpy(label)

        except Exception as exc:
            logging.error(f"Error loading Case {case_id}: {exc}")
            dummy = torch.zeros(1, *self.target_size)
            return dummy, dummy

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resize_volume(
        volume: np.ndarray,
        target_size: tuple[int, int, int],
    ) -> np.ndarray:
        """Resize 3D volume to *target_size* using trilinear zoom."""
        if volume.shape == target_size:
            return volume
        zoom_factors = [t / s for t, s in zip(target_size, volume.shape)]
        return zoom(volume, zoom_factors, order=1)


# ── Split helpers ─────────────────────────────────────────────────────────────

def load_splits() -> dict[str, list[int]]:
    """
    Load the canonical train / val / test case-ID lists from splits_final.json.
    This is the same JSON used by RASNet, ensuring identical data splits.
    """
    if not os.path.exists(_SPLITS_FILE):
        raise FileNotFoundError(
            f"splits_final.json not found at {_SPLITS_FILE}. "
            "Ensure Phase3_Local_Integration is accessible."
        )
    with open(_SPLITS_FILE, "r") as fh:
        splits = json.load(fh)
    logging.info(
        f"Splits loaded — train: {len(splits['train'])}, "
        f"val: {len(splits.get('val', []))}, "
        f"test: {len(splits['test'])} cases."
    )
    return splits


def create_data_loaders(
    batch_size: int = 2,
    num_workers: int = 4,
    target_size: tuple[int, int, int] = (128, 128, 128),
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders using the canonical splits_final.json split.

    Args:
        batch_size:   Training and validation batch size.
        num_workers:  DataLoader worker processes (use 4 for RTX 4080).
        target_size:  Spatial dimensions each volume is resampled to.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    splits = load_splits()

    train_ids = [int(x) for x in splits["train"]]
    val_ids   = [int(x) for x in splits.get("val", [])]
    test_ids  = [int(x) for x in splits["test"]]

    # If the JSON has no separate val split, carve 10 % off train
    if not val_ids:
        import random
        random.seed(42)
        random.shuffle(train_ids)
        n_val = max(1, len(train_ids) // 10)
        val_ids   = train_ids[:n_val]
        train_ids = train_ids[n_val:]
        logging.warning(
            f"No 'val' key in splits_final.json. "
            f"Using first {n_val} shuffled train cases as validation."
        )

    train_ds = MedicalImage3DDataset(train_ids, target_size)
    val_ds   = MedicalImage3DDataset(val_ids,   target_size)
    test_ds  = MedicalImage3DDataset(test_ids,  target_size)

    _common = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **_common
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **_common
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, drop_last=False, **_common
    )

    logging.info(
        f"DataLoaders ready — train batches: {len(train_loader)}, "
        f"val batches: {len(val_loader)}, test batches: {len(test_loader)}."
    )
    return train_loader, val_loader, test_loader


def get_dataset_info() -> dict:
    """Return split sizes and a sample shape for logging / sanity checks."""
    splits = load_splits()
    img_path = dataset_paths.find_image(int(splits["train"][0]))
    sample_shape = None
    if img_path:
        import nibabel as nib
        sample_shape = nib.load(img_path).shape
    return {
        "n_train": len(splits["train"]),
        "n_val":   len(splits.get("val", [])),
        "n_test":  len(splits["test"]),
        "sample_shape": sample_shape,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    info = get_dataset_info()
    print("Dataset info:", info)
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=2, num_workers=0, target_size=(64, 64, 64)
    )
    for images, labels in train_loader:
        print(f"Batch — images: {images.shape}, labels: {labels.shape}, "
              f"range [{images.min():.3f}, {images.max():.3f}]")
        break
    print("dataset_3d.py smoke-test passed.")
