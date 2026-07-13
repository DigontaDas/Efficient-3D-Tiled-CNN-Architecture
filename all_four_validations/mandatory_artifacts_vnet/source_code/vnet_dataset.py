#!/usr/bin/env python3
"""
V-Net Dataset and DataLoader for 3D Medical Image Segmentation
Designed for c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Dataset_Main dataset structure
"""

import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import os
from typing import Tuple, List, Optional, Dict, Any
import glob
import random
from torch.nn.functional import pad


def vnet_collate_fn(batch):
    """
    Custom collate function for V-Net that handles variable-depth volumes
    
    Args:
        batch: List of tuples (image, label) where each has shape (1, H, W, D)
        
    Returns:
        Tuple of (batched_images, batched_labels) with consistent depth
    """
    images, labels = zip(*batch)
    
    # Find maximum depth in the batch (depth is the last dimension)
    max_depth = max(img.shape[-1] for img in images)  # Shape is (1, H, W, D)
    
    # Pad all volumes to max_depth
    padded_images = []
    padded_labels = []
    
    for img, label in zip(images, labels):
        current_depth = img.shape[-1]
        if current_depth < max_depth:
            # Calculate padding for depth dimension (last dimension)
            pad_depth = max_depth - current_depth
            padding = (0, pad_depth)  # Only pad the last dimension (depth)
            
            padded_img = pad(img, padding, mode='constant', value=0)
            padded_label = pad(label, padding, mode='constant', value=0)
        else:
            padded_img = img
            padded_label = label
        
        padded_images.append(padded_img)
        padded_labels.append(padded_label)
    
    # Stack into batches
    batched_images = torch.stack(padded_images, dim=0)  # (B, 1, H, W, D)
    batched_labels = torch.stack(padded_labels, dim=0)  # (B, 1, H, W, D)
    
    return batched_images, batched_labels

class VNetMedicalDataset(Dataset):
    """
    PyTorch Dataset for V-Net 3D medical image segmentation
    
    Dataset Assumptions:
    - Images stored in c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Dataset_Main\images_img\
    - Labels stored in c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Dataset_Main\images_label\
    - File naming: {number}.img.nii.gz and {number}.label.nii.gz
    - 3D volumes with shape (D, H, W) where D varies (206-275 slices)
    - Binary segmentation labels (0, 1)
    - All volumes have consistent in-plane resolution (512×512)
    """
    
    def __init__(
        self,
        data_root: str = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Dataset_Main",
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        cache_data: bool = False,
        preload: bool = False,
        slice_range: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize V-Net Dataset
        
        Args:
            data_root: Root directory containing images_img and images_label
            transform: Optional transform to apply to images
            target_transform: Optional transform to apply to labels
            cache_data: Whether to cache loaded volumes in memory
            preload: Whether to preload all data at initialization
            slice_range: Optional tuple (start, end) to limit depth dimension
        """
        self.data_root = data_root
        self.transform = transform
        self.target_transform = target_transform
        self.cache_data = cache_data
        self.slice_range = slice_range
        
        # Define paths
        self.image_dir = os.path.join(data_root, "images_img")
        self.label_dir = os.path.join(data_root, "images_label")
        
        # Verify directories exist
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")
        
        # Get all image files and extract IDs
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.nii.gz")))
        self.sample_ids = []
        
        for img_file in self.image_files:
            # Extract sample ID from filename (e.g., "1.img.nii.gz" -> "1")
            filename = os.path.basename(img_file)
            sample_id = filename.replace(".img.nii.gz", "")
            
            # Verify corresponding label exists
            label_file = os.path.join(self.label_dir, f"{sample_id}.label.nii.gz")
            if os.path.exists(label_file):
                self.sample_ids.append(sample_id)
            else:
                print(f"Warning: Missing label for sample {sample_id}")
        
        print(f"Found {len(self.sample_ids)} valid image-label pairs")
        
        # Data cache for faster access
        self.data_cache = {} if cache_data else None
        
        # Preload data if requested
        if preload:
            print("Preloading all data...")
            self._preload_data()
    
    def _preload_data(self):
        """Preload all volumes into memory"""
        if self.data_cache is None:
            self.data_cache = {}
        
        for sample_id in self.sample_ids:
            self.data_cache[sample_id] = self._load_volume(sample_id)
        
        print(f"Preloaded {len(self.data_cache)} volumes")
    
    def _load_volume(self, sample_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image and label volumes for a given sample ID
        
        Args:
            sample_id: Sample identifier (e.g., "1", "10", "100")
            
        Returns:
            Tuple of (image_volume, label_volume) as numpy arrays
        """
        # Construct file paths
        img_path = os.path.join(self.image_dir, f"{sample_id}.img.nii.gz")
        label_path = os.path.join(self.label_dir, f"{sample_id}.label.nii.gz")
        
        # Load volumes
        img_nii = nib.load(img_path)
        label_nii = nib.load(label_path)
        
        # Get data as numpy arrays
        img_volume = img_nii.get_fdata()
        label_volume = label_nii.get_fdata()
        
        # Apply slice range if specified
        if self.slice_range is not None:
            start, end = self.slice_range
            img_volume = img_volume[:, :, start:end]
            label_volume = label_volume[:, :, start:end]
        
        return img_volume, label_volume
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, label_tensor) with shapes:
            - image: (C, D, H, W) where C=1 for single channel
            - label: (C, D, H, W) where C=1 for single channel
        """
        sample_id = self.sample_ids[idx]
        
        # Load from cache or disk
        if self.data_cache is not None and sample_id in self.data_cache:
            img_volume, label_volume = self.data_cache[sample_id]
        else:
            img_volume, label_volume = self._load_volume(sample_id)
            
            # Cache if enabled
            if self.data_cache is not None:
                self.data_cache[sample_id] = (img_volume, label_volume)
        
        # Convert to float32 for images and float32 for labels
        img_volume = img_volume.astype(np.float32)
        label_volume = label_volume.astype(np.float32)
        
        # Apply transforms if specified
        if self.transform:
            img_volume = self.transform(img_volume)
        if self.target_transform:
            label_volume = self.target_transform(label_volume)
        
        # Convert to PyTorch tensors and add channel dimension
        # Shape: (D, H, W) -> (C, D, H, W) where C=1
        img_tensor = torch.from_numpy(img_volume).unsqueeze(0)  # (1, D, H, W)
        label_tensor = torch.from_numpy(label_volume).unsqueeze(0)  # (1, D, H, W)
        
        return img_tensor, label_tensor
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get detailed information about a sample
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing sample metadata
        """
        sample_id = self.sample_ids[idx]
        img_path = os.path.join(self.image_dir, f"{sample_id}.img.nii.gz")
        label_path = os.path.join(self.label_dir, f"{sample_id}.label.nii.gz")
        
        # Load header information
        img_nii = nib.load(img_path)
        label_nii = nib.load(label_path)
        
        return {
            "sample_id": sample_id,
            "image_shape": img_nii.shape,
            "label_shape": label_nii.shape,
            "voxel_size": img_nii.header.get_zooms(),
            "data_type": img_nii.get_fdata().dtype,
            "unique_labels": np.unique(label_nii.get_fdata()),
            "image_path": img_path,
            "label_path": label_path
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics
        
        Returns:
            Dictionary containing dataset-wide statistics
        """
        if len(self.sample_ids) == 0:
            return {}
        
        # Sample first few volumes for statistics
        sample_shapes = []
        voxel_sizes = []
        label_distributions = []
        
        for i in range(min(10, len(self.sample_ids))):
            info = self.get_sample_info(i)
            sample_shapes.append(info["image_shape"])
            voxel_sizes.append(info["voxel_size"])
            label_distributions.append(info["unique_labels"])
        
        return {
            "total_samples": len(self.sample_ids),
            "sample_shapes": {
                "min_depth": min(s[2] for s in sample_shapes),
                "max_depth": max(s[2] for s in sample_shapes),
                "height": sample_shapes[0][0],  # Should be consistent
                "width": sample_shapes[0][1]    # Should be consistent
            },
            "voxel_sizes": {
                "min": tuple(min(v[i] for v in voxel_sizes) for i in range(3)),
                "max": tuple(max(v[i] for v in voxel_sizes) for i in range(3)),
                "mean": tuple(np.mean([v[i] for v in voxel_sizes]) for i in range(3))
            },
            "label_values": sorted(set().union(*label_distributions))
        }


def create_vnet_dataloaders(
    data_root: str = r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Dataset_Main",
    batch_size: int = 2,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    cache_data: bool = False,
    random_seed: int = 42,
    target_depth: Optional[int] = None,
    depth_strategy: str = "pad"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders for V-Net training
    
    Args:
        data_root: Root directory of the dataset
        batch_size: Batch size for DataLoaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        cache_data: Whether to cache data in memory
        random_seed: Random seed for reproducible splits
        target_depth: Target depth for all volumes (None for auto-detect)
        depth_strategy: Strategy for handling variable depths ("pad" or "crop")
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Validate split ratios
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    # Create full dataset
    full_dataset = VNetMedicalDataset(
        data_root=data_root,
        cache_data=cache_data
    )
    
    # Get dataset statistics
    stats = full_dataset.get_statistics()
    print("Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Volume shape: {stats['sample_shapes']}")
    print(f"  Voxel size range: {stats['voxel_sizes']['min']} - {stats['voxel_sizes']['max']}")
    print(f"  Label values: {stats['label_values']}")
    
    # Create train/val/test splits
    total_samples = len(full_dataset)
    train_size = int(train_split * total_samples)
    val_size = int(val_split * total_samples)
    test_size = total_samples - train_size - val_size
    
    # Set random seed for reproducible splits
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create DataLoaders with GPU optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches for consistent training
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        collate_fn=vnet_collate_fn  # Custom collate for variable depths
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=vnet_collate_fn  # Custom collate for variable depths
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=vnet_collate_fn  # Custom collate for variable depths
    )
    
    print(f"\nDataLoader Configuration:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")
    print(f"  Batch size: {batch_size}")
    print(f"  GPU optimization: pin_memory={pin_memory}, num_workers={num_workers}")
    
    return train_loader, val_loader, test_loader


# Example usage and testing functions
def test_dataset_and_dataloader():
    """Test the dataset and dataloader functionality"""
    print("🧪 Testing V-Net Dataset and DataLoader")
    print("=" * 50)
    
    try:
        # Create dataset
        dataset = VNetMedicalDataset(
            data_root=r"c:\Thesis_RASNET\Thesis_Trainings\Thesis_Trainings\Dataset_Main",
            cache_data=False
        )
        
        print(f"✅ Dataset created successfully with {len(dataset)} samples")
        
        # Test single sample
        img, label = dataset[0]
        print(f"✅ Sample loaded successfully:")
        print(f"   Image shape: {img.shape} (should be [1, D, H, W])")
        print(f"   Label shape: {label.shape} (should be [1, D, H, W])")
        print(f"   Image dtype: {img.dtype}")
        print(f"   Label dtype: {label.dtype}")
        print(f"   Image device: {img.device}")
        print(f"   Unique label values: {torch.unique(label)}")
        
        # Test sample info
        info = dataset.get_sample_info(0)
        print(f"✅ Sample info: {info['sample_id']}, shape: {info['image_shape']}")
        
        # Test dataloaders
        train_loader, val_loader, test_loader = create_vnet_dataloaders(
            batch_size=2,
            train_split=0.7,
            val_split=0.2,
            test_split=0.1,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"✅ DataLoaders created successfully")
        
        # Test batch loading
        for batch_imgs, batch_labels in train_loader:
            print(f"✅ Batch loaded successfully:")
            print(f"   Batch images shape: {batch_imgs.shape} (should be [B, 1, D, H, W])")
            print(f"   Batch labels shape: {batch_labels.shape} (should be [B, 1, D, H, W])")
            print(f"   Batch device: {batch_imgs.device}")
            break
        
        print("\n🎉 All tests passed! Dataset and DataLoader ready for V-Net training.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Run tests
    test_dataset_and_dataloader()
