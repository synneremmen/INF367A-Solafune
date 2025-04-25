import os
from glob import glob

import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset, random_split
from utils.preprocessing import normalize
from torchvision import transforms

class DeforestationFileDataset(Dataset):
    """
    Lazily loads one image+mask pair per __getitem__ from disk.
    Expects:
      images_dir/      *.tif     (each with shape [12,H,W])
      masks_dir/       *.tif     (each with shape [1,H,W] or [H,W])
    """
    def __init__(self, images_dir: str, masks_dir: str):
        super().__init__()
        self.images = sorted(glob(os.path.join(images_dir, "*.tif")))
        self.masks  = sorted(glob(os.path.join(masks_dir,  "*.tif")))
        assert len(self.images)==len(self.masks), f"Image/Mask count mismatch {len(self.images)} vs {len(self.masks)}"
        self.normalizer =  self._get_normalizer()  # Compute mean and std across all images in the dataset

    def _get_normalizer(self):
        """Compute mean and std across all images in the dataset"""
        
        # First pass: compute sum and squared sum
        sum_tensor = None
        sum_squared_tensor = None
        pixel_count = 0
        
        for img_path in self.images:
            with rasterio.open(img_path) as src:
                img = src.read().astype(np.float32)      # (C, H, W)
                img = np.nan_to_num(img, nan=0.0)
                img_t = torch.from_numpy(img)
                
                if sum_tensor is None:
                    sum_tensor = torch.zeros(img_t.shape[0])
                    sum_squared_tensor = torch.zeros(img_t.shape[0])
                
                # Sum across spatial dimensions (H,W)
                sum_tensor += img_t.sum(dim=(1, 2))
                sum_squared_tensor += (img_t ** 2).sum(dim=(1, 2))
                pixel_count += img_t.shape[1] * img_t.shape[2]

                # Calculate mean and std
        means = sum_tensor / pixel_count
        var = (sum_squared_tensor / pixel_count) - (means ** 2)
        stds = torch.sqrt(var + 1e-7)  # Add epsilon for numerical stability
        
        return transforms.Normalize(means, stds)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # ---- load image ----
        with rasterio.open(self.images[idx]) as src:
            img = src.read().astype(np.float32)      # (C, H, W)
        # ---- load mask ----
        with rasterio.open(self.masks[idx]) as src:
            mask = src.read(1).astype(np.int64)      # (H, W)

        img = np.nan_to_num(img, nan=0.0)

        img_t  = torch.from_numpy(img)              # float32, (C,H,W)
        mask_t = torch.from_numpy(mask)             # int64,   (H,W)

        img_t = self.normalizer(img_t.unsqueeze(0)).squeeze(0)          # your existing normalize()

        return {"image": img_t, "mask": mask_t}


def build_deforestation_datasets(
    images_dir,
    masks_dir,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    Instantiates the full Dataset, then splits into train/val.
    Computes dataset statistics, creates normalizer, and splits into train/val.
    """
    # Compute dataset statistics
    full_ds = DeforestationFileDataset(images_dir, masks_dir)

    # split
    n_total  = len(full_ds)
    n_train  = int(n_total * train_ratio)
    n_val    = n_total - n_train
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)

    return train_ds, val_ds

