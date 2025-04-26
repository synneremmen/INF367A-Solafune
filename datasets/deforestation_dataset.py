import os
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from torchvision.transforms import Normalize
import random


Sample = Tuple[str, str]


def build_samples(images_dir: str, masks_dir: str) -> List[Sample]:
    """
    Builds a list of (image_path, mask_path) tuples
    """
    imgs = sorted(glob(os.path.join(images_dir, "*.tif")))
    msks = sorted(glob(os.path.join(masks_dir,  "*.tif")))
    if len(imgs) != len(msks):
        raise ValueError(f"Image/mask count mismatch: {len(imgs)} vs {len(msks)}")
    return list(zip(imgs, msks))


def compute_stats(samples: List[Sample],
                  stats_path: str = "stats.npz"
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads or computes population mean/std over the given samples
    """
    if os.path.exists(stats_path):
        data = np.load(stats_path)
        mean = torch.from_numpy(data["mean"])
        std  = torch.from_numpy(data["std"])
    else:
        tensors = []
        for img_path, _ in samples:
            with rasterio.open(img_path) as src:
                img = src.read().astype(np.float32)  # (C, H, W)
            img = np.nan_to_num(img, 0.0)
            tensors.append(torch.from_numpy(img))
        x = torch.stack(tensors, dim=0)  # [N, C, H, W]

        mean = x.mean(dim=(0, 2, 3))
        std  = x.std(dim=(0, 2, 3), unbiased=False) + 1e-7  # population std
        np.savez(stats_path, mean=mean.numpy(), std=std.numpy())

    return mean, std



class DeforestationDataset(Dataset):
    """
    PyTorch Dataset for deforestation segmentation
    """
    def __init__(self, samples: List[Sample], oba_generator, normalizer: Normalize):
        self.samples   = samples
        self.oba_generator = oba_generator
        self.normalizer = normalizer
        self.p_aug     = 0.5

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)

        if self.oba_generator is not None and random.random() < self.p_aug:
            img, mask = self.oba_generator.generate_augmented_sample()

        img = np.nan_to_num(img, 0.0)
        img_t = torch.from_numpy(img)
        img_t = self.normalizer(img_t)

        mask_t = torch.from_numpy(mask)

        return img_t, mask_t



def build_datasets(images_dir: str,
                   masks_dir:  str,
                   train_ratio: float = 0.8,
                   test_ratio:  float = 0.1,
                   oba_generator: object = None,
                   seed:        int   = 42,
                   batch_size:  int   = 8,
                   num_workers: int   = 4,
                   stats_path:  str   = "stats.npz"
                  ):
    """
    Builds dataloaders for training, validation, and testing
    """
    samples = build_samples(images_dir, masks_dir)
    n       = len(samples)
    n_train = int(n * train_ratio)
    n_test  = int(n * test_ratio)
    n_val   = n - n_train - n_test

    perm   = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    train  = [samples[i] for i in perm[:n_train]]
    val    = [samples[i] for i in perm[n_train:n_train + n_val]]
    test   = [samples[i] for i in perm[n_train + n_val:]]

    mean, std = compute_stats(train, stats_path)
    normalizer = Normalize(mean=mean, std=std)

    train_ds = DeforestationDataset(train, oba_generator, normalizer)
    val_ds   = DeforestationDataset(val, oba_generator,  normalizer)
    test_ds  = DeforestationDataset(test, oba_generator, normalizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
