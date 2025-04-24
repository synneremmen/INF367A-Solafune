import os
from pathlib import Path
import numpy as np
import rasterio
import torch

SCALE = 2000
bands_10m = [1, 2, 3, 7]       # B2, B3, B4, B8
bands_20m = [4, 5, 6, 8, 10, 11]  # B5, B6, B7, B8A, B11, B12
bands_60m = [0, 9]             # B1, B9

def super_resolver(input_dir, output_dir, model_20m_path, model_60m_path, device):
    # path input and output directories
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # make output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(list(input_dir.glob("*.tif")))

    print("Superresolving images")

    # Load pretrained model weights
    model_20m = torch.jit.load(model_20m_path, map_location=device).eval()
    model_60m = torch.jit.load(model_60m_path, map_location=device).eval()

    # loop through all tif files in the input directory
    for i, tif_path in enumerate(tif_files, start=1):
        print(f"Super resolving {i}/{len(tif_files)}: {tif_path.name}")

        with rasterio.open(tif_path) as src:
            profile = src.profile
            bands = src.read()
            transform = src.transform
            crs = src.crs

        # assign bands to 10m, 20m, and 60m
        img_10m = bands[bands_10m]
        img_20m = bands[bands_20m]
        img_60m = bands[bands_60m]

        # convert to pytorch tensors and normalize and move to device
        img_10m_tensor = torch.from_numpy(img_10m).unsqueeze(0).float().to(device) / SCALE
        img_20m_tensor = torch.from_numpy(img_20m).unsqueeze(0).float().to(device) / SCALE
        img_60m_tensor = torch.from_numpy(img_60m).unsqueeze(0).float().to(device) / SCALE


        # perform superresolution using the models
        with torch.no_grad():
            sr20 = model_20m(img_10m_tensor, img_20m_tensor).squeeze(0).cpu() * SCALE
            sr60 = model_60m(img_10m_tensor, img_20m_tensor, img_60m_tensor).squeeze(0).cpu() * SCALE

        # move the 10m tensor back to CPU for final stacking
        img_10m_tensor = img_10m_tensor.cpu()

        final_stack = torch.zeros((12, sr20.shape[1], sr20.shape[2]), dtype=torch.float32)
        final_stack[[0, 9]] = sr60
        final_stack[[1, 2, 3, 7]] = img_10m_tensor[0] * SCALE
        final_stack[[4, 5, 6, 8, 10, 11]] = sr20

        # convert from float32 to uint16 for GeoTIFF, also handles NaN values
        final_stack_uint16 = torch.nan_to_num(final_stack, nan=0.0, posinf=65535, neginf=0.0).clamp(0, 65535).to(torch.uint16)

        # update the profile for the output GeoTIFF
        profile.update({
            "count": 12,
            "height": final_stack.shape[1],
            "width": final_stack.shape[2],
            "dtype": 'uint16',
            "transform": transform,
            "crs": crs
        })

        # save the superresolved image
        output_path = output_dir / tif_path.name
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(final_stack_uint16.numpy())

    print(f"Finished superresolving {len(tif_files)} images")
