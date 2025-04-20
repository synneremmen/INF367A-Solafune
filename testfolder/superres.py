import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.transform import Affine
from rasterio.enums import Resampling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Model paths ======
MODEL_20M = "./testfolder/L2A20M.pt"
MODEL_60M = "./testfolder/L2A60M.pt"
SCALE = 2000

# ====== Band indices (0-based) ======
bands_10m = [1, 2, 3, 7]       # B2, B3, B4, B8
bands_20m = [4, 5, 6, 8, 10, 11]  # B5, B6, B7, B8A, B11, B12
bands_60m = [0, 9]             # B1, B9

# ====== Load full .tif ======
with rasterio.open("./data/train_images/train_0.tif") as src:
    profile = src.profile
    bands = src.read()  # shape: [12, H, W]
    crs = src.crs
    transform = src.transform

# ====== Separate bands ======
img_10m = bands[bands_10m]    # [4, H, W]
img_20m = bands[bands_20m]    # [6, H//2, W//2]
img_60m = bands[bands_60m]    # [2, H//6, W//6]

# ====== Upsample to 10m ======
def upsample(img, scale):
    img_torch = torch.from_numpy(img).unsqueeze(0).float() / SCALE  # [1, C, H, W]
    return F.interpolate(img_torch, scale_factor=scale, mode='bilinear', align_corners=False)

img_10m_tensor = torch.from_numpy(img_10m).unsqueeze(0).float().to(device) / SCALE
img_20m_tensor = torch.from_numpy(img_20m).unsqueeze(0).float().to(device) / SCALE
img_60m_tensor = torch.from_numpy(img_60m).unsqueeze(0).float().to(device) / SCALE


# ====== Load models ======
model_20m = torch.jit.load(MODEL_20M, map_location=device).eval()
model_60m = torch.jit.load(MODEL_60M, map_location=device).eval()

# ====== Run inference ======
with torch.no_grad():
    sr20 = model_20m(img_10m_tensor, img_20m_tensor).squeeze(0).cpu() * SCALE
    sr60 = model_60m(img_10m_tensor, img_20m_tensor, img_60m_tensor).squeeze(0).cpu() * SCALE

sr20 = sr20.cpu()
sr60 = sr60.cpu()
img_10m_tensor = img_10m_tensor.cpu()

# ====== Combine output: [B1–B12] all at 10m ======
# Allocate in float32 or int32
final_stack = torch.zeros((12, sr20.shape[1], sr20.shape[2]), dtype=torch.float32)

# Assign bands
final_stack[[0, 9]] = sr60         # B1, B9
final_stack[[1, 2, 3, 7]] = img_10m_tensor[0] * SCALE  # B2, B3, B4, B8
final_stack[[4, 5, 6, 8, 10, 11]] = sr20  # B5, B6, B7, B8A, B11, B12

# Convert to uint16 before saving
final_stack_uint16 = torch.nan_to_num(final_stack, nan=0.0, posinf=65535, neginf=0.0).clamp(0, 65535).to(torch.uint16)

# ====== Save output ======
output_profile = profile.copy()
output_profile.update({
    "count": 12,
    "height": final_stack.shape[1],
    "width": final_stack.shape[2],
    "dtype": 'uint16',
    "transform": transform,  # Should match 10m resolution
    "crs": crs
})

with rasterio.open("superresolved_10m.tif", "w", **output_profile) as dst:
    dst.write(final_stack_uint16.numpy())
