import rasterio
import json
from shapely.geometry import Polygon
import os
from dotenv import load_dotenv
from rasterio.features import rasterize
from rasterio.transform import Affine

load_dotenv()
class_mapping = {'none':0, 'plantation':1, 'logging':2, 'mining':3, 'grassland_shrubland':4}
LABELS_PATH = os.getenv("LABELS_PATH")
IMAGES_PATH = os.getenv("IMAGES_PATH")
MASKED_IMAGES_PATH = os.getenv("MASKED_IMAGES_PATH")

os.makedirs(MASKED_IMAGES_PATH, exist_ok=True)

with open(LABELS_PATH, "r") as f:
    labels_loaded = json.load(f)

labels_list = labels_loaded["images"]

labels = {}
for label in labels_list:
    file_name = label["file_name"]
    labels[file_name] = label


for image_file in os.listdir(IMAGES_PATH):
    if image_file.endswith(".tif"):
        image_path = os.path.join(IMAGES_PATH, image_file)
        with rasterio.open(image_path) as src:
            profile = src.profile.copy()
            height, width = src.height, src.width

        label_entry = labels.get(image_file)

        polygons = []
        for ann in label_entry["annotations"]:
            class_label = ann["class"]
            class_value = class_mapping[class_label]
            coords = ann["segmentation"]
            polygon_coords = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            poly = Polygon(polygon_coords)
            polygons.append((poly, class_value))

        transform = Affine.identity()
        mask = rasterize(
            polygons,
            out_shape=(height, width),
            transform=transform,
            fill=0,
        )

        profile.update({
            "count": 1,    # keep only one band
        })


        mask_path = os.path.join(MASKED_IMAGES_PATH, image_file)
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(mask, 1)

        print(f"Saved masked annotation {image_file} to {mask_path}")