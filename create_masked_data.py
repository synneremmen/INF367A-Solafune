import rasterio
from shapely.geometry import Polygon
import os
from dotenv import load_dotenv
from rasterio.features import rasterize
from rasterio.transform import Affine
from utils.loading import load_labels, load_images

load_dotenv()
class_mapping = {'none':0, 'plantation':1, 'logging':2, 'mining':3, 'grassland_shrubland':4}
LABELS_PATH = os.getenv("LABELS_PATH")
IMAGES_PATH = os.getenv("IMAGES_PATH")
MASKED_IMAGES_PATH = os.getenv("MASKED_IMAGES_PATH")

os.makedirs(MASKED_IMAGES_PATH, exist_ok=True)

labels = load_labels()
images = load_images()

for image in images:
    profile = image["profile"]
    height, width = image["height"], image["width"]
    label = labels.get(image)
    polygons = []

    for elem in label:
        class_label = elem["class"]
        class_value = class_mapping[class_label]
        coords = elem["segmentation"]
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

    if not os.path.exists(MASKED_IMAGES_PATH):
            os.makedirs(MASKED_IMAGES_PATH)

    mask_path = os.path.join(MASKED_IMAGES_PATH, image)
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask, 1)
        
    print(f"Saved masked annotation {image} to {mask_path}")