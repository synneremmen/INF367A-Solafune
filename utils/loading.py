import os
from dotenv import load_dotenv
import json
import rasterio
import subprocess

load_dotenv()
class_mapping = {'none':0, 'plantation':1, 'logging':2, 'mining':3, 'grassland_shrubland':4}
LABELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.getenv("LABELS_PATH")) 
IMAGES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.getenv("IMAGES_PATH")) 
MASKED_IMAGES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.getenv("MASKED_IMAGES_PATH")) 

def load_labels():
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f'File {LABELS_PATH} not found.')
    
    labels_data = dict()

    with open(LABELS_PATH, 'r') as f:
        labels = json.load(f)
        for image in labels["images"]:
            labels_data.update({image["file_name"] : image["annotations"]} )
    return labels_data

    
def load_images():
    if not os.path.exists(IMAGES_PATH):
        raise FileNotFoundError(f'Folder {IMAGES_PATH} not found.')
    
    train_data = dict()

    for file in os.listdir(IMAGES_PATH):
        if file.endswith('.tif'):
            image_path = os.path.join(IMAGES_PATH, file)
            with rasterio.open(image_path) as src:
                train_data.update({ file : {"image": src.read(), "profile": src.profile.copy(), "height": src.height, "width": src.width} })
    return train_data
    
def load_masked_images():
    if not os.path.exists(MASKED_IMAGES_PATH):
        subprocess.run(["python", "create_masked_data.py"], check=True)

    masked_data = dict()

    for file in os.listdir(MASKED_IMAGES_PATH):
        if file.endswith('.tif'):
            image_path = os.path.join(MASKED_IMAGES_PATH, file)
            with rasterio.open(image_path) as src:
                masked_data.update({ file : {"image": src.read(), "profile": src.profile.copy(), "height": src.height, "width": src.width} })
    return masked_data