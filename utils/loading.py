import os
import sys
from dotenv import load_dotenv
import json
import rasterio
import subprocess

load_dotenv()
class_mapping = {'none':0, 'plantation':1, 'logging':2, 'mining':3, 'grassland_shrubland':4}
LABELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.getenv("LABELS_PATH")) 
IMAGES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.getenv("IMAGES_PATH")) 
MASKED_IMAGES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.getenv("MASKED_IMAGES_PATH")) 
IMAGES_SUBSET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.getenv("IMAGES_SUBSET_PATH")) 
MASKED_IMAGES_SUBSET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.getenv("MASKED_IMAGES_SUBSET_PATH")) 

def load_labels():
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f'File {LABELS_PATH} not found.')
    
    labels_data = dict()

    with open(LABELS_PATH, 'r') as f:
        labels = json.load(f)
        for image in labels["images"]:
            labels_data.update({image["file_name"] : image["annotations"]} )
    print(f"Loaded {len(labels_data)} labels.")
    return labels_data

    
def load_images(subset=False):
    if not os.path.exists(IMAGES_PATH):
        raise FileNotFoundError(f'Folder {IMAGES_PATH} not found.')
    
    if subset:
        path = IMAGES_SUBSET_PATH
        if not os.path.exists(path):
            sys.path.append(os.path.abspath(".."))
            subprocess.run(["python", os.path.join(os.path.dirname(__file__), "create_subsets.py")], check=True)
    else:
        path = IMAGES_PATH
    
    train_data = dict()

    for file in os.listdir(path):
        if file.endswith('.tif'):
            image_path = os.path.join(path, file)
            with rasterio.open(image_path) as src:
                train_data.update({ file : {"image": src.read(), "profile": src.profile} })
    print(f"Loaded {len(train_data)} images.")
    return train_data
    
def load_masked_images(subset=False):
    if not os.path.exists(MASKED_IMAGES_PATH):
        subprocess.run(["python", os.path.join(os.path.dirname(__file__), "create_masked_data.py")], check=True)
    
    if subset:
        path = MASKED_IMAGES_SUBSET_PATH
        if not os.path.exists(MASKED_IMAGES_SUBSET_PATH):
            subprocess.run(["python", os.path.join(os.path.dirname(__file__), "create_subsets.py")], check=True)
    else:
        path = MASKED_IMAGES_PATH

    masked_data = dict()

    for file in os.listdir(path):
        if file.endswith('.tif'):
            image_path = os.path.join(path, file)
            with rasterio.open(image_path) as src:
                masked_data.update({ file : {"image": src.read(), "profile": src.profile} })
    print(f"Loaded {len(masked_data)} masked images.")
    return masked_data
