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
EVAL_IMAGES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.getenv("EVAL_IMAGES_PATH"))
EVAL_IMAGES_SUBSET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.getenv("EVAL_IMAGES_SUBSET_PATH"))
SR_IMAGES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.getenv("SR_IMAGES_PATH"))

def load_labels(subset=False, object_based_augmentation=False):
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f'File {LABELS_PATH} not found.')
    
    labels_data = dict()

    if subset:
        file_names = os.listdir(IMAGES_SUBSET_PATH)

    with open(LABELS_PATH, 'r') as f:
        id = 0
        labels = json.load(f)
        for image in labels["images"]:
            if subset:
                if image["file_name"] not in file_names:
                    continue

            if object_based_augmentation: 
                # extract all objects
                for annotation in image["annotations"]:
                    if labels_data.get(image["file_name"]) is None:
                        labels_data.update({image["file_name"]: []})
                        
                    labels_data[image["file_name"]].append((annotation, id))
                    id += 1
            else:
                # extract the annotations for the image
                labels_data.update({image["file_name"] : image["annotations"]} )

    print(f"Loaded {len(labels_data)} labels.")
    return labels_data

    
def load_images(subset=False, use_SR=False, type="train"):   
    
    if use_SR:
        path = SR_IMAGES_PATH
        if not os.path.exists(path):
            sys.path.append(os.path.abspath(".."))
            subprocess.run(["python", os.path.join(os.path.dirname(__file__),"..", "SR", "superresolution.py")], check=True)

    elif type == "train":
        if subset:
            path = IMAGES_SUBSET_PATH
            if not os.path.exists(path):
                sys.path.append(os.path.abspath(".."))
                subprocess.run(["python", os.path.join(os.path.dirname(__file__), "create_subsets.py")], check=True)
        else:
            path = IMAGES_PATH 

    elif type == "eval":
        if subset:
            path = EVAL_IMAGES_SUBSET_PATH
            if not os.path.exists(path):
                sys.path.append(os.path.abspath(".."))
                subprocess.run(["python", os.path.join(os.path.dirname(__file__), "create_subsets.py")], check=True)
        else:
            path = EVAL_IMAGES_PATH 

    else:
        raise ValueError("Invalid type. Use 'train' or 'eval' or use_SR=True.")

    if not os.path.exists(path):
        raise FileNotFoundError(f'Folder {path} not found. Please ensure your .env variables are correct.')

    data = dict()

    for file in os.listdir(path):
        if file.endswith('.tif'):
            image_path = os.path.join(path, file)
            with rasterio.open(image_path) as src:
                data.update({ file : {"image": src.read(), "profile": src.profile} })
    print(f"Loaded {len(data)} images.")
    return data
    
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
