import os
from dotenv import load_dotenv
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
import matplotlib.colors as mcolors
import numpy as np
import rasterio

load_dotenv()
class_mapping = {'none':0, 'plantation':1, 'logging':2, 'mining':3, 'grassland_shrubland':4}
LABELS_PATH = os.getenv("LABELS_PATH")
IMAGES_PATH = os.getenv("IMAGES_PATH")
MASKED_IMAGES_PATH = os.getenv("MASKED_IMAGES_PATH")
EVAL_LABELS_PATH = os.getenv("EVAL_LABELS_PATH")
EVAL_IMAGES_PATH = os.getenv("EVAL_IMAGES_PATH")

def get_unique_classes(image_label):
    unique_classes = set()
    for elem in image_label:
        unique_classes.add(elem["class"])
    return unique_classes


def plot_image(filename, num_plots=2, band=5, no_nan=False, labels=None, polygons=True):
    image_label = None

    labels_path = LABELS_PATH
    images_path = IMAGES_PATH
    
    print(f"Labels path: {labels_path}")
    print(f"Images path: {images_path}")

    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.exists(labels_path):
        print(f"Cant find path: {os.getcwd()}")
        raise FileNotFoundError("No labels found.")
    
    with open(labels_path, 'r') as file:
        labels = json.load(file)

    for image in labels["images"]:
        if image["file_name"] == filename:
            print(f"Found {filename}")
            image_label = image["annotations"] 
            break
    
    if image_label:
        if not os.path.exists(images_path):
            raise FileNotFoundError("No images found.")
    
        with rasterio.open(os.path.join(images_path, filename)) as src:
            _, ax = plt.subplots(1, num_plots, figsize=(num_plots*5, 5))

            unique_classes = ', '.join(get_unique_classes(image_label))
            
            if no_nan:
                title = f'{filename} with class(es): {unique_classes} with enhanced NaN values'
            else:
                title = f'{filename} with class(es): {unique_classes}'
            plt.suptitle(title)

            if num_plots == 1:
                ax = [ax]

            for i in range(1,num_plots+1):
                print((band+i) % 12)
                ax[i].set_title(f"Band {(band+i) % 12}")
                val = src.read((band+i) % 12)

                if no_nan:
                    cmap = plt.cm.viridis
                    cmap.set_bad(color='yellow')
                    val = np.ma.masked_invalid(val)
                    cmap.set_bad(color='yellow')
                    ax[i].imshow(val, cmap=cmap)
                
                else:
                    ax[i].imshow(val)
                
                if polygons:
                    for polygon in image_label:
                        class_name = polygon['class']
                        segmentation = polygon['segmentation']
                        
                        coords = [(segmentation[j], segmentation[j+1]) for j in range(0, len(segmentation), 2)]
                        polygon = Polygon(coords, edgecolor='red', lw=2, facecolor="red", alpha=0.3, label=class_name)
                        ax[i].add_patch(polygon)            
        
            plt.show()
    else:
        print(f'No annotations found for {filename}.')

def plot_images(folder, type="train", amount=20, num_plots=2, band=5):
    count = 0
    for filename in os.listdir(folder):
        if count >= amount:
            print("Too many images, ending session...")
            break
        else:
            if filename.endswith('.tif') :
                print(f"Plotting {filename}")
                plot_image(filename, num_plots=num_plots, band=band, type=type) 
                count += 1
            else:
                print(f'{filename} is not a .tif file.')


def plot_masked_image(filename):
    path = f"{MASKED_IMAGES_PATH}/{filename}"
    src = rasterio.open(path)
    mask_data = src.read(1)

    colors = ['gainsboro', 'forestgreen', 'sienna', 'darkred', 'dodgerblue']
    cmap = mcolors.ListedColormap(colors)

    fig, ax = plt.subplots()
    im = ax.imshow(mask_data, cmap=cmap, vmin=0, vmax=len(class_mapping)-1)

    patches = []
    for label, class_val in class_mapping.items():
        patch_color = cmap(class_val)
        patches.append(Patch(color=patch_color, label=label))

    ax.legend(handles=patches, bbox_to_anchor=(1, 1), loc='upper left')
    ax.axis('off')
    ax.set_title("Polygons with classes")
    plt.show()
    src.close()


def plot_predictions(filename, band=1):
    image_label = None
    image_pred = None

    images_path = IMAGES_PATH
    labels_path = LABELS_PATH
    predictions_path = EVAL_LABELS_PATH

    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.exists(labels_path):
        print(f"Cant find path: {os.getcwd()}")
        raise FileNotFoundError("No labels found.")
    
    if not os.path.exists(predictions_path):
        print(f"Cant find path: {os.getcwd()}")
        raise FileNotFoundError("No predictions found.")
    
    with open(labels_path, 'r') as file:
        labels = json.load(file)

    with open(predictions_path, 'r') as file:
        predictions = json.load(file)

    for image in labels["images"]:
        if image["file_name"] == filename:
            print(f"Found ground truth for {filename}")
            image_label = image["annotations"] 
            break

    for image in predictions["images"]:
        if image["file_name"] == filename:
            print(f"Found prediction for {filename}")
            image_pred = image["annotations"] 
            break
    
    if image_label and image_pred:
        if not os.path.exists(images_path):
            raise FileNotFoundError("No images found.")
    
        with rasterio.open(os.path.join(images_path, filename)) as src:
            _, ax = plt.subplots(1, 2, figsize=(15, 5))

            unique_classes = ', '.join(get_unique_classes(image_label))
            
            title = f'{filename} with class(es): {unique_classes}'
            plt.suptitle(title)

            ax[0].set_title("Ground truth")
            ax[1].set_title("Predictions")

            val = src.read(band)
            ax[0].imshow(val)
            ax[1].imshow(val)
        
            for polygon in image_label:
                class_name = polygon['class']
                segmentation = polygon['segmentation']
                
                coords = [(segmentation[j], segmentation[j+1]) for j in range(0, len(segmentation), 2)]
                polygon = Polygon(coords, edgecolor='red', lw=2, facecolor="red", alpha=0.3, label=class_name)
                ax[0].add_patch(polygon)         

            for polygon in image_pred:
                class_name = polygon['class']
                segmentation = polygon['segmentation']
                
                coords = [(segmentation[j], segmentation[j+1]) for j in range(0, len(segmentation), 2)]
                polygon = Polygon(coords, edgecolor='red', lw=2, facecolor="red", alpha=0.3, label=class_name)
                ax[1].add_patch(polygon)         

            plt.show()
    else:
        print(f'No annotations found for {filename}.')


def plot_all_bands(filename):
    image_label = None

    labels_path = LABELS_PATH
    images_path = IMAGES_PATH
    
    print(f"Labels path: {labels_path}")
    print(f"Images path: {images_path}")

    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.exists(labels_path):
        print(f"Cant find path: {os.getcwd()}")
        raise FileNotFoundError("No labels found.")
    
    with open(labels_path, 'r') as file:
        labels = json.load(file)

    for image in labels["images"]:
        if image["file_name"] == filename:
            print(f"Found {filename}")
            image_label = image["annotations"] 
            break
    
    if image_label:
        if not os.path.exists(images_path):
            raise FileNotFoundError("No images found.")
    
        with rasterio.open(os.path.join(images_path, filename)) as src:
            _, ax = plt.subplots(3, 4, figsize=(30, 20))

            unique_classes = ', '.join(get_unique_classes(image_label))
            
            title = f'{filename} with class(es): {unique_classes}'
            plt.suptitle(title)

            for i in range(src.count):
                row, col = divmod(i, 4)
                ax[row, col].set_title(f"Band {i+1}")
                val = src.read(i+1)
                ax[row, col].imshow(val)

            plt.show()
    else:
        print(f'No annotations found for {filename}.')