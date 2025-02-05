import os
from dotenv import load_dotenv
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
import matplotlib.colors as mcolors
import rasterio

load_dotenv()
class_mapping = {'none':0, 'plantation':1, 'logging':2, 'mining':3, 'grassland_shrubland':4}
LABELS_PATH = os.getenv("LABELS_PATH")
IMAGES_PATH = os.getenv("IMAGES_PATH")
MASKED_IMAGES_PATH = os.getenv("MASKED_IMAGES_PATH")

def get_unique_classes(image_label):
    unique_classes = set()
    for elem in image_label:
        unique_classes.add(elem["class"])
    return unique_classes


def plot_image(filename, num_plots, band=5):
    
    with open(LABELS_PATH, 'r') as file:
        labels = json.load(file)

    image_label = None
    for image in labels["images"]:
        if image["file_name"] == filename:
            image_label = image["annotations"]
            break
    
    if image_label:
        with rasterio.open( IMAGES_PATH + filename ) as src:
            _, ax = plt.subplots(1, num_plots, figsize=(15, 15))

            unique_classes = ', '.join(get_unique_classes(image_label))
            plt.title(f'{filename} with class(es): {unique_classes}')

            for i in range(num_plots):
                ax[i].imshow(src.read((band+i*2) % 13))
            
                for polygon in image_label:
                    class_name = polygon['class']
                    segmentation = polygon['segmentation']
                    
                    coords = [(segmentation[j], segmentation[j+1]) for j in range(0, len(segmentation), 2)]
                    polygon = Polygon(coords, edgecolor='red', lw=2, facecolor="red", alpha=0.3, label=class_name)
                    ax[i].add_patch(polygon)
            
            
            plt.show()
    else:
        print(f'No annotations found for {filename}.')


def plot_images(folder, amount=20, num_plots=2, band=5):
    count = 0
    for filename in os.listdir(folder):
        if count >= amount:
            print("Too many images, ending session...")
            break
        else:
            if filename.endswith('.tif') :
                plot_image(filename, num_plots, band) 
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
