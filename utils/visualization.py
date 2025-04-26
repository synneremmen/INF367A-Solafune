import math
import os
from dotenv import load_dotenv
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
import matplotlib.colors as mcolors
import numpy as np
import rasterio

# from matplotlib import rcParams
# rcParams.update({
#     'font.size': 14,
#     'axes.titlesize': 16,
#     'axes.labelsize': 14,
#     'xtick.labelsize': 12,
#     'ytick.labelsize': 12,
#     'legend.fontsize': 12,
#     'figure.dpi': 150,
#     'axes.edgecolor': 'black',
#     'axes.linewidth': 1,
#     'axes.grid': False,
# })

class_mapping = {'none':0, 'plantation':1, 'logging':2, 'mining':3, 'grassland_shrubland':4}

# Load environment variables
load_dotenv()
LABELS_PATH = os.getenv("LABELS_PATH")
IMAGES_PATH = os.getenv("IMAGES_PATH")
MASKED_IMAGES_PATH = os.getenv("MASKED_IMAGES_PATH")
PREDICTIONS_PATH = os.getenv("PREDICTIONS_PATH")
EVAL_IMAGES_PATH = os.getenv("EVAL_IMAGES_PATH")

# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------

def get_json(path):
    """
    Load a JSON file from the specified path.
    Args:
        path: The path to the JSON file.
    Returns:
        The loaded JSON data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, 'r') as file:
        return json.load(file)


def find_annotations(labels_json, filename):
    """
    Find annotations for a specific image in the labels JSON.
    Args:
        labels_json: The labels JSON data.
        filename: The name of the image file.
    Returns:
        The annotations for the specified image, or None if not found.
    """
    for image in labels_json.get("images", []): # for image in labels["images"]:
        if image["file_name"] == filename:
            return image["annotations"]
    return None


def get_unique_classes(image_label):
    """
    Get unique classes from the image label.
    Args:
        image_label: The image label containing annotations.
    Returns:
        A set of unique classes.
    """
    return set(elem["class"] for elem in image_label)


def read_band(src, band, no_nan=False):
    """
    Read a specific band from the raster file.
    Args:
        src: The rasterio dataset.
        band: The band number to read.
        no_nan: Whether to mask NaN values in the image.
    Returns:
        The band data as a numpy array.
    """
    val = src.read(band)
    if no_nan:
        val = np.ma.masked_invalid(val)
    return val


def draw_polygons(ax, annotations, alpha=0.3, color='red'):
    """
    Draw polygons on the given axes.
    Args:
        ax: The axes to draw on.
        annotations: The annotations containing polygon data.
        alpha: The transparency of the polygons.
        color: The color of the polygons.
    """
    for poly_data in annotations:
        coords = [(poly_data['segmentation'][i], poly_data['segmentation'][i+1]) 
                  for i in range(0, len(poly_data['segmentation']), 2)]
        polygon = Polygon(coords, edgecolor=color, lw=2, facecolor=color, alpha=alpha, label=poly_data['class'])
        ax.add_patch(polygon)

def get_subplot_grid(num_plots, max_cols=4):
    cols = min(num_plots, max_cols)
    rows = math.ceil(num_plots / cols)
    return rows, cols


# ---------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------

def plot_image(filename: str, num_plots: int = 2, band: int = 5, no_nan: bool = False, polygons: bool = True):
    """
    Plot a single image with or without annotations.
    Args:
        filename (str): The name of the image file to plot.
        num_plots (int): The number of plots to display for each image.
        band (int): The band number to plot.
        no_nan (bool): Whether to mask NaN values in the image.
        polygons (bool): Whether to draw polygons on the image.
    """
    os.chdir(os.path.join(os.path.dirname(__file__), '..')) # trengs denne?

    labels = get_json(LABELS_PATH)
    annotations = find_annotations(labels, filename)

    if not annotations:
        print(f"No annotations found for {filename}")
        return
    
    with rasterio.open(os.path.join(IMAGES_PATH, filename)) as src:
        rows, cols = get_subplot_grid(num_plots)
        fig, ax = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        ax = ax.flatten() if num_plots > 1 else [ax]

        unique = ', '.join(get_unique_classes(annotations))
        fig.suptitle(
            f"{filename} | Class(es): {unique}" + (" | NaNs masked" if no_nan else ""),
            fontsize=18, fontweight='bold'
        )

        for i in range(num_plots):
            band_idx = ((band - 1 + i) % src.count) + 1
            val = read_band(src, band_idx, no_nan)
            cmap = plt.cm.viridis if no_nan else None

            if no_nan and cmap:
                cmap.set_bad(color='magenta')

            ax[i].imshow(val, cmap=cmap)
            ax[i].set_title(f"Band {band_idx}")
            ax[i].axis('off') # Hides axes

            if polygons:
                draw_polygons(ax[i], annotations)
            
        plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
        plt.show()
    

def plot_images(folder: str, type: str = "train", amount: int = 10, num_plots: int = 2, band: int = 5):
    """
    Plot images from a specified folder.
    Args:
        folder (str): The folder containing the images.
        type (str): The type of images to plot (train or eval).
        amount (int): The maximum number of images to plot.
        num_plots (int): The number of plots to display for each image.
        band (int): The band number to plot.
    """
    os.chdir(os.path.join(os.path.dirname(__file__), '..')) # trengs denne?
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


def plot_masked_image(filename: str):
    """
    Plot a the mask for an image with class mapping.
    Args:
        filename (str): The name of the image file to plot.
    """
    os.chdir(os.path.join(os.path.dirname(__file__), '..')) # trengs denne?
    path = os.path.join(MASKED_IMAGES_PATH, filename)
    with rasterio.open(path) as src:
        mask_data = src.read(1)

        colors = ['gainsboro', 'forestgreen', 'sienna', 'darkred', 'dodgerblue']
        cmap = mcolors.ListedColormap(colors)

        _, ax = plt.subplots()
        ax.imshow(mask_data, cmap=cmap, vmin=0, vmax=len(class_mapping)-1)
        ax.axis('off')
        ax.set_title("Polygons with classes")

        patches = [Patch(color=cmap(val), label=label) for label, val in class_mapping.items()]
        ax.legend(handles=patches, bbox_to_anchor=(1, 1), loc='upper left')
        plt.show()


def plot_masked_prediction(filename: str, path: str):
    """
    Plot predictions for a given image with masked annotations.
    Args:
        filename (str): The name of the image file to plot.
        path (str): The path to the predictions JSON file.
    """
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))  # trengs denne?
    labels = get_json(LABELS_PATH)
    predictions = get_json(os.path.join(PREDICTIONS_PATH, path))
    gt_ann = find_annotations(labels, filename)
    pred_ann = find_annotations(predictions, filename)

    if not (gt_ann and pred_ann):
        print(f"Missing ground truth or prediction for {filename}")
        return

    gt_path = os.path.join(MASKED_IMAGES_PATH, filename)

    with rasterio.open(gt_path) as gt_src:
        gt_mask = gt_src.read(1)

        colors = ['gainsboro', 'forestgreen', 'sienna', 'darkred', 'dodgerblue']
        cmap = mcolors.ListedColormap(colors)

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        # Plot ground truth mask
        ax[0].imshow(gt_mask, cmap=cmap, vmin=0, vmax=len(class_mapping) - 1)
        ax[0].axis('off')
        ax[0].set_title("Ground Truth Mask")

        # Create a blank canvas for predictions
        pred_canvas = np.zeros_like(gt_mask, dtype=np.uint8)

        # Draw polygons for predictions with class-based colors
        for poly_data in pred_ann:
            coords = [(poly_data['segmentation'][i], poly_data['segmentation'][i + 1])
                      for i in range(0, len(poly_data['segmentation']), 2)]
            class_id = class_mapping.get(poly_data['class'], 0)
            color = colors[class_id] if class_id < len(colors) else 'blue'
            polygon = Polygon(coords, edgecolor=color, lw=2, facecolor=color, alpha=1)
            ax[1].add_patch(polygon)

        # Plot prediction mask
        ax[1].imshow(pred_canvas, cmap=cmap, vmin=0, vmax=len(class_mapping) - 1)
        ax[1].axis('off')
        ax[1].set_title("Prediction Mask")

        # Add legend
        patches = [Patch(color=cmap(val), label=label) for label, val in class_mapping.items()]
        ax[1].legend(handles=patches, bbox_to_anchor=(1, 1), loc='upper left')

        plt.suptitle(f"{filename} - Ground Truth vs Prediction")
        plt.show()

def plot_prediction(filename: str, path: str, band: int = 1):
    """
    Plot predictions for a given image with annotations.
    Args:
        filename (str): The name of the image file to plot.
        band (int): The band number to plot.
    """
    os.chdir(os.path.join(os.path.dirname(__file__), '..')) # trengs denne?
    labels = get_json(LABELS_PATH)
    predictions = get_json(os.path.join(PREDICTIONS_PATH, path))
    gt_ann = find_annotations(labels, filename)
    pred_ann = find_annotations(predictions, filename)

    if not (gt_ann and pred_ann):
        print(f"Missing ground truth or prediction for {filename}")
        return

    with rasterio.open(os.path.join(IMAGES_PATH, filename)) as src:
        _, ax = plt.subplots(1, 2, figsize=(15, 5))
        val = src.read(band)

        ax[0].imshow(val)
        ax[1].imshow(val)

        ax[0].set_title("Ground truth")
        ax[1].set_title("Predictions")

        draw_polygons(ax[0], gt_ann)
        draw_polygons(ax[1], pred_ann)

        plt.suptitle(f"{filename} with class(es): {', '.join(get_unique_classes(gt_ann))}")
        plt.show()