import torch
import torch.nn.functional as F
import numpy as np
from shapely.geometry import Polygon
from skimage import measure
import os
import json

class_names = ['plantation', 'logging', 'mining', 'grassland_shrubland']

def outputs_to_polygons(outputs, num_channels=5, min_area=0, threshold=0.5):
    # Comes in the shape (B, C, H, W)
    outputs = outputs[:, 1:, :, :] # remove first channel (None)
    print(f"Outputs shape: {outputs.shape}")
    polygons = []
    for output in outputs: # Shape (C, H, W)
        num_channels = output.shape[0]
        polygons_by_class = {}
        
        for class_idx in range(1,num_channels):
            class_mask = output[class_idx]
            contours = measure.find_contours(class_mask, threshold)
            class_polygons = []
            for contour in contours:
                if len(contour) < 4:
                    continue
                # Convert (row, col) to (x, y) for Shapely.
                poly = Polygon([(pt[1], pt[0]) for pt in contour])
                if poly.area >= min_area and poly.is_valid:
                    class_polygons.append(poly)
            polygons_by_class[class_idx] = class_polygons
        polygons.append(polygons_by_class)
        
    return polygons

def labels_to_polygons(labels, num_channels=5, threshold=0.5):
    # Comes in the shape (B, H, W)
    polygons = []
    for label in labels: # Shape (H, W)
        polygons_by_class = {}

        for class_idx in range(1, num_channels):  # class indices start from 1, 0 is None
            mask = (label == class_idx).astype(np.uint8)
            contours = measure.find_contours(mask, threshold)  # 0.5 threshold for contour detection
            class_polygons = []
            for contour in contours:
                if len(contour) < 4:
                    continue  # Ignore very small contours
                poly = Polygon([(pt[1], pt[0]) for pt in contour])
                if poly.is_valid:
                    class_polygons.append(poly)
            polygons_by_class[class_idx] = class_polygons
        polygons.append(polygons_by_class)

    return polygons


def polygons_to_json(polygons_by_class, file_name="train"):

    json_dict = {
        "images": [
        ]
    }
    
    for idx, output in enumerate(polygons_by_class):
        json_dict["images"].append({
            "file_name": file_name + f"_{idx}.tif",
            "annotations": []
        })
        for class_idx, polygons in output.items():
            for poly in polygons:
                # Get the exterior coordinates and flatten them.
                coords = list(poly.exterior.coords)
                # Flatten the list of tuples into a single list of numbers.
                segmentation = [int(coordinate) for point in coords for coordinate in point]
                annotation = {
                    "class": class_names[class_idx-1],
                    "segmentation": segmentation
                }
                json_dict["images"][idx]["annotations"].append(annotation)
    
    return json_dict


def save_json_to_folder(json_data, folder_path, filename="output.json"):
    """
    Save JSON data to a specified folder.
    
    Args:
        json_data (dict): The JSON data to save.
        folder_path (str): The folder where the file should be saved.
        filename (str): The name of the JSON file.
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Create the full file path
    file_path = os.path.join(folder_path, filename)
    
    # Write the JSON file
    with open(file_path, "w") as f:
        json.dump(json_data, f, indent=4)
    
    print(f"JSON saved to {file_path}")
