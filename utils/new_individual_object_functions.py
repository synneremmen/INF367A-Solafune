import os
from typing import Tuple, Dict
from rasterio.transform import Affine
from rasterio.features import rasterize
import numpy as np
import rasterio
from shapely.geometry import Polygon
import cv2
from rasterio.plot import reshape_as_image
#from rasterio.mask import mask
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from utils.preprocessing import get_processed_data
from utils.augmentation import augment 
import random
from torch.utils.data import TensorDataset
from utils.loading import load_images, load_labels
from dotenv import load_dotenv

load_dotenv()
IMAGES_PATH = os.getenv("IMAGES_PATH")

class Generator:
    def __init__(self, batch_size, images=None, masks=None):
        self.val = False

        self.images = images
        self.masks = masks
        self.augm = False
        self.color_augm = False
        self.color_augm_prob = 0
        self.object_augm = False
        self.object_augm_prob = 0.6
        self.extra_background_prob = 0.6
        self.shadows = False
        self.extra_objects = 0  
        # self.stands_id = False # vet ikke hva denne gjør. Hvilke id?
        self.batch_size = batch_size

        self.channels_background = ['Aerosols', 'Blue', 'Green', 'Red', 'Red Edge 1', 'Red Edge 2', 'Red Edge 3', 'NIR', 'Red Edge 4', 'Water vapor', 'SWIR1', 'SWIR2']

        self.extracted_objects = self.create_OBA_masked_data()
        self.num_of_extracted_objects = len(self.extracted_objects)

    def load_dataset(self, subset=True) -> TensorDataset:
        """
        Loads the dataset and returns a TensorDataset object.
        Args:
            subset (bool): If True, loads a subset of the dataset.
        Returns:
            TensorDataset: A dataset containing the images and masks.
        """
        return get_processed_data(subset=subset)


    def create_OBA_masked_data(self): # ikke laste opp alle bildene for å unngå å ta opp så mye plass?
        """
        Creates object-based augmented data by rasterizing the object masks.
        Returns:
            list: A list of tuples containing image names, masks, and object IDs.
        """
        # TODO: importer class_mapping (?)
        class_mapping = {'none':0, 'plantation':1, 'logging':2, 'mining':3, 'grassland_shrubland':4}

        labels = load_labels(object_based_augmentation=True)
        images = load_images()

        all_object_masks = []

        for image_name, label in labels.items():
            profile = images[image_name]["profile"]
            height, width = profile["height"], profile["width"]
            transform = Affine.identity()

            for annotation, obj_id in label:
                class_label = annotation["class"]
                class_value = class_mapping[class_label]
                coords = annotation["segmentation"]
                polygon_coords = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                poly = Polygon(polygon_coords)

                mask = rasterize(
                    [(poly, class_value)],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                )

                all_object_masks.append((image_name, mask, obj_id))
        return all_object_masks


    def split_dataset(self, images, masks, test_size=0.2) -> list:
        """
        Splits the dataset into training and testing sets.
        Args:
            images (list): List of images.
            masks (list): List of masks.
            test_size (float): Proportion of the dataset to include in the test split.
        Returns:
            list: A list containing the training and testing datasets.
        """
        return train_test_split(images, masks, test_size=test_size, random_state=42)


    def add_shadows(self, img, mask):
        """
        Adds shadows to the image based on the mask.
        The shadows are created by shifting the mask and applying a random alpha value.

        Args:
            img (numpy.ndarray): The input image.
            mask (numpy.ndarray): The mask used to create shadows.
        Returns:
            numpy.ndarray: The image with shadows added.
        """

        mask_shift = mask[:, :, 0] * 1
        shift_x = random.randint(0,6) 
        shift_y = random.randint(0,6) # de brukte: 4

        for i_sh in range(1, shift_x):
            mask_shift[:-i_sh, :-i_sh - shift_y] += mask[i_sh:, i_sh + shift_y:, 0]

        shadow = (mask_shift > 0) * (mask[:, :, 0] == 0)
        alpha = random.choice([0.4, .3, .45]) # endre?

        for i, channel in enumerate(self.channels_background):
            if channel in ['Blue', 'Green', 'Red']:  # Only apply to visible channels, hvis jeg ikke ønsker, fjern denne if
                img[:, :, i] = img[:, :, i] * (shadow == 0) + (alpha * (shadow > 0) * img[:, :, i])

        return img
        

    def augment_background_object(self, background, mask):
        """
        Augments the background and object mask using the augment function.
        Args:
            background (numpy.ndarray): The background image.
            mask (numpy.ndarray): The object mask.
        Returns:
            tuple: The augmented background and mask.
        """
        # Optional per-channel transforms (e.g., histogram stretch for NIR)
        augm_background, augm_mask  = augment(background, mask, self.color_augm_prob)
        return augm_background, augm_mask #.astype(np.uint8) # trengs dette? Kun med cv2?
    

    def crop_target_object(self, object_id: int):
        """
        Crops the target object from the image and mask based on the object ID.
        Args:
            object_id (int): The ID of the target object.
        Returns:
            tuple: The cropped image and mask of the target object.
        """
        # loacalize target object given objet id
        for obj in self.extracted_objects:
            if obj[-1] == object_id:
                image_name = obj[0]
                mask = obj[1]
                break
        else:
            raise ValueError(f"Object ID {object_id} not found.")

        # read image
        image_path = os.path.join(IMAGES_PATH, image_name)
        with rasterio.open(image_path) as src:
            image = src.read()

        # crop the target object
        image = image.transpose(1, 2, 0).astype(np.uint16) 
        mask = np.array(mask, dtype=np.uint8)
        cropped_object = np.where(mask[..., None] != 0, image, 0)

        # return target object and mask
        return cropped_object.transpose(2, 0, 1).astype(np.float32), mask


    def get_img_mask(self, image, mask, background):

        # crop target object
        if self.object_augm: #and random.random() < self.object_augm_prob:
            target_key = random.choice(list(self.extracted_objects))
            image, mask = self.crop_target_object(target_key[-1])
            print("Cropped target object")
            print(image.shape)
            print(mask.shape)

            if self.extra_objects:
                for _ in range(random.randint(0, self.extra_objects)):
                    intersection = True
                    iteration = 0

                    while intersection and iteration < 10:
                        iteration += 1

                        extra_key = random.choice(list(self.extracted_objects))
                        if extra_key == target_key:
                            continue  # avoid adding the same object again

                        extra_image, extra_mask = self.crop_target_object(extra_key[-1])

                        if np.sum(extra_mask * mask) == 0:
                            # no intersection
                            intersection = False
                            print("No intersection")
                            mask += extra_mask

                            # blend new object into all channels
                            for c in range(image.shape[-1]):
                                image[:, :, c] = np.where(extra_mask > 0,
                                                        extra_image[:, :, c],
                                                        image[:, :, c])
        return image, mask

        # augment background and mask
        if self.augm:
            """
            def augment(image, mask, color_augm_prob):
                # Assuming image is (H, W, C)
                visible_indices = [1, 2, 3]  # Assuming ['Blue', 'Green', 'Red'] are at index 1-3
                if random.random() < color_augm_prob:
                    for i in visible_indices:
                        alpha = random.uniform(0.9, 1.1)  # contrast
                        beta = random.randint(-10, 10)    # brightness
                        image[:, :, i] = np.clip(alpha * image[:, :, i] + beta, 0, 65535)
                # Apply flipping or other geometric aug here
                if random.random() > 0.5:
                    image = np.flip(image, axis=1)
                    mask = np.flip(mask, axis=1)
                return image, mask
            """
            background, mask = self.augment_background_object(background, mask)

        # add shadows
        if self.shadows:
            self.add_shadows(image, mask)

        