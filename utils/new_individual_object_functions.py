import os
from matplotlib import pyplot as plt
from torchvision import transforms
from rasterio.transform import Affine
from rasterio.features import rasterize
import numpy as np
import rasterio
from shapely.geometry import Polygon
from sklearn.model_selection import train_test_split
import torch
from utils.normalize import normalize
from utils.augmentation import augment 
import random
from torch.utils.data import TensorDataset
from utils.loading import load_images, load_labels, load_masked_images
from dotenv import load_dotenv

load_dotenv()
IMAGES_PATH = os.getenv("IMAGES_PATH")

class Generator:
    def __init__(self, batch_size, masked_images=None, images=None, subset=False, min_area=1000):
        self.val = False

        # load all masks for images
        if masked_images is None:
            self.masks = load_masked_images(subset=subset)
        else:
            self.masks = masked_images

        self.class_mapping = {'none':0, 'plantation':1, 'logging':2, 'mining':3, 'grassland_shrubland':4}

        self.background_list = []
        self.file_name_list = list(self.masks.keys())
        print(f"Found {len(self.file_name_list)} file names")

        self.base_augm = False

        self.object_augm = True
        self.object_augm_prob = 0.6

        self.extra_background_prob = 0.6
        self.background_augm_prob = 0.6
        
        self.augm_seperately = False
        self.flag_object_augm = False

        self.shadows = False
        self.extra_objects = 0  

        # values for image-mask augmentation
        self.augm_prob = 0.9
        self.geometric_augm_prob = 0.6
        self.color_augm_prob = 0

        # self.stands_id = False # vet ikke hva denne gjør. Hvilke id?
        self.batch_size = batch_size

        self.channels_background = ['Aerosols', 'Blue', 'Green', 'Red', 'Red Edge 1', 'Red Edge 2', 'Red Edge 3', 'NIR', 'Red Edge 4', 'Water vapor', 'SWIR1', 'SWIR2']

        self.extracted_objects = self.extract_objects(images=images, min_area=min_area, subset=subset)
        self.num_of_extracted_objects = len(self.extracted_objects)


    def visualize(self, image, mask=None):
        """
        Visualizes the image and mask.
        Args:
            image (numpy.ndarray): The image to visualize.
            mask (numpy.ndarray): The mask to visualize.
            original_image (numpy.ndarray): The original image to visualize.
            original_mask (numpy.ndarray): The original mask to visualize.
        """
        fontsize = 18

        if mask is None:
            f, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(image)
            
        else:
            f, ax = plt.subplots(2, 1, figsize=(8, 8))

            ax[0, 0].imshow(image)
            ax[0, 0].set_title('Image', fontsize=fontsize)

            ax[1, 0].imshow(mask)
            ax[1, 0].set_title('Mask', fontsize=fontsize)

        plt.show()


    def extract_objects(self, images=None, min_area=100, subset=False): # ikke laste opp alle bildene for å unngå å ta opp så mye plass?
        """
        Creates object-based augmented data by rasterizing the object masks.
        Returns:
            list: A list of tuples containing image names, masks, and object IDs.
        """
        # load labels with masks for objects
        labels = load_labels(subset=subset, object_based_augmentation=True)
        if images is None:
            images = load_images(subset=subset)

        all_object_masks = []

        for image_name, label in labels.items():
            image = images[image_name]["image"]
            profile = images[image_name]["profile"]
            height, width = profile["height"], profile["width"]
            transform = Affine.identity()

            for annotation, obj_id in label:
                class_label = annotation["class"]
                class_value = self.class_mapping[class_label]
                coords = annotation["segmentation"]
                polygon_coords = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                polygon = Polygon(polygon_coords)

                if polygon.area > min_area and polygon.is_valid:
                    mask = rasterize(
                        [(polygon, class_value)],
                        out_shape=(height, width),
                        transform=transform,
                        fill=0,
                    )

                    all_object_masks.append((image_name, image, mask, obj_id))

        print(f"Extracted {len(all_object_masks)} objects from {len(images)} images.")
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
                mask = obj[2]
                break
        else:
            raise ValueError(f"Object ID {object_id} not found.")

        # read image
        image_path = os.path.join(IMAGES_PATH, image_name)
        with rasterio.open(image_path) as src:
            image = src.read()

        # crop the target object
        cropped_object = np.zeros_like(image)
        #mask = np.where(mask > 0, mask, 0)

        for c in range(image.shape[0]):
            cropped_object[c] = np.where(mask > 0, image[c], 0)

        # return target object and mask
        return cropped_object, mask # (12, 1024, 1024), (1024, 1024)
    

    def generate_augmented_sample(self):
        self.flag_object_augm = False
        # -------------------------------------------------------------------
        # find the background
        # -------------------------------------------------------------------
        background, mask, rnd_choice = self._find_background()
        self.visualize(background[2], mask)
        background, mask = self._prepare_image_and_mask(background, mask)

        if random.random() < self.background_augm_prob and self.augm_seperately:
            # by some probability, augment background
            background, mask = self._augment_image_mask(background, mask)
        
        # -------------------------------------------------------------------
        # object-based augmentation
        # -------------------------------------------------------------------
        if self.object_augm:
            print("Object-based augmentation")
            objects_image, objects_mask = self._apply_object_augmentation(rnd_choice, background, mask)
            image, mask = self._insert_objects_to_background(objects_image, objects_mask, background, mask)

            # -------------------------------------------------------------------
            # add shadows
            # -------------------------------------------------------------------
            if self.shadows:
                print("Adding shadows")
                objects_image = self._add_shadows(image, objects_mask)

        # -------------------------------------------------------------------
        # base augmentation (if no object-based augmentation)
        # -------------------------------------------------------------------
        if self.base_augm and not self.flag_object_augm:
            print("Base augmentation")
            image, mask = self._augment_image_mask(background, mask)

        # -------------------------------------------------------------------
        # normalize images
        # -------------------------------------------------------------------
        if not self.base_augm and not self.object_augm:
            raise ValueError("No augmentation applied. Please set base_augm or object_augm to True.")

        return image, mask
    

    def _find_background(self):
        if len(self.background_list) and random.random() < self.extra_background_prob:
            print("I should no be in here...")
            # TODO: not implemented extra backgrounds
            rnd_choice = random.choice(self.background_list)
        else:
            rnd_choice = random.choice(self.file_name_list)
            with rasterio.open(IMAGES_PATH + "/" + rnd_choice) as src:
                background = src.read()
                background = np.nan_to_num(background, nan=0.0)

            mask = self.masks[rnd_choice]["image"][0] # mask has shape (1, 1024, 1024)
        return background, mask, rnd_choice
    
    def _prepare_image_and_mask(self, background, mask):
        background = background.clone() if torch.is_tensor(background) else background.copy()
        mask  = mask.clone() if torch.is_tensor(mask) else mask.copy()
        background = background.numpy() if torch.is_tensor(background) else background
        mask  = mask.numpy() if torch.is_tensor(mask) else mask
        return background, mask

    def _apply_object_augmentation(self, name, background, mask):
        temp_mask = mask.copy()
        temp_image = np.zeros_like(background)
        target_ids = [obj[-1] for obj in self.extracted_objects if obj[0] == name]
        num_of_objects = 0

        if self.extra_objects > 0:
            print("Adding extra objects")
            random_num = random.randint(0, self.extra_objects)
            print(f"Random number of objects: {random_num}")
            for _ in range(random_num):
                intersection = True
                iteration = 0

                while intersection and iteration < 10: # iteration: how mmany attempts to try to add an object
                    iteration += 1

                    extra_obj = random.choice(self.extracted_objects)
                    extra_id = extra_obj[-1]

                    if extra_id in target_ids:
                        # do not duplicate objects
                        continue

                    cropped_obj, cropped_mask = self.crop_target_object(extra_id)

                    if np.sum(cropped_mask * temp_mask) == 0:
                        if random.random() < self.object_augm_prob and self.augm_seperately:
                            # by some probability, augment object
                            cropped_obj, cropped_mask = self._augment_image_mask(cropped_obj, cropped_mask)
                            self.flag_object_augm = True
                            if np.sum(cropped_mask * temp_mask) != 0:
                                # ensure it does not overlap withany objects
                                continue

                        intersection = False
                        temp_mask = np.maximum(cropped_mask, temp_mask)
                        for c in range(temp_image.shape[0]):
                            temp_image[c] = np.where(cropped_mask > 0, cropped_obj[c], temp_image[c])

                        target_ids.append(extra_id)
                        num_of_objects += 1
            print(f"Added {num_of_objects} objects")
            temp_mask = np.where(mask > 0, 0, temp_mask)
        # else:
        #     # should be an object selected, not image. So this should be changed
        #     cropped_obj, cropped_mask = self.crop_target_object(target_ids[0]) 
        return temp_image, temp_mask # returns image and mask of cropped objects

    
    def _add_shadows(self, img, mask):
        if mask.ndim == 3:
            mask = mask[0] # ensure correct dimensions

        shadow_mask = mask.copy()

        shift_x = random.randint(0, 4)
        shift_y = random.randint(0, 4)

        shifted_mask = np.zeros_like(shadow_mask)
        h, w = shadow_mask.shape
        shifted_mask[shift_x:, shift_y:] = shadow_mask[:h - shift_x, :w - shift_y]

        shadow = (shifted_mask > 0) & (mask == 0)
        
        # plt.title("Shadow mask")
        # plt.imshow(shadow)
        # plt.show()

        alpha = random.choice([0.4, 0.5]) #([0.3, 0.4, 0.45])
        for idx, _ in enumerate(self.channels_background):
            img[idx] = np.where(shadow > 0, img[idx] * alpha, img[idx])

        return img # returns objects with shadows

    def _augment_image_mask(self, image, mask):
        augm_image, augm_mask  = augment(image, mask, self.augm_prob, self.color_augm_prob, self.geometric_augm_prob)
        return augm_image, augm_mask
    
    def _insert_objects_to_background(self, objects_image, objects_mask, background, mask):
        mask = np.where(objects_mask > 0, objects_mask, mask)
        for c in range(background.shape[0]):
            background[c] = np.where(objects_mask > 0, objects_image[c], background[c])
        return background, mask
    

# -------------------------------------------------------------------
#  Create OBA dataset
# -------------------------------------------------------------------
def create_OBA_dataset(
        prob_of_OBA=0.5, 
        subset=False, 
        base_augm=True, 
        object_augm=False, 
        extra_background_prob=0,
        background_augm_prob=0,
        augm_seperately=False,
        shadows=False,
        extra_objects=5,
        object_augm_prob=0,
        augm_prob=0.9,
        geometric_augm_prob=0.6,
        color_augm_prob=0,
        batch_size=1,
        min_area=1000,
        ) -> TensorDataset:

    x_train_dict = load_images(subset=subset)
    y_train_dict = load_masked_images(subset=subset)

    generator = Generator(batch_size=batch_size, masked_images=y_train_dict, images=x_train_dict, subset=subset, min_area=min_area)
    
    # set parameters
    generator.base_augm = base_augm
    generator.object_augm = object_augm
    generator.extra_background_prob = extra_background_prob
    generator.background_augm_prob = background_augm_prob
    generator.augm_seperately = augm_seperately
    generator.shadows = shadows
    generator.extra_objects = extra_objects
    generator.object_augm_prob = object_augm_prob
    generator.augm_prob = augm_prob
    generator.geometric_augm_prob = geometric_augm_prob
    generator.color_augm_prob = color_augm_prob
    
    num = 0
    # generate sample by a given probability
    for _ in range(len(x_train_dict.items())):
        if random.random() < prob_of_OBA:
            num += 1
            sample_image, sample_mask = generator.generate_augmented_sample()
            sample_mask = sample_mask[np.newaxis, ...]  # get correct dimensions (should be (1, 1024, 1024))
            x_train_dict.update({f"OBA_{num}": {"image": sample_image}})
            y_train_dict.update({f"OBA_{num}": {"image": sample_mask}})
            generator.visualize(sample_image[2], sample_mask[0])

    x_train = [torch.tensor(each['image']) for each in x_train_dict.values()]
    y_train = [torch.tensor(each['image']) for each in y_train_dict.values()]

    x_train_tensor = torch.stack(x_train, dim=0)  # Shape: [num_samples, 12, 1024, 1024]
    y_train_tensor = torch.stack(y_train, dim=0).squeeze(1).long()   # Shape: [num_samples, 1, 1024, 1024]

    x_train_tensor = torch.nan_to_num(x_train_tensor, nan=0.0)
    x_train_tensor = normalize(x_train_tensor)

    print(f"Created dataset with {num} generated samples and {len(x_train_tensor)-num} original samples.")

    return TensorDataset(x_train_tensor, y_train_tensor)



    # augmentation
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
