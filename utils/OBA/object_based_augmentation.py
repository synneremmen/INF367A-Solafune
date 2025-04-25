import os
import numpy as np
import random
from matplotlib import pyplot as plt
import rasterio
from rasterio.transform import Affine
from rasterio.features import rasterize
from shapely.geometry import Polygon
import torch
from torch.utils.data import TensorDataset
from utils.normalize import normalize
from utils.OBA.augmentation import augment
from utils.loading import load_images, load_labels, load_masked_images
from dotenv import load_dotenv

load_dotenv()
IMAGES_PATH = os.getenv("IMAGES_PATH")


class Generator:
    def __init__(
        self, batch_size, masked_images=None, images=None, subset=False, min_area=1000
    ):
        self.val = False

        # load all masks for images
        if masked_images is None:
            self.masks = load_masked_images(subset=subset)
        else:
            self.masks = masked_images

        self.class_mapping = {
            "none": 0,
            "plantation": 1,
            "logging": 2,
            "mining": 3,
            "grassland_shrubland": 4,
        }

        self.background_list = []
        self.file_name_list = list(self.masks.keys())
        print(f"Found {len(self.file_name_list)} file names")

        self.augm = True
        self.object_augm = True
        self.object_augm_prob = 0.6

        self.extra_background_prob = 0.6
        self.background_augm_prob = 0.6

        self.flag_as_augm = False

        self.shadows = False
        self.extra_objects = 3

        # values for image-mask augmentation
        self.augm_prob = 0.9
        self.geometric_augm_prob = 0.5
        self.color_augm_prob = 0

        self.batch_size = batch_size

        self.channels_background = [
            "Aerosols",
            "Blue",
            "Green",
            "Red",
            "Red Edge 1",
            "Red Edge 2",
            "Red Edge 3",
            "NIR",
            "Red Edge 4",
            "Water vapor",
            "SWIR1",
            "SWIR2",
        ]

        self.extracted_objects = self._extract_objects(
            images=images, min_area=min_area, subset=subset
        )
        self.num_of_extracted_objects = len(self.extracted_objects)

    def visualize(self, image, mask=None):
        """
        Visualizes the image and mask.
        Args:
            image (numpy.ndarray): The image to visualize.
            mask (numpy.ndarray, optional): The mask to visualize. Defaults to None.
        """
        fontsize = 18

        if mask is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(image)

        else:
            _, ax = plt.subplots(2, 1, figsize=(8, 8))

            ax[0].imshow(image)
            ax[0].set_title("Image", fontsize=fontsize)

            ax[1].imshow(mask)
            ax[1].set_title("Mask", fontsize=fontsize)

        plt.show()

    def _extract_objects(
        self, images=None, min_area=100, subset=False
    ):  # ikke laste opp alle bildene for å unngå å ta opp så mye plass?
        """
        Creates object-based augmented data by rasterizing the object masks.
        Returns:
            list: A list of tuples containing image names, masks, and object IDs.
        """
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
                polygon_coords = [
                    (coords[i], coords[i + 1]) for i in range(0, len(coords), 2)
                ]
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

    def _crop_target_object(self, object_id: int):
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

        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            raise ValueError("Object mask is empty.")

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1  # +1 because slicing is exclusive

        # Crop the image and mask
        cropped_image = image[:, y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        return cropped_image, cropped_mask

    def _find_background(self):
        """
        Finds a random background image and its corresponding mask.
        Returns:
            tuple: The background image, mask, and the name of the image.
        """
        if len(self.background_list) and random.random() < self.extra_background_prob:
            print("I should no be in here...")
            # TODO: not implemented extra backgrounds
            rnd_choice = random.choice(self.background_list)
        else:
            rnd_choice = random.choice(self.file_name_list)
            with rasterio.open(IMAGES_PATH + "/" + rnd_choice) as src:
                background = src.read()
                background = np.nan_to_num(background, nan=0.0)

            mask = self.masks[rnd_choice]["image"][0]  # mask has shape (1, 1024, 1024)
        return background, mask, rnd_choice

    def _prepare_image_and_mask(self, background, init_mask):
        """
        Prepares the image and mask for augmentation.
        Args:
            background (numpy.ndarray): The background image.
            mask (numpy.ndarray): The mask of the background.
        Returns:
            tuple: The prepared image and mask.
        """
        image = (
            background.clone() if torch.is_tensor(background) else background.copy()
        )
        mask = init_mask.clone() if torch.is_tensor(init_mask) else init_mask.copy()
        background = image.numpy() if torch.is_tensor(image) else image
        mask = mask.numpy() if torch.is_tensor(mask) else mask
        return background, mask

    def _apply_object_augmentation(self, name, background, mask):
        """
        Applies object-based augmentation by adding extra objects to the background.
        Args:
            name (str): The name of the image.
            background (numpy.ndarray): The background image.
            mask (numpy.ndarray): The mask of the background.
        Returns:
            tuple: The augmented image and mask.
        """
        temp_mask = mask.copy()
        temp_image = np.zeros_like(background)
        target_ids = [obj[-1] for obj in self.extracted_objects if obj[0] == name]
        num_of_objects = 0

        if self.extra_objects > 0:
            for _ in range(random.randint(0, self.extra_objects)):
                intersection = True
                iteration = 0

                while (
                    intersection and iteration < 30
                ):  # iteration: how mmany attempts to try to add an object
                    iteration += 1

                    extra_obj = random.choice(self.extracted_objects)
                    extra_id = extra_obj[-1]

                    if extra_id in target_ids:
                        # do not duplicate objects
                        continue

                    cropped_obj, cropped_mask = self._crop_target_object(extra_id)
                    if random.random() < self.object_augm_prob and self.augm:
                        cropped_obj, cropped_mask = self._augment_image_mask(
                            cropped_obj, cropped_mask
                        )

                    # get bounding box of the cropped mask
                    ys, xs = np.where(cropped_mask > 0)
                    if len(xs) == 0 or len(ys) == 0:
                        continue  # Skip empty masks

                    bbox_h, bbox_w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1

                    max_x = background.shape[2] - bbox_w
                    max_y = background.shape[1] - bbox_h
                    if max_x <= 0 or max_y <= 0:
                        continue

                    top = random.randint(0, max_y)
                    left = random.randint(0, max_x)

                    crop_h, crop_w = cropped_mask.shape
                    roi_mask = temp_mask[top : top + crop_h, left : left + crop_w]

                    tight_crop_mask = cropped_mask[ys.min():ys.max()+1, xs.min():xs.max()+1]

                    if np.any((roi_mask > 0) & (tight_crop_mask > 0)):
                        continue

                    for c in range(background.shape[0]):
                        temp_image[c, top : top + crop_h, left : left + crop_w] = (
                            np.where(
                                cropped_mask > 0,
                                cropped_obj[c],
                                temp_image[c, top : top + crop_h, left : left + crop_w],
                            )
                        )

                    temp_mask[top : top + crop_h, left : left + crop_w] = np.where(
                        cropped_mask > 0,
                        cropped_mask,
                        temp_mask[top : top + crop_h, left : left + crop_w],
                    )

                    target_ids.append(extra_id)
                    num_of_objects += 1
                    intersection = False

            temp_mask = np.where(mask > 0, 0, temp_mask)
        return temp_image, temp_mask  # returns image and mask of cropped objects

    def _add_shadows(self, img, mask):
        """
        Adds shadows to the image based on the mask.
        Args:
            img (numpy.ndarray): The image to add shadows to.
            mask (numpy.ndarray): The mask indicating where to add shadows.
        Returns:
            numpy.ndarray: The image with shadows added.
        """
        if mask.ndim == 3:
            mask = mask[0]  # ensure correct dimensions

        shadow_mask = mask.copy()

        shift_x = random.randint(0, 4)
        shift_y = random.randint(0, 4)

        shifted_mask = np.zeros_like(shadow_mask)
        h, w = shadow_mask.shape
        shifted_mask[shift_x:, shift_y:] = shadow_mask[: h - shift_x, : w - shift_y]

        shadow = (shifted_mask > 0) & (mask == 0)

        alpha = random.choice([0.3, 0.4, 0.45])
        for idx, _ in enumerate(self.channels_background):
            img[idx] = np.where(shadow > 0, img[idx] * alpha, img[idx])

        return img  # returns objects with shadows

    def _augment_image_mask(self, image, mask):
        """
        Augments the image and mask using the specified augmentation probabilities.
        Args:
            image (numpy.ndarray): The image to augment.
            mask (numpy.ndarray): The mask to augment.
        Returns:
            tuple: The augmented image and mask.
        """
        augm_image, augm_mask = augment(
            image, mask, self.augm_prob, self.color_augm_prob, self.geometric_augm_prob
        )
        return augm_image, augm_mask

    def _insert_objects_to_background(
        self, objects_image, objects_mask, background, mask
    ):
        """
        Inserts the objects into the background image and mask.
        Args:
            objects_image (numpy.ndarray): The objects to insert.
            objects_mask (numpy.ndarray): The mask of the objects.
            background (numpy.ndarray): The background image.
            mask (numpy.ndarray): The mask of the background.
        Returns:
            tuple: The combined image and mask.
        """
        mask = np.where(objects_mask > 0, objects_mask, mask)
        for c in range(background.shape[0]):
            background[c] = np.where(objects_mask > 0, objects_image[c], background[c])
        return background, mask

    def generate_augmented_sample(self):
        """
        Generates an augmented sample by applying object-based augmentation and background augmentation.
        Returns:
            tuple: The augmented image and mask.
        """
        self.flag_as_augm = False
        # -------------------------------------------------------------------
        # find the background
        # -------------------------------------------------------------------
        background, init_mask, rnd_choice = self._find_background()
        image, mask = self._prepare_image_and_mask(background, init_mask)

        if random.random() < self.background_augm_prob and self.augm:
            # by some probability, augment background
            image, mask = self._augment_image_mask(image, mask)
            self.flag_as_augm = True

        # -------------------------------------------------------------------
        # object-based augmentation
        # -------------------------------------------------------------------
        if self.object_augm and random.random() < self.object_augm_prob:
            objects_image, objects_mask = self._apply_object_augmentation(
                rnd_choice, image, mask
            )
            self.flag_as_augm = True
            image, mask = self._insert_objects_to_background(
                objects_image, objects_mask, image, mask
            )

            # -------------------------------------------------------------------
            # add shadows
            # -------------------------------------------------------------------
            if self.shadows:
                objects_image = self._add_shadows(image, objects_mask)

        # -------------------------------------------------------------------
        # base augmentation (if no object-based augmentation)
        # -------------------------------------------------------------------
        if self.augm and not self.flag_as_augm:
            image, mask = self._augment_image_mask(image, mask)

        # self.visualize(background[5], init_mask)
        # self.visualize(image[5], mask)
        return image, mask


# -------------------------------------------------------------------
#  Create OBA dataset
# -------------------------------------------------------------------
def create_OBA_dataset(
    prob_of_OBA=0.5,
    subset=False,
    augm=True,
    object_augm=False,
    extra_background_prob=0,
    background_augm_prob=0,
    shadows=False,
    extra_objects=3,
    object_augm_prob=0,
    augm_prob=0.8,
    geometric_augm_prob=0.6,
    color_augm_prob=0.6,
    batch_size=10,
    min_area=1000,
) -> TensorDataset:
    """
    Create a dataset with object-based augmentation (OBA) samples.
    Args:
        prob_of_OBA (float): Probability of generating an OBA sample.
        subset (bool): Whether to use a subset of the data.
        augm (bool): Whether to apply augmentation.
        object_augm (bool): Whether to apply object-based augmentation.
        extra_background_prob (float): Probability of using an extra background.
        background_augm_prob (float): Probability of augmenting the background.
        shadows (bool): Whether to add shadows to the objects.
        extra_objects (int): Number of extra objects to add.
        object_augm_prob (float): Probability of augmenting the objects.
        augm_prob (float): Probability of applying augmentation.
        geometric_augm_prob (float): Probability of applying geometric augmentation.
        color_augm_prob (float): Probability of applying color augmentation.
        batch_size (int): Batch size for the dataset.
        min_area (int): Minimum area for the objects to be considered.
    Returns:
        TensorDataset: A dataset containing the augmented samples.
    """

    x_train_dict = load_images(subset=subset)
    y_train_dict = load_masked_images(subset=subset)

    generator = Generator(
        batch_size=batch_size,
        masked_images=y_train_dict,
        images=x_train_dict,
        subset=subset,
        min_area=min_area,
    )

    # set parameters
    generator.augm = augm
    generator.object_augm = object_augm
    generator.extra_background_prob = extra_background_prob
    generator.background_augm_prob = background_augm_prob
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

            sample_image, sample_mask = generator.generate_augmented_sample()
            if sample_image is None:
                # if no augmentation as happend, skip this sample
                continue

            num += 1
            sample_mask = sample_mask[
                np.newaxis, ...
            ]  # get correct dimensions (should be (1, 1024, 1024))
            x_train_dict.update({f"OBA_{num}": {"image": sample_image}})
            y_train_dict.update({f"OBA_{num}": {"image": sample_mask}})

    x_train = [torch.tensor(each["image"]) for each in x_train_dict.values()]
    y_train = [torch.tensor(each["image"]) for each in y_train_dict.values()]

    x_train_tensor = torch.stack(x_train, dim=0)  # Shape: [num_samples, 12, 1024, 1024]
    y_train_tensor = (
        torch.stack(y_train, dim=0).squeeze(1).long()
    )  # Shape: [num_samples, 1, 1024, 1024]

    x_train_tensor = torch.nan_to_num(x_train_tensor, nan=0.0)
    x_train_tensor = normalize(x_train_tensor)

    print(
        f"Created dataset with {num} generated samples and {len(x_train_tensor)-num} original samples."
    )

    return TensorDataset(x_train_tensor, y_train_tensor)