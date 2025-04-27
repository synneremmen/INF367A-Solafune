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
from OBA.augmentation import augment
from utils.loading import load_images, load_labels, load_masked_images
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
IMAGES_PATH = os.getenv("IMAGES_PATH")
OBA_IMAGES_PATH = os.getenv("OBA_IMAGES_PATH")
OBA_MASKED_IMAGES_PATH = os.getenv("OBA_MASKED_IMAGES_PATH")
BACKGROUND_IMAGES_PATH = os.getenv("BACKGROUND_IMAGES_PATH")

# -------------------------------------------------------------------
# Object-Based Augmentation Generator class
# -------------------------------------------------------------------
class Generator:
    """
    Generator class for object-based augmentation of satellite images.

    This class provides functionality to load and preprocess satellite images and their corresponding masks,
    apply various augmentations, and generate augmented samples for training machine learning models.

    Attributes:
        masks (dict): Dictionary of masks for the images.
        images (dict): Dictionary of images.
        class_mapping (dict): Mapping of class names to integer labels.
        background_list (list): List of background images.
        file_name_list (list): List of file names for the images.
        augm (bool): Flag to enable/disable augmentation.
        object_augm (bool): Flag to enable/disable object-based augmentation.
        object_augm_prob (float): Probability of applying object-based augmentation.
        extra_background_prob (float): Probability of using extra background images.
        background_augm_prob (float): Probability of applying background augmentation.
        flag_as_augm (bool): Flag indicating if the sample is augmented.
        shadows (bool): Flag to enable/disable shadow augmentation.
        extra_objects (int): Number of extra objects to add during augmentation.
        augm_prob (float): Probability of applying augmentation.
        geometric_augm_prob (float): Probability of applying geometric augmentation.
        color_augm_prob (float): Probability of applying color augmentation.
        batch_size (int): Batch size for data generation.
        channels_background (list): List of background channels.
        extracted_objects (list): List of extracted objects from the images.
        num_of_extracted_objects (int): Number of extracted objects.

    Methods:
        __init__(batch_size, masked_images=None, images=None, subset=False, min_area=1000):
            Initializes the Generator class with the given parameters.

        visualize(image, mask=None):
            Visualizes an image and its corresponding mask.

        _extract_objects(images=None, min_area=100, subset=False):
            Extracts objects from the images based on the provided masks and annotations.

        _crop_target_object(object_id):
            Crops a specific object from the extracted objects based on its ID.

        _find_background():
            Finds a background image and its corresponding mask.

        _prepare_image_and_mask(background, init_mask):
            Prepares the image and mask for augmentation by cloning or copying them.

        _apply_object_augmentation(name, background, mask):
            Applies object-based augmentation by adding extra objects to the background.

        _add_shadows(img, mask):
            Adds shadows to the image based on the provided mask.

        _augment_image_mask(image, mask):
            Applies augmentation to the image and mask.

        _insert_objects_to_background(objects_image, objects_mask, background, mask):
            Inserts objects into the background image and updates the mask.

        generate_augmented_sample():
            Generates an augmented sample by applying various augmentations to the image and mask.
    """

    def __init__(
        self, batch_size, masked_images=None, images=None, extra_backgrounds=False, subset=False, min_area=100
    ):
        self.masks = (
            load_masked_images(subset=subset)
            if masked_images is None
            else masked_images
        )

        self.images = load_images(subset=subset) if images is None else images

        self.class_mapping = {
            "none": 0,
            "plantation": 1,
            "logging": 2,
            "mining": 3,
            "grassland_shrubland": 4,
        }

        if extra_backgrounds:
            self.background_list = os.listdir(BACKGROUND_IMAGES_PATH)
        else:
            self.background_list = [] 
            
        self.file_name_list = list(self.masks.keys())
        print(f"Found {len(self.file_name_list)} file names")

        # Augmentation settings
        self.augm = True
        self.object_augm = True
        self.object_augm_prob = 0.6
        self.extra_background_prob = 0.6
        self.background_augm_prob = 0.6
        self.flag_as_augm = False
        self.shadows = False
        self.extra_objects = 3
        self.augm_prob = 0.9
        self.geometric_augm_prob = 0.6
        self.color_augm_prob = 0.6

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
            images=self.images, min_area=min_area, subset=subset
        )
        self.num_of_extracted_objects = len(self.extracted_objects)

    def visualize(self, image, mask=None):
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

    def _extract_objects(self, images=None, min_area=100, subset=False):
        labels = load_labels(subset=subset, object_based_augmentation=True)
        all_object_masks = []

        for file_name in images.keys():
            if file_name not in labels:
                continue  # No annotated polygons for this image

            with rasterio.open(os.path.join(IMAGES_PATH, file_name)) as src:
                image = src.read().astype(np.float32)
                height, width = src.height, src.width
                transform = Affine.identity()

            for annotation, obj_id in labels[file_name]:
                class_value = self.class_mapping[annotation["class"]]
                coords = annotation["segmentation"]
                poly_coords = [
                    (coords[i], coords[i + 1]) for i in range(0, len(coords), 2)
                ]
                poly = Polygon(poly_coords)

                if not poly.is_valid or poly.area < min_area:
                    continue

                mask = rasterize(
                    [(poly, class_value)],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                )

                all_object_masks.append((file_name, image, mask, obj_id))

        print(f"Extracted {len(all_object_masks)} objects from {len(file_name)} files.")
        return all_object_masks

    def _crop_target_object(self, object_id: int):
        for obj in self.extracted_objects:
            if obj[-1] == object_id:
                image_name = obj[0]
                mask = obj[2]
                break
        else:
            raise ValueError(f"Object ID {object_id} not found.")

        image_path = os.path.join(IMAGES_PATH, image_name)
        with rasterio.open(image_path) as src:
            image = src.read()

        coords = np.argwhere(mask > 0)
        if coords.size == 0:
            raise ValueError("Object mask is empty.")

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1

        cropped_image = image[:, y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        return cropped_image, cropped_mask
    
    def _tight_crop_object(self, temp_image, temp_mask, object_image, object_mask):
        ys, xs = np.where(object_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None, None

        bbox_h, bbox_w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        max_x = temp_image.shape[2] - bbox_w
        max_y = temp_image.shape[1] - bbox_h

        if max_x <= 0 or max_y <= 0:
            return None, None
        
        top = random.randint(0, max_y)
        left = random.randint(0, max_x)

        crop_h, crop_w = object_mask.shape
        roi_mask = temp_mask[top : top + crop_h, left : left + crop_w]
        tight_crop_mask = object_mask[ys.min():ys.max()+1, xs.min():xs.max()+1]

        if np.any((roi_mask > 0) & (tight_crop_mask > 0)):
            return None, None
        
        for c in range(temp_image.shape[0]):
            temp_image[c, top : top + crop_h, left : left + crop_w] = (
                np.where(
                    object_mask > 0,
                    object_image[c],
                    temp_image[c, top : top + crop_h, left : left + crop_w],
                )
            )

        temp_mask[top : top + crop_h, left : left + crop_w] = np.where(
            object_mask > 0,
            object_mask,
            temp_mask[top : top + crop_h, left : left + crop_w],
        )
        return temp_image, temp_mask

    def _find_background(self):
        if len(self.background_list) and random.random() < self.extra_background_prob:
            rnd_choice = random.choice(self.background_list)
            with rasterio.open(BACKGROUND_IMAGES_PATH + rnd_choice) as src:
                background = src.read()
                background = np.nan_to_num(background, nan=0.0)
                profile = src.profile
        else:
            rnd_choice = random.choice(self.file_name_list)
            with rasterio.open(IMAGES_PATH + rnd_choice) as src:
                background = src.read()
                background = np.nan_to_num(background, nan=0.0)
                profile = src.profile

            mask = self.masks[rnd_choice]["image"][0]
        return background, mask, rnd_choice, profile

    def _prepare_image_and_mask(self, background, init_mask):
        if len(init_mask.shape) == 3: # ensure mask is (1024, 1024) not (1, 1024, 1024)
            mask = init_mask[0]
        else:
            mask = init_mask

        image = background.clone() if torch.is_tensor(background) else background.copy()
        mask = mask.clone() if torch.is_tensor(mask) else mask.copy()
        image = image.numpy() if torch.is_tensor(image) else image
        mask = mask.numpy() if torch.is_tensor(mask) else mask
        return image, mask

    def _apply_object_augmentation(self, name, background, mask, verbose=False):
        temp_mask = mask.copy()
        temp_image = np.zeros_like(background)
        target_ids = [obj[-1] for obj in self.extracted_objects if obj[0] == name]
        num_of_objects = 0

        if self.extra_objects > 0:
            for _ in range(random.randint(0, self.extra_objects)):
                intersection = True
                iteration = 0

                while intersection and iteration < 30:
                    iteration += 1
                    extra_obj = random.choice(self.extracted_objects)
                    extra_id = extra_obj[-1]

                    if extra_id in target_ids:
                        continue

                    cropped_obj, cropped_mask = self._crop_target_object(extra_id)
                    if random.random() < self.object_augm_prob and self.augm:
                        cropped_obj, cropped_mask = self._augment_image_mask(
                            cropped_obj, cropped_mask
                        )

                    tested_temp_image, tested_temp_mask = self._tight_crop_object(temp_image, temp_mask, cropped_obj, cropped_mask)
                    
                    if temp_image is None: # if the object does not fit
                        continue

                    temp_image = tested_temp_image
                    temp_mask = tested_temp_mask

                    target_ids.append(extra_id)
                    num_of_objects += 1
                    intersection = False

                if verbose:
                    print(f"Added {num_of_objects} objects to background {name}")

            temp_mask = np.where(mask > 0, 0, temp_mask)
        return temp_image, temp_mask

    def _add_shadows(self, img, mask):
        if mask.ndim == 3:
            mask = mask[0]

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

        return img

    def _augment_image_mask(self, image, mask):
        augm_image, augm_mask = augment(
            image, mask, self.augm_prob, self.color_augm_prob, self.geometric_augm_prob
        )
        return augm_image, augm_mask

    def _insert_objects_to_background(
        self, objects_image, objects_mask, background, mask
    ):
        mask = np.where(objects_mask > 0, objects_mask, mask)
        for c in range(background.shape[0]):
            background[c] = np.where(objects_mask > 0, objects_image[c], background[c])
        return background, mask

    def generate_augmented_sample(self, verbose=False):
        self.flag_as_augm = False

        background, init_mask, rnd_choice, profile = self._find_background()
        image, mask = self._prepare_image_and_mask(background, init_mask)

        if random.random() < self.background_augm_prob and self.augm:
            image, mask = self._augment_image_mask(image, mask)
            self.flag_as_augm = True

        if self.object_augm and random.random() < self.object_augm_prob:
            objects_image, objects_mask = self._apply_object_augmentation(
                rnd_choice, image, mask, verbose=verbose
            )
            self.flag_as_augm = True
            image, mask = self._insert_objects_to_background(
                objects_image, objects_mask, image, mask
            )

            if self.shadows:
                objects_image = self._add_shadows(image, objects_mask)

        if self.augm and not self.flag_as_augm:
            image, mask = self._augment_image_mask(image, mask)

        mask = mask[np.newaxis, ...]
        return image, mask, profile
    
# -------------------------------------------------------------------
# Function to create a tensor OBA dataset
# -------------------------------------------------------------------
def create_OBA_tensor_dataset(
    prob_of_OBA=0.0,
    subset=False,
    augm=True,
    object_augm=True,
    extra_background_prob=0,
    background_augm_prob=0.6,
    shadows=False,
    extra_objects=3,
    object_augm_prob=0,
    augm_prob=0.8,
    geometric_augm_prob=0.6,
    color_augm_prob=0.6,
    batch_size=10,
    min_area=1000,
    use_SR=False,
    generator=None,
) -> TensorDataset:
    x_train_dict = load_images(subset=subset, use_SR=use_SR)
    y_train_dict = load_masked_images(subset=subset)

    if generator is None:
        generator = Generator(
        batch_size=batch_size,
        masked_images=y_train_dict,
        images=x_train_dict,
        extra_backgrounds=False,
        subset=subset,
        min_area=min_area,
    )

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

    for _ in range(len(x_train_dict.items())):
        if random.random() < prob_of_OBA:
            sample_image, sample_mask, _ = generator.generate_augmented_sample()
            if sample_image is None:
                continue

            num += 1
            sample_mask = sample_mask[np.newaxis, ...]
            x_train_dict.update({f"OBA_{num}": {"image": sample_image}})
            y_train_dict.update({f"OBA_{num}": {"image": sample_mask}})

    x_train = [torch.tensor(each["image"]) for each in x_train_dict.values()]
    y_train = [torch.tensor(each["image"]) for each in y_train_dict.values()]

    x_train_tensor = torch.stack(x_train, dim=0)
    y_train_tensor = torch.stack(y_train, dim=0).squeeze(1).long()

    x_train_tensor = torch.nan_to_num(x_train_tensor, nan=0.0)
    x_train_tensor = normalize(x_train_tensor)

    print(
        f"Created dataset with {num} generated samples and {len(x_train_tensor) - num} original samples."
    )
    return TensorDataset(x_train_tensor, y_train_tensor)

# -------------------------------------------------------------------
# Function to create and save OBA images
# -------------------------------------------------------------------

def create_save_OBA_images(
    prob_of_OBA=0.5,
    subset=False,
    augm=True,
    object_augm=True,
    extra_background_prob=0,
    background_augm_prob=0.6,
    shadows=False,
    extra_objects=3,
    object_augm_prob=0,
    augm_prob=0.8,
    geometric_augm_prob=0.6,
    color_augm_prob=0.6,
    batch_size=10,
    min_area=1000,
    use_SR=False,
    generator=None,
):
    x_train_dict = load_images(subset=subset, use_SR=use_SR)
    y_train_dict = load_masked_images(subset=subset)

    if generator is None:
        generator = Generator(
            batch_size=batch_size,
            masked_images=y_train_dict,
            images=x_train_dict,
            extra_backgrounds=False,
            subset=subset,
            min_area=min_area,
        )

    if not os.path.exists(OBA_IMAGES_PATH):
        os.makedirs(OBA_IMAGES_PATH)
    else:
        for file in os.listdir(OBA_IMAGES_PATH):
            if file.startswith("OBA"):
                os.remove(os.path.join(OBA_IMAGES_PATH, file))

    if not os.path.exists(OBA_MASKED_IMAGES_PATH):
        os.makedirs(OBA_MASKED_IMAGES_PATH)
    else:
        for file in os.listdir(OBA_MASKED_IMAGES_PATH):
            if file.startswith("OBA"):
                os.remove(os.path.join(OBA_MASKED_IMAGES_PATH, file))

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

    for idx, (key, value) in enumerate(x_train_dict.items()):
        image_name = f"Original_{idx + 1}.tif"
        mask_name = f"Original_{idx + 1}_mask.tif"

        image_path = os.path.join(OBA_IMAGES_PATH, image_name)
        mask_path = os.path.join(OBA_MASKED_IMAGES_PATH, mask_name)

        profile = {
            "driver": "GTiff",
            "height": value["image"].shape[1],
            "width": value["image"].shape[2],
            "count": value["image"].shape[0],
            "dtype": value["image"].dtype,
        }

        with rasterio.open(image_path, "w", **profile) as dst:
            dst.write(value["image"])

        mask_profile = profile.copy()
        mask_profile.update({"count": 1, "dtype": y_train_dict[key]["image"].dtype})
        with rasterio.open(mask_path, "w", **mask_profile) as dst:
            dst.write(y_train_dict[key]["image"][0], 1)

        print(f"Saved {image_name} to {image_path}")
        print(f"Saved {mask_name} to {mask_path}")

    for idx in range(len(x_train_dict.items())):
        if random.random() < prob_of_OBA:
            sample_image, sample_mask, profile = generator.generate_augmented_sample()
            if sample_image is None or sample_mask is None:
                print(f"Skipping sample {idx + 1} due to failed augmentation.")
                continue

            image_name = f"OBA_{idx + 1}.tif"
            mask_name = f"OBA_{idx + 1}_mask.tif"

            image_path = os.path.join(OBA_IMAGES_PATH, image_name)
            mask_path = os.path.join(OBA_MASKED_IMAGES_PATH, mask_name)

            profile.update(
                {
                    "driver": "GTiff",
                    "height": sample_image.shape[1],
                    "width": sample_image.shape[2],
                    "count": sample_image.shape[0],
                    "dtype": sample_image.dtype,
                }
            )

            with rasterio.open(image_path, "w", **profile) as dst:
                dst.write(sample_image)

            mask_profile = profile.copy()
            mask_profile.update({"count": 1, "dtype": sample_mask.dtype})
            with rasterio.open(mask_path, "w", **mask_profile) as dst:
                dst.write(sample_mask[0], 1)

            print(f"Saved {image_name} to {image_path}")
            print(f"Saved {mask_name} to {mask_path}")
