import numpy as np
from albumentations import (
    HorizontalFlip, Perspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine,
    Sharpen, Emboss, RandomBrightnessContrast, OneOf, Compose, RandomGamma,
    ChannelShuffle, RGBShift, HorizontalFlip, VerticalFlip, Rotate, MultiplicativeNoise
)

def aug(p, color_aug_prob, geometric_aug_prob):
    return Compose([
        OneOf([
            HorizontalFlip(),
            VerticalFlip(),
            RandomRotate90(), 
        ], p=geometric_aug_prob),
        OneOf([
            OpticalDistortion(), 
            MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True),
            #GaussNoise(var_limit=(1.0, 5.0), per_channel=True)
        ], p=color_aug_prob),
    ], 
        p=p, 
        #seed=42,
        additional_targets={'mask': 'mask'}
        )

def augment(img, mask, augm_prob=0.9, color_aug_prob=0.6, geometric_aug_prob=0.6):
    # our images are on the form (12, 1024, 1024)
    # want input on the form (1024, 1024, 12)
    img = img.transpose(1, 2, 0)

    img = img.astype(np.float32)
    mask = mask.astype(np.float32)

    aug_func = aug(augm_prob, color_aug_prob, geometric_aug_prob)
    data = {"image": img, "mask": mask}
    augmented = aug_func(**data)

    # transpose the image back to (12, 1024, 1024)
    augmented["image"] = augmented["image"].transpose(2,0,1).astype(np.float64)
    augmented["mask"] = augmented["mask"].astype(np.float64)
    
    return augmented["image"], augmented["mask"]