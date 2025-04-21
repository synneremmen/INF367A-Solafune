import numpy as np
from albumentations import (
    HorizontalFlip, Perspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine,
    Sharpen, Emboss, RandomBrightnessContrast, OneOf, Compose, RandomGamma,
    ChannelShuffle, RGBShift, HorizontalFlip, VerticalFlip
)

def aug(p, color_aug_prob, geometric_aug_prob):
    # funker ikke med GuassNoise, HueSaturationValue, CLAHE
    # dårlig med RandomBrightnessContrast, RandomGamma, Emboss, MotionBlur

    return Compose([
        OneOf([
            HorizontalFlip(),
            VerticalFlip()
        ], p=geometric_aug_prob),
        OneOf([
            OpticalDistortion(p=1), 
        ], p=color_aug_prob)
    ], p=p, seed=42)
    # return Compose([
    #     RandomRotate90(p=0.8),
    #     OneOf([
    #         HorizontalFlip(p=1),
    #         VerticalFlip(p=0)
    #     ]),
    #     OneOf([
    #         GaussNoise(p=1), 
    #         HueSaturationValue(p=1), 
    #         CLAHE(p=1),
    #         OpticalDistortion(p=1), 
    #         RandomBrightnessContrast(p=1),
    #         RandomGamma(p=1),
    #         Emboss(p=1),
    #         MotionBlur(p=1),
    #     ], p=color_aug_prob)
    # ], p=p, seed=42)

def augment(img, mask, color_aug_prob, geometric_aug_prob=0.6, p=0.9):
    
    # our images are in the form (12, 1024, 1024)
    # want input on the form (1024, 1024, 12)
    img = img.transpose(2,1,0) 
    mask = mask.transpose(2,1,0)

    aug_func = aug(p, color_aug_prob, geometric_aug_prob) # lager en funksjon for å kjøre Compose på img - img_mask pair
    
    data = {"image": img, "mask": mask} # img.astype(np.float32)/255
    
    augmented = aug_func(**data)

    # transpose the image back to (12, 1024, 1024)
    augmented["image"] = augmented["image"].transpose(2,1,0)
    augmented["mask"] = augmented["mask"].transpose(2,1,0)
    
    return augmented