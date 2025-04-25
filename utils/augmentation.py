import numpy as np
from albumentations import (
    HorizontalFlip,
    RandomRotate90,
    Transpose,
    OneOf,
    Compose,
    VerticalFlip,
    Lambda,
)


def aug(p, color_aug_prob, geometric_aug_prob):
    return Compose(
        [
            OneOf(
                [
                    HorizontalFlip(),
                    VerticalFlip(),
                    RandomRotate90(),
                    Transpose(),
                ],
                p=geometric_aug_prob,
            ),
            OneOf(
                [
                    Lambda(
                        image=lambda x, **kwargs: x
                        + np.random.normal(0, 0.01, x.shape).astype(np.float32)
                    ),
                    Lambda(
                        image=lambda x, **kwargs: x
                        * np.random.uniform(0.9, 1.1, x.shape).astype(np.float32)
                    ),
                ],
                p=color_aug_prob,
            ),
        ],
        p=p,
        additional_targets={"mask": "mask"},
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
    augmented["image"] = augmented["image"].transpose(2, 0, 1).astype(np.float64)
    augmented["mask"] = augmented["mask"].astype(np.float64)

    return augmented["image"], augmented["mask"]
