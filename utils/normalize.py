import torch
from torchvision import transforms

def normalize(x_train:torch.Tensor):
    means = x_train.mean(dim=(0, 2, 3))
    stds = x_train.std(dim=(0, 2, 3))

    eps = 1e-7
    stds_fixed = stds + eps

    normalizer_pipe = transforms.Normalize(means, stds_fixed)

    preprocessor = transforms.Compose([
        normalizer_pipe
    ])

    x_train = [preprocessor(img) for img in x_train]
    x_train = torch.stack(x_train, dim=0)
    return x_train
