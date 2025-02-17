from torch.utils.data import TensorDataset
import torch
from utils.normalize import normalize
from utils.loading import load_images, load_masked_images

def get_processed_data(subset=False):
    x_train_dict = load_images(subset=subset)
    y_train_dict = load_masked_images(subset=subset)

    x_train = [torch.tensor(each['image']) for each in x_train_dict.values()]
    y_train = [torch.tensor(each['image']) for each in y_train_dict.values()]

    x_train_tensor = torch.stack(x_train, dim=0)  # Shape: [num_samples, 12, 1024, 1024]
    y_train_tensor = torch.stack(y_train, dim=0).squeeze(1).long()   # Shape: [num_samples, 1, 1024, 1024]

    x_train_tensor = torch.nan_to_num(x_train_tensor, nan=0.0)
    x_train_tensor = normalize(x_train_tensor)

    return TensorDataset(x_train_tensor, y_train_tensor)