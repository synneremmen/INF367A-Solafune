from utils.loading import load_images, load_masked_images
from train.train import train
from train.normalize import normalize
import torch
from torch.utils.data import TensorDataset
from models.simple_convnet import SimpleConvNet
from torch.utils.data import DataLoader
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)

def main():
    print("Loading data...")
    x_train_dict = load_images()
    y_train_dict = load_masked_images()

    x_train = [torch.tensor(each['image']) for each in x_train_dict.values()]
    y_train = [torch.tensor(each['image']) for each in y_train_dict.values()]

    x_train_tensor = torch.stack(x_train, dim=0)  # Shape: [num_samples, 12, 1024, 1024]
    y_train_tensor = torch.stack(y_train, dim=0).squeeze(1).long()   # Shape: [num_samples, 1, 1024, 1024]

    x_train_tensor = torch.nan_to_num(x_train_tensor, nan=0.0)

    x_train_tensor = normalize(x_train_tensor)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    batch_size = 10

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("Training model...")
    model = SimpleConvNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    n_epochs = 10

    losses_train = train(n_epochs, optimizer, model, loss_fn, train_loader, device=DEVICE)
    print(losses_train)

if __name__ == "__main__":
    main()