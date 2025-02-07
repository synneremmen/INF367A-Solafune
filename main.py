from utils.preprocessing import get_processed_data
from train.train import train
import torch
from models.simple_convnet import SimpleConvNet
from torch.utils.data import DataLoader
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)

def main():
    print("Loading data...")
    train_dataset = get_processed_data(subset=True)
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