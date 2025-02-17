from utils.preprocessing import get_processed_data
from train.train import train
import torch
from models.simple_convnet import SimpleConvNet
import torch.nn as nn
from train.loader import get_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)

def main():
    print("Loading data...")
    dataset = get_processed_data(subset=True)
    train_loader, val_loader, test_loader = get_loader(dataset, batch_size=10)

    print("Training model...\n\n")
    model = SimpleConvNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    n_epochs = 10

    losses_train = train(n_epochs, optimizer, model, loss_fn, train_loader, device=DEVICE)
    print(losses_train)

if __name__ == "__main__":
    main()