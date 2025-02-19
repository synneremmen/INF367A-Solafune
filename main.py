from utils.preprocessing import get_processed_data, get_processed_evaluation_data
from train.train import train
import torch
from models.simple_convnet import SimpleConvNet
import torch.nn as nn
from train.loader import get_loader
from utils.postprocessing import run_evaluation
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)

def main():
    print("Loading data...")
    dataset = get_processed_data(subset=True)
    train_loader, val_loader, test_loader = get_loader(dataset, batch_size=10)

    model = SimpleConvNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    n_epochs = 5

    print()
    print("Training model...")
    losses_train = train(n_epochs, optimizer, model, loss_fn, train_loader, device=DEVICE)

    print()
    print("Training completed. Training losses:")
    print(losses_train)

    print()
    print("Running evaluation...")
    eval_dataset = get_processed_evaluation_data(subset=True)
    eval_loader = DataLoader(eval_dataset, batch_size=10, shuffle=False)
    run_evaluation(model, eval_loader, device=DEVICE)

if __name__ == "__main__":
    main()