from utils.preprocessing import get_processed_data, get_processed_evaluation_data
from train.train import train
import torch
from models.simple_convnet import SimpleConvNet
from models.UNet import UNet
import torch.nn as nn
from train.loader import get_loader
from utils.postprocessing import run_evaluation
from torch.utils.data import DataLoader
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)

# model path
MODEL_PATH = "models/saved_model.pth"

def main():
    subset = False

    print("\n\nLoading data...\n\n")
    dataset = get_processed_data(subset=subset)
    train_loader, val_loader, test_loader = get_loader(dataset, batch_size=10)

    print("Size of training dataset: ", len(train_loader.dataset))
    print("Size of validation dataset: ", len(val_loader.dataset))
    print("Size of test dataset: ", len(test_loader.dataset))

    model = SimpleConvNet() # UNet() if you want to use the UNet model

    if os.path.exists(MODEL_PATH):
        print("\n\nLoading saved model...\n\n")
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        n_epochs = 30

        print("\n\nTraining model...\n\n")
        losses_train = train(n_epochs, optimizer, model, loss_fn, train_loader, device=DEVICE)

        print("\n\nTraining completed. Training losses:\n\n")
        print(losses_train)

        print("\n\nSaving model...\n\n")
        torch.save(model.state_dict(), MODEL_PATH)

    print("\n\nRunning evaluation...\n\n")
    eval_dataset = get_processed_evaluation_data(subset=subset)
    eval_loader = DataLoader(eval_dataset, batch_size=10, shuffle=False)
    torch.cuda.empty_cache()
    run_evaluation(model, eval_loader, device=DEVICE)

if __name__ == "__main__":
    main()