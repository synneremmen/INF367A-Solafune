from utils.preprocessing import get_processed_data, get_processed_evaluation_data
from train.train import train
import torch
# from models.simple_convnet import SimpleConvNet
# from models.UNet import UNet
# from models.UNetLight import UNet_Light
from models.resnet import UNetResNet18
import torch.nn as nn
from train.loader import get_loader
from utils.evaluation import run_evaluation
from torch.utils.data import DataLoader
import os
from torch.optim.lr_scheduler import StepLR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)

# model path
MODEL_PATH = "models/test_saved_model.pth"

def main():
    subset = True
    model = UNetResNet18().to(DEVICE)# UNet_Light().to(DEVICE) # SimpleConvNet().to(DEVICE) # UNet().to(DEVICE) # UNet_Light().to(DEVICE) # UNetResNet18().to(DEVICE)

    if os.path.exists(MODEL_PATH):
        print("\n\nLoading saved model...")
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    else:
        print("\n\nLoading data...")
        dataset = get_processed_data(subset=subset)
        train_loader, val_loader, test_loader = get_loader(dataset, batch_size=6)

        print("Size of training dataset: ", len(train_loader.dataset))
        print("Size of validation dataset: ", len(val_loader.dataset))
        print("Size of test dataset: ", len(test_loader.dataset))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # endret lr fra 0.001 til 0.0001
        loss_fn = nn.CrossEntropyLoss()
        n_epochs = 20

        # Add learning rate scheduler (reduces LR every 10 epochs by factor of 0.1)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  

        print("\n\nTraining model...")
        losses_train = train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler=scheduler, device=DEVICE)

        print("\n\nTraining completed. Training losses:")
        print(losses_train)

        print("\n\nSaving model...")
        torch.save(model.state_dict(), MODEL_PATH)

    print("\n\nRunning evaluation...")
    eval_dataset = get_processed_evaluation_data(subset=subset)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)
    torch.cuda.empty_cache()
    run_evaluation(model, eval_loader, device=DEVICE)
    print("\n\nEvaluation completed.\n\n")

if __name__ == "__main__":
    main()