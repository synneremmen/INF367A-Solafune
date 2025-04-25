from utils.preprocessing import get_processed_data
from train.train import train
import torch
from models.simple_convnet import SimpleConvNet
from models.UNet import UNet
from models.resnet import UNetResNet18
import torch.nn as nn
from train.loader import get_loader
from train.selection import train_model_selection
from utils.evaluation import run_evaluation
import os
from torch.optim.lr_scheduler import StepLR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)

# model path
MODEL_PATH = "models/saved_model.pth"

def main(model_selection=False, subset=False):
    """
    Main function to train and evaluate the model.
    Args:
        model_selection (bool): If True, perform model selection with hyperparameter search.
        subset (bool): If True, use a subset of the data for training and evaluation.
    """

    model = SimpleConvNet().to(
        DEVICE
    )  # SimpleConvNet().to(DEVICE) # UNet().to(DEVICE) # UNetResNet18().to(DEVICE)

    print("\n\nLoading data...")
    dataset = get_processed_data(subset=subset)
    train_loader, val_loader, test_loader = get_loader(dataset, batch_size=6)
    print("Size of training dataset: ", len(train_loader.dataset))
    print("Size of validation dataset: ", len(val_loader.dataset))
    print("Size of test dataset: ", len(test_loader.dataset))

    if os.path.exists(MODEL_PATH):
        print("\n\nLoading saved model...")
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        n_epochs = 30
        
        if model_selection: # perform model selection with hyperparameter search
            print("\n\nModel selection...")

            param_grid = {
                'lr': [0.1, 0.001, 0.001],
                'decay': [0.01, 0.001],
                'mom': [0.9, 0.99],
            }
            models = {
                "SimpleConvNet": SimpleConvNet,
                 "UNet": UNet,
                 "UNetResNet18": UNetResNet18,
            }

            print("\n\nTraining model selection...")
            best_model, best_model_name, best_params, best_val_score, best_train_losses, best_val_losses = train_model_selection(
                models,
                param_grid,
                n_epochs,
                loss_fn,
                train_loader,
                val_loader,
                device=DEVICE,
                early_stopping=True, 
                patience=5,  # early stopping if it doesn't improve for 5 epochs
            )

            model = best_model

        else: # train a model
            print("\n\nTraining model...")

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1, verbose=False)

            losses_train = train(
                n_epochs,
                optimizer,
                model,
                loss_fn,
                train_loader,
                scheduler=scheduler,
                device=DEVICE,
            )

            print("\n\nTraining completed. Training losses:")
            print(losses_train)

    print("\n\nSaving model...")
    torch.save(model.state_dict(), MODEL_PATH)

    print("\n\nRunning evaluation...")
    torch.cuda.empty_cache()
    run_evaluation(model, test_loader, device=DEVICE)

    print("\n\nEvaluation completed.\n\n")

if __name__ == "__main__":
    main(model_selection=True, subset=True)
