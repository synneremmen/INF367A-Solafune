from utils.OBA.object_based_augmentation import create_OBA_dataset
from utils.preprocessing import get_processed_data
from utils.evaluation import run_evaluation
from train.train import train
from train.loader import get_loader
from train.selection import train_model_selection, get_dataset
from models.simple_convnet import SimpleConvNet
from models.UNet import UNet
from models.resnet import UNetResNet18
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import os
from datasets.deforestation_dataset import build_datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)


def main(model_selection=False, subset=False):
    """
    Main function to train and evaluate the model.
    Args:
        model_selection (bool): If True, perform model selection with hyperparameter search.
        subset (bool): If True, use a subset of the data for training and evaluation.
    """
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    n_epochs = 30
    batch_size = 6
    MODEL_PATH = "models/SimpleConvNet_paramset_0.01_0.01_0.9.pth"

    if MODEL_PATH and os.path.exists(MODEL_PATH):
        # use saved model if it exists
        print("\n\nLoading saved model...")

        model = SimpleConvNet().to(DEVICE)  # SimpleConvNet().to(DEVICE) # UNet().to(DEVICE) # UNetResNet18().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=DEVICE))
        dataset = get_dataset("SR_OBA", subset=subset)

        train_loader, val_loader, test_loader = get_loader(dataset, batch_size=batch_size)
        print("Size of training dataset: ", len(train_loader.dataset))
        print("Size of validation dataset: ", len(val_loader.dataset))
        print("Size of test dataset: ", len(test_loader.dataset))

        print("\n\nRunning evaluation...")
        torch.cuda.empty_cache()
        run_evaluation(model, test_loader, device=DEVICE, save=False)
        print("\n\nEvaluation completed of saved model.\n\n")

    elif model_selection:
        # perform model selection with hyperparameter search on different models and/or with different datasets
        for dataset in ["normal", "OBA", "SR", "SR_OBA"]:
            print(f"\n\nModel selection on {dataset} dataset...")
            dataset = get_dataset(dataset, subset=subset)

            train_loader, val_loader, test_loader = get_loader(dataset, batch_size=batch_size)
            print("Size of training dataset: ", len(train_loader.dataset))
            print("Size of validation dataset: ", len(val_loader.dataset))
            print("Size of test dataset: ", len(test_loader.dataset))

            # For når vi leverer koden
            # param_grid = {
            #     'lr': [0.1, 0.001],
            #     'decay': [0.01, 0.001],
            #     'mom': [0.9, 0.99],
            # }
            # models = {
            #     "SimpleConvNet": SimpleConvNet,
            #     "UNet": UNet,
            #     "UNetResNet18": UNetResNet18,
            # }

            # For testing
            param_grid = {
                'lr': [0.001],
                'decay': [0.01],
                'mom': [0.9],
            }
            models = {
                "SimpleConvNet": SimpleConvNet,
                # "UNet": UNet,
                # "UNetResNet18": UNetResNet18,
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

            print("\n\nRunning evaluation...")
            torch.cuda.empty_cache()
            run_evaluation(best_model, test_loader, device=DEVICE, save=False)
            print("\n\nModel selection and evaluation completed.\n\n")

    else: # train a single model
        print("\n\nTraining model...")

        model = SimpleConvNet().to(
            DEVICE
        )  # SimpleConvNet().to(DEVICE) # UNet().to(DEVICE) # UNetResNet18().to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1, verbose=False)
        dataset = get_dataset("normal", subset=subset)

        train_loader, val_loader, test_loader = get_loader(dataset, batch_size=batch_size)
        print("Size of training dataset: ", len(train_loader.dataset))
        print("Size of validation dataset: ", len(val_loader.dataset))
        print("Size of test dataset: ", len(test_loader.dataset))

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
        run_evaluation(model, test_loader, device=DEVICE, save=True)
        print("\nTraining and evaluation completed.\n\n")

if __name__ == "__main__":
    main(model_selection=True, subset=True)
