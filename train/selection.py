import torch
import torch.nn as nn
from train.train import train
from utils.evaluation import run_evaluation
from torch import optim
from itertools import product
from torch.optim.lr_scheduler import StepLR
import os
from utils.visualization import plot_loss

def selection(models:list[nn.Module], val_loader, device) -> None:
    best_model = None
    best_score = float('-inf')

    for model in models:
        # Compute validation score using `evaluate()`
        val_score = run_evaluation(model, val_loader, device, save=False)

        print(f"Model: {model}, validation score: {val_score}")
    
        if val_score > best_score:
            best_score = val_score
            best_model = model
    
    print(f"Best model: {best_model}, score: {best_score}")

    return best_model, best_score

def train_model_selection(models, param_grid, n_epochs, loss_fn, train_loader, val_loader, dataset, device='cpu', early_stopping=True, patience=5):
    best_val_score = float('-inf')
    best_model = None
    best_model_name = None
    best_params = None
    best_train_losses = None
    best_val_losses = None

    # get all possible combinations of hyperparameters
    param_combinations = list(product(param_grid['lr'], param_grid['decay'], param_grid['mom']))

    # iterate over all models and hyperparameter combinations
    for name, model_fn in models.items():
        for idx, (lr, decay, mom) in enumerate(param_combinations):

            model_save_path = f"models/{name}_{dataset}_paramset_{lr}_{decay}_{mom}.pth"
            
            if os.path.exists(model_save_path):
                print(f"\nFound checkpoint for {name} param set {idx}, loading instead of training.")
                # instantiate & load
                model = model_fn().to(device)
                model.load_state_dict(torch.load(model_save_path, map_location=device))
                model.eval()

                # evaluate loaded model
                result_dict = run_evaluation(model, val_loader, device, save=False)
                val_score = result_dict["Overall"]["F1"]
                print(f"Loaded model validation F1: {val_score}")

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model = model
                    best_model_name = f"{name}_paramset_{idx}"
                    best_params = {'lr': lr, 'decay': decay, 'mom': mom}
                    best_train_losses = None
                    best_val_losses = None

                    # skip training entirely
                continue


            # instantiate the model and optimizer
            model = model_fn().to(device)
            model.float()
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(mom, 0.999), weight_decay=decay)
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1, verbose=False)

            # train the model
            print(f"\n\nTraining {name} with param set {idx}: lr={lr}, decay={decay}, mom={mom}")
            train_losses, val_losses = train(model, optimizer, loss_fn, train_loader, val_loader, device, n_epochs, scheduler, early_stopping=early_stopping, patience=patience)

            plot_loss(train_losses, val_losses, save_path=f"losses/{name}_{dataset}_paramset_{idx}.png")

            result_dict = run_evaluation(model, val_loader, device, save=False)
            val_score = result_dict["Overall"]["F1"]  # Only take Overall F1 score

            print(f"Model: {name}, Param set {idx}, Validation score (F1): {val_score}")

            print("=" * 50)

            print("\n\nSaving model...")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

            # update the best model if the current one is better
            if val_score > best_val_score:
                best_val_score = val_score
                best_model = model
                best_model_name = f"{name}_paramset_{idx}"
                best_params = {'lr': lr, 'decay': decay, 'mom': mom}
                best_train_losses = train_losses
                best_val_losses = val_losses

    print('\n\nModel selection completed')
    print(f"Best model: {best_model_name}, score: {best_val_score}")
    print(f"Best params: {best_params}")

    return best_model, best_model_name, best_params, best_val_score, best_train_losses, best_val_losses