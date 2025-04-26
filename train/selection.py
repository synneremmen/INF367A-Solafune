import torch
import torch.nn as nn
from train.train import train
from utils.evaluation import run_evaluation
from utils.OBA.object_based_augmentation import create_OBA_dataset
from utils.preprocessing import get_processed_data
from torch import optim
from itertools import product
from torch.optim.lr_scheduler import StepLR

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

def train_model_selection(models, param_grid, n_epochs, loss_fn, train_loader, val_loader, device='cpu', early_stopping=True, patience=5):
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

            # instantiate the model and optimizer
            model = model_fn().to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(mom, 0.999), weight_decay=decay)
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1, verbose=False)

            # train the model
            print(f"\n\nTraining {name} with param set {idx}: lr={lr}, decay={decay}, mom={mom}")
            train_losses, val_losses = train(model, optimizer, loss_fn, train_loader, val_loader, device, n_epochs, scheduler, early_stopping=early_stopping, patience=patience)

            result_dict = run_evaluation(model, val_loader, device, save=False)
            val_score = result_dict["Overall"]["F1"]  # Only take Overall F1 score

            print(f"Model: {name}, Param set {idx}, Validation score (F1): {val_score}")

            print("=" * 50)

            print("\n\nSaving model...")
            model_save_path = f"models/{name}_paramset_{lr}_{decay}_{mom}.pth"
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


def get_dataset(dataset_type, subset=False):
    print("\n\nLoading data...")
    use_OBA = False
    use_SR = False

    if dataset_type == "normal":
        dataset = get_processed_data(subset=subset)
    elif dataset_type == "OBA":
        use_OBA = True
        use_SR = False
    elif dataset_type == "SR":
        use_OBA = False
        use_SR = True
    elif dataset_type == "SR_OBA":
        use_OBA = True
        use_SR = True
    else:
        raise ValueError("Invalid dataset type. Choose from 'normal', 'OBA', 'SR', or 'SR_OBA'.")
    
    if use_OBA:
        print(f"Using OBA{' and SR' if use_SR else ''} dataset")
        dataset = create_OBA_dataset(
            prob_of_OBA=0.5, # how much OBA data to generate
            subset=True,
            augm=True,
            object_augm=True,
            extra_background_prob=0, # not in use
            background_augm_prob=0.6,
            shadows=False, # not to be used
            extra_objects=3,
            object_augm_prob=0.6,
            augm_prob=0.8,
            geometric_augm_prob=0.6,
            color_augm_prob=0.6,
            batch_size=10,
            min_area=1000, # how much area to be considered as an object
            use_SR=use_SR,
        )

    if use_SR:
        print("Using SR dataset")
        dataset = get_processed_data(use_SR=True, subset=False)

    return dataset