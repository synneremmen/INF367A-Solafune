import torch.nn as nn
#   from train.optimizer import adam, sgd
#   from train.loss import 
from train import train
from utils.evaluation import run_evaluation
from itertools import product

def selection(models:list[nn.Module], params:dict, train_loader, val_loader, device) -> None:

    best_model = None
    best_hyperparams = None
    best_score = float('-inf')
    loss_fn = nn.CrossEntropyLoss()

    for model in models:
        for n_epochs, optimizer, lr, batch_size in product(*params.values()):
            hyperparams = {
                "n_epochs": n_epochs,
                "optimizer": optimizer,
                "lr": lr,
                "batch_size": batch_size
            }

            model, train_loss, val_loss = train(n_epochs, optimizer, model, loss_fn, train_loader, device, val_loader=None)

             # Compute validation score using `evaluate()`
            val_score = run_evaluation(model, val_loader, device, save=False)

            print(f"Model: {type(model).__name__}, LR: {lr}, Epochs: {n_epochs}, Val Loss: {val_score}")
    
            if val_score > best_score:
                best_score = val_score
                best_model = model
                best_hyperparams = hyperparams

    return best_model, best_hyperparams, best_score