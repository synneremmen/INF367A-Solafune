import torch.nn as nn
#   from train.optimizer import adam, sgd
#   from train.loss import 
from train import train
from utils.evaluation import run_evaluation
from itertools import product

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