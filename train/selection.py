import torch.nn as nn
#   from train.optimizer import adam, sgd
#   from train.loss import 
from train import train
from utils.evaluation import run_evaluation
from itertools import product
# optim adam
from torch import optim

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

def train_model_selection(models, params, n_epochs, loss_fn, train_loader, val_loader, scheduler,device='cpu'):
    best_val_score = float('-inf')
    best_model = None
    best_model_name = None
    best_params = None
    best_train_losses = None
    best_val_losses = None
    i = 0

    # loop over params

    
    for name, model in models.items(): # refine model selection loop (presently we need to define a model for eacgh set of hyperparams)
        if i%2 == 0:
            param = params[0]
        else:
            param = params[1]
    
        optimizer = optim.Adam(model.parameters(), lr=param['lr'], betas=param['mom'], weight_decay=param['decay'])
        print("Current hyperparamters:")
        print(param)

        print(f"Training {name} using train...")
        train_losses, val_losses = train(model, optimizer,scheduler,n_epochs, loss_fn, train_loader, val_loader, device)

        print()

        val_score = run_evaluation(model, val_loader, device, save=False)

        print(f"Model: {name}, validation score: {val_score}")

        print("=" * 50)

        # update best best_model
        if val_score > best_val_score:
            best_val_score = val_score
            best_model = model
            best_model_name = name
            best_params = param
            best_train_losses = train_losses
            best_val_losses = val_losses
        i += 1

        print(f'Model selection completed')
        print(f"Best model: {best_model_name}, score: {best_val_score}")
        print(f"Best params: {best_params}")

    return best_model, best_model_name, best_params, best_val_score, best_train_losses, best_val_losses

### example hyperparams
"""
# global hyperparams
batch_size =  512
n_epoch =  5
loss_fn =  localization_loss

# define the data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# instance hyperparams
hyperparams = [
    {'lr': 0.001,
     'mom': (0.9, 0.999),
     'decay': 0.0001   
    },
    {'lr': 0.0001,
     'mom': (0.9, 0.999),
     'decay': 0.001   
    },
]

# models, defined under the models definition section
models = {'model_small_1': MyNetSmall(),
          'model_small_2': MyNetSmall(),
          'model_medium_1': MyNetMedium(),
          'model_medium_2': MyNetMedium(),
          'model_large_1': MyNetLarge(),
          'model_large_2': MyNetLarge()}
"""