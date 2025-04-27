from OBA.object_based_augmentation import create_save_OBA_images
from utils.evaluation import run_evaluation
from train.train import train
from train.loader import get_loader, get_dataset
from train.selection import train_model_selection
from models.simple_convnet import SimpleConvNet
from models.UNet import UNet
from models.resnet import UNetResNet18
from satmae_pp.vit_large import make_vit_finetune
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import os
from datasets.deforestation_dataset import build_datasets
from OBA.object_based_augmentation import Generator
import sys
import subprocess
from functools import partial

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)

IMAGES_PATH = os.getenv("IMAGES_PATH")
MASKED_IMAGES_PATH = os.getenv("MASKED_IMAGES_PATH")
IMAGES_SUBSET_PATH = os.getenv("IMAGES_SUBSET_PATH")
MASKED_IMAGES_SUBSET_PATH = os.getenv("MASKED_IMAGES_SUBSET_PATH")
OBA_IMAGES_PATH = os.getenv("OBA_IMAGES_PATH")
OBA_MASKED_IMAGES_PATH = os.getenv("OBA_MASKED_IMAGES_PATH")

def main(subset=False):
    """
    Main function to train and/or evaluate a model. Can be used for model selection with hyperparameter search 
    or to train a single model.
    Args:
        model_selection (bool): If True, perform model selection with hyperparameter search.
        subset (bool): If True, use a subset of the data for training and evaluation.
    """
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    n_epochs = 30
    batch_size = 8
    MODEL_PATH = None#"models/UNet_normal_paramset_0.001_0.01_0.9.pth"

    if subset:
        image_path = IMAGES_SUBSET_PATH
        masked_image_path = MASKED_IMAGES_SUBSET_PATH
    else:
        image_path = IMAGES_PATH
        masked_image_path = MASKED_IMAGES_PATH

    # perform model selection with hyperparameter search on different models and/or with different datasets
    for dataset in ["normal", "SR"]:# ["normal", "OBA", "SR", "SR_OBA"]:
        print(f"\n\nModel selection on {dataset} dataset...")
        
        if dataset == "SR":
            image_path = os.getenv("SR_IMAGES_PATH")
            masked_image_path = MASKED_IMAGES_PATH
            if not os.path.exists(image_path):
                sr_script = os.path.join(
                    os.path.dirname(__file__),
                    "SR",
                    "superresolution.py"
                )
                subprocess.run([sys.executable, sr_script], check=True)

        elif dataset == "OBA":
            #create_save_OBA_images(subset=subset) # create new OBA images each time for randomness
            image_path = os.getenv("OBA_IMAGES_PATH")
            masked_image_path = os.getenv("OBA_MASKED_IMAGES_PATH")

        elif dataset == "SR_OBA":
            #create_save_OBA_images(subset=subset, use_SR=True) # create new OBA images each time for randomness
            image_path = os.getenv("SR_OBA_IMAGES_PATH")
            masked_image_path = os.getenv("SR_OBA_MASKED_IMAGES_PATH")
        else:
            image_path = IMAGES_PATH
            masked_image_path = MASKED_IMAGES_PATH

        oba_generator = None
        # not used in the current implementation, but can be used for on-the-fly OBA generation
        # if dataset == "OBA" or dataset == "SR_OBA":
        #     oba_generator = Generator(batch_size=batch_size)

        train_loader, val_loader, test_loader = build_datasets(
            images_dir=image_path,
            masks_dir=masked_image_path,
            oba_generator=oba_generator,
            num_workers=4,
            batch_size=batch_size)

        print("Size of training dataset: ", len(train_loader.dataset))
        print("Size of validation dataset: ", len(val_loader.dataset))
        print("Size of test dataset: ", len(test_loader.dataset))

        # For når vi leverer koden
        # param_grid = {
        #     'lr': [0.1, 0.001],
        #     'decay': [0.01, 0.001],
        #     'mom': [0.9, 0.99],
        # }
        models = {
            "SimpleConvNet": SimpleConvNet,
            "UNet": UNet,
            "UNetResNet18": UNetResNet18,
            # "ViT-finetune": partial(  # refers to finetune ViT-large 
            #     make_vit_finetune,
            #     ckpt_path="satmae_pp/checkpoint_ViT-L_pretrain_fmow_sentinel.pth",
            #     n_trainable_layers=2
            # )
        }

        # For testing
        param_grid = {
            'lr': [0.001],
            'decay': [0.01],
            'mom': [0.9],
        }

        print("\n\nTraining model selection...")
        best_model, best_model_name, best_params, best_val_score, best_train_losses, best_val_losses = train_model_selection(
            models,
            param_grid,
            n_epochs,
            loss_fn,
            train_loader,
            val_loader,
            dataset=dataset,
            device=DEVICE,
            early_stopping=True, 
            patience=5,  # early stopping if it doesn't improve for 5 epochs
        )

        print("\n\nBest model: ", best_model_name)

        print("\n\nRunning evaluation...")
        torch.cuda.empty_cache()
        run_evaluation(best_model, test_loader, device=DEVICE, save=True)
        print("\n\nModel selection and evaluation completed.\n\n")

if __name__ == "__main__":
    main(subset=False)
