from utils.evaluation import run_evaluation
from train.train import train
from train.loader import get_loader, get_dataset
from train.selection import train_model_selection
from models.simple_convnet import SimpleConvNet
from models.UNet import UNet
from models.resnet import UNetResNet18
from models.vit_large import vit_seg_large_patch16, make_vit_finetune
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import os
from datasets.deforestation_dataset import build_datasets
from utils.OBA.object_based_augmentation import Generator
import sys
import subprocess
from functools import partial

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)

IMAGES_PATH = os.getenv("IMAGES_PATH")
MASKED_IMAGES_PATH = os.getenv("MASKED_IMAGES_PATH")
IMAGES_SUBSET_PATH = os.getenv("IMAGES_SUBSET_PATH")
MASKED_IMAGES_SUBSET_PATH = os.getenv("MASKED_IMAGES_SUBSET_PATH")

def main(model_selection=False, subset=False):
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
    MODEL_PATH = "models/UNet_normal_paramset_0.001_0.01_0.9.pth"

    if subset:
        image_path = IMAGES_SUBSET_PATH
        masked_image_path = MASKED_IMAGES_SUBSET_PATH
    else:
        image_path = IMAGES_PATH
        masked_image_path = MASKED_IMAGES_PATH

    if MODEL_PATH and os.path.exists(MODEL_PATH):
        # use saved model if it exists
        print("\n\nLoading saved model...")

        model = UNet().to(DEVICE)  # SimpleConvNet().to(DEVICE) # UNet().to(DEVICE) # UNetResNet18().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=DEVICE))

        train_loader, val_loader, test_loader = build_datasets(
            images_dir=image_path,
            masks_dir=masked_image_path,
            oba_generator=None,
            num_workers=4,
            batch_size=batch_size)

        print("\n\nRunning evaluation...")
        torch.cuda.empty_cache()
        run_evaluation(model, test_loader, device=DEVICE, save=True)
        print("\n\nEvaluation completed of saved model.\n\n")

    elif model_selection:
        # perform model selection with hyperparameter search on different models and/or with different datasets
        for dataset in ["OBA"]:# ["normal", "OBA", "SR", "SR_OBA"]:
            print(f"\n\nModel selection on {dataset} dataset...")
            
            if dataset == "SR" or dataset == "SR_OBA":
                image_path = os.getenv("SR_IMAGES_PATH")
                if not os.path.exists(image_path):
                    sr_script = os.path.join(
                        os.path.dirname(__file__),
                        "utils",
                        "superresolution.py"
                    )
                    subprocess.run([sys.executable, sr_script], check=True)

            else:
                image_path = image_path

            oba_generator = None
            if dataset == "OBA" or dataset == "SR_OBA":
                print("Using OBA generator for augmentation...")
                oba_generator = Generator(batch_size=batch_size)
                print("Finished creating OBA generator...")

            train_loader, val_loader, test_loader = build_datasets(
                images_dir=image_path,
                masks_dir=masked_image_path,
                oba_generator=oba_generator,
                num_workers=24,
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
                "ViT-finetune": partial(
                    make_vit_finetune,
                    num_classes=5,
                    patch_size=16,
                    img_size=1024,
                    in_chans=12,
                    ckpt_path="checkpoint_ViT-L_pretrain_fmow_sentinel.pth",
                    n_trainable_layers=2
                )
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

    else: # train a single model
        print("\n\nTraining model...")

        model = SimpleConvNet().to(
            DEVICE
        )  # SimpleConvNet().to(DEVICE) # UNet().to(DEVICE) # UNetResNet18().to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1, verbose=False)
        train_loader, val_loader, test_loader = build_datasets(
            images_dir=image_path,
            masks_dir=masked_image_path,
            oba_generator=oba_generator,
            num_workers=4,
            batch_size=batch_size)
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
    main(model_selection=True, subset=False)
