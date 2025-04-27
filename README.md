# INF367A-Identifying Deforestation drivers

This project contains our submission to [Solafune competition](https://solafune.com/competitions/68ad4759-4686-4bb3-94b8-7063f755b43d?menu=about&tab=&topicId=a5e978e7-7759-4433-b1a5-063760451ff5).


## Project Description

As described in the competition deforestation is when previously forested areas are turned into non-forrested. This is usually due to human activity were forested areas are turned into cultivated land in some manor.

This heavily affets rainforest such as the amazons. The competition wishes to identify deforestation drivers by using drone images.

The project also contains 3 individuals implementations:
- superresolution
    - location: SR/
- object based augmentaion
    - location: OBA/
- transfomers (SatMAE-PP)
    - location: satmae_pp/

## File Structure

project/
├── main.py
├── exploratory_data_analysis.ipynb
├── .env
├── requirements.txt
├── README.md
├── train/
│   ├── loader.py
│   ├── selection.py
│   └── train.py
├── utils/
│   ├── create_masked_data.py
│   ├── create_subsets.py
│   ├── evaluation.py
│   ├── loading.py
│   ├── normalize.py
│   ├── postprocessing.py
│   ├── preprocessing.py
│   └── visualization.py
├── SR/
│   ├── superres/
│   │   ├── L2A20M.pt
│   │   └── L2A60M.pt
│   ├── superresolution.py
│   ├── README.md
│   └── SR-inspect.ipynb
├── satmae_pp/
│   ├── README.md
│   ├── satemae_pp.py
│   └── vit_large.py
├── OBA/
│   ├── README.md
│   ├── augmentation.py
│   ├── object_based_augmentation.py
│   └── inspect_object_based_augmentation.ipynb
├── data/
│   ├── train_images/
│   ├── masked_annotations/
│   └── train_annotations.json
├── datasets/
│   └── deforestation_dataset.py
└── models/
    ├── simple_convnet.py
    ├── resnet.py
    └── UNet.py

## Create .env

Create an .env file with the following information:

```plaintext
LABELS_PATH = path/to/labels/
IMAGES_PATH = path/to/images/
MASKED_IMAGES_PATH = path/to/masked/images/
EVAL_IMAGES_PATH = path/to/evaluation/images/
```

If training using a subset, these are also required:

```plaintext
IMAGES_SUBSET_PATH = path/to/subset/images
MASKED_IMAGES_SUBSET_PATH = path/to/subset/images
EVAL_IMAGES_SUBSET_PATH = path/to/evaluation/images/subset
```

If training with superresolution images

```plaintext
SR_IMAGES_PATH = path/to/superresolved_images
SR_20M_PATH = path/to/superresolution_20m_model
SR_60M_PATH = path/to/superresolution_60m_model
```

If training with object-based augmentation
```plaintext
OBA_IMAGES_PATH = path/to/oba_images_path/
OBA_MASKED_IMAGES_PATH = path/to/oba_masked_images_path/
BACKGROUND_IMAGES_PATH = path/to/background_images/
```

## Dataset

Contains a training dataset of 176 datapoints:
We split this into train, val and test for training and model evaluation.

## Methodology

- **Evaluation Metric**  
  - Pixel-wise IoU F1-score was used to measure model performance, balancing precision and recall by evaluating the overlap between predicted and ground truth segments.

- **Model Architectures**  
  - Implemented three main model types:
    - A baseline convolutional neural network (CNN).
    - A ResNet-based encoder-decoder network.
    - A novel transformer-based architecture (SatMAE).

- **Hyperparameter Search**  
  - Performed grid search over key hyperparameters (e.g., learning rate, batch size) for each architecture.

- **Data Augmentation**  
  - Trained models on datasets with various augmentations:
    - **Super-resolution**: Low-resolution bands were enhanced to higher resolution before training.
    - **Object-based augmentation**: Techniques applied to improve segmentation around objects of interest.

- **Model Selection**  
  - Selected the best model based on highest validation F1-score.

- **Final Evaluation**  
  - Evaluated the selected model on a separate, unseen test set to assess generalization performance.

## Required packages

[solafune-tools](https://github.com/Solafune-Inc/solafune-tools/tree/main)

Run the following code:

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

## How to run project (ADD MORE DETAIL)

1. Run the create_masked_data.py to get images in the masked_annotations folder. 
These tif files contains only one band of size 1024x1024. Each pixel in the data contains the class value (0-4, where 0 is none.)

## Contributors

Aurora Ingebrigtsen, Synne Remmen Andreassen, Christian Bontveit

## References

[solafune-tools](https://github.com/Solafune-Inc/solafune-tools/tree/main) by Solafune team