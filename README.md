# INF367A-Identifying Deforestation drivers

This project is a submission to [Solafune competition](https://solafune.com/competitions/68ad4759-4686-4bb3-94b8-7063f755b43d?menu=about&tab=&topicId=a5e978e7-7759-4433-b1a5-063760451ff5).


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

## Project Description

As described in the competition deforestation is when previously forested areas are turned into non-forrested. This is usually due to human activity were forested areas are turned into cultivated land in some manor.

This heavily affets rainforest such as the amazons. The competition wishes to identify deforestation drivers by using drone images.

## Dataset

Contains a training dataset of 176 datapoints:
- dsda
- fsdadsad
- dsadsa

Evaluation_images
- Contains

## Methodology (ADD MORE DETAIL)

CNN, ect, Using a combination of remote sensing data and advanced machine learning techniques, competitors will analyze satellite images to distinguish between different land-use changes that lead to deforestation, such as agricultural expansion, mining, or other factors LOREM IPSUM LOREM.

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

## Submission and evaluation

## Contributors

Aurora Ingebrigtsen, Synne Remmen Andreassen, Christian Bontveit

## References

[solafune-tools](https://github.com/Solafune-Inc/solafune-tools/tree/main) by Solafune team