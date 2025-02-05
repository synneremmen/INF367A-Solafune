# INF367A-Identifying Deforestation drivers


This project is a submission to [Solafune competition](https://solafune.com/competitions/68ad4759-4686-4bb3-94b8-7063f755b43d?menu=about&tab=&topicId=a5e978e7-7759-4433-b1a5-063760451ff5).

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

## Required packages (ADD MORE IF NECESSARY)

[solafune-tools](https://github.com/Solafune-Inc/solafune-tools/tree/main)

pytorch

torchvision

numpy

matplotlib

## How to run project (ADD MORE DETAIL)

1. Run the create_masked_data.py to get images in the masked_annotations folder. 
These tif files contains only one band of size 1024x1024. Each pixel in the data contains the class value (0-4, where 0 is none.)

## Submission and evaluation

## Contributors

Aurora Ingebrigtsen, Synne Remmen Andreassen, Christian Bontveit

## References

[solafune-tools](https://github.com/Solafune-Inc/solafune-tools/tree/main) by Solafune team

[solafune-baseline](https://github.com/motokimura/solafune_deforestation_baseline/tree/main) by Moto Kimura
