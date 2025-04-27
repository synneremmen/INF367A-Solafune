# SatMAE++ Individual Implementation

## Overview

The SatMAE++ framework was introduced in ["Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery"](https://arxiv.org/abs/2403.05419) (Noman et al., 2024). This folder contains the code, installation and setup guide necessary for running the finetuning part of the framework on the Solafune deforestation drivers competition. I use the pretrained model with weights provided by the authors on GitHub at [techmn/satmae_pp.](https://github.com/techmn/satmae_pp).

## Method

DESCRIPTION

<img width="1096" alt="image" src="overall_architecture.png">




## Installation and Setup

1. **Install global requirements**

    Ensure dependencies is installed by the global requirements.txt

2. **Download the ViT-Large [pretrained weights](https://huggingface.co/mubashir04/checkpoint_ViT-L_pretrain_fmow_sentinel) from hugging face**
    The weights should be placed inside the satmae_pp folder.
    ```bash
    wget -O checkpoint_ViT-L_pretrain_fmow_sentinel.pth \
    https://huggingface.co/mubashir04/checkpoint_ViT-L_pretrain_fmow_sentinel/resolve/main/pytorch_model.bin
    ```



## Usage
To reproduce the finetuning run the following command


## Citation

```
@inproceedings{satmaepp2024rethinking,
      title={Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery}, 
      author={Mubashir Noman and Muzammal Naseer and Hisham Cholakkal and Rao Muhammad Anwar and Salman Khan and Fahad Shahbaz Khan},
      year={2024},
      booktitle={CVPR}
}
```

> techmn. _satmae_pp_. GitHub. 2025. https://github.com/techmn/satmae_pp
