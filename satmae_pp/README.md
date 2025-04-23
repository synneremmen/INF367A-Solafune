# SatMAE++ Implementation

# Overview

The SatMAE++ framework was introduced in ["Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery"](https://arxiv.org/abs/2403.05419) (Noman et al., 2024). This folder contains the code, installation and setup guide necessary for running the finetuning part of the framework on the Solafune deforestation drivers competition. I use the pretrained model with weights provided by the authors on GitHub at [techmn/satmae_pp.](https://github.com/techmn/satmae_pp).

<img width="1096" alt="image" src="images/overall_architecture.png">

## Installation and Setup

1. **Clone the SatMAE++ repository and install dependencies**

   ```bash
   git clone https://github.com/techmn/satmae_pp.git
   cd satmae_pp
   python -m venv .venv
   source .venv/bin/activate    # Linux/macOS
   # .\.venv\Scripts\activate   # Windows 
   pip install -r requirements.txt
    ```

2. **Download the ViT-Large [pretrained weights](https://huggingface.co/mubashir04/checkpoint_ViT-L_pretrain_fmow_sentinel) from hugging face**
    ```bash
    wget -O checkpoint_ViT-L_pretrain_fmow_sentinel.pth \
    https://huggingface.co/mubashir04/checkpoint_ViT-L_pretrain_fmow_sentinel/resolve/main/pytorch_model.bin
    ```

3. **Move the weights into the repo**
    ```bash
    mv checkpoint_ViT-L_pretrain_fmow_sentinel.pth .
    ```

4. **Copy custom scripts into the repo**<br>
    Copy the custom engine, entrypoint and models from this folder into the satmae_pp folder.


## Usage
To reproduce the finetuning run the following command

```bash
python main_segmentation.py \
  --device cpu \
  --input_size 1024 \
  --patch_size 16 \
  --nb_classes 5 \
  --finetune ./checkpoint_ViT-L_finetune_fmow_sentinel.pth \
  --epochs 10 \
  --batch_size 8 \
  --mixup 0.0 \
  --cutmix 0.0
```

Alternatively, load the finetuned weights from PATH.

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
