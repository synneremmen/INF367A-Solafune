from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer as _VisionTransformer
from satmae_pp.satmae_pp import get_2d_sincos_pos_embed, load_freeze_layers


class VisionTransformerSeg(_VisionTransformer):
    """
    Extends the VisionTransformer to include a segmentation head
    """
    def __init__(self, num_classes, patch_size, img_size, in_chans, **kwargs):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            **kwargs
        )
        # reinitialize positional embedding to 2D sincos
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.patch_size = patch_size
        self.img_size   = img_size
        self.num_classes = num_classes
        embed_dim = self.embed_dim

        # 1x1 conv head
        self.seg_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.patch_embed(x)
        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = x[:, 1:, :].transpose(1, 2)  # (B, D, N)
        h = w = self.img_size // self.patch_size
        x = x.reshape(B, -1, h, w)  # (B, D, h, w)
        x = self.seg_head(x)  # (B, num_classes, h, w)
        # upsample to original size
        return F.interpolate(
            x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False
        )


def vit_seg_large_patch16(num_classes, patch_size, img_size, in_chans, **kwargs):
    """
    Creates a large ViT model for the deforestation dataset
    """
    return VisionTransformerSeg(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        patch_size=patch_size,
        img_size=img_size,
        in_chans=in_chans,
        **kwargs
    )

def make_vit_finetune(
    ckpt_path, num_classes:int=5, patch_size:int=16, img_size:int=1024, in_chans:int=12, n_trainable_layers:int=2
):
    """
    Creates a ViT model for pretraining on the deforestation dataset 
    """
    model = vit_seg_large_patch16(
        num_classes=num_classes,
        patch_size=patch_size,
        img_size=img_size,
        in_chans=in_chans
    )
    return load_freeze_layers(model, n_trainable_layers=n_trainable_layers, ckpt_path=ckpt_path)