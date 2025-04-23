from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer as _VisionTransformer
from util.pos_embed import get_2d_sincos_pos_embed


class VisionTransformerSeg(_VisionTransformer):
    """
    Extends the VisionTransformer to produce a segmentation map via a 1x1 conv head.
    """
    def __init__(self, num_classes, patch_size, img_size, **kwargs):
        super().__init__(
            img_size=img_size,   # ← forward this
            patch_size=patch_size, # ← and this
            in_chans=12,    # ← and the channel count
            **kwargs             # your embed_dim, depth, num_heads, etc.
        )
        # reinitialize positional embedding to 2D sincos
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # segmentation parameters
        self.patch_size = patch_size
        self.img_size   = img_size
        self.num_classes = num_classes
        embed_dim = self.embed_dim

        # define a lightweight 1x1 conv head
        self.seg_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        # embed into patches
        x = self.patch_embed(x)
        B = x.shape[0]
        # add cls token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # drop cls token, reshape
        x = x[:, 1:, :].transpose(1, 2)  # (B, D, N)
        h = w = self.img_size // self.patch_size
        x = x.reshape(B, -1, h, w)       # (B, D, h, w)
        # segmentation logits
        x = self.seg_head(x)             # (B, num_classes, h, w)
        # upsample to original size
        return F.interpolate(
            x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False
        )


def vit_seg_base_patch16(num_classes, patch_size, img_size, **kwargs):
    return VisionTransformerSeg(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        patch_size=patch_size,
        img_size=img_size,
        **kwargs
    )

def vit_seg_large_patch16(num_classes, patch_size, img_size, **kwargs):
    return VisionTransformerSeg(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        patch_size=patch_size,
        img_size=img_size,
        **kwargs
    )