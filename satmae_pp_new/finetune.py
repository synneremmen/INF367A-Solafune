def freeze_layers(model, num_layers_to_freeze):
    """
    Freeze the specified number of layers in the model.
    
    Args:
        model: The model whose layers are to be frozen.
        num_layers_to_freeze: The number of layers to freeze.
    """
    # 2a) load the checkpoint
    ckpt = torch.load("checkpoint_ViT-L_pretrain_fmow_sentinel.pth", map_location="cpu")["model"]

    # 2b) filter out keys that don’t match (your seg head, old cls head, any decoder weights)
    model_dict = model.state_dict()
    filtered_ckpt = {
        k: v for k, v in ckpt.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    # 2c) interpolate pos_embed if resolution changed
    from util.pos_embed import interpolate_pos_embed
    interpolate_pos_embed(model, filtered_ckpt)

    # 2d) load
    model.load_state_dict(filtered_ckpt, strict=False)

    for p in model.parameters():
        p.requires_grad = False

    # choose how many of the last transformer blocks to unfreeze—for example the last 4 out of 24:
    for blk in model.blocks[-4:]:
        for p in blk.parameters():
            p.requires_grad = True

    # and of course your new segmentation head:
    for p in model.seg_head.parameters():
        p.requires_grad = True

