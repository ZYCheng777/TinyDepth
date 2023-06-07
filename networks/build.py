# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------


from .tiny_vit import TinyViT


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'tiny_vit':
        M = config.MODEL.TINY_VIT
        model = TinyViT(
                        img_width=640,
                        img_height=192,
                        in_chans=M.IN_CHANS,
                        embed_dims=M.EMBED_DIMS,
                        depths=M.DEPTHS,
                        num_heads=M.NUM_HEADS,
                        window_sizes=M.WINDOW_SIZES,
                        mlp_ratio=M.MLP_RATIO,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        use_checkpoint=False,
                        mbconv_expand_ratio=M.MBCONV_EXPAND_RATIO,
                        local_conv_size=M.LOCAL_CONV_SIZE,
                        layer_lr_decay=config.TRAIN.LAYER_LR_DECAY,
                        pretrain_path=config.MODEL.PRETRAINED,
                        )

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model

