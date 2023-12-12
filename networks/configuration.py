# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'


import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size

# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()



#tiny-ViT
_C.MODEL.TYPE = 'tiny_vit'
# Model name
_C.MODEL.NAME = 'tiny_vit'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = '/home/ace/workspace/TinyDepth/networks/tiny_vit_5m_22k_distill_depth.pth'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# TinyViT Model
_C.MODEL.TINY_VIT = CN()
_C.MODEL.TINY_VIT.IN_CHANS = 3
_C.MODEL.TINY_VIT.DEPTHS = [2, 2, 6, 2]
_C.MODEL.TINY_VIT.NUM_HEADS = [2, 4, 5, 10]
_C.MODEL.TINY_VIT.WINDOW_SIZES = [7, 7, 14, 7]
_C.MODEL.TINY_VIT.EMBED_DIMS = [64, 128, 160, 320]
_C.MODEL.TINY_VIT.MLP_RATIO = 4.
_C.MODEL.TINY_VIT.MBCONV_EXPAND_RATIO = 4.0
_C.MODEL.TINY_VIT.LOCAL_CONV_SIZE = 3

_C.TRAIN = CN()
_C.TRAIN.LAYER_LR_DECAY = 0.05





def update_config(config, args):
    '''
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    '''


    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
