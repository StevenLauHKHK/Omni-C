# --------------------------------------------------------
# Re-parameterizing Your Optimizers rather than Architectures (https://arxiv.org/abs/2205.15242)
# Github source: https://github.com/DingXiaoH/RepOptimizers
# Licensed under The MIT License [see LICENSE for details]
# The training script is based on the code of Swin Transformer (https://github.com/microsoft/Swin-Transformer)
# --------------------------------------------------------

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
_C.DATA.IMAGE_DATA_PATH = '/data_ssd/DATA/imagenet1k'
_C.DATA.IMAGE_DATASET = 'imagenet'
_C.DATA.AUDIO_DATA_PATH = '/data_ssd/DATA/audioset'
_C.DATA.AUDIO_DATA_PATH_2 = '/data1/DATA/LibriSpeech'
_C.DATA.AUDIO_DATASET = 'audioset-librispeech'
_C.DATA.TEXT_DATA_PATH = '/data_ssd/DATA/AG_NEWS'
_C.DATA.TEXT_DATASET = 'agnews'

# Input image size
_C.DATA.IMG_SIZE = 64
_C.DATA.NUM_CLASSES = None

_C.DATA.TEST_BATCH_SIZE = 512
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bilinear'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = False
# Number of data loading threads
_C.DATA.NUM_WORKERS = 32
_C.DATA.RETURN_TYPE = 'no_pt'  # Options: 'no_pt', 'pt'


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300


# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adam'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# Optimizer Weight decay
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 0.05
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.OPTIMIZER.BASE_LR = 1e-3
_C.TRAIN.OPTIMIZER.LR = None
_C.TRAIN.OPTIMIZER.MIN_LR = 0.0


# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.WARMUP_EPOCHS = 40

# -----------------------------------------------------------------------------
# Audio settings
# -----------------------------------------------------------------------------
_C.AUDIO = CN()
_C.AUDIO.MELBINS = 128
_C.AUDIO.FREQM = 0
_C.AUDIO.TIMEM = 0 # 128 # 512
_C.AUDIO.TARGET_LENGTH = 256
_C.AUDIO.MIXUP = 0
_C.AUDIO.NORM_MEAN = -4.2677393
_C.AUDIO.NORM_STD = 4.5689974
_C.AUDIO.SKIP_NORM = False
_C.AUDIO.NOISE = False
_C.AUDIO.CLUSTER = True
_C.AUDIO.TASK = 'mae'
_C.AUDIO.MAE_WEIGHT = 10.0
_C.AUDIO.PRETRAIN_CROP_AND_AUGMENT = False


# -----------------------------------------------------------------------------
# Text settings
# -----------------------------------------------------------------------------
_C.TEXT = CN()
_C.TEXT.ENABLE_NSP = True
_C.TEXT.MAX_SEQ_LENGTH = 256
_C.TEXT.NSP_WEIGHT = 1.0
_C.TEXT.TASK = 'nsp_mae'


# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.0
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 0.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

_C.AUG.PRESET = 'none'    # If use AUG.PRESET (e.g., 'raug15'), use the pre-defined preprocessing, ignoring the following settings.
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1


# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = False


def update_config(config, args):
    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.image_data_path:
        config.DATA.IMAGE_DATA_PATH = args.image_data_path
    if args.audio_data_path:
        config.DATA.AUDIO_DATA_PATH = args.audio_data_path
    if args.text_data_path:
        config.DATA.TEXT_DATA_PATH = args.text_data_path
    if args.test_batch_size:
        config.DATA.TEST_BATCH_SIZE = args.test_batch_size
    if args.image_dataset:
        config.DATA.IMAGE_DATASET = args.image_dataset
    if args.audio_dataset:
        config.DATA.AUDIO_DATASET = args.audio_dataset
    if args.text_dataset:
        config.DATA.TEXT_DATASET = args.text_dataset
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
