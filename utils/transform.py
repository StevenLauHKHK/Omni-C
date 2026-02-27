import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
try:
    from timm.data.transforms import str_to_pil_interp as _pil_interp
except:
    from timm.data.transforms import _pil_interp



def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        if config.AUG.PRESET is None:
            transform = create_transform(
                input_size=config.DATA.IMG_SIZE,
                is_training=True,
                color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG.REPROB,
                re_mode=config.AUG.REMODE,
                re_count=config.AUG.RECOUNT,
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            print('=============================== original AUG! ', config.AUG.AUTO_AUGMENT)
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        
        elif config.AUG.PRESET.strip() == 'simclr':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
            print('---------------------- SimCLR-style augmentations!')
        
        elif config.AUG.PRESET.strip() == 'raug15':
            from randaug import RandAugPolicy
            transform = transforms.Compose([
                transforms.RandomResizedCrop(config.DATA.IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                RandAugPolicy(magnitude=15),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
            print('---------------------- RAND AUG 15 distortion!')

        elif config.AUG.PRESET.strip() == 'weak':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(config.DATA.IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
        elif config.AUG.PRESET.strip() == 'none':
            transform = transforms.Compose([
                transforms.Resize(config.DATA.IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(config.DATA.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ])
        else:
            raise ValueError('???' + config.AUG.PRESET)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR))
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            #   default for testing
            t.append(transforms.Resize(config.DATA.IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR))
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    trans = transforms.Compose(t)
    return trans
