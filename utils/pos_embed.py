# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, cls_token=False):
    """
    Generate 2D sine-cosine positional embeddings for non-square grids.
    Args:
        embed_dim (int): Embedding dimension.
        grid_h (int): Grid height.
        grid_w (int): Grid width.
        cls_token (bool): Whether to include a class token.
    Returns:
        pos_embed (np.ndarray): Positional embeddings of shape (grid_h * grid_w, embed_dim).
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even."

    # Generate grid coordinates
    grid_h = np.arange(grid_h, dtype=float)
    grid_w = np.arange(grid_w, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # (2, grid_h, grid_w)
    grid = np.stack(grid, axis=0)  # (2, grid_h, grid_w)

    grid = grid.reshape(2, -1)  # (2, 1, grid_h, grid_w)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    # Add cls token if required
    if cls_token:
        cls_pos_embed = np.zeros((1, embed_dim), dtype=float)
        pos_embed = np.concatenate([cls_pos_embed, pos_embed], axis=0)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model, source='image', new_grid_size=(4,8)):
    if source=='image' and 'image_pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['image_pos_embed'] # [1, 50, 768]
        embedding_size = pos_embed_checkpoint.shape[-1] # 768 for ViT-B/32
        num_patches = model.audio_patch_embed.num_patches # 32 (4*8)
        num_extra_tokens = model.audio_pos_embed.shape[-2] - num_patches # 33 - 32 = 1
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)  # sqrt(50 - 1) = 7
        new_size = new_grid_size  # (4, 8) for 128x256, 32x32 patches'

        if orig_size ** 2 != num_patches or num_patches != new_size[0] * new_size[1]:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size[0], new_size[1]))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]  # [1, 1, 768]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]  # [1, 49, 768]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)  # [1, 768, 7, 7]
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=new_size, mode='bicubic', align_corners=False)  # [1, 768, 4, 8]
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)  # [1, 32, 768]
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)  # [1, 33, 768]
            checkpoint_model['audio_pos_embed'] = new_pos_embed

    elif source=='audio' and 'audio_pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['audio_pos_embed'] # [1, 33, 768]
        embedding_size = pos_embed_checkpoint.shape[-1] # 768 for ViT-B/32
        num_patches = model.image_patch_embed.num_patches # 49 (7*7)
        num_extra_tokens = model.image_pos_embed.shape[-2] - num_patches # 50 - 49 = 1
        orig_size = (4, 8)
        new_size = new_grid_size # (7, 7) for 224x224, 32x32 patches
        if orig_size[0] * orig_size[1] != num_patches or num_patches != new_size[0] * new_size[1]:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size[0], orig_size[1], new_size[0], new_size[1]))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens] # [1, 1, 768]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:] # [1, 32, 768]
            pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)  # [1, 768, 4, 8]
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=new_size, mode='bicubic', align_corners=False)  # [1, 768, 7, 7] 
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)  # [1, 49, 768]
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)  # [1, 50, 768]
            checkpoint_model['image_pos_embed'] = new_pos_embed