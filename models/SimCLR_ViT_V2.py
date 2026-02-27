import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from functools import partial
import numpy as np
from utils.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


class Conv2dPatchEmbed(nn.Module):
    """2D Convolutional Patch Embedding"""
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size if img_size[0] != 1 else 1, img_size[1] // patch_size if img_size[1] != 1 else 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class SimCLRViTV2(nn.Module):
    """SimCLR with VisionTransformer backbone for multimodal inputs"""
    def __init__(self, 
                 img_size=(64, 64), audio_size=(512, 128), text_seq_len=128, 
                 patch_size=8, img_in_chans=3, audio_in_chans=1, text_vocab_size=30522,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, split_projector=False, directclr=False):
        super().__init__()

        self.split_projector = split_projector
        self.directclr = directclr

        # Shared encoder specifics
        self.embed_dim = embed_dim

        # Image patch embedding
        self.image_patch_embed = Conv2dPatchEmbed(img_size, patch_size, img_in_chans, embed_dim)
        num_image_patches = self.image_patch_embed.num_patches

        # Audio patch embedding
        self.audio_patch_embed = Conv2dPatchEmbed(audio_size, patch_size, audio_in_chans, embed_dim)
        num_audio_patches = self.audio_patch_embed.num_patches

        # Text embedding
        self.text_embed = Conv2dPatchEmbed((text_seq_len, 1), 1, 1, embed_dim)
        num_text_patches = self.text_embed.num_patches
        self.text_pos_embed = nn.Parameter(torch.zeros(1, num_text_patches + 1, embed_dim), requires_grad=False)

        # Positional embeddings for image and audio
        self.image_pos_embed = nn.Parameter(torch.zeros(1, num_image_patches + 1, embed_dim), requires_grad=False)
        self.audio_pos_embed = nn.Parameter(torch.zeros(1, num_audio_patches + 1, embed_dim), requires_grad=False)

        # Shared cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Shared transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

        # Projection head for contrastive learning
        if self.split_projector:
            self.projection_head_image = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            self.projection_head_audio = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            self.projection_head_text = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )

    def initialize_weights(self):
        # Initialize positional embeddings for image (encoder)
        image_pos_embed = get_2d_sincos_pos_embed(
            self.image_pos_embed.shape[-1],
            int(self.image_patch_embed.grid_size[0]), 
            int(self.image_patch_embed.grid_size[1]), 
            cls_token=True
        )
        self.image_pos_embed.data.copy_(torch.from_numpy(image_pos_embed).float().unsqueeze(0))

        # Initialize positional embeddings for audio (encoder, non-square)
        audio_pos_embed = get_2d_sincos_pos_embed(
            self.audio_pos_embed.shape[-1], 
            self.audio_patch_embed.grid_size[0], 
            self.audio_patch_embed.grid_size[1], 
            cls_token=True
        )
        self.audio_pos_embed.data.copy_(torch.from_numpy(audio_pos_embed).float().unsqueeze(0))

        # Initialize text positional embedding (encoder) using 1D sine-cosine
        text_pos_embed = get_2d_sincos_pos_embed(
            self.text_pos_embed.shape[-1], 
            self.text_embed.grid_size[0], 
            self.text_embed.grid_size[1], 
            cls_token=True
        )
        self.text_pos_embed.data.copy_(torch.from_numpy(text_pos_embed).float().unsqueeze(0))

        # Initialize cls token and mask token
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # Initialize transformer weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, modality):
        """
        Forward pass through the encoder.
        Args:
            x: Input data (image, audio, or text).
            modality: Modality type ('image', 'audio', or 'text').
        Returns:
            cls_representation: CLS token representation.
        """
        if modality == 'image':
            x = self.image_patch_embed(x)
            pos_embed = self.image_pos_embed
        elif modality == 'audio':
            x = self.audio_patch_embed(x)
            pos_embed = self.audio_pos_embed
        elif modality == 'text':
            x = x.unsqueeze(1).unsqueeze(1).float() # [B, seq_len] → [B, 1, 1, seq_len] or reshape to 2D
            x = self.text_embed(x)
            pos_embed = self.text_pos_embed
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        if pos_embed is not None:
            x = x + pos_embed[:, 1:, :]

        cls_token = self.cls_token + (pos_embed[:, :1, :] if pos_embed is not None else 0)
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Use the CLS token as the representation
        cls_representation = x[:, 0]
        return cls_representation

    def forward_projector(self, x, modality):
        cls_token = self.forward_encoder(x, modality)

        if self.split_projector:
            if modality == 'image':
                cls_token = self.projection_head_image(cls_token)
            elif modality == 'audio':
                cls_token = self.projection_head_audio(cls_token)
            elif modality == 'text':
                cls_token = self.projection_head_text(cls_token)
        else:
            cls_token = self.projection_head(cls_token)
        return cls_token

    def forward_embedding(self, x, modality):
        """
        Forward pass to get embeddings before projection head.
        Args:
            x: Input data (image, audio, or text).
            modality: Modality type ('image', 'audio', or 'text').
        Returns:
            cls_representation: CLS token representation.
        """

        h = self.forward_encoder(x, modality)
        if self.split_projector:
            if modality == 'image':
                z = self.projection_head_image(h)
            elif modality == 'audio':
                z = self.projection_head_audio(h)
            elif modality == 'text':
                z = self.projection_head_text(h)
        elif self.directclr:
            z = h
        else:
            z = self.projection_head(h)
        return z

    def forward(self, x1, x2, modality):
        """
        Forward pass for SimCLR.
        Args:
            x1: First augmented view of the input.
            x2: Second augmented view of the input.
            modality: Modality (e.g., 'image', 'audio', 'text').
        Returns:
            z1, z2: Projected embeddings for the two views.
        """
        # Encode both views
        h1 = self.forward_encoder(x1, modality)
        h2 = self.forward_encoder(x2, modality)

        # Project embeddings
        if self.split_projector:
            if modality == 'image':
                z1 = self.projection_head_image(h1)
                z2 = self.projection_head_image(h2)
            elif modality == 'audio':
                z1 = self.projection_head_audio(h1)
                z2 = self.projection_head_audio(h2)
            elif modality == 'text':
                z1 = self.projection_head_text(h1)
                z2 = self.projection_head_text(h2)
        elif self.directclr:
            z1 = h1
            z2 = h2
        else:
            z1 = self.projection_head(h1)
            z2 = self.projection_head(h2)

        return z1, z2