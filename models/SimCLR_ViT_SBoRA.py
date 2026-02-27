import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from functools import partial
import numpy as np
from utils.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


class SBoRALayer(nn.Module):
    """SBoRA (Standard-Basis LoRA) layer"""
    def __init__(self, in_features, out_features, rank=4, alpha=16, dropout=0.1, mode='FA'):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.mode = mode
        self.in_features = in_features
        self.out_features = out_features

        if mode == 'FA':
            # Freeze A (use standard basis), train B
            # Select random indices for standard basis A
            indices = torch.randperm(in_features)[:rank]
            self.register_buffer('A_indices', indices)

            # Trainable B matrix (zero-initialized)
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            
        elif mode == 'FB':
            # Freeze B (use standard basis), train A
            # Trainable A matrix (zero-initialized)
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))

            # Store indices instead of one-hot matrix
            indices = torch.randperm(out_features)[:rank]
            self.register_buffer('B_indices', indices)

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'FA' or 'FB'")
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        # x: [batch_size, seq_len, in_features]
        if self.mode == 'FA':
            # Direct indexing: much faster than matrix multiplication
            # Select specific input dimensions using advanced indexing
            x_selected = self.dropout(x)[..., self.A_indices]  # [B, S, rank]
            result = x_selected @ self.lora_B.T  # [B, S, out_features]
            
        elif self.mode == 'FB':
            # Use einsum for efficient computation
            x_dropped = self.dropout(x)
            temp = torch.einsum('bsi,ri->bsr', x_dropped, self.lora_A)  # [B, S, rank]
            result = torch.zeros(x.shape[0], x.shape[1], self.out_features, device=x.device, dtype=x.dtype)
            result[..., self.B_indices] = temp
            
        return result * self.scaling


class SBoRALinear(nn.Module):
    """Linear layer with SBoRA adaptation"""
    def __init__(self, linear_layer, rank=4, alpha=16, dropout=0.1, mode='FA'):
        super().__init__()
        self.linear = linear_layer
        self.sbora = SBoRALayer(
            linear_layer.in_features, 
            linear_layer.out_features, 
            rank, alpha, dropout, mode
        )
        self.use_sbora = True
        
        # Freeze original parameters
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        result = self.linear(x)
        if self.use_sbora:
            result += self.sbora(x)
        return result
    
    def enable_sbora(self):
        self.use_sbora = True
        
    def disable_sbora(self):
        self.use_sbora = False


class SBoRABlock(Block):
    """Vision Transformer Block with SBoRA adaptation"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, proj_drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sbora_config=None):
        super().__init__(
            dim=dim, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            drop_path=drop_path, 
            act_layer=act_layer, 
            norm_layer=norm_layer
        )
        
        if sbora_config is not None:
            # Replace attention layers with SBoRA versions
            self.attn.qkv = SBoRALinear(
                self.attn.qkv, 
                rank=sbora_config.get('rank', 4),
                alpha=sbora_config.get('alpha', 16),
                dropout=sbora_config.get('dropout', 0.1),
                mode=sbora_config.get('mode', 'FA')
            )
            self.attn.proj = SBoRALinear(
                self.attn.proj,
                rank=sbora_config.get('rank', 4),
                alpha=sbora_config.get('alpha', 16),
                dropout=sbora_config.get('dropout', 0.1),
                mode=sbora_config.get('mode', 'FA')
            )
            
            # Replace MLP layers with SBoRA versions
            self.mlp.fc1 = SBoRALinear(
                self.mlp.fc1,
                rank=sbora_config.get('rank', 4),
                alpha=sbora_config.get('alpha', 16),
                dropout=sbora_config.get('dropout', 0.1),
                mode=sbora_config.get('mode', 'FA')
            )
            self.mlp.fc2 = SBoRALinear(
                self.mlp.fc2,
                rank=sbora_config.get('rank', 4),
                alpha=sbora_config.get('alpha', 16),
                dropout=sbora_config.get('dropout', 0.1),
                mode=sbora_config.get('mode', 'FA')
            )


class Conv2dPatchEmbed(nn.Module):
    """2D Convolutional Patch Embedding"""
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class SimCLRViT_SBoRA(nn.Module):
    """SimCLR with VisionTransformer backbone for multimodal inputs"""
    def __init__(self, 
                 img_size=(64, 64), audio_size=(512, 128), text_seq_len=128, 
                 patch_size=8, img_in_chans=3, audio_in_chans=1, text_vocab_size=30522,
                 embed_dim=1024, out_dim=768, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, sbora_config=None, split_projector=False, directclr=False):
        super().__init__()

        self.split_projector = split_projector
        self.directclr = directclr

        # Shared encoder specifics
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        # Image patch embedding
        self.image_patch_embed = Conv2dPatchEmbed(img_size, patch_size, img_in_chans, embed_dim)
        num_image_patches = self.image_patch_embed.num_patches

        # Audio patch embedding
        self.audio_patch_embed = Conv2dPatchEmbed(audio_size, patch_size, audio_in_chans, embed_dim)
        num_audio_patches = self.audio_patch_embed.num_patches

        # Text embedding
        self.text_embed = nn.Embedding(text_vocab_size, embed_dim)
        self.text_pos_embed = nn.Parameter(torch.zeros(1, text_seq_len, embed_dim), requires_grad=False)

        # Positional embeddings for image and audio
        self.image_pos_embed = nn.Parameter(torch.zeros(1, num_image_patches + 1, embed_dim), requires_grad=False)
        self.audio_pos_embed = nn.Parameter(torch.zeros(1, num_audio_patches + 1, embed_dim), requires_grad=False)

        # Shared cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Store SBoRA config
        self.sbora_config = sbora_config

        # Shared transformer blocks with optional SBoRA
        if sbora_config is not None:
            self.blocks = nn.ModuleList([
                SBoRABlock(
                    dim=embed_dim, 
                    num_heads=num_heads, 
                    mlp_ratio=mlp_ratio, 
                    qkv_bias=True,
                    norm_layer=norm_layer, 
                    sbora_config=sbora_config
                )
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(depth)
            ])
        
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

        # Projection head for contrastive learning with optional SBoRA
        if sbora_config is not None:
            if self.split_projector:
                self.projection_head_image = nn.Sequential(
                    SBoRALinear(nn.Linear(embed_dim, embed_dim), 
                               rank=sbora_config.get('rank', 4),
                               alpha=sbora_config.get('alpha', 16),
                               dropout=sbora_config.get('dropout', 0.1),
                               mode=sbora_config.get('mode', 'FA')),
                    nn.ReLU(),
                    SBoRALinear(nn.Linear(embed_dim, out_dim), 
                               rank=sbora_config.get('rank', 4),
                               alpha=sbora_config.get('alpha', 16),
                               dropout=sbora_config.get('dropout', 0.1),
                               mode=sbora_config.get('mode', 'FA')),
                )
                self.projection_head_audio = nn.Sequential(
                    SBoRALinear(nn.Linear(embed_dim, embed_dim), 
                               rank=sbora_config.get('rank', 4),
                               alpha=sbora_config.get('alpha', 16),
                               dropout=sbora_config.get('dropout', 0.1),
                               mode=sbora_config.get('mode', 'FA')),
                    nn.ReLU(),
                    SBoRALinear(nn.Linear(embed_dim, out_dim), 
                               rank=sbora_config.get('rank', 4),
                               alpha=sbora_config.get('alpha', 16),
                               dropout=sbora_config.get('dropout', 0.1),
                               mode=sbora_config.get('mode', 'FA')),
                )
                self.projection_head_text = nn.Sequential(
                    SBoRALinear(nn.Linear(embed_dim, embed_dim), 
                               rank=sbora_config.get('rank', 4),
                               alpha=sbora_config.get('alpha', 16),
                               dropout=sbora_config.get('dropout', 0.1),
                               mode=sbora_config.get('mode', 'FA')),
                    nn.ReLU(),
                    SBoRALinear(nn.Linear(embed_dim, out_dim), 
                               rank=sbora_config.get('rank', 4),
                               alpha=sbora_config.get('alpha', 16),
                               dropout=sbora_config.get('dropout', 0.1),
                               mode=sbora_config.get('mode', 'FA')),
                )
            elif not self.directclr:
                projection_linear_1 = nn.Linear(embed_dim, embed_dim)
                projection_linear_2 = nn.Linear(embed_dim, out_dim)
                self.projection_head = nn.Sequential(
                    SBoRALinear(projection_linear_1, 
                            rank=sbora_config.get('rank', 4),
                            alpha=sbora_config.get('alpha', 16),
                            dropout=sbora_config.get('dropout', 0.1),
                            mode=sbora_config.get('mode', 'FA')),
                    nn.ReLU(),
                    SBoRALinear(projection_linear_2, 
                            rank=sbora_config.get('rank', 4),
                            alpha=sbora_config.get('alpha', 16),
                            dropout=sbora_config.get('dropout', 0.1),
                            mode=sbora_config.get('mode', 'FA')),
                )
        else:
            if self.split_projector:
                self.projection_head_image = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, out_dim)
                )
                self.projection_head_audio = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, out_dim)
                )
                self.projection_head_text = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, out_dim)
                )
            elif not self.directclr:
                self.projection_head = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, out_dim)
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
        text_pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.text_pos_embed.shape[-1], 
            np.arange(self.text_pos_embed.shape[1], dtype=float)
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
            x = self.text_embed(x) + self.text_pos_embed
            pos_embed = None
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
            z2 = h2s
        else:
            z1 = self.projection_head(h1)
            z2 = self.projection_head(h2)

        return z1, z2
    
    def enable_sbora(self):
        """Enable SBoRA adaptation"""
        for module in self.modules():
            if isinstance(module, SBoRALinear):
                module.enable_sbora()
                
    def disable_sbora(self):
        """Disable SBoRA adaptation"""
        for module in self.modules():
            if isinstance(module, SBoRALinear):
                module.disable_sbora()
    
    def get_sbora_parameters(self):
        """Get all SBoRA parameters for optimization during training"""
        sbora_params = []
        sbora_keys = []
        for name, module in self.named_modules():
            if isinstance(module, SBoRALayer):
                # Collect trainable parameters based on mode
                if module.mode == 'FA' and hasattr(module, 'lora_B'):
                    sbora_params.append(module.lora_B)
                    sbora_keys.append(f"{name}.lora_B")
                elif module.mode == 'FB' and hasattr(module, 'lora_A'):
                    sbora_params.append(module.lora_A)
                    sbora_keys.append(f"{name}.lora_A")

        print(f"Found {len(sbora_params)} SBoRA parameters for optimization")
        return sbora_params, sbora_keys
    
    def freeze_base_model(self):
        """Freeze all non-LoRA parameters"""
        for name, param in self.named_parameters():
            if 'sbora' not in name and 'patch_embed' not in name:
                param.requires_grad = False
            else:
                print(f"Parameter {name} is trainable")
    
    def unfreeze_base_model(self):
        """Unfreeze all non-LoRA parameters"""
        for name, param in self.named_parameters():
            if 'sbora' not in name and 'pos_embed' not in name:
                param.requires_grad = True

    def freeze_sbora_parameters(self):
        """Freeze all LoRA parameters"""
        for name, param in self.named_parameters():
            if 'lora_B' in name or 'lora_A' in name:
                param.requires_grad = False
                print(f"SBoRA Parameter {name} is {param.requires_grad}")

    def unfreeze_sbora_parameters(self):
        """Freeze all LoRA parameters"""
        for name, param in self.named_parameters():
            if 'lora_B' in name or 'lora_A' in name:
                param.requires_grad = True

    def freeze_patch_embedding(self):
        """Freeze patch embedding layers"""
        for name, param in self.named_parameters():
            if 'patch_embed' in name:
                param.requires_grad = False
                print(f"Patch Embedding Parameter {name} is {param.requires_grad}")
            elif 'text_embed' in name:
                param.requires_grad = False
                print(f"Text Embedding Parameter {name} is {param.requires_grad}")

    def unfreeze_patch_embedding(self):
        """Unfreeze patch embedding layers"""
        for name, param in self.named_parameters():
            if 'patch_embed' in name:
                param.requires_grad = True
                print(f"Patch Embedding Parameter {name} is {param.requires_grad}")
            
    def save_sbora_weights(self, path):
        """Save only SBoRA weights and buffers"""
        sbora_state_dict = {}
        
        # Save SBoRA parameters
        for name, param in self.named_parameters():
            if 'lora_' in name:
                sbora_state_dict[name] = param.data
        
        # Save SBoRA buffers (A_indices, B_indices)
        for name, buffer in self.named_buffers():
            if 'indices' in name:
                sbora_state_dict[name] = buffer
                
        torch.save(sbora_state_dict, path)
        print(f"Saved SBoRA weights and buffers to {path}")

    def load_sbora_weights(self, path, device='cpu'):
        """Load SBoRA weights"""
        sbora_state_dict = torch.load(path, map_location=device)
        missing_keys, unexpected_keys = self.load_state_dict(sbora_state_dict, strict=False)
        print(f"Loaded SBoRA weights. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
    
    def load_pretrained_weights(self, checkpoint_path, device='cpu'):
        """
        Simple method to load pretrained SimCLR weights
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        print(f"Loading pretrained weights from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract state dict from different checkpoint formats
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint
        
        # Remove 'module.' prefix if present
        clean_dict = {}
        for key, value in pretrained_dict.items():
            clean_key = key.replace('module.', '') if key.startswith('module.') else key
            # attn.qkv in key, we replace as attn.qkv.linear
            clean_key = clean_key.replace('attn.qkv', 'attn.qkv.linear') if 'attn.qkv' in key else clean_key
            clean_key = clean_key.replace('attn.proj', 'attn.proj.linear') if 'attn.proj' in key else clean_key
            clean_key = clean_key.replace('mlp.fc1', 'mlp.fc1.linear') if 'mlp.fc1' in key else clean_key
            clean_key = clean_key.replace('mlp.fc2', 'mlp.fc2.linear') if 'mlp.fc2' in key else clean_key
            clean_key = clean_key.replace('projection_head.0', 'projection_head.0.linear') if 'projection_head.0' in key else clean_key
            clean_key = clean_key.replace('projection_head.2', 'projection_head.2.linear') if 'projection_head.2' in key else clean_key
            clean_key = clean_key.replace('projection_head_image.0', 'projection_head_image.0.linear') if 'projection_head_image.0' in key else clean_key
            clean_key = clean_key.replace('projection_head_image.2', 'projection_head_image.2.linear') if 'projection_head_image.2' in key else clean_key
            clean_key = clean_key.replace('projection_head_audio.0', 'projection_head_audio.0.linear') if 'projection_head_audio.0' in key else clean_key
            clean_key = clean_key.replace('projection_head_audio.2', 'projection_head_audio.2.linear') if 'projection_head_audio.2' in key else clean_key
            clean_key = clean_key.replace('projection_head_text.0', 'projection_head_text.0.linear') if 'projection_head_text.0' in key else clean_key
            clean_key = clean_key.replace('projection_head_text.2', 'projection_head_text.2.linear') if 'projection_head_text.2' in key else clean_key

            clean_dict[clean_key] = value
        
        filtered_dict = {}
        for k, v in self.state_dict().items():
            if k in clean_dict and v.shape == clean_dict[k].shape:
                filtered_dict[k] = clean_dict[k]
            else:
                filtered_dict[k] = v
                print(f"From Model SBoRA Loading Function: Skipping parameter {k} with shape mismatch or not found")
        
        # Load weights (ignore missing LoRA parameters)
        self.load_state_dict(filtered_dict, strict=False)
        
        print(f"Loaded pretrained weights successfully!")
        
        return self

    def get_parameter_count(self):
        """Get parameter counts for base model vs SBoRA parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        sbora_param_objects, _ = self.get_sbora_parameters()
        sbora_params = sum(p.numel() for p in sbora_param_objects)
        
        base_params = total_params - sbora_params
        
        return {
            'total_params': total_params,
            'base_params': base_params, 
            'lora_params': sbora_params,  # Keep same key name for compatibility
            'trainable_params': trainable_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
        }

    def setup_for_sbora_finetuning(self, pretrained_path=None, freeze_base_model=True, device='cpu'):
        """
        One-step setup for SBoRA fine-tuning
        Args:
            pretrained_path: Optional path to pretrained weights
        """
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path, device=device)
        
        # Freeze base model and print info
        if freeze_base_model:
            self.freeze_base_model()

        self.print_parameter_info()
        
        return self
    
    def print_parameter_info(self):
        """Print detailed parameter information"""
        info = self.get_parameter_count()
        print(f"Parameter Information:")
        print(f"  Total parameters: {info['total_params']:,}")
        print(f"  Base model parameters: {info['base_params']:,}")
        print(f"  SBoRA parameters: {info['lora_params']:,}")
        print(f"  Trainable parameters: {info['trainable_params']:,}")
        print(f"  Trainable ratio: {info['trainable_ratio']:.2%}")
        print(f"  Parameter reduction: {(1 - info['trainable_ratio']):.2%}")

    def create_merged_model(self):
        """Create a clean merged model with SBoRA weights merged into base weights"""
        from .SimCLR_ViT import SimCLRViT  # Import the base model
        
        # Create a clean SimCLR model without SBoRA
        merged_model = SimCLRViT(
            img_size=self.image_patch_embed.img_size,
            audio_size=self.audio_patch_embed.img_size,
            text_seq_len=self.text_pos_embed.shape[1],
            patch_size=self.image_patch_embed.patch_size,
            img_in_chans=self.image_patch_embed.proj.in_channels,
            audio_in_chans=self.audio_patch_embed.proj.in_channels,
            text_vocab_size=self.text_embed.num_embeddings,
            embed_dim=self.embed_dim,
            depth=len(self.blocks),
            num_heads=self.blocks[0].attn.num_heads,
            mlp_ratio=4.,  # Standard ratio
            norm_layer=type(self.norm),
            split_projector=self.split_projector,
        )
        
        # Copy base weights first
        merged_state_dict = {}
        
        # Get current state dict and filter out SBoRA-specific parameters
        current_state_dict = self.state_dict()
        
        for key, value in current_state_dict.items():
            # Skip SBoRA parameters and buffers
            if 'lora_' in key or 'indices' in key:
                continue
            
            # Map SBoRA linear layers back to regular linear layers
            clean_key = key.replace('.linear.', '.')
            merged_state_dict[clean_key] = value.clone()
        
        # Load the base weights into the merged model
        merged_model.load_state_dict(merged_state_dict, strict=False)
        
        # Now merge SBoRA deltas into the corresponding layers
        for name, module in self.named_modules():
            if isinstance(module, SBoRALinear):
                # Find the corresponding layer in the merged model
                clean_name = name.replace('.linear', '')
                target_module = merged_model
                
                # Navigate to the target module
                for attr in clean_name.split('.'):
                    target_module = getattr(target_module, attr)
                
                # Compute SBoRA delta and add to base weights
                delta = self._compute_sbora_delta(module.sbora)
                
                with torch.no_grad():
                    target_module.weight.data += delta
                
                print(f"Merged SBoRA weights for layer: {clean_name}")
        
        print("Successfully created merged model without SBoRA components")
        return merged_model
    
    def _compute_sbora_delta(self, sbora_layer):
        """Compute the weight delta from a SBoRA layer"""
        with torch.no_grad():
            if sbora_layer.mode == 'FA':
                # For FA mode: delta = B @ A_standard_basis
                # Create standard basis matrix A from indices
                A_matrix = torch.zeros(sbora_layer.rank, sbora_layer.in_features, 
                                     device=sbora_layer.lora_B.device, 
                                     dtype=sbora_layer.lora_B.dtype)
                A_matrix[torch.arange(sbora_layer.rank), sbora_layer.A_indices] = 1.0
                # Compute delta: B @ A
                delta = sbora_layer.lora_B @ A_matrix * sbora_layer.scaling
                
            elif sbora_layer.mode == 'FB':
                # For FB mode: delta = B_standard_basis @ A
                # Create standard basis matrix B from indices  
                B_matrix = torch.zeros(sbora_layer.out_features, sbora_layer.rank,
                                     device=sbora_layer.lora_A.device,
                                     dtype=sbora_layer.lora_A.dtype)
                B_matrix[sbora_layer.B_indices, torch.arange(len(sbora_layer.B_indices))] = 1.0
                
                # Compute delta: B @ A
                delta = B_matrix @ sbora_layer.lora_A * sbora_layer.scaling
                
            else:
                raise ValueError(f"Unknown SBoRA mode: {sbora_layer.mode}")
        
        return delta
    
    def merge_sbora_weights_inplace(self):
        """Merge SBoRA weights into base weights in-place (modifies current model)"""
        print("Merging SBoRA weights into base linear layers in-place...")
        
        merged_count = 0
        for name, module in self.named_modules():
            if isinstance(module, SBoRALinear):
                # Get the SBoRA delta weights
                sbora_delta = self._compute_sbora_delta(module.sbora)
                
                # Add delta to the original linear layer weights
                with torch.no_grad():
                    module.linear.weight.data += sbora_delta
                
                # Disable SBoRA to use only the merged weights
                module.use_sbora = False
                merged_count += 1
                
                print(f"Merged SBoRA weights for layer: {name}")
        
        print(f"Successfully merged {merged_count} SBoRA layers into base weights")
        return self






