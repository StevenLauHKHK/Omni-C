import torch
import torch.nn as nn
import numpy as np
import random
from random import randrange
from timm.models.vision_transformer import Block
from functools import partial
from utils.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
import torch.nn.functional as F
import os
# from pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


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


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone for multimodal inputs"""
    def __init__(self, 
                 img_size=(64, 64), audio_size=(512, 128), text_seq_len=128, 
                 patch_size=8, img_in_chans=3, audio_in_chans=1, text_vocab_size=30522,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, enable_nsp=False, text_nsp_weight=1.0, audio_mae_weight=10.0, pretrain=True):
        super().__init__()

        self.pretrain = pretrain
        self.enable_nsp = enable_nsp  # Enable Next Sentence Prediction (NSP) for text
        self.text_vocab_size = text_vocab_size
        self.text_nsp_weight = text_nsp_weight
        self.audio_mae_weight = audio_mae_weight

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # Image patch embedding
        self.embed_dim = embed_dim
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

        # Shared transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        # self.pre_layrnorm = norm_layer(embed_dim)

        # for b in self.blocks:
        #     b.attn.scale = 1 ** -0.5  
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        if self.pretrain:
            # MAE decoder specifics
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            self.decoder_image_pos_embed = nn.Parameter(torch.zeros(1, num_image_patches + 1, decoder_embed_dim), requires_grad=False)
            self.decoder_audio_pos_embed = nn.Parameter(torch.zeros(1, num_audio_patches + 1, decoder_embed_dim), requires_grad=False)
            self.decoder_text_pos_embed = nn.Parameter(torch.zeros(1, text_seq_len + 1, decoder_embed_dim), requires_grad=False)

            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(decoder_depth)
            ])

            self.decoder_norm = norm_layer(decoder_embed_dim)

            # Separate predictor layers for each modality
            self.decoder_pred_image = nn.Linear(decoder_embed_dim, patch_size**2 * img_in_chans, bias=True)  # Image
            self.decoder_pred_audio = nn.Linear(decoder_embed_dim, patch_size**2 * audio_in_chans, bias=True)  # Audio
            self.decoder_pred_audio_mpc = nn.Linear(decoder_embed_dim, patch_size**2 * audio_in_chans, bias=True)  # Audio
            self.decoder_pred_text = nn.Linear(decoder_embed_dim, text_vocab_size, bias=True)  # Text

            # SSAST audio pretext task components
            # MPC (Masked Patch Classification) prediction layer
            self.audio_mpc_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, self.image_patch_embed.num_patches)  # 256 is the dimension used in SSAST
            )

            if self.enable_nsp:
                self.nsp_head = nn.Linear(embed_dim, 2) # Binary classification: IsNext or NotNext
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        
        self.initialize_weights()

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

        if self.pretrain:
            # Initialize decoder positional embeddings for image
            decoder_image_pos_embed = get_2d_sincos_pos_embed(
                self.decoder_image_pos_embed.shape[-1], 
                int(self.image_patch_embed.grid_size[0]), 
                int(self.image_patch_embed.grid_size[1]), 
                cls_token=True
            )
            self.decoder_image_pos_embed.data.copy_(torch.from_numpy(decoder_image_pos_embed).float().unsqueeze(0))

            # Initialize decoder positional embeddings for audio (non-square)
            decoder_audio_pos_embed = get_2d_sincos_pos_embed(
                self.decoder_audio_pos_embed.shape[-1], 
                self.audio_patch_embed.grid_size[0], 
                self.audio_patch_embed.grid_size[1], 
                cls_token=True
            )
            self.decoder_audio_pos_embed.data.copy_(torch.from_numpy(decoder_audio_pos_embed).float().unsqueeze(0))

            # Initialize decoder positional embeddings for text using 1D sine-cosine
            decoder_text_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.decoder_text_pos_embed.shape[-1], 
                np.arange(self.decoder_text_pos_embed.shape[1], dtype=float)
            )
            self.decoder_text_pos_embed.data.copy_(torch.from_numpy(decoder_text_pos_embed).float().unsqueeze(0))
            # Initialize mask token
            torch.nn.init.normal_(self.mask_token, std=0.02)
        
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

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        audio: (N, 1, F, T)
        x: (N, L, patch_size**2 * 3)
        """
        p = self.image_patch_embed.patch_size
        
        c = imgs.shape[1]
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p

        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x, modality='image'):
        """
        Reconstruct the original input from patches.
        Args:
            x: (N, L, patch_size**2 * C), where L is the number of patches.
            modality: 'image' or 'audio'.
        Returns:
            imgs: (N, C, H, W) for images.
            audio: (N, C, F, T) for audio spectrograms.
        """
        p = self.image_patch_embed.patch_size  # Patch size (assumes square patches)
        
        if modality == 'image':
            c = 3  # Number of channels for images
            h = w = int(x.shape[1]**0.5)  # Assume square grid
            assert h * w == x.shape[1], "Number of patches does not match a square grid."
        elif modality == 'audio':
            c = 1  # Number of channels for audio
            h = self.audio_patch_embed.grid_size[0]  # Height of the spectrogram grid
            w = self.audio_patch_embed.grid_size[1]  # Width of the spectrogram grid
            assert h * w == x.shape[1], "Number of patches does not match the audio grid."
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio, modality='image'):
        """
        Perform per-sample random masking by per-sample shuffling.
        x: [N, L, D], sequence

        For text, it performs BERT-style masking.
        Args:
            x: [N, L, D], sequence of token embeddings
            mask_ratio: percentage of tokens to mask (typically 0.15 for BERT)
            vocab_size: size of vocabulary for random token replacement
        """
        if modality == 'text':
            N, L, D = x.shape  # batch, length, dim
            len_keep = int(L * (1 - mask_ratio))
            len_mask = L - len_keep
            
            # Create random indices for potential masking
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            
            # Sort noise to determine which tokens to mask
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)  # restore original order
            
            # Select tokens to mask (15% of all tokens)
            ids_mask = ids_shuffle[:, len_keep:]
            
            # Initialize mask (0=keep, 1=mask)
            mask = torch.zeros([N, L], device=x.device)
            mask.scatter_(1, ids_mask, 1)
            
            # Keep a copy of the original embeddings for loss calculation
            x_orig = x.clone()

            # BERT masking strategy
            # For each masked token:
            # - 80% replace with [MASK] token (we'll use zeros, decoder will learn mask representation)
            # - 10% replace with random token
            # - 10% keep the original token

            # Create a new random tensor for BERT-style decisions
            mask_decision = torch.rand(N, L, device=x.device)
            for i in range(N):
                for j in range(L):
                    if mask[i, j] == 1:  # If this token is selected for masking
                        if mask_decision[i, j] < 0.8:
                            # 80% chance: replace with zeros (will be handled by mask_token in decoder)
                            x[i, j] = torch.zeros(D, device=x.device)
                        elif mask_decision[i, j] < 0.9:
                            # 10% chance: replace with random token
                            # Generate random token ID
                            random_id = torch.randint(0, self.text_vocab_size, (1,), device=x.device)
                            # Get embedding for this random token (assuming self.text_embed is accessible)
                            random_token_embedding = self.text_embed(random_id)
                            x[i, j] = random_token_embedding
                        # else: 10% chance: keep as is (do nothing)

            # Keep unmasked tokens and sort them according to noise
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            return x_masked, mask, ids_restore

        else:
            N, L, D = x.shape  # batch, length, dim
            len_keep = int(L * (1 - mask_ratio))
            
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            
            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)
            
            return x_masked, mask, ids_restore

    def gen_maskid_patch(self, sequence_len, f_dim, t_dim, mask_ratio=0.75, cluster_range=3):
        """Generate mask indices with clustering for audio patches"""
        mask_size = int(round(sequence_len * mask_ratio))
        mask_id = []
        cur_clus = randrange(cluster_range) + 3  # Random clustering factor in [3,6)

        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)
            cur_mask = []
            for i in range(cur_clus):
                for j in range(cur_clus):
                    mask_cand = start_id + t_dim * i + j
                    if 0 <= mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id.extend(cur_mask)
        
        # Convert to set to remove duplicates, then back to list
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)

    def gen_maskid_frame(self, sequence_len, mask_ratio=0.75):
        """Generate random mask indices for frames"""
        mask_size = int(round(sequence_len * mask_ratio))
        mask_id = random.sample(range(sequence_len), mask_size)
        return torch.tensor(mask_id)

    def cluster_masking(self, x, mask_ratio=0.75, cluster=True):
        """
        Perform clustered masking using SSAST strategy
        x: [N, L, D], sequence
        """

        N, L, D = x.shape  # batch, length, dim
        mask_size = int(round(L * mask_ratio))
        len_keep = L - mask_size

        # Get the grid dimensions
        f_dim = self.audio_patch_embed.grid_size[0]
        t_dim = self.audio_patch_embed.grid_size[1]

        # Initialize with all zeros (keep all tokens)
        mask = torch.zeros([N, L], device=x.device)  # 0=keep, 1=remove
        ids_restore = torch.arange(L, device=x.device).unsqueeze(0).repeat(N, 1)
        x_masked_list = []

        for i in range(N):
            if cluster:
                mask_indices = self.gen_maskid_patch(L, f_dim, t_dim, mask_ratio, cluster_range=3).to(x.device)
            else:
                mask_indices = self.gen_maskid_frame(L, mask_ratio).to(x.device)
            
            # Set masked positions to 1
            mask[i, mask_indices] = 1
            
            # Get indices to keep (where mask is 0)
            keep_indices = torch.nonzero(mask[i] == 0).squeeze(1)
            
            # Keep only the non-masked embeddings
            x_masked_sample = x[i, keep_indices]
            
            # Store the masked sample
            x_masked_list.append(x_masked_sample)

        # Create a padded tensor for the masked embeddings
        x_masked = torch.zeros(N, len_keep, D, device=x.device)
        for i, sample in enumerate(x_masked_list):
            x_masked[i, :sample.shape[0]] = sample

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, modality, cluster_mask=False, pretrain=True, ref_embed=[None, None, None]):
        """
        Forward pass through the encoder
        x: Input tensor
        mask_ratio: Ratio of tokens to mask
        modality: 'image', 'audio', or 'text'
        cluster_mask: Whether to use SSAST-style clustered masking
        """
        if modality == 'image':
            x = self.image_patch_embed(x)
            # pos_embed = self.image_pos_embed
            # x = self.audio_patch_embed(x)
            pos_embed = self.image_pos_embed

        elif modality == 'audio':
            x = self.audio_patch_embed(x)
            pos_embed = self.audio_pos_embed
            
            # x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
            # x = x.repeat(1, 3, 1, 1)
            # x = self.image_patch_embed(x)
            # pos_embed = self.audio_pos_embed
            
        elif modality == 'text':
            x = self.text_embed(x) + self.text_pos_embed
            pos_embed = None
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        if pos_embed is not None:
            x = x + pos_embed[:, 1:, :]

        if self.pretrain:
            # Choose masking strategy
            if modality == 'text':
                # Use BERT-style masking for text
                x, mask, ids_restore = self.random_masking(x, mask_ratio, modality='text')
            elif cluster_mask and modality == 'audio':
                x, mask, ids_restore = self.cluster_masking(x, mask_ratio, cluster=True)
            else:
                x, mask, ids_restore = self.random_masking(x, mask_ratio)

            cls_token = self.cls_token + (pos_embed[:, :1, :] if pos_embed is not None else 0)
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)

            return x, mask, ids_restore
        
        else:

            # For inference or non-pretraining tasks, just return the embeddings
            cls_token = self.cls_token + (pos_embed[:, :1, :] if pos_embed is not None else 0)
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)

            return x, None, None

            # # if ref_embed[0] is not None:   
            # #     print(x.shape, ref_embed[0].shape)
            # #     diff = x - ref_embed[0]
            # #     print("Patch embedding difference:", diff.sum().item())
            # #     x_ref = self.pre_layrnorm(ref_embed[0])
            # #     diffs = x_ref - ref_embed[1]
            # #     print("Patch embedding norm difference 1:", diffs.sum().item())
            # #     if not os.path.exists("patch_embeddings_self.pt"):
            # #         torch.save(x, "patch_embeddings_self.pt")

            # # x = self.pre_layrnorm(x)

            # # if ref_embed[1] is not None:   
            # #     print(x.shape, ref_embed[1].shape)
            # #     diff = x - ref_embed[1]
            # #     print("Patch embedding norm difference:", diff.sum().item())
            # #     if not os.path.exists("patch_embeddings_norm_self.pt"):
            # #         torch.save(x, "patch_embeddings_norm_self.pt")

            # for i, blk in enumerate(self.blocks):
            #     x = blk(x)
            
            #     if ref_embed[2] is not None and i == 0:
            #         diff = x - ref_embed[2]
            #         print("embedding difference:", diff.sum().item())
            # # x = self.norm(x)

            # return x, None, None

    def forward_decoder(self, x, ids_restore, modality, mpc=False):
        x = self.decoder_embed(x)

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1) # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1) # append cls token

        if modality == 'image':
            x = x + self.decoder_image_pos_embed
        elif modality == 'audio':
            x = x + self.decoder_audio_pos_embed
        elif modality == 'text':
            x = x + self.decoder_text_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        if modality == 'image':
            x = self.decoder_pred_image(x)
        elif modality == 'audio':
            if mpc:
                x = self.decoder_pred_audio_mpc(x)
            else:
                x = self.decoder_pred_audio(x)
        elif modality == 'text':
            x = self.decoder_pred_text(x)

        x = x[:, 1:, :]
        return x

    def forward_mpc_mae(self, x, mask_ratio=0.75, cluster_mask=True):
        """
        SSAST-style MPC using MAE's encoder-decoder architecture
        """
        B = x.shape[0]
        # 1. Use existing encoder
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio, 'audio', cluster_mask)

        # 2. Use existing decoder
        pred = self.forward_decoder(latent, ids_restore, 'audio', mpc=True)

        # 3. Get raw patches for NCE loss
        # Extract patch size and create unfold operation
        p = self.audio_patch_embed.patch_size
        unfold = nn.Unfold(kernel_size=p, stride=p)
        input_patches = unfold(x).transpose(1, 2)  # [B, num_patches, p^2*C]

        # 4. Find masked patch indices for contrastive loss
        masked_indices = torch.nonzero(mask.to(torch.bool), as_tuple=True)
        
        batch_indices = masked_indices[0]  # Which batch item
        patch_indices = masked_indices[1]  # Which patch position

        # Reshape to group by batch item
        patch_indices_by_batch = []
        for b in range(B):
            batch_mask = (batch_indices == b)
            patches_in_batch = patch_indices[batch_mask]
            patch_indices_by_batch.append(patches_in_batch)
    
        # 5. Calculate NCE loss
        nce_loss = torch.tensor(0.0, device=x.device)
        correct = torch.tensor(0, device=x.device)
        total_patches = 0

        for i in range(B):
            if len(patch_indices_by_batch[i]) == 0:
                continue

            # Extract predictions for masked patches (from decoder output)
            masked_preds = pred[i, patch_indices_by_batch[i]]
            
            # Extract raw patches as targets (no need for extra encoding, like in SSAST)
            raw_patches = input_patches[i, patch_indices_by_batch[i]]
            
            # Calculate similarity matrix (cross-correlation)
            sim_matrix = torch.mm(masked_preds, raw_patches.transpose(0, 1))
        
            # Apply temperature scaling
            temperature = 0.1  # From SSAST paper
            sim_matrix = sim_matrix / temperature
            
            # Calculate accuracy (diagonal should be highest)
            pred_labels = torch.argmax(sim_matrix, dim=1)
            target_labels = torch.arange(len(patch_indices_by_batch[i]), device=x.device)
            correct += torch.sum(pred_labels == target_labels)
            
            # Calculate NCE loss
            nce_loss += nn.CrossEntropyLoss()(sim_matrix, target_labels)

            total_patches += len(patch_indices_by_batch[i])

        # Average over all masked patches
        accuracy = correct.float() / total_patches if total_patches > 0 else torch.tensor(0.0, device=x.device)
        nce_loss = nce_loss / B
        
        return nce_loss, pred, mask, accuracy

    def forward_nsp(self, x1, x2, is_next_sentence):
        """
        Forward pass for Next Sentence Prediction task
        
        Args:
            x1: First sentence tokens [B, L1]
            x2: Second sentence tokens [B, L2]
            is_next_sentence: Binary labels [B] (1 = is next, 0 = random)
        
        Returns:
            nsp_loss: NSP loss
            nsp_acc: NSP accuracy
        """

        batch_size = x1.size(0)
        # Get embeddings for first sentence
        x1_embed = self.text_embed(x1)
        # Get embeddings for second sentence
        x2_embed = self.text_embed(x2)

        # Add position embeddings (note: might need to handle length differences)
        x1_pos_embed = self.text_pos_embed[:, :x1.size(1), :]
        x2_pos_embed = self.text_pos_embed[:, :x2.size(1), :]

        x1_embed = x1_embed + x1_pos_embed
        x2_embed = x2_embed + x2_pos_embed

        # Concatenate sentences with [SEP] token in between
        # We'll use the cls_token for simplicity as our separator
        sep_token = self.cls_token.expand(batch_size, 1, -1)
        cls_token = self.cls_token.expand(batch_size, 1, -1)
        combined = torch.cat([cls_token, x1_embed, sep_token, x2_embed], dim=1)

        # Process through transformer
        for blk in self.blocks:
            combined = blk(combined)
        combined = self.norm(combined)

        # Use [CLS] token for prediction
        cls_output = combined[:, 0]
        nsp_logits = self.nsp_head(cls_output)

        # Calculate loss and accuracy
        nsp_loss = nn.CrossEntropyLoss()(nsp_logits, is_next_sentence)
        nsp_preds = torch.argmax(nsp_logits, dim=1)
        
        nsp_acc = (nsp_preds == is_next_sentence).float().mean()

        return nsp_loss, nsp_acc



    def forward_loss(self, x, pred, mask, modality):
        if modality in ['image', 'audio']:
            target = self.patchify(x)
            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum()
        elif modality == 'text':
            # Reshape pred and x for CrossEntropyLoss
            pred = pred.reshape(-1, pred.size(-1))  # [N * T, vocab_size]
            x = x.reshape(-1)  # [N * T]
            loss = nn.CrossEntropyLoss()(pred, x)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        return loss

    def forward(self, x, mask_ratio=0.75, modality='image', cluster_mask=False, task='mae', x2=None, is_next_sentence=None):

        if modality == 'text' and self.enable_nsp and task == 'nsp_mae':
            # Next Sentence Prediction task and MAE combined
            latent, mask, ids_restore = self.forward_encoder(x, mask_ratio, modality, cluster_mask and modality=='text')
            pred = self.forward_decoder(latent, ids_restore, modality)
            mae_loss = self.forward_loss(x, pred, mask, modality)

            # Get NSP loss
            nsp_loss, nsp_acc = self.forward_nsp(x, x2, is_next_sentence)
            # Combine losses
            loss = mae_loss + self.text_nsp_weight * nsp_loss

            return loss, pred, mask, {'mlm_loss': mae_loss, 'nsp_loss': nsp_loss, 'nsp_acc': nsp_acc}

        
        elif task == 'mae' or task is None:
            # Standard MAE approach
            latent, mask, ids_restore = self.forward_encoder(x, mask_ratio, modality, cluster_mask and modality=='audio')
            pred = self.forward_decoder(latent, ids_restore, modality)
            loss = self.forward_loss(x, pred, mask, modality)
            return loss, pred, mask

        elif task == 'mpc' and modality == 'audio':
            # Use the merged encoder-decoder with contrastive objective
            nce_loss, pred, mask, accuracy = self.forward_mpc_mae(x, mask_ratio, cluster_mask)
            return nce_loss, pred, mask, accuracy
        
        elif task == 'joint' and modality == 'audio':
            # Combined MAE + MPC training (joint training)
            # 1. Run the merged approach to get both reconstruction and NCE loss
            nce_loss, pred, mask, accuracy = self.forward_mpc_mae(x, mask_ratio, cluster_mask)
            
            # 2. Get standard MAE reconstruction loss
            mae_loss = self.forward_loss(x, pred, mask, modality)
            
            # 3. Combine losses (weight as needed)
            total_loss = nce_loss + self.audio_mae_weight * mae_loss

            return total_loss, mae_loss, nce_loss, accuracy, pred, mask

        else:
            raise ValueError(f"Unsupported task '{task}' for modality '{modality}'")

if __name__ == '__main__':
    # Example usage
    model_params = {
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'decoder_embed_dim': 768,
        'decoder_depth': 8,
        'decoder_num_heads': 16,
        'patch_size': 16
    }
    model = MaskedAutoencoderViT(
        img_size=(224,224),  # Tiny ImageNet images are 64x64
        audio_size=(1024, 128),  # VGGSound spectrogram size
        text_seq_len=128,  # AGNews max sequence length
        patch_size=model_params['patch_size'],  # Set patch size to 8
        embed_dim=model_params['embed_dim'],
        depth=model_params['depth'],
        num_heads=model_params['num_heads'],
        decoder_embed_dim=model_params['decoder_embed_dim'],
        decoder_depth=model_params['decoder_depth'],
        decoder_num_heads=model_params['decoder_num_heads'],
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_pix_loss=True
    )

    
    # img_test_input = torch.zeros([1, 3, 224, 224])
    # model.forward(img_test_input, mask_ratio=0.75, modality='image')

    input_tdim = 1024
    audio_test_input = torch.zeros([5, 1, input_tdim, 128])
    model.forward(audio_test_input, mask_ratio=0.1953, modality='audio', cluster_mask=True, task='mpc')