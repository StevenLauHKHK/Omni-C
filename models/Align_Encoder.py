import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import Block
from functools import partial
import numpy as np

class AlignEncoder(torch.nn.Module):
    def __init__(self, embed_dim, out_dim=768, unified_model=None, image_model=None, text_model=None, audio_model=None, keep_lang=False, logit_scale_enable=True, split_projector=True, single_projector=True):
        super().__init__()

        self.out_dim = out_dim
        self.unified_model = unified_model
        self.image_model = image_model
        self.text_model = text_model
        self.audio_model = audio_model
        self.logit_scale_enable = logit_scale_enable
        self.split_projector = split_projector
        self.single_projector = single_projector

        if self.logit_scale_enable:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            
            # Initialize scale to log(20) ≈ 2.996
            self.sail_scale = nn.Parameter(torch.ones([]) * math.log(20.0))
            # Initialize bias to -10 (regular space, not log)
            self.sail_bias = nn.Parameter(torch.zeros([]))


        if self.split_projector:
            if self.single_projector:
                self.single_projection_head_image = nn.Linear(embed_dim, out_dim)
                self.single_projection_head_audio = nn.Linear(embed_dim, out_dim)
                self.single_projection_head_text = nn.Linear(embed_dim, out_dim)
            else:
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
        else:
            if self.single_projector:
                self.single_projection_head = nn.Linear(embed_dim, out_dim)
            else:
                self.projection_head = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, out_dim)
                )

    def forward(self, x, modality):

        if self.split_projector:
            if modality == 'image':
                if self.single_projector:
                    z = self.single_projection_head_image(x)
                else:
                    z = self.projection_head_image(x)
            elif modality == 'audio':
                if self.single_projector:
                    z = self.single_projection_head_audio(x)
                else:
                    z = self.projection_head_audio(x)
            elif modality == 'text':
                if self.single_projector:
                    z = self.single_projection_head_text(x)
                else:
                    z = self.projection_head_text(x)
        else:
            if self.single_projector:
                z = self.single_projection_head(x)
            else:
                z = self.projection_head(x)
        return z

    def forward_embedding(self, x, modality, unified=False):
        if unified:
            x = self.unified_model.forward_embedding(x, modality)
            return x

        if modality == 'image':
            x = self.image_model.forward_embedding(x, modality)
        elif modality == 'text':
            x = self.text_model.forward_embedding(x, modality)
        elif modality == 'audio':
            x = self.audio_model.forward_embedding(x, modality)

        return x

    def forward_encoder(self, x, modality, unified=False):
        
        if unified:
            x = self.unified_model.forward_encoder(x, modality)
            return x

        if modality == 'image':
            x = self.image_model.forward_encoder(x, modality)
        elif modality == 'text':
            x = self.text_model.forward_encoder(x, modality)
        elif modality == 'audio':
            x = self.audio_model.forward_encoder(x, modality)

        return x

    
    def forward_align(self, x1, x2, scale_logit=True):
        x1 = x1 / x1.norm(dim=-1, keepdim=True)
        x2 = x2 / x2.norm(dim=-1, keepdim=True)

        if scale_logit:
            # Clip logit scale to prevent explosion (CLIP uses max value ~100)
            logit_scale = self.logit_scale.clamp(0, math.log(100)).exp()
        else:
            logit_scale = 1.0
        # x1 and x2 are expected to be normalized embeddings
        logits = logit_scale * torch.matmul(x1, x2.t())
        return logits
    


        
        
    

