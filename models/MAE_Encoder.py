import torch
import open_clip
from utils import utils
from transformers import AutoImageProcessor, AutoModel, ViTMAEModel
import torch.nn.functional as F

class ImageMAEEncoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        print(f'Loading MAE ViT model: {args.model}')
        self.model = ViTMAEModel.from_pretrained('facebook/' + args.model, cache_dir=args.cache_dir)
        self.train_preprocess = AutoImageProcessor.from_pretrained('facebook/' + args.model, cache_dir=args.cache_dir)
        self.val_preprocess = AutoImageProcessor.from_pretrained('facebook/' + args.model, cache_dir=args.cache_dir)
        self.cache_dir = args.cache_dir

        # Initialize projection layer
        hidden_dim = 768
        self.projection = torch.nn.Linear(hidden_dim, 512)
        clip_model, _, _ = open_clip.create_model_and_transforms(args.clip_model, pretrained='openai')
        # Get CLIP's projection layer
        if hasattr(clip_model.visual, 'proj'):
            clip_proj = clip_model.visual.proj
            
            # Check if dimensions match
            if clip_proj.shape[0] == 512 and clip_proj.shape[1] == hidden_dim:
                # Direct copy - weights are already in correct shape
                self.projection.weight.data.copy_(clip_proj)
                print(f"Copied CLIP projection weights: {clip_proj.shape} -> {self.projection.weight.shape}")
            elif clip_proj.shape[1] == 512 and clip_proj.shape[0] == hidden_dim:
                # Need to transpose weights
                self.projection.weight.data.copy_(clip_proj.T)
                print(f"Copied and transposed CLIP projection weights: {clip_proj.shape} -> {self.projection.weight.shape}")
            else:
                print(f"CLIP projection dimensions ({clip_proj.shape}) don't match required dimensions ({hidden_dim}, 512)")
                print("Using random initialization for projection layer")
        else:
            print("CLIP model doesn't have a 'proj' attribute in its visual module")
        

    def forward(self, images):
        assert self.model is not None
        # MAE ViT model returns a tuple; we take the last hidden state for encoding
        outputs = self.model(images)
        cls_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        return self.projection(cls_features)  # [batch_size, 512]
    
    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name, filename):
        print(f'Loading image encoder from {filename}')
        state_dict = torch.load(filename)
        return cls.load(model_name, state_dict)

    @classmethod
    def load_from_state_dict(cls, model_name, state_dict):
        self.model = ViTMAEModel.from_pretrained('facebook/' + model_name, cache_dir=args.cache_dir)
        self.model.load_from_state_dict(state_dict)