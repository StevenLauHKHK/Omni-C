import torch
import open_clip
from utils import utils
from transformers import AutoImageProcessor, AutoModel, ViTMAEModel
import torch.nn.functional as F
from utils.templates import get_templates

class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        print(f'Loading {args.model} pre-trained weights.')
        if '__pretrained__' in args.model:
            name, pretrained = args.model.split('__pretrained__')
        else:
            name = args.model
            pretrained = 'openai'
        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained)
        
        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)
    
    def __call__(self, inputs):
        return self.forward(inputs)

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
        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained)
        self.model.load_from_state_dict(state_dict)

class TextEncoder(torch.nn.Module):
    def __init__(self, args, model_type='clip', keep_vision=False):
        super().__init__()

        self.args = args
        self.keep_vision = keep_vision
        self.model_type = model_type
        self.cache_dir = getattr(args, 'cache_dir', None)

        if model_type == 'clip':
            self._init_clip_encoder(args)
        else:
            raise ValueError(f'Unknown model_type: {model_type}')

    
    def _init_clip_encoder(self, args):
        """Initialize CLIP-based text encoder"""
        print(f'Loading {args.text_model} CLIP text encoder pre-trained weights.')
        if '__pretrained__' in args.text_model:
            name, pretrained = args.text_model.split('__pretrained__')
        else:
            name = args.text_model
            pretrained = 'openai'

        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=self.cache_dir
        )
        # Get tokenizer
        self.tokenizer = open_clip.get_tokenizer(name)
        # Remove vision encoder to save memory if not needed
        if not self.keep_vision and hasattr(self.model, 'visual'):
            delattr(self.model, 'visual')
    
    def forward(self, texts, modality, return_individual=False):
        """
        Forward pass for text encoding
        Args:
            texts: Can be either:
                - List of strings (will be tokenized)
                - Pre-tokenized tensor [batch_size, seq_len]
        Returns:
            Text embeddings [batch_size, embedding_dim]
        """
        if modality == 'image':
            dataset_name = self.args.image_dataset
        elif modality == 'audio':
            dataset_name = self.args.audio_dataset
        elif modality == 'text':
            dataset_name = self.args.text_dataset
        else:
            raise ValueError(f"Unknown modality: {modality}")

        text_templates = get_templates(dataset_name)
        
        if self.model_type == 'clip':
            return self._forward_clip_with_templates(texts, text_templates, return_individual)
        else:
            raise ValueError(f"Forward not implemented for {self.model_type}")

    def _forward_clip_with_templates(self, class_names, text_templates, return_individual=False):
        """Forward pass for CLIP text encoder with template averaging"""
        device = next(self.model.parameters()).device
        # Generate all text descriptions
        all_texts = []
        class_template_counts = []

        for class_name in class_names:
            texts = [template(class_name) for template in text_templates]
            all_texts.extend(texts)
            class_template_counts.append(len(texts))

        # Tokenize all texts at once for efficiency
        tokenized_texts = self.tokenizer(all_texts).to(device)

        # Encode all texts
        text_features = self.model.encode_text(tokenized_texts)
        text_features = F.normalize(text_features, dim=-1)  # Normalize each template feature

        # Reshape to [num_classes, num_templates, feature_dim]
        reshaped_features = []
        start_idx = 0

        for i, class_name in enumerate(class_names):
            num_templates = class_template_counts[i]
            end_idx = start_idx + num_templates
            class_features = text_features[start_idx:end_idx]  # [num_templates, feature_dim]
            reshaped_features.append(class_features)
            start_idx = end_idx
        
        # Stack to create [num_classes, num_templates, feature_dim]
        stacked_features = torch.stack(reshaped_features, dim=0)

        if return_individual:
            return stacked_features  # Return all individual template features
        else:
            # Average across templates for each class
            mean_features = stacked_features.mean(dim=1)  # [num_classes, feature_dim]
            # Re-normalize after averaging (important!)
            mean_features = F.normalize(mean_features, dim=-1)
            return mean_features
        
    
    def _forward_clip(self, texts):
        """Forward pass for CLIP text encoder"""
        if isinstance(texts, list):
            # Tokenize input texts
            texts = self.tokenizer(texts).to(next(self.model.parameters()).device)
        return self.model.encode_text(texts)

    def get_text_model_logit_scale(self):
        """Get logit_scale parameter from text model (if CLIP)"""
        if self.model_type == 'clip':
            return self.model.logit_scale
        else:
            raise ValueError(f'Logit scale not available for model_type: {self.model_type}')


    def encode_texts(self, texts, modality, return_individual=False):
        return self.forward(texts, modality, return_individual)

    def get_tokenizer(self):
        return self.tokenizer

    def __call__(self, inputs, modality, return_individual=False):
        return self.forward(inputs, modality, return_individual)

    def save(self, filename):
        print(f'Saving text encoder to {filename}')
        utils.torch_save(self, filename)

    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False

    @classmethod
    def load(cls, model_name, filename):
        print(f'Loading text encoder from {filename}')
        state_dict = torch.load(filename)
        return cls.load(model_name, state_dict)

    @classmethod
    def load_from_state_dict(cls, model_name, state_dict):
        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained)
        self.model.load_from_state_dict(state_dict)


class CLIPEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=True):
        super().__init__()

        print(f'Loading {args.model} pre-trained weights.')
        if '__pretrained__' in args.model:
            name, pretrained = args.model.split('__pretrained__')
        else:
            name = args.model
            pretrained = 'openai'
        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained)
        
        self.cache_dir = args.cache_dir

        self.tokenizer = open_clip.get_tokenizer(name)

    def forward_images(self, images):
        return self.model.encode_image(images)

    def forward_texts(self, texts):
        return self.model.encode_text(texts)

    def forward(self,images, texts):
        assert self.model is not None
        return self.model.encode_image(images), self.model.encode_text(texts)

    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving clip encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name, filename):
        print(f'Loading clip encoder from {filename}')
        state_dict = torch.load(filename)
        return cls.load(model_name, state_dict)

    @classmethod
    def load_from_state_dict(cls, model_name, state_dict):
        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained)
        self.model.load_from_state_dict(state_dict)

class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def freeze_head(self):
        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)


class MultiHeadImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_heads):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_heads = torch.nn.ModuleList(classification_heads)
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        for idx in range(len(self.classification_heads)):
            self.classification_heads[idx].weight.requires_grad_(False)
            self.classification_heads[idx].bias.requires_grad_(False)

    def forward(self, inputs, head_idx):
        features = self.image_encoder(inputs)
        outputs = self.classification_heads[head_idx](features)
        return outputs

    def __call__(self, inputs, head_idx):
        return self.forward(inputs, head_idx)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)
