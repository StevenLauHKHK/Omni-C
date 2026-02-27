import torch
from utils import utils
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F

class HFImageEncoder(torch.nn.Module):
    """
    Image encoder using pretrained models from Hugging Face.
    Supports various vision models like ViT, DeiT, BEiT, etc.
    """
    def __init__(self, args):
        super().__init__()
        
        # Default to ViT if not specified
        self.model_name = getattr(args, 'hf_model_name', 'google/vit-base-patch16-224')
        
        print(f'Loading pretrained model: {self.model_name} from Hugging Face')
        
        # Load the image processor for preprocessing
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name, 
            cache_dir=getattr(args, 'cache_dir', None)
        )
        
        # Load the model
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            cache_dir=getattr(args, 'cache_dir', None)
        )
        
        # Create standardized preprocessing transforms
        self._create_transforms()
        
        self.cache_dir = getattr(args, 'cache_dir', None)
        self.pooling_strategy = getattr(args, 'pooling_strategy', 'cls')  # Options: cls, mean, max
        
    def _create_transforms(self):
        """Create train and validation preprocessing transforms based on the processor"""
        import torchvision.transforms as transforms
        from PIL import Image
        
        # Extract preprocessing parameters from the processor
        size = self.processor.size.get("height", 224)
        if isinstance(size, dict):
            size = size.get("height", 224)
            
        mean = self.processor.image_mean if hasattr(self.processor, "image_mean") else [0.485, 0.456, 0.406]
        std = self.processor.image_std if hasattr(self.processor, "image_std") else [0.229, 0.224, 0.225]
        
        # Create validation/inference transform
        self.val_preprocess = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        # Create training transform with augmentations
        self.train_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        # Store original processor for cases where we need it directly
        self.original_processor = self.processor
    
    def preprocess_batch(self, images):
        """Process a batch of PIL images or tensors using the HF processor"""
        if isinstance(images[0], torch.Tensor):
            # Already a tensor, assume properly normalized
            return images
        else:
            # Use the HF processor directly
            return self.processor(images=images, return_tensors="pt")["pixel_values"]
    
    def forward(self, images):
        """
        Forward pass through the model
        Returns embeddings according to the specified pooling strategy
        """
        outputs = self.model(images, output_hidden_states=True)
        
        # Different pooling strategies
        if self.pooling_strategy == 'cls':
            # Use CLS token (first token) from last layer
            return outputs.last_hidden_state[:, 0]
        
        elif self.pooling_strategy == 'mean':
            # Mean pooling over all tokens (excluding CLS if present)
            return outputs.last_hidden_state[:, 1:].mean(dim=1)
        
        elif self.pooling_strategy == 'max':
            # Max pooling over all tokens
            return outputs.last_hidden_state.max(dim=1)[0]
        
        elif self.pooling_strategy == 'mean_all_layers':
            # Average the CLS token from the last 4 layers
            last_four = outputs.hidden_states[-4:]
            cls_embeddings = [layer[:, 0] for layer in last_four]
            return torch.stack(cls_embeddings).mean(dim=0)
            
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def forward_features(self, images):
        """Return all token features from the last layer"""
        outputs = self.model(images)
        return outputs.last_hidden_state
    
    def __call__(self, inputs):
        return self.forward(inputs)
    
    def save(self, filename):
        print(f'Saving HF image encoder to {filename}')
        utils.torch_save(self, filename)
    
    @classmethod
    def load(cls, args, filename):
        print(f'Loading HF image encoder from {filename}')
        encoder = cls(args)
        state_dict = torch.load(filename)
        encoder.load_state_dict(state_dict)
        return encoder

class HFMAEEncoder(HFImageEncoder):
    """
    Specialized encoder that loads MAE-pretrained models from Hugging Face
    """
    def __init__(self, args):
        # Default to Facebook's MAE model if not specified
        args.hf_model_name = getattr(args, 'hf_model_name', 'facebook/vit-mae-base')
        
        # Call parent constructor
        super().__init__(args)
    
    def forward(self, images):
        """
        Forward pass returning MAE features
        """
        # For MAE models, we typically want the CLS token
        outputs = self.model(images, output_hidden_states=True)
        
        # Use CLS token from last hidden state by default
        return outputs.last_hidden_state[:, 0]

# Common model identifiers for reference
HF_MODEL_OPTIONS = {
    'vit-base': 'google/vit-base-patch16-224',
    'vit-large': 'google/vit-large-patch16-224',
    'vit-huge': 'google/vit-huge-patch14-224-in21k',
    'beit-base': 'microsoft/beit-base-patch16-224',
    'beit-large': 'microsoft/beit-large-patch16-224',
    'deit-base': 'facebook/deit-base-patch16-224',
    'deit-small': 'facebook/deit-small-patch16-224',
    'mae-base': 'facebook/vit-mae-base',
    'mae-large': 'facebook/vit-mae-large',
    'dino-base': 'facebook/dino-vitb16',
    'dino-small': 'facebook/dino-vits16',
    'swin-base': 'microsoft/swin-base-patch4-window7-224',
    'swin-large': 'microsoft/swin-large-patch4-window7-224',
}
