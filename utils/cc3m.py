import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.templates import get_templates

class CC3MDataset(Dataset):
    """CC3M Dataset for multimodal training"""
    
    def __init__(self, json_file, image_dir, metadata_dir, transform=None, 
                 text_seq_len=256, clip_align=False):
        self.modality = 'image'  # Primary modality
        self.image_dir = image_dir
        self.metadata_dir = metadata_dir
        self.transform = transform
        self.text_seq_len = text_seq_len
        self.clip_align = clip_align
        
        # Load data
        with open(json_file, 'r') as f:
            self.data = json.load(f)
            
        if self.clip_align:
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __len__(self):
        print('cc3m have len:', len(self.data))
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, sample['image_path'])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.clip_align:
            # For CLIP-style alignment training
            caption = sample['caption_text']
            
            # Tokenize caption
            encoded = self.tokenizer.encode_plus(
                caption,
                add_special_tokens=True,
                max_length=self.text_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            caption_tokens = encoded['input_ids'].squeeze(0)
            
            return image, caption_tokens, self.modality
        else:
            # For regular training
            label = sample['label']
            return image, label, self.modality