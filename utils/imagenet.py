import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.templates import get_templates


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, json_file, class_map_json=None, transform=None, return_double_augmentations=False, clip_align=False):
        self.modality = 'image'
        self.root_dir = root_dir
        self.json_file = json_file
        self.image_paths, self.labels = self._load_json_file()
        self.transform = transform
        self.return_double_augmentations = return_double_augmentations
        self.clip_align = clip_align
        self.class_map_json = class_map_json

        if self.clip_align:
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.text_templates = get_templates('imagenet')
        # Load the class map
        self.class_map = self._load_class_map()

    def _load_json_file(self):
        img_files = []
        img_labels = []
        with open(self.json_file,'r') as json_file:
            # Load json data
            json_data = json.load(json_file)
            for d in json_data:
                img_files.append(d['name'])
                img_labels.append(d['label'])
        return img_files, img_labels

    def _load_class_map(self):
        """
        Load the class map from JSON file
        Returns: Dictionary mapping class names to indices
        """
        if self.class_map_json is None:
            print("Warning: No class map JSON file provided")
            return {}
            
        if not os.path.exists(self.class_map_json):
            print(f"Warning: Class map file {self.class_map_json} not found")
            return {}
        
        with open(self.class_map_json, 'r') as f:
            class_map = json.load(f)
        
        print(f"Loaded {len(class_map)} classes from class map")
        return class_map


    def __len__(self):
        return len(self.image_paths)
    
    def tokenize_text(self, text, max_len=256):
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoded['input_ids'].squeeze(0)  # Remove batch dimension

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image_1 = self.transform(image)
            image_2 = self.transform(image)
        else:
            image_1 = image
            image_2 = image

        label = self.labels[idx]

        if self.class_map:
            label_name = self.class_map[str(label)]

        if self.return_double_augmentations:
            return image_1, image_2, label, self.modality
        elif self.clip_align and self.class_map:
            # random get a sentence from the templates
            template = self.text_templates[torch.randint(len(self.text_templates), (1,)).item()]
            text = template(label_name)

            token_sentence = self.tokenize_text(text)
            return image_1, token_sentence, self.modality
        else:
            return image_1, label, self.modality