import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class DownStreamDataset(Dataset):
    def __init__(self, root_base, json_file, prefix, return_type='no_pt', transform=None):
        self.modality = 'image'
        self.root_dir = os.path.join(root_base, prefix)
        self.root_base = root_base
        self.json_file = json_file
        self.image_paths, self.labels, self.classes, self.class_to_idx = self._load_json_file()
        self.transform = transform
        self.return_type = return_type
        

    def _load_json_file(self):
        img_files = []
        img_labels = []
        with open(self.json_file,'r') as json_file:
            # Load json data
            json_data = json.load(json_file)
            for d in json_data:
                img_files.append(d['name'])
                img_labels.append(d['label'])

        # Load class information
        with open(os.path.join(self.root_base , "classes.json"), "r") as f:
            class_info = json.load(f)
            classes = [cls_data["name"] for cls_data in class_info]
            class_to_idx = {cls_data["name"]: cls_data["id"] for cls_data in class_info}

        return img_files, img_labels, classes, class_to_idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        target = self.labels[idx]
        pil_image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            if self.return_type == 'pt':
                pil_image = self.transform(images=pil_image, return_tensors="pt")['pixel_values'][0]
            else:
                pil_image = self.transform(pil_image)
                
        return pil_image, target, self.modality