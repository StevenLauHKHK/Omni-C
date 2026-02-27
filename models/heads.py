import os
import torch
from tqdm import tqdm

import open_clip

from utils.templates import get_templates
from models.CLIP_Encoder import ClassificationHead, ImageEncoder


def build_classification_head(model, dataset_name, template, classnames, logit_scale_enable=True):
    template = get_templates(dataset_name)

    if logit_scale_enable:
        logit_scale = model.model.logit_scale

    model.eval()

    device = next(model.model.parameters()).device
    
    print('Building classification head.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = open_clip.tokenize(texts).to(device) # tokenize
            embeddings = model.model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()
            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        if logit_scale_enable:
            zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)
        
    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def get_classification_head(args, dataset, classnames):
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, f'head_{dataset}.pt')
    if os.path.exists(filename):
        print(f'Classification head for {args.model} on {dataset} exists at {filename}')
        return ClassificationHead.load(filename)
    print(f'Did not find classification head for {args.model} on {dataset} at {filename}, building one from scratch.')
    model = ImageEncoder(args, keep_lang=True).model
    
    template = get_templates(dataset)
    classification_head = build_classification_head(model, dataset, template, classnames)
    print(f'Saving classification head to {filename}')
    classification_head.save(filename)
    return classification_head

def get_mae_classification_head(args, dataset, classnames):
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, f'head_{dataset}.pt')
    if os.path.exists(filename):
        print(f'Classification head for {args.model} on {dataset} exists at {filename}')
        return ClassificationHead.load(filename)
    print(f'Did not find classification head for {args.model} on {dataset} at {filename}, building one from scratch.')
    
    model, _, _ = open_clip.create_model_and_transforms(args.clip_model, pretrained='openai')
    
    template = get_templates(dataset)
    classification_head = build_classification_head(model, dataset, template, classnames)
    print(f'Saving classification head to {filename}')
    classification_head.save(filename)
    return classification_head


def get_vit_classification_head(args, text_model, dataset, classnames, logit_scale_enable=True):
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, f'head_{dataset}.pt')

    if os.path.exists(filename):
        print(f'Classification head for {args.model} on {dataset} exists at {filename}')
        return ClassificationHead.load(filename)
    print(f'Did not find classification head for {args.model} on {dataset} at {filename}, building one from scratch.')

    template = get_templates(dataset)
    classification_head = build_classification_head(text_model, dataset, template, classnames, logit_scale_enable)
    print(f'Saving classification head to {filename}')
    classification_head.save(filename)
    return classification_head