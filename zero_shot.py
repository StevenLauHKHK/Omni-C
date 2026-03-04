import argparse
import os
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from datetime import datetime
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torch.nn.functional as F

from models.ViT import MaskedAutoencoderViT
from models.SimCLR_ViT import SimCLRViT
from models.SimCLR_ViT_SBoRA import SimCLRViT_SBoRA
from models.Align_Encoder import AlignEncoder
from utils.dataloader import MultimodalDataLoader, SingleModalityDataLoader
from config.config import get_config
from utils.simclr_loss import NTXentLoss
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.scheduler import adjust_learning_rate
from utils.lars import LARS
import timm.optim.optim_factory as optim_factory
from timm.models.layers import trunc_normal_
from utils.pos_embed import interpolate_pos_embed

# from transformers import ViTModel, CLIPModel, ViTMAEForPreTraining, CLIPVisionModel





def parse_option():
    parser = argparse.ArgumentParser('Multimodal MAE training script', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--batch-size', default=128, type=int, help="Batch size for single GPU")
    parser.add_argument('--test-batch-size', default=128, type=int, help="Test batch size for single GPU")
    parser.add_argument('--accum-iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--num-workers', default=32, type=int, help="Number of data loading threads")
    parser.add_argument('--resume-checkpoint', default=None, help='Resume checkpoint path')
    parser.add_argument('--checkpoint-dir', default="checkpoints", type=str, help="Directory to save checkpoints")
    parser.add_argument('--debug-dir', default="debug_outputs", type=str, help="Directory to save debug outputs")
    parser.add_argument('--seed', default=42, type=int, help="Random seed for reproducibility")
    
    # Dataset arguments
    parser.add_argument('--image-data-path', default='/path/to/tiny-imagenet/', type=str, help='Path to image dataset')
    parser.add_argument('--image-dataset', default='tiny_imagenet', type=str, help='Image dataset name')
    parser.add_argument('--multi-image-data-paths', nargs=2, type=str, help='Paths to two image datasets (e.g., Tiny-ImageNet and CIFAR-100)')
    parser.add_argument('--multi-image-datasets', nargs=2, type=str, help='Names of the two image datasets (e.g., tiny_imagenet and cifar100)')
    parser.add_argument('--audio-data-path', default='/path/to/audioset/', type=str, help='Path to audio dataset')
    parser.add_argument('--audio-data-path-2', default='/path/to/librispeech/', type=str, help='Path to audio dataset')
    parser.add_argument('--audio-dataset', default='vggsound', type=str, help='Audio dataset name')
    parser.add_argument('--text-data-path', default='/path/to/AG_NEWS/', type=str, help='Path to text dataset')
    parser.add_argument('--text-dataset', default='agnews', type=str, help='Text dataset name')
    
    
    # training arguments
    parser.add_argument('--early_stop_patience', default=3, type=int, help='Early stopping patience epochs')
    parser.add_argument('--eval_every_n', default=1, type=int, help='Number of epochs for evaluation')
    parser.add_argument('--save_every_n', default=1, type=int, help='Number of epochs for saving a checkpoint')
    parser.add_argument('--image-only-epochs', default=0, type=int, help='Number of epochs to train on image modality only before multimodal training')
    parser.add_argument('--single-modality', type=str, choices=['image', 'audio', 'text'], help='Specify which modality to train on')
    parser.add_argument('--selected-multimodality', type=str, nargs='+', choices=['image', 'audio', 'text'], help='Specify which modalities to include in multimodal training (e.g., --selected-multimodality image audio)')
    parser.add_argument('--max-train-batches', default=None, type=int, help='Maximum number of batches to train per epoch')
    parser.add_argument('--log-file', default="training_log.txt", type=str, help="Path to the log file")  # New argument
    

    # zero shot evaluation arguments
    parser.add_argument('--zero-shot', action='store_true', help='Perform zero shot evaluation only')
    parser.add_argument('--zero-shot-remove-head', action='store_true', help='Remove projection head for zero shot evaluation')

    # model arguments
    parser.add_argument('--model', default='vit', type=str, help='Model architecture')
    parser.add_argument('--model-size', default='small', choices=['small', 'medium', 'base'], 
                        help="Choose the model size: 'small', 'medium', or 'base'")

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


# Define the lambda function for the learning rate schedule
def lr_lambda(epoch):
    if epoch < 30:
        return 1.0  # Keep the learning rate at its initial value
    elif epoch < 40:
        return 0.1  # Reduce the learning rate by 10x at epoch 30
    else:
        return 0.01  # Reduce the learning rate by another 10x at epoch 40

def get_scheduler(optimizer, total_epochs, warmup_epochs, scheduler_type='cosine'):
    """
    Create a learning rate scheduler with a warmup phase followed by cosine annealing.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        total_epochs: Total number of training epochs.
        warmup_epochs: Number of warmup epochs.
        scheduler_type: Type of scheduler to use ('cosine', 'cosine_custom', 'linear', etc.)
    
    Returns:
        A learning rate scheduler.
    """
    if scheduler_type == 'cosine_custom':
        # Use the cosine_lr from utils.py
        from utils.utils import cosine_lr
        base_lr = optimizer.param_groups[0]['lr']
        return cosine_lr(optimizer, base_lr, warmup_length=warmup_epochs, steps=total_epochs)
    
    elif scheduler_type == 'cosine':
        # The default implementation using LambdaLR
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))  # Linear warmup
            return 0.5 * (1. + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))  # Cosine decay
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'cosine_restarts':
        # Cosine annealing with warm restarts
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=total_epochs // 3,  # Restart every T_0 epochs
            T_mult=1,               # Multiply T_0 by this factor after each restart
            eta_min=0.0             # Minimum learning rate
        )
    
    elif scheduler_type == 'step':
        # Step LR with gamma decay at milestones
        milestones = [total_epochs // 3, 2 * total_epochs // 3]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.1  # Reduce LR by factor of 10 at each milestone
        )
    
    else:
        # Default to standard cosine implementation
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))  # Linear warmup
            return 0.5 * (1. + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))  # Cosine decay
        
        return LambdaLR(optimizer, lr_lambda)


def zero_shot_evaluation(model, train_loader, valid_loader, epoch, device, config, top_k=(1, 5), modality='image', remove_projection_head=False):
    """
    Perform zero-shot evaluation on a pretrained multimodal model.
    Only reports top-k accuracy without generating visualizations.
    
    Args:
        model: The pretrained multimodal model
        train_loader: Data loader for source data (used to get class representations)
        valid_loader: Data loader for evaluation data
        epoch: Current epoch number (for logging)
        device: Device to run inference on
        config: Configuration object
        top_k: Tuple of k values for which to compute accuracy
        modality: Modality to use for evaluation ('image', 'audio', or 'text')
        debug_dir: Directory to save any debug outputs
    
    Returns:
        Dict containing top-k accuracies
    """
    
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Zero-shot Validation Epoch: [{}]'.format(epoch)
    print_freq = 20

    print(f"Running zero-shot evaluation for {modality} modality...")
    
    # Step 1: Extract class prototypes from the source dataset
    class_prototypes = {}
    class_counts = {}
    
    print("Extracting class prototypes from source dataset...")
    
    with torch.no_grad():
        for data_iter_step, (batch, labels, modalities) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
            
            # Move data to device
            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            modality = modalities[0]
            
            # Extract features
            if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                if remove_projection_head:
                    embedding = model.module.forward_encoder(batch, modality=modality)
                else:
                    embedding = model.module.forward_embedding(batch, modality=modality)
            else:
                if remove_projection_head:
                    embedding = model.forward_encoder(batch, modality=modality)
                else:
                    embedding = model.forward_embedding(batch, modality=modality)
            
            # Normalize features
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            # Accumulate class prototypes
            for i in range(batch.size(0)):
                label = labels[i].item()
                if label not in class_prototypes:
                    class_prototypes[label] = embedding[i].cpu()
                    class_counts[label] = 1
                else:
                    class_prototypes[label] += embedding[i].cpu()
                    class_counts[label] += 1

    # Normalize the prototypes
    for label in class_prototypes:
        class_prototypes[label] = class_prototypes[label] / class_counts[label]
        # Normalize again
        class_prototypes[label] = class_prototypes[label] / class_prototypes[label].norm()

    # Convert class prototypes to tensor
    prototype_labels = list(class_prototypes.keys())
    prototype_features = torch.stack([class_prototypes[label] for label in prototype_labels]).to(device)
    
    print(f"Created prototypes for {len(prototype_labels)} classes")
    
    # Step 2: Evaluate on the target dataset
    top_k_correct = {k: 0 for k in top_k}
    total = 0

    print("Evaluating on target dataset...")

    with torch.no_grad():
        for data_iter_step, (batch, labels, modalities) in enumerate(metric_logger.log_every(valid_loader, print_freq, header)):  
            # Move data to device
            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            modality = modalities[0]
            
            # Extract features
            if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                if remove_projection_head:
                    embedding = model.module.forward_encoder(batch, modality=modality)
                else:
                    embedding = model.module.forward_embedding(batch, modality=modality)
            else:
                if remove_projection_head:
                    embedding = model.forward_encoder(batch, modality=modality)
                else:
                    embedding = model.forward_embedding(batch, modality=modality)
            
            # Normalize features
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            # Compute similarity to prototypes
            similarity = embedding @ prototype_features.T  # [batch_size, num_prototypes]

            # Get predictions and scores
            _, predicted_indices = similarity.topk(max(top_k), dim=1)
            predictions = torch.tensor([prototype_labels[idx] for idx in predicted_indices.cpu().numpy().flatten()], 
                                      device=device).view(predicted_indices.shape)
            
            # Check if true labels are in top-k predictions
            for k in top_k:
                # Check if the true label is in the top-k predictions
                correct = predictions[:, :k].eq(labels.unsqueeze(1)).any(dim=1)
                top_k_correct[k] += correct.sum().item()

            total += batch.size(0)

    if total == 0:
        print(f"No samples found for modality {modality} in validation data")
        return {'error': f'No samples found for modality {modality}'}
    
    # Calculate metrics
    results = {f"top_{k}_acc": top_k_correct[k] / total * 100 for k in top_k}
    
    print(f"Zero-shot evaluation complete for {modality} modality.")
    print(f"Top-1 Accuracy: {results['top_1_acc']:.2f}%")
    if len(top_k) > 1:
        print(f"Top-5 Accuracy: {results['top_5_acc']:.2f}%")

    return results



def main(args, config, log_file="training_log.txt"):

    misc.init_distributed_mode(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the random seed for reproducibility
    seed = args.seed + misc.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Multi-GPU training requires GPUs.")
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    patience = args.early_stop_patience # stop after args.early_stop_patience epochs without improvement

    train_loader = SingleModalityDataLoader(
        modality = args.single_modality,
        dataset={'image': config.DATA.IMAGE_DATASET, 'audio': config.DATA.AUDIO_DATASET, 'text': config.DATA.TEXT_DATASET},
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=args.num_workers,
        shuffle=True,
        config=config,
        is_train=True,
        is_train_img_transform=True,
        audio_augmentation=False,
        text_augmentation=False,
        max_batches=args.max_train_batches,
        distributed=args.distributed,
        num_tasks=num_tasks,
        text_seq_len=config.TEXT.MAX_SEQ_LENGTH if args.single_modality == 'text' else None,
        rank=global_rank,
        enable_nsp=config.TEXT.ENABLE_NSP if args.single_modality == 'text' else False,
        return_double_augmentations=False,
    )
    
    
    valid_loader = SingleModalityDataLoader(
        modality = args.single_modality,
        dataset={'image': config.DATA.IMAGE_DATASET, 'audio': config.DATA.AUDIO_DATASET, 'text': config.DATA.TEXT_DATASET},
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=args.num_workers,
        shuffle=False,
        config=config,
        is_train=False,
        is_train_img_transform=False,
        audio_augmentation=False,
        text_augmentation=False,
        distributed=args.distributed,
        num_tasks=num_tasks,
        text_seq_len=config.TEXT.MAX_SEQ_LENGTH if args.single_modality == 'text' else None,
        rank=global_rank,
        enable_nsp=config.TEXT.ENABLE_NSP if args.single_modality == 'text' else False,
        return_double_augmentations=False,
    )
        

    # Configure model parameters based on the selected size
    if args.model_size == 'base':
        if 'B-32' in args.model:
            model_params = {
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'decoder_embed_dim': 512,
                'decoder_depth': 8,
                'decoder_num_heads': 16,
                'patch_size': 32,
                'out_dim': 768
            }
        elif 'B-16' in args.model:
            model_params = {
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'decoder_embed_dim': 512,
                'decoder_depth': 8,
                'decoder_num_heads': 16,
                'patch_size': 16,
                'out_dim': 768
            }
        else:
            raise ValueError(f"Unknown base model variant: {args.model}")
    elif args.model_size == 'medium':
        if 'M-32' in args.model:
            model_params = {
                'embed_dim': 512,  # Intermediate embedding dimension
                'depth': 12,       # Intermediate number of transformer layers
                'num_heads': 8,    # Intermediate number of attention heads
                'decoder_embed_dim': 512,
                'decoder_depth': 8,
                'decoder_num_heads': 8,
                'patch_size': 32,
                'out_dim': 768
            }
        elif 'M-16' in args.model:
            model_params = {
                'embed_dim': 512,  # Intermediate embedding dimension
                'depth': 12,       # Intermediate number of transformer layers
                'num_heads': 8,    # Intermediate number of attention heads
                'decoder_embed_dim': 512,
                'decoder_depth': 8,
                'decoder_num_heads': 8,
                'patch_size': 16,
                'out_dim': 768
            }
        else:
            raise ValueError(f"Unknown medium model variant: {args.model}")
    elif args.model_size == 'small':
        if 'S-32' in args.model:
            model_params = {
                'embed_dim': 384,
                'depth': 12,
                'num_heads': 6,
                'decoder_embed_dim': 384,
                'decoder_depth': 8,
                'decoder_num_heads': 6,
                'patch_size': 32,
                'out_dim': 768
            }
        elif 'S-16' in args.model:
            model_params = {
                'embed_dim': 384,
                'depth': 12,
                'num_heads': 6,
                'decoder_embed_dim': 384,
                'decoder_depth': 8,
                'decoder_num_heads': 6,
                'patch_size': 16,
                'out_dim': 768
            }
        else:
            raise ValueError(f"Unknown small model variant: {args.model}")
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

                
    model = SimCLRViT(
        img_size=(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),  # ImageNet images are 224x224
        audio_size=(config.AUDIO.MELBINS, config.AUDIO.TARGET_LENGTH),  # Audioset spectrogram size
        text_seq_len=config.TEXT.MAX_SEQ_LENGTH,  # AGNews max sequence length
        patch_size=model_params['patch_size'],  # Set patch size to 8
        embed_dim=model_params['embed_dim'],
        out_dim=model_params['out_dim'],
        depth=model_params['depth'],
        num_heads=model_params['num_heads'],
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        split_projector=True
    ).to(device)

    
    # Log the model structure to the log file
    if misc.is_main_process():
        if not os.path.exists(args.log_file):
            os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        with open(args.log_file, 'a') as f:
            f.write("\nModel Structure:\n")
            print(model, file=f)  # Log the model structure

    model_without_ddp = model

    encoder_params = 0
    other_params = 0

    
    # Print encoder and decoder parameters separately
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name, param in model.named_parameters():
        if "blocks" in name and "decoder_blocks" not in name:
            encoder_params += param.numel()
        else:
            other_params += param.numel()
    
    if misc.is_main_process():
        print('number of trainable params:', n_parameters)
        print(f"\nTotal Encoder Parameters: {encoder_params}")
        print(f"Total Other Parameters: {other_params}")
    
        # Log to the file
        with open(args.log_file, 'a') as f:
            f.write(f"\nTotal Encoder Parameters: {encoder_params}\n")
            f.write(f"Total Other Parameters: {other_params}\n")
            f.write(f"Total Parameters: {n_parameters}\n")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if config.TRAIN.OPTIMIZER.LR is None:
        config.defrost()
        config.TRAIN.OPTIMIZER.LR = config.TRAIN.OPTIMIZER.BASE_LR * eff_batch_size / 256
        print(f"Setting learning rate to {config.TRAIN.OPTIMIZER.LR} based on effective batch size {eff_batch_size}")
        config.freeze()

    # write all config to the log file
    if misc.is_main_process():
        with open(args.log_file, 'a') as f:
            f.write("\nTraining Configuration:\n")
            print(config, file=f)
            f.write("\n")
            # write args to the log file
            f.write("Command-line Arguments:\n")
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
            f.write("\n")

    if num_gpus > 1 and not args.distributed:
        model = nn.DataParallel(model)
    
    elif args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True)
        model_without_ddp = model.module

    # Load pretrained weights if available
    start_epoch = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"Loading model for evaluation from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        # Get the checkpoint state dict
        pretrained_dict = checkpoint['model']
        model_dict = model_without_ddp.state_dict()
        filtered_dict = {}
        for k, v in model_dict.items():
            if k in pretrained_dict and v.shape == pretrained_dict[k].shape:
                filtered_dict[k] = pretrained_dict[k]
            else:
                filtered_dict[k] = v
                print(f"Skipping parameter {k} with shape mismatch or not found")

        model_without_ddp.load_state_dict(filtered_dict)
        start_epoch = checkpoint['epoch'] + 1

    
    if misc.is_main_process():
        print("Running zero-shot evaluation...")
        if config.DATA.NUM_CLASSES >= 5:
            top_k = (1, 5)
        else:
            top_k = (1,)
        results = zero_shot_evaluation(model, train_loader, valid_loader, start_epoch, device, config, top_k=top_k, modality=args.single_modality, remove_projection_head=args.zero_shot_remove_head) 
        
        with open(args.log_file, 'a') as f:
            f.write(f"Zero-shot evaluation results at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n")
            for k in top_k:
                f.write(f"Top-{k} Accuracy: {results[f'top_{k}_acc']:.2f}%\n")
        
        return

if __name__ == "__main__":
    args, cfg = parse_option()
    main(args, cfg)