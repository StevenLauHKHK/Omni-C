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
from models.SimCLR_ViT_V2 import SimCLRViTV2
from models.SimCLR_ViT_SBoRA import SimCLRViT_SBoRA
from models.Align_Encoder import AlignEncoder
from utils.dataloader import MultimodalDataLoader
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
    parser.add_argument('--audio-data-path', default='/path/to/audioset/', type=str, help='Path to audio dataset')
    parser.add_argument('--audio-dataset', default='vggsound', type=str, help='Audio dataset name')
    parser.add_argument('--text-data-path', default='/path/to/AG_NEWS/', type=str, help='Path to text dataset')
    parser.add_argument('--text-dataset', default='agnews', type=str, help='Text dataset name')
    
    # training arguments
    parser.add_argument('--early_stop_patience', default=3, type=int, help='Early stopping patience epochs')
    parser.add_argument('--eval_every_n', default=1, type=int, help='Number of epochs for evaluation')
    parser.add_argument('--save_every_n', default=1, type=int, help='Number of epochs for saving a checkpoint')
    parser.add_argument('--single-modality', type=str, choices=['image', 'audio', 'text'], help='Specify which modality to train on')
    parser.add_argument('--selected-multimodality', type=str, nargs='+', choices=['image', 'audio', 'text'], help='Specify which modalities to include in multimodal training (e.g., --selected-multimodality image audio)')
    parser.add_argument('--simclr', action='store_true', help='Enable SimCLR training')
    parser.add_argument('--max-train-batches', default=None, type=int, help='Maximum number of batches to train per epoch')
    parser.add_argument('--log-file', default="training_log.txt", type=str, help="Path to the log file")  # New argument
    
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


def save_augmented_images(batch1, batch2, labels, save_dir, batch_idx, num_images=8, mean=None, std=None):
    """
    Save augmented image pairs for SimCLR contrastive learning visualization.
    
    Args:
        batch1: First augmented batch [B, C, H, W]
        batch2: Second augmented batch [B, C, H, W]
        labels: Labels for the batch [B]
        save_dir: Directory to save images
        batch_idx: Batch index for naming
        num_images: Number of image pairs to save
        mean: Normalization mean for denormalization
        std: Normalization std for denormalization
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Use default ImageNet normalization if not provided
    if mean is None:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(batch1.device)
    if std is None:
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(batch1.device)
    
    # Denormalize images
    batch1_denorm = batch1 * std + mean
    batch2_denorm = batch2 * std + mean
    
    # Clamp to valid range
    batch1_denorm = torch.clamp(batch1_denorm, 0, 1)
    batch2_denorm = torch.clamp(batch2_denorm, 0, 1)
    
    # Limit number of images to save
    num_images = min(num_images, batch1.size(0))

        # Save individual image pairs
    for i in range(num_images):
        label = labels[i].item() if labels is not None else 'unknown'
        
        # Save first augmentation
        torchvision.utils.save_image(
            batch1_denorm[i], 
            os.path.join(save_dir, f'batch_{batch_idx}_img_{i}_label_{label}_aug1.png')
        )
        
        # Save second augmentation
        torchvision.utils.save_image(
            batch2_denorm[i], 
            os.path.join(save_dir, f'batch_{batch_idx}_img_{i}_label_{label}_aug2.png')
        )
    
    # Create a comparison grid showing pairs side by side
    if num_images >= 4:
        # Create grid with pairs side by side
        grid_images = []
        for i in range(min(4, num_images)):  # Show up to 4 pairs
            grid_images.extend([batch1_denorm[i], batch2_denorm[i]])
        
        # Create 2x4 grid (2 rows, 4 columns) showing 4 pairs
        grid = torchvision.utils.make_grid(grid_images, nrow=4, padding=2, pad_value=1.0)
        torchvision.utils.save_image(
            grid, 
            os.path.join(save_dir, f'batch_{batch_idx}_augmentation_pairs_grid.png')
        )
    
    print(f"Saved {num_images} augmented image pairs to {save_dir}")

def save_augmented_spectrograms(batch1, batch2, labels, save_dir, batch_idx, num_spectrograms=5):
    """
    Save augmented spectrogram pairs for SimCLR contrastive learning visualization.
    
    Args:
        batch1: First augmented batch [B, C, F, T]
        batch2: Second augmented batch [B, C, F, T]
        labels: Labels for the batch [B] or [B, num_classes] for one-hot
        save_dir: Directory to save spectrograms
        batch_idx: Batch index for naming
        num_spectrograms: Number of spectrogram pairs to save
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Denormalization parameters (adjust based on your audio config)
    norm_mean = -4.2677393
    norm_std = 4.5689974
    
    # Denormalize spectrograms
    batch1_denorm = batch1 * (norm_std * 2) + norm_mean
    batch2_denorm = batch2 * (norm_std * 2) + norm_mean
    
    # Limit number of spectrograms to save
    num_spectrograms = min(num_spectrograms, batch1.size(0))
    
    for i in range(num_spectrograms):
        if labels is not None:
            # Handle one-hot encoded labels
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                # One-hot encoded labels - get indices of all active classes
                active_classes = torch.where(labels[i] == 1)[0]
                if len(active_classes) > 0:
                    # Convert to list of integers and join with underscores
                    label_str = '_'.join([str(cls.item()) for cls in active_classes])
                else:
                    label_str = 'no_label'
            else:
                # Regular integer labels
                label_str = str(labels[i].item())
        else:
            label_str = 'unknown'
        
        # Create figure for this spectrogram pair
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # First augmentation
        axes[0].imshow(
            batch1_denorm[i].cpu().detach().numpy().squeeze(), 
            aspect='auto', origin='lower', cmap='viridis'
        )
        axes[0].set_title(f"Augmentation 1 (Label: {label_str})")
        axes[0].set_xlabel("Time Frames")
        axes[0].set_ylabel("Frequency Bins")
        
        # Second augmentation  
        axes[1].imshow(
            batch2_denorm[i].cpu().detach().numpy().squeeze(), 
            aspect='auto', origin='lower', cmap='viridis'
        )
        axes[1].set_title(f"Augmentation 2 (Label: {label_str})")
        axes[1].set_xlabel("Time Frames")
        axes[1].set_ylabel("Frequency Bins")
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f'batch_{batch_idx}_spec_{i}_label_{label_str}_aug_pair.png'), 
            dpi=150, bbox_inches='tight'
        )
        plt.close()
    
    print(f"Saved {num_spectrograms} augmented spectrogram pairs to {save_dir}")

def save_augmented_text(batch1, batch2, labels, save_dir, batch_idx, vocab, num_sentences=5):
    """
    Save augmented text pairs for SimCLR contrastive learning visualization.
    
    Args:
        batch1: First augmented batch [B, L] - token IDs
        batch2: Second augmented batch [B, L] - token IDs  
        labels: Labels for the batch [B]
        save_dir: Directory to save text
        batch_idx: Batch index for naming
        vocab: Vocabulary dictionary mapping token IDs to tokens
        num_sentences: Number of sentence pairs to save
    """
    if vocab is None:
        print("Vocabulary not provided, skipping text augmentation saving")
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert token IDs to text
    batch1_tokens = batch1.cpu().numpy()
    batch2_tokens = batch2.cpu().numpy()
    
    # Limit number of sentences to save
    num_sentences = min(num_sentences, batch1.size(0))
    
    with open(os.path.join(save_dir, f'batch_{batch_idx}_augmented_text_pairs.txt'), 'w') as f:
        f.write("SimCLR Augmented Text Pairs\n")
        f.write("=" * 60 + "\n\n")
        
        for i in range(num_sentences):
            label = labels[i].item() if labels is not None else 'unknown'
            
            # Convert first augmentation to text
            tokens1 = [vocab.get(int(t), "[UNK]") for t in batch1_tokens[i] if t != 0]  # Skip padding
            text1 = " ".join(tokens1)
            
            # Convert second augmentation to text
            tokens2 = [vocab.get(int(t), "[UNK]") for t in batch2_tokens[i] if t != 0]  # Skip padding
            text2 = " ".join(tokens2)

            f.write(f"Pair {i+1} (Label: {label})\n")
            f.write(f"Augmentation 1: {text1}\n")
            f.write(f"Augmentation 2: {text2}\n")
            f.write("-" * 60 + "\n\n")
    
    print(f"Saved {num_sentences} augmented text pairs to {save_dir}")


def train_simclr_one_epoch(model, model_without_ddp, train_loader, optimizer, epoch, device, loss_scaler, config, modalities=['image', 'audio', 'text'], accumulation_steps=1, debug_dir="debug_outputs", max_batches=None, is_single_modality=False, temperature=0.5):
    """
    Train the model using SimCLR contrastive learning for one epoch.
    """
    print(f"Starting SimCLR training for epoch {epoch + 1}...")
    os.makedirs(debug_dir, exist_ok=True)

    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    
    # Initialize counts for each modality
    iter_counters = {}
    for modality in modalities:
        iter_counters[modality] = 0

    if is_single_modality:
        num_modalities = 1
    else:
        num_modalities = len(iter_counters)


    
    contrastive_loss_fn = NTXentLoss(temperature=temperature)

    optimizer.zero_grad()

    for data_iter_step, (batch1, batch2, labels, modalities) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
        if max_batches is not None and data_iter_step >= max_batches:
            print(f"Stopping training after {max_batches} batches (max-train-batches reached).")
            break

        batch1 = batch1.to(device, non_blocking=True)
        batch2 = batch2.to(device, non_blocking=True)
        modality = modalities[0]

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accumulation_steps == 0:
            adjust_learning_rate(optimizer, (iter_counters[modality] / len(train_loader) * num_modalities) + epoch, config)
            # print(f"Adjusting learning rate for {modality} at step {data_iter_step}, epoch {((iter_counters[modality] / len(train_loader) / num_modalities) + epoch):.2f}, counter {iter_counters[modality]}")

        # Increment the counter for the current modality
        iter_counters[modality] += 1

        with torch.cuda.amp.autocast():
            # Forward pass
            z1, z2 = model(batch1, batch2, modality)
            # Compute contrastive loss
            loss = contrastive_loss_fn(z1, z2)

        
        # Ensure loss is a scalar
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Normalize loss for gradient accumulation
        loss /= accumulation_steps
        loss_scaler(loss, optimizer, parameters=model.parameters(), 
                    update_grad=(data_iter_step + 1) % accumulation_steps == 0)

        # Zero gradients after accumulation step
        if (data_iter_step + 1) % accumulation_steps == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # Reduce loss across processes for distributed training
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        
        # Update MetricLogger with modality-specific losses
        metric_logger.update(**{f"loss_{modality}": loss_value_reduce})
        metric_logger.update(loss=loss_value_reduce)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # Save debug outputs every 100 batches
        if iter_counters[modality] % 100 == 0 and misc.is_main_process():
            save_dir = os.path.join(debug_dir, f"epoch_{epoch+1}_modality_{modality}_iter_{iter_counters[modality]}")
            os.makedirs(save_dir, exist_ok=True)

            if modality == 'image':
                save_augmented_images(batch1, batch2, labels, save_dir, data_iter_step, num_images=8)
            elif modality == 'audio':
                save_augmented_spectrograms(batch1, batch2, labels, save_dir, data_iter_step, num_spectrograms=5)
            elif modality == 'text':
                vocab = train_loader.loaders['text'].dataset.vocab  # Ensure vocab is loaded
                save_augmented_text(batch1, batch2, labels, save_dir, data_iter_step, vocab, num_sentences=5)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # Log metrics
    print(f"Epoch {epoch} training completed. {metric_logger}")


    return metric_logger.loss.global_avg, {mod: metric_logger.meters[f"loss_{mod}"].global_avg for mod in ['image', 'audio', 'text']}




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

    
    train_loader = MultimodalDataLoader(
        dataset={'image': config.DATA.IMAGE_DATASET, 'audio': config.DATA.AUDIO_DATASET, 'text': config.DATA.TEXT_DATASET},
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=args.num_workers,
        shuffle=True,
        config=config,
        is_train=True,
        is_train_img_transform=True,
        audio_augmentation=True,
        text_augmentation=True,
        distributed=args.distributed,
        num_tasks=num_tasks,
        text_seq_len=config.TEXT.MAX_SEQ_LENGTH,
        rank=global_rank,
        accumulate_steps = args.accum_iter,
        selected_modalities = args.selected_multimodality,
        return_double_augmentations=True if args.simclr else False,
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
        split_projector=True,
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
    
    # Print encoder parameters
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


    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, config.TRAIN.OPTIMIZER.WEIGHT_DECAY)

    # Optimizer
    optimizer = optim.AdamW(
        param_groups,
        lr=config.TRAIN.OPTIMIZER.LR,
        betas=config.TRAIN.OPTIMIZER.BETAS,
        eps=config.TRAIN.OPTIMIZER.EPS,
        weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY
    )
    loss_scaler = NativeScaler()
    
    # Load pretrained weights if available
    start_epoch = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        try:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from checkpoint: {args.resume_checkpoint}, starting at epoch {start_epoch + 1}")
        except (IndexError, ValueError):
            print(f"Could not parse epoch from {args.resume_checkpoint}, starting from epoch 1")
            start_epoch = 0
    
    
    # Use the checkpoint directory from the argument
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if misc.is_main_process():
        print(f"Starting pretraining for {config.TRAIN.EPOCHS} epochs...")
        with open(args.log_file, 'a') as f:  # Use the log_file argument from args
            f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):

        for loader in train_loader.loaders.values():
            loader.sampler.set_epoch(epoch)
        
        avg_loss, avg_modality_losses = train_simclr_one_epoch(
            model, model_without_ddp, train_loader, optimizer, epoch, device, loss_scaler, config, modalities=args.selected_multimodality, debug_dir=args.debug_dir, max_batches=args.max_train_batches, temperature=0.5, accumulation_steps=args.accum_iter
        )
            
        if misc.is_main_process():
            # Basic training progress log
            log_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch+1}/{config.TRAIN.EPOCHS}, Loss: {avg_loss:.4f}, Image Loss: {avg_modality_losses['image']:.4f}, Audio Loss: {avg_modality_losses['audio']:.4f}, Text Loss: {avg_modality_losses['text']:.4f}"
            print(log_msg)
            with open(args.log_file, 'a') as f:
                f.write(log_msg + "\n")

        # Save checkpoint
        if (epoch + 1) % args.save_every_n == 0:
            state_dict = model.module.state_dict() if num_gpus > 1 else model.state_dict()
            misc.save_model(checkpoint_dir, config, epoch, model, model_without_ddp, optimizer, loss_scaler)
            print(f"Saved checkpoint: {checkpoint_dir}")


if __name__ == "__main__":
    args, cfg = parse_option()
    main(args, cfg)