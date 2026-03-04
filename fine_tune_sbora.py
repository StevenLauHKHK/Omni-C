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
    parser.add_argument('--resume-fine-tune-checkpoint', default=None, help='Resume checkpoint path')
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
    
    # eval arguments
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    
    # model arguments
    parser.add_argument('--model', default='vit', type=str, help='Model architecture')
    parser.add_argument('--model-size', default='small', choices=['small', 'medium', 'base'], 
                        help="Choose the model size: 'small', 'medium', or 'base'")

    # sbora arguments
    parser.add_argument('--sbora-rank', default=4, type=int, help='SBoRA rank')
    parser.add_argument('--sbora-alpha', default=16, type=int, help='SBoRA alpha scaling factor')
    parser.add_argument('--sbora-dropout', default=0.1, type=float, help='SBoRA dropout rate')
    parser.add_argument('--sbora-mode', default='FA', choices=['FA', 'FB'], help='SBoRA mode: FA or FB')
    parser.add_argument('--sbora-base-checkpoint', default=None, type=str, help='Path to base model checkpoint for loading')
    parser.add_argument('--disable-sbora-gradients', action='store_true', help='Disable SBoRA gradient')
    parser.add_argument('--freeze-patch-embedding', action='store_true', help='Freeze patch embedding layers')

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


def train_fine_tune_one_epoch(model, train_loader, optimizer, epoch, device, config, loss_scaler, accumulation_steps=1, is_single_modality=True, max_batches=None):
    """
    Conduct fine tuning on a pretrained simclr model.
    """
    print("Starting fine tuning...")

    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch + 1)
    print_freq = 20

    # Initialize counts for each modality
    iter_counters = {'image': 0, 'audio': 0, 'text': 0}
    if is_single_modality:
        num_modalities = 1
    else:
        num_modalities = len(iter_counters)

    # Initialize tracking variables for accuracy
    total = 0
    correct = 0

    optimizer.zero_grad()

    for data_iter_step, (batch, labels, modalities) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
        if max_batches is not None and data_iter_step >= max_batches:
            print(f"Stopping training after {max_batches} batches (max-train-batches reached).")
            break

        # Move data to device
        batch = batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        modality = modalities[0]

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accumulation_steps == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(train_loader) + epoch, config)

        # Increment the counter for the current modality
        iter_counters[modality] += 1

        # Only the forward/backward through classifier needs gradients
        # and can use mixed precision
        with torch.cuda.amp.autocast():
            # Forward pass through encoder (without gradient computation)
            if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                cls_token = model.module.forward_encoder(batch, modality=modality)
            else:
                cls_token = model.forward_encoder(batch, modality=modality)

            if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                logits = model.module.linear_classifier(cls_token)
            else:
                logits = model.linear_classifier(cls_token)

            loss = nn.CrossEntropyLoss()(logits, labels)
            
        # Calculate accuracy
        _, predicted = logits.max(1)
        batch_size = labels.size(0)
        total += batch_size
        correct += predicted.eq(labels).sum().item()
        accuracy = 100. * correct / total
            

        # Ensure loss is a scalar
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        # Normalize loss for gradient accumulation
        loss /= accumulation_steps
        loss_scaler(loss, optimizer, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accumulation_steps == 0)

        # Zero gradients after accumulation step
        if (data_iter_step + 1) % accumulation_steps == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # Reduce loss across processes for distributed training
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        accuracy_value_reduce = misc.all_reduce_mean(accuracy)

        # Update MetricLogger with modality-specific losses
        metric_logger.update(**{f"loss_{modality}": loss_value_reduce})
        metric_logger.update(loss=loss_value_reduce)
        metric_logger.update(accuracy=accuracy_value_reduce)

        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group['lr'])
            max_lr = max(max_lr, group['lr'])

        metric_logger.update(lr=max_lr)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # Log metrics
    print(f"Epoch {epoch + 1} fine tuning completed. {metric_logger}")

    return metric_logger.loss.global_avg, metric_logger.accuracy.global_avg


def validate_fine_tune(model, valid_loader, epoch, device, config, modality, is_single_modality=True, max_batches=None):
    """
    Validate the linear probe / fine tuning model on the validation set.
    """

    print("Starting validation...")
    
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Validation Epoch: [{}]'.format(epoch)
    print_freq = 20

    # Initialize counts for each modality
    iter_counters = {'image': 0, 'audio': 0, 'text': 0}

    if is_single_modality:
        num_modalities = 1
    else:
        num_modalities = len(iter_counters)

    # Keep classifier in eval mode too for validation
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.linear_classifier.eval()
    else:
        model.linear_classifier.eval()

    # Initialize counters
    total = 0
    correct = 0

    with torch.no_grad():
        for data_iter_step, (batch, labels, modalities) in enumerate(metric_logger.log_every(valid_loader, print_freq, header)):
            
            # Move data to device
            batch = batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            modality = modalities[0]

            # Forward pass through encoder
            if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                cls_token = model.module.forward_encoder(batch, modality=modality)
            else:
                cls_token = model.forward_encoder(batch, modality=modality)
                
                
                
            if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                logits = model.module.linear_classifier(cls_token)
            else:
                logits = model.linear_classifier(cls_token)
            
            loss = nn.CrossEntropyLoss()(logits, labels)

            # Calculate accuracy
            _, predicted = logits.max(1)
            batch_size = labels.size(0)
            total += batch_size
            correct += predicted.eq(labels).sum().item()
            accuracy = 100. * correct / total

            # Update metrics
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            torch.cuda.synchronize()

            # Reduce loss across processes for distributed training
            loss_value_reduce = misc.all_reduce_mean(loss_value)
            accuracy_value_reduce = misc.all_reduce_mean(accuracy)

            # Update metric logger
            metric_logger.update(**{f"loss_{modality}": loss_value_reduce})
            metric_logger.update(loss=loss_value_reduce)
            metric_logger.update(accuracy=accuracy_value_reduce)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # Calculate final metrics
    avg_loss = metric_logger.loss.global_avg
    final_accuracy = metric_logger.accuracy.global_avg
    
    print(f"Validation - Epoch {epoch} - Loss: {avg_loss:.4f}, Accuracy: {final_accuracy:.2f}%")
    
    return avg_loss, final_accuracy



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

    # Define the dataloaders
    if not args.eval:
        train_loader = SingleModalityDataLoader(
            modality = args.single_modality,
            dataset={'image': config.DATA.IMAGE_DATASET, 'audio': config.DATA.AUDIO_DATASET, 'text': config.DATA.TEXT_DATASET},
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=args.num_workers,
            shuffle=True,
            config=config,
            is_train=True,
            is_train_img_transform=True,
            audio_augmentation=True,
            text_augmentation=True,
            max_batches=args.max_train_batches,
            distributed=args.distributed,
            num_tasks=num_tasks,
            text_seq_len=config.TEXT.MAX_SEQ_LENGTH if args.single_modality == 'text' else None,
            rank=global_rank,
            enable_nsp=config.TEXT.ENABLE_NSP if args.single_modality == 'text' else False,
            return_double_augmentations=False
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
        return_double_augmentations=False
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


    sbora_config = {
        'rank': args.sbora_rank,
        'alpha': args.sbora_alpha,
        'dropout': args.sbora_dropout,
        'mode': args.sbora_mode
    }

    model = SimCLRViT_SBoRA(
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
        sbora_config=sbora_config,
        split_projector=True,
    ).to(device)
    
    start_epoch = 0
    model.linear_classifier = nn.Linear(model.embed_dim, config.DATA.NUM_CLASSES).to(device)

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


    # Setup SBoRA finetuning if applicable, freeze base weights
    if args.sbora_base_checkpoint:
        model_without_ddp.setup_for_sbora_finetuning(args.sbora_base_checkpoint, freeze_base_model=True, device=device)
    
    if args.freeze_patch_embedding:
        print("Freezing patch embedding as per user request.")
        model_without_ddp.freeze_patch_embedding()

    
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

    if num_gpus > 1 and not args.distributed:
        modal_param = model_without_ddp.module.parameters()
    else:
        modal_param = model_without_ddp.parameters()

    # Optimizer
    optimizer = optim.AdamW(
        modal_param,
        lr=config.TRAIN.OPTIMIZER.LR,
        betas=config.TRAIN.OPTIMIZER.BETAS,
        eps=config.TRAIN.OPTIMIZER.EPS,
        weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY
    )

    loss_scaler = NativeScaler()

    # Load pretrained weights if available
    start_epoch = 0
    if args.resume_fine_tune_checkpoint and os.path.exists(args.resume_fine_tune_checkpoint):
        print(f"Resuming fine tuning from checkpoint: {args.resume_fine_tune_checkpoint}")
        checkpoint = torch.load(args.resume_fine_tune_checkpoint, map_location=device)

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

        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        try:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from checkpoint: {args.resume_fine_tune_checkpoint}, starting at epoch {start_epoch + 1}")
        except (IndexError, ValueError):
            print(f"Could not parse epoch from {args.resume_fine_tune_checkpoint}, starting from epoch 1")
            start_epoch = 0

    if args.eval:
        modality = args.single_modality
            
        val_loss, accuracy = validate_fine_tune(model, valid_loader, start_epoch, device, config, modality, is_single_modality=True)
            
        if misc.is_main_process():
            log_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%"
            print(log_msg)
            with open(args.log_file, 'a') as f:
                f.write(log_msg + "\n")
        return

    # Use the checkpoint directory from the argument
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    if misc.is_main_process():
        print(f"Starting fine tuning for {config.TRAIN.EPOCHS} epochs...")
        with open(args.log_file, 'a') as f:  # Use the log_file argument from args
            f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        
        if args.distributed:
            train_loader.loader.sampler.set_epoch(epoch)
        

        print(f"fine tune training for epoch {epoch + 1}...")
        
        avg_loss, accuracy = train_fine_tune_one_epoch(model, train_loader, optimizer, epoch, device, config, loss_scaler)

        if misc.is_main_process():
            # Log training progress
            log_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch+1}/{config.TRAIN.EPOCHS}, Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%"
            print(log_msg)
            with open(args.log_file, 'a') as f:  # Use the log_file argument from args
                f.write(log_msg + "\n")

        # Save checkpoint
        if (epoch + 1) % args.save_every_n == 0:
            state_dict = model.module.state_dict() if num_gpus > 1 else model.state_dict()
            if args.linear_probe or args.simclr_linear_probe:
                misc.save_model(checkpoint_dir, config, epoch, model, model_without_ddp, optimizer, loss_scaler)
                print(f"Saved checkpoint: {checkpoint_dir}")
            else:
                misc.save_model(checkpoint_dir, config, epoch, model, model_without_ddp, optimizer, loss_scaler)
                print(f"Saved checkpoint: {checkpoint_dir}")

        # Evaluation (if needed)
        if (epoch + 1) % args.eval_every_n == 0:
            print(f"Evaluating model at epoch {epoch+1}...")
            # Add evaluation logic here if required

            modality = args.single_modality
            
            val_loss, accuracy = validate_fine_tune(model, valid_loader, epoch, device, config, modality, is_single_modality=True)
                
            if misc.is_main_process():
                log_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%"
                print(log_msg)
                with open(args.log_file, 'a') as f:
                    f.write(log_msg + "\n")

                if accuracy > best_val_accuracy:
                    best_val_accuracy = accuracy
                    epochs_without_improvement = 0
                    # Save the best model
                    misc.save_model(checkpoint_dir, config, epoch, model, model_without_ddp, optimizer, loss_scaler, best=True)
                    print(f"New best model saved with accuracy: {best_val_accuracy:.2f}%")
                else:
                    epochs_without_improvement += 1
                    print(f"No improvement in validation accuracy for {epochs_without_improvement} evaluation(s). Best accuracy remains: {best_val_accuracy:.2f}%")
    
                with open(args.log_file, 'a') as f:
                    f.write(f"Best validation accuracy so far: {best_val_accuracy:.2f}%\n")

                # Early stopping check
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epochs_without_improvement} evaluations without improvement.")
                    with open(args.log_file, 'a') as f:
                        f.write(f"Early stopping triggered after {epochs_without_improvement} evaluations without improvement at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    return
                
                    
                        
                        


if __name__ == "__main__":
    args, cfg = parse_option()
    main(args, cfg)