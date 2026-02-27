import torch
import torch.nn as nn
import torch.optim as optim
import math

# Custom Warmup + Cosine Scheduler
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr
        )
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.cosine_scheduler.step(self.current_epoch - self.warmup_epochs)
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]



def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config.TRAIN.LR_SCHEDULER.WARMUP_EPOCHS:
        lr = config.TRAIN.OPTIMIZER.LR * epoch / config.TRAIN.LR_SCHEDULER.WARMUP_EPOCHS
    else:
        lr = config.TRAIN.OPTIMIZER.MIN_LR + (config.TRAIN.OPTIMIZER.LR - config.TRAIN.OPTIMIZER.MIN_LR) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config.TRAIN.LR_SCHEDULER.WARMUP_EPOCHS) / (config.TRAIN.EPOCHS - config.TRAIN.LR_SCHEDULER.WARMUP_EPOCHS)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr