import torch
import numpy as np
import math



class GradientNormBalancer:
    def __init__(self, alpha=0.9, initial_weights=None):
        self.alpha = alpha
        self.grad_norm_ema = initial_weights or {'image': 1.0, 'audio': 1.0, 'text': 1.0}
        
    def get_weights(self, current_grad_norms):
        """Update EMA of gradient norms and return inverse weights"""
        # Update exponential moving average
        for mod in current_grad_norms:
            if mod in self.grad_norm_ema:
                self.grad_norm_ema[mod] = (
                    self.alpha * self.grad_norm_ema[mod] + 
                    (1 - self.alpha) * current_grad_norms[mod]
                )
        
        # Compute weights: higher weight for modalities with smaller gradient norms
        max_norm = max(self.grad_norm_ema.values())
        weights = {}
        for mod in self.grad_norm_ema:
            # Inverse relationship: smaller grad norm → higher weight
            weights[mod] = max_norm / (self.grad_norm_ema[mod] + 1e-8)
            # Clamp weights to reasonable range
            weights[mod] = min(max(weights[mod], 0.5), 2.0)
        
        return weights