import torch
import torch.nn as nn
import torchsort
import numpy as np
from config import cfg

# utils ==================================================================================

def get_loss(loss, mse_tails_pow, mse_tails_mult_fact):
    if loss == "mse":
        return MSELoss()
    elif loss == "mse_tails":
        return MSETailsLoss(mse_tails_pow, mse_tails_mult_fact)
    elif loss == "l1":
        return L1Loss()
    elif loss == "spearman":
        return SpearmanLoss()
    elif loss == "mse_to_spearman":
        return MSEToSpearman()

# loss ===================================================================================

# MSE

class MSELoss(nn.Module):
    """MSE loss"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred, target, **kw):
        return self.mse(pred, target)

# MSE Tails

class MSETailsLoss(nn.Module):
    def __init__(self, pow, mult_fact):
        super().__init__()
        print("Warning! mse_tails is only available for the Memento10k dataset.")
        self.mse = nn.MSELoss(reduction='none')
        self.pow = pow
        self.mult_fact = mult_fact
    def forward(self, pred, target, **kw):
        weights = self.mult_fact * torch.abs(target - cfg.DATASET.MEAN_MEM_SCORE.MEMENTO10K) ** self.pow
        weights = weights.to(dtype=torch.float).cuda()
        weights = (1 + weights)
        weights = weights / weights.mean()
        return (self.mse(pred, target) * weights).mean()

# L1

class L1Loss(nn.Module):
    """L1 loss"""
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()
    def forward(self, pred, target, **kw):
        return self.mae(pred, target)

# Spearman RC

def spearman_diff(pred, target):
    pred = pred.T
    target = target.T
    pred = torchsort.soft_rank(pred)
    target = torchsort.soft_rank(target)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()

class SpearmanLoss(nn.Module):
    """Spearman rank correlation loss"""
    def forward(self, pred, target, **kw):
        return - spearman_diff(pred, target) / 100

# combo

class MSEToSpearman(nn.Module):
    """Convex combination of MSE and Spearman rank correlation loss"""
    def __init__(self):
        super().__init__()
        self.mse      = nn.MSELoss()
        self.spearman = SpearmanLoss()

    def forward(self, pred, target, **kw):
        alpha = kw['alpha']
        return (1 - alpha) * self.mse(pred, target) + alpha * self.spearman(pred, target)


# distributions metrics ==================================================================

def kl_divergence(pred, target):
    return (pred * np.log(pred / target)).sum()

def js_divergence(pred, target):
    mean = (pred + target) / 2
    return (kl_divergence(pred, mean) + kl_divergence(target, mean)) / 2

