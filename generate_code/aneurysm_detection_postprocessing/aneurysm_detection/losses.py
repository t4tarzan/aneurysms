"""Loss functions for training detection models."""
import torch
import torch.nn.functional as F

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for classification."""
    pt = torch.sigmoid(pred)
    ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    loss = ce_loss * ((1 - pt) ** gamma)
    loss = (alpha * target + (1 - alpha) * (1 - target)) * loss
    return loss.mean()

def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss for regression."""
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean()
