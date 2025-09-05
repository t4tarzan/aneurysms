"""
Loss Functions for Aneurysm Detection Models

This module implements specialized loss functions for 3D aneurysm detection,
supporting both CPM-Net (center-point detection) and 3D-CNN-TR (hybrid detection) approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in object detection.
    Particularly useful for aneurysm detection where positive samples are rare.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (positive samples)
            gamma: Focusing parameter to down-weight easy examples
            reduction: Specifies the reduction to apply to the output
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predictions (logits) [N, C, ...]
            targets: Ground truth labels [N, ...]
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CenterPointLoss(nn.Module):
    """
    Loss function for CPM-Net center-point detection.
    Combines heatmap loss, size regression loss, and offset regression loss.
    """
    
    def __init__(self, 
                 heatmap_weight: float = 1.0,
                 size_weight: float = 0.1,
                 offset_weight: float = 1.0):
        """
        Initialize center-point detection loss.
        
        Args:
            heatmap_weight: Weight for heatmap loss
            size_weight: Weight for size regression loss
            offset_weight: Weight for offset regression loss
        """
        super(CenterPointLoss, self).__init__()
        self.heatmap_weight = heatmap_weight
        self.size_weight = size_weight
        self.offset_weight = offset_weight
        self.focal_loss = FocalLoss(alpha=2.0, gamma=4.0)
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute center-point detection loss.
        
        Args:
            predictions: Model predictions containing 'heatmap', 'size', 'offset'
            targets: Ground truth targets containing 'heatmap', 'size', 'offset', 'mask'
            
        Returns:
            Dictionary containing individual losses and total loss
        """
        # Heatmap loss (focal loss for center point detection)
        heatmap_loss = self.focal_loss(predictions['heatmap'], targets['heatmap'])
        
        # Size regression loss (L1 loss for bounding box dimensions)
        size_mask = targets['mask'].unsqueeze(1).expand_as(predictions['size'])
        size_loss = F.l1_loss(
            predictions['size'] * size_mask,
            targets['size'] * size_mask,
            reduction='sum'
        ) / (size_mask.sum() + 1e-6)
        
        # Offset regression loss (L1 loss for center point offsets)
        offset_mask = targets['mask'].unsqueeze(1).expand_as(predictions['offset'])
        offset_loss = F.l1_loss(
            predictions['offset'] * offset_mask,
            targets['offset'] * offset_mask,
            reduction='sum'
        ) / (offset_mask.sum() + 1e-6)
        
        # Total loss
        total_loss = (self.heatmap_weight * heatmap_loss + 
                     self.size_weight * size_loss + 
                     self.offset_weight * offset_loss)
        
        return {
            'total_loss': total_loss,
            'heatmap_loss': heatmap_loss,
            'size_loss': size_loss,
            'offset_loss': offset_loss
        }


class DetectionLoss(nn.Module):
    """
    General detection loss for bounding box regression and classification.
    Used for 3D-CNN-TR and other detection approaches.
    """
    
    def __init__(self, 
                 classification_weight: float = 1.0,
                 regression_weight: float = 1.0,
                 iou_weight: float = 2.0):
        """
        Initialize detection loss.
        
        Args:
            classification_weight: Weight for classification loss
            regression_weight: Weight for bounding box regression loss
            iou_weight: Weight for IoU loss
        """
        super(DetectionLoss, self).__init__()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.iou_weight = iou_weight
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute detection loss.
        
        Args:
            predictions: Model predictions containing 'classification', 'regression'
            targets: Ground truth targets containing 'labels', 'boxes', 'valid_mask'
            
        Returns:
            Dictionary containing individual losses and total loss
        """
        # Classification loss (focal loss for aneurysm vs background)
        classification_loss = self.focal_loss(
            predictions['classification'], 
            targets['labels']
        )
        
        # Regression loss (smooth L1 loss for bounding box coordinates)
        valid_mask = targets['valid_mask'].unsqueeze(-1).expand_as(predictions['regression'])
        regression_loss = F.smooth_l1_loss(
            predictions['regression'] * valid_mask,
            targets['boxes'] * valid_mask,
            reduction='sum'
        ) / (valid_mask.sum() + 1e-6)
        
        # IoU loss for better localization
        iou_loss = self._compute_iou_loss(
            predictions['regression'], 
            targets['boxes'], 
            targets['valid_mask']
        )
        
        # Total loss
        total_loss = (self.classification_weight * classification_loss + 
                     self.regression_weight * regression_loss + 
                     self.iou_weight * iou_loss)
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'regression_loss': regression_loss,
            'iou_loss': iou_loss
        }
    
    def _compute_iou_loss(self, pred_boxes: torch.Tensor, 
                         target_boxes: torch.Tensor, 
                         valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU loss for better bounding box localization.
        
        Args:
            pred_boxes: Predicted bounding boxes [N, 6] (x, y, z, w, h, d)
            target_boxes: Target bounding boxes [N, 6]
            valid_mask: Mask for valid boxes [N]
            
        Returns:
            IoU loss value
        """
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
        
        # Extract valid boxes
        valid_pred = pred_boxes[valid_mask]
        valid_target = target_boxes[valid_mask]
        
        # Convert center format to corner format for IoU calculation
        pred_corners = self._center_to_corners(valid_pred)
        target_corners = self._center_to_corners(valid_target)
        
        # Compute 3D IoU
        iou = self._compute_3d_iou(pred_corners, target_corners)
        
        # IoU loss (1 - IoU)
        iou_loss = 1.0 - iou.mean()
        
        return iou_loss
    
    def _center_to_corners(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert center format (x, y, z, w, h, d) to corner format.
        
        Args:
            boxes: Boxes in center format [N, 6]
            
        Returns:
            Boxes in corner format [N, 6] (x1, y1, z1, x2, y2, z2)
        """
        x, y, z, w, h, d = boxes.unbind(-1)
        x1 = x - w / 2
        y1 = y - h / 2
        z1 = z - d / 2
        x2 = x + w / 2
        y2 = y + h / 2
        z2 = z + d / 2
        return torch.stack([x1, y1, z1, x2, y2, z2], dim=-1)
    
    def _compute_3d_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D IoU between two sets of boxes.
        
        Args:
            boxes1: First set of boxes [N, 6] (x1, y1, z1, x2, y2, z2)
            boxes2: Second set of boxes [N, 6]
            
        Returns:
            IoU values [N]
        """
        # Intersection coordinates
        inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        inter_z1 = torch.max(boxes1[:, 2], boxes2[:, 2])
        inter_x2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        inter_y2 = torch.min(boxes1[:, 4], boxes2[:, 4])
        inter_z2 = torch.min(boxes1[:, 5], boxes2[:, 5])
        
        # Intersection volume
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_d = torch.clamp(inter_z2 - inter_z1, min=0)
        inter_volume = inter_w * inter_h * inter_d
        
        # Box volumes
        volume1 = ((boxes1[:, 3] - boxes1[:, 0]) * 
                   (boxes1[:, 4] - boxes1[:, 1]) * 
                   (boxes1[:, 5] - boxes1[:, 2]))
        volume2 = ((boxes2[:, 3] - boxes2[:, 0]) * 
                   (boxes2[:, 4] - boxes2[:, 1]) * 
                   (boxes2[:, 5] - boxes2[:, 2]))
        
        # Union volume
        union_volume = volume1 + volume2 - inter_volume
        
        # IoU
        iou = inter_volume / (union_volume + 1e-6)
        return iou


class AuxiliaryLoss(nn.Module):
    """
    Auxiliary loss for models with additional supervision (e.g., artery segmentation).
    Used in 3D-CNN-TR model for vessel-aware training.
    """
    
    def __init__(self, weight: float = 0.5):
        """
        Initialize auxiliary loss.
        
        Args:
            weight: Weight for auxiliary loss relative to main detection loss
        """
        super(AuxiliaryLoss, self).__init__()
        self.weight = weight
        self.dice_loss = DiceLoss()
    
    def forward(self, aux_predictions: torch.Tensor, 
                aux_targets: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary loss for artery segmentation.
        
        Args:
            aux_predictions: Auxiliary predictions (artery segmentation) [N, C, D, H, W]
            aux_targets: Auxiliary targets (artery masks) [N, D, H, W]
            
        Returns:
            Auxiliary loss value
        """
        # Dice loss for segmentation
        aux_loss = self.dice_loss(aux_predictions, aux_targets)
        return self.weight * aux_loss


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    """
    
    def __init__(self, smooth: float = 1e-6):
        """
        Initialize Dice loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            predictions: Predicted segmentation [N, C, ...]
            targets: Target segmentation [N, ...]
            
        Returns:
            Dice loss value
        """
        # Apply softmax to predictions
        predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot encoding
        num_classes = predictions.shape[1]
        targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, -1, *range(1, len(targets.shape)))
        targets_one_hot = targets_one_hot.float()
        
        # Flatten tensors
        predictions_flat = predictions.view(predictions.shape[0], predictions.shape[1], -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
        
        # Compute Dice coefficient
        intersection = (predictions_flat * targets_flat).sum(dim=-1)
        union = predictions_flat.sum(dim=-1) + targets_flat.sum(dim=-1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss (1 - Dice coefficient)
        dice_loss = 1.0 - dice.mean()
        return dice_loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that can handle multiple loss components.
    Suitable for complex models with multiple outputs.
    """
    
    def __init__(self, 
                 loss_components: Dict[str, nn.Module],
                 loss_weights: Dict[str, float]):
        """
        Initialize combined loss.
        
        Args:
            loss_components: Dictionary of loss functions
            loss_weights: Dictionary of weights for each loss component
        """
        super(CombinedLoss, self).__init__()
        self.loss_components = nn.ModuleDict(loss_components)
        self.loss_weights = loss_weights
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary containing individual losses and total loss
        """
        losses = {}
        total_loss = 0.0
        
        for loss_name, loss_fn in self.loss_components.items():
            if loss_name in self.loss_weights:
                loss_value = loss_fn(predictions, targets)
                if isinstance(loss_value, dict):
                    # Handle complex loss functions that return multiple components
                    for sub_loss_name, sub_loss_value in loss_value.items():
                        losses[f"{loss_name}_{sub_loss_name}"] = sub_loss_value
                        if sub_loss_name == 'total_loss':
                            total_loss += self.loss_weights[loss_name] * sub_loss_value
                else:
                    losses[loss_name] = loss_value
                    total_loss += self.loss_weights[loss_name] * loss_value
        
        losses['total_loss'] = total_loss
        return losses


# Factory functions for creating loss functions
def create_cpm_net_loss(heatmap_weight: float = 1.0,
                       size_weight: float = 0.1,
                       offset_weight: float = 1.0) -> CenterPointLoss:
    """
    Create loss function for CPM-Net model.
    
    Args:
        heatmap_weight: Weight for heatmap loss
        size_weight: Weight for size regression loss
        offset_weight: Weight for offset regression loss
        
    Returns:
        Configured CenterPointLoss instance
    """
    return CenterPointLoss(
        heatmap_weight=heatmap_weight,
        size_weight=size_weight,
        offset_weight=offset_weight
    )


def create_cnn_tr_3d_loss(classification_weight: float = 1.0,
                         regression_weight: float = 1.0,
                         iou_weight: float = 2.0,
                         auxiliary_weight: float = 0.5) -> CombinedLoss:
    """
    Create loss function for 3D-CNN-TR model.
    
    Args:
        classification_weight: Weight for classification loss
        regression_weight: Weight for regression loss
        iou_weight: Weight for IoU loss
        auxiliary_weight: Weight for auxiliary loss
        
    Returns:
        Configured CombinedLoss instance
    """
    loss_components = {
        'detection': DetectionLoss(
            classification_weight=classification_weight,
            regression_weight=regression_weight,
            iou_weight=iou_weight
        ),
        'auxiliary': AuxiliaryLoss(weight=auxiliary_weight)
    }
    
    loss_weights = {
        'detection': 1.0,
        'auxiliary': 1.0
    }
    
    return CombinedLoss(loss_components, loss_weights)


# Loss configuration dictionaries
LOSS_CONFIGS = {
    'cpm_net': {
        'loss_type': 'center_point',
        'heatmap_weight': 1.0,
        'size_weight': 0.1,
        'offset_weight': 1.0
    },
    'cnn_tr_3d': {
        'loss_type': 'combined',
        'classification_weight': 1.0,
        'regression_weight': 1.0,
        'iou_weight': 2.0,
        'auxiliary_weight': 0.5
    }
}