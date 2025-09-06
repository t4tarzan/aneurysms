"""
CPM-Net: Center Points Matching for 3D Object Detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .base_detector import BaseDetector

class BasicBlock3D(nn.Module):
    """Basic 3D convolution block with batch norm and ReLU."""
    def __init__(self, in_c, out_c, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm3d(out_c)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CPMBackbone(nn.Module):
    """3D CNN backbone for CPM-Net."""
    def __init__(self, in_c=1, base_c=32):
        super().__init__()
        # Initial convolution
        self.stem = nn.Sequential(
            BasicBlock3D(in_c, base_c, 7, 2, 3),
            nn.MaxPool3d(3, 2, 1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(base_c, base_c, 2)
        self.layer2 = self._make_layer(base_c, base_c*2, 2, 2)
        self.layer3 = self._make_layer(base_c*2, base_c*4, 2, 2)
        self.layer4 = self._make_layer(base_c*4, base_c*8, 2, 2)
        
        self.out_channels = base_c * 8
    
    def _make_layer(self, in_c, out_c, blocks, s=1):
        layers = [BasicBlock3D(in_c, out_c, 3, s, 1)]
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_c, out_c, 3, 1, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)        # 1/4
        x = self.layer1(x)      # 1/4
        x = self.layer2(x)      # 1/8
        x = self.layer3(x)      # 1/16
        x = self.layer4(x)      # 1/32
        return x

class CPMMatchingHead(nn.Module):
    """Head for center point matching."""
    def __init__(self, in_c, num_classes):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Conv3d(in_c, in_c, 3, 1, 1, bias=False),
            nn.BatchNorm3d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_c, num_classes, 1)
        )
        self.reg_head = nn.Sequential(
            nn.Conv3d(in_c, in_c, 3, 1, 1, bias=False),
            nn.BatchNorm3d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_c, 6, 1)  # dx, dy, dz, w, h, d
        )
    
    def forward(self, x):
        return {
            'cls': self.cls_head(x),
            'reg': self.reg_head(x).exp()  # Ensure positive values for dims
        }

class CPMNet(BaseDetector):
    """CPM-Net for 3D object detection."""
    def __init__(self, in_c=1, base_c=32, num_classes=1):
        super().__init__()
        self.backbone = CPMBackbone(in_c, base_c)
        self.head = CPMMatchingHead(self.backbone.out_channels, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
    
    def loss(self, preds, targets):
        from ..losses import focal_loss, smooth_l1_loss
        
        cls_pred = preds['cls']  # [B, C, D, H, W]
        reg_pred = preds['reg']  # [B, 6, D, H, W]
        
        # Classification loss (Focal Loss)
        cls_target = targets['cls_map']  # [B, C, D, H, W]
        loss_cls = focal_loss(cls_pred, cls_target)
        
        # Regression loss (Smooth L1)
        reg_target = targets['reg_map']  # [B, 6, D, H, W]
        pos_mask = (cls_target > 0).float()  # [B, C, D, H, W]
        num_pos = max(1.0, pos_mask.sum().item())
        
        # Apply position mask and compute regression loss
        reg_loss = smooth_l1_loss(
            reg_pred * pos_mask,
            reg_target * pos_mask
        ) / num_pos
        
        return {
            'loss_cls': loss_cls,
            'loss_reg': reg_loss,
            'loss': loss_cls + reg_loss
        }
    
    def post_process(self, preds, score_thr=0.3, nms_thr=0.5, max_per_img=100):
        cls_scores = preds['cls'].sigmoid()  # [B, C, D, H, W]
        reg_preds = preds['reg']  # [B, 6, D, H, W]
        
        batch_size = cls_scores.size(0)
        detections = []
        
        for i in range(batch_size):
            scores = cls_scores[i]  # [C, D, H, W]
            boxes = reg_preds[i]    # [6, D, H, W]
            
            # Filter by score threshold
            mask = scores > score_thr
            if not mask.any():
                detections.append([])
                continue
                
            # Get indices of high-scoring locations
            indices = torch.nonzero(mask)  # [N, 4] (class, d, h, w)
            scores = scores[mask]  # [N]
            
            # Convert to (x1, y1, z1, x2, y2, z2) format
            locs = indices[:, 1:].float()  # [N, 3] (d, h, w)
            pred_boxes = torch.cat([
                locs - boxes[:3][:, mask].t(),
                locs + boxes[3:][:, mask].t()
            ], dim=1)  # [N, 6]
            
            # Apply NMS per class
            keep = batched_nms(pred_boxes, scores, nms_thr)
            keep = keep[:max_per_img]
            
            detections.append({
                'boxes': pred_boxes[keep],
                'scores': scores[keep],
                'labels': indices[keep, 0]
            })
            
        return detections

def create_cpm_net(in_c=1, base_c=32, num_classes=1):
    """Create CPM-Net model."""
    return CPMNet(in_c, base_c, num_classes)
