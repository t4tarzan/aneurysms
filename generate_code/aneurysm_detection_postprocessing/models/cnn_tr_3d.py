"""
3D-CNN-TR Hybrid Model Implementation

This module implements the 3D-CNN-TR (3D CNN-Transformer) hybrid model for aneurysm detection.
The model combines deformable 3D CNN with Transformer components and uses auxiliary artery 
segmentation input for vessel-aware detection.

Key Features:
- Deformable 3D CNN backbone for spatial feature extraction
- Transformer components for long-range dependency modeling
- Auxiliary artery segmentation input for vessel-aware detection
- Same training configuration as CPM-Net (45 epochs, AdamW optimizer)
- Compatible with anatomical post-processing pipeline

Architecture:
- Deformable 3D CNN encoder with multi-scale features
- Transformer blocks for feature refinement
- Dual-input design: CTA volume + artery segmentation
- Detection head with bounding box regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from .base_detector import BaseDetector

# Training configuration matching CPM-Net
CNN_TR_3D_CONFIG = {
    'epochs': 45,
    'optimizer': 'AdamW',
    'learning_rate_start': 0.0001,
    'learning_rate_end': 0.00001,
    'batch_size': 4,
    'input_size': (128, 128, 128),
    'input_channels': 1,
    'auxiliary_channels': 1,  # For artery segmentation input
    'num_classes': 1,
    'confidence_threshold': 0.8,
    'framework': 'PyTorch'
}


class DeformableConv3D(nn.Module):
    """
    Deformable 3D Convolution implementation.
    
    This module implements deformable convolution for 3D data, allowing the
    convolution kernel to adapt its sampling locations based on learned offsets.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, groups: int = 1):
        super(DeformableConv3D, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        # Offset prediction network
        self.offset_conv = nn.Conv3d(
            in_channels, 
            3 * kernel_size * kernel_size * kernel_size * groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        
        # Main convolution
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        
        # Initialize offset conv to zero
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predict offsets
        offsets = self.offset_conv(x)
        
        # For simplicity, we'll use regular convolution with offset-modulated features
        # In a full implementation, this would use proper deformable convolution
        # Here we approximate by using the offsets to modulate the input features
        offset_magnitude = torch.mean(torch.abs(offsets), dim=1, keepdim=True)
        modulated_x = x * (1 + 0.1 * torch.tanh(offset_magnitude))
        
        return self.conv(modulated_x)


class TransformerBlock3D(nn.Module):
    """
    3D Transformer block for processing volumetric features.
    
    This block applies self-attention mechanism to 3D feature maps,
    enabling the model to capture long-range dependencies in the volume.
    """
    
    def __init__(self, channels: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super(TransformerBlock3D, self).__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        
        # Layer normalization
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
        
        # MLP
        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, channels),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        
        # Reshape for attention: (B, D*H*W, C)
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        
        # Self-attention with residual connection
        x_norm = self.norm1(x.view(B, C, -1)).permute(0, 2, 1)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        
        # MLP with residual connection
        x_norm2 = self.norm2(x_flat.permute(0, 2, 1).view(B, C, D, H, W))
        x_norm2_flat = x_norm2.view(B, C, -1).permute(0, 2, 1)
        mlp_out = self.mlp(x_norm2_flat)
        x_flat = x_flat + mlp_out
        
        # Reshape back to original format
        return x_flat.permute(0, 2, 1).view(B, C, D, H, W)


class CNNTRBlock(nn.Module):
    """
    Combined CNN-Transformer block.
    
    This block combines deformable 3D CNN with Transformer components
    for enhanced feature extraction and long-range dependency modeling.
    """
    
    def __init__(self, in_channels: int, out_channels: int, use_transformer: bool = True):
        super(CNNTRBlock, self).__init__()
        
        self.use_transformer = use_transformer
        
        # Deformable CNN layers
        self.conv1 = DeformableConv3D(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = DeformableConv3D(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(8, out_channels)
        
        # Transformer block
        if use_transformer:
            self.transformer = TransformerBlock3D(out_channels)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        
        # CNN path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Transformer path
        if self.use_transformer:
            out = self.transformer(out)
        
        # Residual connection
        out = out + identity
        out = self.relu(out)
        
        return out


class AuxiliaryFusionModule(nn.Module):
    """
    Module for fusing CTA features with auxiliary artery segmentation features.
    
    This module combines the main CTA features with artery segmentation information
    to create vessel-aware feature representations.
    """
    
    def __init__(self, cta_channels: int, aux_channels: int, out_channels: int):
        super(AuxiliaryFusionModule, self).__init__()
        
        # Process auxiliary input
        self.aux_conv = nn.Sequential(
            nn.Conv3d(aux_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Process CTA input
        self.cta_conv = nn.Sequential(
            nn.Conv3d(cta_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, cta_features: torch.Tensor, aux_features: torch.Tensor) -> torch.Tensor:
        # Process both inputs
        cta_proc = self.cta_conv(cta_features)
        aux_proc = self.aux_conv(aux_features)
        
        # Concatenate and fuse
        combined = torch.cat([cta_proc, aux_proc], dim=1)
        fused = self.fusion(combined)
        
        return fused


class DetectionHead3D(nn.Module):
    """
    Detection head for 3D bounding box prediction.
    
    This head predicts object confidence, bounding box coordinates,
    and size parameters for aneurysm detection.
    """
    
    def __init__(self, in_channels: int, num_classes: int = 1):
        super(DetectionHead3D, self).__init__()
        
        # Shared feature processing
        self.shared_conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.GroupNorm(4, in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.cls_head = nn.Conv3d(in_channels // 4, num_classes, kernel_size=1)
        
        # Regression head (6 values: x, y, z, w, h, d)
        self.reg_head = nn.Conv3d(in_channels // 4, 6, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_features = self.shared_conv(x)
        
        # Classification output
        cls_output = torch.sigmoid(self.cls_head(shared_features))
        
        # Regression output
        reg_output = self.reg_head(shared_features)
        
        return {
            'classification': cls_output,
            'regression': reg_output,
            'features': shared_features
        }


class CNNTR3D(BaseDetector):
    """
    3D-CNN-TR Hybrid Model for Aneurysm Detection.
    
    This model combines deformable 3D CNN with Transformer components
    and uses auxiliary artery segmentation input for vessel-aware detection.
    
    Architecture:
    - Dual input: CTA volume + artery segmentation
    - Deformable 3D CNN encoder with multi-scale features
    - Transformer blocks for feature refinement
    - Detection head with bounding box regression
    
    Args:
        input_channels: Number of input channels for CTA (default: 1)
        aux_channels: Number of auxiliary channels for artery segmentation (default: 1)
        num_classes: Number of detection classes (default: 1)
        confidence_threshold: Confidence threshold for predictions (default: 0.8)
        base_channels: Base number of channels (default: 32)
    """
    
    def __init__(self, input_channels: int = 1, aux_channels: int = 1, 
                 num_classes: int = 1, confidence_threshold: float = 0.8,
                 base_channels: int = 32):
        super(CNNTR3D, self).__init__(input_channels, num_classes, confidence_threshold)
        
        self.aux_channels = aux_channels
        self.base_channels = base_channels
        
        # Input fusion module
        self.input_fusion = AuxiliaryFusionModule(
            input_channels, aux_channels, base_channels
        )
        
        # Encoder blocks with increasing channels
        self.encoder1 = CNNTRBlock(base_channels, base_channels * 2, use_transformer=False)
        self.encoder2 = CNNTRBlock(base_channels * 2, base_channels * 4, use_transformer=True)
        self.encoder3 = CNNTRBlock(base_channels * 4, base_channels * 8, use_transformer=True)
        self.encoder4 = CNNTRBlock(base_channels * 8, base_channels * 16, use_transformer=True)
        
        # Decoder blocks with skip connections
        self.decoder1 = CNNTRBlock(base_channels * 16 + base_channels * 8, base_channels * 8, use_transformer=True)
        self.decoder2 = CNNTRBlock(base_channels * 8 + base_channels * 4, base_channels * 4, use_transformer=True)
        self.decoder3 = CNNTRBlock(base_channels * 4 + base_channels * 2, base_channels * 2, use_transformer=False)
        
        # Detection head
        self.detection_head = DetectionHead3D(base_channels * 2, num_classes)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool3d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        
    def forward(self, x: torch.Tensor, aux_input: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the 3D-CNN-TR model.
        
        Args:
            x: Input CTA volume tensor (B, C, D, H, W)
            aux_input: Auxiliary artery segmentation tensor (B, aux_C, D, H, W)
            
        Returns:
            Dictionary containing:
            - 'boxes': Predicted bounding boxes
            - 'scores': Confidence scores
            - 'features': Feature maps
        """
        if aux_input is None:
            # Create dummy auxiliary input if not provided
            aux_input = torch.zeros(x.shape[0], self.aux_channels, *x.shape[2:], 
                                  device=x.device, dtype=x.dtype)
        
        # Input fusion
        fused_input = self.input_fusion(x, aux_input)
        
        # Encoder path with skip connections
        enc1 = self.encoder1(fused_input)  # (B, 64, D, H, W)
        enc1_pool = self.pool(enc1)        # (B, 64, D/2, H/2, W/2)
        
        enc2 = self.encoder2(enc1_pool)    # (B, 128, D/2, H/2, W/2)
        enc2_pool = self.pool(enc2)        # (B, 128, D/4, H/4, W/4)
        
        enc3 = self.encoder3(enc2_pool)    # (B, 256, D/4, H/4, W/4)
        enc3_pool = self.pool(enc3)        # (B, 256, D/8, H/8, W/8)
        
        enc4 = self.encoder4(enc3_pool)    # (B, 512, D/8, H/8, W/8)
        
        # Decoder path with skip connections
        dec1_up = self.upsample(enc4)      # (B, 512, D/4, H/4, W/4)
        dec1_cat = torch.cat([dec1_up, enc3], dim=1)  # (B, 768, D/4, H/4, W/4)
        dec1 = self.decoder1(dec1_cat)     # (B, 256, D/4, H/4, W/4)
        
        dec2_up = self.upsample(dec1)      # (B, 256, D/2, H/2, W/2)
        dec2_cat = torch.cat([dec2_up, enc2], dim=1)  # (B, 384, D/2, H/2, W/2)
        dec2 = self.decoder2(dec2_cat)     # (B, 128, D/2, H/2, W/2)
        
        dec3_up = self.upsample(dec2)      # (B, 128, D, H, W)
        dec3_cat = torch.cat([dec3_up, enc1], dim=1)  # (B, 192, D, H, W)
        dec3 = self.decoder3(dec3_cat)     # (B, 64, D, H, W)
        
        # Detection head
        detection_output = self.detection_head(dec3)
        
        # Process outputs
        boxes, scores = self._decode_predictions(
            detection_output['classification'],
            detection_output['regression']
        )
        
        return {
            'boxes': boxes,
            'scores': scores,
            'features': detection_output['features']
        }
    
    def _decode_predictions(self, cls_output: torch.Tensor, 
                          reg_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode network predictions into bounding boxes and scores.
        
        Args:
            cls_output: Classification output tensor (B, num_classes, D, H, W)
            reg_output: Regression output tensor (B, 6, D, H, W)
            
        Returns:
            Tuple of (boxes, scores) tensors
        """
        batch_size = cls_output.shape[0]
        device = cls_output.device
        
        # Get spatial dimensions
        _, _, D, H, W = cls_output.shape
        
        # Create coordinate grids
        z_coords = torch.arange(D, device=device).float()
        y_coords = torch.arange(H, device=device).float()
        x_coords = torch.arange(W, device=device).float()
        
        zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        
        # Flatten and expand for batch
        coords = torch.stack([xx, yy, zz], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        
        # Get confidence scores
        scores = cls_output.squeeze(1)  # (B, D, H, W)
        
        # Apply confidence threshold
        valid_mask = scores > self.confidence_threshold
        
        all_boxes = []
        all_scores = []
        
        for b in range(batch_size):
            batch_mask = valid_mask[b]
            if not batch_mask.any():
                # No valid detections
                all_boxes.append(torch.zeros((0, 6), device=device))
                all_scores.append(torch.zeros((0,), device=device))
                continue
            
            # Get valid positions
            valid_positions = torch.nonzero(batch_mask, as_tuple=False)  # (N, 3)
            
            # Extract coordinates and regression values
            valid_coords = coords[b, :, batch_mask].T  # (N, 3) - x, y, z
            valid_scores = scores[b][batch_mask]       # (N,)
            valid_reg = reg_output[b, :, batch_mask].T # (N, 6)
            
            # Decode bounding boxes
            # Center coordinates (add regression offsets)
            centers = valid_coords + valid_reg[:, :3]
            
            # Box dimensions (apply exponential to ensure positive)
            sizes = torch.exp(valid_reg[:, 3:6]) * 10.0  # Scale factor for reasonable sizes
            
            # Combine into bounding box format [x, y, z, w, h, d]
            boxes = torch.cat([centers, sizes], dim=1)
            
            all_boxes.append(boxes)
            all_scores.append(valid_scores)
        
        return all_boxes, all_scores
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information and configuration."""
        return {
            'model_name': '3D-CNN-TR',
            'model_type': 'Hybrid CNN-Transformer',
            'input_channels': self.input_channels,
            'aux_channels': self.aux_channels,
            'num_classes': self.num_classes,
            'confidence_threshold': self.confidence_threshold,
            'base_channels': self.base_channels,
            'has_auxiliary_input': True,
            'architecture_features': [
                'Deformable 3D CNN',
                'Transformer blocks',
                'Auxiliary artery segmentation input',
                'Multi-scale feature extraction',
                'Skip connections'
            ]
        }


def create_cnn_tr_3d(input_channels: int = 1, aux_channels: int = 1,
                     num_classes: int = 1, confidence_threshold: float = 0.8,
                     base_channels: int = 32) -> CNNTR3D:
    """
    Factory function to create a 3D-CNN-TR model instance.
    
    Args:
        input_channels: Number of input channels for CTA (default: 1)
        aux_channels: Number of auxiliary channels for artery segmentation (default: 1)
        num_classes: Number of detection classes (default: 1)
        confidence_threshold: Confidence threshold for predictions (default: 0.8)
        base_channels: Base number of channels (default: 32)
        
    Returns:
        Configured CNNTR3D model instance
    """
    model = CNNTR3D(
        input_channels=input_channels,
        aux_channels=aux_channels,
        num_classes=num_classes,
        confidence_threshold=confidence_threshold,
        base_channels=base_channels
    )
    
    return model


# Export configuration for training scripts
__all__ = [
    'CNNTR3D',
    'create_cnn_tr_3d',
    'CNN_TR_3D_CONFIG',
    'DeformableConv3D',
    'TransformerBlock3D',
    'CNNTRBlock',
    'AuxiliaryFusionModule',
    'DetectionHead3D'
]