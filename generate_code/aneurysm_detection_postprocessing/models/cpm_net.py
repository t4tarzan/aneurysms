"""
CPM-Net: Center-Points Matching Network for 3D Aneurysm Detection

Implementation of the CPM-Net model from the paper:
"Automated anatomy-based post-processing reduces false positives and improved 
interpretability of deep learning intracranial aneurysm detection"

This model uses a 3D CNN architecture with center-points matching mechanism
for detecting intracranial aneurysms in CTA volumes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from .base_detector import BaseDetector


class CPMBlock(nn.Module):
    """Center-Points Matching Block for feature extraction and matching"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(CPMBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection if dimensions don't match
        self.skip_conv = None
        if in_channels != out_channels:
            self.skip_conv = nn.Conv3d(in_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
        
        out += identity
        out = self.relu(out)
        
        return out


class CenterPointsHead(nn.Module):
    """Head for center points detection and bounding box regression"""
    
    def __init__(self, in_channels: int, num_classes: int = 1):
        super(CenterPointsHead, self).__init__()
        
        # Center heatmap prediction
        self.center_head = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, num_classes, 1),
            nn.Sigmoid()
        )
        
        # Bounding box size regression (width, height, depth)
        self.size_head = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 3, 1),
            nn.ReLU(inplace=True)  # Ensure positive sizes
        )
        
        # Offset regression for precise localization
        self.offset_head = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 3, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        center_heatmap = self.center_head(x)
        size_pred = self.size_head(x)
        offset_pred = self.offset_head(x)
        
        return {
            'center_heatmap': center_heatmap,
            'size_pred': size_pred,
            'offset_pred': offset_pred
        }


class CPMNet(BaseDetector):
    """
    CPM-Net: Center-Points Matching Network for 3D Aneurysm Detection
    
    A CNN-based 3D aneurysm detector using center-points matching mechanism.
    Inherits from BaseDetector to provide standardized interface.
    """
    
    def __init__(self, 
                 input_channels: int = 1,
                 num_classes: int = 1,
                 confidence_threshold: float = 0.8,
                 base_channels: int = 32):
        """
        Initialize CPM-Net model
        
        Args:
            input_channels: Number of input channels (1 for CTA)
            num_classes: Number of detection classes (1 for aneurysm)
            confidence_threshold: Confidence threshold for filtering detections
            base_channels: Base number of channels for the network
        """
        super(CPMNet, self).__init__(input_channels, num_classes, confidence_threshold)
        
        self.base_channels = base_channels
        
        # Encoder pathway with downsampling
        self.encoder1 = CPMBlock(input_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)
        
        self.encoder2 = CPMBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)
        
        self.encoder3 = CPMBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2)
        
        self.encoder4 = CPMBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = CPMBlock(base_channels * 8, base_channels * 16)
        
        # Decoder pathway with upsampling
        self.upconv4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.decoder4 = CPMBlock(base_channels * 16, base_channels * 8)
        
        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.decoder3 = CPMBlock(base_channels * 8, base_channels * 4)
        
        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.decoder2 = CPMBlock(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
        self.decoder1 = CPMBlock(base_channels * 2, base_channels)
        
        # Detection head
        self.detection_head = CenterPointsHead(base_channels, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of CPM-Net
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Dictionary containing:
            - boxes: Predicted bounding boxes
            - scores: Confidence scores
            - features: Feature maps for analysis
        """
        # Encoder pathway
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        bottleneck = self.bottleneck(p4)
        
        # Decoder pathway with skip connections
        up4 = self.upconv4(bottleneck)
        up4 = torch.cat([up4, e4], dim=1)
        d4 = self.decoder4(up4)
        
        up3 = self.upconv3(d4)
        up3 = torch.cat([up3, e3], dim=1)
        d3 = self.decoder3(up3)
        
        up2 = self.upconv2(d3)
        up2 = torch.cat([up2, e2], dim=1)
        d2 = self.decoder2(up2)
        
        up1 = self.upconv1(d2)
        up1 = torch.cat([up1, e1], dim=1)
        d1 = self.decoder1(up1)
        
        # Detection head
        detection_outputs = self.detection_head(d1)
        
        # Convert to bounding boxes and scores
        boxes, scores = self._decode_predictions(detection_outputs)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'features': d1,
            'center_heatmap': detection_outputs['center_heatmap'],
            'size_pred': detection_outputs['size_pred'],
            'offset_pred': detection_outputs['offset_pred']
        }
    
    def _decode_predictions(self, predictions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode center-point predictions to bounding boxes
        
        Args:
            predictions: Dictionary with center_heatmap, size_pred, offset_pred
            
        Returns:
            boxes: Tensor of shape (N, 6) with format [x, y, z, w, h, d]
            scores: Tensor of shape (N,) with confidence scores
        """
        center_heatmap = predictions['center_heatmap']
        size_pred = predictions['size_pred']
        offset_pred = predictions['offset_pred']
        
        batch_size = center_heatmap.shape[0]
        device = center_heatmap.device
        
        all_boxes = []
        all_scores = []
        
        for b in range(batch_size):
            # Get center points above threshold
            heatmap = center_heatmap[b, 0]  # Shape: (D, H, W)
            
            # Find local maxima
            local_maxima = self._find_local_maxima(heatmap, threshold=0.1)
            
            if len(local_maxima) == 0:
                # No detections
                boxes = torch.zeros((0, 6), device=device)
                scores = torch.zeros((0,), device=device)
            else:
                # Extract predictions at detected points
                z_coords, y_coords, x_coords = local_maxima[:, 0], local_maxima[:, 1], local_maxima[:, 2]
                
                # Get confidence scores
                scores = heatmap[z_coords, y_coords, x_coords]
                
                # Get size predictions
                sizes = size_pred[b, :, z_coords, y_coords, x_coords].T  # Shape: (N, 3)
                
                # Get offset predictions
                offsets = offset_pred[b, :, z_coords, y_coords, x_coords].T  # Shape: (N, 3)
                
                # Compute final center coordinates with offsets
                centers = local_maxima.float() + offsets
                
                # Combine to form bounding boxes [x, y, z, w, h, d]
                boxes = torch.cat([
                    centers[:, [2, 1, 0]],  # x, y, z (swap to match expected order)
                    sizes
                ], dim=1)
            
            all_boxes.append(boxes)
            all_scores.append(scores)
        
        # Concatenate all batches
        if len(all_boxes) > 0:
            final_boxes = torch.cat(all_boxes, dim=0)
            final_scores = torch.cat(all_scores, dim=0)
        else:
            final_boxes = torch.zeros((0, 6), device=device)
            final_scores = torch.zeros((0,), device=device)
        
        return final_boxes, final_scores
    
    def _find_local_maxima(self, heatmap: torch.Tensor, threshold: float = 0.1, 
                          kernel_size: int = 3) -> torch.Tensor:
        """
        Find local maxima in the heatmap
        
        Args:
            heatmap: 3D heatmap tensor
            threshold: Minimum confidence threshold
            kernel_size: Size of local maximum kernel
            
        Returns:
            Tensor of local maxima coordinates (N, 3)
        """
        # Apply threshold
        mask = heatmap > threshold
        
        # Find local maxima using max pooling
        pad = kernel_size // 2
        local_max = F.max_pool3d(
            heatmap.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=pad
        ).squeeze()
        
        # Points that are both above threshold and local maxima
        peaks = (heatmap == local_max) & mask
        
        # Get coordinates of peaks
        coords = torch.nonzero(peaks, as_tuple=False)
        
        return coords
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information"""
        return {
            'name': 'CPM-Net',
            'type': 'Center-Points Matching Network',
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'confidence_threshold': self.confidence_threshold,
            'base_channels': self.base_channels,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def create_cpm_net(input_channels: int = 1, 
                   num_classes: int = 1,
                   confidence_threshold: float = 0.8,
                   base_channels: int = 32) -> CPMNet:
    """
    Factory function to create CPM-Net model
    
    Args:
        input_channels: Number of input channels
        num_classes: Number of detection classes
        confidence_threshold: Confidence threshold for filtering
        base_channels: Base number of channels
        
    Returns:
        CPMNet model instance
    """
    return CPMNet(
        input_channels=input_channels,
        num_classes=num_classes,
        confidence_threshold=confidence_threshold,
        base_channels=base_channels
    )


# Training configuration for CPM-Net as specified in the paper
CPM_NET_CONFIG = {
    'epochs': 45,
    'optimizer': 'AdamW',
    'learning_rate_start': 0.0001,
    'learning_rate_end': 0.00001,
    'lr_schedule': 'linear_decay',
    'batch_size': 4,  # Typical for 3D medical imaging
    'input_size': (64, 64, 64),  # Subvolume size
    'overlap': 0.5,  # Overlap for subvolume processing
    'confidence_threshold': 0.8
}


if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_cpm_net()
    print(f"Model info: {model.get_model_info()}")
    
    # Test forward pass
    x = torch.randn(1, 1, 64, 64, 64)
    with torch.no_grad():
        outputs = model(x)
        print(f"Output shapes:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")