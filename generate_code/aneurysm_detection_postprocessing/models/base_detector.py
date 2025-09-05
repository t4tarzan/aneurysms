"""
Base detector class for aneurysm detection models.
Provides common functionality for CPM-Net and 3D-CNN-TR models.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional


class BaseDetector(nn.Module, ABC):
    """
    Abstract base class for 3D aneurysm detection models.
    
    This class provides common functionality for:
    - Model initialization
    - Forward pass structure
    - Bounding box processing
    - Confidence score handling
    """
    
    def __init__(self, 
                 input_channels: int = 1,
                 num_classes: int = 1,
                 confidence_threshold: float = 0.8):
        """
        Initialize base detector.
        
        Args:
            input_channels: Number of input channels (typically 1 for CTA)
            num_classes: Number of detection classes (1 for aneurysm)
            confidence_threshold: Minimum confidence for detections
        """
        super(BaseDetector, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the detection model.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Dictionary containing:
                - 'boxes': Predicted bounding boxes (B, N, 6) - [x, y, z, w, h, d]
                - 'scores': Confidence scores (B, N)
                - 'features': Optional feature maps
        """
        pass
    
    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on input volume.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            boxes: Numpy array of bounding boxes (N, 6)
            scores: Numpy array of confidence scores (N,)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            boxes = outputs['boxes'].cpu().numpy()
            scores = outputs['scores'].cpu().numpy()
            
            # Filter by confidence threshold
            valid_indices = scores >= self.confidence_threshold
            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
            
        return boxes, scores
    
    def process_subvolumes(self, 
                          volume: np.ndarray,
                          subvolume_size: Tuple[int, int, int] = (64, 64, 64),
                          overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process full volume by cropping into overlapping subvolumes.
        
        Args:
            volume: Full CTA volume (D, H, W)
            subvolume_size: Size of each subvolume
            overlap: Overlap ratio between adjacent subvolumes
            
        Returns:
            all_boxes: Combined bounding boxes from all subvolumes
            all_scores: Combined confidence scores
        """
        d, h, w = volume.shape
        sub_d, sub_h, sub_w = subvolume_size
        
        # Calculate step sizes with overlap
        step_d = int(sub_d * (1 - overlap))
        step_h = int(sub_h * (1 - overlap))
        step_w = int(sub_w * (1 - overlap))
        
        all_boxes = []
        all_scores = []
        
        # Iterate through subvolumes
        for z in range(0, d - sub_d + 1, step_d):
            for y in range(0, h - sub_h + 1, step_h):
                for x in range(0, w - sub_w + 1, step_w):
                    # Extract subvolume
                    subvol = volume[z:z+sub_d, y:y+sub_h, x:x+sub_w]
                    
                    # Convert to tensor and add batch dimension
                    subvol_tensor = torch.FloatTensor(subvol).unsqueeze(0).unsqueeze(0)
                    
                    # Make prediction
                    boxes, scores = self.predict(subvol_tensor)
                    
                    # Adjust box coordinates to global coordinate system
                    if len(boxes) > 0:
                        boxes[:, 0] += x  # x coordinate
                        boxes[:, 1] += y  # y coordinate
                        boxes[:, 2] += z  # z coordinate
                        
                        all_boxes.append(boxes)
                        all_scores.append(scores)
        
        # Combine all detections
        if all_boxes:
            all_boxes = np.concatenate(all_boxes, axis=0)
            all_scores = np.concatenate(all_scores, axis=0)
        else:
            all_boxes = np.empty((0, 6))
            all_scores = np.empty((0,))
            
        return all_boxes, all_scores
    
    def non_maximum_suppression(self, 
                               boxes: np.ndarray, 
                               scores: np.ndarray,
                               iou_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections.
        
        Args:
            boxes: Bounding boxes (N, 6) - [x, y, z, w, h, d]
            scores: Confidence scores (N,)
            iou_threshold: IoU threshold for suppression
            
        Returns:
            filtered_boxes: Boxes after NMS
            filtered_scores: Scores after NMS
        """
        if len(boxes) == 0:
            return boxes, scores
            
        # Sort by confidence scores
        sorted_indices = np.argsort(scores)[::-1]
        
        keep_indices = []
        
        while len(sorted_indices) > 0:
            # Keep the box with highest confidence
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
                
            # Calculate IoU with remaining boxes
            current_box = boxes[current_idx]
            remaining_boxes = boxes[sorted_indices[1:]]
            
            ious = self._calculate_3d_iou(current_box, remaining_boxes)
            
            # Remove boxes with high IoU
            keep_mask = ious < iou_threshold
            sorted_indices = sorted_indices[1:][keep_mask]
        
        return boxes[keep_indices], scores[keep_indices]
    
    def _calculate_3d_iou(self, box1: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Calculate 3D IoU between one box and multiple boxes.
        
        Args:
            box1: Single box [x, y, z, w, h, d]
            boxes: Multiple boxes (N, 6)
            
        Returns:
            ious: IoU values (N,)
        """
        # Convert to corner coordinates
        box1_min = box1[:3] - box1[3:] / 2
        box1_max = box1[:3] + box1[3:] / 2
        
        boxes_min = boxes[:, :3] - boxes[:, 3:] / 2
        boxes_max = boxes[:, :3] + boxes[:, 3:] / 2
        
        # Calculate intersection
        inter_min = np.maximum(box1_min, boxes_min)
        inter_max = np.minimum(box1_max, boxes_max)
        
        inter_dims = np.maximum(0, inter_max - inter_min)
        inter_volume = np.prod(inter_dims, axis=1)
        
        # Calculate union
        box1_volume = np.prod(box1[3:])
        boxes_volume = np.prod(boxes[:, 3:], axis=1)
        union_volume = box1_volume + boxes_volume - inter_volume
        
        # Calculate IoU
        ious = inter_volume / (union_volume + 1e-8)
        
        return ious
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get model information and parameters.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.__class__.__name__,
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'confidence_threshold': self.confidence_threshold,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }