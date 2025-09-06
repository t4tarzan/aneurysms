"""Base class for 3D object detection models."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


class BaseDetector(nn.Module, ABC):
    """Abstract base class for 3D object detection models.
    
    This class defines the interface that all detection models must implement.
    """
    
    def __init__(self, **kwargs):
        """Initialize the base detector.
        
        Args:
            **kwargs: Additional model-specific arguments.
        """
        super().__init__()
        self.model_name = self.__class__.__name__
    
    @abstractmethod
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, depth, height, width)
            
        Returns:
            Dictionary containing model outputs, typically including:
                - 'cls_scores': Classification scores
                - 'bbox_preds': Bounding box predictions
                - 'centerness': Centerness predictions (if applicable)
        """
        pass
    
    @abstractmethod
    def loss(
        self,
        preds: Dict[str, Tensor],
        targets: List[Dict[str, Tensor]],
        **kwargs
    ) -> Dict[str, Tensor]:
        """Compute the loss for training.
        
        Args:
            preds: Model predictions from forward pass
            targets: List of target dictionaries, one per image in the batch
            **kwargs: Additional arguments for loss computation
            
        Returns:
            Dictionary of loss values
        """
        pass
    
    def predict(
        self,
        x: Tensor,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        **kwargs
    ) -> List[Dict[str, Tensor]]:
        """Generate predictions for inference.
        
        Args:
            x: Input tensor of shape (batch_size, channels, depth, height, width)
            score_threshold: Minimum score to consider a detection
            nms_threshold: NMS IoU threshold
            **kwargs: Additional arguments for prediction
            
        Returns:
            List of prediction dictionaries, one per image in the batch
        """
        self.eval()
        with torch.no_grad():
            preds = self.forward(x)
        
        # Apply thresholding and NMS
        return self.post_process(preds, score_threshold, nms_threshold, **kwargs)
    
    @abstractmethod
    def post_process(
        self,
        preds: Dict[str, Tensor],
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        **kwargs
    ) -> List[Dict[str, Tensor]]:
        """Post-process model outputs to get final detections.
        
        Args:
            preds: Raw model predictions
            score_threshold: Minimum score to consider a detection
            nms_threshold: NMS IoU threshold
            **kwargs: Additional arguments for post-processing
            
        Returns:
            List of prediction dictionaries, each containing:
                - 'boxes': Detected bounding boxes in (x1, y1, z1, x2, y2, z2) format
                - 'scores': Confidence scores for each box
                - 'labels': Class labels for each box
        """
        pass
    
    def get_optimizer(
        self,
        optimizer_type: str = "adamw",
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        **kwargs
    ) -> torch.optim.Optimizer:
        """Get optimizer for training.
        
        Args:
            optimizer_type: Type of optimizer ('adamw', 'adam', or 'sgd')
            lr: Learning rate
            weight_decay: Weight decay factor
            **kwargs: Additional optimizer arguments
            
        Returns:
            Configured optimizer
        """
        if optimizer_type.lower() == 'adamw':
            return torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_type.lower() == 'adam':
            return torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_type.lower() == 'sgd':
            return torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def load_weights(self, checkpoint_path: str) -> None:
        """Load model weights from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DataParallel or DDP wrapping
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        self.load_state_dict(state_dict, strict=True)
