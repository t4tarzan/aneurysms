"""
Model service for loading and running the 3D-CNN-TR aneurysm detection model.
"""
import os
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import SimpleITK as sitk
import torch.nn.functional as F

# Import model architecture
import sys
sys.path.append(str(Path(__file__).parents[2] / "models"))
from cnn_tr_3d import CNNTR3D, create_cnn_tr_3d

logger = logging.getLogger(__name__)

class AneurysmDetectionModel:
    """Service for loading and running the 3D-CNN-TR aneurysm detection model."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the model service.
        
        Args:
            model_path: Path to the trained model weights
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.model = None
        self.artery_segmentation_model = None
        self.confidence_threshold = 0.8
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load the 3D-CNN-TR model from the specified path.
        
        Args:
            model_path: Path to the trained model weights
        """
        try:
            # Initialize model with the same architecture as training
            self.model = create_cnn_tr_3d(
                input_channels=1,
                aux_channels=1,
                num_classes=1,
                confidence_threshold=self.confidence_threshold
            )
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded 3D-CNN-TR model from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise
    
    def preprocess_volume(self, volume: np.ndarray, spacing: Tuple[float, float, float]) -> torch.Tensor:
        """
        Preprocess the input volume for the model.
        
        Args:
            volume: 3D numpy array of shape (H, W, D)
            spacing: Tuple of (x, y, z) voxel spacing in mm
            
        Returns:
            Preprocessed torch.Tensor ready for model input
        """
        # Convert to float32 and normalize to [0, 1] if not already
        volume = volume.astype(np.float32)
        if volume.max() > 1.0:
            volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        # Add channel dimension: (H, W, D) -> (1, H, W, D)
        volume = np.expand_dims(volume, axis=0)
        
        # Resample to target spacing if needed
        target_spacing = (0.4, 0.4, 0.4)  # Match training spacing
        if not np.allclose(spacing, target_spacing):
            volume = self._resample_volume(volume, spacing, target_spacing)
        
        # Convert to tensor and add batch dimension: (1, H, W, D) -> (1, 1, H, W, D)
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).to(self.device)
        
        return volume_tensor
    
    def _resample_volume(self, volume: np.ndarray, 
                        input_spacing: Tuple[float, float, float],
                        target_spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Resample the volume to the target voxel spacing using SimpleITK.
        
        Args:
            volume: Input volume as numpy array (1, H, W, D)
            input_spacing: Input voxel spacing (x, y, z) in mm
            target_spacing: Target voxel spacing (x, y, z) in mm
            
        Returns:
            Resampled volume as numpy array
        """
        # Convert numpy array to SimpleITK image
        sitk_image = sitk.GetImageFromArray(volume[0])
        sitk_image.SetSpacing(input_spacing[::-1])  # SimpleITK uses z, y, x order
        
        # Calculate new size
        input_size = np.array(sitk_image.GetSize())
        output_size = (input_size * np.array(input_spacing) / np.array(target_spacing)).astype(int)
        
        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize([int(s) for s in output_size])
        resampler.SetOutputSpacing(target_spacing[::-1])
        resampler.SetOutputOrigin(sitk_image.GetOrigin())
        resampler.SetOutputDirection(sitk_image.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        
        resampled_image = resampler.Execute(sitk_image)
        
        # Convert back to numpy and add channel dimension
        return np.expand_dims(sitk.GetArrayFromImage(resampled_image), axis=0)
    
    def detect_aneurysms(self, volume: torch.Tensor, 
                        artery_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Detect aneurysms in the input volume.
        
        Args:
            volume: Input volume tensor of shape (1, 1, H, W, D)
            artery_mask: Optional artery segmentation mask tensor of same shape as volume
            
        Returns:
            Dictionary containing detection results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        with torch.no_grad():
            # If no artery mask is provided, use zeros
            if artery_mask is None:
                artery_mask = torch.zeros_like(volume)
            
            # Run model inference
            outputs = self.model(volume, artery_mask)
            
            # Process outputs
            detections = []
            
            # Extract bounding boxes, scores, and features
            boxes = outputs.get('boxes', torch.empty((0, 6)))
            scores = outputs.get('scores', torch.empty((0,)))
            
            # Filter detections by confidence threshold
            keep = scores >= self.confidence_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            
            # Convert to numpy and format detections
            for box, score in zip(boxes.cpu().numpy(), scores.cpu().numpy()):
                detections.append({
                    'bbox': box.tolist(),  # [x, y, z, w, h, d]
                    'score': float(score),
                    'class_name': 'aneurysm',
                })
            
            return {
                'detections': detections,
                'num_detections': len(detections),
                'volume_shape': volume.shape[2:],  # (D, H, W)
            }
    
    def postprocess_detections(self, detections: Dict[str, Any], 
                             original_shape: Tuple[int, int, int],
                             original_spacing: Tuple[float, float, float]) -> Dict[str, Any]:
        """
        Post-process detections to original image space.
        
        Args:
            detections: Raw detections from detect_aneurysms()
            original_shape: Shape of the original volume (H, W, D)
            original_spacing: Voxel spacing of the original volume (x, y, z) in mm
            
        Returns:
            Post-processed detections in original image space
        """
        # TODO: Implement post-processing steps from the paper
        # This would include applying the anatomical filters and other post-processing
        
        # For now, just return the detections as-is
        return detections
