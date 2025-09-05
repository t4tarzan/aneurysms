"""
Data Augmentation and Normalization Transforms for CTA Aneurysm Detection

This module provides comprehensive data transformation utilities for 3D medical imaging,
including augmentation techniques and normalization methods specifically designed for
intracranial aneurysm detection in CTA volumes.

Key Features:
- 3D spatial transformations (rotation, scaling, translation)
- Intensity-based augmentations (noise, contrast, brightness)
- Medical imaging specific normalizations
- Bounding box coordinate transformations
- Elastic deformations for realistic anatomical variations
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any, Union
from dataclasses import dataclass
import random
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransformConfig:
    """Configuration for data transformations"""
    # Spatial augmentation parameters
    rotation_range: Tuple[float, float, float] = (15.0, 15.0, 15.0)  # degrees
    scaling_range: Tuple[float, float] = (0.9, 1.1)
    translation_range: Tuple[float, float, float] = (10.0, 10.0, 5.0)  # pixels
    
    # Intensity augmentation parameters
    noise_std: float = 0.05
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    brightness_range: Tuple[float, float] = (-0.1, 0.1)
    gamma_range: Tuple[float, float] = (0.8, 1.2)
    
    # Elastic deformation parameters
    elastic_alpha: float = 100.0
    elastic_sigma: float = 10.0
    
    # Normalization parameters
    normalize_intensity: bool = True
    clip_range: Tuple[float, float] = (-1000.0, 3000.0)  # HU units for CTA
    target_mean: float = 0.0
    target_std: float = 1.0
    
    # Augmentation probabilities
    prob_rotation: float = 0.5
    prob_scaling: float = 0.5
    prob_translation: float = 0.5
    prob_noise: float = 0.3
    prob_contrast: float = 0.3
    prob_brightness: float = 0.3
    prob_gamma: float = 0.2
    prob_elastic: float = 0.2
    prob_flip: float = 0.5


class IntensityNormalizer:
    """Handles intensity normalization for CTA volumes"""
    
    def __init__(self, config: TransformConfig):
        self.config = config
        
    def normalize_cta_intensity(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize CTA volume intensity values
        
        Args:
            volume: Input CTA volume
            
        Returns:
            Normalized volume
        """
        # Clip to reasonable HU range for CTA
        volume = np.clip(volume, self.config.clip_range[0], self.config.clip_range[1])
        
        if self.config.normalize_intensity:
            # Z-score normalization
            mean = np.mean(volume)
            std = np.std(volume)
            if std > 0:
                volume = (volume - mean) / std
                # Scale to target mean and std
                volume = volume * self.config.target_std + self.config.target_mean
        
        return volume.astype(np.float32)
    
    def denormalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """Reverse intensity normalization"""
        if self.config.normalize_intensity:
            # Reverse z-score normalization
            volume = (volume - self.config.target_mean) / self.config.target_std
            # Note: Cannot fully reverse without original statistics
        
        return volume


class SpatialTransforms:
    """Handles 3D spatial transformations"""
    
    def __init__(self, config: TransformConfig):
        self.config = config
        
    def random_rotation_matrix(self) -> np.ndarray:
        """Generate random 3D rotation matrix"""
        angles = [
            np.random.uniform(-self.config.rotation_range[i], self.config.rotation_range[i])
            for i in range(3)
        ]
        rotation = Rotation.from_euler('xyz', angles, degrees=True)
        return rotation.as_matrix()
    
    def random_scaling_matrix(self) -> np.ndarray:
        """Generate random scaling matrix"""
        scale = np.random.uniform(self.config.scaling_range[0], self.config.scaling_range[1])
        return np.eye(3) * scale
    
    def random_translation_vector(self) -> np.ndarray:
        """Generate random translation vector"""
        return np.array([
            np.random.uniform(-self.config.translation_range[i], self.config.translation_range[i])
            for i in range(3)
        ])
    
    def apply_spatial_transform(self, volume: np.ndarray, 
                              bboxes: Optional[List] = None,
                              transform_matrix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[List]]:
        """
        Apply spatial transformation to volume and bounding boxes
        
        Args:
            volume: Input volume
            bboxes: List of bounding boxes [x, y, z, w, h, d]
            transform_matrix: Optional pre-computed transformation matrix
            
        Returns:
            Transformed volume and bounding boxes
        """
        if transform_matrix is None:
            # Generate random transformation
            rotation_matrix = self.random_rotation_matrix()
            scaling_matrix = self.random_scaling_matrix()
            translation = self.random_translation_vector()
            
            # Combine transformations
            transform_matrix = scaling_matrix @ rotation_matrix
        else:
            translation = np.zeros(3)
        
        # Apply transformation to volume
        transformed_volume = self._transform_volume(volume, transform_matrix, translation)
        
        # Transform bounding boxes if provided
        transformed_bboxes = None
        if bboxes is not None:
            transformed_bboxes = self._transform_bboxes(bboxes, transform_matrix, translation, volume.shape)
        
        return transformed_volume, transformed_bboxes
    
    def _transform_volume(self, volume: np.ndarray, transform_matrix: np.ndarray, 
                         translation: np.ndarray) -> np.ndarray:
        """Apply transformation to volume using scipy"""
        # Create coordinate grids
        coords = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]
        coords = coords.reshape(3, -1)
        
        # Center coordinates
        center = np.array(volume.shape) / 2
        coords_centered = coords - center.reshape(3, 1)
        
        # Apply transformation
        coords_transformed = transform_matrix @ coords_centered + translation.reshape(3, 1)
        coords_transformed += center.reshape(3, 1)
        
        # Reshape back to grid format
        coords_transformed = coords_transformed.reshape(3, *volume.shape)
        
        # Interpolate
        transformed_volume = map_coordinates(volume, coords_transformed, order=1, 
                                           mode='constant', cval=0.0)
        
        return transformed_volume
    
    def _transform_bboxes(self, bboxes: List, transform_matrix: np.ndarray, 
                         translation: np.ndarray, volume_shape: Tuple) -> List:
        """Transform bounding box coordinates"""
        transformed_bboxes = []
        center = np.array(volume_shape) / 2
        
        for bbox in bboxes:
            if len(bbox) >= 6:  # [x, y, z, w, h, d, ...]
                # Extract center and dimensions
                center_coords = np.array(bbox[:3])
                dimensions = np.array(bbox[3:6])
                
                # Transform center coordinates
                center_centered = center_coords - center
                center_transformed = transform_matrix @ center_centered + translation + center
                
                # Transform dimensions (approximate scaling)
                scale_factor = np.cbrt(np.linalg.det(transform_matrix))
                dimensions_transformed = dimensions * scale_factor
                
                # Create transformed bbox
                transformed_bbox = list(center_transformed) + list(dimensions_transformed)
                if len(bbox) > 6:  # Preserve additional attributes
                    transformed_bbox.extend(bbox[6:])
                
                transformed_bboxes.append(transformed_bbox)
        
        return transformed_bboxes


class IntensityAugmentation:
    """Handles intensity-based augmentations"""
    
    def __init__(self, config: TransformConfig):
        self.config = config
        
    def add_gaussian_noise(self, volume: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to volume"""
        noise = np.random.normal(0, self.config.noise_std, volume.shape)
        return volume + noise
    
    def adjust_contrast(self, volume: np.ndarray) -> np.ndarray:
        """Adjust contrast of volume"""
        factor = np.random.uniform(self.config.contrast_range[0], self.config.contrast_range[1])
        mean = np.mean(volume)
        return (volume - mean) * factor + mean
    
    def adjust_brightness(self, volume: np.ndarray) -> np.ndarray:
        """Adjust brightness of volume"""
        factor = np.random.uniform(self.config.brightness_range[0], self.config.brightness_range[1])
        return volume + factor
    
    def gamma_correction(self, volume: np.ndarray) -> np.ndarray:
        """Apply gamma correction"""
        gamma = np.random.uniform(self.config.gamma_range[0], self.config.gamma_range[1])
        
        # Normalize to [0, 1] for gamma correction
        volume_min, volume_max = volume.min(), volume.max()
        if volume_max > volume_min:
            volume_norm = (volume - volume_min) / (volume_max - volume_min)
            volume_gamma = np.power(volume_norm, gamma)
            return volume_gamma * (volume_max - volume_min) + volume_min
        
        return volume


class ElasticDeformation:
    """Handles elastic deformation for realistic anatomical variations"""
    
    def __init__(self, config: TransformConfig):
        self.config = config
        
    def elastic_transform(self, volume: np.ndarray, 
                         bboxes: Optional[List] = None) -> Tuple[np.ndarray, Optional[List]]:
        """
        Apply elastic deformation to volume
        
        Args:
            volume: Input volume
            bboxes: Optional bounding boxes to transform
            
        Returns:
            Deformed volume and bounding boxes
        """
        shape = volume.shape
        
        # Generate random displacement fields
        dx = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), 
            self.config.elastic_sigma, 
            mode="constant", 
            cval=0
        ) * self.config.elastic_alpha
        
        dy = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), 
            self.config.elastic_sigma, 
            mode="constant", 
            cval=0
        ) * self.config.elastic_alpha
        
        dz = gaussian_filter(
            (np.random.rand(*shape) * 2 - 1), 
            self.config.elastic_sigma, 
            mode="constant", 
            cval=0
        ) * self.config.elastic_alpha
        
        # Create coordinate grids
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]), indexing='ij')
        
        # Apply displacement
        indices = [y + dy, x + dx, z + dz]
        
        # Transform volume
        deformed_volume = map_coordinates(volume, indices, order=1, mode='reflect')
        
        # Transform bounding boxes (simplified - just apply average displacement)
        transformed_bboxes = None
        if bboxes is not None:
            transformed_bboxes = []
            for bbox in bboxes:
                if len(bbox) >= 6:
                    x_center, y_center, z_center = bbox[:3]
                    
                    # Sample displacement at bbox center
                    x_idx = int(np.clip(x_center, 0, shape[1] - 1))
                    y_idx = int(np.clip(y_center, 0, shape[0] - 1))
                    z_idx = int(np.clip(z_center, 0, shape[2] - 1))
                    
                    dx_center = dx[y_idx, x_idx, z_idx]
                    dy_center = dy[y_idx, x_idx, z_idx]
                    dz_center = dz[y_idx, x_idx, z_idx]
                    
                    # Apply displacement to center
                    new_bbox = [
                        x_center + dx_center,
                        y_center + dy_center,
                        z_center + dz_center
                    ] + list(bbox[3:])  # Keep dimensions and other attributes
                    
                    transformed_bboxes.append(new_bbox)
        
        return deformed_volume, transformed_bboxes


class CompositeTransform:
    """Main transform class that combines all augmentations"""
    
    def __init__(self, config: TransformConfig, mode: str = 'train'):
        self.config = config
        self.mode = mode
        
        # Initialize transform components
        self.normalizer = IntensityNormalizer(config)
        self.spatial_transforms = SpatialTransforms(config)
        self.intensity_augmentation = IntensityAugmentation(config)
        self.elastic_deformation = ElasticDeformation(config)
        
    def __call__(self, volume: np.ndarray, 
                 bboxes: Optional[List] = None,
                 apply_augmentation: bool = None) -> Tuple[np.ndarray, Optional[List]]:
        """
        Apply complete transformation pipeline
        
        Args:
            volume: Input CTA volume
            bboxes: Optional bounding boxes
            apply_augmentation: Override augmentation based on mode
            
        Returns:
            Transformed volume and bounding boxes
        """
        if apply_augmentation is None:
            apply_augmentation = (self.mode == 'train')
        
        # Always normalize intensity
        volume = self.normalizer.normalize_cta_intensity(volume)
        
        if not apply_augmentation:
            return volume, bboxes
        
        # Apply augmentations with probability
        transformed_bboxes = bboxes
        
        # Spatial transformations
        if random.random() < self.config.prob_rotation or \
           random.random() < self.config.prob_scaling or \
           random.random() < self.config.prob_translation:
            volume, transformed_bboxes = self.spatial_transforms.apply_spatial_transform(
                volume, transformed_bboxes
            )
        
        # Elastic deformation
        if random.random() < self.config.prob_elastic:
            volume, transformed_bboxes = self.elastic_deformation.elastic_transform(
                volume, transformed_bboxes
            )
        
        # Intensity augmentations
        if random.random() < self.config.prob_noise:
            volume = self.intensity_augmentation.add_gaussian_noise(volume)
        
        if random.random() < self.config.prob_contrast:
            volume = self.intensity_augmentation.adjust_contrast(volume)
        
        if random.random() < self.config.prob_brightness:
            volume = self.intensity_augmentation.adjust_brightness(volume)
        
        if random.random() < self.config.prob_gamma:
            volume = self.intensity_augmentation.gamma_correction(volume)
        
        # Random flipping
        if random.random() < self.config.prob_flip:
            volume, transformed_bboxes = self._random_flip(volume, transformed_bboxes)
        
        return volume, transformed_bboxes
    
    def _random_flip(self, volume: np.ndarray, 
                    bboxes: Optional[List] = None) -> Tuple[np.ndarray, Optional[List]]:
        """Apply random flipping"""
        # Choose random axis to flip (avoid z-axis for medical data)
        axis = random.choice([0, 1])
        
        volume = np.flip(volume, axis=axis)
        
        if bboxes is not None:
            transformed_bboxes = []
            for bbox in bboxes:
                if len(bbox) >= 6:
                    new_bbox = list(bbox)
                    # Flip coordinate
                    if axis == 0:
                        new_bbox[1] = volume.shape[0] - bbox[1]  # y-coordinate
                    elif axis == 1:
                        new_bbox[0] = volume.shape[1] - bbox[0]  # x-coordinate
                    
                    transformed_bboxes.append(new_bbox)
            return volume, transformed_bboxes
        
        return volume, bboxes


class TestTimeAugmentation:
    """Test-time augmentation for improved inference"""
    
    def __init__(self, config: TransformConfig, n_augmentations: int = 4):
        self.config = config
        self.n_augmentations = n_augmentations
        self.normalizer = IntensityNormalizer(config)
        
    def augment_volume(self, volume: np.ndarray) -> List[np.ndarray]:
        """
        Generate multiple augmented versions for test-time augmentation
        
        Args:
            volume: Input volume
            
        Returns:
            List of augmented volumes
        """
        augmented_volumes = []
        
        # Original normalized volume
        normalized_volume = self.normalizer.normalize_cta_intensity(volume)
        augmented_volumes.append(normalized_volume)
        
        # Generate additional augmentations
        for _ in range(self.n_augmentations - 1):
            # Apply light augmentations
            aug_volume = normalized_volume.copy()
            
            # Random flipping
            if random.random() < 0.5:
                axis = random.choice([0, 1])
                aug_volume = np.flip(aug_volume, axis=axis)
            
            # Light rotation
            if random.random() < 0.5:
                angle = random.uniform(-5, 5)  # Small rotation
                # Apply rotation (simplified)
                aug_volume = self._light_rotation(aug_volume, angle)
            
            augmented_volumes.append(aug_volume)
        
        return augmented_volumes
    
    def _light_rotation(self, volume: np.ndarray, angle: float) -> np.ndarray:
        """Apply light rotation for TTA"""
        # Simplified rotation around z-axis
        from scipy.ndimage import rotate
        return rotate(volume, angle, axes=(0, 1), reshape=False, order=1)


def create_train_transform(config: Optional[TransformConfig] = None) -> CompositeTransform:
    """Create training transform with augmentation"""
    if config is None:
        config = TransformConfig()
    return CompositeTransform(config, mode='train')


def create_val_transform(config: Optional[TransformConfig] = None) -> CompositeTransform:
    """Create validation transform without augmentation"""
    if config is None:
        config = TransformConfig()
    return CompositeTransform(config, mode='val')


def create_test_transform(config: Optional[TransformConfig] = None) -> CompositeTransform:
    """Create test transform without augmentation"""
    if config is None:
        config = TransformConfig()
    return CompositeTransform(config, mode='test')


def create_tta_transform(config: Optional[TransformConfig] = None, 
                        n_augmentations: int = 4) -> TestTimeAugmentation:
    """Create test-time augmentation transform"""
    if config is None:
        config = TransformConfig()
    return TestTimeAugmentation(config, n_augmentations)


# Example usage and testing functions
def test_transforms():
    """Test transform functionality"""
    print("Testing data transforms...")
    
    # Create sample volume and bounding boxes
    volume = np.random.randn(128, 128, 64).astype(np.float32)
    bboxes = [
        [64, 64, 32, 10, 10, 8, 0.9, 'aneurysm'],  # [x, y, z, w, h, d, conf, label]
        [80, 80, 40, 8, 8, 6, 0.8, 'aneurysm']
    ]
    
    # Test configuration
    config = TransformConfig(
        prob_rotation=1.0,  # Always apply for testing
        prob_noise=1.0,
        prob_contrast=1.0
    )
    
    # Test training transform
    train_transform = create_train_transform(config)
    transformed_volume, transformed_bboxes = train_transform(volume, bboxes)
    
    print(f"Original volume shape: {volume.shape}")
    print(f"Transformed volume shape: {transformed_volume.shape}")
    print(f"Original volume range: [{volume.min():.3f}, {volume.max():.3f}]")
    print(f"Transformed volume range: [{transformed_volume.min():.3f}, {transformed_volume.max():.3f}]")
    
    if transformed_bboxes:
        print(f"Original bboxes: {len(bboxes)}")
        print(f"Transformed bboxes: {len(transformed_bboxes)}")
        print(f"First bbox transformation: {bboxes[0][:6]} -> {transformed_bboxes[0][:6]}")
    
    # Test validation transform (no augmentation)
    val_transform = create_val_transform(config)
    val_volume, val_bboxes = val_transform(volume, bboxes)
    
    print(f"Validation volume range: [{val_volume.min():.3f}, {val_volume.max():.3f}]")
    
    # Test TTA
    tta_transform = create_tta_transform(config, n_augmentations=3)
    tta_volumes = tta_transform.augment_volume(volume)
    
    print(f"TTA generated {len(tta_volumes)} augmented volumes")
    
    print("Transform testing completed successfully!")


if __name__ == "__main__":
    test_transforms()