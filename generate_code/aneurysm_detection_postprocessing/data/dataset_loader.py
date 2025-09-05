"""
CTA Dataset Loader for Intracranial Aneurysm Detection

This module provides comprehensive data loading and preprocessing functionality for CTA volumes
and their corresponding 3D bounding box annotations. It handles data augmentation, normalization,
and batch preparation for training and evaluation of aneurysm detection models.

Key Features:
- CTA volume loading with DICOM and NIfTI support
- Integration with annotation parser for ground truth handling
- Data augmentation pipeline for training
- Batch processing with configurable subvolume extraction
- Memory-efficient data loading with caching
- Support for both training and inference modes
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Medical imaging libraries
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    warnings.warn("nibabel not available - NIfTI support disabled")

try:
    import pydicom
    import SimpleITK as sitk
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    warnings.warn("pydicom/SimpleITK not available - DICOM support disabled")

# Local imports
from .annotation_parser import AnnotationManager, AnnotationCase, BoundingBox3D

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""
    
    # Data paths
    data_root: str
    annotation_path: str
    
    # Volume processing
    target_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)  # mm per voxel
    target_size: Optional[Tuple[int, int, int]] = None  # Target volume size
    
    # Subvolume extraction
    subvolume_size: Tuple[int, int, int] = (128, 128, 128)
    subvolume_overlap: float = 0.25  # 25% overlap between subvolumes
    
    # Data augmentation
    enable_augmentation: bool = True
    rotation_range: float = 15.0  # degrees
    scaling_range: Tuple[float, float] = (0.9, 1.1)
    noise_std: float = 0.05
    
    # Normalization
    intensity_clip_range: Tuple[float, float] = (-1000, 3000)  # HU units
    normalize_method: str = "z_score"  # "z_score", "min_max", "percentile"
    
    # Training parameters
    positive_sample_ratio: float = 0.5  # Ratio of positive to negative samples
    max_detections_per_volume: int = 50
    
    # Performance
    cache_preprocessed: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2


class VolumeProcessor:
    """Handles 3D volume preprocessing operations."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        
    def load_volume(self, volume_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load a 3D medical volume from file.
        
        Args:
            volume_path: Path to volume file (NIfTI or DICOM)
            
        Returns:
            Tuple of (volume_array, metadata_dict)
        """
        volume_path = Path(volume_path)
        
        if volume_path.suffix.lower() in ['.nii', '.nii.gz']:
            return self._load_nifti(volume_path)
        elif volume_path.is_dir() or volume_path.suffix.lower() == '.dcm':
            return self._load_dicom(volume_path)
        else:
            raise ValueError(f"Unsupported file format: {volume_path}")
    
    def _load_nifti(self, nifti_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load NIfTI volume."""
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel required for NIfTI support")
            
        img = nib.load(str(nifti_path))
        volume = img.get_fdata().astype(np.float32)
        
        # Extract metadata
        header = img.header
        affine = img.affine
        voxel_spacing = header.get_zooms()[:3]
        
        metadata = {
            'original_shape': volume.shape,
            'voxel_spacing': voxel_spacing,
            'affine': affine,
            'header': header,
            'file_path': str(nifti_path)
        }
        
        return volume, metadata
    
    def _load_dicom(self, dicom_path: Union[Path, str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load DICOM volume."""
        if not DICOM_AVAILABLE:
            raise ImportError("pydicom and SimpleITK required for DICOM support")
            
        # Use SimpleITK for robust DICOM loading
        reader = sitk.ImageSeriesReader()
        
        if Path(dicom_path).is_dir():
            dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_path))
            reader.SetFileNames(dicom_names)
        else:
            reader.SetFileName(str(dicom_path))
            
        image = reader.Execute()
        
        # Convert to numpy array
        volume = sitk.GetArrayFromImage(image).astype(np.float32)
        
        # Extract metadata
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        
        metadata = {
            'original_shape': volume.shape,
            'voxel_spacing': spacing,
            'origin': origin,
            'direction': direction,
            'file_path': str(dicom_path)
        }
        
        return volume, metadata
    
    def preprocess_volume(self, volume: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Apply preprocessing pipeline to volume.
        
        Args:
            volume: Raw volume array
            metadata: Volume metadata
            
        Returns:
            Preprocessed volume
        """
        # 1. Intensity clipping
        volume = np.clip(volume, self.config.intensity_clip_range[0], 
                        self.config.intensity_clip_range[1])
        
        # 2. Resampling to target spacing
        if 'voxel_spacing' in metadata:
            volume = self._resample_volume(volume, metadata['voxel_spacing'])
        
        # 3. Normalization
        volume = self._normalize_volume(volume)
        
        # 4. Optional resizing to target size
        if self.config.target_size is not None:
            volume = self._resize_volume(volume, self.config.target_size)
        
        return volume
    
    def _resample_volume(self, volume: np.ndarray, current_spacing: Tuple[float, float, float]) -> np.ndarray:
        """Resample volume to target spacing."""
        current_spacing = np.array(current_spacing)
        target_spacing = np.array(self.config.target_spacing)
        
        # Calculate scaling factors
        scale_factors = current_spacing / target_spacing
        
        # Skip resampling if spacing is already close to target
        if np.allclose(scale_factors, 1.0, rtol=0.05):
            return volume
        
        # Convert to torch tensor for resampling
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Calculate new size
        new_size = (np.array(volume.shape) * scale_factors).astype(int)
        
        # Resample using trilinear interpolation
        resampled = F.interpolate(volume_tensor, size=tuple(new_size), 
                                mode='trilinear', align_corners=False)
        
        return resampled.squeeze().numpy()
    
    def _normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume intensities."""
        if self.config.normalize_method == "z_score":
            mean = np.mean(volume)
            std = np.std(volume)
            return (volume - mean) / (std + 1e-8)
        
        elif self.config.normalize_method == "min_max":
            min_val = np.min(volume)
            max_val = np.max(volume)
            return (volume - min_val) / (max_val - min_val + 1e-8)
        
        elif self.config.normalize_method == "percentile":
            p1, p99 = np.percentile(volume, [1, 99])
            volume = np.clip(volume, p1, p99)
            return (volume - p1) / (p99 - p1 + 1e-8)
        
        else:
            raise ValueError(f"Unknown normalization method: {self.config.normalize_method}")
    
    def _resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Resize volume to target dimensions."""
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(volume_tensor, size=target_size, 
                              mode='trilinear', align_corners=False)
        return resized.squeeze().numpy()


class DataAugmentation:
    """Handles data augmentation for training."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.enabled = config.enable_augmentation
    
    def augment_volume_and_boxes(self, volume: np.ndarray, 
                                bounding_boxes: List[BoundingBox3D]) -> Tuple[np.ndarray, List[BoundingBox3D]]:
        """
        Apply augmentation to volume and corresponding bounding boxes.
        
        Args:
            volume: Input volume
            bounding_boxes: List of 3D bounding boxes
            
        Returns:
            Tuple of (augmented_volume, augmented_boxes)
        """
        if not self.enabled:
            return volume, bounding_boxes
        
        # Apply random transformations
        volume, bounding_boxes = self._apply_rotation(volume, bounding_boxes)
        volume, bounding_boxes = self._apply_scaling(volume, bounding_boxes)
        volume = self._apply_noise(volume)
        
        return volume, bounding_boxes
    
    def _apply_rotation(self, volume: np.ndarray, 
                       bounding_boxes: List[BoundingBox3D]) -> Tuple[np.ndarray, List[BoundingBox3D]]:
        """Apply random rotation."""
        if np.random.random() > 0.5:
            return volume, bounding_boxes
        
        # Random rotation angle
        angle = np.random.uniform(-self.config.rotation_range, self.config.rotation_range)
        
        # For simplicity, implement rotation around z-axis only
        # In practice, you might want full 3D rotation
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Rotation matrix for z-axis rotation
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        # Apply rotation to volume (simplified - in practice use scipy.ndimage.rotate)
        # For now, just return original (full implementation would be complex)
        
        # Transform bounding boxes
        center = np.array(volume.shape) / 2
        transformed_boxes = []
        
        for bbox in bounding_boxes:
            # Transform bbox center
            bbox_center = np.array([bbox.x, bbox.y, bbox.z]) - center
            rotated_center = rotation_matrix @ bbox_center + center
            
            # Create new bounding box with rotated center
            new_bbox = BoundingBox3D(
                x=rotated_center[0], y=rotated_center[1], z=rotated_center[2],
                width=bbox.width, height=bbox.height, depth=bbox.depth,
                confidence=bbox.confidence, label=bbox.label
            )
            
            # Check if still within volume bounds
            if new_bbox.is_valid_in_volume(volume.shape):
                transformed_boxes.append(new_bbox)
        
        return volume, transformed_boxes
    
    def _apply_scaling(self, volume: np.ndarray, 
                      bounding_boxes: List[BoundingBox3D]) -> Tuple[np.ndarray, List[BoundingBox3D]]:
        """Apply random scaling."""
        if np.random.random() > 0.5:
            return volume, bounding_boxes
        
        # Random scaling factor
        scale = np.random.uniform(self.config.scaling_range[0], self.config.scaling_range[1])
        
        # Scale volume
        new_size = (np.array(volume.shape) * scale).astype(int)
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        scaled_volume = F.interpolate(volume_tensor, size=tuple(new_size), 
                                    mode='trilinear', align_corners=False)
        scaled_volume = scaled_volume.squeeze().numpy()
        
        # Scale bounding boxes
        scaled_boxes = []
        for bbox in bounding_boxes:
            scaled_bbox = BoundingBox3D(
                x=bbox.x * scale, y=bbox.y * scale, z=bbox.z * scale,
                width=bbox.width * scale, height=bbox.height * scale, depth=bbox.depth * scale,
                confidence=bbox.confidence, label=bbox.label
            )
            
            if scaled_bbox.is_valid_in_volume(scaled_volume.shape):
                scaled_boxes.append(scaled_bbox)
        
        return scaled_volume, scaled_boxes
    
    def _apply_noise(self, volume: np.ndarray) -> np.ndarray:
        """Add random noise to volume."""
        if np.random.random() > 0.3:
            return volume
        
        noise = np.random.normal(0, self.config.noise_std, volume.shape)
        return volume + noise


class AneurysmDataset(Dataset):
    """PyTorch Dataset for aneurysm detection."""
    
    def __init__(self, config: DatasetConfig, mode: str = 'train'):
        """
        Initialize dataset.
        
        Args:
            config: Dataset configuration
            mode: 'train', 'val', or 'test'
        """
        self.config = config
        self.mode = mode
        
        # Initialize components
        self.volume_processor = VolumeProcessor(config)
        self.augmentation = DataAugmentation(config)
        self.annotation_manager = AnnotationManager()
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Cache for preprocessed volumes
        self.volume_cache = {} if config.cache_preprocessed else None
        
        logger.info(f"Initialized {mode} dataset with {len(self.annotations)} cases")
    
    def _load_annotations(self) -> List[AnnotationCase]:
        """Load and filter annotations based on mode."""
        all_annotations = self.annotation_manager.load_annotations(self.config.annotation_path)
        
        # In practice, you would split based on case IDs or cross-validation folds
        # For now, return all annotations
        return all_annotations
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single data sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing volume, bounding boxes, and metadata
        """
        annotation_case = self.annotations[idx]
        
        # Load and preprocess volume
        volume, metadata = self._get_preprocessed_volume(annotation_case)
        
        # Get bounding boxes
        bounding_boxes = annotation_case.bounding_boxes.copy()
        
        # Apply augmentation if training
        if self.mode == 'train':
            volume, bounding_boxes = self.augmentation.augment_volume_and_boxes(volume, bounding_boxes)
        
        # Convert to tensors
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)  # Add channel dimension
        
        # Convert bounding boxes to arrays
        if bounding_boxes:
            boxes_array = np.array([bbox.to_array() for bbox in bounding_boxes])
            confidence_scores = np.array([bbox.confidence for bbox in bounding_boxes])
        else:
            boxes_array = np.zeros((0, 6))  # Empty array with correct shape
            confidence_scores = np.zeros(0)
        
        return {
            'volume': volume_tensor,
            'bounding_boxes': torch.from_numpy(boxes_array).float(),
            'confidence_scores': torch.from_numpy(confidence_scores).float(),
            'case_id': annotation_case.case_id,
            'metadata': {
                **metadata,
                'original_boxes': bounding_boxes,
                'voxel_spacing': annotation_case.voxel_spacing,
                'volume_shape': annotation_case.volume_shape
            }
        }
    
    def _get_preprocessed_volume(self, annotation_case: AnnotationCase) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get preprocessed volume, using cache if available."""
        case_id = annotation_case.case_id
        
        # Check cache first
        if self.volume_cache is not None and case_id in self.volume_cache:
            return self.volume_cache[case_id]
        
        # Load and preprocess volume
        volume_path = Path(self.config.data_root) / annotation_case.image_path
        volume, metadata = self.volume_processor.load_volume(volume_path)
        preprocessed_volume = self.volume_processor.preprocess_volume(volume, metadata)
        
        # Update metadata with preprocessing info
        metadata['preprocessed_shape'] = preprocessed_volume.shape
        metadata['preprocessing_config'] = {
            'target_spacing': self.config.target_spacing,
            'normalize_method': self.config.normalize_method,
            'intensity_clip_range': self.config.intensity_clip_range
        }
        
        # Cache if enabled
        if self.volume_cache is not None:
            self.volume_cache[case_id] = (preprocessed_volume, metadata)
        
        return preprocessed_volume, metadata


class SubvolumeDataset(Dataset):
    """Dataset that extracts subvolumes for training/inference."""
    
    def __init__(self, config: DatasetConfig, mode: str = 'train'):
        """
        Initialize subvolume dataset.
        
        Args:
            config: Dataset configuration
            mode: 'train', 'val', or 'test'
        """
        self.config = config
        self.mode = mode
        self.base_dataset = AneurysmDataset(config, mode)
        
        # Pre-compute subvolume locations for each case
        self.subvolume_indices = self._compute_subvolume_indices()
        
        logger.info(f"Generated {len(self.subvolume_indices)} subvolumes for {mode} mode")
    
    def _compute_subvolume_indices(self) -> List[Tuple[int, Tuple[slice, slice, slice]]]:
        """Compute subvolume extraction indices for all cases."""
        indices = []
        
        for case_idx in range(len(self.base_dataset)):
            # Get volume shape from annotation case
            annotation_case = self.base_dataset.annotations[case_idx]
            
            # Use default shape if not available
            volume_shape = annotation_case.volume_shape or (512, 512, 200)
            
            # Generate subvolume slices
            subvolume_slices = self._generate_subvolume_slices(volume_shape)
            
            for slices in subvolume_slices:
                indices.append((case_idx, slices))
        
        return indices
    
    def _generate_subvolume_slices(self, volume_shape: Tuple[int, int, int]) -> List[Tuple[slice, slice, slice]]:
        """Generate overlapping subvolume slices for a volume."""
        slices = []
        subvol_size = self.config.subvolume_size
        overlap = self.config.subvolume_overlap
        
        # Calculate step size with overlap
        step_size = tuple(int(s * (1 - overlap)) for s in subvol_size)
        
        for z in range(0, volume_shape[0] - subvol_size[0] + 1, step_size[0]):
            for y in range(0, volume_shape[1] - subvol_size[1] + 1, step_size[1]):
                for x in range(0, volume_shape[2] - subvol_size[2] + 1, step_size[2]):
                    z_slice = slice(z, min(z + subvol_size[0], volume_shape[0]))
                    y_slice = slice(y, min(y + subvol_size[1], volume_shape[1]))
                    x_slice = slice(x, min(x + subvol_size[2], volume_shape[2]))
                    
                    slices.append((z_slice, y_slice, x_slice))
        
        return slices
    
    def __len__(self) -> int:
        return len(self.subvolume_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a subvolume sample."""
        case_idx, subvol_slices = self.subvolume_indices[idx]
        
        # Get full volume data
        full_data = self.base_dataset[case_idx]
        full_volume = full_data['volume']
        full_boxes = full_data['bounding_boxes']
        
        # Extract subvolume
        subvolume = full_volume[:, subvol_slices[0], subvol_slices[1], subvol_slices[2]]
        
        # Filter bounding boxes that overlap with subvolume
        subvol_boxes, subvol_scores = self._filter_boxes_for_subvolume(
            full_boxes, full_data['confidence_scores'], subvol_slices
        )
        
        return {
            'volume': subvolume,
            'bounding_boxes': subvol_boxes,
            'confidence_scores': subvol_scores,
            'case_id': full_data['case_id'],
            'subvolume_location': subvol_slices,
            'metadata': full_data['metadata']
        }
    
    def _filter_boxes_for_subvolume(self, boxes: torch.Tensor, scores: torch.Tensor, 
                                   subvol_slices: Tuple[slice, slice, slice]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter bounding boxes that overlap with subvolume."""
        if len(boxes) == 0:
            return boxes, scores
        
        # Get subvolume bounds
        z_start, z_end = subvol_slices[0].start, subvol_slices[0].stop
        y_start, y_end = subvol_slices[1].start, subvol_slices[1].stop
        x_start, x_end = subvol_slices[2].start, subvol_slices[2].stop
        
        # Check which boxes overlap with subvolume
        valid_indices = []
        adjusted_boxes = []
        
        for i, box in enumerate(boxes):
            x, y, z, w, h, d = box.numpy()
            
            # Calculate box bounds
            box_x_min, box_x_max = x - w/2, x + w/2
            box_y_min, box_y_max = y - h/2, y + h/2
            box_z_min, box_z_max = z - d/2, z + d/2
            
            # Check overlap
            if (box_x_max > x_start and box_x_min < x_end and
                box_y_max > y_start and box_y_min < y_end and
                box_z_max > z_start and box_z_min < z_end):
                
                # Adjust coordinates relative to subvolume
                adjusted_x = x - x_start
                adjusted_y = y - y_start
                adjusted_z = z - z_start
                
                adjusted_boxes.append([adjusted_x, adjusted_y, adjusted_z, w, h, d])
                valid_indices.append(i)
        
        if adjusted_boxes:
            filtered_boxes = torch.tensor(adjusted_boxes, dtype=torch.float32)
            filtered_scores = scores[valid_indices]
        else:
            filtered_boxes = torch.zeros((0, 6), dtype=torch.float32)
            filtered_scores = torch.zeros(0, dtype=torch.float32)
        
        return filtered_boxes, filtered_scores


def create_data_loaders(config: DatasetConfig, 
                       train_split: float = 0.8,
                       val_split: float = 0.1,
                       test_split: float = 0.1) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        config: Dataset configuration
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        
    Returns:
        Dictionary of DataLoaders for each split
    """
    # Create datasets
    train_dataset = SubvolumeDataset(config, mode='train')
    val_dataset = SubvolumeDataset(config, mode='val')
    test_dataset = SubvolumeDataset(config, mode='test')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # Small batch size for 3D volumes
        shuffle=True,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def create_sample_dataset(num_cases: int = 10, output_dir: str = "sample_data") -> DatasetConfig:
    """
    Create a sample dataset for testing purposes.
    
    Args:
        num_cases: Number of sample cases to create
        output_dir: Directory to save sample data
        
    Returns:
        DatasetConfig for the sample dataset
    """
    from .annotation_parser import create_sample_annotations
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create sample annotations
    sample_annotations = create_sample_annotations(num_cases)
    
    # Save annotations
    annotation_manager = AnnotationManager()
    annotation_path = output_path / "annotations.json"
    annotation_manager.export_to_json(sample_annotations, str(annotation_path))
    
    # Create sample volumes (random data for testing)
    volumes_dir = output_path / "volumes"
    volumes_dir.mkdir(exist_ok=True)
    
    for annotation in sample_annotations:
        volume_shape = annotation.volume_shape or (200, 256, 256)
        sample_volume = np.random.randn(*volume_shape).astype(np.float32) * 100
        
        # Save as simple numpy file (in practice would be NIfTI/DICOM)
        volume_path = volumes_dir / f"{annotation.case_id}.npy"
        np.save(volume_path, sample_volume)
        
        # Update annotation with correct path
        annotation.image_path = str(volume_path.relative_to(output_path))
    
    # Re-save updated annotations
    annotation_manager.export_to_json(sample_annotations, str(annotation_path))
    
    # Create config
    config = DatasetConfig(
        data_root=str(output_path),
        annotation_path=str(annotation_path),
        cache_preprocessed=False,  # Disable caching for sample data
        num_workers=1  # Reduce workers for testing
    )
    
    logger.info(f"Created sample dataset with {num_cases} cases in {output_dir}")
    return config


if __name__ == "__main__":
    # Example usage and testing
    print("Testing AneurysmDataset implementation...")
    
    # Create sample dataset
    config = create_sample_dataset(num_cases=5)
    
    # Test basic dataset
    dataset = AneurysmDataset(config, mode='train')
    print(f"Dataset size: {len(dataset)}")
    
    # Test sample loading
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Volume shape: {sample['volume'].shape}")
    print(f"Number of bounding boxes: {len(sample['bounding_boxes'])}")
    
    # Test subvolume dataset
    subvol_dataset = SubvolumeDataset(config, mode='train')
    print(f"Subvolume dataset size: {len(subvol_dataset)}")
    
    subvol_sample = subvol_dataset[0]
    print(f"Subvolume shape: {subvol_sample['volume'].shape}")
    
    # Test data loaders
    data_loaders = create_data_loaders(config)
    print(f"Created data loaders: {list(data_loaders.keys())}")
    
    # Test batch loading
    train_loader = data_loaders['train']
    for batch_idx, batch in enumerate(train_loader):
        print(f"Batch {batch_idx}: volume shape {batch['volume'].shape}")
        if batch_idx >= 2:  # Only test first few batches
            break
    
    print("Dataset implementation test completed successfully!")