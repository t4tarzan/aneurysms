"""
Mask processing utilities for anatomical segmentation in aneurysm detection.

This module provides utility functions for processing anatomical masks including
dilation, erosion, intersection operations, and mask validation used across
all segmentation components.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, binary_closing, binary_opening
from typing import Tuple, Optional, Union, List, Dict, Any
import warnings


class MaskProcessor:
    """
    Utility class for processing anatomical masks with various morphological operations.
    
    Provides methods for dilation, erosion, intersection, union, and validation
    of 3D anatomical masks used in aneurysm detection post-processing.
    """
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)):
        """
        Initialize mask processor with voxel spacing information.
        
        Args:
            voxel_spacing: Voxel spacing in mm for (x, y, z) dimensions
        """
        self.voxel_spacing = voxel_spacing
    
    def dilate_mask(self, mask: np.ndarray, distance_mm: float) -> np.ndarray:
        """
        Dilate mask by specified distance in millimeters.
        
        Args:
            mask: Binary mask array (3D)
            distance_mm: Dilation distance in millimeters
            
        Returns:
            Dilated binary mask
        """
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        
        # Convert distance to voxels for each dimension
        voxel_distances = [
            int(np.ceil(distance_mm / spacing)) 
            for spacing in self.voxel_spacing
        ]
        
        # Create structuring element
        structure = self._create_spherical_kernel(voxel_distances)
        
        # Perform dilation
        dilated_mask = binary_dilation(mask.astype(bool), structure=structure)
        
        return dilated_mask.astype(np.uint8)
    
    def erode_mask(self, mask: np.ndarray, distance_mm: float) -> np.ndarray:
        """
        Erode mask by specified distance in millimeters.
        
        Args:
            mask: Binary mask array (3D)
            distance_mm: Erosion distance in millimeters
            
        Returns:
            Eroded binary mask
        """
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        
        # Convert distance to voxels for each dimension
        voxel_distances = [
            int(np.ceil(distance_mm / spacing)) 
            for spacing in self.voxel_spacing
        ]
        
        # Create structuring element
        structure = self._create_spherical_kernel(voxel_distances)
        
        # Perform erosion
        eroded_mask = binary_erosion(mask.astype(bool), structure=structure)
        
        return eroded_mask.astype(np.uint8)
    
    def _create_spherical_kernel(self, voxel_distances: List[int]) -> np.ndarray:
        """
        Create spherical structuring element for morphological operations.
        
        Args:
            voxel_distances: Distances in voxels for each dimension
            
        Returns:
            3D spherical kernel
        """
        # Create coordinate grids
        z_size, y_size, x_size = [2 * d + 1 for d in voxel_distances]
        z, y, x = np.ogrid[:z_size, :y_size, :x_size]
        
        # Center coordinates
        z_center, y_center, x_center = voxel_distances
        
        # Create spherical kernel
        kernel = ((z - z_center) ** 2 / voxel_distances[0] ** 2 + 
                 (y - y_center) ** 2 / voxel_distances[1] ** 2 + 
                 (x - x_center) ** 2 / voxel_distances[2] ** 2) <= 1
        
        return kernel.astype(bool)
    
    def intersect_masks(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """
        Compute intersection of two masks.
        
        Args:
            mask1: First binary mask
            mask2: Second binary mask
            
        Returns:
            Intersection mask
        """
        if mask1.shape != mask2.shape:
            raise ValueError(f"Mask shapes must match: {mask1.shape} vs {mask2.shape}")
        
        intersection = np.logical_and(mask1.astype(bool), mask2.astype(bool))
        return intersection.astype(np.uint8)
    
    def union_masks(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """
        Compute union of two masks.
        
        Args:
            mask1: First binary mask
            mask2: Second binary mask
            
        Returns:
            Union mask
        """
        if mask1.shape != mask2.shape:
            raise ValueError(f"Mask shapes must match: {mask1.shape} vs {mask2.shape}")
        
        union = np.logical_or(mask1.astype(bool), mask2.astype(bool))
        return union.astype(np.uint8)
    
    def subtract_masks(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """
        Subtract mask2 from mask1.
        
        Args:
            mask1: Base mask
            mask2: Mask to subtract
            
        Returns:
            Subtraction result (mask1 - mask2)
        """
        if mask1.shape != mask2.shape:
            raise ValueError(f"Mask shapes must match: {mask1.shape} vs {mask2.shape}")
        
        result = np.logical_and(mask1.astype(bool), 
                               np.logical_not(mask2.astype(bool)))
        return result.astype(np.uint8)
    
    def clean_mask(self, mask: np.ndarray, min_component_size: int = 100) -> np.ndarray:
        """
        Clean mask by removing small connected components.
        
        Args:
            mask: Binary mask to clean
            min_component_size: Minimum size of components to keep
            
        Returns:
            Cleaned mask
        """
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        
        # Label connected components
        labeled_mask, num_components = ndimage.label(mask.astype(bool))
        
        # Find component sizes
        component_sizes = ndimage.sum(mask, labeled_mask, range(1, num_components + 1))
        
        # Create mask for components to keep
        keep_components = np.where(component_sizes >= min_component_size)[0] + 1
        
        # Create cleaned mask
        cleaned_mask = np.isin(labeled_mask, keep_components)
        
        return cleaned_mask.astype(np.uint8)
    
    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Fill holes in binary mask.
        
        Args:
            mask: Binary mask with potential holes
            
        Returns:
            Mask with holes filled
        """
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        
        # Fill holes using binary closing
        filled_mask = binary_closing(mask.astype(bool))
        
        return filled_mask.astype(np.uint8)


class BoundingBoxProcessor:
    """
    Utility class for processing 3D bounding boxes and their interactions with masks.
    
    Handles bounding box overlap calculations, mask intersection computations,
    and spatial relationship analysis for post-processing pipeline.
    """
    
    def __init__(self):
        """Initialize bounding box processor."""
        pass
    
    def calculate_overlap_ratio(self, bbox: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate overlap ratio between bounding box and mask.
        
        Args:
            bbox: Bounding box in format [x, y, z, w, h, d] (center + dimensions)
            mask: Binary mask array
            
        Returns:
            Overlap ratio (intersection volume / bbox volume)
        """
        # Convert center-based bbox to corner coordinates
        x_center, y_center, z_center, width, height, depth = bbox
        
        x_min = int(max(0, x_center - width // 2))
        x_max = int(min(mask.shape[2], x_center + width // 2))
        y_min = int(max(0, y_center - height // 2))
        y_max = int(min(mask.shape[1], y_center + height // 2))
        z_min = int(max(0, z_center - depth // 2))
        z_max = int(min(mask.shape[0], z_center + depth // 2))
        
        # Extract bbox region from mask
        bbox_region = mask[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Calculate overlap
        intersection_volume = np.sum(bbox_region)
        bbox_volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        
        if bbox_volume == 0:
            return 0.0
        
        overlap_ratio = intersection_volume / bbox_volume
        return float(overlap_ratio)
    
    def is_bbox_center_in_mask(self, bbox: np.ndarray, mask: np.ndarray) -> bool:
        """
        Check if bounding box center is inside mask.
        
        Args:
            bbox: Bounding box in format [x, y, z, w, h, d]
            mask: Binary mask array
            
        Returns:
            True if center is in mask, False otherwise
        """
        x_center, y_center, z_center = bbox[:3]
        
        # Convert to integer coordinates
        x_idx = int(round(x_center))
        y_idx = int(round(y_center))
        z_idx = int(round(z_center))
        
        # Check bounds
        if (0 <= z_idx < mask.shape[0] and 
            0 <= y_idx < mask.shape[1] and 
            0 <= x_idx < mask.shape[2]):
            return bool(mask[z_idx, y_idx, x_idx])
        
        return False
    
    def calculate_bbox_mask_intersection(self, bbox: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract mask region corresponding to bounding box.
        
        Args:
            bbox: Bounding box in format [x, y, z, w, h, d]
            mask: Binary mask array
            
        Returns:
            Mask region within bounding box
        """
        # Convert center-based bbox to corner coordinates
        x_center, y_center, z_center, width, height, depth = bbox
        
        x_min = int(max(0, x_center - width // 2))
        x_max = int(min(mask.shape[2], x_center + width // 2))
        y_min = int(max(0, y_center - height // 2))
        y_max = int(min(mask.shape[1], y_center + height // 2))
        z_min = int(max(0, z_center - depth // 2))
        z_max = int(min(mask.shape[0], z_center + depth // 2))
        
        # Extract and return region
        bbox_region = mask[z_min:z_max, y_min:y_max, x_min:x_max]
        return bbox_region.copy()
    
    def compare_overlaps(self, bbox: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> Tuple[float, float]:
        """
        Compare overlap ratios of bounding box with two different masks.
        
        Args:
            bbox: Bounding box in format [x, y, z, w, h, d]
            mask1: First mask for comparison
            mask2: Second mask for comparison
            
        Returns:
            Tuple of (overlap_ratio_1, overlap_ratio_2)
        """
        overlap1 = self.calculate_overlap_ratio(bbox, mask1)
        overlap2 = self.calculate_overlap_ratio(bbox, mask2)
        
        return overlap1, overlap2


class MaskValidator:
    """
    Utility class for validating mask properties and consistency.
    
    Provides methods to check mask validity, consistency across different
    anatomical regions, and quality metrics for segmentation results.
    """
    
    def __init__(self):
        """Initialize mask validator."""
        pass
    
    def validate_mask_format(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Validate mask format and properties.
        
        Args:
            mask: Mask array to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'properties': {}
        }
        
        # Check if mask is numpy array
        if not isinstance(mask, np.ndarray):
            validation_results['errors'].append("Mask must be numpy array")
            validation_results['is_valid'] = False
            return validation_results
        
        # Check dimensions
        if mask.ndim != 3:
            validation_results['errors'].append(f"Mask must be 3D, got {mask.ndim}D")
            validation_results['is_valid'] = False
        
        # Check data type
        if mask.dtype not in [np.uint8, np.bool_, bool]:
            validation_results['warnings'].append(f"Mask dtype {mask.dtype} not binary")
        
        # Check value range
        unique_values = np.unique(mask)
        if len(unique_values) > 2:
            validation_results['warnings'].append(f"Mask has {len(unique_values)} unique values")
        
        # Store properties
        validation_results['properties'] = {
            'shape': mask.shape,
            'dtype': str(mask.dtype),
            'unique_values': unique_values.tolist(),
            'total_voxels': mask.size,
            'positive_voxels': int(np.sum(mask > 0)),
            'fill_ratio': float(np.sum(mask > 0) / mask.size)
        }
        
        return validation_results
    
    def check_mask_consistency(self, masks: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Check consistency across multiple anatomical masks.
        
        Args:
            masks: Dictionary of mask name -> mask array
            
        Returns:
            Consistency check results
        """
        consistency_results = {
            'is_consistent': True,
            'errors': [],
            'warnings': [],
            'shape_consistency': True,
            'overlap_analysis': {}
        }
        
        if not masks:
            consistency_results['errors'].append("No masks provided")
            consistency_results['is_consistent'] = False
            return consistency_results
        
        # Check shape consistency
        shapes = [mask.shape for mask in masks.values()]
        if len(set(shapes)) > 1:
            consistency_results['errors'].append(f"Inconsistent shapes: {shapes}")
            consistency_results['is_consistent'] = False
            consistency_results['shape_consistency'] = False
        
        # Analyze overlaps between masks
        mask_names = list(masks.keys())
        for i, name1 in enumerate(mask_names):
            for j, name2 in enumerate(mask_names[i+1:], i+1):
                mask1, mask2 = masks[name1], masks[name2]
                
                if mask1.shape == mask2.shape:
                    intersection = np.sum(np.logical_and(mask1, mask2))
                    union = np.sum(np.logical_or(mask1, mask2))
                    
                    overlap_key = f"{name1}_vs_{name2}"
                    consistency_results['overlap_analysis'][overlap_key] = {
                        'intersection_voxels': int(intersection),
                        'union_voxels': int(union),
                        'jaccard_index': float(intersection / union) if union > 0 else 0.0
                    }
        
        return consistency_results


# Utility functions for common operations
def create_brain_mask_with_dilation(brain_mask: np.ndarray, 
                                  dilation_mm: float = 3.6,
                                  voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)) -> np.ndarray:
    """
    Create dilated brain mask as specified in the paper.
    
    Args:
        brain_mask: Original brain mask
        dilation_mm: Dilation distance in mm (default: 3.6mm as per paper)
        voxel_spacing: Voxel spacing in mm
        
    Returns:
        Dilated brain mask
    """
    processor = MaskProcessor(voxel_spacing)
    dilated_mask = processor.dilate_mask(brain_mask, dilation_mm)
    return dilated_mask


def create_cvs_mask_with_expansion(cvs_region: np.ndarray,
                                 venous_mask: np.ndarray,
                                 expansion_mm: float = 3.2,
                                 voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)) -> np.ndarray:
    """
    Create CVS mask with expansion as specified in the paper.
    
    Args:
        cvs_region: CVS region bounding box mask
        venous_mask: Venous segmentation mask
        expansion_mm: Expansion distance in mm (default: 3.2mm as per paper)
        voxel_spacing: Voxel spacing in mm
        
    Returns:
        Final CVS mask (expanded_box ∩ venous_mask)
    """
    processor = MaskProcessor(voxel_spacing)
    
    # Expand CVS region
    expanded_cvs = processor.dilate_mask(cvs_region, expansion_mm)
    
    # Intersect with venous mask
    final_cvs_mask = processor.intersect_masks(expanded_cvs, venous_mask)
    
    return final_cvs_mask


def subtract_cvs_from_vein_mask(vein_mask: np.ndarray, cvs_mask: np.ndarray) -> np.ndarray:
    """
    Create modified vein mask by subtracting CVS mask as per paper methodology.
    
    Args:
        vein_mask: Original vein segmentation mask
        cvs_mask: CVS mask to subtract
        
    Returns:
        Modified vein mask (vein_mask - cvs_mask)
    """
    processor = MaskProcessor()
    modified_vein_mask = processor.subtract_masks(vein_mask, cvs_mask)
    return modified_vein_mask


# Export main classes and functions
__all__ = [
    'MaskProcessor',
    'BoundingBoxProcessor', 
    'MaskValidator',
    'create_brain_mask_with_dilation',
    'create_cvs_mask_with_expansion',
    'subtract_cvs_from_vein_mask'
]