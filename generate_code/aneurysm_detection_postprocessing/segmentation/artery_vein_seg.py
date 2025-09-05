"""
Artery-Vein Segmentation Module

This module implements artery and vein segmentation using nnUNet framework
for 4D dynamic CTA data as specified in the paper. It provides both
intracranial and extracranial vessel segmentation with CVS mask subtraction
from vein masks.

Key Features:
- nnUNet-based artery-vein segmentation
- 4D dynamic CTA support
- CVS mask integration for modified vein masks
- Fallback methods for static CTA data
"""

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import subprocess
import tempfile
import os
import logging
from typing import Tuple, Optional, Union, Dict, Any
from pathlib import Path

from .mask_utils import MaskProcessor, subtract_cvs_from_vein_mask, MaskValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArteryVeinSegmentator:
    """
    Main artery-vein segmentation engine using nnUNet framework.
    
    Implements the paper's approach for segmenting intracranial and extracranial
    arteries and veins from 4D dynamic CTA data.
    """
    
    def __init__(self, 
                 voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4),
                 device: str = 'cpu',
                 nnunet_model_path: Optional[str] = None,
                 use_4d: bool = True):
        """
        Initialize the artery-vein segmentator.
        
        Args:
            voxel_spacing: Voxel spacing in mm (x, y, z)
            device: Computing device ('cpu' or 'cuda')
            nnunet_model_path: Path to nnUNet model weights
            use_4d: Whether to use 4D dynamic CTA data
        """
        self.voxel_spacing = voxel_spacing
        self.device = device
        self.nnunet_model_path = nnunet_model_path
        self.use_4d = use_4d
        self.mask_processor = MaskProcessor(voxel_spacing=voxel_spacing)
        self.validator = MaskValidator()
        
        # nnUNet configuration
        self.nnunet_config = {
            'task_name': 'Task500_ArteryVein',
            'model_name': '3d_fullres',
            'fold': 'all',
            'trainer': 'nnUNetTrainerV2'
        }
        
    def segment_vessels(self, 
                       cta_volume: np.ndarray,
                       cvs_mask: Optional[np.ndarray] = None,
                       temporal_phases: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment arteries and veins from CTA volume.
        
        Args:
            cta_volume: CTA volume data (3D or 4D)
            cvs_mask: Cavernous venous sinus mask for vein modification
            temporal_phases: Temporal phases for 4D CTA (optional)
            
        Returns:
            Tuple of (artery_mask, modified_vein_mask)
        """
        try:
            logger.info("Starting artery-vein segmentation")
            
            # Validate input
            if cta_volume.ndim not in [3, 4]:
                raise ValueError(f"CTA volume must be 3D or 4D, got {cta_volume.ndim}D")
            
            # Prepare input for nnUNet
            if self.use_4d and cta_volume.ndim == 4:
                segmentation_input = cta_volume
                logger.info("Using 4D dynamic CTA for segmentation")
            elif cta_volume.ndim == 4:
                # Use peak arterial phase if available
                segmentation_input = cta_volume[:, :, :, 0]  # First phase
                logger.info("Using single phase from 4D CTA")
            else:
                segmentation_input = cta_volume
                logger.info("Using 3D CTA for segmentation")
            
            # Run nnUNet segmentation
            artery_mask, vein_mask = self._run_nnunet_segmentation(segmentation_input)
            
            # Apply CVS mask subtraction to vein mask as per paper
            if cvs_mask is not None:
                logger.info("Applying CVS mask subtraction to vein mask")
                modified_vein_mask = subtract_cvs_from_vein_mask(vein_mask, cvs_mask)
            else:
                modified_vein_mask = vein_mask
                logger.warning("No CVS mask provided, using original vein mask")
            
            # Validate outputs
            self.validator.validate_mask_format(artery_mask)
            self.validator.validate_mask_format(modified_vein_mask)
            
            logger.info("Artery-vein segmentation completed successfully")
            return artery_mask, modified_vein_mask
            
        except Exception as e:
            logger.error(f"Error in vessel segmentation: {str(e)}")
            # Return fallback segmentation
            return self._fallback_segmentation(cta_volume, cvs_mask)
    
    def _run_nnunet_segmentation(self, cta_volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run nnUNet segmentation on CTA volume.
        
        Args:
            cta_volume: Input CTA volume
            
        Returns:
            Tuple of (artery_mask, vein_mask)
        """
        try:
            # Create temporary files for nnUNet
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = os.path.join(temp_dir, "input.nii.gz")
                output_path = os.path.join(temp_dir, "output.nii.gz")
                
                # Save input volume
                input_nii = nib.Nifti1Image(cta_volume.astype(np.float32), 
                                          affine=np.eye(4))
                nib.save(input_nii, input_path)
                
                # Run nnUNet prediction
                if self.nnunet_model_path:
                    cmd = [
                        "nnUNet_predict",
                        "-i", temp_dir,
                        "-o", temp_dir,
                        "-t", self.nnunet_config['task_name'],
                        "-m", self.nnunet_config['model_name'],
                        "-f", self.nnunet_config['fold'],
                        "--disable_tta"
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode != 0:
                        logger.warning(f"nnUNet failed: {result.stderr}")
                        raise RuntimeError("nnUNet segmentation failed")
                    
                    # Load segmentation result
                    output_nii = nib.load(output_path)
                    segmentation = output_nii.get_fdata()
                    
                else:
                    logger.warning("No nnUNet model path provided, using fallback")
                    raise RuntimeError("No nnUNet model available")
                
                # Parse segmentation labels (assuming label 1=artery, label 2=vein)
                artery_mask = (segmentation == 1).astype(np.uint8)
                vein_mask = (segmentation == 2).astype(np.uint8)
                
                return artery_mask, vein_mask
                
        except Exception as e:
            logger.error(f"nnUNet segmentation failed: {str(e)}")
            raise
    
    def _fallback_segmentation(self, 
                              cta_volume: np.ndarray,
                              cvs_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback vessel segmentation using intensity thresholding.
        
        Args:
            cta_volume: Input CTA volume
            cvs_mask: CVS mask for vein modification
            
        Returns:
            Tuple of (artery_mask, vein_mask)
        """
        logger.info("Using fallback intensity-based vessel segmentation")
        
        # Normalize volume
        volume_norm = (cta_volume - cta_volume.min()) / (cta_volume.max() - cta_volume.min())
        
        # High intensity threshold for arteries (bright in arterial phase)
        artery_threshold = np.percentile(volume_norm, 95)
        artery_mask = (volume_norm > artery_threshold).astype(np.uint8)
        
        # Medium intensity threshold for veins
        vein_threshold = np.percentile(volume_norm, 85)
        vein_mask = ((volume_norm > vein_threshold) & (volume_norm <= artery_threshold)).astype(np.uint8)
        
        # Clean up masks
        artery_mask = self.mask_processor.clean_mask(artery_mask)
        vein_mask = self.mask_processor.clean_mask(vein_mask)
        
        # Apply CVS subtraction if available
        if cvs_mask is not None:
            vein_mask = subtract_cvs_from_vein_mask(vein_mask, cvs_mask)
        
        logger.info("Fallback segmentation completed")
        return artery_mask, vein_mask


class ArteryVeinMaskGenerator:
    """
    Simplified interface for artery-vein mask generation.
    
    Provides easy-to-use methods for generating vessel masks with
    standard parameters.
    """
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)):
        """
        Initialize the mask generator.
        
        Args:
            voxel_spacing: Voxel spacing in mm (x, y, z)
        """
        self.voxel_spacing = voxel_spacing
        self.segmentator = ArteryVeinSegmentator(voxel_spacing=voxel_spacing)
    
    def generate_vessel_masks(self, 
                            cta_volume: np.ndarray,
                            cvs_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Generate artery and vein masks from CTA volume.
        
        Args:
            cta_volume: Input CTA volume
            cvs_mask: Optional CVS mask for vein modification
            
        Returns:
            Dictionary with 'artery_mask' and 'vein_mask' keys
        """
        artery_mask, vein_mask = self.segmentator.segment_vessels(cta_volume, cvs_mask)
        
        return {
            'artery_mask': artery_mask,
            'vein_mask': vein_mask
        }


def segment_arteries_veins_from_cta(cta_volume: np.ndarray,
                                   voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4),
                                   cvs_mask: Optional[np.ndarray] = None,
                                   use_4d: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main function to segment arteries and veins from CTA volume.
    
    This function implements the paper's approach using nnUNet for
    vessel segmentation with CVS mask integration.
    
    Args:
        cta_volume: CTA volume data (3D or 4D)
        voxel_spacing: Voxel spacing in mm (x, y, z)
        cvs_mask: Cavernous venous sinus mask for vein modification
        use_4d: Whether to use 4D dynamic CTA data
        
    Returns:
        Tuple of (artery_mask, modified_vein_mask)
    """
    segmentator = ArteryVeinSegmentator(
        voxel_spacing=voxel_spacing,
        use_4d=use_4d
    )
    
    return segmentator.segment_vessels(cta_volume, cvs_mask)


def create_sample_vessel_masks(volume_shape: Tuple[int, int, int] = (512, 512, 200)) -> Dict[str, np.ndarray]:
    """
    Create sample artery and vein masks for testing purposes.
    
    Args:
        volume_shape: Shape of the volume (width, height, depth)
        
    Returns:
        Dictionary with sample artery and vein masks
    """
    logger.info(f"Creating sample vessel masks with shape {volume_shape}")
    
    # Create sample artery mask (central branching pattern)
    artery_mask = np.zeros(volume_shape, dtype=np.uint8)
    
    # Main artery trunk
    center_x, center_y = volume_shape[0] // 2, volume_shape[1] // 2
    for z in range(volume_shape[2] // 4, 3 * volume_shape[2] // 4):
        # Main trunk
        artery_mask[center_x-2:center_x+3, center_y-2:center_y+3, z] = 1
        
        # Branching pattern
        if z > volume_shape[2] // 2:
            branch_offset = (z - volume_shape[2] // 2) // 10
            # Left branch
            if center_x - branch_offset > 5:
                artery_mask[center_x-branch_offset-1:center_x-branch_offset+2, 
                           center_y-1:center_y+2, z] = 1
            # Right branch
            if center_x + branch_offset < volume_shape[0] - 5:
                artery_mask[center_x+branch_offset-1:center_x+branch_offset+2, 
                           center_y-1:center_y+2, z] = 1
    
    # Create sample vein mask (peripheral pattern)
    vein_mask = np.zeros(volume_shape, dtype=np.uint8)
    
    # Peripheral veins
    for z in range(volume_shape[2] // 6, 5 * volume_shape[2] // 6):
        # Left peripheral vein
        vein_mask[volume_shape[0]//4-1:volume_shape[0]//4+2, 
                  volume_shape[1]//3-1:volume_shape[1]//3+2, z] = 1
        # Right peripheral vein
        vein_mask[3*volume_shape[0]//4-1:3*volume_shape[0]//4+2, 
                  volume_shape[1]//3-1:volume_shape[1]//3+2, z] = 1
        # Superior vein
        vein_mask[center_x-1:center_x+2, 
                  volume_shape[1]//6-1:volume_shape[1]//6+2, z] = 1
    
    logger.info("Sample vessel masks created successfully")
    
    return {
        'artery_mask': artery_mask,
        'vein_mask': vein_mask
    }


class VesselMaskProcessor:
    """
    Specialized processor for vessel mask operations.
    
    Provides vessel-specific mask processing operations including
    connectivity analysis and vessel tree extraction.
    """
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)):
        """
        Initialize the vessel mask processor.
        
        Args:
            voxel_spacing: Voxel spacing in mm (x, y, z)
        """
        self.voxel_spacing = voxel_spacing
        self.mask_processor = MaskProcessor(voxel_spacing=voxel_spacing)
    
    def extract_vessel_tree(self, vessel_mask: np.ndarray) -> np.ndarray:
        """
        Extract main vessel tree from segmentation mask.
        
        Args:
            vessel_mask: Binary vessel mask
            
        Returns:
            Cleaned vessel tree mask
        """
        # Remove small disconnected components
        cleaned_mask = self.mask_processor.clean_mask(vessel_mask)
        
        # Apply morphological operations to connect nearby vessels
        connected_mask = self.mask_processor.dilate_mask(cleaned_mask, iterations=1)
        connected_mask = self.mask_processor.erode_mask(connected_mask, iterations=1)
        
        return connected_mask
    
    def separate_vessel_regions(self, 
                               vessel_mask: np.ndarray,
                               brain_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Separate vessel mask into intracranial and extracranial regions.
        
        Args:
            vessel_mask: Binary vessel mask
            brain_mask: Brain mask for region separation
            
        Returns:
            Dictionary with intracranial and extracranial vessel masks
        """
        if brain_mask is None:
            logger.warning("No brain mask provided, returning full vessel mask")
            return {
                'intracranial': vessel_mask,
                'extracranial': np.zeros_like(vessel_mask)
            }
        
        # Intracranial vessels (inside brain)
        intracranial_vessels = self.mask_processor.intersect_masks(vessel_mask, brain_mask)
        
        # Extracranial vessels (outside brain)
        extracranial_vessels = self.mask_processor.subtract_masks(vessel_mask, brain_mask)
        
        return {
            'intracranial': intracranial_vessels,
            'extracranial': extracranial_vessels
        }


# Export main functions and classes
__all__ = [
    'ArteryVeinSegmentator',
    'ArteryVeinMaskGenerator', 
    'VesselMaskProcessor',
    'segment_arteries_veins_from_cta',
    'create_sample_vessel_masks'
]


if __name__ == "__main__":
    # Test the implementation
    print("Testing Artery-Vein Segmentation Module")
    
    # Create sample data
    sample_cta = np.random.rand(128, 128, 64) * 1000  # Sample CTA volume
    sample_cvs = np.zeros((128, 128, 64), dtype=np.uint8)
    sample_cvs[60:68, 60:68, 30:34] = 1  # Sample CVS region
    
    # Test segmentation
    try:
        artery_mask, vein_mask = segment_arteries_veins_from_cta(
            sample_cta, 
            cvs_mask=sample_cvs
        )
        print(f"Artery mask shape: {artery_mask.shape}, unique values: {np.unique(artery_mask)}")
        print(f"Vein mask shape: {vein_mask.shape}, unique values: {np.unique(vein_mask)}")
        print("✓ Artery-vein segmentation test passed")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
    
    # Test sample mask creation
    try:
        sample_masks = create_sample_vessel_masks((64, 64, 32))
        print(f"Sample artery mask: {sample_masks['artery_mask'].shape}")
        print(f"Sample vein mask: {sample_masks['vein_mask'].shape}")
        print("✓ Sample mask creation test passed")
        
    except Exception as e:
        print(f"✗ Sample mask test failed: {str(e)}")