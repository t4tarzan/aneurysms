"""
Brain Segmentation Module for Aneurysm Detection Post-processing

This module implements brain mask generation using TotalSegmentator with
3.6mm dilation as specified in the paper for anatomical post-processing.

Key Features:
- TotalSegmentator integration for brain mask generation
- 3.6mm dilation in all directions as per paper specifications
- CVS region integration for skull base inclusion
- Robust error handling and validation
"""

import numpy as np
import nibabel as nib
import subprocess
import tempfile
import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Union
import torch
import torch.nn.functional as F

from .mask_utils import MaskProcessor, create_brain_mask_with_dilation, MaskValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrainSegmentator:
    """
    Brain segmentation using TotalSegmentator with paper-specific post-processing.
    
    Implements the brain mask generation pipeline described in the paper:
    1. Generate brain mask using TotalSegmentator
    2. Dilate mask by 3.6mm (9 pixels) in all 3 dimensions
    3. Combine with CVS region bounding box for skull base inclusion
    """
    
    def __init__(self, 
                 voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4),
                 dilation_mm: float = 3.6,
                 device: str = 'cpu',
                 totalsegmentator_path: Optional[str] = None):
        """
        Initialize brain segmentator.
        
        Args:
            voxel_spacing: Voxel spacing in mm (x, y, z)
            dilation_mm: Dilation amount in mm (paper specifies 3.6mm)
            device: Device for computation ('cpu' or 'cuda')
            totalsegmentator_path: Path to TotalSegmentator executable
        """
        self.voxel_spacing = voxel_spacing
        self.dilation_mm = dilation_mm
        self.device = device
        self.totalsegmentator_path = totalsegmentator_path or "TotalSegmentator"
        
        # Initialize mask processor
        self.mask_processor = MaskProcessor(voxel_spacing=voxel_spacing)
        self.validator = MaskValidator()
        
        logger.info(f"Initialized BrainSegmentator with {dilation_mm}mm dilation")
    
    def segment_brain(self, 
                     cta_volume: Union[np.ndarray, str, Path],
                     cvs_bounding_box: Optional[np.ndarray] = None,
                     output_dir: Optional[str] = None) -> np.ndarray:
        """
        Generate brain mask from CTA volume using TotalSegmentator.
        
        Args:
            cta_volume: CTA volume as numpy array or path to NIfTI file
            cvs_bounding_box: Optional CVS region bounding box for skull base inclusion
            output_dir: Optional directory for intermediate files
            
        Returns:
            Dilated brain mask as numpy array
        """
        try:
            # Handle input volume
            if isinstance(cta_volume, (str, Path)):
                volume_path = str(cta_volume)
                volume_array = self._load_volume(volume_path)
            else:
                volume_array = cta_volume
                # Save to temporary file for TotalSegmentator
                volume_path = self._save_temp_volume(volume_array, output_dir)
            
            # Validate input volume
            self.validator.validate_mask_format(volume_array, "input_volume")
            
            # Run TotalSegmentator
            brain_mask = self._run_totalsegmentator(volume_path, output_dir)
            
            # Apply dilation as specified in paper (3.6mm)
            dilated_mask = create_brain_mask_with_dilation(
                brain_mask, 
                dilation_mm=self.dilation_mm,
                voxel_spacing=self.voxel_spacing
            )
            
            # Include CVS region if provided
            if cvs_bounding_box is not None:
                dilated_mask = self._include_cvs_region(dilated_mask, cvs_bounding_box)
            
            # Validate output
            self.validator.validate_mask_format(dilated_mask, "brain_mask")
            
            logger.info(f"Generated brain mask with shape {dilated_mask.shape}")
            return dilated_mask
            
        except Exception as e:
            logger.error(f"Error in brain segmentation: {str(e)}")
            raise
    
    def _run_totalsegmentator(self, 
                            input_path: str, 
                            output_dir: Optional[str] = None) -> np.ndarray:
        """
        Run TotalSegmentator to generate brain mask.
        
        Args:
            input_path: Path to input NIfTI file
            output_dir: Output directory for segmentation results
            
        Returns:
            Brain mask as numpy array
        """
        try:
            # Create temporary output directory if not provided
            if output_dir is None:
                temp_dir = tempfile.mkdtemp()
                output_dir = temp_dir
            else:
                os.makedirs(output_dir, exist_ok=True)
            
            # Run TotalSegmentator command
            cmd = [
                self.totalsegmentator_path,
                "-i", input_path,
                "-o", output_dir,
                "--roi_subset", "brain"  # Only segment brain region
            ]
            
            logger.info(f"Running TotalSegmentator: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.warning(f"TotalSegmentator failed: {result.stderr}")
                # Fallback to simple brain mask generation
                return self._generate_fallback_brain_mask(input_path)
            
            # Load brain mask from TotalSegmentator output
            brain_mask_path = os.path.join(output_dir, "brain.nii.gz")
            if os.path.exists(brain_mask_path):
                brain_mask = self._load_volume(brain_mask_path)
            else:
                logger.warning("Brain mask not found in TotalSegmentator output, using fallback")
                brain_mask = self._generate_fallback_brain_mask(input_path)
            
            return brain_mask.astype(np.uint8)
            
        except subprocess.TimeoutExpired:
            logger.warning("TotalSegmentator timed out, using fallback method")
            return self._generate_fallback_brain_mask(input_path)
        except Exception as e:
            logger.warning(f"TotalSegmentator error: {str(e)}, using fallback method")
            return self._generate_fallback_brain_mask(input_path)
    
    def _generate_fallback_brain_mask(self, input_path: str) -> np.ndarray:
        """
        Generate a simple brain mask as fallback when TotalSegmentator fails.
        
        Args:
            input_path: Path to input volume
            
        Returns:
            Simple brain mask based on intensity thresholding
        """
        logger.info("Generating fallback brain mask using intensity thresholding")
        
        # Load volume
        volume = self._load_volume(input_path)
        
        # Simple brain extraction using intensity thresholding
        # This is a basic fallback - in practice, more sophisticated methods would be used
        
        # Calculate intensity statistics
        mean_intensity = np.mean(volume[volume > 0])
        std_intensity = np.std(volume[volume > 0])
        
        # Create mask based on intensity range
        lower_threshold = mean_intensity - 2 * std_intensity
        upper_threshold = mean_intensity + 3 * std_intensity
        
        brain_mask = ((volume >= lower_threshold) & (volume <= upper_threshold)).astype(np.uint8)
        
        # Clean up mask using morphological operations
        brain_mask = self.mask_processor.clean_mask(brain_mask)
        brain_mask = self.mask_processor.fill_holes(brain_mask)
        
        logger.info(f"Generated fallback brain mask with {np.sum(brain_mask)} voxels")
        return brain_mask
    
    def _include_cvs_region(self, 
                          brain_mask: np.ndarray, 
                          cvs_bounding_box: np.ndarray) -> np.ndarray:
        """
        Include CVS region bounding box in brain mask for skull base coverage.
        
        Args:
            brain_mask: Current brain mask
            cvs_bounding_box: CVS region bounding box [x_min, y_min, z_min, x_max, y_max, z_max]
            
        Returns:
            Brain mask with CVS region included
        """
        try:
            # Create CVS region mask
            cvs_mask = np.zeros_like(brain_mask)
            x_min, y_min, z_min, x_max, y_max, z_max = cvs_bounding_box.astype(int)
            
            # Ensure bounds are within volume
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            z_min = max(0, z_min)
            x_max = min(brain_mask.shape[0], x_max)
            y_max = min(brain_mask.shape[1], y_max)
            z_max = min(brain_mask.shape[2], z_max)
            
            cvs_mask[x_min:x_max, y_min:y_max, z_min:z_max] = 1
            
            # Union with brain mask
            combined_mask = self.mask_processor.union_masks(brain_mask, cvs_mask)
            
            logger.info("Included CVS region in brain mask")
            return combined_mask
            
        except Exception as e:
            logger.warning(f"Failed to include CVS region: {str(e)}")
            return brain_mask
    
    def _load_volume(self, volume_path: str) -> np.ndarray:
        """Load medical volume from file."""
        try:
            nii = nib.load(volume_path)
            volume = nii.get_fdata()
            return volume.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to load volume from {volume_path}: {str(e)}")
            raise
    
    def _save_temp_volume(self, 
                         volume: np.ndarray, 
                         output_dir: Optional[str] = None) -> str:
        """Save volume to temporary NIfTI file."""
        try:
            if output_dir is None:
                temp_dir = tempfile.mkdtemp()
            else:
                temp_dir = output_dir
                os.makedirs(temp_dir, exist_ok=True)
            
            temp_path = os.path.join(temp_dir, "temp_volume.nii.gz")
            
            # Create NIfTI image with proper spacing
            affine = np.eye(4)
            affine[0, 0] = self.voxel_spacing[0]
            affine[1, 1] = self.voxel_spacing[1]
            affine[2, 2] = self.voxel_spacing[2]
            
            nii = nib.Nifti1Image(volume, affine)
            nib.save(nii, temp_path)
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to save temporary volume: {str(e)}")
            raise


class BrainMaskGenerator:
    """
    Simplified interface for brain mask generation.
    
    Provides a streamlined API for generating brain masks with the
    paper-specified parameters.
    """
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)):
        """Initialize brain mask generator."""
        self.segmentator = BrainSegmentator(voxel_spacing=voxel_spacing)
    
    def generate_brain_mask(self, 
                          cta_volume: Union[np.ndarray, str, Path],
                          cvs_bounding_box: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate brain mask with paper specifications.
        
        Args:
            cta_volume: CTA volume or path to volume file
            cvs_bounding_box: Optional CVS region for skull base inclusion
            
        Returns:
            Dilated brain mask (3.6mm dilation)
        """
        return self.segmentator.segment_brain(cta_volume, cvs_bounding_box)


def create_sample_brain_mask(volume_shape: Tuple[int, int, int] = (512, 512, 200)) -> np.ndarray:
    """
    Create a sample brain mask for testing purposes.
    
    Args:
        volume_shape: Shape of the volume
        
    Returns:
        Sample brain mask
    """
    # Create ellipsoidal brain mask
    center = np.array(volume_shape) // 2
    
    # Create coordinate grids
    z, y, x = np.ogrid[:volume_shape[0], :volume_shape[1], :volume_shape[2]]
    
    # Define ellipsoid parameters (brain-like shape)
    a, b, c = volume_shape[0] // 3, volume_shape[1] // 3, volume_shape[2] // 2.5
    
    # Create ellipsoidal mask
    mask = ((x - center[2])**2 / a**2 + 
            (y - center[1])**2 / b**2 + 
            (z - center[0])**2 / c**2) <= 1
    
    # Apply dilation to match paper specifications
    mask_processor = MaskProcessor()
    dilated_mask = create_brain_mask_with_dilation(
        mask.astype(np.uint8), 
        dilation_mm=3.6,
        voxel_spacing=(0.4, 0.4, 0.4)
    )
    
    logger.info(f"Created sample brain mask with shape {dilated_mask.shape}")
    return dilated_mask


# Main interface functions
def segment_brain_from_cta(cta_volume: Union[np.ndarray, str, Path],
                          voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4),
                          cvs_bounding_box: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Main function to segment brain from CTA volume.
    
    Args:
        cta_volume: CTA volume or path to volume file
        voxel_spacing: Voxel spacing in mm
        cvs_bounding_box: Optional CVS region for skull base inclusion
        
    Returns:
        Dilated brain mask with paper specifications
    """
    generator = BrainMaskGenerator(voxel_spacing=voxel_spacing)
    return generator.generate_brain_mask(cta_volume, cvs_bounding_box)


if __name__ == "__main__":
    # Test brain segmentation
    print("Testing Brain Segmentation Module...")
    
    # Create sample volume
    sample_volume = np.random.randn(256, 256, 100) * 100 + 500
    sample_volume = np.clip(sample_volume, 0, 1000).astype(np.float32)
    
    # Test brain mask generation
    try:
        brain_mask = segment_brain_from_cta(sample_volume)
        print(f"✅ Brain mask generated successfully: {brain_mask.shape}")
        print(f"   Mask volume: {np.sum(brain_mask)} voxels")
        print(f"   Mask percentage: {np.sum(brain_mask) / brain_mask.size * 100:.2f}%")
        
        # Test sample brain mask
        sample_mask = create_sample_brain_mask()
        print(f"✅ Sample brain mask created: {sample_mask.shape}")
        
    except Exception as e:
        print(f"❌ Error in brain segmentation: {str(e)}")
        raise