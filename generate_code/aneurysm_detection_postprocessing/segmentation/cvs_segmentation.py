"""
CVS (Cavernous Venous Sinus) Segmentation Module

This module implements CVS mask generation using ANT (Advanced Normalize Tool) registration
as described in the paper for anatomical post-processing of aneurysm detection results.

Key Features:
- ANT-based affine registration of CVS atlas template to target CTA
- 3.2mm expansion of CVS region as specified in paper
- Integration with venous mask for final CVS mask creation
- Fallback methods for robust CVS region detection

Paper Reference:
"Automated anatomy-based post-processing reduces false positives and improved 
interpretability of deep learning intracranial aneurysm detection"
"""

import numpy as np
import nibabel as nib
import subprocess
import tempfile
import os
import logging
from typing import Tuple, Optional, Dict, Any
import torch
from pathlib import Path

from .mask_utils import MaskProcessor, create_cvs_mask_with_expansion, MaskValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CVSSegmentator:
    """
    Main CVS segmentation engine using ANT registration framework.
    
    Implements the paper's method:
    1. Perform affine registration of template to target CTA using ANT
    2. Transform CVS region annotation to target space
    3. Expand CVS region box by 3.2mm (8 pixels) in all directions
    4. Create final mask: cvs_mask = expanded_box ∩ venous_mask
    """
    
    def __init__(self, 
                 voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4),
                 expansion_mm: float = 3.2,
                 device: str = 'cpu',
                 ant_path: Optional[str] = None,
                 cvs_atlas_path: Optional[str] = None):
        """
        Initialize CVS segmentator.
        
        Args:
            voxel_spacing: Voxel spacing in mm (x, y, z)
            expansion_mm: CVS region expansion in mm (paper specifies 3.2mm)
            device: Computing device ('cpu' or 'cuda')
            ant_path: Path to ANT installation
            cvs_atlas_path: Path to CVS atlas template
        """
        self.voxel_spacing = voxel_spacing
        self.expansion_mm = expansion_mm
        self.device = device
        self.ant_path = ant_path or self._find_ant_path()
        self.cvs_atlas_path = cvs_atlas_path
        
        # Initialize mask processor
        self.mask_processor = MaskProcessor(voxel_spacing=voxel_spacing)
        self.validator = MaskValidator()
        
        logger.info(f"CVS Segmentator initialized with expansion: {expansion_mm}mm")
    
    def _find_ant_path(self) -> Optional[str]:
        """Find ANT installation path."""
        try:
            result = subprocess.run(['which', 'antsRegistration'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                ant_path = os.path.dirname(result.stdout.strip())
                logger.info(f"Found ANT at: {ant_path}")
                return ant_path
        except Exception as e:
            logger.warning(f"Could not find ANT installation: {e}")
        return None
    
    def segment_cvs(self, 
                    cta_volume: np.ndarray,
                    venous_mask: Optional[np.ndarray] = None,
                    cvs_atlas: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Segment CVS region using ANT registration.
        
        Args:
            cta_volume: Input CTA volume (H, W, D)
            venous_mask: Venous segmentation mask (optional)
            cvs_atlas: CVS atlas template (optional)
            
        Returns:
            cvs_mask: CVS segmentation mask
        """
        try:
            logger.info("Starting CVS segmentation with ANT registration")
            
            # Validate inputs
            if not self.validator.validate_mask_format(cta_volume):
                raise ValueError("Invalid CTA volume format")
            
            # Use ANT registration if available
            if self.ant_path and cvs_atlas is not None:
                cvs_mask = self._segment_with_ant_registration(
                    cta_volume, cvs_atlas, venous_mask
                )
            else:
                logger.warning("ANT not available, using fallback method")
                cvs_mask = self._segment_with_fallback_method(
                    cta_volume, venous_mask
                )
            
            # Validate output
            if not self.validator.validate_mask_format(cvs_mask):
                raise ValueError("Generated CVS mask has invalid format")
            
            logger.info(f"CVS segmentation completed. Mask shape: {cvs_mask.shape}")
            return cvs_mask
            
        except Exception as e:
            logger.error(f"CVS segmentation failed: {e}")
            # Return fallback mask
            return self._create_fallback_cvs_mask(cta_volume.shape)
    
    def _segment_with_ant_registration(self, 
                                     cta_volume: np.ndarray,
                                     cvs_atlas: np.ndarray,
                                     venous_mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Perform CVS segmentation using ANT registration.
        
        Args:
            cta_volume: Target CTA volume
            cvs_atlas: CVS atlas template
            venous_mask: Venous mask for intersection
            
        Returns:
            cvs_mask: Registered and expanded CVS mask
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            # Save volumes as NIfTI files
            target_path = temp_dir / "target.nii.gz"
            atlas_path = temp_dir / "atlas.nii.gz"
            
            # Create NIfTI images with proper affine
            affine = np.eye(4)
            affine[0, 0] = self.voxel_spacing[0]
            affine[1, 1] = self.voxel_spacing[1]
            affine[2, 2] = self.voxel_spacing[2]
            
            target_img = nib.Nifti1Image(cta_volume.astype(np.float32), affine)
            atlas_img = nib.Nifti1Image(cvs_atlas.astype(np.float32), affine)
            
            nib.save(target_img, target_path)
            nib.save(atlas_img, atlas_path)
            
            # Perform ANT registration
            transform_path = temp_dir / "transform"
            registered_path = temp_dir / "registered.nii.gz"
            
            registration_cmd = [
                os.path.join(self.ant_path, "antsRegistration"),
                "--dimensionality", "3",
                "--float", "0",
                "--output", f"[{transform_path},{registered_path}]",
                "--interpolation", "Linear",
                "--winsorize-image-intensities", "[0.005,0.995]",
                "--use-histogram-matching", "0",
                "--initial-moving-transform", f"[{target_path},{atlas_path},1]",
                "--transform", "Rigid[0.1]",
                "--metric", f"MI[{target_path},{atlas_path},1,32,Regular,0.25]",
                "--convergence", "[1000x500x250x100,1e-6,10]",
                "--shrink-factors", "8x4x2x1",
                "--smoothing-sigmas", "3x2x1x0vox",
                "--transform", "Affine[0.1]",
                "--metric", f"MI[{target_path},{atlas_path},1,32,Regular,0.25]",
                "--convergence", "[1000x500x250x100,1e-6,10]",
                "--shrink-factors", "8x4x2x1",
                "--smoothing-sigmas", "3x2x1x0vox"
            ]
            
            try:
                subprocess.run(registration_cmd, check=True, capture_output=True)
                
                # Load registered atlas
                registered_img = nib.load(registered_path)
                registered_atlas = registered_img.get_fdata()
                
                # Create CVS region mask from registered atlas
                cvs_region = (registered_atlas > 0.5).astype(np.uint8)
                
                # Apply expansion and intersection with venous mask
                cvs_mask = create_cvs_mask_with_expansion(
                    cvs_region, venous_mask, 
                    expansion_mm=self.expansion_mm,
                    voxel_spacing=self.voxel_spacing
                )
                
                return cvs_mask
                
            except subprocess.CalledProcessError as e:
                logger.error(f"ANT registration failed: {e}")
                return self._segment_with_fallback_method(cta_volume, venous_mask)
    
    def _segment_with_fallback_method(self, 
                                    cta_volume: np.ndarray,
                                    venous_mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Fallback CVS segmentation method using anatomical priors.
        
        Args:
            cta_volume: Input CTA volume
            venous_mask: Venous mask for guidance
            
        Returns:
            cvs_mask: Estimated CVS mask
        """
        logger.info("Using fallback CVS segmentation method")
        
        # Create CVS region based on anatomical location
        # CVS is typically located in the skull base region
        height, width, depth = cta_volume.shape
        
        # Define CVS region based on anatomical knowledge
        # CVS is located in the central skull base, lateral to sella turcica
        cvs_region = np.zeros_like(cta_volume, dtype=np.uint8)
        
        # Central region with bilateral CVS locations
        center_x, center_y = width // 2, height // 2
        cvs_size = int(20 / self.voxel_spacing[0])  # ~20mm region
        
        # Bilateral CVS regions
        for side_offset in [-cvs_size//2, cvs_size//2]:
            x_start = max(0, center_x + side_offset - cvs_size//4)
            x_end = min(width, center_x + side_offset + cvs_size//4)
            y_start = max(0, center_y - cvs_size//4)
            y_end = min(height, center_y + cvs_size//4)
            z_start = max(0, depth//4)  # Lower part of volume
            z_end = min(depth, 3*depth//4)
            
            cvs_region[y_start:y_end, x_start:x_end, z_start:z_end] = 1
        
        # Apply expansion and intersection with venous mask
        cvs_mask = create_cvs_mask_with_expansion(
            cvs_region, venous_mask,
            expansion_mm=self.expansion_mm,
            voxel_spacing=self.voxel_spacing
        )
        
        return cvs_mask
    
    def _create_fallback_cvs_mask(self, volume_shape: Tuple[int, int, int]) -> np.ndarray:
        """Create a basic fallback CVS mask."""
        logger.warning("Creating fallback CVS mask")
        
        height, width, depth = volume_shape
        cvs_mask = np.zeros(volume_shape, dtype=np.uint8)
        
        # Small central region as fallback
        center_x, center_y = width // 2, height // 2
        size = 10  # Small region
        
        x_start = max(0, center_x - size)
        x_end = min(width, center_x + size)
        y_start = max(0, center_y - size)
        y_end = min(height, center_y + size)
        z_start = max(0, depth//3)
        z_end = min(depth, 2*depth//3)
        
        cvs_mask[y_start:y_end, x_start:x_end, z_start:z_end] = 1
        
        return cvs_mask


class CVSMaskGenerator:
    """
    Simplified interface for CVS mask generation.
    
    Provides easy-to-use methods for generating CVS masks
    without requiring detailed knowledge of the registration process.
    """
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)):
        """
        Initialize CVS mask generator.
        
        Args:
            voxel_spacing: Voxel spacing in mm (x, y, z)
        """
        self.voxel_spacing = voxel_spacing
        self.segmentator = CVSSegmentator(voxel_spacing=voxel_spacing)
    
    def generate_cvs_mask(self, 
                         cta_volume: np.ndarray,
                         venous_mask: Optional[np.ndarray] = None,
                         use_ant: bool = True) -> np.ndarray:
        """
        Generate CVS mask from CTA volume.
        
        Args:
            cta_volume: Input CTA volume
            venous_mask: Optional venous mask for intersection
            use_ant: Whether to use ANT registration (if available)
            
        Returns:
            cvs_mask: CVS segmentation mask
        """
        if use_ant and self.segmentator.ant_path:
            return self.segmentator.segment_cvs(cta_volume, venous_mask)
        else:
            return self.segmentator._segment_with_fallback_method(cta_volume, venous_mask)


class CVSAtlasManager:
    """
    Manages CVS atlas templates for registration.
    
    Handles loading, preprocessing, and management of CVS atlas data
    used in the registration process.
    """
    
    def __init__(self, atlas_path: Optional[str] = None):
        """
        Initialize CVS atlas manager.
        
        Args:
            atlas_path: Path to CVS atlas directory
        """
        self.atlas_path = atlas_path
        self.loaded_atlases = {}
    
    def load_cvs_atlas(self, atlas_name: str = "default") -> Optional[np.ndarray]:
        """
        Load CVS atlas template.
        
        Args:
            atlas_name: Name of atlas to load
            
        Returns:
            cvs_atlas: CVS atlas volume or None if not available
        """
        if atlas_name in self.loaded_atlases:
            return self.loaded_atlases[atlas_name]
        
        if self.atlas_path and os.path.exists(self.atlas_path):
            try:
                atlas_file = os.path.join(self.atlas_path, f"{atlas_name}.nii.gz")
                if os.path.exists(atlas_file):
                    atlas_img = nib.load(atlas_file)
                    atlas_data = atlas_img.get_fdata()
                    self.loaded_atlases[atlas_name] = atlas_data
                    logger.info(f"Loaded CVS atlas: {atlas_name}")
                    return atlas_data
            except Exception as e:
                logger.error(f"Failed to load CVS atlas {atlas_name}: {e}")
        
        # Return synthetic atlas if no real atlas available
        logger.warning(f"CVS atlas {atlas_name} not found, creating synthetic atlas")
        return self._create_synthetic_atlas()
    
    def _create_synthetic_atlas(self, 
                              volume_shape: Tuple[int, int, int] = (256, 256, 128)) -> np.ndarray:
        """
        Create a synthetic CVS atlas for testing/fallback.
        
        Args:
            volume_shape: Shape of synthetic atlas
            
        Returns:
            synthetic_atlas: Synthetic CVS atlas
        """
        height, width, depth = volume_shape
        atlas = np.zeros(volume_shape, dtype=np.float32)
        
        # Create bilateral CVS regions
        center_x, center_y = width // 2, height // 2
        cvs_size = 15
        
        for side_offset in [-25, 25]:  # Bilateral locations
            x_center = center_x + side_offset
            y_center = center_y
            z_center = depth // 2
            
            # Create ellipsoidal CVS region
            for z in range(max(0, z_center - cvs_size), min(depth, z_center + cvs_size)):
                for y in range(max(0, y_center - cvs_size//2), min(height, y_center + cvs_size//2)):
                    for x in range(max(0, x_center - cvs_size//2), min(width, x_center + cvs_size//2)):
                        # Ellipsoidal distance
                        dx = (x - x_center) / (cvs_size//2)
                        dy = (y - y_center) / (cvs_size//2)
                        dz = (z - z_center) / cvs_size
                        
                        if dx*dx + dy*dy + dz*dz <= 1.0:
                            atlas[y, x, z] = 1.0
        
        logger.info(f"Created synthetic CVS atlas with shape: {volume_shape}")
        return atlas


# Main functions for external use
def segment_cvs_from_cta(cta_volume: np.ndarray,
                        voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4),
                        venous_mask: Optional[np.ndarray] = None,
                        cvs_atlas: Optional[np.ndarray] = None,
                        use_ant: bool = True) -> np.ndarray:
    """
    Main function to segment CVS from CTA volume.
    
    This is the primary interface for CVS segmentation as described in the paper:
    1. Perform affine registration of template to target CTA using ANT
    2. Transform CVS region annotation to target space  
    3. Expand CVS region box by 3.2mm in all directions
    4. Create final mask: cvs_mask = expanded_box ∩ venous_mask
    
    Args:
        cta_volume: Input CTA volume (H, W, D)
        voxel_spacing: Voxel spacing in mm (x, y, z)
        venous_mask: Optional venous mask for intersection
        cvs_atlas: Optional CVS atlas template
        use_ant: Whether to use ANT registration
        
    Returns:
        cvs_mask: CVS segmentation mask
    """
    segmentator = CVSSegmentator(voxel_spacing=voxel_spacing)
    return segmentator.segment_cvs(cta_volume, venous_mask, cvs_atlas)


def create_sample_cvs_mask(volume_shape: Tuple[int, int, int] = (512, 512, 200)) -> np.ndarray:
    """
    Create a sample CVS mask for testing purposes.
    
    Args:
        volume_shape: Shape of the volume (H, W, D)
        
    Returns:
        sample_cvs_mask: Sample CVS mask
    """
    height, width, depth = volume_shape
    cvs_mask = np.zeros(volume_shape, dtype=np.uint8)
    
    # Create bilateral CVS regions in skull base
    center_x, center_y = width // 2, height // 2
    cvs_size = 20
    
    # Bilateral CVS locations
    for side_offset in [-30, 30]:
        x_center = center_x + side_offset
        y_center = center_y + 10  # Slightly posterior
        z_start = depth // 4
        z_end = 3 * depth // 4
        
        # Create CVS region
        x_start = max(0, x_center - cvs_size//2)
        x_end = min(width, x_center + cvs_size//2)
        y_start = max(0, y_center - cvs_size//3)
        y_end = min(height, y_center + cvs_size//3)
        
        cvs_mask[y_start:y_end, x_start:x_end, z_start:z_end] = 1
    
    logger.info(f"Created sample CVS mask with shape: {volume_shape}")
    return cvs_mask


if __name__ == "__main__":
    # Test CVS segmentation
    print("Testing CVS Segmentation Module")
    
    # Create test data
    test_volume = np.random.rand(256, 256, 128).astype(np.float32)
    test_venous_mask = np.random.randint(0, 2, (256, 256, 128)).astype(np.uint8)
    
    # Test CVS segmentation
    cvs_mask = segment_cvs_from_cta(
        test_volume, 
        voxel_spacing=(0.4, 0.4, 0.4),
        venous_mask=test_venous_mask,
        use_ant=False  # Use fallback method for testing
    )
    
    print(f"CVS mask shape: {cvs_mask.shape}")
    print(f"CVS mask dtype: {cvs_mask.dtype}")
    print(f"CVS mask range: [{cvs_mask.min()}, {cvs_mask.max()}]")
    print(f"CVS mask volume: {np.sum(cvs_mask)} voxels")
    
    # Test sample mask creation
    sample_mask = create_sample_cvs_mask((512, 512, 200))
    print(f"Sample CVS mask shape: {sample_mask.shape}")
    print(f"Sample CVS mask volume: {np.sum(sample_mask)} voxels")
    
    print("CVS Segmentation Module test completed successfully!")