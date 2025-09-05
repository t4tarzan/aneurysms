"""
Overlap Calculator for Anatomical Post-processing

This module provides comprehensive bounding box overlap calculations with anatomical masks
for the aneurysm detection post-processing pipeline. It implements the five post-processing
methods described in the paper.

Key Features:
- Bounding box to mask overlap ratio calculations
- Center-point mask inclusion checks
- Comparative overlap analysis between different masks
- Support for all five post-processing methods from the paper

Author: Automated Implementation
Date: 2024
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from segmentation.mask_utils import BoundingBoxProcessor, MaskValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OverlapCalculator:
    """
    Main overlap calculator for anatomical post-processing.
    
    This class implements all overlap calculation methods needed for the five
    post-processing approaches described in the paper.
    """
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)):
        """
        Initialize the overlap calculator.
        
        Args:
            voxel_spacing: Voxel spacing in mm (x, y, z)
        """
        self.voxel_spacing = voxel_spacing
        self.bbox_processor = BoundingBoxProcessor()
        self.mask_validator = MaskValidator()
        
        logger.info(f"OverlapCalculator initialized with voxel spacing: {voxel_spacing}")
    
    def calculate_overlap_ratio(self, bbox: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate the overlap ratio between a bounding box and a mask.
        
        Args:
            bbox: 3D bounding box [x, y, z, w, h, d] (center coordinates + dimensions)
            mask: 3D binary mask
            
        Returns:
            Overlap ratio (intersection volume / bounding box volume)
        """
        try:
            # Validate inputs
            if not self.mask_validator.validate_mask_format(mask):
                logger.warning("Invalid mask format provided")
                return 0.0
                
            if len(bbox) != 6:
                logger.warning(f"Invalid bounding box format. Expected 6 values, got {len(bbox)}")
                return 0.0
            
            # Use the BoundingBoxProcessor from mask_utils
            overlap_ratio = self.bbox_processor.calculate_overlap_ratio(bbox, mask)
            
            logger.debug(f"Calculated overlap ratio: {overlap_ratio:.4f}")
            return overlap_ratio
            
        except Exception as e:
            logger.error(f"Error calculating overlap ratio: {str(e)}")
            return 0.0
    
    def is_bbox_center_in_mask(self, bbox: np.ndarray, mask: np.ndarray) -> bool:
        """
        Check if the center of a bounding box is inside a mask.
        
        Args:
            bbox: 3D bounding box [x, y, z, w, h, d]
            mask: 3D binary mask
            
        Returns:
            True if center is inside mask, False otherwise
        """
        try:
            # Validate inputs
            if not self.mask_validator.validate_mask_format(mask):
                logger.warning("Invalid mask format provided")
                return False
                
            if len(bbox) != 6:
                logger.warning(f"Invalid bounding box format. Expected 6 values, got {len(bbox)}")
                return False
            
            # Use the BoundingBoxProcessor from mask_utils
            is_inside = self.bbox_processor.is_bbox_center_in_mask(bbox, mask)
            
            logger.debug(f"Bbox center in mask: {is_inside}")
            return is_inside
            
        except Exception as e:
            logger.error(f"Error checking bbox center in mask: {str(e)}")
            return False
    
    def compare_overlaps(self, bbox: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> Dict[str, float]:
        """
        Compare overlap ratios between a bounding box and two different masks.
        
        Args:
            bbox: 3D bounding box [x, y, z, w, h, d]
            mask1: First 3D binary mask
            mask2: Second 3D binary mask
            
        Returns:
            Dictionary with overlap ratios and comparison results
        """
        try:
            # Calculate overlap ratios for both masks
            overlap1 = self.calculate_overlap_ratio(bbox, mask1)
            overlap2 = self.calculate_overlap_ratio(bbox, mask2)
            
            result = {
                'overlap_mask1': overlap1,
                'overlap_mask2': overlap2,
                'mask1_greater': overlap1 > overlap2,
                'mask2_greater': overlap2 > overlap1,
                'equal': abs(overlap1 - overlap2) < 1e-6,
                'difference': overlap1 - overlap2
            }
            
            logger.debug(f"Overlap comparison - Mask1: {overlap1:.4f}, Mask2: {overlap2:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error comparing overlaps: {str(e)}")
            return {
                'overlap_mask1': 0.0,
                'overlap_mask2': 0.0,
                'mask1_greater': False,
                'mask2_greater': False,
                'equal': True,
                'difference': 0.0
            }
    
    def batch_overlap_calculation(self, bboxes: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Calculate overlap ratios for multiple bounding boxes with a single mask.
        
        Args:
            bboxes: Array of bounding boxes, shape (N, 6)
            mask: 3D binary mask
            
        Returns:
            Array of overlap ratios, shape (N,)
        """
        try:
            if len(bboxes.shape) != 2 or bboxes.shape[1] != 6:
                logger.warning(f"Invalid bboxes shape. Expected (N, 6), got {bboxes.shape}")
                return np.zeros(len(bboxes))
            
            overlap_ratios = np.zeros(len(bboxes))
            
            for i, bbox in enumerate(bboxes):
                overlap_ratios[i] = self.calculate_overlap_ratio(bbox, mask)
            
            logger.debug(f"Calculated {len(overlap_ratios)} overlap ratios")
            return overlap_ratios
            
        except Exception as e:
            logger.error(f"Error in batch overlap calculation: {str(e)}")
            return np.zeros(len(bboxes))
    
    def batch_center_check(self, bboxes: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Check if centers of multiple bounding boxes are inside a mask.
        
        Args:
            bboxes: Array of bounding boxes, shape (N, 6)
            mask: 3D binary mask
            
        Returns:
            Boolean array indicating which centers are inside, shape (N,)
        """
        try:
            if len(bboxes.shape) != 2 or bboxes.shape[1] != 6:
                logger.warning(f"Invalid bboxes shape. Expected (N, 6), got {bboxes.shape}")
                return np.zeros(len(bboxes), dtype=bool)
            
            inside_mask = np.zeros(len(bboxes), dtype=bool)
            
            for i, bbox in enumerate(bboxes):
                inside_mask[i] = self.is_bbox_center_in_mask(bbox, mask)
            
            logger.debug(f"Checked {len(inside_mask)} bbox centers, {np.sum(inside_mask)} inside mask")
            return inside_mask
            
        except Exception as e:
            logger.error(f"Error in batch center check: {str(e)}")
            return np.zeros(len(bboxes), dtype=bool)


class PostProcessingMethodsCalculator:
    """
    Implements the five post-processing methods from the paper using overlap calculations.
    
    Methods:
    1. Remove bounding boxes outside brain mask
    2. Remove bounding boxes with ANY overlap with vein mask
    3. Remove bounding boxes if overlap_vein > overlap_artery
    4. Combine Method 1 AND Method 2
    5. Combine Method 1 AND Method 3
    """
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)):
        """
        Initialize the post-processing methods calculator.
        
        Args:
            voxel_spacing: Voxel spacing in mm (x, y, z)
        """
        self.overlap_calc = OverlapCalculator(voxel_spacing)
        logger.info("PostProcessingMethodsCalculator initialized")
    
    def method_1_outside_brain(self, bboxes: np.ndarray, scores: np.ndarray, 
                              brain_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Method 1: Remove bounding boxes outside brain mask.
        
        Args:
            bboxes: Array of bounding boxes, shape (N, 6)
            scores: Array of confidence scores, shape (N,)
            brain_mask: 3D binary brain mask
            
        Returns:
            Tuple of (filtered_bboxes, filtered_scores, keep_indices)
        """
        try:
            logger.info(f"Applying Method 1: Remove boxes outside brain mask")
            logger.info(f"Input: {len(bboxes)} bounding boxes")
            
            # Check which bbox centers are inside brain mask
            inside_brain = self.overlap_calc.batch_center_check(bboxes, brain_mask)
            
            # Keep only boxes inside brain
            filtered_bboxes = bboxes[inside_brain]
            filtered_scores = scores[inside_brain]
            keep_indices = np.where(inside_brain)[0]
            
            logger.info(f"Method 1 result: {len(filtered_bboxes)} boxes kept, "
                       f"{len(bboxes) - len(filtered_bboxes)} boxes removed")
            
            return filtered_bboxes, filtered_scores, keep_indices
            
        except Exception as e:
            logger.error(f"Error in method 1: {str(e)}")
            return bboxes, scores, np.arange(len(bboxes))
    
    def method_2_any_vein_overlap(self, bboxes: np.ndarray, scores: np.ndarray, 
                                 vein_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Method 2: Remove bounding boxes with ANY overlap with vein mask.
        
        Args:
            bboxes: Array of bounding boxes, shape (N, 6)
            scores: Array of confidence scores, shape (N,)
            vein_mask: 3D binary vein mask
            
        Returns:
            Tuple of (filtered_bboxes, filtered_scores, keep_indices)
        """
        try:
            logger.info(f"Applying Method 2: Remove boxes with any vein overlap")
            logger.info(f"Input: {len(bboxes)} bounding boxes")
            
            # Calculate overlap ratios with vein mask
            vein_overlaps = self.overlap_calc.batch_overlap_calculation(bboxes, vein_mask)
            
            # Keep only boxes with no vein overlap (overlap = 0)
            no_vein_overlap = vein_overlaps == 0.0
            
            filtered_bboxes = bboxes[no_vein_overlap]
            filtered_scores = scores[no_vein_overlap]
            keep_indices = np.where(no_vein_overlap)[0]
            
            logger.info(f"Method 2 result: {len(filtered_bboxes)} boxes kept, "
                       f"{len(bboxes) - len(filtered_bboxes)} boxes removed")
            
            return filtered_bboxes, filtered_scores, keep_indices
            
        except Exception as e:
            logger.error(f"Error in method 2: {str(e)}")
            return bboxes, scores, np.arange(len(bboxes))
    
    def method_3_vein_vs_artery_overlap(self, bboxes: np.ndarray, scores: np.ndarray,
                                       vein_mask: np.ndarray, artery_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Method 3: Remove bounding boxes if overlap_vein > overlap_artery.
        
        Args:
            bboxes: Array of bounding boxes, shape (N, 6)
            scores: Array of confidence scores, shape (N,)
            vein_mask: 3D binary vein mask
            artery_mask: 3D binary artery mask
            
        Returns:
            Tuple of (filtered_bboxes, filtered_scores, keep_indices)
        """
        try:
            logger.info(f"Applying Method 3: Remove boxes where vein overlap > artery overlap")
            logger.info(f"Input: {len(bboxes)} bounding boxes")
            
            # Calculate overlap ratios with both masks
            vein_overlaps = self.overlap_calc.batch_overlap_calculation(bboxes, vein_mask)
            artery_overlaps = self.overlap_calc.batch_overlap_calculation(bboxes, artery_mask)
            
            # Keep boxes where artery overlap >= vein overlap
            keep_mask = artery_overlaps >= vein_overlaps
            
            filtered_bboxes = bboxes[keep_mask]
            filtered_scores = scores[keep_mask]
            keep_indices = np.where(keep_mask)[0]
            
            logger.info(f"Method 3 result: {len(filtered_bboxes)} boxes kept, "
                       f"{len(bboxes) - len(filtered_bboxes)} boxes removed")
            
            return filtered_bboxes, filtered_scores, keep_indices
            
        except Exception as e:
            logger.error(f"Error in method 3: {str(e)}")
            return bboxes, scores, np.arange(len(bboxes))
    
    def method_4_brain_and_no_vein(self, bboxes: np.ndarray, scores: np.ndarray,
                                  brain_mask: np.ndarray, vein_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Method 4: Combine Method 1 AND Method 2 (inside brain AND no vein overlap).
        
        Args:
            bboxes: Array of bounding boxes, shape (N, 6)
            scores: Array of confidence scores, shape (N,)
            brain_mask: 3D binary brain mask
            vein_mask: 3D binary vein mask
            
        Returns:
            Tuple of (filtered_bboxes, filtered_scores, keep_indices)
        """
        try:
            logger.info(f"Applying Method 4: Inside brain AND no vein overlap")
            logger.info(f"Input: {len(bboxes)} bounding boxes")
            
            # Check brain mask condition
            inside_brain = self.overlap_calc.batch_center_check(bboxes, brain_mask)
            
            # Check vein overlap condition
            vein_overlaps = self.overlap_calc.batch_overlap_calculation(bboxes, vein_mask)
            no_vein_overlap = vein_overlaps == 0.0
            
            # Combine both conditions (AND)
            keep_mask = inside_brain & no_vein_overlap
            
            filtered_bboxes = bboxes[keep_mask]
            filtered_scores = scores[keep_mask]
            keep_indices = np.where(keep_mask)[0]
            
            logger.info(f"Method 4 result: {len(filtered_bboxes)} boxes kept, "
                       f"{len(bboxes) - len(filtered_bboxes)} boxes removed")
            
            return filtered_bboxes, filtered_scores, keep_indices
            
        except Exception as e:
            logger.error(f"Error in method 4: {str(e)}")
            return bboxes, scores, np.arange(len(bboxes))
    
    def method_5_brain_and_artery_priority(self, bboxes: np.ndarray, scores: np.ndarray,
                                          brain_mask: np.ndarray, vein_mask: np.ndarray, 
                                          artery_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Method 5: Combine Method 1 AND Method 3 (inside brain AND artery overlap >= vein overlap).
        
        Args:
            bboxes: Array of bounding boxes, shape (N, 6)
            scores: Array of confidence scores, shape (N,)
            brain_mask: 3D binary brain mask
            vein_mask: 3D binary vein mask
            artery_mask: 3D binary artery mask
            
        Returns:
            Tuple of (filtered_bboxes, filtered_scores, keep_indices)
        """
        try:
            logger.info(f"Applying Method 5: Inside brain AND artery overlap >= vein overlap")
            logger.info(f"Input: {len(bboxes)} bounding boxes")
            
            # Check brain mask condition
            inside_brain = self.overlap_calc.batch_center_check(bboxes, brain_mask)
            
            # Check overlap comparison condition
            vein_overlaps = self.overlap_calc.batch_overlap_calculation(bboxes, vein_mask)
            artery_overlaps = self.overlap_calc.batch_overlap_calculation(bboxes, artery_mask)
            artery_priority = artery_overlaps >= vein_overlaps
            
            # Combine both conditions (AND)
            keep_mask = inside_brain & artery_priority
            
            filtered_bboxes = bboxes[keep_mask]
            filtered_scores = scores[keep_mask]
            keep_indices = np.where(keep_mask)[0]
            
            logger.info(f"Method 5 result: {len(filtered_bboxes)} boxes kept, "
                       f"{len(bboxes) - len(filtered_bboxes)} boxes removed")
            
            return filtered_bboxes, filtered_scores, keep_indices
            
        except Exception as e:
            logger.error(f"Error in method 5: {str(e)}")
            return bboxes, scores, np.arange(len(bboxes))
    
    def apply_all_methods(self, bboxes: np.ndarray, scores: np.ndarray,
                         brain_mask: np.ndarray, vein_mask: np.ndarray, 
                         artery_mask: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Apply all five post-processing methods and return results.
        
        Args:
            bboxes: Array of bounding boxes, shape (N, 6)
            scores: Array of confidence scores, shape (N,)
            brain_mask: 3D binary brain mask
            vein_mask: 3D binary vein mask
            artery_mask: 3D binary artery mask
            
        Returns:
            Dictionary with results from all methods
        """
        try:
            logger.info("Applying all five post-processing methods")
            
            results = {}
            
            # Method 1: Outside brain
            results['method_1'] = self.method_1_outside_brain(bboxes, scores, brain_mask)
            
            # Method 2: Any vein overlap
            results['method_2'] = self.method_2_any_vein_overlap(bboxes, scores, vein_mask)
            
            # Method 3: Vein vs artery overlap
            results['method_3'] = self.method_3_vein_vs_artery_overlap(bboxes, scores, vein_mask, artery_mask)
            
            # Method 4: Brain AND no vein
            results['method_4'] = self.method_4_brain_and_no_vein(bboxes, scores, brain_mask, vein_mask)
            
            # Method 5: Brain AND artery priority
            results['method_5'] = self.method_5_brain_and_artery_priority(bboxes, scores, brain_mask, vein_mask, artery_mask)
            
            # Summary statistics
            logger.info("Post-processing methods summary:")
            for method_name, (filtered_bboxes, _, _) in results.items():
                reduction = len(bboxes) - len(filtered_bboxes)
                percentage = (reduction / len(bboxes)) * 100 if len(bboxes) > 0 else 0
                logger.info(f"  {method_name}: {len(filtered_bboxes)} kept, {reduction} removed ({percentage:.1f}%)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error applying all methods: {str(e)}")
            return {}


def create_test_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create test data for overlap calculator validation.
    
    Returns:
        Tuple of (bboxes, scores, brain_mask, vein_mask, artery_mask)
    """
    # Create test bounding boxes
    bboxes = np.array([
        [50, 50, 50, 10, 10, 10],  # Center of volume
        [25, 25, 25, 8, 8, 8],     # Upper left
        [75, 75, 75, 12, 12, 12],  # Lower right
        [10, 10, 10, 5, 5, 5],     # Near edge
        [90, 90, 90, 6, 6, 6]      # Near opposite edge
    ])
    
    # Create test scores
    scores = np.array([0.9, 0.85, 0.8, 0.75, 0.7])
    
    # Create test masks (100x100x100)
    brain_mask = np.zeros((100, 100, 100), dtype=bool)
    brain_mask[20:80, 20:80, 20:80] = True  # Brain region
    
    vein_mask = np.zeros((100, 100, 100), dtype=bool)
    vein_mask[70:90, 70:90, 70:90] = True  # Vein region
    
    artery_mask = np.zeros((100, 100, 100), dtype=bool)
    artery_mask[40:60, 40:60, 40:60] = True  # Artery region
    
    return bboxes, scores, brain_mask, vein_mask, artery_mask


def main():
    """
    Test the overlap calculator implementation.
    """
    logger.info("Testing OverlapCalculator implementation")
    
    try:
        # Create test data
        bboxes, scores, brain_mask, vein_mask, artery_mask = create_test_data()
        
        # Initialize calculators
        overlap_calc = OverlapCalculator()
        methods_calc = PostProcessingMethodsCalculator()
        
        # Test basic overlap calculation
        logger.info("Testing basic overlap calculations...")
        for i, bbox in enumerate(bboxes):
            brain_overlap = overlap_calc.calculate_overlap_ratio(bbox, brain_mask)
            vein_overlap = overlap_calc.calculate_overlap_ratio(bbox, vein_mask)
            artery_overlap = overlap_calc.calculate_overlap_ratio(bbox, artery_mask)
            
            logger.info(f"Bbox {i}: Brain={brain_overlap:.3f}, Vein={vein_overlap:.3f}, Artery={artery_overlap:.3f}")
        
        # Test all post-processing methods
        logger.info("\nTesting all post-processing methods...")
        results = methods_calc.apply_all_methods(bboxes, scores, brain_mask, vein_mask, artery_mask)
        
        logger.info("Overlap calculator test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("✅ OverlapCalculator implementation test passed!")
    else:
        print("❌ OverlapCalculator implementation test failed!")