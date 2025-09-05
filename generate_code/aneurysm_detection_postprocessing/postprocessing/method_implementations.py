"""
Post-processing Method Implementations for Aneurysm Detection

This module provides clean interfaces for the five anatomical post-processing methods
described in the paper. It serves as a high-level wrapper around the overlap calculator
to provide method-specific implementations with clear documentation.

Methods implemented:
1. Remove bounding boxes outside brain mask
2. Remove bounding boxes with ANY overlap with vein mask  
3. Remove bounding boxes if overlap_vein > overlap_artery
4. Combine Method 1 AND Method 2
5. Combine Method 1 AND Method 3

Author: Implementation based on paper requirements
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

from .overlap_calculator import PostProcessingMethodsCalculator, OverlapCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnatomicalPostProcessor:
    """
    High-level interface for anatomical post-processing methods.
    
    This class provides a clean API for applying the five post-processing methods
    described in the paper, with detailed logging and validation.
    """
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)):
        """
        Initialize the anatomical post-processor.
        
        Args:
            voxel_spacing: Voxel spacing in mm (z, y, x)
        """
        self.voxel_spacing = voxel_spacing
        self.calculator = PostProcessingMethodsCalculator(voxel_spacing=voxel_spacing)
        self.overlap_calc = OverlapCalculator(voxel_spacing=voxel_spacing)
        
        logger.info(f"Initialized AnatomicalPostProcessor with voxel spacing: {voxel_spacing}")
    
    def method_1_brain_filter(self, 
                             bboxes: np.ndarray, 
                             scores: np.ndarray, 
                             brain_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Method 1: Remove bounding boxes outside brain mask.
        
        Algorithm: if bbox_center not in brain_mask: remove_bbox
        
        Args:
            bboxes: Array of bounding boxes (N, 6) - [z_min, y_min, x_min, z_max, y_max, x_max]
            scores: Array of confidence scores (N,)
            brain_mask: 3D binary brain mask
            
        Returns:
            Tuple of (filtered_bboxes, filtered_scores, statistics)
        """
        logger.info(f"Applying Method 1: Brain filter on {len(bboxes)} detections")
        
        # Apply method 1
        filtered_bboxes, filtered_scores = self.calculator.method_1_outside_brain(
            bboxes, scores, brain_mask
        )
        
        # Calculate statistics
        original_count = len(bboxes)
        filtered_count = len(filtered_bboxes)
        removed_count = original_count - filtered_count
        
        stats = {
            'method': 'Method 1: Brain Filter',
            'original_detections': original_count,
            'filtered_detections': filtered_count,
            'removed_detections': removed_count,
            'removal_rate': removed_count / original_count if original_count > 0 else 0.0,
            'description': 'Remove detections outside dilated brain mask'
        }
        
        logger.info(f"Method 1 results: {original_count} -> {filtered_count} "
                   f"({removed_count} removed, {stats['removal_rate']:.2%} removal rate)")
        
        return filtered_bboxes, filtered_scores, stats
    
    def method_2_vein_filter(self, 
                            bboxes: np.ndarray, 
                            scores: np.ndarray, 
                            vein_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Method 2: Remove bounding boxes with ANY overlap with vein mask.
        
        Algorithm: if overlap(bbox, vein_mask) > 0: remove_bbox
        
        Args:
            bboxes: Array of bounding boxes (N, 6)
            scores: Array of confidence scores (N,)
            vein_mask: 3D binary vein mask (modified - CVS subtracted)
            
        Returns:
            Tuple of (filtered_bboxes, filtered_scores, statistics)
        """
        logger.info(f"Applying Method 2: Vein filter on {len(bboxes)} detections")
        
        # Apply method 2
        filtered_bboxes, filtered_scores = self.calculator.method_2_any_vein_overlap(
            bboxes, scores, vein_mask
        )
        
        # Calculate statistics
        original_count = len(bboxes)
        filtered_count = len(filtered_bboxes)
        removed_count = original_count - filtered_count
        
        stats = {
            'method': 'Method 2: Vein Filter',
            'original_detections': original_count,
            'filtered_detections': filtered_count,
            'removed_detections': removed_count,
            'removal_rate': removed_count / original_count if original_count > 0 else 0.0,
            'description': 'Remove detections with any vein overlap'
        }
        
        logger.info(f"Method 2 results: {original_count} -> {filtered_count} "
                   f"({removed_count} removed, {stats['removal_rate']:.2%} removal rate)")
        
        return filtered_bboxes, filtered_scores, stats
    
    def method_3_artery_priority_filter(self, 
                                       bboxes: np.ndarray, 
                                       scores: np.ndarray, 
                                       artery_mask: np.ndarray, 
                                       vein_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Method 3: Remove bounding boxes if overlap_vein > overlap_artery.
        
        Algorithm: 
          overlap_vein = volume(bbox ∩ vein_mask) / volume(bbox)
          overlap_artery = volume(bbox ∩ artery_mask) / volume(bbox)
          if overlap_vein > overlap_artery: remove_bbox
        
        Args:
            bboxes: Array of bounding boxes (N, 6)
            scores: Array of confidence scores (N,)
            artery_mask: 3D binary artery mask
            vein_mask: 3D binary vein mask (modified - CVS subtracted)
            
        Returns:
            Tuple of (filtered_bboxes, filtered_scores, statistics)
        """
        logger.info(f"Applying Method 3: Artery priority filter on {len(bboxes)} detections")
        
        # Apply method 3
        filtered_bboxes, filtered_scores = self.calculator.method_3_vein_vs_artery_overlap(
            bboxes, scores, artery_mask, vein_mask
        )
        
        # Calculate statistics
        original_count = len(bboxes)
        filtered_count = len(filtered_bboxes)
        removed_count = original_count - filtered_count
        
        stats = {
            'method': 'Method 3: Artery Priority Filter',
            'original_detections': original_count,
            'filtered_detections': filtered_count,
            'removed_detections': removed_count,
            'removal_rate': removed_count / original_count if original_count > 0 else 0.0,
            'description': 'Remove detections where vein overlap > artery overlap'
        }
        
        logger.info(f"Method 3 results: {original_count} -> {filtered_count} "
                   f"({removed_count} removed, {stats['removal_rate']:.2%} removal rate)")
        
        return filtered_bboxes, filtered_scores, stats
    
    def method_4_brain_and_vein_filter(self, 
                                      bboxes: np.ndarray, 
                                      scores: np.ndarray, 
                                      brain_mask: np.ndarray, 
                                      vein_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Method 4: Combine Method 1 AND Method 2.
        
        Algorithm: Apply both conditions (outside brain OR any vein overlap)
        
        Args:
            bboxes: Array of bounding boxes (N, 6)
            scores: Array of confidence scores (N,)
            brain_mask: 3D binary brain mask
            vein_mask: 3D binary vein mask (modified - CVS subtracted)
            
        Returns:
            Tuple of (filtered_bboxes, filtered_scores, statistics)
        """
        logger.info(f"Applying Method 4: Brain + Vein filter on {len(bboxes)} detections")
        
        # Apply method 4
        filtered_bboxes, filtered_scores = self.calculator.method_4_brain_and_no_vein(
            bboxes, scores, brain_mask, vein_mask
        )
        
        # Calculate statistics
        original_count = len(bboxes)
        filtered_count = len(filtered_bboxes)
        removed_count = original_count - filtered_count
        
        stats = {
            'method': 'Method 4: Brain + Vein Filter',
            'original_detections': original_count,
            'filtered_detections': filtered_count,
            'removed_detections': removed_count,
            'removal_rate': removed_count / original_count if original_count > 0 else 0.0,
            'description': 'Remove detections outside brain OR with any vein overlap'
        }
        
        logger.info(f"Method 4 results: {original_count} -> {filtered_count} "
                   f"({removed_count} removed, {stats['removal_rate']:.2%} removal rate)")
        
        return filtered_bboxes, filtered_scores, stats
    
    def method_5_brain_and_artery_priority_filter(self, 
                                                 bboxes: np.ndarray, 
                                                 scores: np.ndarray, 
                                                 brain_mask: np.ndarray, 
                                                 artery_mask: np.ndarray, 
                                                 vein_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Method 5: Combine Method 1 AND Method 3.
        
        Algorithm: Apply both conditions (outside brain OR vein > artery overlap)
        
        Args:
            bboxes: Array of bounding boxes (N, 6)
            scores: Array of confidence scores (N,)
            brain_mask: 3D binary brain mask
            artery_mask: 3D binary artery mask
            vein_mask: 3D binary vein mask (modified - CVS subtracted)
            
        Returns:
            Tuple of (filtered_bboxes, filtered_scores, statistics)
        """
        logger.info(f"Applying Method 5: Brain + Artery priority filter on {len(bboxes)} detections")
        
        # Apply method 5
        filtered_bboxes, filtered_scores = self.calculator.method_5_brain_and_artery_priority(
            bboxes, scores, brain_mask, artery_mask, vein_mask
        )
        
        # Calculate statistics
        original_count = len(bboxes)
        filtered_count = len(filtered_bboxes)
        removed_count = original_count - filtered_count
        
        stats = {
            'method': 'Method 5: Brain + Artery Priority Filter',
            'original_detections': original_count,
            'filtered_detections': filtered_count,
            'removed_detections': removed_count,
            'removal_rate': removed_count / original_count if original_count > 0 else 0.0,
            'description': 'Remove detections outside brain OR where vein overlap > artery overlap'
        }
        
        logger.info(f"Method 5 results: {original_count} -> {filtered_count} "
                   f"({removed_count} removed, {stats['removal_rate']:.2%} removal rate)")
        
        return filtered_bboxes, filtered_scores, stats
    
    def apply_all_methods(self, 
                         bboxes: np.ndarray, 
                         scores: np.ndarray, 
                         brain_mask: np.ndarray, 
                         artery_mask: np.ndarray, 
                         vein_mask: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
        """
        Apply all five post-processing methods and return results.
        
        Args:
            bboxes: Array of bounding boxes (N, 6)
            scores: Array of confidence scores (N,)
            brain_mask: 3D binary brain mask
            artery_mask: 3D binary artery mask
            vein_mask: 3D binary vein mask (modified - CVS subtracted)
            
        Returns:
            Dictionary with results for each method
        """
        logger.info(f"Applying all 5 post-processing methods on {len(bboxes)} detections")
        
        results = {}
        
        # Method 1: Brain filter
        results['method_1'] = self.method_1_brain_filter(bboxes, scores, brain_mask)
        
        # Method 2: Vein filter
        results['method_2'] = self.method_2_vein_filter(bboxes, scores, vein_mask)
        
        # Method 3: Artery priority filter
        results['method_3'] = self.method_3_artery_priority_filter(bboxes, scores, artery_mask, vein_mask)
        
        # Method 4: Brain + Vein filter
        results['method_4'] = self.method_4_brain_and_vein_filter(bboxes, scores, brain_mask, vein_mask)
        
        # Method 5: Brain + Artery priority filter
        results['method_5'] = self.method_5_brain_and_artery_priority_filter(
            bboxes, scores, brain_mask, artery_mask, vein_mask
        )
        
        # Log summary
        logger.info("All methods applied successfully:")
        for method_name, (filtered_bboxes, _, stats) in results.items():
            logger.info(f"  {method_name}: {stats['original_detections']} -> "
                       f"{stats['filtered_detections']} ({stats['removal_rate']:.2%} removed)")
        
        return results


class MethodComparator:
    """
    Utility class for comparing the effectiveness of different post-processing methods.
    """
    
    def __init__(self):
        """Initialize the method comparator."""
        self.results_history = []
        logger.info("Initialized MethodComparator")
    
    def compare_methods(self, 
                       method_results: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, Any]]],
                       ground_truth_bboxes: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compare the effectiveness of different post-processing methods.
        
        Args:
            method_results: Results from apply_all_methods()
            ground_truth_bboxes: Optional ground truth for accuracy calculation
            
        Returns:
            Comparison statistics
        """
        logger.info("Comparing post-processing method effectiveness")
        
        comparison = {
            'method_statistics': {},
            'ranking_by_removal_rate': [],
            'ranking_by_remaining_detections': []
        }
        
        # Extract statistics for each method
        for method_name, (filtered_bboxes, filtered_scores, stats) in method_results.items():
            comparison['method_statistics'][method_name] = {
                'original_count': stats['original_detections'],
                'filtered_count': stats['filtered_detections'],
                'removed_count': stats['removed_detections'],
                'removal_rate': stats['removal_rate'],
                'description': stats['description']
            }
        
        # Rank by removal rate (highest first)
        comparison['ranking_by_removal_rate'] = sorted(
            comparison['method_statistics'].items(),
            key=lambda x: x[1]['removal_rate'],
            reverse=True
        )
        
        # Rank by remaining detections (lowest first)
        comparison['ranking_by_remaining_detections'] = sorted(
            comparison['method_statistics'].items(),
            key=lambda x: x[1]['filtered_count']
        )
        
        # Log comparison results
        logger.info("Method comparison results:")
        logger.info("Ranking by removal rate (most aggressive first):")
        for i, (method, stats) in enumerate(comparison['ranking_by_removal_rate'], 1):
            logger.info(f"  {i}. {method}: {stats['removal_rate']:.2%} removal rate")
        
        logger.info("Ranking by remaining detections (most conservative first):")
        for i, (method, stats) in enumerate(comparison['ranking_by_remaining_detections'], 1):
            logger.info(f"  {i}. {method}: {stats['filtered_count']} detections remaining")
        
        return comparison
    
    def save_comparison_report(self, 
                              comparison: Dict[str, Any], 
                              output_path: str) -> None:
        """
        Save comparison report to file.
        
        Args:
            comparison: Comparison results from compare_methods()
            output_path: Path to save the report
        """
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("Post-processing Method Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Method Statistics:\n")
            f.write("-" * 20 + "\n")
            for method, stats in comparison['method_statistics'].items():
                f.write(f"{method}:\n")
                f.write(f"  Description: {stats['description']}\n")
                f.write(f"  Original detections: {stats['original_count']}\n")
                f.write(f"  Filtered detections: {stats['filtered_count']}\n")
                f.write(f"  Removed detections: {stats['removed_count']}\n")
                f.write(f"  Removal rate: {stats['removal_rate']:.2%}\n\n")
            
            f.write("Ranking by Removal Rate (Most Aggressive First):\n")
            f.write("-" * 45 + "\n")
            for i, (method, stats) in enumerate(comparison['ranking_by_removal_rate'], 1):
                f.write(f"{i}. {method}: {stats['removal_rate']:.2%}\n")
            
            f.write("\nRanking by Remaining Detections (Most Conservative First):\n")
            f.write("-" * 55 + "\n")
            for i, (method, stats) in enumerate(comparison['ranking_by_remaining_detections'], 1):
                f.write(f"{i}. {method}: {stats['filtered_count']} detections\n")
        
        logger.info(f"Comparison report saved to: {report_path}")


def create_test_scenario() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a test scenario with sample data for method validation.
    
    Returns:
        Tuple of (bboxes, scores, brain_mask, artery_mask, vein_mask)
    """
    logger.info("Creating test scenario for method validation")
    
    # Create sample bounding boxes (N, 6) - [z_min, y_min, x_min, z_max, y_max, x_max]
    bboxes = np.array([
        [10, 20, 30, 15, 25, 35],  # Inside brain, overlaps artery
        [50, 60, 70, 55, 65, 75],  # Outside brain
        [25, 35, 45, 30, 40, 50],  # Inside brain, overlaps vein
        [80, 90, 100, 85, 95, 105],  # Outside brain, overlaps vein
        [15, 25, 35, 20, 30, 40],  # Inside brain, overlaps both
    ], dtype=np.float32)
    
    # Create sample confidence scores
    scores = np.array([0.9, 0.8, 0.85, 0.75, 0.95], dtype=np.float32)
    
    # Create sample masks (100x100x100)
    mask_shape = (100, 100, 100)
    
    # Brain mask - covers central region
    brain_mask = np.zeros(mask_shape, dtype=bool)
    brain_mask[5:60, 10:80, 20:90] = True
    
    # Artery mask - specific regions
    artery_mask = np.zeros(mask_shape, dtype=bool)
    artery_mask[8:18, 18:28, 28:38] = True  # Overlaps with bbox 0
    artery_mask[13:23, 23:33, 33:43] = True  # Overlaps with bbox 4
    
    # Vein mask - different regions
    vein_mask = np.zeros(mask_shape, dtype=bool)
    vein_mask[23:33, 33:43, 43:53] = True  # Overlaps with bbox 2
    vein_mask[78:88, 88:98, 98:108] = True  # Overlaps with bbox 3 (outside brain)
    vein_mask[13:23, 23:33, 33:43] = True  # Overlaps with bbox 4 (same as artery)
    
    logger.info(f"Created test scenario with {len(bboxes)} bounding boxes")
    logger.info(f"Mask shapes: brain={brain_mask.shape}, artery={artery_mask.shape}, vein={vein_mask.shape}")
    
    return bboxes, scores, brain_mask, artery_mask, vein_mask


def main():
    """
    Main function for testing the post-processing methods.
    """
    logger.info("Starting post-processing method implementations test")
    
    # Create test data
    bboxes, scores, brain_mask, artery_mask, vein_mask = create_test_scenario()
    
    # Initialize post-processor
    processor = AnatomicalPostProcessor()
    
    # Apply all methods
    results = processor.apply_all_methods(bboxes, scores, brain_mask, artery_mask, vein_mask)
    
    # Compare methods
    comparator = MethodComparator()
    comparison = comparator.compare_methods(results)
    
    # Save comparison report
    comparator.save_comparison_report(comparison, "method_comparison_report.txt")
    
    logger.info("Post-processing method implementations test completed successfully")


if __name__ == "__main__":
    main()