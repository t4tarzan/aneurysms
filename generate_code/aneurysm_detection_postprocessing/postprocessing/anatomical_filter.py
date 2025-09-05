"""
Anatomical Filter - Main Post-processing Pipeline

This module implements the main anatomical filtering pipeline that integrates
detection results with anatomical masks to reduce false positives in aneurysm
detection. It serves as the primary interface for applying the five post-processing
methods described in the paper.

Key Features:
- Integration of detection results with anatomical masks
- Application of confidence thresholding (0.8 as per paper)
- Orchestration of all five post-processing methods
- Comprehensive logging and performance tracking
- Batch processing capabilities for multiple cases

Author: Implementation based on paper methodology
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import time
from dataclasses import dataclass

from .method_implementations import AnatomicalPostProcessor, MethodComparator
from .overlap_calculator import OverlapCalculator, PostProcessingMethodsCalculator
from ..segmentation.mask_utils import MaskValidator, BoundingBoxProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnatomicalMasks:
    """Container for anatomical masks used in post-processing."""
    brain_mask: np.ndarray
    artery_mask: np.ndarray
    vein_mask: np.ndarray
    cvs_mask: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate mask consistency after initialization."""
        validator = MaskValidator()
        
        # Validate each mask
        for mask_name, mask in [
            ("brain_mask", self.brain_mask),
            ("artery_mask", self.artery_mask), 
            ("vein_mask", self.vein_mask)
        ]:
            if not validator.validate_mask_format(mask):
                raise ValueError(f"Invalid format for {mask_name}")
        
        # Check consistency between masks
        if not validator.check_mask_consistency([
            self.brain_mask, self.artery_mask, self.vein_mask
        ]):
            logger.warning("Anatomical masks have inconsistent dimensions")


@dataclass
class DetectionResults:
    """Container for detection results before post-processing."""
    bounding_boxes: np.ndarray  # Shape: (N, 6) - [x_min, y_min, z_min, x_max, y_max, z_max]
    confidence_scores: np.ndarray  # Shape: (N,)
    case_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate detection results after initialization."""
        if self.bounding_boxes.shape[0] != self.confidence_scores.shape[0]:
            raise ValueError("Number of bounding boxes must match number of confidence scores")
        
        if self.bounding_boxes.shape[1] != 6:
            raise ValueError("Bounding boxes must have 6 coordinates [x_min, y_min, z_min, x_max, y_max, z_max]")


@dataclass
class FilteredResults:
    """Container for post-processed detection results."""
    original_detections: DetectionResults
    filtered_detections: Dict[str, DetectionResults]
    processing_stats: Dict[str, Dict[str, Any]]
    confidence_threshold: float
    processing_time: float
    
    def get_method_results(self, method_name: str) -> Optional[DetectionResults]:
        """Get results for a specific method."""
        return self.filtered_detections.get(method_name)
    
    def get_reduction_stats(self) -> Dict[str, float]:
        """Calculate false positive reduction statistics."""
        original_count = len(self.original_detections.bounding_boxes)
        reduction_stats = {}
        
        for method_name, results in self.filtered_detections.items():
            filtered_count = len(results.bounding_boxes)
            reduction_rate = (original_count - filtered_count) / original_count if original_count > 0 else 0.0
            reduction_stats[method_name] = {
                'original_count': original_count,
                'filtered_count': filtered_count,
                'reduction_rate': reduction_rate,
                'removed_count': original_count - filtered_count
            }
        
        return reduction_stats


class AnatomicalFilter:
    """
    Main anatomical filtering pipeline for aneurysm detection post-processing.
    
    This class orchestrates the application of anatomical masks to filter
    detection results, implementing the methodology described in the paper.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.8,
                 voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4),
                 enable_logging: bool = True):
        """
        Initialize the anatomical filter.
        
        Args:
            confidence_threshold: Confidence threshold for initial filtering (default: 0.8)
            voxel_spacing: Voxel spacing in mm (default: (0.4, 0.4, 0.4))
            enable_logging: Whether to enable detailed logging
        """
        self.confidence_threshold = confidence_threshold
        self.voxel_spacing = voxel_spacing
        self.enable_logging = enable_logging
        
        # Initialize processing components
        self.post_processor = AnatomicalPostProcessor(voxel_spacing=voxel_spacing)
        self.method_calculator = PostProcessingMethodsCalculator(voxel_spacing=voxel_spacing)
        self.overlap_calculator = OverlapCalculator(voxel_spacing=voxel_spacing)
        self.bbox_processor = BoundingBoxProcessor()
        self.method_comparator = MethodComparator()
        
        # Processing statistics
        self.processing_history = []
        
        if self.enable_logging:
            logger.info(f"AnatomicalFilter initialized with confidence_threshold={confidence_threshold}")
    
    def apply_confidence_threshold(self, 
                                 detections: DetectionResults) -> DetectionResults:
        """
        Apply confidence threshold to filter detections.
        
        Args:
            detections: Original detection results
            
        Returns:
            Filtered detection results above confidence threshold
        """
        # Find detections above threshold
        valid_indices = detections.confidence_scores >= self.confidence_threshold
        
        # Filter bounding boxes and scores
        filtered_boxes = detections.bounding_boxes[valid_indices]
        filtered_scores = detections.confidence_scores[valid_indices]
        
        if self.enable_logging:
            original_count = len(detections.bounding_boxes)
            filtered_count = len(filtered_boxes)
            logger.info(f"Confidence filtering: {original_count} -> {filtered_count} detections "
                       f"(threshold={self.confidence_threshold})")
        
        return DetectionResults(
            bounding_boxes=filtered_boxes,
            confidence_scores=filtered_scores,
            case_id=detections.case_id
        )
    
    def filter_detections(self,
                         detections: DetectionResults,
                         masks: AnatomicalMasks,
                         methods: Optional[List[str]] = None) -> FilteredResults:
        """
        Apply anatomical filtering to detection results.
        
        Args:
            detections: Detection results to filter
            masks: Anatomical masks for filtering
            methods: List of methods to apply (default: all methods)
            
        Returns:
            Complete filtered results with statistics
        """
        start_time = time.time()
        
        if methods is None:
            methods = ['method_1', 'method_2', 'method_3', 'method_4', 'method_5']
        
        # Apply confidence threshold first
        thresholded_detections = self.apply_confidence_threshold(detections)
        
        if len(thresholded_detections.bounding_boxes) == 0:
            logger.warning("No detections remain after confidence thresholding")
            return self._create_empty_results(detections, methods, time.time() - start_time)
        
        # Apply each post-processing method
        filtered_results = {}
        processing_stats = {}
        
        for method_name in methods:
            method_start_time = time.time()
            
            try:
                # Apply the specific method
                filtered_boxes, filtered_scores, method_stats = self._apply_method(
                    method_name, thresholded_detections, masks
                )
                
                # Store results
                filtered_results[method_name] = DetectionResults(
                    bounding_boxes=filtered_boxes,
                    confidence_scores=filtered_scores,
                    case_id=detections.case_id
                )
                
                # Store processing statistics
                method_time = time.time() - method_start_time
                processing_stats[method_name] = {
                    **method_stats,
                    'processing_time': method_time,
                    'original_count': len(thresholded_detections.bounding_boxes),
                    'filtered_count': len(filtered_boxes)
                }
                
                if self.enable_logging:
                    logger.info(f"{method_name}: {len(thresholded_detections.bounding_boxes)} -> "
                               f"{len(filtered_boxes)} detections ({method_time:.3f}s)")
                
            except Exception as e:
                logger.error(f"Error applying {method_name}: {str(e)}")
                # Create empty result for failed method
                filtered_results[method_name] = DetectionResults(
                    bounding_boxes=np.empty((0, 6)),
                    confidence_scores=np.empty(0),
                    case_id=detections.case_id
                )
                processing_stats[method_name] = {
                    'error': str(e),
                    'processing_time': time.time() - method_start_time
                }
        
        total_time = time.time() - start_time
        
        # Create comprehensive results
        results = FilteredResults(
            original_detections=detections,
            filtered_detections=filtered_results,
            processing_stats=processing_stats,
            confidence_threshold=self.confidence_threshold,
            processing_time=total_time
        )
        
        # Store in processing history
        self.processing_history.append(results)
        
        if self.enable_logging:
            logger.info(f"Anatomical filtering completed in {total_time:.3f}s")
        
        return results
    
    def _apply_method(self,
                     method_name: str,
                     detections: DetectionResults,
                     masks: AnatomicalMasks) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Apply a specific post-processing method.
        
        Args:
            method_name: Name of the method to apply
            detections: Detection results to filter
            masks: Anatomical masks
            
        Returns:
            Tuple of (filtered_boxes, filtered_scores, method_stats)
        """
        boxes = detections.bounding_boxes
        scores = detections.confidence_scores
        
        # Apply the appropriate method
        if method_name == 'method_1':
            # Remove bounding boxes outside brain mask
            valid_indices, stats = self.method_calculator.method_1_outside_brain(
                boxes, masks.brain_mask
            )
        elif method_name == 'method_2':
            # Remove bounding boxes with any vein overlap
            valid_indices, stats = self.method_calculator.method_2_any_vein_overlap(
                boxes, masks.vein_mask
            )
        elif method_name == 'method_3':
            # Remove bounding boxes if vein overlap > artery overlap
            valid_indices, stats = self.method_calculator.method_3_vein_vs_artery_overlap(
                boxes, masks.vein_mask, masks.artery_mask
            )
        elif method_name == 'method_4':
            # Combine method 1 and method 2
            valid_indices, stats = self.method_calculator.method_4_brain_and_no_vein(
                boxes, masks.brain_mask, masks.vein_mask
            )
        elif method_name == 'method_5':
            # Combine method 1 and method 3
            valid_indices, stats = self.method_calculator.method_5_brain_and_artery_priority(
                boxes, masks.brain_mask, masks.vein_mask, masks.artery_mask
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        # Filter results
        filtered_boxes = boxes[valid_indices]
        filtered_scores = scores[valid_indices]
        
        return filtered_boxes, filtered_scores, stats
    
    def _create_empty_results(self,
                            original_detections: DetectionResults,
                            methods: List[str],
                            processing_time: float) -> FilteredResults:
        """Create empty results when no detections remain after thresholding."""
        filtered_results = {}
        processing_stats = {}
        
        for method_name in methods:
            filtered_results[method_name] = DetectionResults(
                bounding_boxes=np.empty((0, 6)),
                confidence_scores=np.empty(0),
                case_id=original_detections.case_id
            )
            processing_stats[method_name] = {
                'original_count': 0,
                'filtered_count': 0,
                'processing_time': 0.0
            }
        
        return FilteredResults(
            original_detections=original_detections,
            filtered_detections=filtered_results,
            processing_stats=processing_stats,
            confidence_threshold=self.confidence_threshold,
            processing_time=processing_time
        )
    
    def batch_filter_detections(self,
                              detection_list: List[DetectionResults],
                              mask_list: List[AnatomicalMasks],
                              methods: Optional[List[str]] = None) -> List[FilteredResults]:
        """
        Apply anatomical filtering to multiple cases in batch.
        
        Args:
            detection_list: List of detection results for multiple cases
            mask_list: List of anatomical masks for multiple cases
            methods: List of methods to apply (default: all methods)
            
        Returns:
            List of filtered results for each case
        """
        if len(detection_list) != len(mask_list):
            raise ValueError("Number of detection cases must match number of mask cases")
        
        results = []
        total_start_time = time.time()
        
        for i, (detections, masks) in enumerate(zip(detection_list, mask_list)):
            if self.enable_logging:
                logger.info(f"Processing case {i+1}/{len(detection_list)}: {detections.case_id}")
            
            case_results = self.filter_detections(detections, masks, methods)
            results.append(case_results)
        
        total_time = time.time() - total_start_time
        
        if self.enable_logging:
            logger.info(f"Batch processing completed: {len(detection_list)} cases in {total_time:.3f}s")
        
        return results
    
    def compare_methods(self,
                       results: FilteredResults,
                       save_report: bool = False,
                       report_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Compare the effectiveness of different post-processing methods.
        
        Args:
            results: Filtered results to analyze
            save_report: Whether to save comparison report
            report_path: Path to save report (optional)
            
        Returns:
            Method comparison statistics
        """
        return self.method_comparator.compare_methods(
            results.filtered_detections,
            results.original_detections,
            save_report=save_report,
            report_path=report_path
        )
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get summary of all processing operations performed.
        
        Returns:
            Summary statistics of processing history
        """
        if not self.processing_history:
            return {"message": "No processing operations performed yet"}
        
        total_cases = len(self.processing_history)
        total_time = sum(result.processing_time for result in self.processing_history)
        
        # Aggregate statistics across all cases
        method_stats = {}
        for result in self.processing_history:
            for method_name, stats in result.processing_stats.items():
                if method_name not in method_stats:
                    method_stats[method_name] = {
                        'total_original': 0,
                        'total_filtered': 0,
                        'total_time': 0.0,
                        'cases_processed': 0
                    }
                
                method_stats[method_name]['total_original'] += stats.get('original_count', 0)
                method_stats[method_name]['total_filtered'] += stats.get('filtered_count', 0)
                method_stats[method_name]['total_time'] += stats.get('processing_time', 0.0)
                method_stats[method_name]['cases_processed'] += 1
        
        # Calculate reduction rates
        for method_name, stats in method_stats.items():
            if stats['total_original'] > 0:
                stats['reduction_rate'] = (
                    (stats['total_original'] - stats['total_filtered']) / stats['total_original']
                )
            else:
                stats['reduction_rate'] = 0.0
        
        return {
            'total_cases_processed': total_cases,
            'total_processing_time': total_time,
            'average_time_per_case': total_time / total_cases if total_cases > 0 else 0.0,
            'confidence_threshold': self.confidence_threshold,
            'method_statistics': method_stats
        }


def create_test_scenario() -> Tuple[DetectionResults, AnatomicalMasks]:
    """
    Create test scenario for anatomical filtering.
    
    Returns:
        Tuple of (test_detections, test_masks)
    """
    # Create test detection results
    test_boxes = np.array([
        [10, 10, 10, 20, 20, 20],  # Inside brain, overlaps artery
        [50, 50, 50, 60, 60, 60],  # Outside brain
        [30, 30, 30, 40, 40, 40],  # Inside brain, overlaps vein
        [70, 70, 70, 80, 80, 80],  # Inside brain, overlaps both
        [5, 5, 5, 15, 15, 15],     # Edge case
    ])
    
    test_scores = np.array([0.9, 0.85, 0.95, 0.82, 0.88])
    
    test_detections = DetectionResults(
        bounding_boxes=test_boxes,
        confidence_scores=test_scores,
        case_id="test_case_001"
    )
    
    # Create test anatomical masks
    mask_shape = (100, 100, 100)
    
    # Brain mask - covers central region
    brain_mask = np.zeros(mask_shape, dtype=bool)
    brain_mask[5:85, 5:85, 5:85] = True
    
    # Artery mask - covers some regions
    artery_mask = np.zeros(mask_shape, dtype=bool)
    artery_mask[8:25, 8:25, 8:25] = True
    artery_mask[65:82, 65:82, 65:82] = True
    
    # Vein mask - covers different regions
    vein_mask = np.zeros(mask_shape, dtype=bool)
    vein_mask[25:45, 25:45, 25:45] = True
    vein_mask[65:85, 65:85, 65:85] = True
    
    test_masks = AnatomicalMasks(
        brain_mask=brain_mask,
        artery_mask=artery_mask,
        vein_mask=vein_mask
    )
    
    return test_detections, test_masks


if __name__ == "__main__":
    # Test the anatomical filter
    print("Testing Anatomical Filter...")
    
    # Create test scenario
    test_detections, test_masks = create_test_scenario()
    
    # Initialize filter
    filter_pipeline = AnatomicalFilter(
        confidence_threshold=0.8,
        enable_logging=True
    )
    
    # Apply filtering
    results = filter_pipeline.filter_detections(test_detections, test_masks)
    
    # Print results
    print("\nFiltering Results:")
    print(f"Original detections: {len(test_detections.bounding_boxes)}")
    
    for method_name, filtered_detections in results.filtered_detections.items():
        print(f"{method_name}: {len(filtered_detections.bounding_boxes)} detections remaining")
    
    # Get reduction statistics
    reduction_stats = results.get_reduction_stats()
    print("\nReduction Statistics:")
    for method_name, stats in reduction_stats.items():
        print(f"{method_name}: {stats['reduction_rate']:.2%} reduction "
              f"({stats['removed_count']} removed)")
    
    # Compare methods
    comparison = filter_pipeline.compare_methods(results)
    print(f"\nProcessing completed in {results.processing_time:.3f}s")
    
    print("Anatomical Filter test completed successfully!")