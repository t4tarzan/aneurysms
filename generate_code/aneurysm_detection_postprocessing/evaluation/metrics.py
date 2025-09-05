"""
Evaluation metrics for intracranial aneurysm detection with post-processing.

This module implements comprehensive evaluation metrics for assessing the performance
of aneurysm detection models before and after anatomical post-processing, including
TP/FP/FN calculations, sensitivity, precision, and false positive analysis.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from collections import defaultdict

# Internal imports
from ..data.annotation_parser import BoundingBox3D, AnnotationCase, AnnotationManager
from ..postprocessing.anatomical_filter import DetectionResults, FilteredResults


@dataclass
class DetectionMetrics:
    """Container for detection performance metrics."""
    
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_ground_truth: int = 0
    total_detections: int = 0
    
    # Per-case metrics
    case_metrics: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # IoU threshold used for matching
    iou_threshold: float = 0.1
    
    # Confidence threshold applied
    confidence_threshold: float = 0.8
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        if self.true_positives < 0 or self.false_positives < 0 or self.false_negatives < 0:
            raise ValueError("Metrics values cannot be negative")
        
        if self.total_ground_truth < 0 or self.total_detections < 0:
            raise ValueError("Total counts cannot be negative")
    
    @property
    def sensitivity(self) -> float:
        """Calculate sensitivity (recall/true positive rate)."""
        if self.total_ground_truth == 0:
            return 0.0
        return self.true_positives / self.total_ground_truth
    
    @property
    def precision(self) -> float:
        """Calculate precision (positive predictive value)."""
        if self.total_detections == 0:
            return 0.0
        return self.true_positives / self.total_detections
    
    @property
    def f1_score(self) -> float:
        """Calculate F1 score (harmonic mean of precision and recall)."""
        if self.precision + self.sensitivity == 0:
            return 0.0
        return 2 * (self.precision * self.sensitivity) / (self.precision + self.sensitivity)
    
    @property
    def false_positive_rate(self) -> float:
        """Calculate false positive rate per case."""
        total_cases = len(self.case_metrics)
        if total_cases == 0:
            return 0.0
        return self.false_positives / total_cases
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'total_ground_truth': self.total_ground_truth,
            'total_detections': self.total_detections,
            'sensitivity': self.sensitivity,
            'precision': self.precision,
            'f1_score': self.f1_score,
            'false_positive_rate': self.false_positive_rate,
            'iou_threshold': self.iou_threshold,
            'confidence_threshold': self.confidence_threshold,
            'total_cases': len(self.case_metrics)
        }


@dataclass
class ComparisonMetrics:
    """Container for comparing metrics before and after post-processing."""
    
    before_processing: DetectionMetrics
    after_processing: DetectionMetrics
    method_name: str = "Unknown"
    
    @property
    def fp_reduction(self) -> int:
        """Calculate absolute false positive reduction."""
        return self.before_processing.false_positives - self.after_processing.false_positives
    
    @property
    def fp_reduction_rate(self) -> float:
        """Calculate false positive reduction rate."""
        if self.before_processing.false_positives == 0:
            return 0.0
        return self.fp_reduction / self.before_processing.false_positives
    
    @property
    def sensitivity_change(self) -> float:
        """Calculate change in sensitivity."""
        return self.after_processing.sensitivity - self.before_processing.sensitivity
    
    @property
    def precision_improvement(self) -> float:
        """Calculate precision improvement."""
        return self.after_processing.precision - self.before_processing.precision
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """Get comprehensive comparison summary."""
        return {
            'method_name': self.method_name,
            'before_processing': self.before_processing.get_summary(),
            'after_processing': self.after_processing.get_summary(),
            'fp_reduction': self.fp_reduction,
            'fp_reduction_rate': self.fp_reduction_rate,
            'sensitivity_change': self.sensitivity_change,
            'precision_improvement': self.precision_improvement
        }


class IoUCalculator:
    """Calculate Intersection over Union (IoU) for 3D bounding boxes."""
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)):
        """
        Initialize IoU calculator.
        
        Args:
            voxel_spacing: Voxel spacing in mm (x, y, z)
        """
        self.voxel_spacing = np.array(voxel_spacing)
        self.logger = logging.getLogger(__name__)
    
    def calculate_iou_3d(self, bbox1: Union[BoundingBox3D, np.ndarray], 
                        bbox2: Union[BoundingBox3D, np.ndarray]) -> float:
        """
        Calculate 3D IoU between two bounding boxes.
        
        Args:
            bbox1: First bounding box (BoundingBox3D or array [x, y, z, w, h, d])
            bbox2: Second bounding box (BoundingBox3D or array [x, y, z, w, h, d])
            
        Returns:
            IoU value between 0 and 1
        """
        try:
            # Convert to arrays if needed
            if isinstance(bbox1, BoundingBox3D):
                bbox1_array = bbox1.to_array()
            else:
                bbox1_array = np.array(bbox1)
            
            if isinstance(bbox2, BoundingBox3D):
                bbox2_array = bbox2.to_array()
            else:
                bbox2_array = np.array(bbox2)
            
            # Extract center coordinates and dimensions
            center1, dims1 = bbox1_array[:3], bbox1_array[3:6]
            center2, dims2 = bbox2_array[:3], bbox2_array[3:6]
            
            # Convert to corner coordinates
            min1 = center1 - dims1 / 2
            max1 = center1 + dims1 / 2
            min2 = center2 - dims2 / 2
            max2 = center2 + dims2 / 2
            
            # Calculate intersection
            intersection_min = np.maximum(min1, min2)
            intersection_max = np.minimum(max1, max2)
            
            # Check if there's an intersection
            if np.any(intersection_min >= intersection_max):
                return 0.0
            
            # Calculate intersection volume
            intersection_dims = intersection_max - intersection_min
            intersection_volume = np.prod(intersection_dims)
            
            # Calculate union volume
            volume1 = np.prod(dims1)
            volume2 = np.prod(dims2)
            union_volume = volume1 + volume2 - intersection_volume
            
            # Calculate IoU
            if union_volume == 0:
                return 0.0
            
            iou = intersection_volume / union_volume
            return float(iou)
            
        except Exception as e:
            self.logger.error(f"Error calculating IoU: {e}")
            return 0.0
    
    def batch_iou_calculation(self, detections: np.ndarray, 
                             ground_truths: np.ndarray) -> np.ndarray:
        """
        Calculate IoU matrix between all detections and ground truths.
        
        Args:
            detections: Array of detection bboxes [N, 6] (x, y, z, w, h, d)
            ground_truths: Array of ground truth bboxes [M, 6] (x, y, z, w, h, d)
            
        Returns:
            IoU matrix [N, M]
        """
        n_detections = len(detections)
        n_ground_truths = len(ground_truths)
        
        iou_matrix = np.zeros((n_detections, n_ground_truths))
        
        for i, detection in enumerate(detections):
            for j, gt in enumerate(ground_truths):
                iou_matrix[i, j] = self.calculate_iou_3d(detection, gt)
        
        return iou_matrix


class DetectionMatcher:
    """Match detections to ground truth annotations using IoU threshold."""
    
    def __init__(self, iou_threshold: float = 0.1, 
                 voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)):
        """
        Initialize detection matcher.
        
        Args:
            iou_threshold: Minimum IoU for positive match
            voxel_spacing: Voxel spacing in mm
        """
        self.iou_threshold = iou_threshold
        self.iou_calculator = IoUCalculator(voxel_spacing)
        self.logger = logging.getLogger(__name__)
    
    def match_detections(self, detections: np.ndarray, ground_truths: np.ndarray,
                        confidence_scores: Optional[np.ndarray] = None) -> Tuple[List[int], List[int], List[int]]:
        """
        Match detections to ground truths using IoU threshold.
        
        Args:
            detections: Detection bounding boxes [N, 6]
            ground_truths: Ground truth bounding boxes [M, 6]
            confidence_scores: Optional confidence scores for detections
            
        Returns:
            Tuple of (true_positive_indices, false_positive_indices, false_negative_indices)
        """
        if len(detections) == 0:
            # No detections - all ground truths are false negatives
            return [], [], list(range(len(ground_truths)))
        
        if len(ground_truths) == 0:
            # No ground truths - all detections are false positives
            return [], list(range(len(detections))), []
        
        # Calculate IoU matrix
        iou_matrix = self.iou_calculator.batch_iou_calculation(detections, ground_truths)
        
        # Find matches using Hungarian algorithm approach (greedy matching)
        matched_detections = set()
        matched_ground_truths = set()
        true_positives = []
        
        # Sort detections by confidence if available
        if confidence_scores is not None:
            detection_order = np.argsort(confidence_scores)[::-1]  # Descending order
        else:
            detection_order = range(len(detections))
        
        # Greedy matching: highest confidence detections first
        for det_idx in detection_order:
            if det_idx in matched_detections:
                continue
            
            # Find best matching ground truth
            best_gt_idx = -1
            best_iou = self.iou_threshold
            
            for gt_idx in range(len(ground_truths)):
                if gt_idx in matched_ground_truths:
                    continue
                
                iou = iou_matrix[det_idx, gt_idx]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # If match found, mark as true positive
            if best_gt_idx >= 0:
                true_positives.append(det_idx)
                matched_detections.add(det_idx)
                matched_ground_truths.add(best_gt_idx)
        
        # Remaining detections are false positives
        false_positives = [i for i in range(len(detections)) if i not in matched_detections]
        
        # Remaining ground truths are false negatives
        false_negatives = [i for i in range(len(ground_truths)) if i not in matched_ground_truths]
        
        return true_positives, false_positives, false_negatives


class MetricsCalculator:
    """Main class for calculating detection metrics."""
    
    def __init__(self, iou_threshold: float = 0.1, 
                 confidence_threshold: float = 0.8,
                 voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)):
        """
        Initialize metrics calculator.
        
        Args:
            iou_threshold: IoU threshold for positive matches
            confidence_threshold: Confidence threshold for detections
            voxel_spacing: Voxel spacing in mm
        """
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.matcher = DetectionMatcher(iou_threshold, voxel_spacing)
        self.logger = logging.getLogger(__name__)
    
    def calculate_case_metrics(self, detections: np.ndarray, ground_truths: np.ndarray,
                              confidence_scores: np.ndarray, case_id: str = "unknown") -> Dict[str, int]:
        """
        Calculate metrics for a single case.
        
        Args:
            detections: Detection bounding boxes [N, 6]
            ground_truths: Ground truth bounding boxes [M, 6]
            confidence_scores: Confidence scores for detections
            case_id: Case identifier
            
        Returns:
            Dictionary with TP, FP, FN counts
        """
        # Apply confidence threshold
        valid_indices = confidence_scores >= self.confidence_threshold
        filtered_detections = detections[valid_indices]
        filtered_scores = confidence_scores[valid_indices]
        
        # Match detections to ground truths
        tp_indices, fp_indices, fn_indices = self.matcher.match_detections(
            filtered_detections, ground_truths, filtered_scores
        )
        
        case_metrics = {
            'true_positives': len(tp_indices),
            'false_positives': len(fp_indices),
            'false_negatives': len(fn_indices),
            'total_detections': len(filtered_detections),
            'total_ground_truth': len(ground_truths)
        }
        
        self.logger.debug(f"Case {case_id} metrics: {case_metrics}")
        return case_metrics
    
    def calculate_dataset_metrics(self, detection_results: List[DetectionResults],
                                 annotation_cases: List[AnnotationCase]) -> DetectionMetrics:
        """
        Calculate metrics for entire dataset.
        
        Args:
            detection_results: List of detection results for each case
            annotation_cases: List of ground truth annotations
            
        Returns:
            Overall dataset metrics
        """
        # Create case ID mapping
        annotation_dict = {case.case_id: case for case in annotation_cases}
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_detections = 0
        total_ground_truth = 0
        case_metrics = {}
        
        for detection_result in detection_results:
            case_id = detection_result.case_id or "unknown"
            
            # Get corresponding annotations
            if case_id not in annotation_dict:
                self.logger.warning(f"No ground truth found for case {case_id}")
                continue
            
            annotation_case = annotation_dict[case_id]
            ground_truths = annotation_case.get_bounding_boxes_array()
            
            # Calculate case metrics
            case_result = self.calculate_case_metrics(
                detection_result.bounding_boxes,
                ground_truths,
                detection_result.confidence_scores,
                case_id
            )
            
            # Accumulate totals
            total_tp += case_result['true_positives']
            total_fp += case_result['false_positives']
            total_fn += case_result['false_negatives']
            total_detections += case_result['total_detections']
            total_ground_truth += case_result['total_ground_truth']
            
            case_metrics[case_id] = case_result
        
        return DetectionMetrics(
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
            total_ground_truth=total_ground_truth,
            total_detections=total_detections,
            case_metrics=case_metrics,
            iou_threshold=self.iou_threshold,
            confidence_threshold=self.confidence_threshold
        )
    
    def compare_before_after_processing(self, 
                                       original_results: List[DetectionResults],
                                       filtered_results: List[FilteredResults],
                                       annotation_cases: List[AnnotationCase],
                                       method_name: str = "Unknown") -> ComparisonMetrics:
        """
        Compare metrics before and after post-processing.
        
        Args:
            original_results: Original detection results
            filtered_results: Post-processed detection results
            annotation_cases: Ground truth annotations
            method_name: Name of post-processing method
            
        Returns:
            Comparison metrics
        """
        # Calculate metrics before processing
        before_metrics = self.calculate_dataset_metrics(original_results, annotation_cases)
        
        # Extract post-processed detections
        after_results = []
        for filtered_result in filtered_results:
            # Get filtered detections for the method
            method_results = filtered_result.get_method_results()
            if method_name in method_results:
                method_data = method_results[method_name]
                after_result = DetectionResults(
                    bounding_boxes=method_data['filtered_bboxes'],
                    confidence_scores=method_data['filtered_scores'],
                    case_id=filtered_result.original_detections.case_id
                )
                after_results.append(after_result)
        
        # Calculate metrics after processing
        after_metrics = self.calculate_dataset_metrics(after_results, annotation_cases)
        
        return ComparisonMetrics(
            before_processing=before_metrics,
            after_processing=after_metrics,
            method_name=method_name
        )


class PerformanceAnalyzer:
    """Analyze performance across different post-processing methods."""
    
    def __init__(self, metrics_calculator: MetricsCalculator):
        """
        Initialize performance analyzer.
        
        Args:
            metrics_calculator: Configured metrics calculator
        """
        self.metrics_calculator = metrics_calculator
        self.logger = logging.getLogger(__name__)
    
    def analyze_all_methods(self, original_results: List[DetectionResults],
                           filtered_results: List[FilteredResults],
                           annotation_cases: List[AnnotationCase]) -> Dict[str, ComparisonMetrics]:
        """
        Analyze performance for all post-processing methods.
        
        Args:
            original_results: Original detection results
            filtered_results: Post-processed results
            annotation_cases: Ground truth annotations
            
        Returns:
            Dictionary mapping method names to comparison metrics
        """
        method_names = [
            "Method 1: Outside Brain",
            "Method 2: Any Vein Overlap", 
            "Method 3: Vein vs Artery Overlap",
            "Method 4: Brain AND No Vein",
            "Method 5: Brain AND Artery Priority"
        ]
        
        results = {}
        
        for method_name in method_names:
            try:
                comparison = self.metrics_calculator.compare_before_after_processing(
                    original_results, filtered_results, annotation_cases, method_name
                )
                results[method_name] = comparison
                
                self.logger.info(f"Analyzed {method_name}: "
                               f"FP reduction: {comparison.fp_reduction}, "
                               f"Sensitivity change: {comparison.sensitivity_change:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error analyzing {method_name}: {e}")
                continue
        
        return results
    
    def generate_performance_report(self, method_comparisons: Dict[str, ComparisonMetrics]) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            method_comparisons: Results from analyze_all_methods
            
        Returns:
            Comprehensive performance report
        """
        report = {
            'summary': {},
            'method_details': {},
            'best_methods': {},
            'overall_statistics': {}
        }
        
        # Analyze each method
        fp_reductions = []
        sensitivity_changes = []
        precision_improvements = []
        
        for method_name, comparison in method_comparisons.items():
            method_summary = comparison.get_comparison_summary()
            report['method_details'][method_name] = method_summary
            
            fp_reductions.append(comparison.fp_reduction)
            sensitivity_changes.append(comparison.sensitivity_change)
            precision_improvements.append(comparison.precision_improvement)
        
        # Find best methods
        if method_comparisons:
            best_fp_reduction = max(method_comparisons.items(), 
                                  key=lambda x: x[1].fp_reduction)
            best_precision = max(method_comparisons.items(), 
                               key=lambda x: x[1].precision_improvement)
            
            report['best_methods'] = {
                'best_fp_reduction': {
                    'method': best_fp_reduction[0],
                    'reduction': best_fp_reduction[1].fp_reduction
                },
                'best_precision_improvement': {
                    'method': best_precision[0],
                    'improvement': best_precision[1].precision_improvement
                }
            }
        
        # Overall statistics
        if fp_reductions:
            report['overall_statistics'] = {
                'mean_fp_reduction': np.mean(fp_reductions),
                'std_fp_reduction': np.std(fp_reductions),
                'mean_sensitivity_change': np.mean(sensitivity_changes),
                'std_sensitivity_change': np.std(sensitivity_changes),
                'mean_precision_improvement': np.mean(precision_improvements),
                'std_precision_improvement': np.std(precision_improvements)
            }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """
        Save performance report to JSON file.
        
        Args:
            report: Performance report from generate_performance_report
            output_path: Path to save the report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Performance report saved to {output_path}")


def create_test_metrics() -> Tuple[List[DetectionResults], List[AnnotationCase]]:
    """
    Create test data for metrics validation.
    
    Returns:
        Tuple of (detection_results, annotation_cases)
    """
    # Create sample detection results
    detection_results = []
    
    # Case 1: Perfect detection
    case1_detections = DetectionResults(
        bounding_boxes=np.array([[50, 50, 50, 10, 10, 10]]),
        confidence_scores=np.array([0.9]),
        case_id="case_001"
    )
    detection_results.append(case1_detections)
    
    # Case 2: Mixed results
    case2_detections = DetectionResults(
        bounding_boxes=np.array([
            [30, 30, 30, 8, 8, 8],    # TP
            [100, 100, 100, 5, 5, 5], # FP
            [70, 70, 70, 12, 12, 12]  # TP
        ]),
        confidence_scores=np.array([0.85, 0.75, 0.95]),
        case_id="case_002"
    )
    detection_results.append(case2_detections)
    
    # Create corresponding annotations
    from ..data.annotation_parser import create_sample_annotations
    annotation_cases = create_sample_annotations()
    
    return detection_results, annotation_cases


if __name__ == "__main__":
    # Test the metrics implementation
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    detection_results, annotation_cases = create_test_metrics()
    
    # Initialize metrics calculator
    calculator = MetricsCalculator(iou_threshold=0.1, confidence_threshold=0.8)
    
    # Calculate metrics
    metrics = calculator.calculate_dataset_metrics(detection_results, annotation_cases)
    
    print("Test Metrics Results:")
    print(f"Sensitivity: {metrics.sensitivity:.3f}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"F1 Score: {metrics.f1_score:.3f}")
    print(f"False Positives: {metrics.false_positives}")
    print(f"True Positives: {metrics.true_positives}")
    print(f"False Negatives: {metrics.false_negatives}")