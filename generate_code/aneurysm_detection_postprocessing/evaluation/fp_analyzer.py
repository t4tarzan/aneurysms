"""
False Positive Analysis Module for Intracranial Aneurysm Detection

This module provides comprehensive analysis and categorization of false positive detections
to understand the effectiveness of anatomical post-processing methods. It analyzes FP
patterns, anatomical locations, and provides insights for method improvement.

Key Features:
- Categorizes false positives by anatomical location
- Analyzes FP reduction patterns across post-processing methods
- Provides detailed statistics and visualization data
- Supports batch analysis across multiple cases
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from collections import defaultdict, Counter

from ..data.annotation_parser import BoundingBox3D, AnnotationCase
from ..postprocessing.anatomical_filter import DetectionResults, FilteredResults, AnatomicalMasks
from .metrics import DetectionMetrics, ComparisonMetrics, DetectionMatcher


@dataclass
class FalsePositiveCase:
    """Represents a single false positive detection with analysis metadata"""
    detection_bbox: BoundingBox3D
    confidence_score: float
    case_id: str
    anatomical_location: str = "unknown"
    removed_by_methods: List[str] = field(default_factory=list)
    overlap_with_brain: float = 0.0
    overlap_with_artery: float = 0.0
    overlap_with_vein: float = 0.0
    overlap_with_cvs: float = 0.0
    distance_to_nearest_tp: float = float('inf')
    
    def __post_init__(self):
        """Validate false positive case data"""
        if not isinstance(self.detection_bbox, BoundingBox3D):
            raise ValueError("detection_bbox must be BoundingBox3D instance")
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")


@dataclass
class FPAnalysisResults:
    """Container for false positive analysis results"""
    total_fps: int
    fps_by_location: Dict[str, int]
    fps_by_method: Dict[str, int]
    reduction_effectiveness: Dict[str, float]
    anatomical_distribution: Dict[str, float]
    confidence_distribution: List[float]
    overlap_statistics: Dict[str, Dict[str, float]]
    method_comparison: Dict[str, Dict[str, Any]]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of FP analysis"""
        return {
            "total_false_positives": self.total_fps,
            "location_distribution": self.fps_by_location,
            "method_effectiveness": self.reduction_effectiveness,
            "anatomical_percentages": self.anatomical_distribution,
            "overlap_stats": self.overlap_statistics,
            "method_comparison": self.method_comparison
        }


class AnatomicalLocationClassifier:
    """Classifies false positives by anatomical location based on overlap patterns"""
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)):
        self.voxel_spacing = voxel_spacing
        self.logger = logging.getLogger(__name__)
    
    def classify_location(self, fp_case: FalsePositiveCase, masks: AnatomicalMasks) -> str:
        """
        Classify anatomical location of false positive
        
        Args:
            fp_case: False positive case to classify
            masks: Anatomical masks for classification
            
        Returns:
            Anatomical location category
        """
        try:
            # Calculate overlaps with different anatomical structures
            brain_overlap = fp_case.overlap_with_brain
            artery_overlap = fp_case.overlap_with_artery
            vein_overlap = fp_case.overlap_with_vein
            cvs_overlap = fp_case.overlap_with_cvs
            
            # Classification logic based on overlap patterns
            if brain_overlap < 0.1:
                return "extracranial"
            elif cvs_overlap > 0.3:
                return "cavernous_sinus"
            elif vein_overlap > 0.5 and artery_overlap < 0.2:
                return "venous_structure"
            elif artery_overlap > 0.3 and vein_overlap < artery_overlap:
                return "arterial_structure"
            elif brain_overlap > 0.8 and artery_overlap < 0.1 and vein_overlap < 0.1:
                return "brain_parenchyma"
            else:
                return "mixed_structure"
                
        except Exception as e:
            self.logger.error(f"Error classifying FP location: {e}")
            return "unknown"
    
    def batch_classify(self, fp_cases: List[FalsePositiveCase], 
                      masks: AnatomicalMasks) -> Dict[str, str]:
        """
        Classify multiple false positive cases
        
        Args:
            fp_cases: List of false positive cases
            masks: Anatomical masks
            
        Returns:
            Dictionary mapping case IDs to locations
        """
        classifications = {}
        for fp_case in fp_cases:
            location = self.classify_location(fp_case, masks)
            classifications[f"{fp_case.case_id}_{fp_case.detection_bbox.annotation_id}"] = location
            fp_case.anatomical_location = location
        
        return classifications


class FPReductionAnalyzer:
    """Analyzes false positive reduction effectiveness across different methods"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_method_effectiveness(self, original_fps: List[FalsePositiveCase],
                                   filtered_results: Dict[str, FilteredResults]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze effectiveness of each post-processing method
        
        Args:
            original_fps: Original false positive cases
            filtered_results: Results after applying each method
            
        Returns:
            Method effectiveness analysis
        """
        method_analysis = {}
        
        for method_name, results in filtered_results.items():
            try:
                # Calculate reduction statistics
                original_count = len(original_fps)
                remaining_count = len(results.filtered_detections.get(method_name, []))
                reduction_count = original_count - remaining_count
                reduction_rate = reduction_count / original_count if original_count > 0 else 0.0
                
                # Analyze which FPs were removed
                removed_fps = self._identify_removed_fps(original_fps, results, method_name)
                
                # Categorize removed FPs by location
                location_distribution = Counter([fp.anatomical_location for fp in removed_fps])
                
                method_analysis[method_name] = {
                    "total_fps_removed": reduction_count,
                    "reduction_rate": reduction_rate,
                    "remaining_fps": remaining_count,
                    "removed_by_location": dict(location_distribution),
                    "effectiveness_score": self._calculate_effectiveness_score(
                        reduction_rate, location_distribution
                    )
                }
                
            except Exception as e:
                self.logger.error(f"Error analyzing method {method_name}: {e}")
                method_analysis[method_name] = {"error": str(e)}
        
        return method_analysis
    
    def _identify_removed_fps(self, original_fps: List[FalsePositiveCase],
                             results: FilteredResults, method_name: str) -> List[FalsePositiveCase]:
        """Identify which FPs were removed by a specific method"""
        # This is a simplified implementation - in practice, you'd need to match
        # bounding boxes between original and filtered results
        removed_fps = []
        
        try:
            # Get remaining detections after filtering
            remaining_detections = results.filtered_detections.get(method_name, [])
            remaining_bbox_ids = {bbox.annotation_id for bbox in remaining_detections}
            
            # Find FPs that were removed
            for fp in original_fps:
                if fp.detection_bbox.annotation_id not in remaining_bbox_ids:
                    fp.removed_by_methods.append(method_name)
                    removed_fps.append(fp)
                    
        except Exception as e:
            self.logger.error(f"Error identifying removed FPs for {method_name}: {e}")
        
        return removed_fps
    
    def _calculate_effectiveness_score(self, reduction_rate: float,
                                     location_distribution: Counter) -> float:
        """Calculate overall effectiveness score for a method"""
        # Weight reduction rate by quality of removals
        # Higher score for removing extracranial and venous FPs
        quality_weights = {
            "extracranial": 1.0,
            "venous_structure": 0.9,
            "cavernous_sinus": 0.8,
            "mixed_structure": 0.6,
            "arterial_structure": 0.4,  # Lower weight as these might be near real aneurysms
            "brain_parenchyma": 0.3,
            "unknown": 0.5
        }
        
        if not location_distribution:
            return reduction_rate
        
        total_removed = sum(location_distribution.values())
        weighted_score = sum(
            count * quality_weights.get(location, 0.5) 
            for location, count in location_distribution.items()
        ) / total_removed
        
        return reduction_rate * weighted_score


class FalsePositiveAnalyzer:
    """Main class for comprehensive false positive analysis"""
    
    def __init__(self, iou_threshold: float = 0.1,
                 voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)):
        self.iou_threshold = iou_threshold
        self.voxel_spacing = voxel_spacing
        self.location_classifier = AnatomicalLocationClassifier(voxel_spacing)
        self.reduction_analyzer = FPReductionAnalyzer()
        self.detection_matcher = DetectionMatcher(iou_threshold, voxel_spacing)
        self.logger = logging.getLogger(__name__)
    
    def analyze_false_positives(self, detection_results: DetectionResults,
                               ground_truth: AnnotationCase,
                               anatomical_masks: AnatomicalMasks,
                               filtered_results: Optional[Dict[str, FilteredResults]] = None) -> FPAnalysisResults:
        """
        Comprehensive analysis of false positive detections
        
        Args:
            detection_results: Original detection results
            ground_truth: Ground truth annotations
            anatomical_masks: Anatomical masks for analysis
            filtered_results: Results after post-processing (optional)
            
        Returns:
            Comprehensive FP analysis results
        """
        try:
            # Identify false positives using detection matcher
            fp_cases = self._identify_false_positives(
                detection_results, ground_truth, anatomical_masks
            )
            
            # Classify FPs by anatomical location
            self.location_classifier.batch_classify(fp_cases, anatomical_masks)
            
            # Analyze location distribution
            location_counts = Counter([fp.anatomical_location for fp in fp_cases])
            total_fps = len(fp_cases)
            location_percentages = {
                loc: count / total_fps * 100 if total_fps > 0 else 0
                for loc, count in location_counts.items()
            }
            
            # Analyze method effectiveness if filtered results provided
            method_analysis = {}
            if filtered_results:
                method_analysis = self.reduction_analyzer.analyze_method_effectiveness(
                    fp_cases, filtered_results
                )
            
            # Calculate overlap statistics
            overlap_stats = self._calculate_overlap_statistics(fp_cases)
            
            # Create analysis results
            results = FPAnalysisResults(
                total_fps=total_fps,
                fps_by_location=dict(location_counts),
                fps_by_method={method: data.get("total_fps_removed", 0) 
                              for method, data in method_analysis.items()},
                reduction_effectiveness={method: data.get("effectiveness_score", 0.0)
                                       for method, data in method_analysis.items()},
                anatomical_distribution=location_percentages,
                confidence_distribution=[fp.confidence_score for fp in fp_cases],
                overlap_statistics=overlap_stats,
                method_comparison=method_analysis
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in FP analysis: {e}")
            raise
    
    def _identify_false_positives(self, detection_results: DetectionResults,
                                 ground_truth: AnnotationCase,
                                 anatomical_masks: AnatomicalMasks) -> List[FalsePositiveCase]:
        """Identify false positive detections"""
        fp_cases = []
        
        try:
            # Use detection matcher to find unmatched detections (FPs)
            matches = self.detection_matcher.match_detections(
                detection_results.bounding_boxes,
                ground_truth.bounding_boxes,
                detection_results.confidence_scores
            )
            
            # Find unmatched detections (these are false positives)
            matched_detection_indices = set(match[0] for match in matches if match[0] is not None)
            
            for i, bbox in enumerate(detection_results.bounding_boxes):
                if i not in matched_detection_indices:
                    # This is a false positive
                    fp_case = FalsePositiveCase(
                        detection_bbox=bbox,
                        confidence_score=detection_results.confidence_scores[i],
                        case_id=detection_results.case_id or "unknown"
                    )
                    
                    # Calculate overlaps with anatomical structures
                    fp_case.overlap_with_brain = self._calculate_bbox_mask_overlap(
                        bbox, anatomical_masks.brain_mask
                    )
                    fp_case.overlap_with_artery = self._calculate_bbox_mask_overlap(
                        bbox, anatomical_masks.artery_mask
                    )
                    fp_case.overlap_with_vein = self._calculate_bbox_mask_overlap(
                        bbox, anatomical_masks.vein_mask
                    )
                    if anatomical_masks.cvs_mask is not None:
                        fp_case.overlap_with_cvs = self._calculate_bbox_mask_overlap(
                            bbox, anatomical_masks.cvs_mask
                        )
                    
                    fp_cases.append(fp_case)
                    
        except Exception as e:
            self.logger.error(f"Error identifying false positives: {e}")
        
        return fp_cases
    
    def _calculate_bbox_mask_overlap(self, bbox: BoundingBox3D, mask: np.ndarray) -> float:
        """Calculate overlap ratio between bounding box and mask"""
        try:
            # Convert bbox to voxel coordinates
            x_min = max(0, int(bbox.x - bbox.width/2))
            x_max = min(mask.shape[0], int(bbox.x + bbox.width/2))
            y_min = max(0, int(bbox.y - bbox.height/2))
            y_max = min(mask.shape[1], int(bbox.y + bbox.height/2))
            z_min = max(0, int(bbox.z - bbox.depth/2))
            z_max = min(mask.shape[2], int(bbox.z + bbox.depth/2))
            
            # Extract bbox region from mask
            bbox_region = mask[x_min:x_max, y_min:y_max, z_min:z_max]
            
            # Calculate overlap ratio
            overlap_voxels = np.sum(bbox_region > 0)
            total_voxels = bbox_region.size
            
            return overlap_voxels / total_voxels if total_voxels > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating bbox-mask overlap: {e}")
            return 0.0
    
    def _calculate_overlap_statistics(self, fp_cases: List[FalsePositiveCase]) -> Dict[str, Dict[str, float]]:
        """Calculate overlap statistics for false positives"""
        if not fp_cases:
            return {}
        
        overlap_data = {
            "brain": [fp.overlap_with_brain for fp in fp_cases],
            "artery": [fp.overlap_with_artery for fp in fp_cases],
            "vein": [fp.overlap_with_vein for fp in fp_cases],
            "cvs": [fp.overlap_with_cvs for fp in fp_cases if fp.overlap_with_cvs > 0]
        }
        
        statistics = {}
        for structure, overlaps in overlap_data.items():
            if overlaps:
                statistics[structure] = {
                    "mean": float(np.mean(overlaps)),
                    "std": float(np.std(overlaps)),
                    "median": float(np.median(overlaps)),
                    "min": float(np.min(overlaps)),
                    "max": float(np.max(overlaps))
                }
        
        return statistics
    
    def batch_analyze(self, detection_cases: List[DetectionResults],
                     ground_truth_cases: List[AnnotationCase],
                     anatomical_masks_cases: List[AnatomicalMasks],
                     filtered_results_cases: Optional[List[Dict[str, FilteredResults]]] = None) -> Dict[str, FPAnalysisResults]:
        """
        Analyze false positives across multiple cases
        
        Args:
            detection_cases: List of detection results
            ground_truth_cases: List of ground truth annotations
            anatomical_masks_cases: List of anatomical masks
            filtered_results_cases: List of filtered results (optional)
            
        Returns:
            Dictionary of analysis results per case
        """
        results = {}
        
        for i, (detections, gt, masks) in enumerate(zip(
            detection_cases, ground_truth_cases, anatomical_masks_cases
        )):
            case_id = detections.case_id or f"case_{i}"
            filtered_results = None
            if filtered_results_cases and i < len(filtered_results_cases):
                filtered_results = filtered_results_cases[i]
            
            try:
                analysis = self.analyze_false_positives(
                    detections, gt, masks, filtered_results
                )
                results[case_id] = analysis
                
            except Exception as e:
                self.logger.error(f"Error analyzing case {case_id}: {e}")
                continue
        
        return results
    
    def generate_analysis_report(self, analysis_results: Dict[str, FPAnalysisResults],
                               output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report
        
        Args:
            analysis_results: Analysis results from batch_analyze
            output_path: Optional path to save report
            
        Returns:
            Comprehensive analysis report
        """
        # Aggregate statistics across all cases
        total_fps = sum(result.total_fps for result in analysis_results.values())
        
        # Aggregate location distributions
        aggregated_locations = defaultdict(int)
        for result in analysis_results.values():
            for location, count in result.fps_by_location.items():
                aggregated_locations[location] += count
        
        # Calculate average method effectiveness
        method_effectiveness = defaultdict(list)
        for result in analysis_results.values():
            for method, score in result.reduction_effectiveness.items():
                method_effectiveness[method].append(score)
        
        avg_effectiveness = {
            method: np.mean(scores) if scores else 0.0
            for method, scores in method_effectiveness.items()
        }
        
        # Create comprehensive report
        report = {
            "summary": {
                "total_cases_analyzed": len(analysis_results),
                "total_false_positives": total_fps,
                "average_fps_per_case": total_fps / len(analysis_results) if analysis_results else 0,
                "location_distribution": dict(aggregated_locations),
                "method_effectiveness": avg_effectiveness
            },
            "per_case_results": {
                case_id: result.get_summary()
                for case_id, result in analysis_results.items()
            },
            "recommendations": self._generate_recommendations(analysis_results)
        }
        
        # Save report if path provided
        if output_path:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                self.logger.info(f"Analysis report saved to {output_path}")
            except Exception as e:
                self.logger.error(f"Error saving report: {e}")
        
        return report
    
    def _generate_recommendations(self, analysis_results: Dict[str, FPAnalysisResults]) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Analyze overall patterns
        total_fps = sum(result.total_fps for result in analysis_results.values())
        if total_fps == 0:
            return ["No false positives detected - excellent performance!"]
        
        # Location-based recommendations
        location_counts = defaultdict(int)
        for result in analysis_results.values():
            for location, count in result.fps_by_location.items():
                location_counts[location] += count
        
        dominant_location = max(location_counts, key=location_counts.get)
        
        if dominant_location == "extracranial":
            recommendations.append("Consider improving brain mask generation or expanding dilation parameters")
        elif dominant_location == "venous_structure":
            recommendations.append("Vein segmentation appears effective - consider Method 2 or 4")
        elif dominant_location == "arterial_structure":
            recommendations.append("Many FPs in arterial regions - review artery segmentation quality")
        
        # Method effectiveness recommendations
        method_scores = defaultdict(list)
        for result in analysis_results.values():
            for method, score in result.reduction_effectiveness.items():
                method_scores[method].append(score)
        
        if method_scores:
            best_method = max(method_scores, key=lambda m: np.mean(method_scores[m]))
            recommendations.append(f"Method {best_method} shows highest effectiveness")
        
        return recommendations


def create_test_fp_analysis() -> Tuple[List[DetectionResults], List[AnnotationCase], List[AnatomicalMasks]]:
    """Create test data for FP analysis validation"""
    from ..data.annotation_parser import create_sample_annotations
    from ..postprocessing.anatomical_filter import create_test_scenario
    
    # Create sample data
    detection_results, masks = create_test_scenario()
    annotations = create_sample_annotations()
    
    return [detection_results], annotations[:1], [masks]


if __name__ == "__main__":
    # Test the FP analyzer
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create test data
        detections, annotations, masks = create_test_fp_analysis()
        
        # Initialize analyzer
        analyzer = FalsePositiveAnalyzer()
        
        # Run analysis
        results = analyzer.batch_analyze(detections, annotations, masks)
        
        # Generate report
        report = analyzer.generate_analysis_report(results)
        
        print("FP Analysis Test Results:")
        print(f"Total cases: {report['summary']['total_cases_analyzed']}")
        print(f"Total FPs: {report['summary']['total_false_positives']}")
        print(f"Location distribution: {report['summary']['location_distribution']}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise