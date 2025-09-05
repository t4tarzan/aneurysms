"""
Performance Evaluator for Aneurysm Detection Post-processing

This module provides the main evaluation pipeline that orchestrates comprehensive
performance analysis of aneurysm detection systems with anatomical post-processing.
It integrates metrics calculation, false positive analysis, and method comparison
to generate complete evaluation reports as described in the paper.

Key Features:
- Complete evaluation pipeline orchestration
- Integration of metrics calculation and FP analysis
- Multi-method performance comparison
- Comprehensive reporting and visualization
- Dataset-level and case-level analysis
- Statistical significance testing
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Internal imports
from .metrics import (
    MetricsCalculator, PerformanceAnalyzer, DetectionMetrics, 
    ComparisonMetrics, create_test_metrics
)
from .fp_analyzer import (
    FalsePositiveAnalyzer, FPAnalysisResults, 
    create_test_fp_analysis
)
from ..postprocessing.anatomical_filter import (
    AnatomicalFilter, DetectionResults, FilteredResults, 
    AnatomicalMasks, create_test_scenario
)
from ..data.annotation_parser import AnnotationCase, BoundingBox3D


@dataclass
class EvaluationConfig:
    """Configuration for performance evaluation"""
    iou_threshold: float = 0.1
    confidence_threshold: float = 0.8
    voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)
    enable_fp_analysis: bool = True
    enable_visualization: bool = True
    save_detailed_results: bool = True
    output_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.iou_threshold <= 0 or self.iou_threshold > 1:
            raise ValueError("IoU threshold must be between 0 and 1")
        if self.confidence_threshold <= 0 or self.confidence_threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        if len(self.voxel_spacing) != 3 or any(v <= 0 for v in self.voxel_spacing):
            raise ValueError("Voxel spacing must be 3 positive values")


@dataclass
class CaseEvaluationResults:
    """Results for a single case evaluation"""
    case_id: str
    original_metrics: DetectionMetrics
    method_results: Dict[str, DetectionMetrics] = field(default_factory=dict)
    method_comparisons: Dict[str, ComparisonMetrics] = field(default_factory=dict)
    fp_analysis: Optional[FPAnalysisResults] = None
    processing_time: float = 0.0
    
    def get_best_method(self) -> Tuple[str, DetectionMetrics]:
        """Get the method with best F1 score"""
        if not self.method_results:
            return "original", self.original_metrics
        
        best_method = "original"
        best_f1 = self.original_metrics.f1_score
        
        for method_name, metrics in self.method_results.items():
            if metrics.f1_score > best_f1:
                best_method = method_name
                best_f1 = metrics.f1_score
        
        return best_method, self.method_results.get(best_method, self.original_metrics)


@dataclass
class DatasetEvaluationResults:
    """Results for complete dataset evaluation"""
    dataset_name: str
    total_cases: int
    case_results: List[CaseEvaluationResults] = field(default_factory=list)
    aggregated_metrics: Dict[str, DetectionMetrics] = field(default_factory=dict)
    aggregated_comparisons: Dict[str, ComparisonMetrics] = field(default_factory=dict)
    fp_analysis_summary: Optional[Dict[str, Any]] = None
    evaluation_time: float = 0.0
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the dataset"""
        if not self.case_results:
            return {}
        
        # Calculate method performance statistics
        method_stats = defaultdict(list)
        for case_result in self.case_results:
            method_stats["original"].append(case_result.original_metrics.f1_score)
            for method_name, metrics in case_result.method_results.items():
                method_stats[method_name].append(metrics.f1_score)
        
        # Calculate statistics
        summary = {}
        for method_name, f1_scores in method_stats.items():
            summary[method_name] = {
                "mean_f1": np.mean(f1_scores),
                "std_f1": np.std(f1_scores),
                "median_f1": np.median(f1_scores),
                "min_f1": np.min(f1_scores),
                "max_f1": np.max(f1_scores)
            }
        
        return summary


class PerformanceEvaluator:
    """
    Main performance evaluation orchestrator for aneurysm detection post-processing.
    
    This class coordinates the complete evaluation pipeline including:
    - Metrics calculation for original and post-processed results
    - False positive analysis and categorization
    - Method comparison and statistical analysis
    - Report generation and visualization
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize the performance evaluator.
        
        Args:
            config: Evaluation configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize component analyzers
        self.metrics_calculator = MetricsCalculator(
            iou_threshold=config.iou_threshold,
            confidence_threshold=config.confidence_threshold,
            voxel_spacing=config.voxel_spacing
        )
        
        self.performance_analyzer = PerformanceAnalyzer(self.metrics_calculator)
        
        if config.enable_fp_analysis:
            self.fp_analyzer = FalsePositiveAnalyzer(
                iou_threshold=config.iou_threshold,
                voxel_spacing=config.voxel_spacing
            )
        else:
            self.fp_analyzer = None
        
        # Initialize anatomical filter
        self.anatomical_filter = AnatomicalFilter(
            confidence_threshold=config.confidence_threshold,
            voxel_spacing=config.voxel_spacing
        )
        
        # Setup output directory
        if config.output_dir:
            self.output_dir = Path(config.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path("evaluation_results")
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_single_case(
        self,
        detections: DetectionResults,
        ground_truth: List[AnnotationCase],
        anatomical_masks: AnatomicalMasks,
        case_id: str = "unknown"
    ) -> CaseEvaluationResults:
        """
        Evaluate a single case with all post-processing methods.
        
        Args:
            detections: Original detection results
            ground_truth: Ground truth annotations
            anatomical_masks: Anatomical masks for post-processing
            case_id: Case identifier
            
        Returns:
            Complete evaluation results for the case
        """
        start_time = time.time()
        
        self.logger.info(f"Evaluating case: {case_id}")
        
        # Calculate original metrics
        original_metrics = self.metrics_calculator.calculate_case_metrics(
            detections, ground_truth
        )
        
        # Apply post-processing methods
        filtered_results = self.anatomical_filter.filter_detections(
            detections, anatomical_masks
        )
        
        # Calculate metrics for each method
        method_results = {}
        method_comparisons = {}
        
        for method_name, method_detections in filtered_results.filtered_detections.items():
            # Calculate post-processed metrics
            method_metrics = self.metrics_calculator.calculate_case_metrics(
                method_detections, ground_truth
            )
            method_results[method_name] = method_metrics
            
            # Calculate comparison metrics
            comparison = self.metrics_calculator.compare_before_after_processing(
                original_metrics, method_metrics, method_name
            )
            method_comparisons[method_name] = comparison
        
        # Perform false positive analysis if enabled
        fp_analysis = None
        if self.fp_analyzer and self.config.enable_fp_analysis:
            try:
                fp_analysis = self.fp_analyzer.analyze_false_positives(
                    detections, ground_truth, anatomical_masks, case_id
                )
            except Exception as e:
                self.logger.warning(f"FP analysis failed for case {case_id}: {e}")
        
        processing_time = time.time() - start_time
        
        return CaseEvaluationResults(
            case_id=case_id,
            original_metrics=original_metrics,
            method_results=method_results,
            method_comparisons=method_comparisons,
            fp_analysis=fp_analysis,
            processing_time=processing_time
        )
    
    def evaluate_dataset(
        self,
        detection_cases: List[DetectionResults],
        ground_truth_cases: List[List[AnnotationCase]],
        anatomical_masks_cases: List[AnatomicalMasks],
        dataset_name: str = "unknown_dataset"
    ) -> DatasetEvaluationResults:
        """
        Evaluate complete dataset with all cases.
        
        Args:
            detection_cases: List of detection results for each case
            ground_truth_cases: List of ground truth annotations for each case
            anatomical_masks_cases: List of anatomical masks for each case
            dataset_name: Dataset identifier
            
        Returns:
            Complete dataset evaluation results
        """
        start_time = time.time()
        
        self.logger.info(f"Evaluating dataset: {dataset_name}")
        self.logger.info(f"Total cases: {len(detection_cases)}")
        
        if not (len(detection_cases) == len(ground_truth_cases) == len(anatomical_masks_cases)):
            raise ValueError("All input lists must have the same length")
        
        # Evaluate each case
        case_results = []
        for i, (detections, ground_truth, masks) in enumerate(
            zip(detection_cases, ground_truth_cases, anatomical_masks_cases)
        ):
            case_id = detections.case_id or f"case_{i:04d}"
            case_result = self.evaluate_single_case(
                detections, ground_truth, masks, case_id
            )
            case_results.append(case_result)
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{len(detection_cases)} cases")
        
        # Calculate aggregated metrics
        aggregated_metrics = self._calculate_aggregated_metrics(case_results)
        aggregated_comparisons = self._calculate_aggregated_comparisons(case_results)
        
        # Aggregate FP analysis if available
        fp_analysis_summary = None
        if self.fp_analyzer and self.config.enable_fp_analysis:
            fp_analysis_summary = self._aggregate_fp_analysis(case_results)
        
        evaluation_time = time.time() - start_time
        
        return DatasetEvaluationResults(
            dataset_name=dataset_name,
            total_cases=len(detection_cases),
            case_results=case_results,
            aggregated_metrics=aggregated_metrics,
            aggregated_comparisons=aggregated_comparisons,
            fp_analysis_summary=fp_analysis_summary,
            evaluation_time=evaluation_time
        )
    
    def _calculate_aggregated_metrics(
        self, 
        case_results: List[CaseEvaluationResults]
    ) -> Dict[str, DetectionMetrics]:
        """Calculate aggregated metrics across all cases"""
        # Collect all method names
        all_methods = set(["original"])
        for case_result in case_results:
            all_methods.update(case_result.method_results.keys())
        
        aggregated = {}
        
        for method_name in all_methods:
            total_tp = total_fp = total_fn = 0
            total_gt = total_det = 0
            
            for case_result in case_results:
                if method_name == "original":
                    metrics = case_result.original_metrics
                else:
                    metrics = case_result.method_results.get(method_name)
                    if metrics is None:
                        continue
                
                total_tp += metrics.true_positives
                total_fp += metrics.false_positives
                total_fn += metrics.false_negatives
                total_gt += metrics.total_ground_truth
                total_det += metrics.total_detections
            
            aggregated[method_name] = DetectionMetrics(
                true_positives=total_tp,
                false_positives=total_fp,
                false_negatives=total_fn,
                total_ground_truth=total_gt,
                total_detections=total_det
            )
        
        return aggregated
    
    def _calculate_aggregated_comparisons(
        self, 
        case_results: List[CaseEvaluationResults]
    ) -> Dict[str, ComparisonMetrics]:
        """Calculate aggregated comparison metrics"""
        aggregated_comparisons = {}
        
        # Get all method names
        all_methods = set()
        for case_result in case_results:
            all_methods.update(case_result.method_comparisons.keys())
        
        for method_name in all_methods:
            # Get aggregated metrics for comparison
            original_aggregated = None
            method_aggregated = None
            
            for method, metrics in self._calculate_aggregated_metrics(case_results).items():
                if method == "original":
                    original_aggregated = metrics
                elif method == method_name:
                    method_aggregated = metrics
            
            if original_aggregated and method_aggregated:
                aggregated_comparisons[method_name] = ComparisonMetrics(
                    before_processing=original_aggregated,
                    after_processing=method_aggregated,
                    method_name=method_name
                )
        
        return aggregated_comparisons
    
    def _aggregate_fp_analysis(
        self, 
        case_results: List[CaseEvaluationResults]
    ) -> Dict[str, Any]:
        """Aggregate false positive analysis results"""
        if not case_results or not case_results[0].fp_analysis:
            return {}
        
        # Collect FP analysis data
        total_fps = 0
        fps_by_location = defaultdict(int)
        reduction_effectiveness = defaultdict(list)
        
        for case_result in case_results:
            if case_result.fp_analysis:
                fp_analysis = case_result.fp_analysis
                total_fps += fp_analysis.total_fps
                
                for location, count in fp_analysis.fps_by_location.items():
                    fps_by_location[location] += count
                
                for method, effectiveness in fp_analysis.reduction_effectiveness.items():
                    reduction_effectiveness[method].append(effectiveness)
        
        # Calculate average reduction effectiveness
        avg_reduction_effectiveness = {}
        for method, values in reduction_effectiveness.items():
            if values:
                avg_reduction_effectiveness[method] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values)
                }
        
        return {
            "total_fps": total_fps,
            "fps_by_location": dict(fps_by_location),
            "avg_reduction_effectiveness": avg_reduction_effectiveness
        }
    
    def generate_evaluation_report(
        self, 
        results: DatasetEvaluationResults,
        save_to_file: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Dataset evaluation results
            save_to_file: Whether to save report to file
            
        Returns:
            Complete evaluation report
        """
        self.logger.info("Generating evaluation report...")
        
        report = {
            "dataset_info": {
                "name": results.dataset_name,
                "total_cases": results.total_cases,
                "evaluation_time": results.evaluation_time,
                "config": {
                    "iou_threshold": self.config.iou_threshold,
                    "confidence_threshold": self.config.confidence_threshold,
                    "voxel_spacing": self.config.voxel_spacing
                }
            },
            "aggregated_metrics": {},
            "method_comparisons": {},
            "summary_statistics": results.get_summary_statistics(),
            "fp_analysis": results.fp_analysis_summary
        }
        
        # Add aggregated metrics
        for method_name, metrics in results.aggregated_metrics.items():
            report["aggregated_metrics"][method_name] = {
                "sensitivity": metrics.sensitivity,
                "precision": metrics.precision,
                "f1_score": metrics.f1_score,
                "true_positives": metrics.true_positives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives
            }
        
        # Add method comparisons
        for method_name, comparison in results.aggregated_comparisons.items():
            report["method_comparisons"][method_name] = {
                "fp_reduction": comparison.fp_reduction,
                "fp_reduction_rate": comparison.fp_reduction_rate,
                "sensitivity_change": comparison.sensitivity_change,
                "precision_change": comparison.precision_change,
                "f1_change": comparison.f1_change
            }
        
        # Save to file if requested
        if save_to_file and self.config.save_detailed_results:
            report_path = self.output_dir / f"{results.dataset_name}_evaluation_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Report saved to: {report_path}")
        
        return report
    
    def create_visualizations(
        self, 
        results: DatasetEvaluationResults,
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Create evaluation visualizations.
        
        Args:
            results: Dataset evaluation results
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary of plot information
        """
        if not self.config.enable_visualization:
            return {}
        
        self.logger.info("Creating evaluation visualizations...")
        
        plots_info = {}
        
        try:
            # 1. Method comparison bar plot
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            methods = list(results.aggregated_metrics.keys())
            f1_scores = [results.aggregated_metrics[m].f1_score for m in methods]
            
            bars = ax.bar(methods, f1_scores, alpha=0.7)
            ax.set_ylabel('F1 Score')
            ax.set_title('Method Performance Comparison')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, score in zip(bars, f1_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_plots:
                plot_path = self.output_dir / f"{results.dataset_name}_method_comparison.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plots_info["method_comparison"] = str(plot_path)
            
            plt.close()
            
            # 2. False positive reduction plot
            if results.fp_analysis_summary:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                
                reduction_data = results.fp_analysis_summary.get("avg_reduction_effectiveness", {})
                if reduction_data:
                    methods = list(reduction_data.keys())
                    reductions = [reduction_data[m]["mean"] for m in methods]
                    errors = [reduction_data[m]["std"] for m in methods]
                    
                    bars = ax.bar(methods, reductions, yerr=errors, alpha=0.7, capsize=5)
                    ax.set_ylabel('FP Reduction Rate')
                    ax.set_title('False Positive Reduction Effectiveness')
                    ax.set_ylim(0, 1)
                    
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    if save_plots:
                        plot_path = self.output_dir / f"{results.dataset_name}_fp_reduction.png"
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plots_info["fp_reduction"] = str(plot_path)
                    
                    plt.close()
            
            self.logger.info(f"Created {len(plots_info)} visualizations")
            
        except Exception as e:
            self.logger.warning(f"Visualization creation failed: {e}")
        
        return plots_info


def create_test_evaluation() -> Tuple[List[DetectionResults], List[List[AnnotationCase]], List[AnatomicalMasks]]:
    """
    Create test data for performance evaluation validation.
    
    Returns:
        Tuple of (detection_cases, ground_truth_cases, anatomical_masks_cases)
    """
    # Create test detection results
    detection_cases = []
    ground_truth_cases = []
    anatomical_masks_cases = []
    
    for i in range(3):  # Create 3 test cases
        # Create detection results
        detections = DetectionResults(
            bounding_boxes=[
                BoundingBox3D(10 + i*5, 10 + i*5, 10 + i*5, 20, 20, 20),
                BoundingBox3D(50 + i*5, 50 + i*5, 50 + i*5, 15, 15, 15),
                BoundingBox3D(80 + i*5, 80 + i*5, 80 + i*5, 25, 25, 25)
            ],
            confidence_scores=[0.9, 0.85, 0.75],
            case_id=f"test_case_{i:03d}"
        )
        detection_cases.append(detections)
        
        # Create ground truth
        ground_truth = [
            AnnotationCase(
                case_id=f"test_case_{i:03d}",
                aneurysm_bboxes=[
                    BoundingBox3D(12 + i*5, 12 + i*5, 12 + i*5, 18, 18, 18),
                    BoundingBox3D(52 + i*5, 52 + i*5, 52 + i*5, 13, 13, 13)
                ]
            )
        ]
        ground_truth_cases.append(ground_truth)
        
        # Create anatomical masks
        brain_mask = np.ones((100, 100, 100), dtype=bool)
        artery_mask = np.zeros((100, 100, 100), dtype=bool)
        artery_mask[10:30, 10:30, 10:30] = True
        artery_mask[50:70, 50:70, 50:70] = True
        
        vein_mask = np.zeros((100, 100, 100), dtype=bool)
        vein_mask[80:100, 80:100, 80:100] = True
        
        masks = AnatomicalMasks(
            brain_mask=brain_mask,
            artery_mask=artery_mask,
            vein_mask=vein_mask
        )
        anatomical_masks_cases.append(masks)
    
    return detection_cases, ground_truth_cases, anatomical_masks_cases


def main():
    """Main function for testing the performance evaluator"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Testing Performance Evaluator...")
    
    try:
        # Create test configuration
        config = EvaluationConfig(
            iou_threshold=0.1,
            confidence_threshold=0.8,
            enable_fp_analysis=True,
            enable_visualization=True,
            output_dir=Path("test_evaluation_output")
        )
        
        # Initialize evaluator
        evaluator = PerformanceEvaluator(config)
        
        # Create test data
        detection_cases, ground_truth_cases, anatomical_masks_cases = create_test_evaluation()
        
        logger.info(f"Created test data: {len(detection_cases)} cases")
        
        # Run evaluation
        results = evaluator.evaluate_dataset(
            detection_cases, 
            ground_truth_cases, 
            anatomical_masks_cases,
            "test_dataset"
        )
        
        logger.info(f"Evaluation completed in {results.evaluation_time:.2f} seconds")
        
        # Generate report
        report = evaluator.generate_evaluation_report(results)
        
        # Create visualizations
        plots = evaluator.create_visualizations(results)
        
        logger.info("Performance Evaluator test completed successfully!")
        logger.info(f"Results: {len(results.case_results)} cases evaluated")
        logger.info(f"Report keys: {list(report.keys())}")
        logger.info(f"Plots created: {list(plots.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance Evaluator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)