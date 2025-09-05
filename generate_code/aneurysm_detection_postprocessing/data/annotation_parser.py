"""
3D Bounding Box Annotation Parser for Aneurysm Detection

This module handles parsing and processing of 3D bounding box annotations for intracranial
aneurysms from various annotation formats (JSON, XML, CSV) and converts them to standardized
format for training and evaluation.

Key Features:
- Support for multiple annotation formats
- 3D bounding box validation and normalization
- Coordinate system conversion (world to voxel coordinates)
- Annotation quality checks and filtering
- Ground truth preparation for training and evaluation

Author: Implementation based on paper requirements
"""

import json
import xml.etree.ElementTree as ET
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox3D:
    """
    3D bounding box representation for aneurysm annotations.
    
    Uses center-based coordinate system: [x, y, z, width, height, depth]
    where (x, y, z) is the center point and (width, height, depth) are dimensions.
    """
    x: float  # Center x coordinate
    y: float  # Center y coordinate  
    z: float  # Center z coordinate
    width: float   # Width (x dimension)
    height: float  # Height (y dimension)
    depth: float   # Depth (z dimension)
    confidence: float = 1.0  # Confidence score (1.0 for ground truth)
    label: str = "aneurysm"  # Class label
    annotation_id: Optional[str] = None  # Unique annotation identifier
    
    def __post_init__(self):
        """Validate bounding box parameters."""
        if self.width <= 0 or self.height <= 0 or self.depth <= 0:
            raise ValueError(f"Bounding box dimensions must be positive: {self.width}, {self.height}, {self.depth}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1: {self.confidence}")
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array format [x, y, z, w, h, d]."""
        return np.array([self.x, self.y, self.z, self.width, self.height, self.depth])
    
    def to_corners(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to corner coordinates (min_corner, max_corner)."""
        min_corner = np.array([
            self.x - self.width / 2,
            self.y - self.height / 2, 
            self.z - self.depth / 2
        ])
        max_corner = np.array([
            self.x + self.width / 2,
            self.y + self.height / 2,
            self.z + self.depth / 2
        ])
        return min_corner, max_corner
    
    def get_volume(self) -> float:
        """Calculate bounding box volume."""
        return self.width * self.height * self.depth
    
    def is_valid_in_volume(self, volume_shape: Tuple[int, int, int]) -> bool:
        """Check if bounding box is valid within given volume dimensions."""
        min_corner, max_corner = self.to_corners()
        return (min_corner >= 0).all() and (max_corner < np.array(volume_shape)).all()


@dataclass
class AnnotationCase:
    """Container for all annotations in a single CTA case."""
    case_id: str
    image_path: str
    bounding_boxes: List[BoundingBox3D] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    voxel_spacing: Tuple[float, float, float] = (0.4, 0.4, 0.4)  # Default spacing
    volume_shape: Optional[Tuple[int, int, int]] = None
    
    def __post_init__(self):
        """Validate annotation case."""
        if not self.case_id:
            raise ValueError("Case ID cannot be empty")
        if not self.image_path:
            raise ValueError("Image path cannot be empty")
    
    def add_bounding_box(self, bbox: BoundingBox3D) -> None:
        """Add a bounding box to this case."""
        if self.volume_shape and not bbox.is_valid_in_volume(self.volume_shape):
            logger.warning(f"Bounding box {bbox.annotation_id} is outside volume bounds for case {self.case_id}")
        self.bounding_boxes.append(bbox)
    
    def get_bounding_boxes_array(self) -> np.ndarray:
        """Get all bounding boxes as numpy array."""
        if not self.bounding_boxes:
            return np.empty((0, 6))
        return np.array([bbox.to_array() for bbox in self.bounding_boxes])
    
    def get_confidence_scores(self) -> np.ndarray:
        """Get confidence scores for all bounding boxes."""
        if not self.bounding_boxes:
            return np.empty(0)
        return np.array([bbox.confidence for bbox in self.bounding_boxes])
    
    def filter_by_confidence(self, min_confidence: float = 0.5) -> 'AnnotationCase':
        """Create new case with filtered bounding boxes by confidence."""
        filtered_boxes = [bbox for bbox in self.bounding_boxes if bbox.confidence >= min_confidence]
        new_case = AnnotationCase(
            case_id=self.case_id,
            image_path=self.image_path,
            bounding_boxes=filtered_boxes,
            metadata=self.metadata.copy(),
            voxel_spacing=self.voxel_spacing,
            volume_shape=self.volume_shape
        )
        return new_case
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about annotations in this case."""
        if not self.bounding_boxes:
            return {"num_annotations": 0}
        
        volumes = [bbox.get_volume() for bbox in self.bounding_boxes]
        confidences = [bbox.confidence for bbox in self.bounding_boxes]
        
        return {
            "num_annotations": len(self.bounding_boxes),
            "mean_volume": np.mean(volumes),
            "std_volume": np.std(volumes),
            "min_volume": np.min(volumes),
            "max_volume": np.max(volumes),
            "mean_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences)
        }


class AnnotationParser(ABC):
    """Abstract base class for annotation parsers."""
    
    @abstractmethod
    def parse_file(self, file_path: Union[str, Path]) -> List[AnnotationCase]:
        """Parse annotation file and return list of annotation cases."""
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        pass


class JSONAnnotationParser(AnnotationParser):
    """Parser for JSON format annotations."""
    
    def get_supported_extensions(self) -> List[str]:
        return ['.json']
    
    def parse_file(self, file_path: Union[str, Path]) -> List[AnnotationCase]:
        """
        Parse JSON annotation file.
        
        Expected JSON format:
        {
            "cases": [
                {
                    "case_id": "case_001",
                    "image_path": "/path/to/image.nii.gz",
                    "voxel_spacing": [0.4, 0.4, 0.4],
                    "volume_shape": [512, 512, 300],
                    "annotations": [
                        {
                            "id": "ann_001",
                            "center": [256, 256, 150],
                            "size": [10, 12, 8],
                            "confidence": 1.0,
                            "label": "aneurysm"
                        }
                    ]
                }
            ]
        }
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        cases = []
        for case_data in data.get('cases', []):
            case = AnnotationCase(
                case_id=case_data['case_id'],
                image_path=case_data['image_path'],
                voxel_spacing=tuple(case_data.get('voxel_spacing', [0.4, 0.4, 0.4])),
                volume_shape=tuple(case_data['volume_shape']) if 'volume_shape' in case_data else None,
                metadata=case_data.get('metadata', {})
            )
            
            for ann_data in case_data.get('annotations', []):
                center = ann_data['center']
                size = ann_data['size']
                bbox = BoundingBox3D(
                    x=center[0], y=center[1], z=center[2],
                    width=size[0], height=size[1], depth=size[2],
                    confidence=ann_data.get('confidence', 1.0),
                    label=ann_data.get('label', 'aneurysm'),
                    annotation_id=ann_data.get('id')
                )
                case.add_bounding_box(bbox)
            
            cases.append(case)
        
        logger.info(f"Parsed {len(cases)} cases from {file_path}")
        return cases


class XMLAnnotationParser(AnnotationParser):
    """Parser for XML format annotations (e.g., PASCAL VOC style)."""
    
    def get_supported_extensions(self) -> List[str]:
        return ['.xml']
    
    def parse_file(self, file_path: Union[str, Path]) -> List[AnnotationCase]:
        """
        Parse XML annotation file.
        
        Expected XML format (similar to PASCAL VOC but for 3D):
        <annotation>
            <case_id>case_001</case_id>
            <image_path>/path/to/image.nii.gz</image_path>
            <voxel_spacing>0.4,0.4,0.4</voxel_spacing>
            <volume_shape>512,512,300</volume_shape>
            <object>
                <name>aneurysm</name>
                <bndbox3d>
                    <center_x>256</center_x>
                    <center_y>256</center_y>
                    <center_z>150</center_z>
                    <width>10</width>
                    <height>12</height>
                    <depth>8</depth>
                </bndbox3d>
                <confidence>1.0</confidence>
            </object>
        </annotation>
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {file_path}")
        
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        case_id = root.find('case_id').text
        image_path = root.find('image_path').text
        
        voxel_spacing_text = root.find('voxel_spacing').text
        voxel_spacing = tuple(map(float, voxel_spacing_text.split(',')))
        
        volume_shape_elem = root.find('volume_shape')
        volume_shape = tuple(map(int, volume_shape_elem.text.split(','))) if volume_shape_elem is not None else None
        
        case = AnnotationCase(
            case_id=case_id,
            image_path=image_path,
            voxel_spacing=voxel_spacing,
            volume_shape=volume_shape
        )
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox3d')
            
            bbox = BoundingBox3D(
                x=float(bndbox.find('center_x').text),
                y=float(bndbox.find('center_y').text),
                z=float(bndbox.find('center_z').text),
                width=float(bndbox.find('width').text),
                height=float(bndbox.find('height').text),
                depth=float(bndbox.find('depth').text),
                confidence=float(obj.find('confidence').text) if obj.find('confidence') is not None else 1.0,
                label=name
            )
            case.add_bounding_box(bbox)
        
        logger.info(f"Parsed 1 case with {len(case.bounding_boxes)} annotations from {file_path}")
        return [case]


class CSVAnnotationParser(AnnotationParser):
    """Parser for CSV format annotations."""
    
    def get_supported_extensions(self) -> List[str]:
        return ['.csv']
    
    def parse_file(self, file_path: Union[str, Path]) -> List[AnnotationCase]:
        """
        Parse CSV annotation file.
        
        Expected CSV format:
        case_id,image_path,center_x,center_y,center_z,width,height,depth,confidence,label,voxel_spacing_x,voxel_spacing_y,voxel_spacing_z
        case_001,/path/to/image.nii.gz,256,256,150,10,12,8,1.0,aneurysm,0.4,0.4,0.4
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Group by case_id
        cases = []
        for case_id, group in df.groupby('case_id'):
            first_row = group.iloc[0]
            
            voxel_spacing = (
                first_row.get('voxel_spacing_x', 0.4),
                first_row.get('voxel_spacing_y', 0.4),
                first_row.get('voxel_spacing_z', 0.4)
            )
            
            case = AnnotationCase(
                case_id=case_id,
                image_path=first_row['image_path'],
                voxel_spacing=voxel_spacing
            )
            
            for _, row in group.iterrows():
                bbox = BoundingBox3D(
                    x=row['center_x'],
                    y=row['center_y'],
                    z=row['center_z'],
                    width=row['width'],
                    height=row['height'],
                    depth=row['depth'],
                    confidence=row.get('confidence', 1.0),
                    label=row.get('label', 'aneurysm')
                )
                case.add_bounding_box(bbox)
            
            cases.append(case)
        
        logger.info(f"Parsed {len(cases)} cases from {file_path}")
        return cases


class AnnotationManager:
    """
    Main class for managing annotation parsing and processing.
    
    Supports multiple annotation formats and provides unified interface
    for loading, processing, and converting annotations.
    """
    
    def __init__(self):
        self.parsers = {
            '.json': JSONAnnotationParser(),
            '.xml': XMLAnnotationParser(),
            '.csv': CSVAnnotationParser()
        }
        self.cases: List[AnnotationCase] = []
    
    def load_annotations(self, file_path: Union[str, Path]) -> List[AnnotationCase]:
        """Load annotations from file using appropriate parser."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension not in self.parsers:
            raise ValueError(f"Unsupported annotation format: {extension}. "
                           f"Supported formats: {list(self.parsers.keys())}")
        
        parser = self.parsers[extension]
        cases = parser.parse_file(file_path)
        self.cases.extend(cases)
        
        return cases
    
    def load_multiple_files(self, file_paths: List[Union[str, Path]]) -> List[AnnotationCase]:
        """Load annotations from multiple files."""
        all_cases = []
        for file_path in file_paths:
            cases = self.load_annotations(file_path)
            all_cases.extend(cases)
        
        return all_cases
    
    def get_case_by_id(self, case_id: str) -> Optional[AnnotationCase]:
        """Get annotation case by ID."""
        for case in self.cases:
            if case.case_id == case_id:
                return case
        return None
    
    def filter_cases_by_annotation_count(self, min_count: int = 1, max_count: Optional[int] = None) -> List[AnnotationCase]:
        """Filter cases by number of annotations."""
        filtered_cases = []
        for case in self.cases:
            count = len(case.bounding_boxes)
            if count >= min_count and (max_count is None or count <= max_count):
                filtered_cases.append(case)
        
        return filtered_cases
    
    def convert_to_detection_format(self, case: AnnotationCase) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert annotation case to detection format compatible with models.
        
        Returns:
            Tuple of (bounding_boxes, confidence_scores) as numpy arrays
        """
        bounding_boxes = case.get_bounding_boxes_array()
        confidence_scores = case.get_confidence_scores()
        
        return bounding_boxes, confidence_scores
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the loaded dataset."""
        if not self.cases:
            return {"num_cases": 0, "total_annotations": 0}
        
        total_annotations = sum(len(case.bounding_boxes) for case in self.cases)
        case_annotation_counts = [len(case.bounding_boxes) for case in self.cases]
        
        all_volumes = []
        all_confidences = []
        for case in self.cases:
            for bbox in case.bounding_boxes:
                all_volumes.append(bbox.get_volume())
                all_confidences.append(bbox.confidence)
        
        stats = {
            "num_cases": len(self.cases),
            "total_annotations": total_annotations,
            "mean_annotations_per_case": np.mean(case_annotation_counts),
            "std_annotations_per_case": np.std(case_annotation_counts),
            "min_annotations_per_case": np.min(case_annotation_counts),
            "max_annotations_per_case": np.max(case_annotation_counts),
        }
        
        if all_volumes:
            stats.update({
                "mean_annotation_volume": np.mean(all_volumes),
                "std_annotation_volume": np.std(all_volumes),
                "min_annotation_volume": np.min(all_volumes),
                "max_annotation_volume": np.max(all_volumes),
                "mean_confidence": np.mean(all_confidences),
                "min_confidence": np.min(all_confidences),
                "max_confidence": np.max(all_confidences)
            })
        
        return stats
    
    def export_to_json(self, output_path: Union[str, Path]) -> None:
        """Export all loaded cases to JSON format."""
        output_path = Path(output_path)
        
        export_data = {
            "cases": []
        }
        
        for case in self.cases:
            case_data = {
                "case_id": case.case_id,
                "image_path": case.image_path,
                "voxel_spacing": list(case.voxel_spacing),
                "volume_shape": list(case.volume_shape) if case.volume_shape else None,
                "metadata": case.metadata,
                "annotations": []
            }
            
            for i, bbox in enumerate(case.bounding_boxes):
                ann_data = {
                    "id": bbox.annotation_id or f"ann_{i:03d}",
                    "center": [bbox.x, bbox.y, bbox.z],
                    "size": [bbox.width, bbox.height, bbox.depth],
                    "confidence": bbox.confidence,
                    "label": bbox.label
                }
                case_data["annotations"].append(ann_data)
            
            export_data["cases"].append(case_data)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(self.cases)} cases to {output_path}")


def create_sample_annotations() -> List[AnnotationCase]:
    """Create sample annotation data for testing purposes."""
    cases = []
    
    # Case 1: Single aneurysm
    case1 = AnnotationCase(
        case_id="test_case_001",
        image_path="/data/cta/case_001.nii.gz",
        voxel_spacing=(0.4, 0.4, 0.4),
        volume_shape=(512, 512, 300)
    )
    bbox1 = BoundingBox3D(
        x=256, y=256, z=150,
        width=12, height=10, depth=8,
        confidence=1.0,
        annotation_id="ann_001"
    )
    case1.add_bounding_box(bbox1)
    cases.append(case1)
    
    # Case 2: Multiple aneurysms
    case2 = AnnotationCase(
        case_id="test_case_002", 
        image_path="/data/cta/case_002.nii.gz",
        voxel_spacing=(0.4, 0.4, 0.4),
        volume_shape=(512, 512, 280)
    )
    bbox2a = BoundingBox3D(
        x=200, y=300, z=140,
        width=8, height=12, depth=10,
        confidence=1.0,
        annotation_id="ann_002a"
    )
    bbox2b = BoundingBox3D(
        x=350, y=180, z=160,
        width=15, height=8, depth=12,
        confidence=1.0,
        annotation_id="ann_002b"
    )
    case2.add_bounding_box(bbox2a)
    case2.add_bounding_box(bbox2b)
    cases.append(case2)
    
    return cases


def main():
    """Example usage of the annotation parser."""
    # Create sample data
    sample_cases = create_sample_annotations()
    
    # Create annotation manager
    manager = AnnotationManager()
    manager.cases = sample_cases
    
    # Get statistics
    stats = manager.get_dataset_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Convert to detection format
    for case in sample_cases:
        bboxes, scores = manager.convert_to_detection_format(case)
        print(f"\nCase {case.case_id}:")
        print(f"  Bounding boxes shape: {bboxes.shape}")
        print(f"  Confidence scores shape: {scores.shape}")
        print(f"  Bounding boxes:\n{bboxes}")
        print(f"  Confidence scores: {scores}")


if __name__ == "__main__":
    main()