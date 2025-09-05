"""
DICOM processing service for handling DICOM files and preparing them for analysis.
"""
import os
import pydicom
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from pydicom.pixel_data_handlers.util import apply_voi_lut
import SimpleITK as sitk
import numpy as np

logger = logging.getLogger(__name__)

class DICOMProcessor:
    """Service for processing DICOM files and preparing them for analysis."""
    
    def __init__(self, upload_dir: Path):
        """Initialize the DICOM processor with the upload directory."""
        self.upload_dir = upload_dir
        
    async def load_dicom_series(self, upload_id: str) -> Dict[str, Any]:
        """
        Load and process a DICOM series from the upload directory.
        
        Args:
            upload_id: The unique ID of the upload
            
        Returns:
            Dictionary containing the processed DICOM data
        """
        upload_path = self.upload_dir / upload_id
        if not upload_path.exists():
            raise FileNotFoundError(f"Upload directory {upload_id} not found")
            
        # Get all DICOM files in the upload directory
        dicom_files = list(upload_path.glob("*.dcm")) + list(upload_path.glob("*.dicom"))
        if not dicom_files:
            raise ValueError("No DICOM files found in the upload directory")
            
        # Load DICOM files
        dicom_datasets = [pydicom.dcmread(str(f), stop_before_pixels=False) for f in dicom_files]
        
        # Sort slices by position
        dicom_datasets = self._sort_dicom_slices(dicom_datasets)
        
        # Extract pixel data and metadata
        pixel_data = self._extract_pixel_data(dicom_datasets)
        metadata = self._extract_metadata(dicom_datasets)
        
        return {
            "pixel_data": pixel_data,
            "metadata": metadata,
            "series_uid": metadata.get("series_instance_uid", ""),
            "modality": metadata.get("modality", "CT")
        }
    
    def _sort_dicom_slices(self, datasets: List[pydicom.Dataset]) -> List[pydicom.Dataset]:
        """Sort DICOM slices by position along the z-axis."""
        try:
            # Try to sort by ImagePositionPatient (3D position)
            return sorted(datasets, key=lambda ds: float(ds.ImagePositionPatient[2]))
        except (AttributeError, TypeError):
            # Fall back to InstanceNumber if available
            try:
                return sorted(datasets, key=lambda ds: int(ds.InstanceNumber))
            except (AttributeError, TypeError):
                # Default to original order if no sortable attribute is found
                return datasets
    
    def _extract_pixel_data(self, datasets: List[pydicom.Dataset]) -> np.ndarray:
        """Extract and process pixel data from DICOM datasets."""
        # Initialize 3D volume
        first_ds = datasets[0]
        rows = first_ds.Rows
        cols = first_ds.Columns
        num_slices = len(datasets)
        
        # Create empty volume
        volume = np.zeros((rows, cols, num_slices), dtype=np.float32)
        
        # Process each slice
        for i, ds in enumerate(datasets):
            # Get pixel array
            pixel_array = ds.pixel_array
            
            # Apply VOI LUT if available
            if hasattr(ds, 'WindowWidth') and hasattr(ds, 'WindowCenter'):
                pixel_array = apply_voi_lut(pixel_array, ds)
            
            # Normalize to [0, 1]
            pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array) + 1e-8)
            
            # Add to volume
            volume[:, :, i] = pixel_array
            
        return volume
    
    def _extract_metadata(self, datasets: List[pydicom.Dataset]) -> Dict[str, Any]:
        """Extract relevant metadata from DICOM datasets."""
        if not datasets:
            return {}
            
        # Get metadata from first dataset
        ds = datasets[0]
        metadata = {
            "patient_id": getattr(ds, 'PatientID', ''),
            "patient_name": getattr(ds, 'PatientName', ''),
            "study_instance_uid": getattr(ds, 'StudyInstanceUID', ''),
            "series_instance_uid": getattr(ds, 'SeriesInstanceUID', ''),
            "study_date": getattr(ds, 'StudyDate', ''),
            "modality": getattr(ds, 'Modality', 'CT'),
            "rows": getattr(ds, 'Rows', 0),
            "columns": getattr(ds, 'Columns', 0),
            "pixel_spacing": list(getattr(ds, 'PixelSpacing', [1.0, 1.0])),
            "slice_thickness": float(getattr(ds, 'SliceThickness', 1.0)),
            "manufacturer": getattr(ds, 'Manufacturer', ''),
            "institution_name": getattr(ds, 'InstitutionName', ''),
            "number_of_slices": len(datasets)
        }
        
        return metadata
    
    def convert_to_nifti(self, pixel_data: np.ndarray, metadata: Dict[str, Any], output_path: Path) -> Path:
        """
        Convert DICOM pixel data to NIfTI format.
        
        Args:
            pixel_data: 3D numpy array of pixel data
            metadata: DICOM metadata dictionary
            output_path: Path to save the NIfTI file
            
        Returns:
            Path to the saved NIfTI file
        """
        # Create SimpleITK image
        image = sitk.GetImageFromArray(pixel_data)
        
        # Set metadata
        image.SetSpacing([
            float(metadata.get("pixel_spacing", [1.0, 1.0])[0]),
            float(metadata.get("pixel_spacing", [1.0, 1.0])[1]),
            float(metadata.get("slice_thickness", 1.0))
        ])
        
        # Save as NIfTI
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(output_path))
        
        return output_path
