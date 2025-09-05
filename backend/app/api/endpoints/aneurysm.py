"""
API endpoints for aneurysm detection and analysis.
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import uuid
import numpy as np

from ....core.config import settings
from ....services.dicom_processor import DICOMProcessor
from ....services.model_service import AneurysmDetectionModel

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
dicom_processor = DICOMProcessor(settings.UPLOAD_DIR)
model_service = AneurysmDetectionModel()  # Will be initialized with actual model path

# In-memory storage for processing status and results
processing_status = {}

@router.post("/upload", response_model=Dict[str, Any])
async def upload_dicom_files(
    files: List[UploadFile] = File(...)
):
    """
    Upload DICOM files for aneurysm detection.
    
    Args:
        files: List of DICOM files from a single study
        
    Returns:
        Upload status and ID for tracking processing
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    # Generate unique upload ID
    upload_id = str(uuid.uuid4())
    upload_path = settings.UPLOAD_DIR / upload_id
    upload_path.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded files
    saved_files = []
    for file in files:
        try:
            file_path = upload_path / file.filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            saved_files.append({
                "filename": file.filename,
                "size": len(content),
                "content_type": file.content_type
            })
        except Exception as e:
            logger.error(f"Error saving file {file.filename}: {str(e)}")
            # Continue with other files even if one fails
    
    if not saved_files:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save any files"
        )
    
    # Initialize processing status
    processing_status[upload_id] = {
        "status": "uploaded",
        "progress": 0,
        "message": "Files uploaded successfully",
        "results": None,
        "error": None
    }
    
    return {
        "status": "success",
        "upload_id": upload_id,
        "file_count": len(saved_files),
        "files": saved_files
    }

@router.post("/process/{upload_id}", response_model=Dict[str, Any])
async def process_aneurysm_detection(
    upload_id: str,
    background_tasks: BackgroundTasks
):
    """
    Start background processing of uploaded DICOM files for aneurysm detection.
    
    Args:
        upload_id: The unique ID of the upload to process
        
    Returns:
        Initial processing status
    """
    if upload_id not in processing_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload ID {upload_id} not found"
        )
    
    # Update status to processing
    processing_status[upload_id].update({
        "status": "processing",
        "progress": 0,
        "message": "Starting DICOM processing",
        "error": None
    })
    
    # Start background task
    background_tasks.add_task(
        process_detection_pipeline,
        upload_id=upload_id
    )
    
    return {
        "status": "processing_started",
        "upload_id": upload_id,
        "message": "Aneurysm detection processing started"
    }

@router.get("/status/{upload_id}", response_model=Dict[str, Any])
async def get_processing_status(upload_id: str):
    """
    Get the current status of a processing job.
    
    Args:
        upload_id: The unique ID of the upload
        
    Returns:
        Current processing status and results if available
    """
    if upload_id not in processing_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload ID {upload_id} not found"
        )
    
    status_info = processing_status[upload_id].copy()
    
    # Don't return full results in status endpoint
    if "results" in status_info:
        status_info["has_results"] = status_info["results"] is not None
        status_info.pop("results", None)
    
    return status_info

@router.get("/results/{upload_id}", response_model=Dict[str, Any])
async def get_detection_results(upload_id: str):
    """
    Get the results of a completed processing job.
    
    Args:
        upload_id: The unique ID of the upload
        
    Returns:
        Detection results if processing is complete
    """
    if upload_id not in processing_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload ID {upload_id} not found"
        )
    
    status_info = processing_status[upload_id]
    
    if status_info["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail=f"Processing not complete. Current status: {status_info['status']}"
        )
    
    if "results" not in status_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No results available"
        )
    
    return {
        "status": "completed",
        "upload_id": upload_id,
        **status_info["results"]
    }

# Background task function
async def process_detection_pipeline(upload_id: str):
    """Background task to process DICOM files and detect aneurysms."""
    try:
        # Update status
        processing_status[upload_id].update({
            "status": "processing",
            "progress": 10,
            "message": "Loading DICOM files"
        })
        
        # Step 1: Load and preprocess DICOM files
        dicom_data = await dicom_processor.load_dicom_series(upload_id)
        
        # Update status
        processing_status[upload_id].update({
            "progress": 30,
            "message": "Running aneurysm detection model"
        })
        
        # Step 2: Run aneurysm detection
        # Note: In a real implementation, we would use the actual artery segmentation
        volume = dicom_data["pixel_data"]
        spacing = dicom_data["metadata"].get("pixel_spacing", [1.0, 1.0])
        spacing = spacing + [dicom_data["metadata"].get("slice_thickness", 1.0)]  # Add z-spacing
        
        # Preprocess volume for model
        volume_tensor = model_service.preprocess_volume(volume, spacing)
        
        # Run detection (with None for artery mask in this example)
        detections = model_service.detect_aneurysms(volume_tensor)
        
        # Post-process detections
        original_shape = volume.shape
        processed_detections = model_service.postprocess_detections(
            detections, original_shape, spacing
        )
        
        # Format results
        results = {
            "findings": {
                "aneurysm_detected": len(processed_detections.get("detections", [])) > 0,
                "detection_count": len(processed_detections.get("detections", [])),
                "detections": processed_detections.get("detections", [])
            },
            "metadata": dicom_data["metadata"],
            "visualization": {
                "volume_shape": volume.shape,
                "spacing": spacing,
                # Add visualization data here
            }
        }
        
        # Update status with results
        processing_status[upload_id].update({
            "status": "completed",
            "progress": 100,
            "message": "Processing complete",
            "results": results,
            "error": None
        })
        
    except Exception as e:
        logger.error(f"Error processing {upload_id}: {str(e)}", exc_info=True)
        processing_status[upload_id].update({
            "status": "error",
            "progress": 0,
            "message": f"Error during processing: {str(e)}",
            "error": str(e)
        })

# Include the router in the main FastAPI app
# This will be done in app/main.py
