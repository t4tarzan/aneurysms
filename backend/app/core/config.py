"""Configuration settings for the Aneurysm Detection API."""
from pydantic import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Aneurysm Detection API"
    DEBUG: bool = True
    
    # File upload settings
    UPLOAD_DIR: Path = Path("uploads")
    MAX_UPLOAD_SIZE: int = 1024 * 1024 * 500  # 500MB
    ALLOWED_FILE_TYPES: List[str] = ["application/dicom", "application/octet-stream"]
    
    # Model settings
    MODEL_CONFIDENCE_THRESHOLD: float = 0.8
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        case_sensitive = True

# Create instance of settings
settings = Settings()

# Ensure upload directory exists
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
