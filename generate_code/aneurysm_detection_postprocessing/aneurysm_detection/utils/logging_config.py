"""Logging configuration for the aneurysm detection project."""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

def setup_logging(output_dir: str = "logs", name: str = None):
    """Setup logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger(name or "aneurysm_detection")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    log_file = f"{name or 'training'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(Path(output_dir) / log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
