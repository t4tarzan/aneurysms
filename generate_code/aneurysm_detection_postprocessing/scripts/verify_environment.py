#!/usr/bin/env python3
"""Verify that the environment is set up correctly."""
import sys
import torch
import numpy as np
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aneurysm_detection.utils.logging_config import setup_logging

def check_python_version() -> bool:
    """Check Python version."""
    print("\n=== Python Version ===")
    print(f"Python {sys.version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print("✅ Python version is compatible")
    return True

def check_pytorch() -> bool:
    """Check PyTorch installation and GPU availability."""
    print("\n=== PyTorch ===")
    print(f"Version: {torch.__version__}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {'✅' if cuda_available else '❌'}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA devices: {torch.cuda.device_count()}")
    
    return cuda_available

def check_imports() -> bool:
    """Check if required packages can be imported."""
    print("\n=== Required Packages ===")
    packages = [
        'numpy', 'scipy', 'matplotlib', 'seaborn',
        'nibabel', 'pydicom', 'SimpleITK', 'skimage',
        'tqdm', 'pandas', 'wandb'
    ]
    
    all_imported = True
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg} not found")
            all_imported = False
    
    return all_imported

def check_optional_imports() -> bool:
    """Check if optional packages can be imported."""
    print("\n=== Optional Packages ===")
    packages = [
        'totalsegmentator',
        'nnunet',
        'antspy'
    ]
    
    all_imported = True
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg}")
        except ImportError:
            print(f"⚠️  {pkg} not found (optional)")
            all_imported = False
    
    return all_imported

def check_logging() -> bool:
    """Test logging configuration."""
    print("\n=== Logging Test ===")
    try:
        logger = setup_logging(name="test_logger")
        logger.info("Logging test successful")
        print("✅ Logging test passed")
        return True
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def main():
    """Run all environment checks."""
    print("\n🚀 Starting Environment Verification\n")
    
    checks = {
        "Python Version": check_python_version(),
        "PyTorch": check_pytorch(),
        "Required Packages": check_imports(),
        "Optional Packages": check_optional_imports(),
        "Logging": check_logging()
    }
    
    print("\n=== Summary ===")
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
    
    if all(checks.values()):
        print("\n🎉 All checks passed! Your environment is ready.")
        return 0
    else:
        print("\n❌ Some checks failed. Please address the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
