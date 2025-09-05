#!/bin/bash

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install SimpleITK for DICOM processing
pip install SimpleITK

# Install development dependencies
pip install black isort pytest pytest-cov

echo "Setup complete! Don't forget to activate the virtual environment with 'source venv/bin/activate'"
