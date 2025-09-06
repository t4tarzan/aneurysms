#!/bin/bash

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support if available
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install core requirements
echo "Installing core requirements..."
pip install -r requirements.txt

# Install optional dependencies
echo "Installing optional dependencies..."
pip install totalsegmentator nnunet

# Install development dependencies
echo "Installing development dependencies..."
pip install -e .[dev]

echo "\nSetup complete! Activate the virtual environment with:"
echo "source venv/bin/activate"
