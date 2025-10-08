#!/bin/bash

echo "========================================"
echo "Voice Chat Environment Setup (Linux/Mac)"
echo "========================================"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (for GPU)
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
echo "Installing voice chat requirements..."
pip install -r requirements.txt

# Fix numpy version conflict
echo "Fixing numpy version..."
pip install "numpy>=1.26.0" --upgrade

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run server:"
echo "  python server.py"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
