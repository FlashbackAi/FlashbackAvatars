#!/bin/bash
# Cloud GPU Development Setup Script for EchoMimic v3
# Works on RunPod, Vast.ai, or any Ubuntu cloud instance

echo "🌩️ Setting up EchoMimic v3 Cloud Development Environment"
echo "=================================================="

# Check GPU
echo "🔍 Checking GPU availability..."
nvidia-smi

# Update system
echo "📦 Updating system packages..."
apt-get update -y
apt-get install -y git wget curl unzip

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip

# Install all dependencies from requirements.txt
echo "📦 Installing all dependencies from requirements.txt..."
pip install -r requirements.txt

# Download models
echo "🎭 Downloading EchoMimic v3 models..."
python download_models.py

# Set up environment
echo "⚙️ Setting up environment..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Test GPU memory
echo "🧪 Testing GPU setup..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"

echo "✅ Setup complete! Ready for EchoMimic v3 development"
echo ""
echo "🚀 To start the server:"
echo "cd FlashbackAvatars/services/renderer"
echo "uvicorn server:app --host 0.0.0.0 --port 9000"
echo ""
echo "📱 Access your server at the provided cloud URL"
