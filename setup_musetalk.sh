#!/bin/bash

# MuseTalk Setup Script for FlashbackAvatars
# Real-time lip-sync animation (30+ FPS)

set -e

echo "🎤 Setting up MuseTalk for real-time lip-sync..."

# Clone MuseTalk
cd third_party
if [ ! -d "MuseTalk" ]; then
    echo "📥 Cloning MuseTalk repository..."
    git clone https://github.com/TMElyralab/MuseTalk.git
else
    echo "✅ MuseTalk already cloned"
fi

cd MuseTalk

# Create conda environment
echo "🐍 Creating conda environment..."
if ! conda env list | grep -q "musetalk"; then
    conda create -n musetalk python=3.10 -y
fi

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate musetalk

# Install PyTorch
echo "🔥 Installing PyTorch with CUDA 11.7..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Install MMlab packages
echo "🔬 Installing MMlab packages..."
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"

# Download MuseTalk models
echo "📥 Downloading MuseTalk models..."
if [ -f "scripts/download_models.py" ]; then
    python scripts/download_models.py
else
    echo "⚠️  Warning: download_models.py not found. Download models manually from:"
    echo "   https://huggingface.co/TMElyralab/MuseTalk"
fi

echo ""
echo "✅ MuseTalk setup complete!"
echo ""
echo "📋 Test MuseTalk:"
echo "   conda activate musetalk"
echo "   python inference.py --audio_path <audio.wav> --video_path <reference.mp4>"
echo ""
echo "🎯 Performance: 30+ FPS on NVIDIA Tesla V100, even faster on H200"
echo ""
