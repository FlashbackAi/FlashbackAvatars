#!/bin/bash

# SplattingAvatar Setup Script for FlashbackAvatars
# Creates high-quality shoulders-up avatar for Anam.ai-like experience

set -e

echo "🎬 Setting up SplattingAvatar for real-time avatar..."

# Clone SplattingAvatar
cd third_party
if [ ! -d "SplattingAvatar" ]; then
    echo "📥 Cloning SplattingAvatar repository..."
    git clone --recursive https://github.com/initialneil/SplattingAvatar.git
else
    echo "✅ SplattingAvatar already cloned"
fi

cd SplattingAvatar

# Create conda environment
echo "🐍 Creating conda environment..."
if ! conda env list | grep -q "splatting_avatar"; then
    conda create -n splatting_avatar python=3.9 -y
fi

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate splatting_avatar

# Install PyTorch
echo "🔥 Installing PyTorch with CUDA 11.7..."
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install pytorch3d
echo "📐 Installing pytorch3d..."
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install other dependencies
echo "📦 Installing dependencies..."
pip install opencv-python trimesh scikit-image tqdm pyyaml plyfile

# Create data directory for FLAME models
mkdir -p data/FLAME2020

echo ""
echo "✅ SplattingAvatar setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Download FLAME models:"
echo "   - Go to: https://flame.is.tue.mpg.de/"
echo "   - Register and download FLAME 2020 model"
echo "   - Place files in: third_party/SplattingAvatar/data/FLAME2020/"
echo ""
echo "2. Record Vinay's video:"
echo "   - Duration: 2-3 minutes"
echo "   - Framing: Shoulders-up, centered"
echo "   - Content: Various expressions and head movements"
echo "   - Save as: avatar_input/vinay_shoulders_up.mp4"
echo ""
echo "3. Train avatar:"
echo "   conda activate splatting_avatar"
echo "   python train_splatting_avatar.py --config configs/head_avatar.yaml --dat_dir data/vinay_head"
echo ""
