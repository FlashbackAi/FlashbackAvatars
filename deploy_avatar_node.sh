#!/bin/bash
# 🚀 Complete Avatar System Deployment Script
# Deploys real-time avatar system on a fresh Ubuntu node

set -e  # Exit on any error

echo "🚀 Starting Avatar System Deployment..."
echo "========================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. System Information
print_status "Checking system information..."
echo "OS: $(lsb_release -d | cut -f2)"
echo "Architecture: $(uname -m)"
echo "Available RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "Available Disk: $(df -h / | awk 'NR==2 {print $4}')"

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB"
else
    print_warning "No NVIDIA GPU detected. CPU-only mode will be slower."
fi

# 2. Update System
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# 3. Install Python 3.11
print_status "Installing Python 3.11 and pip..."
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
sudo apt install -y build-essential cmake git curl wget

# 4. Install Node.js (for potential web enhancements)
print_status "Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# 5. Install FFmpeg (for video processing)
print_status "Installing FFmpeg and media libraries..."
sudo apt install -y ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0 libsndfile1

# 6. Install CUDA (if NVIDIA GPU detected)
if command -v nvidia-smi &> /dev/null; then
    print_status "Installing CUDA toolkit..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt update
    sudo apt install -y cuda-toolkit-12-1

    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
fi

# 7. Install Ollama
print_status "Installing Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
sudo systemctl enable ollama
sudo systemctl start ollama

# Wait for Ollama to start
sleep 5

# 8. Download Llama 3.2 3B model
print_status "Downloading Llama 3.2 3B model (this may take a while)..."
ollama pull llama3.2:3b

print_success "Llama 3.2 3B model downloaded successfully!"

# 9. Clone the FlashbackAvatars repository
# print_status "Cloning FlashbackAvatars repository..."
# if [ -d "FlashbackAvatars" ]; then
#     print_warning "FlashbackAvatars directory exists, pulling latest changes..."
#     cd FlashbackAvatars
#     git pull
# else
#     git clone https://github.com/FlashbackAi/FlashbackAvatars.git  # Replace with actual repo
#     cd FlashbackAvatars
# fi

# 10. Create Python virtual environment
print_status "Creating Python virtual environment..."
python3.11 -m venv avatar_env
source avatar_env/bin/activate

# 11. Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# 12. Install PyTorch with CUDA support (if available)
if command -v nvidia-smi &> /dev/null; then
    print_status "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    print_status "Installing PyTorch (CPU-only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 13. Install Python dependencies
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# 14. Download EchoMimic v3 models
print_status "Downloading EchoMimic v3 models..."
mkdir -p third_party/echomimic_v3/models

# Create download script for models (you'll need to implement this)
python3 download_models.py

# 15. Create avatar input directory with sample files
print_status "Setting up avatar input directory..."
mkdir -p avatar_input

# If you have sample files, add them here
# cp /path/to/sample/vinay_intro.mp4 avatar_input/
# cp /path/to/sample/vinay_audio.wav avatar_input/

# 16. Run system setup
print_status "Running avatar system setup..."
python3 setup_realtime_avatar.py

# 17. Generate initial frame banks (this takes time)
print_status "Generating initial frame banks (this may take 30+ minutes)..."
python3 frame_bank_generator.py

# 18. Create systemd service for auto-start
print_status "Creating systemd service..."
sudo tee /etc/systemd/system/avatar-server.service > /dev/null << EOF
[Unit]
Description=Real-Time Avatar Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/avatar_env/bin
ExecStart=$(pwd)/avatar_env/bin/python realtime_avatar_server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable avatar-server
sudo systemctl start avatar-server

# 19. Setup firewall
print_status "Configuring firewall..."
sudo ufw allow 8000/tcp
sudo ufw allow ssh
sudo ufw --force enable

# 20. Create environment file
print_status "Creating environment configuration..."
cat > .env << EOF
# Avatar System Configuration
AVATAR_HOST=0.0.0.0
AVATAR_PORT=8000
LLM_MODEL=llama3.2:3b
RAG_DATABASE_PATH=rag_database
FRAME_BANK_PATH=frame_banks
LOG_LEVEL=INFO
EOF

# 21. Test system
print_status "Testing system components..."

# Test Ollama
echo "Testing Ollama..."
if ollama run llama3.2:3b "Hello" --timeout 30; then
    print_success "Ollama and Llama 3.2 3B working!"
else
    print_error "Ollama test failed!"
    exit 1
fi

# Test Python imports
echo "Testing Python dependencies..."
python3 -c "
import torch
import chromadb
import ollama
import fastapi
import numpy as np
from PIL import Image
print('All Python dependencies working!')
"

# 22. Display completion information
echo ""
echo "========================================"
print_success "🎉 Avatar System Deployment Complete!"
echo "========================================"
echo ""
echo "📋 System Information:"
echo "   • Ollama Model: llama3.2:3b"
echo "   • Vector DB: ChromaDB (rag_database/)"
echo "   • Frame Banks: frame_banks/"
echo "   • Service: avatar-server.service"
echo "   • Web Interface: http://$(curl -s ifconfig.me):8000"
echo ""
echo "🚀 Quick Start Commands:"
echo "   • Check service: sudo systemctl status avatar-server"
echo "   • View logs: journalctl -u avatar-server -f"
echo "   • Restart service: sudo systemctl restart avatar-server"
echo "   • Manual start: source avatar_env/bin/activate && python realtime_avatar_server.py"
echo ""
echo "🔧 Configuration Files:"
echo "   • Environment: .env"
echo "   • Service: /etc/systemd/system/avatar-server.service"
echo "   • Logs: /var/log/avatar-server/"
echo ""
echo "💡 Next Steps:"
echo "   1. Add your video files to avatar_input/"
echo "   2. Customize knowledge base in local_llm_rag.py"
echo "   3. Access web interface at http://$(curl -s ifconfig.me):8000"
echo ""
print_success "Your real-time avatar is ready for conversations!"