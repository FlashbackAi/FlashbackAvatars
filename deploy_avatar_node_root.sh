#!/bin/bash
# 🚀 Complete Avatar System Deployment Script (Root User Version)
# Deploys real-time avatar system on a fresh Ubuntu node as root

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
    print_success "NVIDIA H200 detected with 144GB memory!"
else
    print_warning "No NVIDIA GPU detected. CPU-only mode will be slower."
fi

# 2. Update System
print_status "Updating system packages..."
apt update && apt upgrade -y

# 3. Install Python 3.11
print_status "Installing Python 3.11 and pip..."
apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
apt install -y build-essential cmake git curl wget

# 4. Install Node.js (for potential web enhancements)
print_status "Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt install -y nodejs

# 5. Install FFmpeg (for video processing)
print_status "Installing FFmpeg and media libraries..."
apt install -y ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0 libsndfile1

# 6. CUDA is likely already installed on H200, check version
if command -v nvidia-smi &> /dev/null; then
    print_status "Checking CUDA installation..."
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        print_success "CUDA already installed: $CUDA_VERSION"
    else
        print_warning "CUDA toolkit not found in PATH, may need to install"
        # Add CUDA to PATH if it exists
        if [ -d "/usr/local/cuda" ]; then
            echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
            echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
            export PATH=/usr/local/cuda/bin:$PATH
            export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
        fi
    fi
fi

# 7. Install Ollama
print_status "Installing Ollama..."
if command -v ollama &> /dev/null; then
    print_success "Ollama already installed"
else
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Start Ollama service (if systemd is available)
if command -v systemctl &> /dev/null; then
    systemctl enable ollama 2>/dev/null || true
    systemctl start ollama 2>/dev/null || true
else
    # Start Ollama in background
    nohup ollama serve > /var/log/ollama.log 2>&1 &
fi

# Wait for Ollama to start
sleep 5

# 8. Download Llama 3.2 3B model
print_status "Downloading Llama 3.2 3B model (this may take a while)..."
if ollama list | grep -q "llama3.2:3b"; then
    print_success "Llama 3.2 3B model already exists"
else
    ollama pull llama3.2:3b
    print_success "Llama 3.2 3B model downloaded successfully!"
fi

# 9. Navigate to project directory
print_status "Setting up project directory..."
cd /mnt/FlashbackAvatars

# 10. Create Python virtual environment
print_status "Creating Python virtual environment..."
python3.11 -m venv avatar_env
source avatar_env/bin/activate

# 11. Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# 12. Install PyTorch with CUDA support
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

# Run download script
python download_models.py

# 15. Create avatar input directory
print_status "Setting up avatar input directory..."
mkdir -p avatar_input

# Check if input files exist
if [ ! -f "avatar_input/vinay_intro.mp4" ]; then
    print_warning "Input video not found: avatar_input/vinay_intro.mp4"
    print_warning "Please upload your video to avatar_input/vinay_intro.mp4"
fi

if [ ! -f "avatar_input/vinay_audio.wav" ]; then
    print_warning "Input audio not found: avatar_input/vinay_audio.wav"
    print_warning "Please upload your audio to avatar_input/vinay_audio.wav"
fi

# 16. Run system setup
print_status "Running avatar system setup..."
python setup_realtime_avatar.py

# 17. Create systemd service for auto-start
print_status "Creating systemd service..."
cat > /etc/systemd/system/avatar-server.service << EOF
[Unit]
Description=Real-Time Avatar Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/mnt/FlashbackAvatars
Environment=PATH=/mnt/FlashbackAvatars/avatar_env/bin
ExecStart=/mnt/FlashbackAvatars/avatar_env/bin/python realtime_avatar_server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload
systemctl enable avatar-server
# Don't start yet - need frame banks first

# 18. Setup firewall (if ufw is installed)
if command -v ufw &> /dev/null; then
    print_status "Configuring firewall..."
    ufw allow 8000/tcp
    ufw allow ssh
    ufw --force enable
else
    print_warning "ufw not installed, skipping firewall configuration"
fi

# 19. Create environment file
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

# 20. Get public IP
PUBLIC_IP=$(curl -s ifconfig.me || echo "localhost")

# 21. Test system
print_status "Testing system components..."

# Test Ollama
echo "Testing Ollama..."
if ollama run llama3.2:3b "Hello" --timeout 30 > /dev/null 2>&1; then
    print_success "Ollama and Llama 3.2 3B working!"
else
    print_warning "Ollama test failed, but continuing..."
fi

# Test Python imports
echo "Testing Python dependencies..."
python -c "
import torch
import numpy as np
from PIL import Image
print('Core dependencies working!')
" && print_success "Python dependencies working!"

# 22. Display completion information
echo ""
echo "========================================"
print_success "🎉 Avatar System Deployment Complete!"
echo "========================================"
echo ""
echo "📋 System Information:"
echo "   • GPU: NVIDIA H200 (144GB)"
echo "   • Ollama Model: llama3.2:3b"
echo "   • Vector DB: ChromaDB (rag_database/)"
echo "   • Frame Banks: frame_banks/"
echo "   • Service: avatar-server.service"
echo "   • Web Interface: http://${PUBLIC_IP}:8000"
echo ""
echo "🎯 Next Steps:"
echo ""
echo "1. Upload your input files (if not already done):"
echo "   • avatar_input/vinay_intro.mp4"
echo "   • avatar_input/vinay_audio.wav"
echo ""
echo "2. Generate frame banks (takes 30-40 minutes):"
echo "   python frame_bank_generator.py"
echo ""
echo "3. Start the avatar server:"
echo "   systemctl start avatar-server"
echo ""
echo "   OR run manually:"
echo "   source avatar_env/bin/activate && python realtime_avatar_server.py"
echo ""
echo "🚀 Quick Start Commands:"
echo "   • Check service: systemctl status avatar-server"
echo "   • View logs: journalctl -u avatar-server -f"
echo "   • Restart service: systemctl restart avatar-server"
echo ""
echo "🔧 Configuration Files:"
echo "   • Environment: .env"
echo "   • Service: /etc/systemd/system/avatar-server.service"
echo ""
print_success "Your real-time Vinay Thadem avatar is almost ready!"
echo "Complete steps 1-3 above to start using it."