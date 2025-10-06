#!/bin/bash
# Quick start script for Flashback Avatar Server

echo "🚀 Starting Flashback Avatar Server..."
echo "========================================"

# Check if we're in the right directory
if [ ! -f "flashback_avatar_server.py" ]; then
    echo "❌ Error: Not in FlashbackAvatars directory"
    echo "Run: cd /mnt/FlashbackAvatars"
    exit 1
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip install -q pyyaml 2>/dev/null || true

# Create logs directory
mkdir -p logs

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not installed. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.2:3b
fi

# Check if PM2 is installed
if ! command -v pm2 &> /dev/null; then
    echo "⚠️  PM2 not installed. Installing..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
    npm install -g pm2
fi

# Start services with PM2
echo "🎭 Starting services..."
pm2 start ecosystem.config.js

# Save PM2 configuration
pm2 save

# Show status
echo ""
echo "✅ Flashback Avatar Server Started!"
echo "========================================"
pm2 list
echo ""
echo "📱 Access at: http://localhost:8000"
echo "📊 View logs: pm2 logs flashback-avatar"
echo "🔧 Health: http://localhost:8000/health"
echo ""
echo "Commands:"
echo "  pm2 logs flashback-avatar  - View logs"
echo "  pm2 restart flashback-avatar - Restart server"
echo "  pm2 stop all - Stop all services"
echo "========================================"
