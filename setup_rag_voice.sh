#!/bin/bash
# Setup RAG + Voice for Flashback Avatar

echo "🚀 Setting up RAG + Voice Features..."
echo "========================================"

cd /mnt/FlashbackAvatars

# Install RAG dependencies
echo "📦 Installing RAG dependencies..."
pip install chromadb sentence-transformers pyyaml

# Install voice dependencies (already have edge-tts)
echo "🎤 Checking voice dependencies..."
pip install edge-tts --upgrade

# Stop existing services
echo "⏹️  Stopping existing services..."
pm2 stop flashback-avatar 2>/dev/null || true

# Start new RAG+Voice server
echo "🎭 Starting Flashback Avatar (RAG + Voice)..."
pm2 delete flashback-avatar 2>/dev/null || true
pm2 start flashback_avatar_rag_voice.py --name flashback-avatar --interpreter python3

# Save configuration
pm2 save

echo ""
echo "✅ Setup Complete!"
echo "========================================"
pm2 list
echo ""
echo "📱 Access at: http://localhost:8000"
echo "🎤 Voice input: Click microphone button"
echo "🔊 Voice output: Avatar speaks automatically"
echo "📚 RAG: Knowledge base enabled"
echo ""
echo "Commands:"
echo "  pm2 logs flashback-avatar     - View logs"
echo "  pm2 restart flashback-avatar  - Restart"
echo "  pm2 monit                      - Monitor resources"
echo "========================================"
