#!/bin/bash
# Setup Git Submodules for Third-Party Dependencies
# This keeps third-party repos separate and manageable

echo "🔗 Setting up Git Submodules for Third-Party Dependencies"
echo "========================================================="

# Remove existing third_party directory if it exists
if [ -d "third_party" ]; then
    echo "🗑️  Removing existing third_party directory..."
    rm -rf third_party
fi

# Create third_party directory
mkdir -p third_party

# Add EchoMimic v3 as submodule
echo "📥 Adding EchoMimic v3 submodule..."
git submodule add https://github.com/antgroup/echomimic_v3.git third_party/echomimic_v3

# Add other submodules as needed
echo "📥 Adding TTS submodule..."
# git submodule add https://github.com/coqui-ai/TTS.git third_party/TTS

echo "📥 Adding VLLM submodule..."
# git submodule add https://github.com/vllm-project/vllm.git third_party/vllm

echo "📥 Adding Ion SFU submodule..."
# git submodule add https://github.com/pion/ion-sfu.git third_party/ion-sfu

# Initialize and update submodules
echo "🔄 Initializing submodules..."
git submodule init
git submodule update

echo "✅ Submodules setup complete!"
echo ""
echo "📋 To clone this repo with submodules in the future:"
echo "git clone --recursive https://github.com/FlashbackAi/FlashbackAvatars.git"
echo ""
echo "📋 To update submodules:"
echo "git submodule update --remote"
