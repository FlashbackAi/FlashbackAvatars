#!/bin/bash
# Setup Git Submodules for Third-Party Dependencies
# This keeps third-party repos separate and manageable

echo "🔗 Setting up Git Submodules for Third-Party Dependencies"
echo "========================================================="

# Create third_party directory
mkdir -p third_party

# Function to add submodule or clone directly
add_dependency() {
    local name=$1
    local url=$2
    local path=$3

    if [ ! -d "$path" ]; then
        echo "📥 Adding $name..."
        if git rev-parse --git-dir > /dev/null 2>&1; then
            # We're in a git repo, use submodules
            git submodule add "$url" "$path" 2>/dev/null || echo "Submodule $name already exists"
        else
            # Not in a git repo, clone directly
            git clone "$url" "$path"
        fi
    else
        echo "✅ $name already exists"
    fi
}

# Add EchoMimic v3
add_dependency "EchoMimic v3" "https://github.com/antgroup/echomimic_v3.git" "third_party/echomimic_v3"

# Add other dependencies as needed (uncomment when required)
# add_dependency "TTS" "https://github.com/coqui-ai/TTS.git" "third_party/TTS"
# add_dependency "VLLM" "https://github.com/vllm-project/vllm.git" "third_party/vllm"
# add_dependency "Ion SFU" "https://github.com/pion/ion-sfu.git" "third_party/ion-sfu"

# Initialize and update submodules
echo "🔄 Initializing submodules..."
git submodule init
git submodule update

echo "✅ Submodules setup complete!"
