#!/usr/bin/env python3
"""
Setup Script for Real-Time Avatar System
Initializes frame banks, validates dependencies, and prepares the system
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")

    required_packages = {
        'fastapi': 'FastAPI web framework',
        'uvicorn': 'ASGI server',
        'websockets': 'WebSocket support',
        'numpy': 'Numerical computing',
        'PIL': 'Image processing',
        'torch': 'PyTorch',
        'librosa': 'Audio processing',
        'soundfile': 'Audio file I/O',
    }

    optional_packages = {
        'webrtcvad': 'Voice Activity Detection (install with: pip install webrtcvad)',
        'speech_recognition': 'Speech-to-Text (install with: pip install SpeechRecognition)',
        'pyttsx3': 'Text-to-Speech (install with: pip install pyttsx3)',
    }

    missing_required = []
    missing_optional = []

    # Check required packages
    for package, description in required_packages.items():
        try:
            if package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"  ✅ {package}: {description}")
        except ImportError:
            missing_required.append(package)
            print(f"  ❌ {package}: {description} - MISSING")

    # Check optional packages
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package}: {description}")
        except ImportError:
            missing_optional.append(package)
            print(f"  ⚠️  {package}: {description} - OPTIONAL")

    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Run: pip install -r requirements.txt")
        return False

    if missing_optional:
        print(f"\n⚠️  Optional packages missing: {', '.join(missing_optional)}")
        print("Some features may be limited.")

    print("\n✅ All required dependencies are available!")
    return True

def validate_input_files():
    """Validate that required input files exist"""
    print("\n🎬 Validating input files...")

    required_files = {
        'avatar_input/vinay_intro.mp4': 'Base video for avatar generation',
        'avatar_input/vinay_audio.wav': 'Base audio for avatar generation'
    }

    missing_files = []

    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}: {description}")
        else:
            missing_files.append(file_path)
            print(f"  ❌ {file_path}: {description} - MISSING")

    if missing_files:
        print(f"\n❌ Missing input files: {', '.join(missing_files)}")
        print("Please ensure your video and audio files are in the avatar_input/ directory")
        return False

    print("\n✅ All input files found!")
    return True

def check_echomimic_models():
    """Check if EchoMimic v3 models are available"""
    print("\n🤖 Checking EchoMimic v3 models...")

    model_paths = {
        'third_party/echomimic_v3/config/config.yaml': 'Configuration file',
        'third_party/echomimic_v3/models/Wan2.1-Fun-V1.1-1.3B-InP': 'Main model directory',
        'third_party/echomimic_v3/models/wav2vec2-base-960h': 'Wav2Vec model directory'
    }

    missing_models = []

    for model_path, description in model_paths.items():
        if os.path.exists(model_path):
            print(f"  ✅ {model_path}: {description}")
        else:
            missing_models.append(model_path)
            print(f"  ❌ {model_path}: {description} - MISSING")

    if missing_models:
        print(f"\n❌ Missing EchoMimic v3 components: {', '.join(missing_models)}")
        print("Please ensure EchoMimic v3 models are downloaded and placed in third_party/echomimic_v3/")
        return False

    print("\n✅ EchoMimic v3 models found!")
    return True

async def generate_initial_frame_bank():
    """Generate initial frame bank for the avatar"""
    print("\n🎭 Generating initial frame bank...")

    try:
        from frame_bank_generator import FrameBankGenerator

        # Check if frame bank already exists
        if os.path.exists("frame_banks/frame_bank_metadata.json"):
            print("  ✅ Frame bank already exists!")
            return True

        print("  🔄 Creating frame bank (this may take a while)...")

        generator = FrameBankGenerator("avatar_input/vinay_intro.mp4")

        # For now, just create the directory structure and metadata
        # The actual frame generation would happen here with EchoMimic v3

        metadata = await generator.generate_frame_bank()

        print("  ✅ Frame bank generated successfully!")
        print(f"     📁 Expressions: {len(metadata['expressions'])}")

        return True

    except Exception as e:
        print(f"  ❌ Frame bank generation failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")

    directories = [
        'outputs',
        'frame_banks',
        'logs',
        'temp'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ Created: {directory}/")

def test_basic_functionality():
    """Test basic system functionality"""
    print("\n🧪 Testing basic functionality...")

    try:
        # Test frame bank player
        print("  🔄 Testing frame bank player...")
        from frame_bank_generator import FrameBankPlayer

        if os.path.exists("frame_banks/frame_bank_metadata.json"):
            player = FrameBankPlayer("frame_banks")
            print("  ✅ Frame bank player initialized")
        else:
            print("  ⚠️  Frame bank not found, skipping player test")

        # Test server components
        print("  🔄 Testing server components...")
        from realtime_avatar_server import RealTimeAvatarServer

        server = RealTimeAvatarServer()
        print("  ✅ Server components initialized")

        return True

    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        return False

async def main():
    """Main setup routine"""
    print("🚀 Real-Time Avatar System Setup")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Validate input files
    if not validate_input_files():
        print("\n💡 Create avatar_input/ directory and add your video/audio files")
        sys.exit(1)

    # Check EchoMimic models
    if not check_echomimic_models():
        print("\n💡 Download EchoMimic v3 models using download_models.py")
        sys.exit(1)

    # Create directories
    create_directories()

    # Generate frame bank
    if not await generate_initial_frame_bank():
        print("\n⚠️  Frame bank generation failed, but system can still run")

    # Test functionality
    if not test_basic_functionality():
        print("\n⚠️  Some functionality tests failed")

    print("\n" + "=" * 50)
    print("✅ Setup Complete!")
    print("\n🎯 Next Steps:")
    print("1. Generate frame banks: python frame_bank_generator.py")
    print("2. Start the server: python realtime_avatar_server.py")
    print("3. Open browser: http://localhost:8000")
    print("\n📱 For production:")
    print("- Add OpenAI/Anthropic API keys for LLM")
    print("- Configure audio devices for microphone input")
    print("- Optimize frame generation settings")

    return True

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⛔ Setup interrupted by user")
        sys.exit(1)