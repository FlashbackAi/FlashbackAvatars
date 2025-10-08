#!/usr/bin/env python3
"""
Download Voice Chat Models from Hugging Face
Downloads XTTS v2 and Whisper models only (not avatar models)
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
import shutil

# Configuration
REPO_ID = "flashback-labs/flashback-avatar-models"

# Model paths - use local models directory
VOICE_MODELS_DIR = Path(__file__).parent / "flashback_voice_chat/models"
XTTS_DIR = VOICE_MODELS_DIR / "xtts_v2"
WHISPER_DIR = VOICE_MODELS_DIR / "whisper"

def download_xtts():
    """Download XTTS v2 model"""
    print("=" * 70)
    print("📥 Downloading XTTS v2 (Voice Cloning Model)...")
    print("=" * 70)
    print(f"Target directory: {XTTS_DIR}")
    print()

    # Create directory
    XTTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Download entire XTTS folder from Hugging Face
        snapshot_download(
            repo_id=REPO_ID,
            allow_patterns="voice_models/xtts_v2/*",
            local_dir="temp_voice_models",
            local_dir_use_symlinks=False
        )

        # Move files to correct location
        temp_xtts = Path("temp_voice_models/voice_models/xtts_v2")
        if temp_xtts.exists():
            for file in temp_xtts.rglob('*'):
                if file.is_file():
                    dest = XTTS_DIR / file.relative_to(temp_xtts)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, dest)
                    print(f"  ✅ {file.name}")

            # Cleanup temp directory
            shutil.rmtree("temp_voice_models")

        print(f"\n✅ XTTS v2 downloaded to: {XTTS_DIR}")
        print(f"   Size: {get_dir_size(XTTS_DIR):.2f} GB\n")

    except Exception as e:
        print(f"❌ XTTS download failed: {e}\n")
        if Path("temp_voice_models").exists():
            shutil.rmtree("temp_voice_models")

def download_whisper():
    """Download Whisper base model"""
    print("=" * 70)
    print("📥 Downloading Whisper base (Speech-to-Text Model)...")
    print("=" * 70)
    print(f"Target directory: {WHISPER_DIR}")
    print()

    # Create directory
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Download Whisper base model
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="voice_models/whisper/base.pt",
            local_dir="temp_voice_models",
            local_dir_use_symlinks=False
        )

        # Move to correct location
        dest = WHISPER_DIR / "base.pt"
        shutil.copy2(downloaded_path, dest)

        # Cleanup temp directory
        shutil.rmtree("temp_voice_models")

        print(f"  ✅ base.pt")
        print(f"\n✅ Whisper base downloaded to: {dest}")
        print(f"   Size: {dest.stat().st_size / (1024**2):.2f} MB\n")

    except Exception as e:
        print(f"❌ Whisper download failed: {e}\n")
        if Path("temp_voice_models").exists():
            shutil.rmtree("temp_voice_models")

def get_dir_size(path):
    """Get directory size in GB"""
    total = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
    return total / (1024**3)

def check_existing_models():
    """Check which models are already downloaded"""
    print("🔍 Checking existing models...\n")

    xtts_exists = XTTS_DIR.exists() and any(XTTS_DIR.iterdir())
    whisper_exists = (WHISPER_DIR / "base.pt").exists()

    print(f"XTTS v2: {'✅ Already downloaded' if xtts_exists else '❌ Not found'}")
    if xtts_exists:
        print(f"  Location: {XTTS_DIR}")

    print(f"\nWhisper (base): {'✅ Already downloaded' if whisper_exists else '❌ Not found'}")
    if whisper_exists:
        print(f"  Location: {WHISPER_DIR / 'base.pt'}")

    print()
    return xtts_exists, whisper_exists

def main():
    print("=" * 70)
    print("Voice Chat Models Download from Hugging Face")
    print("=" * 70)
    print(f"Repository: {REPO_ID}")
    print("=" * 70)
    print()

    # Check existing models
    xtts_exists, whisper_exists = check_existing_models()

    # Ask what to download
    print("=" * 70)
    print("What would you like to download?")
    print("=" * 70)
    print("1. XTTS v2 only (~1.8 GB)")
    print("2. Whisper base only (~140 MB)")
    print("3. Both models (~2 GB total)")
    print("4. Skip (models already exist)")
    print()

    choice = input("Enter choice (1-4): ").strip()

    if choice == "1" or choice == "3":
        if xtts_exists:
            response = input("\n⚠️  XTTS v2 already exists. Re-download? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("⏭️  Skipping XTTS v2")
            else:
                download_xtts()
        else:
            download_xtts()

    if choice == "2" or choice == "3":
        if whisper_exists:
            response = input("\n⚠️  Whisper base already exists. Re-download? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("⏭️  Skipping Whisper base")
            else:
                download_whisper()
        else:
            download_whisper()

    if choice == "4":
        print("⏭️  Skipping download")

    print()
    print("=" * 70)
    print("🎉 Download complete!")
    print("=" * 70)
    print()
    print("Models installed:")
    if XTTS_DIR.exists() and any(XTTS_DIR.iterdir()):
        print(f"  ✅ XTTS v2: {XTTS_DIR}")
    if (WHISPER_DIR / "base.pt").exists():
        print(f"  ✅ Whisper base: {WHISPER_DIR / 'base.pt'}")
    print()
    print("You can now run the voice chat server:")
    print("  cd flashback_voice_chat")
    print("  python3 server.py")

if __name__ == "__main__":
    main()
