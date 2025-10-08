#!/usr/bin/env python3
"""
Download Voice Models from Original Sources
Downloads XTTS v2 and Whisper models to flashback_voice_chat/models/
"""

import os
from pathlib import Path
import shutil

# Target directory
MODELS_DIR = Path(__file__).parent / "flashback_voice_chat/models"

def download_xtts():
    """Download XTTS v2 from Coqui TTS"""
    print("=" * 70)
    print("📥 Downloading XTTS v2 (Voice Cloning Model)...")
    print("=" * 70)

    target_dir = MODELS_DIR / "xtts_v2"
    print(f"Target: {target_dir}\n")

    try:
        from TTS.api import TTS

        # Set environment variable
        os.environ["COQUI_TOS_AGREED"] = "1"

        # Download model (will go to default cache first)
        print("Downloading model files...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

        # Find where it downloaded (default cache location)
        home = Path.home()
        if os.name == 'nt':  # Windows
            cache_dir = home / "AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2"
        else:  # Linux/Mac
            cache_dir = home / ".local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"

        if cache_dir.exists():
            print(f"Found model in cache: {cache_dir}")
            print(f"Copying to: {target_dir}")

            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy all files
            for file in cache_dir.rglob('*'):
                if file.is_file():
                    rel_path = file.relative_to(cache_dir)
                    dest = target_dir / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, dest)
                    print(f"  ✅ {rel_path}")

            print(f"\n✅ XTTS v2 downloaded to: {target_dir}")

            # Calculate size
            total_size = sum(f.stat().st_size for f in target_dir.rglob('*') if f.is_file())
            print(f"   Size: {total_size / (1024**3):.2f} GB\n")
        else:
            print(f"❌ Cache directory not found: {cache_dir}")

    except ImportError:
        print("❌ TTS library not installed!")
        print("   Install with: pip install TTS==0.21.3")
    except Exception as e:
        print(f"❌ Error: {e}")

def download_whisper():
    """Download Whisper base model"""
    print("=" * 70)
    print("📥 Downloading Whisper base (Speech-to-Text Model)...")
    print("=" * 70)

    target_dir = MODELS_DIR / "whisper"
    target_file = target_dir / "base.pt"
    print(f"Target: {target_file}\n")

    try:
        import whisper

        # Download model (will go to default cache first)
        print("Downloading model file...")
        model = whisper.load_model("base")

        # Find where it downloaded
        home = Path.home()
        cache_file = home / ".cache/whisper/base.pt"

        if cache_file.exists():
            print(f"Found model in cache: {cache_file}")
            print(f"Copying to: {target_file}")

            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy file
            shutil.copy2(cache_file, target_file)
            print(f"  ✅ base.pt")

            print(f"\n✅ Whisper base downloaded to: {target_file}")
            print(f"   Size: {target_file.stat().st_size / (1024**2):.2f} MB\n")
        else:
            print(f"❌ Cache file not found: {cache_file}")

    except ImportError:
        print("❌ Whisper library not installed!")
        print("   Install with: pip install openai-whisper==20231117")
    except Exception as e:
        print(f"❌ Error: {e}")

def check_existing():
    """Check what's already downloaded"""
    print("🔍 Checking existing models in flashback_voice_chat/models/\n")

    xtts_dir = MODELS_DIR / "xtts_v2"
    whisper_file = MODELS_DIR / "whisper/base.pt"

    xtts_exists = xtts_dir.exists() and any(xtts_dir.iterdir()) if xtts_dir.exists() else False
    whisper_exists = whisper_file.exists()

    print(f"XTTS v2: {'✅ Already exists' if xtts_exists else '❌ Not found'}")
    if xtts_exists:
        print(f"  Location: {xtts_dir}")

    print(f"\nWhisper base: {'✅ Already exists' if whisper_exists else '❌ Not found'}")
    if whisper_exists:
        print(f"  Location: {whisper_file}")

    print()
    return xtts_exists, whisper_exists

def main():
    print("=" * 70)
    print("Download Voice Models from Original Sources")
    print("=" * 70)
    print(f"Target directory: {MODELS_DIR}")
    print("=" * 70)
    print()

    # Check existing
    xtts_exists, whisper_exists = check_existing()

    # Show menu
    print("=" * 70)
    print("What would you like to download?")
    print("=" * 70)
    print("1. XTTS v2 only (~1.8 GB)")
    print("2. Whisper base only (~140 MB)")
    print("3. Both models (~2 GB total)")
    print("4. Skip")
    print()

    choice = input("Enter choice (1-4): ").strip()

    if choice == "1" or choice == "3":
        if xtts_exists:
            response = input("\n⚠️  XTTS v2 already exists. Re-download? (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                download_xtts()
            else:
                print("⏭️  Skipping XTTS v2")
        else:
            download_xtts()

    if choice == "2" or choice == "3":
        if whisper_exists:
            response = input("\n⚠️  Whisper base already exists. Re-download? (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                download_whisper()
            else:
                print("⏭️  Skipping Whisper base")
        else:
            download_whisper()

    if choice == "4":
        print("⏭️  Skipping download")

    # Final status
    print()
    print("=" * 70)
    print("📁 Final Directory Structure")
    print("=" * 70)
    print()
    print("flashback_voice_chat/models/")

    xtts_dir = MODELS_DIR / "xtts_v2"
    if xtts_dir.exists():
        print("├── xtts_v2/")
        for file in sorted(xtts_dir.rglob('*'))[:5]:  # Show first 5 files
            if file.is_file():
                print(f"│   ├── {file.relative_to(xtts_dir)}")
        if len(list(xtts_dir.rglob('*'))) > 5:
            print("│   └── ...")
    else:
        print("├── xtts_v2/ ❌ (not downloaded)")

    whisper_file = MODELS_DIR / "whisper/base.pt"
    if whisper_file.exists():
        print("└── whisper/")
        print("    └── base.pt ✅")
    else:
        print("└── whisper/ ❌ (not downloaded)")

    print()
    print("=" * 70)
    print("✅ Setup complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Upload to Hugging Face: python upload_voice_models.py")
    print("2. Start voice chat: cd flashback_voice_chat && python3 server.py")

if __name__ == "__main__":
    main()
