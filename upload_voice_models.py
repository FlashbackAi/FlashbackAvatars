#!/usr/bin/env python3
"""
Upload Voice Chat Models to Hugging Face
Uploads XTTS v2 and Whisper models to the same repo as Avatar models
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import shutil

# Configuration
REPO_ID = "flashback-labs/flashback-avatar-models"  # Same repo as avatar
HF_TOKEN = os.environ.get("HF_TOKEN")

# Model paths - use local models directory
VOICE_MODELS_DIR = Path(__file__).parent / "flashback_voice_chat/models"
XTTS_DIR = VOICE_MODELS_DIR / "xtts_v2"
WHISPER_DIR = VOICE_MODELS_DIR / "whisper"

def check_models_exist():
    """Check if models are downloaded"""
    print("🔍 Checking for models...\n")

    xtts_exists = XTTS_DIR.exists() and any(XTTS_DIR.iterdir()) if XTTS_DIR.exists() else False
    whisper_base = WHISPER_DIR / "base.pt"
    whisper_exists = whisper_base.exists()

    print(f"XTTS v2: {'✅ Found' if xtts_exists else '❌ Not found'}")
    if xtts_exists:
        print(f"  Location: {XTTS_DIR}")
        print(f"  Size: {get_dir_size(XTTS_DIR):.2f} GB")

    print(f"\nWhisper (base): {'✅ Found' if whisper_exists else '❌ Not found'}")
    if whisper_exists:
        print(f"  Location: {whisper_base}")
        print(f"  Size: {whisper_base.stat().st_size / (1024**2):.2f} MB")

    print()

    if not xtts_exists or not whisper_exists:
        print("❌ Models not found! Place them in flashback_voice_chat/models/")
        print("\nExpected structure:")
        print("  flashback_voice_chat/models/")
        print("  ├── xtts_v2/")
        print("  │   ├── model.pth")
        print("  │   ├── config.json")
        print("  │   ├── vocab.json")
        print("  │   └── speakers_xtts.pth")
        print("  └── whisper/")
        print("      └── base.pt")
        print()
        return False

    return True

def get_dir_size(path):
    """Get directory size in GB"""
    total = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
    return total / (1024**3)

def upload_models():
    """Upload models to Hugging Face"""

    if not HF_TOKEN:
        print("❌ HF_TOKEN environment variable not set!")
        print("   Set it with: export HF_TOKEN=your_token_here")
        return

    print(f"📤 Uploading to: {REPO_ID}\n")

    api = HfApi(token=HF_TOKEN)

    # Create repo if doesn't exist (likely already exists from avatar upload)
    try:
        create_repo(REPO_ID, token=HF_TOKEN, exist_ok=True)
        print(f"✅ Repository ready: {REPO_ID}\n")
    except Exception as e:
        print(f"⚠️  Repo might already exist: {e}\n")

    # Upload XTTS v2
    print("=" * 70)
    print("📤 Uploading XTTS v2 (Voice Cloning Model)...")
    print("=" * 70)

    try:
        api.upload_folder(
            folder_path=str(XTTS_DIR),
            repo_id=REPO_ID,
            path_in_repo="voice_models/xtts_v2",
            token=HF_TOKEN
        )
        print("✅ XTTS v2 uploaded successfully!\n")
    except Exception as e:
        print(f"❌ XTTS upload failed: {e}\n")

    # Upload Whisper base model
    print("=" * 70)
    print("📤 Uploading Whisper base (Speech-to-Text Model)...")
    print("=" * 70)

    try:
        api.upload_file(
            path_or_fileobj=str(WHISPER_DIR / "base.pt"),
            path_in_repo="voice_models/whisper/base.pt",
            repo_id=REPO_ID,
            token=HF_TOKEN
        )
        print("✅ Whisper base uploaded successfully!\n")
    except Exception as e:
        print(f"❌ Whisper upload failed: {e}\n")

    print("=" * 70)
    print("🎉 Voice models upload complete!")
    print("=" * 70)
    print(f"\nModels available at: https://huggingface.co/{REPO_ID}/tree/main/voice_models")
    print("\nUploaded:")
    print("  📁 voice_models/xtts_v2/          - XTTS v2 voice cloning model")
    print("  📄 voice_models/whisper/base.pt   - Whisper speech-to-text model")

def main():
    print("=" * 70)
    print("Voice Chat Models Upload to Hugging Face")
    print("=" * 70)
    print()

    # Check if models exist
    if not check_models_exist():
        return

    # Confirm upload
    print("=" * 70)
    response = input("Upload models to Hugging Face? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        upload_models()
    else:
        print("❌ Upload cancelled")

if __name__ == "__main__":
    main()
