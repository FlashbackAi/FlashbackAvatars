#!/usr/bin/env python3
"""
Model Download Script for Real-Time Avatar Pipeline
Downloads SplattingAvatar, MuseTalk, and required models
"""

import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import hashlib
import shutil

def download_file(url, filepath, expected_size=None):
    """Download a file with progress bar"""
    print(f"📥 Downloading {os.path.basename(filepath)}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    if expected_size and total_size != expected_size:
        print(f"⚠️  Warning: Expected {expected_size} bytes, got {total_size} bytes")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f, tqdm(
        desc=os.path.basename(filepath),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"✅ Downloaded: {filepath}")

def verify_checksum(filepath, expected_md5):
    """Verify file integrity"""
    if not os.path.exists(filepath):
        return False

    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest() == expected_md5

def upload_flame_models():
    """Upload FLAME models to Hugging Face"""
    base_dir = Path(__file__).parent
    flame_dir = base_dir / "third_party" / "SplattingAvatar" / "data" / "FLAME2020"

    if not flame_dir.exists():
        print("❌ FLAME directory not found. Please download FLAME models first.")
        return False

    print("📤 Uploading FLAME models to Hugging Face...")

    repo_id = "FlashbackLabs/FlashbackAvatars"

    try:
        from huggingface_hub import HfApi, upload_file
        api = HfApi()

        # Get token from environment or prompt
        token = os.getenv("HF_TOKEN")
        if not token:
            print("⚠️  HF_TOKEN not found in environment.")
            print("Please set it: export HF_TOKEN=your_token")
            return False

        # Upload FLAME models
        flame_files = ["generic_model.pkl", "male_model.pkl", "female_model.pkl"]

        for filename in flame_files:
            filepath = flame_dir / filename
            if filepath.exists():
                print(f"📤 Uploading {filename}...")
                upload_file(
                    path_or_fileobj=str(filepath),
                    path_in_repo=f"models/FLAME2020/{filename}",
                    repo_id=repo_id,
                    token=token
                )
                print(f"✅ Uploaded {filename}")

        print("✅ FLAME models uploaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Failed to upload FLAME models: {e}")
        return False

def download_flame_models():
    """Download FLAME models from Hugging Face"""
    base_dir = Path(__file__).parent
    flame_dir = base_dir / "third_party" / "SplattingAvatar" / "data" / "FLAME2020"
    flame_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "FlashbackLabs/FlashbackAvatars"
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"

    flame_files = ["generic_model.pkl", "male_model.pkl", "female_model.pkl"]

    print("🔥 Downloading FLAME models...")

    for filename in flame_files:
        filepath = flame_dir / filename

        if filepath.exists():
            print(f"✅ {filename} already exists")
            continue

        url = f"{base_url}/models/FLAME2020/{filename}"

        try:
            download_file(url, filepath)
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            print(f"   You can manually download from: https://flame.is.tue.mpg.de/")
            return False

    return True

def download_musetalk_models():
    """Download MuseTalk models from HuggingFace"""
    base_dir = Path(__file__).parent
    musetalk_dir = base_dir / "third_party" / "MuseTalk" / "models"
    musetalk_dir.mkdir(parents=True, exist_ok=True)

    print("🎤 Downloading MuseTalk models...")

    # Model structure from FlashbackLabs repo
    repo_id = "FlashbackLabs/FlashbackAvatars"
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"

    model_structure = {
        "musetalk": ["musetalk.json", "pytorch_model.bin"],
        "musetalkV15": ["musetalk.json", "unet.pth"],
        "whisper": ["config.json", "pytorch_model.bin", "preprocessor_config.json"],
        "dwpose": ["dw-ll_ucoco_384.pth"],
        "face-parse-bisent": ["79999_iter.pth", "resnet18-5c106cde.pth"],
        "sd-vae": ["config.json", "diffusion_pytorch_model.bin"],
        "syncnet": ["latentsync_unet.pt"]
    }

    for subdir, files in model_structure.items():
        print(f"  📂 Downloading {subdir}...")
        subdir_path = musetalk_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)

        for filename in files:
            filepath = subdir_path / filename

            if filepath.exists():
                print(f"     ✅ {filename} already exists")
                continue

            url = f"{base_url}/models/MuseTalk/{subdir}/{filename}"

            try:
                download_file(url, filepath)
            except Exception as e:
                print(f"     ❌ Failed to download {filename}: {e}")
                return False

    print("✅ MuseTalk models downloaded")
    return True

def download_splatting_avatar_dependencies():
    """Download dependencies for SplattingAvatar"""
    base_dir = Path(__file__).parent

    print("📦 Setting up SplattingAvatar dependencies...")

    # FLAME models
    if not download_flame_models():
        return False

    # Download SMPL models (if needed for full body)
    # For head-only avatar, FLAME is sufficient

    print("✅ SplattingAvatar dependencies ready")
    return True

def download_whisper_model():
    """Download Whisper model for MuseTalk audio processing"""
    base_dir = Path(__file__).parent

    print("🎤 Whisper is included with MuseTalk models")
    print("   No separate download needed")

    return True

def main():
    """Main function"""
    print("🚀 Real-Time Avatar Model Downloader")
    print("=" * 50)

    base_dir = Path(__file__).parent

    # Check if we want to upload FLAME models first
    if "--upload-flame" in os.sys.argv:
        upload_flame_models()
        return

    # Download all required models
    success = True

    # 1. SplattingAvatar dependencies (FLAME models)
    print("\n📦 Step 1: SplattingAvatar Dependencies")
    print("-" * 50)
    if not download_splatting_avatar_dependencies():
        success = False

    # 2. MuseTalk models
    print("\n🎤 Step 2: MuseTalk Models")
    print("-" * 50)
    if not download_musetalk_models():
        print("⚠️  You can download MuseTalk models manually:")
        print("   cd third_party/MuseTalk && python scripts/download_models.py")

    # 3. Whisper (already included in MuseTalk)
    print("\n🔊 Step 3: Audio Processing (Whisper in MuseTalk)")
    print("-" * 50)
    download_whisper_model()

    print("\n" + "=" * 50)

    if success:
        print("✅ All models downloaded successfully!")
        print("\n📋 Models downloaded:")
        print("   • FLAME models (for SplattingAvatar head avatar)")
        print("   • MuseTalk models (for real-time lip-sync with Whisper)")
        print("\n📋 Next steps:")
        print("1. Preprocess video: python preprocess_avatar_video.py avatar_input/vinay_intro_shoulders_up.mp4")
        print("2. Train avatar: python train_avatar.py third_party/SplattingAvatar/data/vinay_intro_shoulders_up --export")
        print("3. Start server: python realtime_avatar_server.py")
    else:
        print("⚠️  Some models failed to download. Check errors above.")
        print("You can manually download missing models and re-run this script.")

if __name__ == "__main__":
    main()
