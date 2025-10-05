#!/usr/bin/env python3
"""
Upload Models to Hugging Face Hub
Repository: https://huggingface.co/FlashbackLabs/FlashbackAvatars

Uploads:
1. FLAME models (must download manually first from https://flame.is.tue.mpg.de/)
2. MuseTalk models (all subdirectories)
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
import hashlib
from tqdm import tqdm

def calculate_md5(filepath):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def upload_flame_models(api, repo_id, token, base_dir):
    """Upload FLAME models to Hugging Face"""

    flame_dir = base_dir / "third_party" / "SplattingAvatar" / "data" / "FLAME2020"

    if not flame_dir.exists():
        print(f"❌ FLAME directory not found: {flame_dir}")
        print("\n📋 Instructions:")
        print("1. Go to: https://flame.is.tue.mpg.de/")
        print("2. Register with academic email")
        print("3. Download FLAME 2020 models")
        print("4. Extract to: third_party/SplattingAvatar/data/FLAME2020/")
        print("5. Run this script again")
        return False

    print("\n🔥 Uploading FLAME Models")
    print("=" * 60)

    # FLAME files to upload
    flame_files = [
        "generic_model.pkl",
        "male_model.pkl",
        "female_model.pkl"
    ]

    for filename in flame_files:
        filepath = flame_dir / filename
        path_in_repo = f"models/FLAME2020/{filename}"

        if not filepath.exists():
            print(f"⚠️  File not found: {filename}")
            continue

        # Check if file already exists on HF
        try:
            files_in_repo = api.list_repo_files(repo_id, token=token)
            if path_in_repo in files_in_repo:
                print(f"✅ {filename} already exists, skipping...")
                continue
        except Exception as e:
            # If we can't check, just try to upload
            print(f"   ⚠️  Could not check if file exists: {e}")
            pass

        file_size = filepath.stat().st_size / (1024**2)  # MB
        print(f"📤 Uploading {filename} ({file_size:.2f} MB)...")

        try:
            upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=token,
                commit_message=f"Upload FLAME model {filename}"
            )
            print(f"   ✅ Uploaded successfully")
        except Exception as e:
            print(f"   ❌ Upload failed: {e}")
            return False

    return True

def upload_avatar_input(api, repo_id, token, base_dir):
    """Upload avatar input videos to Hugging Face"""

    avatar_input_dir = base_dir / "avatar_input"

    if not avatar_input_dir.exists():
        print(f"❌ avatar_input directory not found: {avatar_input_dir}")
        return False

    print("\n🎬 Uploading Avatar Input Videos")
    print("=" * 60)

    # Get all video files in avatar_input
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(avatar_input_dir.glob(f"*{ext}"))

    if not video_files:
        print("   ⚠️  No video files found in avatar_input/")
        return False

    for video_file in video_files:
        filename = video_file.name
        path_in_repo = f"avatar_input/{filename}"

        # Check if file already exists on HF
        try:
            files_in_repo = api.list_repo_files(repo_id, token=token)
            if path_in_repo in files_in_repo:
                print(f"   ✅ {filename} already exists, skipping...")
                continue
        except Exception as e:
            print(f"   ⚠️  Could not check if file exists: {e}")
            pass

        file_size = video_file.stat().st_size / (1024**2)  # MB
        print(f"   📤 Uploading {filename} ({file_size:.2f} MB)...")

        try:
            upload_file(
                path_or_fileobj=str(video_file),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=token,
                commit_message=f"Upload avatar input video {filename}"
            )
            print(f"   ✅ Uploaded")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    return True

def upload_musetalk_models(api, repo_id, token, base_dir):
    """Upload MuseTalk models to Hugging Face"""

    musetalk_dir = base_dir / "third_party" / "MuseTalk" / "models"

    if not musetalk_dir.exists():
        print(f"❌ MuseTalk models not found: {musetalk_dir}")
        print("Run download_weights.sh in third_party/MuseTalk first")
        return False

    print("\n🎤 Uploading MuseTalk Models")
    print("=" * 60)

    # Model subdirectories and their files
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
        print(f"\n📂 Uploading {subdir}...")

        for filename in files:
            filepath = musetalk_dir / subdir / filename
            path_in_repo = f"models/MuseTalk/{subdir}/{filename}"

            if not filepath.exists():
                print(f"   ⚠️  File not found: {subdir}/{filename}")
                continue

            # Check if file already exists on HF
            try:
                files_in_repo = api.list_repo_files(repo_id, token=token)
                if path_in_repo in files_in_repo:
                    print(f"   ✅ {filename} already exists, skipping...")
                    continue
            except Exception as e:
                # If we can't check, just try to upload
                print(f"   ⚠️  Could not check if file exists: {e}")
                pass

            file_size = filepath.stat().st_size / (1024**2)  # MB
            print(f"   📤 Uploading {filename} ({file_size:.2f} MB)...")

            try:
                upload_file(
                    path_or_fileobj=str(filepath),
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    token=token,
                    commit_message=f"Upload MuseTalk {subdir}/{filename}"
                )
                print(f"   ✅ Uploaded")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                # Don't fail completely, continue with other files

    return True

def main():
    print("🤗 FlashbackAvatars Model Uploader")
    print("=" * 60)
    print("Repository: FlashbackLabs/FlashbackAvatars")
    print()

    # Configuration
    repo_id = "FlashbackLabs/FlashbackAvatars"

    # Get token from environment or prompt
    token = os.getenv("HF_TOKEN")
    if not token:
        token = input("Enter your Hugging Face token (or set HF_TOKEN env var): ").strip()

    if not token:
        print("❌ Hugging Face token required!")
        print("Get your token from: https://huggingface.co/settings/tokens")
        print("Or set: export HF_TOKEN=your_token")
        return

    # Initialize API
    api = HfApi(token=token)

    # Ensure repo exists
    try:
        api.repo_info(repo_id)
        print(f"✅ Repository {repo_id} exists")
    except:
        print(f"Creating repository {repo_id}...")
        create_repo(repo_id, token=token, repo_type="model")

    base_dir = Path(__file__).parent

    # Check what models/videos are available
    flame_dir = base_dir / "third_party" / "SplattingAvatar" / "data" / "FLAME2020"
    musetalk_dir = base_dir / "third_party" / "MuseTalk" / "models"
    avatar_input_dir = base_dir / "avatar_input"

    has_flame = flame_dir.exists() and (flame_dir / "generic_model.pkl").exists()
    has_musetalk = musetalk_dir.exists() and (musetalk_dir / "musetalk" / "pytorch_model.bin").exists()
    has_avatar_input = avatar_input_dir.exists() and any(avatar_input_dir.glob("*.mp4"))

    if not has_flame and not has_musetalk and not has_avatar_input:
        print("\n❌ No models or videos found to upload!")
        print("\n📋 To upload FLAME:")
        print("1. Download from: https://flame.is.tue.mpg.de/")
        print("2. Extract to: third_party/SplattingAvatar/data/FLAME2020/")
        print("\n📋 To upload MuseTalk:")
        print("1. cd third_party/MuseTalk")
        print("2. bash download_weights.sh")
        print("\n📋 To upload avatar videos:")
        print("1. Place videos in: avatar_input/")
        return

    print("\n📦 Found to upload:")
    if has_flame:
        print("   ✅ FLAME models")
    else:
        print("   ❌ FLAME models (not found)")

    if has_musetalk:
        print("   ✅ MuseTalk models")
    else:
        print("   ❌ MuseTalk models (not found)")

    if has_avatar_input:
        video_count = len(list(avatar_input_dir.glob("*.mp4")))
        print(f"   ✅ Avatar input videos ({video_count} files)")
    else:
        print("   ❌ Avatar input videos (not found)")

    print()
    response = input("Continue with upload? (y/N): ")
    if response.lower() != 'y':
        print("Upload cancelled.")
        return

    # Upload models and videos
    success = True

    if has_flame:
        if not upload_flame_models(api, repo_id, token, base_dir):
            success = False

    if has_musetalk:
        if not upload_musetalk_models(api, repo_id, token, base_dir):
            print("⚠️  Some MuseTalk files failed to upload")

    if has_avatar_input:
        upload_avatar_input(api, repo_id, token, base_dir)

    print("\n" + "=" * 60)

    if success:
        print("🎉 Upload complete!")
        print(f"📁 Repository: https://huggingface.co/{repo_id}")
        print("\n✅ Team members can now download with:")
        print("   python download_models.py")
    else:
        print("⚠️  Upload completed with some errors")
        print("Check the output above for details")

if __name__ == "__main__":
    main()
