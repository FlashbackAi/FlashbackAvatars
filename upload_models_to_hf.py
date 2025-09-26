#!/usr/bin/env python3
"""
Upload EchoMimic v3 models to Hugging Face Hub
Repository: https://huggingface.co/FlashbackLabs/FlashbackAvatars
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

def check_file_exists_on_hf(api, repo_id, path_in_repo):
    """Check if a file already exists on Hugging Face"""
    try:
        info = api.get_paths_info(repo_id, paths=[path_in_repo])
        print(f"   🔍 Checking {path_in_repo}: Found")
        return True
    except Exception as e:
        print(f"   🔍 Checking {path_in_repo}: Not found ({str(e)[:50]}...)")
        return False

def upload_models():
    """Upload all model files to Hugging Face"""
    
    # Configuration
    repo_id = "FlashbackLabs/FlashbackAvatars"
    token = input("Enter your Hugging Face token: ").strip()
    
    if not token:
        print("❌ Hugging Face token required!")
        print("Get your token from: https://huggingface.co/settings/tokens")
        return False
    
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
    models_dir = base_dir / "services" / "renderer" / "models" / "pretrained"
    
    print(f"🚀 Uploading models from {models_dir}")
    print("=" * 60)
    
    # Upload main model files
    main_model_dir = models_dir / "Wan2.1-Fun-V1.1-1.3B-InP"
    
    if not main_model_dir.exists():
        print(f"❌ Model directory not found: {main_model_dir}")
        return False
    
    # Files to upload with their paths
    files_to_upload = [
        "LICENSE.txt",
        "config.json", 
        "Wan2.1_VAE.pth",
        "diffusion_pytorch_model.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    ]
    
    # Tokenizer files in google/umt5-xxl/ folder
    tokenizer_files = [
        "google/umt5-xxl/special_tokens_map.json",
        "google/umt5-xxl/spiece.model",
        "google/umt5-xxl/tokenizer.json",
        "google/umt5-xxl/tokenizer_config.json"
    ]
    
    # Calculate checksums and upload
    checksums = {}
    
    # Upload main model files
    for filename in files_to_upload:
        filepath = main_model_dir / filename
        path_in_repo = f"models/Wan2.1-Fun-V1.1-1.3B-InP/{filename}"
        
        if not filepath.exists():
            print(f"⚠️  File not found: {filename}")
            continue
        
        # Check if file already exists on HF
        if check_file_exists_on_hf(api, repo_id, path_in_repo):
            print(f"✅ {filename} already exists on Hugging Face, skipping...")
            continue
        
        file_size = filepath.stat().st_size / (1024**3)  # GB
        print(f"📤 Uploading {filename} ({file_size:.2f} GB)...")
        
        # Calculate MD5
        print("   Calculating checksum...")
        md5_hash = calculate_md5(filepath)
        checksums[filename] = md5_hash
        print(f"   MD5: {md5_hash}")
        
        try:
            # Upload to HF
            upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=token,
                commit_message=f"Upload {filename}"
            )
            print(f"   ✅ Uploaded successfully")
            
        except Exception as e:
            print(f"   ❌ Upload failed: {e}")
            return False
    
    # Upload tokenizer files
    for filename in tokenizer_files:
        filepath = main_model_dir / filename
        path_in_repo = f"models/Wan2.1-Fun-V1.1-1.3B-InP/{filename}"
        
        if not filepath.exists():
            print(f"⚠️  Tokenizer file not found: {filename}")
            continue
        
        # Check if file already exists on HF
        if check_file_exists_on_hf(api, repo_id, path_in_repo):
            print(f"✅ {filename} already exists on Hugging Face, skipping...")
            continue
        
        file_size = filepath.stat().st_size / (1024**2)  # MB
        print(f"📤 Uploading {filename} ({file_size:.2f} MB)...")
        
        # Calculate MD5
        md5_hash = calculate_md5(filepath)
        checksums[filename] = md5_hash
        print(f"   MD5: {md5_hash}")
        
        try:
            # Upload to HF
            upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=token,
                commit_message=f"Upload tokenizer file {filename}"
            )
            print(f"   ✅ Uploaded successfully")
            
        except Exception as e:
            print(f"   ❌ Upload failed: {e}")
            return False
    
    # Upload wav2vec2 model
    wav2vec_dir = models_dir / "wav2vec2-base-960h"
    if wav2vec_dir.exists():
        print(f"📤 Uploading wav2vec2 model...")
        try:
            upload_folder(
                folder_path=str(wav2vec_dir),
                path_in_repo="models/wav2vec2-base-960h",
                repo_id=repo_id,
                token=token,
                commit_message="Upload wav2vec2 model"
            )
            print("   ✅ wav2vec2 uploaded successfully")
        except Exception as e:
            print(f"   ❌ wav2vec2 upload failed: {e}")
    
    # Save checksums to file
    checksums_file = base_dir / "model_checksums.json"
    import json
    with open(checksums_file, 'w') as f:
        json.dump(checksums, f, indent=2)
    
    print(f"\n🎉 Upload complete!")
    print(f"📁 Repository: https://huggingface.co/{repo_id}")
    print(f"🔐 Checksums saved to: {checksums_file}")
    
    return True

def main():
    print("🤗 Hugging Face Model Uploader")
    print("=" * 40)
    print("Repository: FlashbackLabs/FlashbackAvatars")
    print()
    
    # Check if models exist
    models_dir = Path(__file__).parent / "services" / "renderer" / "models" / "pretrained"
    main_model_dir = models_dir / "Wan2.1-Fun-V1.1-1.3B-InP"
    
    if not main_model_dir.exists():
        print(f"❌ Models not found at: {main_model_dir}")
        print("Please ensure your models are downloaded first.")
        return
    
    print("⚠️  This will upload ~20GB of data to Hugging Face.")
    print("Make sure you have:")
    print("1. A Hugging Face account")
    print("2. Write access to FlashbackLabs/FlashbackAvatars")
    print("3. A stable internet connection")
    print()
    
    response = input("Continue with upload? (y/N): ")
    if response.lower() != 'y':
        print("Upload cancelled.")
        return
    
    success = upload_models()
    
    if success:
        print("\n✅ All models uploaded successfully!")
        print("You can now use the download script on cloud instances.")
    else:
        print("\n❌ Upload failed. Please check errors above.")

if __name__ == "__main__":
    main()
