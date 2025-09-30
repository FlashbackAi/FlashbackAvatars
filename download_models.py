#!/usr/bin/env python3
"""
Model Download Script for EchoMimic v3
Downloads pre-trained models from Hugging Face Hub or other sources
"""

import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import hashlib

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

def download_echomimic_models():
    """Download EchoMimic v3 models"""

    base_dir = Path(__file__).parent
    models_dir = base_dir / "third_party" / "echomimic_v3" / "models"
    
    # Model URLs from FlashbackLabs Hugging Face repository
    repo_id = "FlashbackLabs/FlashbackAvatars"
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"
    
    models = {
        "Wan2.1-Fun-V1.1-1.3B-InP": {
            "files": {
                "LICENSE.txt": {
                    "url": f"{base_url}/models/Wan2.1-Fun-V1.1-1.3B-InP/LICENSE.txt",
                    "md5": None  # Will be updated after upload
                },
                "config.json": {
                    "url": f"{base_url}/models/Wan2.1-Fun-V1.1-1.3B-InP/config.json",
                    "md5": None
                },
                "Wan2.1_VAE.pth": {
                    "url": f"{base_url}/models/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth",
                    "md5": None
                },
                "models_t5_umt5-xxl-enc-bf16.pth": {
                    "url": f"{base_url}/models/Wan2.1-Fun-V1.1-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth",
                    "md5": None
                },
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": {
                    "url": f"{base_url}/models/Wan2.1-Fun-V1.1-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                    "md5": None
                },
                # Tokenizer files
                "google/umt5-xxl/special_tokens_map.json": {
                    "url": f"{base_url}/models/Wan2.1-Fun-V1.1-1.3B-InP/google/umt5-xxl/special_tokens_map.json",
                    "md5": None
                },
                "google/umt5-xxl/spiece.model": {
                    "url": f"{base_url}/models/Wan2.1-Fun-V1.1-1.3B-InP/google/umt5-xxl/spiece.model",
                    "md5": None
                },
                "google/umt5-xxl/tokenizer.json": {
                    "url": f"{base_url}/models/Wan2.1-Fun-V1.1-1.3B-InP/google/umt5-xxl/tokenizer.json",
                    "md5": None
                },
                "google/umt5-xxl/tokenizer_config.json": {
                    "url": f"{base_url}/models/Wan2.1-Fun-V1.1-1.3B-InP/google/umt5-xxl/tokenizer_config.json",
                    "md5": None
                }
            }
        },
        "transformers": {
            "files": {
                "diffusion_pytorch_model.safetensors": {
                    "url": f"{base_url}/models/transformer/diffusion_pytorch_model.safetensors",
                    "md5": None
                }
            }
        },
        "wav2vec2-base-960h": {
            "huggingface_repo": "facebook/wav2vec2-base-960h"
        }
    }
    
    print("🎭 Downloading EchoMimic v3 Models")
    print("=" * 50)
    
    # Download main model files
    main_model_dir = models_dir / "Wan2.1-Fun-V1.1-1.3B-InP"
    main_model_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, info in models["Wan2.1-Fun-V1.1-1.3B-InP"]["files"].items():
        filepath = main_model_dir / filename
        
        # Skip if file exists and checksum matches (if checksum available)
        if filepath.exists():
            if info["md5"] and verify_checksum(filepath, info["md5"]):
                print(f"✅ {filename} already exists and verified")
                continue
            elif not info["md5"]:
                print(f"✅ {filename} already exists (checksum verification skipped)")
                continue
        
        try:
            download_file(info["url"], filepath)
            
            # Verify checksum (if available)
            if info["md5"] and not verify_checksum(filepath, info["md5"]):
                print(f"❌ Checksum mismatch for {filename}")
                os.remove(filepath)
                return False
            elif not info["md5"]:
                print(f"   ⚠️  Checksum verification skipped (not available yet)")
                
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            return False

    # Download transformers model files
    transformers_model_dir = models_dir / "transformers"
    transformers_model_dir.mkdir(parents=True, exist_ok=True)

    for filename, info in models["transformers"]["files"].items():
        filepath = transformers_model_dir / filename

        # Skip if file exists and checksum matches (if checksum available)
        if filepath.exists():
            if info["md5"] and verify_checksum(filepath, info["md5"]):
                print(f"✅ {filename} already exists and verified")
                continue
            elif not info["md5"]:
                print(f"✅ {filename} already exists (checksum verification skipped)")
                continue

        try:
            download_file(info["url"], filepath)

            # Verify checksum (if available)
            if info["md5"] and not verify_checksum(filepath, info["md5"]):
                print(f"❌ Checksum mismatch for {filename}")
                os.remove(filepath)
                return False
            elif not info["md5"]:
                print(f"   ⚠️  Checksum verification skipped (not available yet)")

        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            return False

    # Download Wav2Vec2 model using transformers
    wav2vec_dir = models_dir / "wav2vec2-base-960h"
    
    # Check if wav2vec2 model already exists
    if wav2vec_dir.exists() and (wav2vec_dir / "config.json").exists() and (wav2vec_dir / "model.safetensors").exists():
        print("✅ Wav2Vec2 model already exists, skipping download...")
    else:
        print("🔄 Downloading Wav2Vec2 model...")
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            
            # This will download and cache the model
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            
            # Save to local directory
            processor.save_pretrained(wav2vec_dir)
            model.save_pretrained(wav2vec_dir)
            
            print("✅ Wav2Vec2 model downloaded")
            
        except Exception as e:
            print(f"❌ Failed to download Wav2Vec2: {e}")
            return False
    
    print("\n🎉 All models downloaded successfully!")
    print(f"📁 Models saved to: {models_dir}")
    
    return True

def main():
    """Main function"""
    print("🚀 EchoMimic v3 Model Downloader")
    print("=" * 40)

    # Check if models already exist
    models_dir = Path(__file__).parent / "third_party" / "echomimic_v3" / "models"
    main_model_dir = models_dir / "Wan2.1-Fun-V1.1-1.3B-InP"
    
    # Check if key model files exist
    key_files = [
        "config.json",
        "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    ]

    # Check transformer file in separate directory
    transformer_dir = models_dir / "transformers"
    transformer_file = transformer_dir / "diffusion_pytorch_model.safetensors"
    
    all_exist = (main_model_dir.exists() and
                  all((main_model_dir / f).exists() for f in key_files) and
                  transformer_file.exists())
    
    if all_exist:
        print("✅ Main model files already exist.")
        response = input("Re-download all models? (y/N): ")
        if response.lower() != 'y':
            print("Skipping download. Use existing models.")
            return
    
    success = download_echomimic_models()
    
    if success:
        print("\n✅ Setup complete! You can now run:")
        print("python setup_realtime_avatar.py")
        print("python realtime_avatar_server.py")
    else:
        print("\n❌ Download failed. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
