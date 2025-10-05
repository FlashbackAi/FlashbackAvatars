#!/usr/bin/env python3
"""
MuseTalk Avatar Setup Script
Sets up Vinay's avatar for real-time inference
"""

import os
import sys
import shutil
from pathlib import Path
import yaml

def setup_musetalk_avatar():
    """Set up MuseTalk for Vinay's avatar"""

    base_dir = Path(__file__).parent
    musetalk_dir = base_dir / "third_party" / "MuseTalk"

    # Check if MuseTalk exists
    if not musetalk_dir.exists():
        print("❌ MuseTalk directory not found!")
        return False

    # Check if video exists
    video_path = base_dir / "avatar_input" / "vinay_intro_shoulders_up.mp4"
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        print("Please place Vinay's video at: avatar_input/vinay_intro_shoulders_up.mp4")
        return False

    # Copy video to MuseTalk data directory
    musetalk_data_video = musetalk_dir / "data" / "video" / "vinay.mp4"
    musetalk_data_video.parent.mkdir(parents=True, exist_ok=True)

    if not musetalk_data_video.exists():
        print(f"📹 Copying video to MuseTalk data directory...")
        shutil.copy(video_path, musetalk_data_video)
        print(f"✅ Video copied to: {musetalk_data_video}")
    else:
        print(f"✅ Video already exists: {musetalk_data_video}")

    # Check for sample audio
    audio_path = base_dir / "avatar_input" / "vinay_audio.wav"
    if audio_path.exists():
        musetalk_data_audio = musetalk_dir / "data" / "audio" / "vinay.wav"
        musetalk_data_audio.parent.mkdir(parents=True, exist_ok=True)

        if not musetalk_data_audio.exists():
            print(f"🎵 Copying audio to MuseTalk data directory...")
            shutil.copy(audio_path, musetalk_data_audio)
            print(f"✅ Audio copied to: {musetalk_data_audio}")

    # Create config for Vinay's avatar
    config_path = musetalk_dir / "configs" / "inference" / "vinay_realtime.yaml"

    config_content = {
        'vinay_avatar': {
            'preparation': True,  # First time: prepare avatar
            'bbox_shift': 0,  # Adjust if face detection is off
            'video_path': 'data/video/vinay.mp4',
            'audio_clips': {
                'audio_0': 'data/audio/vinay.wav'
            }
        }
    }

    with open(config_path, 'w') as f:
        yaml.dump(config_content, f, default_flow_style=False)

    print(f"✅ Created config: {config_path}")

    # Check if models are downloaded
    models_dir = musetalk_dir / "models"
    required_models = ['musetalkV15', 'whisper', 'dwpose', 'face-parse-bisent', 'sd-vae']

    missing_models = []
    for model in required_models:
        if not (models_dir / model).exists():
            missing_models.append(model)

    if missing_models:
        print(f"\n⚠️  Missing models: {', '.join(missing_models)}")
        print("Run: python download_models.py")
        return False

    print("\n✅ MuseTalk setup complete!")
    print("\n📋 Next steps:")
    print("\n1. Test MuseTalk inference:")
    print("   cd third_party/MuseTalk")
    print("   python -m scripts.realtime_inference \\")
    print("     --inference_config configs/inference/vinay_realtime.yaml \\")
    print("     --result_dir ../../results/vinay_avatar \\")
    print("     --unet_model_path models/musetalkV15/unet.pth \\")
    print("     --unet_config models/musetalkV15/musetalk.json \\")
    print("     --version v15 --fps 25")
    print("\n2. After first run, set 'preparation: False' in config for faster inference")
    print("\n3. Integrate with your UI server")

    return True


if __name__ == "__main__":
    print("🎬 MuseTalk Avatar Setup")
    print("=" * 60)

    success = setup_musetalk_avatar()

    if not success:
        sys.exit(1)
