#!/usr/bin/env python3
"""
Test script to verify EchoMimic v3 imports and basic functionality
"""

import os
import sys

def test_basic_imports():
    """Test basic Python imports"""
    print("Testing basic imports...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}")
        
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
        
        import diffusers
        print(f"✅ Diffusers {diffusers.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Basic import failed: {e}")
        return False

def test_echomimic_imports():
    """Test EchoMimic v3 specific imports"""
    print("\nTesting EchoMimic v3 imports...")
    
    # Add EchoMimic v3 to path
    echomimic_path = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'echomimic_v3')
    sys.path.append(echomimic_path)
    
    try:
        from omegaconf import OmegaConf
        print("✅ OmegaConf imported")
        
        from src.wan_vae import AutoencoderKLWan
        print("✅ AutoencoderKLWan imported")
        
        from src.wan_image_encoder import CLIPModel
        print("✅ CLIPModel imported")
        
        from src.wan_text_encoder import WanT5EncoderModel
        print("✅ WanT5EncoderModel imported")
        
        from src.wan_transformer3d_audio import WanTransformerAudioMask3DModel
        print("✅ WanTransformerAudioMask3DModel imported")
        
        from src.pipeline_wan_fun_inpaint_audio import WanFunInpaintAudioPipeline
        print("✅ WanFunInpaintAudioPipeline imported")
        
        from src.utils import filter_kwargs
        print("✅ filter_kwargs imported")
        
        return True
    except ImportError as e:
        print(f"❌ EchoMimic import failed: {e}")
        return False

def test_model_paths():
    """Test if model files exist"""
    print("\nTesting model file paths...")
    
    model_dir = os.path.join(os.path.dirname(__file__), "models", "pretrained", "Wan2.1-Fun-V1.1-1.3B-InP")
    wav2vec_dir = os.path.join(os.path.dirname(__file__), "models", "pretrained", "wav2vec2-base-960h")
    
    print(f"Model directory: {model_dir}")
    print(f"Wav2Vec directory: {wav2vec_dir}")
    
    # Check key files
    key_files = [
        os.path.join(model_dir, "Wan2.1_VAE.pth"),
        os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(model_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        os.path.join(model_dir, "diffusion_pytorch_model.safetensors"),
        os.path.join(wav2vec_dir, "config.json"),
        os.path.join(wav2vec_dir, "model.safetensors"),
    ]
    
    all_exist = True
    for file_path in key_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✅ {os.path.basename(file_path)} ({file_size:.1f} MB)")
        else:
            print(f"❌ Missing: {os.path.basename(file_path)}")
            all_exist = False
    
    return all_exist

def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        from omegaconf import OmegaConf
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'echomimic_v3', 'config', 'config.yaml')
        
        if os.path.exists(config_path):
            cfg = OmegaConf.load(config_path)
            print(f"✅ Config loaded: {config_path}")
            print(f"   Pipeline: {cfg.get('pipeline', 'N/A')}")
            print(f"   Format: {cfg.get('format', 'N/A')}")
            return True
        else:
            print(f"❌ Config file not found: {config_path}")
            return False
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 EchoMimic v3 Setup Diagnostic")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("EchoMimic Imports", test_echomimic_imports),
        ("Model Files", test_model_paths),
        ("Config Loading", test_config_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to start the server.")
    else:
        print("\n⚠️  Some tests failed. Please install missing dependencies or check file paths.")
    
    return all_passed

if __name__ == "__main__":
    main()
