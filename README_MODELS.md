# 🎭 EchoMimic v3 Model Setup

This project uses large pre-trained models that are not stored in Git due to size constraints.

## 📦 Model Files Required

### Main Model (Wan2.1-Fun-V1.1-1.3B-InP)
- `LICENSE.txt` (11 KB)
- `config.json` (249 bytes)
- `Wan2.1_VAE.pth` (507 MB)
- `diffusion_pytorch_model.safetensors` (3.4 GB)
- `models_t5_umt5-xxl-enc-bf16.pth` (11.3 GB)
- `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` (4.7 GB)
- `google/umt5-xxl/` (Tokenizer files - 21 MB total)
  - `special_tokens_map.json`
  - `spiece.model`
  - `tokenizer.json`
  - `tokenizer_config.json`

### Audio Model (wav2vec2-base-960h)
- Downloaded automatically from Hugging Face

**Total Size: ~20GB**

## 🚀 Quick Setup

### Option 1: Automatic Download (Recommended)
```bash
python download_models.py
```

### Option 2: Manual Download
1. Download models from [source]
2. Extract to `services/renderer/models/pretrained/`
3. Verify directory structure:
```
services/renderer/models/pretrained/
├── Wan2.1-Fun-V1.1-1.3B-InP/
│   ├── LICENSE.txt
│   ├── config.json
│   ├── Wan2.1_VAE.pth
│   ├── diffusion_pytorch_model.safetensors
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
│   └── google/umt5-xxl/
│       ├── special_tokens_map.json
│       ├── spiece.model
│       ├── tokenizer.json
│       └── tokenizer_config.json
└── wav2vec2-base-960h/
    ├── config.json
    ├── model.safetensors
    └── [other files]
```

## 🌩️ Cloud Development

For cloud platforms (RunPod, Colab, etc.):

1. **Clone the repository** (models excluded)
2. **Run download script** on cloud instance
3. **Models download directly** to cloud storage

## 🔐 Model Sources

- **EchoMimic v3**: [Original Repository](https://github.com/antgroup/echomimic_v3)
- **Wav2Vec2**: [Hugging Face](https://huggingface.co/facebook/wav2vec2-base-960h)

## ⚠️ Important Notes

- Models are **not included** in Git repository
- First run will download models (may take 30+ minutes)
- Requires **20GB+ free space**
- **GPU with 12GB+ VRAM** recommended for optimal performance
