# 🚀 FlashbackAvatars Setup Guide

Complete setup guide for EchoMimic v3 development with cloud GPU support.

## 📋 Prerequisites

- Python 3.10+
- Git
- Hugging Face account (for model hosting)
- Cloud GPU access (RunPod, Colab, etc.) - **Recommended for development**

## 🎯 Quick Setup Process

### 1. **Upload Models to Hugging Face** (One-time setup)

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Upload your models (requires HF token)
python upload_models_to_hf.py
```

### 2. **Set up Git Repository**

```bash
# Initialize Git (if not done)
git init

# Set up submodules for third-party repos
chmod +x setup_submodules.sh
./setup_submodules.sh

# Add your remote
git remote add origin https://github.com/FlashbackAi/FlashbackAvatars.git

# Commit everything (models excluded via .gitignore)
git add .
git commit -m "feat: Initial FlashbackAvatars setup with EchoMimic v3"
git push -u origin main
```

### 3. **Cloud Development Setup**

#### Option A: RunPod (Recommended)
1. Sign up at [runpod.io](https://runpod.io)
2. Choose RTX 4090 (24GB) template
3. Clone your repo: `git clone --recursive https://github.com/FlashbackAi/FlashbackAvatars.git`
4. Run setup: `./cloud_setup.sh`

#### Option B: Google Colab
1. Upload `colab_development.ipynb` to Colab
2. Run all cells
3. Models download automatically

## 📁 Repository Structure

```
FlashbackAvatars/
├── services/
│   └── renderer/
│       ├── echomimic_engine.py      # Main EchoMimic v3 engine
│       ├── server.py                # FastAPI server
│       └── models/pretrained/       # Models (downloaded, not in Git)
├── third_party/                     # Git submodules
│   └── echomimic_v3/               # EchoMimic v3 source
├── cloud_setup.sh                  # Cloud environment setup
├── download_models.py               # Download models from HF
├── upload_models_to_hf.py          # Upload models to HF
└── README_MODELS.md                # Model documentation
```

## 🌩️ Development Workflow

### Local Development (Code Only)
- Edit code locally
- Commit changes to Git
- Models stay on Hugging Face

### Cloud Testing (Full GPU)
- Pull latest code: `git pull`
- Models download automatically
- Test with full 24GB GPU

### Model Management
- **Storage**: Hugging Face Hub (FlashbackLabs/FlashbackAvatars)
- **Download**: Automatic via `download_models.py`
- **Updates**: Re-upload via `upload_models_to_hf.py`

## 🔧 Key Features

### GPU Memory Optimization
- Auto-detects GPU memory (<12GB enables optimizations)
- Sequential model loading with cache clearing
- Memory usage monitoring

### Cloud-First Architecture
- Models hosted separately from code
- Automatic dependency installation
- Port forwarding for web access

### Third-Party Management
- Git submodules for clean dependency management
- Version pinning for reproducibility
- Easy updates without repo bloat

## 🚨 Important Notes

1. **Models are NOT in Git** - They're downloaded from Hugging Face
2. **First run takes time** - ~20GB download
3. **GPU memory required** - 12GB+ recommended, 8GB with optimizations
4. **Third-party repos** - Managed as Git submodules

## 🆘 Troubleshooting

### Out of Memory Error
- Use cloud GPU with 24GB VRAM
- Or force CPU mode: `device="cpu"` in server.py

### Models Not Found
- Run `python download_models.py`
- Check Hugging Face repository access

### Submodule Issues
- Run `git submodule update --init --recursive`
- Or re-run `./setup_submodules.sh`

## 💰 Cost Optimization

- **Development**: Use RunPod (~$0.80/hour)
- **Testing**: Pause instances when not in use
- **Storage**: Models on HF (free), code on GitHub (free)

---

Ready to start? Follow the setup steps above! 🚀
