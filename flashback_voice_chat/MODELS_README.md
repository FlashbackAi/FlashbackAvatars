# Voice Chat Models

## 📁 Models Directory

All models are stored in `flashback_voice_chat/models/`:

```
models/
├── xtts_v2/              # Voice cloning model (~1.8 GB)
│   ├── model.pth
│   ├── config.json
│   ├── vocab.json
│   └── speakers_xtts.pth
└── whisper/              # Speech-to-text model (~140 MB)
    └── base.pt
```

## 📥 How to Get Models

### Option 1: Download from Hugging Face (Recommended for deployment)

```bash
# From project root
python download_voice_models.py
```

This downloads models from `flashback-labs/flashback-avatar-models` to `flashback_voice_chat/models/`.

### Option 2: Download from original sources (First time setup)

```bash
# From project root
python download_models_from_source.py
```

This downloads models from Coqui/OpenAI and places them in `flashback_voice_chat/models/`.

## 🔄 Models are Auto-Downloaded

If models are missing, the code will automatically download them on first run:
- `voice_cloning.py` will download XTTS v2 if missing
- `server.py` will download Whisper base if missing

However, **pre-downloading is recommended** for faster startup.

## ⚠️ Important

- Models directory is **gitignored** (not committed to repo)
- Download models separately on each deployment
- Total size: ~2 GB
- Models are loaded from local directory first, then auto-downloaded if missing

## 📚 Documentation

See project root for detailed guides:
- `VOICE_CHAT_MODELS_README.md` - Quick start guide
- `VOICE_MODELS_GUIDE.md` - Complete documentation
