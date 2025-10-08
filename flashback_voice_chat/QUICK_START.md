# Voice Chat - Quick Start Guide

## 🚀 Setup (First Time)

### Option 1: Using Virtual Environment (RECOMMENDED)

This isolates voice chat dependencies from your other projects.

**Windows:**
```bash
cd flashback_voice_chat
setup_env.bat
```

**Linux/Mac:**
```bash
cd flashback_voice_chat
chmod +x setup_env.sh
./setup_env.sh
```

### Option 2: Fix Existing Environment

If you already installed in global Python:

```bash
pip install "numpy>=1.26.0" --upgrade
pip install "transformers>=4.30.0" --upgrade
```

---

## 📥 Download Models

From project root:

```bash
# Activate virtual environment (if using)
# Windows: flashback_voice_chat\venv\Scripts\activate
# Linux/Mac: source flashback_voice_chat/venv/bin/activate

# Download models
python download_models_from_source.py
# Choose option 3 (Both models)
```

---

## ▶️ Run Server

**With Virtual Environment:**

Windows:
```bash
cd flashback_voice_chat
venv\Scripts\activate
python server.py
```

Linux/Mac:
```bash
cd flashback_voice_chat
source venv/bin/activate
python server.py
```

**Without Virtual Environment:**
```bash
cd flashback_voice_chat
python server.py
```

---

## 🧪 Test

1. **Check GPU is detected:**
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
   ```

   **Expected**: `CUDA: True, GPU: NVIDIA GeForce RTX 4070`

2. **Open browser:** `http://localhost:8001`

3. **Click "Connect"**

4. **Type message:** "Hi, what's your name?"

5. **Expected:**
   - Response in 1-2 seconds (GPU)
   - Vinay's cloned voice plays
   - Waveform animation shows

---

## 🔧 Troubleshooting

### Error: "module 'numpy' has no attribute 'dtypes'"

**Solution:** Use virtual environment (setup_env.bat/sh) or upgrade numpy:
```bash
pip install "numpy>=1.26.0" --upgrade
```

### Error: "CUDA not available"

**Solution:** Install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Error: "Models not found"

**Solution:** Download models:
```bash
python download_models_from_source.py
```

### Server starts but slow responses (5-10s)

**Cause:** Running on CPU instead of GPU

**Check:**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**If False:** Reinstall PyTorch with CUDA support

---

## 📊 Performance on RTX 4070

- **Model loading**: ~5s (first request only)
- **Response time**: 1-2s (with GPU)
- **Memory usage**: ~2.5 GB VRAM
- **Concurrent users**: 2-3 max

---

## 🎯 Next Steps

1. **Test locally** with your 4070 ✅
2. **Upload models** to Hugging Face (optional):
   ```bash
   export HF_TOKEN=your_token
   python upload_voice_models.py
   ```
3. **Deploy to production** server

---

## 💡 Tips

- **Always use virtual environment** to avoid dependency conflicts
- **Monitor GPU usage:** `nvidia-smi -l 1` (updates every second)
- **First request is slow** (model loading), subsequent requests are fast
- **Test with different messages** to verify voice quality

---

## 📝 Commands Summary

```bash
# Setup (one time)
cd flashback_voice_chat
setup_env.bat  # Windows
# OR
./setup_env.sh  # Linux/Mac

# Download models (one time)
python download_models_from_source.py

# Run server
cd flashback_voice_chat
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Linux/Mac
python server.py

# Test
# Open http://localhost:8001
```

---

## ✅ Checklist

- [ ] Virtual environment created
- [ ] Dependencies installed (no errors)
- [ ] PyTorch CUDA available
- [ ] Models downloaded to `models/` directory
- [ ] Server starts without errors
- [ ] Browser UI loads at http://localhost:8001
- [ ] Can connect and send messages
- [ ] Voice response plays correctly
- [ ] Response time is 1-2 seconds

All green? You're ready to deploy! 🚀
