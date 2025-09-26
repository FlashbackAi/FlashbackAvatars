# Avatar Animation System Setup Guide

## Overview

This guide explains how to run the high-quality avatar animation system individually using EchoMimic v3, and how the system learns your expressions.

## What is EchoMimic v3?

EchoMimic v3 is a **1.3B parameter model** that creates photoreal talking head animations from:
- **Your portrait photo** (single image)
- **Audio input** (speech or any audio)

### Key Features:
- **12GB VRAM minimum** for high-quality generation
- **512×512 resolution** at 15 FPS
- **Real-time lip-sync** with smart buffering
- **Expression learning** from single portrait image
- **Cinematic quality** with built-in styling

## How Expression Learning Works

### 1. **Single Image Input Method**
EchoMimic v3 uses a **diffusion-based approach** that learns from just ONE high-quality portrait:

```python
# From echomimic_engine.py:19-30
def generate_frame(self, image, audio_array, steps=8, cfg=2.5):
    # Your portrait becomes the base for all expressions
    audio_feats = self.audio_proc(audio_array, sampling_rate=16000,
                                  return_tensors="pt").input_values.to(self.device)
    with torch.no_grad():
        out = self.pipe(
            image=image,  # Your portrait photo
            audio_features=audio_feats,  # Drives lip movement & expressions
            num_inference_steps=steps,
            guidance_scale=cfg
        )
    return out.images[0]
```

### 2. **Audio-Driven Expression Generation**
- **Wav2Vec2 audio encoder** extracts speech features from audio
- **Diffusion pipeline** translates audio features into facial movements
- **No training required** - works with pre-trained 1.3B parameter model

### 3. **What the System Learns**
From your single portrait, it automatically generates:
- **Lip synchronization** matching audio phonemes
- **Natural expressions** (smiles, frowns, eyebrow movements)
- **Head movements** and micro-motions
- **Blinking patterns** during idle states
- **Gaze variations** for natural appearance

## Individual Avatar System Setup

### Prerequisites
```bash
# NVIDIA GPU with 12GB+ VRAM (RTX 3080/4070 or better)
# CUDA 12.1+ installed
# Docker with nvidia-container-toolkit
```

### Step 1: Prepare Your Portrait

**Photo Requirements:**
- **Head-and-shoulders framing** (like a passport photo)
- **Neutral expression** works best as base
- **Good lighting** (soft, even lighting)
- **High resolution** (at least 512×512, preferably 1024×1024)
- **Clear face visibility** (no glasses glare, shadows)
- **Frontal pose** (slight angles OK, but avoid profile views)

**Recommended Setup:**
```bash
mkdir -p avatar_input
# Place your portrait as: avatar_input/portrait.jpg
```

### Step 2: Run Avatar Renderer Only

**Option A: Docker Approach (Recommended)**
```bash
# Build only the renderer service
docker build -f docker/renderer.Dockerfile -t avatar-renderer .

# Run with GPU support
docker run --gpus all -p 9000:9000 \
  -v ./avatar_input:/app/input \
  -v ./services/renderer/models:/app/services/renderer/models \
  avatar-renderer
```

**Option B: Direct Python Approach**
```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate xformers fastapi uvicorn

# Run the renderer server
cd services/renderer
uvicorn server:app --host 0.0.0.0 --port 9000
```

### Step 3: Test Avatar Generation

**Simple test script:**
```python
import requests
import base64
from PIL import Image
import numpy as np

# Load your portrait
portrait = Image.open("avatar_input/portrait.jpg")
portrait_array = np.array(portrait)

# Test with sample audio (you can use any WAV file)
audio_file = "test_audio.wav"

# Send to renderer
response = requests.post("http://localhost:9000/render",
                        json={
                            "image": portrait_array.tolist(),
                            "audio_file": audio_file
                        })

# Get animated frame
animated_frame = response.json()["frame"]
```

## Quality Settings

### High Quality Configuration (from .env):
```bash
# Rendering settings
TARGET_FPS=15                    # Smooth playback
BUFFER_SECONDS=2.5              # Reduces stuttering
num_inference_steps=8           # Higher = better quality (slower)
STYLE_CFG=2.5                   # Guidance scale for style

# Cinematic style prompt
STYLE_PROMPT="cinematic portrait, soft studio key light, shallow depth of field, filmic color"
```

### Performance vs Quality Trade-offs:
- **steps=6**: Fast (2-3 sec/frame), good quality
- **steps=8**: Balanced (3-4 sec/frame), high quality
- **steps=12**: Slow (5-6 sec/frame), maximum quality

## Advanced Expression Control

### 1. **Audio-Based Expression Enhancement**
- Use **expressive speech** for more dynamic animations
- **Emotional audio** (happy, sad, excited) influences facial expressions
- **Music** can drive rhythmic head movements

### 2. **Idle Motion System** (services/renderer/idle_motion.py)
```python
# Generates micro-motions when no audio
- Blinking: 12-20 times per minute
- Micro-gaze shifts: every 2-5 seconds
- Subtle head movements
```

### 3. **Custom Style Prompts**
Modify the style for different looks:
```bash
# Professional
STYLE_PROMPT="professional headshot, clean lighting, sharp focus"

# Artistic
STYLE_PROMPT="artistic portrait, dramatic lighting, shallow DOF"

# Natural
STYLE_PROMPT="natural portrait, soft daylight, realistic skin"
```

## Hardware Requirements

### Minimum Setup:
- **GPU**: RTX 3080 (12GB VRAM)
- **CPU**: 8-core modern CPU
- **RAM**: 16GB system RAM
- **Storage**: 10GB for models

### Optimal Setup:
- **GPU**: RTX 4080/4090 (16-24GB VRAM)
- **CPU**: 12+ cores
- **RAM**: 32GB system RAM
- **Storage**: NVMe SSD

## Troubleshooting

### Common Issues:

1. **Out of Memory Error**
   - Reduce `num_inference_steps` to 6
   - Use `torch.float16` instead of `float32`
   - Enable gradient checkpointing

2. **Slow Generation**
   - Check GPU utilization with `nvidia-smi`
   - Enable xformers memory optimization
   - Use TensorRT optimization (advanced)

3. **Poor Expression Quality**
   - Use higher resolution portrait (1024×1024)
   - Ensure good lighting in source photo
   - Try different audio inputs

## Next Steps

Once your individual avatar system works:
1. **Integrate with TTS** for voice synthesis
2. **Add RAG system** for conversational AI
3. **Connect WebRTC** for real-time streaming
4. **Build web interface** for user interaction

This guide focuses on the core avatar animation. The full system integration follows the main README.md instructions.