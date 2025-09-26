# FlashbackAvatars

Live, persistent, photoreal talking head with RAG. Self-hosted. EchoMimic v3 + smart buffering at ~15 FPS. One fixed cinematic style. No per-frame stylization.

## What you get
- Persistent avatar video track on your page (WebRTC SFU).
- Real-time lip-sync while speaking grounded answers.
- Barge-in: user speech cancels avatar mid-utterance.
- “Only trained context” RAG guardrail.
- One cinematic preset (v3 text prompt + CFG + GPU LUT).

## Prereqs
- NVIDIA driver + CUDA runtime.
- Docker + nvidia-container-toolkit.

## Quick Setup

### Local Development
```bash
git clone https://github.com/FlashbackAi/FlashbackAvatars.git && cd FlashbackAvatars

# Automated setup (downloads models and 3rd party libraries)
./setup_submodules.sh
python download_models.py

# Build RAG index
docker compose -f docker/compose.yml run --rm rag-llm python /app/services/rag-llm/ingest.py /data/corpus

# Bring up stack
docker compose -f docker/compose.yml up --build
```

### Cloud GPU Setup (RunPod/Vast.ai)
```bash
git clone https://github.com/FlashbackAi/FlashbackAvatars.git && cd FlashbackAvatars

# One-shot cloud setup (installs dependencies, downloads models)
chmod +x cloud_setup.sh
./cloud_setup.sh

# Start the server
cd services/renderer
uvicorn server:app --host 0.0.0.0 --port 9000
```

## Model Downloads
The `download_models.py` script automatically downloads:
- EchoMimic v3 pretrained weights
- Required model checkpoints
- See `README_MODELS.md` for model details and manual download options
