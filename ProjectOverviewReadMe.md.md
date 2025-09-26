# FlashbackAvatars

FlashbackAvatars is a **self-hosted, real-time interactive avatar system**.  
It creates a **persistent, photoreal talking head** that lip-syncs to speech,  
answers questions grounded in your private documents (RAG),  
and maintains a **cinematic look** throughout the conversation.

Unlike API-based services, this runs entirely on your own GPU (H200, A100, 4090, etc.)  
and is packaged for deployment in Docker on cloud GPU nodes (e.g., Akash).

---

## ✨ Features
- **Persistent avatar**: Avatar is always visible in the browser (WebRTC video track).
- **Real-time lip-sync**: EchoMimic v3 generates 12–18 FPS video with smart buffering.
- **Grounded answers only**: RAG (vector search + LLM) ensures the avatar only responds  
  with context you provide (no open hallucinations).
- **Voice cloning**: Avatar speaks in the cloned voice of the user.
- **Cinematic quality**: Fixed cinematic style (via EchoMimic v3 text prompt + LUT).
- **Barge-in**: User can interrupt; avatar cancels speech and returns to listening state.
- **Self-hosted**: No external APIs; fully local model weights.

---

## 🏗 Architecture
```
browser (WebRTC) ⇄ sfu-gateway (Pion ion-sfu)
                      └─ orchestrator-api (FastAPI)
                         ├─ rag-llm   (vLLM + FAISS, context-only answers)
                         ├─ tts       (XTTS v2, PCM + visemes)
                         ├─ renderer  (EchoMimic v3, buffered streaming @15 FPS)
                         └─ postfx    (GPU LUT, cinematic preset)
```

- **Renderer**: EchoMimic v3 diffusion pipeline (audio+image+prompt → video frames).
- **TTS**: XTTS v2 (Coqui) generates audio + viseme timestamps.
- **RAG-LLM**: vLLM serving + FAISS vector store.
- **Orchestrator**: Manages sessions, pacing, barge-in, style control.
- **SFU Gateway**: Handles WebRTC media (video/audio tracks).
- **PostFX**: Real-time cinematic LUT/tone-mapping.

---

## 📂 Project Structure
```
FlashbackAvatars/
├─ README.md
├─ NEXT_STEPS.md
├─ docker/                  # Dockerfiles + compose
├─ services/
│  ├─ orchestrator-api/     # FastAPI orchestrator
│  ├─ sfu-gateway/          # ion-sfu config
│  ├─ rag-llm/              # vLLM + FAISS ingestion
│  ├─ tts/                  # XTTS v2 voice cloning
│  └─ renderer/             # EchoMimic v3 wrapper
├─ third_party/
│  ├─ echomimic_v3/         # vendor model code
│  ├─ vllm/                 # LLM serving
│  ├─ TTS/                  # Coqui XTTS
│  └─ ion-sfu/              # SFU
├─ corpus/                  # your docs for RAG
└─ web-samples/
   └─ minimal-webrtc-client/ # sample frontend
```

---

## ⚙️ Setup

### 1. Prerequisites
- NVIDIA GPU (24 GB VRAM+, H200 recommended for production)
- CUDA 12.1 runtime + drivers
- Docker + nvidia-container-toolkit
- Python 3.10+ (for local tests)

### 2. Get model weights
Download and place under `services/renderer/models/pretrained/`:

- **Wan2.1-Fun-V1.1-1.3B-InP/**
  - `diffusion_pytorch_model.safetensors`
  - `Wan2.1_VAE.pth`
  - `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`
  - `models_t5_umt5-xxl-enc-bf16.pth`
  - configs (`config.json`, `configuration.json`)

- **wav2vec2-base-960h/**
  - `model.safetensors` (or `pytorch_model.bin`)
  - all configs (`config.json`, `tokenizer_config.json`, etc.)

Structure:
```
services/renderer/models/pretrained/
├─ Wan2.1-Fun-V1.1-1.3B-InP/...
└─ wav2vec2-base-960h/...
```

Also prepare:
- `neutral.png` (reference portrait)
- 30–60 s clean voice audio for cloning (XTTS)

### 3. Build RAG index
Place your docs in `corpus/` and run:
```bash
docker compose run --rm rag-llm python /app/services/rag-llm/ingest.py /data/corpus
```

### 4. Launch stack
```bash
docker compose -f docker/compose.yml up --build
```

### 5. Test locally
Visit [http://localhost:5173](http://localhost:5173) (sample client).
- Avatar is visible in idle state.
- Ask a question → avatar speaks with lip-sync in ~1s.

---

## 🧪 Local quick test (without Docker)
Generate one frame:
```bash
cd services/renderer
python test_local.py
# uses neutral.png + tts.wav → outputs out.png
```

Generate a sequence:
```bash
ffmpeg -framerate 15 -i frames/%04d.png -pix_fmt yuv420p out.mp4
```

---

## 🚀 Deployment (Akash / Cloud GPU)
- Provision H200 node with nvidia-container-toolkit.
- Push this repo as Docker build context.
- Deploy with `docker compose` or Helm chart.
- Ensure TURN servers if running across NAT.

---

## 📊 Performance Targets
- **Renderer**: 6–8 diffusion steps, 15 FPS, 512×512.
- **Latency**: 0.8–1.6 s to first word visible.
- **A/V sync skew**: <120 ms.
- **Idle loop**: blink + micro gaze to mask buffering.

---

## 🗺 Roadmap
- Emotion bias: map LLM sentiment → subtle expressions.
- Live upscaler: ESRGAN → 720p.
- Offline “hero re-renders” with EchoMimic v3 higher steps.

---

## 🔒 Security / Consent
- Each user persona isolated under `personas/{id}/`.
- Require consent for use of voice/likeness.
- RAG ensures out-of-context queries are rejected.

---

## 📌 Next Steps
See [NEXT_STEPS.md](./NEXT_STEPS.md) for:
- Persona onboarding
- TTS voice cloning
- RAG guardrails
- Cinematic style configuration
