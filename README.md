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

## Setup
```bash
git clone https://github.com/FlashbackAi/FlashbackAvatars.git && cd FlashbackAvatars

# Vendor EchoMimic v3
mkdir -p third_party && \
  git clone https://github.com/antgroup/echomimic_v3 third_party/echomimic_v3

# Place v3 weights
mkdir -p services/renderer/models/pretrained
# Put downloaded weights under services/renderer/models/pretrained/

# Build RAG index
docker compose -f docker/compose.yml run --rm rag-llm python /app/services/rag-llm/ingest.py /data/corpus

# Bring up stack
docker compose -f docker/compose.yml up --build
