This document is for creating this project structure and building a form (will be deleted once the form is built)

### Exact repos to clone (only these)

git clone https://github.com/antgroup/echomimic_v3       # renderer backbone (vendor into /third_party)
git clone https://github.com/vllm-project/vllm            # LLM serving
git clone https://github.com/coqui-ai/TTS                 # XTTS-v2 TTS
git clone https://github.com/pion/ion-sfu                 # WebRTC SFU


You will vendor echomimic_v3 under /third_party/echomimic_v3 and build a thin server on top. The others run as separate containers.


### Layout

```
FlashbackAvatars/
├─ README.md
├─ NEXT_STEPS.md
├─ .env
├─ docker/
│  ├─ compose.yml
│  ├─ renderer.Dockerfile
│  ├─ orchestrator.Dockerfile
│  ├─ sfu.Dockerfile
│  ├─ rag.Dockerfile
│  └─ tts.Dockerfile
├─ services/
│  ├─ orchestrator-api/
│  │  ├─ app.py                 # FastAPI: session FSM, barge-in, pacing
│  │  ├─ routes/
│  │  │  ├─ session.py          # /session (SDP), /answer, /cancel
│  │  │  └─ style.py            # set_style (fixed preset params)
│  │  └─ lib/
│  │     ├─ contracts.py        # schemas: audio chunks, visemes
│  │     ├─ bus.py              # redis pub/sub
│  │     └─ state.py            # listening/thinking/speaking
│  ├─ sfu-gateway/
│  │  ├─ main.go                # ion-sfu with config.yaml
│  │  └─ config.yaml
│  ├─ rag-llm/
│  │  ├─ server.py              # vLLM proxy + RAG router
│  │  ├─ ingest.py              # build FAISS index from ./corpus
│  │  └─ config.yaml
│  ├─ tts/
│  │  ├─ server.py              # XTTS v2 HTTP: PCM + viseme timestamps
│  │  └─ voices/                # cloned voice per user
│  └─ renderer/
│     ├─ server.py              # EchoMimic v3 FastAPI (streaming)
│     ├─ echomimic_engine.py    # load weights, steps=6–8, CFG=2–3
│     ├─ streaming_buffer.py    # 2–3 s deque; audio→frames loop @15 FPS
│     ├─ idle_motion.py         # blink/gaze micro-motions
│     └─ models/                # mount weights here
├─ third_party/
│  └─ echomimic_v3/             # vendor repo (no ComfyUI)
├─ corpus/                      # your RAG docs
└─ web-samples/
   └─ minimal-webrtc-client/    # vanilla sample; later swap to your React/Vite

```

### Docker Compose (pinned services)

```
# docker/compose.yml
version: "3.9"
services:
  sfu-gateway:
    build: { context: .., dockerfile: docker/sfu.Dockerfile }
    network_mode: host
    restart: always
    environment: [ SFU_BIND=${SFU_BIND} ]

  orchestrator-api:
    build: { context: .., dockerfile: docker/orchestrator.Dockerfile }
    ports: ["8080:8080"]
    environment:
      - RAG_URL=http://rag-llm:8000
      - TTS_URL=http://tts:7000
      - RENDER_URL=http://renderer:9000
      - TARGET_FPS=${TARGET_FPS}
      - BUFFER_SECONDS=${BUFFER_SECONDS}
      - STYLE_PROMPT=${STYLE_PROMPT}
      - STYLE_CFG=${STYLE_CFG}
    depends_on: [sfu-gateway, rag-llm, tts, renderer]

  rag-llm:
    build: { context: .., dockerfile: docker/rag.Dockerfile }
    ports: ["8000:8000"]
    volumes: ["rag_index:/data/index","../corpus:/data/corpus"]

  tts:
    build: { context: .., dockerfile: docker/tts.Dockerfile }
    ports: ["7000:7000"]
    volumes: ["voices:/app/voices"]

  renderer:
    build: { context: .., dockerfile: docker/renderer.Dockerfile }
    ports: ["9000:9000"]
    deploy:
      resources:
        reservations:
          devices: [{ capabilities: ["gpu"] }]
    volumes:
      - models:/app/services/renderer/models
      - ../third_party/echomimic_v3:/app/third_party/echomimic_v3:ro

volumes: { rag_index: {}, voices: {}, models: {} }
```

### Renderer service specifics (EchoMimic v3 only)

Fixed output: 512×512 @ 15 FPS.

num_inference_steps=6–8, cfg=2–3.

Single style: STYLE_PROMPT baked at startup; no runtime per-frame style.

Accepts 160–320 ms audio chunks + visemes.

Maintains 2.5 s frame buffer; audio is master clock; A/V skew clamp ≤120 ms.

Idle micro-motion when no audio.


### .env (checked in with placeholders)


SFU_BIND=0.0.0.0:7001
TARGET_FPS=15
BUFFER_SECONDS=2.5
STYLE_PROMPT="cinematic portrait, soft studio key light, shallow depth of field, filmic color"
STYLE_CFG=2.5



### Dev test

Open the sample client at web-samples/minimal-webrtc-client (serves a simple page).

Grant mic. You should see the persistent avatar idle.

Send a question → avatar speaks within ~1 s, streams ~15 FPS.


### APIs

POST /session → SDP exchange (persistent tracks)

WS /control → user_speaking|start_reply|cancel|set_style

POST /answer { "text": "..." } → RAG→TTS→Render


### TTS → Renderer payload
{
  "session":"s1",
  "sr":16000,
  "chunks":[{"t0":0,"t1":160,"pcm_b64":"..."}],
  "visemes":[{"t":0,"v":"AA"},{"t":90,"v":"EH"}]
}