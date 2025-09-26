# Next Steps

## 1) Persona onboarding
- Capture clean head-and-shoulders video + audio.
- Extract neutral portrait PNG.
- Record 30–60 s clean speech for voice clone (XTTS/OpenVoice).
- Add voice under `services/tts/voices/{user_id}/`.

## 2) RAG corpus
- Place documents in /corpus (markdown, PDFs → text).
- Run `ingest.py`. Enable "answer only if retrieved" guardrail.

## 3) Renderer tuning
- `num_inference_steps=6–8`, `cfg=2–3`, fixed `seed` per utterance.
- Buffer 2.5 s; 15 FPS; 512×512.
- Idle: blink 12–20/min; micro-gaze every 2–5 s.

## 4) Cinematic preset
- Keep single prompt in .env:  
  `"cinematic portrait, soft studio key light, shallow DoF, filmic color"`
- Live PostFX LUT (2383-like), intensity 0.35.

## 5) Barge-in & pacing
- Client VAD → `user_speaking` → orchestrator sends `cancel`.
- Renderer switches to idle immediately (no black frames).
- Resume listen → think → speak.

## 6) Production
- Metrics: FPS, start latency, A/V skew, drop-frame%.
- Preload models at container start to avoid cold start.
- One renderer container per GPU.

## 7) Integrate with your React/Vite site
- Keep this stack as backend.
- In your React app, add a WebRTC client to attach the persistent video track and the control WebSocket.
- Reuse the same `/session` and `/answer` contracts.



- Clone the upstream repos listed in ProjectStartingSteps.md.
- Vendor EchoMimic v3 into `third_party/echomimic_v3` and wire up weights.
- Implement service skeletons in `services/`.
- Flesh out Dockerfiles and compose resources per service requirements.
- Replace `web-samples/minimal-webrtc-client` with the preferred frontend once ready.
