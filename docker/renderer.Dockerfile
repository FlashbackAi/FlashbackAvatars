FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app
COPY services/renderer/ /app/services/renderer/
COPY third_party/echomimic_v3/ /app/third_party/echomimic_v3/

# TODO: Install renderer dependencies and manage model weights.
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir fastapi uvicorn torch torchvision torchaudio

CMD ["uvicorn", "services.renderer.server:app", "--host", "0.0.0.0", "--port", "9000"]