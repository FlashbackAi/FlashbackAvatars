FROM python:3.11-slim

WORKDIR /app
COPY services/tts/ /app/

# TODO: Add XTTS-v2 specific dependencies and GPU runtime.
RUN pip install --no-cache-dir fastapi uvicorn

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7000"]