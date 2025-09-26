FROM python:3.11-slim

WORKDIR /app
COPY services/orchestrator-api/ /app/

# TODO: Add precise dependency pins once service requirements are defined.
RUN pip install --no-cache-dir fastapi uvicorn redis pydantic

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]