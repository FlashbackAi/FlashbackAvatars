FROM python:3.11-slim

WORKDIR /app
COPY services/rag-llm/ /app/

# TODO: Install vLLM runtime and RAG dependencies when packaging strategy is finalized.
RUN pip install --no-cache-dir fastapi uvicorn faiss-cpu

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]