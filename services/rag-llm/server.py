from fastapi import FastAPI

app = FastAPI(title="Flashback RAG Proxy")


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate")
async def generate() -> dict[str, str]:
    """Placeholder endpoint for vLLM integration."""
    return {"detail": "not yet implemented"}