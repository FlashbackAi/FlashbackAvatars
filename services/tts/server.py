from fastapi import FastAPI

app = FastAPI(title="Flashback TTS Service")


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/synthesize")
async def synthesize() -> dict[str, str]:
    """Placeholder endpoint for XTTS-v2 inference."""
    return {"detail": "not yet implemented"}