from fastapi import FastAPI

from routes import session, style

app = FastAPI(title="Flashback Orchestrator")
app.include_router(session.router)
app.include_router(style.router)


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    """Basic liveness endpoint for compose startup sequencing."""
    return {"status": "ok"}