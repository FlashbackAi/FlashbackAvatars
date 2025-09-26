from fastapi import APIRouter

router = APIRouter(prefix="/session", tags=["session"])


@router.post("/")
async def negotiate_session() -> dict[str, str]:
    """Handle SDP negotiation with the SFU."""
    return {"detail": "not yet implemented"}


@router.post("/answer")
async def submit_answer() -> dict[str, str]:
    """Accept LLM answer payloads from the RAG backend."""
    return {"detail": "not yet implemented"}


@router.post("/cancel")
async def cancel_session() -> dict[str, str]:
    """Tear down active session state."""
    return {"detail": "not yet implemented"}