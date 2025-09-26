from fastapi import APIRouter

router = APIRouter(prefix="/style", tags=["style"])


@router.post("/set")
async def set_style() -> dict[str, str]:
    """Assign a style preset to the active session."""
    return {"detail": "not yet implemented"}