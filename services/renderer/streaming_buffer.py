from collections import deque
from typing import Deque


class StreamingBuffer:
    """Maintains rolling audio/frame buffers to align with renderer output."""

    def __init__(self, capacity_seconds: float = 2.5) -> None:
        self.capacity_seconds = capacity_seconds
        self.frames: Deque[bytes] = deque()

    def push(self, frame: bytes) -> None:
        # TODO: Enforce capacity based on TARGET_FPS / BUFFER_SECONDS.
        self.frames.append(frame)

    def pop(self) -> bytes | None:
        return self.frames.popleft() if self.frames else None