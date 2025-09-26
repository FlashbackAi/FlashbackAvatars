class IdleMotionController:
    """Generates micro-motions (blink, gaze) when no audio is playing."""

    def next_frame(self) -> bytes:
        raise NotImplementedError("Idle motion synthesis not implemented")