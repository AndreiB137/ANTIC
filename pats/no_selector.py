from .pats import PATS

class NoneSelector(PATS):
    """Trivial selector that keeps all snapshots."""

    def __init__(self):
        super().__init__()

    def compute_activity(self, *args, **kwargs) -> float:
        """Return a constant activity value."""
        return 0.0

    def _decide(self, *args, **kwargs) -> bool:
        return True
