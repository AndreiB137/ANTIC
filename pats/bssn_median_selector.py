
from .pats import PATS

from typing import Any
import jax.numpy as jnp
import numpy as np
from collections import deque

class BSSN_MedianSelector(PATS):
    """Selector using persistent-median surge detection.

    Samples densely during *surges* (activity spikes) and skips
    ahead during quiet baselines.  Originally designed for the BSSN
    Weyl-scalar |Ψ₄| but works with any scalar activity signal.

    Parameters
    ----------
    window_size : int
        Step size when skipping during quiet phases.
    surge_factor : float
        Multiplicative factor above the running median that triggers
        dense (surge) sampling.
    history_len : int
        Max length of the rolling median window.
    patience_factor : int
        After ``patience_factor * window_size`` consecutive surge
        steps, the median baseline is updated to accept the new
        regime.
    warmup : int
        Initial frames always kept (to seed the median).
    """

    def __init__(
        self,
        window_size: int = 5,
        surge_factor: float = 1.55,
        history_len: int = 60,
        patience_factor: int = 5,
        warmup: int = 20,
        **kwargs: Any,
    ):
        super().__init__(warmup=warmup, **kwargs)
        self.window_size = window_size
        self.surge_factor = surge_factor
        self.history_len = history_len
        self.patience_factor = patience_factor

        self._median_history: deque[float] = deque(maxlen=history_len)
        self._surge_count: int = 0

    def reset(self) -> None:
        super().reset()
        self._median_history.clear()
        self._surge_count = 0

    def compute_activity(self, field: jnp.ndarray) -> float:
        """Default: L2 norm of the field.  Override for domain-specific."""
        return jnp.linalg.norm(field)

    def decide(self, timestep: int) -> bool:
        """Keep if in a surge; attempt to skip during quiet phases."""
        a = self._activity_history[timestep] if timestep < len(self._activity_history) else self._activity_history[-1]
        median = np.median(list(self._median_history)) if self._median_history else a

        patience_limit = self.window_size * self.patience_factor

        if a > median * self.surge_factor:
            self._surge_count += 1
            if self._surge_count > patience_limit:
                self._median_history.append(a)
                self._surge_count = 0
            return True
        else:
            self._surge_count = 0
            self._median_history.append(a)
            return False
        


