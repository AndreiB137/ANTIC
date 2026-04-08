
import jax
import json
from pathlib import Path

from .pats import PATS

from typing import Any
import jax.numpy as jnp
import numpy as np
from collections import deque
from typing import Dict
import pickle
try:
    from solver.bssn import BSSNSolver
except ImportError:
    BSSNSolver = None
    print("Warning: BSSNSolver not found. BSSN_MedianSelector will not work without JAX_NR.")

class BSSN_MedianSelector(PATS):
    """Selector using persistent median surge detection.

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
    """

    def __init__(
        self,
        window_size: int = 5,
        surge_factor: float = 1.55,
        history_len: int = 60,
        patience_factor: int = 5,
    ):
        super().__init__()

        self.window_size = window_size
        self.surge_factor = surge_factor
        self.history_len = history_len
        self.patience_factor = patience_factor

        self.median_history: deque[float] = deque(maxlen=history_len)
        self.surge_count: int = 0
        self.patience_limit = self.window_size * self.patience_factor
        self.skip_remaining: int = 0

    def reset(self) -> None:
        """Reset all internal state (history, surge counter, skip window)."""
        super().reset()
        self.median_history.clear()
        self.surge_count = 0
        self.skip_remaining = 0

    def compute_activity(self, bssn_variables: Dict[str, jnp.ndarray], bssn_solver: BSSNSolver) -> float:
        """Compute the absolute Weyl scalar |Psi4| at the extraction radius as the activity signal."""
        extract_psi4_jit = jax.jit(lambda bssn_variables: bssn_solver.extract_psi4(bssn_variables, 14.3))
        return float(jnp.abs(extract_psi4_jit(bssn_variables)))

    def _decide(self, bssn_variables: Dict[str, jnp.ndarray], bssn_solver: BSSNSolver) -> bool:
        """In-situ persistent-median surge detector.

        Keeps every frame during warmup and surges (dense sampling).
        During quiet baselines, keeps one frame then skips the next
        ``window_size`` frames before keeping again.  If a surge is
        detected mid-skip the skip is cancelled and dense sampling
        resumes immediately.
        """
        a = self.compute_activity(bssn_variables, bssn_solver)

        # 1. Warmup: history not yet full → keep every frame to seed median
        if len(self.median_history) < self.median_history.maxlen:
            self.median_history.append(a)
            return True

        # 2. Establish current baseline
        median = float(np.median(list(self.median_history)))
        in_surge = a > median * self.surge_factor

        # 3. Inside a skip window (baseline skip-ahead)
        if self.skip_remaining > 0:
            self.skip_remaining -= 1
            if in_surge:
                # Surge detected mid-skip → stop skipping, keep frame
                self.skip_remaining = 0
                self.median_history.append(a)
                return True
            if self.skip_remaining == 0:
                # Reached end of skip window → keep this frame, start new skip
                self.median_history.append(a)
                self.skip_remaining = self.window_size
                return True
            return False

        # 4. Not in a skip window
        if in_surge:
            # Dense sampling during surge; don't update median
            self.surge_count += 1
            if self.surge_count > self.patience_limit:
                # Prolonged surge → accept as new baseline
                self.median_history.append(a)
                self.surge_count = 0
            return True
        else:
            # Baseline: keep this frame, then skip ahead
            self.surge_count = 0
            self.median_history.append(a)
            self.skip_remaining = self.window_size
            return True

    def _save_state(self, base_dict: dict[str, Any], path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "median_history": self.median_history,
            "surge_count": self.surge_count,
            "skip_remaining": self.skip_remaining,
        }
        state.update(base_dict)
        with open(path / "state.pkl", "wb") as f:
            pickle.dump(state, f)

    def _load_state(self, path: Path) -> None:
        path = Path(path)
        with open(path / "state.pkl", "rb") as f:
            state = pickle.load(f)
        self.median_history = state["median_history"]
        self.surge_count = state["surge_count"]
        self.skip_remaining = state["skip_remaining"]
        


