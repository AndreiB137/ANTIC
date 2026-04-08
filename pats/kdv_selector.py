
from __future__ import annotations
from collections import deque
from pathlib import Path
import json

from .pats import PATS
from typing import Any
import jax.numpy as jnp

from solver.kdv import KDVSolver
import pickle

class KdVActivitySelector(PATS):
    """Physics-aware selector for KdV using the time-derivative
    |u_t| = |-u_xxx - α·u·u_x| as activity signal.

    Parameters
    ----------
    high_quantile : float
        Upper quantile threshold for activity to trigger selection.
    low_quantile : float
        Lower quantile threshold for activity to trigger selection.
    window_size : int
        Rolling window for adaptive threshold.
    """

    def __init__(
        self,
        high_quantile: float = 0.95,
        low_quantile: float = 0.05,
        window_size: int = 10,
    ):
        super().__init__()
        self.high_quantile = high_quantile
        self.low_quantile = low_quantile
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def init(self, initial_field: jnp.ndarray, solver: KDVSolver) -> None:
        """Seed the selector with the initial field and bootstrap the activity history."""
        self.history.clear()
        self.history.append(self.compute_activity(initial_field, solver))

    def compute_activity(self, field: jnp.ndarray, solver: KDVSolver) -> float:
        """Compute the maximum absolute PDE right-hand-side as the KdV activity signal."""
        u_t = solver.pde_rhs(field)
        u_t = solver.to_real(u_t)
        return jnp.max(jnp.abs(u_t))

    def _decide(self, field: jnp.ndarray, solver: KDVSolver) -> bool:
        """Keep the snapshot if its activity falls outside the rolling quantile window."""
        activity = self.compute_activity(field, solver)
        recent = list(self.history)
        q_high = jnp.quantile(jnp.array(recent), self.high_quantile)
        q_low = jnp.quantile(jnp.array(recent), self.low_quantile)
        self.history.append(activity)
        return activity > q_high or activity < q_low

    def _save_state(self, base_dict: dict[str, Any], path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "history": list(self.history),
        }
        state.update(base_dict)
        with open(path / "state.pkl", "wb") as f:
            pickle.dump(state, f)

    def _load_state(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Selector state directory {path} not found.")
        with open(path / "state.pkl", "rb") as f:
            state = pickle.load(f)
        self.history = deque(state["history"], maxlen=self.window_size)
        