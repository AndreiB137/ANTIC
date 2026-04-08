
import json
from pathlib import Path

from solver.kolmogorov import KolmogorovSolver
from temporal_selector.metrics import get_metric

from .pats import PATS

from typing import Any, Callable
import jax.numpy as jnp
import numpy as np
from collections import deque
from temporal_selector import _pearson_corr
import pickle

class EnstrophySelector(PATS):
    """Physics-aware selector for Navier-Stokes based on enstrophy flux.

    Uses enstrophy change + Pearson correlation to decide whether two
    snapshots are close enough to skip one.

    Parameters
    ----------
    metric : str or Callable[[jnp.ndarray, jnp.ndarray], float]
        Metric to quantify enstrophy change between snapshots.  Can be
        "max_abs" (default) or a custom function that takes two arrays and returns a scalar.
    window_size : int
        Number of consecutive frames within bounds to trigger a selection even if the criteria is not broken, 
        to ensure progress.
    corr_threshold : float
        Minimum Pearson correlation between the current frame and the last selected frame to keep skipping.  
        If the correlation drops below this threshold, the criteria is considered broken.
    """

    def __init__(
        self,
        metric: str | Callable = "max_abs",
        window_size: int = 10,
        corr_threshold: float = 0.90,
    ):
        super().__init__()
        self._metric_name = metric if isinstance(metric, str) else "custom"
        self.metric = get_metric(metric)
        self.corr_metric = get_metric("pearson")
        self.window_size = window_size
        self.corr_threshold = corr_threshold
        # queue stores (field, enstrophy, step_flux) tuples
        self.queue: deque = deque()
        self.maxx_flux = 0.0
        self.flux_sum = 0.0
        self.last_selected = None
        self.last_selected_e = None
        self._e_diff0 = None  # enstrophy diff of first step after last selection
        # previous within-bounds frame for anchor-to-previous on break
        self._prev_field = None
        self._prev_e = None
    
    def init(self, initial_field: jnp.ndarray) -> None:
        """Seed the selector with the initial field and reset internal queues."""
        self.last_selected = initial_field
        self.last_selected_e = self.compute_activity(initial_field)
        self.queue.clear()
        self.maxx_flux = 0.0
        self.flux_sum = 0.0
        self._e_diff0 = None
        self._prev_field = None
        self._prev_e = None

    def compute_activity(self, field: jnp.ndarray) -> float:
        """Compute the enstrophy of the given field."""
        return jnp.mean(field**2)

    def _decide(self, field: jnp.ndarray) -> bool:
        """

        A bit different from other selectors, the True or False decision is not 
        to mean the *field* itself is compressed, but whether to select the *last_selected* frame.
        The selection criteria follows an adaptive flux-based window, where the selector keeps skipping while both hold:
        1. The enstrophy change from the last selected frame is within a factor of the initial change (first step after last selection), 
        where the factor is adapted based on the observed enstrophy flux statistics in the current window (since last selection).
        2. The Pearson correlation between the current frame and the last selected frame is above a threshold.
            If the queue reaches the window_size, the current frame is selected regardless of the criteria to ensure progress.
            If the criteria breaks, the last selected frame is returned and the current frame becomes the new anchor for the next window.
            This way, we can adaptively skip frames during periods of low activity while ensuring we capture important transitions when the enstrophy changes significantly or the field decorrelates.
        
        Since the decision is about whether to select the last_selected frame, the method returns True when the criteria breaks (select last_selected) and False when still within bounds (keep skipping).
        Because the first frame that breaks the criteria is the one that triggers selection, we cannot select the current frame,
        but the *last_selected* frame which is the anchor of the next window.

        Parameters
        ----------
        field : jnp.ndarray
            The current field snapshot to evaluate. Should be in real space (not Fourier).

        """
        curr_e = self.compute_activity(field)

        # --- track per-step enstrophy flux (accumulated since last selection) ---
        if len(self.queue) > 0:
            step_flux = self.metric(curr_e, self.queue[-1][1])
        else:
            step_flux = self.metric(curr_e, self.last_selected_e)

        self.maxx_flux = max(self.maxx_flux, step_flux)
        self.flux_sum += step_flux
        self.queue.append((field, curr_e, step_flux))

        # record first-step diff from anchor
        if self._e_diff0 is None:
            self._e_diff0 = self.metric(curr_e, self.last_selected_e)

        # initial warmup only: need at least window_size frames to
        # bootstrap flux statistics; once maxx_flux > 0 we have history
        if self.maxx_flux == 0.0 and len(self.queue) < self.window_size:
            self._prev_field = field
            self._prev_e = curr_e
            return False

        # --- adaptive factor from flux statistics since last selection ---
        avg_e_flux = self.flux_sum / len(self.queue)
        factor = jnp.sqrt(self.maxx_flux / (avg_e_flux + 1e-8)).item()

        # same criterion as offline: keep skipping while both hold
        e_change = self.metric(curr_e, self.last_selected_e)
        corr_val = self.corr_metric(field, self.last_selected)

        if (e_change / (self._e_diff0 + 1e-8)) <= factor and corr_val >= self.corr_threshold:
            # Still within bounds.  But if the queue has reached
            # window_size, force a selection
            if len(self.queue) < self.window_size:
                self._prev_field = field
                self._prev_e = curr_e
                return False

            # Queue full, all frames passed — select current 
            self.last_selected = field
            self.last_selected_e = curr_e
            self._e_diff0 = None
            self.queue.clear()
            self.flux_sum = 0.0
            self._prev_field = None
            self._prev_e = None
            return True

        # --- criterion broke: select current frame, anchor to previous ---
        if self._prev_field is not None:
            # Anchor to the previous frame (last within bounds)
            self.last_selected = self._prev_field
            self.last_selected_e = self._prev_e
            # Start new window with current frame already in queue
            new_flux = self.metric(curr_e, self.last_selected_e)
            self._e_diff0 = new_flux
            self.queue.clear()
            self.flux_sum = new_flux
            self.queue.append((field, curr_e, new_flux))
        else:
            # No buffered previous (first frame after anchor) — anchor
            # to current, same as original behaviour.
            self.last_selected = field
            self.last_selected_e = curr_e
            self._e_diff0 = None
            self.queue.clear()
            self.flux_sum = 0.0

        self._prev_field = field
        self._prev_e = curr_e
        return True

    def _save_state(self, base_dict: dict[str, Any], path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "queue": self.queue,
            "maxx_flux": float(self.maxx_flux),
            "flux_sum": float(self.flux_sum),
            "last_selected": self.last_selected,
            "last_selected_e": float(self.last_selected_e) if self.last_selected_e is not None else None,
            "_e_diff0": float(self._e_diff0) if self._e_diff0 is not None else None,
            "_prev_field": self._prev_field,
            "_prev_e": float(self._prev_e) if self._prev_e is not None else None,
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
        self.queue = state["queue"]
        self.maxx_flux = state["maxx_flux"]
        self.flux_sum = state["flux_sum"]
        self.last_selected = state["last_selected"]
        self.last_selected_e = state["last_selected_e"]
        self._e_diff0 = state.get("_e_diff0")
        self._prev_field = state.get("_prev_field")
        self._prev_e = state.get("_prev_e")
    
