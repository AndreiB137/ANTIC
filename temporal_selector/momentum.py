"""Distance-based temporal selector (threshold / adaptive).

The :class:`DistanceSelector` uses a configurable distance metric
to decide whether consecutive snapshots are different enough to keep.
It supports both a static threshold and an adaptive (momentum-style)
threshold computed from a rolling window of recent distances.

Works both **in-situ** (via :meth:`step` / :meth:`run`) and
**offline** (via :meth:`select`), with no dependency on any solver.
"""

from __future__ import annotations

from dataclasses import field
from typing import Callable, Optional

import jax.numpy as jnp
from networkx import sigma
import numpy as np

from .base import TemporalSelector
from .metrics import get_metric
from collections import deque

class MomentumSelector(TemporalSelector):
    """
        Select snapshots based on the KdV time-derivative activity.
    """
    def __init__(
        self,
        metric: str | Callable = "max_abs",
        threshold: float = 1e-3,
        window_size: int = 10,
        k: float = 1.0,
    ):
        super().__init__()
        self.metric = get_metric(metric)
        self.threshold = threshold
        self.window_size = window_size
        self.k = k
        self.last_selected = None
        self.mu = None
        self.sigma = None
        
        self.history: deque[float] = deque(maxlen=window_size)

    def decide(self, field: jnp.ndarray) -> bool:
        """Keep the snapshot if its distance from the last selected exceeds mu + k * sigma."""

        if len(self.history) < self.window_size:
            # Warmup phase: just collect stats
            self.warmup_step(field)
            self.mu = jnp.mean(jnp.array(list(self.history)))
            self.sigma = jnp.std(jnp.array(list(self.history)))
            return True

        e_t = self.metric(field, self.last_selected)

        self.history.append(e_t)

        if e_t > self.mu + self.k * self.sigma:
            self.selected_snapshots.append(field)
            self.last_selected = field
            self.mu = jnp.mean(jnp.array(list(self.history)))
            self.sigma = jnp.std(jnp.array(list(self.history)))
            return True
        
        return False

    def init_selector(self, initial_field: jnp.ndarray) -> None:
        """Seed the selector with the initial field and reset running statistics."""
        self.last_selected = initial_field
        self.mu = 0.0
        self.sigma = 0.0
