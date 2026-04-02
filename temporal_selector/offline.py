from __future__ import annotations

from typing import Callable, Literal, Optional

import jax.numpy as jnp
import numpy as np

from .metrics import resolve_metric
from ..solver import Solver, State

class OfflineSelector:
    """Select keyframes from a pre-computed trajectory using distance
    metrics.

    Parameters
    ----------
    metric : str or callable
        Distance function (same semantics as :class:`InSituSelector`).
    method : str
        ``"threshold"`` or ``"momentum"``.
    threshold : float
        Used when ``method="threshold"``.
    window_size : int
        Rolling window for ``method="momentum"``.
    k : float
        Standard-deviation multiplier for ``method="momentum"``.

    Examples
    --------
    ::

        sel = OfflineSelector(metric="l2", method="momentum", window_size=10)
        indices, distances = sel.select(trajectory)
    """

    def __init__(
        self,
        metric: str | Callable = "max_abs",
        method: Literal["threshold", "momentum"] = "threshold",
        threshold: float = 1e-3,
        window_size: int = 10,
        k: float = 1.0,
    ):
        self._metric = resolve_metric(metric)
        self.method = method
        self.threshold = threshold
        self.window_size = window_size
        self.k = k

    def select(
        self,
        trajectory: jnp.ndarray | np.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(keyframe_indices, distances)`` for *trajectory*.

        Parameters
        ----------
        trajectory : array, shape ``(T, ...)``

        Returns
        -------
        indices : jnp.ndarray, shape ``(K,)``
        distances : jnp.ndarray, shape ``(T,)``
        """
        trajectory = jnp.asarray(trajectory)
        T = trajectory.shape[0]
        flat = trajectory.reshape(T, -1)

        if self.method == "threshold":
            return self._select_threshold(flat, T)
        elif self.method == "momentum":
            return self._select_momentum(flat, T)
        else:
            raise ValueError(f"Unknown method {self.method!r}")

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _select_threshold(
        self, flat: jnp.ndarray, T: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        selected = [0]
        distances = np.zeros(T, dtype=np.float32)

        for t in range(1, T):
            ref = flat[selected[-1]]
            dist = self._metric(flat[t], ref)
            distances[t] = dist
            if dist > self.threshold:
                selected.append(t)

        return jnp.array(selected, dtype=jnp.int32), jnp.array(distances)

    def _select_momentum(
        self, flat: jnp.ndarray, T: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        selected = [0]
        distances = np.zeros(T, dtype=np.float32)
        history: list[float] = []

        warmup = min(self.window_size, T - 1)
        for t in range(1, warmup + 1):
            ref = flat[selected[-1]]
            dist = self._metric(flat[t], ref)
            distances[t] = dist
            history.append(dist)
            selected.append(t)

        last_selected = selected[-1]

        for t in range(warmup + 1, T):
            dist = self._metric(flat[t], flat[last_selected])
            distances[t] = dist
            history.append(dist)

            recent = history[-self.window_size:]
            mu = np.mean(recent)
            sigma = np.std(recent)

            if dist > mu + self.k * sigma:
                selected.append(t)
                last_selected = t

        return jnp.array(selected, dtype=jnp.int32), jnp.array(distances)

    def __repr__(self) -> str:
        return (
            f"OfflineSelector(method={self.method!r}, "
            f"metric={self._metric.__name__!r})"
        )
