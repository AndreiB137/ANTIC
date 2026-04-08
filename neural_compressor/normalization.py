"""Online normalization statistics trackers for streaming PDE data."""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import pickle

import jax.numpy as jnp
import numpy as np

# ======================================================================
# Array-aware online trackers for per-feature PDE normalization
# ======================================================================

def _merge_welford(
    n_a: jnp.ndarray, mean_a: jnp.ndarray, m2_a: jnp.ndarray,
    n_b: jnp.ndarray, mean_b: jnp.ndarray, m2_b: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Merge two Welford states (parallel combine)."""
    n = n_a + n_b
    delta = mean_b - mean_a
    mean = mean_a + delta * n_b / jnp.maximum(n, 1)
    m2 = m2_a + m2_b + delta ** 2 * n_a * n_b / jnp.maximum(n, 1)
    return n, mean, m2


def _unmerge_welford(
    n: jnp.ndarray, mean: jnp.ndarray, m2: jnp.ndarray,
    n_b: jnp.ndarray, mean_b: jnp.ndarray, m2_b: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Inverse of :func:`_merge_welford` — remove *b* from combined state."""
    n_a = n - n_b
    mean_a = jnp.where(n_a > 0, (n * mean - n_b * mean_b) / n_a, 0.0)
    delta = mean_b - mean_a
    m2_a = jnp.maximum(m2 - m2_b - delta ** 2 * n_a * n_b / jnp.maximum(n, 1), 0.0)
    return n_a, mean_a, m2_a


class WelfordArrayOnline:
    """Per-feature Welford tracker for streaming arrays of shape ``(N, F)``.

    Each call to :meth:`update` ingests all *N* collocation points for
    every feature, so the running mean / std reflect the distribution
    over ``(T * N,)`` per feature — **not** the mean-of-means.

    Optionally maintains a sliding window of the last ``window_size``
    *snapshots* (not individual points).  Per-snapshot sufficient
    statistics ``(n, mean, m2)`` are stored in a deque and merged on
    each update to recompute window statistics.  Set ``window_size=0``
    (default) for cumulative (no window) mode.

    Parameters
    ----------
    n_features : int
        Number of features *F*.
    window_size : int
        Number of past snapshots whose statistics are kept.
        ``0`` means cumulative (infinite window).
    """

    def __init__(self, n_features: int, window_size: int = 0):
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        self.n_features = n_features
        self.window_size = window_size

        # cumulative state — shape (F,) each
        self._count = jnp.zeros(n_features)
        self._mean = jnp.zeros(n_features)
        self._m2 = jnp.zeros(n_features)

        # per-snapshot summaries for windowed mode
        # each entry: (n: jnp.ndarray, mean: jnp.ndarray, m2: jnp.ndarray)
        self._window: deque[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = deque()

    def update(self, snapshot: jnp.ndarray) -> None:
        """Incorporate a snapshot of shape ``(N, F)`` or ``(N,)`` (F=1)."""
        if snapshot.ndim == 1:
            snapshot = snapshot[:, None]
        n_points = snapshot.shape[0]

        # sufficient stats for this snapshot
        n_b = jnp.full(self.n_features, n_points, dtype=jnp.float32)
        mean_b = jnp.mean(snapshot, axis=0)
        m2_b = jnp.var(snapshot, axis=0) * n_points  # sum of squared devs

        if self.window_size > 0:
            self._window.append((n_b, mean_b, m2_b))
            self._count, self._mean, self._m2 = _merge_welford(
                self._count, self._mean, self._m2, n_b, mean_b, m2_b,
            )
            if len(self._window) > self.window_size:
                old_n, old_mean, old_m2 = self._window.popleft()
                self._count, self._mean, self._m2 = _unmerge_welford(
                    self._count, self._mean, self._m2, old_n, old_mean, old_m2,
                )
        else:
            self._count, self._mean, self._m2 = _merge_welford(
                self._count, self._mean, self._m2, n_b, mean_b, m2_b,
            )

    # ------------------------------------------------------------------
    # Properties — all return shape (F,)
    # ------------------------------------------------------------------

    @property
    def mean(self) -> jnp.ndarray:
        return self._mean

    @property
    def variance(self) -> jnp.ndarray:
        return jnp.where(self._count > 0, self._m2 / self._count, 0.0)

    @property
    def std(self) -> jnp.ndarray:
        return jnp.sqrt(self.variance)

    @property
    def count(self) -> jnp.ndarray:
        return self._count

    def stats(self) -> dict[str, jnp.ndarray]:
        """Return dict compatible with ``BaseCompressor.norm_stats``."""
        return {"method": "z-score", "mean": self.mean, "std": self.std}

    def normalize(self, x: jnp.ndarray) -> jnp.ndarray:
        """Z-score normalize *x* of shape ``(N, F)`` or ``(N,)``."""
        return (x - self.mean) / (self.std + 1e-8)

    def denormalize(self, x: jnp.ndarray) -> jnp.ndarray:
        """Inverse of :meth:`normalize`."""
        return x * (self.std + 1e-8) + self.mean

    def reset(self) -> None:
        self._count = jnp.zeros(self.n_features)
        self._mean = jnp.zeros(self.n_features)
        self._m2 = jnp.zeros(self.n_features)
        self._window.clear()

    def save_stats(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "n_features": self.n_features,
                "window_size": self.window_size,
                "count": np.asarray(self._count),
                "mean": np.asarray(self._mean),
                "m2": np.asarray(self._m2),
                "window": [
                    (np.asarray(n), np.asarray(m), np.asarray(m2))
                    for n, m, m2 in self._window
                ],
            }, f)

    def load_stats(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Stats file {path} not found.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.n_features = data["n_features"]
        self.window_size = data["window_size"]
        self._count = jnp.array(data["count"])
        self._mean = jnp.array(data["mean"])
        self._m2 = jnp.array(data["m2"])
        self._window = deque(
            (jnp.array(n), jnp.array(m), jnp.array(m2))
            for n, m, m2 in data["window"]
        )


class MinMaxArrayOnline:
    """Per-feature online min / max tracker for arrays of shape ``(N, F)``.

    Tracks the global (or windowed) min and max across all collocation
    points for each feature independently.

    Parameters
    ----------
    n_features : int
        Number of features *F*.
    window_size : int
        Number of past snapshots to keep.  ``0`` = cumulative.
    """

    def __init__(self, n_features: int, window_size: int = 0):
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        self.n_features = n_features
        self.window_size = window_size

        self._min = jnp.full(n_features, jnp.inf)
        self._max = jnp.full(n_features, -jnp.inf)

        # per-snapshot (min, max) for windowed mode
        self._window: deque[tuple[jnp.ndarray, jnp.ndarray]] = deque()

    def update(self, snapshot: jnp.ndarray) -> None:
        """Incorporate a snapshot of shape ``(N, F)`` or ``(N,)`` (F=1)."""
        if snapshot.ndim == 1:
            snapshot = snapshot[:, None]

        snap_min = jnp.min(snapshot, axis=0)
        snap_max = jnp.max(snapshot, axis=0)

        if self.window_size > 0:
            self._window.append((snap_min, snap_max))
            if len(self._window) > self.window_size:
                self._window.popleft()
            self._rebuild_from_window()
        else:
            self._min = jnp.minimum(self._min, snap_min)
            self._max = jnp.maximum(self._max, snap_max)

    # ------------------------------------------------------------------
    # Properties — shape (F,)
    # ------------------------------------------------------------------

    @property
    def min(self) -> jnp.ndarray:
        return self._min

    @property
    def max(self) -> jnp.ndarray:
        return self._max

    def stats(self) -> dict[str, jnp.ndarray]:
        """Return dict compatible with ``BaseCompressor.norm_stats``."""
        return {"method": "min-max", "min": self.min, "max": self.max}

    def normalize(self, x: jnp.ndarray) -> jnp.ndarray:
        """Min-max normalize *x* of shape ``(N, F)`` or ``(N,)`` to [0, 1]."""
        r = self.max - self.min
        return (x - self.min) / (r + 1e-8)

    def denormalize(self, x: jnp.ndarray) -> jnp.ndarray:
        """Inverse of :meth:`normalize`."""
        r = self.max - self.min
        return x * (r + 1e-8) + self.min

    def reset(self) -> None:
        self._min = jnp.full(self.n_features, jnp.inf)
        self._max = jnp.full(self.n_features, -jnp.inf)
        self._window.clear()

    def save_stats(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "n_features": self.n_features,
                "window_size": self.window_size,
                "min": np.asarray(self._min),
                "max": np.asarray(self._max),
                "window": [
                    (np.asarray(mn), np.asarray(mx))
                    for mn, mx in self._window
                ],
            }, f)

    def load_stats(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Stats file {path} not found.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.n_features = data["n_features"]
        self.window_size = data["window_size"]
        self._min = jnp.array(data["min"])
        self._max = jnp.array(data["max"])
        self._window = deque(
            (jnp.array(mn), jnp.array(mx))
            for mn, mx in data["window"]
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_from_window(self) -> None:
        mn = jnp.full(self.n_features, jnp.inf)
        mx = jnp.full(self.n_features, -jnp.inf)
        for s_min, s_max in self._window:
            mn = jnp.minimum(mn, s_min)
            mx = jnp.maximum(mx, s_max)
        self._min = mn
        self._max = mx
