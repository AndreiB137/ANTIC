"""Online normalization statistics trackers for streaming PDE data."""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import pickle

import jax.numpy as jnp
import numpy as np

class WelfordOnline:
    """Sliding-window Welford online algorithm for mean and standard deviation.

    Maintains a fixed-size deque of observations.  When the window is full,
    the oldest sample is removed before the newest one is incorporated so
    that the statistics always reflect the most recent ``window_size``
    values.

    Parameters
    ----------
    window_size : int
        Maximum number of samples kept in the sliding window.

    Examples
    --------
    >>> tracker = WelfordOnline(window_size=1000)
    >>> tracker.update(3.5)
    >>> tracker.update(4.2)
    >>> tracker.mean, tracker.std
    """

    def __init__(self, window_size: int = 1024):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self._buffer: deque[float] = deque(maxlen=window_size)
        self._mean: float = 0.0
        self._m2: float = 0.0  # sum of squared deviations from the mean
        self._count: int = 0

    def update(self, value: float) -> None:
        """Add a new observation, evicting the oldest if the window is full."""
        if self._count == self.window_size:
            # Remove the oldest sample (inverse Welford step)
            old = self._buffer[0]  # will be popped by deque.append
            self._remove(old)

        self._add(value)
        self._buffer.append(value)  # deque handles eviction automatically

    def update_batch(self, values) -> None:
        """Add multiple observations at once."""
        for v in values:
            self.update(float(v))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mean(self) -> float:
        """Current running mean."""
        return self._mean

    @property
    def variance(self) -> float:
        """Current *population* variance over the window."""
        if self._count < 1:
            return 0.0
        return self._m2 / self._count

    @property
    def std(self) -> float:
        """Current population standard deviation over the window."""
        return math.sqrt(self.variance)

    @property
    def count(self) -> int:
        """Number of samples currently in the window."""
        return self._count

    def stats(self) -> dict[str, float]:
        """Return a dict compatible with ``BaseCompressor.norm_stats``."""
        return {"method": "z-score", "mean": self.mean, "std": self.std}

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self._buffer.clear()
        self._mean = 0.0
        self._m2 = 0.0
        self._count = 0

    def save_stats(self, path: str | Path) -> None:
        """Save current stats to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "mean": self._mean,
                "m2": self._m2,
                "count": self._count,
                "buffer": self._buffer,
            }, f)

    def load_stats(self, path: str | Path) -> None:
        """Load stats from a file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Stats file {path} not found.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._mean = data["mean"]
        self._m2 = data["m2"]
        self._count = data["count"]
        self._buffer = data["buffer"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add(self, value: float) -> None:
        """Standard Welford accumulation step."""
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    def _remove(self, value: float) -> None:
        """Inverse Welford step — remove *value* from the running stats."""
        if self._count <= 1:
            self.reset()
            return
        old_mean = self._mean
        self._count -= 1
        delta = value - old_mean
        self._mean = (old_mean * (self._count + 1) - value) / self._count
        delta2 = value - self._mean
        self._m2 -= delta * delta2

class MinMaxOnline:
    """Sliding-window online min / max tracker.

    Keeps a fixed-size deque and efficiently tracks the running minimum
    and maximum using two monotonic deques (indices), giving O(1)
    amortised per-update min/max queries.

    Parameters
    ----------
    window_size : int
        Maximum number of samples kept in the sliding window.

    Examples
    --------
    >>> tracker = MinMaxOnline(window_size=500)
    >>> tracker.update(1.0)
    >>> tracker.update(5.0)
    >>> tracker.min, tracker.max
    """

    def __init__(self, window_size: int = 1024):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = window_size
        self._buffer: deque[float] = deque(maxlen=window_size)
        # Monotonic deques store *indices* into the logical stream.
        self._min_deque: deque[int] = deque()  # front = index of current min
        self._max_deque: deque[int] = deque()  # front = index of current max
        self._head: int = 0  # total number of values ever pushed

    def update(self, value: float) -> None:
        """Add a new observation, evicting the oldest if the window is full."""
        idx = self._head

        # Maintain ascending monotonic deque for min
        while self._min_deque and self._values_at(self._min_deque[-1]) >= value:
            self._min_deque.pop()
        self._min_deque.append(idx)

        # Maintain descending monotonic deque for max
        while self._max_deque and self._values_at(self._max_deque[-1]) <= value:
            self._max_deque.pop()
        self._max_deque.append(idx)

        self._buffer.append(value)
        self._head += 1

        # Evict indices that have fallen out of the window
        oldest_valid = self._head - len(self._buffer)
        while self._min_deque and self._min_deque[0] < oldest_valid:
            self._min_deque.popleft()
        while self._max_deque and self._max_deque[0] < oldest_valid:
            self._max_deque.popleft()

    def update_batch(self, values) -> None:
        """Add multiple observations at once."""
        for v in values:
            self.update(float(v))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def min(self) -> float:
        """Current minimum in the window."""
        if not self._buffer:
            raise ValueError("No data — call update() first")
        return self._values_at(self._min_deque[0])

    @property
    def max(self) -> float:
        """Current maximum in the window."""
        if not self._buffer:
            raise ValueError("No data — call update() first")
        return self._values_at(self._max_deque[0])

    @property
    def count(self) -> int:
        """Number of samples currently in the window."""
        return len(self._buffer)

    def stats(self) -> dict[str, float]:
        """Return a dict compatible with ``BaseCompressor.norm_stats``."""
        return {"method": "min-max", "min": self.min, "max": self.max}

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self._buffer.clear()
        self._min_deque.clear()
        self._max_deque.clear()
        self._head = 0

    def save_stats(self, path: str | Path) -> None:
        """Save current stats to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "buffer": self._buffer,
                "min_deque": self._min_deque,
                "max_deque": self._max_deque,
                "head": self._head,
            }, f)
    
    def load_stats(self, path: str | Path) -> None:
        """Load stats from a file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Stats file {path} not found.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._buffer = data["buffer"]
        self._min_deque = data["min_deque"]
        self._max_deque = data["max_deque"]
        self._head = data["head"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _values_at(self, absolute_idx: int) -> float:
        """Look up a value by its absolute stream index."""
        oldest_abs = self._head - len(self._buffer)
        return self._buffer[absolute_idx - oldest_abs]


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
