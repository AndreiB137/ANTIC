"""Information-theoretic temporal selectors.

Provides divergence / entropy functions and an ``InformationSelector``
class that selects keyframes from a trajectory using one of:

    * ``"jsd"``      — Jensen-Shannon divergence of field histograms.
    * ``"residual"`` — differential entropy of the frame-to-frame residual.
    * ``"spectral"`` — absolute change in radial spectral entropy.
    * ``"mi"``       — normalised mutual information (keep when NMI
      **drops below** the threshold, i.e. fields decorrelate).
"""

from __future__ import annotations

from typing import Literal, Optional

import jax.numpy as jnp
import numpy as np


# ======================================================================
# Low-level helpers
# ======================================================================

def _joint_histogram(
    field_a: jnp.ndarray,
    field_b: jnp.ndarray,
    n_bins: int = 128,
) -> jnp.ndarray:
    a_np = np.asarray(field_a.ravel())
    b_np = np.asarray(field_b.ravel())
    lo = min(a_np.min(), b_np.min())
    hi = max(a_np.max(), b_np.max())
    if hi == lo:
        hi = lo + 1.0
    hist, _, _ = np.histogram2d(a_np, b_np, bins=n_bins, range=[[lo, hi], [lo, hi]])
    hist = hist.astype(np.float64) + 1e-12
    hist /= hist.sum()
    return jnp.array(hist, dtype=jnp.float32)


def _entropy(p: jnp.ndarray) -> jnp.ndarray:
    return -jnp.sum(p * jnp.log(p))


def _kl_divergence(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(p * jnp.log(p / q))


# ======================================================================
# Public divergence / entropy functions
# ======================================================================

def jensen_shannon_divergence(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def normalised_mutual_information(
    field_a: jnp.ndarray, field_b: jnp.ndarray, n_bins: int = 128,
) -> jnp.ndarray:
    joint = _joint_histogram(field_a, field_b, n_bins)
    marginal_a = jnp.sum(joint, axis=1)
    marginal_b = jnp.sum(joint, axis=0)
    return (_entropy(marginal_a) + _entropy(marginal_b)) / _entropy(joint.ravel())


def residual_differential_entropy(
    field_a: jnp.ndarray, field_b: jnp.ndarray, n_bins: int = 256,
) -> jnp.ndarray:
    residual = field_b - field_a
    flat = residual.ravel()
    lo, hi = jnp.min(flat), jnp.max(flat)
    hi = jnp.where(hi == lo, lo + 1.0, hi)
    bin_width = (hi - lo) / n_bins
    counts, _ = jnp.histogram(flat, bins=n_bins, range=(lo.item(), hi.item()))
    counts = counts.astype(jnp.float32) + 1e-12
    p = counts / jnp.sum(counts)
    return _entropy(p) + jnp.log(bin_width)


def spectral_entropy(field: jnp.ndarray, resolution: int) -> jnp.ndarray:
    fft2 = jnp.fft.fft2(field.reshape(resolution, resolution))
    power = jnp.abs(fft2) ** 2
    kx = jnp.fft.fftfreq(resolution, d=1.0 / resolution)
    ky = jnp.fft.fftfreq(resolution, d=1.0 / resolution)
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing="ij")
    k_magnitude = jnp.sqrt(kx_grid**2 + ky_grid**2).ravel()
    max_k = resolution // 2
    shell_idx = jnp.clip(jnp.floor(k_magnitude).astype(jnp.int32), 0, max_k - 1)
    spectrum = jnp.zeros(max_k, dtype=jnp.float32).at[shell_idx].add(power.ravel()) + 1e-30
    p = spectrum / jnp.sum(spectrum)
    return _entropy(p)


# ======================================================================
# InformationSelector
# ======================================================================


class InformationSelector:
    """Select keyframes using information-theoretic divergences.

    Parameters
    ----------
    method : str
        ``"jsd"``, ``"residual"``, ``"spectral"``, or ``"mi"``.
    threshold : float or None
        Selection threshold.  ``None`` uses a per-method default.
    n_bins : int
        Histogram bins (JSD / residual / MI).
    resolution : int or None
        Spatial resolution for spectral entropy (inferred if ``None``).

    Examples
    --------
    ::

        sel = InformationSelector(method="jsd", threshold=0.05)
        indices, distances = sel.select(trajectory)
    """

    _DEFAULTS: dict[str, float] = {
        "jsd": 0.05,
        "residual": 2.0,
        "spectral": 0.1,
        "mi": 1.5,
    }

    def __init__(
        self,
        method: Literal["jsd", "residual", "spectral", "mi"] = "jsd",
        threshold: float | None = None,
        n_bins: int = 256,
        resolution: int | None = None,
    ):
        self.method = method
        self.threshold = threshold if threshold is not None else self._DEFAULTS.get(method, 0.05)
        self.n_bins = n_bins
        self.resolution = resolution

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

        dispatch = {
            "jsd": self._select_jsd,
            "residual": self._select_residual,
            "spectral": self._select_spectral,
            "mi": self._select_mi,
        }
        if self.method not in dispatch:
            raise ValueError(f"Unknown method {self.method!r}")
        return dispatch[self.method](flat, T)

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _infer_resolution(self, flat: jnp.ndarray) -> int:
        if self.resolution is not None:
            return self.resolution
        return int(np.sqrt(flat.shape[1]))

    def _select_jsd(
        self, flat: jnp.ndarray, T: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        selected = [0]
        distances = np.zeros(T, dtype=np.float32)

        for t in range(1, T):
            ref = flat[selected[-1]]
            cur = flat[t]
            lo = min(jnp.min(ref), jnp.min(cur))
            hi = max(jnp.max(ref), jnp.max(cur))
            if hi == lo:
                hi = lo + 1.0
            ref_h, _ = jnp.histogram(ref, bins=self.n_bins, range=(lo, hi))
            cur_h, _ = jnp.histogram(cur, bins=self.n_bins, range=(lo, hi))
            ref_h = ref_h.astype(jnp.float32) + 1e-12
            cur_h = cur_h.astype(jnp.float32) + 1e-12
            ref_h = ref_h / jnp.sum(ref_h)
            cur_h = cur_h / jnp.sum(cur_h)
            d = jensen_shannon_divergence(ref_h, cur_h)
            distances[t] = d
            if d > self.threshold:
                selected.append(t)

        return jnp.array(selected, dtype=jnp.int32), jnp.array(distances)

    def _select_residual(
        self, flat: jnp.ndarray, T: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        selected = [0]
        distances = np.zeros(T, dtype=np.float32)

        for t in range(1, T):
            d = residual_differential_entropy(
                flat[selected[-1]], flat[t], self.n_bins,
            )
            distances[t] = d
            if d > self.threshold:
                selected.append(t)

        return jnp.array(selected, dtype=jnp.int32), jnp.array(distances)

    def _select_spectral(
        self, flat: jnp.ndarray, T: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        res = self._infer_resolution(flat)
        selected = [0]
        distances = np.zeros(T, dtype=np.float32)
        ref_se = spectral_entropy(flat[0], res)

        for t in range(1, T):
            cur_se = spectral_entropy(flat[t], res)
            d = jnp.abs(cur_se - ref_se)
            distances[t] = d
            if d > self.threshold:
                selected.append(t)
                ref_se = cur_se

        return jnp.array(selected, dtype=jnp.int32), jnp.array(distances)

    def _select_mi(
        self, flat: jnp.ndarray, T: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        selected = [0]
        distances = np.zeros(T, dtype=np.float32)

        for t in range(1, T):
            d = normalised_mutual_information(
                flat[selected[-1]], flat[t], n_bins=self.n_bins,
            )
            distances[t] = d
            if d < self.threshold:
                selected.append(t)

        return jnp.array(selected, dtype=jnp.int32), jnp.array(distances)

    def __repr__(self) -> str:
        return (
            f"InformationSelector(method={self.method!r}, "
            f"threshold={self.threshold})"
        )
