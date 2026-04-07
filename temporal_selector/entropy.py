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

from .base import TemporalSelector

def _joint_histogram(
    field_a: jnp.ndarray,
    field_b: jnp.ndarray,
    n_bins: int = 128,
) -> jnp.ndarray:
    """Compute a normalised 2-D joint histogram of two fields."""
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
    """Shannon entropy of a discrete probability distribution."""
    return -jnp.sum(p * jnp.log(p))


def _kl_divergence(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """Kullback-Leibler divergence from distribution *q* to *p*."""
    return jnp.sum(p * jnp.log(p / q))

def jensen_shannon_divergence(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """Symmetric Jensen-Shannon divergence between two distributions."""
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def normalised_mutual_information(
    field_a: jnp.ndarray, field_b: jnp.ndarray, n_bins: int = 128,
) -> jnp.ndarray:
    """Compute the normalised mutual information (NMI) between two fields via their joint histogram."""
    joint = _joint_histogram(field_a, field_b, n_bins)
    marginal_a = jnp.sum(joint, axis=1)
    marginal_b = jnp.sum(joint, axis=0)
    return (_entropy(marginal_a) + _entropy(marginal_b)) / _entropy(joint.ravel())


def residual_differential_entropy(
    field_a: jnp.ndarray, field_b: jnp.ndarray, n_bins: int = 256,
) -> jnp.ndarray:
    """Estimate the differential entropy of the residual (field_b - field_a) via histogram."""
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
    """Compute the Shannon entropy of the radially-binned power spectrum of a 2-D field."""
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


class InformationSelector(TemporalSelector):
    """Select keyframes using information-theoretic divergences.

    Inherits from :class:`TemporalSelector` so it works both in-situ
    (via :meth:`step` / :meth:`run`) and offline (via :meth:`select`).

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
    on_keep : callable, optional
        ``on_keep(timestep, field)`` — called whenever a snapshot is kept.
    on_skip : callable, optional
        ``on_skip(timestep, field)`` — called for skipped snapshots.

    Examples
    --------
    **Offline**::

        sel = InformationSelector(method="jsd", threshold=0.05)
        indices, kept = sel.select(trajectory)

    **In-situ**::

        sel = InformationSelector(method="residual")
        for t, field in enumerate(fields):
            kept = sel.step(field, t)
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
        super().__init__()
        self.method = method
        self.threshold = threshold if threshold is not None else self._DEFAULTS.get(method, 0.05)
        self.n_bins = n_bins
        self.resolution = resolution

        self._distances: list[float] = []
        self._ref_spectral_entropy: float | None = None

    # ------------------------------------------------------------------
    # TemporalSelector interface
    # ------------------------------------------------------------------

    def _decide(self, field: jnp.ndarray):
        """Dispatch to the active method-specific decision function (JSD, residual, spectral, or MI)."""
        ref = self._ref_field
        flat_cur = field.ravel()
        flat_ref = ref.ravel()

        if self.method == "jsd":
            return self._decide_jsd(flat_cur, flat_ref)
        elif self.method == "residual":
            return self._decide_residual(flat_cur, flat_ref)
        elif self.method == "spectral":
            return self._decide_spectral(field)
        elif self.method == "mi":
            return self._decide_mi(flat_cur, flat_ref)
        else:
            raise ValueError(f"Unknown method {self.method!r}")

    def _decide_jsd(self, cur: jnp.ndarray, ref: jnp.ndarray) -> bool:
        """Keep the snapshot if the Jensen-Shannon divergence from the reference exceeds the threshold."""
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
        self._distances.append(float(d))
        return d > self.threshold

    def _decide_residual(self, cur: jnp.ndarray, ref: jnp.ndarray) -> bool:
        """Keep the snapshot if the differential entropy of the residual exceeds the threshold."""
        d = residual_differential_entropy(ref, cur, self.n_bins)
        self._distances.append(float(d))
        return d > self.threshold

    def _decide_spectral(self, field: jnp.ndarray) -> bool:
        """Keep the snapshot if the absolute change in spectral entropy exceeds the threshold."""
        res = self._infer_resolution(field.ravel())
        cur_se = spectral_entropy(field, res)
        if self._ref_spectral_entropy is None:
            self._ref_spectral_entropy = float(spectral_entropy(self._ref_field, res))
        d = jnp.abs(cur_se - self._ref_spectral_entropy)
        self._distances.append(float(d))
        keep = d > self.threshold
        if keep:
            self._ref_spectral_entropy = float(cur_se)
        return keep

    def _decide_mi(self, cur: jnp.ndarray, ref: jnp.ndarray) -> bool:
        """Keep the snapshot if the normalised mutual information drops below the threshold (fields decorrelate)."""
        d = normalised_mutual_information(ref, cur, n_bins=self.n_bins)
        self._distances.append(float(d))
        return d < self.threshold

    def _infer_resolution(self, flat: jnp.ndarray) -> int:
        """Return the spatial resolution, inferring it as sqrt(N) if not explicitly set."""
        if self.resolution is not None:
            return self.resolution
        return int(np.sqrt(flat.shape[0]))
