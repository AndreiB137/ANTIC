
from __future__ import annotations

from .pats import PATS
from typing import Any
import jax.numpy as jnp

class KdVActivitySelector(PATS):
    """Physics-aware selector for KdV using the time-derivative
    |u_t| = |-u_xxx - α·u·u_x| as activity signal.

    Parameters
    ----------
    domain_length : float
        Periodic domain size.
    nonlinearity : float
        Coefficient α in front of the nonlinear term u·u_x.
    threshold_quantile : float
        Quantile of the running activity distribution above which a
        snapshot is kept (0–1).
    window_size : int
        Rolling window for adaptive threshold.
    warmup : int
        Initial frames always kept.
    """

    def __init__(
        self,
        domain_length: float = 10.0 * jnp.pi,
        nonlinearity: float = 1.0,
        threshold_quantile: float = 0.75,
        window_size: int = 20,
        warmup: int = 5,
        **kwargs: Any,
    ):
        super().__init__(warmup=warmup, **kwargs)
        self.domain_length = domain_length
        self.nonlinearity = nonlinearity
        self.threshold_quantile = threshold_quantile
        self.window_size = window_size

    def compute_activity(self, field: jnp.ndarray) -> float:
        """Mean absolute KdV time-derivative over the spatial domain."""
        act = kdv_activity(field, self.domain_length, self.nonlinearity)
        return jnp.mean(act)

    def decide(self, timestep: int) -> bool:
        """Keep if current activity exceeds the rolling quantile."""
        h = self._activity_history
        recent = h[-self.window_size:]
        q = jnp.quantile(jnp.array(recent), self.threshold_quantile)
        return h[-1] > q
    

def _spectral_derivative_1d(
    field: jnp.ndarray, order: int, domain_length: float,
) -> jnp.ndarray:
    n = field.shape[-1]
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=domain_length / n)
    return jnp.fft.ifft((1j * k) ** order * jnp.fft.fft(field, axis=-1), axis=-1).real


def kdv_activity(
    field: jnp.ndarray,
    domain_length: float = 10.0 * jnp.pi,
    nonlinearity: float = 1.0,
) -> jnp.ndarray:
    """Pointwise absolute KdV time-derivative ``|u_t|``."""
    u_x = _spectral_derivative_1d(field, 1, domain_length)
    u_xxx = _spectral_derivative_1d(field, 3, domain_length)
    return jnp.abs(-u_xxx - nonlinearity * field * u_x)