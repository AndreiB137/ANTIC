"""Shared distance metrics for temporal selectors."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


@jax.jit
def _max_abs(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Maximum absolute element-wise difference between two arrays."""
    return jnp.max(jnp.abs(x - y))


@jax.jit
def _l2_norm(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """L2 norm of the difference between two arrays."""
    return jnp.linalg.norm(x - y)


@jax.jit
def _mae(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Mean absolute error between two arrays."""
    return jnp.mean(jnp.abs(x - y))


@jax.jit
def _mse(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error between two arrays."""
    return jnp.mean((x - y) ** 2)


@jax.jit
def _pearson_corr(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Pearson correlation coefficient between two flattened arrays."""
    return jnp.corrcoef(x.ravel(), y.ravel())[0, 1]


METRIC_PRESETS: dict[str, Callable] = {
    "max_abs": _max_abs,
    "l2": _l2_norm,
    "mae": _mae,
    "mse": _mse,
    "pearson": _pearson_corr,
}


def get_metric(
    metric: str | Callable,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Turn a string preset or callable into a metric function."""
    if callable(metric):
        return metric
    if metric not in METRIC_PRESETS:
        raise ValueError(
            f"Unknown metric {metric!r}. Available: {sorted(METRIC_PRESETS)}"
        )
    return METRIC_PRESETS[metric]
