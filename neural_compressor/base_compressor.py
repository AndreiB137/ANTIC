"""Neural-network compressors for PDE snapshots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

from flax import nnx
import jax
import jax.numpy as jnp

from neural_compressor.utils import FilterSpec, resolve_filter

class BaseCompressor(ABC):
    """Base class for any neural-network-based compressor.

    Subclasses must implement :meth:`compress` and :meth:`reconstruct`.

    Parameters
    ----------
    model : nnx.Module
        The neural-network architecture.
    norm_stats : dict, optional
        Normalization statistics (e.g. ``{"mean": …, "std": …}``).
        When set, :meth:`reconstruct` / :meth:`decompress` can use them
        to map the network output back to the original (unnormalised)
        scale.
    """

    def __init__(
        self,
        model: nnx.Module,
        norm_stats: dict[str, Any] | None = None,
    ):
        self.model = model

        # Normalization bookkeeping
        self.norm_stats: dict[str, Any] = norm_stats or {}

    def count_params(self, filter: nnx.Variable = None) -> int:
        """Count the number of parameters in the model.

        Parameters
        ----------
        filter : str or callable, optional
            If a string, only count parameters whose path contains
            that substring.  If a callable ``(name, array) -> bool``,
            only count parameters for which it returns ``True``.
        """
        if filter is None:
            state = nnx.state(self.model, nnx.Param)
            return sum(p.size for p in jax.tree_util.tree_leaves(state))
        else:
            state = nnx.state(self.model, filter)
            return sum(
                p.size for p in jax.tree_util.tree_leaves(state)
            )

    @abstractmethod
    def compress(self, *args, **kwargs):
        """Compress data.  Subclasses define the signature."""
        ...

    @abstractmethod
    def decompress(self, *args, **kwargs):
        """Reconstruct data from the compressed representation."""
        ...
