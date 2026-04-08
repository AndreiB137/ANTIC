"""Neural-network compressors for PDE snapshots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union

from flax import nnx
import jax
import jax.numpy as jnp

from neural_compressor.utils import FilterSpec, resolve_filter

class BaseCompressor(ABC):
    """Base class for any neural network based compressor.

    Every compressor should define a :meth:`compress` and a :meth:`decompress` method,
    which can be the training procedure and the model forward pass. For simplicity,
    one can also implement the __call__ method as an alias for :meth:`decompress`, so that
    the compressor can be used as a drop-in replacement for the original solver's forward pass.

    It is highly recommended you stick with NNX API for defining the model architecture,
    so that you can take advantage of the built-in state management and filtering utilities, but
    more importantly the PyTorch-style API provided by NNX.

    Parameters
    ----------
    model : nnx.Module
        The neural network architecture. 
    norm_stats : dict, optional
        Normalization statistics (e.g. ``{"method": "z-score", "mean": …, "std": …}``).
        When set, and :meth:`decompress` can use them
        to map the network output back to the original (unnormalised)
        scale.
    """

    def __init__(
        self,
        model: nnx.Module,
        norm_stats: dict[str, Any] | None = None,
    ):
        self.model = model

        self.norm_stats: dict[str, Any] = norm_stats or {}

    def count_params(self, filter: nnx.Variable = None) -> int:
        """Count the number of parameters in the model.

        Parameters
        ----------
        filter : nnx.Variable, optional
            If provided, only count parameters matching the filter spec. If None, count all parameters.
            Filter must be of nnx.Variable type or a valid subclass (e.g. nnx.Param, nnx.LoRAParam, etc.).
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
        """Compress original data into the neural representation. For example this can be the training loop of the model."""
        ...

    @abstractmethod
    def decompress(self, *args, **kwargs):
        """Reconstruct data from the compressed representation. For example this can be the forward pass of the model."""
        ...
