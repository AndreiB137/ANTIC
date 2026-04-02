"""Neural-network compressors for PDE snapshots."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from ..temporal_param import ParamManager, FilterSpec

class BaseCompressor(ABC):
    """Base class for any neural-network-based compressor.

    Subclasses must implement :meth:`compress` and :meth:`reconstruct`.

    Parameters
    ----------
    model : nnx.Module
        The neural-network architecture.
    checkpoint_dir : str or Path, optional
        Where to persist parameters.  If ``None``, checkpointing is disabled.
    optimizer_fn : callable, optional
        ``(model) -> nnx.Optimizer``.  Defaults to AdamW + cosine schedule.
    keyframe_every : int, optional
        When set, :meth:`save` automatically stores a full-state keyframe
        every *keyframe_every* timesteps.  This enables cheap random-access
        :meth:`restore` in LoRA workflows instead of sequentially applying
        every delta from the beginning.
    """

    def __init__(
        self,
        model: nnx.Module,
        checkpoint_dir: str | Path | None = None,
        optimizer: nnx.Optimizer | None = None,
        keyframe_every: int | None = None,
    ):
        self.model = model
        self.optimizer = optimizer

        # Checkpoint store
        self._store: ParamManager | None = None
        if checkpoint_dir is not None:
            self._store = ParamManager(
                checkpoint_dir,
                config={"compressor_class": type(self).__qualname__},
            )

        # LoRA bookkeeping
        self._lora_enabled = False
        self._lora_rank: int | None = None

        # Keyframe strategy
        self._keyframe_every = keyframe_every

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compress(self, *args, **kwargs):
        """Compress data.  Subclasses define the signature."""
        ...

    @abstractmethod
    def decompress(self, *args, **kwargs):
        """Alias for :meth:`reconstruct`."""
        return self.reconstruct(*args, **kwargs)

    @abstractmethod
    def reconstruct(self, *args, **kwargs):
        """Reconstruct data from the compressed representation."""
        ...

    # ------------------------------------------------------------------
    # LoRA management
    # ------------------------------------------------------------------

    def enable_lora(
        self,
        rank: int = 4,
        initial_keyframe_timestep: int | None = None,
        **kwargs,
    ) -> None:
        """Attach LoRA adapters to the hidden layers.

        After this call, :meth:`compress` trains **only** the LoRA parameters.

        Parameters
        ----------
        rank : int
            LoRA rank.
        initial_keyframe_timestep : int, optional
            If given *and* a checkpoint store is configured, immediately
            saves a full-state keyframe at this timestep with the new LoRA
            tree structure (base weights + zeroed LoRA).  This anchors the
            delta chain so that :meth:`restore` can reach any subsequent
            timestep.
        """
        from nnx_models import add_lora_to_model

        add_lora_to_model(self.model, lora_rank=rank, **kwargs)
        self._lora_enabled = True
        self._lora_rank = rank

        if initial_keyframe_timestep is not None and self._store is not None:
            self._store.save(
                self.model, initial_keyframe_timestep, filter="all",
            )

    def merge_lora(self) -> None:
        """Fold LoRA updates into the base weights (irreversible)."""
        from nnx_models import merge_lora_params

        merge_lora_params(self.model)

    def reset_lora(self, key: jax.Array | None = None) -> None:
        """Re-initialise LoRA matrices (keeps base weights)."""
        from nnx_models import reset_lora_params

        if key is None:
            key = jax.random.PRNGKey(0)
        reset_lora_params(
            self.model,
            lora_rank=self._lora_rank,
            lora_layers=self.model.num_hidden_layers,
            key=key,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(
        self,
        timestep: int,
        filter: FilterSpec = "all",
        extra: dict | None = None,
    ) -> Path | None:
        """Persist the current parameters.

        Parameters
        ----------
        timestep : int
            Snapshot index / logical time.
        filter : FilterSpec
            Which parameters to save.  Accepts a preset name
            (``"all"``, ``"lora"``), an ``nnx`` variable type
            (e.g. ``nnx.LoRAParam``), or any callable filter spec
            accepted by ``nnx.state``.  Full-parameter saves
            (``filter="all"``) act as anchors that :meth:`restore`
            can load directly; LoRA saves are stacked on top.
        extra : dict, optional
            Arbitrary metadata saved alongside.

        Returns
        -------
        Path or None
            Checkpoint directory, or ``None`` if no store is configured.
        """
        if self._store is None:
            return None

        # Save with the requested filter
        path = self._store.save(
            self.model, timestep, filter=filter, extra=extra,
        )

        # When keyframe_every triggers and we only saved a subset,
        # additionally save a full checkpoint as an anchor for restore.
        if (self._keyframe_every
                and timestep % self._keyframe_every == 0
                and filter != "all"):
            self._save_full_checkpoint(timestep, extra)

        return path

    def _save_full_checkpoint(self, timestep: int, extra: dict | None = None) -> None:
        """Save a full-parameter checkpoint, merging LoRA into base if active.

        Temporarily merges LoRA into the base weights, resets LoRA to zero,
        saves the full state, then restores the original model state.
        """
        if not self._lora_enabled:
            self._store.save(
                self.model, timestep, filter="all", extra=extra,
            )
            return

        from nnx_models import merge_lora_params, reset_lora_params

        # Backup the full state (JAX arrays are immutable; safe to hold refs)
        original_state = nnx.state(self.model)

        # Merge LoRA → base, then reset LoRA to zeros
        merge_lora_params(self.model)
        reset_lora_params(
            self.model,
            lora_rank=self._lora_rank,
            lora_layers=self.model.num_hidden_layers,
            key=jax.random.PRNGKey(0),
        )

        # Save the clean merged state
        self._store.save(
            self.model, timestep, filter="all", extra=extra,
        )

        # Restore exactly the state we had before
        nnx.update(self.model, original_state)

    def load(
        self,
        timestep: int,
        filter: FilterSpec = "all",
    ) -> None:
        """Load a specific checkpoint directly.

        Parameters
        ----------
        timestep : int
            Which checkpoint to load.
        filter : FilterSpec
            Must correspond to the filter used when saving.
        """
        if self._store is None:
            raise RuntimeError("No checkpoint_dir configured.")
        self._store.load(self.model, timestep, filter=filter)

    def restore(self, timestep: int) -> None:
        """Restore to *timestep* using the cheapest restore plan.

        When LoRA is enabled, loads the nearest full checkpoint
        (``filter="all"``) then sequentially applies intermediate LoRA
        deltas.  The final delta's LoRA parameters are loaded but
        **not** merged, leaving the model ready for inference or
        further training.

        When LoRA is not enabled, falls back to a direct full-state load.

        .. note::

            Requires at least one full checkpoint at or before *timestep*
            when LoRA is enabled.  Use ``save(…, filter="all")`` or
            ``enable_lora(initial_keyframe_timestep=…)`` to create one.
        """
        if self._store is None:
            raise RuntimeError("No checkpoint_dir configured.")

        if not self._lora_enabled:
            self._store.load(self.model, timestep, filter="all")
            return

        plan = self._store.get_restore_plan(timestep)

        # Load the latest full checkpoint as the base
        self._store.load(self.model, plan.keyframe, filter="all")

        if not plan.deltas:
            return

        # Apply intermediate LoRA deltas: load → merge into base
        from nnx_models import merge_lora_params

        for delta_t in plan.deltas[:-1]:
            self._store.load(self.model, delta_t, filter="lora")
            merge_lora_params(self.model)

        # Load final delta (active LoRA — do not merge)
        self._store.load(self.model, plan.deltas[-1], filter="lora")

    @property
    def saved_timesteps(self) -> list[int]:
        if self._store is None:
            return []
        return self._store.saved_timesteps

    @property
    def keyframes(self) -> list[int]:
        """Sorted list of keyframe timesteps."""
        if self._store is None:
            return []
        return self._store.keyframes

    # ------------------------------------------------------------------
    # Snapshot state (low-level)
    # ------------------------------------------------------------------

    def get_state(self, filter: FilterSpec = "all") -> nnx.State:
        """Return current model state, optionally filtered."""
        from ..temporal_param import resolve_filter, extract_state as _extract

        resolved = resolve_filter(filter)
        return _extract(self.model, resolved)

    def set_state(self, state: nnx.State) -> None:
        """Overwrite model state in-place."""
        nnx.update(self.model, state)

    def __repr__(self) -> str:
        cls = type(self.model).__name__
        lora = f", lora_rank={self._lora_rank}" if self._lora_enabled else ""
        return f"{type(self).__name__}(model={cls}{lora})"