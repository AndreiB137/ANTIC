"""Checkpoint store — flexible save / load with filter and keyframe support.

Supports ``nnx`` filter specs for saving arbitrary parameter subsets, and a
keyframe / delta strategy for efficient random-access restore of LoRA
checkpoint chains.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

from flax import nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp


# ======================================================================
# Filter resolution
# ======================================================================

FilterSpec = Union[str, type, Callable, None]
"""User-facing filter: a preset name, an ``nnx`` variable type, or callable."""

FILTER_PRESETS: dict[str, Any] = {
    "all": nnx.Param,
    "lora": nnx.LoRAParam,
    "hidden_layers": nnx.PathContains('hidden_layers'),
}


def resolve_filter(spec: FilterSpec) -> Any:
    """Turn a user-facing *FilterSpec* into an nnx-compatible filter.

    Returns ``None`` when the full (unfiltered) state should be used.
    """
    if spec is None or spec == "all":
        return nnx.Param
    if isinstance(spec, str):
        if spec not in FILTER_PRESETS:
            raise ValueError(
                f"Unknown filter preset {spec!r}. "
                f"Available: {sorted(FILTER_PRESETS)}"
            )
        return FILTER_PRESETS[spec]
    return spec


def extract_state(model: nnx.Module, resolved_filter: nnx.Variable | None) -> nnx.State:
    """Extract state from *model*, optionally narrowed by *resolved_filter*."""
    if resolved_filter is None:
        return nnx.state(model)
    return nnx.state(model, resolved_filter)


# ======================================================================
# Restore plan
# ======================================================================

@dataclass
class RestorePlan:
    """Describes how to efficiently restore a specific timestep.

    Attributes
    ----------
    keyframe : int
        The nearest keyframe timestep to load first.
    deltas : list[int]
        Ordered LoRA-delta timesteps to apply *after* the keyframe.
    """
    keyframe: int
    deltas: list[int] = field(default_factory=list)


# ======================================================================
# Checkpoint store
# ======================================================================


class ParamStore:
    """Manages parameter checkpoints on disk.

    Supports flexible ``nnx``-style filter specs and a *keyframe / delta*
    strategy for efficient random-access restore in LoRA workflows.

    Parameters
    ----------
    root_dir : str or Path
        Directory where checkpoints are persisted.
    """

    def __init__(self, root_dir: str | Path, config: dict[str, Any] | None = None):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.root_dir / "meta.json"
        self._meta: dict[str, Any] = self._load_meta()
        if not os.path.join(self.root_dir, "config.json").exists():
            raise FileNotFoundError(f"Checkpoint directory {self.root_dir} is missing config.json")

        

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(
        self,
        model: nnx.Module,
        timestep: int,
        filter: FilterSpec = "all",
        overwrite: bool = False,
    ):
        """Save model parameters for a given *timestep*.

        Parameters
        ----------
        model : nnx.Module
            The model to checkpoint.
        timestep : int
            Logical timestep / snapshot index.
        filter : FilterSpec
            Which parameters to save.  Accepts a preset name
            (``"all"``, ``"lora"``), an ``nnx`` variable type
            (e.g. ``nnx.LoRAParam``), or any callable filter spec
            accepted by ``nnx.state``.  Entries saved with
            ``filter="all"`` act as full-parameter anchors for the
            restore-plan logic.

        Returns
        -------
        None
        """
        resolved = resolve_filter(filter)
        state = extract_state(model, resolved)

        filter_name = filter if isinstance(filter, str) else type(filter).__name__
        ckpt_dir = self.root_dir / f"step_{timestep:06d}"
        if ckpt_dir.exists() and not overwrite:
            raise FileExistsError(f"Checkpoint directory {ckpt_dir} already exists.")
        self.checkpointer.save(ckpt_dir / "state", state)

        save_metadata = {
            "timestep": timestep,
            "filter": filter_name,
        }

        json.dump(save_metadata, ckpt_dir / "metadata.json", indent=2)
        

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, 
             model: nnx.Module,
             timestep: int,
    ):
        """Restore parameters from a checkpoint **in-place**.

        Parameters
        ----------
        model : nnx.Module
            Target model whose state is overwritten.
        timestep : int
            Which checkpoint to load.
        filter : FilterSpec
            Must correspond to the filter used when saving, so the
            abstract state tree matches for deserialisation.
        """

        filter_name = filter if isinstance(filter, str) else type(filter).__name__
        ckpt_dir = self.root_dir / f"step_{timestep:06d}"
        restored = self.checkpointer.restore(ckpt_dir, abstract)
        nnx.update(model, restored)

    def load_state(
        self,
        model: nnx.Module,
        timestep: int,
        filter: FilterSpec = "all",
        label: str = "params",
    ) -> nnx.State:
        """Load and return a state *without* applying it to the model."""
        resolved = resolve_filter(filter)
        abstract = extract_state(model, resolved)

        ckpt_dir = self.root_dir / f"step_{timestep:06d}"
        return self.checkpointer.restore(ckpt_dir, abstract)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def saved_timesteps(self) -> list[int]:
        """Sorted list of all checkpointed timesteps."""
        entries = self._meta.get("checkpoints", [])
        return sorted({e["timestep"] for e in entries})

    @property
    def full_checkpoints(self) -> list[int]:
        """Sorted list of timesteps where all parameters were saved."""
        entries = self._meta.get("checkpoints", [])
        return sorted({e["timestep"] for e in entries if e.get("filter") == "all"})

    @property
    def keyframes(self) -> list[int]:
        """Alias for :attr:`full_checkpoints`."""
        return self.full_checkpoints

    @property
    def compressor_class(self) -> str | None:
        """The compressor class name that created these checkpoints."""
        return self._meta.get("config", {}).get("compressor_class")

    def has_timestep(self, timestep: int) -> bool:
        return timestep in self.saved_timesteps

    def has_full_checkpoint(self, timestep: int) -> bool:
        """Check whether *timestep* has a full-parameter checkpoint."""
        return timestep in self.full_checkpoints

    def has_keyframe(self, timestep: int) -> bool:
        """Alias for :meth:`has_full_checkpoint`."""
        return self.has_full_checkpoint(timestep)

    def entries_for(self, timestep: int) -> list[dict]:
        """Return all metadata entries for a given *timestep*."""
        return [
            e for e in self._meta.get("checkpoints", [])
            if e["timestep"] == timestep
        ]

    def get_restore_plan(self, timestep: int) -> RestorePlan:
        """Compute the cheapest plan to restore *timestep*.

        Finds the latest full checkpoint (``filter="all"``) at or before
        *timestep*, then collects the intermediate LoRA-delta timesteps
        that must be applied sequentially after it.
        """
        fulls = self.full_checkpoints
        if not fulls:
            return RestorePlan(keyframe=timestep)

        candidates = [k for k in fulls if k <= timestep]
        if not candidates:
            return RestorePlan(keyframe=timestep)

        anchor = max(candidates)
        if anchor == timestep:
            return RestorePlan(keyframe=anchor)

        all_ts = self.saved_timesteps
        deltas = sorted(t for t in all_ts if anchor < t <= timestep)
        return RestorePlan(keyframe=anchor, deltas=deltas)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_meta(self) -> dict:
        if self._meta_path.exists():
            with open(self._meta_path) as f:
                return json.load(f)
        return {
            "snapshots": [],
            "lora_snapshots": [],
        }

    def _flush_meta(self) -> None:
        with open(self._meta_path, "w") as f:
            json.dump(self._meta, f, indent=2, default=str)

    def __repr__(self) -> str:
        n = len(self.saved_timesteps)
        nf = len(self.full_checkpoints)
        cls = self.compressor_class
        parts = [f"root={self.root_dir!r}", f"checkpoints={n}", f"full={nf}"]
        if cls:
            parts.append(f"compressor={cls!r}")
        return f"CheckpointStore({', '.join(parts)})"
