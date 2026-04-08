"""Checkpoint store — flexible save / load with filter and keyframe support.

Supports ``nnx`` filter specs for saving arbitrary parameter subsets
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from nnx_models import model_fn
from nnx_models.lora.utils_lora import add_lora_to_model, apply_lora_state
from neural_compressor import NeuralFieldCompressor
from neural_compressor.utils import FilterSpec, resolve_filter, extract_state
import yaml
import pickle

class ParamManager:
    """

    Manages saving and loading of neural field model parameters 
    with support for checkpoints where only a subset of parameters
    (e.g. LoRA matrices, only hidden layers or other specific components) are saved.

    The clas is best used during the full ANTIC compression loop, it does not 
    implement a general saving and loading interface for an independent run,
    but rather is designed to work in the full compression workflow.

    If a checkpoint involves only a subset of paramaters, the manager automatically loads the
    most recent checkpoint involving the full parameter set and updates it with the current subset checkpoint.
    This is done by keeping track of last_full_save variable, which is updated whenever a full checkpoint is saved.

    No matter if a full or partial checkpoint is loaded, the returned model parameter state will always have the same pytree structure
    as the original model. Moreover, since saved parameters correspond to the actual compression at a particular snapshot, these
    are restored in O(1) time, without needing to additional reads to reconstruct the compressed state.

    Parameters
    ----------
    root_dir : str or Path
        Directory where compressed parameters are stored.
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.compressed_dir = self.root_dir / "compressed"
        self.compressed_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.root_dir / "metadata.json"
        self._meta: dict[str, Any] = self._load_meta()
        if not (self.root_dir / "train_config.yaml").exists():
            raise FileNotFoundError(f"Checkpoint directory {self.root_dir} is missing train_config.yaml")
        self.last_full_save = 0

        with open(self.root_dir / "train_config.yaml") as f:
            self.model_cfg = yaml.safe_load(f).get("model", {})
        with open(self.root_dir / "train_config.yaml") as f:
            self.train_cfg = yaml.safe_load(f).get("training", {})
        self.model_name = self.model_cfg.get("name", None)
        self.model_cfg.pop("name", None)
        self.checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())

        self.sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        self.sharding_pytree = lambda x : x.update(sharding=self.sharding) if isinstance(x, jax.ShapeDtypeStruct) else x

        # Constructor kwargs: everything except bookkeeping keys.
        _ctor_keys = {"name", "rank"}
        ctor_kwargs = {k: v for k, v in self.model_cfg.items() if k not in _ctor_keys}

        if self.train_cfg.get("filter") == 'lora':
            def make_lora_model(rank=self.train_cfg.get("rank"), rngs=None, **kwargs):
                base_model = model_fn[self.model_name](**kwargs, rngs=rngs)
                add_lora_to_model(base_model, rank=rank, rngs=rngs)
                return base_model
        
            lora_model_abstract = nnx.eval_shape(lambda : make_lora_model(
                rank=self.train_cfg.get("rank"), rngs=nnx.Rngs(0), **ctor_kwargs))
            
            self.lora_model_state = jax.tree_util.tree_map(
                self.sharding_pytree,
                nnx.state(lora_model_abstract, nnx.LoRAParam)
            )


        model_abstract_state = nnx.eval_shape(lambda : model_fn[self.model_name](**ctor_kwargs, rngs=nnx.Rngs(0)))
        self.model_abstract_state = jax.tree_util.tree_map(
            self.sharding_pytree,
            nnx.state(model_abstract_state)
        )

    def save(
        self,
        model_compressor: NeuralFieldCompressor,
        idx: int,
        elapsed_time: int,
        filter: FilterSpec = "all",
        overwrite: bool = False,
    ):
        """Save model parameters for a given *idx* and *elapsed_time*.

        Parameters
        ----------
        model_compressor : NeuralFieldCompressor
            The model to checkpoint.
        idx : int
            Logical snapshot index. This corresponds to an identifier for the current compression state.
            During ANTIC, this index is typically updated by the temporal selector. 
        elapsed_time : float
            The simulation elapsed time corresponding to this checkpoint, for metadata purposes.
        filter : FilterSpec
            Which parameters to save.  Accepts a preset name
            (``"all"``, ``"lora"``), an ``nnx`` variable type
            (e.g. ``nnx.LoRAParam``), or any callable filter spec
            accepted by ``nnx.state``.  Entries saved with
            ``filter="all"`` act as full parameter anchors for the
            restore-plan logic.

        Returns
        -------
        None
        """ 

        if filter == "all":
            self.last_full_save = idx
        
        resolved = resolve_filter(filter)
        state = extract_state(model_compressor.model, resolved)

        filter_name = filter if isinstance(filter, str) else type(filter).__name__
        ckpt_dir = self.compressed_dir / f"snapshot_{idx}"
        if ckpt_dir.exists() and not overwrite:
            raise FileExistsError(f"Checkpoint directory {ckpt_dir} already exists.")

        if overwrite and ckpt_dir.exists():
            print(f"Overwriting existing checkpoint at {ckpt_dir}...")
            self.checkpointer.save(os.path.abspath(ckpt_dir / "state"), state, force=True)
        else:
            self.checkpointer.save(os.path.abspath(ckpt_dir / "state"), state)
        

        save_metadata = {
            "idx": idx,
            "elapsed_time": elapsed_time,
            "filter": filter_name,
            "last_full_save": self.last_full_save,
            "norm_stats": model_compressor.norm_stats,
        }
        if filter_name == "lora":
            save_metadata["rank"] = self.model_cfg.get("rank")

        with open(ckpt_dir / "metadata.pkl", "wb") as f:
            pickle.dump(save_metadata, f)
        

    def load(
        self,
        id: int
    ) -> NeuralFieldCompressor:
        """Restore parameters from a compressed snapshot.

        If the compressed snapshot corresponds to a full paramater save, then these are directly loaded. 
        If instead the snapshot corresponds to a partial save (e.g. only LoRA parameters or hidden layers), 
        then the most recent full parameter save is first loaded, then updated with the current restored state.

        If the metadata file inside the compressed snapshot directory contained normalization stats,
        these are also loaded and returned as part of the `NeuralFieldCompressor` instance.

        Parameters
        ----------
        id : int
            Which compressed snapshot to load (i.e. which idx was used during saving). Must correspond to an existing compressed snapshot.

        Returns
        -------
        model : NeuralFieldCompressor
            A `NeuralFieldCompressor` instance from training config with parameters loaded from disk.
        """

        model = model_fn[self.model_name](**self.model_cfg, rngs=nnx.Rngs(0))
        with open(os.path.abspath(self.compressed_dir / f"snapshot_{id}" / "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
        if metadata["filter"] == "all":
            full_state = self.checkpointer.restore(
                os.path.abspath(self.compressed_dir / f"snapshot_{id}" / "state"),
                self.model_abstract_state
            )
            nnx.update(model, full_state)
        elif metadata["filter"] == "lora":
            last_full_save = metadata.get("last_full_save")
            full_state = self.checkpointer.restore(
                os.path.abspath(self.compressed_dir / f"snapshot_{last_full_save}" / "state"),
                self.model_abstract_state
            )
            nnx.update(model, full_state)

            lora_state = self.checkpointer.restore(
                os.path.abspath(self.compressed_dir / f"snapshot_{id}" / "state"),
                self.lora_model_state,
            )

            apply_lora_state(model, lora_state)
        elif metadata["filter"] == "hidden_layers":
            last_full_save = metadata.get("last_full_save")
            full_state = self.checkpointer.restore(
                os.path.abspath(self.compressed_dir / f"snapshot_{last_full_save}" / "state"),
                self.model_abstract_state
            )
            nnx.update(model, full_state)
            curr_state = self.checkpointer.restore(
                os.path.abspath(self.compressed_dir / f"snapshot_{id}" / "state"),
                jax.tree_util.tree_map(
                    self.sharding_pytree,
                    extract_state(nnx.eval_shape(lambda: model_fn[self.model_name](**self.model_cfg, rngs=nnx.Rngs(0))), nnx.PathContains('hidden_layers'))
                )
            )
            nnx.update(model, curr_state)
        else:
            raise ValueError(f"Unsupported filter {metadata['filter']} in checkpoint metadata.")

        model_comp = NeuralFieldCompressor(model, metadata.get("norm_stats", None))

        return model_comp

    def _load_meta(self) -> dict:
        """Load the compressed snapshot metadata .pkl from disk, or return an empty template."""
        if self._meta_path.exists():
            with open(self._meta_path, "rb") as f:
                return pickle.load(f)
        return {
            "snapshots": [],
        }
