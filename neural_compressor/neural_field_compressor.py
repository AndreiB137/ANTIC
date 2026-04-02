
from __future__ import annotations

from .base_compressor import BaseCompressor, FilterSpec
from flax import nnx
from typing import Optional, Callable
import jax
import jax.numpy as jnp
from pathlib import Path

class NeuralFieldCompressor(BaseCompressor):
    """Compressor that represents PDE snapshots as neural fields.

    A *neural field* maps spatial coordinates to field values.  This class
    trains such a network on one snapshot at a time and manages checkpoints,
    LoRA adapters, and keyframe-based random-access restore.

    Parameters
    ----------
    model : nnx.Module
        The neural-field architecture (MLP, SIREN, WIRE, …).
    checkpoint_dir : str or Path, optional
        Where to persist parameters.  If ``None``, checkpointing is disabled.
    optimizer : nnx.Optimizer, optional
        Optimizer instance.  Defaults to AdamW + cosine schedule.
    loss_fn : callable, optional
        ``(model, x, y) -> scalar``.  Defaults to MSE.
    keyframe_every : int, optional
        Auto-save full-state keyframes every *N* timesteps for fast
        random-access :meth:`restore`.

    Examples
    --------
    ::

        from nnx_models import MLP
        model = MLP(input_dim=2, output_dim=1, hidden_dim=256,
                    num_hidden_layers=5, fourier_emb_scale=7.0)

        compressor = NeuralFieldCompressor(
            model, checkpoint_dir="ckpts/", keyframe_every=5,
        )

        # --- Warmup: train & save full model ---
        compressor.compress(coords, snapshot_0, epochs=500)
        compressor.save(timestep=0)  # filter="all" saves full params

        # --- Switch to LoRA (anchors delta chain at t=0) ---
        compressor.enable_lora(rank=4, initial_keyframe_timestep=0)

        for t in range(1, 20):
            compressor.compress(coords, snapshots[t], epochs=300)
            compressor.save(t, filter="lora")   # auto-keyframe at 5, 10, 15
            compressor.merge_lora()
            compressor.reset_lora()

        # --- Fast random-access restore ---
        compressor.restore(timestep=7)
        # Loads keyframe at t=5, applies LoRA deltas for t=6 & t=7
        reconstructed = compressor.reconstruct(coords)
    """

    def __init__(
        self,
        model: nnx.Module,
        checkpoint_dir: str | Path | None = None,
        optimizer : nnx.Optimizer | None = None,
        loss_fn: Optional[Callable] = None,
        keyframe_every: int | None = None,
    ):
        super().__init__(
            model=model,
            checkpoint_dir=checkpoint_dir,
            optimizer=optimizer,
            keyframe_every=keyframe_every,
        )

    def compress(
        self,
        coords: jnp.ndarray,
        target: jnp.ndarray,
        jac_target: Optional[jnp.ndarray] = None,
        epochs: int = 500,
        batch_size: int | None = None,
        filter: Optional[FilterSpec] = "all",
        verbose: bool = False,
    ) -> float:
        """Train the neural field on a single snapshot.

        Parameters
        ----------
        coords : jnp.ndarray
            Spatial coordinates array (``(N, D)``).
        target : jnp.ndarray
            The field values at ``coords``.
        jac_target : jnp.ndarray, optional
            If given, the Jacobian of the field at ``coords``.
        epochs : int
            Number of gradient-descent iterations.
        batch_size : int, optional
            If given, mini-batch training is used.
        verbose : bool
            Print loss every 100 epochs.

        Returns
        -------
        float
            Final loss value.
        """

        if jac_target is not None:
            loss = [0.0, 0.0]
        else:
            loss = 0.0

        if batch_size is None:
            batch_size = coords.shape[0]
        key = jax.random.PRNGKey(0)

        for epoch in range(epochs):
            key, _ = jax.random.split(key)
            perm = jax.random.permutation(key, coords.shape[0])
            for batch in range(0, coords.shape[0], batch_size):
                if batch + batch_size > coords.shape[0]:
                    continue
                coords_batch = coords[perm[batch:batch + batch_size]]
                target_batch = target[perm[batch:batch + batch_size]]
                jac_batch = jac_target[perm[batch:batch + batch_size]] if jac_target is not None else None
                loss_batch = train_step(self.model, self.optimizer, coords_batch, target_batch, jac_batch)
                if jac_target is not None:
                    loss[0] += loss_batch[0]
                    loss[1] += loss_batch[1]
                else:
                    loss += loss_batch

            loss = loss / (coords.shape[0] // batch_size)

            if verbose and epoch % 100 == 0:
                if jac_target is not None:
                    print(f"Epoch {epoch}: Total loss = {loss[0] + loss[1]:.2e}, Target Loss = {loss[0]:.2e}, Jacobian Loss = {loss[1]:.2e}")
                else:
                    print(f"Epoch {epoch}: Loss = {loss:.2e}")
        

        return loss

    def reconstruct(self, coords: jnp.ndarray) -> jnp.ndarray:
        """Run a forward pass through the current model.

        Parameters
        ----------
        coords : jnp.ndarray
            Query coordinates.
        """
        return self.model(coords)