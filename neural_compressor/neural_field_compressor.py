
from __future__ import annotations

from typing import Any, Optional, Callable

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .base_compressor import BaseCompressor
from .utils import FilterSpec, resolve_filter

class NeuralFieldCompressor(BaseCompressor):
    """Compressor that represents PDE snapshots as neural fields.

    A *neural field* maps spatial coordinates to field values. By having less parameters
    than the original discretized representation of the original snapshot, the neural field can be seen as a compressed representation of the original data. 

    Parameters
    ----------
    model : nnx.Module
        The neural field architecture (MLP, SIREN, WIRE, …).
    norm_stats : dict[str, Any], optional
        If given, should contain normalization statistics and method for the target field, e.g. ``{"method": "z-score", "mean": mean, "std": std}``.
        If provided, the compressor will automatically normalize the target data during training and denormalize the model output during inference.  
        This can help with training stability, especially for fields with large dynamic ranges.
        
    Examples
    --------
    ::

        from nnx_models import MLP
        model = MLP(input_dim=2, output_dim=1, hidden_dim=256,
                    num_hidden_layers=5, fourier_emb_scale=7.0)

        compressor = NeuralFieldCompressor(model)

        # --- Warmup: train & save full model ---
        compressor.compress(optimizer, coords, snapshot_0, epochs=500)

        # --- With normalization ---
        compressor.norm_stats = {"mean": mean, "std": std}
        reconstructed = compressor.decompress(coords)  # auto-denormalized
        reconstructed = compressor(coords)  # same as above

    """

    def __init__(
        self,
        model: nnx.Module,
        norm_stats: dict[str, Any] | None = None,
    ):
        super().__init__(
            model=model,
            norm_stats=norm_stats,
        )

    def compress(
        self,
        optimizer: nnx.Optimizer,
        coords: jnp.ndarray,
        target: jnp.ndarray,
        jac_target: Optional[jnp.ndarray] = None,
        epochs: int = 500,
        batch_size: int | None = None,
        num_devices: int = 1,
        verbose: bool = False,
    ) -> float:
        """Train the neural field on a single discrete spatial snapshot.

        Parameters
        ----------
        coords : jnp.ndarray
            Spatial coordinates array. Should have shape ``(N, D)`` with D the number of spatial dimensions.
        target : jnp.ndarray
            The field values at ``coords``. Should have shape ``(N, F)`` with F the number of features.
        jac_target : jnp.ndarray, optional
            If given, the Jacobian of the field at ``coords``. Should have shape ``(N, D, F)`` with D the number of spatial dimensions and F the number of features.
        epochs : int
            Number of gradient descent iterations.
        batch_size : int, optional
            If given, mini-batch training is used.  When ``num_devices > 1``
            each device will receive a batch of size ``batch_size`` // ``num_devices``.
        num_devices : int
            Number of devices (GPUs/TPUs) to use for multi-device training.
            Defaults to ``1`` (single device).
        verbose : bool
            Print loss every 10 epochs.

        Returns
        -------
        float
            Final loss value.
        """

        if coords.ndim != 2:
            raise ValueError(f"Expected coords to have shape (N, D), got {coords.shape}")
        if target.ndim != 2:
            raise ValueError(f"Expected target to have shape (N, F) with F the number of features, got {target.shape}")
        if jac_target is not None and jac_target.ndim != 3:
            raise ValueError(f"Expected jac_target to have shape (N, D, F) with D the number of spatial dimensions and F the number of features, got {jac_target.shape}")
        
        if target.shape[0] != coords.shape[0]:
            raise ValueError(f"Expected target and coords to have the same number of points, got {target.shape[0]} and {coords.shape[0]}")


        if jac_target is not None:
            loss = [0.0, 0.0]
        else:
            loss = 0.0

        if batch_size is None:
            batch_size = coords.shape[0]

        # ── Multi-device sharding setup ──────────────────────────────
        available_devices = jax.local_devices()
        num_devices = min(num_devices, len(available_devices))
        use_sharding = num_devices > 1

        filter = optimizer.wrt

        if use_sharding:
            devices = available_devices[:num_devices]
            mesh = Mesh(devices, axis_names=('batch',))
            data_sharding = NamedSharding(mesh, P('batch'))
            replicated_sharding = NamedSharding(mesh, P())

            if batch_size % num_devices != 0:
                raise ValueError(f"Batch size {batch_size} must be divisible by number of devices {num_devices}.")

            # Replicate model & optimizer state across devices.
            model_state = nnx.state(self.model)
            model_state = jax.device_put(model_state, replicated_sharding)
            nnx.update(self.model, model_state)

            opt_state = nnx.state(optimizer)
            opt_state = jax.device_put(opt_state, replicated_sharding)
            nnx.update(optimizer, opt_state)

        @nnx.jit
        def train_step_jac_jit(model, optimizer, coords, target, jac_target):
            return train_step_with_jac(model, optimizer, coords, target, jac_target, filter)
        
        @nnx.jit
        def train_step_jit(model, optimizer, coords, target):
            return train_step(model, optimizer, coords, target, filter)

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

                # Shard data batches across devices.
                if use_sharding:
                    coords_batch = jax.device_put(coords_batch, data_sharding)
                    target_batch = jax.device_put(target_batch, data_sharding)
                    if jac_batch is not None:
                        jac_batch = jax.device_put(jac_batch, data_sharding)

                if jac_target is not None:
                    loss_batch = train_step_jac_jit(self.model, optimizer, coords_batch, target_batch, jac_batch)
                    loss[0] += loss_batch[0]
                    loss[1] += loss_batch[1]
                else:
                    loss_batch = train_step_jit(self.model, optimizer, coords_batch, target_batch)
                    loss += loss_batch

            loss = loss / (coords.shape[0] // batch_size)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                if jac_target is not None:
                    print(f"Epoch {epoch + 1}: Total loss = {loss[0] + loss[1]:.2e}, Target Loss = {loss[0]:.2e}, Jacobian Loss = {loss[1]:.2e}")
                else:
                    print(f"Epoch {epoch + 1}: Loss = {loss:.2e}")
        
        # After multi-device training, gather model back to a single device.
        if use_sharding:
            single_device = jax.sharding.SingleDeviceSharding(jax.devices()[0])
            model_state = jax.device_put(nnx.state(self.model), single_device)
            nnx.update(self.model, model_state)

        return loss

    def compute_extra_metrics(self, 
                              coords: jnp.ndarray, 
                              target: jnp.ndarray, 
                              jac_target: Optional[jnp.ndarray] = None,
                              batch_size: int = 100_000) -> dict[str, float]:
        """Test the neural field compression by comparing it to the ground truth
        on different metrics. If ``jac_target`` is given, metrics are computed for both the field values and the Jacobian.

        The list of metrics includes:
        - PSNR
        - Max absolute error
        - Mean absolute error
        - Relative L2 error
        - L2 error

        Parameters
        ----------
        coords : jnp.ndarray
            Spatial coordinates array (``(N, D)``).
        target : jnp.ndarray
            The field values at ``coords``. Should have shape ``(N, F)`` with F the number of features.
            If any normalization was used, ``target`` should be the original unnormalized field values for correct metric computation.
        jac_target : jnp.ndarray, optional
            If given, the Jacobian of the field at ``coords``. Should have shape ``(N, D, F)`` with D the number of spatial dimensions and F the number of features.

        Returns
        -------
        dict[str, float] or tuple[dict[str, float], dict[str, float]]
            A dictionary or tuple of two dictionaries of metric names to their computed values.
        """

        metrics_target = {}
        metrics_jac = {}
        pred = []
        pred_jac = [] if jac_target is not None else None

        if batch_size is None:
            batch_size = coords.shape[0]
        
        for batch in range(0, coords.shape[0], batch_size):
            coords_batch = coords[batch:batch + batch_size]
            pred.append(self.decompress(coords_batch))
            if jac_target is not None:
                pred = jnp.transpose(jax.jacfwd(lambda x: self.decompress(x))(coords_batch), (0, 2, 1))
                pred_jac.append(pred)

        pred = jnp.concatenate(pred, axis=0)
        pred_jac = jnp.concatenate(pred_jac, axis=0) if jac_target is not None else None

        metrics_target["psnr"] = 10 * jnp.log10(jnp.max(jnp.abs(target))**2 / jnp.mean((pred - target) ** 2))
        metrics_target["max_abs"] = jnp.max(jnp.abs(pred - target))
        metrics_target["mean_abs"] = jnp.mean(jnp.abs(pred - target))
        metrics_target["rel_l2"] = jnp.linalg.norm(pred - target) / jnp.linalg.norm(target)
        metrics_target["l2"] = jnp.linalg.norm(pred - target)

        if jac_target is not None:
            metrics_jac["max_abs"] = jnp.max(jnp.abs(pred_jac - jac_target))
            metrics_jac["mean_abs"] = jnp.mean(jnp.abs(pred_jac - jac_target))
            metrics_jac["rel_l2"] = jnp.linalg.norm(pred_jac - jac_target) / jnp.linalg.norm(jac_target)
            metrics_jac["l2"] = jnp.linalg.norm(pred_jac - jac_target)

            return metrics_target, metrics_jac
        
        return metrics_target

    def decompress(self, coords: jnp.ndarray) -> jnp.ndarray:
        """Run a forward pass through the current model.

        If :attr:`norm_stats` is not None, the output is denormalized before being returned.

        Parameters
        ----------
        coords : jnp.ndarray
            Query coordinates. Should have shape ``(N, D)`` with D the number of spatial dimensions.
        """
        out = self.model(coords)
        if self.norm_stats.get("method", None) is None:
            return out
        elif self.norm_stats["method"] == "z-score":
            out = out * self.norm_stats["std"] + self.norm_stats["mean"]
        elif self.norm_stats["method"] == "min-max":
            out = out * (self.norm_stats["max"] - self.norm_stats["min"]) + self.norm_stats["min"]
        else:
            raise ValueError(f"Unknown norm_stats method {self.norm_stats['method']!r}")
        return out

    def validate_downcast(self, 
                          coords: jnp.ndarray, 
                          target: jnp.ndarray, 
                          jac_target: jnp.ndarray | None = None, 
                          batch_size: int = 100_000,
                          tolerance: float = 2.0,
                          filter: Optional[str | nnx.Variable] = nnx.Param) -> bool:
        """Check whether downcasting model weights to float16 keeps the loss within a tolerance factor.

        Compares the loss of the original model against a float16-downcasted copy.
        Returns the downcasted state if the loss ratio is acceptable, otherwise ``False``.

        Parameters
        ----------
        coords : jnp.ndarray
            Spatial coordinates for evaluation. Should have shape ``(N, D)`` with D the number of spatial dimensions.
        target : jnp.ndarray
            Ground truth field values. Should have shape ``(N, F)`` with F the number of features.
        jac_target : jnp.ndarray, optional
            Ground truth Jacobian values (included in the loss if given). Should have shape ``(N, D, F)`` with D the number of spatial dimensions and F the number of features.
        batch_size : int
            Number of points per evaluation batch.
        tolerance : float
            Maximum acceptable ratio of downcasted loss to original loss.
        filter : FilterSpec
            Which parameters to downcast.

        Returns
        -------
        nnx.State or False
            The downcasted parameter state if validation passes, ``False`` otherwise.
        """

        resolved_filter = resolve_filter(filter) if isinstance(filter, str) else filter

        if batch_size is None:
            batch_size = coords.shape[0]

        if filter == nnx.Param:
            downcasted_state = jax.tree_util.tree_map(lambda p: p.astype(jnp.float16), nnx.state(self.model, nnx.PathContains('hidden_layers')))
        else:
            downcasted_state = jax.tree_util.tree_map(lambda p: p.astype(jnp.float16), nnx.state(self.model, resolved_filter))

        graphdef, _ = nnx.split(self.model)
        model_downcast = nnx.merge(graphdef, downcasted_state)

        base_loss = 0.0
        downcast_loss = 0.0

        for batch in range(0, coords.shape[0], batch_size):
            coords_batch = coords[batch:batch + batch_size]
            target_batch = target[batch:batch + batch_size]
            jac_batch = jac_target[batch:batch + batch_size] if jac_target is not None else None

            out = target_loss(self.model, coords_batch, target_batch)
            out_downcast = target_loss(model_downcast, coords_batch, target_batch)
            base_loss += out
            downcast_loss += out_downcast

            if jac_batch is not None:
                out_jac = _jac_loss(self.model, coords_batch, jac_batch)
                out_downcast_jac = _jac_loss(model_downcast, coords_batch, jac_batch)
                base_loss += out_jac
                downcast_loss += out_downcast_jac
                base_loss += out + out_jac

        downcast_loss /= (coords.shape[0] // batch_size)
        base_loss /= (coords.shape[0] // batch_size)

        if downcast_loss / base_loss <= tolerance:
            return downcasted_state

        return False

    def __call__(self, coords: jnp.ndarray) -> jnp.ndarray:
        """Alias for :meth:`decompress`."""
        return self.decompress(coords)

        
    
def _mse_loss(x, y):
    """Compute the mean squared error between two arrays."""
    return jnp.mean((x - y) ** 2)

@nnx.jit
def target_loss(model, x, y):
    """Compute the MSE between the model's prediction at ``x`` and the target ``y``."""
    return _mse_loss(model(x), y)

@nnx.jit
def _jac_loss(model, x, dy):
    """Compute the MSE between the model's Jacobian at ``x`` and the target Jacobian ``dy``."""
    model_jac = jnp.transpose(jax.jacfwd(lambda x: model(x))(x), (0, 2, 1))
    return _mse_loss(model_jac, dy)

def train_step(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        coords: jnp.ndarray,
        target: jnp.ndarray,
        filter: Optional[nnx.Variable] = nnx.Param,
) -> float:
    """Perform a single gradient-descent step minimising the target MSE loss.

    Parameters
    ----------
    model : nnx.Module
        The neural field model to train.
    optimizer : nnx.Optimizer
        Optimizer that updates model parameters.
    coords : jnp.ndarray
        Input coordinates batch.
    target : jnp.ndarray
        Target field values batch.
    filter : nnx.Variable, optional
        Which parameter subset to differentiate through.

    Returns
    -------
    float
        The loss value for this step.
    """
    loss, grads = nnx.value_and_grad(target_loss, argnums=nnx.DiffState(0, filter=filter))(model, coords, target)
    optimizer.update(model, grads)

    return loss

def train_step_with_jac(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        coords: jnp.ndarray,
        target: jnp.ndarray,
        jac_target: jnp.ndarray,
        filter: Optional[nnx.Variable] = nnx.Param,
) -> tuple[float, float]:
    """Perform a single gradient-descent step using both target and Jacobian losses.

    The gradients from the target loss and the Jacobian loss are normalised
    independently and summed, so both objectives contribute equally
    regardless of their magnitudes.

    Parameters
    ----------
    model : nnx.Module
        The neural field model to train.
    optimizer : nnx.Optimizer
        Optimizer that updates model parameters.
    coords : jnp.ndarray
        Input coordinates batch.
    target : jnp.ndarray
        Target field values batch.
    jac_target : jnp.ndarray
        Target Jacobian values batch.
    filter : nnx.Variable, optional
        Which parameter subset to differentiate through.

    Returns
    -------
    tuple[float, float]
        ``(target_loss, jacobian_loss)`` for this step.
    """
    
    loss_y, grads_y = nnx.value_and_grad(target_loss, argnums=nnx.DiffState(0, filter=filter))(model, coords, target)
    loss_dy, grads_dy = nnx.value_and_grad(_jac_loss, argnums=nnx.DiffState(0, filter=filter))(model, coords, jac_target)

    grads_y_flatt, unflatten = jax.flatten_util.ravel_pytree(grads_y)
    grads_dy_flatt = jax.flatten_util.ravel_pytree(grads_dy)[0]

    grad_norm_y = jnp.linalg.norm(grads_y_flatt)
    grad_norm_dy = jnp.linalg.norm(grads_dy_flatt)

    grad = grads_y_flatt / grad_norm_y + grads_dy_flatt / grad_norm_dy

    grad = unflatten(grad)

    optimizer.update(model, grad)

    return loss_y, loss_dy