from flax import nnx
import jax.numpy as jnp
import typing
import jax
from flax.typing import Dtype, Initializer
from flax.nnx.nn import initializers
from nnx_models import SirenLayer
from nnx_models import RealGaborLayer

default_a_initializer = initializers.he_uniform()
default_b_initializer = initializers.zeros

class LoRALinear(nnx.Linear):
    """LoRA-augmented Linear that preserves the original pytree structure.

    Subclasses nnx.Linear so kernel/bias stay at their original paths.
    lora_a and lora_b are added alongside them (like PyTorch's approach).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int,
        *,
        use_bias: bool = True,
        dtype: typing.Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        kernel_init: Initializer = initializers.lecun_normal(),
        bias_init: Initializer = initializers.zeros_init(),
        a_initializer: Initializer = default_a_initializer,
        b_initializer: Initializer = default_b_initializer,
        lora_param_type: typing.Type[nnx.variablelib.Variable] = nnx.LoRAParam,
        rngs: nnx.rnglib.Rngs = nnx.rnglib.Rngs(0),
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
        )
        self.lora_rank = lora_rank
        self.a_initializer = a_initializer
        self.b_initializer = b_initializer

        self.lora_a = lora_param_type(
            a_initializer(rngs.params(), (in_features, lora_rank), param_dtype)
        )
        self.lora_b = lora_param_type(
            b_initializer(rngs.params(), (lora_rank, out_features), param_dtype)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        base_out = super().__call__(x)
        lora_a = self.lora_a.value
        lora_b = self.lora_b.value
        if self.dtype:
            x = x.astype(self.dtype)
            lora_a = lora_a.astype(self.dtype)
            lora_b = lora_b.astype(self.dtype)
        lora_out = x @ lora_a @ lora_b
        return base_out + lora_out

    @classmethod
    def from_linear(
        cls,
        linear: nnx.Linear,
        lora_rank: int,
        *,
        a_initializer: Initializer = default_a_initializer,
        b_initializer: Initializer = default_b_initializer,
        lora_param_type: typing.Type[nnx.variablelib.Variable] = nnx.LoRAParam,
        rngs: nnx.rnglib.Rngs = nnx.rnglib.Rngs(0),
    ) -> "LoRALinear":
        """Create a LoRALinear from an existing nnx.Linear, preserving its weights."""
        lora_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            lora_rank=lora_rank,
            use_bias=linear.use_bias,
            param_dtype=linear.kernel.value.dtype,
            dtype=getattr(linear, "dtype", None),
            a_initializer=a_initializer,
            b_initializer=b_initializer,
            lora_param_type=lora_param_type,
            rngs=rngs,
        )
        # Transfer original weights (shares the same Variable objects)
        lora_linear.kernel = linear.kernel
        if linear.use_bias:
            lora_linear.bias = linear.bias
        return lora_linear


# Keep backward-compatible alias
LoRA = LoRALinear


def _linear_from_lora(
    lora_linear: LoRALinear,
    merge: bool,
) -> nnx.Linear:
    """Convert a LoRALinear back to a plain nnx.Linear."""
    linear = nnx.Linear(
        in_features=lora_linear.in_features,
        out_features=lora_linear.out_features,
        use_bias=lora_linear.use_bias,
        dtype=getattr(lora_linear, "dtype", None),
        param_dtype=lora_linear.kernel.value.dtype,
        rngs=nnx.rnglib.Rngs(0),
    )
    linear.kernel = lora_linear.kernel
    if merge:
        linear.kernel.value += lora_linear.lora_a.value @ lora_linear.lora_b.value
    if lora_linear.use_bias:
        linear.bias = lora_linear.bias
    return linear


def _get_lora_linears(model: nnx.Module) -> list[LoRALinear]:
    """Collects all LoRALinear instances from the model's hidden layers."""
    result = []
    for layer in model.hidden_layers:
        if isinstance(layer, LoRALinear):
            result.append(layer)
        elif isinstance(layer, SirenLayer) and isinstance(layer.linear, LoRALinear):
            result.append(layer.linear)
        elif isinstance(layer, RealGaborLayer):
            if isinstance(layer.freqs, LoRALinear):
                result.append(layer.freqs)
            if isinstance(layer.scales, LoRALinear):
                result.append(layer.scales)
    return result


def add_lora_to_model(
    model: nnx.Module,
    rank: int,
    rngs: nnx.rnglib.Rngs = nnx.rnglib.Rngs(0)
):
    """Adds LoRA layers to a given model, preserving the original pytree structure."""

    for i, layer in enumerate(model.hidden_layers):
        if isinstance(layer, LoRALinear):
            continue
        elif isinstance(layer, SirenLayer):
            if not isinstance(layer.linear, LoRALinear):
                layer.linear = LoRALinear.from_linear(
                    layer.linear, rank, rngs=rngs
                )
        elif isinstance(layer, RealGaborLayer):
            if not isinstance(layer.freqs, LoRALinear):
                layer.freqs = LoRALinear.from_linear(
                    layer.freqs, rank, rngs=rngs
                )
            if not isinstance(layer.scales, LoRALinear):
                layer.scales = LoRALinear.from_linear(
                    layer.scales, rank, rngs=rngs
                )
        elif isinstance(layer, nnx.Linear):
            model.hidden_layers[i] = LoRALinear.from_linear(
                layer, rank, rngs=rngs
            )
        else:
            raise ValueError(
                f"Unsupported layer type: {type(layer)}. "
                "LoRA can only be applied to Linear, SirenLayer, and RealGaborLayer."
            )


def remove_lora_from_model(
    model: nnx.Module,
    merge: bool = True,
):
    """Remove LoRA layers from a model, optionally merging them into the base weights."""

    lora_removed = False

    for i, layer in enumerate(model.hidden_layers):
        if isinstance(layer, LoRALinear):
            model.hidden_layers[i] = _linear_from_lora(layer, merge=merge)
            lora_removed = True
        elif isinstance(layer, SirenLayer):
            if isinstance(layer.linear, LoRALinear):
                layer.linear = _linear_from_lora(layer.linear, merge=merge)
                lora_removed = True
        elif isinstance(layer, RealGaborLayer):
            if isinstance(layer.freqs, LoRALinear):
                layer.freqs = _linear_from_lora(layer.freqs, merge=merge)
                lora_removed = True
            if isinstance(layer.scales, LoRALinear):
                layer.scales = _linear_from_lora(layer.scales, merge=merge)
                lora_removed = True
        elif not isinstance(layer, nnx.Linear):
            raise ValueError(
                f"Unsupported layer type: {type(layer)}. "
                "LoRA can only be removed from Linear, SirenLayer, and RealGaborLayer."
            )

    if not lora_removed:
        raise ValueError("No LoRA parameters found in the model.")


def reset_lora_leaf(path, val, key):
    """Check if the path corresponds to LoRA parameters."""
    if path[-2].key == "lora_a":
        return jax.nn.initializers.lecun_normal()(key, val.shape, val.dtype)
    elif path[-2].key == "lora_b":
        return jax.nn.initializers.zeros(key, val.shape, val.dtype)


@nnx.jit(static_argnames=('rank',))
def reset_lora_params(
    model: nnx.Module,
    rank: int,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
):
    """Resets the LoRA parameters of the model."""
    lora_linears = _get_lora_linears(model)
    if not lora_linears:
        raise ValueError("No LoRA parameters found in the model.")
    keys = jax.random.split(key, len(lora_linears))
    for i, lora_linear in enumerate(lora_linears):
        lora_linear.lora_a.value = lora_linear.a_initializer(
            keys[i], (lora_linear.in_features, rank), lora_linear.lora_a.value.dtype
        )
        lora_linear.lora_b.value = lora_linear.b_initializer(
            keys[i], (rank, lora_linear.out_features), lora_linear.lora_b.value.dtype
        )


@nnx.jit
def merge_lora_params(model: nnx.Module):
    """Merges LoRA parameters into the base kernel weights."""
    lora_linears = _get_lora_linears(model)
    if not lora_linears:
        raise ValueError("No LoRA parameters found in the model.")

    for lora_linear in lora_linears:
        lora_linear.kernel.value += lora_linear.lora_a.value @ lora_linear.lora_b.value


def apply_lora_state(model: nnx.Module, lora_state: nnx.State):
    """Fold external LoRA deltas into a plain model's kernel weights.

    Given a model **without** LoRA layers and a separately stored
    ``nnx.State`` that mirrors the model's ``hidden_layers`` structure
    but contains only ``lora_a`` / ``lora_b`` arrays, this function
    computes ``kernel += lora_a @ lora_b`` for every hidden layer.

    Parameters
    ----------
    model : nnx.Module
        A model whose hidden layers are plain ``nnx.Linear``,
        ``SirenLayer``, or ``RealGaborLayer`` (no LoRA wrappers).
    lora_state : nnx.State
        State containing only ``lora_a`` and ``lora_b`` entries,
        structured to mirror ``model.hidden_layers``.  Typically
        obtained via ``nnx.state(lora_model, nnx.LoRAParam)``.
    """
    for i, layer in enumerate(model.hidden_layers):
        if isinstance(layer, SirenLayer):
            ls = lora_state.hidden_layers[i]
            layer.linear.kernel.value += ls.lora_a.value @ ls.lora_b.value
        elif isinstance(layer, RealGaborLayer):
            fs = lora_state.hidden_layers[i].freqs
            layer.freqs.kernel.value += fs.lora_a.value @ fs.lora_b.value
            ss = lora_state.hidden_layers[i].scales
            layer.scales.kernel.value += ss.lora_a.value @ ss.lora_b.value
        elif isinstance(layer, nnx.Linear):
            ls = lora_state.hidden_layers[i]
            layer.kernel.value += ls.lora_a.value @ ls.lora_b.value
        else:
            raise ValueError(
                f"Unsupported layer type: {type(layer)}. "
                "LoRA can only be applied to Linear, SirenLayer, and RealGaborLayer."
            )
