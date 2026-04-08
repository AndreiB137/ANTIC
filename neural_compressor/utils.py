from typing import Any, Union
from flax import nnx
import jax
import jax.numpy as jnp

FilterSpec = Union[str, nnx.Variable]
"""User-facing filter: a preset name, an ``nnx`` variable type, or callable."""

FILTER_PRESETS: dict[str, Any] = {
    "all": nnx.Param,
    "lora": nnx.LoRAParam,
    "hidden_layers": nnx.PathContains('hidden_layers'),
}


def resolve_filter(spec: FilterSpec) -> Any:
    """Turn a *FilterSpec* into an nnx-compatible filter.

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
    if isinstance(spec, type) and issubclass(spec, nnx.Variable):
        return spec
    
    raise TypeError(f"Invalid filter spec {spec!r}: must be a preset name or nnx.Variable subclass.")


def extract_state(model: nnx.Module, resolved_filter: nnx.Variable | None) -> nnx.State:
    """Extract state from *model*, optionally narrowed by *resolved_filter*."""
    if resolved_filter is None:
        return nnx.state(model)
    return nnx.state(model, resolved_filter)
    
def check_nan(model: nnx.Module) -> bool:
    """Return True if any model parameter contains NaN."""
    leaves = jax.tree_util.tree_leaves(nnx.state(model))
    return any(jnp.any(jnp.isnan(leaf)).item() for leaf in leaves)