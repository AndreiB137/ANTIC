import jax
import math
from typing import Any, Dict, Sequence, Union

import jax.numpy as jnp
from jax import dtypes, random
from jax.nn.initializers import Initializer
from typing import Callable
from flax import linen as nn
from flax import nnx

class FourierLinear(nnx.Module):
    def __init__(self, input_dim: int, output_dim: int, embed_scale: float, dtype: jnp.dtype = jnp.float32, rngs: nnx.Rngs = nnx.Rngs(0)):
        kernel_key = rngs.params()
        self.in_features = input_dim
        self.out_features = output_dim
        self.kernel = nnx.Param(nnx.initializers.normal(embed_scale, dtype=dtype)(kernel_key, (input_dim, output_dim // 2), dtype=dtype))

    def __call__(self, x):
        return x @ self.kernel


def _compute_fans(
    shape: tuple,
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Union[int, Sequence[int]] = (),
):
    """Compute effective input and output sizes for a linear or convolutional layer.

    Axes not in in_axis, out_axis, or batch_axis are assumed to constitute the "receptive field" of
    a convolution (kernel spatial dimensions).
    """
    if len(shape) <= 1:
        raise ValueError(
            f"Can't compute input and output sizes of a {shape.rank}"
            "-dimensional weights tensor. Must be at least 2D."
        )

    if isinstance(in_axis, int):
        in_size = shape[in_axis]
    else:
        in_size = math.prod([shape[i] for i in in_axis])
    if isinstance(out_axis, int):
        out_size = shape[out_axis]
    else:
        out_size = math.prod([shape[i] for i in out_axis])
    if isinstance(batch_axis, int):
        batch_size = shape[batch_axis]
    else:
        batch_size = math.prod([shape[i] for i in batch_axis])
    receptive_field_size = math.prod(shape) / in_size / out_size / batch_size
    fan_in = in_size * receptive_field_size
    fan_out = out_size * receptive_field_size
    return fan_in, fan_out


def custom_uniform(
    numerator: float = 6,
    mode: str = "fan_in",
    dtype: jnp.dtype = jnp.float_,
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    distribution: str = "uniform",
) -> Initializer:
    """Builds an initializer that returns real uniformly-distributed random arrays.

    :param numerator: the numerator of the range of the random distribution.
    :type numerator: float
    :param mode: the mode for computing the range of the random distribution.
    :type mode: str
    :param dtype: optional; the initializer's default dtype.
    :type dtype: jnp.dtype
    :param in_axis: the axis or axes that specify the input size.
    :type in_axis: Union[int, Sequence[int]]
    :param out_axis: the axis or axes that specify the output size.
    :type out_axis: Union[int, Sequence[int]]
    :param batch_axis: the axis or axes that specify the batch size.
    :type batch_axis: Sequence[int]
    :param distribution: the distribution of the random distribution.
    :type distribution: str

    :return: An initializer that returns arrays whose values are uniformly distributed in
        the range ``[-range, range)``.
    :rtype: Initializer
    """

    def init(key: jax.random.key, shape: tuple, dtype: Any = dtype) -> Any:
        dtype = dtypes.canonicalize_dtype(dtype)
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis, batch_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(f"invalid mode for variance scaling initializer: {mode}")
        if distribution == "uniform":
            return random.uniform(
                key,
                shape,
                dtype,
                minval=-jnp.sqrt(numerator / denominator),
                maxval=jnp.sqrt(numerator / denominator),
            )
        elif distribution == "normal":
            return random.normal(key, shape, dtype) * jnp.sqrt(numerator / denominator)
        elif distribution == "uniform_squared":
            return random.uniform(
                key, shape, dtype, minval=-numerator / denominator, maxval=numerator / denominator
            )
        else:
            raise ValueError(
                f"invalid distribution for variance scaling initializer: {distribution}"
            )

    return init
