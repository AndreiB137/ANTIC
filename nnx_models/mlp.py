from flax import nnx
import jax.numpy as jnp
from typing import Callable
from nnx_models.utils import FourierLinear

class MLP(nnx.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_dim: int, 
                 num_hidden_layers: int = 2, 
                 act: Callable = nnx.silu,
                 fourier_emb_dim: int = 128,
                 fourier_emb_scale: float = 7.0, 
                 dtype: jnp.dtype = jnp.float32, 
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.act = act
        self.in_layer = FourierLinear(input_dim,
                                    fourier_emb_dim,
                                    fourier_emb_scale,
                                    dtype=dtype,
                                    rngs=rngs)
        self.out_layer = nnx.Linear(
            hidden_dim, output_dim, use_bias=True, param_dtype=dtype, rngs=rngs
        )
        self.hidden_layers = nnx.List()
        self.hidden_layers.append(nnx.Linear(fourier_emb_dim, hidden_dim, use_bias=True, param_dtype=dtype, rngs=rngs))
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nnx.Linear(hidden_dim, hidden_dim, use_bias=True, param_dtype=dtype, rngs=rngs))

    def __call__(self, x):
        x = self.in_layer(x)
        x = jnp.concatenate([
            jnp.cos(x), jnp.sin(x)
        ], axis=-1)
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        x = self.out_layer(x)
        return x