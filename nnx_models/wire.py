from flax import nnx
import jax
import jax.numpy as jnp
from typing import Any

from nnx_models.utils import custom_uniform

def complex_kernel_uniform_init(numerator : float = 6,
                                mode : str = "fan_in",
                                dtype : jnp.dtype = jnp.float32,
                                distribution: str = "uniform") -> jax.nn.initializers.Initializer:
    def init(key: jax.random.key, shape: tuple, dtype: jax.random.key) -> Any:
        key1, key2 = jax.random.split(key)
        if dtype == jnp.complex64:
            real_kernel = custom_uniform(numerator=numerator, mode=mode, distribution=distribution)(key1, shape, jnp.float32)
            imag_kernel = custom_uniform(numerator=numerator, mode=mode, distribution=distribution)(key2, shape, jnp.float32)
        elif dtype == jnp.complex128:
            real_kernel = custom_uniform(numerator=numerator, mode=mode, distribution=distribution)(key1, shape, jnp.float64)
            imag_kernel = custom_uniform(numerator=numerator, mode=mode, distribution=distribution)(key2, shape, jnp.float64)

        return real_kernel + 1j * imag_kernel
        
    return init

class WIRE(nnx.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 num_hidden_layers: int,
                 first_omega: float = 1.0,
                 hidden_omega: float = 1.0,
                 scale: float = 1.0,
                 complexgabor: bool = False,
                 dtype: jnp.dtype = jnp.float32,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.first_omega = first_omega
        self.hidden_omega = hidden_omega
        self.scale = scale
        self.complexgabor = complexgabor
        self.dtype = dtype
        self.rngs = rngs
        if self.complexgabor:
            WIRElayer = ComplexGaborLayer
        else:
            WIRElayer = RealGaborLayer
        self.in_layer = WIRElayer(
            input_dim=input_dim,
            output_dim=hidden_dim,
            omega_0=first_omega,
            s_0=scale,
            is_first_layer=True,
            dtype=dtype,
            rngs=rngs
        )
        self.out_layer = WIRElayer(
            input_dim=hidden_dim,
            output_dim=output_dim,
            omega_0=hidden_omega,
            s_0=scale,
            is_first_layer=False,
            dtype=dtype,
            rngs=rngs
        )
        self.hidden_layers = nnx.List([
            WIRElayer(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                omega_0=hidden_omega,
                s_0=scale,
                is_first_layer=False,
                dtype=dtype,
                rngs=rngs
            ) for _ in range(num_hidden_layers)
        ])
    def __call__(self, x):
        x = self.in_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.out_layer(x)
        return x

class ComplexGaborLayer(nnx.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int, 
                 omega_0: float, 
                 s_0: float, 
                 is_first_layer: bool = False,
                 dtype: jnp.dtype = jnp.float32,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        
        self.omega_0 = omega_0
        self.s_0 = s_0
        c = 1 if is_first_layer else 6 / self.omega_0**2
        if dtype == jnp.float32:
            dtype = jnp.complex64
        elif dtype == jnp.float64:
            dtype = jnp.complex128
        distrib = "uniform_squared" if is_first_layer else "uniform"
        self.linear = nnx.Linear(
            in_features=input_dim,
            out_features=output_dim,
            use_bias=True,
            kernel_init=complex_kernel_uniform_init(numerator=c, mode="fan_in", distribution=distrib, dtype=dtype),
            param_dtype=dtype,
            rngs=rngs
        )
    def __call__(self, x):
        omega = self.omega_0 * self.linear(x)
        scale = self.s_0 * self.linear(x)

        return jnp.exp(1j * omega - (jnp.abs(scale)**2))
    
class RealGaborLayer(nnx.Module):
    def __init__(self, 
                input_dim: int,
                output_dim: int, 
                omega_0: float, 
                s_0: float, 
                is_first_layer: bool = False,
                dtype: jnp.dtype = jnp.float32,
                rngs: nnx.Rngs = nnx.Rngs(0)):
        
        self.omega_0 = omega_0
        self.s_0 = s_0
        c = 1 if is_first_layer else 6 / self.omega_0**2
        distrib = "uniform_squared" if is_first_layer else "uniform"
        self.freqs = nnx.Linear(
            in_features=input_dim,
            out_features=output_dim,
            use_bias=True,
            kernel_init=custom_uniform(numerator=c, mode="fan_in", distribution=distrib, dtype=dtype),
            param_dtype=dtype,
            rngs=rngs
        )
        self.scales = nnx.Linear(
            in_features=input_dim,
            out_features=output_dim,
            use_bias=True,
            kernel_init=custom_uniform(numerator=c, mode="fan_in", distribution=distrib, dtype=dtype),
            param_dtype=dtype,
            rngs=rngs
        )

    def __call__(self, x):
        omega = self.omega_0 * self.freqs(x)
        scale = self.s_0 * self.scales(x)

        return jnp.cos(omega) * jnp.exp(-(scale**2))
    

if __name__ == "__main__":
    # Example usage
    model = WIRE(input_dim=3, output_dim=4, hidden_dim=5, num_hidden_layers=2, first_omega=1.0, hidden_omega=1.0, scale=1.0, complexgabor=False)
    x = jnp.ones((10, 3))

    output = model(x)
    print(nnx.state(model, nnx.Param))
    print(output.shape)  # Should print (10, 4)