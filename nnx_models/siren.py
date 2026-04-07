import jax.numpy as jnp
from flax import nnx
from nnx_models.utils import custom_uniform

class SIREN(nnx.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_dim: int = 64, 
                 num_hidden_layers: int = 3, 
                 first_omega: float = 1.0,
                 hidden_omega: float = 1.0,
                 dtype: jnp.dtype = jnp.float32,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.first_omega = first_omega
        self.hidden_omega = hidden_omega
        self.dtype = dtype
        self.rngs = rngs
        self.in_layer = SirenLayer(
            input_dim=input_dim,
            output_dim=hidden_dim,
            omega_0=first_omega,
            is_first_layer=True,
            dtype=dtype,
            rngs=rngs
        )
        self.out_layer = nnx.Linear(
            in_features=hidden_dim,
            out_features=output_dim,
            use_bias=True,
            param_dtype=dtype,
            rngs=rngs
        )

        self.hidden_layers = nnx.List([
            SirenLayer(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                omega_0=hidden_omega,
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

class SirenLayer(nnx.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 omega_0: float = 1.0,
                 is_first_layer: bool = False,
                 dtype: jnp.dtype = jnp.float32,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        
        self.omega_0 = omega_0
        self.is_first_layer = is_first_layer
        c = 1 if is_first_layer else 6 / self.omega_0**2
        self.linear = nnx.Linear(
            input_dim, 
            output_dim, 
            use_bias=True, 
            param_dtype=dtype,
            kernel_init=custom_uniform(numerator=c, mode='fan_in', distribution='uniform', dtype=dtype),
            bias_init=nnx.initializers.zeros,
            rngs=rngs
        )
    def __call__(self, x):
        after_linear = self.omega_0 * self.linear(x)
        return jnp.sin(after_linear) 

if __name__ == "__main__":
    # Example usage
    model = SIREN(
        input_dim=2,
        output_dim=3,
        hidden_dim=64,
        num_hidden_layers=3,
        first_omega=30.0,
        hidden_omega=30.0
    )

    x = jnp.ones((10, 2))  # Example input
    print(nnx.state(model, nnx.Param))  # Print model parameters    
    y = model(x)
    print(y)