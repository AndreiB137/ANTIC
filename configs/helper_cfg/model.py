from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Annotated

class CommonModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_dim: int = 1
    output_dim: int = 1
    hidden_dim: int = 64
    num_hidden_layers: int = 3
    dtype: str = "float32"

class MLPConfig(CommonModelConfig):
    name: Literal["mlp", "mlp_plus"]
    fourier_emb_dim: int = 128
    fourier_emb_scale: float = 7.0

class SIRENConfig(CommonModelConfig):
    name: Literal["siren"]
    first_omega: float = 1.0
    hidden_omega: float = 1.0

class WIREConfig(CommonModelConfig):
    name: Literal["wire"]
    first_omega: float = 1.0
    hidden_omega: float = 1.0
    scale: float = 1.0
    complexgabor: bool = False

ModelConfig = Annotated[
    MLPConfig | SIRENConfig | WIREConfig,
    Field(discriminator="name")
]