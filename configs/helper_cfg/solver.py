from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Literal, Annotated, Union

class BaseSolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

class KdVSolverConfig(BaseSolverConfig):
    """KdV solver parameters."""
    name: Literal["kdv"]
    N: int = 512
    systemsize: float = 31.41592653589793  # 10π
    nonlinparameter: float = 1.0
    dt: float = 1e-5
    total_time: float = 2.0


class KolmogorovSolverConfig(BaseSolverConfig):
    """Kolmogorov (2-D Navier–Stokes) solver parameters."""
    name: Literal["kolmogorov"]
    ns: int = 256
    viscosity: float = 1e-3
    max_velocity: float = 5.0
    anti_aliasing: bool = True
    outer_steps: int = 1000
    total_time: float = 25.0

try:
    from JAX_NR.utils_config.config import (
        GridConfig as _GridConfig,
        EvolutionConfig as _EvolutionConfig,
        DampingArgsConfig as _DampingArgsConfig,
    )
    _JAX_NR_AVAILABLE = True
except ImportError:
    _JAX_NR_AVAILABLE = False
    _GridConfig = Any  # type: ignore[assignment,misc]
    _EvolutionConfig = Any  # type: ignore[assignment,misc]
    _DampingArgsConfig = Any  # type: ignore[assignment,misc]


class BSSNSolverConfig(BaseSolverConfig):
    """BSSN (numerical relativity) solver parameters."""
    name: Literal["bssn"]

    grid: _GridConfig
    evolution: _EvolutionConfig
    damping_args: _DampingArgsConfig
    initial_data_path: str


SolverConfig = Annotated[
    Union[KdVSolverConfig, KolmogorovSolverConfig, BSSNSolverConfig],
    Field(discriminator="name"),
]