"""Pydantic configuration schema for ANTIC experiments.

All experiment parameters live here with validation and defaults.
YAML files are loaded and validated against these models.
"""

from __future__ import annotations

from enum import Enum
import json
import os
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, DirectoryPath, FilePath
from .helper_cfg import ModelConfig, SelectorConfig, SolverConfig

class SchedulerType(str, Enum):
    constant = "constant"
    cosine_decay = "cosine_decay"
    warmup_cosine_decay = "warmup_cosine_decay"
    exponential_decay = "exponential_decay"

class SelectorType(str, Enum):
    kdv = "kdv"
    kolmogorov = "kolmogorov"
    bssn = "bssn"
    none = "none"
    
class OptimizerType(str, Enum):
    adamw = "adamw"
    soap = "soap"

class NormalizationMethod(str, Enum):
    standard = "z-score"
    minmax = "min-max"

class FilterType(str, Enum):
    all = "all"
    lora = "lora"
    hidden_layers = "hidden_layers"


class SchedulerConfig(BaseModel):
    """Learning-rate schedule."""
    type: SchedulerType = SchedulerType.cosine_decay
    warmup_percentage: float = 0.1
    init_value: float = 0.0
    end_lr: float = 1e-5
    transition_steps: int = 1
    decay_rate: float = 0.99

    class Config:
        extra = "forbid"


class OptimizerConfig(BaseModel):
    """Optimizer parameters."""
    name: OptimizerType = OptimizerType.adamw
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    
    class Config:
        extra = "forbid"


class TrainingConfig(BaseModel):
    """Training loop parameters."""
    initial_epochs: int = 500
    subsequent_epochs: int = 300
    batch_size: Optional[int] = None
    filter: FilterType = FilterType.all
    rank: int = 4
    reset_every_n: int = 100
    verbose: bool = False
    stop_at: float | Literal['inf'] = 'inf'
    save_dir: str = "ckpts/"
    num_devices: int = 1

    class Config:
        extra = "forbid"

    @model_validator(mode="after")
    def check_batch_devices(self):
        if self.batch_size is not None and self.batch_size % self.num_devices != 0:
            raise ValueError(f"batch_size={self.batch_size} must be divisible by num_devices={self.num_devices}.")
        return self

    @model_validator(mode="after")
    def check_stop_at(self):
        ckpt_dir = Path(self.save_dir) / "checkpoint"
        if self.stop_at != 'inf' and ckpt_dir.exists():
            info_path = ckpt_dir / "info.json"
            if info_path.exists():
                with open(info_path, "r") as f:
                    info = json.load(f)
                if self.stop_at <= info["elapsed_time"]:
                    raise ValueError(f"stop_at={self.stop_at} must be greater than the solver's elapsed_time={info['elapsed_time']}.")
                elif self.stop_at <= info["stop_at"]:
                    raise ValueError(f"stop_at={self.stop_at} must be greater than the stop_at={info['stop_at']} from the last checkpoint.")
        return self


class NormalizationConfig(BaseModel):
    """Neural-field compressor"""
    enabled: bool = False
    method: NormalizationMethod = NormalizationMethod.standard
    window_size: int = 100
    n_features: int = 1

    class Config:
        extra = "forbid"

class WandbConfig(BaseModel):
    """Weights & Biases logging."""
    enabled: bool = False
    project: str = "antic"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    log_every: int = 1

    class Config:
        extra = "forbid"


# ======================================================================
# Top-level experiment config
# ======================================================================

class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    seed: int = 42
    solver: SolverConfig
    model: ModelConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    selector: SelectorConfig = Field(default_factory=SelectorConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


# ======================================================================
# Loader
# ======================================================================

def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML config file.

    If ``solver`` is a plain string (e.g. ``solver: kdv``), the
    corresponding file ``configs/solver/<name>.yaml`` is loaded and
    merged automatically.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    # Resolve solver shorthand: "solver: <name>" -> load configs/solver/<name>.yaml
    solver_val = raw.get("solver")
    if isinstance(solver_val, str):
        solver_name = solver_val
        solver_cfg_path = path.parent / "solver" / f"{solver_name}.yaml"
        with open(solver_cfg_path) as f:
            solver_raw = yaml.safe_load(f) or {}
        solver_raw["name"] = solver_name
        raw["solver"] = solver_raw

    return ExperimentConfig(**raw)
