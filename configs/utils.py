"""Utility functions for building components from config."""
import json
from pathlib import Path

from temporal_selector.no_selector import NoneSelector

from .schema import NormalizationMethod, OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType, NormalizationConfig
from .helper_cfg import ModelConfig, SelectorConfig, SolverConfig
import optax
from nnx_models import model_fn
from flax import nnx
import jax.numpy as jnp
from neural_compressor.normalization import WelfordArrayOnline, MinMaxArrayOnline

def build_schedule(opt_cfg: OptimizerConfig, schedule_cfg: SchedulerConfig, decay_steps: int | None = None):
    """Build an optax learning-rate schedule from config."""
    if schedule_cfg.type == "constant":
        return optax.constant_schedule(opt_cfg.learning_rate)
    elif schedule_cfg.type == "cosine_decay":
        return optax.cosine_decay_schedule(
            init_value=opt_cfg.learning_rate,
            decay_steps=decay_steps,
            alpha=schedule_cfg.end_lr / opt_cfg.learning_rate,
        )
    elif schedule_cfg.type == "warmup_cosine_decay":
        return optax.warmup_cosine_decay_schedule(
            init_value=schedule_cfg.init_value,
            peak_value=opt_cfg.learning_rate,
            warmup_steps=int(schedule_cfg.warmup_percentage * decay_steps) if decay_steps is not None else 0,
            decay_steps=decay_steps,
        )
    elif schedule_cfg.type == "exponential_decay":
        return optax.exponential_decay(
            init_value=opt_cfg.learning_rate,
            transition_steps=schedule_cfg.transition_steps,
            decay_rate=schedule_cfg.decay_rate,
            end_value=schedule_cfg.end_lr,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {schedule_cfg.type}. Supported types are: {list(SchedulerType)}")


def build_optimizer(opt_cfg: OptimizerConfig, schedule_cfg: SchedulerConfig, decay_steps: int):
    """Build an optax optimizer from config."""
    schedule = build_schedule(opt_cfg, schedule_cfg, decay_steps)
    if opt_cfg.name == "soap":
        from soap_jax import soap
        return soap(
            learning_rate=schedule,
            precondition_frequency=1,
            weight_decay=opt_cfg.weight_decay,
        )
    elif opt_cfg.name == "adamw":
        return optax.adamw(learning_rate=schedule, weight_decay=opt_cfg.weight_decay)
    else:
        raise NotImplementedError(f"Unknown optimizer type: {opt_cfg.name}. Not implemented yet. Supported types are: {list(OptimizerType)}")


def build_solver(cfg: SolverConfig):
    """Instantiate the PDE solver from config."""
    if cfg.name == "kdv":
        from solver.kdv import KDVSolver
        return KDVSolver(
            N=cfg.N,
            systemsize=cfg.systemsize,
            nonlinparameter=cfg.nonlinparameter,
            dt=cfg.dt,
        )
    elif cfg.name == "kolmogorov":
        from solver.kolmogorov import KolmogorovSolver
        return KolmogorovSolver(
            ns=cfg.ns,
            viscosity=cfg.viscosity,
            max_velocity=cfg.max_velocity,
            anti_aliasing=cfg.anti_aliasing,
            outer_steps=cfg.outer_steps,
        )
    elif cfg.name == "bssn":
        from solver.bssn import BSSNSolver
        return BSSNSolver(config=cfg)
    raise ValueError(f"Unknown solver: {cfg.name}")


def build_model(cfg: ModelConfig, seed: int = 42) -> nnx.Module:
    """Create the neural-field model from config."""
    name = cfg.name
    kwargs = cfg.model_dump(exclude={"name"})
    model = model_fn[name](rngs=nnx.Rngs(seed), **kwargs)
    return model

def build_normalization(cfg: NormalizationConfig):
    """Create the normalization object from config."""
    if not cfg.enabled:
        return None
    if cfg.method == "z-score":
        return WelfordArrayOnline(n_features=cfg.n_features, window_size=cfg.window_size)
    elif cfg.method == "min-max":
        return MinMaxArrayOnline(n_features=cfg.n_features, window_size=cfg.window_size)
    else:
        raise ValueError(f"Unknown normalization method: {cfg.method}. Supported methods are: {list(NormalizationMethod)}")

def build_selector(cfg: SelectorConfig):
    """Instantiate the temporal selector from config."""
    if cfg.type == "none":
        return NoneSelector()
    elif cfg.type == "kdv":
        from pats.kdv_selector import KdVActivitySelector
        kwargs = cfg.model_dump(exclude={"type"})
        return KdVActivitySelector(**kwargs)
    elif cfg.type == "kolmogorov":
        from pats.enstrophy_selector import EnstrophySelector
        kwargs = cfg.model_dump(exclude={"type"})
        return EnstrophySelector(**kwargs)
    elif cfg.type == "bssn":
        from pats.bssn_median_selector import BSSN_MedianSelector
        kwargs = cfg.model_dump(exclude={"type"})
        return BSSN_MedianSelector(**kwargs)
    raise ValueError(f"Unknown selector type: {cfg.type}")

def compute_decay_steps(coords: jnp.ndarray, batch_size: int | None, epochs: int) -> int:
    """Number of optimiser update steps for the scheduler."""
    n = coords.shape[0]
    if batch_size is None or batch_size >= n:
        return epochs
    return (n // batch_size) * epochs
