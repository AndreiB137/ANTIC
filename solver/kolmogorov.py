
"""2-D Navier–Stokes (Kolmogorov flow) solver wrapping JAX-CFD."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import yaml
import numpy as np
import logging

import jax
import jax.numpy as jnp

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral

from solver.solver import Solver


class KolmogorovSolver(Solver):
    """2-D Navier–Stokes solver via JAX-CFD pseudo-spectral method.

    State is the vorticity field in Fourier space (rfft2 output with shape
    ``(ns, ns // 2 + 1)``).  The :meth:`extract` method converts back to
    real-space vorticity of shape ``(ns, ns)``.

    Parameters
    ----------
    ns : int
        Grid resolution (square: ns × ns).
    domain : tuple of tuple of float
        Physical domain extents, e.g. ``((0, 2π), (0, 2π))``.
    viscosity : float
        Kinematic viscosity.
    max_velocity : float
        Maximum velocity magnitude (used for CFL-based dt).
    anti_aliasing : bool
        Whether to apply a 2/3-rule dealiasing filter.
    mol : callable
        JAX-CFD time-stepping method (e.g.
        ``spectral.time_stepping.crank_nicolson_rk4``).
    """

    def __init__(
        self,
        ns: int = 256,
        domain: tuple = ((0.0, 2.0 * jnp.pi), (0.0, 2.0 * jnp.pi)),
        total_time: float = 25.0,
        viscosity: float = 1e-3,
        max_velocity: float = 5.0,
        anti_aliasing: bool = True,
        outer_steps: int = 1000,
        mol=spectral.time_stepping.crank_nicolson_rk4,
    ):
        self.ns = ns
        self.domain = domain
        self.viscosity = viscosity
        self.max_velocity = max_velocity
        self.anti_aliasing = anti_aliasing

        # Build JAX-CFD grid and integrator
        self.grid = grids.Grid((ns, ns), domain=domain)
        dt = cfd.equations.stable_time_step(max_velocity, 0.5, viscosity, self.grid)
        self.inner_steps = int((total_time / dt) / outer_steps)
        step_fn = mol(
            spectral.equations.NavierStokes2D(viscosity, self.grid, smooth=anti_aliasing),
            dt,
        )
        self.elapsed_time = 0.0
        self.outer_step_fn = cfd.funcutils.repeated(step_fn, self.inner_steps)
        self.dt = dt
        self.tf = total_time
        self.mol = mol
        self.coords = jnp.stack(
            jnp.meshgrid(
                jnp.linspace(0, 2 * jnp.pi, ns, endpoint=False),
                jnp.linspace(0, 2 * jnp.pi, ns, endpoint=False),
            ),
            axis=-1
        ).reshape(-1, 2)

        self.step_fn = step_fn

    def initialize(self, seed: int = 42, peak_wavenumber: int = 4) -> jnp.ndarray:
        """Create a random initial vorticity field in Fourier space.

        Returns
        -------
        vorticity_hat : jnp.ndarray, shape ``(ns, ns // 2 + 1)``
        """
        v0 = cfd.initial_conditions.filtered_velocity_field(
            jax.random.PRNGKey(seed), self.grid, self.max_velocity, peak_wavenumber,
        )
        vorticity0 = cfd.finite_differences.curl_2d(v0).data
        return jnp.fft.rfftn(vorticity0)
    
    def step(self, state: jnp.ndarray) -> jnp.ndarray:
        """Advance the Navier-Stokes vorticity by one outer time step."""
        self.elapsed_time += self.inner_steps * self.dt
        return jax.jit(cfd.funcutils.repeated(self.outer_step_fn, 1))(state)
    
    def extract(self, state: jnp.ndarray) -> jnp.ndarray:
        """Convert Fourier-space vorticity to a real-space column vector."""
        return jnp.fft.irfftn(state, s=(self.ns, self.ns), axes=(0, 1)).reshape(-1, 1)

    def rollout(self, state: jnp.ndarray, n_steps: int):
        """Advance the Fourier-space vorticity by *n_steps* outer steps."""
        return jax.jit(cfd.funcutils.repeated(self.outer_step_fn, n_steps))(state)

    def prepare_coords(self):
        return self.coords

    def save_state(self, state: jnp.ndarray, directory: str):
        """Save the Fourier-space state and elapsed time on disk."""
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, f"solver")
        jnp.save(filename + "/state.npy", state)
        json.dump({"elapsed_time": self.elapsed_time}, open(filename + "/metadata.json", "w"), indent=2)

    def load_state(self, directory: str) -> jnp.ndarray:
        """Load a previously saved Fourier-space state and elapsed time from a saved directory."""
        if os.path.exists(directory):
            with open(directory + "/metadata.json", "r") as f:
                metadata = json.load(f)
            self.elapsed_time = metadata.get("elapsed_time", 0.0)
            return jnp.load(directory + "/state.npy")
        else:
            raise FileNotFoundError(f"State file {directory} not found.")

    def trajectory(self, state: jnp.ndarray, outer_steps: int, inner_steps: int):
        """Run a full trajectory, saving snapshots every *inner_steps*.

        Parameters
        ----------
        state : jnp.ndarray
            Initial vorticity in Fourier space.
        outer_steps : int
            Number of snapshots to save.
        inner_steps : int
            Number of integrator steps between each snapshot.

        Returns
        -------
        final_state : jnp.ndarray
            State after the full integration.
        trajectory : jnp.ndarray, shape ``(outer_steps, ns, ns // 2 + 1)``
            Saved Fourier-space vorticity snapshots.
        """
        trajectory_fn = cfd.funcutils.trajectory(
            cfd.funcutils.repeated(self.step, inner_steps), outer_steps,
        )
        return trajectory_fn(state)


# ------------------------------------------------------------------
# Standalone data-generation utility
# ------------------------------------------------------------------
def generate_training_and_validation_grid(config: dict) -> None:
    """Generate a 2-D NS vorticity dataset and save to disk.

    Parameters
    ----------
    config : dict
        Must contain at least ``ns``, ``viscosity``, ``max_velocity``,
        ``seed``, and ``data_save_dir``.  See the ``if __name__`` block
        for a full example.
    """
    ns = config["ns"]
    tf = config.get("tf", 25.0)
    outer_steps = config.get("outer_steps", 100)
    snapshots = config.get("temporal_snapshots", outer_steps // 10)
    temporal_slice = config.get(
        "vorticity_temporal_slice", slice(0, outer_steps, max(1, snapshots)),
    )
    precision = config.get("precision", jnp.float32)
    data_save_dir = Path(config.get("data_save_dir", "."))
    data_save_dir.mkdir(parents=True, exist_ok=True)

    solver = KolmogorovSolver(
        ns=ns,
        domain=config.get("domain", ((0.0, 2.0 * jnp.pi), (0.0, 2.0 * jnp.pi))),
        viscosity=config.get("viscosity", 1e-3),
        max_velocity=config.get("max_velocity", 5.0),
        anti_aliasing=config.get("anti_aliasing", True),
        outer_steps=config.get("outer_steps", 1000),
        mol=config.get("mol", spectral.time_stepping.crank_nicolson_rk4),
    )

    inner_steps = int((tf // solver.dt) // outer_steps)
    vorticity_hat0 = solver.initialize(seed=config.get("seed", 42))
    _, traj = solver.trajectory(vorticity_hat0, outer_steps, inner_steps)

    coords = solver.coords

    traj = jnp.fft.irfftn(traj[temporal_slice], s=(ns, ns))

    jnp.save(
        data_save_dir / "coord.npy",
        jnp.array(coords, dtype=precision),
        allow_pickle=True,
    )
    jnp.save(
        data_save_dir / "vorticity_trajectory.npy",
        traj.astype(precision),
        allow_pickle=True,
    )

    space_label = "real" 
    logging.info(
        f"Saved 2D NS vorticity grid ({ns}x{ns}) in {space_label} space "
        f"to {data_save_dir}, {outer_steps} steps, slice {temporal_slice}."
    )

if __name__ == "__main__":
    os.environ["JAX_PLATFORMS"] = "gpu"
    jax.config.update("jax_enable_x64", False)

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO,
    )

    ns = 256
    domain = ((0, 2 * jnp.pi), (0, 2 * jnp.pi))
    eta = 1e-3
    v = 5
    tf = 25.0
    out_steps = 1000
    snapshots = 1000

    config = {
        "ns": ns,
        "domain": domain,
        "viscosity": eta,
        "max_velocity": v,
        "tf": tf,
        "outer_steps": out_steps,
        "seed": 42,
        "temporal_snapshots": snapshots,
        "vorticity_temporal_slice": slice(0, out_steps, out_steps // snapshots),
        "anti_aliasing": True,
        "mol": spectral.time_stepping.crank_nicolson_rk4,
        "precision": jnp.float32,
        "data_save_dir": "..",
    }

    generate_training_and_validation_grid(config)

    config_path = Path(config["data_save_dir"]) / "config.yml"
    with open(config_path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)

    logging.info(f"Stored config file at {config_path}")
    logging.info("Simulation data and config storage complete.")



