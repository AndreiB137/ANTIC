"""BSSN (3+1 numerical relativity) solver wrapping JAX_NR."""

from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from pydantic import BaseModel
from JAX_NR.initial_conditions import setup_grid, create_grid_masks
from JAX_NR.finite_difference import distance_to_boundary
from JAX_NR.bssn_evolution import (
    backward_euler,
    create_jac_hessian_variables,
    conf_gamma,
    get_jacobians,
    bssn_to_adm,
    hamiltonian_constraint_error,
    momentum_constraint_error,
    conf_gamma_error,
)
from JAX_NR.tensor_operations import christoffel_first_kind, christoffel_second_kind
from JAX_NR.utils import inv_33_fn
from JAX_NR.utils.utils import bssn_dict_to_array, jac_dict_to_array
from JAX_NR.wave_extraction import (
    weyl_tensor,
    build_np_m_from_coords_and_metric,
    weyl_scalar_outgoing_field,
    weyl_scalar_swsh_coefficients_at_extraction_radii,
)
from JAX_NR.utils_config import load_config

from solver.solver import Solver

State = Dict[str, jax.Array]


class BSSNSolver(Solver):
    """BSSN 3+1 numerical relativity solver wrapping JAX_NR.

    State is a dictionary of BSSN variables with keys:
    ``'lapse'``, ``'beta'``, ``'conf_metric'``, ``'conf_a'``,
    ``'K'``, ``'W'``, ``'conf_gamma'``.

    The :meth:`extract` method returns the main BSSN variables
    excluding shift vector (``'beta'``), while ``lapse`` is kept
    for visualization purposes, but as a gauge variable it is not
    present in the calculation of the Weyl scalar, the quantity 
    of interest in black hole merger simulations. 

    Parameters
    ----------
    config_path : str or Path
        Path to a JAX_NR YAML config file, found in ``configs/solver/bssn.yaml``
    """

    def __init__(self, config : BaseModel):
        self.config = config
        self.n = config.grid.n

        # ---- grid setup ----------------------------------------------------
        grid_info = setup_grid(self.n, config.grid.domain_width)
        self.coords = grid_info["coords"]
        self.coord_linspaces = grid_info["coord_linspaces"]
        self.grid = grid_info["grid"]
        self.scale = grid_info["scale"]
        self.grid_dims = grid_info["grid_dims"]
        self.mask, self.axis_mask = create_grid_masks(self.n)
        self.dist = distance_to_boundary(self.grid, self.grid.shape[0])

        # ---- damping / evolution params ------------------------------------
        self.damping_args = config.damping_args.model_dump()
        self.dt = config.evolution.dt
        self.tf = config.evolution.total_time

        # ---- flat-space boundary variables ---------------------------------
        # (will be shaped correctly once initial data is loaded)
        self.boundary_variables : State = None

        # ---- JIT-compiled helpers ------------------------------------------
        self._backward_euler_jit = jax.jit(
            backward_euler, static_argnames=("dt", "scale")
        )
        self._create_jac_hessian_jit = jax.jit(
            create_jac_hessian_variables, static_argnames=("scale",)
        )
        self._copy_state_jit = jax.jit(
            lambda s: jax.tree.map(lambda x: x.copy(), s)
        )
        self.get_jacobian_jit = jax.jit(
            lambda s: get_jacobians(s, self.axis_mask, self.scale)
        )

    def _build_boundary_variables(self, bssn_variables: State) -> None:
        """Construct boundary conditions given the BSSN variables."""
        gd = self.grid_dims
        self.boundary_variables = {
            "conf_metric": jnp.tile(
                jnp.diag(jnp.array([1.0, 1.0, 1.0])), gd + (1, 1)
            ),
            "conf_a": jnp.zeros_like(bssn_variables["conf_a"]),
            "K": jnp.zeros_like(bssn_variables["K"]),
            "W": jnp.ones_like(bssn_variables["W"]),
            "conf_gamma": jnp.zeros_like(bssn_variables["conf_gamma"]),
            "lapse": jnp.ones_like(bssn_variables["lapse"]),
            "beta": jnp.zeros_like(bssn_variables["beta"]),
        }

    @staticmethod
    def _convert_bssn_var_to_array(bssn_variables: State) -> jnp.ndarray:
        """Pack the BSSN variable dictionary into a single concatenated array shape (N, 18) (excluding the shift vector)."""
        bssn_array = bssn_dict_to_array(bssn_variables)
        return jnp.concatenate([
            bssn_array[..., :3],
            bssn_array[..., 6:],
        ], axis=-1)
    
    @staticmethod
    def _convert_jac_var_to_dict(jac_variables: State) -> jnp.ndarray:
        """Pack the Jacobian variable dictionary into a single concatenated array shape (N, 3, 18) (excluding the shift vector)."""
        jac_array = jac_dict_to_array(jac_variables)
        return jnp.concatenate([
            jac_array[..., :3],
            jac_array[..., 6:],
        ], axis=-1)

    def step(self, bssn_variables: State) -> State:
        """Advance the BSSN system by one backward Euler time step."""
        old = self._copy_state_jit(bssn_variables)
        new = self._backward_euler_jit(
            old,
            bssn_variables,
            self.boundary_variables,
            self.damping_args,
            self.coords,
            self.mask,
            self.axis_mask,
            self.elapsed_time,
            self.dist,
            self.dt,
            self.scale,
        )
        self.elapsed_time += self.dt
        return new

    def extract(self, bssn_variables: State, get_jacobian: bool = False) -> jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Extract the BSSN variable snapshot from *bssn_variables*.

        Parameters
        ----------
        bssn_variables : dict
            The full BSSN variable dictionary.
        get_jacobian : bool, optional
            If ``True``, also extract the Jacobian variables.

        Returns
        -------
        jnp.ndarray or tuple of (snapshot, jacobian_dict)
            The snapshot is a concatenated array of the main BSSN variables.

        """
        if get_jacobian:
            return jax.jit(self._convert_bssn_var_to_array)(bssn_variables), jax.jit(self._convert_jac_var_to_dict)(self.get_jacobian_jit(bssn_variables))
        return jax.jit(self._convert_bssn_var_to_array)(bssn_variables)

    def rollout(self, state: State, n_steps: int) -> State:
        """Advance the BSSN state by *n_steps* backward-Euler time steps."""
        for _ in range(n_steps):
            state = self.step(state)
        return state

    def prepare_coords(self):
        return self.coords.reshape(-1, 3)
    
    def save_state(self, bssn_variables: State, directory: str | Path) -> None:
        """Store the BSSN variables and elapsed time to *directory*."""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "bssn_variables.pkl", "wb") as f:
            pickle.dump(bssn_variables, f)
        with open(save_dir / "elapsed_time.pkl", "wb") as f:
            pickle.dump(self.elapsed_time, f)

    def load_state(self, directory: str | Path) -> State:
        """Load BSSN variables from a checkpoint or initial conditions file.

        Also builds the flat-space boundary variables and resets the elapsed
        time counter.

        Parameters
        ----------
        directory : str or Path
            Directory containing ``bssn_variables.pkl`` (and optionally
            ``elapsed_time.pkl``). If ``elapsed_time.pkl`` is not found, 
            it will be initialized to 0.0, assuming a fresh run.

        Returns
        -------
        state : dict[str, jax.Array]
            The BSSN variable dictionary.
        """

        cp = Path(directory)

        with open(cp / "bssn_variables.pkl", "rb") as f:
            bssn_variables: State = pickle.load(f)

        elapsed_time = 0.0
        elapsed_path = cp / "elapsed_time.pkl"
        if elapsed_path.exists():
            with open(elapsed_path, "rb") as f:
                elapsed_time = pickle.load(f)

        self.elapsed_time = elapsed_time
        self._build_boundary_variables(bssn_variables)
        return bssn_variables

    def compute_constraints(
        self, bssn_variables: State
    ) -> Dict[str, float]:
        """Compute Hamiltonian, momentum, and conformal-Gamma constraint errors.

        Returns
        -------
        dict with keys ``'hamiltonian'``, ``'momentum'``, ``'gamma'``
            Each value is a scalar (log10 L2 norm).
        """
        jac_vars, hess_vars = self._create_jac_hessian_jit(
            bssn_variables, self.axis_mask, self.scale
        )
        inv_metric_conf = inv_33_fn(bssn_variables["conf_metric"])
        christoffel_1st = christoffel_first_kind(jac_vars["conf_metric"])
        christoffel_2nd = christoffel_second_kind(
            jac_vars["conf_metric"], inv_metric_conf
        )
        conf_gamma_und = conf_gamma(jac_vars["conf_metric"], inv_metric_conf)

        ham = hamiltonian_constraint_error(
            bssn_variables, jac_vars, hess_vars,
            inv_metric_conf, christoffel_1st, christoffel_2nd, conf_gamma_und,
        )
        mom = momentum_constraint_error(
            bssn_variables, jac_vars, inv_metric_conf, self.axis_mask, self.scale,
        )
        gam = conf_gamma_error(
            bssn_variables["conf_gamma"], jac_vars["conf_metric"], inv_metric_conf,
        )
        return {"hamiltonian": ham, "momentum": mom, "gamma": gam}

    def compute_adm_variables(self, bssn_variables: State) -> Dict[str, jax.Array]:
        """Reconstruct ADM variables from the current BSSN state.

        Returns
        -------
        dict with keys ``'spatial_metric'``, ``'extrinsic_curvature'``,
        ``'extrinsic_curvature_jac'``, ``'christoffel_second_kind'``,
        ``'ricci_tensor'``.
        """
        jac_vars, hess_vars = self._create_jac_hessian_jit(
            bssn_variables, self.axis_mask, self.scale
        )
        inv_metric_conf = inv_33_fn(bssn_variables["conf_metric"])
        christoffel_1st = christoffel_first_kind(jac_vars["conf_metric"])
        christoffel_2nd = christoffel_second_kind(
            jac_vars["conf_metric"], inv_metric_conf
        )
        conf_gamma_und = conf_gamma(jac_vars["conf_metric"], inv_metric_conf)

        return bssn_to_adm(
            bssn_variables, jac_vars, hess_vars,
            inv_metric_conf, (christoffel_1st, christoffel_2nd), conf_gamma_und,
        )

    def extract_psi4(
        self,
        bssn_variables: State,
        extraction_radius: float,
        n_theta: int = 512,
        n_phi: int = 512,
    ) -> jax.Array:
        """Compute the Weyl scalar Ψ₄ SWSH coefficients at given extraction radius.

        Parameters
        ----------
        bssn_variables : dict
            Current BSSN variables.
        extraction_radius : float
            Radius at which to extract the waveform.
        n_theta, n_phi : int
            Angular resolution for the spherical interpolation grid.

        Returns
        -------
        jax.Array
            Integrated spin-weighted spherical-harmonic coefficient of Ψ₄ over the extraction sphere.
        """
        adm = self.compute_adm_variables(bssn_variables)

        weyl = weyl_tensor(
            K=bssn_variables["K"],
            adm_spatial_metric=adm["spatial_metric"],
            adm_ricci_tensor=adm["ricci_tensor"],
            extrinsic_curvature=adm["extrinsic_curvature"],
            extrinsic_curvature_jac=adm["extrinsic_curvature_jac"],
            christoffel_second_kind=adm["christoffel_second_kind"],
        )

        m_triad = build_np_m_from_coords_and_metric(
            self.coords, adm["spatial_metric"]
        )
        psi4_cart = weyl_scalar_outgoing_field(weyl, m_triad)

        x, y, z = self.coord_linspaces
        return weyl_scalar_swsh_coefficients_at_extraction_radii(
            extraction_radius, psi4_cart, (x, y, z), (n_theta, n_phi),
        )
