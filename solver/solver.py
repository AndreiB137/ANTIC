"""Solver abstraction — wraps any user-provided PDE time-stepper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import jax.numpy as jnp


# The state can be anything the user's PDE needs: a jax Array, a dict, a
# NamedTuple, etc.  We keep it generic so the abstraction works with any PDE.
State = Any


class Solver:
    """Wraps an arbitrary PDE time-stepping function.

    The user only needs to supply a *step function* with the signature::

        next_state = step_fn(state)

    and an *extract function* that pulls the field data (a single array) out
    of ``state`` so the selector / compressor can analyse it::

        field = extract_fn(state)   # -> jnp.ndarray

    Parameters
    ----------
    step_fn : callable
        ``step_fn(state) -> state``.  Advances the PDE by one time step.
        This can come from any source: a pure-Python integrator, a JAX-CFD
        routine, a C++/pybind11 binding, etc.
    extract_fn : callable, optional
        ``extract_fn(state) -> jnp.ndarray``.  Extracts the field snapshot
        from ``state``.  If ``state`` already *is* the snapshot array you
        can leave this as ``None`` (identity is used).
    dt : float, optional
        Physical time step size (informational; not used internally).
    metadata : dict, optional
        Any extra information about the solver (PDE name, parameters, …).

    Examples
    --------
    **Minimal — state is the field itself**::

        solver = Solver(step_fn=my_rk4_step)

    **JAX-CFD style — state is a Fourier coefficient array**::

        import jax.numpy as jnp
        solver = Solver(
            step_fn=jaxcfd_step,
            extract_fn=lambda vhat: jnp.fft.irfftn(vhat, s=(N, N)),
            dt=dt,
        )

    **C++ binding**::

        solver = Solver(
            step_fn=cpp_module.advance,
            extract_fn=lambda s: jnp.array(s.field),
        )
    """

    def __init__(
        self,
        step_fn: Callable[[State], State],
        extract_fn: Optional[Callable[[State], jnp.ndarray]] = None,
        dt: Optional[float] = None,
        metadata: Optional[dict] = None,
    ):
        self._step_fn = step_fn
        self._extract_fn = extract_fn if extract_fn is not None else _identity
        self.dt = dt
        self.metadata = metadata or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, state: State) -> State:
        """Advance the PDE by one time step."""
        return self._step_fn(state)

    def extract(self, state: State) -> jnp.ndarray:
        """Extract the physical-space field from *state*."""
        return self._extract_fn(state)

    def rollout(self, state: State, n_steps: int) -> State:
        """Apply ``step`` *n_steps* times and return the final state."""
        for _ in range(n_steps):
            state = self.step(state)
        return state

    def __repr__(self) -> str:
        name = self.metadata.get("name", "custom")
        return f"Solver(name={name!r}, dt={self.dt})"


def _identity(x):
    return x
