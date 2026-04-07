"""Solver abstraction — wraps any user-provided PDE time-stepper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import jax.numpy as jnp


# The state can be anything the user's PDE needs: a jax Array, a dict, a
# NamedTuple, etc.  We keep it generic so the abstraction works with any PDE.
State = Any


class Solver(ABC):
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

    @abstractmethod
    def step(self, *args, **kwargs) -> State:
        """Advance the PDE by one time step and return the new state."""
        pass

    @abstractmethod
    def extract(self, *args, **kwargs) -> jnp.ndarray:
        """Extract the physical-space field snapshot from the current state."""
        pass

    @abstractmethod
    def rollout(self, *args, **kwargs) -> State:
        """Advance the state by multiple time steps and return the final state."""
        pass

    @abstractmethod
    def prepare_coords(self, *args, **kwargs) -> jnp.ndarray:
        """Prepare the coordinates for the compressor."""
        pass

    @abstractmethod
    def save_state(self, *args, **kwargs):
        """Save the current state to disk."""
        pass

    @abstractmethod
    def load_state(self, *args, **kwargs) -> State:
        """Load a state from disk."""
        pass
    

def _identity(x):
    """Identity function used as a default extract_fn."""
    return x
