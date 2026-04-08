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

    The user needs to supply a *step function* with the signature::

        next_state = step_fn(state)

    and an *extract function* that pulls the field data (a single array) out
    of ``state`` so the selector or compressor can work with it::

        field = extract_fn(state)   # -> jnp.ndarray

    The solver abstraction is designed to be as flexible as possible, so the
    user can use any PDE, any time-stepping method, and any state structure they like.
    
    In addition, the user should define a ``rollout`` method that advances the state by multiple time steps, 
    a ``prepare_coords`` method that prepares the coordinates for the neural compressor, either normalized
    or in their original form. Finally, the user should implement ``save_state`` and ``load_state`` methods 
    to allow saving and loading the solver state to disk, which is necessary for checkpointing.

    Examples
    --------
    ::
    
        solver = MySolver(...)
        selector = MySelector(...)
        state = solver.load_state(...) / solver.init(...) / solver.init_condition(...)
        while True:
            state = solver.step(state)
            field = solver.extract(state)
            keep = selector.step(field or state):
            
            if not keep:
                continue  # Skip compression this snapshot
                
            # Compress and save the snapshot here
            compress(field, ...)

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
