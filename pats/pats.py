"""Physics-Aware Temporal Selectors (PATS).

PATS is an abstract base class for selectors that leverage
domain-specific physical scalars (enstrophy, Weyl invariants,
conserved-quantity deviations, …) to decide which PDE snapshots
are worth keeping.

Concrete selectors:
    EnstrophySelector     – Navier-Stokes enstrophy-flux selector.
    SurgeDetectorSelector – median-baseline + surge detection
                            (e.g. BSSN Weyl scalar |Ψ₄|).
    KdVActivitySelector   – KdV |u_t| activity selector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Optional

import jax.numpy as jnp
import numpy as np

from ..temporal_selector import _pearson_corr
from ..solver import Solver, State


# ======================================================================
# Physics helpers
# ======================================================================

def compute_enstrophy(vorticity_field: jnp.ndarray) -> jnp.ndarray:
    """Mean squared vorticity per frame.  ``(T, NX, NY) -> (T,)``."""
    return jnp.mean(vorticity_field ** 2, axis=tuple(range(1, vorticity_field.ndim)))



# ======================================================================
# PATS — abstract base
# ======================================================================


class PATS(ABC):
    """Abstract base for physics-aware temporal selectors.

    Subclasses provide a domain-specific *activity scalar* (e.g. an
    enstrophy, a Weyl-scalar magnitude, a conserved-quantity
    deviation) or physical quantity of interest, and a *decision policy* that uses that quantity's
    history to decide whether a snapshot is important.

    The base class provides both **in-situ** (wrapping a
    :class:`Solver`) and **offline** (operating on a pre-computed
    trajectory) entry points.

    To create a custom physics-aware selector, subclass ``PATS`` and
    implement :meth:`compute_activity` and :meth:`decide`.

    Parameters
    ----------
    warmup : int
        Number of initial snapshots to always keep (to bootstrap
        statistics).
    on_keep : callable, optional
        ``on_keep(timestep, field, state)`` — fired on every kept
        snapshot.
    on_skip : callable, optional
        ``on_skip(timestep, field, state)`` — fired on every skipped
        snapshot.
    """

    def __init__(
        self,
        warmup: int = 1,
        on_keep: Optional[Callable] = None,
        on_skip: Optional[Callable] = None,
    ):
        self.warmup = warmup
        self.on_keep = on_keep
        self.on_skip = on_skip

        self._activity_history: list[float] = []
        self._kept_indices: list[int] = []
        self._ref_field: jnp.ndarray | None = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_activity(self, field: jnp.ndarray) -> float:
        """Compute a physics-specific activity scalar from *field*.

        The returned value summarises *how active* the current
        snapshot is — e.g. total enstrophy, Weyl-scalar magnitude,
        ``|u_t|`` norm, etc.
        """
        ...

    @abstractmethod
    def decide(self, timestep: int) -> bool:
        """Decide whether the snapshot at *timestep* should be kept.

        When called, :attr:`_activity_history` already contains the
        activity for the current timestep (appended just before this
        call).  Subclasses can inspect the full history to implement
        any stateful policy (running mean, surge detection, …).
        """
        ...

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear internal state for a new run."""
        self._activity_history.clear()
        self._kept_indices.clear()
        self._ref_field = None

    @property
    def kept_indices(self) -> list[int]:
        return list(self._kept_indices)

    @property
    def activity_history(self) -> list[float]:
        return list(self._activity_history)

    def run(
        self,
        solver: Solver,
        state: State,
        n_steps: int,
        start_timestep: int = 0,
    ) -> tuple[State, list[int]]:
        """Drive *solver* for *n_steps*, selecting snapshots.

        Parameters
        ----------
        solver : Solver
            PDE solver.
        state : State
            Initial state.
        n_steps : int
            Number of solver steps.
        start_timestep : int
            Logical offset.

        Returns
        -------
        state : State
            Final PDE state.
        kept : list[int]
            Selected timestep indices.
        """
        self.reset()

        # Always keep timestep 0
        field = solver.extract(state)
        self._ref_field = field
        a = float(self.compute_activity(field))
        self._activity_history.append(a)
        self._kept_indices.append(start_timestep)
        if self.on_keep is not None:
            self.on_keep(start_timestep, field, state)

        for i in range(n_steps):
            t = start_timestep + i + 1
            state = solver.step(state)
            field = solver.extract(state)

            a = float(self.compute_activity(field))
            self._activity_history.append(a)

            if len(self._kept_indices) < self.warmup or self.decide(t):
                self._ref_field = field
                self._kept_indices.append(t)
                if self.on_keep is not None:
                    self.on_keep(t, field, state)
            else:
                if self.on_skip is not None:
                    self.on_skip(t, field, state)

        return state, list(self._kept_indices)

    def select_offline(
        self,
        trajectory: jnp.ndarray | np.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Select keyframes from a pre-computed trajectory.

        Calls :meth:`compute_activity` on every frame, then runs
        :meth:`decide` to select keyframes.

        Returns ``(selected_indices, activity_per_frame)``.
        """
        self.reset()
        trajectory = jnp.asarray(trajectory)
        T = trajectory.shape[0]
        flat = trajectory.reshape(T, -1)

        self._ref_field = flat[0]
        a0 = float(self.compute_activity(flat[0]))
        self._activity_history.append(a0)
        self._kept_indices.append(0)

        for t in range(1, T):
            a = float(self.compute_activity(flat[t]))
            self._activity_history.append(a)

            if len(self._kept_indices) < self.warmup or self.decide(t):
                self._ref_field = flat[t]
                self._kept_indices.append(t)

        return (
            jnp.array(self._kept_indices, dtype=jnp.int32),
            jnp.array(self._activity_history),
        )







