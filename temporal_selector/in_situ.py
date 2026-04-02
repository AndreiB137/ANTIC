"""Generic temporal selectors — distance-based snapshot selection.

Classes:
    InSituSelector  – wraps a Solver; decides on-the-fly after every
                      step whether to keep or skip the snapshot.
    OfflineSelector – selects keyframes from a pre-computed trajectory
                      using threshold or momentum methods.

For information-theoretic selectors see :mod:`.information`.
For physics-aware selectors (PATS) see :mod:`.pats`.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import jax.numpy as jnp
import numpy as np

from .metrics import resolve_metric
from ..solver import Solver, State

class InSituSelector:
    """Wraps a :class:`Solver` to select snapshots on-the-fly.

    After every ``step_fn`` call the selector evaluates a distance
    metric between the current snapshot and the last *kept* snapshot.
    If the distance exceeds a (possibly adaptive) threshold the
    snapshot is kept; otherwise it is skipped.

    Parameters
    ----------
    solver : Solver
        The PDE solver whose ``step`` output is monitored.
    metric : str or callable
        Distance function ``metric(field_a, field_b) -> scalar``.
        Pass a string for a built-in preset (``"max_abs"``, ``"l2"``,
        ``"mae"``, ``"mse"``, ``"pearson"``) or any callable with the
        same signature.
    threshold : float
        Static threshold — a snapshot is kept when
        ``metric(current, reference) > threshold``.
        Ignored when *adaptive* is ``True``.
    adaptive : bool
        When ``True``, the threshold is computed as
        ``mean + k * std`` over a rolling window of recent distances
        (momentum-style selector).
    window_size : int
        Rolling-window length used when *adaptive* is ``True``.
    k : float
        Number of standard deviations above the mean for the adaptive
        threshold (only used when *adaptive* is ``True``).
    keep_first_n : int
        Always keep the first *keep_first_n* snapshots (used to warm
        up statistics for the adaptive mode).
    on_keep : callable, optional
        ``on_keep(timestep, field, state)`` — called whenever a
        snapshot is kept.  Use this to feed the snapshot into a
        compressor, write it to disk, etc.
    on_skip : callable, optional
        ``on_skip(timestep, field, state)`` — called for skipped
        snapshots (useful for logging).

    Examples
    --------
    ::

        selector = InSituSelector(
            solver,
            metric="max_abs",
            adaptive=True,
            window_size=10,
            on_keep=lambda t, f, s: compressor.compress(f, timestep=t),
        )
        state, kept = selector.run(initial_state, n_steps=1000)
    """

    def __init__(
        self,
        solver: Solver,
        metric: str | Callable = "max_abs",
        threshold: float = 1e-3,
        adaptive: bool = False,
        window_size: int = 10,
        k: float = 1.0,
        keep_first_n: int | None = None,
        on_keep: Optional[Callable] = None,
        on_skip: Optional[Callable] = None,
    ):
        self.solver = solver
        self._metric = resolve_metric(metric)
        self.threshold = threshold
        self.adaptive = adaptive
        self.window_size = window_size
        self.k = k
        self.keep_first_n = keep_first_n if keep_first_n is not None else (
            window_size if adaptive else 1
        )
        self.on_keep = on_keep
        self.on_skip = on_skip

        self._history: list[float] = []
        self._kept_indices: list[int] = []
        self._ref_field: jnp.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, state: State, timestep: int) -> tuple[State, bool]:
        """Advance the solver by one step and decide keep / skip.

        Returns ``(new_state, kept)`` where *kept* is ``True`` when
        the snapshot was selected.
        """
        state = self.solver.step(state)
        field = self.solver.extract(state)

        kept = self._decide(field, timestep)

        if kept:
            self._ref_field = field
            self._kept_indices.append(timestep)
            if self.on_keep is not None:
                self.on_keep(timestep, field, state)
        else:
            if self.on_skip is not None:
                self.on_skip(timestep, field, state)

        return state, kept

    def run(
        self,
        state: State,
        n_steps: int,
        start_timestep: int = 0,
    ) -> tuple[State, list[int]]:
        """Run the solver for *n_steps*, selecting snapshots along the way.

        Parameters
        ----------
        state : State
            Initial PDE state.
        n_steps : int
            Total number of solver steps.
        start_timestep : int
            Logical timestep offset (useful for restarts).

        Returns
        -------
        state : State
            Final PDE state.
        kept_indices : list[int]
            Timestep indices that were kept.
        """
        self.reset()

        # Always keep the initial snapshot (timestep 0)
        field = self.solver.extract(state)
        self._ref_field = field
        self._kept_indices.append(start_timestep)
        if self.on_keep is not None:
            self.on_keep(start_timestep, field, state)

        for i in range(n_steps):
            timestep = start_timestep + i + 1
            state, _ = self.step(state, timestep)

        return state, list(self._kept_indices)

    def reset(self) -> None:
        """Clear internal bookkeeping (call before a new run)."""
        self._history.clear()
        self._kept_indices.clear()
        self._ref_field = None

    @property
    def kept_indices(self) -> list[int]:
        return list(self._kept_indices)

    @property
    def distances(self) -> list[float]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decide(self, field: jnp.ndarray, timestep: int) -> bool:
        """Return ``True`` if *field* should be kept."""
        if len(self._kept_indices) < self.keep_first_n:
            return True

        dist = self._metric(field, self._ref_field)
        self._history.append(dist)

        if self.adaptive:
            recent = self._history[-self.window_size:]
            mu = np.mean(recent)      
            sigma = np.std(recent)
            return dist > mu + self.k * sigma

        return dist > self.threshold

    def __repr__(self) -> str:
        mode = "adaptive" if self.adaptive else f"threshold={self.threshold}"
        return f"InSituSelector({mode}, metric={self._metric.__name__!r})"
