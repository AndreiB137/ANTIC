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

from abc import abstractmethod
from collections import deque
from pathlib import Path
import pickle
from typing import Any, Callable, Optional

import jax.numpy as jnp
import numpy as np

from temporal_selector import _pearson_corr
from temporal_selector.base import TemporalSelector
from solver.solver import Solver

# ======================================================================
# Physics helpers
# ======================================================================

def compute_enstrophy(vorticity_field: jnp.ndarray) -> jnp.ndarray:
    """Mean squared vorticity per frame.  ``(T, NX, NY) -> (T,)``."""
    return jnp.mean(vorticity_field ** 2, axis=tuple(range(1, vorticity_field.ndim)))



# ======================================================================
# PATS — abstract base
# ======================================================================


class PATS(TemporalSelector):
    """Abstract base for physics-aware temporal selectors.

    Subclasses provide a domain-specific *activity scalar* (e.g. an
    enstrophy, a Weyl-scalar magnitude, a conserved-quantity
    deviation) or physical quantity of interest, and a *decision policy*
    that uses that quantity's history to decide whether a snapshot is
    important.

    Inherits from :class:`TemporalSelector` so it supports both
    **in-situ** (via :meth:`step` / :meth:`run`) and **offline**
    (via :meth:`select`) workflows with no solver dependency.

    To create a custom physics-aware selector, subclass ``PATS`` and
    implement :meth:`compute_activity` and :meth:`decide`.

    Parameters
    ----------
    warmup : int
        Number of initial snapshots to always keep (to bootstrap
        statistics).
    on_keep : callable, optional
        ``on_keep(timestep, field)`` — fired on every kept snapshot.
    on_skip : callable, optional
        ``on_skip(timestep, field)`` — fired on every skipped snapshot.
    """

    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def compute_activity(self, *args, **kwargs) -> float:
        """Compute a physics-specific activity scalar from *field*.

        The returned value summarises *how active* the current
        snapshot is — e.g. total enstrophy, Weyl-scalar magnitude,
        ``|u_t|`` norm, etc.
        """
        pass

    def save_state(self, path: str | Path) -> None:
        """Save selector type, constructor args, and internal state to *path* directory."""
        common_state = {
            "physical_time": self.physical_time,
            "selected_snapshots": self.selected_snapshots,
            "idx": self.idx,
            "total_num": self.total_num,
            "selected_num": self.selected_num,
        }

        self._save_state(common_state, path)

    def load_state(self, path: str | Path) -> None:
        """Load internal state from *path* directory."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Selector state directory {path} not found.")
        with open(path / "state.pkl", "rb") as f:
            state = pickle.load(f)
        self.physical_time = state["physical_time"]
        self.selected_snapshots = state["selected_snapshots"]
        self.idx = state["idx"]
        self.total_num = state["total_num"]
        self.selected_num = state["selected_num"]

        self._load_state(path)
    
    @abstractmethod
    def _save_state(self, base_dict: dict[str, Any], path: str | Path, *args, **kwargs) -> None:
        """Save selector type, constructor args, and internal state to *path* directory.

        The directory will contain at minimum a ``metadata.json`` with
        the selector class name and the constructor kwargs needed to
        re-instantiate it.
        """
        pass

    @abstractmethod
    def _load_state(self, *args, **kwargs) -> None:
        """Load internal state from *path* directory.

        The directory will contain a ``metadata.json`` with the selector
        class name and constructor kwargs, as well as any additional
        files needed to restore the internal state (e.g. activity
        history queues).
        """
        pass







