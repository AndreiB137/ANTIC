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

    Apart from the `TemporalSelector` interface, PATS requires
    the implementation of a `compute_activity` method that produces a
    physics specific activity scalar or quantity of interest from the current snapshot,
    which is the main criterion for selection. For example, this could be the total
    enstrophy for Navier-Stokes, the magnitude of the Weyl scalar Ψ₄ for BSSN.

    Subclasses can also override the `_save_state` and `_load_state` methods to store
    any additional internal state variables that are needed to resume the selector's operation
    after loading a solver checkpoint.

    """

    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def compute_activity(self, *args, **kwargs):
        """
        Compute a scalar or other quantity of interest with a specific physical meaning, usually derived
        from the spatial representation at a particular time step. This value is used to determine
        the importance of the current snapshot.

        """
        pass

    def save_state(self, path: str | Path) -> None:
        """Save internal state to *path* directory, while concatenating with any specific state from ``_save_state``."""
        common_state = {
            "physical_time": self.physical_time,
            "selected_snapshots": self.selected_snapshots,
            "idx": self.idx,
            "total_num": self.total_num,
            "selected_num": self.selected_num,
        }

        self._save_state(common_state, path)

    def load_state(self, path: str | Path) -> None:
        """Load the whole internal state from *path* directory and assign attributes to the selector instance."""
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
    
    def _save_state(self, base_dict: dict[str, Any], path: str | Path, *args, **kwargs) -> None:
        """
        Save any state specific to a subclass to *path* directory, while keeping the common state in *base_dict*.
        Subclasses with extra internal state can override this method,
        extend ``base_dict``, and write the combined payload to the same
        ``state.pkl`` file.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "state.pkl", "wb") as f:
            pickle.dump(base_dict, f)

    def _load_state(self, *args, **kwargs) -> None:
        """Load any state specific to a subclass from *path* directory.

        The common state is already restored by :meth:`load_state`.
        Subclasses only need to override this if they persist additional
        fields.
        """
        return None







