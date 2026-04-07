"""Base class for all temporal selectors.

Every temporal selector in the library inherits from
:class:`TemporalSelector`, which provides both **in-situ** (one field
at a time) and **offline** (full trajectory array) entry points —
without depending on any solver.

Subclasses must implement :meth:`_decide`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional

import jax.numpy as jnp
import numpy as np


class TemporalSelector(ABC):
    """Solver-independent base for temporal snapshot selection.

    Subclasses implement :meth:`_decide` to express their selection
    logic. 

    Parameters
    ----------
    """

    def __init__(
        self,
    ):
        self.idx = 0
        self.selected_snapshots : list[int] = [0]
        self.physical_time : list[float] = [0.0]
        self.selected_num : int = 1
        self.total_num: int = 1

    def compress_ratio(self) -> float:
        """Return the current compression ratio."""
        if self.total_num == 0:
            return 1.0
        return self.total_num / self.selected_num
    
    def decide(self, *args, **kwargs) -> bool:
        """
        Return ``True`` if the current snapshot should be kept.

        Implementations can rely on ``self._ref_field`` (last kept
        snapshot) and ``self._kept_indices`` being up-to-date when
        this is called.
        """
        self.total_num += 1
        decision = self._decide(*args, **kwargs)
        self.idx += 1
        if decision:
            self.selected_num += 1
            self.selected_snapshots.append(self.idx)
        return decision

    @abstractmethod
    def _decide(self, *args, **kwargs) -> bool:
        """
        Return ``True`` if *field* at *timestep* should be kept.

        Implementations can rely on ``self._ref_field`` (last kept
        snapshot) and ``self._kept_indices`` being up-to-date when
        this is called.
        """
        pass