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
    """
    Base class for any in-situ temporal selector.

    The structure is kept simple on purpose, to allow maximum flexibility in the selection logic.
    The only requirement is that the ``decide`` method is called sequentially on each snapshot
    (e.g. at each time step of the solver), and that it updates the internal state accordingly, 
    so that the next call to ``decide`` can rely on an up-to-date internal state.

    ``decide`` is the main entry point for in-situ selection, and it calls the abstract method ``_decide``, 
    which must be implemented by subclasses to define the actual selection logic, which is a True or False decision.

    The ``compress_ratio`` method can be used to query the current temporal compression ratio.

    A list of all attributes and their meaning is provided below:
    - ``idx``: int, index of the current snapshot, an integer identifier of the snapshot.
    - ``selected_snapshots``: list of int, indices ``idx`` of the snapshots that have been selected.
    - ``physical_time``: list of float, physical time corresponding to each selected snapshot (same length as ``selected_snapshots``).
    - ``selected_num``: int, number of selected snapshots so far. Assumes it starts at 1, with the initial condition snapshot being selected.
    - ``total_num``: int, total number of snapshots processed so far. Assumes it always starts with an initial condition snapshot,
        which is obviously selected.

    If the selector needs a warmup stage, it can simply be implemented inside the decision method,
    by checking if any window of statistics or history of selected has been filled up yet, and selecting
    everything until then. Once the warmup is done, the selector can switch to the main logic.

    Parameters
    ----------
    None

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
        """Return the current temporal compression ratio."""
        if self.total_num == 0:
            return 1.0
        return self.total_num / self.selected_num
    
    def decide(self, *args, **kwargs) -> bool:
        """
        Return ``True`` if the current snapshot should be kept.

        This is the main entry point for in-situ selection, and it updates the internal state 
        (e.g. selected snapshots, counts, etc.) accordingly. This is what the user should call
        at each time step of the solver, passing in any relevant information (e.g. current field, physical time, etc.) as arguments.

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
        This is the main decision block that is used for the selection process.
        It is called by the ``decide`` method, which updates the main internal states accordingly. 
        Subclasses must implement this method to define the actual selection logic, which is a True or False decision.

        """
        pass