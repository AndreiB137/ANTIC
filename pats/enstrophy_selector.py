
from .pats import PATS

from typing import Any, Callable
import jax.numpy as jnp
import numpy as np
from collections import deque
from ..temporal_selector import _pearson_corr

class EnstrophySelector(PATS):
    """Physics-aware selector for Navier-Stokes based on enstrophy flux.

    Uses enstrophy change + Pearson correlation to decide whether two
    snapshots are close enough to skip one.

    Parameters
    ----------
    queue_size : int
        Look-ahead window size.
    corr_threshold : float
        Pearson correlation below which a snapshot is forced to be
        kept.
    warmup : int
        Initial frames always kept.
    """

    def __init__(
        self,
        queue_size: int = 5,
        corr_threshold: float = 0.90,
        warmup: int = 2,
        **kwargs: Any,
    ):
        super().__init__(warmup=warmup, **kwargs)
        self.queue_size = queue_size
        self.corr_threshold = corr_threshold

    def compute_activity(self, field: jnp.ndarray) -> float:
        """Enstrophy = mean(field²)."""
        return jnp.mean(field ** 2)

    def decide(self, timestep: int) -> bool:
        """Keep if enstrophy flux exceeds the local adaptive factor."""
        h = self._activity_history
        if len(h) < 3:
            return True

        diffs = [abs(h[i] - h[i - 1]) for i in range(max(1, len(h) - self.queue_size), len(h))]
        avg_flux = np.mean(diffs) if diffs else 1e-12
        max_flux = np.max(diffs) if diffs else 1e-12
        factor = np.sqrt(max_flux / (avg_flux + 1e-8))

        last_kept_idx = self._kept_indices[-1]
        e_change = abs(h[timestep] - h[last_kept_idx])
        e_base = abs(h[min(last_kept_idx + 1, timestep)] - h[last_kept_idx])

        if e_base < 1e-12:
            return True

        return (e_change / (e_base + 1e-8)) > factor

    def select_with_enstrophy(
        self,
        trajectory: jnp.ndarray,
        enstrophy: jnp.ndarray,
        corr_metric: Callable = _pearson_corr,
    ) -> jnp.ndarray:
        """Full enstrophy-flux selection with correlation, matching the
        original ``temporal_selection_enstrophy_flux`` algorithm.

        Parameters
        ----------
        trajectory : shape ``(T, ...)``
        enstrophy : shape ``(T,)``, pre-computed enstrophy per frame.
        corr_metric : correlation function.

        Returns
        -------
        jnp.ndarray — selected timestep indices.
        """
        T = trajectory.shape[0]
        flat = trajectory.reshape(T, -1)
        queue: deque[int] = deque(maxlen=self.queue_size)
        queue.append(1)

        def e_diff(i: int, j: int) -> float:
            return jnp.abs(enstrophy[i] - enstrophy[j])

        maxx_e_flux = e_diff(1, 0)
        curr_e_flux_sum = e_diff(1, 0)
        selected = [0]
        start_idx = 0
        last_idx = 1

        while start_idx < T - 1:
            if len(queue) == self.queue_size or queue[-1] == T - 1:
                avg_e = curr_e_flux_sum / len(queue)
                factor = jnp.sqrt(maxx_e_flux / (avg_e + 1e-8))
                best_idx = start_idx + 1
                e_diff0 = e_diff(best_idx, start_idx)

                for idx in queue:
                    e_change = e_diff(idx, start_idx)
                    corr = corr_metric(flat[idx], flat[start_idx])
                    if (e_change / (e_diff0 + 1e-8)) <= factor and corr >= self.corr_threshold:
                        best_idx = idx
                    else:
                        break

                start_idx = best_idx
                selected.append(best_idx)

                while len(queue) > 0 and queue[0] <= start_idx:
                    prev = queue.popleft()
                    curr_e_flux_sum -= e_diff(prev, max(0, prev - 1))

            if last_idx + 1 < T:
                queue.append(last_idx + 1)
                last_idx += 1
                step_flux = e_diff(queue[-1], queue[-1] - 1)
                maxx_e_flux = max(maxx_e_flux, step_flux)
                curr_e_flux_sum += step_flux

        return jnp.array(selected, dtype=jnp.int32)