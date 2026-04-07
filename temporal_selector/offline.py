import jax.numpy as jnp
from collections import deque
from .base import TemporalSelector
from .metrics import _pearson_corr
from typing import Callable
import jax

def offline_enstrophy_selector(
        trajectory: jax.Array,
        queue_size: int,
        enstrophy_weight: jax.Array, # Now Mandatory
        corr_metric: Callable[[jax.Array, jax.Array], jax.Array] = _pearson_corr,
        corr_treshold: float = 0.90
    ) -> jax.Array:
        
    """

    Selects time steps where the primary criterion is the accumulation of 

    Enstrophy Flux rather than raw state deviation.

    """
    start_idx = 0
    queue = deque(maxlen=queue_size)
    T = trajectory.shape[0]
    queue.append(1)

    # NEW: maxx_diff is now based on the maximum change in enstrophy between steps

    # This identifies the most 'physically violent' transition in the current window

    def e_diff(idx1, idx2):
        return jnp.abs(enstrophy_weight[idx1] - enstrophy_weight[idx2])
    
    maxx_e_flux = e_diff(1, 0)
    curr_e_flux_sum = e_diff(1, 0)
    selected_timesteps = [0]
    last_idx = 1

    while start_idx < T - 1:

        if len(queue) == queue_size or queue[-1] == T - 1:
            # The 'budget' for how much enstrophy can change before we need a new snapshot
            avg_e_flux = curr_e_flux_sum / len(queue) # [commented for now]
            # The more 'unstable' the enstrophy (high max vs avg), the smaller the factor
            factor = jnp.sqrt(maxx_e_flux/(avg_e_flux + 1e-8)).item() # [commented for now] 

            best_idx = start_idx + 1

            # Initial enstrophy change from the last selected point

            e_diff0 = e_diff(best_idx, start_idx)

            for idx in queue:

                # 1. Physical Criterion: Has enstrophy changed too much?

                e_change = e_diff(idx, start_idx)
                # 2. Statistical Criterion: Is the field still correlated?

                corr_val = corr_metric(trajectory[idx], trajectory[start_idx])
                # We stop when the enstrophy has shifted significantly 

                # relative to the local flux 'factor'

                if e_change / (e_diff0 + 1e-8) <= factor and corr_val >= corr_treshold:

                    best_idx = idx

                else:

                    break
            start_idx = best_idx

            selected_timesteps.append(best_idx)
            # Reset queue stats for the next window

            while len(queue) > 0 and queue[0] <= start_idx:

                prev_idx = queue.popleft()

                curr_e_flux_sum -= e_diff(prev_idx, max(0, prev_idx - 1))


        if last_idx + 1 < T:

            queue.append(last_idx + 1)

            last_idx += 1

            step_flux = e_diff(queue[-1], queue[-1] - 1)

            maxx_e_flux = max(maxx_e_flux, step_flux)

            curr_e_flux_sum += step_flux

    return jnp.array(selected_timesteps)

def persistent_median_policy(t: int, 
                             T: int, 
                             activity_array: jnp.ndarray, 
                             history: deque,
                             window_size: int = 5,
                             surge_count: int = 0) -> tuple[int, int]:
    """Decide the next timestep using a persistent-median surge detector.

    Skips ahead by *window_size* during quiet baselines, but forces
    dense (step-by-step) sampling whenever the activity exceeds a
    multiplicative factor above the running median.

    Returns
    -------
    tuple[int, int]
        ``(next_timestep, updated_surge_count)``.
    """
    
    # 1. Initialization
    if len(history) < history.maxlen:
        return t + 1, 0 # Return next_t and updated surge_count

    # 2. Establish the current baseline
    # We use the median of the 'History' which only contains 'accepted' baseline values
    local_median = jnp.median(jnp.array(history))
    
    current_activity = activity_array[t+1]
    
    # 3. Decision Logic
    if current_activity > local_median * 1.55: # 20% buffer above median
        # WE ARE IN A SURGE (e.g., the merger at t=150 in your plot)
        # Force dense sampling and increment counter
        new_surge_count = surge_count + 1
        return t + 1, new_surge_count
    else:
        # WE ARE IN THE BASELINE
        # Attempt to jump
        new_surge_count = 0 # Reset because we are back to 'normal'
        best_idx = min(t + window_size, T - 1)
        
        # Check ahead to ensure we don't jump OVER the start of a merger
        for candidate_idx in range(t + 1, best_idx + 1):
            if activity_array[candidate_idx] > local_median * 1.55: # this is a median value threshold, below which frames are selected with low frequency. 
                return candidate_idx, 0 # Stop early if merger starts mid-window
                
        return best_idx, new_surge_count

def offline_bssn_selector(activity_array: jnp.ndarray, 
                          window_size : int = 5,
                          history_size: int = 20) -> jnp.ndarray:
    """Run the persistent-median policy offline on a pre-computed activity array.

    Iterates through the full activity signal, applying
    :func:`persistent_median_policy` at each step, and returns the
    indices of selected frames.  Suitable for BSSN or any scalar
    activity signal.

    Returns
    -------
    jnp.ndarray
        Integer indices of selected timesteps.
    """
    T = len(activity_array)
    t = 0
    selected = [0]
    history = deque(maxlen=history_size) # 50
    surge_count = 0
    patience_limit = window_size * 5 # 6

    while t < T - 1:
        # Get decision and updated counter
        next_t, surge_count = persistent_median_policy(
            t, T, activity_array, history, 
            window_size=window_size, surge_count=surge_count)
        
        # Update pointer
        t = int(next_t)
        selected.append(t)
        
        # LOGIC: Only update the median history if:
        # 1. We are in a 'quiet' phase (surge_count == 0)
        # 2. OR the surge has lasted so long it's the new baseline (surge_count > patience)
        if surge_count == 0 or surge_count > patience_limit:
            history.append(activity_array[t])
            if surge_count > patience_limit:
                surge_count = 0 # Reset after updating baseline
                
    return jnp.array(selected)
