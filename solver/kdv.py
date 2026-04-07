import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU usage for JAX
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # prevent JAX from preallocating all GPU memory

import jax
import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import json
from solver.solver import Solver


class KDVSolver(Solver):
    """KdV equation solver using pseudo-spectral method with RK4 time stepping.

    State is stored as Fourier coefficients (rfft output). The extract method
    converts back to physical space via irfft.

    Parameters
    ----------
    N : int
        Number of spatial grid points.
    systemsize : float
        Physical domain length.
    nonlinparameter : float
        Nonlinear coefficient in the KdV equation.
    dt : float
        Time step size for RK4 integration.
    """

    def __init__(self, N=512, systemsize=10.0 * jnp.pi, nonlinparameter=1.0, total_time=2.0, dt=1e-5):
        self.N = N
        self.systemsize = systemsize
        self.nonlinparameter = nonlinparameter
        self.tf = total_time
        self.inner_steps = int((total_time / dt) / 1000)
        self.coords = jnp.linspace(0.0, systemsize, N, endpoint=False)

        k_indices = jnp.arange(N // 2 + 1)
        self.k_vec = 2.0 * jnp.pi / systemsize * k_indices
        self.dt = dt
        self.elapsed_time = 0.0
        # Capture in local vars for JAX-jittable closures
        k_vec = self.k_vec
        nlp = nonlinparameter
        n = N

        def to_fourier(r):
            """Transform a real-space field to Fourier coefficients (normalised rfft)."""
            return jnp.fft.rfft(r) / n

        def to_real(c):
            """Transform Fourier coefficients back to a real-space field (inverse rfft)."""
            return jnp.fft.irfft(c * n, n=n)

        def pde_rhs(c_input):
            """Evaluate the KdV right-hand side in Fourier space."""
            r = to_real(c_input)
            c_squared = to_fourier(r * r)
            return 1j * ((k_vec ** 3) * c_input - 0.5 * nlp * k_vec * c_squared)

        @jax.jit
        def step_fn(state):
            """Advance the Fourier-space state by one RK4 time step."""
            k1 = self.pde_rhs(state)
            k2 = self.pde_rhs(state + 0.5 * self.dt * k1)
            k3 = self.pde_rhs(state + 0.5 * self.dt * k2)
            k4 = self.pde_rhs(state + self.dt * k3)
            return state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Store helpers as instance methods for external use
        self.to_fourier = to_fourier
        self.to_real = to_real
        self.pde_rhs = pde_rhs
        self.step_fn = step_fn

    def init_condition(self, velocities: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
        """Create a Fourier-space initial state from a superposition of solitons at given velocities and positions."""
        r_components = jax.vmap(self.initialize_soliton)(velocities, positions)
        r_total = jnp.sum(r_components, axis=0)
        return self.to_fourier(r_total)

    def initialize_soliton(self, velocity: float, position: float) -> jnp.ndarray:
        """Create a single KdV soliton profile centred at *position* on the periodic domain."""
        arg_base = self.coords - position
        amp_prefactor = 3.0 * velocity / self.nonlinparameter
        coeff = 0.5 * jnp.sqrt(jnp.abs(velocity))
        return amp_prefactor * (
            1.0 / jnp.cosh(coeff * arg_base) ** 2
            + 1.0 / jnp.cosh(coeff * (arg_base - self.systemsize)) ** 2
            + 1.0 / jnp.cosh(coeff * (arg_base + self.systemsize)) ** 2
        )
    
    def step(self, state: jnp.ndarray) -> jnp.ndarray:
        """Advance the KdV equation by one macro time step (200 RK4 sub-steps)."""
        self.elapsed_time += self.inner_steps * self.dt
        return self.rollout(state, n_steps=200)  # Advance by 200 steps for each frame
    
    def extract(self, state: jnp.ndarray) -> jnp.ndarray:
        """Convert Fourier-space state to a physical-space field via inverse FFT."""
        return self.to_real(state).reshape(-1, 1)

    def rollout(self, state: jnp.ndarray, n_steps: int) -> jnp.ndarray:
        """Advance the Fourier-space state by *n_steps* RK4 sub-steps using ``lax.scan``."""

        def body_fn(c, _):
            return self.step_fn(c), None

        c_final, _ = lax.scan(body_fn, state, None, length=n_steps)
        return c_final

    def prepare_coords(self) -> jnp.ndarray:
        """Prepare the coordinates for the compressor (just the spatial grid)."""
        return ((self.coords - self.coords.min()) / (self.coords.max() - self.coords.min())).reshape(-1, 1)  # Shape (N, 1) for compatibility with compressor
    
    def save_state(self, state: jnp.ndarray, directory: str):
        """Save the Fourier-space state and elapsed time on disk."""
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        jnp.save(os.path.join(directory, "state.npy"), state)
        json.dump({"elapsed_time": self.elapsed_time}, open(os.path.join(directory, "metadata.json"), "w"), indent=2)

    def load_state(self, directory: str) -> jnp.ndarray:
        """Load a previously saved Fourier-space state and elapsed time from a saved directory."""
        if os.path.exists(directory):
            with open(directory + "/metadata.json", "r") as f:
                metadata = json.load(f)
            self.elapsed_time = metadata.get("elapsed_time", 0.0)
            return jnp.load(directory + "/state.npy")
        else:
            raise FileNotFoundError(f"State file {directory} not found.")
            

# -------------------------------
# Main simulation
# -------------------------------
def run_simulation():
    """Run a four-soliton KdV simulation and save the result as an animated GIF."""
    solver = KDVSolver(N=512, systemsize=10.0 * jnp.pi, nonlinparameter=1.0, dtime=1e-5)
    PI = jnp.pi

    velocities = jnp.array([50.0, 20.0, 4.0, 0.5])
    positions  = jnp.array([2.0 * PI, 6.0 * PI, 12.0 * PI, 18.0 * PI])

    r_components = jax.vmap(solver.initialize_soliton)(velocities, positions)
    r_total = jnp.sum(r_components, axis=0)

    c_components = jax.vmap(solver.to_fourier)(r_components)
    c_total = solver.to_fourier(r_total)

    n_frames = 600
    steps_per_frame = 200

    def frame_step(state, _):
        c_arr, c_tot = state
        c_new = jax.vmap(lambda c: solver.rollout(c, steps_per_frame))(c_arr)
        c_total_new = solver.rollout(c_tot, steps_per_frame)

        frames = jax.vmap(solver.to_real)(c_new)
        total_field = solver.to_real(c_total_new)
        return (c_new, c_total_new), (frames, total_field)

    (_, _), (frames_components, frames_total) = lax.scan(
        frame_step, (c_components, c_total), None, length=n_frames
    )

    frames_components = jnp.array(frames_components)
    frames_total = jnp.array(frames_total)

    # Move to host for animation (JAX → numpy)
    frames_components_np = jax.device_get(frames_components)
    frames_total_np = jax.device_get(frames_total)
    x_np = jax.device_get(solver.x)

    # --- Matplotlib animation ---
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:red", "tab:orange", "tab:green", "tab:blue"]
    lines_individual = [
        ax.plot(x_np, frames_components_np[0, i], color=colors[i], lw=1.2, alpha=0.8, label=f"Soliton {i+1}")[0]
        for i in range(4)
    ]
    line_total, = ax.plot(x_np, frames_total_np[0], color="k", lw=2.0, label="Total field")

    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, solver.systemsize)
    ax.set_ylim(-1, 200)
    title = ax.text(0.5, 1.03, "", transform=ax.transAxes, ha="center")

    def init():
        for i in range(4):
            lines_individual[i].set_ydata(frames_components_np[0, i])
        line_total.set_ydata(frames_total_np[0])
        title.set_text("t = 0")
        return lines_individual + [line_total, title]

    def update(i):
        for j in range(4):
            lines_individual[j].set_ydata(frames_components_np[i, j])
        line_total.set_ydata(frames_total_np[i])
        title.set_text(f"t = {i}")
        return lines_individual + [line_total, title]

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=True, interval=40)
    anim.save("kdv_solitons.gif", writer=PillowWriter(fps=20), dpi=150)
    return anim


if __name__ == "__main__":
    anim = run_simulation()
