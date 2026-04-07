"""BSSN (numerical relativity) experiment for ANTIC.

Runs an in-situ compression loop: advance the BSSN solver one time-step at
a time, apply the persistent-median surge detector (PATS), and train the
neural field compressor on selected snapshots.

Supports LoRA fine-tuning with periodic merge/reset, NaN debugging via JAX,
and early stopping based on a user-specified maximum physical time.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from configs import ExperimentConfig
from configs.utils import (
    build_model,
    build_optimizer,
    build_selector,
    build_solver,
)
from neural_compressor import NeuralFieldCompressor
from nnx_models.lora.utils_lora import (
    add_lora_to_model,
    merge_lora_params,
    remove_lora_from_model,
    reset_lora_params,
)


def run_bssn(cfg: ExperimentConfig):
    """Run the full BSSN ANTIC compression experiment."""

    # ---- wandb --------------------------------------------------
    if cfg.wandb.enabled:
        import wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            config=cfg.model_dump(),
        )

    # ---- solver -------------------------------------------------
    solver = build_solver(cfg)
    print(f"[solver] bssn  n={cfg.solver.grid.n}  dt={solver.dt}")

    # ---- model --------------------------------------------------
    model = build_model(cfg)
    print(f"[model]  {cfg.model.name.value}  hidden_dim={cfg.model.hidden_dim}  "
          f"layers={cfg.model.num_hidden_layers}")

    # ---- compressor ---------------------------------------------
    compressor = NeuralFieldCompressor(model)

    # ---- load initial data / state ------------------------------
    state = solver.load_state(cfg.solver.initial_data_path)

    # ---- coordinates: 3-D grid (flattened) ----------------------
    # solver.grid has shape (n, n, n, 3) — flatten to (n³, 3)
    grid_flat = solver.grid.reshape(-1, 3)
    coords = grid_flat

    # ---- selector -----------------------------------------------
    selector = build_selector(cfg)

    # ---- LoRA setup ---------------------------------------------
    if cfg.lora.enabled:
        add_lora_to_model(model, rank=cfg.lora.rank, rngs=nnx.Rngs(cfg.seed))
        print(f"[lora]   rank={cfg.lora.rank}  reset_every_n={cfg.lora.reset_every_n}")

    # ---- checkpoint directory -----------------------------------
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_config = {
        "model": {"name": cfg.model.name.value, **cfg.model.build_kwargs()},
    }
    if cfg.lora.enabled:
        train_config["model"]["rank"] = cfg.lora.rank
    with open(ckpt_dir / "train_config.json", "w") as f:
        json.dump(train_config, f, indent=2)

    # ---- training loop ------------------------------------------
    physical_time = solver.elapsed_time
    snap_idx = 0
    selected_count = 0
    batch_size = cfg.training.batch_size

    # Determine which variables the optimizer should track
    wrt = nnx.LoRAParam if cfg.lora.enabled and cfg.training.filter.value == "lora" else nnx.Param

    # Train on the initial snapshot
    initial_snapshot = solver.extract(state)
    # Flatten the BSSN snapshot to (n_points, n_components)
    if initial_snapshot.ndim > 2:
        n_components = initial_snapshot.shape[0]
        target = initial_snapshot.reshape(n_components, -1).T  # (n³, n_components)
    elif initial_snapshot.ndim == 1:
        target = initial_snapshot[:, None]
    else:
        target = initial_snapshot

    epochs = cfg.training.initial_epochs
    decay_steps = _compute_decay_steps(coords, batch_size, epochs)
    opt_tx = build_optimizer(cfg, decay_steps)
    optimizer = nnx.Optimizer(model, opt_tx, wrt=wrt)

    loss = compressor.compress(
        optimizer=optimizer,
        coords=coords,
        target=target,
        epochs=epochs,
        batch_size=batch_size,
        filter=cfg.training.filter.value if cfg.lora.enabled else "all",
        verbose=cfg.training.verbose,
    )
    print(f"[snap {snap_idx:>4d}]  t={physical_time:.6f}  loss={float(loss):.4e}  (initial)")

    if cfg.wandb.enabled:
        import wandb
        wandb.log({"snapshot": snap_idx, "loss": float(loss), "physical_time": physical_time})

    snap_idx += 1

    # ---- in-situ stepping loop ----------------------------------
    while True:
        state = solver.step(state)
        physical_time = solver.elapsed_time

        # Check physical time limit
        if cfg.training.max_physical_time is not None and physical_time >= cfg.training.max_physical_time:
            solver.save_state(state, ckpt_dir / "solver_states")
            if selector is not None:
                selector.save_state(ckpt_dir / "selector_state")
            print(f"[stop]   physical_time={physical_time:.6f} >= max_physical_time={cfg.training.max_physical_time}")
            break

        # Run selector (BSSN median selector needs bssn_variables + solver)
        if selector is not None:
            keep = selector.decide(state, solver)
            if not keep:
                continue

        selected_count += 1

        # Extract and reshape for compressor
        snapshot = solver.extract(state)
        if snapshot.ndim > 2:
            n_components = snapshot.shape[0]
            target = snapshot.reshape(n_components, -1).T
        elif snapshot.ndim == 1:
            target = snapshot[:, None]
        else:
            target = snapshot

        # ---- LoRA periodic reset ----
        if cfg.lora.enabled and cfg.lora.reset_every_n > 0:
            if selected_count % cfg.lora.reset_every_n == 0 and selected_count > 0:
                print(f"[lora]   reset_every_n reached ({selected_count}): merging & removing LoRA, retraining from scratch")
                merge_lora_params(model)
                remove_lora_from_model(model, merge=False)
                add_lora_to_model(model, rank=cfg.lora.rank, rngs=nnx.Rngs(cfg.seed + selected_count))

        # Build fresh optimizer for this snapshot
        epochs = cfg.training.subsequent_epochs
        decay_steps = _compute_decay_steps(coords, batch_size, epochs)
        opt_tx = build_optimizer(cfg, decay_steps)
        optimizer = nnx.Optimizer(model, opt_tx, wrt=wrt)

        loss = compressor.compress(
            optimizer=optimizer,
            coords=coords,
            target=target,
            epochs=epochs,
            batch_size=batch_size,
            filter=cfg.training.filter.value if cfg.lora.enabled else "all",
            verbose=cfg.training.verbose,
        )

        # ---- NaN debugging ----
        if _check_nan(model):
            print(f"[warn]   NaN detected at snap {snap_idx} (t={physical_time:.6f}). "
                  "Removing LoRA adapters and resetting parameters.")
            if cfg.lora.enabled:
                remove_lora_from_model(model, merge=False)
                add_lora_to_model(model, rank=cfg.lora.rank, rngs=nnx.Rngs(cfg.seed + snap_idx))
            decay_steps = _compute_decay_steps(coords, batch_size, epochs)
            opt_tx = build_optimizer(cfg, decay_steps)
            optimizer = nnx.Optimizer(model, opt_tx, wrt=wrt)
            loss = compressor.compress(
                optimizer=optimizer,
                coords=coords,
                target=target,
                epochs=epochs,
                batch_size=batch_size,
                filter=cfg.training.filter.value if cfg.lora.enabled else "all",
                verbose=cfg.training.verbose,
            )

        log_msg = f"[snap {snap_idx:>4d}]  t={physical_time:.6f}  loss={float(loss):.4e}  selected={selected_count}"
        print(log_msg)

        if cfg.wandb.enabled and snap_idx % cfg.wandb.log_every == 0:
            import wandb
            wandb.log({"snapshot": snap_idx, "loss": float(loss),
                       "physical_time": physical_time, "selected_count": selected_count})

        # LoRA merge + reset for next snapshot
        if cfg.lora.enabled:
            merge_lora_params(model)
            reset_lora_params(model, rank=cfg.lora.rank)

        snap_idx += 1

    # ---- cleanup ------------------------------------------------
    if cfg.wandb.enabled:
        import wandb
        wandb.finish()

    print(f"[done]  {snap_idx} snapshots saved, {selected_count} selected, t_final={physical_time:.6f}")
