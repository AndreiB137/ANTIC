from __future__ import annotations

import json
from pathlib import Path
import os

import jax
import jax.numpy as jnp
from flax import nnx
import pickle

from configs import ExperimentConfig
from configs.utils import (
    build_model,
    build_normalization,
    build_optimizer,
    build_selector,
    build_solver,
    compute_decay_steps,
)
from neural_compressor import NeuralFieldCompressor
from neural_compressor.utils import FilterSpec, resolve_filter
from nnx_models.lora.utils_lora import (
    add_lora_to_model,
    merge_lora_params,
    remove_lora_from_model,
    reset_lora_params,
)
from param_manager import ParamManager


def run_bssn(cfg: ExperimentConfig, run_checkpoint: bool = False):
    """Run the BSSN ANTIC compression experiment."""

    # ---- wandb --------------------------------------------------
    if cfg.wandb.enabled:
        import wandb
        if os.path.exists(cfg.training.save_dir + "/checkpoint"):
            with open(cfg.training.save_dir + "/checkpoint/info.json", "r") as f:
                info = json.load(f)
            wandb_info = info["wandb"]
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.name,
                tags=cfg.wandb.tags,
                id=wandb_info["id"],
                config=cfg.model_dump(),
                resume="must",
            )
            print(f"[wandb] Resumed run: {wandb.run.id}  previous_stop_at={info['stop_at']}")
        else:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.name,
                tags=cfg.wandb.tags,
                config=cfg.model_dump(),
            )


    # ---- Param Manager ------------------------------------------

    param_manager = ParamManager(
        root_dir = cfg.training.save_dir
    )

    if run_checkpoint:
        with open(cfg.training.save_dir + "/checkpoint/info.json", "r") as f:
            info = json.load(f)
        last_full_save = info.get("last_full_save")
        param_manager.last_full_save = last_full_save

    # ---- solver -------------------------------------------------
    solver = build_solver(cfg.solver)
    print(f"[solver] bssn  N={solver.grid.n}  domain_width={solver.grid.domain_width} dt={solver.evolution.dt}")

    # ---- model --------------------------------------------------

    norm_stats = None
    if run_checkpoint:
        nf_compressor = param_manager.load(id=info["snapshot_id"])
        model = nf_compressor.model
        if cfg.normalization.enabled:
            norm_stats = build_normalization(cfg.normalization)
            norm_stats.load_stats(cfg.training.save_dir + "/checkpoint/norm_stats.pkl")
    else:
        model = build_model(cfg.model, seed=cfg.seed)
        if cfg.normalization.enabled:
            norm_stats = build_normalization(cfg.normalization)

    print(f"[model]  {cfg.model.name}  hidden_dim={cfg.model.hidden_dim}  "
          f"layers={cfg.model.num_hidden_layers}")

    # ---- compressor ---------------------------------------------
    if not run_checkpoint:
        nf_compressor = NeuralFieldCompressor(model)

    # ---- coordinates --------------------------------------------
    coords = solver.prepare_coords()

    # ---- initial condition or load checkpoint --------------------------
    if run_checkpoint:
        state = solver.load_state(
            cfg.training.save_dir + "/checkpoint/solver"
        )
    else:
        state = solver.load_state(
            cfg.solver.initial_data_path
        )

    # ---- selector -----------------------------------------------
    selector = build_selector(cfg.selector)

    if cfg.selector.type != "none":
        if run_checkpoint:
            selector.load_state(cfg.training.save_dir + "/checkpoint/selector")

    # Determine which variables the optimizer should track
    nnx_filter = resolve_filter(cfg.training.filter)

    # Train on the initial snapshot (snapshot 0)
    batch_size = cfg.training.batch_size
    decay_steps_init = compute_decay_steps(coords, batch_size, cfg.training.initial_epochs)
    decay_steps_after = compute_decay_steps(coords, batch_size, cfg.training.subsequent_epochs)
    opt_tx_init = build_optimizer(cfg.optimizer, cfg.optimizer.scheduler, decay_steps_init)
    opt_tx_after = build_optimizer(cfg.optimizer, cfg.optimizer.scheduler, decay_steps_after)

    @nnx.jit(static_argnames=["optim", "wrt"])
    def build_nnx_opt(model: nnx.Module, optim, wrt):
        optimizer = nnx.Optimizer(model, optim, wrt=wrt)
        return optimizer

    if not run_checkpoint:

        target = solver.extract(state)

        if cfg.normalization.enabled:
            norm_stats.update(target)
            target = norm_stats.normalize(target)
            nf_compressor.norm_stats = norm_stats.stats()

        optimizer = build_nnx_opt(nf_compressor.model, opt_tx_init, wrt=nnx.Param)

        loss = nf_compressor.compress(
            optimizer=optimizer,
            coords=coords,
            target=target,
            epochs=cfg.training.initial_epochs,
            batch_size=batch_size,
            num_devices=cfg.training.num_devices,
            verbose=cfg.training.verbose,
        )

        param_manager.save(
            nf_compressor,
            idx=0,
            elapsed_time=solver.elapsed_time,
            filter='all',
        )

        print(f"[Snapshot {selector.idx:>4d}]  t={0.0:.4e}  loss={float(loss):.4e}  (initial)")

        if cfg.wandb.enabled:
            wandb.log({"snapshot": selector.idx, "loss": float(loss), "physical_time": 0.0})
    
    # ---- LoRA setup ---------------------------------------------
    if cfg.training.filter == 'lora':
        add_lora_to_model(model, rank=cfg.training.rank, rngs=nnx.Rngs(cfg.seed))
        print(f"[lora]   rank={cfg.training.rank}  reset_every_n={cfg.training.reset_every_n}")

    while True:
        # Advance PDE
        state = solver.step(state)

        keep = selector.decide(state, solver)

        if not keep:
            continue

        # Extract target field for compression
        jac_target = None
        if cfg.training.use_jac:
            target, jac_target = solver.extract(state, cfg.training.use_jac)
        else:
            target = solver.extract(state)

        # ---- LoRA periodic reset (merge + remove + retrain from scratch) ----
        if cfg.training.filter == 'lora' and cfg.training.reset_every_n > 0:
            if selector.selected_num % cfg.training.reset_every_n == 0:
                print(f"[lora]   reset_every_n reached ({selector.selected_num}): merging & removing LoRA, retraining from scratch")
                remove_lora_from_model(nf_compressor.model, merge=True)
                nnx_filter = nnx.Param

        if cfg.normalization.enabled:
            norm_stats.update(target)
            target = norm_stats.normalize(target)
            nf_compressor.norm_stats = norm_stats.stats()

        optimizer = build_nnx_opt(nf_compressor.model, opt_tx_after, wrt=nnx_filter)

        loss = nf_compressor.compress(
            optimizer=optimizer,
            coords=coords,
            target=target,
            jac_target=jac_target,
            epochs=cfg.training.subsequent_epochs,
            batch_size=batch_size,
            num_devices=cfg.training.num_devices,
            verbose=cfg.training.verbose,
        )   

        # ---- NaN debugging: reset compressor if NaN detected ----
        if jnp.isnan(loss):
            
            print(f"[warn]   NaN detected at snapshot {selector.idx} (t={solver.elapsed_time:.6f}). "
                  "Removing LoRA adapters and resetting parameters.")
            
            if cfg.training.filter == 'lora':
                if len(nnx.state(nf_compressor.model, nnx.LoRAParam)) > 0:
                    remove_lora_from_model(nf_compressor.model, merge=False)

            nf_compressor.model = build_model(cfg.model, seed=cfg.seed)  # Rebuild the base model to reset parameters

            nnx_filter = nnx.Param

            optimizer = build_nnx_opt(nf_compressor.model, opt_tx_after, wrt=nnx_filter)
            loss = nf_compressor.compress(
                optimizer=optimizer,
                coords=coords,
                target=target,
                jac_target=jac_target,
                epochs=cfg.training.subsequent_epochs,
                batch_size=batch_size,
                num_devices=cfg.training.num_devices,
                verbose=cfg.training.verbose,
            )

        save_filter = cfg.training.filter
        if nnx_filter == nnx.Param:
            save_filter = "all"

        param_manager.save(
            nf_compressor,
            idx=selector.selected_num - 1,
            elapsed_time=solver.elapsed_time,
            filter=save_filter,
            overwrite=True
        )

        log_msg = f"[Snapshot {selector.idx:>4d}]  t={solver.elapsed_time:.6f}  loss={float(loss):.4e}, temporal_compression_ratio = {selector.compress_ratio():.4f}"
        log_msg += f" , neural_compression_ratio={len(target) / nf_compressor.count_params(nnx_filter):.2f}"
        print(log_msg)

        target = norm_stats.denormalize(target) if cfg.normalization.enabled else target

        extra_metrics = nf_compressor.compute_extra_metrics(coords, target, batch_size=batch_size)
        extra_metrics_str = ", ".join([f"{k}={v:.4e}" for k, v in extra_metrics.items()])
        print(f"[extra metrics]: {extra_metrics_str}")

        if nnx_filter != nnx.LoRAParam:
            if cfg.training.filter == 'lora':
                add_lora_to_model(nf_compressor.model, rank=cfg.training.rank, rngs=nnx.Rngs(cfg.seed))

        if cfg.wandb.enabled and selector.selected_num % cfg.wandb.log_every == 0:
            wandb.log({"snapshot": selector.idx, 
                       "loss": float(loss),
                       "physical_time": solver.elapsed_time, 
                       "selected_count": selector.selected_num,
                       "temporal_compression_ratio": selector.compress_ratio(),
                       "neural_compression_ratio": len(target) / nf_compressor.count_params(nnx_filter),
                       **extra_metrics,
                       })
        

        nnx_filter = resolve_filter(cfg.training.filter)  # recompute nnx_filter in case filter is dynamic (e.g. LoRA reset)
        # Check physical time limit
        if cfg.training.stop_at != 'inf' and solver.elapsed_time >= cfg.training.stop_at:
            # Save state so user can resume later
            ckpt_dir = cfg.training.save_dir + "/checkpoint"
            os.makedirs(ckpt_dir, exist_ok=True)
            solver.save_state(state, ckpt_dir + "/solver")
            selector.save_state(ckpt_dir + "/selector")
            if cfg.normalization.enabled and norm_stats is not None:
                norm_stats.save_stats(ckpt_dir + "/norm_stats.pkl")
            info_save = {
                "stop_at": cfg.training.stop_at,
                "elapsed_time": solver.elapsed_time,
                "snapshot_id": selector.selected_num - 1,
                "last_full_save": param_manager.last_full_save,
                "wandb": {
                    "id": wandb.run.id if cfg.wandb.enabled else None
                }
            }
            with open(ckpt_dir + "/info.json", "w") as f:
                json.dump(info_save, f, indent=2)
            print(f"[checkpoint] saved at t={solver.elapsed_time:.6f}")
            break
        
        if solver.elapsed_time >= cfg.solver.evolution.total_time:
            print(f"[done] Reached end of solver integration at t={solver.elapsed_time:.6f}")
            break

    # ---- cleanup ------------------------------------------------
    if cfg.wandb.enabled:
        wandb.finish()

    param_manager.checkpointer.close()

    print(f"[done]  {selector.selected_num} snapshots saved, t_final={solver.elapsed_time:.6f}")

