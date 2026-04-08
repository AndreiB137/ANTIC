"""ANTIC — main entry point.

Usage
-----
    python main.py --config configs/kdv.yaml
    python main.py --config configs/kolmogorov.yaml
    python main.py --config configs/bssn.yaml
    python main.py --config configs/kdv.yaml --override solver.N=1024 training.initial_epochs=1000
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
os.environ["JAX_PLATFORMS"] = "cpu"

from configs import ExperimentConfig
from experiments import run_kdv, run_kolmogorov

try:
    from experiments import run_bssn
except ImportError:
    run_bssn = None

def run(cfg: ExperimentConfig, run_checkpoint: bool = False):
    """Run the experiment corresponding to the solver type."""
    solver_name = cfg.solver.name
    if solver_name == "kdv":
        run_kdv(cfg, run_checkpoint)
    elif solver_name == "kolmogorov":
        run_kolmogorov(cfg, run_checkpoint)
    elif solver_name == "bssn":
        run_bssn(cfg, run_checkpoint)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


def _apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """Apply dot-separated key=value overrides to the raw dict."""
    for ov in overrides:
        key, _, value = ov.partition("=")
        if not value:
            raise ValueError(f"Override must be key=value, got: {ov!r}")
        parts = key.split(".")
        d = raw
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        # Try to cast to int/float/bool
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                continue
        else:
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.lower() == "null":
                value = None
        d[parts[-1]] = value
    return raw


def _resolve_solver_config(raw: dict, config_path: str | Path) -> dict:
    """Resolve solver shorthand or merge solver defaults with inline overrides."""
    import yaml

    config_path = Path(config_path)
    solver_val = raw.get("solver")
    solver_name = None
    inline_solver = {}

    if isinstance(solver_val, str):
        solver_name = solver_val
    elif isinstance(solver_val, dict) and isinstance(solver_val.get("name"), str):
        solver_name = solver_val["name"]
        inline_solver = {k: v for k, v in solver_val.items() if k != "name"}

    if solver_name is None:
        return raw

    solver_cfg_path = config_path.parent / "solver" / f"{solver_name}.yaml"
    with open(solver_cfg_path) as f:
        solver_raw = yaml.safe_load(f) or {}

    solver_raw.update(inline_solver)
    solver_raw["name"] = solver_name
    raw["solver"] = solver_raw
    return raw


def main():
    parser = argparse.ArgumentParser(description="ANTIC — Adaptive Neural Temporal In-situ Compressor")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Dot-separated overrides, e.g. solver.N=1024 training.initial_epochs=1000")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        raw = yaml.safe_load(f)

    raw = _resolve_solver_config(raw, args.config)

    if args.override:
        raw = _apply_overrides(raw, args.override)

    cfg = ExperimentConfig(**raw)

    save_dir = Path(cfg.training.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "train_config.yaml", "w") as f:
        yaml.safe_dump(cfg.model_dump(mode="json", exclude={"solver"}), f, sort_keys=False)

    with open(save_dir / "solver_config.yaml", "w") as f:
        yaml.safe_dump(cfg.solver.model_dump(mode="json"), f, sort_keys=False)


    print(f"[config] Loaded {args.config}")
    run_checkpoint = False
    if os.path.exists(save_dir / "checkpoint"):
        print(f"[checkpoint] Checkpoint found in {save_dir / 'checkpoint'}, resuming training.")
        run_checkpoint = True

    run(cfg, run_checkpoint)


if __name__ == "__main__":
    main()
