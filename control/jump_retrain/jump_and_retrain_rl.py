#!/usr/bin/env python3
"""
Jump & Retrain for RL policies (PufferLib + Walker2d-v4 / HalfCheetah-v4).

Given a trained controlled_task_<base_id>:
  - Load best_model.pt as θ0
  - Sample random directions v (L2-normalized in parameter space)
  - For each step_size and direction, construct θ_jump = θ0 + step_size * v
  - Initialize from θ_jump and continue training for extra_steps
  - Use the same PufferLib environment & config pipeline as the original training
  - Each run writes to a new controlled_task_<base>_jr<idx> directory

Note: This script does not implement PHATE/TDA; it only performs "Jump & Retrain" sampling.
"""

import sys
from pathlib import Path

# Ensure the control/ directory is on PYTHONPATH when running from subfolders
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch

# Heavy imports (pufferlib + training pipeline) are intentionally loaded lazily
# so that `--help` works even when optional compiled extensions are unavailable.
pufferl = None
LandscapeAnalyzer = None
load_config_manually = None
train_with_landscape_loop = None
load_sb3_config = None
apply_overrides = None


def _lazy_imports() -> None:
    global pufferl, LandscapeAnalyzer, load_config_manually, train_with_landscape_loop
    global load_sb3_config, apply_overrides

    if pufferl is not None:
        return

    import pufferlib.pufferl as _pufferl
    from train_landscape_controlled import (
        LandscapeAnalyzer as _LandscapeAnalyzer,
        load_config_manually as _load_config_manually,
        train_with_landscape_loop as _train_with_landscape_loop,
    )
    from train_sb3_lstm_landscape import load_sb3_config as _load_sb3_config, apply_overrides as _apply_overrides

    pufferl = _pufferl
    LandscapeAnalyzer = _LandscapeAnalyzer
    load_config_manually = _load_config_manually
    train_with_landscape_loop = _train_with_landscape_loop
    load_sb3_config = _load_sb3_config
    apply_overrides = _apply_overrides


def load_base_task_metadata(task_dir: Path) -> dict:
    """Load train_seed / init_seed / best_reward metadata from a controlled_task_x directory."""
    best_info_path = task_dir / "best_info.json"
    summary_path = task_dir / "training_summary.json"

    if not best_info_path.exists():
        raise FileNotFoundError(f"missing {best_info_path}")

    with best_info_path.open() as f:
        best_info = json.load(f)

    summary = {}
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)

    meta = {
        "env_name": summary.get("env_name", None),
        "best_reward": float(best_info.get("best_reward", float("nan"))),
        "best_step": int(best_info.get("best_step", 0)),
        "init_seed": int(best_info.get("init_seed", summary.get("init_seed", 0))),
        "train_seed": int(best_info.get("train_seed", summary.get("train_seed", 0))),
    }
    if meta["env_name"] is None:
        raise RuntimeError("Missing env_name in training_summary.json")
    return meta


def build_base_config(env_name: str,
                      config_path: Path,
                      total_steps: int,
                      train_seed: int,
                      device: str,
                      data_dir: str) -> dict:
    """
    Build a PufferLib config consistent with the original training and apply SB3 overrides.
    Note: If your original training used ppo_lstm_chatgpt.yml, set config_path to that file.
    """
    _lazy_imports()
    base_cfg = load_config_manually(env_name)
    sb3_cfg = load_sb3_config(str(config_path), env_name)
    apply_overrides(base_cfg, sb3_cfg, total_steps)

    train_cfg = base_cfg.setdefault("train", {})
    vec_cfg = base_cfg.setdefault("vec", {})

    train_cfg["env"] = env_name
    train_cfg["seed"] = train_seed
    train_cfg["device"] = device
    train_cfg["use_rnn"] = True
    train_cfg["total_timesteps"] = total_steps
    train_cfg["data_dir"] = data_dir

    vec_cfg["seed"] = train_seed
    vec_cfg.pop("overwork", None)  # backward compatibility with older configs

    return base_cfg


def flatten_params(state_dict: dict) -> Tuple[torch.Tensor, List[Tuple[str, Tuple[int, ...]]]]:
    """
    Flatten all tensor parameters in state_dict into a single vector and record (key, shape) metadata.

    Returns:
      flat_vec: [D] tensor
      metadata: [(key, shape), ...] for reconstruction
    """
    flats = []
    meta = []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        meta.append((k, v.shape))
        flats.append(v.reshape(-1))
    if not flats:
        raise RuntimeError("Empty state_dict? No tensor parameters found.")
    flat_vec = torch.cat(flats, dim=0)
    return flat_vec, meta


def unflatten_params(flat_vec: torch.Tensor,
                     meta: List[Tuple[str, Tuple[int, ...]]]) -> dict:
    """
    Reconstruct a state_dict from a flat parameter vector using metadata from flatten_params.
    Returns a new {key: tensor} dict (buffers are not included).
    """
    state_dict = {}
    idx = 0
    for k, shape in meta:
        numel = int(np.prod(shape))
        chunk = flat_vec[idx: idx + numel]
        state_dict[k] = chunk.view(*shape).clone()
        idx += numel
    return state_dict
def sample_random_direction_filter_norm(state_dict: dict,
                                        device: torch.device,
                                        rng: np.random.Generator) -> torch.Tensor:
    """
    Approximate the paper's filter-wise normalization:
    - For each parameter tensor W:
        1) Sample same-shaped noise D ~ N(0, 1)
        2) Normalize D so ||D||_2 = 1
        3) Rescale so ||D||_2 matches ||W||_2
    - Finally, flatten and concatenate all tensors into a direction vector v.
    """
    flats = []
    for k, w in state_dict.items():
        if not isinstance(w, torch.Tensor):
            continue
        w = w.to(device)
        shape = w.shape
        numel = w.numel()

        noise = rng.standard_normal(size=(numel,)).astype(np.float32)
        d = torch.from_numpy(noise).to(device).view(*shape)

        # Step 1: normalize within this tensor
        d_norm = torch.linalg.norm(d)
        if d_norm > 0:
            d = d / d_norm

        # Step 2: rescale to match the original weight norm
        w_norm = torch.linalg.norm(w)
        if w_norm > 0:
            d = d * w_norm

        flats.append(d.reshape(-1))

    if not flats:
        raise RuntimeError("No usable tensor parameters found in state_dict.")

    v = torch.cat(flats, dim=0)
    # Final L2 normalization to make step_size easier to interpret
    v = v / (torch.linalg.norm(v) + 1e-8)
    return v


def sample_random_direction_like(state_dict: dict,
                                 device: torch.device,
                                 rng: np.random.Generator) -> torch.Tensor:
    """
    Generate a random vector matching the parameter-space dimensionality and L2-normalize it.
    """
    flat0, meta = flatten_params(state_dict)
    dim = flat0.numel()
    # Sample with numpy then convert to torch for RNG control
    noise = rng.standard_normal(size=(dim,)).astype(np.float32)
    noise_t = torch.from_numpy(noise).to(device)

    norm = torch.linalg.norm(noise_t)
    if norm < 1e-8:
        raise RuntimeError("Sampled direction norm too small; sampling failed.")
    v = noise_t / norm
    return v  # shape [D]


def make_jump_state_dict(theta0: torch.Tensor,
                         direction_v: torch.Tensor,
                         step_size: float,
                         meta: List[Tuple[str, Tuple[int, ...]]]) -> dict:
    """
    Given base parameters θ0, direction v, and step_size, construct θ_jump's state_dict.
    """
    theta_jump = theta0 + step_size * direction_v
    return unflatten_params(theta_jump, meta)


def run_single_jump_and_retrain(
    base_task_id: str,
    base_task_dir: Path,
    base_meta: dict,
    base_model_path: Path,
    config_path: Path,
    env_name: str,
    device: str,
    step_size: float,
    jr_index: int,
    extra_steps: int,
    rng_seed: int,
    output_root: Path,
    models_root: Path,
):
    """
    Run Jump & Retrain for a single controlled_task_<base_task_id>:
      - Use step_size and a random direction to build θ_jump
      - Build new task id: <base_task_id>_jr<jr_index>
      - Continue training from θ_jump for extra_steps
    """
    _lazy_imports()

    train_seed = base_meta["train_seed"]
    init_seed = base_meta["init_seed"]

    run_label = f"{base_task_id}_jr{jr_index}_s{step_size:.2f}"
    new_task_id = run_label  # task id used by LandscapeAnalyzer
    output_root = Path(output_root)
    models_root = Path(models_root)
    output_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    # === 1. Load base config & policy (θ0) ===
    data_dir = output_root / f"jr_{env_name.replace('-', '_')}_{run_label}"
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    cfg = build_base_config(
        env_name=env_name,
        config_path=config_path,
        total_steps=extra_steps,
        train_seed=train_seed,
        device=device,
        data_dir=str(data_dir),
    )

    vecenv = pufferl.load_env(env_name, cfg)
    
    # Try loading normalization stats from the base model (if available)
    from train_landscape_controlled import load_vec_stats
    base_vec_stats_path = base_task_dir / "vec_stats.npz"
    if base_vec_stats_path.exists():
        if load_vec_stats(vecenv.driver_env, base_vec_stats_path):
            print(f"[J&R] Loaded base normalization stats: {base_vec_stats_path}")
        else:
            print(f"[J&R] Warning: failed to load normalization stats: {base_vec_stats_path}")
    else:
        print("[J&R] Note: base model has no vec_stats.npz; will use new normalization stats")
    
    policy = pufferl.load_policy(cfg, vecenv, env_name)

    checkpoint = torch.load(base_model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    # Strip "module." prefix
    state_dict = {k.replace("module.", ""): v.to(device) for k, v in state_dict.items()}

    # Flatten θ0
    theta0_flat, meta = flatten_params(state_dict)

    # === 2. Sample direction v and build θ_jump ===
    rng = np.random.default_rng(rng_seed)
    v = sample_random_direction_filter_norm(state_dict, device=torch.device(device), rng=rng)

    theta_jump_sd = make_jump_state_dict(theta0_flat, v, step_size, meta)

    # Load θ_jump into the policy
    # Note: only parameters are overwritten; buffers (e.g., running mean) are left as-is
    own_state = policy.state_dict()
    for k, v_param in theta_jump_sd.items():
        if k in own_state and own_state[k].shape == v_param.shape:
            own_state[k] = v_param
    policy.load_state_dict(own_state)

    # === 3. LandscapeAnalyzer & training loop ===
    analyzer = LandscapeAnalyzer()  # same analyzer as before
    print(f"[J&R] base_task={base_task_id}, new_task={new_task_id}, "
          f"step_size={step_size}, extra_steps={extra_steps}, rng_seed={rng_seed}")
    print(f"[J&R] Jumping from {base_model_path} and starting retrain ...")

    train_with_landscape_loop(
        env_name,
        cfg,
        policy,
        vecenv,
        analyzer,
        new_task_id,    # task_id / run_label
        train_seed,
        init_seed,
        models_root=str(models_root),
    )

    vecenv.close()
    print(f"[J&R] Task {new_task_id} completed.")


def main():
    parser = argparse.ArgumentParser("Jump & Retrain for PufferLib RL policies")
    parser.add_argument("--models-dir", type=Path, default=Path("models"),
                        help="Path containing controlled_task_* directories.")
    parser.add_argument("--base-task-id", required=True,
                        help="Base task id (e.g., 10 for controlled_task_10).")
    parser.add_argument("--env-name", type=str, required=True,
                        choices=["Walker2d-v4", "HalfCheetah-v4"],
                        help="Environment name (must match the base task).")
    parser.add_argument("--config", type=Path,
                        default=Path("config/ppo_lstm_chatgpt.yml"),
                        help="SB3-style PPO-LSTM config file (should match original training).")
    parser.add_argument("--step-sizes", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75, 1.0],
                        help="Jump step sizes (parameter-space L2 scaling).")
    parser.add_argument("--num-directions", type=int, default=3,
                        help="Number of random directions per step_size.")
    parser.add_argument("--extra-steps", type=int, default=500_000,
                        help="How many env steps to train per retrain run.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Training device (e.g., cpu / cuda / cuda:0).")
    parser.add_argument("--rng-seed-base", type=int, default=12345,
                        help="Base random seed for parameter-space directions.")
    parser.add_argument("--jr-index", type=int, default=None,
                        help="Manually set jr_index (for array jobs). If omitted, auto-increments from 0.")
    parser.add_argument("--checkpoint-type", choices=["best", "final"], default="best",
                        help="Choose best_model or final_model as θ0.")
    parser.add_argument("--output-root", type=Path, default=Path("results/jump_retrain_runs"),
                        help="Root directory for retrain outputs (data_dir).")
    parser.add_argument("--models-root", type=Path, default=Path("models/jump_retrain_runs"),
                        help="Root directory for retrain models.")
    args = parser.parse_args()

    base_task_dir = args.models_dir / f"controlled_task_{args.base_task_id}"
    if not base_task_dir.exists():
        raise FileNotFoundError(f"Base task directory not found: {base_task_dir}")

    base_meta = load_base_task_metadata(base_task_dir)
    if base_meta["env_name"] != args.env_name:
        raise RuntimeError(
            f"Base task env_name = {base_meta['env_name']} does not match "
            f"--env-name={args.env_name}. Please check."
        )

    # Select best_model or final_model as θ0
    if args.checkpoint_type == "final":
        base_model_path = base_task_dir / "final_model.pt"
        if not base_model_path.exists():
            raise FileNotFoundError(f"Missing final_model.pt: {base_model_path}")
    else:
        base_model_path = base_task_dir / "best_model.pt"
        if not base_model_path.exists():
            alt = base_task_dir / "final_model.pt"
            if not alt.exists():
                raise FileNotFoundError(
                    f"Neither best_model.pt nor final_model.pt found under {base_task_dir}"
                )
            base_model_path = alt

    print(f"[Base] task={args.base_task_id}, env={args.env_name}, "
          f"best_model={base_model_path.name}, "
          f"best_reward={base_meta['best_reward']:.2f}, "
          f"train_seed={base_meta['train_seed']}")

    # If jr_index is specified, use it; otherwise auto-increment from 0
    if args.jr_index is not None:
        # Single-run mode (for array jobs)
        if len(args.step_sizes) != 1 or args.num_directions != 1:
            raise ValueError("When --jr-index is set, you must provide exactly one step_size and num_directions=1")
        jr_index = args.jr_index
        step_size = args.step_sizes[0]
        rng_seed = args.rng_seed_base
        run_single_jump_and_retrain(
            base_task_id=str(args.base_task_id),
            base_task_dir=base_task_dir,
            base_meta=base_meta,
            base_model_path=base_model_path,
            config_path=args.config,
            env_name=args.env_name,
            device=args.device,
            step_size=step_size,
            jr_index=jr_index,
            extra_steps=args.extra_steps,
            rng_seed=rng_seed,
            output_root=args.output_root,
            models_root=args.models_root,
        )
    else:
        # Batch mode
        jr_index = 0
        for step_size in args.step_sizes:
            for d_i in range(args.num_directions):
                rng_seed = args.rng_seed_base + jr_index
                run_single_jump_and_retrain(
                    base_task_id=str(args.base_task_id),
                    base_task_dir=base_task_dir,
                    base_meta=base_meta,
                    base_model_path=base_model_path,
                    config_path=args.config,
                    env_name=args.env_name,
                    device=args.device,
                    step_size=step_size,
                    jr_index=jr_index,
                    extra_steps=args.extra_steps,
                    rng_seed=rng_seed,
                    output_root=args.output_root,
                    models_root=args.models_root,
                )
                jr_index += 1


if __name__ == "__main__":
    main()
