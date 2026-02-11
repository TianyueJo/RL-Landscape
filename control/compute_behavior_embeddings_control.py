#!/usr/bin/env python3
"""
Compute behavior vectors for control (MuJoCo) policies trained with PufferLib,
then reduce with PCA (pure NumPy SVD) and save embeddings in the same JSON format
as grid_world/compute_behavior_embeddings.py so we can reuse:

- grid_world/compute_embedding_distances.py (L1/L2 distance matrices)
- grid_world/plot_distance_graphs.py (mean/GMM threshold visualization + PNG export)

This script is designed for the default controlled_task_0..31 layout:
- tasks 0..15: Walker2d-v4
- tasks 16..31: HalfCheetah-v4

It will auto-split tasks by env_name from each task's training_summary.json.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Ensure the control/ directory is on PYTHONPATH when running from elsewhere
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

try:
    import pufferlib.pufferl as pufferl
    from train_landscape_controlled import load_config_manually
    from record_policy_video import load_vec_stats
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import PufferLib/control dependencies. "
        "Please run in the environment used for control/."
    ) from e


def pca_numpy(behavior_matrix: np.ndarray, dim: int) -> np.ndarray:
    """Pure NumPy PCA via SVD. Returns projected coordinates with shape (N, dim)."""
    X = np.asarray(behavior_matrix, dtype=np.float64)
    Xc = X - np.mean(X, axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:dim].T
    return (Xc @ components).astype(np.float64)


def load_policy_and_env(env_name: str, model_path: Path, device: str) -> Tuple[torch.nn.Module, object]:
    """Load a policy + vecenv (Serial, 1 env) matching existing control scripts."""
    args = load_config_manually(env_name)
    # Keep env on CPU; policy can be on cuda if requested/available
    args["train"]["device"] = device if device == "cuda" else "cpu"
    args["vec"]["backend"] = "Serial"
    args["vec"]["num_envs"] = 1
    args["vec"]["num_workers"] = 1
    args["vec"]["batch_size"] = 1

    vecenv = pufferl.load_env(env_name, args)
    policy = pufferl.load_policy(args, vecenv, env_name)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)

    policy.to(device)
    policy.eval()
    return policy, vecenv


def select_model_path(task_dir: Path, checkpoint_type: str) -> Path:
    """Pick best_model.pt (fallback final_model.pt) or force final/best."""
    best_path = task_dir / "best_model.pt"
    final_path = task_dir / "final_model.pt"
    if checkpoint_type == "best":
        return best_path if best_path.exists() else final_path
    if checkpoint_type == "final":
        return final_path
    raise ValueError(f"Unknown checkpoint_type={checkpoint_type!r}")


def task_env_name(task_dir: Path) -> str:
    """Read env_name from training_summary.json (required for auto split)."""
    summary_path = task_dir / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing training_summary.json: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    env = data.get("env_name")
    if not env:
        raise ValueError(f"Missing env_name in {summary_path}")
    return str(env)


def task_reward(task_dir: Path) -> float:
    """Best-effort reward used for node coloring (best_info.json preferred)."""
    best_info = task_dir / "best_info.json"
    if best_info.exists():
        try:
            with best_info.open("r", encoding="utf-8") as f:
                data = json.load(f)
            r = data.get("best_reward")
            if r is not None:
                return float(r)
        except Exception:
            pass
    summary = task_dir / "training_summary.json"
    if summary.exists():
        with summary.open("r", encoding="utf-8") as f:
            data = json.load(f)
        r = data.get("best_reward", 0.0)
        return float(r)
    return 0.0


def collect_representative_states(
    env_name: str,
    model_files: List[Tuple[int, Path]],
    n_episodes_per_policy: int,
    max_steps_per_episode: int,
    n_representative_states: int,
    device: str,
) -> np.ndarray:
    """
    Roll out each policy (deterministic mean action) and collect observations,
    then randomly subsample to representative states.
    Observations come from the vecenv (may be normalized depending on vec_stats).
    """
    buffer_states: List[np.ndarray] = []

    for task_id, model_path in model_files:
        policy, vecenv = load_policy_and_env(env_name, model_path, device=device)

        # Load normalization stats if available
        vec_stats_path = model_path.parent / "vec_stats.npz"
        if vec_stats_path.exists():
            try:
                load_vec_stats(vecenv, vec_stats_path)
            except Exception:
                pass

        for ep in range(n_episodes_per_policy):
            obs, _ = vecenv.reset(seed=int(task_id) * 1000 + ep)
            lstm_h, lstm_c = None, None

            for _ in range(max_steps_per_episode):
                buffer_states.append(np.asarray(obs, dtype=np.float32).copy().reshape(-1))

                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                state_dict = {"lstm_h": lstm_h, "lstm_c": lstm_c}
                with torch.no_grad():
                    action_dist, _ = policy.forward_eval(obs_tensor, state_dict)
                lstm_h = state_dict.get("lstm_h", None)
                lstm_c = state_dict.get("lstm_c", None)

                # Deterministic action (prefer mean)
                if isinstance(action_dist, (tuple, list)):
                    action_dist = action_dist[0]
                if hasattr(action_dist, "mean"):
                    action = action_dist.mean
                elif hasattr(action_dist, "sample"):
                    action = action_dist.sample()
                else:
                    action = action_dist

                obs, _, terminated, truncated, _ = vecenv.step(action.detach().cpu().numpy())
                if bool(terminated[0]) or bool(truncated[0]):
                    break

        try:
            vecenv.close()
        except Exception:
            pass

    if not buffer_states:
        raise RuntimeError("No representative states collected.")

    if len(buffer_states) > n_representative_states:
        idx = random.sample(range(len(buffer_states)), n_representative_states)
        rep = np.array([buffer_states[i] for i in idx], dtype=np.float32)
    else:
        rep = np.array(buffer_states, dtype=np.float32)
    return rep


def behavior_vector_on_states(policy, representative_states: np.ndarray, device: str) -> np.ndarray:
    """
    For each representative state, run policy.forward_eval with zeroed LSTM state
    and extract distribution parameters. Concatenate per-state features.
    """
    feats: List[np.ndarray] = []

    # Try batching for speed; keep a conservative batch size to avoid GPU/CPU OOM.
    batch_size = 256
    policy_device = next(policy.parameters()).device

    for start in range(0, representative_states.shape[0], batch_size):
        batch = representative_states[start : start + batch_size]
        obs_tensor = torch.as_tensor(batch, dtype=torch.float32, device=policy_device)
        # Zero LSTM state for all states (treat them independently)
        state_dict = {"lstm_h": None, "lstm_c": None}
        with torch.no_grad():
            action_dist, _ = policy.forward_eval(obs_tensor, state_dict)
        if isinstance(action_dist, (tuple, list)):
            action_dist = action_dist[0]

        # Continuous action distributions (commonly have mean + log_std)
        if hasattr(action_dist, "mean") and hasattr(action_dist, "log_std"):
            mean = action_dist.mean.detach().cpu().numpy()
            log_std = action_dist.log_std.detach().cpu().numpy()
            feats.append(np.concatenate([mean, log_std], axis=-1).reshape(batch.shape[0], -1))
        elif hasattr(action_dist, "mean") and hasattr(action_dist, "stddev"):
            mean = action_dist.mean.detach().cpu().numpy()
            std = action_dist.stddev.detach().cpu().numpy()
            feats.append(np.concatenate([mean, std], axis=-1).reshape(batch.shape[0], -1))
        elif hasattr(action_dist, "mean"):
            mean = action_dist.mean.detach().cpu().numpy()
            feats.append(mean.reshape(batch.shape[0], -1))
        elif hasattr(action_dist, "probs"):
            probs = action_dist.probs.detach().cpu().numpy()
            feats.append(probs.reshape(batch.shape[0], -1))
        elif hasattr(action_dist, "logits"):
            logits = action_dist.logits.detach().cpu().numpy()
            feats.append(logits.reshape(batch.shape[0], -1))
        else:
            # Last resort: sample action as feature
            sample = action_dist.sample().detach().cpu().numpy()
            feats.append(sample.reshape(batch.shape[0], -1))

    mat = np.concatenate(feats, axis=0)  # (S, F_state)
    return mat.reshape(-1)  # flatten to (S*F_state,)


def save_embeddings_json(
    output_dir: Path,
    method: str,
    dim: int,
    case_id: int,
    seeds: List[int],
    model_paths: List[str],
    embeddings: np.ndarray,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{method}_dim{dim}_case{case_id}.json"
    payload = {
        "case_id": int(case_id),
        "method": str(method),
        "dim": int(dim),
        "seeds": [int(s) for s in seeds],
        "model_paths": list(model_paths),
        "embeddings": [[float(x) for x in row] for row in np.asarray(embeddings)],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return json_path


def save_eval_json(output_dir: Path, case_to_seed_returns: Dict[int, Dict[int, float]]) -> Path:
    """
    Write eval JSON in the format expected by grid_world/plot_distance_graphs.py:
    { "case_1": {"seeds_data": {"seed_0": {"mean_return": ...}, ...}}, ... }
    """
    out = {}
    for case_id, seed_map in case_to_seed_returns.items():
        out[f"case_{case_id}"] = {
            "seeds_data": {f"seed_{seed}": {"mean_return": float(ret)} for seed, ret in seed_map.items()}
        }
    path = output_dir / "evaluation_results_control.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute behavior embeddings (PCA) for control/ controlled_task_* policies."
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Models root directory.")
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=list(range(32)),
        help="Task IDs to include (default: 0..31).",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[2, 3, 6, 9, 12],
        help="PCA output dimensions to compute.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("behavior_space_embeddings_control"), help="Output directory.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for policy inference.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--checkpoint-type", type=str, default="best", choices=["best", "final"], help="Which checkpoint to load.")
    parser.add_argument("--n-episodes-per-policy", type=int, default=10, help="Episodes per policy for state sampling.")
    parser.add_argument("--max-steps-per-episode", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--n-representative-states", type=int, default=1000, help="Representative states to sample.")
    parser.add_argument(
        "--reuse-representative-states",
        action="store_true",
        default=True,
        help="Reuse saved representative_states_caseX.npy if present (default: enabled).",
    )
    parser.add_argument(
        "--no-reuse-representative-states",
        action="store_false",
        dest="reuse_representative_states",
        help="Disable reuse of saved representative states and resample instead.",
    )
    parser.add_argument(
        "--case-map",
        type=str,
        default="auto",
        help="Case mapping mode: 'auto' (split by env_name) or 'single' (all tasks same env; requires --env-name).",
    )
    parser.add_argument("--env-name", type=str, default=None, help="Env name when using --case-map single.")
    args = parser.parse_args()

    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    models_dir: Path = args.models_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build cases: case_id -> (env_name, task_ids)
    case_to_tasks: Dict[int, Tuple[str, List[int]]] = {}
    if args.case_map == "single":
        if not args.env_name:
            raise ValueError("--env-name is required when --case-map single")
        case_to_tasks[1] = (args.env_name, list(args.task_ids))
    elif args.case_map == "auto":
        env_to_tasks: Dict[str, List[int]] = {}
        for tid in args.task_ids:
            task_dir = models_dir / f"controlled_task_{tid}"
            env = task_env_name(task_dir)
            env_to_tasks.setdefault(env, []).append(int(tid))
        # Stable case ids for common envs
        preferred = ["Walker2d-v4", "HalfCheetah-v4"]
        ordered_envs = [e for e in preferred if e in env_to_tasks] + sorted(
            [e for e in env_to_tasks.keys() if e not in preferred]
        )
        for i, env in enumerate(ordered_envs, start=1):
            case_to_tasks[i] = (env, sorted(env_to_tasks[env]))
    else:
        raise ValueError("--case-map must be 'auto' or 'single'")

    case_to_seed_returns: Dict[int, Dict[int, float]] = {}

    for case_id, (env_name, task_ids) in case_to_tasks.items():
        print(f"\n=== [Case {case_id}] env={env_name} tasks={task_ids[:3]}... total={len(task_ids)} ===")

        model_files: List[Tuple[int, Path]] = []
        seed_returns: Dict[int, float] = {}
        for tid in task_ids:
            task_dir = models_dir / f"controlled_task_{tid}"
            model_path = select_model_path(task_dir, args.checkpoint_type)
            if not model_path.exists():
                raise FileNotFoundError(f"Missing model for task {tid}: {model_path}")
            model_files.append((int(tid), model_path))
            seed_returns[int(tid)] = task_reward(task_dir)
        case_to_seed_returns[case_id] = seed_returns

        rep_path = output_dir / f"representative_states_case{case_id}.npy"
        if args.reuse_representative_states and rep_path.exists():
            representative_states = np.load(rep_path)
            print(f"[Reuse] {rep_path} shape={representative_states.shape}")
        else:
            representative_states = collect_representative_states(
                env_name=env_name,
                model_files=model_files,
                n_episodes_per_policy=args.n_episodes_per_policy,
                max_steps_per_episode=args.max_steps_per_episode,
                n_representative_states=args.n_representative_states,
                device=args.device,
            )
            np.save(rep_path, representative_states)
            print(f"[Saved] {rep_path} shape={representative_states.shape}")

        # Compute behavior vectors for all tasks in this case
        behavior_vectors: List[np.ndarray] = []
        seeds: List[int] = []
        model_paths: List[str] = []

        for tid, model_path in model_files:
            print(f"  [Behavior] task={tid} model={model_path.name}")
            policy, vecenv = load_policy_and_env(env_name, model_path, device=args.device)
            # Load vec_stats if available (important for consistent inference)
            vec_stats_path = model_path.parent / "vec_stats.npz"
            if vec_stats_path.exists():
                try:
                    load_vec_stats(vecenv, vec_stats_path)
                except Exception:
                    pass
            vec = behavior_vector_on_states(policy, representative_states, device=args.device)
            behavior_vectors.append(vec.astype(np.float64, copy=False))
            seeds.append(int(tid))
            model_paths.append(str(model_path))
            try:
                vecenv.close()
            except Exception:
                pass

        behavior_matrix = np.vstack(behavior_vectors)  # (K, F)
        print(f"[OK] Behavior matrix case{case_id} shape={behavior_matrix.shape}")

        # Save PCA embeddings for requested dims
        for dim in args.dims:
            if dim > min(behavior_matrix.shape[0], behavior_matrix.shape[1]):
                raise ValueError(
                    f"Requested dim={dim} is too large for behavior_matrix shape={behavior_matrix.shape}"
                )
            emb = pca_numpy(behavior_matrix, dim=dim)
            json_path = save_embeddings_json(
                output_dir=output_dir,
                method="pca",
                dim=int(dim),
                case_id=int(case_id),
                seeds=seeds,
                model_paths=model_paths,
                embeddings=emb,
            )
            print(f"[Saved] {json_path}")

    eval_path = save_eval_json(output_dir, case_to_seed_returns)
    print(f"\n[Saved] {eval_path}")


if __name__ == "__main__":
    main()


