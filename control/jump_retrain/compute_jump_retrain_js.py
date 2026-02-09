#!/usr/bin/env python3
"""
Compute Jensen-Shannon (JS) divergence between the final model of a single
HalfCheetah controlled_task and its Jump & Retrain models.

Depends on utility functions from compute_policy_similarity_matrix.py:
  - load_policy
  - sample_states
  - compute_average_js
"""

import sys
from pathlib import Path

# Ensure the control/ directory is on PYTHONPATH when running from subfolders
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from typing import List

from compute_policy_similarity_matrix import (
    load_policy,
    sample_states,
    compute_average_js,
)


def format_step(step: float) -> str:
    """Format step size to match retrain directory naming (2 decimal places)."""
    return f"{step:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute JS divergence between a final model and its retrain models."
    )
    parser.add_argument("--env-name", type=str, default="HalfCheetah-v4")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument(
        "--base-models-dir",
        type=Path,
        default=Path("models"),
        help="Root directory containing controlled_task_* folders (where final_model.pt lives).",
    )
    parser.add_argument(
        "--retrain-models-dir",
        type=Path,
        default=Path("models/jump_retrain_hc_step_sweep"),
        help="Root directory containing Jump & Retrain models.",
    )
    parser.add_argument(
        "--step-sizes",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 15.0, 45.0, 135.0],
        help="Step sizes to compare; should match s{step:.2f} in retrain directories.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of states to sample for JS estimation.",
    )
    parser.add_argument(
        "--states-seed",
        type=int,
        default=12345,
        help="Random seed used when sampling states.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/jump_retrain_js"),
        help="Output directory for JSON results.",
    )
    return parser.parse_args()


def build_retrain_model_path(
    retrain_root: Path, task_id: int, jr_index: int, step_size: float
) -> Path:
    """Build retrain model path (final_model.pt) for a given task/step/jr_index."""
    label = f"controlled_task_{task_id}_jr{jr_index}_s{format_step(step_size)}"
    return retrain_root / label / "final_model.pt"


def main() -> None:
    args = parse_args()
    step_sizes: List[float] = list(args.step_sizes)

    # === Load base final model ===
    base_model_path = (
        args.base_models_dir / f"controlled_task_{args.task_id}" / "final_model.pt"
    )
    if not base_model_path.exists():
        raise FileNotFoundError(f"Missing final_model.pt: {base_model_path}")

    final_policy, final_vecenv = load_policy(args.env_name, str(base_model_path))
    states = sample_states(
        final_vecenv, final_policy, num_samples=args.num_samples, seed=args.states_seed
    )
    final_vecenv.close()

    results = {
        "task_id": args.task_id,
        "env_name": args.env_name,
        "base_model": str(base_model_path),
        "num_states": int(states.shape[0]),
        "state_shape": list(states.shape),
        "step_results": [],
    }

    for jr_index, step in enumerate(step_sizes):
        model_path = build_retrain_model_path(
            args.retrain_models_dir, args.task_id, jr_index, step
        )
        if not model_path.exists():
            print(f"[Warning] Retrain model not found: {model_path}")
            continue

        retrain_policy, retrain_vecenv = load_policy(
            args.env_name, str(model_path)
        )
        js_value = compute_average_js(final_policy, retrain_policy, states)
        retrain_vecenv.close()

        results["step_results"].append(
            {
                "step_size": step,
                "jr_index": jr_index,
                "model_path": str(model_path),
                "js_divergence": float(js_value),
            }
        )
        print(
            f"[Result] task {args.task_id}, step {step:g}: JS = {js_value:.6f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"task_{args.task_id:02d}_js.json"
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()


