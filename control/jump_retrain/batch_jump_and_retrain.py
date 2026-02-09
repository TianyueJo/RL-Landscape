#!/usr/bin/env python3
"""
Run Jump & Retrain in batch for Walker2d-v4 / HalfCheetah-v4 final models.
Iterate controlled_task_0-31 in the order: HalfCheetah first, then Walker,
and run a fixed number of random-direction retrains per task.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from jump_and_retrain_rl import (
    load_base_task_metadata,
    run_single_jump_and_retrain,
)

ALLOWED_ENVS = ("HalfCheetah-v4", "Walker2d-v4")


def discover_tasks(models_dir: Path) -> List[Tuple[str, int, Path]]:
    tasks: List[Tuple[str, int, Path]] = []
    for summary_path in sorted(models_dir.glob("controlled_task_*/training_summary.json")):
        task_dir = summary_path.parent
        task_id_str = task_dir.name.split("_")[-1]
        if not task_id_str.isdigit():
            continue
        task_id = int(task_id_str)
        with summary_path.open() as f:
            summary = json.load(f)
        env_name = summary.get("env_name")
        if env_name not in ALLOWED_ENVS:
            continue
        tasks.append((env_name, task_id, task_dir))
    # HalfCheetah first, then Walker; within each env sort by task id ascending
    tasks.sort(key=lambda item: (0 if item[0] == "HalfCheetah-v4" else 1, item[1]))
    return tasks


def main():
    parser = argparse.ArgumentParser("Batch Jump & Retrain for all controlled_task_* models")
    parser.add_argument("--models-dir", type=Path, default=Path("models"),
                        help="Root directory containing controlled_task_* subfolders")
    parser.add_argument("--config", type=Path, default=Path("config/ppo_lstm_chatgpt.yml"),
                        help="SB3-style PPO-LSTM config file")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Training device (e.g., cpu / cuda / cuda:0)")
    parser.add_argument("--step-size", type=float, default=0.5,
                        help="Jump step size (parameter-space L2 scaling)")
    parser.add_argument("--num-directions", type=int, default=5,
                        help="Number of random directions per task")
    parser.add_argument("--extra-steps", type=int, default=500_000,
                        help="Env steps per retrain run")
    parser.add_argument("--rng-seed-base", type=int, default=12345,
                        help="Global base seed for random directions")
    parser.add_argument("--checkpoint-type", choices=["final", "best"], default="final",
                        help="Use final_model or best_model as θ0 (default: final)")
    parser.add_argument("--output-root", type=Path, default=Path("results/jump_retrain_batch"),
                        help="Root directory for retrain outputs (data_dir)")
    parser.add_argument("--models-root", type=Path, default=Path("models/jump_retrain_batch"),
                        help="Root directory for retrain models")
    args = parser.parse_args()

    models_dir = args.models_dir.resolve()
    tasks = discover_tasks(models_dir)
    if not tasks:
        print("[Info] No HalfCheetah/Walker controlled_task_* found.")
        return

    print(f"[Batch J&R] Planning to process {len(tasks)} tasks; {args.num_directions} directions per task.")
    global_run_idx = 0

    for env_name, task_id, task_dir in tasks:
        base_meta = load_base_task_metadata(task_dir)
        if base_meta["env_name"] != env_name:
            print(f"[Skip] task {task_id}: env mismatch in training summary ({base_meta['env_name']} vs {env_name})")
            continue

        if args.checkpoint_type == "final":
            ckpt_path = task_dir / "final_model.pt"
        else:
            ckpt_path = task_dir / "best_model.pt"
        if not ckpt_path.exists():
            print(f"[Skip] {task_dir.name}: missing {ckpt_path.name}")
            continue

        print(f"\n========== Task {task_id} | {env_name} | Model {ckpt_path.name} ==========")
        for direction_idx in range(args.num_directions):
            rng_seed = args.rng_seed_base + global_run_idx
            run_single_jump_and_retrain(
                base_task_id=str(task_id),
                base_task_dir=task_dir,
                base_meta=base_meta,
                base_model_path=ckpt_path,
                config_path=args.config,
                env_name=env_name,
                device=args.device,
                step_size=args.step_size,
                jr_index=direction_idx,
                extra_steps=args.extra_steps,
                rng_seed=rng_seed,
                output_root=args.output_root,
                models_root=args.models_root,
            )
            global_run_idx += 1


if __name__ == "__main__":
    main()

