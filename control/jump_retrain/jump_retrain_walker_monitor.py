#!/usr/bin/env python3
"""
Run Walker2d-v4 Jump & Retrain and monitor episode reward.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_jump_retrain_with_monitoring(
    base_task_id: int,
    extra_steps: int = 2_000_000,
    step_sizes: list = [0.50],
    num_directions: int = 5,
    device: str = "cuda",
    models_dir: Path = Path("models"),
    output_root: Path = Path("results/jump_retrain_runs"),
    models_root: Path = Path("models/jump_retrain_all"),
    config: Path = Path("config/ppo_lstm_chatgpt.yml"),
):
    """
    Run jump-and-retrain and monitor the training process.
    """
    script_path = Path(__file__).parent / "jump_and_retrain_rl.py"
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        "--base-task-id", str(base_task_id),
        "--env-name", "Walker2d-v4",
        "--extra-steps", str(extra_steps),
        "--step-sizes"] + [str(s) for s in step_sizes] + [
        "--num-directions", str(num_directions),
        "--device", device,
        "--models-dir", str(models_dir),
        "--output-root", str(output_root),
        "--models-root", str(models_root),
        "--config", str(config),
    ]
    
    print("=" * 70)
    print(f"Running Walker2d-v4 Jump & Retrain (Task {base_task_id})")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print(f"Training steps: {extra_steps:,} steps")
    print(f"Step sizes: {step_sizes}")
    print(f"Directions per step_size: {num_directions}")
    print("=" * 70)
    print()
    
    # Run command
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # stream output
            text=True,
        )
        print()
        print("=" * 70)
        print(f"Task {base_task_id} completed!")
        print("=" * 70)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 70)
        print(f"Task {base_task_id} failed!")
        print("=" * 70)
        return False


def main():
    parser = argparse.ArgumentParser("Walker2d-v4 Jump & Retrain with Monitoring")
    parser.add_argument("--base-task-id", type=int, required=True,
                        help="Base task id (0-15)")
    parser.add_argument("--extra-steps", type=int, default=2_000_000,
                        help="Env steps per retrain run (default: 2M)")
    parser.add_argument("--step-sizes", type=float, nargs="+", default=[0.50],
                        help="Jump step sizes (default: [0.50])")
    parser.add_argument("--num-directions", type=int, default=5,
                        help="Random directions per step_size (default: 5)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Training device (default: cuda)")
    parser.add_argument("--models-dir", type=Path, default=Path("models"),
                        help="Directory containing base models")
    parser.add_argument("--output-root", type=Path, default=Path("results/jump_retrain_runs"),
                        help="Root directory for retrain outputs")
    parser.add_argument("--models-root", type=Path, default=Path("models/jump_retrain_all"),
                        help="Root directory for retrain models")
    parser.add_argument("--config", type=Path, default=Path("config/ppo_lstm_chatgpt.yml"),
                        help="Config file path")
    
    args = parser.parse_args()
    
    success = run_jump_retrain_with_monitoring(
        base_task_id=args.base_task_id,
        extra_steps=args.extra_steps,
        step_sizes=args.step_sizes,
        num_directions=args.num_directions,
        device=args.device,
        models_dir=args.models_dir,
        output_root=args.output_root,
        models_root=args.models_root,
        config=args.config,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()














