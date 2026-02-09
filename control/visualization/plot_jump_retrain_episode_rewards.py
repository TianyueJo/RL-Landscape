#!/usr/bin/env python3
"""
Plot episode returns for HalfCheetah Jump & Retrain step sweep results:
 - X-axis: controlled_task id (16~31)
 - Y-axis: episode return of each retrain run
 - Point color: num_steps recorded in feature files (gradient 0~500k)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

FILENAME_REGEX = re.compile(
    r"^features_task_(?P<label>\d+_jr\d+_s(?P<step>[\d\.]+))_step_(?P<idx>\d+)\.json$"
)
LABEL_REGEX = re.compile(r"^(?P<task>\d+)_jr(?P<jr>\d+)_s(?P<step>[\d\.]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Jump & Retrain episode rewards for multiple step sizes."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("landscape_data"),
        help="Root directory containing features_task_*.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("landscape_plots/jump_retrain_step_sweep"),
        help="Directory to save figures",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=list(range(16, 32)),
        help="HalfCheetah controlled_task ids to plot",
    )
    parser.add_argument(
        "--step-sizes",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 15.0, 45.0, 135.0],
        help="Step sizes to plot",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500_000,
        help="Max steps for color normalization (default: 500k)",
    )
    return parser.parse_args()


def _match_step(value: float, targets: List[float], tol: float = 1e-3) -> Optional[float]:
    for target in targets:
        if abs(value - target) <= tol:
            return target
    return None


def load_records(
    data_dir: Path, task_ids: List[int], step_sizes: List[float]
) -> Dict[float, List[Dict]]:
    records: Dict[float, List[Dict]] = {s: [] for s in step_sizes}

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for path in data_dir.glob("features_task_*_step_*.json"):
        match = FILENAME_REGEX.match(path.name)
        if not match:
            continue
        label = match.group("label")
        label_match = LABEL_REGEX.match(label)
        if not label_match:
            continue

        task_id = int(label_match.group("task"))
        if task_id not in task_ids:
            continue

        raw_step = float(label_match.group("step"))
        step_size = _match_step(raw_step, step_sizes)
        if step_size is None:
            continue

        with path.open() as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        episode_reward = data.get("episode_reward")
        num_steps = data.get("num_steps")
        if episode_reward is None or num_steps is None:
            continue

        records[step_size].append(
            {
                "task_id": task_id,
                "num_steps": int(num_steps),
                "episode_reward": float(episode_reward),
                "file": path,
            }
        )

    return records


def make_plots_by_step_size(
    records: Dict[float, List[Dict]],
    task_ids: List[int],
    output_dir: Path,
    max_steps: int,
) -> None:
    """
    One figure per step size: x=task id, y=episode return.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=max_steps)

    x_positions = {task_id: idx for idx, task_id in enumerate(sorted(task_ids))}
    xticks = [x_positions[t] for t in sorted(task_ids)]
    xtick_labels = [str(t) for t in sorted(task_ids)]

    for step_size, items in records.items():
        if not items:
            print(f"[Warning] step size={step_size} has no valid data; skipping.")
            continue

        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        xs = [x_positions[item["task_id"]] for item in items]
        ys = [item["episode_reward"] for item in items]
        colors = [cmap(norm(min(item["num_steps"], max_steps))) for item in items]

        ax.scatter(xs, ys, c=colors, s=40, edgecolors="none")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=0)
        ax.set_xlabel("HalfCheetah controlled task id")
        ax.set_ylabel("Episode return")
        ax.set_title(f"Step size = {step_size:g} | Jump & Retrain Episode Rewards")

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Training steps (num_steps)")

        output_path = output_dir / f"halfcheetah_step{step_size:g}_episode_rewards.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"[Saved] {output_path}")


def make_plots_by_task(
    records: Dict[float, List[Dict]],
    task_ids: List[int],
    step_sizes: List[float],
    output_dir: Path,
    max_steps: int,
) -> None:
    """
    One figure per task id: x=step size (even spacing), y=episode return.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=max_steps)

    sorted_steps = sorted(step_sizes)
    x_positions = {step: idx for idx, step in enumerate(sorted_steps)}
    xticks = [x_positions[s] for s in sorted_steps]
    xtick_labels = [str(s) for s in sorted_steps]

    # Build task -> all records
    task_records: Dict[int, List[Dict]] = {task: [] for task in task_ids}
    for step_size, items in records.items():
        for item in items:
            task_records[item["task_id"]].append(
                {
                    "step_size": step_size,
                    "episode_reward": item["episode_reward"],
                    "num_steps": item["num_steps"],
                }
            )

    for task_id in sorted(task_ids):
        items = task_records.get(task_id, [])
        if not items:
            print(f"[Warning] task {task_id} has no data; skipping.")
            continue

        fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
        xs = [x_positions[item["step_size"]] for item in items]
        ys = [item["episode_reward"] for item in items]
        colors = [cmap(norm(min(item["num_steps"], max_steps))) for item in items]

        ax.scatter(xs, ys, c=colors, s=40, edgecolors="none")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel("Step size")
        ax.set_ylabel("Episode return")
        ax.set_title(f"HalfCheetah task {task_id} | Jump & Retrain episode rewards")

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Training steps (num_steps)")

        output_path = output_dir / f"halfcheetah_task{task_id}_episode_rewards.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"[Saved] {output_path}")


def main() -> None:
    args = parse_args()
    step_sizes = sorted({round(s, 3) for s in args.step_sizes})
    records = load_records(args.data_dir, args.task_ids, step_sizes)
    step_output = args.output_dir / "by_step"
    task_output = args.output_dir / "by_task"
    make_plots_by_step_size(records, args.task_ids, step_output, args.max_steps)
    make_plots_by_task(records, args.task_ids, step_sizes, task_output, args.max_steps)


if __name__ == "__main__":
    main()

