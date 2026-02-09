#!/usr/bin/env python3
"""
Plot Jump & Retrain results for tasks 22 and 27.
- X-axis: step size
- Y-axis: episode return
- Point color: episode return at different num_steps
- For the same step size, different random seeds are shown side-by-side
- A horizontal line shows the original task episode return
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

FILENAME_REGEX = re.compile(
    r"^features_task_(?P<label>\d+_jr\d+_s(?P<step>[\d\.]+))_step_(?P<idx>\d+)\.json$"
)
LABEL_REGEX = re.compile(r"^(?P<task>\d+)_jr(?P<jr>\d+)_s(?P<step>[\d\.]+)$")


def load_original_reward(task_id: int, models_dir: Path) -> Optional[float]:
    """Load the original task's best_reward."""
    best_info_path = models_dir / f"controlled_task_{task_id}" / "best_info.json"
    if best_info_path.exists():
        with best_info_path.open() as f:
            data = json.load(f)
            return data.get("best_reward")
    return None


def load_retrain_data(
    models_dir: Path, task_ids: List[int], step_sizes: List[float]
) -> Dict[int, Dict[float, List[Dict]]]:
    """
    Load retrain data.
    Search for features_task_*.json files from several possible locations.
    These files contain intermediate training-step records.
    Returns: {task_id: {step_size: [records]}}
    """
    results: Dict[int, Dict[float, List[Dict]]] = {
        task_id: {step_size: [] for step_size in step_sizes} for task_id in task_ids
    }

    # Search several possible locations
    base_dir = models_dir.parent if models_dir.name == "models" else models_dir
    search_dirs = [
        base_dir / "results" / "jump_retrain_22_27",
        base_dir / "landscape_data",
        models_dir / "jump_retrain_22_27",
    ]
    
    # Also search subdirectories under landscape_data
    landscape_data_dir = base_dir / "landscape_data"
    if landscape_data_dir.exists():
        for subdir in landscape_data_dir.iterdir():
            if subdir.is_dir():
                search_dirs.append(subdir)

    found_files = 0
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Find all features_task_*.json files
        for json_file in search_dir.rglob("features_task_*.json"):
            match = FILENAME_REGEX.match(json_file.name)
            if not match:
                continue

            label = match.group("label")
            label_match = LABEL_REGEX.match(label)
            if not label_match:
                continue

            task_id = int(label_match.group("task"))
            if task_id not in task_ids:
                continue

            jr_index = int(label_match.group("jr"))
            raw_step = float(label_match.group("step"))
            
            # Match step_size
            step_size = None
            for s in step_sizes:
                if abs(raw_step - s) < 0.01:
                    step_size = s
                    break
            
            if step_size is None:
                continue

            try:
                with json_file.open() as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            episode_reward = data.get("episode_reward")
            num_steps = data.get("num_steps")
            if episode_reward is None or num_steps is None:
                continue

            results[task_id][step_size].append(
                {
                    "episode_reward": float(episode_reward),
                    "num_steps": int(num_steps),
                    "jr_index": jr_index,
                    "step_size": step_size,
                }
            )
            found_files += 1

    if found_files == 0:
        print("Warning: no features_task_*.json files found")
        print("  Searched these directories:")
        for d in search_dirs:
            print(f"    - {d}")

    return results


def plot_task_results(
    task_id: int,
    data: Dict[float, List[Dict]],
    original_reward: Optional[float],
    step_sizes: List[float],
    output_path: Path,
    max_steps: int = 500_000,
) -> None:
    """Plot results for a single task."""
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=max_steps)

    # Prepare data: group by step_size and jr_index
    sorted_step_sizes = sorted(step_sizes)
    x_positions = {step: idx for idx, step in enumerate(sorted_step_sizes)}

    all_x = []
    all_y = []
    all_colors = []

    for step_size in sorted_step_sizes:
        items = data.get(step_size, [])
        if not items:
            continue

        # Group by jr_index (same random seed)
        seed_groups = defaultdict(list)
        for item in items:
            seed_groups[item["jr_index"]].append(item)

        # Assign x positions per seed (side-by-side)
        num_seeds = len(seed_groups)
        base_x = x_positions[step_size]
        
        if num_seeds == 1:
            seed_x_positions = {list(seed_groups.keys())[0]: base_x}
        else:
            # Spread seeds evenly around the step_size position
            jitter_width = 0.3  # total width
            seed_x_positions = {}
            for idx, jr_index in enumerate(sorted(seed_groups.keys())):
                seed_x_positions[jr_index] = base_x + (idx - (num_seeds - 1) / 2) * (jitter_width / max(1, num_seeds - 1))

        # For each seed, all points across training steps share the same x position
        for jr_index, seed_items in seed_groups.items():
            seed_x = seed_x_positions[jr_index]
            for item in seed_items:
                all_x.append(seed_x)
                all_y.append(item["episode_reward"])
                all_colors.append(cmap(norm(min(item["num_steps"], max_steps))))

    # Scatter
    if all_x:
        ax.scatter(all_x, all_y, c=all_colors, s=60, edgecolors="black", linewidths=0.5, alpha=0.7)

    # Original reward line
    if original_reward is not None:
        ax.axhline(
            y=original_reward,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Original Reward: {original_reward:.2f}",
            alpha=0.8,
        )

    # Axes
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels([str(s) for s in sorted_step_sizes])
    ax.set_xlabel("Noise Scale", fontsize=12)
    ax.set_ylabel("Episode Return", fontsize=12)
    ax.set_title(
        f"HalfCheetah Task {task_id} | Jump & Retrain Episode Returns\n"
        f"(Different colors = different training steps, grouped by random seed)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Training Steps (num_steps)", fontsize=11)

    # Legend
    if original_reward is not None:
        ax.legend(loc="best", fontsize=10)

    # Statistics
    total_points = sum(len(items) for items in data.values())
    ax.text(
        0.02,
        0.98,
        f"Total points: {total_points}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Jump & Retrain results for tasks 22 and 27"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Models directory path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("landscape_plots/jump_retrain_22_27"),
        help="Output directory",
    )
    parser.add_argument(
        "--step-sizes",
        type=float,
        nargs="+",
        default=[20.0, 40.0, 60.0],
        help="Step sizes list",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500_000,
        help="Max steps for color normalization",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=[22, 27],
        help="Task id list",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from: {args.models_dir}")
    step_sizes = sorted({round(s, 3) for s in args.step_sizes})
    retrain_data = load_retrain_data(args.models_dir, args.task_ids, step_sizes)

    # Plot each task
    for task_id in args.task_ids:
        print(f"\nProcessing task {task_id}...")
        task_data = retrain_data.get(task_id, {})
        
        # Data summary
        total_points = sum(len(items) for items in task_data.values())
        print(f"  Found {total_points} data points")
        for step_size, items in task_data.items():
            print(f"    Step size {step_size}: {len(items)} points")

        # Load original reward
        original_reward = load_original_reward(task_id, args.models_dir)
        if original_reward is not None:
            print(f"  Original reward: {original_reward:.2f}")
        else:
            print("  Warning: original reward not found")

        # Plot
        output_path = args.output_dir / f"task_{task_id}_jump_retrain.png"
        plot_task_results(
            task_id,
            task_data,
            original_reward,
            step_sizes,
            output_path,
            args.max_steps,
        )

    print(f"\n✓ All figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

