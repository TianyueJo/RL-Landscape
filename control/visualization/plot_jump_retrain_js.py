#!/usr/bin/env python3
"""
Visualize HalfCheetah Jump & Retrain JS divergence:
  - X-axis: controlled_task id (16-31)
  - Y-axis: JS divergence (log scale)
  - One color per step size

Input: analysis_outputs/jump_retrain_js/task_<id>_js.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

DEFAULT_STEPS = [0.0, 5.0, 15.0, 45.0, 135.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot JS divergence between final and retrain policies."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("analysis_outputs/jump_retrain_js"),
        help="Directory containing task_x_js.json files",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=list(range(16, 32)),
        help="List of controlled_task ids to plot",
    )
    parser.add_argument(
        "--step-sizes",
        type=float,
        nargs="+",
        default=DEFAULT_STEPS,
        help="Step sizes to plot (must match the JSON)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/jump_retrain_js_plot.png"),
        help="Output image path",
    )
    return parser.parse_args()


def load_js_results(results_dir: Path, task_ids: List[int]) -> Dict[int, Dict[float, float]]:
    """Return {task_id: {step_size: js_divergence}}."""
    data: Dict[int, Dict[float, float]] = {}
    for task_id in task_ids:
        path = results_dir / f"task_{task_id:02d}_js.json"
        if not path.exists():
            print(f"[Warning] Missing file: {path}")
            continue
        with path.open() as f:
            payload = json.load(f)
        step_map: Dict[float, float] = {}
        for entry in payload.get("step_results", []):
            step = float(entry.get("step_size"))
            js_value = float(entry.get("js_divergence", 0.0))
            step_map[step] = js_value
        data[task_id] = step_map
    return data


def plot_js_divergence(
    data: Dict[int, Dict[float, float]],
    task_ids: List[int],
    step_sizes: List[float],
    output_path: Path,
) -> None:
    if not data:
        raise RuntimeError("No JS results loaded; cannot plot.")

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    colors = plt.cm.tab10.colors

    sorted_tasks = sorted(task_ids)
    x_positions = {task: idx for idx, task in enumerate(sorted_tasks)}

    for idx, step in enumerate(step_sizes):
        xs = []
        ys = []
        for task_id in sorted_tasks:
            step_map = data.get(task_id, {})
            if step not in step_map:
                continue
            xs.append(x_positions[task_id])
            ys.append(step_map[step])
        if not xs:
            print(f"[Info] step size={step} has no data; skipping.")
            continue
        ax.scatter(
            xs,
            ys,
            label=f"step {step:g}",
            color=colors[idx % len(colors)],
            s=40,
        )

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels([str(t) for t in sorted_tasks])
    ax.set_xlabel("HalfCheetah controlled task id")
    ax.set_ylabel("JS divergence (log scale)")
    ax.set_yscale("log")
    ax.set_title("HalfCheetah final vs retrain policy JS divergence")
    ax.legend(title="Step size", ncol=3)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {output_path}")


def main() -> None:
    args = parse_args()
    step_sizes = [float(s) for s in args.step_sizes]
    js_data = load_js_results(args.results_dir, args.task_ids)
    plot_js_divergence(js_data, args.task_ids, step_sizes, args.output)


if __name__ == "__main__":
    main()


