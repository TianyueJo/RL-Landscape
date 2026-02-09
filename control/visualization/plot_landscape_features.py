import argparse
import os
import re
import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import matplotlib.pyplot as plt


LANDSCAPE_DIR = os.path.join(os.path.dirname(__file__), 'landscape_data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'landscape_plots')
LOGS_DIR = os.path.join(os.path.dirname(__file__), 'slurm_logs')

AXIS_MIN_STEPS = 0.0
AXIS_MAX_STEPS = 10_000_000.0


def format_step_range() -> str:
    return f"{AXIS_MIN_STEPS / 1_000_000:.1f}-{AXIS_MAX_STEPS / 1_000_000:.1f}M steps"

ENV_TASK_ID_RANGES = {
    # Walker2d-v4: 0-15
    'Walker2d-v4': [range(0, 16)],
    # HalfCheetah-v4: 16-31 (new task ids) and 100-115 (legacy task ids)
    'HalfCheetah-v4': [range(16, 32), range(100, 116)],
    'Ant-v4': [range(200, 216)],
    'Humanoid-v4': [range(300, 316)],
}


FILENAME_RE = re.compile(r"features_task_(?P<task>\d+)_step_(?P<k>\d+)\.json$")


def task_to_env(task_id: int) -> str:
    """Map task_id to env name (supports new 16-31 and legacy 100-115 task id ranges)."""
    for env, ranges in ENV_TASK_ID_RANGES.items():
        for r in ranges:
            if task_id in r:
                return env
    # Default to Walker2d-v4 (backward compatibility)
    return 'Walker2d-v4'


def task_to_seed_index(task_id: int) -> int:
    """Compute seed index from task_id (0-15 within each env)."""
    env = task_to_env(task_id)
    for r in ENV_TASK_ID_RANGES.get(env, []):
        if task_id in r:
            return task_id - r.start
    return task_id


def task_to_group(task_id: int) -> str:
    """Assign group label by task_id (local id 0-7 / 8-15)."""
    env = task_to_env(task_id)
    if env not in ENV_TASK_ID_RANGES:
        return 'Unknown'
    
    local_id = get_local_id(task_id)
    if 0 <= local_id <= 7:
        return 'Group1_FixedInit'
    elif 8 <= local_id <= 15:
        return 'Group2_FixedTrain'
    else:
        return 'Unknown'


def get_local_id(task_id: int) -> int:
    """Get local id (0-15) by offsetting within ENV_TASK_ID_RANGES."""
    env = task_to_env(task_id)
    for r in ENV_TASK_ID_RANGES.get(env, []):
        if task_id in r:
            return task_id - r.start
    return task_id


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def get_task_ids_for_envs(envs: Set[str]) -> Set[int]:
    task_ids: Set[int] = set()
    for env in envs:
        ranges = ENV_TASK_ID_RANGES.get(env, [])
        for r in ranges:
            task_ids.update(r)
    return task_ids


def load_all_features(dir_path: str, min_steps: float = 0.0, max_steps: float = float('inf'),
                      legacy_task_ids: Optional[Set[int]] = None) -> Dict[int, List[Dict]]:
    """Aggregate feature records by task_id and return {task_id: [records...]}.

    Each record contains: num_steps, step_index, timestamp, and scalar feature keys.
    Also loads early data from all_features.jsonl to include the first training runs.

    Args:
        dir_path: Feature directory
        max_steps: Max steps; only load records <= this (default: 10M)
    """
    per_task: Dict[int, List[Dict]] = defaultdict(list)
    if not os.path.isdir(dir_path):
        return per_task

    if legacy_task_ids:
        for tid in legacy_task_ids:
            per_task.setdefault(tid, [])

    # 1) Load from per-step JSON files
    for fname in os.listdir(dir_path):
        if not fname.endswith('.json') or fname == 'all_features.jsonl':
            continue
        m = FILENAME_RE.match(fname)
        if not m:
            continue
        task_id = int(m.group('task'))
        step_index = int(m.group('k'))
        fpath = os.path.join(dir_path, fname)
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
        except Exception:
            continue

        num_steps = data.get('num_steps') or data.get('global_steps') or np.nan
        if np.isfinite(num_steps):
            if num_steps > max_steps:
                continue
            if num_steps < min_steps:
                continue

        record = {
            'step_index': step_index,
            'num_steps': num_steps,
            'timestamp': data.get('timestamp', np.nan),
        }
        # Copy other scalar features
        for k, v in data.items():
            if k in ('step_index', 'num_steps', 'global_steps', 'timestamp'):
                continue
            if isinstance(v, (int, float)):
                record[k] = float(v)
        per_task[task_id].append(record)

    # 2) Load early data (0-5M) from all_features.jsonl using legacy assignment logic
    jsonl_path = os.path.join(dir_path, 'all_features.jsonl')
    if legacy_task_ids and min_steps < 5_000_000:
        assign_legacy_initial_data(per_task, jsonl_path, legacy_task_ids,
                                   max_initial_steps=min(max_steps, 5_000_000))

    # Sort by steps
    for task_id, recs in per_task.items():
        recs.sort(key=lambda r: (math.inf if r.get('num_steps') is None else r.get('num_steps'), r.get('step_index', -1)))
    
    return per_task


def assign_legacy_initial_data(per_task: Dict[int, List[Dict]], jsonl_path: str, target_task_ids: Set[int],
                               max_initial_steps: float = 5_000_000, gap_threshold: int = 200_000) -> None:
    """Assign 0-5M-step legacy data to target task_ids using the original script logic."""
    if not target_task_ids or not os.path.exists(jsonl_path):
        return

    existing_task_ids = [tid for tid in per_task.keys() if tid in target_task_ids]
    if not existing_task_ids:
        return

    task_timestamp_ranges: Dict[int, Tuple[float, float]] = {}
    task_num_steps_ranges: Dict[int, Tuple[float, float]] = {}
    for task_id in existing_task_ids:
        recs = per_task.get(task_id, [])
        timestamps = [r.get('timestamp', 0) for r in recs if r.get('timestamp', 0) > 0]
        num_steps_list = [r.get('num_steps', 0) for r in recs if r.get('num_steps', 0) > 0]
        if timestamps:
            task_timestamp_ranges[task_id] = (min(timestamps), max(timestamps))
        if num_steps_list:
            task_num_steps_ranges[task_id] = (min(num_steps_list), max(num_steps_list))

    if not task_num_steps_ranges:
        return

    early_data_by_timestamp: List[Tuple[float, float, Dict]] = []
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
            except Exception:
                continue
            num_steps = data.get('num_steps', 0)
            step_count = data.get('step_count', -1)
            timestamp = data.get('timestamp', 0)

            if not (0 < num_steps < max_initial_steps):
                continue

            record = {
                'step_index': step_count,
                'num_steps': num_steps,
                'timestamp': timestamp,
            }
            for k, v in data.items():
                if k in ('step_index', 'num_steps', 'global_steps', 'timestamp', 'step_count'):
                    continue
                if isinstance(v, (int, float)):
                    record[k] = float(v)
            early_data_by_timestamp.append((timestamp, num_steps, record))

    if not early_data_by_timestamp:
        return

    early_data_by_timestamp.sort(key=lambda x: (x[1], x[0]))
    assigned_records: Set[Tuple[float, float]] = set()
    task_early_data: Dict[int, List[Dict]] = defaultdict(list)

    for timestamp, record_num_steps, record in early_data_by_timestamp:
        record_key = (timestamp, record_num_steps)
        if record_key in assigned_records:
            continue

        best_task_id = None
        min_gap = float('inf')
        candidates = []
        for task_id, (min_ns, _max_ns) in task_num_steps_ranges.items():
            if task_id not in target_task_ids:
                continue
            if record_num_steps < min_ns:
                gap = min_ns - record_num_steps
                if gap < gap_threshold:
                    candidates.append((task_id, gap))

        if candidates:
            candidates.sort(key=lambda x: (x[1], x[0]))
            best_task_id, min_gap = candidates[0]

        if best_task_id is None:
            min_time_diff = float('inf')
            for task_id, (min_ts, _max_ts) in task_timestamp_ranges.items():
                if task_id not in target_task_ids:
                    continue
                if timestamp < min_ts:
                    time_diff = min_ts - timestamp
                    if time_diff < min_time_diff and time_diff < 2_592_000:  # 30 days
                        min_time_diff = time_diff
                        best_task_id = task_id

        if best_task_id is not None:
            task_early_data[best_task_id].append(record)
            assigned_records.add(record_key)

    for task_id, records in task_early_data.items():
        per_task[task_id].extend(records)


def plot_lines(xs: np.ndarray, ys_map: Dict[str, np.ndarray], title: str, out_path: str,
               xlabel: str = 'steps', ylabel: str = 'value') -> None:
    plt.figure(figsize=(10, 6))
    for name, ys in ys_map.items():
        plt.plot(xs, ys, label=name, linewidth=1.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if len(ys_map) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def nan_filter_pair(xs: List[float], ys: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    arr_x = np.asarray(xs, dtype=float)
    arr_y = np.asarray(ys, dtype=float)
    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    return arr_x[mask], arr_y[mask]


def plot_per_task(per_task: Dict[int, List[Dict]]) -> None:
    # Disabled: per-task core plots
    return

        # Disabled: plot other features separately


def aggregate_by_env(per_task: Dict[int, List[Dict]]) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    """
    Return structure: {
        env_name: {
            feature_key: [(num_steps, value), ...]
        }
    }
    (concatenated across all tasks in the same env; used for quick trend visualization)
    """
    agg: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for task_id, recs in per_task.items():
        env_name = task_to_env(task_id)
        for r in recs:
            steps = r.get('num_steps', np.nan)
            for k, v in r.items():
                if k in ('num_steps', 'step_index', 'timestamp'):
                    continue
                if isinstance(v, (int, float)) and np.isfinite(v) and np.isfinite(steps):
                    agg[env_name][k].append((float(steps), float(v)))
    return agg


def _parse_steps_token(token: str) -> float:
    # Parse TUI "Steps" token (e.g., 688.1K, 4.9M) to a float
    try:
        token = token.strip()
        if token.endswith(('K', 'M', 'G')):
            num = float(token[:-1])
            suffix = token[-1].upper()
            mul = {'K': 1e3, 'M': 1e6, 'G': 1e9}.get(suffix, 1.0)
            return num * mul
        return float(token)
    except Exception:
        return float('nan')


def parse_episode_from_log(task_id: int) -> List[Tuple[float, float]]:
    """Parse (steps, episode_return) from slurm logs.

    Logic: keep tracking the most recent "Steps" value; when an episode_return line appears,
    record (last_steps, value).
    """
    log_path = os.path.join(LOGS_DIR, f"walker_landscape_{task_id}.out")
    pairs: List[Tuple[float, float]] = []
    if not os.path.isfile(log_path):
        return pairs
    last_steps = float('nan')
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if '│  Steps' in line:
                    # Example: "│  Steps             4.9M      Copy        8s   1%"
                    parts = line.strip().split()
                    # Find a token like 4.9K/M
                    for tok in parts:
                        if any(tok.endswith(suf) for suf in ('K', 'M', 'G')) or tok.replace('.', '', 1).isdigit():
                            val = _parse_steps_token(tok)
                            if np.isfinite(val):
                                last_steps = val
                            break
                elif 'episode_return' in line:
                    # Example: "│  episode_return              395.360    episode_length              223.507  │"
                    try:
                        parts = line.strip().split()
                        # Take the number right after "episode_return"
                        for i, tok in enumerate(parts):
                            if tok == 'episode_return' and i + 1 < len(parts):
                                val = float(parts[i + 1])
                                if np.isfinite(last_steps):
                                    pairs.append((last_steps, val))
                                break
                    except Exception:
                        continue
    except Exception:
        return pairs
    # Deduplicate and sort by steps
    pairs = sorted(list({(float(s), float(v)) for s, v in pairs}), key=lambda x: x[0])
    return pairs


def merge_episode_from_logs(per_task: Dict[int, List[Dict]], env_seed: Dict[str, Dict[int, Dict[str, List[Tuple[float, float]]]]]) -> None:
    """If a seed's episode_reward series is missing/too short, try filling it from logs."""
    for task_id in per_task.keys():
        env = task_to_env(task_id)
        seed_idx = task_to_seed_index(task_id)
        series_map = env_seed[env][seed_idx]
        cur = series_map.get('episode_reward', [])
        if len(cur) >= 3:
            continue
        pairs = parse_episode_from_log(task_id)
        if pairs:
            series_map['episode_reward'] = pairs

def plot_per_env(agg_env: Dict[str, Dict[str, List[Tuple[float, float]]]]) -> None:
    # No longer generate per-env single-feature aggregate plots
    return


def build_env_seed_series(per_task: Dict[int, List[Dict]]) -> Dict[str, Dict[int, Dict[str, List[Tuple[float, float]]]]]:
    """
    Return { env: { seed_idx: { feature_key: [(steps, value), ...] } } }.
    Used to overlay multiple seeds within the same env on one figure.
    """
    env_seed: Dict[str, Dict[int, Dict[str, List[Tuple[float, float]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for task_id, recs in per_task.items():
        env = task_to_env(task_id)
        seed_idx = task_to_seed_index(task_id)
        for r in recs:
            steps = r.get('num_steps', np.nan)
            if not np.isfinite(steps):
                continue
            for k, v in r.items():
                if k in ('num_steps', 'step_index', 'timestamp'):
                    continue
                if isinstance(v, (int, float)) and np.isfinite(v):
                    env_seed[env][seed_idx][k].append((float(steps), float(v)))
    # Sort
    for env, seeds in env_seed.items():
        for seed_idx, fmap in seeds.items():
            for k, pairs in fmap.items():
                pairs.sort(key=lambda x: x[0])
    return env_seed


def build_env_group_series(per_task: Dict[int, List[Dict]]) -> Dict[str, Dict[str, Dict[int, Dict[str, List[Tuple[float, float]]]]]]:
    """
    Return { env: { group: { local_id: { feature_key: [(steps, value), ...] } } } }.
    Used to visualize by group.
    """
    env_group: Dict[str, Dict[str, Dict[int, Dict[str, List[Tuple[float, float]]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for task_id, recs in per_task.items():
        env = task_to_env(task_id)
        group = task_to_group(task_id)
        local_id = get_local_id(task_id)
        
        if group == 'Unknown':
            continue
            
        for r in recs:
            steps = r.get('num_steps', np.nan)
            if not np.isfinite(steps):
                continue
            for k, v in r.items():
                if k in ('num_steps', 'step_index', 'timestamp'):
                    continue
                if isinstance(v, (int, float)) and np.isfinite(v):
                    env_group[env][group][local_id][k].append((float(steps), float(v)))
    
    # Sort
    for env, groups in env_group.items():
        for group, local_ids in groups.items():
            for local_id, fmap in local_ids.items():
                for k, pairs in fmap.items():
                    pairs.sort(key=lambda x: x[0])
    return env_group


def plot_env_multiseed(env_seed: Dict[str, Dict[int, Dict[str, List[Tuple[float, float]]]]]) -> None:
    """
    Create one multi-panel figure per env (x=steps, y=feature value), overlaying seeds with colors/linestyles.
    Core panel keys: sharpness / lambda_max_10 / hessian_trace_10 / fim_trace / parameter_norm / gradient_smoothness / episode_reward
    """
    ensure_dir(OUTPUT_DIR)
    core_keys = ['sharpness', 'lambda_max_10', 'hessian_trace_10', 'fim_trace', 'parameter_norm', 'gradient_smoothness', 'episode_reward']

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])

    for env, seeds in env_seed.items():
        # Core features (including episode_reward); dynamic rows/cols
        n = len(core_keys)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 + 4 * rows), sharex=True)
        axes = axes.ravel() if isinstance(axes, np.ndarray) else [axes]
        for idx, key in enumerate(core_keys):
            ax = axes[idx]
            any_plotted = False
            for si, fmap in sorted(seeds.items()):
                if key not in fmap:
                    continue
                pairs = fmap[key]
                if not pairs:
                    continue
                xs = np.asarray([p[0] for p in pairs], dtype=float)
                ys = np.asarray([p[1] for p in pairs], dtype=float)
                # Keep points within configured x-axis range
                mask = np.isfinite(xs) & np.isfinite(ys) & (xs <= AXIS_MAX_STEPS) & (xs >= AXIS_MIN_STEPS)
                xs, ys = xs[mask], ys[mask]
                if len(xs) < 2:
                    continue
                
                # Deduplicate by x: take mean y for identical x
                unique_xs = np.unique(xs)
                unique_ys = np.zeros_like(unique_xs)
                for i, x in enumerate(unique_xs):
                    mask_x = np.abs(xs - x) < 1  # allow small floating error
                    unique_ys[i] = np.mean(ys[mask_x])
                
                # Sort by x
                sort_idx = np.argsort(unique_xs)
                unique_xs = unique_xs[sort_idx]
                unique_ys = unique_ys[sort_idx]
                
                if len(unique_xs) < 2:
                    continue
                
                ax.plot(unique_xs, unique_ys, label=f'seed{100+si}', color=color_cycle[si % len(color_cycle)], linewidth=1.4)
                any_plotted = True
            ax.set_title(key)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=AXIS_MIN_STEPS, right=AXIS_MAX_STEPS)
            if idx % cols == 0:
                ax.set_ylabel('value')
            if idx >= (rows - 1) * cols:
                ax.set_xlabel('agent steps')
            if any_plotted and idx == 0:
                ax.legend(loc='best', fontsize=9)
        # Hide unused axes
        for k in range(n, rows * cols):
            fig.delaxes(axes[k])
        range_label = format_step_range()
        fig.suptitle(f'{env} | Original PufferLib (LSTM) - Core Features (Multi-seed, {range_label})', fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        out_core = os.path.join(OUTPUT_DIR, f'env_{env}_core_multiseed.png')
        fig.savefig(out_core, dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved: {out_core}")


def plot_env_by_group(env_group: Dict[str, Dict[str, Dict[int, Dict[str, List[Tuple[float, float]]]]]]) -> None:
    """
    Create one multi-panel figure per env per group.
    Group1: fixed init (200), varying train seeds (200-207)
    Group2: varying init (200-207), fixed train seed (200)
    """
    ensure_dir(OUTPUT_DIR)
    core_keys = ['sharpness', 'lambda_max_10', 'hessian_trace_10', 'fim_trace', 'parameter_norm', 'gradient_smoothness', 'episode_reward']

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])

    group_labels = {
        'Group1_FixedInit': 'Group1 (Fixed Init, Varying Train Seed)',
        'Group2_FixedTrain': 'Group2 (Varying Init Seed, Fixed Train)'
    }

    for env, groups in env_group.items():
        for group_name, local_ids in groups.items():
            if group_name not in group_labels:
                continue
                
            # Core features (including episode_reward); dynamic rows/cols
            n = len(core_keys)
            cols = 3
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(14, 4 + 4 * rows), sharex=True)
            axes = axes.ravel() if isinstance(axes, np.ndarray) else [axes]
            
            for idx, key in enumerate(core_keys):
                ax = axes[idx]
                any_plotted = False
                for local_id, fmap in sorted(local_ids.items()):
                    if key not in fmap:
                        continue
                    pairs = fmap[key]
                    if not pairs:
                        continue
                    xs = np.asarray([p[0] for p in pairs], dtype=float)
                    ys = np.asarray([p[1] for p in pairs], dtype=float)
                    # Keep points within configured x-axis range
                    mask = np.isfinite(xs) & np.isfinite(ys) & (xs <= AXIS_MAX_STEPS) & (xs >= AXIS_MIN_STEPS)
                    xs, ys = xs[mask], ys[mask]
                    if len(xs) < 2:
                        # If no in-range data, fall back to all data (still show within the axis range)
                        xs = np.asarray([p[0] for p in pairs], dtype=float)
                        ys = np.asarray([p[1] for p in pairs], dtype=float)
                        mask = np.isfinite(xs) & np.isfinite(ys)
                        xs, ys = xs[mask], ys[mask]
                        if len(xs) < 2:
                            continue
                    
                    # Deduplicate by x: take mean y for identical x
                    unique_xs = np.unique(xs)
                    unique_ys = np.zeros_like(unique_xs)
                    for i, x in enumerate(unique_xs):
                        mask_x = np.abs(xs - x) < 1  # allow small floating error
                        unique_ys[i] = np.mean(ys[mask_x])
                    
                    # Sort by x
                    sort_idx = np.argsort(unique_xs)
                    unique_xs = unique_xs[sort_idx]
                    unique_ys = unique_ys[sort_idx]
                    
                    if len(unique_xs) < 2:
                        continue
                    
                    # Group1 uses local_id 0-7, Group2 uses 8-15, but display both starting from 0
                    display_id = local_id if group_name == 'Group1_FixedInit' else (local_id - 8)
                    ax.plot(unique_xs, unique_ys, label=f'Task {display_id}', color=color_cycle[display_id % len(color_cycle)], linewidth=1.4)
                    any_plotted = True
                    # Set x-axis range
                    if idx == 0:  # only set once to avoid repetition
                        ax.set_xlim(left=AXIS_MIN_STEPS, right=AXIS_MAX_STEPS)
                ax.set_title(key)
                ax.grid(True, alpha=0.3)
                if idx % cols == 0:
                    ax.set_ylabel('value')
                if idx >= (rows - 1) * cols:
                    ax.set_xlabel('agent steps')
                if any_plotted and idx == 0:
                    ax.legend(loc='best', fontsize=9)
            
            # Hide unused axes
            for k in range(n, rows * cols):
                fig.delaxes(axes[k])
            
            range_label = format_step_range()
            fig.suptitle(f'{env} | {group_labels[group_name]} - Core Features ({range_label})', fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            # Filename
            group_suffix = 'Group1_FixedInit' if group_name == 'Group1_FixedInit' else 'Group2_FixedTrain'
            out_core = os.path.join(OUTPUT_DIR, f'env_{env}_{group_suffix}.png')
            fig.savefig(out_core, dpi=150)
            plt.close(fig)
            print(f"  ✓ Saved: {out_core}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot landscape features per environment.')
    parser.add_argument('--min-steps', type=float, default=0.0,
                        help='Minimum agent steps to include (default: 0).')
    parser.add_argument('--max-steps', type=float, default=10_000_000.0,
                        help='Maximum agent steps to include (default: 10M).')
    parser.add_argument('--envs', nargs='*', default=['HalfCheetah-v4', 'Ant-v4'],
                        help='List of environment names to include.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(OUTPUT_DIR)
    target_envs = set(args.envs)
    global AXIS_MIN_STEPS, AXIS_MAX_STEPS
    AXIS_MIN_STEPS = args.min_steps
    AXIS_MAX_STEPS = args.max_steps
    legacy_task_ids = None
    if args.min_steps < 5_000_000:
        legacy_task_ids = get_task_ids_for_envs(target_envs)
    per_task = load_all_features(LANDSCAPE_DIR, min_steps=args.min_steps, max_steps=args.max_steps,
                                 legacy_task_ids=legacy_task_ids)
    if not per_task:
        print(f"No feature files found in: {LANDSCAPE_DIR}")
        return

    print(f"Loaded tasks: {sorted(per_task.keys())}")
    print(f"Total feature records: {sum(len(recs) for recs in per_task.values())}")
    
    filtered_per_task = {}
    for task_id, recs in per_task.items():
        env = task_to_env(task_id)
        if env in target_envs:
            filtered_per_task[task_id] = recs
    
    if not filtered_per_task:
        print(f"No tasks found for target environments: {target_envs}")
        return
    
    print(f"Filtered tasks for target environments: {sorted(filtered_per_task.keys())}")
    print(f"Filtered feature records: {sum(len(recs) for recs in filtered_per_task.values())}")
    
    # Organize data by groups
    print("\nOrganizing data by groups...")
    env_group = build_env_group_series(filtered_per_task)
    
    # Group summary
    for env, groups in env_group.items():
        print(f"\n{env}:")
        for group_name, local_ids in groups.items():
            print(f"  {group_name}: {sorted(local_ids.keys())} (total {len(local_ids)} tasks)")
    
    # Supplement: fill episode_reward series from slurm logs (requires env_seed first)
    print("\nParsing episode rewards from logs...")
    env_seed = build_env_seed_series(filtered_per_task)
    merge_episode_from_logs(filtered_per_task, env_seed)
    
    # Merge episode_reward back into env_group
    for task_id in filtered_per_task.keys():
        env = task_to_env(task_id)
        group = task_to_group(task_id)
        local_id = get_local_id(task_id)
        seed_idx = task_to_seed_index(task_id)
        
        if group != 'Unknown' and env in env_seed and seed_idx in env_seed[env]:
            if 'episode_reward' in env_seed[env][seed_idx]:
                env_group[env][group][local_id]['episode_reward'] = env_seed[env][seed_idx]['episode_reward']
    
    print("\nGenerating group-specific plots...")
    plot_env_by_group(env_group)
    
    print("\nGenerating multiseed plots...")
    plot_env_multiseed(env_seed)
    
    print(f"\n✅ All plots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()







