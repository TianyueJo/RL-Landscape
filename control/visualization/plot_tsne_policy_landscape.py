#!/usr/bin/env python3
"""
Use t-SNE to reduce policy parameters to 2D and visualize in 3D:
- x, y: t-SNE 2D coordinates
- z: episode reward
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px


def flatten_params(state_dict: dict):
    """
    Flatten all tensor parameters in state_dict into a single vector.

    Returns:
      flat_vec: [D] numpy array
    """
    flats = []
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            flats.append(v.detach().cpu().reshape(-1).numpy())
    if not flats:
        raise RuntimeError("Empty state_dict? No tensor parameters found.")
    flat_vec = np.concatenate(flats, axis=0)
    return flat_vec

FILENAME_REGEX = re.compile(
    r"^features_task_(?P<label>\d+_jr\d+_s(?P<step>[\d\.]+))_step_(?P<idx>\d+)\.json$"
)
LABEL_REGEX = re.compile(r"^(?P<task>\d+)_jr(?P<jr>\d+)_s(?P<step>[\d\.]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize policy parameter space and episode reward using t-SNE"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Models root directory containing controlled_task_* folders",
    )
    parser.add_argument(
        "--retrain-models-dir",
        type=Path,
        default=None,
        help="Directory containing retrained model files (e.g., jump_retrain_all), if not specified, search in models-dir",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("landscape_data"),
        help="Landscape data root containing features_task_*.json files",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("landscape_plots/tsne_policy_landscape_3d.png"),
        help="Output image path (PNG)",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("landscape_plots/tsne_policy_landscape_3d.html"),
        help="Output interactive HTML path",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=list(range(16, 32)),
        help="HalfCheetah controlled_task ids to process",
    )
    parser.add_argument(
        "--step-sizes",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 15.0, 45.0, 135.0],
        help="Step sizes to process",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity",
    )
    parser.add_argument(
        "--tsne-random-state",
        type=int,
        default=42,
        help="t-SNE random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used when loading models",
    )
    return parser.parse_args()


def _match_step(value: float, targets: List[float], tol: float = 1e-3) -> Optional[float]:
    """Match a step size to one of the targets within tolerance."""
    for target in targets:
        if abs(value - target) <= tol:
            return target
    return None


def load_policy_params(
    models_dir: Path,
    task_id: int,
    device: str = "cpu",
    checkpoint_type: str = "best",
) -> Optional[torch.Tensor]:
    """Load policy parameters and flatten them"""
    task_dir = models_dir / f"controlled_task_{task_id}"
    
    if checkpoint_type == "best":
        model_path = task_dir / "best_model.pt"
        if not model_path.exists():
            model_path = task_dir / "final_model.pt"
    else:
        model_path = task_dir / "final_model.pt"
    
    if not model_path.exists():
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Strip "module." prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Flatten parameters
        flat_params = flatten_params(state_dict)
        return flat_params
    except Exception as e:
        print(f"[Warning] Failed to load model for task {task_id}: {e}")
        return None


def load_retrain_policy_params(
    models_dir: Path,
    task_id: int,
    jr_index: int,
    step_size: float,
    device: str = "cpu",
    retrain_models_dir: Optional[Path] = None,
) -> Optional[torch.Tensor]:
    """Load retrain policy parameters"""
    # Prioritize loading from retrain_models_dir
    if retrain_models_dir is not None and retrain_models_dir.exists():
        # Load from jump_retrain_all directory, format: controlled_task_{task_id}_jr{jr_index}_s0.50
        task_label = f"{task_id}_jr{jr_index}_s0.50"
        task_dir = retrain_models_dir / f"controlled_task_{task_label}"
        model_path = task_dir / "final_model.pt"
        
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
                
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                flat_params = flatten_params(state_dict)
                return flat_params
            except Exception as e:
                print(f"  [Warning] Failed to load {model_path}: {e}")
    
    # Fallback to original logic: search in models_dir
    task_label = f"{task_id}_jr{jr_index}_s{step_size:.2f}"
    task_dir = models_dir / f"controlled_task_{task_label}"
    
    # Use final_model.pt directly
    model_path = task_dir / "final_model.pt"
    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            flat_params = flatten_params(state_dict)
            return flat_params
        except Exception as e:
            print(f"  [Warning] Failed to load {model_path}: {e}")
    
    return None


def get_episode_reward_from_features(
    data_dir: Path,
    task_id: int,
    jr_index: Optional[int],
    step_size: Optional[float],
) -> Optional[float]:
    """Get episode reward from feature files."""
    if jr_index is None:
        # Original policy, find features_task_{task_id}_step_*.json (files without jr)
        # Or fall back to best_info.json
        pattern = f"features_task_{task_id}_step_*.json"
        files = [f for f in data_dir.glob(pattern) if "_jr" not in f.name]
    else:
        # Retrain policy
        label = f"{task_id}_jr{jr_index}_s{step_size:.2f}"
        pattern = f"features_task_{label}_step_*.json"
        files = list(data_dir.glob(pattern))
    
    if not files:
        return None
    
    # Get the latest step (usually the last one)
    step_numbers = []
    for f in files:
        match = re.search(r"_step_(\d+)\.json$", f.name)
        if match:
            step_numbers.append((int(match.group(1)), f))
    
    if not step_numbers:
        return None
    
    # Use the file with the largest step index
    _, latest_file = max(step_numbers, key=lambda x: x[0])
    
    try:
        with latest_file.open() as f:
            data = json.load(f)
        return data.get("episode_reward")
    except Exception as e:
        print(f"[Warning] Failed to read {latest_file}: {e}")
        return None


def get_episode_reward_from_best_info(
    models_dir: Path,
    task_id: int,
) -> Optional[float]:
    """Get episode reward from best_info.json."""
    task_dir = models_dir / f"controlled_task_{task_id}"
    best_info_path = task_dir / "best_info.json"
    
    if not best_info_path.exists():
        return None
    
    try:
        with best_info_path.open() as f:
            data = json.load(f)
        return data.get("best_reward")
    except Exception as e:
        return None


def collect_all_policies(
    models_dir: Path,
    data_dir: Path,
    task_ids: List[int],
    step_sizes: List[float],
    device: str = "cpu",
    retrain_models_dir: Optional[Path] = None,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Collect parameters and metadata for all policies."""
    all_params = []
    all_metadata = []
    
    # 1. Load original policies (16)
    print("[1/2] Loading original policies...")
    for task_id in task_ids:
        params = load_policy_params(models_dir, task_id, device, "best")
        if params is not None:
            all_params.append(params)
            # Try features first; fall back to best_info.json
            episode_reward = get_episode_reward_from_features(
                data_dir, task_id, None, None
            )
            if episode_reward is None:
                episode_reward = get_episode_reward_from_best_info(models_dir, task_id)
            all_metadata.append({
                "task_id": task_id,
                "jr_index": None,
                "step_size": None,
                "episode_reward": episode_reward,
                "type": "original",
            })
            print(f"  Task {task_id}: parameter dimension {params.shape}, reward={episode_reward}")
        else:
            print(f"  [Warning] Model not found for task {task_id}")
    
    # 2. Load retrain policies
    print(f"[2/2] Loading retrain policies...")
    
    # Prioritize scanning model files from retrain_models_dir
    if retrain_models_dir is not None and retrain_models_dir.exists():
        print(f"  Scanning retrain models from {retrain_models_dir}...")
        retrain_model_dirs = list(retrain_models_dir.glob("controlled_task_*_jr*_s0.50"))
        
        for model_dir in sorted(retrain_model_dirs):
            # Parse directory name: controlled_task_{task_id}_jr{jr_index}_s0.50
            dir_name = model_dir.name
            match = re.match(r"controlled_task_(\d+)_jr(\d+)_s0\.50", dir_name)
            if not match:
                continue
            
            task_id = int(match.group(1))
            jr_index = int(match.group(2))
            
            if task_id not in task_ids:
                continue
            
            # Load model
            model_path = model_dir / "final_model.pt"
            if not model_path.exists():
                continue
            
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
                
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                params = flatten_params(state_dict)
                
                # Find corresponding step_size and reward from feature files
                # Search for all matching feature files
                episode_reward = None
                step_size = None
                
                for path in data_dir.glob(f"features_task_{task_id}_jr{jr_index}_s*_step_*.json"):
                    match_feat = FILENAME_REGEX.match(path.name)
                    if not match_feat:
                        continue
                    label = match_feat.group("label")
                    label_match = LABEL_REGEX.match(label)
                    if not label_match:
                        continue
                    raw_step = float(label_match.group("step"))
                    matched_step = _match_step(raw_step, step_sizes)
                    if matched_step is not None:
                        step_size = matched_step
                        # Get the latest reward
                        try:
                            with path.open() as f:
                                data = json.load(f)
                            reward = data.get("episode_reward")
                            if reward is not None:
                                episode_reward = reward
                        except:
                            pass
                
                all_params.append(params)
                all_metadata.append({
                    "task_id": task_id,
                    "jr_index": jr_index,
                    "step_size": step_size,
                    "episode_reward": episode_reward,
                    "type": "retrain",
                })
                print(
                    f"  Task {task_id}_jr{jr_index}_s{step_size}: "
                    f"parameter dimension {params.shape}, reward={episode_reward}"
                )
            except Exception as e:
                print(f"  [Warning] Failed to load {model_path}: {e}")
    else:
        # Fallback to original logic: find all retrain policies from feature files
        retrain_tasks = set()
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
            jr_index = int(label_match.group("jr"))
            raw_step = float(label_match.group("step"))
            step_size = _match_step(raw_step, step_sizes)
            if step_size is not None:
                retrain_tasks.add((task_id, jr_index, step_size))
        
        print(f"  Found {len(retrain_tasks)} retrain tasks from feature files")
        
        for task_id, jr_index, step_size in sorted(retrain_tasks):
            params = load_retrain_policy_params(
                models_dir, task_id, jr_index, step_size, device, retrain_models_dir
            )
            if params is not None:
                all_params.append(params)
                episode_reward = get_episode_reward_from_features(
                    data_dir, task_id, jr_index, step_size
                )
                all_metadata.append({
                    "task_id": task_id,
                    "jr_index": jr_index,
                    "step_size": step_size,
                    "episode_reward": episode_reward,
                    "type": "retrain",
                })
                print(
                    f"  Task {task_id}_jr{jr_index}_s{step_size}: "
                    f"parameter dimension {params.shape}, reward={episode_reward}"
                )
            else:
                # If model file is missing, try to get reward from feature files
                episode_reward = get_episode_reward_from_features(
                    data_dir, task_id, jr_index, step_size
                )
                if episode_reward is not None:
                    # If features exist but model is missing, use original policy params as placeholder
                    base_params = load_policy_params(models_dir, task_id, device, "best")
                    if base_params is not None:
                        all_params.append(base_params)
                        all_metadata.append({
                            "task_id": task_id,
                            "jr_index": jr_index,
                            "step_size": step_size,
                            "episode_reward": episode_reward,
                            "type": "retrain",
                        })
                        print(
                            f"  Task {task_id}_jr{jr_index}_s{step_size}: "
                            f"using original policy parameters as placeholder, reward={episode_reward}"
                        )
                    else:
                        print(
                            f"  [Warning] Task {task_id}_jr{jr_index}_s{step_size}: "
                            f"model file not found, skipping"
                        )
    
    return all_params, all_metadata


def main() -> None:
    args = parse_args()
    
    # Collect all policies
    all_params, all_metadata = collect_all_policies(
        args.models_dir,
        args.data_dir,
        args.task_ids,
        args.step_sizes,
        args.device,
        args.retrain_models_dir,
    )
    
    if len(all_params) == 0:
        print("[Error] No policy parameters found")
        return
    
    print(f"\nTotal policies collected: {len(all_params)}")
    
    # Check if parameter dimensions are consistent
    param_dims = [p.shape[0] for p in all_params]
    if len(set(param_dims)) > 1:
        print(f"[Warning] Parameter dimensions inconsistent: {set(param_dims)}")
        # Use minimum dimension
        min_dim = min(param_dims)
        all_params = [p[:min_dim] for p in all_params]
        print(f"  Unified to dimension: {min_dim}")
    
    # Convert to numpy array
    params_matrix = np.vstack(all_params)
    print(f"Parameter matrix shape: {params_matrix.shape}")
    
    # Use t-SNE for dimensionality reduction
    print("\nReducing dimensions to 2D using t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        random_state=args.tsne_random_state,
        max_iter=1000,
        verbose=1,
    )
    params_2d = tsne.fit_transform(params_matrix)
    print(f"Reduced dimension shape: {params_2d.shape}")
    
    # Extract episode rewards
    rewards = [m.get("episode_reward") for m in all_metadata]
    valid_indices = [i for i, r in enumerate(rewards) if r is not None]
    
    if len(valid_indices) == 0:
        print("[Error] No valid episode rewards found")
        return
    
    print(f"\nValid data points: {len(valid_indices)}/{len(all_metadata)}")
    
    # Define 5 distinct colors for 5 step sizes
    step_size_colors = {
        0.0: "#FF0000",    # red
        5.0: "#00FF00",    # green
        15.0: "#0000FF",   # blue
        45.0: "#FF00FF",   # magenta
        135.0: "#FFA500",  # orange
    }
    
    # Separate original and retrain policies
    original_indices = [
        i for i in valid_indices if all_metadata[i]["type"] == "original"
    ]
    retrain_indices = [
        i for i in valid_indices if all_metadata[i]["type"] == "retrain"
    ]
    
    # Create interactive 3D plot using plotly
    print("\nCreating interactive 3D visualization...")
    fig = go.Figure()
    
    # Plot original policies
    if original_indices:
        orig_x = params_2d[original_indices, 0]
        orig_y = params_2d[original_indices, 1]
        orig_z = [rewards[i] for i in original_indices]
        fig.add_trace(go.Scatter3d(
            x=orig_x,
            y=orig_y,
            z=orig_z,
            mode='markers',
            marker=dict(
                size=8,
                color='#000000',  # black for original policy
                opacity=0.8,
                line=dict(width=1, color='white'),
            ),
            name='Original Policy',
            text=[f"Task {all_metadata[i]['task_id']}" for i in original_indices],
            hovertemplate='<b>Original Policy</b><br>' +
                         'Task: %{text}<br>' +
                         't-SNE X: %{x:.2f}<br>' +
                         't-SNE Y: %{y:.2f}<br>' +
                         'Reward: %{z:.2f}<extra></extra>',
        ))
    
    # Plot retrain policies grouped by step_size
    for step_size in sorted(args.step_sizes):
        step_indices = [
            i for i in retrain_indices 
            if all_metadata[i].get("step_size") == step_size
        ]
        if not step_indices:
            continue
        
        retrain_x = params_2d[step_indices, 0]
        retrain_y = params_2d[step_indices, 1]
        retrain_z = [rewards[i] for i in step_indices]
        color = step_size_colors.get(step_size, "#808080")
        
        fig.add_trace(go.Scatter3d(
            x=retrain_x,
            y=retrain_y,
            z=retrain_z,
            mode='markers',
            marker=dict(
                size=6,
                color=color,
                opacity=0.7,
                line=dict(width=0.5, color='white'),
            ),
            name=f'Step Size {step_size}',
            text=[f"Task {all_metadata[i]['task_id']}_jr{all_metadata[i]['jr_index']}" 
                  for i in step_indices],
            hovertemplate=f'<b>Step Size {step_size}</b><br>' +
                         'Task: %{text}<br>' +
                         't-SNE X: %{x:.2f}<br>' +
                         't-SNE Y: %{y:.2f}<br>' +
                         'Reward: %{z:.2f}<extra></extra>',
        ))
    
    # Set layout
    fig.update_layout(
        title=dict(
            text=f'Policy Parameter Space t-SNE Visualization (3D)<br>' +
                 f'<sub>Original Policies: {len(original_indices)}, ' +
                 f'Retrain Policies: {len(retrain_indices)}</sub>',
            x=0.5,
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title='t-SNE Dimension 1',
            yaxis_title='t-SNE Dimension 2',
            zaxis_title='Episode Reward',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
        ),
        width=1200,
        height=800,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1,
        ),
    )
    
    # Save interactive HTML
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.output_html))
    print(f"\n[Saved] Interactive HTML: {args.output_html}")
    
    # Also save static PNG image (using matplotlib)
    print("Creating static PNG image...")
    fig_static = plt.figure(figsize=(14, 10))
    ax = fig_static.add_subplot(111, projection="3d")
    
    # Plot original policies
    if original_indices:
        orig_x = params_2d[original_indices, 0]
        orig_y = params_2d[original_indices, 1]
        orig_z = [rewards[i] for i in original_indices]
        ax.scatter(
            orig_x,
            orig_y,
            orig_z,
            c="black",
            s=100,
            alpha=0.8,
            label="Original Policy",
            edgecolors="white",
            linewidths=1,
        )
    
    # Plot retrain policies grouped by step_size
    for step_size in sorted(args.step_sizes):
        step_indices = [
            i for i in retrain_indices 
            if all_metadata[i].get("step_size") == step_size
        ]
        if not step_indices:
            continue
        
        retrain_x = params_2d[step_indices, 0]
        retrain_y = params_2d[step_indices, 1]
        retrain_z = [rewards[i] for i in step_indices]
        color = step_size_colors.get(step_size, "#808080")
        
        ax.scatter(
            retrain_x,
            retrain_y,
            retrain_z,
            c=color,
            s=50,
            alpha=0.7,
            label=f"Step Size {step_size}",
            edgecolors="white",
            linewidths=0.5,
        )
    
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_zlabel("Episode Reward", fontsize=12)
    ax.set_title(
        "Policy Parameter Space t-SNE Visualization (3D)\n"
        f"Original Policies: {len(original_indices)}, "
        f"Retrain Policies: {len(retrain_indices)}",
        fontsize=14,
    )
    ax.legend(loc='upper left')
    
    # Save static image
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_static.savefig(args.output_path, dpi=200, bbox_inches="tight")
    plt.close(fig_static)
    print(f"[Saved] Static PNG: {args.output_path}")
    
    # Save data (optional)
    data_output = args.output_path.with_suffix(".npz")
    np.savez(
        data_output,
        params_2d=params_2d,
        rewards=np.array(rewards),
        metadata=all_metadata,
    )
    print(f"[Saved] Data file: {data_output}")


if __name__ == "__main__":
    main()

