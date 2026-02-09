#!/usr/bin/env python3
"""
Visualize policy behavior space using t-SNE:
- Collect representative states using original policies
- Compute behavior vectors for all policies (action distributions on representative states)
- Apply t-SNE to reduce behavior space to 2D
- Visualize in 3D: (t-SNE_x, t-SNE_y, episode_reward)
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

# Import pufferlib - this may fail if not in the correct environment
# The script should be run in an environment where pufferlib is properly installed
try:
    import pufferlib.pufferl as pufferl
    from train_landscape_controlled import load_config_manually
    from record_policy_video import load_vec_stats
except ImportError as e:
    print(f"[Error] Failed to import pufferlib: {e}")
    print("This script requires pufferlib to be properly installed.")
    print("Please run this script in the correct environment where pufferlib is available.")
    raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize policy behavior space using t-SNE"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Root path containing controlled_task_* directories",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("landscape_data"),
        help="Root path containing features_task_*.json landscape data",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("landscape_plots/behavior_space_tsne_3d.png"),
        help="Path to save PNG image",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("landscape_plots/behavior_space_tsne_3d.html"),
        help="Path to save interactive HTML",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=list(range(16, 32)),
        help="List of HalfCheetah controlled_task IDs to process",
    )
    parser.add_argument(
        "--step-sizes",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 15.0, 45.0, 135.0],
        help="List of step sizes to plot",
    )
    parser.add_argument(
        "--n-episodes-per-policy",
        type=int,
        default=10,
        help="Number of episodes to collect states per original policy",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--n-representative-states",
        type=int,
        default=1000,
        help="Number of representative states to sample",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use when loading models",
    )
    parser.add_argument(
        "--retrain-models-dir",
        type=Path,
        default=None,
        help="Directory containing retrained model files (e.g., jump_retrain_all)",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="HalfCheetah-v4",
        help="Environment name",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity parameter (default: 30.0)",
    )
    parser.add_argument(
        "--tsne-random-state",
        type=int,
        default=42,
        help="t-SNE random seed (default: 42)",
    )
    parser.add_argument(
        "--include-retrain-in-states",
        action="store_true",
        help="Include retrain policies when collecting representative states (for broader coverage)",
    )
    return parser.parse_args()


FILENAME_REGEX = re.compile(
    r"^features_task_(?P<label>\d+_jr\d+_s(?P<step>[\d\.]+))_step_(?P<idx>\d+)\.json$"
)
LABEL_REGEX = re.compile(r"^(?P<task>\d+)_jr(?P<jr>\d+)_s(?P<step>[\d\.]+)$")


def load_policy(model_path: Path, env_name: str, device: str = "cpu"):
    """Load a policy from model file"""
    args = load_config_manually(env_name)
    # Use CPU for environment, but GPU for policy if available
    args['train']['device'] = device if device == "cuda" else "cpu"
    args['vec']['backend'] = 'Serial'
    args['vec']['num_envs'] = 1
    args['vec']['num_workers'] = 1
    args['vec']['batch_size'] = 1
    
    vecenv = pufferl.load_env(env_name, args)
    policy = pufferl.load_policy(args, vecenv, env_name)
    
    if model_path.exists():
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


def collect_representative_states(
    orig_policies: List[Tuple[Path, int]],
    env_name: str,
    n_episodes_per_policy: int,
    max_steps_per_episode: int,
    n_representative_states: int,
    device: str = "cpu",
    include_retrain: bool = False,
    retrain_policies: Optional[List[Tuple[Path, int, int, float]]] = None,
) -> np.ndarray:
    """
    Step 1: Collect representative states using original policies (and optionally retrain policies)
    
    Args:
        include_retrain: If True, also collect states from retrain policies to get broader coverage.
                         Default is False (use only original policies).
    """
    print(f"\n[Step 1] Collecting representative states...")
    if include_retrain and retrain_policies:
        print(f"  Using {len(orig_policies)} original policies + {len(retrain_policies)} retrain policies")
    else:
        print(f"  Using {len(orig_policies)} original policies only")
    
    buffer_states = []
    
    # Collect from original policies
    for model_path, task_id in orig_policies:
        print(f"  Collecting states from task {task_id}...")
        try:
            policy, vecenv = load_policy(model_path, env_name, device)
            
            # Load vec_stats if available
            vec_stats_path = model_path.parent / "vec_stats.npz"
            if vec_stats_path.exists():
                load_vec_stats(vecenv, vec_stats_path)
            
            for episode in range(n_episodes_per_policy):
                obs, _ = vecenv.reset(seed=task_id * 1000 + episode)
                lstm_h, lstm_c = None, None
                
                for step in range(max_steps_per_episode):
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    state_dict = {'lstm_h': lstm_h, 'lstm_c': lstm_c}
                    
                    with torch.no_grad():
                        action_dist, _ = policy.forward_eval(obs_tensor, state_dict)
                        lstm_h = state_dict.get('lstm_h', None)
                        lstm_c = state_dict.get('lstm_c', None)
                    
                    # Store state
                    buffer_states.append(obs.copy().flatten())
                    
                    # Sample action (use mean for deterministic behavior)
                    if hasattr(action_dist, 'mean'):
                        action = action_dist.mean
                    elif hasattr(action_dist, 'sample'):
                        action = action_dist.sample()
                    else:
                        # Fallback: try to get action from distribution
                        action = action_dist
                    
                    vecenv.send(action.detach().cpu().numpy())
                    obs, _, _, _, _, _, done = vecenv.recv()
                    
                    if done.any():
                        break
            
            vecenv.close()
        except Exception as e:
            print(f"  [Warning] Failed to collect states from task {task_id}: {e}")
            continue
    
    # Optionally collect from retrain policies for broader state coverage
    if include_retrain and retrain_policies:
        print(f"  Collecting states from retrain policies...")
        # Sample a subset of retrain policies to avoid too many states
        n_retrain_samples = min(len(retrain_policies), 20)  # Sample up to 20 retrain policies
        sampled_retrain = np.random.choice(len(retrain_policies), n_retrain_samples, replace=False)
        
        for idx in sampled_retrain:
            model_path, task_id, jr_index, step_size = retrain_policies[idx]
            print(f"    Collecting states from task {task_id}_jr{jr_index}_s{step_size}...")
            try:
                policy, vecenv = load_policy(model_path, env_name, device)
                vec_stats_path = model_path.parent / "vec_stats.npz"
                if vec_stats_path.exists():
                    load_vec_stats(vecenv, vec_stats_path)
                
                # Collect fewer episodes from retrain policies
                for episode in range(max(1, n_episodes_per_policy // 2)):
                    obs, _ = vecenv.reset(seed=task_id * 10000 + jr_index * 100 + episode)
                    lstm_h, lstm_c = None, None
                    
                    for step in range(max_steps_per_episode):
                        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                        state_dict = {'lstm_h': lstm_h, 'lstm_c': lstm_c}
                        
                        with torch.no_grad():
                            action_dist, _ = policy.forward_eval(obs_tensor, state_dict)
                            lstm_h = state_dict.get('lstm_h', None)
                            lstm_c = state_dict.get('lstm_c', None)
                        
                        buffer_states.append(obs.copy().flatten())
                        
                        if hasattr(action_dist, 'mean'):
                            action = action_dist.mean
                        elif hasattr(action_dist, 'sample'):
                            action = action_dist.sample()
                        else:
                            action = action_dist
                        
                        vecenv.send(action.detach().cpu().numpy())
                        obs, _, _, _, _, _, done = vecenv.recv()
                        
                        if done.any():
                            break
                
                vecenv.close()
            except Exception as e:
                print(f"    [Warning] Failed to collect states from retrain policy: {e}")
            continue
    
    print(f"  Collected {len(buffer_states)} states total")
    
    # Sample representative states
    if len(buffer_states) > n_representative_states:
        indices = np.random.choice(len(buffer_states), n_representative_states, replace=False)
        representative_states = np.array([buffer_states[i] for i in indices])
    else:
        representative_states = np.array(buffer_states)
    
    print(f"  Selected {len(representative_states)} representative states")
    return representative_states


def compute_behavior_vector(
    policy,
    vecenv,
    representative_states: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Compute behavior vector for a policy on representative states"""
    # Note: pufferl and load_vec_stats are already imported in calling functions
    behavior_features = []
    
    # Use GPU for policy inference if available, but keep states on CPU for batch processing
    policy_device = next(policy.parameters()).device
    
    for state in representative_states:
        obs_tensor = torch.as_tensor(state, dtype=torch.float32, device=policy_device).unsqueeze(0)
        state_dict = {'lstm_h': None, 'lstm_c': None}
        
        with torch.no_grad():
            action_dist, _ = policy.forward_eval(obs_tensor, state_dict)
        
        # Extract distribution parameters
        if hasattr(action_dist, 'mean') and hasattr(action_dist, 'log_std'):
            # Continuous action: Normal distribution
            mean = action_dist.mean.cpu().numpy().flatten()
            log_std = action_dist.log_std.cpu().numpy().flatten()
            behavior_features.append(np.concatenate([mean, log_std]))
        elif hasattr(action_dist, 'mean'):
            # Only mean available
            mean = action_dist.mean.cpu().numpy().flatten()
            behavior_features.append(mean)
        elif hasattr(action_dist, 'probs'):
            # Discrete action: categorical distribution
            probs = action_dist.probs.cpu().numpy().flatten()
            behavior_features.append(probs)
        else:
            # Fallback: try to get logits or sample
            if hasattr(action_dist, 'logits'):
                logits = action_dist.logits.cpu().numpy().flatten()
                behavior_features.append(logits)
            else:
                # Last resort: sample and use as feature
                sample = action_dist.sample().cpu().numpy().flatten()
                behavior_features.append(sample)
    
    return np.concatenate(behavior_features)


def get_episode_reward_from_features(
    data_dir: Path,
    task_id: int,
    jr_index: Optional[int],
    step_size: Optional[float],
) -> Optional[float]:
    """Get episode reward from feature files"""
    if jr_index is None:
        pattern = f"features_task_{task_id}_step_*.json"
        files = [f for f in data_dir.glob(pattern) if "_jr" not in f.name]
    else:
        label = f"{task_id}_jr{jr_index}_s{step_size:.2f}"
        pattern = f"features_task_{label}_step_*.json"
        files = list(data_dir.glob(pattern))
    
    if not files:
        return None
    
    step_numbers = []
    for f in files:
        match = re.search(r"_step_(\d+)\.json$", f.name)
        if match:
            step_numbers.append((int(match.group(1)), f))
    
    if not step_numbers:
        return None
    
    _, latest_file = max(step_numbers, key=lambda x: x[0])
    
    try:
        with latest_file.open() as f:
            data = json.load(f)
        return data.get("episode_reward")
    except Exception as e:
        return None


def get_episode_reward_from_best_info(models_dir: Path, task_id: int) -> Optional[float]:
    """Get episode reward from best_info.json"""
    task_dir = models_dir / f"controlled_task_{task_id}"
    best_info_path = task_dir / "best_info.json"
    
    if not best_info_path.exists():
        return None
    
    try:
        with best_info_path.open() as f:
            data = json.load(f)
        return data.get("best_reward")
    except:
        return None


def main() -> None:
    global args
    args = parse_args()
    
    # Step 0: Prepare policy collection
    print("[Step 0] Preparing policy collection...")
    
    orig_policies = []
    for task_id in args.task_ids:
        task_dir = args.models_dir / f"controlled_task_{task_id}"
        model_path = task_dir / "best_model.pt"
        if not model_path.exists():
            model_path = task_dir / "final_model.pt"
        if model_path.exists():
            orig_policies.append((model_path, task_id))
    
    print(f"  Found {len(orig_policies)} original policies")
    
    retrain_policies = []
    retrain_metadata = []
    
    if args.retrain_models_dir and args.retrain_models_dir.exists():
        print(f"  Scanning retrain models from {args.retrain_models_dir}...")
        retrain_model_dirs = list(args.retrain_models_dir.glob("controlled_task_*_jr*_s0.50"))
        
        for model_dir in sorted(retrain_model_dirs):
            match = re.match(r"controlled_task_(\d+)_jr(\d+)_s0\.50", model_dir.name)
            if not match:
                continue
            
            task_id = int(match.group(1))
            jr_index = int(match.group(2))
            
            if task_id not in args.task_ids:
                continue
            
            model_path = model_dir / "final_model.pt"
            if model_path.exists():
                # Find step_size from feature files
                step_size = None
                for path in args.data_dir.glob(f"features_task_{task_id}_jr{jr_index}_s*_step_*.json"):
                    match_feat = FILENAME_REGEX.match(path.name)
                    if match_feat:
                        label = match_feat.group("label")
                        label_match = LABEL_REGEX.match(label)
                        if label_match:
                            raw_step = float(label_match.group("step"))
                            for target_step in args.step_sizes:
                                if abs(raw_step - target_step) <= 1e-3:
                                    step_size = target_step
                                    break
                    if step_size is not None:
                        break
                
                if step_size is not None:
                    retrain_policies.append((model_path, task_id, jr_index, step_size))
                    episode_reward = get_episode_reward_from_features(
                        args.data_dir, task_id, jr_index, step_size
                    )
                    retrain_metadata.append({
                        "task_id": task_id,
                        "jr_index": jr_index,
                        "step_size": step_size,
                        "episode_reward": episode_reward,
                    })
    
    print(f"  Found {len(retrain_policies)} retrain policies")
    
    # Step 1: Collect representative states
    # Option 1: Use only original policies (current approach - may have bias)
    # Option 2: Include retrain policies for broader state coverage
    representative_states = collect_representative_states(
        orig_policies,
        args.env_name,
        args.n_episodes_per_policy,
        args.max_steps_per_episode,
        args.n_representative_states,
        args.device,
        include_retrain=args.include_retrain_in_states,
        retrain_policies=retrain_policies if args.include_retrain_in_states else None,
    )
    
    # Step 2: Compute behavior vectors for all policies
    print(f"\n[Step 2] Computing behavior vectors for {len(orig_policies) + len(retrain_policies)} policies...")
    
    behavior_vectors = []
    all_rewards = []
    all_metadata = []
    
    # Original policies
    for model_path, task_id in orig_policies:
        print(f"  Computing behavior vector for original policy task {task_id}...")
        try:
            policy, vecenv = load_policy(model_path, args.env_name, args.device)
            vec_stats_path = model_path.parent / "vec_stats.npz"
            if vec_stats_path.exists():
                load_vec_stats(vecenv, vec_stats_path)
            
            behavior_vec = compute_behavior_vector(
                policy, vecenv, representative_states, args.device
            )
            behavior_vectors.append(behavior_vec)
            
            reward = get_episode_reward_from_features(args.data_dir, task_id, None, None)
            if reward is None:
                reward = get_episode_reward_from_best_info(args.models_dir, task_id)
            
            all_rewards.append(reward)
            all_metadata.append({
                "task_id": task_id,
                "type": "original",
                "step_size": None,
            })
            
            vecenv.close()
        except Exception as e:
            print(f"  [Warning] Failed to compute behavior vector for task {task_id}: {e}")
    
    # Retrain policies
    for model_path, task_id, jr_index, step_size in retrain_policies:
        print(f"  Computing behavior vector for retrain policy task {task_id}_jr{jr_index}_s{step_size}...")
        try:
            policy, vecenv = load_policy(model_path, args.env_name, args.device)
            vec_stats_path = model_path.parent / "vec_stats.npz"
            if vec_stats_path.exists():
                load_vec_stats(vecenv, vec_stats_path)
            
            behavior_vec = compute_behavior_vector(
                policy, vecenv, representative_states, args.device
            )
            behavior_vectors.append(behavior_vec)
            
            reward = get_episode_reward_from_features(
                args.data_dir, task_id, jr_index, step_size
            )
            all_rewards.append(reward)
            all_metadata.append({
                "task_id": task_id,
                "type": "retrain",
                "step_size": step_size,
            })
            
            vecenv.close()
        except Exception as e:
            print(f"  [Warning] Failed to compute behavior vector for task {task_id}_jr{jr_index}: {e}")
    
    if len(behavior_vectors) == 0:
        print("[Error] No behavior vectors computed")
        return
    
    # Stack behavior vectors
    behavior_matrix = np.vstack(behavior_vectors)
    print(f"  Behavior matrix shape: {behavior_matrix.shape}")
    
    # Step 3: Apply t-SNE
    print(f"\n[Step 3] Applying t-SNE to reduce behavior space to 2D...")
    tsne_perplexity = getattr(args, 'tsne_perplexity', 30.0)
    tsne_random_state = getattr(args, 'tsne_random_state', 42)
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        random_state=tsne_random_state,
        max_iter=1000,
        verbose=1,
    )
    behavior_2d = tsne.fit_transform(behavior_matrix)
    print(f"  Reduced dimension shape: {behavior_2d.shape}")
    
    # Filter valid data points
    valid_indices = [i for i, r in enumerate(all_rewards) if r is not None]
    print(f"\nValid data points: {len(valid_indices)}/{len(all_rewards)}")
    
    # Step 4: Visualize
    print(f"\n[Step 4] Creating 3D visualization...")
    
    # Define colors
    step_size_colors = {
        0.0: "#FF0000",    # Red
        5.0: "#00FF00",    # Green
        15.0: "#0000FF",   # Blue
        45.0: "#FF00FF",   # Magenta
        135.0: "#FFA500",  # Orange
    }
    
    original_indices = [i for i in valid_indices if all_metadata[i]["type"] == "original"]
    retrain_indices = [i for i in valid_indices if all_metadata[i]["type"] == "retrain"]
    
    # Create interactive plot
    fig = go.Figure()
    
    # Original policies
    if original_indices:
        orig_x = behavior_2d[original_indices, 0]
        orig_y = behavior_2d[original_indices, 1]
        orig_z = [all_rewards[i] for i in original_indices]
        orig_text = [f"Task {all_metadata[i]['task_id']}" for i in original_indices]
        fig.add_trace(go.Scatter3d(
            x=orig_x,
            y=orig_y,
            z=orig_z,
            mode='markers',
            marker=dict(size=8, color='#000000', opacity=0.8, line=dict(width=1, color='white')),
            name='Original Policy',
            text=orig_text,
            hovertemplate='<b>Original Policy</b><br>Task: %{text}<br>t-SNE X: %{x:.2f}<br>t-SNE Y: %{y:.2f}<br>Reward: %{z:.2f}<extra></extra>',
        ))
    
    # Retrain policies by step size
    for step_size in sorted(args.step_sizes):
        step_indices = [
            i for i in retrain_indices 
            if all_metadata[i].get("step_size") == step_size
        ]
        if not step_indices:
            continue
        
        retrain_x = behavior_2d[step_indices, 0]
        retrain_y = behavior_2d[step_indices, 1]
        retrain_z = [all_rewards[i] for i in step_indices]
        color = step_size_colors.get(step_size, "#808080")
        
        retrain_text = [f"Task {all_metadata[i]['task_id']}_jr{all_metadata[i].get('jr_index', '?')}" 
                        for i in step_indices]
        fig.add_trace(go.Scatter3d(
            x=retrain_x,
            y=retrain_y,
            z=retrain_z,
            mode='markers',
            marker=dict(size=6, color=color, opacity=0.7, line=dict(width=0.5, color='white')),
            name=f'Step Size {step_size}',
            text=retrain_text,
            hovertemplate=f'<b>Step Size {step_size}</b><br>Task: %{{text}}<br>t-SNE X: %{{x:.2f}}<br>t-SNE Y: %{{y:.2f}}<br>Reward: %{{z:.2f}}<extra></extra>',
        ))
    
    fig.update_layout(
        title=dict(
            text=f'Policy Behavior Space t-SNE Visualization (3D)<br>' +
                 f'<sub>Original Policies: {len(original_indices)}, ' +
                 f'Retrain Policies: {len(retrain_indices)}</sub>',
            x=0.5,
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title='t-SNE Dimension 1',
            yaxis_title='t-SNE Dimension 2',
            zaxis_title='Episode Reward',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        width=1200,
        height=800,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.8)', bordercolor='black', borderwidth=1),
    )
    
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.output_html))
    print(f"[Saved] Interactive HTML: {args.output_html}")
    
    # Static PNG
    fig_static = plt.figure(figsize=(14, 10))
    ax = fig_static.add_subplot(111, projection="3d")
    
    if original_indices:
        orig_x = behavior_2d[original_indices, 0]
        orig_y = behavior_2d[original_indices, 1]
        orig_z = [all_rewards[i] for i in original_indices]
        ax.scatter(orig_x, orig_y, orig_z, c="black", s=100, alpha=0.8, 
                  label="Original Policy", edgecolors="white", linewidths=1)
    
    for step_size in sorted(args.step_sizes):
        step_indices = [i for i in retrain_indices if all_metadata[i].get("step_size") == step_size]
        if not step_indices:
            continue
        retrain_x = behavior_2d[step_indices, 0]
        retrain_y = behavior_2d[step_indices, 1]
        retrain_z = [all_rewards[i] for i in step_indices]
        color = step_size_colors.get(step_size, "#808080")
        ax.scatter(retrain_x, retrain_y, retrain_z, c=color, s=50, alpha=0.7,
                  label=f"Step Size {step_size}", edgecolors="white", linewidths=0.5)
    
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_zlabel("Episode Reward", fontsize=12)
    ax.set_title(
        f"Policy Behavior Space t-SNE Visualization (3D)\n"
        f"Original Policies: {len(original_indices)}, Retrain Policies: {len(retrain_indices)}",
        fontsize=14,
    )
    ax.legend(loc='upper left')
    
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_static.savefig(args.output_path, dpi=200, bbox_inches="tight")
    plt.close(fig_static)
    print(f"[Saved] Static PNG: {args.output_path}")
    
    # Save data
    data_output = args.output_path.with_suffix(".npz")
    np.savez(
        data_output,
        behavior_2d=behavior_2d,
        rewards=np.array(all_rewards),
        metadata=all_metadata,
        representative_states=representative_states,
    )
    print(f"[Saved] Data file: {data_output}")


if __name__ == "__main__":
    main()

