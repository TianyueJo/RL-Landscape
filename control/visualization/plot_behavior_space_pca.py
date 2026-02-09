#!/usr/bin/env python3
"""
Visualize policy behavior space using PCA:
- Collect representative states using original policies
- Compute behavior vectors for all policies (action distributions on representative states)
- Apply PCA to reduce behavior space to 2D
- Visualize in 3D: (PCA_x, PCA_y, episode_reward)
"""

import sys
from pathlib import Path

# Ensure the control/ directory is on PYTHONPATH when running from subfolders
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import networkx as nx
from collections import defaultdict

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

# Import PCA and L2 distance computation
from compute_behavior_space_pca_L2_matrix import pca_and_pairwise_l2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize policy behavior space using PCA"
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
        default=Path("landscape_plots/behavior_space_pca_3d.png"),
        help="Path to save PNG image",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("landscape_plots/behavior_space_pca_3d.html"),
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
        "--visualize-graph",
        action="store_true",
        help="Visualize policy similarity as a graph based on L2 distance in PCA space",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=None,
        help="Distance threshold for graph edges (if None, will use median distance)",
    )
    parser.add_argument(
        "--pca-dims",
        type=int,
        nargs="+",
        default=[2, 6, 10],
        help="PCA dimensions to compute for graph visualization",
    )
    parser.add_argument(
        "--graph-layout",
        type=str,
        default="spring",
        choices=["spring", "circular", "kamada_kawai", "spectral"],
        help="Graph layout algorithm",
    )
    parser.add_argument(
        "--separate-components",
        action="store_true",
        help="Draw each connected component in a separate subplot",
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
) -> np.ndarray:
    """Step 1: Collect representative states using original policies"""
    print(f"\n[Step 1] Collecting representative states using {len(orig_policies)} original policies...")
    
    buffer_states = []
    
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


def build_graph_from_distance_matrix(
    distance_matrix: np.ndarray,
    threshold: float,
    metadata: List[Dict],
) -> Tuple[nx.Graph, List]:
    """
    Build a graph from an L2 distance matrix.
    
    Args:
        distance_matrix: L2 distance matrix (symmetric, diagonal is 0)
        threshold: Add an edge if distance < threshold
        metadata: List of policy metadata dicts
    
    Returns:
        G: networkx Graph
        edges: List of edges [(node_i, node_j, distance), ...]
    """
    n = len(distance_matrix)
    G = nx.Graph()
    
    # Add nodes; use metadata to build labels
    for i, meta in enumerate(metadata):
        if meta.get("type") == "original":
            node_label = f"T{meta['task_id']}"
        else:
            node_label = f"T{meta['task_id']}_jr{meta.get('jr_index', '?')}_s{meta.get('step_size', '?')}"
        G.add_node(i, label=node_label, metadata=meta)
    
    # Add edges: if distance < threshold, connect nodes
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] < threshold:
                G.add_edge(i, j, 
                          weight=distance_matrix[i, j],
                          distance=distance_matrix[i, j])
                edges.append((i, j, distance_matrix[i, j]))
    
    return G, edges


def visualize_behavior_graph(
    G: nx.Graph,
    env_name: str,
    output_path: Path,
    threshold: float,
    layout: str = 'spring',
    separate_components: bool = False,
    distance_matrix: np.ndarray = None,
    pca_dim: int = None,
):
    """
    Visualize the behavior-space graph structure.
    
    Args:
        G: networkx Graph
        env_name: Environment name
        output_path: Output path
        threshold: Threshold used to build the graph
        layout: Layout algorithm
        separate_components: Whether to draw connected components separately
        distance_matrix: Distance matrix (used for summary stats)
    """
    num_components = nx.number_connected_components(G)
    
    if separate_components and num_components > 1:
        # Create one subplot per connected component
        components = list(nx.connected_components(G))
        n_components = len(components)
        
        n_cols = int(np.ceil(np.sqrt(n_components)))
        n_rows = int(np.ceil(n_components / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_components == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        component_colors = plt.cm.tab20(np.linspace(0, 1, min(n_components, 20)))
        if n_components > 20:
            component_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        for idx, component in enumerate(components):
            ax = axes[idx] if n_components > 1 else axes[0]
            subgraph = G.subgraph(component)
            
            if layout == 'spring':
                pos = nx.spring_layout(subgraph, k=1.5, iterations=50, seed=42)
            elif layout == 'circular':
                pos = nx.circular_layout(subgraph)
            elif layout == 'kamada_kawai':
                try:
                    pos = nx.kamada_kawai_layout(subgraph)
                except:
                    pos = nx.spring_layout(subgraph, seed=42)
            else:
                pos = nx.spring_layout(subgraph, seed=42)
            
            edges = subgraph.edges()
            if edges:
                edge_weights = [subgraph[u][v].get('distance', 1.0) for u, v in edges]
                max_weight = max(edge_weights) if edge_weights else 1.0
                edge_colors = plt.cm.viridis_r([w / max_weight if max_weight > 0 else 0.5 
                                               for w in edge_weights])
                edge_widths = [3.0 + 5.0 * (1.0 - w / max_weight) if max_weight > 0 else 5.0 
                              for w in edge_weights]
                
                nx.draw_networkx_edges(subgraph, pos, 
                                      edge_color=edge_colors,
                                      width=edge_widths,
                                      alpha=0.7,
                                      style='solid',
                                      ax=ax)
            
            node_color = component_colors[idx % len(component_colors)]
            nx.draw_networkx_nodes(subgraph, pos,
                                  node_color=[node_color],
                                  node_size=2000,
                                  alpha=0.9,
                                  edgecolors='black',
                                  linewidths=2,
                                  ax=ax)
            
            labels = {node: subgraph.nodes[node].get('label', str(node)) for node in subgraph.nodes()}
            nx.draw_networkx_labels(subgraph, pos, labels,
                                   font_size=10,
                                   font_weight='bold',
                                   font_color='white',
                                   ax=ax)
            
            component_list = sorted([subgraph.nodes[node].get('label', str(node)) for node in component])
            ax.set_title(f"Component {idx+1}\nNodes: {component_list}\nSize: {len(component)}", 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
        
        for idx in range(n_components, len(axes)):
            axes[idx].axis('off')
        
        pca_text = f"PCA Dimension: {pca_dim} | " if pca_dim else ""
        fig.suptitle(f"{env_name} Behavior Space Graph - Separate Components\n"
                    f"{pca_text}Threshold: {threshold:.4f} | "
                    f"Total Nodes: {G.number_of_nodes()} | "
                    f"Total Edges: {G.number_of_edges()} | "
                    f"Components: {num_components}",
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {output_path}")
        plt.close()
        return
    
    # Unified view
    # Use a larger figure to accommodate left/right colorbars
    fig = plt.figure(figsize=(16, 10))
    
    if layout == 'spring':
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.spring_layout(G, seed=42)
    elif layout == 'spectral':
        try:
            pos = nx.spectral_layout(G)
        except:
            pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Use GridSpec for better colorbar layout control
    from matplotlib.gridspec import GridSpec
    
    # Create main plot area and colorbar areas
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.06, 0.88, 0.06], hspace=0.1, wspace=0.05)
    
    # Main plot area (center)
    ax_main = plt.subplot(gs[0, 1])
    
    # Compute edge colors and widths
    edges = G.edges()
    if edges:
        edge_weights = [G[u][v].get('distance', 1.0) for u, v in edges]
        max_weight = max(edge_weights) if edge_weights else 1.0
        edge_colors = plt.cm.viridis_r([w / max_weight if max_weight > 0 else 0.5 
                                       for w in edge_weights])
        edge_widths = [3.0 + 5.0 * (1.0 - w / max_weight) if max_weight > 0 else 5.0 
                      for w in edge_weights]
        
        nx.draw_networkx_edges(G, pos, 
                              edge_color=edge_colors,
                              width=edge_widths,
                              alpha=0.7,
                              style='solid',
                              ax=ax_main)
    else:
        print("Warning: no edges (all pairwise distances exceed the threshold).")
    
    # Set node colors based on reward (if available)
    if use_reward_colors:
        node_rewards = [G.nodes[node].get('reward') for node in G.nodes()]
        if any(r is not None for r in node_rewards):
            # Color by reward
            valid_rewards = [r for r in node_rewards if r is not None]
            if valid_rewards:
                min_reward = min(valid_rewards)
                max_reward = max(valid_rewards)
                norm = plt.Normalize(vmin=min_reward, vmax=max_reward)
                cmap = plt.cm.viridis  # viridis colormap
                
                node_colors = []
                for node in G.nodes():
                    reward = G.nodes[node].get('reward')
                    if reward is not None:
                        node_colors.append(cmap(norm(reward)))
                    else:
                        node_colors.append('gray')
            else:
                # If no valid rewards, color by connected component
                components = list(nx.connected_components(G))
                component_colors = plt.cm.tab20(np.linspace(0, 1, min(len(components), 20)))
                node_to_color = {}
                for idx, component in enumerate(components):
                    color = component_colors[idx % len(component_colors)]
                    for node in component:
                        node_to_color[node] = color
                node_colors = [node_to_color[node] for node in G.nodes()]
        else:
            # If reward info missing, color by connected component
            components = list(nx.connected_components(G))
            component_colors = plt.cm.tab20(np.linspace(0, 1, min(len(components), 20)))
            node_to_color = {}
            for idx, component in enumerate(components):
                color = component_colors[idx % len(component_colors)]
                for node in component:
                    node_to_color[node] = color
            node_colors = [node_to_color[node] for node in G.nodes()]
    else:
        # Color by connected component
        components = list(nx.connected_components(G))
        component_colors = plt.cm.tab20(np.linspace(0, 1, min(len(components), 20)))
        node_to_color = {}
        for idx, component in enumerate(components):
            color = component_colors[idx % len(component_colors)]
            for node in component:
                node_to_color[node] = color
        node_colors = [node_to_color[node] for node in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=1500,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=2,
                          ax=ax_main)
    
    # Draw labels
    labels = {node: G.nodes[node].get('label', str(node)) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels,
                           font_size=10,
                           font_weight='bold',
                           font_color='black',
                           ax=ax_main)
    
    # Title
    pca_text = f"PCA Dimension: {pca_dim} | " if pca_dim else ""
    title = f"{env_name} Behavior Space Graph (L2 Distance in PCA{pca_dim if pca_dim else ''} Space)\n"
    title += f"{pca_text}Threshold: {threshold:.4f} | "
    title += f"Nodes: {G.number_of_nodes()} | "
    title += f"Edges: {G.number_of_edges()} | "
    title += f"Connected Components: {num_components}"
    
    ax_main.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax_main.axis('off')
    
    # Left colorbar: edge similarity (L2 distance)
    if edges:
        edge_weights_list = [G[u][v].get('distance', 1.0) for u, v in edges]
        ax_cbar_left = plt.subplot(gs[0, 0])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r, 
                                   norm=plt.Normalize(vmin=min(edge_weights_list), 
                                                     vmax=max(edge_weights_list)))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=ax_cbar_left, orientation='vertical')
        cbar.set_label('Edge Similarity\n(L2 Distance)', rotation=90, labelpad=20, fontsize=11, fontweight='bold')
        # Ensure ticks are shown and formatted
        from matplotlib.ticker import MaxNLocator
        tick_locator = MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        # Ensure tick labels are visible
        cbar.ax.tick_params(labelsize=10, which='major', direction='out', length=4, width=1, pad=5)
        cbar.ax.yaxis.set_visible(True)
    
    # Right colorbar: node performance (training reward)
    if use_reward_colors:
        node_rewards = [G.nodes[node].get('reward') for node in G.nodes()]
        valid_rewards = [r for r in node_rewards if r is not None]
        if valid_rewards:
            ax_cbar_right = plt.subplot(gs[0, 2])
            sm_reward = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                             norm=plt.Normalize(vmin=min(valid_rewards),
                                                               vmax=max(valid_rewards)))
            sm_reward.set_array([])
            cbar_reward = plt.colorbar(sm_reward, cax=ax_cbar_right, orientation='vertical')
            cbar_reward.set_label('Node Performance\n(Training Reward)', rotation=90, labelpad=20, fontsize=11, fontweight='bold')
            
            # Ticks (more reliable approach)
            from matplotlib.ticker import MaxNLocator, FuncFormatter
            tick_locator = MaxNLocator(nbins=5)
            cbar_reward.locator = tick_locator
            cbar_reward.update_ticks()
            
            # Ensure tick labels are visible
            cbar_reward.ax.yaxis.set_major_locator(tick_locator)
            cbar_reward.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}'))
            cbar_reward.ax.tick_params(labelsize=10, which='major', direction='out', length=4, width=1, pad=5)
            cbar_reward.ax.yaxis.set_visible(True)
            
            # Ensure the colorbar and ticks are visible
            ax_cbar_right.set_visible(True)
            # Force a refresh of tick labels
            cbar_reward.ax.yaxis.set_tick_params(which='major', labelsize=10)
        else:
            # If no valid rewards, keep an empty area to preserve layout
            ax_cbar_right = plt.subplot(gs[0, 2])
            ax_cbar_right.set_visible(False)
    else:
        # If use_reward_colors=False, keep an empty area to preserve layout
        ax_cbar_right = plt.subplot(gs[0, 2])
        ax_cbar_right.set_visible(False)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")
    plt.close()


def print_behavior_graph_statistics(G, distance_matrix, threshold, metadata):
    """Print behavior-space graph statistics."""
    print("\n" + "="*70)
    print("Behavior-space graph statistics")
    print("="*70)
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Connected components: {nx.number_connected_components(G)}")
    print(f"Threshold: {threshold:.4f}")
    
    edges = G.edges()
    if edges:
        edge_distances = [G[u][v].get('distance', 0.0) for u, v in edges]
        print("\nEdge L2 distance stats:")
        print(f"  Min: {min(edge_distances):.4f}")
        print(f"  Max: {max(edge_distances):.4f}")
        print(f"  Mean: {np.mean(edge_distances):.4f}")
        print(f"  Median: {np.median(edge_distances):.4f}")
    
    degrees = [G.degree(node) for node in G.nodes()]
    print("\nNode degree stats:")
    print(f"  Min degree: {min(degrees)}")
    print(f"  Max degree: {max(degrees)}")
    print(f"  Mean degree: {np.mean(degrees):.2f}")
    
    components = list(nx.connected_components(G))
    print("\nConnected components:")
    for idx, component in enumerate(components, 1):
        component_labels = [G.nodes[node].get('label', str(node)) for node in component]
        print(f"  Component {idx}: {sorted(component_labels)} (size: {len(component)})")
    
    if edges:
        print(f"\nAll edges (total {len(edges)}):")
        sorted_edges = sorted(edges, key=lambda e: G[e[0]][e[1]].get('distance', float('inf')))
        for u, v in sorted_edges[:20]:  # show first 20
            dist = G[u][v].get('distance', 0.0)
            label_u = G.nodes[u].get('label', str(u))
            label_v = G.nodes[v].get('label', str(v))
            print(f"  {label_u} <-> {label_v}: L2 distance = {dist:.4f}")
        if len(edges) > 20:
            print(f"  ... ({len(edges) - 20} more edges)")
    print("="*70)


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
    representative_states = collect_representative_states(
        orig_policies,
        args.env_name,
        args.n_episodes_per_policy,
        args.max_steps_per_episode,
        args.n_representative_states,
        args.device,
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
    
    # Step 3: Apply PCA and compute L2 distance matrices
    print(f"\n[Step 3] Applying PCA and computing L2 distance matrices...")
    
    # Compute PCA embeddings and L2 distance matrices for multiple dimensions
    embeddings_dict, distances_dict, pca_dict = pca_and_pairwise_l2(
        behavior_matrix,
        dims=args.pca_dims,
        center=True,
    )
    
    # For visualization, use 2D PCA
    pca = PCA(n_components=2, random_state=42)
    behavior_2d = pca.fit_transform(behavior_matrix)
    print(f"  2D PCA shape: {behavior_2d.shape}")
    print(f"  2D Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"  2D Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Print distance statistics for each dimension
    for k in args.pca_dims:
        dist_mat = distances_dict[k]
        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices_from(dist_mat, k=1)
        distances = dist_mat[triu_indices]
        print(f"  PCA{k} L2 Distance range: [{distances.min():.4f}, {distances.max():.4f}], "
              f"mean: {distances.mean():.4f}, median: {np.median(distances):.4f}")
    
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
            hovertemplate='<b>Original Policy</b><br>Task: %{text}<br>PCA X: %{x:.2f}<br>PCA Y: %{y:.2f}<br>Reward: %{z:.2f}<extra></extra>',
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
            hovertemplate=f'<b>Step Size {step_size}</b><br>Task: %{{text}}<br>PCA X: %{{x:.2f}}<br>PCA Y: %{{y:.2f}}<br>Reward: %{{z:.2f}}<extra></extra>',
        ))
    
    fig.update_layout(
        title=dict(
            text=f'Policy Behavior Space PCA Visualization (3D)<br>' +
                 f'<sub>Original Policies: {len(original_indices)}, ' +
                 f'Retrain Policies: {len(retrain_indices)}<br>' +
                 f'Explained Variance: {pca.explained_variance_ratio_.sum()*100:.2f}%</sub>',
            x=0.5,
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title='PCA Component 1',
            yaxis_title='PCA Component 2',
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
    
    ax.set_xlabel("PCA Component 1", fontsize=12)
    ax.set_ylabel("PCA Component 2", fontsize=12)
    ax.set_zlabel("Episode Reward", fontsize=12)
    ax.set_title(
        f"Policy Behavior Space PCA Visualization (3D)\n"
        f"Original Policies: {len(original_indices)}, Retrain Policies: {len(retrain_indices)}\n"
        f"Explained Variance: {pca.explained_variance_ratio_.sum()*100:.2f}%",
        fontsize=14,
    )
    ax.legend(loc='upper left')
    
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_static.savefig(args.output_path, dpi=200, bbox_inches="tight")
    plt.close(fig_static)
    print(f"[Saved] Static PNG: {args.output_path}")
    
    # Step 5: Visualize graph if requested
    if args.visualize_graph:
        print(f"\n[Step 5] Visualizing behavior space graph...")
        
        # Use the first PCA dimension for graph visualization (or user can specify)
        # Default to the largest dimension for better representation
        graph_dim = max(args.pca_dims)
        distance_matrix = distances_dict[graph_dim]
        
        # Determine threshold
        if args.distance_threshold is None:
            # Use median distance as default threshold
            triu_indices = np.triu_indices_from(distance_matrix, k=1)
            all_distances = distance_matrix[triu_indices]
            threshold = np.median(all_distances)
            print(f"  Using default threshold (median distance): {threshold:.4f}")
        else:
            threshold = args.distance_threshold
            print(f"  Using specified threshold: {threshold:.4f}")
        
        # Build graph for each PCA dimension
        for k in args.pca_dims:
            print(f"\n  Building PCA{k} graph...")
            dist_mat = distances_dict[k]
            
            G, edges = build_graph_from_distance_matrix(
                dist_mat,
                threshold,
                all_metadata,
            )
            
            print_behavior_graph_statistics(G, dist_mat, threshold, all_metadata)
            
            # Visualize
            graph_output = args.output_path.parent / f"behavior_graph_pca{k}_threshold_{threshold:.4f}.png"
            visualize_behavior_graph(
                G,
                args.env_name,
                graph_output,
                threshold,
                layout=args.graph_layout,
                separate_components=args.separate_components,
                distance_matrix=dist_mat,
                pca_dim=k,
            )
    
    # Save data
    data_output = args.output_path.with_suffix(".npz")
    save_dict = {
        "behavior_2d": behavior_2d,
        "rewards": np.array(all_rewards),
        "metadata": all_metadata,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "representative_states": representative_states,
    }
    
    # Add PCA embeddings and distance matrices
    for k in args.pca_dims:
        save_dict[f"embeddings_pca{k}"] = embeddings_dict[k]
        save_dict[f"distances_pca{k}"] = distances_dict[k]
    
    np.savez(data_output, **save_dict)
    print(f"[Saved] Data file: {data_output}")


if __name__ == "__main__":
    main()

