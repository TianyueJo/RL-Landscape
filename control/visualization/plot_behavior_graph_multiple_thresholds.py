#!/usr/bin/env python3
"""
Generate multiple behavior-space graph visualizations using different thresholds.
Loads an existing distance matrix from a data file and builds graphs under multiple thresholds.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def load_rewards_from_models(models_dir: Path, metadata: List[Dict]) -> List[Optional[float]]:
    """
    Load each policy's training reward from the models directory.
    
    Args:
        models_dir: Models root directory
        metadata: List of policy metadata dicts
    
    Returns:
        rewards: List of rewards aligned with metadata
    """
    rewards = []
    
    for meta in metadata:
        task_id = meta.get("task_id")
        if task_id is None:
            rewards.append(None)
            continue
        
        # Try best_info.json or training_summary.json
        task_dir = models_dir / f"controlled_task_{task_id}"
        
        # Prefer best_info.json
        best_info_path = task_dir / "best_info.json"
        if best_info_path.exists():
            try:
                with best_info_path.open() as f:
                    info = json.load(f)
                reward = info.get("best_reward")
                rewards.append(reward)
                continue
            except:
                pass
        
        # Fallback: training_summary.json
        summary_path = task_dir / "training_summary.json"
        if summary_path.exists():
            try:
                with summary_path.open() as f:
                    summary = json.load(f)
                reward = summary.get("best_reward")
                rewards.append(reward)
                continue
            except:
                pass
        
        # If neither exists/works, return None
        rewards.append(None)
    
    return rewards


def build_graph_from_distance_matrix(
    distance_matrix: np.ndarray,
    threshold: float,
    metadata: List[Dict],
    rewards: List[Optional[float]] = None,
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
    
    # Add nodes; build labels from metadata
    for i, meta in enumerate(metadata):
        if meta.get("type") == "original":
            node_label = f"T{meta['task_id']}"
        else:
            node_label = f"T{meta['task_id']}_jr{meta.get('jr_index', '?')}_s{meta.get('step_size', '?')}"
        
        node_attrs = {"label": node_label, "metadata": meta}
        
        # Attach reward info
        if rewards is not None and i < len(rewards):
            node_attrs["reward"] = rewards[i]
        
        G.add_node(i, **node_attrs)
    
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
    pca_dim: int = None,
    use_reward_colors: bool = True,
):
    """
    Visualize the behavior-space graph structure.
    """
    num_components = nx.number_connected_components(G)
    
    if separate_components and num_components > 1:
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
            
            # Set node colors by reward (if available)
            if use_reward_colors:
                node_rewards = [subgraph.nodes[node].get('reward') for node in subgraph.nodes()]
                if any(r is not None for r in node_rewards):
                    # Color by reward
                    valid_rewards = [r for r in node_rewards if r is not None]
                    if valid_rewards:
                        min_reward = min(valid_reward for valid_reward in valid_rewards)
                        max_reward = max(valid_reward for valid_reward in valid_rewards)
                        norm = plt.Normalize(vmin=min_reward, vmax=max_reward)
                        cmap = plt.cm.viridis  # viridis colormap
                        
                        node_colors_list = []
                        for node in subgraph.nodes():
                            reward = subgraph.nodes[node].get('reward')
                            if reward is not None:
                                node_colors_list.append(cmap(norm(reward)))
                            else:
                                node_colors_list.append('gray')
                    else:
                        node_colors_list = [component_colors[idx % len(component_colors)]]
                else:
                    node_colors_list = [component_colors[idx % len(component_colors)]]
            else:
                node_colors_list = [component_colors[idx % len(component_colors)]]
            
            nx.draw_networkx_nodes(subgraph, pos,
                                  node_color=node_colors_list,
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
    
    # Unified view (use a larger figure to fit left/right colorbars)
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(16, 10))
    
    # Use GridSpec for better colorbar layout control
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.06, 0.88, 0.06], hspace=0.1, wspace=0.05)
    
    # Main plot area (center)
    ax_main = plt.subplot(gs[0, 1])
    
    # Layout
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
    
    # Set node colors by reward (if available)
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
        # Tick formatting
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
            min_reward = min(valid_rewards)
            max_reward = max(valid_rewards)
            sm_reward = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                             norm=plt.Normalize(vmin=min_reward,
                                                               vmax=max_reward))
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
            print(
                f"  [Debug] Right colorbar: reward range=[{min_reward:.2f}, {max_reward:.2f}], "
                f"num ticks={len(cbar_reward.ax.get_yticks())}"
            )
        else:
            # If no valid rewards, keep an empty area to preserve layout
            ax_cbar_right = plt.subplot(gs[0, 2])
            ax_cbar_right.set_visible(False)
            print("  [Warning] No valid reward data; right colorbar not created")
    else:
        # If use_reward_colors=False, keep an empty area to preserve layout
        ax_cbar_right = plt.subplot(gs[0, 2])
        ax_cbar_right.set_visible(False)
        print("  [Info] use_reward_colors=False; right colorbar not created")
    
    # Note: do not call axis('off') on the entire figure; it would hide colorbars
    ax_main.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")
    plt.close()


def print_graph_statistics(G, distance_matrix, threshold, metadata):
    """Print graph statistics."""
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
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Generate behavior-space graph visualizations under multiple thresholds")
    parser.add_argument("--data-file", type=Path, 
                       default=Path("analysis_outputs/behavior_space_pca_3d.npz"),
                       help="Path to the data file")
    parser.add_argument("--output-dir", type=Path,
                       default=Path("analysis_outputs/behavior_graphs"),
                       help="Output directory")
    parser.add_argument("--env-name", type=str, default="Walker2d-v4",
                       help="Environment name")
    parser.add_argument("--pca-dims", type=int, nargs="+", default=[2, 6, 10],
                       help="PCA dimensions to process")
    parser.add_argument("--thresholds", type=float, nargs="+", default=None,
                       help="Threshold list (if None, use percentiles)")
    parser.add_argument("--layout", type=str, default="spring",
                       choices=["spring", "circular", "kamada_kawai", "spectral"],
                       help="Graph layout algorithm")
    parser.add_argument("--separate-components", action="store_true",
                       help="Plot connected components separately")
    parser.add_argument("--models-dir", type=Path, default=Path("models"),
                       help="Models directory (used to load rewards)")
    parser.add_argument("--no-reward-colors", action="store_true",
                       help="Disable reward-based node colors (use component colors)")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data file: {args.data_file}")
    if not args.data_file.exists():
        print(f"Error: data file does not exist: {args.data_file}")
        return
    
    data = np.load(args.data_file, allow_pickle=True)
    
    # Check distance matrices exist
    has_distances = any(f"distances_pca{k}" in data.files for k in args.pca_dims)
    if not has_distances:
        print("Error: no distance matrices found in data file")
        print(f"Available keys: {list(data.files)}")
        return
    
    # Load metadata
    if "metadata" in data.files:
        metadata = data["metadata"].tolist()
    else:
        # Try to infer metadata
        n_policies = len(data["behavior_2d"]) if "behavior_2d" in data.files else 16
        metadata = [{"task_id": i, "type": "original", "step_size": None} 
                   for i in range(n_policies)]
        print(f"Warning: metadata not found; using default metadata ({n_policies} policies)")
    
    print(f"✓ Loaded metadata for {len(metadata)} policies")
    
    # Load rewards
    rewards = None
    if not args.no_reward_colors and args.models_dir.exists():
        print(f"\nLoading rewards from: {args.models_dir}")
        rewards = load_rewards_from_models(args.models_dir, metadata)
        valid_rewards = [r for r in rewards if r is not None]
        if valid_rewards:
            print(f"✓ Loaded rewards for {len(valid_rewards)}/{len(rewards)} policies")
            print(f"  Reward range: [{min(valid_rewards):.2f}, {max(valid_rewards):.2f}]")
        else:
            print("⚠ No reward information found")
            rewards = None
    else:
        if args.no_reward_colors:
            print("Using component colors (--no-reward-colors)")
        else:
            print(f"⚠ Models directory does not exist: {args.models_dir}; using component colors")
    
    # Generate plots for each PCA dimension
    for k in args.pca_dims:
        key = f"distances_pca{k}"
        if key not in data.files:
            print(f"⚠ Skipping PCA{k}: distance matrix not found")
            continue
        
        distance_matrix = data[key]
        print(f"\nProcessing PCA{k}; distance matrix shape: {distance_matrix.shape}")
        
        # Distance stats
        triu_indices = np.triu_indices_from(distance_matrix, k=1)
        all_distances = distance_matrix[triu_indices]
        
        print("Distance stats:")
        print(f"  Min: {all_distances.min():.4f}")
        print(f"  Max: {all_distances.max():.4f}")
        print(f"  Mean: {all_distances.mean():.4f}")
        print(f"  Median: {np.median(all_distances):.4f}")
        print(f"  25th percentile: {np.percentile(all_distances, 25):.4f}")
        print(f"  75th percentile: {np.percentile(all_distances, 75):.4f}")
        
        # Thresholds
        if args.thresholds:
            thresholds = args.thresholds
        else:
            # Use multiple percentiles as thresholds
            thresholds = [
                np.percentile(all_distances, 10),
                np.percentile(all_distances, 25),
                np.percentile(all_distances, 50),  # median
                np.percentile(all_distances, 75),
                np.percentile(all_distances, 90),
            ]
            print(f"\nUsing percentile thresholds: {[f'{t:.4f}' for t in thresholds]}")
        
        # Generate plots for each threshold
        for threshold in thresholds:
            print(f"\nGenerating PCA{k} plot with threshold: {threshold:.4f}")
            G, edges = build_graph_from_distance_matrix(
                distance_matrix, threshold, metadata, rewards=rewards
            )
            
            print_graph_statistics(G, distance_matrix, threshold, metadata)
            
            # Filename
            threshold_str = f"{threshold:.4f}".replace('.', '_')
            output_path = args.output_dir / f"behavior_graph_pca{k}_threshold_{threshold_str}.png"
            
            visualize_behavior_graph(
                G, args.env_name, output_path, threshold,
                layout=args.layout,
                separate_components=args.separate_components,
                pca_dim=k,
                use_reward_colors=not args.no_reward_colors and rewards is not None,
            )
    
    print(f"\n✓ All figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

