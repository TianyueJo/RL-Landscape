#!/usr/bin/env python3
"""
Build and visualize a policy graph from a JS-divergence matrix.
Each policy is a node; an edge is added when JS divergence between two policies is below a threshold.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
from pathlib import Path
from matplotlib.gridspec import GridSpec


def load_js_matrix(data_dir):
    """Load a JS-divergence matrix and metadata."""
    data_dir = Path(data_dir)
    
    # Load matrix
    matrix_path = data_dir / "HalfCheetah-v4_similarity_matrix.npy"
    if not matrix_path.exists():
        # Try Walker2d
        matrix_path = data_dir / "Walker2d-v4_similarity_matrix.npy"
    
    if not matrix_path.exists():
        raise FileNotFoundError(f"Could not find similarity matrix file: {matrix_path}")
    
    similarity_matrix = np.load(matrix_path)
    
    # Load metadata
    summary_path = data_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            metadata = json.load(f)
        task_ids = metadata.get('task_ids', list(range(len(similarity_matrix))))
        env_name = metadata.get('env_name', 'Unknown')
    else:
        task_ids = list(range(len(similarity_matrix)))
        env_name = 'Unknown'
    
    return similarity_matrix, task_ids, env_name


def build_graph_from_matrix(similarity_matrix, threshold, task_ids=None):
    """
    Build a graph from a similarity (JS divergence) matrix.
    
    Args:
        similarity_matrix: JS divergence matrix (symmetric, diagonal is 0)
        threshold: Add an edge if divergence < threshold
        task_ids: Task id list; if None, use indices
    
    Returns:
        G: networkx Graph
    """
    n = len(similarity_matrix)
    if task_ids is None:
        task_ids = list(range(n))
    
    # Undirected graph
    G = nx.Graph()
    
    # Add nodes
    for i, task_id in enumerate(task_ids):
        G.add_node(task_id, label=f"Task {task_id}")
    
    # Add edges: if JS divergence < threshold
    # Note: smaller JS divergence => more similar policies, hence "< threshold"
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] < threshold:
                G.add_edge(task_ids[i], task_ids[j], 
                          weight=similarity_matrix[i, j],
                          divergence=similarity_matrix[i, j])
                edges.append((task_ids[i], task_ids[j], similarity_matrix[i, j]))
    
    return G, edges


def load_rewards_from_models(models_dir, task_ids):
    """
    Load training reward information from the models directory.
    
    Args:
        models_dir: Models root directory
        task_ids: Task id list
    
    Returns:
        rewards: List of rewards aligned with task_ids (None if missing)
    """
    models_dir = Path(models_dir)
    rewards = []
    
    for task_id in task_ids:
        # Try best_info.json first
        best_info_path = models_dir / f"controlled_task_{task_id}" / "best_info.json"
        training_summary_path = models_dir / f"controlled_task_{task_id}" / "training_summary.json"
        
        reward = None
        if best_info_path.exists():
            try:
                with open(best_info_path, 'r') as f:
                    data = json.load(f)
                    reward = data.get('best_reward')
            except:
                pass
        
        if reward is None and training_summary_path.exists():
            try:
                with open(training_summary_path, 'r') as f:
                    data = json.load(f)
                    reward = data.get('best_reward')
            except:
                pass
        
        rewards.append(reward)
    
    return rewards


def visualize_graph(G, env_name, output_path, threshold, layout='spring', separate_components=False, 
                    use_reward_colors=False, rewards=None, models_dir=None):
    """
    Visualize graph structure.
    
    Args:
        G: networkx Graph
        env_name: Environment name
        output_path: Output path
        threshold: Threshold used to build edges
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
        separate_components: Whether to draw connected components separately
    """
    num_components = nx.number_connected_components(G)
    
    if separate_components and num_components > 1:
        # Create one subplot per connected component
        components = list(nx.connected_components(G))
        n_components = len(components)
        
        # Compute subplot grid (roughly square)
        n_cols = int(np.ceil(np.sqrt(n_components)))
        n_rows = int(np.ceil(n_components / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_components == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        # Assign a color per component
        component_colors = plt.cm.tab20(np.linspace(0, 1, min(n_components, 20)))
        if n_components > 20:
            # If too many components, cycle colors
            component_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        for idx, component in enumerate(components):
            ax = axes[idx] if n_components > 1 else axes[0]
            
            # Subgraph
            subgraph = G.subgraph(component)
            
            # Layout
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
            
            # Draw edges
            edges = subgraph.edges()
            if edges:
                edge_weights = [subgraph[u][v].get('divergence', 1.0) for u, v in edges]
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
            
            # Draw nodes (component color)
            node_color = component_colors[idx % len(component_colors)]
            nx.draw_networkx_nodes(subgraph, pos,
                                  node_color=[node_color],
                                  node_size=2000,
                                  alpha=0.9,
                                  edgecolors='black',
                                  linewidths=2,
                                  ax=ax)
            
            # Draw labels
            labels = {node: f"T{node}" for node in subgraph.nodes()}
            nx.draw_networkx_labels(subgraph, pos, labels,
                                   font_size=12,
                                   font_weight='bold',
                                   font_color='white',
                                   ax=ax)
            
            # Title
            component_list = sorted(list(component))
            ax.set_title(f"Component {idx+1}\nNodes: {component_list}\nSize: {len(component)}", 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # Hide unused axes
        for idx in range(n_components, len(axes)):
            axes[idx].axis('off')
        
        # Global title
        fig.suptitle(f"{env_name} Policy Graph - Separate Components\n"
                    f"Threshold: {threshold:.4f} | "
                    f"Total Nodes: {G.number_of_nodes()} | "
                    f"Total Edges: {G.number_of_edges()} | "
                    f"Components: {num_components}",
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {output_path}")
        plt.close()
        return
    
    # Unified view (all components in one plot). Use GridSpec for colorbars.
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.06, 0.88, 0.06], hspace=0.1, wspace=0.05)
    ax_main = plt.subplot(gs[0, 1])
    
    # Layout algorithm
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
    
    # Draw edges
    edges = G.edges()
    if edges:
        # Set edge color/width based on divergence
        edge_weights = [G[u][v].get('divergence', 1.0) for u, v in edges]
        edge_colors = plt.cm.viridis_r([w / max(edge_weights) if max(edge_weights) > 0 else 0.5 
                                        for w in edge_weights])
        # Thicker edges: min width 3.0, max width 8.0
        edge_widths = [3.0 + 5.0 * (1.0 - w / max(edge_weights)) if max(edge_weights) > 0 else 5.0 
                      for w in edge_weights]
        
        nx.draw_networkx_edges(G, pos, 
                              edge_color=edge_colors,
                              width=edge_widths,
                              alpha=0.7,
                              style='solid',
                              ax=ax_main)
    else:
        print("Warning: no edges (all divergences exceed the threshold).")
    
    # Draw nodes - color by reward or by connected component
    if use_reward_colors and rewards is not None:
        # Color nodes by reward
        # Build node->reward mapping
        node_to_reward = {}
        for i, node in enumerate(G.nodes()):
            if i < len(rewards) and rewards[i] is not None:
                node_to_reward[node] = rewards[i]
            else:
                node_to_reward[node] = None
        
        node_rewards = [node_to_reward[node] for node in G.nodes()]
        valid_rewards = [r for r in node_rewards if r is not None]
        
        if valid_rewards:
            min_reward = min(valid_rewards)
            max_reward = max(valid_rewards)
            norm_reward = plt.Normalize(vmin=min_reward, vmax=max_reward)
            cmap_reward = plt.cm.viridis
            
            node_colors = []
            for node in G.nodes():
                reward = node_to_reward[node]
                if reward is not None:
                    node_colors.append(cmap_reward(norm_reward(reward)))
                else:
                    node_colors.append('gray')
        else:
            # If no valid rewards, fallback to component colors
            use_reward_colors = False
    
    if not use_reward_colors:
        # Color by connected component
        components = list(nx.connected_components(G))
        num_components = len(components)
        
        # Assign colors
        component_colors = plt.cm.tab20(np.linspace(0, 1, min(num_components, 20)))
        if num_components > 20:
            # If too many components, cycle colors
            component_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        # Node -> color map
        node_to_color = {}
        for idx, component in enumerate(components):
            color = component_colors[idx % len(component_colors)]
            for node in component:
                node_to_color[node] = color
        
        node_colors = [node_to_color[node] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=1500,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=2,
                          ax=ax_main)
    
    # Draw labels
    labels = {node: f"T{node}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels,
                           font_size=10,
                           font_weight='bold',
                           font_color='black',
                           ax=ax_main)
    
    # Title
    title = f"{env_name} Policy Graph\n"
    title += f"Threshold: {threshold:.4f} | "
    title += f"Nodes: {G.number_of_nodes()} | "
    title += f"Edges: {G.number_of_edges()} | "
    title += f"Connected Components: {nx.number_connected_components(G)}"
    
    ax_main.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax_main.axis('off')
    
    # Left colorbar: edge similarity (JS Divergence)
    if edges:
        edge_weights_list = [G[u][v].get('divergence', 1.0) for u, v in edges]
        ax_cbar_left = plt.subplot(gs[0, 0])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r, 
                                   norm=plt.Normalize(vmin=min(edge_weights_list), 
                                                     vmax=max(edge_weights_list)))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=ax_cbar_left, orientation='vertical')
        cbar.set_label('Edge Similarity\n(JS Divergence)', rotation=90, labelpad=20, fontsize=11, fontweight='bold')
        # Tick formatting
        from matplotlib.ticker import MaxNLocator
        tick_locator = MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=10, which='major', direction='out', length=4, width=1, pad=5)
        cbar.ax.yaxis.set_visible(True)
    
    # Right colorbar: node performance (training reward)
    if use_reward_colors and rewards is not None:
        node_rewards_list = [node_to_reward[node] for node in G.nodes()]
        valid_rewards_list = [r for r in node_rewards_list if r is not None]
        if valid_rewards_list:
            ax_cbar_right = plt.subplot(gs[0, 2])
            min_reward = min(valid_rewards_list)
            max_reward = max(valid_rewards_list)
            sm_reward = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                             norm=plt.Normalize(vmin=min_reward,
                                                               vmax=max_reward))
            sm_reward.set_array([])
            cbar_reward = plt.colorbar(sm_reward, cax=ax_cbar_right, orientation='vertical')
            cbar_reward.set_label('Node Performance\n(Training Reward)', rotation=90, labelpad=20, fontsize=11, fontweight='bold')
            
            # Tick formatting
            from matplotlib.ticker import MaxNLocator, FuncFormatter
            tick_locator = MaxNLocator(nbins=5)
            cbar_reward.locator = tick_locator
            cbar_reward.update_ticks()
            cbar_reward.ax.yaxis.set_major_locator(tick_locator)
            cbar_reward.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}'))
            cbar_reward.ax.tick_params(labelsize=10, which='major', direction='out', length=4, width=1, pad=5)
            cbar_reward.ax.yaxis.set_visible(True)
            ax_cbar_right.set_visible(True)
            cbar_reward.ax.yaxis.set_tick_params(which='major', labelsize=10)
        else:
            ax_cbar_right = plt.subplot(gs[0, 2])
            ax_cbar_right.set_visible(False)
    else:
        ax_cbar_right = plt.subplot(gs[0, 2])
        ax_cbar_right.set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")
    plt.close()


def print_graph_statistics(G, similarity_matrix, threshold, task_ids):
    """Print graph statistics."""
    print("\n" + "="*70)
    print("Graph statistics")
    print("="*70)
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Connected components: {nx.number_connected_components(G)}")
    print(f"Threshold: {threshold:.4f}")
    
    if G.number_of_edges() > 0:
        edge_weights = [G[u][v].get('divergence', 0) for u, v in G.edges()]
        print("\nEdge JS divergence stats:")
        print(f"  Min: {min(edge_weights):.4f}")
        print(f"  Max: {max(edge_weights):.4f}")
        print(f"  Mean: {np.mean(edge_weights):.4f}")
        print(f"  Median: {np.median(edge_weights):.4f}")
    
    # Degree stats
    degrees = [G.degree(node) for node in G.nodes()]
    print("\nNode degree stats:")
    print(f"  Min degree: {min(degrees)}")
    print(f"  Max degree: {max(degrees)}")
    print(f"  Mean degree: {np.mean(degrees):.2f}")
    
    # Connected components
    if nx.number_connected_components(G) > 1:
        print("\nConnected components:")
        for i, component in enumerate(nx.connected_components(G), 1):
            component_list = sorted(list(component))
            print(f"  Component {i}: {component_list} (size: {len(component_list)})")
    
    # List edges
    if G.number_of_edges() > 0:
        print(f"\nAll edges (total {G.number_of_edges()}):")
        edges_sorted = sorted(G.edges(data=True), 
                            key=lambda x: x[2].get('divergence', 0))
        for u, v, data in edges_sorted:
            div = data.get('divergence', 0)
            print(f"  Task {u} <-> Task {v}: JS divergence = {div:.4f}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Build and visualize a policy graph from a JS-divergence matrix'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing similarity_matrix.npy and summary.json')
    parser.add_argument('--threshold', type=float, default=None,
                       help='JS divergence threshold (add edge if divergence < threshold). '
                            'If omitted, uses the median of matrix values.')
    parser.add_argument('--layout', type=str, default='spring',
                       choices=['spring', 'circular', 'kamada_kawai', 'spectral'],
                       help='Graph layout algorithm')
    parser.add_argument('--output', type=str, default=None,
                       help='Output image path (default: {data_dir}/policy_graph_threshold_{threshold}.png)')
    parser.add_argument('--separate-components', action='store_true',
                       help='Plot each connected component separately (one subplot per component)')
    parser.add_argument('--models-dir', type=str, default=None,
                       help='Models directory (used to load rewards for node coloring)')
    parser.add_argument('--no-reward-colors', action='store_true',
                       help='Disable reward-based node colors (use component colors instead)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.data_dir}")
    similarity_matrix, task_ids, env_name = load_js_matrix(args.data_dir)
    print(f"Env: {env_name}")
    print(f"Task IDs: {task_ids}")
    print(f"Matrix shape: {similarity_matrix.shape}")
    print(f"JS Divergence range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
    print(
        f"Mean: {similarity_matrix.mean():.4f}, "
        f"Median: {np.median(similarity_matrix[similarity_matrix > 0]):.4f}"
    )
    
    # Threshold
    if args.threshold is None:
        # Use median of non-diagonal elements as default threshold
        mask = ~np.eye(len(similarity_matrix), dtype=bool)
        non_diag_values = similarity_matrix[mask]
        threshold = np.median(non_diag_values)
        print(f"\nNo threshold provided; using median: {threshold:.4f}")
    else:
        threshold = args.threshold
        print(f"\nUsing specified threshold: {threshold:.4f}")
    
    # Build graph
    print(f"\nBuilding graph (threshold = {threshold:.4f})...")
    G, edges = build_graph_from_matrix(similarity_matrix, threshold, task_ids)
    
    # Stats
    print_graph_statistics(G, similarity_matrix, threshold, task_ids)
    
    # Output path
    if args.output is None:
        output_dir = Path(args.data_dir)
        output_path = output_dir / f"policy_graph_threshold_{threshold:.4f}.png"
    else:
        output_path = Path(args.output)
    
    # Load rewards (optional)
    rewards = None
    use_reward_colors = False
    if args.models_dir and not args.no_reward_colors:
        models_dir = Path(args.models_dir)
        if models_dir.exists():
            print(f"\nLoading rewards from: {args.models_dir}")
            rewards = load_rewards_from_models(models_dir, task_ids)
            valid_rewards = [r for r in rewards if r is not None]
            if valid_rewards:
                print(f"✓ Loaded rewards for {len(valid_rewards)}/{len(rewards)} policies")
                print(f"  Reward range: [{min(valid_rewards):.2f}, {max(valid_rewards):.2f}]")
                use_reward_colors = True
            else:
                print("⚠ No reward information found; using component colors")
        else:
            print(f"⚠ Models directory does not exist: {args.models_dir}; using component colors")
    
    # Visualize
    print("Visualizing graph...")
    visualize_graph(G, env_name, output_path, threshold, layout=args.layout, 
                   separate_components=args.separate_components,
                   use_reward_colors=use_reward_colors,
                   rewards=rewards,
                   models_dir=args.models_dir)
    
    print("\n✓ Done!")


if __name__ == '__main__':
    main()

