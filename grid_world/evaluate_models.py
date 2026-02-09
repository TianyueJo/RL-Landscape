"""
Evaluate trained GridWorld PPO models.
"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

# Try importing plotting libraries
HAS_MATPLOTLIB = False
HAS_PLOTLY = False

try:
    import matplotlib
    matplotlib.use('Agg')  # use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except (ImportError, RuntimeError):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        HAS_PLOTLY = True
    except ImportError:
        pass

from env import make_env
from ppo_simple import PPO, PPOAgent


def load_model(model_path: Path, device: str = "cpu"):
    """Load a saved PPOAgent checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    obs_dim = checkpoint['obs_dim']
    action_dim = checkpoint['action_dim']
    
    # Create agent
    agent = PPOAgent(obs_dim, action_dim, hidden_dim=64, lr=3e-4)
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    
    # Create a PPO-like wrapper with a predict() API
    class ModelWrapper:
        def __init__(self, agent):
            self.agent = agent
        
        def predict(self, obs: np.ndarray, deterministic: bool = True):
            action, _, _ = self.agent.get_action(obs, deterministic=deterministic)
            return action, None
    
    return ModelWrapper(agent), checkpoint


def evaluate_model(model, env, n_episodes: int = 50, deterministic: bool = True):
    """Evaluate a model over n_episodes and return summary statistics."""
    ep_returns = []
    ep_lens = []
    success_count = 0
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        reached_goal = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(int(action))
            total_reward += r
            steps += 1
            
            if info.get("event") == "goal":
                reached_goal = True
            
            done = terminated or truncated
        
        ep_returns.append(total_reward)
        ep_lens.append(steps)
        if reached_goal:
            success_count += 1
    
    return {
        'mean_return': np.mean(ep_returns),
        'std_return': np.std(ep_returns),
        'min_return': np.min(ep_returns),
        'max_return': np.max(ep_returns),
        'mean_length': np.mean(ep_lens),
        'success_rate': success_count / n_episodes,
        'returns': ep_returns,
    }


def plot_performance_distribution(results: dict, output_path: Path):
    """Plot performance distribution: bar chart of mean return per seed."""
    if HAS_PLOTLY:
        plot_performance_distribution_plotly(results, output_path)
    elif HAS_MATPLOTLIB:
        plot_performance_distribution_matplotlib(results, output_path)
    else:
        print("Warning: no plotting library available; skipping plot.")
        return


def plot_performance_distribution_plotly(results: dict, output_path: Path):
    """Plot performance distribution with Plotly (mean return per seed)."""
    import plotly.graph_objects as go
    
    cases = sorted(results.keys())
    # Support an arbitrary number of cases
    palette = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]
    colors = {case_id: palette[(int(case_id) - 1) % len(palette)] for case_id in cases}
    
    fig = go.Figure()
    
    # Build bars for each case
    x_positions = []
    x_labels = []
    y_values = []
    bar_colors = []
    
    for case in cases:
        seeds_data = results[case]['seeds_data']  # per-seed data
        sorted_seeds = sorted(seeds_data.keys())
        
        for seed in sorted_seeds:
            x_positions.append(f"Case {case}\nSeed {seed}")
            x_labels.append(f"S{seed}")
            y_values.append(seeds_data[seed]['mean_return'])
            bar_colors.append(colors[int(case)])
    
    # Create bar chart
    fig.add_trace(go.Bar(
        x=x_positions,
        y=y_values,
        marker_color=bar_colors,
        text=[f'{v:.2f}' for v in y_values],
        textposition='outside',
        name='Episode Return',
        showlegend=False
    ))
    
    # Add case separators
    case_boundaries = []
    current_case = None
    for i, pos in enumerate(x_positions):
        case = int(pos.split()[1])
        if current_case is None:
            current_case = case
        elif case != current_case:
            case_boundaries.append(i - 0.5)
            current_case = case
    
    # Add separator lines
    for boundary in case_boundaries:
        fig.add_vline(
            x=boundary,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text="",
        )
    
    fig.update_layout(
        title={
            'text': 'GridWorld PPO Model Performance - Average Episode Return per Random Seed',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'weight': 'bold'}
        },
        xaxis=dict(
            title='Case and Random Seed',
            tickangle=-45,
            title_font=dict(size=12)
        ),
        yaxis=dict(
            title='Average Episode Return',
            title_font=dict(size=12)
        ),
        height=600,
        width=1200,
        margin=dict(b=100),
    )
    
    # Add legend annotation (dynamic)
    legend_lines = ["<b>Legend:</b>"]
    for case in cases:
        legend_lines.append(f"Case {case}: <span style='color:{colors[int(case)]}'>■</span>")
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text="<br>".join(legend_lines),
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="black",
        borderwidth=1,
    )
    
    # Save HTML
    html_path = output_path.with_suffix('.html')
    fig.write_html(str(html_path))
    print(f"Performance plot saved to: {html_path}")
    
    # Also try saving PNG if possible
    try:
        fig.write_image(str(output_path))
        print(f"Performance plot saved to: {output_path}")
    except Exception as e:
        print("Could not save PNG (requires kaleido); HTML was saved instead.")


def plot_performance_distribution_matplotlib(results: dict, output_path: Path):
    """Plot performance distribution with Matplotlib (mean return per seed)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    cases = sorted(results.keys())
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    colors = {case_id: palette[(int(case_id) - 1) % len(palette)] for case_id in cases}
    
    x_positions = []
    x_labels = []
    y_values = []
    bar_colors = []
    
    for case in cases:
        seeds_data = results[case]['seeds_data']
        sorted_seeds = sorted(seeds_data.keys())
        
        for seed in sorted_seeds:
            x_positions.append(f"Case {case}\nSeed {seed}")
            x_labels.append(f"S{seed}")
            y_values.append(seeds_data[seed]['mean_return'])
            bar_colors.append(colors[int(case)])
    
    # Create bar chart
    bars = ax.bar(range(len(x_positions)), y_values, color=bar_colors, 
                  edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, y_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # Add case separators
    case_boundaries = []
    current_case = None
    for i, pos in enumerate(x_positions):
        case = int(pos.split()[1])
        if current_case is None:
            current_case = case
        elif case != current_case:
            case_boundaries.append(i - 0.5)
            current_case = case
    
    for boundary in case_boundaries:
        ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Case and Random Seed', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Episode Return', fontsize=12, fontweight='bold')
    ax.set_title('GridWorld PPO Model Performance - Average Episode Return per Random Seed\n(Each seed evaluated 5 times, averaged)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(x_positions)))
    ax.set_xticklabels(x_labels, rotation=0, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[int(case)], label=f'Case {case}') for case in cases]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Performance plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate GridWorld PPO models")
    parser.add_argument("--models-dir", type=str, default="models", help="Models directory")
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=5,
        help="Episodes per model evaluation (each seed averaged over N episodes)",
    )
    parser.add_argument("--output", type=str, default="evaluation_results.png", help="Output image path")
    parser.add_argument(
        "--action-mode",
        type=str,
        default="deterministic",
        choices=["deterministic", "stochastic", "both"],
        help=(
            "Action selection for evaluation: "
            "deterministic=argmax; stochastic=sample from policy; both=evaluate both and save."
        ),
    )
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"Error: models directory does not exist: {models_dir}")
        return
    
    # Collect model files
    model_files = defaultdict(list)
    for model_file in models_dir.glob("gridworld_ppo_case_*_seed_*.pt"):
        # Parse filename: gridworld_ppo_case_{case_id}_seed_{seed}.pt
        parts = model_file.stem.split('_')
        case_id = int(parts[3])
        seed = int(parts[5])
        model_files[case_id].append((seed, model_file))
    
    if not model_files:
        print(f"Error: no model files found under {models_dir}")
        return
    
    print(f"Found {sum(len(files) for files in model_files.values())} model files")
    for case_id, files in sorted(model_files.items()):
        print(f"  Case {case_id}: {len(files)} models")
    
    # Evaluate all models
    all_results = defaultdict(lambda: {'seeds_data': {}})  # per-seed results
    
    for case_id in sorted(model_files.keys()):
        print(f"\n{'='*60}")
        print(f"Evaluating models for Case {case_id}...")
        print(f"{'='*60}")
        
        seeds_data = {}
        for seed, model_path in sorted(model_files[case_id]):
            print(f"  Evaluating: {model_path.name}...", end=" ", flush=True)
            
            # Load model
            model, checkpoint = load_model(model_path)
            
            # Evaluate in the corresponding case environment
            eval_env = make_env(seed=seed, case_id=case_id)  # match training seed
            
            if args.action_mode in ("deterministic", "stochastic"):
                deterministic = args.action_mode == "deterministic"
                result = evaluate_model(
                    model, eval_env, n_episodes=args.n_episodes, deterministic=deterministic
                )
                seeds_data[seed] = {
                    "mean_return": result["mean_return"],
                    "std_return": result["std_return"],
                    "success_rate": result["success_rate"],
                }
                mode_str = "Det" if deterministic else "Sto"
                print(
                    f"[{mode_str}] Return: {result['mean_return']:.2f}±{result['std_return']:.2f}, "
                    f"Success: {result['success_rate']*100:.1f}%"
                )
            else:
                # both
                det_res = evaluate_model(
                    model, eval_env, n_episodes=args.n_episodes, deterministic=True
                )
                sto_res = evaluate_model(
                    model, eval_env, n_episodes=args.n_episodes, deterministic=False
                )
                seeds_data[seed] = {
                    "deterministic": {
                        "mean_return": det_res["mean_return"],
                        "std_return": det_res["std_return"],
                        "success_rate": det_res["success_rate"],
                    },
                    "stochastic": {
                        "mean_return": sto_res["mean_return"],
                        "std_return": sto_res["std_return"],
                        "success_rate": sto_res["success_rate"],
                    },
                }
                print(
                    f"[Det] Return: {det_res['mean_return']:.2f}±{det_res['std_return']:.2f}, "
                    f"Success: {det_res['success_rate']*100:.1f}% | "
                    f"[Sto] Return: {sto_res['mean_return']:.2f}±{sto_res['std_return']:.2f}, "
                    f"Success: {sto_res['success_rate']*100:.1f}%"
                )
        
        # Store results for this case
        all_results[case_id] = {
            'seeds_data': seeds_data,
        }
        
        # Summary statistics
        if args.action_mode == "both":
            seed_returns = [data["deterministic"]["mean_return"] for data in seeds_data.values()]
        else:
            seed_returns = [data['mean_return'] for data in seeds_data.values()]
        print(f"\nCase {case_id} summary:")
        print(f"  Mean return: {np.mean(seed_returns):.2f} ± {np.std(seed_returns):.2f}")
        print(f"  Range: [{np.min(seed_returns):.2f}, {np.max(seed_returns):.2f}]")
        if args.action_mode == "both":
            mean_sr = np.mean([data["deterministic"]["success_rate"] for data in seeds_data.values()])
        else:
            mean_sr = np.mean([data['success_rate'] for data in seeds_data.values()])
        print(f"  Mean success rate: {mean_sr*100:.1f}%")
    
    # Plot performance distribution
    output_path = Path(args.output)
    if args.action_mode != "both":
        plot_performance_distribution(all_results, output_path)
    else:
        # For "both": write deterministic and stochastic plots separately (different seeds_data format)
        det_results = defaultdict(lambda: {"seeds_data": {}})
        sto_results = defaultdict(lambda: {"seeds_data": {}})
        for case_id in sorted(all_results.keys()):
            det_results[case_id]["seeds_data"] = {
                seed: payload["deterministic"] for seed, payload in all_results[case_id]["seeds_data"].items()
            }
            sto_results[case_id]["seeds_data"] = {
                seed: payload["stochastic"] for seed, payload in all_results[case_id]["seeds_data"].items()
            }
        plot_performance_distribution(det_results, output_path.with_suffix("").with_name(output_path.stem + "_det").with_suffix(output_path.suffix))
        plot_performance_distribution(sto_results, output_path.with_suffix("").with_name(output_path.stem + "_sto").with_suffix(output_path.suffix))
    
    # Save results to JSON
    json_output_path = output_path.with_suffix('.json')
    # Build JSON payload
    json_data = {}
    json_data["action_mode"] = args.action_mode
    for case_id in sorted(all_results.keys()):
        case_data = all_results[case_id]
        json_data[f'case_{case_id}'] = {
            'seeds_data': {}
        }
        for seed in sorted(case_data['seeds_data'].keys()):
            seed_data = case_data['seeds_data'][seed]
            if args.action_mode == "both":
                json_data[f'case_{case_id}']['seeds_data'][f'seed_{seed}'] = {
                    "deterministic": {
                        "mean_return": float(seed_data["deterministic"]["mean_return"]),
                        "std_return": float(seed_data["deterministic"]["std_return"]),
                        "success_rate": float(seed_data["deterministic"]["success_rate"]),
                    },
                    "stochastic": {
                        "mean_return": float(seed_data["stochastic"]["mean_return"]),
                        "std_return": float(seed_data["stochastic"]["std_return"]),
                        "success_rate": float(seed_data["stochastic"]["success_rate"]),
                    },
                }
            else:
                json_data[f'case_{case_id}']['seeds_data'][f'seed_{seed}'] = {
                    'mean_return': float(seed_data['mean_return']),
                    'std_return': float(seed_data['std_return']),
                    'success_rate': float(seed_data['success_rate']),
                }
        # Add summary statistics
        if args.action_mode == "both":
            seed_returns = [data["deterministic"]["mean_return"] for data in case_data["seeds_data"].values()]
            mean_success_rate = float(np.mean([data["deterministic"]["success_rate"] for data in case_data["seeds_data"].values()]))
        else:
            seed_returns = [data['mean_return'] for data in case_data['seeds_data'].values()]
            mean_success_rate = float(np.mean([data['success_rate'] for data in case_data['seeds_data'].values()]))
        json_data[f'case_{case_id}']['summary'] = {
            'mean_return': float(np.mean(seed_returns)),
            'std_return': float(np.std(seed_returns)),
            'min_return': float(np.min(seed_returns)),
            'max_return': float(np.max(seed_returns)),
            'mean_success_rate': mean_success_rate,
        }
    
    with open(json_output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Evaluation results saved to JSON: {json_output_path}")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

