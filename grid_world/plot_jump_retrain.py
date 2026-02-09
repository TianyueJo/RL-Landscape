#!/usr/bin/env python3
"""
Plot Jump & Retrain results.
"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

# Try importing plotting libraries
HAS_PLOTLY = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False

# Try importing environment/model helpers
try:
    from env import make_env
    from ppo_simple import PPOAgent
    HAS_ENV = True
except ImportError:
    HAS_ENV = False


class ObservationWrapper:
    """Convert MultiDiscrete observations to a Box of floats."""
    def __init__(self, env):
        self.env = env
        from gymnasium import spaces
        if hasattr(env, 'spec') and hasattr(env.spec, 'width') and hasattr(env.spec, 'height'):
            self.width = env.spec.width
            self.height = env.spec.height
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(2,), dtype=np.float32
            )
        else:
            self.observation_space = env.observation_space
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._transform_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._transform_obs(obs), reward, terminated, truncated, info
    
    def _transform_obs(self, obs):
        """Convert (x, y) position to normalized floats."""
        if isinstance(obs, np.ndarray) and obs.dtype in (np.int64, np.int32):
            return obs.astype(np.float32) / np.array([self.width - 1, self.height - 1], dtype=np.float32)
        return obs.astype(np.float32)
    
    def __getattr__(self, name):
        return getattr(self.env, name)


def load_original_model(model_path: Path, device: str = "cpu"):
    """Load an original (base) PPO policy model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    obs_dim = checkpoint['obs_dim']
    action_dim = checkpoint['action_dim']
    
    agent = PPOAgent(obs_dim, action_dim, hidden_dim=64, lr=3e-4)
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    return agent


def evaluate_original_model(agent, env, n_episodes: int = 5):
    """Evaluate an original (base) policy model."""
    ep_returns = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            action, _, _ = agent.get_action(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(int(action))
            total_reward += r
            done = terminated or truncated
        
        ep_returns.append(total_reward)
    
    return np.mean(ep_returns)


def load_original_policy_returns(models_dir: Path) -> Dict:
    """
    Evaluate original/base policies and return their mean returns.

    Returns: {case_id: {seed: mean_return}}
    """
    if not HAS_ENV:
        print("Warning: Cannot load environment modules, skipping original policy evaluation")
        return {}
    
    results = defaultdict(dict)
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        print(f"Warning: Models directory not found: {models_dir}")
        return {}
    
    # Collect all model files
    model_files = defaultdict(list)
    for model_file in models_dir.glob("gridworld_ppo_case_*_seed_*.pt"):
        parts = model_file.stem.split('_')
        case_id = int(parts[3])
        seed = int(parts[5])
        model_files[case_id].append((seed, model_file))
    
    print(f"Evaluating original policies from: {models_dir}")
    for case_id in sorted(model_files.keys()):
        for seed, model_path in sorted(model_files[case_id]):
            try:
                # Load model
                agent = load_original_model(model_path)
                
                # Evaluate in the corresponding case environment
                eval_env = make_env(seed=42, case_id=case_id)  # fixed eval seed
                
                # Evaluate (average over 5 episodes)
                mean_return = evaluate_original_model(agent, eval_env, n_episodes=5)
                results[case_id][seed] = mean_return
                
                print(f"  Case {case_id}, Seed {seed}: {mean_return:.2f}")
            except Exception as e:
                print(f"Warning: Failed to evaluate {model_path}: {e}")
                continue
    
    return results


def load_retrain_logs(results_dir: Path) -> Dict:
    """
    Load all retrain_log.json files.

    Returns: {case_id: {seed: {step_size: log_data}}}
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for log_dir in results_dir.iterdir():
        if not log_dir.is_dir():
            continue
        
        # Parse directory name: case_X_seed_Y_stepZ_jrI
        parts = log_dir.name.split('_')
        if len(parts) < 5 or parts[0] != 'case':
            continue
        
        try:
            case_id = int(parts[1])
            seed = int(parts[3])
            step_size_str = parts[4].replace('step', '')
            step_size = float(step_size_str)
        except (ValueError, IndexError):
            continue
        
        log_path = log_dir / "retrain_log.json"
        if not log_path.exists():
            continue
        
        try:
            with log_path.open() as f:
                log_data = json.load(f)
            results[case_id][seed][step_size] = log_data
        except Exception as e:
            print(f"Warning: Failed to load {log_path}: {e}")
            continue
    
    return results


def plot_jump_retrain_plotly(results: Dict, output_path: Path, original_returns: Dict = None):
    """Plot Jump & Retrain results with Plotly (bar chart of final-step mean return)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    cases = sorted(results.keys())
    step_sizes = [5, 10, 15, 20, 25, 30]
    seeds = list(range(10))
    
    # Define a color gradient for step_size from light to dark (blues).
    # Light: (135, 206, 250) -> Dark: (25, 25, 112)
    light_color_rgb = (135, 206, 250)  # LightSkyBlue
    dark_color_rgb = (25, 25, 112)     # MidnightBlue
    colors = []
    for step_size in step_sizes:
        # Compute depth: step_size=5 is lightest, 30 is darkest
        depth = (step_size - 5) / (30 - 5)  # 0.0 to 1.0
        # Linear interpolation from light to dark
        r = int(light_color_rgb[0] + (dark_color_rgb[0] - light_color_rgb[0]) * depth)
        g = int(light_color_rgb[1] + (dark_color_rgb[1] - light_color_rgb[1]) * depth)
        b = int(light_color_rgb[2] + (dark_color_rgb[2] - light_color_rgb[2]) * depth)
        # Clamp to valid RGB range
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        color_hex = f'#{r:02x}{g:02x}{b:02x}'
        colors.append(color_hex)
    
    # Create subplots for each case
    fig = make_subplots(
        rows=1, cols=len(cases),
        subplot_titles=[f'Case {case_id}' for case_id in cases],
        horizontal_spacing=0.15
    )
    
    for col_idx, case_id in enumerate(cases):
        case_data = results[case_id]
        
        # Collect final-step mean_return for each step_size
        for step_size_idx, step_size in enumerate(step_sizes):
            # Only process step_sizes that exist
            available_step_sizes = set()
            for seed_data in case_data.values():
                available_step_sizes.update(seed_data.keys())
            
            if step_size not in available_step_sizes:
                continue
            
            x_positions = []
            y_values = []
            hover_texts = []
            
            for seed in seeds:
                if seed not in case_data:
                    continue
                
                seed_data = case_data[seed]
                if step_size not in seed_data:
                    continue
                
                log_data = seed_data[step_size]
                episode_rewards = log_data.get('episode_rewards', [])
                
                # Extract the final (last) step mean_return
                if episode_rewards:
                    final_entry = episode_rewards[-1]
                    final_step = final_entry.get('step', 0)
                    mean_return = final_entry.get('mean_return', 0)
                    
                    x_positions.append(seed)
                    y_values.append(mean_return)
                    hover_texts.append(
                        f"Seed: {seed}<br>Noise Scale: {step_size}<br>Final Step: {final_step}<br>Return: {mean_return:.2f}"
                    )
            
            if x_positions:
                # Offset x positions so different step_sizes are shown side-by-side
                bar_width = 0.12
                x_offset = (step_size_idx - len(step_sizes) / 2 + 0.5) * bar_width
                x_positions_offset = [x + x_offset for x in x_positions]
                
                fig.add_trace(
                    go.Bar(
                        x=x_positions_offset,
                        y=y_values,
                        name=f'Noise Scale {step_size}',
                        marker=dict(
                            color=colors[step_size_idx % len(colors)],
                            opacity=0.8,
                            line=dict(width=1, color='black')
                        ),
                        text=[f'{v:.2f}' for v in y_values],
                        textposition='outside',
                        textfont=dict(size=9),
                        hovertemplate='%{text}<extra></extra>',
                        showlegend=(col_idx == 0),  # show legend only in the first subplot
                        width=bar_width * 0.8,
                    ),
                    row=1, col=col_idx + 1
                )
        
        # X axis
        fig.update_xaxes(
            title_text="Random Seed",
            tickmode='linear',
            tick0=0,
            dtick=1,
            range=[-0.5, 9.5],
            row=1, col=col_idx + 1
        )
        
        # Add original policy as dashed segments (one per seed)
        if original_returns and case_id in original_returns:
            case_original = original_returns[case_id]
            for seed in seeds:
                if seed in case_original:
                    original_return = case_original[seed]
                    # Draw a short horizontal dashed segment around the seed position
                    x_segment = [seed - 0.4, seed + 0.4]
                    y_segment = [original_return, original_return]
                    fig.add_trace(
                        go.Scatter(
                            x=x_segment,
                            y=y_segment,
                            mode='lines',
                            line=dict(
                                dash='8px 3px',  # custom dash pattern: shorter gaps
                                color='red',
                                width=4  # thicker line
                            ),
                            name='Original',
                            showlegend=(col_idx == 0 and seed == seeds[0]),  # legend once
                            legendgroup='original',
                            hoverinfo='skip',
                            opacity=0.8,
                        ),
                        row=1, col=col_idx + 1
                    )
        
        # Y axis
        fig.update_yaxes(
            title_text="Episode Return (Final Step)",
            row=1, col=col_idx + 1
        )
    
    fig.update_layout(
        title={
            'text': 'Jump & Retrain Results - Final Episode Return vs Random Seed',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'weight': 'bold'}
        },
        height=600,
        width=1400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        barmode='group'  # grouped bars
    )
    
    # Save HTML
    html_path = output_path.with_suffix('.html')
    fig.write_html(str(html_path))
    print(f"Plot saved to: {html_path}")
    
    # Try saving PNG
    try:
        fig.write_image(str(output_path))
        print(f"Plot saved to: {output_path}")
    except Exception as e:
        print(f"Could not save PNG (need kaleido), saved HTML instead")


def plot_jump_retrain_matplotlib(results: Dict, output_path: Path, original_returns: Dict = None):
    """Plot Jump & Retrain results with Matplotlib (bar chart of final-step mean return)."""
    fig, axes = plt.subplots(1, len(results), figsize=(16, 6))
    if len(results) == 1:
        axes = [axes]
    
    step_sizes = [5, 10, 15, 20, 25, 30]
    seeds = list(range(10))
    
    # Define a color gradient for step_size from light to dark (blues).
    light_color_rgb = (135, 206, 250)  # LightSkyBlue
    dark_color_rgb = (25, 25, 112)     # MidnightBlue
    colors = []
    for step_size in step_sizes:
        depth = (step_size - 5) / (30 - 5)  # 0.0 to 1.0
        # Linear interpolation from light to dark
        r = int(light_color_rgb[0] + (dark_color_rgb[0] - light_color_rgb[0]) * depth)
        g = int(light_color_rgb[1] + (dark_color_rgb[1] - light_color_rgb[1]) * depth)
        b = int(light_color_rgb[2] + (dark_color_rgb[2] - light_color_rgb[2]) * depth)
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        color_hex = f'#{r:02x}{g:02x}{b:02x}'
        colors.append(color_hex)
    
    for ax_idx, (case_id, ax) in enumerate(zip(sorted(results.keys()), axes)):
        case_data = results[case_id]
        
        # Prepare bar chart data
        x = np.arange(len(seeds))
        width = 0.12  # bar width
        available_step_sizes = []
        
        # Collect final-step mean_return for each step_size
        step_size_data = {}
        for step_size in step_sizes:
            y_values = []
            for seed in seeds:
                if seed not in case_data:
                    y_values.append(None)
                    continue
                
                seed_data = case_data[seed]
                if step_size not in seed_data:
                    y_values.append(None)
                    continue
                
                log_data = seed_data[step_size]
                episode_rewards = log_data.get('episode_rewards', [])
                
                # Extract the final (last) step mean_return
                if episode_rewards:
                    final_entry = episode_rewards[-1]
                    mean_return = final_entry.get('mean_return', 0)
                    y_values.append(mean_return)
                else:
                    y_values.append(None)
            
            # Only include step_sizes that have data
            if any(v is not None for v in y_values):
                step_size_data[step_size] = y_values
                available_step_sizes.append(step_size)
        
        # Plot bars
        for step_size_idx, step_size in enumerate(available_step_sizes):
            y_values = step_size_data[step_size]
            offset = (step_size_idx - len(available_step_sizes) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, y_values,
                width=width * 0.8,
                label=f'Noise Scale {step_size}',
                color=colors[step_sizes.index(step_size) % len(colors)],
                alpha=0.8,
                edgecolor='black',
                linewidth=1
            )
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, y_values)):
                if val is not None:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.2f}',
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Random Seed', fontsize=12, fontweight='bold')
        ax.set_ylabel('Episode Return (Final Step)', fontsize=12, fontweight='bold')
        ax.set_title(f'Case {case_id}', fontsize=14, fontweight='bold')
        # Add original policy as dashed segments
        if original_returns and case_id in original_returns:
            case_original = original_returns[case_id]
            for seed_idx, seed in enumerate(seeds):
                if seed in case_original:
                    original_return = case_original[seed]
                    # One horizontal dashed segment per seed
                    ax.axhline(
                        y=original_return,
                        xmin=(seed_idx - 0.5) / len(seeds),
                        xmax=(seed_idx + 0.5) / len(seeds),
                        color='red',
                        linestyle='--',
                        dashes=(5, 2),  # custom dash pattern: shorter gaps
                        linewidth=4,  # thicker line
                        alpha=0.8,
                        label='Original' if seed_idx == 0 and ax_idx == 0 else ''
                    )
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'S{s}' for s in seeds])
        ax.grid(True, alpha=0.3, axis='y')
        if ax_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    fig.suptitle('Jump & Retrain Results - Final Episode Return vs Random Seed', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Jump & Retrain results")
    parser.add_argument("--results-dir", type=Path, default=Path("results/jump_retrain"),
                        help="Results directory containing jump_retrain subdirectories")
    parser.add_argument("--models-dir", type=Path, default=Path("models"),
                        help="Models directory containing original trained models")
    parser.add_argument("--output", type=str, default="jump_retrain_plot.png",
                        help="Output plot path")
    args = parser.parse_args()
    
    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print(f"Loading retrain logs from: {results_dir}")
    results = load_retrain_logs(results_dir)
    
    if not results:
        print("Error: No retrain logs found")
        return
    
    print(f"Loaded data for {len(results)} cases:")
    for case_id in sorted(results.keys()):
        num_seeds = len(results[case_id])
        print(f"  Case {case_id}: {num_seeds} seeds")
    
    # Load/evaluate original policy returns
    original_returns = {}
    if args.models_dir.exists():
        print(f"\nLoading original policy returns from: {args.models_dir}")
        original_returns = load_original_policy_returns(args.models_dir)
    else:
        print(f"\nWarning: Models directory not found: {args.models_dir}, skipping original policy evaluation")
    
    output_path = Path(args.output)
    
    if HAS_PLOTLY:
        plot_jump_retrain_plotly(results, output_path, original_returns)
    elif HAS_MATPLOTLIB:
        plot_jump_retrain_matplotlib(results, output_path, original_returns)
    else:
        print("Error: No plotting library available (need plotly or matplotlib)")
        return
    
    print("Plotting completed!")


if __name__ == "__main__":
    main()

