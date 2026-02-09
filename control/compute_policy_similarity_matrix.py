#!/usr/bin/env python3
"""
Compute a policy similarity matrix for multiple tasks.
Adapted from compute_similarity_baseline.py.
"""

import os
import sys
import json
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Add pufferlib path
pufferlib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if pufferlib_path in sys.path:
    sys.path.remove(pufferlib_path)
sys.path.insert(0, pufferlib_path)

from train_landscape_controlled import load_config_manually
import pufferlib.pufferl as pufferl
import gymnasium

JS_MAX = math.log(2.0)


def _reduce_log_prob(log_prob: torch.Tensor) -> torch.Tensor:
    """Sum log_prob along the last dimension until it becomes 1D."""
    while log_prob.ndim > 1:
        log_prob = log_prob.sum(dim=-1)
    return log_prob


def compute_js_divergence_distributions(dist_a, dist_b, num_samples: int = 1024) -> float:
    """Compute Jensen-Shannon divergence between two distributions via sampling."""
    try:
        sample_fn = dist_a.rsample if hasattr(dist_a, 'rsample') else dist_a.sample
        samples_a = sample_fn((num_samples,))
        log_p_a = _reduce_log_prob(dist_a.log_prob(samples_a))
        log_q_a = _reduce_log_prob(dist_b.log_prob(samples_a))
        log_m_a = torch.logaddexp(log_p_a, log_q_a) - math.log(2.0)
        kl_a = (log_p_a - log_m_a).mean()

        sample_fn_b = dist_b.rsample if hasattr(dist_b, 'rsample') else dist_b.sample
        samples_b = sample_fn_b((num_samples,))
        log_q_b = _reduce_log_prob(dist_b.log_prob(samples_b))
        log_p_b = _reduce_log_prob(dist_a.log_prob(samples_b))
        log_m_b = torch.logaddexp(log_q_b, log_p_b) - math.log(2.0)
        kl_b = (log_q_b - log_m_b).mean()

        js = 0.5 * (kl_a + kl_b)
        return float(max(js.item(), 0.0))
    except Exception as e:
        print(f"Warning: JS divergence computation failed: {e}")
        return 0.0


def load_policy(env_name, model_path=None):
    """Load policy."""
    args = load_config_manually(env_name)
    args['train']['device'] = 'cpu'
    args['vec']['backend'] = 'Serial'
    args['vec']['num_envs'] = 1
    args['vec']['num_workers'] = 1
    args['vec']['batch_size'] = 1
    
    vecenv = pufferl.load_env(env_name, args)
    policy = pufferl.load_policy(args, vecenv, env_name)
    
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        policy.load_state_dict(state_dict)
    
    policy.to('cpu')
    policy.eval()
    
    return policy, vecenv


def sample_states(vecenv, policy, num_samples=500, seed=42):
    """Sample states."""
    states = []
    obs, _ = vecenv.reset(seed=seed)
    lstm_h, lstm_c = None, None
    
    for _ in range(num_samples):
        states.append(obs.copy().flatten())
        
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        state_dict = {'lstm_h': lstm_h, 'lstm_c': lstm_c}
        
        with torch.no_grad():
            action_dist, _ = policy.forward_eval(obs_tensor, state_dict)
            lstm_h = state_dict.get('lstm_h', None)
            lstm_c = state_dict.get('lstm_c', None)
            
            if isinstance(action_dist, (tuple, list)):
                action_dist = action_dist[0]
            if hasattr(action_dist, 'sample'):
                action = action_dist.sample()
            else:
                action = action_dist
        
        obs, _, terminated, truncated, _ = vecenv.step(action.cpu().numpy())
        if terminated or truncated:
            obs, _ = vecenv.reset()
            lstm_h, lstm_c = None, None
    
    return np.array(states)


def compute_average_js(policy_a, policy_b, states):
    """Compute average JS divergence over a given set of states."""
    divergences = []
    
    for state in states:
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        state_dict_a = {'lstm_h': None, 'lstm_c': None}
        state_dict_b = {'lstm_h': None, 'lstm_c': None}
        
        with torch.no_grad():
            dist_a, _ = policy_a.forward_eval(state_tensor, state_dict_a)
            dist_b, _ = policy_b.forward_eval(state_tensor, state_dict_b)
        
        if isinstance(dist_a, (tuple, list)):
            dist_a = dist_a[0]
        if isinstance(dist_b, (tuple, list)):
            dist_b = dist_b[0]
        
        js = compute_js_divergence_distributions(dist_a, dist_b)
        divergences.append(js)
    
    return np.mean(divergences)


def compute_similarity_matrix(env_name, task_ids, model_dir_template, num_samples=500):
    """
    Compute the similarity matrix for all policies in an environment.
    
    Args:
        env_name: Environment name
        task_ids: Task id list
        model_dir_template: Model path template, e.g. 'models/controlled_task_{}/final_model.pt'
        num_samples: Number of sampled states
    """
    print(f"\n{'='*80}")
    print(f"Computing similarity matrix for {env_name}")
    print(f"{'='*80}")
    
    # Load all policies
    print(f"Loading {len(task_ids)} policies...")
    policies = []
    vecenvs = []
    
    for task_id in tqdm(task_ids, desc="Loading policies"):
        model_path = model_dir_template.format(task_id)
        if not os.path.exists(model_path):
            print(f"Warning: Model not found: {model_path}")
            return None
        
        policy, vecenv = load_policy(env_name, model_path)
        policies.append(policy)
        vecenvs.append(vecenv)
    
    # Sample states using the first policy (all policies share the same state set)
    print(f"\nSampling {num_samples} states...")
    states = sample_states(vecenvs[0], policies[0], num_samples=num_samples)
    print(f"State shape: {states.shape}")
    
    # Compute similarity matrix
    print(f"\nComputing {len(policies)}x{len(policies)} similarity matrix...")
    similarity_matrix = np.zeros((len(policies), len(policies)))
    
    for i in tqdm(range(len(policies)), desc="Computing similarities"):
        for j in range(i, len(policies)):
            if i == j:
                similarity_matrix[i, j] = 0.0  # self distance
            else:
                js_div = compute_average_js(policies[i], policies[j], states)
                similarity_matrix[i, j] = js_div
                similarity_matrix[j, i] = js_div  # symmetric
    
    # Close envs
    for vecenv in vecenvs:
        vecenv.close()
    
    return similarity_matrix


def plot_similarity_matrix(matrix, env_name, task_ids, output_dir):
    """Plot similarity matrix heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Heatmap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', origin='upper', vmin=0.0, vmax=JS_MAX)
    
    # Labels
    labels = [f'T{task_id}' for task_id in task_ids]
    ax.set_xticks(np.arange(len(task_ids)))
    ax.set_yticks(np.arange(len(task_ids)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    # Value annotations
    for i in range(len(task_ids)):
        for j in range(len(task_ids)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=7)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('JS Divergence (0 - log2)', rotation=270, labelpad=20)
    cbar.set_ticks([0.0, JS_MAX / 2.0, JS_MAX])
    cbar.set_ticklabels(['0', f'{JS_MAX/2.0:.3f}', f'{JS_MAX:.3f}'])
    
    # Title
    ax.set_title(f'{env_name} - Policy Similarity Matrix\n(Lower = More Similar)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f'{env_name}_similarity_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Similarity matrix figure saved: {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser("Compute JS similarity matrix for tasks")
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument(
        "--task-ids", type=int, nargs="+", required=True, help="Task id list"
    )
    parser.add_argument(
        "--model-template",
        type=str,
        default="models/controlled_task_{}/final_model.pt",
        help="Model path template (use {} as task-id placeholder).",
    )
    parser.add_argument(
        "--num-samples", type=int, default=500, help="Number of sampled states"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_outputs/hc_similarity",
        help="Output directory (stores matrix + plot).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env_name = args.env_name
    task_ids = args.task_ids
    model_template = args.model_template
    num_samples = args.num_samples
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    missing_models = [
        task_id
        for task_id in task_ids
        if not os.path.exists(model_template.format(task_id))
    ]
    if missing_models:
        raise FileNotFoundError(f"Missing models: {missing_models}")

    similarity_matrix = compute_similarity_matrix(
        env_name, task_ids, model_template, num_samples=num_samples
    )
    matrix_path = os.path.join(output_dir, f"{env_name}_similarity_matrix.npy")
    np.save(matrix_path, similarity_matrix)
    print(f"✓ Similarity matrix saved: {matrix_path}")

    plot_path = plot_similarity_matrix(
        similarity_matrix, env_name, task_ids, output_dir
    )

    upper_tri = similarity_matrix[np.triu_indices(len(task_ids), k=1)]
    stats = {
        "mean": float(np.mean(upper_tri)),
        "std": float(np.std(upper_tri)),
        "min": float(np.min(upper_tri)),
        "max": float(np.max(upper_tri)),
        "median": float(np.median(upper_tri)),
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"env_name": env_name, "task_ids": task_ids, "stats": stats}, f, indent=2)
    print(f"✓ Summary saved: {summary_path}")

    print("\nStats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

