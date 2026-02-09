#!/usr/bin/env python3
"""
Evaluate Walker2d policy performance.
For each policy, run multiple episodes and compute mean episode return.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add pufferlib path
pufferlib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if pufferlib_path in sys.path:
    sys.path.remove(pufferlib_path)
sys.path.insert(0, pufferlib_path)

from train_landscape_controlled import load_config_manually
import pufferlib.pufferl as pufferl


def load_policy(env_name, model_path=None, vec_stats_path=None):
    """Load policy."""
    args = load_config_manually(env_name)
    args['train']['device'] = 'cpu'
    args['vec']['backend'] = 'Serial'
    args['vec']['num_envs'] = 1
    args['vec']['num_workers'] = 1
    args['vec']['batch_size'] = 1
    
    vecenv = pufferl.load_env(env_name, args)
    
    # Load normalization stats (if available)
    if vec_stats_path and os.path.exists(vec_stats_path):
        from train_landscape_controlled import load_vec_stats
        if load_vec_stats(vecenv.driver_env, vec_stats_path):
            print(f"  Loaded normalization stats: {vec_stats_path}")
        else:
            print(f"  Warning: failed to load normalization stats: {vec_stats_path}")
    
    policy = pufferl.load_policy(args, vecenv, env_name)
    
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        policy.load_state_dict(state_dict)
    
    policy.to('cpu')
    policy.eval()
    
    return policy, vecenv


def evaluate_policy(policy, vecenv, num_episodes=10, seed=42):
    """
    Evaluate policy performance.
    
    Args:
        policy: Policy network
        vecenv: Vectorized environment
        num_episodes: Number of episodes to evaluate
        seed: Random seed
    
    Returns:
        dict: mean return, std, and per-episode returns
    """
    episode_returns = []
    episode_lengths = []
    
    # Reset env
    obs, _ = vecenv.reset(seed=seed)
    lstm_h, lstm_c = None, None
    
    current_episode_return = 0.0
    current_episode_length = 0
    
    for _ in range(num_episodes * 1000):  # run enough steps at most
        # Action
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        state_dict = {'lstm_h': lstm_h, 'lstm_c': lstm_c}
        
        with torch.no_grad():
            action_dist, _ = policy.forward_eval(obs_tensor, state_dict)
            lstm_h = state_dict.get('lstm_h', None)
            lstm_c = state_dict.get('lstm_c', None)
            
            # Use deterministic action (mean) instead of sampling
            if isinstance(action_dist, (tuple, list)):
                action_dist = action_dist[0]
            if hasattr(action_dist, 'mean'):
                action = action_dist.mean
            elif hasattr(action_dist, 'sample'):
                action = action_dist.sample()
            else:
                action = action_dist
        
        # Step env
        obs, reward, terminated, truncated, info = vecenv.step(action.cpu().numpy())
        
        # Try reading raw episode return from info (if available)
        # Training may use episode_return_raw or episode_return
        episode_return = None
        if info and len(info) > 0:
            info_entry = info[0] if isinstance(info, (list, tuple)) else info
            if isinstance(info_entry, dict):
                # Prefer raw (unnormalized) episode return
                for key in ("episode_return_raw", "raw_episode_return", "episode_return"):
                    if key in info_entry:
                        episode_return = float(info_entry[key])
                        break
        
        # Fallback: accumulate rewards
        if episode_return is None:
            current_episode_return += float(reward[0])
        else:
            # If info provides episode return, use it
            current_episode_return = episode_return
        
        current_episode_length += 1
        
        # Episode end
        if terminated[0] or truncated[0]:
            episode_returns.append(current_episode_return)
            episode_lengths.append(current_episode_length)
            
            if len(episode_returns) >= num_episodes:
                break
            
            # Reset env
            obs, _ = vecenv.reset()
            lstm_h, lstm_c = None, None
            current_episode_return = 0.0
            current_episode_length = 0
    
    if len(episode_returns) == 0:
        return {
            'mean_return': 0.0,
            'std_return': 0.0,
            'min_return': 0.0,
            'max_return': 0.0,
            'num_episodes': 0,
            'episode_returns': [],
            'episode_lengths': [],
        }
    
    return {
        'mean_return': float(np.mean(episode_returns)),
        'std_return': float(np.std(episode_returns)),
        'min_return': float(np.min(episode_returns)),
        'max_return': float(np.max(episode_returns)),
        'num_episodes': len(episode_returns),
        'episode_returns': [float(r) for r in episode_returns],
        'episode_lengths': [int(l) for l in episode_lengths],
    }


def evaluate_all_policies(env_name, task_ids, model_template, num_episodes=10, output_dir='evaluation_results'):
    """
    Evaluate all policies.
    
    Args:
        env_name: Environment name
        task_ids: Task id list
        model_template: Model path template, e.g. 'models/controlled_task_{}/final_model.pt'
        num_episodes: Episodes per policy
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"Evaluating {len(task_ids)} policies for {env_name}")
    print(f"{'='*80}\n")
    
    for task_id in tqdm(task_ids, desc="Evaluating policies"):
        model_path = model_template.format(task_id)
        
        if not os.path.exists(model_path):
            print(f"Warning: model not found: {model_path}")
            results[task_id] = {
                'error': f'Model not found: {model_path}',
                'model_path': model_path,
            }
            continue
        
        try:
            # Load policy (and normalization stats)
            vec_stats_path = model_template.format(task_id).replace('final_model.pt', 'vec_stats.npz').replace('best_model.pt', 'vec_stats.npz')
            vec_stats_path = os.path.join(os.path.dirname(model_path), 'vec_stats.npz')
            policy, vecenv = load_policy(env_name, model_path, vec_stats_path=vec_stats_path)
            
            # Evaluate
            eval_result = evaluate_policy(policy, vecenv, num_episodes=num_episodes, seed=42)
            
            # Save result
            results[task_id] = {
                'task_id': task_id,
                'model_path': model_path,
                'env_name': env_name,
                'num_episodes': eval_result['num_episodes'],
                'mean_return': eval_result['mean_return'],
                'std_return': eval_result['std_return'],
                'min_return': eval_result['min_return'],
                'max_return': eval_result['max_return'],
                'episode_returns': eval_result['episode_returns'],
                'episode_lengths': eval_result['episode_lengths'],
            }
            
            # Close env
            vecenv.close()
            
            print(f"Task {task_id}: mean_return={eval_result['mean_return']:.2f} ± {eval_result['std_return']:.2f}")
            
        except Exception as e:
            print(f"Error: evaluation failed for task {task_id}: {e}")
            results[task_id] = {
                'error': str(e),
                'model_path': model_path,
            }
    
    # Save all results
    output_file = os.path.join(output_dir, f'{env_name}_evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Evaluation results saved: {output_file}")
    
    # Summary stats
    valid_results = [r for r in results.values() if 'mean_return' in r]
    if valid_results:
        mean_returns = [r['mean_return'] for r in valid_results]
        print("\nSummary:")
        print(f"  Mean return: {np.mean(mean_returns):.2f} ± {np.std(mean_returns):.2f}")
        print(f"  Min return: {np.min(mean_returns):.2f}")
        print(f"  Max return: {np.max(mean_returns):.2f}")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser("Evaluate Walker2d policies")
    parser.add_argument("--env-name", type=str, default="Walker2d-v4")
    parser.add_argument(
        "--task-ids", type=int, nargs="+", default=list(range(16)),
        help="Task id list"
    )
    parser.add_argument(
        "--model-template",
        type=str,
        default="models/controlled_task_{}/final_model.pt",
        help="Model path template (use {} as task-id placeholder).",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=10,
        help="Episodes per policy"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    evaluate_all_policies(
        env_name=args.env_name,
        task_ids=args.task_ids,
        model_template=args.model_template,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

