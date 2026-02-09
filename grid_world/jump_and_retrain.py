#!/usr/bin/env python3
"""
Jump & Retrain for GridWorld PPO policies.

Jump & Retrain procedure:
  - Load the base model as \(\theta_0\)
  - Sample a random direction \(v\) in parameter space (filter-wise normalization)
  - Construct the jump point \(\theta_{jump} = \theta_0 + \text{step\_size} \cdot v\)
  - Continue training from \(\theta_{jump}\) for extra_steps
  - Log episode rewards every log_interval steps
"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict
from collections import deque

from env import make_env
from ppo_simple import PPO, PPOAgent


def flatten_params(state_dict: dict) -> Tuple[torch.Tensor, List[Tuple[str, Tuple[int, ...]]]]:
    """
    Flatten all tensor parameters in a state_dict into a single vector, and record
    (key, shape) metadata for reconstruction.
    """
    flats = []
    meta = []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        meta.append((k, v.shape))
        flats.append(v.reshape(-1))
    if not flats:
        raise RuntimeError("Empty state_dict? No tensor parameters found.")
    flat_vec = torch.cat(flats, dim=0)
    return flat_vec, meta


def unflatten_params(flat_vec: torch.Tensor,
                     meta: List[Tuple[str, Tuple[int, ...]]]) -> dict:
    """
    Reconstruct a state_dict from a flat parameter vector using the metadata
    returned by flatten_params.
    """
    state_dict = {}
    idx = 0
    for k, shape in meta:
        numel = int(np.prod(shape))
        chunk = flat_vec[idx: idx + numel]
        state_dict[k] = chunk.view(*shape).clone()
        idx += numel
    return state_dict


def sample_random_direction_filter_norm(state_dict: dict,
                                        device: torch.device,
                                        rng: np.random.Generator) -> torch.Tensor:
    """
    Filter-wise normalization：
    - For each parameter tensor W:
        1) Sample same-shaped noise D ~ N(0, 1)
        2) Normalize D so that ||D||_2 = 1
        3) Rescale so that ||D||_2 matches ||W||_2
    - Finally, flatten and concatenate all tensors into a direction vector v.
    """
    flats = []
    for k, w in state_dict.items():
        if not isinstance(w, torch.Tensor):
            continue
        w = w.to(device)
        shape = w.shape
        numel = w.numel()

        noise = rng.standard_normal(size=(numel,)).astype(np.float32)
        d = torch.from_numpy(noise).to(device).view(*shape)

        # Step 1: normalize within this tensor
        d_norm = torch.linalg.norm(d)
        if d_norm > 0:
            d = d / d_norm

        # Step 2: rescale to match the original weight norm
        w_norm = torch.linalg.norm(w)
        if w_norm > 0:
            d = d * w_norm

        flats.append(d.reshape(-1))

    if not flats:
        raise RuntimeError("No usable tensor parameters found in state_dict.")

    v = torch.cat(flats, dim=0)
    # Final global L2 normalization
    v = v / (torch.linalg.norm(v) + 1e-8)
    return v


def make_jump_state_dict(theta0: torch.Tensor,
                         direction_v: torch.Tensor,
                         step_size: float,
                         meta: List[Tuple[str, Tuple[int, ...]]]) -> dict:
    """
    Given base parameter vector \(\theta_0\), direction v, and step_size,
    construct the state_dict for \(\theta_{jump}\).
    """
    theta_jump = theta0 + step_size * direction_v
    return unflatten_params(theta_jump, meta)


def load_model(model_path: Path, device: str = "cpu"):
    """Load a saved PPOAgent model checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    obs_dim = checkpoint['obs_dim']
    action_dim = checkpoint['action_dim']
    
    # Create agent
    agent = PPOAgent(obs_dim, action_dim, hidden_dim=64, lr=3e-4)
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.to(device)
    
    return agent, checkpoint


def run_jump_and_retrain(
    base_model_path: Path,
    case_id: int,
    seed: int,
    step_size: float,
    jr_index: int,
    extra_steps: int,
    log_interval: int,
    output_dir: Path,
    device: str = "cpu",
):
    """
    Run one Jump & Retrain experiment:
      - Load base model as \(\theta_0\)
      - Sample random direction v, build \(\theta_{jump} = \theta_0 + step\_size \cdot v\)
      - Continue training from \(\theta_{jump}\) for extra_steps
      - Log episode rewards every log_interval steps
    """
    print(f"\n{'='*60}")
    print(f"Jump & Retrain: Case {case_id}, Seed {seed}, Step Size {step_size}")
    print(f"{'='*60}")
    
    # === 1. Load base model ===
    print(f"Loading base model: {base_model_path}")
    base_agent, base_checkpoint = load_model(base_model_path, device=device)
    
    obs_dim = base_checkpoint['obs_dim']
    action_dim = base_checkpoint['action_dim']
    base_seed = base_checkpoint.get('seed', seed)
    
    # === 2. Create environment and PPO wrapper ===
    env = make_env(seed=base_seed, case_id=case_id)
    model = PPO(
        env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        value_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        device=device,
    )
    
    # === 3. Get \(\theta_0\) and sample a random direction ===
    base_state_dict = base_agent.state_dict()
    theta0_flat, meta = flatten_params(base_state_dict)
    
    rng = np.random.default_rng(seed=jr_index * 1000 + int(step_size * 100))
    direction_v = sample_random_direction_filter_norm(
        base_state_dict, 
        device=torch.device(device), 
        rng=rng
    )
    
    # === 4. Construct \(\theta_{jump}\) and load into the model ===
    theta_jump_sd = make_jump_state_dict(theta0_flat, direction_v, step_size, meta)
    
    # Load into PPO agent
    own_state = model.agent.state_dict()
    for k, v_param in theta_jump_sd.items():
        if k in own_state and own_state[k].shape == v_param.shape:
            own_state[k] = v_param
    model.agent.load_state_dict(own_state)
    
    print(f"Jump distance: {step_size:.2f} (L2 norm in parameter space)")
    print(f"Starting retrain from θ_jump for {extra_steps} steps...")
    
    # === 5. Train and log ===
    output_dir.mkdir(parents=True, exist_ok=True)
    log_data = {
        'case_id': case_id,
        'seed': seed,
        'step_size': step_size,
        'jr_index': jr_index,
        'extra_steps': extra_steps,
        'log_interval': log_interval,
        'episode_rewards': [],  # [(step, mean_reward, std_reward), ...]
    }
    
    num_rollouts = (extra_steps + model.n_steps - 1) // model.n_steps  # ceil
    episode_returns = deque(maxlen=100)
    total_steps = 0
    last_log_step = 0
    
    print(f"Training: {num_rollouts} rollouts, {extra_steps} total steps")
    
    for rollout in range(num_rollouts):
        model.collect_rollout()
        rollout_steps = len(model.buffer["dones"])
        total_steps += rollout_steps
        model.update()
        
        # Track episode information
        episode_return = 0
        for i, done in enumerate(model.buffer["dones"]):
            episode_return += model.buffer["rewards"][i]
            if done:
                episode_returns.append(episode_return)
                episode_return = 0
        
        # Log every log_interval steps
        if total_steps - last_log_step >= log_interval or rollout == num_rollouts - 1:
            if len(episode_returns) > 0:
                mean_return = np.mean(episode_returns)
                std_return = np.std(episode_returns) if len(episode_returns) > 1 else 0.0
                log_data['episode_rewards'].append({
                    'step': total_steps,
                    'mean_return': float(mean_return),
                    'std_return': float(std_return),
                    'num_episodes': len(episode_returns),
                })
                print(f"Step {total_steps}/{extra_steps} | "
                      f"Mean Return: {mean_return:.2f} ± {std_return:.2f} | "
                      f"Episodes: {len(episode_returns)}")
                last_log_step = total_steps
        
        # Stop early if we've reached the target step budget
        if total_steps >= extra_steps:
            break
    
    # === 6. Save model and logs ===
    model_path = output_dir / f"jump_retrain_model.pt"
    torch.save({
        'model_state_dict': model.agent.state_dict(),
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'case_id': case_id,
        'seed': seed,
        'step_size': step_size,
        'jr_index': jr_index,
        'extra_steps': extra_steps,
        'total_steps': total_steps,
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    log_path = output_dir / "retrain_log.json"
    with log_path.open('w') as f:
        json.dump(log_data, f, indent=2)
    print(f"Log saved to: {log_path}")
    
    print(f"Jump & Retrain completed: {output_dir.name}")
    return log_data


def main():
    parser = argparse.ArgumentParser(description="Jump & Retrain for GridWorld PPO")
    parser.add_argument("--models-dir", type=Path, default=Path("models"),
                        help="Directory containing trained models.")
    parser.add_argument("--case-id", type=int, required=True, choices=[1, 2],
                        help="Case ID (1 or 2).")
    parser.add_argument("--seed", type=int, required=True,
                        help="Base model seed (0-9).")
    parser.add_argument("--step-sizes", type=float, nargs="+",
                        default=[10.0, 30.0, 90.0],
                        help="List of jump step sizes.")
    parser.add_argument("--extra-steps", type=int, default=10000,
                        help="Number of retraining steps.")
    parser.add_argument("--log-interval", type=int, default=1000,
                        help="Log every N steps.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Training device.")
    parser.add_argument("--output-root", type=Path, default=Path("results/jump_retrain"),
                        help="Output root directory.")
    parser.add_argument("--jr-index", type=int, default=None,
                        help="Jump-Retrain index (for array jobs).")
    args = parser.parse_args()
    
    # Load base model
    base_model_path = args.models_dir / f"gridworld_ppo_case_{args.case_id}_seed_{args.seed}.pt"
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")
    
    # If jr_index is specified, run only one step_size
    if args.jr_index is not None:
        if len(args.step_sizes) != 1:
            raise ValueError("When --jr-index is set, you must provide exactly one step_size.")
        step_size = args.step_sizes[0]
        jr_index = args.jr_index
        output_dir = args.output_root / f"case_{args.case_id}_seed_{args.seed}_step{step_size:.0f}_jr{jr_index}"
        run_jump_and_retrain(
            base_model_path=base_model_path,
            case_id=args.case_id,
            seed=args.seed,
            step_size=step_size,
            jr_index=jr_index,
            extra_steps=args.extra_steps,
            log_interval=args.log_interval,
            output_dir=output_dir,
            device=args.device,
        )
    else:
        # Run all step_sizes
        for jr_idx, step_size in enumerate(args.step_sizes):
            output_dir = args.output_root / f"case_{args.case_id}_seed_{args.seed}_step{step_size:.0f}_jr{jr_idx}"
            run_jump_and_retrain(
                base_model_path=base_model_path,
                case_id=args.case_id,
                seed=args.seed,
                step_size=step_size,
                jr_index=jr_idx,
                extra_steps=args.extra_steps,
                log_interval=args.log_interval,
                output_dir=output_dir,
                device=args.device,
            )


if __name__ == "__main__":
    main()

