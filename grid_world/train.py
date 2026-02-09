# train_ppo.py
import argparse
import numpy as np
import torch
from pathlib import Path

from env import make_env
from ppo_simple import PPO


def eval_policy(env, model, n_episodes=30, deterministic: bool = True):
    ep_returns = []
    ep_lens = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, truncated, info = env.step(int(action))
            total += r
            steps += 1
            done = terminated or truncated
        ep_returns.append(total)
        ep_lens.append(steps)
    return np.mean(ep_returns), np.std(ep_returns), np.mean(ep_lens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GridWorld PPO")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=20_000, help="Total training timesteps")
    parser.add_argument(
        "--case-id",
        type=int,
        default=1,
        help="Environment spec ID (env_specs/case_{id}.json)",
    )
    args = parser.parse_args()
    
    seed = args.seed
    total_timesteps = args.total_timesteps
    case_id = args.case_id
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Case ID: {case_id}, seed: {seed}, total timesteps: {total_timesteps}")
    
    env = make_env(seed, case_id=case_id)
    
    # Get observation/action dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Observation dim: {obs_dim}, action dim: {action_dim}")
    
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
        device="cpu",
    )
    
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps)
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"gridworld_ppo_case_{case_id}_seed_{seed}.pt"
    torch.save({
        'model_state_dict': model.agent.state_dict(),
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'seed': seed,
        'total_timesteps': total_timesteps,
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Evaluation (report deterministic vs stochastic to match training-vs-eval conventions)
    eval_env = make_env(seed=123, case_id=case_id)
    mean_r_det, std_r_det, mean_len_det = eval_policy(eval_env, model, n_episodes=50, deterministic=True)
    mean_r_sto, std_r_sto, mean_len_sto = eval_policy(eval_env, model, n_episodes=50, deterministic=False)
    print("\nEvaluation (Deterministic=True):")
    print(f"Mean return: {mean_r_det:.2f} ± {std_r_det:.2f}")
    print(f"Mean episode length: {mean_len_det:.1f}")
    print("\nEvaluation (Deterministic=False / sampled actions):")
    print(f"Mean return: {mean_r_sto:.2f} ± {std_r_sto:.2f}")
    print(f"Mean episode length: {mean_len_sto:.1f}")
    
    # Optional: print one trajectory (e.g., to diagnose candy loops)
    print("\nExample trajectory:")
    obs, _ = eval_env.reset()
    for t in range(60):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = eval_env.step(int(action))
        if info.get("event"):
            print(f"Step {t}: event={info['event']}, reward={r:.2f}, pos={eval_env.env.agent_pos}")
        if terminated or truncated:
            print(f"Finished: step {t}, final_reward={r:.2f}")
            break
