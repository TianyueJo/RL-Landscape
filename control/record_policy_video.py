#!/usr/bin/env python3
"""
Record policy rollout videos in Mujoco environments.
"""

import os
import sys
import argparse
import copy
import numpy as np
import torch
import imageio
import time
from pathlib import Path
from typing import Optional, Tuple

# Add pufferlib path
pufferlib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if pufferlib_path in sys.path:
    sys.path.remove(pufferlib_path)
sys.path.insert(0, pufferlib_path)

import pufferlib.pufferl as pufferl
import gymnasium
from gymnasium.wrappers import NormalizeObservation, NormalizeReward

from train_landscape_controlled import load_config_manually
from train_sb3_lstm_landscape import load_sb3_config, apply_overrides

DEFAULT_STATS_WARMUP_STEPS = 65_536
DEFAULT_STATS_NUM_ENVS = 32


def iter_wrapped_envs(env):
    """Yield env and all nested .env wrappers to help locate Normalize wrappers."""
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        yield current
        visited.add(id(current))
        current = getattr(current, "env", None)


def get_normalize_wrappers(puffer_env) -> Tuple[Optional[NormalizeObservation], Optional[NormalizeReward]]:
    obs_wrapper = None
    rew_wrapper = None
    for wrapper in iter_wrapped_envs(puffer_env):
        if obs_wrapper is None and isinstance(wrapper, NormalizeObservation):
            obs_wrapper = wrapper
        if rew_wrapper is None and isinstance(wrapper, NormalizeReward):
            rew_wrapper = wrapper
        if obs_wrapper is not None and rew_wrapper is not None:
            break
    return obs_wrapper, rew_wrapper


def _find_episode_info(info_obj) -> Optional[dict]:
    """Recursively find an info dict containing episode_return (from EpisodeStats)."""
    if isinstance(info_obj, dict):
        if 'episode_return' in info_obj:
            return info_obj
        for value in info_obj.values():
            found = _find_episode_info(value)
            if found:
                return found
    elif isinstance(info_obj, (list, tuple)):
        for item in info_obj:
            found = _find_episode_info(item)
            if found:
                return found
    return None


def save_vec_stats(puffer_env, stats_path: Path) -> bool:
    """Persist NormalizeObservation/Reward statistics to disk."""
    obs_wrapper, rew_wrapper = get_normalize_wrappers(puffer_env)
    data = {}
    if obs_wrapper is not None:
        data["obs_mean"] = obs_wrapper.obs_rms.mean
        data["obs_var"] = obs_wrapper.obs_rms.var
        data["obs_count"] = np.array([obs_wrapper.obs_rms.count], dtype=np.float64)
    if rew_wrapper is not None:
        data["rew_mean"] = np.array([rew_wrapper.return_rms.mean], dtype=np.float64)
        data["rew_var"] = np.array([rew_wrapper.return_rms.var], dtype=np.float64)
        data["rew_count"] = np.array([rew_wrapper.return_rms.count], dtype=np.float64)
    if not data:
        return False
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(stats_path, **data)
    return True


def load_vec_stats(puffer_env, stats_path: Path) -> bool:
    """Load saved NormalizeObservation/Reward statistics into environment wrappers."""
    if stats_path is None or not stats_path.exists():
        return False
    obs_wrapper, rew_wrapper = get_normalize_wrappers(puffer_env)
    if obs_wrapper is None and rew_wrapper is None:
        return False
    data = np.load(stats_path, allow_pickle=False)
    if obs_wrapper is not None and all(key in data for key in ("obs_mean", "obs_var", "obs_count")):
        obs_wrapper.obs_rms.mean = data["obs_mean"]
        obs_wrapper.obs_rms.var = data["obs_var"]
        obs_wrapper.obs_rms.count = float(data["obs_count"].item())
    if rew_wrapper is not None and all(key in data for key in ("rew_mean", "rew_var", "rew_count")):
        rew_wrapper.return_rms.mean = float(data["rew_mean"].item())
        rew_wrapper.return_rms.var = float(data["rew_var"].item())
        rew_wrapper.return_rms.count = float(data["rew_count"].item())
    return True


def sample_action_from_dist(action_obj, deterministic: bool = False):
    """Handle the different return types from policy.forward_eval."""
    dist_obj = action_obj
    if isinstance(dist_obj, (tuple, list)):
        dist_obj = dist_obj[0]
    if hasattr(dist_obj, "sample"):
        if deterministic:
            if hasattr(dist_obj, "mean"):
                action = dist_obj.mean
            elif hasattr(dist_obj, "probs"):
                action = torch.argmax(dist_obj.probs, dim=-1)
            elif hasattr(dist_obj, "loc"):  # e.g., Normal distribution alias
                action = dist_obj.loc
            else:
                action = dist_obj.sample()
        else:
            action = dist_obj.sample()
    else:
        action = dist_obj
    return action


def rollout_vecenv_for_stats(policy,
                             vecenv,
                             total_steps: int,
                             deterministic: bool = True,
                             seed: Optional[int] = None):
    """Run a short deterministic rollout on the large vectorized env to accumulate stats."""
    device = next(policy.parameters()).device
    state_dict = {"lstm_h": None, "lstm_c": None}
    reset_seed = 0 if seed is None else seed
    vecenv.async_reset(seed=reset_seed)
    obs, rewards, terms, truncs, infos, env_ids, masks = vecenv.recv()
    agents_per_batch = getattr(vecenv, "agents_per_batch", obs.shape[0])
    total_steps = max(total_steps, agents_per_batch)
    steps = 0
    while steps < total_steps:
        obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)
        with torch.no_grad():
            action_obj, _ = policy.forward_eval(obs_tensor, state_dict)
        actions = sample_action_from_dist(action_obj, deterministic=deterministic)
        if isinstance(actions, torch.Tensor):
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = np.asarray(actions)
        vecenv.send(actions_np)
        obs, rewards, terms, truncs, infos, env_ids, masks = vecenv.recv()
        steps += obs.shape[0]


def _configure_stats_env(args: dict,
                         stats_num_envs: Optional[int]) -> dict:
    """Return a deep-copied args dict configured for local Serial rollout."""
    stats_args = copy.deepcopy(args)
    vec_cfg = stats_args.setdefault('vec', {})
    original_envs = int(vec_cfg.get('num_envs', 1) or 1)
    target_envs = original_envs
    if stats_num_envs is not None and stats_num_envs > 0:
        target_envs = min(original_envs, stats_num_envs)
    else:
        target_envs = min(original_envs, DEFAULT_STATS_NUM_ENVS)

    vec_cfg['backend'] = 'Serial'
    vec_cfg['num_envs'] = target_envs
    vec_cfg['num_workers'] = target_envs
    vec_cfg['batch_size'] = target_envs
    return stats_args


def generate_vec_stats(env_name: str,
                       config_path: str,
                       stats_path: Path,
                       model_path: str,
                       warmup_steps: int,
                       train_seed: int,
                       deterministic: bool = True,
                       device: str = "cpu",
                       stats_num_envs: Optional[int] = None):
    """Approximate training-time Normalize stats by running the policy on the original vectorized env."""
    args = load_config_manually(env_name)
    env_cfg = load_sb3_config(config_path, env_name)
    apply_overrides(args, env_cfg, warmup_steps)
    args["train"]["seed"] = train_seed
    args["vec"]["seed"] = train_seed
    args["train"]["env"] = env_name
    args["train"]["device"] = device
    args["train"]["use_rnn"] = True
    args["train"]["data_dir"] = os.path.join("results", f"stats_warmup_{env_name}_{train_seed}")

    stats_args = _configure_stats_env(args, stats_num_envs)
    vecenv = pufferl.load_env(env_name, stats_args)
    policy = pufferl.load_policy(stats_args, vecenv, env_name)

    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict)
    policy.eval()

    rollout_vecenv_for_stats(
        policy,
        vecenv,
        warmup_steps,
        deterministic=deterministic,
        seed=train_seed,
    )
    if save_vec_stats(vecenv.driver_env, stats_path):
        print(f"[Normalization] Generated stats file: {stats_path}")
    else:
        print(f"[Normalization] Normalize wrapper not found; cannot generate stats: {stats_path}")
    vecenv.close()


def load_policy_from_checkpoint(env_name: str, model_path: str, device: str = 'cpu',
                                env_seed: int | None = None):
    """Load a saved policy model."""
    print(f"Loading policy from {model_path}")
    
    # Load config (avoid parsing CLI args)
    import sys
    old_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]  # keep script name only
        args = pufferl.load_config(env_name)
    finally:
        sys.argv = old_argv
    
    # Device and seed
    args['train']['device'] = device
    if env_seed is not None:
        train_cfg = args.setdefault('train', {})
        vec_cfg = args.setdefault('vec', {})
        env_cfg = args.setdefault('env', {})
        train_cfg['seed'] = env_seed
        vec_cfg['seed'] = env_seed
        env_cfg.pop('seed', None)
    
    # Create env (to get obs/action spaces)
    args['vec']['backend'] = 'Serial'
    args['vec']['num_envs'] = 1
    args['vec']['num_workers'] = 1
    
    # Ensure env supports rgb_array rendering
    if 'env' not in args:
        args['env'] = {}
    args['env']['render_mode'] = 'rgb_array'
    
    vecenv = pufferl.load_env(env_name, args)
    
    # Load policy
    policy = pufferl.load_policy(args, vecenv, env_name)
    
    # Load weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        # Strip possible module prefix
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
        print("  ✓ Model weights loaded")
    else:
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    
    policy.to(device)
    policy.eval()
    
    return policy, vecenv


def create_render_env(env_name: str, render_mode: str = 'rgb_array'):
    """Create a rendering environment (without pufferlib wrappers)."""
    import gymnasium
    import os
    from gymnasium.wrappers import OrderEnforcing
    
    # Try setting a headless rendering backend
    if 'MUJOCO_GL' not in os.environ:
        # Try egl first; fall back to osmesa
        os.environ['MUJOCO_GL'] = 'egl'
    try:
        env = gymnasium.make(env_name, render_mode=render_mode)
        # If env is wrapped by OrderEnforcing, disable render order enforcing
        if isinstance(env, OrderEnforcing):
            env.disable_render_order_enforcing = True
        # Reset first, then test render
        env.reset()
        frame = env.render()
        if frame is not None:
            return env
        else:
            env.close()
            return None
    except Exception as e:
        # If egl fails, try osmesa
        if os.environ.get('MUJOCO_GL') == 'egl':
            os.environ['MUJOCO_GL'] = 'osmesa'
            try:
                env = gymnasium.make(env_name, render_mode=render_mode)
                if isinstance(env, OrderEnforcing):
                    env.disable_render_order_enforcing = True
                env.reset()
                frame = env.render()
                if frame is not None:
                    return env
                else:
                    env.close()
                    return None
            except Exception as e2:
                pass
        # If all fail, return None; recording will skip rendering
        print(f"  Warning: failed to create render env ({e}); skipping video recording")
        return None


def record_video(policy, vecenv, env_name: str, output_path: str,
                 num_episodes: int = 1, max_steps: int = 1000, fps: int = 30,
                 env_seed: int | None = None, deterministic: bool = False):
    """Record policy rollout videos."""
    
    device = next(policy.parameters()).device
    
    # Try using vecenv.driver_env for rendering
    driver_env = None
    if hasattr(vecenv, 'driver_env'):
        driver_env = vecenv.driver_env
    elif hasattr(vecenv, 'envs') and len(vecenv.envs) > 0:
        # Try the first env in envs list
        driver_env = vecenv.envs[0]
        # May need to unwrap nested wrappers
        while hasattr(driver_env, 'env') or hasattr(driver_env, 'unwrapped'):
            if hasattr(driver_env, 'unwrapped'):
                driver_env = driver_env.unwrapped
            elif hasattr(driver_env, 'env'):
                driver_env = driver_env.env
            else:
                break
    
    # Try rendering from driver_env
    can_render_from_driver = False
    if driver_env is not None:
        try:
            # Check render capability
            if hasattr(driver_env, 'render'):
                # Try reading render mode
                render_mode = getattr(driver_env, 'render_mode', None)
                if render_mode is None and hasattr(driver_env, 'unwrapped'):
                    render_mode = getattr(driver_env.unwrapped, 'render_mode', None)
                if render_mode == 'rgb_array' or render_mode is None:
                    # Test render once
                    test_frame = driver_env.render()
                    if test_frame is not None:
                        can_render_from_driver = True
        except Exception as e:
            print(f"  Warning: driver_env render test failed ({e})")
    
    # If driver_env cannot render, create a separate render env
    if not can_render_from_driver:
        print("  Using a separate render env...")
        render_env = create_render_env(env_name, render_mode='rgb_array')
        use_separate_env = (render_env is not None)
        if not use_separate_env:
            print("  ⚠ Failed to create render env; will only log reward info")
    else:
        print("  Using driver_env for rendering")
        render_env = None
        use_separate_env = False
    
    all_frames = []
    episode_rewards = []
    
    print(f"\nRecording {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        print(f"  Episode {ep + 1}/{num_episodes}")

        current_seed = env_seed
        reset_kwargs = {}
        if current_seed is not None:
            reset_kwargs["seed"] = current_seed

        # Reset env - keep both envs in sync with the same seed
        try:
            obs_vec, info_vec = vecenv.reset(**reset_kwargs)
        except TypeError:
            obs_vec, info_vec = vecenv.reset()
        if use_separate_env and render_env is not None:
            try:
                # Use the same seed to synchronize initial state
                seed = current_seed
                if seed is None and isinstance(info_vec, dict):
                    seed = info_vec.get('seed')
                if seed is None:
                    seed = ep  # use episode index as seed
                obs_render, info_render = render_env.reset(seed=seed)
                # Ensure initial states match (if possible)
                if hasattr(render_env, 'set_state') and hasattr(driver_env, 'get_state'):
                    try:
                        state = driver_env.get_state()
                        render_env.set_state(state)
                    except Exception:
                        pass
            except Exception as e:
                print(f"    Warning: render env reset failed ({e}); skipping video recording")
                render_env = None
                use_separate_env = False
        
        frames = []
        episode_reward_norm = 0.0
        episode_info_raw = None
        lstm_h, lstm_c = None, None
        
        for step in range(max_steps):
            # Render current frame (before applying the action)
            frame = None
            if not use_separate_env and driver_env is not None:
                # Render directly from driver_env (synced with vecenv)
                try:
                    frame = driver_env.render()
                except Exception as e:
                    if step == 0:
                        print(f"    Render warning: {e}")
                    frame = None
            elif use_separate_env and render_env is not None:
                # Render from the separate render env
                try:
                    frame = render_env.render()
                except Exception as e:
                    if step == 0:
                        print(f"    Render warning: {e}")
                    frame = None
            
            if frame is not None:
                frames.append(frame)
            
            # Policy forward pass
            obs_tensor = torch.as_tensor(obs_vec, device=device, dtype=torch.float32)
            
            state_dict = {'lstm_h': lstm_h, 'lstm_c': lstm_c}
            with torch.no_grad():
                action_obj, value = policy.forward_eval(obs_tensor, state_dict)
            
            # Update LSTM state
            lstm_h = state_dict.get('lstm_h', None)
            lstm_c = state_dict.get('lstm_c', None)
            
            # Sample/select action
            action = sample_action_from_dist(action_obj, deterministic=deterministic)
            
            # Convert to numpy
            if isinstance(action, torch.Tensor):
                action_np = action.cpu().numpy()
            else:
                action_np = action
            
            # Ensure action shape matches action_space
            if action_np.ndim == 0:
                action_np = np.array([action_np])
            elif action_np.ndim > 1:
                action_np = action_np.squeeze()
                if action_np.ndim == 0:
                    action_np = np.array([action_np])
            
            # Clamp action into valid range
            action_space = vecenv.action_space
            if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
                action_np = np.clip(action_np, action_space.low, action_space.high)
            
            # Step env - ensure both envs use the same action
            obs_vec, reward_vec, terminated_vec, truncated_vec, info_vec = vecenv.step(action_np)
            
            # If using a separate render env, step it too to keep in sync
            if use_separate_env and render_env is not None:
                try:
                    # Step render env with the same action
                    obs_render, reward_render, terminated_render, truncated_render, info_render = render_env.step(action_np)
                    # Use vecenv reward (normalized; log only)
                    episode_reward_norm += float(reward_vec) if isinstance(reward_vec, (int, float)) else float(reward_vec[0])
                    done = terminated_vec or truncated_vec
                except Exception as e:
                    # If render env errors, fall back to vecenv only
                    episode_reward_norm += float(reward_vec) if isinstance(reward_vec, (int, float)) else float(reward_vec[0])
                    done = terminated_vec or truncated_vec
                    if step == 0:
                        print(f"    Warning: render env step failed ({e}); using vecenv only")
            else:
                # Use vecenv directly (driver_env stays in sync)
                episode_reward_norm += float(reward_vec) if isinstance(reward_vec, (int, float)) else float(reward_vec[0])
                done = terminated_vec or truncated_vec

            info_entry = _find_episode_info(info_vec)
            if info_entry is not None:
                episode_info_raw = info_entry
            
            if done:
                # Render last frame
                if use_separate_env and render_env is not None:
                    try:
                        final_frame = render_env.render()
                        if final_frame is not None:
                            frames.append(final_frame)
                    except:
                        pass
                elif not use_separate_env and driver_env is not None:
                    try:
                        final_frame = driver_env.render()
                        if final_frame is not None:
                            frames.append(final_frame)
                    except:
                        pass
                break
        
        all_frames.extend(frames)
        final_reward = None
        if episode_info_raw and 'episode_return' in episode_info_raw:
            final_reward = float(episode_info_raw['episode_return'])
        else:
            final_reward = episode_reward_norm

        reward_note = "(raw reward)" if episode_info_raw else "(normalized reward)"
        episode_rewards.append(final_reward)
        if episode_info_raw and 'episode_length' in episode_info_raw:
            length_note = int(episode_info_raw['episode_length'])
        else:
            length_note = len(frames)
        print(f"    Episode {ep + 1} done: {len(frames)} frames, reward = {final_reward:.2f} {reward_note}")
    
    if use_separate_env and render_env is not None:
        try:
            render_env.close()
        except:
            pass
    
    # Save video
    if all_frames:
        print(f"\nSaving video to {output_path} ({len(all_frames)} frames)...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            imageio.mimsave(output_path, all_frames, fps=fps, codec='libx264', quality=8)
            print("  ✓ Video saved")
        except Exception as e:
            print(f"  ✗ Failed to save video: {e}")
        print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    else:
        print("\n  ⚠ No frames collected (rendering env unavailable)")
        print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print("  Note: policy ran fine, but video could not be recorded. Please check rendering setup.")
    
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Record policy rollout videos in Mujoco environments")
    parser.add_argument('--env', type=str, required=True,
                       choices=['Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4', 'Humanoid-v4', 'BipedalWalker-v3'],
                       help='Environment name')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Policy checkpoint path (best_model.pt)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (default: videos/{env_name}_task_{task_id}.mp4)')
    parser.add_argument('--task-id', type=int, default=None,
                       help='Task id (used to auto-generate output path)')
    parser.add_argument('--num-episodes', type=int, default=1,
                       help='Number of episodes to record')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Max steps per episode')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video FPS')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Compute device')
    parser.add_argument('--env-seed', type=int, default=None,
                       help='Environment seed (match training)')
    parser.add_argument('--vec-stats-path', type=str, default=None,
                        help='NormalizeObservation/Reward stats file path (.npz)')
    parser.add_argument('--generate-vec-stats', action='store_true',
                        help='If stats file is missing, generate it by running a warmup rollout')
    parser.add_argument('--force-generate-vec-stats', action='store_true',
                        help='Regenerate stats even if the file exists')
    parser.add_argument('--stats-warmup-steps', type=int, default=DEFAULT_STATS_WARMUP_STEPS,
                        help='Warmup steps used to generate stats (agent steps)')
    parser.add_argument('--stats-num-envs', type=int, default=DEFAULT_STATS_NUM_ENVS,
                        help='Number of Serial envs used to generate stats (capped by training parallelism)')
    parser.add_argument('--train-config', type=str, default='config/ppo_lstm_chatgpt.yml',
                        help='Training YAML config (used when generating stats)')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic actions at inference (mean/mode)')
    
    args = parser.parse_args()
    
    # Output path
    if args.output is None:
        if args.task_id is not None:
            output_path = f"videos/{args.env}_task_{args.task_id}.mp4"
        else:
            # Extract task_id from model path
            model_dir = os.path.dirname(args.model_path)
            task_id = os.path.basename(model_dir).replace('controlled_task_', '')
            output_path = f"videos/{args.env}_task_{task_id}.mp4"
    else:
        output_path = args.output

    stats_path: Optional[Path] = None
    if args.vec_stats_path:
        stats_path = Path(args.vec_stats_path)
    elif args.task_id is not None:
        stats_path = Path(f"models/controlled_task_{args.task_id}/vec_stats.npz")

    if args.generate_vec_stats and stats_path is not None:
        need_generate = args.force_generate_vec_stats or not stats_path.exists()
        if need_generate:
            if args.env_seed is None:
                raise ValueError("Generating normalization stats requires --env-seed to reproduce experiment randomness")
            print(f"\n[Preprocess] Generating normalization stats for {args.env} -> {stats_path}")
            generate_vec_stats(
                args.env,
                args.train_config,
                stats_path,
                args.model_path,
                args.stats_warmup_steps,
                train_seed=args.env_seed,
                deterministic=args.deterministic,
                device=args.device,
                stats_num_envs=args.stats_num_envs,
            )
    
    print("=" * 80)
    print("Policy video recording")
    print("=" * 80)
    print(f"Env: {args.env}")
    print(f"Model path: {args.model_path}")
    print(f"Output path: {output_path}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"FPS: {args.fps} fps")
    print("=" * 80)
    
    # Load policy
    policy, vecenv = load_policy_from_checkpoint(
        args.env, args.model_path, device=args.device, env_seed=args.env_seed
    )

    if stats_path is not None and stats_path.exists():
        if load_vec_stats(vecenv.driver_env, stats_path):
            print(f"[Normalization] Loaded stats file: {stats_path}")
        else:
            print(f"[Normalization] No usable Normalize wrapper found from {stats_path}; continuing with default stats")
    
    # Record video
    try:
        rewards = record_video(
            policy, vecenv, args.env, output_path,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            fps=args.fps,
            env_seed=args.env_seed,
            deterministic=args.deterministic,
        )
        print(f"\n✓ Done! Video saved to: {output_path}")
    finally:
        vecenv.close()


if __name__ == '__main__':
    main()

