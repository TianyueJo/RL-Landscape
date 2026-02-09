#!/usr/bin/env python3
"""
Landscape feature training with controlled randomness.
Separates the effects of model initialization seed vs. training seed.
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
import configparser
import glob

import numpy as np
import torch
import gymnasium
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
import pufferlib.vector
import pufferlib.ocean
from pufferlib import pufferl


def iter_wrapped_envs(env):
    """Yield env and all nested .env wrappers to help locate Normalize wrappers."""
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        yield current
        visited.add(id(current))
        current = getattr(current, "env", None)


def get_normalize_wrappers(puffer_env):
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


def load_vec_stats(puffer_env, stats_path: Path) -> bool:
    """Load saved NormalizeObservation/Reward statistics into environment wrappers."""
    if stats_path is None or not Path(stats_path).exists():
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
            elif hasattr(dist_obj, "loc"):
                action = dist_obj.loc
            else:
                action = dist_obj.sample()
        else:
            action = dist_obj.sample()
    else:
        action = dist_obj
    return action

# Reuse LandscapeAnalyzer
class LandscapeAnalyzer:
    def _selected_log_prob(self, policy, obs, act):
        # Initialize LSTM state (if using RNN)
        state = {
            'lstm_h': None,
            'lstm_c': None,
        }
        out = policy.forward_eval(obs, state)
        # Unify unpacking: tuple/list -> first item; dict -> prefer distribution / logits / mean
        if isinstance(out, (tuple, list)):
            logits = out[0]
        elif isinstance(out, dict):
            if 'distribution' in out:
                logits = out['distribution']
            elif 'logits' in out:
                logits = out['logits']
            elif 'mean' in out:
                logits = out['mean']
            else:
                logits = out
        else:
            logits = out
        # Distribution path (e.g., DiagNormal)
        if hasattr(logits, 'log_prob'):
            # If action is incorrectly flattened, reshape to distribution parameter shape
            try:
                loc = getattr(logits, 'loc', None)
                if loc is not None and isinstance(loc, torch.Tensor) and act.dim() == 1 and loc.dim() >= 2:
                    batch, ad = loc.shape[0], loc.shape[-1]
                    if act.numel() == batch * ad:
                        act = act.view(batch, ad)
            except Exception:
                pass
            logp = logits.log_prob(act.float())
            # Sum for multi-dimensional actions
            if logp.dim() > 1:
                logp = logp.sum(dim=-1)
            return logp
        # Tensor path
        if isinstance(logits, torch.Tensor) and logits.dim() == 2 and act.dtype in (torch.long, torch.int64):
            log_probs_all = torch.log_softmax(logits, dim=-1)
            return log_probs_all.gather(1, act.view(-1, 1)).squeeze(1)
        # Continuous: assume mean + log_std (if available)
        mu = logits
        log_std = getattr(policy, 'log_std', None)
        if log_std is None and isinstance(out, (tuple, list)) and len(out) > 1 and isinstance(mu, torch.Tensor) and hasattr(out[1], 'shape') and out[1].shape[-1] == mu.shape[-1]:
            log_std = out[1]
        if log_std is None:
            # If mu is not a tensor (edge case), fall back to zeros with act shape
            if isinstance(mu, torch.Tensor):
                log_std = torch.zeros_like(mu)
            else:
                log_std = torch.zeros_like(act, dtype=torch.float32, device=act.device)
        std = torch.exp(log_std)
        var = std * std
        # Ensure action shape matches mean
        if isinstance(mu, torch.Tensor) and act.dim() == 1 and mu.dim() >= 2 and act.numel() == mu.shape[0] * mu.shape[-1]:
            act = act.view(mu.shape[0], mu.shape[-1])
        mu_t = mu if isinstance(mu, torch.Tensor) else act
        logp = -0.5 * (((act.float() - mu_t) ** 2) / (var + 1e-8) + 2 * log_std + torch.log(torch.tensor(2 * np.pi, device=act.device)))
        return logp.sum(dim=-1)
    """Landscape feature analyzer - computes features similar to train_cartpole_landscape.py."""
    
    def __init__(self, start_step=0):
        self.gradient_history = []
        self.parameter_history = []
        self.loss_history = []
        self.step_count = start_step
        
    def compute_sharpness_gradient(self, policy, obs, act, adv):
        """Compute Sharpness/Flatness metric (gradient-norm version)."""
        device = next(policy.parameters()).device
        obs = obs.to(device)
        act = act.to(device)
        adv = adv.to(device)
        for p in policy.parameters():
            if p.grad is not None:
                p.grad.zero_()
        selected_logp = self._selected_log_prob(policy, obs, act)
        policy_loss = -(selected_logp * adv).mean()
        policy_loss.backward(retain_graph=True)
        
        # Gradient norm
        total_grad_norm = 0.0
        for param in policy.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        # Sharpness metric
        sharpness = total_grad_norm / (1 + abs(policy_loss.item()))
        
        return sharpness
    
    def compute_gradient_smoothness(self, policy):
        """Compute gradient path smoothness."""
        if len(self.parameter_history) < 2:
            return 0.0
        
        current_params = []
        for param in policy.parameters():
            current_params.append(param.data.flatten())
        current_param = torch.cat(current_params)
        
        prev_param = self.parameter_history[-1]
        prev_prev_param = self.parameter_history[-2]
        
        param_change_1 = torch.norm(current_param - prev_param).item()
        param_change_2 = torch.norm(prev_param - prev_prev_param).item()
        
        if param_change_2 == 0:
            return 0.0
        
        ratio = param_change_1 / param_change_2
        smoothness = np.tanh(abs(ratio - 1.0))
        return smoothness
    
    def update_history(self, policy, loss):
        """Update history buffers."""
        current_grads = []
        for param in policy.parameters():
            if param.grad is not None:
                current_grads.append(param.grad.data.flatten().clone())
        
        if current_grads:
            self.gradient_history.append(torch.cat(current_grads))
            if len(self.gradient_history) > 10:
                self.gradient_history.pop(0)
        
        current_params = []
        for param in policy.parameters():
            current_params.append(param.data.flatten().clone())
        
        self.parameter_history.append(torch.cat(current_params))
        if len(self.parameter_history) > 10:
            self.parameter_history.pop(0)
        
        self.loss_history.append(loss)
        if len(self.loss_history) > 1000:
            self.loss_history.pop(0)
    
    def hessian_vector_product(self, loss, params, v):
        """Compute Hessian-vector product."""
        g1 = torch.autograd.grad(
            loss, params, create_graph=True, retain_graph=True, allow_unused=True
        )
        g1 = [torch.zeros_like(p, device=p.device) if gi is None else gi for gi, p in zip(g1, params)]
        g1_flat = torch.cat([gi.reshape(-1) for gi in g1])

        if v.device != g1_flat.device:
            v = v.to(g1_flat.device)
        if v.dtype != g1_flat.dtype:
            v = v.to(dtype=g1_flat.dtype)

        gv = torch.dot(g1_flat, v)
        g2 = torch.autograd.grad(gv, params, retain_graph=True, allow_unused=True)
        g2 = [torch.zeros_like(p, device=p.device) if gi is None else gi for gi, p in zip(g2, params)]
        Hv_flat = torch.cat([gi.reshape(-1) for gi in g2])
        return Hv_flat
    
    def power_iteration_lambda_max(self, loss, params, n_iter=10):
        """Power iteration to estimate the largest eigenvalue."""
        try:
            ref = next(p for p in params if p.requires_grad)
            device, dtype = ref.device, ref.dtype
            total = sum(p.numel() for p in params)
            v = torch.randn(total, device=device, dtype=dtype)
            v_norm = torch.norm(v)
            if v_norm == 0 or torch.isnan(v_norm):
                v = torch.randn_like(v)
            else:
                v = v / (v_norm + 1e-12)

            for _ in range(n_iter):
                Hv = self.hessian_vector_product(loss, params, v)
                hv_norm = torch.norm(Hv)
                if hv_norm < 1e-20 or torch.isnan(hv_norm):
                    Hv = torch.randn_like(v)
                    hv_norm = torch.norm(Hv)
                v = Hv / (hv_norm + 1e-12)

            Hv = self.hessian_vector_product(loss, params, v)
            lam = torch.dot(v, Hv).item()
            return float(lam) if np.isfinite(lam) else 0.0
        except Exception:
            return 0.0
    
    def hutchinson_trace(self, loss, params, m=10):
        """Hutchinson estimator for Hessian trace."""
        try:
            ref = next(p for p in params if p.requires_grad)
            device, dtype = ref.device, ref.dtype
            total = sum(p.numel() for p in params)
            acc = 0.0
            for _ in range(m):
                z = torch.randn(total, device=device, dtype=dtype)
                Hz = self.hessian_vector_product(loss, params, z)
                acc += torch.dot(z, Hz).item()
            est = acc / float(m)
            return float(est) if np.isfinite(est) else 0.0
        except Exception:
            return 0.0

    def compute_fim_trace(self, policy, obs, act, max_samples=256):
        """Estimate the trace of the Fisher Information Matrix (FIM)."""
        try:
            device = next(policy.parameters()).device
            obs = obs.to(device)
            act = act.to(device)
            n = obs.shape[0]
            if n == 0:
                return 0.0
            if n > max_samples:
                idx = torch.randperm(n, device=device)[:max_samples]
                obs = obs.index_select(0, idx)
                act = act.index_select(0, idx)

            fim_acc = 0.0
            params = [p for p in policy.parameters() if p.requires_grad]
            for i in range(obs.shape[0]):
                logp_vec = self._selected_log_prob(policy, obs[i:i+1], act[i:i+1])
                logp = logp_vec.view(-1).sum()
                grads = torch.autograd.grad(
                    logp, params, retain_graph=False, create_graph=False, allow_unused=True
                )
                flat = []
                for g, p in zip(grads, params):
                    if g is None:
                        flat.append(torch.zeros_like(p).reshape(-1))
                    else:
                        flat.append(g.reshape(-1))
                gflat = torch.cat(flat)
                fim_acc += float(torch.dot(gflat, gflat).item())

            return fim_acc / float(obs.shape[0])
        except Exception:
            return 0.0
    
    def get_landscape_features(self, policy, obs, act, adv):
        """Get all landscape features."""
        features = {}
        device = next(policy.parameters()).device
        obs = obs.to(device)
        act = act.to(device)
        adv = adv.to(device)
        
        selected_logp = self._selected_log_prob(policy, obs, act)
        policy_loss = -(selected_logp * adv).mean()
        
        features['sharpness'] = self.compute_sharpness_gradient(policy, obs, act, adv)
        features['gradient_smoothness'] = self.compute_gradient_smoothness(policy)
        
        if self.loss_history:
            recent_losses = self.loss_history[-100:]
            features['loss_mean'] = np.mean(recent_losses)
            features['loss_std'] = np.std(recent_losses)
            
            if len(recent_losses) > 1:
                try:
                    x = np.arange(len(recent_losses))
                    y = np.array(recent_losses) + np.random.normal(0, 1e-8, len(recent_losses))
                    features['loss_trend'] = np.polyfit(x, y, 1)[0]
                except:
                    features['loss_trend'] = 0.0
            else:
                features['loss_trend'] = 0.0
        
        total_param_norm = 0.0
        for param in policy.parameters():
            total_param_norm += param.data.norm(2).item() ** 2
        features['parameter_norm'] = total_param_norm ** 0.5
        
        try:
            was_training = policy.training
            policy.eval()
            selected_logp_2 = self._selected_log_prob(policy, obs, act)
            loss_for_hessian = -(selected_logp_2 * adv).mean()

            all_params = [p for p in policy.parameters() if p.requires_grad]
            features['lambda_max_10'] = self.power_iteration_lambda_max(loss_for_hessian, all_params, n_iter=10)
            features['hessian_trace_10'] = self.hutchinson_trace(loss_for_hessian, all_params, m=10)
        except Exception:
            features['lambda_max_10'] = 0.0
            features['hessian_trace_10'] = 0.0
        finally:
            policy.train(was_training)
        
        features['fim_trace'] = self.compute_fim_trace(policy, obs, act, max_samples=256)
        features['step_count'] = self.step_count
        features['timestamp'] = time.time()
        self.step_count += 1
        
        return features
    
    def save_features(self, features, task_id):
        """Save landscape features."""
        os.makedirs('landscape_data', exist_ok=True)
        
        filename = f'landscape_data/features_task_{task_id}_step_{features["step_count"]}.json'
        with open(filename, 'w') as f:
            json.dump(features, f, indent=2)
        
        with open('landscape_data/all_features.jsonl', 'a') as f:
            f.write(json.dumps(features) + '\n')

def _extract_episode_reward_from_logs(logs):
    """Try parsing episode return (raw reward) from Puffer logs."""
    if not logs:
        return None

    candidate_keys = [
        'environment/episode_return',
        'environment/episode_return_raw',
        'environment/episode/r',
        'environment/episode_reward',
        'environment/reward',
        'environment/score',
        'environment/perf',
        'episode_return',
        'episode_return_raw',
        'episode/r',
        'score',
        'reward',
        'perf',
    ]
    for key in candidate_keys:
        if key not in logs:
            continue
        value = logs[key]
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            value = value[-1]
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def extract_episode_return_from_info(info_entry, fallback_sum: float) -> float:
    """
    Extract episode return from info, matching EpisodeStats behavior as closely as possible.
    """
    if isinstance(info_entry, dict):
        ep = info_entry.get("episode", None)
        if isinstance(ep, dict):
            if "r" in ep:
                try:
                    return float(ep["r"])
                except (TypeError, ValueError):
                    pass
            for key in ("episode_return_raw", "episode_return", "reward"):
                if key in ep:
                    try:
                        return float(ep[key])
                    except (TypeError, ValueError):
                        pass

        for key in ("episode_return_raw", "episode_return", "raw_episode_return"):
            if key in info_entry:
                try:
                    return float(info_entry[key])
                except (TypeError, ValueError):
                    pass

    return float(fallback_sum)


def _build_single_env_args(base_args: dict) -> dict:
    eval_args = copy.deepcopy(base_args)
    train_cfg = eval_args.setdefault('train', {})
    vec_cfg = eval_args.setdefault('vec', {})
    train_cfg['device'] = base_args['train']['device']
    vec_cfg['backend'] = 'Serial'
    vec_cfg['num_envs'] = 1
    vec_cfg['num_workers'] = 1
    vec_cfg['batch_size'] = 1
    vec_cfg['seed'] = base_args['train']['seed']
    vec_cfg.pop('overwork', None)
    return eval_args


def _maybe_load_eval_stats(eval_vecenv, task_id: int) -> None:
    if eval_vecenv is None:
        return
    task_dir = Path("models") / f"controlled_task_{task_id}"
    candidates = [
        task_dir / "vec_stats_final.npz",
        task_dir / "vec_stats.npz",
    ]
    for path in candidates:
        if path.exists():
            if load_vec_stats(eval_vecenv.driver_env, path):
                print(f"[Eval] Loaded vec stats: {path}")
            else:
                print(f"[Eval] Failed to load vec stats: {path}")
            break


def rollout_eval_episode(policy, vecenv, device: str, seed: int) -> float:
    hidden_size = getattr(policy, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Policy is missing hidden_size attribute")

    was_training = policy.training
    policy.eval()

    vecenv.async_reset(seed)
    obs, _, _, _, infos, agent_ids, _ = vecenv.recv()
    num_agents = len(agent_ids)
    lstm_h = torch.zeros(num_agents, hidden_size, device=device)
    lstm_c = torch.zeros_like(lstm_h)
    total_rewards = np.zeros(num_agents, dtype=np.float32)
    final_infos = infos

    while True:
        obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)
        state = {"lstm_h": lstm_h, "lstm_c": lstm_c}
        with torch.no_grad():
            action_obj, _ = policy.forward_eval(obs_tensor, state)
            actions = sample_action_from_dist(action_obj, deterministic=True)
        if isinstance(actions, torch.Tensor):
            action_np = actions.detach().cpu().numpy()
        else:
            action_np = np.asarray(actions)
        vecenv.send(action_np)
        obs, rewards, terms, truncs, infos, agent_ids, masks = vecenv.recv()
        lstm_h = state["lstm_h"]
        lstm_c = state["lstm_c"]
        total_rewards += rewards.astype(np.float32)
        final_infos = infos
        done_flags = np.logical_or(terms, truncs)
        if np.any(done_flags):
            break

    episode_returns = np.zeros(num_agents, dtype=np.float32)
    for idx, info_entry in enumerate(final_infos):
        episode_returns[idx] = extract_episode_return_from_info(
            info_entry, fallback_sum=total_rewards[idx]
        )

    policy.train(was_training)
    return float(np.mean(episode_returns))


def load_config_manually(env_name):
    """Load config files manually."""
    puffer_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(puffer_dir, 'pufferlib/config')
    
    config_files = glob.glob(os.path.join(config_dir, '**/*.ini'), recursive=True)
    default_config = os.path.join(config_dir, 'default.ini')
    config = configparser.ConfigParser()
    
    if os.path.exists(default_config):
        config.read(default_config)
    
    for config_file in config_files:
        temp_config = configparser.ConfigParser()
        temp_config.read([default_config, config_file])
        
        if 'base' in temp_config and 'env_name' in temp_config['base']:
            if env_name in temp_config['base']['env_name'].split():
                config = temp_config
                break
    
    def convert_value(value):
        value = value.replace('_', '')
        try:
            if '.' not in value and 'e' not in value.lower():
                return int(value)
            return float(value)
        except ValueError:
            return value
    
    args_dict = {}
    for section in config.sections():
        section_dict = {}
        for key, value in config[section].items():
            section_dict[key] = convert_value(value)
        args_dict[section] = section_dict
    
    # Defaults
    for key in ['train', 'env', 'vec', 'base']:
        if key not in args_dict:
            args_dict[key] = {}
    
    if 'package' not in args_dict['base']:
        args_dict['base']['package'] = 'ocean'
    if 'env_name' not in args_dict['base']:
        args_dict['base']['env_name'] = env_name
    if 'policy_name' not in args_dict['base']:
        args_dict['base']['policy_name'] = 'Policy'
    if 'rnn_name' not in args_dict['base']:
        args_dict['base']['rnn_name'] = None
    
    args_dict['load_id'] = None
    args_dict['neptune'] = False
    args_dict['wandb'] = False
    args_dict['load_model_path'] = None
    
    if 'torch_deterministic' not in args_dict['train']:
        args_dict['train']['torch_deterministic'] = True
    elif isinstance(args_dict['train']['torch_deterministic'], str):
        args_dict['train']['torch_deterministic'] = args_dict['train']['torch_deterministic'].lower() == 'true'
    
    if 'device' not in args_dict['train']:
        args_dict['train']['device'] = 'cuda'
    if 'cpu_offload' not in args_dict['train']:
        args_dict['train']['cpu_offload'] = False
    elif isinstance(args_dict['train']['cpu_offload'], str):
        args_dict['train']['cpu_offload'] = args_dict['train']['cpu_offload'].lower() == 'true'
    
    args_dict['train']['compile'] = False
    args_dict['train']['compile_mode'] = 'default'
    args_dict['train']['compile_fullgraph'] = False
    
    if 'rnn' not in args_dict:
        args_dict['rnn'] = {}
    if 'hidden_size' not in args_dict['rnn']:
        args_dict['rnn']['hidden_size'] = 128
    
    args_dict['package'] = args_dict['base']['package']
    args_dict['env_name'] = args_dict['base']['env_name']
    args_dict['policy_name'] = args_dict['base']['policy_name']
    args_dict['rnn_name'] = args_dict['base'].get('rnn_name', None)
    args_dict['train']['use_rnn'] = args_dict['rnn_name'] is not None
    
    return args_dict

def train_with_controlled_seeds(env_name: str, task_id: int, init_seed: int, train_seed: int, models_root: str = "models"):
    """Train using separated init seed and train seed."""
    print("=" * 80)
    print(f"Task {task_id}: controlled randomness experiment")
    print(f"  Env: {env_name}")
    print(f"  Init seed (model params): {init_seed}")
    print(f"  Train seed (optimizer/env): {train_seed}")
    print("=" * 80)
    
    analyzer = LandscapeAnalyzer()
    
    # Stage 1: initialize model with init_seed
    print(f"\n[Stage 1] Initializing model with seed {init_seed}...")
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    
    args = load_config_manually(env_name)
    args["train"]["data_dir"] = f"results/controlled_task_{task_id}"
    
    env_steps = os.environ.get('TRAIN_TOTAL_STEPS')
    if env_steps is not None:
        try:
            args['train']['total_timesteps'] = int(env_steps)
        except:
            args['train']['total_timesteps'] = 5_000_000
    else:
        args['train']['total_timesteps'] = int(args['train'].get('total_timesteps', 5_000_000))
    
    print("Creating env and policy...")
    vecenv = pufferl.load_env(env_name, args)
    policy = pufferl.load_policy(args, vecenv, env_name)
    
    print(f"  ✓ Num policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Stage 2: train with train_seed
    print(f"\n[Stage 2] Training with seed {train_seed}...")
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    
    # Reset RNG state for optimizer/env
    args['train']['seed'] = train_seed
    args['vec']['seed'] = train_seed
    
    # Training loop
    train_with_landscape_loop(env_name, args, policy, vecenv, analyzer, task_id, train_seed, init_seed, models_root=models_root)
    
    print(f"\nTask {task_id} training completed!")

def train_with_landscape_loop(env_name, args, policy, vecenv, analyzer, task_id, train_seed, init_seed, num_step_offset=0, models_root="models"):
    """Custom training loop that tracks best model and computes landscape features."""
    from pufferlib.pufferl import PuffeRL
    
    args['train']['env'] = env_name
    trainer = PuffeRL(args['train'], vecenv, policy, logger=None)
    training_completed = False
    skip_all = os.environ.get("PUFFER_SKIP_SAVE", "0") == "1"
    skip_models = skip_all or os.environ.get("PUFFER_SKIP_MODELS", "0") == "1"
    skip_features = skip_all or os.environ.get("PUFFER_SKIP_FEATURES", "0") == "1"
    
    total_timesteps = args['train']['total_timesteps']
    feature_stride = int(os.environ.get('FEATURE_STRIDE', 10_000))
    
    print(f"Starting training, total steps: {total_timesteps:,}")
    print(f"Feature stride: {feature_stride:,} steps")
    
    last_feature_step = 0
    last_logs = None
    last_episode_reward = None
    
    # Best model tracking
    best_reward = -np.inf
    best_step = 0

    eval_vecenv = None
    eval_seed_counter = train_seed
    try:
        eval_args = _build_single_env_args(args)
        eval_vecenv = pufferl.load_env(env_name, eval_args)
        _maybe_load_eval_stats(eval_vecenv, task_id)
    except Exception as eval_err:
        print(f"[Eval] Failed to build single-env eval: {eval_err}")
        eval_vecenv = None
    
    def _collect_rollout_samples(sample_steps=2048, seed_local=None):
        args_samp = dict(args)
        args_samp = {k: (v.copy() if isinstance(v, dict) else v) for k, v in args_samp.items()}
        args_samp['vec']['backend'] = 'Serial'
        args_samp['vec']['num_envs'] = 1
        args_samp['vec']['num_workers'] = 1
        args_samp['vec']['batch_size'] = 1
        for k in ['zero_copy']:
            if k in args_samp['vec']:
                args_samp['vec'].pop(k, None)
        ve = pufferl.load_env(env_name, args_samp)
        device = next(policy.parameters()).device
        
        import pufferlib
        from pufferlib import vector as pvec
        o, _ = pvec.reset(ve, seed_local if seed_local is not None else train_seed)
        obs_list, act_list = [], []
        is_continuous = None
        total = 0
        lstm_h, lstm_c = None, None
        
        with torch.no_grad():
            while total < sample_steps:
                st = torch.as_tensor(o, device=device, dtype=torch.float32)
                state = {
                    'lstm_h': lstm_h,
                    'lstm_c': lstm_c,
                }
                out = policy.forward_eval(st, state)
                lstm_h = state.get('lstm_h', None)
                lstm_c = state.get('lstm_c', None)
                
                if isinstance(out, (tuple, list)):
                    logits = out[0]
                elif isinstance(out, dict):
                    logits = out.get('distribution', out.get('logits', out.get('mean', out)))
                else:
                    logits = out
                    
                if hasattr(logits, 'sample') and hasattr(logits, 'log_prob'):
                    action = logits.sample()
                    is_continuous = True if is_continuous is None else is_continuous
                else:
                    log_std = getattr(policy, 'log_std', None)
                    if log_std is None and isinstance(out, (tuple, list)) and len(out) > 1:
                        log_std = out[1]
                    if log_std is not None and isinstance(logits, torch.Tensor):
                        std = torch.exp(log_std.to(logits.device, logits.dtype))
                        dist = torch.distributions.Normal(logits, std)
                        action = dist.sample()
                        is_continuous = True if is_continuous is None else is_continuous
                    else:
                        action, _, _ = pufferlib.pytorch.sample_logits(logits)
                        if is_continuous is None:
                            is_continuous = False
                            
                ve.send(action.detach().cpu().numpy())
                o, _, _, _, _, _, m = ve.recv()
                obs_list.append(st)
                
                if is_continuous:
                    act_list.append(action.detach().to(dtype=torch.float32).view(1, -1))
                else:
                    act_list.append(action.detach().to(dtype=torch.long).view(-1))
                total += int(m.sum())
                
        ve.close()
        obs = torch.vstack(obs_list)
        if is_continuous:
            act = torch.vstack(act_list)
        else:
            act = torch.cat(act_list)
            
        with torch.no_grad():
            state_val = {'lstm_h': None, 'lstm_c': None}
            _, values = policy.forward_eval(obs, state_val)
        adv = (values.squeeze(-1) - values.mean()) / (values.std() + 1e-8)
        return obs, act, adv

    models_root = os.path.abspath(models_root)

    try:
        while trainer.global_step < total_timesteps:
            if args['train']['device'] == 'cuda':
                torch.compiler.cudagraph_mark_step_begin()

            loop_t0 = time.time()
            trainer.evaluate()
            logs = trainer.train()
            train_iter_time = time.time() - loop_t0
            
            if logs is not None:
                last_logs = logs
                current_reward = _extract_episode_reward_from_logs(logs)
                if current_reward is not None:
                    last_episode_reward = current_reward

                if eval_vecenv is not None:
                    try:
                        eval_seed_counter += 1
                        eval_return = rollout_eval_episode(
                            policy,
                            eval_vecenv,
                            device=args['train']['device'],
                            seed=eval_seed_counter,
                        )
                        last_episode_reward = eval_return
                    except Exception as eval_err:
                        print(f"[Eval] Single-env evaluation failed: {eval_err}")

                if last_episode_reward is not None:
                    best_str = (
                        f"{best_reward:.2f}@{best_step}"
                        if best_reward > -np.inf else "N/A"
                    )
                    print(
                        f"[Train] step {trainer.global_step:,}: "
                        f"episode_return={last_episode_reward:.2f} "
                        f"(best={best_str})"
                    )
                else:
                    print(
                        f"[Train] step {trainer.global_step:,}: "
                        f"episode_return=waiting for EpisodeStats output..."
                    )

                if current_reward is not None and current_reward > best_reward:
                    best_reward = current_reward
                    best_step = trainer.global_step
                    
                    if not skip_models:
                        model_dir = os.path.join(models_root, f'controlled_task_{task_id}')
                        os.makedirs(model_dir, exist_ok=True)
                        torch.save(policy.state_dict(),
                                   os.path.join(model_dir, 'best_model.pt'))
                        
                        best_info = {
                            'best_reward': best_reward,
                            'best_step': best_step,
                            'init_seed': init_seed,
                            'train_seed': train_seed,
                        }
                        with open(os.path.join(model_dir, 'best_info.json'), 'w') as f:
                            json.dump(best_info, f, indent=2)

            # Compute landscape features
            if trainer.global_step - last_feature_step >= feature_stride:
                try:
                    feat_t0 = time.time()
                    obs, act, adv = _collect_rollout_samples(sample_steps=2048, seed_local=train_seed)
                    landscape_features = analyzer.get_landscape_features(policy, obs, act, adv)
                    feat_time = time.time() - feat_t0
                    
                    raw_num_steps = None
                    if last_episode_reward is not None:
                        landscape_features['episode_reward'] = float(last_episode_reward)
                    elif last_logs and isinstance(last_logs, dict):
                        reward_from_last_logs = _extract_episode_reward_from_logs(last_logs)
                        if reward_from_last_logs is not None:
                            landscape_features['episode_reward'] = reward_from_last_logs
                            last_episode_reward = reward_from_last_logs

                    if last_logs and isinstance(last_logs, dict):
                        if 'agent_steps' in last_logs:
                            raw_num_steps = int(last_logs['agent_steps'])
                            
                    if raw_num_steps is None:
                        raw_num_steps = int(trainer.global_step)

                    landscape_features['num_steps'] = int(num_step_offset + raw_num_steps)
                        
                    landscape_features['feature_compute_time_s'] = float(feat_time)
                    landscape_features['train_iter_time_s'] = float(train_iter_time)
                    landscape_features['init_seed'] = init_seed
                    landscape_features['train_seed'] = train_seed
                    landscape_features['best_reward_so_far'] = best_reward
                    
                    policy_loss = landscape_features.get('loss_mean', 0.0)
                    analyzer.update_history(policy, policy_loss)
                    if not skip_features:
                        analyzer.save_features(landscape_features, task_id)
                    
                    print(f"Step {trainer.global_step:,}: Reward={landscape_features.get('episode_reward', 0):.2f}, "
                          f"Best={best_reward:.2f}@{best_step}, "
                          f"Sharpness={landscape_features['sharpness']:.4f}, "
                          f"λ_max={landscape_features['lambda_max_10']:.4f}")
                    last_feature_step = trainer.global_step
                    
                except Exception as e:
                    print(f"Landscape feature computation failed: {e}")
        
        print(f"\nTraining finished: {trainer.global_step:,} steps")
        training_completed = True
        
        if not skip_models:
            model_dir = os.path.join(models_root, f'controlled_task_{task_id}')
            os.makedirs(model_dir, exist_ok=True)
            torch.save(policy.state_dict(), os.path.join(model_dir, 'final_model.pt'))
            
            summary = {
                'task_id': task_id,
                'env_name': env_name,
                'init_seed': init_seed,
                'train_seed': train_seed,
                'best_reward': best_reward,
                'best_step': best_step,
                'final_step': trainer.global_step,
                'training_completed': True
            }
            
            with open(os.path.join(model_dir, 'training_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
    finally:
        # Stop utilization monitor thread to prevent hanging
        try:
            if hasattr(trainer, 'utilization') and trainer.utilization is not None:
                trainer.utilization.stop()
                trainer.utilization.join(timeout=5)
        except Exception as e:
            print(f"Failed to stop Utilization thread: {e}")

        # Close envs (important to avoid hanging processes)
        try:
            vecenv.close()
            print("Environment closed")
        except Exception as e:
            print(f"Error while closing env: {e}")

        if eval_vecenv is not None:
            try:
                eval_vecenv.close()
            except Exception as e:
                print(f"Error while closing eval env: {e}")

        if not training_completed:
            print("Training loop did not finish cleanly; cleanup executed.")

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: python train_landscape_controlled.py <env_name> <task_id> <init_seed> <train_seed>")
        sys.exit(1)
    
    env_name = sys.argv[1]
    task_id = int(sys.argv[2])
    init_seed = int(sys.argv[3])
    train_seed = int(sys.argv[4])
    
    print(f"Args: env_name={env_name}, task_id={task_id}")
    print(f"      init_seed={init_seed}, train_seed={train_seed}")
    
    train_with_controlled_seeds(env_name, task_id, init_seed, train_seed)

