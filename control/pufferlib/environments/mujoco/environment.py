
from pdb import set_trace as T

import functools
import json
import os
import time
from pathlib import Path

import numpy as np
import gymnasium

import pufferlib
import pufferlib.emulation
import pufferlib.environments


class EpisodeReturnRecorder(gymnasium.Wrapper):
    """Record per-environment episode returns to disk for debugging."""

    def __init__(self, env, log_dir, run_prefix, flush_interval=32):
        super().__init__(env)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_prefix = run_prefix or "episode_log"
        self.flush_interval = max(1, int(flush_interval))
        self.env_seed = None
        self.episode_idx = 0
        self._buffer = []
        self._log_path = None

    def _ensure_log_path(self):
        if self._log_path is not None:
            return
        seed_part = f"seed{self.env_seed}" if self.env_seed is not None else f"pid{os.getpid()}_{id(self)}"
        filename = f"{self.run_prefix}_{seed_part}.jsonl"
        self._log_path = self.log_dir / filename

    def _flush(self):
        if not self._buffer:
            return
        self._ensure_log_path()
        with self._log_path.open("a", encoding="utf-8") as f:
            for entry in self._buffer:
                f.write(json.dumps(entry, ensure_ascii=False))
                f.write("\n")
        self._buffer.clear()

    def reset(self, seed=None, options=None):
        if self.env_seed is None and seed is not None:
            try:
                self.env_seed = int(seed)
            except (TypeError, ValueError):
                self.env_seed = None
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if done and isinstance(info, dict):
            ep_return = info.get("episode_return")
            if ep_return is not None:
                entry = {
                    "timestamp": time.time(),
                    "env_seed": self.env_seed,
                    "episode_index": self.episode_idx,
                    "episode_return": float(ep_return),
                    "episode_length": int(info.get("episode_length", 0)),
                }
                self._buffer.append(entry)
                self.episode_idx += 1
                if len(self._buffer) >= self.flush_interval:
                    self._flush()
        return obs, reward, terminated, truncated, info

    def close(self):
        self._flush()
        return super().close()


def single_env_creator(env_name, capture_video, gamma,
        run_name=None, idx=None, obs_norm=True, pufferl=False, render_mode='rgb_array',
        buf=None, seed=0, log_episode_returns=False, episode_log_dir=None,
        episode_log_prefix=None, episode_log_flush=32):
    if capture_video and idx == 0:
        assert run_name is not None, "run_name must be specified when capturing videos"
        env = gymnasium.make(env_name, render_mode="rgb_array")
        env = gymnasium.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gymnasium.make(env_name, render_mode=render_mode)

    env = pufferlib.ClipAction(env)  # NOTE: this changed actions space
    env = pufferlib.EpisodeStats(env)
    if log_episode_returns:
        log_dir = episode_log_dir or os.path.join("episode_logs", env_name)
        prefix = episode_log_prefix or (run_name or env_name)
        env = EpisodeReturnRecorder(env, log_dir=log_dir, run_prefix=prefix, flush_interval=episode_log_flush)

    if obs_norm:
        env = gymnasium.wrappers.NormalizeObservation(env)
        env = gymnasium.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))

    env = gymnasium.wrappers.NormalizeReward(env, gamma=gamma)
    env = gymnasium.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

    if pufferl is True:
        env = pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

    return env


def cleanrl_env_creator(env_name, run_name, capture_video, gamma, idx):
    kwargs = {
        "env_name": env_name,
        "run_name": run_name,
        "capture_video": capture_video,
        "gamma": gamma,
        "idx": idx,
        "pufferl": False,
    }
    return functools.partial(single_env_creator, **kwargs)


# Keep it simple for pufferl demo, for now
def env_creator(env_name="HalfCheetah-v4", gamma=0.99):
    default_kwargs = {
        "env_name": env_name,
        "capture_video": False,
        "gamma": gamma,
        "pufferl": True,
    }
    return functools.partial(single_env_creator, **default_kwargs)
