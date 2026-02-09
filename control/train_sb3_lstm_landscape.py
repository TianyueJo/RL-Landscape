#!/usr/bin/env python3
"""
Train BipedalWalker-v3 and HalfCheetah-v4 with SB3-style PPO-LSTM using PufferLib.
Loads hyper-parameters from config/ppo_lstm.yml, runs for 10M steps
and records landscape features (sharpness, lambda_max, etc.).
"""

import argparse
import glob
import json
import os

import torch
import yaml

from train_landscape_controlled import (
    LandscapeAnalyzer,
    load_config_manually,
    train_with_landscape_loop,
)
from pufferlib import pufferl


def parse_args():
    parser = argparse.ArgumentParser(description="Train SB3-style PPO-LSTM with landscape logging.")
    parser.add_argument("--env-name", required=True, help="Gymnasium environment name.")
    parser.add_argument("--task-id", type=int, required=True, help="Unique task id (used for logging).")
    parser.add_argument("--init-seed", type=int, required=True, help="Seed for model initialization.")
    parser.add_argument("--train-seed", type=int, required=True, help="Seed for optimization/environment.")
    parser.add_argument("--config", type=str, default="config/ppo_lstm.yml",
                        help="Path to rl-zoo style ppo_lstm yaml file.")
    parser.add_argument("--total-steps", type=int, default=10_000_000,
                        help="Total training steps (default 10M).")
    parser.add_argument("--feature-stride", type=int, default=10_000,
                        help="Steps between landscape feature snapshots.")
    parser.add_argument("--rollout-steps", type=int, default=2048,
                        help="Rollout length for feature computation.")
    parser.add_argument("--resume-model-path", type=str, default=None,
                        help="Path to an existing checkpoint to resume from.")
    return parser.parse_args()


def _to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value.startswith("lin_"):
            value = value.split("_", 1)[1]
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_sb3_config(yaml_path, env_name):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if env_name not in data:
        raise ValueError(f"Environment '{env_name}' not found in {yaml_path}")
    cfg = data[env_name]
    return cfg


def apply_overrides(base_args, env_cfg, total_steps):
    vec = base_args.setdefault("vec", {})
    train = base_args.setdefault("train", {})

    vec["seed"] = train.get("seed", 0)

    train["total_timesteps"] = total_steps

    scalar_overrides = {
        "learning_rate": "learning_rate",
        "gamma": "gamma",
        "gae_lambda": "gae_lambda",
        "ent_coef": "ent_coef",
        "clip_range": "clip_coef",
        "vf_coef": "vf_coef",
        "max_grad_norm": "max_grad_norm",
    }
    for src_key, dst_key in scalar_overrides.items():
        if src_key in env_cfg and env_cfg[src_key] is not None:
            train[dst_key] = _to_float(env_cfg[src_key])

    if "n_epochs" in env_cfg:
        train["update_epochs"] = int(env_cfg["n_epochs"])

    train["optimizer"] = "adam"
    train["data_dir"] = os.path.join("results", f"sb3_lstm_{env_cfg.get('policy', 'MlpLstmPolicy')}")


def find_feature_offsets(task_id: int):
    pattern = os.path.join("landscape_data", f"features_task_{task_id}_step_*.json")
    files = glob.glob(pattern)
    if not files:
        return 0, 0

    candidates = []
    for path in files:
        base = os.path.basename(path)
        try:
            suffix = base.split("_step_")[1]
            step_value = int(suffix.split(".")[0])
            candidates.append((step_value, path))
        except (IndexError, ValueError):
            continue

    if not candidates:
        return 0, 0

    last_step, last_path = max(candidates, key=lambda item: item[0])
    num_step_offset = 0
    try:
        with open(last_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            num_step_offset = int(data.get("num_steps", 0))
    except Exception:
        num_step_offset = 0

    return last_step + 1, num_step_offset


def resolve_resume_path(task_id: int):
    base = os.path.join("models", f"controlled_task_{task_id}")
    candidates = [
        os.path.join(base, "final_model.pt"),
        os.path.join(base, "best_model.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main():
    cli = parse_args()
    env_cfg = load_sb3_config(cli.config, cli.env_name)

    feature_step_start, num_step_offset = find_feature_offsets(cli.task_id)
    analyzer = LandscapeAnalyzer(start_step=feature_step_start)
    if feature_step_start > 0 or num_step_offset > 0:
        print(
            f"[Resume] Feature start: step_count={feature_step_start}, num_steps offset={num_step_offset}"
        )
    args = load_config_manually(cli.env_name)
    apply_overrides(args, env_cfg, cli.total_steps)

    args["train"]["seed"] = cli.train_seed
    args["vec"]["seed"] = cli.train_seed
    args["train"]["data_dir"] = f"results/sb3_{cli.env_name.replace('-', '_')}_{cli.task_id}"
    args["train"]["use_rnn"] = True
    args["train"]["env"] = cli.env_name
    args["train"]["total_timesteps"] = cli.total_steps

    os.makedirs("landscape_data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    vecenv = pufferl.load_env(cli.env_name, args)
    policy = pufferl.load_policy(args, vecenv, cli.env_name)

    resume_path = cli.resume_model_path
    if resume_path is not None and not os.path.exists(resume_path):
        raise FileNotFoundError(f"Specified resume model does not exist: {resume_path}")
    if resume_path is None:
        resume_path = resolve_resume_path(cli.task_id)

    if resume_path:
        device = torch.device(args["train"].get("device", "cpu"))
        print(f"[Resume] Loading model from {resume_path} (device={device})")
        state_dict = torch.load(resume_path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
    else:
        print("[Resume] No trained model found; starting from random initialization")

    os.environ["FEATURE_STRIDE"] = str(cli.feature_stride)
    train_with_landscape_loop(
        cli.env_name,
        args,
        policy,
        vecenv,
        analyzer,
        cli.task_id,
        cli.train_seed,
        cli.init_seed,
        num_step_offset=num_step_offset,
        models_root="models",
    )


if __name__ == "__main__":
    main()

