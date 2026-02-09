#!/usr/bin/env python3
"""
Compute behavior vectors for GridWorld policies and reduce with PCA/t-SNE.
Saves per-policy embeddings for each case and dimension.
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from env import make_env
from ppo_simple import PPOAgent


def collect_representative_states(
    env,
    model_files: List[Tuple[int, Path]],
    n_episodes_per_policy: int,
    max_steps_per_episode: int,
    n_representative_states: int,
    device: str = "cpu",
) -> np.ndarray:
    """
    Sample representative states by rolling out all policies in the environment,
    collecting observations, then randomly subsampling.

    Returns normalized states (consistent with the training-time ObservationWrapper).
    """
    buffer_states: List[np.ndarray] = []
    for seed, model_path in model_files:
        agent = load_agent(model_path, device=device)
        for episode in range(n_episodes_per_policy):
            obs, _ = env.reset(seed=seed * 1000 + episode)
            done = False
            steps = 0
            while not done and steps < max_steps_per_episode:
                # Record state
                buffer_states.append(obs.copy())
                # Select action according to the policy (deterministic)
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_logits, _ = agent.forward(obs_tensor)
                    action = torch.argmax(action_logits, dim=-1).item()
                obs, _, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
                steps += 1

    if not buffer_states:
        raise RuntimeError("No representative states collected.")

    # Randomly sample representative states
    if len(buffer_states) > n_representative_states:
        indices = random.sample(range(len(buffer_states)), n_representative_states)
        representative_states = np.array([buffer_states[i] for i in indices], dtype=np.float32)
    else:
        representative_states = np.array(buffer_states, dtype=np.float32)

    return representative_states


def load_agent(model_path: Path, device: str = "cpu") -> PPOAgent:
    """Load a GridWorld PPO policy."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    obs_dim = checkpoint["obs_dim"]
    action_dim = checkpoint["action_dim"]
    agent = PPOAgent(obs_dim, action_dim, hidden_dim=64, lr=3e-4).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    return agent


def compute_behavior_vector(
    agent: PPOAgent,
    representative_states: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Compute behavior vector as action probabilities on representative states"""
    obs_tensor = torch.as_tensor(representative_states, dtype=torch.float32, device=device)
    with torch.no_grad():
        action_logits, _ = agent.forward(obs_tensor)
        probs = torch.softmax(action_logits, dim=-1)
    # behavior matrix shape: (N, A) -> flatten to (A*N,)
    return probs.cpu().numpy().reshape(-1)


def collect_models(models_dir: Path, case_id: int) -> List[Tuple[int, Path]]:
    """Collect model files and return a list of (seed, path)."""
    model_files = []
    for model_path in models_dir.glob(f"gridworld_ppo_case_{case_id}_seed_*.pt"):
        parts = model_path.stem.split("_")
        seed = int(parts[5])
        model_files.append((seed, model_path))
    return sorted(model_files, key=lambda x: x[0])


def run_pca(
    behavior_matrix: np.ndarray,
    dim: int,
    random_state: int,
) -> np.ndarray:
    pca = PCA(n_components=dim, random_state=random_state)
    return pca.fit_transform(behavior_matrix)


def run_tsne(
    behavior_matrix: np.ndarray,
    dim: int,
    random_state: int,
    perplexity: float,
) -> np.ndarray:
    method = "exact" if dim > 3 else "barnes_hut"
    tsne = TSNE(
        n_components=dim,
        perplexity=perplexity,
        random_state=random_state,
        method=method,
        init="pca" if behavior_matrix.shape[0] > dim else "random",
        max_iter=1000,
        verbose=0,
    )
    return tsne.fit_transform(behavior_matrix)


def save_embeddings(
    output_dir: Path,
    method: str,
    dim: int,
    case_id: int,
    seeds: List[int],
    model_paths: List[str],
    embeddings: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{method}_dim{dim}_case{case_id}.npz"
    np.savez(
        npz_path,
        embeddings=embeddings,
        seeds=np.array(seeds, dtype=np.int32),
        model_paths=np.array(model_paths, dtype=object),
        method=method,
        dim=dim,
        case_id=case_id,
    )

    json_path = output_dir / f"{method}_dim{dim}_case{case_id}.json"
    payload = {
        "case_id": case_id,
        "method": method,
        "dim": dim,
        "seeds": [int(s) for s in seeds],
        "model_paths": model_paths,
        # K x D matrix
        "embeddings": [[float(x) for x in row] for row in embeddings],
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute behavior vectors and PCA/t-SNE embeddings for GridWorld policies"
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Models directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("behavior_space_embeddings"),
        help="Output directory.",
    )
    parser.add_argument(
        "--case-ids",
        type=int,
        nargs="+",
        default=[1, 2],
        help="List of case_ids to process (default: 1 2).",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[3, 6, 9, 12],
        help="List of embedding dimensions.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device.")
    parser.add_argument(
        "--n-episodes-per-policy",
        type=int,
        default=10,
        help="Episodes per policy used to sample representative states.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=200,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--n-representative-states",
        type=int,
        default=1000,
        help="Number of representative states.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=None,
        help="t-SNE perplexity (default: auto based on sample count).",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    models_dir = args.models_dir
    output_dir = args.output_dir
    dims = args.dims
    device = args.device

    output_dir.mkdir(parents=True, exist_ok=True)

    for case_id in args.case_ids:
        print(f"\n=== Case {case_id} ===")
        model_files = collect_models(models_dir, case_id)
        if not model_files:
            print(f"[Warning] No models found for case {case_id}")
            continue

        seeds = [seed for seed, _ in model_files]
        model_paths = [str(path) for _, path in model_files]

        env = make_env(seed=0, case_id=case_id)
        representative_states = collect_representative_states(
            env=env,
            model_files=model_files,
            n_episodes_per_policy=args.n_episodes_per_policy,
            max_steps_per_episode=args.max_steps_per_episode,
            n_representative_states=args.n_representative_states,
            device=device,
        )

        behavior_vectors = []
        for seed, model_path in model_files:
            agent = load_agent(model_path, device=device)
            vec = compute_behavior_vector(agent, representative_states, device=device)
            behavior_vectors.append(vec)

        behavior_matrix = np.vstack(behavior_vectors)
        n_samples, n_features = behavior_matrix.shape
        print(f"Behavior matrix shape: {behavior_matrix.shape}")

        max_dim = min(n_samples, n_features)
        for dim in dims:
            if dim > max_dim:
                print(f"[Skip] dim {dim} > max allowed {max_dim} (case {case_id})")
                continue

            # PCA
            pca_embeddings = run_pca(behavior_matrix, dim, args.random_state)
            save_embeddings(
                output_dir,
                method="pca",
                dim=dim,
                case_id=case_id,
                seeds=seeds,
                model_paths=model_paths,
                embeddings=pca_embeddings,
            )

            # t-SNE
            if args.tsne_perplexity is None:
                perplexity = min(30.0, max(2.0, n_samples / 2.0))
                if perplexity >= n_samples:
                    perplexity = max(2.0, n_samples - 1.0)
            else:
                perplexity = args.tsne_perplexity

            tsne_embeddings = run_tsne(
                behavior_matrix, dim, args.random_state, perplexity
            )
            save_embeddings(
                output_dir,
                method="tsne",
                dim=dim,
                case_id=case_id,
                seeds=seeds,
                model_paths=model_paths,
                embeddings=tsne_embeddings,
            )

        # Save representative states for reproducibility
        states_path = output_dir / f"representative_states_case{case_id}.npy"
        np.save(states_path, representative_states)

    print("\nDone.")


if __name__ == "__main__":
    main()

