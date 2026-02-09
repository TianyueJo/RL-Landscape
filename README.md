# landscape-v2 (control + grid_world)

This repository contains two related codebases from `landscape-v2`:

- `control/`: MuJoCo-based controlled-randomness training, evaluation, and analysis utilities (Walker2d / HalfCheetah, Jump & Retrain, similarity, video recording).
- `grid_world/`: GridWorld PPO training/evaluation, Jump & Retrain experiments, behavior embeddings (PCA/t-SNE), and visualization scripts.

Large experiment artifacts (models/results/logs/videos/embeddings outputs) are intentionally excluded from git.

## Quick start

### 1) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

Install dependencies for both subprojects:

```bash
pip install -r control/requirements.txt
pip install -r grid_world/requirements.txt
```

Notes:
- Both requirements files pin `numpy==1.26.4` to avoid binary-compat issues with compiled wheels in some environments.

## Repository layout

```text
.
├─ control/
│  ├─ requirements.txt
│  ├─ train_landscape_controlled.py
│  ├─ evaluate_walker_policies.py
│  ├─ record_policy_video.py
│  ├─ jump_retrain/
│  └─ visualization/
└─ grid_world/
   ├─ requirements.txt
   ├─ env.py
   ├─ train.py
   ├─ evaluate_models.py
   ├─ jump_and_retrain.py
   ├─ compute_behavior_embeddings.py
   ├─ plot_distance_graphs.py
   └─ env_specs/
```

## Common commands (examples)

### GridWorld

Train a policy:

```bash
cd grid_world
python3 train.py --case-id 1 --seed 0
```

Evaluate policies:

```bash
cd grid_world
python3 evaluate_models.py --models-dir models --action-mode both
```

### Control (MuJoCo)

Evaluate Walker2d policies (example):

```bash
cd control
python3 evaluate_walker_policies.py --env-name Walker2d-v4 --num-episodes 10
```

## License

Add a license file if you plan to publish this repository publicly.



