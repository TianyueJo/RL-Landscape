#!/bin/bash
#SBATCH --job-name=behavior_tsne
#SBATCH --partition=cpu-gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/behavior_tsne_%j.out
#SBATCH --error=slurm_logs/behavior_tsne_%j.err

set -euo pipefail

cd /mnt/home/tianyuez/landscape-v2
source .venv/bin/activate
cd control

mkdir -p slurm_logs

echo "=========================================="
echo "[Behavior Space t-SNE] Starting job"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Install required packages if not available
echo "Checking and installing required packages..."
pip install -q scikit-learn plotly 2>/dev/null || echo "Packages may already be installed"

# Run the behavior space t-SNE script
python3 visualization/plot_behavior_space_tsne.py \
    --retrain-models-dir /mnt/home/tianyuez/landscape-v2/control/models/jump_retrain_all \
    --n-episodes-per-policy 10 \
    --n-representative-states 1000 \
    --max-steps-per-episode 1000 \
    --device cuda \
    --env-name HalfCheetah-v4

echo "End time: $(date)"
echo "Job completed"

