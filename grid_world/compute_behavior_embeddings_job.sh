#!/bin/bash
#SBATCH --job-name=gw_behavior_embed
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/behavior_embed_%j.out
#SBATCH --error=slurm_logs/behavior_embed_%j.err

set -euo pipefail

cd /mnt/home/tianyuez/landscape-v2
source .venv/bin/activate
cd grid_world

mkdir -p slurm_logs

echo "=========================================="
echo "[GridWorld Behavior Embeddings] Starting job"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Run behavior embeddings computation
python3 compute_behavior_embeddings.py \
    --models-dir /mnt/home/tianyuez/landscape-v2/grid_world/models \
    --output-dir /mnt/home/tianyuez/landscape-v2/grid_world/behavior_space_embeddings \
    --case-ids 1 2 3 \
    --dims 2 3 6 9 12 \
    --device cpu \
    --n-episodes-per-policy 10 \
    --max-steps-per-episode 200 \
    --n-representative-states 1000

echo "End time: $(date)"
echo "Job completed"

