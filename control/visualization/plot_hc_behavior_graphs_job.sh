#!/bin/bash
#SBATCH --job-name=hc_behavior_graph
#SBATCH --partition=cpu-gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/hc_behavior_graph_%j.out
#SBATCH --error=slurm_logs/hc_behavior_graph_%j.err

set -euo pipefail

cd /mnt/home/tianyuez/landscape-v2
source .venv/bin/activate
cd control

mkdir -p slurm_logs
mkdir -p analysis_outputs/hc_behavior_graphs

echo "=========================================="
echo "[HalfCheetah Behavior Graph] Starting job"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Step 1: generate behavior-space data (if missing)
DATA_FILE="analysis_outputs/behavior_space_pca_3d_hc.npz"

if [ ! -f "$DATA_FILE" ]; then
    echo "Data file missing; generating..."
    python3 visualization/plot_behavior_space_pca.py \
        --task-ids 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
        --models-dir models \
        --data-dir landscape_data \
        --output-path analysis_outputs/behavior_space_pca_3d_hc.png \
        --output-html analysis_outputs/behavior_space_pca_3d_hc.html \
        --env-name HalfCheetah-v4 \
        --device cuda \
        --n-episodes-per-policy 10 \
        --n-representative-states 1000 \
        --max-steps-per-episode 1000 \
        --pca-dims 2 6 10
else
    echo "Data file exists: $DATA_FILE"
fi

# Step 2: generate graphs for multiple thresholds
echo ""
echo "Generating visualizations for multiple thresholds..."
python3 visualization/plot_behavior_graph_multiple_thresholds.py \
    --data-file "$DATA_FILE" \
    --output-dir analysis_outputs/hc_behavior_graphs \
    --env-name "HalfCheetah-v4" \
    --pca-dims 2 6 10 \
    --layout spring \
    --separate-components \
    --models-dir models

echo "End time: $(date)"
echo "Job completed! Results saved to: analysis_outputs/hc_behavior_graphs/"

