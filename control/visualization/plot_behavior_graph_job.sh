#!/bin/bash
#SBATCH --job-name=plot_behavior_graph
#SBATCH --partition=cpu-gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/plot_behavior_graph_%j.out
#SBATCH --error=slurm_logs/plot_behavior_graph_%j.err

set -euo pipefail

cd /mnt/home/tianyuez/landscape-v2
source .venv/bin/activate
cd control

mkdir -p slurm_logs
mkdir -p analysis_outputs

echo "=========================================="
echo "[Behavior Space Graph] Starting job"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Install required packages if not available
echo "Checking and installing required packages..."
pip install -q scikit-learn plotly networkx 2>/dev/null || echo "Packages may already be installed"

# Run the script for tasks 0-15 and enable graph visualization
python3 visualization/plot_behavior_space_pca.py \
    --task-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
    --models-dir models \
    --data-dir landscape_data \
    --output-path analysis_outputs/behavior_space_pca_3d.png \
    --output-html analysis_outputs/behavior_space_pca_3d.html \
    --env-name Walker2d-v4 \
    --visualize-graph \
    --pca-dims 2 6 10 \
    --device cuda \
    --n-episodes-per-policy 10 \
    --n-representative-states 1000 \
    --max-steps-per-episode 1000 \
    --graph-layout spring

echo "End time: $(date)"
echo "Job completed!"

