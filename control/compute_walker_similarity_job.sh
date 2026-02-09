#!/bin/bash
#SBATCH --job-name=walker-js-sim
#SBATCH --partition=cpu-gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/walker_js_similarity_%j.out
#SBATCH --error=slurm_logs/walker_js_similarity_%j.err

set -euo pipefail

cd /mnt/home/tianyuez/landscape-v2
source .venv/bin/activate
cd control

mkdir -p slurm_logs
mkdir -p analysis_outputs/walker_similarity

echo "=========================================="
echo "[JS Divergence] Walker2d tasks 0-15"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

python3 compute_policy_similarity_matrix.py \
  --env-name Walker2d-v4 \
  --task-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --model-template models/controlled_task_{}/final_model.pt \
  --output-dir analysis_outputs/walker_similarity \
  --num-samples 500

echo "[JS Divergence] Walker2d completed."












