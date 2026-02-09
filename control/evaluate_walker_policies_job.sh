#!/bin/bash
#SBATCH --job-name=eval-walker
#SBATCH --partition=cpu-gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/eval_walker_%j.out
#SBATCH --error=slurm_logs/eval_walker_%j.err

set -euo pipefail

cd /mnt/home/tianyuez/landscape-v2
source .venv/bin/activate
cd control

mkdir -p slurm_logs
mkdir -p evaluation_results

echo "=========================================="
echo "[Evaluation] Walker2d tasks 0-15"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

python3 evaluate_walker_policies.py \
  --env-name Walker2d-v4 \
  --task-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --model-template models/controlled_task_{}/final_model.pt \
  --num-episodes 10 \
  --output-dir evaluation_results

echo "[Evaluation] Walker2d evaluation completed."












