#!/bin/bash
#SBATCH --job-name=jump-retrain-all
#SBATCH --partition=cpu-gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/jump_retrain_all_%j.out
#SBATCH --error=slurm_logs/jump_retrain_all_%j.err

set -euo pipefail

cd /mnt/home/tianyuez/landscape-v2
source .venv/bin/activate
cd control

mkdir -p slurm_logs

OUTPUT_ROOT="/mnt/home/tianyuez/landscape-v2/results/jump_retrain_all"
MODELS_ROOT="/mnt/home/tianyuez/landscape-v2/models/jump_retrain_all"

python3 jump_retrain/batch_jump_and_retrain.py \
  --models-dir /mnt/home/tianyuez/landscape-v2/control/models \
  --config /mnt/home/tianyuez/landscape-v2/control/config/ppo_lstm_chatgpt.yml \
  --device cuda \
  --step-size 0.5 \
  --num-directions 5 \
  --extra-steps 500000 \
  --checkpoint-type final \
  --output-root "${OUTPUT_ROOT}" \
  --models-root "${MODELS_ROOT}"




