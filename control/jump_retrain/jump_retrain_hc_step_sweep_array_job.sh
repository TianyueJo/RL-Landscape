#!/bin/bash
#SBATCH --job-name=jr-hc-stepsweep
#SBATCH --partition=cpu-gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=slurm_logs/jr_hc_step_sweep_%A_%a.out
#SBATCH --error=slurm_logs/jr_hc_step_sweep_%A_%a.err

set -euo pipefail

cd /mnt/home/tianyuez/landscape-v2
source .venv/bin/activate
cd control

mkdir -p slurm_logs

TASK_IDS=(16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
TASK_ID="${TASK_IDS[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "${TASK_ID}" ]]; then
  echo "[Error] Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

STEP_SIZES=(0 5 15 45 135)

OUTPUT_ROOT="results/jump_retrain_hc_step_sweep"
MODELS_ROOT="models/jump_retrain_hc_step_sweep"

echo "=========================================="
echo "[Jump & Retrain] HalfCheetah task ${TASK_ID} | step_sizes=${STEP_SIZES[*]}"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

python3 jump_retrain/jump_and_retrain_rl.py \
  --models-dir models \
  --base-task-id "${TASK_ID}" \
  --env-name HalfCheetah-v4 \
  --config config/ppo_lstm_chatgpt.yml \
  --step-sizes "${STEP_SIZES[@]}" \
  --num-directions 1 \
  --extra-steps 500000 \
  --device cuda \
  --checkpoint-type final \
  --output-root "${OUTPUT_ROOT}" \
  --models-root "${MODELS_ROOT}"

echo "[Jump & Retrain] HalfCheetah task ${TASK_ID} step sweep completed."


