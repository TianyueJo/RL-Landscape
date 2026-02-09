#!/bin/bash
#SBATCH --job-name=js-hc-array
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=slurm_logs/js_hc_array_%A_%a.out
#SBATCH --error=slurm_logs/js_hc_array_%A_%a.err

set -euo pipefail

cd /mnt/home/tianyuez/landscape-v2
source .venv/bin/activate
cd control

mkdir -p slurm_logs analysis_outputs/jump_retrain_js

TASK_IDS=(16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
TASK_ID="${TASK_IDS[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "${TASK_ID}" ]]; then
  echo "[Error] Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

echo "=========================================="
echo "[JS Divergence] HalfCheetah task ${TASK_ID}"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

python3 jump_retrain/compute_jump_retrain_js.py \
  --env-name HalfCheetah-v4 \
  --task-id "${TASK_ID}" \
  --base-models-dir models \
  --retrain-models-dir models/jump_retrain_hc_step_sweep \
  --step-sizes 0 5 15 45 135 \
  --num-samples 500 \
  --states-seed 12345 \
  --output-dir analysis_outputs/jump_retrain_js

echo "[Done] task ${TASK_ID}"


