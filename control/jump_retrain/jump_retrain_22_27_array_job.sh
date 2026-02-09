#!/bin/bash
#SBATCH --job-name=jr_22_27_arr
#SBATCH --partition=cpu-gpu-v100
#SBATCH --output=slurm_logs/jr_22_27_arr_%A_%a.out
#SBATCH --error=slurm_logs/jr_22_27_arr_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-29

set -euo pipefail

# Activate environment
cd /mnt/home/tianyuez/landscape-v2
source .venv/bin/activate
cd control

mkdir -p slurm_logs

# Configuration
TASK_IDS=(22 27)
STEP_SIZES=(20 40 60)
NUM_SEEDS=5
EXTRA_STEPS=500000
ENV_NAME="HalfCheetah-v4"

# Map SLURM_ARRAY_TASK_ID to (task_id, step_size, seed_idx)
ARRAY_ID=$SLURM_ARRAY_TASK_ID

# Compute indices: task_id_index, step_size_index, seed_idx
# Total tasks = 2 tasks * 3 step_sizes * 5 seeds = 30
# Ordering: all combinations for task0, then all combinations for task1
# task0: (step0,seed0-4), (step1,seed0-4), (step2,seed0-4) = 15
# task1: (step0,seed0-4), (step1,seed0-4), (step2,seed0-4) = 15

TASKS_PER_STEP_SIZE=$NUM_SEEDS  # 5
STEPS_PER_TASK=$(( ${#STEP_SIZES[@]} * TASKS_PER_STEP_SIZE ))  # 3 * 5 = 15

TASK_IDX=$(( ARRAY_ID / STEPS_PER_TASK ))
STEP_IDX=$(( (ARRAY_ID % STEPS_PER_TASK) / TASKS_PER_STEP_SIZE ))
SEED_IDX=$(( ARRAY_ID % TASKS_PER_STEP_SIZE ))

TASK_ID=${TASK_IDS[$TASK_IDX]}
STEP_SIZE=${STEP_SIZES[$STEP_IDX]}

# Compute jr_index and rng_seed
JR_INDEX=$ARRAY_ID
RNG_SEED=$(( 10000 + JR_INDEX ))

echo "=========================================="
echo "[Jump & Retrain] Array Task ${ARRAY_ID}/29"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "=========================================="
echo "Config:"
echo "  Task ID: ${TASK_ID}"
echo "  Step Size: ${STEP_SIZE}"
echo "  Seed Index: ${SEED_IDX}"
echo "  JR Index: ${JR_INDEX}"
echo "  RNG Seed: ${RNG_SEED}"
echo "=========================================="

# Paths
BASE_TASK_DIR="models/controlled_task_${TASK_ID}"
MODELS_DIR="models/jump_retrain_22_27"
OUTPUT_ROOT="results/jump_retrain_22_27"
CONFIG_PATH="config/ppo_lstm_chatgpt.yml"

# Check base task directory
if [[ ! -d "${BASE_TASK_DIR}" ]]; then
    echo "Error: base task directory does not exist: ${BASE_TASK_DIR}"
    exit 1
fi

# Select checkpoint
if [[ -f "${BASE_TASK_DIR}/best_model.pt" ]]; then
    BASE_MODEL="${BASE_TASK_DIR}/best_model.pt"
elif [[ -f "${BASE_TASK_DIR}/final_model.pt" ]]; then
    BASE_MODEL="${BASE_TASK_DIR}/final_model.pt"
else
    echo "Error: could not find checkpoint under ${BASE_TASK_DIR}"
    exit 1
fi

echo "Using model: ${BASE_MODEL}"

# Run a single jump-and-retrain
# Use --jr-index to avoid filename collisions
python3 jump_retrain/jump_and_retrain_rl.py \
    --models-dir models \
    --base-task-id "${TASK_ID}" \
    --env-name "${ENV_NAME}" \
    --config "${CONFIG_PATH}" \
    --step-sizes "${STEP_SIZE}" \
    --num-directions 1 \
    --extra-steps "${EXTRA_STEPS}" \
    --device cuda \
    --rng-seed-base "${RNG_SEED}" \
    --jr-index "${JR_INDEX}" \
    --checkpoint-type best \
    --output-root "${OUTPUT_ROOT}" \
    --models-root "${MODELS_DIR}"

echo "=========================================="
echo "Task ${ARRAY_ID} finished: $(date)"
echo "=========================================="

