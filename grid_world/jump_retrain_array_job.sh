#!/bin/bash
#SBATCH --job-name=gridworld_jr
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-79
#SBATCH --output=slurm_logs/jump_retrain_%A_%a.out
#SBATCH --error=slurm_logs/jump_retrain_%A_%a.err

# Config
# 20 policies (case 1: 10 seeds, case 2: 10 seeds)
# 4 step sizes per policy (5, 15, 20, 25)
# Total tasks = 20 * 4 = 80

CASE_IDS=(1 2)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
STEP_SIZES=(5 15 20 25)

ARRAY_ID=$SLURM_ARRAY_TASK_ID

# Compute case_id, seed, step_size, jr_index
# Each case has 10 seeds; each seed has 4 step sizes
SEEDS_PER_CASE=${#SEEDS[@]}
STEP_SIZES_COUNT=${#STEP_SIZES[@]}

# Total tasks = 2 cases * 10 seeds * 4 step_sizes = 80
TASKS_PER_CASE=$((SEEDS_PER_CASE * STEP_SIZES_COUNT))

CASE_IDX=$((ARRAY_ID / TASKS_PER_CASE))
SEED_IDX=$(( (ARRAY_ID % TASKS_PER_CASE) / STEP_SIZES_COUNT ))
STEP_IDX=$((ARRAY_ID % STEP_SIZES_COUNT))

CASE_ID=${CASE_IDS[$CASE_IDX]}
SEED=${SEEDS[$SEED_IDX]}
STEP_SIZE=${STEP_SIZES[$STEP_IDX]}
JR_INDEX=$STEP_IDX  # use per-seed step_size index as jr_index

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Jump & Retrain Task"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Array ID: $ARRAY_ID"
echo "Case ID: $CASE_ID"
echo "Seed: $SEED"
echo "Step Size: $STEP_SIZE"
echo "JR Index: $JR_INDEX"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd /mnt/home/tianyuez/landscape-v2/grid_world
mkdir -p slurm_logs
mkdir -p results/jump_retrain

python3 jump_and_retrain.py \
    --models-dir models \
    --case-id "$CASE_ID" \
    --seed "$SEED" \
    --step-sizes "$STEP_SIZE" \
    --extra-steps 10000 \
    --log-interval 1000 \
    --device cpu \
    --output-root results/jump_retrain \
    --jr-index "$JR_INDEX"

echo "Task completed: Case $CASE_ID, Seed $SEED, Step Size $STEP_SIZE"

