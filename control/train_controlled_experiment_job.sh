#!/bin/bash
#SBATCH --job-name=landscape-%a
#SBATCH --partition=cpu-gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-31%8
#SBATCH --chdir=/mnt/home/tianyuez/landscape-v2/control
#SBATCH --output=/mnt/home/tianyuez/landscape-v2/control/slurm_logs/controlled_%a.out
#SBATCH --error=/mnt/home/tianyuez/landscape-v2/control/slurm_logs/controlled_%a.err

set -euo pipefail
set -x

echo "JOB_START $(date)" || true
echo "HOST: $(hostname)" || true
pwd || true

# Create required directories
mkdir -p slurm_logs landscape_data models results

TASK_ID=${SLURM_ARRAY_TASK_ID}
ENVS=("Walker2d-v4" "HalfCheetah-v4")
TASKS_PER_ENV=16
ENV_INDEX=$(( TASK_ID / TASKS_PER_ENV ))
ENV_NAME=${ENVS[$ENV_INDEX]}
ENV_TASK_ID=$(( TASK_ID % TASKS_PER_ENV ))

# Seed assignment matches the original controlled tasks: two groups per environment
if [ ${ENV_TASK_ID} -lt 8 ]; then
    GROUP="FixedInit"
    INIT_SEED=200
    TRAIN_SEED=$((200 + ENV_TASK_ID))
else
    GROUP="FixedTrain"
    INIT_SEED=$((200 + ENV_TASK_ID - 8))
    TRAIN_SEED=200
fi

echo "=" | tr '\n' '=' | head -c 80; echo
echo "Task configuration:"
echo "  TASK_ID: $TASK_ID"
echo "  Env: $ENV_NAME"
echo "  Group: $GROUP"
echo "  INIT_SEED: $INIT_SEED"
echo "  TRAIN_SEED: $TRAIN_SEED"
echo "=" | tr '\n' '=' | head -c 80; echo

# Activate virtual environment
cd /mnt/home/tianyuez/landscape-v2
source .venv/bin/activate
python -V

cd /mnt/home/tianyuez/landscape-v2/control

export TRAIN_TOTAL_STEPS=10000000
export FEATURE_STRIDE=10000

python train_sb3_lstm_landscape.py \
    --env-name "${ENV_NAME}" \
    --task-id "${TASK_ID}" \
    --init-seed "${INIT_SEED}" \
    --train-seed "${TRAIN_SEED}" \
    --config "config/ppo_lstm_chatgpt.yml" \
    --total-steps 10000000 \
    --feature-stride "${FEATURE_STRIDE}"

echo "JOB_END $(date)"

