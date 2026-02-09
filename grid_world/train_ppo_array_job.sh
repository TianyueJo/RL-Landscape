#!/bin/bash
#SBATCH --job-name=gridworld_ppo_case3_arr
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --array=0-19
#SBATCH --output=slurm_logs/train_ppo_case3_arr_%A_%a.out
#SBATCH --error=slurm_logs/train_ppo_case3_arr_%A_%a.err

set -euo pipefail

# Set working directory
cd /mnt/home/tianyuez/landscape-v2/grid_world

# Create logs directory
mkdir -p slurm_logs

# Set random seed from SLURM_ARRAY_TASK_ID
SEED=$SLURM_ARRAY_TASK_ID
TOTAL_TIMESTEPS=50000
CASE_ID=3

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Task info:"
echo "  Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "  Seed: $SEED"
echo "  Case ID: $CASE_ID"
echo "  Total timesteps: $TOTAL_TIMESTEPS"
echo "  Node: $(hostname)"
echo "  Time: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run training script
python3 train.py --seed $SEED --total-timesteps $TOTAL_TIMESTEPS --case-id $CASE_ID

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Training completed!"
echo "  Seed: $SEED"
echo "  Finished at: $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"





