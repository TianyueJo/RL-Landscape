#!/bin/bash
#SBATCH --job-name=record-controlled-videos
#SBATCH --partition=cpu-gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/record_controlled_videos_%j.out
#SBATCH --error=slurm_logs/record_controlled_videos_%j.err

set -euo pipefail

cd /mnt/home/tianyuez/landscape-v2
source .venv/bin/activate
cd control

export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
unset DISPLAY || true

mkdir -p slurm_logs
mkdir -p videos/controlled_seedmatch

MODELS_DIR="/mnt/home/tianyuez/landscape-v2/control/models"
VIDEO_DIR="/mnt/home/tianyuez/landscape-v2/control/videos/controlled_seedmatch"

echo "=========================================="
echo "Rendering videos for controlled_task_* models (best_model)"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Output dir: ${VIDEO_DIR}"
echo "MUJOCO_GL=${MUJOCO_GL}"
echo "PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}"
echo "=========================================="

shopt -s nullglob

TARGET_TASKS=("$@")

should_skip_task() {
    local task="$1"
    if [[ ${#TARGET_TASKS[@]} -eq 0 ]]; then
        return 1
    fi
    for t in "${TARGET_TASKS[@]}"; do
        if [[ "${t}" == "${task}" ]]; then
            return 1
        fi
    done
    return 0
}

total_tasks=0
success_tasks=0
failed_tasks=0

for summary_path in "${MODELS_DIR}"/controlled_task_*/training_summary.json; do
    task_dir=$(dirname "${summary_path}")
    task_name=$(basename "${task_dir}")
    task_id=${task_name##*_}

    if should_skip_task "${task_id}"; then
        continue
    fi

    if [[ ! -f "${task_dir}/best_model.pt" ]]; then
        echo "× Skip ${task_name}: missing best_model.pt"
        continue
    fi

    read -r env_name train_seed < <(
        python -c "import json, sys; d=json.load(open(sys.argv[1])); print(d['env_name'], d['train_seed'])" "${summary_path}"
    )

    video_env_dir="${VIDEO_DIR}/${env_name}"
    mkdir -p "${video_env_dir}"
    output_path="${video_env_dir}/${env_name}_task_${task_id}_best.mp4"
    stats_path="${task_dir}/vec_stats.npz"

    echo ""
    echo "------------------------------------------"
    echo "Task ${task_id}"
    echo "  Env: ${env_name}"
    echo "  Train seed: ${train_seed}"
    echo "  Output: ${output_path}"
    echo "------------------------------------------"

    if python record_policy_video.py \
        --env "${env_name}" \
        --model-path "${task_dir}/best_model.pt" \
        --output "${output_path}" \
        --task-id "${task_id}" \
        --num-episodes 3 \
        --max-steps 1000 \
        --fps 30 \
        --device cpu \
        --env-seed "${train_seed}" \
        --vec-stats-path "${stats_path}" \
        --generate-vec-stats \
        --force-generate-vec-stats \
        --stats-warmup-steps 65536 \
        --stats-num-envs 32 \
        --train-config "config/ppo_lstm_chatgpt.yml" \
        --deterministic; then
        echo "  ✓ Done ${task_name}"
        success_tasks=$((success_tasks + 1))
    else
        echo "  ✗ Failed ${task_name}"
        failed_tasks=$((failed_tasks + 1))
    fi

    total_tasks=$((total_tasks + 1))
done

echo ""
echo "=========================================="
echo "Total tasks: ${total_tasks}"
echo "Success: ${success_tasks}, Failed: ${failed_tasks}"
echo "Finished at: $(date)"
echo "=========================================="

