#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM Job Array: SWE-large (num_slices=128, num_ref_points=70), MLP only
#
# Two conditions:
#   Task 0: aug + all features + SWE
#   Task 1: aug + encoder_only + SWE
#
# Submit:
#   sbatch slurm/04_swe_large.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=usv_swe_large
#SBATCH -p scavenger-h200
#SBATCH -A scavenger-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/swe_large_%A_%a.out
#SBATCH --error=logs/swe_large_%A_%a.err

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_DIR="/hpc/group/naderilab/zz394/Mice"
REPO_DIR="${PROJECT_DIR}/usv_classifier"
AUDIO_DIR="${PROJECT_DIR}/USVs/USVRecordingsP7"
DETECTIONS_DIR="${PROJECT_DIR}/dectections_squeakout"
CACHE_DIR="${PROJECT_DIR}/feature_cache"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "${REPO_DIR}/logs" "${CACHE_DIR}"

TASK=${SLURM_ARRAY_TASK_ID}

EXTRA_FLAGS="--pooler swe --mlp_only"
if [ "${TASK}" -eq 0 ]; then
    LABEL="aug_allfeat_swe_L128_M70"
elif [ "${TASK}" -eq 1 ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --encoder_only"
    LABEL="aug_enconly_swe_L128_M70"
else
    echo "ERROR: unknown task id ${TASK}"
    exit 1
fi

echo "============================================================"
echo "  USV SWE-large run (MLP only)"
echo "  Task     : ${TASK}"
echo "  Label    : ${LABEL}"
echo "  Flags    : ${EXTRA_FLAGS}"
echo "  Job ID   : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "  Start    : $(date)"
echo "============================================================"

module purge
module load Anaconda3/2024.02
eval "$(conda shell.bash hook)"
conda activate base

cd "${REPO_DIR}" || { echo "ERROR: could not cd to ${REPO_DIR}"; exit 1; }

python train_enhanced.py \
    --config         config_squeakout.yaml \
    --data_dir       "${AUDIO_DIR}" \
    --detections_dir "${DETECTIONS_DIR}" \
    --cache_dir      "${CACHE_DIR}" \
    --cv_folds       5 \
    --job_id         "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    ${EXTRA_FLAGS}

EXIT_CODE=$?

echo "============================================================"
echo "  ${LABEL} complete (exit code: ${EXIT_CODE})"
echo "  End      : $(date)"
echo "============================================================"

exit ${EXIT_CODE}
