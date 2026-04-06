#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM Job Array: Ablation study — 8 conditions (2x2x2)
#
# Factors:
#   1. Gaussian augmentation:  ON / OFF
#   2. Extra features:         ALL (encoder+spectral+acoustic) / ENCODER_ONLY
#   3. Pooling:                SWE / AVERAGE
#
# Submit:
#   sbatch slurm/03_ablation.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=usv_ablation
#SBATCH -p scavenger-h200
#SBATCH -A scavenger-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_DIR="/hpc/group/naderilab/zz394/Mice"
REPO_DIR="${PROJECT_DIR}/usv_classifier"
AUDIO_DIR="${PROJECT_DIR}/USVs/USVRecordingsP7"
DETECTIONS_DIR="${PROJECT_DIR}/dectections_squeakout"
CACHE_DIR="${PROJECT_DIR}/feature_cache"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "${REPO_DIR}/logs" "${CACHE_DIR}"

# ── Decode array task ID into 3 binary factors ──
# Bit 0: augmentation (0=ON, 1=OFF)
# Bit 1: features     (0=ALL, 1=ENCODER_ONLY)
# Bit 2: pooling      (0=SWE, 1=AVERAGE)
TASK=${SLURM_ARRAY_TASK_ID}

AUG_OFF=$(( (TASK >> 0) & 1 ))
ENC_ONLY=$(( (TASK >> 1) & 1 ))
AVG_POOL=$(( (TASK >> 2) & 1 ))

# Build CLI flags
EXTRA_FLAGS=""
LABEL="aug"

if [ ${AUG_OFF} -eq 1 ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --no_augmentation"
    LABEL="noaug"
else
    LABEL="aug"
fi

if [ ${ENC_ONLY} -eq 1 ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --encoder_only"
    LABEL="${LABEL}_enconly"
else
    LABEL="${LABEL}_allfeat"
fi

if [ ${AVG_POOL} -eq 1 ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --pooler average"
    LABEL="${LABEL}_avg"
else
    EXTRA_FLAGS="${EXTRA_FLAGS} --pooler swe"
    LABEL="${LABEL}_swe"
fi

echo "============================================================"
echo "  USV Ablation Study — Condition ${TASK}/7"
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
echo "  Condition ${LABEL} complete (exit code: ${EXIT_CODE})"
echo "  End      : $(date)"
echo "============================================================"

exit ${EXIT_CODE}
