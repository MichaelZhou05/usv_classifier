#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM Job: Stage 2 — Feature extraction + MLP training
#
# Depends on 01_detect.sh completing successfully (all CSVs present).
#
# First run:   extracts SqueakOut encoder features → saves to CACHE_DIR
#              then trains the MLP classifier
# Re-runs:     skips encoder (loads cached features), only re-trains MLP
#              → fast iteration on hyperparameters
#
# Submit after detection is complete:
#   sbatch slurm/02_train.sh
#
# Or submit with dependency on the detect array job:
#   DETECT_JOB=$(sbatch --parsable slurm/01_detect.sh)
#   sbatch --dependency=afterok:$DETECT_JOB slurm/02_train.sh
#
# Monitor progress (without reading the full log):
#   cat outputs/job_<JOBID>_*/progress.log
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=usv_train
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
# Uncomment if GPU is available on your partition:
# #SBATCH --gres=gpu:1

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_DIR="/hpc/group/naderilab/zz394/Mice"
REPO_DIR="${PROJECT_DIR}/usv_classifier"
AUDIO_DIR="${PROJECT_DIR}/USVs/USVRecordingsP7"
DETECTIONS_DIR="${PROJECT_DIR}/dectections_squeakout"
CACHE_DIR="${PROJECT_DIR}/feature_cache"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "${REPO_DIR}/logs" "${CACHE_DIR}"

echo "============================================================"
echo "  USV Classifier Training"
echo "  Job ID   : ${SLURM_JOB_ID}"
echo "  Start    : $(date)"
echo "  Audio    : ${AUDIO_DIR}"
echo "  Detects  : ${DETECTIONS_DIR}"
echo "  Cache    : ${CACHE_DIR}"
echo "============================================================"

module purge
module load Python/3.11.3-GCCcore-12.3.0

cd "${REPO_DIR}" || { echo "ERROR: could not cd to ${REPO_DIR}"; exit 1; }

# Check that all WAV files have a corresponding detections CSV
N_WAV=$(ls "${AUDIO_DIR}"/*.wav 2>/dev/null | wc -l)
N_CSV=$(ls "${DETECTIONS_DIR}"/*.csv 2>/dev/null | wc -l)
echo "WAV files : ${N_WAV}"
echo "CSVs found: ${N_CSV}"
if [ "${N_CSV}" -lt "${N_WAV}" ]; then
    echo "WARNING: $((N_WAV - N_CSV)) recordings missing detection CSVs."
    echo "Re-run slurm/01_detect.sh for missing files before training."
fi

python train.py \
    --config         config_squeakout.yaml \
    --data_dir       "${AUDIO_DIR}" \
    --detections_dir "${DETECTIONS_DIR}" \
    --cache_dir      "${CACHE_DIR}" \
    --job_id         "${SLURM_JOB_ID}" \
    --cv_folds       5

EXIT_CODE=$?

echo "============================================================"
echo "  Training complete (exit code: ${EXIT_CODE})"
echo "  End      : $(date)"
echo "  Progress : cat ${REPO_DIR}/outputs/job_${SLURM_JOB_ID}_*/progress.log"
echo "  Full log : ${REPO_DIR}/logs/train_${SLURM_JOB_ID}.out"
echo "============================================================"

exit ${EXIT_CODE}
