#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM Job: Stage 1 — USV call detection (sliding window SqueakOut)
#
# Runs detect_calls.py once per WAV file in parallel via job array.
# Each job is independent and takes ~30s–2min per 5-minute recording on CPU.
#
# Submit:
#   sbatch slurm/01_detect.sh
#
# After all jobs finish, check outputs:
#   ls /hpc/group/naderilab/zz394/Mice/dectections_squeakout/*.csv | wc -l
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=usv_detect
#SBATCH -p scavenger-h200
#SBATCH -A scavenger-h200
#SBATCH --gres=gpu:1
#SBATCH --array=0-999%20         # array indices auto-trimmed to actual file count;
                                  # %20 = run max 20 jobs simultaneously
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/detect_%A_%a.out
#SBATCH --error=logs/detect_%A_%a.err

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_DIR="/hpc/group/naderilab/zz394/Mice"
REPO_DIR="${PROJECT_DIR}/usv_classifier"
AUDIO_DIR="${PROJECT_DIR}/USVs/USVRecordingsP7"
DETECTIONS_DIR="${PROJECT_DIR}/dectections_squeakout"
SQUEAKOUT_WEIGHTS="${PROJECT_DIR}/squeakout/squeakout_weights.ckpt"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "${REPO_DIR}/logs" "${DETECTIONS_DIR}"

# Build array of WAV files at submission time
WAV_FILES=("${AUDIO_DIR}"/*.wav)
N_FILES=${#WAV_FILES[@]}

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_FILES}" ]; then
    echo "Task ID ${SLURM_ARRAY_TASK_ID} >= number of files (${N_FILES}), exiting."
    exit 0
fi

AUDIO_FILE="${WAV_FILES[$SLURM_ARRAY_TASK_ID]}"
BASENAME=$(basename "${AUDIO_FILE}" .wav)
OUT_CSV="${DETECTIONS_DIR}/${BASENAME}.csv"

echo "Job      : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "File     : ${AUDIO_FILE}"
echo "Output   : ${OUT_CSV}"

# Skip if already done (allows safe re-submission after partial failures)
if [ -f "${OUT_CSV}" ]; then
    echo "Already exists, skipping."
    exit 0
fi

module purge
module load Python/3.11.3-GCCcore-12.3.0

cd "${REPO_DIR}" || { echo "ERROR: could not cd to ${REPO_DIR}"; exit 1; }

python detect_calls.py \
    --audio   "${AUDIO_FILE}" \
    --weights "${SQUEAKOUT_WEIGHTS}" \
    --out_csv "${OUT_CSV}" \
    --device  cuda \
    --window_sec 0.5 \
    --overlap_sec 0.1 \
    --freq_min 30000 \
    --freq_max 130000

echo "Done. $(wc -l < "${OUT_CSV}") calls detected."
