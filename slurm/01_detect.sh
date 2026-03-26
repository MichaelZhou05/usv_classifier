#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM Job: Stage 1 — USV call detection (sliding window SqueakOut)
#
# Runs detect_calls.py once per WAV file in parallel via job array.
# Each job is independent and takes ~30s–2min per 5-minute recording on CPU.
#
# Submit:
#   sbatch 01_detect.sh
#
# After all jobs finish, check outputs:
#   ls $DETECTIONS_DIR/*.csv | wc -l   # should equal number of WAV files
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=usv_detect
#SBATCH --array=0-999%20         # array indices auto-trimmed to actual file count;
                                  # %20 = run max 20 jobs simultaneously
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00           # 30 min is generous for a 5-min WAV on CPU
#SBATCH --output=logs/detect_%A_%a.out
#SBATCH --error=logs/detect_%A_%a.err

# ── Edit these paths ──────────────────────────────────────────────────────────
REPO_DIR="$HOME/usv_classifier"           # path to cloned usv_classifier repo
AUDIO_DIR="/path/to/wav_files"            # directory of *.wav recordings
DETECTIONS_DIR="/path/to/detections"      # output: one *.csv per recording
SQUEAKOUT_WEIGHTS="$HOME/squeakout/squeakout_weights.ckpt"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs "$DETECTIONS_DIR"

# Build array of WAV files at submission time
WAV_FILES=("$AUDIO_DIR"/*.wav)
N_FILES=${#WAV_FILES[@]}

if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_FILES" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID >= number of files ($N_FILES), exiting."
    exit 0
fi

AUDIO_FILE="${WAV_FILES[$SLURM_ARRAY_TASK_ID]}"
BASENAME=$(basename "$AUDIO_FILE" .wav)
OUT_CSV="$DETECTIONS_DIR/${BASENAME}.csv"

echo "Processing: $AUDIO_FILE"
echo "Output:     $OUT_CSV"

# Skip if already done (allows safe re-submission after partial failures)
if [ -f "$OUT_CSV" ]; then
    echo "Already exists, skipping."
    exit 0
fi

module purge
module load Python/3.11.3-GCCcore-12.3.0   # adjust to your cluster's module name

cd "$REPO_DIR" || exit 1

python detect_calls.py \
    --audio   "$AUDIO_FILE" \
    --weights "$SQUEAKOUT_WEIGHTS" \
    --out_csv "$OUT_CSV" \
    --window_sec 0.5 \
    --overlap_sec 0.1 \
    --freq_min 30000 \
    --freq_max 130000

echo "Done. $(wc -l < "$OUT_CSV") calls detected."
