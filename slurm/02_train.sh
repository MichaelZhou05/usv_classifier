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
#   sbatch 02_train.sh
#
# Or submit with dependency on the detect array job:
#   DETECT_JOB=$(sbatch --parsable 01_detect.sh)
#   sbatch --dependency=afterok:$DETECT_JOB 02_train.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=usv_train
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
# Uncomment if GPU is available on your partition:
# #SBATCH --gres=gpu:1

# ── Edit these paths ──────────────────────────────────────────────────────────
REPO_DIR="$HOME/usv_classifier"
AUDIO_DIR="/path/to/wav_files"
DETECTIONS_DIR="/path/to/detections"
CACHE_DIR="/path/to/feature_cache"        # encoder output cache — reused across runs
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs "$CACHE_DIR"

module purge
module load Python/3.11.3-GCCcore-12.3.0   # adjust to your cluster's module name

cd "$REPO_DIR" || exit 1

# Check that all WAV files have a corresponding detections CSV
N_WAV=$(ls "$AUDIO_DIR"/*.wav 2>/dev/null | wc -l)
N_CSV=$(ls "$DETECTIONS_DIR"/*.csv 2>/dev/null | wc -l)
echo "WAV files:  $N_WAV"
echo "CSVs found: $N_CSV"
if [ "$N_CSV" -lt "$N_WAV" ]; then
    echo "WARNING: $((N_WAV - N_CSV)) recordings missing detection CSVs."
    echo "Re-run 01_detect.sh for missing files before training."
fi

python train.py \
    --config        config_squeakout.yaml \
    --data_dir      "$AUDIO_DIR" \
    --detections_dir "$DETECTIONS_DIR" \
    --cache_dir     "$CACHE_DIR"

echo "Training complete."
