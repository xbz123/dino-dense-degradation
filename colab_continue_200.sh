#!/bin/bash
# =============================================================================
# DINO Colab Continue Training Script
#
# Usage in Colab:
#   from google.colab import drive
#   drive.mount('/content/drive', force_remount=True, timeout_ms=600000)
#
#   %cd /content
#   !rm -rf /content/dino
#   !git clone https://github.com/xbz123/dino-dense-degradation.git /content/dino
#   %cd /content/dino
#   !bash colab_continue_200.sh
#
# What this script does:
#   1. Uses /content/drive/MyDrive/imagenet100.tar when available.
#   2. Extracts ImageNet100 to /content/imagenet100 on local SSD.
#   3. Resumes from /content/drive/MyDrive/dino_colab/dino_output/checkpoint.pth.
#   4. If no output checkpoint exists, copies the latest checkpoint*.pth from
#      /content/drive/MyDrive/dinocheckpoint.
#   5. Continues training to 200 epochs.
# =============================================================================

set -euo pipefail

echo "========================================="
echo "  DINO continue training to 200 epochs"
echo "========================================="

# -----------------------------
# 0. Paths
# -----------------------------
DATA_DIR="/content/imagenet100"
DRIVE_DATA="/content/drive/MyDrive/imagenet100"
DRIVE_TAR="/content/drive/MyDrive/imagenet100.tar"
LOCAL_TAR="/content/imagenet100.tar"

DRIVE_ROOT="/content/drive/MyDrive/dino_colab"
CHECKPOINT_DIR="/content/drive/MyDrive/dinocheckpoint"
OUTPUT_DIR="${DRIVE_ROOT}/dino_output"
RESUME_CKPT="${OUTPUT_DIR}/checkpoint.pth"

mkdir -p "${DRIVE_ROOT}"
mkdir -p "${OUTPUT_DIR}"

echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"

# -----------------------------
# 1. Basic checks
# -----------------------------
if [ ! -d "/content/drive/MyDrive" ]; then
    echo "ERROR: Google Drive is not mounted."
    echo "Run this first in Colab:"
    echo "from google.colab import drive"
    echo "drive.mount('/content/drive', force_remount=True, timeout_ms=600000)"
    exit 1
fi

if [ ! -f "main_dino.py" ]; then
    echo "ERROR: main_dino.py not found. Run this script from /content/dino."
    exit 1
fi

# -----------------------------
# 2. Install dependencies
# -----------------------------
echo "Installing dependencies..."
pip install -q datasets matplotlib

# -----------------------------
# 3. Prepare ImageNet100 on local SSD
# -----------------------------
if [ -d "${DATA_DIR}/train" ] && [ -d "${DATA_DIR}/val" ]; then
    echo "Local ImageNet100 already exists."
else
    echo "Local ImageNet100 not found. Preparing data..."

    rm -rf "${DATA_DIR}"
    rm -f "${LOCAL_TAR}"

    if [ -s "${DRIVE_TAR}" ]; then
        echo "Using tar from Drive: ${DRIVE_TAR}"
        echo "Copying tar to local SSD..."
        cp "${DRIVE_TAR}" "${LOCAL_TAR}"

        echo "Testing tar..."
        tar -tf "${LOCAL_TAR}" >/dev/null

        echo "Extracting tar to /content..."
        tar xf "${LOCAL_TAR}" -C /content
        rm -f "${LOCAL_TAR}"

    elif [ -d "${DRIVE_DATA}/train" ] && [ -d "${DRIVE_DATA}/val" ]; then
        echo "WARNING: ${DRIVE_TAR} not found or empty."
        echo "Creating tar on Drive from ${DRIVE_DATA}."
        echo "This is a one-time slow step because Drive has many small files."

        rm -f "${DRIVE_TAR}"
        tar cf "${DRIVE_TAR}" -C "/content/drive/MyDrive" "imagenet100"

        echo "Copying new tar to local SSD..."
        cp "${DRIVE_TAR}" "${LOCAL_TAR}"

        echo "Testing tar..."
        tar -tf "${LOCAL_TAR}" >/dev/null

        echo "Extracting tar to /content..."
        tar xf "${LOCAL_TAR}" -C /content
        rm -f "${LOCAL_TAR}"

    else
        echo "ERROR: Need either:"
        echo "  ${DRIVE_TAR}"
        echo "or"
        echo "  ${DRIVE_DATA}/train and ${DRIVE_DATA}/val"
        exit 1
    fi
fi

echo "Checking dataset..."
TRAIN_COUNT=$(find "${DATA_DIR}/train" -type f | wc -l)
VAL_COUNT=$(find "${DATA_DIR}/val" -type f | wc -l)
TRAIN_CLASSES=$(find "${DATA_DIR}/train" -mindepth 1 -maxdepth 1 -type d | wc -l)
VAL_CLASSES=$(find "${DATA_DIR}/val" -mindepth 1 -maxdepth 1 -type d | wc -l)

echo "Train files: ${TRAIN_COUNT}"
echo "Val files: ${VAL_COUNT}"
echo "Train classes: ${TRAIN_CLASSES}"
echo "Val classes: ${VAL_CLASSES}"

# -----------------------------
# 4. Prepare resume checkpoint
# -----------------------------
if [ -s "${RESUME_CKPT}" ]; then
    echo "Resume checkpoint already exists in output dir:"
    echo "  ${RESUME_CKPT}"
    echo "Will resume from this checkpoint."
else
    echo "No checkpoint.pth found in output dir."
    echo "Looking for latest checkpoint in ${CHECKPOINT_DIR}..."

    if [ ! -d "${CHECKPOINT_DIR}" ]; then
        echo "ERROR: Checkpoint directory not found: ${CHECKPOINT_DIR}"
        exit 1
    fi

    LATEST_CKPT=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -name "checkpoint*.pth" -type f | sort -V | tail -n 1 || true)

    if [ -z "${LATEST_CKPT}" ]; then
        echo "ERROR: No checkpoint*.pth found in ${CHECKPOINT_DIR}"
        exit 1
    fi

    echo "Copying latest checkpoint:"
    echo "  from: ${LATEST_CKPT}"
    echo "  to:   ${RESUME_CKPT}"
    cp "${LATEST_CKPT}" "${RESUME_CKPT}"
fi

ls -lh "${RESUME_CKPT}"

# -----------------------------
# 5. Train
# -----------------------------
echo "Starting training..."

torchrun --nproc_per_node=1 main_dino.py \
    --arch vit_small \
    --patch_size 16 \
    --epochs 200 \
    --batch_size_per_gpu 64 \
    --accum_steps 2 \
    --warmup_teacher_temp_epochs 30 \
    --data_path "${DATA_DIR}/train" \
    --val_data_path "${DATA_DIR}/val" \
    --output_dir "${OUTPUT_DIR}" \
    --saveckp_freq 10 \
    --keep_last_ckpts 5 \
    --diag_every 5 \
    --attn_viz_every 25 \
    --use_fp16 true \
    --local_crops_number 4 \
    --num_workers 2 \
    --teacher_temp 0.07 \
    --norm_last_layer false
