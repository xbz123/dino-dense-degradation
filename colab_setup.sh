#!/bin/bash
# =============================================================================
# DINO IN-100 Dense Degradation Experiment — Colab Setup Script
#
# Usage (in Colab cell):
#   !bash colab_setup.sh [baseline|high_temp]
#
# Prerequisites:
#   - Google Drive mounted at /content/drive
#   - imagenet100.tar uploaded to /content/drive/MyDrive/
# =============================================================================

set -e

EXPERIMENT=${1:-baseline}
echo "========================================="
echo "  DINO IN-100 Dense Degradation"
echo "  Experiment: ${EXPERIMENT}"
echo "========================================="

# --- 1. Install dependencies ---
pip install -q matplotlib

# --- 2. Copy data from Drive to local SSD (CRITICAL for I/O speed) ---
DATA_DIR="/content/imagenet100"
if [ ! -d "${DATA_DIR}/train" ]; then
    echo "Copying ImageNet-100 from Drive to local SSD..."
    cp /content/drive/MyDrive/imagenet100.tar /content/
    tar xf /content/imagenet100.tar -C /content/
    rm /content/imagenet100.tar
    echo "Data ready: $(find ${DATA_DIR}/train -type f | wc -l) training images"
else
    echo "Data already present: $(find ${DATA_DIR}/train -type f | wc -l) training images"
fi

# --- 3. Set output directory on Drive (for checkpoint persistence) ---
OUTPUT_DIR="/content/drive/MyDrive/dino_in100_${EXPERIMENT}"
mkdir -p "${OUTPUT_DIR}"

# --- 4. Training parameters ---
COMMON_ARGS="
    --arch vit_small
    --patch_size 16
    --epochs 800
    --batch_size_per_gpu 32
    --accum_steps 8
    --warmup_teacher_temp_epochs 30
    --data_path ${DATA_DIR}/train
    --val_data_path ${DATA_DIR}/val
    --output_dir ${OUTPUT_DIR}
    --saveckp_freq 20
    --keep_last_ckpts 3
    --diag_every 10
    --attn_viz_every 50
    --use_fp16 true
    --local_crops_number 6
    --num_workers 2
    --norm_last_layer false
"

if [ "${EXPERIMENT}" == "baseline" ]; then
    echo "Running BASELINE experiment (teacher_temp=0.07)"
    python main_dino.py ${COMMON_ARGS} \
        --teacher_temp 0.07
elif [ "${EXPERIMENT}" == "high_temp" ]; then
    echo "Running HIGH TEMP experiment (teacher_temp=0.09)"
    python main_dino.py ${COMMON_ARGS} \
        --teacher_temp 0.09
else
    echo "Unknown experiment: ${EXPERIMENT}"
    echo "Usage: bash colab_setup.sh [baseline|high_temp]"
    exit 1
fi
