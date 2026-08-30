#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${CLEAN_HORIZON_REPO_DIR:-/kaggle/working/dino}"
OUTPUT_DIR="${CLEAN_HORIZON_OUTPUT_DIR:-/kaggle/working/dino_clean_horizon_seed0}"
DATA_PATH="${CLEAN_HORIZON_DATA_PATH:-/kaggle/input/datasets/wilyzh/imagenet100/ImageNet100/train}"
VAL_PATH="${CLEAN_HORIZON_VAL_PATH:-/kaggle/input/datasets/wilyzh/imagenet100/ImageNet100/val}"
RESUME_FROM="${CLEAN_HORIZON_RESUME_FROM:-}"

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "Missing repository: ${REPO_DIR}" >&2
  exit 2
fi
if [[ ! -d "${DATA_PATH}" || ! -d "${VAL_PATH}" ]]; then
  echo "Missing ImageNet100 train/val input" >&2
  exit 2
fi
if [[ -n "${RESUME_FROM}" && ! -f "${RESUME_FROM}" ]]; then
  echo "Configured resume checkpoint does not exist: ${RESUME_FROM}" >&2
  exit 2
fi

cd "${REPO_DIR}"

python - <<'PY'
import torch

count = torch.cuda.device_count()
print("CUDA devices:", count)
for index in range(count):
    print(index, torch.cuda.get_device_name(index))
if count != 2:
    raise RuntimeError(f"Clean-horizon protocol requires exactly two GPUs, got {count}")
PY

mkdir -p "${OUTPUT_DIR}"

args=(
  --arch vit_small
  --patch_size 16
  --epochs 319
  --batch_size_per_gpu 64
  --accum_steps 2
  --drop_incomplete_accumulation true
  --warmup_teacher_temp_epochs 30
  --data_path "${DATA_PATH}"
  --val_data_path "${VAL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --saveckp_freq 10
  --keep_last_ckpts 0
  --milestone_ckpt_epochs 180 250 318
  --diag_every 5
  --attn_viz_every 25
  --use_fp16 true
  --local_crops_number 4
  --num_workers 2
  --teacher_temp 0.07
  --norm_last_layer false
  --seed 0
  --strict_resume_schedule true
  --expected_world_size 2
  --max_runtime_hours 11.5
  --runtime_reserve_minutes 45
  --run_name dino_v3_clean_horizon_seed0_v1
)

if [[ -n "${RESUME_FROM}" ]]; then
  args+=(--resume_from "${RESUME_FROM}")
fi

torchrun --nproc_per_node=2 main_dino.py "${args[@]}"

python - <<'PY'
import json
import os
from pathlib import Path

output_dir = Path(os.environ.get(
    "CLEAN_HORIZON_OUTPUT_DIR",
    "/kaggle/working/dino_clean_horizon_seed0",
))
summary = output_dir / "clean_horizon_session_summary.json"
if not summary.is_file():
    raise RuntimeError(f"Missing clean-horizon session summary: {summary}")
payload = json.loads(summary.read_text(encoding="utf-8"))
checkpoint = output_dir / "checkpoint.pth"
if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
    raise RuntimeError(f"Missing rolling checkpoint: {checkpoint}")
if payload["rolling_checkpoint"]["size_bytes"] != checkpoint.stat().st_size:
    raise RuntimeError("Rolling checkpoint size does not match the session summary")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
