# DINO + Dense Degradation Diagnostics (ImageNet-100)

> Fork of [facebookresearch/dino](https://github.com/facebookresearch/dino) with added instrumentation to detect **Dense Degradation** during long-horizon self-supervised pre-training.

## What is Dense Degradation?

Dense Degradation is a phenomenon observed in DINO/DINOv2-style self-supervised training where:
- **Global features** (CLS token) continue to improve throughout training
- **Dense/local features** (patch tokens) degrade after a certain training horizon

This creates a *decoupled dynamic*: KNN classification accuracy plateaus or improves, while dense downstream tasks (segmentation, detection) suffer.

**References:**
- DINOv3 (Simeoni et al., 2025) — first identifies dense degradation in long DINO training
- SDD (Dai et al., 2025) — explores structural degradation in dense representations

## Added Features

### 1. Dense Degradation Diagnostics (`dense_diagnostics.py`)
- **Effective Rank**: Tracks the dimensionality of patch token representations via covariance eigenvalue entropy. A sharp drop indicates collapse.
- **CLS-Patch Cosine Similarity**: Monitors feature homogenization between global and local tokens. A sharp rise indicates degradation.
- **Condition Number**: Tracks covariance matrix conditioning.
- **Attention Map Visualization**: Saves CLS→patch attention heatmaps at regular intervals to visually inspect degradation.

### 2. Gradient Accumulation (`--accum_steps`)
Enables training on small GPUs (e.g., T4 16GB) by accumulating gradients over multiple forward passes:
```bash
--batch_size_per_gpu 32 --accum_steps 8  # effective batch size = 256
```

### 3. Checkpoint Management (`--saveckp_freq`, `--keep_last_ckpts`)
Save periodic checkpoints for dense-degradation sweeps. The current evaluation
workflow expects historical checkpoints every 10 epochs and keeps them all:
```bash
--saveckp_freq 10 --keep_last_ckpts 0
```

## Quick Start (Google Colab)

The commands below are historical exploratory examples. They are not the
current registered clean-horizon experiment. Formal continuation now uses the
[clean-horizon protocol](CLEAN_HORIZON_BASELINE_PROTOCOL_2026-08-30.md) and
[Kaggle guide](KAGGLE_GUIDE.md), pinned to source `7404e7f`.

### 1. Setup
```python
# Clone this repo
!git clone https://github.com/xbz123/dino-dense-degradation.git /content/dino
%cd /content/dino

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Prepare ImageNet-100
```python
# Copy from Drive to local SSD (critical for I/O speed!)
!cp /content/drive/MyDrive/imagenet100.tar /content/
!tar xf /content/imagenet100.tar -C /content/
!rm /content/imagenet100.tar
```

### 3. Train (Baseline)
```bash
python main_dino.py \
    --arch vit_small \
    --patch_size 16 \
    --epochs 800 \
    --batch_size_per_gpu 32 \
    --accum_steps 8 \
    --teacher_temp 0.07 \
    --warmup_teacher_temp_epochs 30 \
    --data_path /content/imagenet100/train \
    --val_data_path /content/imagenet100/val \
    --output_dir /content/drive/MyDrive/dino_in100_baseline \
    --saveckp_freq 10 \
    --keep_last_ckpts 0 \
    --diag_every 10 \
    --attn_viz_every 50 \
    --use_fp16 true \
    --local_crops_number 6 \
    --num_workers 2 \
    --norm_last_layer false
```

### 4. Train (High Temperature — Accelerated Degradation)
```bash
# Same as above but with --teacher_temp 0.09
python main_dino.py \
    --arch vit_small \
    --teacher_temp 0.09 \
    ...  # same other args
```

### 5. KNN Evaluation
```bash
python eval_knn.py \
    --pretrained_weights /content/drive/MyDrive/dino_in100_baseline/checkpoint.pth \
    --checkpoint_key teacher \
    --data_path /content/imagenet100
```

### 6. Dense Degradation Evaluation
For the full checkpoint sweep, open:

```
notebooks/colab_dense_degradation_all_checkpoints.ipynb
```

The notebook scans `MyDrive/dinocheckpoint` for all recognizable
`checkpoint*.pth` files, runs PASCAL VOC frozen-backbone linear segmentation,
and exports DSE patch statistics plus fixed-image CLS attention, patch
similarity, and patch feature visualizations. Results are saved under:

```
MyDrive/dino_dense_degradation_eval/to_epoch_XXXX/
```

where `XXXX` is the largest checkpoint epoch found in the Drive folder.

VOC probe randomness is explicit. `eval_voc_dense.py` accepts
`--probe_seed`, resets that seed before every checkpoint head fit, and records
it in `voc_miou_results_global_confusion_v2.json`. Formal runs also pass
`--checkpoint_key teacher` explicitly and record the selected representation.
Metric v2 accumulates intersections and unions over the full validation set
before averaging per-class IoU. Each formal row also records structured
checkpoint identity, probe configuration, dataset identity, and Git
commit/dirty state.
Formal readers require `source_dirty=false`.
The notebooks expose these settings as `VOC_PROBE_SEED` and
`VOC_CHECKPOINT_KEY`; use separate output directories when measuring multiple
seeds.

The historical `voc_miou_results.json` rows use the batch-mean-v1 estimator.
They remain historical evidence only and must not be mixed with
`global_confusion_v2` rows in plots, tables, COCO comparisons, or phenomenon
gates. Formal readers validate metric version, probe seed, checkpoint key,
representation, and provenance before combining results.

The notebook is a thin Colab wrapper around these repository scripts:

- `eval_coco_stuff_dense.py`: runs selected-checkpoint COCO-Stuff
  frozen-backbone linear probing, writes
  `coco_stuff_miou_results_global_confusion_v2.json`, and only compares it
  with VOC rows that have the same v2 metric, probe seed, and checkpoint key.
- `analyze_patch_statistics.py`: scans checkpoints and computes paper-formula
  DSE class separability, effective rank, covariance spectrum, CLS-patch
  cosine, patch norm histograms, CLS attention statistics, fixed-query patch
  similarity maps, and deterministic fixed-basis PCA feature maps. It validates
  checkpoint filename/internal epochs before expensive feature extraction.
- `plot_dense_diagnostics.py`: merges patch diagnostics with VOC mIoU and
  writes the summary figure.
- `make_summary_report.py`: writes a compact Markdown report for the run.
- `audit_training_schedule.py`: audits checkpoint coordinates, schedule
  identities, independent session logs, and reconstructed LR/WD/teacher
  momentum before a trajectory is treated as one training horizon.

Expected output folders:

```
MyDrive/dino_dense_degradation_eval/to_epoch_XXXX/
├── selected_checkpoints.json
├── voc_all_checkpoints/
│   └── voc_miou_results_global_confusion_v2.json
├── patch_attention_dse_all_checkpoints/
│   ├── query_points.json
│   ├── fixed_images.json
│   ├── pca_fixed_basis.pt
│   ├── pca_fixed_basis_config.json
│   └── epoch_XXXX/
├── figures/
└── summary_report.md
```

## New CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--val_data_path` | `''` | Path to validation data for diagnostics |
| `--diag_every` | `10` | Compute diagnostics every N epochs |
| `--attn_viz_every` | `50` | Save attention maps every N epochs |
| `--diag_num_batches` | `50` | Validation batches for diagnostics |
| `--accum_steps` | `1` | Gradient accumulation steps |
| `--drop_incomplete_accumulation` | `false` | Drop a final partial accumulation group |
| `--keep_last_ckpts` | `0` | Keep last N periodic checkpoints; `0` keeps all |
| `--milestone_ckpt_epochs` | empty | Always save selected zero-based labels |
| `--strict_resume_schedule` | `false` | Require an exact saved training contract and RNG state |
| `--expected_world_size` | `0` | Require one distributed world size when non-zero |
| `--max_runtime_hours` | `0` | Enable epoch-boundary session runtime guard |

## Expected Observations

During 800 epochs on IN-100 with ViT-S/16:

| Metric | Early (0-200) | Mid (200-400) | Late (400-800) |
|--------|---------------|---------------|----------------|
| KNN Accuracy | ↑ rapid rise | ↑ slow rise | → plateau |
| Effective Rank | → stable/high | ↓ begins dropping | ↓↓ sharp decline |
| CLS-Patch Cosine | → low | ↑ begins rising | ↑↑ sharp rise |
| Attention Maps | Precise object outlines | Slightly diffuse | Blurry/background |

With `--teacher_temp 0.09`, the degradation onset should shift ~100 epochs earlier.

## T4 16GB Memory Budget

| Config | batch_size | local_crops | Est. VRAM | Status |
|--------|-----------|-------------|-----------|--------|
| ViT-S/16 + AMP + 6 local | 32 | 6 | ~12 GB | ✅ OK |
| ViT-S/16 + AMP + 8 local | 32 | 8 | ~14 GB | ⚠️ Tight |
| ViT-S/16 + AMP + 4 local | 48 | 4 | ~13 GB | ✅ OK |

## Training Time Estimate

- IN-100: ~130K images, effective batch 256 → ~510 iterations/epoch
- T4 with AMP: ~0.5-0.8 sec/step → **4-7 min/epoch**
- 800 epochs → **55-95 hours (~3-4 days)**
- Colab free (12h sessions) → 5-8 resume cycles
- Colab Pro (24h sessions) → 2-4 resume cycles

## Output Structure

```
output_dir/
├── checkpoint.pth                    # latest checkpoint (always kept)
├── checkpoint0000.pth                # periodic checkpoint
├── checkpoint0010.pth
├── checkpoint0020.pth
├── log.txt                           # training log (JSON lines)
├── attention_epoch0000/              # attention maps
│   ├── attn_img00042.png
│   ├── attn_img00123.png
│   └── ...
├── attention_epoch0050/
│   └── ...
└── ...
```

## License

This project inherits the Apache 2.0 license from the original DINO repository.
