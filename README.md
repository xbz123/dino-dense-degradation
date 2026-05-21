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

The notebook is a thin Colab wrapper around these repository scripts:

- `analyze_patch_statistics.py`: scans checkpoints and computes paper-formula
  DSE class separability, effective rank, covariance spectrum, CLS-patch
  cosine, patch norm histograms, CLS attention statistics, fixed-query patch
  similarity maps, and deterministic fixed-basis PCA feature maps. It validates
  checkpoint filename/internal epochs before expensive feature extraction.
- `plot_dense_diagnostics.py`: merges patch diagnostics with VOC mIoU and
  writes the summary figure.
- `make_summary_report.py`: writes a compact Markdown report for the run.

Expected output folders:

```
MyDrive/dino_dense_degradation_eval/to_epoch_XXXX/
├── selected_checkpoints.json
├── voc_all_checkpoints/
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
| `--keep_last_ckpts` | `0` | Keep last N periodic checkpoints; `0` keeps all |

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
