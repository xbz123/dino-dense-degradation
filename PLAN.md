# Dense Degradation Diagnostic Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` for independent implementation tasks, or `superpowers:executing-plans` for inline task execution. Track progress with checklist syntax and verify each step before claiming completion.

**Goal:** Build a reliable dense patch representation diagnostic suite for DINO ViT-S/16 checkpoints trained on ImageNet-100, so the project can evaluate whether SDD-like degradation appears in downstream and structural metrics as training continues beyond epoch 215.

**Architecture:** Kaggle is used for long-running DINO pretraining and checkpoint production. Colab is used for Drive-based checkpoint sweeps: PASCAL VOC linear segmentation, patch/DSE/attention diagnostics, fixed-image qualitative visualizations, summary plots, and Markdown reports. Repository code stays lightweight and reproducible; generated checkpoints, datasets, figures, and papers stay outside Git.

**Tech Stack:** PyTorch, torchvision, DINO ViT-S/16, Kaggle GPU T4 x2, Google Colab, Google Drive, PASCAL VOC, ImageNet-100, Matplotlib, pytest.

---

## Current Scope

The project is currently positioned as a diagnostic suite, not as a claim that the reduced ImageNet-100 setup has already reproduced the full SDD paper degradation curve.

The working claim is:

> Under the current ImageNet-100 + DINO ViT-S/16 + PASCAL VOC linear segmentation setup, VOC mIoU alone has not yet shown a clear downstream dense degradation drop up to the latest evaluated checkpoints. The next step is to inspect structural patch-level diagnostics and continue training to longer horizons.

Current correction:

> The previous raw-feature DSE/class-separability trend is a warning signal, not final evidence of angular structural degradation. Patch feature norms changed strongly during training, so the next validation run must compare raw final-LayerNorm patch tokens against L2-normalized patch tokens.

## Implementation Order

### Phase 1: Current Colab Evaluation Run

Use this phase when new Kaggle checkpoints have been copied or uploaded into Google Drive.

1. Put all available checkpoints into:

   ```text
   /content/drive/MyDrive/dinocheckpoint/
   ```

2. Open and run:

   ```text
   notebooks/colab_dense_degradation_all_checkpoints.ipynb
   ```

3. Keep the default Drive paths unless the folder layout changed:

   ```python
   DRIVE_CHECKPOINT_DIR = '/content/drive/MyDrive/dinocheckpoint'
   OUTPUT_ROOT = '/content/drive/MyDrive/dino_dense_degradation_eval'
   ```

4. Confirm the notebook resolved the ImageNet-100 training image root:

   ```text
   /content/drive/MyDrive/imagenet100/train
   ```

5. Run the full checkpoint sweep after a smoke test passes.

Smoke-test configuration:

```python
CHECKPOINT_EPOCH_FILTER = [215]
NUM_DSE_IMAGES = 128
NUM_VIS_IMAGES = 3
PATCH_DSE_GROUP_STRIDE = 8
VOC_LINEAR_EPOCHS = 1
```

Full raw/L2 structural validation configuration:

```python
CHECKPOINT_EPOCH_FILTER = None
NUM_DSE_IMAGES = 2048
NUM_VIS_IMAGES = 6
PATCH_DSE_GROUP_STRIDE = 1
RUN_VOC_EVAL = False
OUTPUT_RUN_SUFFIX = 'raw_l2'
```

This run skips VOC by default and reuses the existing base-run VOC JSON:

```text
to_epoch_XXXX/voc_all_checkpoints/voc_miou_results.json
```

The new structural validation output is:

```text
to_epoch_XXXX_raw_l2/
```

### Phase 2: Inspect Outputs

For a latest checkpoint `checkpoint0230.pth`, inspect:

```text
/content/drive/MyDrive/dino_dense_degradation_eval/to_epoch_0230_raw_l2/
```

Required files:

```text
patch_attention_dse_all_checkpoints/patch_attention_dse_summary.csv
patch_attention_dse_all_checkpoints/patch_attention_dse_summary.json
patch_attention_dse_all_checkpoints/fixed_images.json
patch_attention_dse_all_checkpoints/query_points.json
figures/combined_dense_summary.csv
figures/fig_dense_diagnostics_summary.png
figures/fig_raw_vs_l2_dse.png
figures/fig_raw_vs_l2_class_sep.png
figures/fig_raw_vs_l2_spectrum.png
summary_report.md
```

Interpret the run in this order:

1. VOC mIoU best vs final checkpoint.
2. Raw DSE vs L2 DSE.
3. Raw class separability vs L2 class separability.
4. Raw effective rank/top eigenvalue ratio vs L2 effective rank/top eigenvalue ratio.
5. Patch norm mean/p90 as the norm-drift confound.
6. DSE and class separability trend.
7. Effective rank and top eigenvalue ratio.
8. CLS-patch cosine trend.
9. Query similarity entropy and early-map correlation.
10. Fixed-image PCA, CLS attention, CLS similarity, and query similarity grids.

Decision logic:

```text
L2 DSE also declines:
    stronger evidence for angular patch-geometry degradation.

Raw DSE declines but L2 DSE does not:
    previous DSE signal was likely magnitude/norm drift artifact.

Raw effective rank rises but L2 effective rank falls:
    raw feature space is spreading while angular geometry may be concentrating.

Neither raw nor L2 metrics decline:
    current ImageNet-100 horizon does not show strong structural degradation.
```

### Phase 3: Global vs Dense Divergence

Add a global quality curve after Phase 1 outputs exist.

Preferred first implementation:

```text
ImageNet-100 CLS kNN accuracy across selected checkpoints.
```

Expected output:

```text
figures/fig_global_vs_dense.png
figures/global_dense_summary.csv
```

The core comparison should show:

```text
epoch -> ImageNet-100 CLS kNN
epoch -> VOC mIoU
epoch -> DSE
epoch -> CLS-patch cosine
epoch -> effective rank
```

### Phase 4: COCO-Stuff Selected Checkpoints

Run COCO-Stuff only after VOC and structural diagnostics are available.

Selected checkpoints:

```text
early checkpoint
VOC best checkpoint
mid checkpoint
latest checkpoint
one checkpoint after epoch 215
one checkpoint near the newest completed training horizon
```

Do not run COCO-Stuff over every checkpoint until selected checkpoints show a useful signal.

### Phase 5: Longer Training

Continue Kaggle training from the latest confirmed checkpoint.

Training constraints:

```text
--saveckp_freq 10
--keep_last_ckpts 0
```

Stop near Kaggle's 12-hour limit only after confirming a current epoch or checkpoint has safely completed. Start the next round only from a clearly verified newest checkpoint.

## Current File Responsibilities

```text
main_dino.py
    DINO pretraining entry point.

eval_voc_dense.py
    Frozen-backbone PASCAL VOC linear segmentation sweep.

dense_eval_utils.py
    Checkpoint discovery, internal epoch reading, and to_epoch_XXXX output naming.

dense_patch_diagnostics.py
    Pure patch diagnostic metric helpers.

analyze_patch_statistics.py
    Colab/offline checkpoint sweep for DSE, patch stats, attention stats, PCA maps, and query maps.

plot_dense_diagnostics.py
    Merge patch diagnostics with VOC mIoU and generate summary figures.

make_summary_report.py
    Generate Markdown reports for each evaluation run.

notebooks/colab_dense_degradation_all_checkpoints.ipynb
    Colab orchestration wrapper for Drive checkpoints and evaluation outputs.

PROJECT_STATUS.md
    Public-facing project status.

PLAN.md
    Current implementation plan.

DECISIONS.md
    Durable technical decisions.

TODO.md
    Task breakdown and progress.

AGENTS.md
    Long-term Codex rules for this repository.
```

## Verification Commands

Run before claiming repository changes are ready:

```bash
pytest -q
python -m py_compile dense_eval_utils.py dense_patch_diagnostics.py analyze_patch_statistics.py plot_dense_diagnostics.py make_summary_report.py dense_results_io.py
python -m json.tool notebooks/colab_dense_degradation_all_checkpoints.ipynb >/dev/null
git diff --check
```
