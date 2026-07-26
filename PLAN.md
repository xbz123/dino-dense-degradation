# Dense Degradation Diagnostic Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` for independent implementation tasks, or `superpowers:executing-plans` for inline task execution. Track progress with checklist syntax and verify each step before claiming completion.

**Goal:** Build a reliable dense patch representation diagnostic suite for DINO ViT-S/16 checkpoints trained on ImageNet-100, so the project can evaluate whether SDD-like degradation appears in downstream and structural metrics as training continues beyond epoch 215.

**Architecture:** Kaggle is used for long-running DINO pretraining and checkpoint production. Colab is used for Drive-based checkpoint sweeps: PASCAL VOC linear segmentation, patch/DSE/attention diagnostics, fixed-image qualitative visualizations, summary plots, and Markdown reports. Repository code stays lightweight and reproducible; generated checkpoints, datasets, figures, and papers stay outside Git.

**Tech Stack:** PyTorch, torchvision, DINO ViT-S/16, Kaggle GPU T4 x2, Google Colab, Google Drive, PASCAL VOC, ImageNet-100, Matplotlib, pytest.

---

## Current Scope

The project is currently positioned as a diagnostic suite, not as a claim that the reduced ImageNet-100 setup has already reproduced the full SDD paper degradation curve.

The working claim is:

> The existing ImageNet-100 + DINO ViT-S/16 VOC sweep contains an apparent
> peak-to-final decline, but those historical rows use the batch-mean-v1 mIoU
> estimator and did not record or reset a probe seed. Until
> `global_confusion_v2` fixed-checkpoint probes bound that noise, the curve is
> a phenomenon signal rather than confirmed SDD evidence.

Current correction:

> The previous raw-feature DSE/class-separability trend is a warning signal, not final evidence of angular structural degradation. Patch feature norms changed strongly during training, so the next validation run must compare raw final-LayerNorm patch tokens against L2-normalized patch tokens.

## Immediate Scientific Gate

Complete these steps before implementing a late-stage mitigation:

1. Retrieve and hash the required checkpoints and every independent training
   session `log.txt`, then run `audit_training_schedule.py`. Only
   `continuous` reaches the clean-horizon scientific gate; `stitched` remains
   exploratory and `unknown` blocks a verdict.
2. Use explicit probe seeds `42`, `1337`, and `2027`. The evaluator resets the
   selected seed before each checkpoint so head initialization and minibatch
   order are matched across the curve. Use `--checkpoint_key teacher`
   explicitly for both VOC and COCO.
3. Rerun VOC at fixed epochs `180`, `250`, and `318`; report every seed, mean,
   sample standard deviation, and paired checkpoint changes from
   `voc_miou_results_global_confusion_v2.json`.
   Accept only rows that record checkpoint SHA256, probe configuration,
   dataset identity, and a Git commit with `source_dirty=false`.
4. Estimate a predeclared post-peak trend instead of selecting the maximum
   after viewing the curve.
5. Run the existing COCO-Stuff evaluator on the same selected checkpoints.
6. Freeze the intervention fork, primary contrast, equivalence margin, stop
   rule, and kill criterion from those measurements.

If the gate passes, the first late-stage experiment uses one fork and only two
matched arms: C0 continuation and CLS-CRR continuation. Epoch 150 or 180 is the
initial prevention candidate; epoch 220 is a later rescue tier. A KoLeo arm is
deferred until C1 shows a positive late-stage effect, which limits the first
migration to one regularizer.

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

This run skips VOC by default and may reuse only a base-run VOC JSON whose v2
metric, probe seed, and checkpoint key match the requested report:

```text
to_epoch_XXXX/voc_all_checkpoints/voc_miou_results_global_confusion_v2.json
```

The historical `voc_miou_results.json` file is batch-mean-v1 evidence. It is
never auto-discovered or mixed into a v2 plot, summary, COCO comparison, or
phenomenon gate. Formal plotting/report commands pass
`--voc_protocol v2 --voc_metric_version global_confusion_v2`,
`--voc_probe_seed`, and `--voc_checkpoint_key` explicitly.

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

1. Paired VOC v2 mIoU changes across the predeclared checkpoints, grouped by
   probe seed and fixed to the teacher representation.
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

Run COCO-Stuff only after matching `global_confusion_v2` VOC and structural
diagnostics are available. Use the same explicit probe seed and checkpoint key
for both datasets; comparison must fail closed on any mismatch.

Formal consistency checkpoints:

```text
180
318
```

Use paired seeds `{42, 1337, 2027}`. A wider selected-checkpoint curve may be
run later as exploratory context, but it cannot replace the registered
180-versus-318 comparison.

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
    Provenance-bound metric-v2 PASCAL VOC linear segmentation sweep.

eval_coco_stuff_dense.py
    Provenance-bound selected-checkpoint COCO-Stuff metric-v2 sweep.

audit_training_schedule.py
    Read-only schedule identity, session-boundary, and log-coverage audit.

dense_eval_utils.py
    Checkpoint discovery, internal epoch reading, and to_epoch_XXXX output naming.

dense_patch_diagnostics.py
    Pure patch diagnostic metric helpers.

analyze_patch_statistics.py
    Colab/offline checkpoint sweep for DSE, patch stats, attention stats, PCA maps, and query maps.

plot_dense_diagnostics.py
    Merge patch diagnostics with explicitly validated VOC v2 rows and generate
    summary figures; historical input requires an explicit legacy mode.

make_summary_report.py
    Generate Markdown reports with explicit VOC protocol identity.

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
python -m py_compile audit_training_schedule.py dense_eval_utils.py dense_patch_diagnostics.py analyze_patch_statistics.py plot_dense_diagnostics.py make_summary_report.py dense_results_io.py eval_voc_dense.py eval_coco_stuff_dense.py
python -m json.tool notebooks/colab_dense_degradation_all_checkpoints.ipynb >/dev/null
python -m json.tool notebooks/kaggle_raw_l2_dense_eval.ipynb >/dev/null
git diff --check
```
