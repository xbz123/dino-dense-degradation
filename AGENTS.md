# Agent Rules for This Repository

This file gives long-term rules for Codex or other coding agents working on
this repository.

## Project Purpose

This repository studies dense representation degradation in DINO-style
self-supervised ViT training.

The current project objective is:

```text
Build a reliable dense patch representation diagnostic suite for DINO ViT-S/16
trained on ImageNet-100, then use it to monitor whether SDD-like degradation
appears as training continues.
```

Do not reframe the project as already proving or disproving SDD unless new
evaluation outputs support that claim.

## Communication Rules

- Use concise, concrete status updates.
- State what was verified and how.
- Do not claim a Colab/Kaggle result exists unless you inspected the output or
  the user supplied the logs/files.
- When a fact comes from an old run or memory, label it as old unless it was
  freshly verified.
- Prefer exact paths, exact commands, and exact output filenames.

## Repository Safety Rules

- Do not commit checkpoints, datasets, generated figures, local Drive exports,
  or paper PDFs.
- Treat `.pth` files as valuable experiment artifacts. Do not delete them unless
  the user explicitly asks.
- Keep generated outputs outside Git:

  ```text
  dino_dense_degradation_eval/
  dense_eval_results/
  dino_eval_checkpoints/
  dino_checkpoint*/
  dino_checkpoints*/
  dinocehckpoint/
  ```

- Before staging, use explicit file paths. Do not use broad `git add -A` if
  large artifacts could be present.

## Branch and Commit Rules

- The user has previously preferred direct updates on `main` for this repo, but
  do not push unless the user asks for a commit/push or the current task clearly
  includes publishing.
- Before committing, run the verification commands in this file.
- Commit messages should be short and specific, for example:

  ```text
  Add dense patch diagnostic suite
  Document evaluation plan
  Fix checkpoint discovery for generic checkpoint files
  ```

## Verification Rules

Run these before claiming local code/docs changes are ready:

```bash
pytest -q
python -m py_compile dense_eval_utils.py dense_patch_diagnostics.py analyze_patch_statistics.py plot_dense_diagnostics.py make_summary_report.py dense_results_io.py
python -m json.tool notebooks/colab_dense_degradation_all_checkpoints.ipynb >/dev/null
git diff --check
```

If only Markdown files changed, this lighter check is acceptable:

```bash
python -m json.tool notebooks/colab_dense_degradation_all_checkpoints.ipynb >/dev/null
git diff --check
git status --short
```

Report any skipped tests and why.

## Training Rules

Use Kaggle for continued DINO pretraining unless the user explicitly switches
platforms.

Preserve these arguments for checkpoint sweeps:

```text
--saveckp_freq 10
--keep_last_ckpts 0
```

When resuming training:

- Prefer an explicit `--resume_from` path.
- Verify checkpoint filename epoch and internal `checkpoint["epoch"]`.
- Use the newest clearly completed checkpoint only.
- Do not guess a checkpoint path.
- Near Kaggle's 12-hour limit, stop after a safe epoch/checkpoint boundary.

## Evaluation Rules

Use Colab for Drive-based evaluation unless the user asks otherwise.

Main notebook:

```text
notebooks/colab_dense_degradation_all_checkpoints.ipynb
```

Default Drive folders:

```text
/content/drive/MyDrive/dinocheckpoint
/content/drive/MyDrive/imagenet100/train
/content/drive/MyDrive/dino_dense_degradation_eval
```

Expected output pattern:

```text
/content/drive/MyDrive/dino_dense_degradation_eval/to_epoch_XXXX/
```

Raw/L2 structural validation output pattern:

```text
/content/drive/MyDrive/dino_dense_degradation_eval/to_epoch_XXXX_raw_l2/
```

Run a raw/L2 smoke test before a full expensive sweep:

```python
CHECKPOINT_EPOCH_FILTER = [215]
NUM_DSE_IMAGES = 128
NUM_VIS_IMAGES = 3
PATCH_DSE_GROUP_STRIDE = 8
RUN_VOC_EVAL = False
OUTPUT_RUN_SUFFIX = 'raw_l2'
```

Then run full raw/L2 structural validation:

```python
CHECKPOINT_EPOCH_FILTER = None
NUM_DSE_IMAGES = 2048
NUM_VIS_IMAGES = 6
PATCH_DSE_GROUP_STRIDE = 1
RUN_VOC_EVAL = False
OUTPUT_RUN_SUFFIX = 'raw_l2'
```

Use the existing base-run VOC JSON unless new checkpoints require a new VOC
sweep:

```text
to_epoch_XXXX/voc_all_checkpoints/voc_miou_results.json
```

## Interpretation Rules

Use conservative language.

Allowed:

```text
VOC mIoU does not show a clear downstream degradation drop in this proxy setup.
Raw structural diagnostics show a warning signal before a visible VOC drop.
L2-normalized diagnostics are needed to confirm angular patch-geometry degradation.
The current setup is not equivalent to the full SDD reference setting.
```

Avoid:

```text
SDD is absent.
The paper is wrong.
The reproduction failed.
VOC alone proves the result.
```

When interpreting outputs, inspect in this order:

1. `summary_report.md`
2. `figures/combined_dense_summary.csv`
3. `figures/fig_dense_diagnostics_summary.png`
4. `figures/fig_raw_vs_l2_dse.png`
5. `figures/fig_raw_vs_l2_class_sep.png`
6. `figures/fig_raw_vs_l2_spectrum.png`
7. `voc_all_checkpoints/voc_miou_results.json` or the reused base-run VOC JSON
8. `patch_attention_dse_all_checkpoints/patch_attention_dse_summary.csv`
9. Fixed-image qualitative figures under `patch_attention_dse_all_checkpoints/epoch_XXXX/`

For DSE/class separability/covariance conclusions:

- Treat unprefixed `dse`, `class_sep_avg`, `effective_rank`, and
  `top1_eigen_ratio` as backward-compatible raw-feature metrics.
- Prefer `raw_*` vs `l2_*` comparisons for current interpretation.
- Do not claim dense angular degradation unless the L2-normalized structural
  track supports it.
- Remember that final LayerNorm is not L2 normalization.

## Code Style Rules

- Prefer small, focused helpers with unit tests.
- Keep notebook cells thin; put reusable logic in `.py` scripts.
- Keep Drive paths configurable in the notebook.
- Use deterministic seeds for fixed images, query points, and PCA basis.
- Preserve teacher/student checkpoint-key configurability, with `teacher` as the
  default for main results.
- Use structured JSON/CSV outputs for metrics; avoid parsing logs when a direct
  output file can be written.

## Important Files

```text
PLAN.md
    Current implementation plan.

DECISIONS.md
    Durable technical decisions and rationale.

TODO.md
    Current task breakdown.

PROJECT_STATUS.md
    Public project status and limitations.

README.md
    Public quick-start instructions.

notebooks/colab_dense_degradation_all_checkpoints.ipynb
    Colab orchestration notebook.

analyze_patch_statistics.py
    Patch/DSE/attention/PCA/query diagnostic sweep.

dense_patch_diagnostics.py
    Pure metric helpers.

dense_eval_utils.py
    Checkpoint discovery and output naming.

eval_voc_dense.py
    PASCAL VOC linear segmentation evaluator.
```
