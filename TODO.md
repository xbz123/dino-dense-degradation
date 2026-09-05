# TODO

This file tracks the current task breakdown for the DINO dense degradation
project. Keep tasks concrete and update checkboxes as work lands in the
repository or external experiment outputs are confirmed.

Updated 2026-09-05. Remote evidence cutoff: 2026-09-01. See
[execution snapshot](RUN_STATUS_2026-09-05.md). P0 clean-horizon continuation is
the critical path; the lower sections are capability/backlog references.

## Priority 0: Reproducibility And Phenomenon Gate

- [x] Add an explicit VOC `--probe_seed`, reset it before every checkpoint
  head fit, and record it in the result JSON.
- [x] Expose `VOC_PROBE_SEED` in the existing Colab and Kaggle notebooks.
- [x] Rerun historical stitched labels `180 / 250 / 318` with probe seeds
  `42 / 1337 / 2027`.
- [x] Report per-seed VOC rows, within-checkpoint mean/sample SD, and paired
  checkpoint changes.
- [x] Estimate a fixed post-peak slope or contrast without selecting the best
  checkpoint after viewing the reruns.
- [ ] Run COCO-Stuff only after a clean-horizon VOC gate reaches a scientific
  verdict; the current `stitched` audit blocks this step.
- [x] Freeze and implement the clean, matched, single-horizon baseline
  reproduction V2 at source `4c16679`; V1 was an excluded engineering failure.
- [ ] Run the clean-horizon baseline to label 318 and independently accept all
  session checkpoints and summaries.
- [ ] Download and independently validate V22 raw artifacts; refresh version
  history and GPU quota before continuing from completed epoch 33.
- [ ] After clean label 318 acceptance, run clean label-180 probes and freeze
  their own noise threshold before opening clean label-250/318 probe outputs.
- [ ] Freeze the first late-stage C0/C1 fork, endpoint, equivalence margin,
  stopping rule, and kill criterion.
- [ ] Do not add a late KoLeo arm or more fork points unless the first C1
  intervention passes its predeclared gate.

## Priority 1: Historical Structural Archive Review

Historical `to_epoch_0215` and `to_epoch_0318_raw_l2_full` output directories
exist through external-storage links. Inventory and review existing files
before considering any rerun. The items below do not declare those runs absent
or authorize a new GPU sweep; unchecked scientific interpretations need review.

- [x] Fix `DSE_IMAGE_ROOT` lookup so lowercase `imagenet100/train` is tried
  before uppercase `ImageNet100/train`.
- [x] Move patch/DSE/attention diagnostic code out of the notebook and into
  reusable repository scripts.
- [x] Generate per-run output directories with `to_epoch_XXXX` naming.
- [x] Add fixed-image PCA, CLS attention, CLS similarity, patch norm, and
  fixed-query patch similarity visualizations.
- [x] Add summary plotting and Markdown report generation scripts.
- [x] Add tests for checkpoint discovery, patch diagnostic helpers, notebook
  script wiring, and report/plot I/O.
- [x] Confirm current raw DSE/class separability used final-LayerNorm patch
  tokens but not L2-normalized patch features.
- [x] Add raw/L2 dual-track structural metrics:

  ```text
  raw_dse
  l2_dse
  raw_class_sep_avg
  l2_class_sep_avg
  raw_effective_rank
  l2_effective_rank
  raw_top1_eigen_ratio
  l2_top1_eigen_ratio
  patch_norm_mean
  patch_norm_p90
  ```

- [x] Add `to_epoch_XXXX_raw_l2` output naming for patch-only raw/L2 validation
  runs.
- [x] Update the Colab notebook to skip VOC by default and reuse the existing
  base-run VOC JSON.
- [x] Generate raw/L2 validation plots from summary CSV:

  ```text
  figures/fig_raw_vs_l2_dse.png
  figures/fig_raw_vs_l2_class_sep.png
  figures/fig_raw_vs_l2_spectrum.png
  ```

- [ ] Reconcile archived historical checkpoint/diagnostic coverage; keep clean
  V2 outputs in a separate run namespace.
- [ ] Only if required archived diagnostics are missing, use this smoke-test
  reference under a separately scoped execution request:

  ```python
  CHECKPOINT_EPOCH_FILTER = [215]
  NUM_DSE_IMAGES = 128
  NUM_VIS_IMAGES = 3
  PATCH_DSE_GROUP_STRIDE = 8
  RUN_VOC_EVAL = False
  OUTPUT_RUN_SUFFIX = 'raw_l2'
  ```

- [ ] Confirm smoke-test output exists:

  ```text
  selected_checkpoints.json
  patch_attention_dse_all_checkpoints/patch_attention_dse_summary.csv
  figures/fig_raw_vs_l2_dse.png
  figures/fig_raw_vs_l2_class_sep.png
  figures/fig_raw_vs_l2_spectrum.png
  figures/fig_dense_diagnostics_summary.png
  summary_report.md
  ```

- [ ] Confirm archived full raw/L2 structural coverage against this recipe
  before deciding whether missing outputs require a rerun:

  ```python
  CHECKPOINT_EPOCH_FILTER = None
  NUM_DSE_IMAGES = 2048
  NUM_VIS_IMAGES = 6
  PATCH_DSE_GROUP_STRIDE = 1
  RUN_VOC_EVAL = False
  OUTPUT_RUN_SUFFIX = 'raw_l2'
  ```

- [ ] Inspect and summarize the latest `summary_report.md`.
- [ ] Keep any historical best-versus-final display descriptive; use only the
  registered label-318 minus label-180 contrast for the formal gate.
- [ ] Compare raw vs L2 DSE, class separability, effective rank, and top
  eigenvalue ratio.
- [ ] Determine whether the raw DSE decline survives L2 normalization.
- [ ] Compare CLS-patch cosine and query similarity entropy across checkpoints.
- [ ] Select representative fixed-image qualitative grids for thesis notes.

## Priority 2: Global vs Dense Divergence

- [ ] Add a checkpoint-sweep script for ImageNet-100 CLS kNN evaluation.
- [ ] Save kNN results under the same `to_epoch_XXXX` output folder.
- [ ] Generate:

  ```text
  figures/fig_global_vs_dense.png
  figures/global_dense_summary.csv
  ```

- [ ] Compare:

  ```text
  ImageNet-100 CLS kNN
  VOC mIoU
  DSE
  effective rank
  CLS-patch cosine
  ```

- [ ] Update `summary_report.md` generation to include global vs dense
  divergence once kNN results are available.

## Priority 3: COCO-Stuff Selected Checkpoint Evaluation

Status: evaluator complete; formal run blocked until a clean-horizon VOC gate
reaches a scientific verdict.

- [ ] Use clean-baseline labels `180 / 318`, probe seeds `42 / 1337 / 2027`,
  and explicit teacher representation only after the registered VOC verdict.
  Do not select checkpoints from the observed VOC peak.

- [x] Implement or adapt a COCO-Stuff linear segmentation evaluator.
- [ ] Run COCO-Stuff only on selected checkpoints first.
- [ ] Save selected-checkpoint results under:

  ```text
  dino_dense_degradation_eval/to_epoch_XXXX/coco_stuff_selected/
  ```

- [ ] Compare COCO-Stuff mIoU with VOC mIoU and structural diagnostics.

## Priority 4: Clean Single-Horizon Training

- [x] Record Kaggle Version 20 as an excluded V1 engineering failure after one
  finite-loss `GradScaler` skip; it is not a scientific result or V2 parent.
- [x] Start the registered V2 seed-0 baseline from epoch 0 as Kaggle Version
  21; no historical or V1 checkpoint is its parent.
- [x] Accept the Version-21 runtime boundary at 18 completed epochs and submit
  Version 22 from its rolling checkpoint with only the resume path changed.
- [x] Inspect V22 successful runtime boundary remotely: completed epochs 33,
  attempts/applied 16302/16299, overflow total/consecutive 3/0.
- [x] Update draft Notebook parent input from V21 to V22 without changing the
  training contract. Record quota-blocked submission, not a launched V23.
- [ ] Independently archive/revalidate V22 and submit Session 3 after refreshing
  quota and checking for intervening runs. No active monitor remains.
- [ ] Preserve:

  ```text
  --epochs 319
  --saveckp_freq 10
  --keep_last_ckpts 0
  --milestone_ckpt_epochs 180 250 318
  --strict_resume_schedule true
  ```

- [ ] Before each new Kaggle round, verify the selected resume checkpoint:

  ```text
  filename epoch or rolling-checkpoint role
  checkpoint["epoch"] internal value
  file size
  structured training contract
  per-rank RNG state count
  output path and preceding Kaggle Version
  ```

- [ ] Near Kaggle's 12-hour limit, stop only after a current epoch or checkpoint
  is safely completed.
- [ ] After each round, confirm the newest checkpoint appears in output or Drive.
- [ ] Preserve each session summary and log without concatenating versions.
- [ ] At labels 180, 250, and 318, copy the accepted milestone checkpoint into
  the clean-baseline evaluation input.
- [ ] Run the registered three-probe-seed VOC gate only after label 318 is
  accepted.

## Priority 5: Stress Tests and Closer Paper Alignment

- [ ] Decide whether the next experiment should be longer ImageNet-100 training
  or a closer SDD-style dataset/evaluation setup.
- [ ] Consider longer ImageNet-100 training in update-step terms rather than
  epoch count alone.
- [ ] Consider selected COCO-Stuff or ADE20K before full multi-dataset sweeps.
- [ ] Keep stress-test experiments clearly labeled so they are not confused
  with baseline DINO reproduction runs.

## Documentation Tasks

- [x] Maintain public artifact policy in `PROJECT_STATUS.md`.
- [x] Add `PLAN.md`.
- [x] Add `DECISIONS.md`.
- [x] Add `TODO.md`.
- [x] Add `AGENTS.md`.
- [ ] After the next Colab run, update `PROJECT_STATUS.md` with the actual
  latest evaluated epoch and result interpretation.
- [ ] Add the final Colab output path and selected figures to the thesis notes
  outside this public repository.
