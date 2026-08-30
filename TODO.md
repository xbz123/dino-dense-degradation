# TODO

This file tracks the current task breakdown for the DINO dense degradation
project. Keep tasks concrete and update checkboxes as work lands in the
repository or external experiment outputs are confirmed.

## Priority 0: Reproducibility And Phenomenon Gate

- [x] Add an explicit VOC `--probe_seed`, reset it before every checkpoint
  head fit, and record it in the result JSON.
- [x] Expose `VOC_PROBE_SEED` in the existing Colab and Kaggle notebooks.
- [x] Rerun epochs `180 / 250 / 318` with probe seeds
  `42 / 1337 / 2027`.
- [x] Report per-seed VOC rows, within-checkpoint mean/sample SD, and paired
  checkpoint changes.
- [x] Estimate a fixed post-peak slope or contrast without selecting the best
  checkpoint after viewing the reruns.
- [ ] Run COCO-Stuff only after a clean-horizon VOC gate reaches a scientific
  verdict; the current `stitched` audit blocks this step.
- [x] Freeze and implement the clean, matched, single-horizon baseline
  reproduction at source `7404e7f`.
- [ ] Run the clean-horizon baseline to label 318 and independently accept all
  session checkpoints and summaries.
- [ ] Freeze the first late-stage C0/C1 fork, endpoint, equivalence margin,
  stopping rule, and kill criterion.
- [ ] Do not add a late KoLeo arm or more fork points unless the first C1
  intervention passes its predeclared gate.

## Priority 1: Colab Evaluation From Current Checkpoints

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

- [ ] Copy or sync the latest Kaggle checkpoints after epoch 215 into
  `MyDrive/dinocheckpoint`.
- [ ] Run a Colab raw/L2 smoke test on one checkpoint:

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

- [ ] Run the full Colab raw/L2 structural validation sweep:

  ```python
  CHECKPOINT_EPOCH_FILTER = None
  NUM_DSE_IMAGES = 2048
  NUM_VIS_IMAGES = 6
  PATCH_DSE_GROUP_STRIDE = 1
  RUN_VOC_EVAL = False
  OUTPUT_RUN_SUFFIX = 'raw_l2'
  ```

- [ ] Inspect and summarize the latest `summary_report.md`.
- [ ] Compare best VOC mIoU vs final VOC mIoU.
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

- [ ] Choose 4-6 checkpoints after the latest VOC/DSE sweep.
- [ ] Use this selection rule:

  ```text
  earliest checkpoint
  VOC best checkpoint
  middle checkpoint
  checkpoint 215
  newest checkpoint after 215
  latest available checkpoint
  ```

- [x] Implement or adapt a COCO-Stuff linear segmentation evaluator.
- [ ] Run COCO-Stuff only on selected checkpoints first.
- [ ] Save selected-checkpoint results under:

  ```text
  dino_dense_degradation_eval/to_epoch_XXXX/coco_stuff_selected/
  ```

- [ ] Compare COCO-Stuff mIoU with VOC mIoU and structural diagnostics.

## Priority 4: Clean Single-Horizon Training

- [ ] Start the registered seed-0 baseline from epoch 0; never use a historical
  stitched checkpoint as its parent.
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
