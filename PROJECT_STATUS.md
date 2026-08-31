# Project Status: DINO Dense Degradation

This document summarizes the current research and engineering status of this
repository. It is intended as a public-facing project status note for
contributors and future maintenance work.

## Objective

This project extends the original DINO training code to study dense
representation degradation during self-supervised Vision Transformer
pretraining. The working setup focuses on ViT-Small pretraining on ImageNet-100
and evaluates whether dense downstream performance can decline even when global
classification-oriented features continue to improve.

The main phenomenon under investigation is self-supervised dense degradation
(SDD): late-stage SSL training can improve global representations while making
patch-level representations less useful for dense prediction tasks such as
semantic segmentation.

## Current Status

### Kaggle Pretraining Workflow

The repository includes a Kaggle-oriented training workflow for running DINO
pretraining on dual T4 GPUs within the 12-hour Kaggle execution window.

Completed work:

- Added a resumable 12-hour `Save & Run All` workflow documented in
  `KAGGLE_GUIDE.md`.
- Configured training to write the latest checkpoint to
  `dino_output/checkpoint.pth` and periodic historical checkpoints such as
  `dino_output/checkpoint0010.pth`.
- Updated checkpoint loading for PyTorch 2.6+ compatibility where
  `torch.load` defaults to `weights_only=True`.
- Removed a distributed startup issue caused by evaluating `torch.hub.list`
  during argument parsing.
- Updated the Kaggle and Colab resume examples to use `--resume_from` directly,
  save periodic checkpoints every 10 epochs, and preserve all periodic
  checkpoints for later dense-degradation sweeps.

### Dense Evaluation Workflow

The repository includes standalone dense evaluation scripts for frozen-backbone
linear segmentation probing:

- `eval_voc_dense.py` runs PASCAL VOC linear probing across checkpoints.
- `eval_coco_stuff_dense.py` runs COCO-Stuff selected-checkpoint linear probing
  for validating whether the VOC trend appears on a denser segmentation
  benchmark.

The evaluation workflow:

- Downloads PASCAL VOC 2012 when needed; COCO-Stuff is provided as an external
  dataset path.
- Loads a series of DINO checkpoints.
- Freezes the DINO backbone.
- Trains a lightweight 1x1 convolutional segmentation head.
- Resets an explicit probe seed before every checkpoint head fit and records
  that seed in the output JSON.
- Loads an explicitly selected `teacher` or `student` checkpoint key and records
  that representation in each output row.
- Reports validation mIoU from a full-validation-set confusion matrix under
  `metric_version=global_confusion_v2`.
- Records structured checkpoint identity, probe configuration, dataset
  identity, and Git commit/dirty state in every result row; formal readers require
  `source_dirty=false`.
- Produces an `mIoU vs. epoch` plot for dense degradation analysis.

Formal VOC and COCO files are
`voc_miou_results_global_confusion_v2.json` and
`coco_stuff_miou_results_global_confusion_v2.json`. Comparisons and report
readers validate metric version, probe seed, checkpoint key, representation,
and provenance before combining rows. Historical batch-mean-v1 JSON remains
readable only through an explicit legacy mode and is never silently mixed with
v2.

`audit_training_schedule.py` separately audits checkpoint coordinates,
schedule identities, independent session logs, log coverage, and reconstructed
LR/weight-decay/teacher-momentum trajectories. Its verdict is `stitched`,
`continuous`, or `unknown`; only sufficient `continuous` evidence supports
interpreting the supplied checkpoints as one clean training horizon.

The verified offline archive currently contains 15 checkpoints spanning
experiment epochs 170 through 318. Their embedded arguments establish target
horizons of 200, 300, and 500 epochs, so the current verdict is `stitched`
with partial evidence. The checkpoint bytes and epoch coordinates are valid,
but the historical curve is exploratory rather than a clean-horizon
phenomenon result. Independent session logs are still needed to localize and
quantify the schedule boundaries.

The registered three-seed EMA-teacher VOC metric-v2 characterization is also
complete at experiment labels `{180, 250, 318}`. Mean mIoU is `38.6871`,
`37.6135`, and `37.2876`, respectively. The paired label-318 minus label-180
change is `-1.39945 +/- 0.01229` points (mean +/- sample SD, `n=3`), with all
three paired changes negative. All rows passed checkpoint, source, seed,
metric, representation, probe, and dataset identity checks. Because the
schedule audit is `stitched`, this is retained as stitched-run exploratory
characterization and is not a clean-horizon SDD verdict.

The clean single-horizon baseline V2 is registered before V2 training at source
`4c16679e915ca1e84842d652c911166f164b5183`. It fixes one 319-completed-epoch
schedule, backbone seed 0, labels `180 / 250 / 318`, T4 x2, effective batch
256, and a fail-closed cross-session resume contract with per-rank RNG state,
dynamic-loss-scaling state, and attempted/applied optimizer-step coordinates.
No V2 clean-horizon output has been accepted yet.

Kaggle Version 20 (`346119135`,
`clean-horizon-seed0-session1-v1`) started correctly but terminated at epoch
17, iteration 793 when one finite-loss accumulation group triggered a standard
`GradScaler` optimizer-step skip. V1 had registered every skip as terminal.
This is an excluded engineering failure, not scientific evidence, and its
checkpoint cannot resume V2. V2 retains every scientific hyperparameter but
recovers one or two consecutive scaler overflows while recording skipped and
applied updates; three consecutive overflows remain terminal.

Kaggle Version 21 (`clean-horizon-seed0-session1-v2`) has now started from
epoch 0 in the same Notebook. The submitted cells pin source
`4c16679e915ca1e84842d652c911166f164b5183`, keep resume empty, attach only
ImageNet100, and use GPU T4 x2. The initial activity status is running with no
immediate failure. This is not accepted evidence until its V2 summary and
rolling checkpoint pass independent inspection.

The repository also includes a Colab notebook for the current Drive-based
evaluation workflow:

- `notebooks/colab_dense_degradation_all_checkpoints.ipynb` mounts Google Drive
  and reads checkpoints from `MyDrive/dinocheckpoint`.
- The notebook automatically discovers all recognizable `checkpoint*.pth` files
  instead of requiring a hard-coded epoch list.
- Checkpoints are normalized into a temporary Colab runtime directory before
  evaluation so the existing evaluator can process them in epoch order.
- Persistent outputs are grouped by the largest discovered checkpoint epoch,
  for example `MyDrive/dino_dense_degradation_eval/to_epoch_0215/`, so separate
  training rounds do not overwrite one another.
- The notebook keeps the dense evaluation protocol aligned with the structural
  degradation paper: frozen ViT-S/16 backbone, projector removed, last-layer
  patch embeddings, and a lightweight linear segmentation head on PASCAL VOC.

### Internal Diagnostics

The repository includes `dense_diagnostics.py` for tracking representation
health during training.

Tracked diagnostics include:

- Effective rank of patch token representations.
- CLS-to-patch cosine similarity.
- Covariance condition number.
- Attention map visualizations for qualitative inspection.

These diagnostics are intended to be compared against dense evaluation results
to determine whether declining segmentation performance correlates with patch
feature collapse or CLS-patch homogenization.

The Colab dense evaluation notebook now delegates patch-level analysis to
standalone repository scripts so the workflow is easier to rerun and debug:

- `analyze_patch_statistics.py` scans checkpoints and computes paper-formula
  DSE class separability, effective rank, covariance spectrum, CLS-patch cosine,
  patch feature magnitude histograms, CLS attention statistics, fixed-query
  patch similarity maps, and deterministic fixed-basis PCA maps of patch
  features.
- `plot_dense_diagnostics.py` merges VOC mIoU with structural diagnostics and
  writes the summary figure after validating the requested metric version,
  probe seed, and checkpoint key.
- `make_summary_report.py` writes a compact Markdown report for each run with
  the VOC protocol identity recorded in the report.

The notebook still controls the Colab environment: it mounts Google Drive,
normalizes checkpoint filenames into a runtime directory, runs VOC linear
segmentation, runs the patch diagnostic suite, and saves persistent outputs
under a final-epoch directory such as
`MyDrive/dino_dense_degradation_eval/to_epoch_0215/`.

The diagnostic runner now records fixed visual images in `fixed_images.json`,
records named query patch coordinates in `query_points.json`, saves one shared
deterministic PCA basis for all fixed-image qualitative maps, and validates
checkpoint filename/internal epoch agreement before expensive feature
extraction.

### Repository Review Status

Current validation coverage:

- Unit tests cover checkpoint discovery, resume-path wiring, Colab notebook
  script references, shared CSV/JSON readers, DSE component calculations,
  fixed-query similarity metrics, and deterministic fixed-basis PCA.
- Syntax checks cover the Python entry points and notebook JSON structure.
- The public documentation avoids committing local checkpoints, datasets,
  generated figures, and reference PDFs.

Known limitations:

- `eval_voc_dense.py` is a lightweight frozen-backbone VOC linear-probing
  runner for trend analysis, not a full reproduction of every downstream
  setting in the SDD paper.
- Historical VOC rows use the batch-mean-v1 mIoU estimator and were generated
  before the evaluator recorded an explicit probe seed. They remain legacy
  evidence and are not mixed with the completed metric-v2 characterization.
- The formal metric-v2 GPU reruns are complete and their raw archives remain
  outside Git. Their repeatable decline applies to a stitched trajectory, not
  to one clean training horizon, so it cannot define an intervention fork.
- The offline schedule audit covers 15 verified checkpoints from labels 170
  through 318 and detects target-horizon changes from 200 to 300 to 500. Its
  verdict is `stitched` with partial evidence because independent session logs
  are not yet archived.
- COCO-Stuff selected-checkpoint evaluation is implemented but was not run:
  the registered protocol permits it only after VOC reaches a scientific
  verdict, which P0 `stitched` blocks. ADE20K and Cityscapes linear
  segmentation remain future work.
- Full Colab/Kaggle runs depend on external data, checkpoints, and GPU runtime,
  so local tests validate code paths and configuration wiring rather than
  reproducing a complete GPU sweep.

## Next Steps

Planned research and engineering work:

1. Retrieve every independent session log into source-specific directories
   and rerun the audit to localize LR/WD boundary behavior. Logs can strengthen
   the boundary evidence but cannot turn the already observed schedule
   identity changes into a clean continuous horizon.
2. Preserve the completed labels `180 / 250 / 318`, seeds
   `42 / 1337 / 2027`, and their predeclared paired changes as stitched-run
   exploratory evidence; do not reinterpret them as a clean-horizon gate.
3. Run the registered clean fixed-horizon phenomenon reproduction without
   changing its source, schedule, endpoint, probe protocol, or seed handling.
4. Run the conditional COCO-Stuff consistency probe only if that clean-horizon
   VOC gate reaches a scientific verdict.
5. Only if that clean-horizon phenomenon gate passes, freeze one intervention
   fork, equivalence margin, stopping rule, and kill criterion, then migrate
   CLS-CRR for a matched C0-versus-C1 run.
6. Defer a late KoLeo arm, additional fork points, and from-scratch
   confirmation until the predeclared first-stage result justifies the
   additional budget.

## Key Files

- `main_dino.py`: Core DINO pretraining entry point.
- `utils.py`: Checkpoint loading, distributed helpers, and shared utilities.
- `dense_diagnostics.py`: Dense representation diagnostics used during
  training.
- `dense_eval_utils.py`: Shared checkpoint discovery and run-output helpers.
- `dense_patch_diagnostics.py`: Shared patch-level metric helpers.
- `analyze_patch_statistics.py`: Offline/Colab patch diagnostic runner.
- `plot_dense_diagnostics.py`: Diagnostic plotting and VOC merge utility.
- `make_summary_report.py`: Markdown report generator for evaluation runs.
- `eval_voc_dense.py`: PASCAL VOC dense evaluation script.
- `eval_coco_stuff_dense.py`: COCO-Stuff selected-checkpoint dense evaluation
  script.
- `audit_training_schedule.py`: Read-only checkpoint/log schedule audit.
- `REVIEW_BASELINE_2026-07-26.md`: Pinned commits, local artifact hashes,
  coordinate contract, and missing external evidence inventory.
- `notebooks/colab_dense_degradation_all_checkpoints.ipynb`: Colab workflow for
  evaluating all Drive checkpoints and exporting mIoU, DSE metrics, and
  qualitative patch/attention diagnostics.
- `KAGGLE_GUIDE.md`: Kaggle training and resume workflow.
- `README.md`: Project overview, features, and quick-start instructions.

## Artifact Policy

Large generated artifacts are intentionally kept out of Git. This includes
checkpoints, local evaluation outputs, generated plots, and local dataset or
runtime outputs. The repository tracks code and documentation; experiment
artifacts should be stored in external storage or regenerated from the documented
workflows.
