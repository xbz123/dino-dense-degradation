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

The repository includes `eval_voc_dense.py`, a standalone dense evaluation
script for PASCAL VOC semantic segmentation linear probing.

The evaluation workflow:

- Downloads PASCAL VOC 2012 when needed.
- Loads a series of DINO checkpoints.
- Freezes the DINO backbone.
- Trains a lightweight 1x1 convolutional segmentation head.
- Reports validation mIoU for each checkpoint.
- Produces an `mIoU vs. epoch` plot for dense degradation analysis.

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
  writes the summary figure.
- `make_summary_report.py` writes a compact Markdown report for each run.

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
- COCO-Stuff, ADE20K, and Cityscapes linear segmentation are still future work;
  current downstream evidence is centered on PASCAL VOC plus structural
  diagnostics.
- Full Colab/Kaggle runs depend on external data, checkpoints, and GPU runtime,
  so local tests validate code paths and configuration wiring rather than
  reproducing a complete GPU sweep.

## Next Steps

Planned research and engineering work:

1. Run dense evaluation across checkpoint snapshots and inspect the resulting
   mIoU curve.
2. Compare dense evaluation results with logged diagnostics such as effective
   rank and CLS-patch cosine similarity.
3. Compare VOC mIoU with the Colab notebook's DSE class separability,
   effective-rank, patch-statistic, and CLS-attention outputs.
4. Add COCO-Stuff linear segmentation on selected checkpoints if VOC remains
   stable but structural diagnostics drift.
5. Continue pretraining toward the target training horizon where needed.
6. Use the confirmed degradation pattern to evaluate mitigation strategies,
   such as dense contrastive objectives or architectural changes that preserve
   local representations.

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
