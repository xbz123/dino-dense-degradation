# Technical Decisions

This file records decisions that should remain stable across future development
unless a new experiment clearly invalidates them.

## Research Framing

### Decision: Treat this as a diagnostic suite, not a guaranteed SDD reproduction

The current project goal is to build a reliable dense patch representation
diagnostic suite for DINO ViT-S/16 training dynamics.

Rationale:

- The current pretraining setup uses ImageNet-100, not ImageNet-1K.
- Current training has reached the 200+ epoch range, while the SDD paper's DINO
  ViT-S/16 reference setting trains much longer.
- PASCAL VOC linear segmentation is useful but may be too coarse to reveal
  early structural degradation.
- Structural metrics and qualitative maps can show representation drift before
  VOC mIoU visibly drops.

Accepted wording:

```text
The historical ImageNet-100 + DINO ViT-S/16 batch-mean-v1 VOC curve contains
an apparent downstream drop, but no metric-v2 degradation window is confirmed
yet. This motivates fixed-seed v2 reruns, structural diagnostics, and only then
a late-stage intervention.
```

Avoid this wording:

```text
DINO ViT-S/16 does not have SDD.
We failed to reproduce SDD.
VOC proves degradation is absent.
```

## Training

### Decision: Keep every 10-epoch checkpoint

Use:

```text
--saveckp_freq 10
--keep_last_ckpts 0
```

Rationale:

- Dense degradation analysis needs historical checkpoints.
- Keeping only the latest checkpoint prevents later curve reconstruction.
- External storage is cheaper than rerunning long training segments.

### Decision: Resume from explicit checkpoint paths

Use `--resume_from` with a verified checkpoint path rather than relying on
generic discovery when restarting Kaggle runs.

Rationale:

- Previous failures came from accidentally selecting the wrong checkpoint.
- Explicit paths make the next training round auditable.
- `checkpoint.pth` stores the next starting epoch internally, so filename and
  internal epoch should both be checked.

### Decision: Use Kaggle for long training, Colab for evaluation

Rationale:

- Kaggle T4 x2 is suitable for continued training in 12-hour rounds.
- Colab is easier for Drive-mounted evaluation, plotting, and report generation.
- Evaluation outputs can be grouped by latest checkpoint epoch in Drive.

## Evaluation

### Decision: Primary downstream metric is PASCAL VOC frozen-backbone linear segmentation

The current evaluator is:

```text
eval_voc_dense.py
```

Protocol:

- Frozen DINO ViT-S/16 backbone.
- Projector removed.
- Last-layer patch embeddings.
- Lightweight 1x1 segmentation head.
- Validation mIoU curve across checkpoints.

Rationale:

- This is the current implemented downstream dense metric.
- It is fast enough to run across many checkpoints.
- It provides a stable baseline for comparing structural diagnostics.

### Decision: Formal dense metrics use global-confusion v2

Formal VOC and COCO results use:

```text
metric_version = global_confusion_v2
voc_miou_results_global_confusion_v2.json
coco_stuff_miou_results_global_confusion_v2.json
```

The evaluator accumulates one confusion matrix over the complete validation
set before computing per-class IoU and mIoU. Every formal row records
`metric_version`, `probe_seed`, `checkpoint_key`, `representation`, checkpoint
path and structured identity, probe configuration, dataset identity, and Git
commit/dirty state. Formal readers require `source_dirty=false`.

Historical `voc_miou_results.json` rows use batch-mean-v1. They remain
readable only through an explicit legacy mode and are not eligible for v2
plots, comparison tables, COCO/VOC joins, or the phenomenon gate. Formal
readers fail closed unless protocol and provenance fields are complete and
homogeneous.

Rationale:

- Averaging per-batch mIoU weights class presence by batch and is not the
  standard whole-validation estimator.
- Explicit provenance prevents a result from being detached from its
  checkpoint, code, data, or probe recipe.
- Teacher and student representations are separate evidence and cannot be
  substituted after viewing results.

### Decision: VOC probe randomness is explicit and matched across checkpoints

Default:

```text
--probe_seed 42
--checkpoint_key teacher
```

`eval_voc_dense.py` resets Python, NumPy, PyTorch, and CUDA RNG state before
every checkpoint's linear-head initialization. It records `probe_seed` and the
explicit checkpoint representation in each output row.

Rationale:

- The previous `torch.randperm` path had no explicit seed, so the same
  checkpoint could change across reruns.
- Resetting per checkpoint gives the sweep common random numbers and prevents
  checkpoint discovery order from changing later head initializations.
- Probe-seed repeats quantify head-fitting noise; they do not replace
  independent backbone-training seeds.
- Exact bitwise CUDA reproducibility is not promised for kernels that are
  nondeterministic on the selected runtime.

### Decision: Audit the training horizon independently of downstream metrics

`audit_training_schedule.py` consumes original checkpoints and independent
session logs and returns `stitched`, `continuous`, or `unknown`. Periodic
filename/log epochs are zero-based; checkpoint `epoch` counts completed epochs.
Teacher momentum is reconstructed because it is not logged.

Only sufficient `continuous` evidence reaches the clean-horizon phenomenon
gate. `stitched` remains exploratory, while `unknown` blocks a verdict.

Rationale:

- `main_dino.py` rebuilds schedules from launch arguments, so a resumed run can
  change schedule identity even when weights load successfully.
- Missing logs or horizon checkpoints are missing evidence, not evidence of
  continuity.
- The schedule audit and metric-v2 rerun are independent; neither is skipped
  because of the other's outcome.

### Decision: Treat the verified epoch-170-to-318 archive as stitched

The offline archive contains 15 byte-verified checkpoints with consistent
epoch coordinates. Embedded training arguments separate them into target
horizons of 200, 300, and 500 epochs. The schedule audit therefore returns
`stitched` with partial evidence.

Consequences:

- Historical VOC and structural curves remain useful exploratory evidence.
- Selected-checkpoint metric-v2 repeats still run to quantify probe noise and
  endpoint behavior, but they cannot establish a clean-horizon phenomenon.
- Independent session logs remain required to localize LR/WD boundary
  behavior, not to reverse the stitched classification.
- A mitigation experiment requires a predeclared clean fixed-horizon
  phenomenon reproduction first.

### Decision: Establish the late-stage phenomenon before migrating mitigation

The first gate uses fixed epochs `180`, `250`, and `318`, probe seeds `42`,
`1337`, and `2027`, a predeclared post-peak trend, and selected-checkpoint
COCO-Stuff.

Rationale:

- A post-hoc peak-versus-final difference is biased toward the selected peak
  and has no meaning until probe variance is bounded.
- Fork-point selection must follow the noise/trend review, not the most
  favorable intervention result.
- The first late-stage mitigation should migrate only CLS-CRR and compare one
  matched C0/C1 fork. A KoLeo arm is useful for specificity but can wait until
  C1 efficacy is established.
- The equivalence margin, stopping rule, and kill criterion must be frozen
  before the intervention run.

### Decision: Run the clean baseline as one 319-completed-epoch contract

The clean baseline uses source
`4c16679e915ca1e84842d652c911166f164b5183`, Kaggle T4 x2, backbone seed 0,
and one target of 319 completed epochs. Experiment labels remain zero-based,
so formal labels `180 / 250 / 318` correspond to internal completed epochs
`181 / 251 / 319`.

Every session uses the same training contract and restores model, optimizer,
scaler, DINO center, and per-rank RNG state. The ImageNet-100 loader exposes
989 micro-batches per epoch; the last incomplete accumulation group is
discarded so all 494 scheduled optimizer-step attempts use effective batch 256.
An isolated dynamic-loss-scaling overflow consumes its schedule slot, skips the
student update and teacher EMA, and is fully recorded; the batch-tied DINO
center update remains. Three consecutive overflows, rank-inconsistent overflow
decisions, non-finite loss, contract mismatch, or incomplete checkpoint
invalidate the session. Attempted and applied update coordinates are persisted
and checked at resume. The full gate is frozen in
`CLEAN_HORIZON_BASELINE_PROTOCOL_2026-08-30.md`.

V1 source `7404e7fcddaa3702574697aa4fa7aa2bb3d1e8b3` treated any
`GradScaler` skip as terminal and failed on one finite-loss overflow at epoch
17, iteration 793. That engineering failure is excluded and cannot seed V2;
V2 restarts from epoch 0 with all scientific hyperparameters unchanged.

### Decision: DSE and patch diagnostics are required companion evidence

The following diagnostics should accompany VOC mIoU:

```text
DSE
class separability
effective rank
covariance spectrum
top eigenvalue ratio
CLS-patch cosine similarity
patch feature norm histograms
CLS attention entropy/concentration
fixed-query patch similarity entropy/correlation
fixed-image PCA maps
CLS attention maps
patch similarity maps
```

Rationale:

- VOC mIoU alone may not capture early dense representation drift.
- DSE-style metrics align with the structural degradation paper.
- CLS-patch cosine and fixed-query similarity maps align with DINOv3-style
  locality erosion analysis.

### Decision: Report raw and L2-normalized structural diagnostics separately

Current raw DSE and covariance metrics are computed from final-LayerNorm patch
tokens, but final LayerNorm does not force unit L2 norm. Because the observed
patch norm changed substantially over training, raw Euclidean/covariance
metrics can be confounded by feature magnitude drift.

Required output columns:

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

Rationale:

- Raw metrics remain useful as a warning signal.
- L2 metrics are the stronger check for angular/cosine patch-geometry
  degradation.
- Effective rank and top-1 eigen ratio are invariant to uniform scaling, but
  can still change under non-uniform norm drift across samples, patches, or
  feature dimensions.
- Spectrum flattening cannot be explained by uniform norm scaling alone, but
  L2-normalized spectrum metrics are needed to separate angular diversification
  from magnitude redistribution.

Accepted wording:

```text
Raw structural metrics show a warning signal, but normalized diagnostics are
required to determine whether this reflects angular dense degradation or
feature magnitude drift.
```

### Decision: Use teacher backbone for main diagnostics

Default:

```text
CHECKPOINT_KEY = 'teacher'
--checkpoint_key teacher
```

Rationale:

- Teacher weights are EMA-smoothed and usually used as the stable DINO
  representation.
- The evaluation should use the same network key across VOC and patch
  diagnostics.
- Formal evaluation never falls back to another checkpoint key when the
  requested representation is missing.

### Decision: Use fixed image/query/PCA basis for qualitative comparisons

Qualitative visualizations must be comparable across checkpoints.

Rules:

- Use the same fixed images across checkpoints.
- Use named fixed query patch coordinates.
- Use one deterministic PCA basis for PCA feature maps.
- Use consistent image size and patch grid.

Rationale:

- Changing images or PCA bases can create visual differences that are not caused
  by model training.
- Fixed query maps make patch-structure drift easier to inspect.

## Drive Layout

### Decision: Checkpoints live in `MyDrive/dinocheckpoint`

Expected path:

```text
/content/drive/MyDrive/dinocheckpoint/
```

Recognized files include:

```text
checkpoint03.pth
checkpoint0020.pth
checkpoint0215.pth
checkpoint.pth
```

Epoch-named files are preferred over generic `checkpoint.pth` if both resolve to
the same epoch.

### Decision: Evaluation outputs are grouped by latest checkpoint epoch

Expected output:

```text
/content/drive/MyDrive/dino_dense_degradation_eval/to_epoch_XXXX/
```

Raw/L2 structural validation output:

```text
/content/drive/MyDrive/dino_dense_degradation_eval/to_epoch_XXXX_raw_l2/
```

Rationale:

- Prevents one evaluation run from overwriting another.
- Makes it obvious which training horizon the result represents.
- Keeps baseline VOC/full-run outputs separate from patch-only raw/L2 validation
  reruns.

### Decision: ImageNet-100 Drive path uses lowercase first

The notebook checks:

```text
/content/drive/MyDrive/imagenet100/train
/content/drive/MyDrive/ImageNet100/train
```

Rationale:

- The active Drive layout used lowercase `imagenet100`.
- Keeping the uppercase fallback avoids breaking older layouts.

## Repository Hygiene

### Decision: Do not commit checkpoints, datasets, papers, or generated outputs

The repository tracks code and documentation only.

Do not commit:

```text
*.pth
dinocehckpoint/
external/
dino_checkpoint*/
dino_checkpoints*/
dino_eval_checkpoints/
dense_eval_results/
dino_dense_degradation_eval/
to_epoch_*/
Exploring Structural Degradation in Dense Representations for Self-supervised Learning .pdf
```

Rationale:

- Checkpoints and datasets are large generated artifacts.
- Papers may contain copyrighted material.
- Generated figures and reports should be reproducible from code and external
  artifacts.

## Testing

### Decision: Local tests validate wiring and pure logic, not full GPU sweeps

Run locally:

```bash
pytest -q
python -m py_compile audit_training_schedule.py dense_eval_utils.py dense_patch_diagnostics.py analyze_patch_statistics.py plot_dense_diagnostics.py make_summary_report.py dense_results_io.py eval_voc_dense.py eval_coco_stuff_dense.py
python -m json.tool notebooks/colab_dense_degradation_all_checkpoints.ipynb >/dev/null
python -m json.tool notebooks/kaggle_raw_l2_dense_eval.ipynb >/dev/null
git diff --check
```

Rationale:

- Full GPU evaluation depends on external checkpoints, Drive folders, and
  datasets.
- Unit tests should still catch checkpoint parsing, notebook wiring, metric
  helper behavior, and report/plot integration mistakes.
