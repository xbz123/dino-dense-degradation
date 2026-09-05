# Clean Single-Horizon Baseline Protocol

Status: `v2_registered_before_first_v2_training_output`

Registration date: 2026-08-30

V2 amendment date: 2026-08-31

Frozen training implementation:
`4c16679e915ca1e84842d652c911166f164b5183`

V2 supersedes V1 only for dynamic-loss-scaling recovery. V1 failed before a
complete epoch-17 checkpoint when one finite-loss accumulation group triggered
a standard `GradScaler` optimizer-step skip. V1 had predeclared every such skip
terminal, so that run is excluded and cannot be resumed. No downstream output
was opened before this amendment.

## Objective And Evidence Level

This experiment asks whether the current ImageNet-100 DINO ViT-S/16 recipe
shows a late dense-transfer decline when the complete optimizer, weight-decay,
teacher-momentum, and temperature schedules are defined by one target horizon.

The first run uses one backbone seed (`0`). It is a clean-horizon phenomenon
baseline, not a multi-backbone-seed or paper-scale confirmation. Training loss,
online diagnostics, and structural proxies cannot replace the frozen
downstream endpoint.

## Frozen Training Contract

- platform: Kaggle `GPU T4 x2`, NCCL, world size `2`;
- dataset: ImageNet-100, exactly `100` classes; the launcher records the
  discovered image count and refuses a resume if it changes;
- model: `vit_small`, patch size `16`, output dimension `65536`;
- backbone seed: `0` (`main_dino --seed 0`);
- total completed epochs: `319`, giving zero-based experiment labels
  `0..318`;
- formal checkpoint labels: `180 / 250 / 318`, whose internal completed-epoch
  coordinates are `181 / 251 / 319`;
- optimizer: AdamW, base LR `0.0005`, minimum LR `0.000001`, warmup `10`;
- weight decay: `0.04 -> 0.4`;
- teacher momentum: `0.996 -> 1.0`;
- teacher temperature: `0.04 -> 0.07`, warmup `30` epochs;
- per-GPU batch `64`, accumulation `2`, effective batch `256`;
- the raw loader has `989` micro-batches per epoch in the current dataset;
  the final incomplete accumulation group is discarded, so every epoch has
  exactly `988` used micro-batches and `494` scheduled optimizer-step attempts;
- local crops `4`, `norm_last_layer=false`, FP16 enabled with dynamic loss
  scaling;
- a recovered AMP overflow consumes its scheduled slot, skips the student
  optimizer update and corresponding teacher EMA update, and is recorded
  separately from applied updates. The DINO center remains batch-tied and is
  not rolled back;
- one or two consecutive AMP overflows are recoverable. Three consecutive
  overflows terminate and invalidate the session;
- periodic checkpoints every `10` zero-based labels, plus forced labels
  `180 / 250 / 318`;
- diagnostics every `5` labels and attention exports every `25` labels.

Changing any item above creates a new experiment and cannot resume this one.

## Multi-Session Continuity

Every Kaggle Save Version must check out the frozen implementation commit and
run `run_clean_horizon_kaggle.sh`. Across sessions, only the explicit resume
path and Kaggle version identity may change.

Each checkpoint stores:

- the complete frozen training contract;
- model, teacher, optimizer, DINO center, and FP16 scaler state;
- AMP total/consecutive overflow counters plus scheduled-attempt and applied-
  update counters;
- internal completed epoch;
- one Python, NumPy, CPU torch, and CUDA RNG state per DDP rank.

Before model updates, a resumed session checks source commit and clean state,
training arguments, dataset size and class count, world size, loader and
optimizer steps per epoch, checkpoint and optimizer-step coordinates, required
state keys, and RNG state count. Any mismatch fails before training. The
runtime guard exits only after an epoch, rolling checkpoint, log row, and
session summary are complete.

The registered session budget is `11.5` hours with `45` minutes reserved. A
session may stop earlier if the observed mean epoch duration indicates that
another epoch could violate the budget. A platform interruption during an
epoch contributes no evidence; the next version resumes from the preceding
accepted rolling checkpoint.

## Invalid-Run And Kill Rules

The current session is invalid and must not be resumed when any of these occur:

- source, dataset, world size, training contract, or checkpoint coordinate
  mismatch;
- missing or unloadable model, optimizer, scaler, center, or RNG state;
- non-finite loss;
- a rank-inconsistent AMP overflow decision;
- three consecutive AMP overflows;
- missing, malformed, or coordinate-inconsistent AMP/optimizer-step state;
- OOM, data-layout failure, checkpoint write failure, or absent session
  summary.

Do not stop early because a training proxy, VOC value, or structural diagnostic
looks favorable or unfavorable.

## Frozen Evaluation Gate

Primary representation: EMA teacher. Student results are secondary and cannot
replace it.

Primary downstream protocol: PASCAL VOC frozen-backbone linear segmentation,
`global_confusion_v2`, probe seeds `42 / 1337 / 2027`, identical probe config
at labels `180 / 250 / 318`.

For each probe seed `s`, define:

```text
d_s = mIoU(label 318, s) - mIoU(label 180, s)
```

Before opening any label-250 or label-318 output, run the three probe seeds at
label 180 and record `sigma_pre`, the sample SD. Freeze:

```text
delta_abs = 0.50 mIoU points
delta_min = max(2 * sigma_pre, delta_abs)
U95 = mean(d_s) + 2.920 * sd(d_s) / sqrt(3)
```

Issue `clean_horizon_decline_confirmed` only if:

1. the complete checkpoint/session audit is `continuous` with sufficient
   evidence;
2. all three `d_s` values are negative;
3. `U95 < 0`;
4. `-mean(d_s) >= delta_min`.

Issue `clean_horizon_decline_not_detected` if `mean(d_s) >= 0` or the observed
decline is smaller than `delta_min`. Use `inconclusive` for every other
structurally valid result that fails at least one confirmation condition, such
as a large negative mean with mixed seed directions or `U95 >= 0`. No
checkpoint, seed, endpoint, threshold, or representation may be changed after
outputs are opened.

COCO-Stuff labels `180 / 318` are secondary consistency evidence and run only
after the VOC gate produces a scientific verdict. They cannot change the VOC
status.

## Consequences

- A confirmed clean-horizon decline permits a separately registered,
  single-fork C0-versus-CLS-CRR prevention experiment.
- A not-detected or inconclusive result blocks late-stage mitigation claims in
  this recipe.
- KoLeo, extra fork points, and additional backbone seeds require separate
  registration and cannot retroactively alter this gate.

No file-content digest is part of training, resume, evaluation, or artifact
acceptance. Acceptance uses checkpoint loadability, internal epoch, structured
training identity, file size, directory structure, source state, and actual
run outputs.

## Execution Record

Session 1 was submitted after registration through the existing Kaggle
Notebook `bingzhouxie/dino-train`:

- Kaggle Version: 20;
- script version: `346119135`;
- version name: `clean-horizon-seed0-session1-v1`;
- accelerator: GPU T4 x2;
- input: ImageNet100 only;
- resume mode: fresh epoch 0;
- source checkout: `7404e7fcddaa3702574697aa4fa7aa2bb3d1e8b3`;
- terminal status: excluded engineering failure;
- failure: one `GradScaler` optimizer-step skip at epoch `17`, iteration `793`;
- accepted completed epoch/checkpoint: none for the V2 contract;
- resume policy: V1 checkpoints are rejected by V2 and must not be used.

The failure occurred before any registered endpoint and contributes no
scientific evidence. Session 1 V2 must restart from epoch 0 after this amended
protocol and its frozen implementation are committed and published.

Session 1 V2 was then submitted through the same Notebook after the amended
protocol and implementation were published:

- Kaggle Version: 21;
- script version: `346212825`;
- version name: `clean-horizon-seed0-session1-v2`;
- accelerator: GPU T4 x2;
- input: ImageNet100 only;
- resume mode: fresh epoch 0;
- source checkout: `4c16679e915ca1e84842d652c911166f164b5183`;
- terminal Kaggle status: successful;
- session status: `partial_runtime_guard`;
- completed epochs: `18` (last zero-based label `17`);
- rolling checkpoint: `checkpoint.pth`, `704834924` bytes, internal completed
  epoch `18`;
- optimizer attempts/applied: `8892 / 8891`, exactly one recovered AMP
  overflow and zero consecutive overflows at the boundary;
- recovered event: epoch `17`, iteration `793`, optimizer slot `8794`, scaler
  `1048576 -> 524288`.

The published summary, event record, file list, source identity, contract
version, and optimizer coordinate are consistent. The rolling checkpoint is
the only V2 parent selected for the next session; strict resume validation must
load and validate it before the first new model update.

Session 2 V2 was submitted through the same Notebook with exactly one source
cell change: `CLEAN_HORIZON_RESUME_FROM` now names the Version-21 rolling
checkpoint mounted from the preceding Notebook output.

- Kaggle Version: 22;
- version name: `clean-horizon-seed0-session2-v2`;
- accelerator: GPU T4 x2;
- inputs: ImageNet100 and Version-21 `dino train` output;
- resume checkpoint:
  `/kaggle/input/notebooks/bingzhouxie/dino-train/dino_clean_horizon_seed0/checkpoint.pth`;
- launch status at the initial check: running.

Neither session is a downstream scientific result. The clean-horizon gate
remains pending until label `318` and the registered probes are complete.

### Execution Addendum 2026-09-05

The frozen contract and decision rules above are unchanged. The last remote
observation on 2026-09-01 supersedes the launch-only status for Version 22:
script version `346345784` succeeded in 37239.0 seconds with
`partial_runtime_guard`, completed epochs 33, last label 32, attempts/applied
16302/16299, total/consecutive overflow 3/0, and rolling size 704835052 bytes.
The Notebook validation cell completed its checkpoint load, contract, epoch,
AMP, coordinate, and two-rank RNG checks. Local independent artifact acceptance
is still pending.

The draft parent input was updated to Version 22, but Session 3 / Version 23
was not created because GPU quota rejected submission. No CPU fallback or
scientific setting change occurred. Quota must be refreshed rather than
assuming that the then-reported three-day wait still applies. See
[execution snapshot](RUN_STATUS_2026-09-05.md) for the next-session checklist.
