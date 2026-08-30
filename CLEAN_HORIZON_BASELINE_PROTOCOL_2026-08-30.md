# Clean Single-Horizon Baseline Protocol

Status: `registered_before_first_training_output`

Registration date: 2026-08-30

Frozen training implementation:
`7404e7fcddaa3702574697aa4fa7aa2bb3d1e8b3`

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
  exactly `988` used micro-batches and `494` optimizer steps;
- local crops `4`, `norm_last_layer=false`, FP16 enabled;
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
- internal completed epoch;
- one Python, NumPy, CPU torch, and CUDA RNG state per DDP rank.

Before model updates, a resumed session checks source commit and clean state,
training arguments, dataset size and class count, world size, loader and
optimizer steps per epoch, checkpoint coordinate, required state keys, and RNG
state count. Any mismatch fails before training. The runtime guard exits only
after an epoch, rolling checkpoint, log row, and session summary are complete.

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
- an AMP overflow that skips an optimizer step;
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
