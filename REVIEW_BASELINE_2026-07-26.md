# Review Baseline - 2026-07-26

Status: pinned execution baseline for the schedule-audit and metric-v2 round.

This record fixes the repository state and external evidence inventory before
the late-stage phenomenon is re-evaluated:

1. pin commits;
2. repair the schedule audit and dense metric protocol;
3. retrieve and hash external checkpoints and per-session logs;
4. run the schedule audit and metric-v2 repeats independently;
5. apply the pre-registered phenomenon gate;
6. only then freeze an intervention protocol.

## Pinned Repositories

| repository | branch | baseline HEAD | baseline message |
| --- | --- | --- | --- |
| `dino_v3` | `codex/raw-l2-diagnostics` | `742190c607a63bff2549adaad13c8416745e142d` | Make VOC probe sweeps reproducible |
| `patch_crr` | `class-aligned-patch-regularization` | `4e2df95870b4b306a9f21b3f638c17122eb24ddc` | Draft staged CRR validation plan |

Both commits were created at `2026-07-26 12:31:02 +0200` while review was in
progress. Reviews after this point must identify a commit before drawing code
conclusions. Fixes are applied between review rounds, not while a pinned state
is being reviewed.

At baseline, `dino_v3` also contained two unrelated, untracked user files:

- `Genrate Fixed-image qualitative grid.py`
- `report.md`

They are outside this execution round and must not be staged or modified.
`patch_crr` was clean at its pinned baseline.

## Local Checkpoint Evidence

The ignored `dinocehckpoint/` directory contains valuable external artifacts.
They remain untracked and must not be repackaged or committed.

| file | filename epoch | checkpoint `epoch` / completed epochs | SHA256 |
| --- | ---: | ---: | --- |
| `checkpoint0180.pth` | 180 | 181 | `3e585ac93c18d75c583a0e69b8ae19e7037630877e0c5b0f8311fb993ff01767` |
| `checkpoint0190.pth` | 190 | 191 | `0f32497f1caf00f400b424ff85597188228855481d9108a9967a154664c806ec` |
| `checkpoint0200.pth` | 200 | 201 | `542f8ac2c9d2af1b740e1f29e04f9b739c2254710ed5f53401de65ed32de22b8` |
| `checkpoint0210.pth` | 210 | 211 | `c892944418418d48ee3cf730a4448abf574de8d790b527f206f781db8a0c9217` |
| `checkpoint0220.pth` | 220 | 221 | `db9208adc3ee67564978737afbdd6bc80e2c0c3e60baa11fc0f19fec2b1c7dac` |
| `checkpoint0230.pth` | 230 | 231 | `e82cacc5c217a6ec8c62f64db0576f2ff1f7ec6d5a6f459db9433d759d1e2cab` |
| `checkpoint0235.pth` | 235 | 236 | `f16d622f7730bc9be7ec2fce95b4abe201f45abd92ac846e6616a22da3c76c6b` |
| `checkpoint199.pth` | 199 | 200 | `74039dacf46c251920c745554542fc748ca5f271244f82e94a9cc031c2065bfd` |
| `checkpoint215.pth` | 215 | 216 | `23c68d05c3189f3378a2218eaeeb8fa0c48908cc120ee438ae6a205786bdcf7b` |

All nine local checkpoints embed the same schedule-defining values:

- target `epochs=300`;
- `lr=0.0005`, `min_lr=0.000001`, `warmup_epochs=10`;
- `weight_decay=0.04`, `weight_decay_end=0.4`;
- `momentum_teacher=0.996`;
- `batch_size_per_gpu=64`, `accum_steps=2`.

This establishes one schedule identity across the supplied epoch-180 to
epoch-235 window. It does not establish what happened before epoch 180 or
after epoch 235, and it does not prove an uninterrupted run.

The coordinate contract is:

- experiment labels and periodic filenames use zero-based epoch indices;
- checkpoint `epoch` stores the number of completed epochs and the next epoch
  to run;
- therefore a repository periodic checkpoint satisfies
  `checkpoint["epoch"] == filename_epoch + 1`.

## External Artifacts Still Required

Retrieve originals into the ignored artifact directory and record SHA256
before audit or evaluation. Do not infer provenance from filenames.

| artifact | purpose | status |
| --- | --- | --- |
| `checkpoint0170.pth` | bracket any pre-180 schedule transition | pending retrieval |
| `checkpoint0250.pth` | registered metric-v2 checkpoint and schedule evidence | pending retrieval |
| `checkpoint0290.pth`, `checkpoint0300.pth`, `checkpoint0310.pth` | bracket a possible late schedule transition | pending retrieval |
| `checkpoint0318.pth` or its verified rolling-checkpoint equivalent | registered metric-v2 endpoint | pending retrieval |
| every independent Colab/Kaggle session `log.txt` | observed LR/WD, session boundaries, and approximate world-size fit | pending retrieval |

The Drive inventory confirms that the named checkpoints exist, but connector
metadata is not a substitute for downloading and hashing the bytes. Every
Kaggle Save Version keeps an independent output; logs must remain separate
inputs to the audit rather than being concatenated or merged by epoch.
Teacher momentum is not logged and is reconstructed from checkpoint arguments.

The connected Drive downloader rejects these approximately 705 MB files
because its per-file limit is 100 MB. Chrome reaches Drive's explicit
"cannot scan for viruses" confirmation, but the controlled download channel
does not materialize the resulting large binary in the workspace. These
attempts verify neither bytes nor hashes; the rows above therefore remain
pending until the originals are placed in the ignored artifact directory.

## Round Rules

1. Historical VOC/COCO values are `batch_mean_v1`. They remain historical
   context and never enter a `global_confusion_v2` decision table.
2. Formal v2 rows must record and match `metric_version`, `probe_seed`,
   `checkpoint_key`, representation, checkpoint path, and checkpoint hash.
3. The schedule audit may return `stitched`, `continuous`, or `unknown`.
   `continuous` means only that sufficient supplied evidence shows one
   schedule; it is not proof that no external session or artifact is missing.
4. The v2 repeats at experiment labels `{180, 250, 318}` run independently of
   the audit result. Their corresponding completed-epoch coordinates are
   `{181, 251, 319}`.
5. A `stitched` audit makes the curve exploratory. An `unknown` audit blocks a
   clean-horizon phenomenon verdict. Neither outcome may be silently treated
   as `continuous`.
6. Code and documents are committed only after targeted tests, the repository
   test suite, notebook JSON validation, and `git diff --check` pass.
