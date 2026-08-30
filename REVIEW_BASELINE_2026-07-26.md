# Review Baseline - 2026-07-26

Status: pinned execution baseline for the schedule-audit and metric-v2 round.

This record fixes the repository state and external evidence inventory before
the late-stage phenomenon is re-evaluated:

1. pin commits;
2. repair the schedule audit and dense metric protocol;
3. retrieve and structurally inventory external checkpoints and per-session logs;
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

## Offline Checkpoint Evidence

The canonical cold archive is stored on the offline volume named `external`
under the relative path
`dino_research/dino_v3_schedule_audit_2026-07-26/`. The archive contains 15
raw checkpoints in `checkpoints/`, a machine-readable inventory in
`manifests/`, and the current audit in `schedule_audit/`. Checkpoints were
copied without repackaging, reopened with PyTorch, checked for coordinate and
training-argument consistency, and then removed from Downloads and the
repository-local staging directory.

| file | filename epoch | checkpoint `epoch` / completed epochs | size bytes |
| --- | ---: | ---: | ---: |
| `checkpoint0170.pth` | 170 | 171 | 704806696 |
| `checkpoint0180.pth` | 180 | 181 | 704809948 |
| `checkpoint0190.pth` | 190 | 191 | 704809948 |
| `checkpoint199.pth` | 199 | 200 | 704806760 |
| `checkpoint0200.pth` | 200 | 201 | 704809948 |
| `checkpoint0210.pth` | 210 | 211 | 704809948 |
| `checkpoint215.pth` | 215 | 216 | 704806760 |
| `checkpoint0220.pth` | 220 | 221 | 704809948 |
| `checkpoint0230.pth` | 230 | 231 | 704809948 |
| `checkpoint0235.pth` | 235 | 236 | 704806760 |
| `checkpoint0250.pth` | 250 | 251 | 704809948 |
| `checkpoint0290.pth` | 290 | 291 | 704809948 |
| `checkpoint0300.pth` | 300 | 301 | 704809948 |
| `checkpoint0310.pth` | 310 | 311 | 704809948 |
| `checkpoint0318.pth` | 318 | 319 | 704806760 |

All 15 checkpoints have consistent filename and internal epoch coordinates.
The audit identifies three schedule identities:

- `checkpoint0170.pth`: target `epochs=200`;
- experiment epochs 180 through 290: target `epochs=300`;
- experiment epochs 300 through 318: target `epochs=500`.

The remaining schedule-defining values match across all three segments:

- `lr=0.0005`, `min_lr=0.000001`, `warmup_epochs=10`;
- `weight_decay=0.04`, `weight_decay_end=0.4`;
- `momentum_teacher=0.996`;
- `batch_size_per_gpu=64`, `accum_steps=2`.

The target changes at completed-epoch intervals 171-180 and 291-300 establish
that the supplied curve is stitched across multiple schedule identities. The
current audit verdict is therefore `stitched` with `evidence_status=partial`.
Without independent session logs, it cannot localize the exact boundary,
measure an LR/WD value jump, or infer world size. Historical late-stage curve
evidence is `stitched-exploratory`, not a clean-horizon phenomenon result.

The coordinate contract is:

- experiment labels and periodic filenames use zero-based epoch indices;
- checkpoint `epoch` stores the number of completed epochs and the next epoch
  to run;
- therefore a repository periodic checkpoint satisfies
  `checkpoint["epoch"] == filename_epoch + 1`.

## External Artifacts Still Required

The checkpoint retrieval requirement is complete. Independent session logs
remain outstanding and must be stored under the cold archive's `logs/`
directory, one source-specific subdirectory per Colab or Kaggle session.
Do not concatenate logs or overwrite records from separate sessions.

| artifact | purpose | status |
| --- | --- | --- |
| all 15 checkpoint files listed above | schedule audit and registered metric-v2 inputs | retrieved, structurally checked, and archived |
| every independent Colab/Kaggle session `log.txt` | observed LR/WD, session boundaries, and approximate world-size fit | pending retrieval |

Every Kaggle Save Version keeps an independent output; logs remain separate
audit inputs rather than being merged by epoch. Teacher momentum is not logged
and is reconstructed from checkpoint arguments.

## Round Rules

1. Historical VOC/COCO values are `batch_mean_v1`. They remain historical
   context and never enter a `global_confusion_v2` decision table.
2. Formal v2 rows must record and match `metric_version`, `probe_seed`,
   `checkpoint_key`, representation, checkpoint path, and structured identity.
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
