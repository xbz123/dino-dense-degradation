# Clean-Horizon Execution Snapshot

Document updated: 2026-09-05. Last remote observation: 2026-09-01.
This is an execution record, not a new registration or a live Kaggle status.

## Evidence Boundary

The clean-horizon V2 baseline has reached **33 of 319 completed epochs** in
the last inspected Kaggle output. It has not reached any registered downstream
endpoint. Its scientific verdict remains pending.

The historical three-probe-seed teacher VOC v2 characterization is complete,
but its training horizon is stitched. Neither those rows nor Patch-CRR's
separate early/reduced-budget experiments establish clean-horizon SDD here.

## Sessions

| Version | Script version | Role and observed result |
| --- | --- | --- |
| 20 | 346119135 | V1 engineering failure; excluded and forbidden as V2 parent |
| 21 | 346212825 | V2 session 1; successful runtime guard at completed epoch 18 |
| 22 | 346345784 | V2 session 2; successful runtime guard at completed epoch 33 |
| 23 | Not created at last check | Session 3 submission blocked by GPU quota |

Sources: [Version 21 logs](https://www.kaggle.com/code/bingzhouxie/dino-train/log?scriptVersionId=346212825)
and [Version 22 logs](https://www.kaggle.com/code/bingzhouxie/dino-train/log?scriptVersionId=346345784).
These are access-controlled run pages, not bundled raw artifacts.

## Version 22 Boundary

| Field | Observed value |
| --- | --- |
| Kaggle terminal status | Successful, 37239.0 seconds |
| Session status | `partial_runtime_guard` |
| Start / completed / last label | 18 / 33 / 32 |
| Scheduled attempts / applied updates | 16302 / 16299 |
| Cumulative / consecutive overflow | 3 / 0 |
| Rolling checkpoint | `checkpoint.pth`, 704835052 bytes |
| Periodic checkpoints | `checkpoint0020.pth`, `checkpoint0030.pth` |
| Contract / backbone seed | 2 / 0 |
| Source | `4c16679e915ca1e84842d652c911166f164b5183`, clean |
| Data / hardware | ImageNet-100, 126689 images, 100 classes, T4 x2 |

The coordinates are consistent: `33 * 494 = 16302` and
`16302 - 16299 = 3`. Session 2 added 15 epochs and two recovered overflows;
three cumulative overflows are not the three-consecutive-overflow kill rule.

The completed Notebook validation cell loaded the rolling checkpoint and
checked its internal epoch, byte size, training contract, source, AMP state,
attempt coordinate, and two rank RNG states against the summary. Startup logs
showed strict model/state loading and training from epoch 18. This is remote
execution evidence. Independent local download and revalidation of the V22
checkpoint, summary, events, and log remain pending; do not mark local archive
acceptance complete based on the web log alone.

## Continuation Readiness

The draft Notebook input was updated from Version 21 to Version 22 using
Kaggle's input-version update. The mounted path remains:

```text
/kaggle/input/notebooks/bingzhouxie/dino-train/dino_clean_horizon_seed0/checkpoint.pth
```

A path alone does not identify its input version. Verify the mounted version
and checkpoint internal state before proceeding. Session 3 was named
`clean-horizon-seed0-session3-v2`, but Save Version was rejected by GPU quota;
no Version 23 was created at the last check. The then-reported reset interval
was three days. That is historical, not evidence of today's quota. CPU was not
selected. The hourly `dino-clean-horizon` automation was deleted; there is no
active monitor documented by this snapshot.

## Next Authorized Execution Checklist

1. Refresh version history, active jobs, quota, and draft inputs before any
   submission. Do not duplicate an intervening user-created run.
2. Download and independently validate V22 artifacts, keeping session logs
   separate. Preserve originals; use structural metadata and loadability,
   not file-content digests.
3. If V22 is still the latest accepted parent, resume from completed epoch 33,
   attempts 16302, applied updates 16299, overflow total 3 / consecutive 0,
   and two RNG states. Use the existing Notebook and Save Version + Run All.
4. Preserve the [frozen V2 contract](CLEAN_HORIZON_BASELINE_PROTOCOL_2026-08-30.md),
   including source, 319-epoch schedule, seed, hardware, crops and batch.
5. Record the actual new version ID and startup state. A submitted or running
   job is not an accepted session or a scientific result.
6. After label 318 is accepted, run the registered label-180 noise study first,
   freeze its threshold, then open labels 250/318. Historical stitched-run
   noise estimates do not substitute for this clean-baseline calibration.
7. Conditional COCO and a separately registered intervention remain behind the
   respective gates in the frozen protocol. No new experiment is registered
   or launched by this documentation update.
