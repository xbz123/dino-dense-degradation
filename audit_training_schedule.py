"""
Training Schedule Audit for Stitched DINO Runs
==============================================
``main_dino.py`` rebuilds the LR, weight-decay, and teacher-momentum schedules
at every launch from the current arguments. A resumed run can therefore be
stitched even when the schedule values happen to be close at the boundary.

The audit keeps three epoch coordinates separate:

* ``filename_epoch``: the zero-based epoch in ``checkpointNNNN.pth``.
* ``completed_epochs``: checkpoint ``epoch`` (the next epoch to run).
* ``log_epoch_index``: the zero-based epoch written to ``log.txt``.

For a periodic checkpoint written by this repository, the expected relation is
``completed_epochs == filename_epoch + 1``.

Each log file remains an independent source. If epochs repeat or move
backwards inside one file, it is split into multiple sessions. A new session's
first logged epoch is the preferred resume boundary. Without that evidence,
the audit reports a boundary interval rather than guessing a point.

World size is not stored in checkpoints. The tool can only fit candidate world
sizes approximately because ``train_lr`` is an epoch-wide microbatch average
while the checkpoint does not record the data-loader length. The report keeps
all candidate fits and returns no inferred world size when residual quality or
candidate separation is inadequate.

Usage:
    python audit_training_schedule.py \
        --ckpt_dir /content/drive/MyDrive/dinocheckpoint \
        --log_files /content/drive/MyDrive/colab/log.txt \
                    /content/drive/MyDrive/kaggle/version_1/log.txt \
        --output_dir /content/drive/MyDrive/schedule_audit

Outputs are ``schedule_audit.json`` and, when matplotlib is available,
``schedule_audit.png``.
"""

import argparse
import json
import math
import os
import re

import numpy as np


CHECKPOINT_NAME_RE = re.compile(r"checkpoint\s*0*(\d+)\.pth$")

SCHEDULE_FIELDS = (
    "epochs",
    "lr",
    "min_lr",
    "warmup_epochs",
    "weight_decay",
    "weight_decay_end",
    "momentum_teacher",
    "batch_size_per_gpu",
    "accum_steps",
)

LR_RATIO_BOUNDS = (0.5, 2.0)
WD_DELTA_LIMIT = 0.02
MOMENTUM_DELTA_LIMIT = 5e-4
REVERSAL_MIN_EPOCH = 15
REVERSAL_FACTOR = 1.2
WORLD_SIZE_MAX_RELATIVE_RESIDUAL = 0.08
WORLD_SIZE_MIN_CANDIDATE_MARGIN = 0.02


def extract_epoch_from_name(filename):
    """Return the zero-based periodic checkpoint epoch, or ``None``."""
    match = CHECKPOINT_NAME_RE.search(os.path.basename(filename))
    return int(match.group(1)) if match else None


def _serializable_args(args_obj):
    if isinstance(args_obj, dict):
        items = args_obj.items()
    else:
        try:
            items = vars(args_obj).items()
        except TypeError:
            return None
    return {
        key: value
        for key, value in items
        if isinstance(value, (int, float, str, bool, type(None)))
    }


def load_checkpoint_record(path):
    """Read checkpoint provenance without conflating its epoch coordinates."""
    import torch

    filename_epoch = extract_epoch_from_name(path)
    record = {
        "file": str(path),
        "filename_epoch": filename_epoch,
        "completed_epochs": None,
        "log_epoch_index": None,
        "args": None,
        "coordinate_status": "unread",
        "usable_for_schedule": False,
        "error": None,
    }
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - preserve the evidence failure
        record["error"] = f"{type(error).__name__}: {error}"
        record["coordinate_status"] = "checkpoint_read_failed"
        return record

    try:
        if not isinstance(checkpoint, dict):
            record["error"] = "checkpoint is not a dict"
            record["coordinate_status"] = "invalid_checkpoint"
            return record

        if "epoch" in checkpoint:
            try:
                record["completed_epochs"] = int(checkpoint["epoch"])
            except (TypeError, ValueError):
                record["error"] = f"invalid checkpoint epoch: {checkpoint['epoch']!r}"
        record["args"] = _serializable_args(checkpoint.get("args"))
    finally:
        del checkpoint

    completed = record["completed_epochs"]
    if completed is None:
        record["coordinate_status"] = "missing_internal_epoch"
    elif completed < 0:
        record["coordinate_status"] = "invalid_internal_epoch"
    else:
        record["log_epoch_index"] = completed - 1
        if filename_epoch is None:
            record["coordinate_status"] = "generic_internal_epoch"
        elif completed == filename_epoch + 1:
            record["coordinate_status"] = "consistent"
        else:
            record["coordinate_status"] = "filename_internal_mismatch"

    record["usable_for_schedule"] = (
        record["args"] is not None
        and record["coordinate_status"] in {"consistent", "generic_internal_epoch"}
        and record["error"] is None
    )
    return record


def schedule_identity(args):
    return tuple(args.get(field) for field in SCHEDULE_FIELDS)


def cosine_value(epoch_position, base, final, total_epochs, warmup_epochs=0):
    """Approximate ``utils.cosine_scheduler`` at a fractional epoch."""
    position = float(epoch_position)
    warmup_epochs = float(warmup_epochs or 0)
    total_epochs = float(total_epochs)
    if warmup_epochs > 0 and position < warmup_epochs:
        return base * (position / warmup_epochs)
    if position >= total_epochs:
        return final
    denominator = max(1e-12, total_epochs - warmup_epochs)
    progress = (position - warmup_epochs) / denominator
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * progress))


def scaled_base_lr(args, world_size):
    effective_batch = (
        args["batch_size_per_gpu"] * args.get("accum_steps", 1) * world_size
    )
    return args["lr"] * effective_batch / 256.0


def reconstruct_metric(args, world_size, metric, epoch_position):
    if metric == "lr":
        return cosine_value(
            epoch_position,
            scaled_base_lr(args, world_size),
            args.get("min_lr", 0.0),
            args["epochs"],
            args.get("warmup_epochs", 0),
        )
    if metric == "wd":
        return cosine_value(
            epoch_position,
            args["weight_decay"],
            args["weight_decay_end"],
            args["epochs"],
        )
    if metric == "momentum":
        return cosine_value(
            epoch_position,
            args["momentum_teacher"],
            1.0,
            args["epochs"],
        )
    raise ValueError(f"Unknown metric: {metric}")


def build_segments(records):
    """Group coordinate-valid checkpoints by contiguous schedule identity."""
    usable = sorted(
        (record for record in records if record["usable_for_schedule"]),
        key=lambda record: (record["log_epoch_index"], record["file"]),
    )
    segments = []
    for record in usable:
        identity = schedule_identity(record["args"])
        if segments and segments[-1]["identity"] == identity:
            segment = segments[-1]
            segment["last_checkpoint_log_epoch_index"] = record["log_epoch_index"]
            segment["last_completed_epochs"] = record["completed_epochs"]
            segment["files"].append(record["file"])
        else:
            segments.append(
                {
                    "identity": identity,
                    "args": record["args"],
                    "first_checkpoint_log_epoch_index": record["log_epoch_index"],
                    "last_checkpoint_log_epoch_index": record["log_epoch_index"],
                    "first_completed_epochs": record["completed_epochs"],
                    "last_completed_epochs": record["completed_epochs"],
                    "files": [record["file"]],
                }
            )
    return segments


def _make_log_session(path, session_index, entries):
    return {
        "source": str(path),
        "session_index": session_index,
        "source_id": f"{path}#session-{session_index}",
        "first_log_epoch_index": entries[0]["log_epoch_index"],
        "last_log_epoch_index": entries[-1]["log_epoch_index"],
        "num_entries": len(entries),
        "entries": entries,
    }


def parse_log_files(paths):
    """Parse logs without merging sessions or overwriting duplicate epochs."""
    sessions = []
    issues = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        except OSError as error:
            issues.append(
                {
                    "source": str(path),
                    "status": "read_failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue

        current = []
        session_index = 0
        valid_rows = 0
        for line_number, line in enumerate(lines, start=1):
            try:
                payload = json.loads(line.strip())
            except (json.JSONDecodeError, TypeError):
                continue
            if "epoch" not in payload:
                continue
            try:
                log_epoch_index = int(payload["epoch"])
            except (TypeError, ValueError):
                continue

            if current and log_epoch_index <= current[-1]["log_epoch_index"]:
                sessions.append(_make_log_session(path, session_index, current))
                current = []
                session_index += 1

            entry = {
                "log_epoch_index": log_epoch_index,
                "line_number": line_number,
            }
            if "train_lr" in payload:
                entry["lr"] = float(payload["train_lr"])
            if "train_wd" in payload:
                entry["wd"] = float(payload["train_wd"])
            current.append(entry)
            valid_rows += 1

        if current:
            sessions.append(_make_log_session(path, session_index, current))
        if valid_rows == 0:
            issues.append(
                {
                    "source": str(path),
                    "status": "no_epoch_records",
                    "error": None,
                }
            )
    return sessions, issues


def _changed_schedule_fields(segment_a, segment_b):
    return [
        field
        for field in SCHEDULE_FIELDS
        if segment_a["args"].get(field) != segment_b["args"].get(field)
    ]


def locate_boundary(segment_a, segment_b, log_sessions):
    """Locate a schedule change, preferring a new log session's first epoch."""
    lower = segment_a["last_completed_epochs"]
    upper = segment_b["first_checkpoint_log_epoch_index"]
    boundary = {
        "schedule_identity_changed": True,
        "changed_fields": _changed_schedule_fields(segment_a, segment_b),
        "boundary_epoch": None,
        "boundary_interval": None,
        "boundary_sources": [],
        "boundary_evidence_status": "insufficient",
    }
    if lower > upper:
        boundary["boundary_interval"] = [upper, lower]
        boundary["boundary_evidence_reason"] = "checkpoint_coordinate_order_invalid"
        return boundary

    starts = {}
    for session in log_sessions:
        start = session["first_log_epoch_index"]
        if (
            lower <= start <= upper
            and session["last_log_epoch_index"] >= upper
        ):
            starts.setdefault(start, []).append(session["source_id"])

    if len(starts) == 1:
        epoch = next(iter(starts))
        boundary["boundary_epoch"] = epoch
        boundary["boundary_interval"] = [epoch, epoch]
        boundary["boundary_sources"] = starts[epoch]
        boundary["boundary_evidence_status"] = "sufficient"
        boundary["boundary_evidence_reason"] = (
            "unique_log_session_start_covering_first_new_checkpoint"
        )
    elif len(starts) > 1:
        boundary["boundary_interval"] = [min(starts), max(starts)]
        boundary["boundary_sources"] = [
            source for epoch in sorted(starts) for source in starts[epoch]
        ]
        boundary["boundary_evidence_status"] = "partial"
        boundary["boundary_evidence_reason"] = (
            "multiple_log_session_starts_covering_first_new_checkpoint"
        )
    elif lower == upper:
        boundary["boundary_epoch"] = lower
        boundary["boundary_interval"] = [lower, upper]
        boundary["boundary_evidence_status"] = "partial"
        boundary["boundary_evidence_reason"] = "adjacent_checkpoint_coordinates"
    else:
        boundary["boundary_interval"] = [lower, upper]
        boundary["boundary_evidence_reason"] = (
            "no_qualifying_log_session_start_in_interval"
        )
    return boundary


def assign_confident_log_intervals(segments, boundaries, log_sessions):
    """Assign only epochs that are unambiguous with respect to each boundary."""
    if len(segments) == 1:
        segments[0]["confident_log_epoch_interval"] = [
            segments[0]["first_checkpoint_log_epoch_index"],
            segments[0]["last_checkpoint_log_epoch_index"],
        ]
        return

    all_starts = [session["first_log_epoch_index"] for session in log_sessions]
    all_ends = [session["last_log_epoch_index"] for session in log_sessions]
    for index, segment in enumerate(segments):
        if index == 0:
            start = min(
                all_starts + [segment["first_checkpoint_log_epoch_index"]]
            )
        else:
            previous = boundaries[index - 1]
            if previous["boundary_epoch"] is not None:
                start = previous["boundary_epoch"]
            else:
                start = previous["boundary_interval"][1]

        if index == len(segments) - 1:
            end = max(all_ends + [segment["last_checkpoint_log_epoch_index"]])
        else:
            following = boundaries[index]
            if following["boundary_epoch"] is not None:
                end = following["boundary_epoch"] - 1
            else:
                end = following["boundary_interval"][0] - 1

        segment["confident_log_epoch_interval"] = [start, end]


def _compress_epochs(epochs):
    intervals = []
    for epoch in sorted(set(epochs)):
        if not intervals or epoch > intervals[-1][1] + 1:
            intervals.append([epoch, epoch])
        else:
            intervals[-1][1] = epoch
    return intervals


def _gaps_within(required_interval, covered_intervals):
    lower, upper = required_interval
    if lower > upper:
        return [[lower, upper]]
    gaps = []
    cursor = lower
    for start, end in covered_intervals:
        if end < lower or start > upper:
            continue
        clipped_start = max(start, lower)
        clipped_end = min(end, upper)
        if clipped_start > cursor:
            gaps.append([cursor, clipped_start - 1])
        cursor = max(cursor, clipped_end + 1)
    if cursor <= upper:
        gaps.append([cursor, upper])
    return gaps


def compute_log_coverage(segments, log_sessions):
    """Report whether logs continuously cover each supplied schedule horizon."""
    coverage_rows = []
    multiple_segments = len(segments) > 1
    for index, segment in enumerate(segments):
        required = (
            list(segment["confident_log_epoch_interval"])
            if multiple_segments
            else [
                segment["first_checkpoint_log_epoch_index"],
                segment["last_checkpoint_log_epoch_index"],
            ]
        )
        covered_epochs = []
        session_rows = []
        for session in log_sessions:
            session_epochs = [
                entry["log_epoch_index"]
                for entry in session["entries"]
                if "lr" in entry
                and required[0] <= entry["log_epoch_index"] <= required[1]
            ]
            if not session_epochs:
                continue
            intervals = _compress_epochs(session_epochs)
            covered_epochs.extend(session_epochs)
            session_rows.append(
                {
                    "source_id": session["source_id"],
                    "covered_intervals": intervals,
                }
            )

        covered_intervals = _compress_epochs(covered_epochs)
        gaps = _gaps_within(required, covered_intervals)
        if required[0] > required[1]:
            status = "invalid"
        elif not covered_intervals:
            status = "missing"
        elif gaps:
            status = "partial"
        else:
            status = "complete"
        coverage_rows.append(
            {
                "segment_index": index,
                "required_interval": required,
                "covered_intervals": covered_intervals,
                "gaps": gaps,
                "status": status,
                "covered_epoch_count": len(set(covered_epochs)),
                "session_intervals": session_rows,
            }
        )

    statuses = [row["status"] for row in coverage_rows]
    if statuses and all(status == "complete" for status in statuses):
        overall_status = "complete"
    elif any(status in {"complete", "partial"} for status in statuses):
        overall_status = "partial"
    else:
        overall_status = "missing"
    return {
        "status": overall_status,
        "segments": coverage_rows,
    }


def _segment_observations(segment, log_sessions):
    lower, upper = segment["confident_log_epoch_interval"]
    observations = []
    for session in log_sessions:
        for entry in session["entries"]:
            epoch = entry["log_epoch_index"]
            if lower <= epoch <= upper and "lr" in entry:
                observations.append(
                    {
                        "source_id": session["source_id"],
                        "log_epoch_index": epoch,
                        "lr": entry["lr"],
                    }
                )
    return observations


def infer_world_size(segment, log_sessions, candidates):
    """Fit candidate world sizes with an explicitly approximate LR model."""
    observations = _segment_observations(segment, log_sessions)
    approximation = (
        "epoch_midpoint approximation; logged train_lr is a microbatch mean, "
        "and exact reconstruction requires data-loader length and accumulation remainder"
    )
    if not observations:
        return {
            "world_size": None,
            "status": "insufficient",
            "confidence": "none",
            "candidate_margin": None,
            "candidates": [
                {
                    "world_size": int(world_size),
                    "rmse": None,
                    "relative_residual": None,
                }
                for world_size in candidates
            ],
            "residuals": {},
            "matched_epochs": 0,
            "matched_sources": [],
            "approximation": approximation,
        }

    observed = np.asarray([row["lr"] for row in observations], dtype=np.float64)
    observed_scale = max(float(np.sqrt(np.mean(np.square(observed)))), 1e-12)
    fits = []
    for world_size in candidates:
        predicted = np.asarray(
            [
                reconstruct_metric(
                    segment["args"],
                    world_size,
                    "lr",
                    row["log_epoch_index"] + 0.5,
                )
                for row in observations
            ],
            dtype=np.float64,
        )
        errors = predicted - observed
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        fits.append(
            {
                "world_size": int(world_size),
                "rmse": rmse,
                "relative_residual": rmse / observed_scale,
            }
        )
    fits.sort(key=lambda row: (row["relative_residual"], row["world_size"]))
    best = fits[0]
    margin = (
        fits[1]["relative_residual"] - best["relative_residual"]
        if len(fits) > 1
        else None
    )

    if best["relative_residual"] > WORLD_SIZE_MAX_RELATIVE_RESIDUAL:
        inferred = None
        status = "low_quality"
        confidence = "low"
    elif margin is not None and margin < WORLD_SIZE_MIN_CANDIDATE_MARGIN:
        inferred = None
        status = "ambiguous"
        confidence = "low"
    else:
        inferred = best["world_size"]
        status = "inferred"
        confidence = (
            "high" if best["relative_residual"] <= 0.02 else "moderate"
        )

    return {
        "world_size": inferred,
        "status": status,
        "confidence": confidence,
        "candidate_margin": margin,
        "candidates": fits,
        "residuals": {
            str(row["world_size"]): row["rmse"] for row in fits
        },
        "matched_epochs": len(observations),
        "matched_sources": sorted(
            {row["source_id"] for row in observations}
        ),
        "approximation": approximation,
    }


def _world_size_candidates(inference, configured_candidates):
    if inference["world_size"] is not None:
        return [inference["world_size"]]
    fitted = [row["world_size"] for row in inference["candidates"]]
    return fitted or list(configured_candidates)


def evaluate_boundary_values(
    boundary,
    segment_a,
    segment_b,
    candidates_a,
    candidates_b,
):
    """Measure boundary jump severity without deciding whether the run is stitched."""
    boundary["combos"] = []
    boundary["boundary_value_jump"] = None
    boundary["discontinuous"] = None
    boundary["discontinuous_any"] = None
    if boundary["boundary_epoch"] is None:
        return boundary

    epoch = boundary["boundary_epoch"]
    combos = []
    for world_size_a in candidates_a:
        for world_size_b in candidates_b:
            row = {
                "world_size_before": world_size_a,
                "world_size_after": world_size_b,
                "jump_metrics": [],
            }
            for metric in ("lr", "wd", "momentum"):
                before = reconstruct_metric(
                    segment_a["args"], world_size_a, metric, epoch
                )
                after = reconstruct_metric(
                    segment_b["args"], world_size_b, metric, epoch
                )
                delta = after - before
                ratio = after / max(abs(before), 1e-12)
                row[metric] = {
                    "before": before,
                    "after": after,
                    "delta": delta,
                    "ratio": ratio,
                }
                if (
                    metric == "lr"
                    and not (LR_RATIO_BOUNDS[0] <= ratio <= LR_RATIO_BOUNDS[1])
                ):
                    row["jump_metrics"].append(metric)
                elif metric == "wd" and abs(delta) > WD_DELTA_LIMIT:
                    row["jump_metrics"].append(metric)
                elif (
                    metric == "momentum"
                    and abs(delta) > MOMENTUM_DELTA_LIMIT
                ):
                    row["jump_metrics"].append(metric)
            row["triggered"] = bool(row["jump_metrics"])
            combos.append(row)

    triggered = [row["triggered"] for row in combos]
    if all(triggered):
        jump = True
    elif not any(triggered):
        jump = False
    else:
        jump = None
    boundary["combos"] = combos
    boundary["boundary_value_jump"] = jump
    boundary["discontinuous"] = jump
    boundary["discontinuous_any"] = any(triggered)
    return boundary


def detect_lr_reversals(log_sessions):
    """Detect LR reversals within each session, never across source boundaries."""
    reversals = []
    for session in log_sessions:
        previous = None
        for entry in session["entries"]:
            if "lr" not in entry:
                continue
            epoch = entry["log_epoch_index"]
            if (
                previous is not None
                and epoch > REVERSAL_MIN_EPOCH
                and entry["lr"] > previous["lr"] * REVERSAL_FACTOR
            ):
                reversals.append(
                    {
                        "source": session["source"],
                        "source_id": session["source_id"],
                        "session_index": session["session_index"],
                        "log_epoch_index": epoch,
                        "previous_log_epoch_index": previous["log_epoch_index"],
                        "lr_before": previous["lr"],
                        "lr_after": entry["lr"],
                    }
                )
            previous = entry
    return reversals


def _evidence_status(
    records,
    segments,
    log_sessions,
    log_issues,
    boundaries,
    segment_world_sizes,
    log_coverage,
):
    usable = [record for record in records if record["usable_for_schedule"]]
    checkpoint_issues = [
        record
        for record in records
        if record["error"] is not None or not record["usable_for_schedule"]
    ]
    fits_usable = all(
        inference["status"] == "inferred"
        for inference in segment_world_sizes
    )
    coverage_complete = (
        bool(log_coverage["segments"])
        and all(
            row["status"] == "complete"
            for row in log_coverage["segments"]
        )
    )

    if not usable:
        return "insufficient"
    if len(segments) > 1:
        boundary_complete = all(
            boundary["boundary_evidence_status"] == "sufficient"
            for boundary in boundaries
        )
        if (
            boundary_complete
            and fits_usable
            and coverage_complete
            and not checkpoint_issues
            and not log_issues
        ):
            return "sufficient"
        return "partial"
    if (
        len(usable) >= 2
        and fits_usable
        and coverage_complete
        and not checkpoint_issues
        and not log_issues
    ):
        return "sufficient"
    return "partial"


def make_verdict(
    records,
    segments,
    log_sessions,
    log_issues,
    boundaries,
    reversals,
    segment_world_sizes,
    log_coverage,
):
    evidence_status = _evidence_status(
        records,
        segments,
        log_sessions,
        log_issues,
        boundaries,
        segment_world_sizes,
        log_coverage,
    )
    observed_identity_change = len(segments) > 1
    if observed_identity_change:
        identity_changed = True
    elif evidence_status == "sufficient":
        identity_changed = False
    else:
        identity_changed = None

    jump_values = [
        boundary["boundary_value_jump"] for boundary in boundaries
    ]
    if reversals or any(value is True for value in jump_values):
        boundary_value_jump = True
    elif any(value is None for value in jump_values):
        boundary_value_jump = None
    elif boundaries:
        boundary_value_jump = False
    elif evidence_status == "sufficient":
        boundary_value_jump = False
    else:
        boundary_value_jump = None

    if observed_identity_change or reversals or boundary_value_jump is True:
        status = "stitched"
    elif (
        evidence_status == "sufficient"
        and identity_changed is False
        and boundary_value_jump is False
    ):
        status = "continuous"
    else:
        status = "unknown"

    return {
        "status": status,
        "schedule_identity_changed": identity_changed,
        "boundary_value_jump": boundary_value_jump,
        "evidence_status": evidence_status,
        "num_boundaries": len(boundaries),
        "num_lr_reversals": len(reversals),
        # Compatibility field: unknown is deliberately not coerced to False.
        "discontinuities_detected": (
            True if status == "stitched" else False if status == "continuous" else None
        ),
    }


def plot_audit(
    segments,
    segment_world_sizes,
    log_sessions,
    boundaries,
    output_path,
):
    if not segments and not log_sessions:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(
            f"WARNING: matplotlib unavailable ({error}); skipping {output_path}. "
            "The JSON report is complete without it."
        )
        return

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    metric_names = ("lr", "wd", "momentum")
    titles = (
        "Learning rate (log scale)",
        "Weight decay",
        "Teacher EMA momentum (reconstructed)",
    )
    for axis, metric, title in zip(axes, metric_names, titles):
        for index, segment in enumerate(segments):
            inference = segment_world_sizes[index]
            candidates = _world_size_candidates(inference, [1])
            lower, upper = segment["confident_log_epoch_interval"]
            if lower > upper:
                continue
            epochs = np.arange(lower, upper + 1)
            for candidate_index, world_size in enumerate(candidates):
                values = [
                    reconstruct_metric(
                        segment["args"], world_size, metric, epoch + 0.5
                    )
                    for epoch in epochs
                ]
                label = None
                if metric == "lr":
                    suffix = "" if inference["world_size"] is not None else " candidate"
                    label = (
                        f"segment {index}: target={segment['args'].get('epochs')} "
                        f"ws={world_size}{suffix}"
                    )
                axis.plot(
                    epochs,
                    values,
                    "--" if inference["world_size"] is None else "-",
                    linewidth=1.3,
                    alpha=0.8 if candidate_index == 0 else 0.5,
                    label=label,
                )

        if metric in ("lr", "wd"):
            for session in log_sessions:
                logged = [
                    (entry["log_epoch_index"], entry[metric])
                    for entry in session["entries"]
                    if metric in entry
                ]
                if not logged:
                    continue
                xs, ys = zip(*logged)
                label = (
                    f"logged {os.path.basename(session['source'])}"
                    f"#{session['session_index']}"
                    if metric == "lr"
                    else None
                )
                axis.scatter(xs, ys, s=9, alpha=0.55, label=label)

        for boundary in boundaries:
            if boundary["boundary_epoch"] is not None:
                axis.axvline(
                    boundary["boundary_epoch"],
                    color="red",
                    linestyle="--",
                    alpha=0.65,
                )
            elif boundary["boundary_interval"] is not None:
                axis.axvspan(
                    boundary["boundary_interval"][0],
                    boundary["boundary_interval"][1],
                    color="orange",
                    alpha=0.16,
                )
        if metric == "lr":
            axis.set_yscale("log")
            axis.legend(fontsize=7)
        axis.set_ylabel(title, fontsize=9)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Log epoch index (zero-based)")
    figure.suptitle("DINO training-schedule audit")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _collect_checkpoint_records(ckpt_dir):
    if not ckpt_dir:
        return [], []
    try:
        names = sorted(os.listdir(ckpt_dir))
    except OSError as error:
        return [], [
            {
                "source": str(ckpt_dir),
                "status": "checkpoint_directory_read_failed",
                "error": f"{type(error).__name__}: {error}",
            }
        ]
    records = []
    for name in names:
        if not name.endswith(".pth"):
            continue
        if extract_epoch_from_name(name) is None and name != "checkpoint.pth":
            continue
        records.append(load_checkpoint_record(os.path.join(ckpt_dir, name)))
    return records, []


def _report_checkpoint(record):
    return {
        "file": record["file"],
        "filename_epoch": record["filename_epoch"],
        "completed_epochs": record["completed_epochs"],
        "log_epoch_index": record["log_epoch_index"],
        "coordinate_status": record["coordinate_status"],
        "usable_for_schedule": record["usable_for_schedule"],
        "error": record["error"],
        "args": (
            {field: record["args"].get(field) for field in SCHEDULE_FIELDS}
            if record["args"] is not None
            else None
        ),
    }


def _report_log_session(session):
    return {
        "source": session["source"],
        "session_index": session["session_index"],
        "source_id": session["source_id"],
        "first_log_epoch_index": session["first_log_epoch_index"],
        "last_log_epoch_index": session["last_log_epoch_index"],
        "num_entries": session["num_entries"],
    }


def run_audit(ckpt_dir, log_files, output_dir, world_size_candidates):
    if not ckpt_dir and not log_files:
        raise ValueError("Provide --ckpt_dir and/or --log_files.")
    os.makedirs(output_dir, exist_ok=True)

    records, checkpoint_collection_issues = _collect_checkpoint_records(ckpt_dir)
    log_sessions, log_issues = parse_log_files(log_files or [])
    segments = build_segments(records)
    boundaries = [
        locate_boundary(segments[index], segments[index + 1], log_sessions)
        for index in range(len(segments) - 1)
    ]
    assign_confident_log_intervals(segments, boundaries, log_sessions)
    log_coverage = compute_log_coverage(segments, log_sessions)

    segment_world_sizes = [
        infer_world_size(segment, log_sessions, world_size_candidates)
        for segment in segments
    ]
    for index, boundary in enumerate(boundaries):
        evaluate_boundary_values(
            boundary,
            segments[index],
            segments[index + 1],
            _world_size_candidates(
                segment_world_sizes[index], world_size_candidates
            ),
            _world_size_candidates(
                segment_world_sizes[index + 1], world_size_candidates
            ),
        )

    reversals = detect_lr_reversals(log_sessions)
    all_log_issues = checkpoint_collection_issues + log_issues
    verdict = make_verdict(
        records,
        segments,
        log_sessions,
        all_log_issues,
        boundaries,
        reversals,
        segment_world_sizes,
        log_coverage,
    )
    notes = [
        "periodic filename and log epochs are zero-based; checkpoint epoch is completed_epochs",
        "teacher momentum is reconstructed because it is not written to log.txt",
        "world-size fitting uses an epoch-midpoint approximation to an epoch-mean train_lr",
        "continuous means no schedule change was observed in sufficient supplied evidence; it is not proof of an uninterrupted run",
    ]

    report = {
        "coordinate_contract": {
            "filename_epoch": "zero-based epoch encoded in checkpointNNNN.pth",
            "completed_epochs": "checkpoint['epoch']; next epoch to run",
            "log_epoch_index": "zero-based epoch in log.txt",
            "periodic_checkpoint_relation": (
                "completed_epochs == filename_epoch + 1"
            ),
        },
        "checkpoints": [_report_checkpoint(record) for record in records],
        "checkpoint_collection_issues": checkpoint_collection_issues,
        "log_sessions": [
            _report_log_session(session) for session in log_sessions
        ],
        "log_issues": log_issues,
        "log_coverage": log_coverage,
        "segments": [
            {
                "index": index,
                "first_checkpoint_log_epoch_index": segment[
                    "first_checkpoint_log_epoch_index"
                ],
                "last_checkpoint_log_epoch_index": segment[
                    "last_checkpoint_log_epoch_index"
                ],
                "first_completed_epochs": segment["first_completed_epochs"],
                "last_completed_epochs": segment["last_completed_epochs"],
                "confident_log_epoch_interval": segment[
                    "confident_log_epoch_interval"
                ],
                "num_checkpoints": len(segment["files"]),
                "args": {
                    field: segment["args"].get(field)
                    for field in SCHEDULE_FIELDS
                },
                "world_size_inference": segment_world_sizes[index],
                "log_coverage_status": log_coverage["segments"][index][
                    "status"
                ],
            }
            for index, segment in enumerate(segments)
        ],
        "boundaries": boundaries,
        "lr_reversals": reversals,
        "verdict": {**verdict, "notes": notes},
    }

    json_path = os.path.join(output_dir, "schedule_audit.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    plot_audit(
        segments,
        segment_world_sizes,
        log_sessions,
        boundaries,
        os.path.join(output_dir, "schedule_audit.png"),
    )

    total_logged_rows = sum(
        session["num_entries"] for session in log_sessions
    )
    print(
        f"Checkpoints read: {len(records)}  segments: {len(segments)}  "
        f"log sessions: {len(log_sessions)}  logged rows: {total_logged_rows}"
    )
    for index, segment in enumerate(segments):
        inference = segment_world_sizes[index]
        best_residual = (
            inference["candidates"][0]["relative_residual"]
            if inference["candidates"]
            else None
        )
        residual_text = (
            f"{best_residual:.3g}" if best_residual is not None else "n/a"
        )
        print(
            f"  segment {index}: checkpoint log epochs "
            f"{segment['first_checkpoint_log_epoch_index']}-"
            f"{segment['last_checkpoint_log_epoch_index']} "
            f"target={segment['args'].get('epochs')} "
            f"world_size={inference['world_size']} ({inference['status']}, "
            f"relative residual={residual_text})"
        )
    for boundary in boundaries:
        location = (
            str(boundary["boundary_epoch"])
            if boundary["boundary_epoch"] is not None
            else str(boundary["boundary_interval"])
        )
        print(
            f"  schedule identity change @ {location}: "
            f"evidence={boundary['boundary_evidence_status']} "
            f"value_jump={boundary['boundary_value_jump']} "
            f"fields={','.join(boundary['changed_fields'])}"
        )
    for reversal in reversals:
        print(
            f"  logged-lr reversal in {reversal['source_id']} @ "
            f"{reversal['log_epoch_index']}: "
            f"{reversal['lr_before']:.3g} -> {reversal['lr_after']:.3g}"
        )
    print(
        f"Verdict: {verdict['status']} "
        f"(identity_changed={verdict['schedule_identity_changed']}, "
        f"boundary_value_jump={verdict['boundary_value_jump']}, "
        f"evidence={verdict['evidence_status']})"
    )
    print(f"Report: {json_path}")
    return report


def main():
    parser = argparse.ArgumentParser("DINO training schedule audit")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="",
        help="Directory with checkpoint*.pth files",
    )
    parser.add_argument(
        "--log_files",
        type=str,
        nargs="+",
        default=[],
        help="Independent log.txt sources with train_lr/train_wd",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./schedule_audit"
    )
    parser.add_argument(
        "--world_size_candidates",
        type=str,
        default="1 2",
        help="Space-separated world sizes to fit approximately",
    )
    args = parser.parse_args()
    candidates = [int(value) for value in args.world_size_candidates.split()]
    if not candidates:
        raise ValueError("--world_size_candidates must not be empty")
    run_audit(
        args.ckpt_dir or None,
        args.log_files,
        args.output_dir,
        candidates,
    )


if __name__ == "__main__":
    main()
