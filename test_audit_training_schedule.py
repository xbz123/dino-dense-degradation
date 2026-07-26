"""Tests for the DINO stitched-schedule audit tool."""

import argparse
import json

import torch

from audit_training_schedule import (
    assign_confident_log_intervals,
    build_segments,
    detect_lr_reversals,
    extract_epoch_from_name,
    infer_world_size,
    load_checkpoint_record,
    parse_log_files,
    reconstruct_metric,
    run_audit,
)


def make_args(epochs, batch=64, accum=2):
    return argparse.Namespace(
        epochs=epochs,
        lr=5e-4,
        min_lr=1e-6,
        warmup_epochs=10,
        weight_decay=0.04,
        weight_decay_end=0.4,
        momentum_teacher=0.996,
        batch_size_per_gpu=batch,
        accum_steps=accum,
        data_path="/tmp/none",
    )


def save_checkpoint(path, completed_epochs, args):
    torch.save(
        {
            "epoch": completed_epochs,
            "args": args,
            "student": {},
        },
        path,
    )


def write_log(path, epochs, args, world_size=2, lr_values=None):
    args_dict = vars(args)
    lines = []
    for index, epoch in enumerate(epochs):
        lr = (
            lr_values[index]
            if lr_values is not None
            else reconstruct_metric(
                args_dict, world_size, "lr", epoch + 0.5
            )
        )
        wd = reconstruct_metric(args_dict, world_size, "wd", epoch + 0.5)
        lines.append(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_lr": lr,
                    "train_wd": wd,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_extract_epoch_from_name():
    assert extract_epoch_from_name("checkpoint0180.pth") == 180
    assert extract_epoch_from_name("checkpoint 23.pth") == 23
    assert extract_epoch_from_name("checkpoint318.pth") == 318
    assert extract_epoch_from_name("checkpoint.pth") is None
    assert extract_epoch_from_name("model_final.pth") is None


def test_periodic_checkpoint_uses_real_dino_epoch_coordinates(tmp_path):
    path = tmp_path / "checkpoint0180.pth"
    save_checkpoint(path, 181, make_args(300))
    record = load_checkpoint_record(path)

    assert record["filename_epoch"] == 180
    assert record["completed_epochs"] == 181
    assert record["log_epoch_index"] == 180
    assert record["coordinate_status"] == "consistent"
    assert record["usable_for_schedule"] is True


def test_filename_internal_epoch_mismatch_is_not_usable(tmp_path):
    path = tmp_path / "checkpoint0180.pth"
    save_checkpoint(path, 180, make_args(300))
    record = load_checkpoint_record(path)

    assert record["coordinate_status"] == "filename_internal_mismatch"
    assert record["usable_for_schedule"] is False


def test_plain_checkpoint_uses_internal_completed_epoch(tmp_path):
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(path, 318, make_args(350))
    record = load_checkpoint_record(path)

    assert record["filename_epoch"] is None
    assert record["completed_epochs"] == 318
    assert record["log_epoch_index"] == 317
    assert record["coordinate_status"] == "generic_internal_epoch"


def test_log_sessions_preserve_sources_and_locate_resume_boundary(tmp_path):
    args_a = make_args(200)
    args_b = make_args(300)
    save_checkpoint(tmp_path / "checkpoint0180.pth", 181, args_a)
    save_checkpoint(tmp_path / "checkpoint0250.pth", 251, args_b)

    old_log = tmp_path / "old" / "log.txt"
    new_log = tmp_path / "new" / "log.txt"
    old_log.parent.mkdir()
    new_log.parent.mkdir()
    write_log(old_log, range(170, 200), args_a)
    write_log(new_log, range(200, 251), args_b)

    report = run_audit(
        str(tmp_path),
        [str(old_log), str(new_log)],
        str(tmp_path / "out"),
        [2],
    )

    assert len(report["log_sessions"]) == 2
    assert report["log_sessions"][0]["source"] == str(old_log)
    assert report["log_sessions"][0]["first_log_epoch_index"] == 170
    boundary = report["boundaries"][0]
    assert boundary["boundary_epoch"] == 200
    assert boundary["boundary_interval"] == [200, 200]
    assert boundary["boundary_evidence_status"] == "sufficient"
    assert boundary["boundary_sources"] == [f"{new_log}#session-0"]
    assert report["log_coverage"]["status"] == "complete"
    assert [
        row["status"] for row in report["log_coverage"]["segments"]
    ] == ["complete", "complete"]
    assert report["verdict"]["status"] == "stitched"
    assert report["verdict"]["evidence_status"] == "sufficient"
    assert report["verdict"]["schedule_identity_changed"] is True


def test_identity_change_is_stitched_even_without_large_value_jump(tmp_path):
    args_a = make_args(300)
    args_b = make_args(310)
    save_checkpoint(tmp_path / "checkpoint0150.pth", 151, args_a)
    save_checkpoint(tmp_path / "checkpoint0200.pth", 201, args_b)

    old_log = tmp_path / "old.log"
    new_log = tmp_path / "new.log"
    write_log(old_log, range(140, 180), args_a)
    write_log(new_log, range(180, 201), args_b)

    report = run_audit(
        str(tmp_path),
        [str(old_log), str(new_log)],
        str(tmp_path / "out"),
        [2],
    )
    boundary = report["boundaries"][0]

    assert boundary["boundary_epoch"] == 180
    assert boundary["boundary_value_jump"] is False
    assert report["verdict"]["status"] == "stitched"
    assert report["verdict"]["schedule_identity_changed"] is True
    assert report["verdict"]["boundary_value_jump"] is False


def test_missing_session_boundary_reports_interval_not_guessed_epoch(tmp_path):
    save_checkpoint(
        tmp_path / "checkpoint0180.pth", 181, make_args(200)
    )
    save_checkpoint(
        tmp_path / "checkpoint0250.pth", 251, make_args(300)
    )

    report = run_audit(
        str(tmp_path), [], str(tmp_path / "out"), [1, 2]
    )
    boundary = report["boundaries"][0]

    assert boundary["boundary_epoch"] is None
    assert boundary["boundary_interval"] == [181, 250]
    assert boundary["boundary_evidence_status"] == "insufficient"
    assert boundary["boundary_value_jump"] is None
    assert report["verdict"]["status"] == "stitched"
    assert report["verdict"]["evidence_status"] == "partial"


def test_session_start_not_covering_new_checkpoint_is_not_exact_boundary(
    tmp_path,
):
    args_a = make_args(200)
    args_b = make_args(300)
    save_checkpoint(tmp_path / "checkpoint0180.pth", 181, args_a)
    save_checkpoint(tmp_path / "checkpoint0250.pth", 251, args_b)
    truncated_log = tmp_path / "truncated.log"
    write_log(truncated_log, range(200, 221), args_b)

    report = run_audit(
        str(tmp_path),
        [str(truncated_log)],
        str(tmp_path / "out"),
        [2],
    )
    boundary = report["boundaries"][0]

    assert boundary["boundary_epoch"] is None
    assert boundary["boundary_interval"] == [181, 250]
    assert boundary["boundary_evidence_status"] == "insufficient"


def test_insufficient_evidence_is_unknown_not_false_continuous(tmp_path):
    save_checkpoint(
        tmp_path / "checkpoint0180.pth", 181, make_args(300)
    )
    report = run_audit(
        str(tmp_path), [], str(tmp_path / "out"), [1, 2]
    )

    assert report["verdict"]["status"] == "unknown"
    assert report["verdict"]["evidence_status"] == "partial"
    assert report["verdict"]["schedule_identity_changed"] is None
    assert report["verdict"]["boundary_value_jump"] is None
    assert report["verdict"]["discontinuities_detected"] is None


def test_unrelated_log_does_not_make_single_identity_continuous(tmp_path):
    args = make_args(300)
    save_checkpoint(tmp_path / "checkpoint0100.pth", 101, args)
    save_checkpoint(tmp_path / "checkpoint0120.pth", 121, args)
    unrelated_log = tmp_path / "unrelated.log"
    write_log(unrelated_log, range(500, 503), args)

    report = run_audit(
        str(tmp_path),
        [str(unrelated_log)],
        str(tmp_path / "out"),
        [2],
    )

    assert report["segments"][0]["world_size_inference"]["matched_epochs"] == 0
    assert report["verdict"]["status"] == "unknown"
    assert report["verdict"]["evidence_status"] == "partial"


def test_same_identity_with_sufficient_evidence_is_continuous(tmp_path):
    args = make_args(300)
    save_checkpoint(tmp_path / "checkpoint0100.pth", 101, args)
    save_checkpoint(tmp_path / "checkpoint0120.pth", 121, args)
    log_path = tmp_path / "log.txt"
    write_log(log_path, range(80, 121), args)

    report = run_audit(
        str(tmp_path),
        [str(log_path)],
        str(tmp_path / "out"),
        [2],
    )

    assert len(report["segments"]) == 1
    assert report["verdict"]["status"] == "continuous"
    assert report["verdict"]["evidence_status"] == "sufficient"
    assert report["verdict"]["schedule_identity_changed"] is False
    assert report["verdict"]["boundary_value_jump"] is False


def test_short_log_cannot_prove_full_checkpoint_horizon_continuous(tmp_path):
    args = make_args(400)
    save_checkpoint(tmp_path / "checkpoint0180.pth", 181, args)
    save_checkpoint(tmp_path / "checkpoint0318.pth", 319, args)
    log_path = tmp_path / "short.log"
    write_log(log_path, [180, 181], args)

    report = run_audit(
        str(tmp_path),
        [str(log_path)],
        str(tmp_path / "out"),
        [2],
    )
    coverage = report["log_coverage"]["segments"][0]

    assert coverage["required_interval"] == [180, 318]
    assert coverage["covered_intervals"] == [[180, 181]]
    assert coverage["gaps"] == [[182, 318]]
    assert coverage["status"] == "partial"
    assert report["verdict"]["status"] == "unknown"
    assert report["verdict"]["evidence_status"] == "partial"


def test_full_log_coverage_proves_supplied_checkpoint_horizon_continuous(
    tmp_path,
):
    args = make_args(400)
    save_checkpoint(tmp_path / "checkpoint0180.pth", 181, args)
    save_checkpoint(tmp_path / "checkpoint0318.pth", 319, args)
    log_path = tmp_path / "full.log"
    write_log(log_path, range(180, 319), args)

    report = run_audit(
        str(tmp_path),
        [str(log_path)],
        str(tmp_path / "out"),
        [2],
    )
    coverage = report["log_coverage"]["segments"][0]

    assert report["log_coverage"]["status"] == "complete"
    assert coverage["required_interval"] == [180, 318]
    assert coverage["covered_intervals"] == [[180, 318]]
    assert coverage["gaps"] == []
    assert coverage["status"] == "complete"
    assert report["verdict"]["status"] == "continuous"
    assert report["verdict"]["evidence_status"] == "sufficient"


def test_distant_sessions_report_unexplained_coverage_gap(tmp_path):
    args = make_args(400)
    save_checkpoint(tmp_path / "checkpoint0180.pth", 181, args)
    save_checkpoint(tmp_path / "checkpoint0318.pth", 319, args)
    first_log = tmp_path / "first.log"
    last_log = tmp_path / "last.log"
    write_log(first_log, range(180, 201), args)
    write_log(last_log, range(300, 319), args)

    report = run_audit(
        str(tmp_path),
        [str(first_log), str(last_log)],
        str(tmp_path / "out"),
        [2],
    )
    coverage = report["log_coverage"]["segments"][0]

    assert coverage["covered_intervals"] == [[180, 200], [300, 318]]
    assert coverage["gaps"] == [[201, 299]]
    assert coverage["status"] == "partial"
    assert report["verdict"]["status"] == "unknown"
    assert report["verdict"]["evidence_status"] == "partial"


def test_adjacent_sessions_form_continuous_coverage(tmp_path):
    args = make_args(400)
    save_checkpoint(tmp_path / "checkpoint0180.pth", 181, args)
    save_checkpoint(tmp_path / "checkpoint0318.pth", 319, args)
    first_log = tmp_path / "first.log"
    second_log = tmp_path / "second.log"
    write_log(first_log, range(180, 251), args)
    write_log(second_log, range(251, 319), args)

    report = run_audit(
        str(tmp_path),
        [str(first_log), str(second_log)],
        str(tmp_path / "out"),
        [2],
    )
    coverage = report["log_coverage"]["segments"][0]

    assert coverage["covered_intervals"] == [[180, 318]]
    assert coverage["gaps"] == []
    assert coverage["status"] == "complete"
    assert len(coverage["session_intervals"]) == 2
    assert report["verdict"]["status"] == "continuous"
    assert report["verdict"]["evidence_status"] == "sufficient"


def test_lr_reversal_detection_never_compares_across_sessions(tmp_path):
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    write_log(
        first,
        [20, 21],
        make_args(300),
        lr_values=[1e-4, 9e-5],
    )
    write_log(
        second,
        [22, 23],
        make_args(300),
        lr_values=[1e-3, 9e-4],
    )
    sessions, issues = parse_log_files([str(first), str(second)])

    assert issues == []
    assert detect_lr_reversals(sessions) == []

    write_log(
        second,
        [22, 23],
        make_args(300),
        lr_values=[1e-4, 2e-4],
    )
    sessions, _ = parse_log_files([str(first), str(second)])
    reversals = detect_lr_reversals(sessions)
    assert len(reversals) == 1
    assert reversals[0]["source"] == str(second)
    assert reversals[0]["log_epoch_index"] == 23


def test_world_size_inference_reports_fit_quality(tmp_path):
    args = make_args(300)
    log_path = tmp_path / "log.txt"
    write_log(log_path, range(20, 61), args, world_size=2)
    sessions, _ = parse_log_files([str(log_path)])

    save_checkpoint(tmp_path / "checkpoint0060.pth", 61, args)
    segments = build_segments(
        [load_checkpoint_record(tmp_path / "checkpoint0060.pth")]
    )
    segments[0]["confident_log_epoch_interval"] = [20, 60]
    inference = infer_world_size(segments[0], sessions, [1, 2])

    assert inference["world_size"] == 2
    assert inference["status"] == "inferred"
    assert inference["confidence"] == "high"
    assert inference["candidates"][0]["relative_residual"] < 1e-12
    assert inference["candidate_margin"] > 0.02
    assert "microbatch mean" in inference["approximation"]


def test_low_quality_world_size_fit_does_not_force_candidate(tmp_path):
    args = make_args(300)
    log_path = tmp_path / "log.txt"
    write_log(
        log_path,
        range(20, 31),
        args,
        lr_values=[0.123] * 11,
    )
    sessions, _ = parse_log_files([str(log_path)])
    segment = {
        "args": vars(args),
        "confident_log_epoch_interval": [20, 30],
    }
    inference = infer_world_size(segment, sessions, [1, 2])

    assert inference["world_size"] is None
    assert inference["status"] == "low_quality"
    assert [row["world_size"] for row in inference["candidates"]] == [2, 1]


def test_ambiguous_world_size_fit_keeps_all_candidates(tmp_path):
    args = make_args(300)
    args_dict = vars(args)
    epoch = 299
    lr_one = reconstruct_metric(args_dict, 1, "lr", epoch + 0.5)
    lr_two = reconstruct_metric(args_dict, 2, "lr", epoch + 0.5)
    log_path = tmp_path / "log.txt"
    write_log(
        log_path,
        [epoch],
        args,
        lr_values=[(lr_one + lr_two) / 2.0],
    )
    sessions, _ = parse_log_files([str(log_path)])
    segment = {
        "args": args_dict,
        "confident_log_epoch_interval": [epoch, epoch],
    }
    inference = infer_world_size(segment, sessions, [1, 2])

    assert inference["world_size"] is None
    assert inference["status"] == "ambiguous"
    assert len(inference["candidates"]) == 2


def test_repeated_epochs_split_one_log_file_into_sessions(tmp_path):
    path = tmp_path / "log.txt"
    rows = [
        {"epoch": 10, "train_lr": 1e-4},
        {"epoch": 11, "train_lr": 9e-5},
        {"epoch": 11, "train_lr": 8e-5},
        {"epoch": 12, "train_lr": 7e-5},
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    sessions, issues = parse_log_files([str(path)])

    assert issues == []
    assert len(sessions) == 2
    assert sessions[0]["first_log_epoch_index"] == 10
    assert sessions[0]["last_log_epoch_index"] == 11
    assert sessions[1]["first_log_epoch_index"] == 11
    assert sessions[1]["last_log_epoch_index"] == 12


def test_assign_confident_intervals_excludes_unknown_boundary_gap():
    segments = [
        {
            "first_checkpoint_log_epoch_index": 180,
            "last_checkpoint_log_epoch_index": 180,
        },
        {
            "first_checkpoint_log_epoch_index": 250,
            "last_checkpoint_log_epoch_index": 250,
        },
    ]
    boundaries = [
        {
            "boundary_epoch": None,
            "boundary_interval": [181, 250],
        }
    ]
    assign_confident_log_intervals(segments, boundaries, [])

    assert segments[0]["confident_log_epoch_interval"] == [180, 180]
    assert segments[1]["confident_log_epoch_interval"] == [250, 250]
