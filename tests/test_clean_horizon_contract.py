import argparse
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from clean_horizon_contract import (
    CleanHorizonContractError,
    build_training_contract,
    accumulation_group_size,
    capture_rank_rng_states,
    restore_rank_rng_state,
    optimizer_steps_per_epoch,
    should_stop_before_next_epoch,
    validate_milestone_epochs,
    validate_resume_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def make_args(**overrides):
    values = {
        "arch": "vit_small",
        "patch_size": 16,
        "out_dim": 65536,
        "norm_last_layer": False,
        "momentum_teacher": 0.996,
        "use_bn_in_head": False,
        "warmup_teacher_temp": 0.04,
        "teacher_temp": 0.07,
        "warmup_teacher_temp_epochs": 30,
        "use_fp16": True,
        "weight_decay": 0.04,
        "weight_decay_end": 0.4,
        "clip_grad": 3.0,
        "batch_size_per_gpu": 64,
        "epochs": 319,
        "freeze_last_layer": 1,
        "lr": 0.0005,
        "warmup_epochs": 10,
        "min_lr": 1e-6,
        "optimizer": "adamw",
        "drop_path_rate": 0.1,
        "global_crops_scale": (0.4, 1.0),
        "local_crops_number": 4,
        "local_crops_scale": (0.05, 0.4),
        "seed": 0,
        "num_workers": 2,
        "accum_steps": 2,
        "drop_incomplete_accumulation": True,
        "saveckp_freq": 10,
        "keep_last_ckpts": 0,
        "diag_every": 5,
        "attn_viz_every": 25,
        "diag_num_batches": 50,
        "milestone_ckpt_epochs": (180, 250, 318),
        "run_name": "dino_v3_clean_horizon_seed0_v1",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def make_contract(args=None):
    return build_training_contract(
        args or make_args(),
        dataset_size=126689,
        class_count=100,
        batches_per_epoch=989,
        optimizer_steps_per_epoch=494,
        world_size=2,
        source_state={"source_commit": "a" * 40, "source_dirty": False},
    )


def save_resume(path, contract, *, completed_epochs=181, rng_count=2, args=None):
    rng_state = capture_rank_rng_states()[0]
    torch.save(
        {
            "student": {},
            "teacher": {},
            "optimizer": {},
            "epoch": completed_epochs,
            "args": args or make_args(),
            "dino_loss": {},
            "fp16_scaler": {},
            "training_contract": contract,
            "rng_states": [rng_state for _ in range(rng_count)],
        },
        path,
    )


def test_contract_records_schedule_dataset_runtime_and_source():
    contract = make_contract()

    assert contract["training_config"]["epochs"] == 319
    assert contract["training_config"]["milestone_ckpt_epochs"] == [180, 250, 318]
    assert contract["dataset"] == {"image_count": 126689, "class_count": 100}
    assert contract["runtime"]["world_size"] == 2
    assert contract["source"]["source_dirty"] is False


def test_resume_accepts_exact_contract_and_coordinate(tmp_path):
    contract = make_contract()
    checkpoint = tmp_path / "checkpoint0180.pth"
    save_resume(checkpoint, contract)

    identity = validate_resume_checkpoint(checkpoint, contract, use_fp16=True)

    assert identity["completed_epochs"] == 181
    assert identity["filename_epoch"] == 180
    assert identity["size_bytes"] > 0


def test_resume_rejects_schedule_contract_change(tmp_path):
    checkpoint = tmp_path / "checkpoint0180.pth"
    save_resume(checkpoint, make_contract())
    changed = make_contract(make_args(epochs=500))

    with pytest.raises(CleanHorizonContractError, match="training contract"):
        validate_resume_checkpoint(checkpoint, changed, use_fp16=True)


def test_resume_rejects_checkpoint_args_that_disagree_with_contract(tmp_path):
    checkpoint = tmp_path / "checkpoint0180.pth"
    save_resume(checkpoint, make_contract(), args=make_args(teacher_temp=0.09))

    with pytest.raises(CleanHorizonContractError, match="teacher_temp"):
        validate_resume_checkpoint(checkpoint, make_contract(), use_fp16=True)


def test_resume_rejects_filename_internal_epoch_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint0180.pth"
    save_resume(checkpoint, make_contract(), completed_epochs=180)

    with pytest.raises(CleanHorizonContractError, match="coordinate mismatch"):
        validate_resume_checkpoint(checkpoint, make_contract(), use_fp16=True)


def test_resume_rejects_missing_rank_rng_state(tmp_path):
    checkpoint = tmp_path / "checkpoint0180.pth"
    save_resume(checkpoint, make_contract(), rng_count=1)

    with pytest.raises(CleanHorizonContractError, match="RNG state count"):
        validate_resume_checkpoint(checkpoint, make_contract(), use_fp16=True)


def test_resume_rejects_incomplete_rng_state(tmp_path):
    checkpoint = tmp_path / "checkpoint0180.pth"
    save_resume(checkpoint, make_contract())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    del payload["rng_states"][0]["torch_cpu"]
    torch.save(payload, checkpoint)

    with pytest.raises(CleanHorizonContractError, match="RNG state.*missing"):
        validate_resume_checkpoint(checkpoint, make_contract(), use_fp16=True)


def test_final_label_318_matches_319_completed_epochs(tmp_path):
    checkpoint = tmp_path / "checkpoint0318.pth"
    save_resume(checkpoint, make_contract(), completed_epochs=319)

    identity = validate_resume_checkpoint(checkpoint, make_contract(), use_fp16=True)

    assert identity["filename_epoch"] == 318
    assert identity["completed_epochs"] == 319


def test_rng_state_round_trip_is_replayable():
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    states = capture_rank_rng_states()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    random.random()
    np.random.rand()
    torch.rand(1)
    restore_rank_rng_state(states, 0)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert actual == pytest.approx(expected)


def test_runtime_guard_reserves_one_estimated_epoch():
    assert should_stop_before_next_epoch(
        elapsed_seconds=36_000,
        mean_epoch_seconds=2_400,
        max_runtime_seconds=41_400,
        reserve_seconds=2_700,
        completed_epochs=15,
        target_epochs=319,
    )


def test_accumulation_schedule_uses_only_full_groups_when_requested():
    assert optimizer_steps_per_epoch(989, 2, drop_incomplete=True) == 494
    assert accumulation_group_size(987, 989, 2, drop_incomplete=True) == 2
    with pytest.raises(ValueError, match="usable accumulation"):
        accumulation_group_size(988, 989, 2, drop_incomplete=True)


def test_accumulation_schedule_scales_a_partial_group_when_retained():
    assert optimizer_steps_per_epoch(989, 2, drop_incomplete=False) == 495
    assert accumulation_group_size(988, 989, 2, drop_incomplete=False) == 1
    assert not should_stop_before_next_epoch(
        elapsed_seconds=1_000,
        mean_epoch_seconds=2_400,
        max_runtime_seconds=41_400,
        reserve_seconds=2_700,
        completed_epochs=1,
        target_epochs=319,
    )


def test_milestones_must_fit_zero_based_horizon():
    validate_milestone_epochs(make_args())
    with pytest.raises(CleanHorizonContractError, match="must be in"):
        validate_milestone_epochs(make_args(milestone_ckpt_epochs=(319,)))


def test_kaggle_launcher_freezes_clean_horizon_contract():
    source = (ROOT / "run_clean_horizon_kaggle.sh").read_text(encoding="utf-8")

    assert "--epochs 319" in source
    assert "--milestone_ckpt_epochs 180 250 318" in source
    assert "--strict_resume_schedule true" in source
    assert "--drop_incomplete_accumulation true" in source
    assert "--expected_world_size 2" in source
    assert "--seed 0" in source
    assert "--saveckp_freq 10" in source
    assert "--keep_last_ckpts 0" in source


def test_training_loop_fails_closed_on_distributed_nonfinite_or_amp_skip():
    source = (ROOT / "main_dino.py").read_text(encoding="utf-8")

    assert "dist.all_reduce(nonfinite_loss, op=dist.ReduceOp.MAX)" in source
    assert "AMP overflow skipped an optimizer step" in source
    assert "if optimizer_stepped:" in source
    assert "optimizer_steps * epoch + it // accum_steps" in source
