"""Fail-closed metadata and resume helpers for clean-horizon DINO runs."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


CONTRACT_VERSION = 2
AMP_OVERFLOW_STATE_VERSION = 1

TRAINING_CONFIG_FIELDS = (
    "arch",
    "patch_size",
    "out_dim",
    "norm_last_layer",
    "momentum_teacher",
    "use_bn_in_head",
    "warmup_teacher_temp",
    "teacher_temp",
    "warmup_teacher_temp_epochs",
    "use_fp16",
    "amp_max_consecutive_overflows",
    "weight_decay",
    "weight_decay_end",
    "clip_grad",
    "batch_size_per_gpu",
    "epochs",
    "freeze_last_layer",
    "lr",
    "warmup_epochs",
    "min_lr",
    "optimizer",
    "drop_path_rate",
    "global_crops_scale",
    "local_crops_number",
    "local_crops_scale",
    "seed",
    "num_workers",
    "accum_steps",
    "drop_incomplete_accumulation",
    "saveckp_freq",
    "keep_last_ckpts",
    "diag_every",
    "attn_viz_every",
    "diag_num_batches",
    "milestone_ckpt_epochs",
    "run_name",
)

REQUIRED_RESUME_KEYS = {
    "student",
    "teacher",
    "optimizer",
    "epoch",
    "args",
    "dino_loss",
    "training_contract",
    "rng_states",
    "amp_overflow_state",
}


class CleanHorizonContractError(RuntimeError):
    """Raised when a resume would change the frozen training horizon."""


def create_amp_overflow_state(max_consecutive_overflows: int) -> dict[str, int]:
    """Create checkpointable state for recoverable dynamic-loss-scaling skips."""
    if max_consecutive_overflows <= 0:
        raise ValueError("max_consecutive_overflows must be positive")
    return {
        "state_version": AMP_OVERFLOW_STATE_VERSION,
        "total_overflows": 0,
        "consecutive_overflows": 0,
        "max_consecutive_overflows": int(max_consecutive_overflows),
        "optimizer_step_attempts": 0,
        "optimizer_steps_applied": 0,
    }


def validate_amp_overflow_state(
    state: Any,
    *,
    expected_max_consecutive_overflows: int,
) -> dict[str, int]:
    """Validate and normalize AMP overflow state restored from a checkpoint."""
    if not isinstance(state, dict):
        raise CleanHorizonContractError("Checkpoint AMP overflow state is not a mapping")
    required = {
        "state_version",
        "total_overflows",
        "consecutive_overflows",
        "max_consecutive_overflows",
        "optimizer_step_attempts",
        "optimizer_steps_applied",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise CleanHorizonContractError(
            f"Checkpoint AMP overflow state is missing {missing}"
        )
    for key in required:
        value = state[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise CleanHorizonContractError(
                f"Checkpoint AMP overflow field {key!r} is not an integer"
            )
    normalized = {key: int(state[key]) for key in required}
    if normalized["state_version"] != AMP_OVERFLOW_STATE_VERSION:
        raise CleanHorizonContractError("Checkpoint AMP overflow state version differs")
    if normalized["max_consecutive_overflows"] != int(
        expected_max_consecutive_overflows
    ):
        raise CleanHorizonContractError(
            "Checkpoint AMP overflow threshold differs from the training contract"
        )
    if normalized["total_overflows"] < 0:
        raise CleanHorizonContractError("Checkpoint AMP total overflow count is negative")
    if not 0 <= normalized["consecutive_overflows"] <= normalized["total_overflows"]:
        raise CleanHorizonContractError(
            "Checkpoint AMP consecutive overflow count is invalid"
        )
    if (
        normalized["consecutive_overflows"]
        >= normalized["max_consecutive_overflows"]
    ):
        raise CleanHorizonContractError(
            "Checkpoint AMP overflow state has already reached the kill limit"
        )
    if not 0 <= normalized["optimizer_steps_applied"] <= normalized[
        "optimizer_step_attempts"
    ]:
        raise CleanHorizonContractError(
            "Checkpoint optimizer-step counters are invalid"
        )
    if (
        normalized["optimizer_step_attempts"]
        - normalized["optimizer_steps_applied"]
        != normalized["total_overflows"]
    ):
        raise CleanHorizonContractError(
            "Checkpoint optimizer-step counters disagree with AMP overflow total"
        )
    return normalized


def record_amp_optimizer_attempt(
    state: dict[str, int],
    *,
    overflowed: bool,
) -> bool:
    """Record one scheduled optimizer attempt and report whether the kill limit hit."""
    state["optimizer_step_attempts"] += 1
    if overflowed:
        state["total_overflows"] += 1
        state["consecutive_overflows"] += 1
    else:
        state["consecutive_overflows"] = 0
        state["optimizer_steps_applied"] += 1
    return (
        state["consecutive_overflows"]
        >= state["max_consecutive_overflows"]
    )


def optimizer_steps_per_epoch(
    num_batches: int,
    accum_steps: int,
    *,
    drop_incomplete: bool,
) -> int:
    if num_batches <= 0 or accum_steps <= 0:
        raise ValueError("num_batches and accum_steps must be positive")
    if drop_incomplete:
        return num_batches // accum_steps
    return (num_batches + accum_steps - 1) // accum_steps


def accumulation_group_size(
    iteration: int,
    num_batches: int,
    accum_steps: int,
    *,
    drop_incomplete: bool,
) -> int:
    steps = optimizer_steps_per_epoch(
        num_batches,
        accum_steps,
        drop_incomplete=drop_incomplete,
    )
    usable_batches = steps * accum_steps if drop_incomplete else num_batches
    if iteration < 0 or iteration >= usable_batches:
        raise ValueError("iteration is outside the usable accumulation range")
    group_start = (iteration // accum_steps) * accum_steps
    return min(accum_steps, usable_batches - group_start)


def _normalize(value: Any) -> Any:
    if isinstance(value, argparse.Namespace):
        return _normalize(vars(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def get_git_state(repo_dir: str | Path) -> dict[str, Any]:
    """Return source provenance without computing an artifact content hash."""
    repo_dir = Path(repo_dir)
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CleanHorizonContractError(
            f"Cannot establish source provenance in {repo_dir}: {exc}"
        ) from exc
    return {"source_commit": commit, "source_dirty": bool(status.strip())}


def build_training_contract(
    args: argparse.Namespace,
    *,
    dataset_size: int,
    class_count: int,
    batches_per_epoch: int,
    optimizer_steps_per_epoch: int,
    world_size: int,
    source_state: dict[str, Any],
) -> dict[str, Any]:
    """Build the immutable contract that every session must reproduce."""
    config = {
        field: _normalize(getattr(args, field))
        for field in TRAINING_CONFIG_FIELDS
    }
    return {
        "contract_version": CONTRACT_VERSION,
        "training_config": config,
        "dataset": {
            "image_count": int(dataset_size),
            "class_count": int(class_count),
        },
        "runtime": {
            "world_size": int(world_size),
            "batches_per_epoch": int(batches_per_epoch),
            "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
        },
        "source": {
            "source_commit": str(source_state["source_commit"]),
            "source_dirty": bool(source_state["source_dirty"]),
        },
    }


def validate_milestone_epochs(args: argparse.Namespace) -> None:
    milestones = list(args.milestone_ckpt_epochs)
    if len(milestones) != len(set(milestones)):
        raise CleanHorizonContractError("Milestone checkpoint epochs must be unique")
    invalid = [epoch for epoch in milestones if epoch < 0 or epoch >= args.epochs]
    if invalid:
        raise CleanHorizonContractError(
            f"Milestone epochs must be in [0, {args.epochs - 1}]: {invalid}"
        )


def _checkpoint_filename_epoch(path: str | Path) -> int | None:
    match = re.fullmatch(r"checkpoint0*(\d+)\.pth", Path(path).name)
    return int(match.group(1)) if match else None


def validate_resume_checkpoint(
    path: str | Path,
    expected_contract: dict[str, Any],
    *,
    use_fp16: bool,
) -> dict[str, Any]:
    """Load resume metadata and reject any contract or coordinate mismatch."""
    path = Path(path)
    if not path.is_file():
        raise CleanHorizonContractError(f"Resume checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise CleanHorizonContractError(f"Resume checkpoint is not a mapping: {path}")

    required = set(REQUIRED_RESUME_KEYS)
    if use_fp16:
        required.add("fp16_scaler")
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise CleanHorizonContractError(
            f"Resume checkpoint is missing required state: {missing}"
        )

    completed_epochs = checkpoint["epoch"]
    if isinstance(completed_epochs, bool) or not isinstance(completed_epochs, int):
        raise CleanHorizonContractError("Checkpoint epoch must be an integer")
    target_epochs = int(expected_contract["training_config"]["epochs"])
    if not 0 < completed_epochs <= target_epochs:
        raise CleanHorizonContractError(
            f"Checkpoint completed_epochs={completed_epochs} is outside (0, {target_epochs}]"
        )

    filename_epoch = _checkpoint_filename_epoch(path)
    if filename_epoch is not None and completed_epochs != filename_epoch + 1:
        raise CleanHorizonContractError(
            "Checkpoint coordinate mismatch: "
            f"filename label {filename_epoch}, internal completed epoch {completed_epochs}"
        )

    actual_contract = _normalize(checkpoint["training_contract"])
    expected_contract = _normalize(expected_contract)
    if actual_contract != expected_contract:
        differing_sections = [
            key
            for key in sorted(set(actual_contract) | set(expected_contract))
            if actual_contract.get(key) != expected_contract.get(key)
        ]
        raise CleanHorizonContractError(
            "Resume checkpoint training contract differs from this launch in: "
            f"{differing_sections}"
        )

    checkpoint_args = checkpoint["args"]
    if isinstance(checkpoint_args, argparse.Namespace):
        checkpoint_args = vars(checkpoint_args)
    if not isinstance(checkpoint_args, dict):
        raise CleanHorizonContractError("Checkpoint args are not a mapping")
    for field, expected_value in expected_contract["training_config"].items():
        if field not in checkpoint_args:
            raise CleanHorizonContractError(
                f"Checkpoint args are missing frozen field {field!r}"
            )
        if _normalize(checkpoint_args[field]) != expected_value:
            raise CleanHorizonContractError(
                f"Checkpoint args field {field!r} differs from the training contract"
            )

    rng_states = checkpoint["rng_states"]
    expected_world_size = int(expected_contract["runtime"]["world_size"])
    if not isinstance(rng_states, list) or len(rng_states) != expected_world_size:
        raise CleanHorizonContractError(
            "Checkpoint RNG state count does not match the frozen world size"
        )
    required_rng_fields = {"python", "numpy", "torch_cpu", "torch_cuda"}
    for rank, state in enumerate(rng_states):
        if not isinstance(state, dict):
            raise CleanHorizonContractError(
                f"Checkpoint RNG state for rank {rank} is not a mapping"
            )
        missing_rng = sorted(required_rng_fields.difference(state))
        if missing_rng:
            raise CleanHorizonContractError(
                f"Checkpoint RNG state for rank {rank} is missing {missing_rng}"
            )

    amp_overflow_state = validate_amp_overflow_state(
        checkpoint["amp_overflow_state"],
        expected_max_consecutive_overflows=int(
            expected_contract["training_config"]["amp_max_consecutive_overflows"]
        ),
    )
    expected_attempts = (
        completed_epochs
        * int(expected_contract["runtime"]["optimizer_steps_per_epoch"])
    )
    if amp_overflow_state["optimizer_step_attempts"] != expected_attempts:
        raise CleanHorizonContractError(
            "Checkpoint optimizer-step coordinate differs from completed epochs: "
            f"{amp_overflow_state['optimizer_step_attempts']} attempts, "
            f"expected {expected_attempts}"
        )

    identity = {
        "basename": path.name,
        "size_bytes": path.stat().st_size,
        "completed_epochs": completed_epochs,
        "filename_epoch": filename_epoch,
        "amp_overflow_state": amp_overflow_state,
    }
    del checkpoint
    return identity


def capture_rank_rng_states() -> list[dict[str, Any]]:
    """Capture one resumable RNG state per distributed rank."""
    local_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    if dist.is_available() and dist.is_initialized():
        states: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(states, local_state)
        return [state for state in states if state is not None]
    return [local_state]


def restore_rank_rng_state(states: list[dict[str, Any]], rank: int) -> None:
    """Restore the state saved for this rank before creating the next iterator."""
    if not isinstance(states, list) or rank < 0 or rank >= len(states):
        raise CleanHorizonContractError(
            f"No RNG state is available for rank {rank}"
        )
    state = states[rank]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise CleanHorizonContractError(
                "Checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state(state["torch_cuda"])


def should_stop_before_next_epoch(
    *,
    elapsed_seconds: float,
    mean_epoch_seconds: float,
    max_runtime_seconds: float,
    reserve_seconds: float,
    completed_epochs: int,
    target_epochs: int,
) -> bool:
    """Stop only after a completed epoch when another epoch risks the budget."""
    if completed_epochs >= target_epochs or max_runtime_seconds <= 0:
        return False
    next_epoch_allowance = max(300.0, mean_epoch_seconds * 1.25)
    return (
        elapsed_seconds + next_epoch_allowance + reserve_seconds
        >= max_runtime_seconds
    )


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_normalize(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
