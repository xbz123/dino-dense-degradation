"""Shared helpers for dense degradation evaluation workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


_CHECKPOINT_RE = re.compile(r"^checkpoint0*(\d+)\.pth$")


@dataclass(frozen=True)
class CheckpointInfo:
    epoch: int
    path: Path
    internal_epoch: int | None
    size_mb: float
    priority: int
    mtime: float


def parse_checkpoint_epoch(name: str) -> int | None:
    """Return the checkpoint epoch encoded in a checkpoint filename."""
    match = _CHECKPOINT_RE.match(Path(name).name)
    if match is None:
        return None
    return int(match.group(1))


def read_checkpoint_internal_epoch(path: str | Path) -> int | None:
    """Read DINO's stored next-start epoch from a checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        return None
    epoch = checkpoint.get("epoch")
    return int(epoch) if epoch is not None else None


def _generic_checkpoint_epoch(path: Path) -> int | None:
    internal_epoch = read_checkpoint_internal_epoch(path)
    if internal_epoch is None:
        return None
    # DINO stores the next epoch to start from. After finishing epoch N,
    # checkpoint.pth usually contains epoch=N+1.
    return max(0, internal_epoch - 1)


def discover_checkpoint_files(
    checkpoint_dir: str | Path,
    epoch_filter: Iterable[int] | None = None,
) -> list[CheckpointInfo]:
    """Discover all recognizable checkpoint files in a directory.

    Epoch-named checkpoints such as ``checkpoint0210.pth`` are preferred over a
    generic ``checkpoint.pth`` when both resolve to the same epoch.
    """
    checkpoint_dir = Path(checkpoint_dir)
    wanted = set(epoch_filter) if epoch_filter is not None else None
    discovered: dict[int, CheckpointInfo] = {}

    for path in sorted(checkpoint_dir.iterdir()):
        if not path.is_file() or path.suffix != ".pth":
            continue

        priority = 2
        epoch = parse_checkpoint_epoch(path.name)
        if epoch is None:
            if path.name != "checkpoint.pth":
                continue
            epoch = _generic_checkpoint_epoch(path)
            priority = 1
            if epoch is None:
                continue

        if wanted is not None and epoch not in wanted:
            continue

        internal_epoch = read_checkpoint_internal_epoch(path)
        stat = path.stat()
        candidate = CheckpointInfo(
            epoch=epoch,
            path=path,
            internal_epoch=internal_epoch,
            size_mb=stat.st_size / 1024 / 1024,
            priority=priority,
            mtime=stat.st_mtime,
        )
        current = discovered.get(epoch)
        if current is None or (candidate.priority, candidate.mtime) > (
            current.priority,
            current.mtime,
        ):
            discovered[epoch] = candidate

    return [discovered[epoch] for epoch in sorted(discovered)]


def build_run_output_root(output_root: str | Path, epochs: Iterable[int]) -> Path:
    """Build the per-run output directory name from the largest epoch."""
    epochs = list(epochs)
    if not epochs:
        raise ValueError("Cannot build output root without checkpoint epochs")
    return Path(output_root) / f"to_epoch_{max(epochs):04d}"
