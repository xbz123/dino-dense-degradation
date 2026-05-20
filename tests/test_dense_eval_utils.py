from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dense_eval_utils import (
    build_run_output_root,
    discover_checkpoint_files,
    parse_checkpoint_epoch,
)


def _write_checkpoint(path: Path, epoch: int):
    torch.save({"epoch": epoch, "teacher": {}}, path)


def test_parse_checkpoint_epoch_handles_mixed_checkpoint_names():
    assert parse_checkpoint_epoch("checkpoint03.pth") == 3
    assert parse_checkpoint_epoch("checkpoint0020.pth") == 20
    assert parse_checkpoint_epoch("checkpoint0210.pth") == 210
    assert parse_checkpoint_epoch("checkpoint215.pth") == 215
    assert parse_checkpoint_epoch("checkpoint.pth") is None
    assert parse_checkpoint_epoch("not_a_checkpoint.pth") is None


def test_discover_checkpoint_files_sorts_filters_and_prefers_named_files(tmp_path):
    _write_checkpoint(tmp_path / "checkpoint03.pth", 3)
    _write_checkpoint(tmp_path / "checkpoint0020.pth", 21)
    _write_checkpoint(tmp_path / "checkpoint0200.pth", 201)
    _write_checkpoint(tmp_path / "checkpoint215.pth", 216)
    _write_checkpoint(tmp_path / "checkpoint.pth", 216)
    (tmp_path / ".DS_Store").write_text("")
    (tmp_path / "notes.txt").write_text("")

    discovered = discover_checkpoint_files(tmp_path)

    assert [item.epoch for item in discovered] == [3, 20, 200, 215]
    assert [item.path.name for item in discovered] == [
        "checkpoint03.pth",
        "checkpoint0020.pth",
        "checkpoint0200.pth",
        "checkpoint215.pth",
    ]
    assert discovered[-1].internal_epoch == 216


def test_discover_checkpoint_files_can_filter_epochs(tmp_path):
    for epoch in [20, 30, 40, 215]:
        _write_checkpoint(tmp_path / f"checkpoint{epoch:04d}.pth", epoch + 1)

    discovered = discover_checkpoint_files(tmp_path, epoch_filter=[30, 215])

    assert [item.epoch for item in discovered] == [30, 215]


def test_build_run_output_root_uses_final_checkpoint_epoch(tmp_path):
    output = build_run_output_root(tmp_path, [3, 20, 215])

    assert output == tmp_path / "to_epoch_0215"
