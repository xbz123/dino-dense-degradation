import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from make_summary_report import (
    read_protocol_voc_rows,
    voc_provenance_lines,
    voc_section,
)


def test_report_reader_validates_v2_protocol(tmp_path):
    path = tmp_path / "voc_v2.json"
    path.write_text(json.dumps([
        {
            "epoch": 180,
            "miou": 30.8,
            "metric_version": "global_confusion_v2",
            "probe_seed": 42,
            "checkpoint_key": "teacher",
            "representation": "ema_teacher",
            "checkpoint": "/checkpoints/checkpoint0180.pth",
            "checkpoint_sha256": "1" * 64,
            "probe_config": {"train_epochs": 15},
            "dataset_identity": {"name": "fixture"},
            "source_commit": "a" * 40,
            "source_dirty": False,
        }
    ]))

    rows = read_protocol_voc_rows(
        path,
        protocol="v2",
        metric_version="global_confusion_v2",
        probe_seed=42,
        checkpoint_key="teacher",
    )
    assert rows[0]["miou"] == 30.8
    provenance = "\n".join(voc_provenance_lines(rows, protocol="v2"))
    assert "VOC source commit" in provenance
    assert "VOC checkpoint hashes recorded: `1`" in provenance

    with pytest.raises(ValueError, match="requires"):
        read_protocol_voc_rows(
            path,
            protocol="v2",
            metric_version=None,
            probe_seed=None,
            checkpoint_key=None,
        )


def test_report_legacy_mode_is_explicitly_historical(tmp_path):
    path = tmp_path / "voc_legacy.json"
    path.write_text(json.dumps([
        {"epoch": 180, "miou": 30.8},
        {"epoch": 318, "miou": 29.7},
    ]))

    rows = read_protocol_voc_rows(
        path,
        protocol="legacy",
        metric_version=None,
        probe_seed=None,
        checkpoint_key=None,
    )
    section = "\n".join(voc_section(rows, protocol="legacy"))
    assert "Historical batch-mean-v1 evidence only" in section
    assert "not eligible" in section


def test_report_legacy_mode_rejects_v2_expectations(tmp_path):
    path = tmp_path / "voc_legacy.json"
    path.write_text(json.dumps([{"epoch": 180, "miou": 30.8}]))

    with pytest.raises(ValueError, match="does not accept"):
        read_protocol_voc_rows(
            path,
            protocol="legacy",
            metric_version="global_confusion_v2",
            probe_seed=42,
            checkpoint_key="teacher",
        )


def test_report_legacy_mode_rejects_v2_rows(tmp_path):
    path = tmp_path / "voc_v2.json"
    path.write_text(json.dumps([
        {
            "epoch": 180,
            "miou": 30.8,
            "metric_version": "global_confusion_v2",
            "probe_seed": 42,
            "checkpoint_key": "teacher",
        }
    ]))

    with pytest.raises(ValueError, match="accepts only"):
        read_protocol_voc_rows(
            path,
            protocol="legacy",
            metric_version=None,
            probe_seed=None,
            checkpoint_key=None,
        )
