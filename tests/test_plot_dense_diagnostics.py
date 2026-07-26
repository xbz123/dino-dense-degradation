from pathlib import Path
import sys
import json

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dense_results_io import read_csv_rows
from plot_dense_diagnostics import (
    plot_raw_l2_figures,
    read_protocol_voc_results,
    write_combined_summary,
)


def test_plot_raw_l2_figures_writes_required_outputs(tmp_path):
    rows = [
        {
            "epoch": 1,
            "raw_dse": 10.0,
            "l2_dse": 8.0,
            "raw_class_sep_avg": -4.0,
            "l2_class_sep_avg": -1.0,
            "raw_effective_rank": 10.0,
            "l2_effective_rank": 5.0,
            "raw_top1_eigen_ratio": 0.4,
            "l2_top1_eigen_ratio": 0.2,
        },
        {
            "epoch": 2,
            "raw_dse": 6.0,
            "l2_dse": 9.0,
            "raw_class_sep_avg": -8.0,
            "l2_class_sep_avg": -1.5,
            "raw_effective_rank": 20.0,
            "l2_effective_rank": 4.0,
            "raw_top1_eigen_ratio": 0.2,
            "l2_top1_eigen_ratio": 0.3,
        },
    ]

    written = plot_raw_l2_figures(rows, tmp_path)

    assert [path.name for path in written] == [
        "fig_raw_vs_l2_dse.png",
        "fig_raw_vs_l2_class_sep.png",
        "fig_raw_vs_l2_spectrum.png",
    ]
    assert all(path.is_file() for path in written)


def test_combined_summary_preserves_raw_l2_columns(tmp_path):
    rows = [
        {
            "epoch": 1,
            "raw_dse": 10.0,
            "l2_dse": 8.0,
        }
    ]
    path = tmp_path / "combined.csv"

    write_combined_summary(
        path,
        rows,
        {1: 33.0},
        voc_metric_version="global_confusion_v2",
        voc_probe_seed=42,
        voc_checkpoint_key="teacher",
    )

    reloaded = read_csv_rows(path)
    assert reloaded[0]["voc_miou"] == 33.0
    assert reloaded[0]["voc_metric_version"] == "global_confusion_v2"
    assert reloaded[0]["voc_probe_seed"] == 42
    assert reloaded[0]["voc_checkpoint_key"] == "teacher"
    assert reloaded[0]["raw_dse"] == 10.0
    assert reloaded[0]["l2_dse"] == 8.0


def test_v2_reader_requires_and_validates_protocol(tmp_path):
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

    with pytest.raises(ValueError, match="requires"):
        read_protocol_voc_results(
            path,
            protocol="v2",
            metric_version=None,
            probe_seed=None,
            checkpoint_key=None,
        )

    values, metadata = read_protocol_voc_results(
        path,
        protocol="v2",
        metric_version="global_confusion_v2",
        probe_seed=42,
        checkpoint_key="teacher",
    )
    assert values == {180: 30.8}
    assert metadata == {
        "metric_version": "global_confusion_v2",
        "probe_seed": 42,
        "checkpoint_key": "teacher",
    }


def test_legacy_reader_requires_explicit_legacy_mode(tmp_path):
    path = tmp_path / "voc_legacy.json"
    path.write_text(json.dumps([{"epoch": 180, "miou": 30.8}]))

    with pytest.raises(ValueError, match="metric_version"):
        read_protocol_voc_results(
            path,
            protocol="v2",
            metric_version="global_confusion_v2",
            probe_seed=42,
            checkpoint_key="teacher",
        )

    values, metadata = read_protocol_voc_results(
        path,
        protocol="legacy",
        metric_version=None,
        probe_seed=None,
        checkpoint_key=None,
    )
    assert values == {180: 30.8}
    assert metadata["metric_version"] == "batch_mean_v1"


def test_legacy_reader_rejects_v2_rows(tmp_path):
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
        read_protocol_voc_results(
            path,
            protocol="legacy",
            metric_version=None,
            probe_seed=None,
            checkpoint_key=None,
        )
