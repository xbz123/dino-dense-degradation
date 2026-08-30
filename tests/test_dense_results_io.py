from pathlib import Path
import sys
import json

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dense_results_io import read_csv_rows, read_voc_results, to_number


def _checkpoint_identity(epoch=180):
    return {
        "basename": f"checkpoint{epoch:04d}.pth",
        "size_bytes": 123,
        "completed_epochs": epoch + 1,
        "training_config": {
            "schedule": {"epochs": 800},
            "model": {"arch": "vit_small", "patch_size": 16},
            "seed": 0,
        },
    }


def test_to_number_parses_numeric_strings_and_preserves_text():
    assert to_number("12") == 12
    assert to_number("12.5") == 12.5
    assert to_number("") != to_number("")
    assert to_number("epoch") == "epoch"


def test_read_csv_rows_sorts_by_epoch(tmp_path):
    path = tmp_path / "summary.csv"
    path.write_text("epoch,value\n30,1.5\n20,1.0\n")

    rows = read_csv_rows(path)

    assert [row["epoch"] for row in rows] == [20, 30]
    assert rows[0]["value"] == 1.0


def test_read_voc_results_can_return_mapping_or_rows(tmp_path):
    path = tmp_path / "voc.json"
    path.write_text(json.dumps([{"epoch": 30, "miou": 31.0}, {"epoch": 20, "miou": 30.0}]))

    assert read_voc_results(path) == {20: 30.0, 30: 31.0}
    assert [row["epoch"] for row in read_voc_results(path, as_rows=True)] == [20, 30]


def test_read_voc_results_enforces_requested_v2_protocol(tmp_path):
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
            "checkpoint_identity": _checkpoint_identity(),
            "probe_config": {"train_epochs": 15},
            "dataset_identity": {"name": "fixture"},
            "source_commit": "a" * 40,
            "source_dirty": False,
        }
    ]))

    rows = read_voc_results(
        path,
        as_rows=True,
        expected_metric_version="global_confusion_v2",
        expected_probe_seed=42,
        expected_checkpoint_key="teacher",
        require_provenance=True,
    )
    assert rows[0]["epoch"] == 180

    with pytest.raises(ValueError, match="probe_seed"):
        read_voc_results(path, expected_probe_seed=1337)
    with pytest.raises(ValueError, match="checkpoint_key"):
        read_voc_results(path, expected_checkpoint_key="student")

    dirty_rows = json.loads(path.read_text())
    dirty_rows[0]["source_dirty"] = True
    path.write_text(json.dumps(dirty_rows))
    with pytest.raises(ValueError, match="dirty source tree"):
        read_voc_results(path, require_provenance=True)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("size_bytes", 0, "size_bytes"),
        ("size_bytes", "123", "size_bytes"),
        ("completed_epochs", 180, "completed_epochs"),
        ("training_config", [], "training_config"),
    ],
)
def test_read_voc_results_validates_checkpoint_identity(tmp_path, field, value, match):
    path = tmp_path / "voc_v2.json"
    row = {
        "epoch": 180,
        "miou": 30.8,
        "metric_version": "global_confusion_v2",
        "probe_seed": 42,
        "checkpoint_key": "teacher",
        "representation": "ema_teacher",
        "checkpoint": "/checkpoints/checkpoint0180.pth",
        "checkpoint_identity": _checkpoint_identity(),
        "probe_config": {"train_epochs": 15},
        "dataset_identity": {"name": "fixture"},
        "source_commit": "a" * 40,
        "source_dirty": False,
    }
    row["checkpoint_identity"][field] = value
    path.write_text(json.dumps([row]))

    with pytest.raises(ValueError, match=match):
        read_voc_results(path, require_provenance=True)


def test_read_voc_results_rejects_missing_checkpoint_identity_field(tmp_path):
    path = tmp_path / "voc_v2.json"
    row = {
        "epoch": 180,
        "miou": 30.8,
        "metric_version": "global_confusion_v2",
        "probe_seed": 42,
        "checkpoint_key": "teacher",
        "representation": "ema_teacher",
        "checkpoint": "/checkpoints/checkpoint0180.pth",
        "checkpoint_identity": _checkpoint_identity(),
        "probe_config": {"train_epochs": 15},
        "dataset_identity": {"name": "fixture"},
        "source_commit": "a" * 40,
        "source_dirty": False,
    }
    del row["checkpoint_identity"]["completed_epochs"]
    path.write_text(json.dumps([row]))

    with pytest.raises(ValueError, match="checkpoint_identity.*missing"):
        read_voc_results(path, require_provenance=True)


def test_read_voc_results_fails_closed_on_missing_v2_provenance(tmp_path):
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

    with pytest.raises(ValueError, match="provenance fields"):
        read_voc_results(path, require_provenance=True)


def test_read_voc_results_fails_closed_on_missing_v2_metadata(tmp_path):
    path = tmp_path / "legacy_voc.json"
    path.write_text(json.dumps([{"epoch": 180, "miou": 30.8}]))

    assert read_voc_results(path) == {180: 30.8}
    with pytest.raises(ValueError, match="metric_version"):
        read_voc_results(
            path,
            expected_metric_version="global_confusion_v2",
        )
