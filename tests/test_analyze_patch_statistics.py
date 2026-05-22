from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_patch_statistics import (
    add_dse_scores,
    analyze_checkpoint,
    build_fixed_image_manifest,
    query_points_for_grid,
    select_metric_and_visual_indices,
)


def test_query_points_for_grid_returns_fixed_named_points():
    points = query_points_for_grid(14, 14)

    assert [point["name"] for point in points] == [
        "center",
        "upper_left",
        "upper_right",
        "lower_left",
        "lower_right",
    ]
    assert [point["index"] for point in points] == [105, 45, 52, 143, 150]


def test_metric_indices_are_sorted_but_visual_indices_keep_random_order():
    metric_indices, visual_indices = select_metric_and_visual_indices(
        dataset_size=100,
        num_metric_images=8,
        num_vis_images=3,
        seed=7,
    )
    expected_draw = np.random.default_rng(7).choice(100, size=8, replace=False).tolist()

    assert metric_indices == sorted(expected_draw)
    assert visual_indices == expected_draw[:3]
    assert visual_indices != sorted(visual_indices)


def test_fixed_image_manifest_records_paths_labels_and_order():
    dataset = SimpleNamespace(
        samples=[
            ("/data/class_a/a.png", 0),
            ("/data/class_b/b.png", 1),
            ("/data/class_a/c.png", 0),
        ],
        classes=["class_a", "class_b"],
    )

    manifest = build_fixed_image_manifest(dataset, [2, 1])

    assert manifest == [
        {
            "position": 0,
            "index": 2,
            "path": "/data/class_a/c.png",
            "class_index": 0,
            "class_name": "class_a",
        },
        {
            "position": 1,
            "index": 1,
            "path": "/data/class_b/b.png",
            "class_index": 1,
            "class_name": "class_b",
        },
    ]


def test_add_dse_scores_populates_raw_and_l2_tracks():
    rows = [
        {
            "epoch": 1,
            "class_sep_avg": -4.0,
            "effective_rank": 10.0,
            "raw_class_sep_avg": -4.0,
            "raw_effective_rank": 10.0,
            "l2_class_sep_avg": -1.0,
            "l2_effective_rank": 5.0,
        },
        {
            "epoch": 2,
            "class_sep_avg": -8.0,
            "effective_rank": 20.0,
            "raw_class_sep_avg": -8.0,
            "raw_effective_rank": 20.0,
            "l2_class_sep_avg": -2.0,
            "l2_effective_rank": 9.0,
        },
    ]

    scored = add_dse_scores(rows)

    assert scored[0]["raw_dse_lambda"] > 0
    assert scored[0]["l2_dse_lambda"] > 0
    assert scored[0]["raw_dse"] == scored[0]["dse"]
    assert scored[1]["raw_dse"] == scored[1]["dse"]
    assert scored[0]["l2_dse"] != scored[0]["raw_dse"]


def test_strict_internal_epoch_validation_happens_before_feature_extraction(tmp_path, monkeypatch):
    import analyze_patch_statistics as module

    info = SimpleNamespace(epoch=215, path=tmp_path / "checkpoint0215.pth", size_mb=1.0)
    args = SimpleNamespace(
        out=str(tmp_path),
        arch="vit_small",
        patch_size=16,
        checkpoint_key="teacher",
        strict_internal_epoch=True,
    )

    def fake_load_vit_backbone(*args, **kwargs):
        return object(), {
            "source": "teacher",
            "internal_epoch": 171,
            "missing_keys": 0,
            "unexpected_keys": 0,
        }

    def fail_if_called(*args, **kwargs):
        raise AssertionError("feature extraction should not run after strict epoch mismatch")

    monkeypatch.setattr(module, "load_vit_backbone", fake_load_vit_backbone)
    monkeypatch.setattr(module, "extract_checkpoint_features", fail_if_called)

    with pytest.raises(ValueError, match="internal epoch 171"):
        analyze_checkpoint(
            info,
            args,
            loader=object(),
            raw_dataset=object(),
            selected_indices=[0],
            vis_indices={0},
            baseline_query_maps=None,
            device="cpu",
        )
