import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_coco_stuff_dense import (
    COCOStuffSegDataset,
    filter_checkpoints_by_epoch,
    parse_epoch_filter,
    write_comparison_csv,
)


def _write_sample(root: Path, split: str, stem: str, mask_values: np.ndarray) -> None:
    image_dir = root / "images" / f"{split}2017"
    mask_dir = root / "annotations" / f"{split}2017"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((mask_values.shape[0], mask_values.shape[1], 3), dtype=np.uint8)
    image[..., 0] = 120
    image[..., 1] = 80
    image[..., 2] = 40
    Image.fromarray(image).save(image_dir / f"{stem}.jpg")
    Image.fromarray(mask_values.astype(np.uint8)).save(mask_dir / f"{stem}.png")


def test_coco_stuff_dataset_pairs_images_and_masks_with_nearest_resize(tmp_path):
    mask = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    _write_sample(tmp_path, "train", "000000000001", mask)

    dataset = COCOStuffSegDataset(
        tmp_path,
        split="train",
        img_size=32,
        patch_size=16,
        num_classes=3,
    )

    image, target = dataset[0]

    assert len(dataset) == 1
    assert tuple(image.shape) == (3, 32, 32)
    assert tuple(target.shape) == (32, 32)
    assert set(target.unique().tolist()) == {0, 1, 2, 255}


def test_coco_stuff_dataset_maps_out_of_range_labels_to_ignore(tmp_path):
    mask = np.array([[0, 1], [200, 255]], dtype=np.uint8)
    _write_sample(tmp_path, "val", "000000000002", mask)

    dataset = COCOStuffSegDataset(
        tmp_path,
        split="val",
        img_size=32,
        patch_size=16,
        num_classes=2,
    )

    _, target = dataset[0]

    assert 200 not in target.unique().tolist()
    assert 255 in target.unique().tolist()


def test_parse_epoch_filter_accepts_commas_and_spaces():
    assert parse_epoch_filter(None) is None
    assert parse_epoch_filter("") is None
    assert parse_epoch_filter("50,80 180") == [50, 80, 180]


def test_filter_checkpoints_by_epoch_requires_all_requested_epochs():
    checkpoints = [(50, "/tmp/checkpoint0050.pth"), (80, "/tmp/checkpoint0080.pth")]

    assert filter_checkpoints_by_epoch(checkpoints, [80]) == [(80, "/tmp/checkpoint0080.pth")]
    with pytest.raises(ValueError, match="180"):
        filter_checkpoints_by_epoch(checkpoints, [80, 180])


def test_write_comparison_csv_merges_coco_voc_and_dense_metrics(tmp_path):
    voc_path = tmp_path / "voc_miou_results.json"
    dense_path = tmp_path / "combined_dense_summary.csv"
    output_path = tmp_path / "comparison.csv"

    voc_path.write_text(json.dumps([{"epoch": 180, "miou": 30.8}]))
    with dense_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "raw_dse",
                "l2_dse",
                "raw_class_sep",
                "l2_class_sep",
                "effective_rank",
                "top1_eigen_ratio",
                "patch_norm_mean",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": 180,
                "raw_dse": "140.0",
                "l2_dse": "0.32",
                "raw_class_sep": "-10.0",
                "l2_class_sep": "-2.0",
                "effective_rank": "300.0",
                "top1_eigen_ratio": "0.04",
                "patch_norm_mean": "42.0",
            }
        )

    write_comparison_csv(
        coco_results=[{"epoch": 180, "miou": 29.5}],
        voc_json_path=voc_path,
        dense_summary_csv_path=dense_path,
        output_path=output_path,
    )

    rows = list(csv.DictReader(output_path.open()))

    assert rows == [
        {
            "epoch": "180",
            "coco_stuff_miou": "29.5",
            "voc_miou": "30.8",
            "raw_dse": "140.0",
            "l2_dse": "0.32",
            "raw_class_sep": "-10.0",
            "l2_class_sep": "-2.0",
            "effective_rank": "300.0",
            "top1_eigen_ratio": "0.04",
            "patch_norm_mean": "42.0",
        }
    ]
