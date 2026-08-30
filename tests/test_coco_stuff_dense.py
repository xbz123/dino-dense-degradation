import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_coco_stuff_dense import (
    COCOStuffSegDataset,
    filter_checkpoints_by_epoch,
    parse_epoch_filter,
    write_comparison_csv,
    write_summary,
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


def _write_image_mask_pair(
    image_dir: Path,
    mask_dir: Path,
    stem: str,
    mask_values: np.ndarray,
) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((mask_values.shape[0], mask_values.shape[1], 3), dtype=np.uint8)
    image[..., 0] = 120
    image[..., 1] = 80
    image[..., 2] = 40
    Image.fromarray(image).save(image_dir / f"{stem}.jpg")
    Image.fromarray(mask_values.astype(np.uint8)).save(mask_dir / f"{stem}.png")


def _write_image_mat_pair(
    image_dir: Path,
    mask_dir: Path,
    stem: str,
    mask_values: np.ndarray,
) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((mask_values.shape[0], mask_values.shape[1], 3), dtype=np.uint8)
    image[..., 0] = 120
    image[..., 1] = 80
    image[..., 2] = 40
    Image.fromarray(image).save(image_dir / f"{stem}.jpg")
    savemat(mask_dir / f"{stem}.mat", {"S": mask_values.astype(np.uint8)})


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


def test_coco_stuff_dataset_supports_flat_10k_v11_layout_with_image_lists(tmp_path):
    dataset_root = tmp_path / "cocostuff-10k-v1-1" / "cocostuff-10k-v1.1"
    mask = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    _write_image_mask_pair(
        dataset_root / "images",
        dataset_root / "annotations",
        "COCO_train2014_000000000077",
        mask,
    )
    _write_image_mask_pair(
        dataset_root / "images",
        dataset_root / "annotations",
        "COCO_train2014_000000000113",
        mask,
    )
    image_lists = dataset_root / "imageLists"
    image_lists.mkdir(parents=True)
    (image_lists / "train.txt").write_text("COCO_train2014_000000000077.jpg\n")
    (image_lists / "val.txt").write_text("COCO_train2014_000000000113\n")

    train_dataset = COCOStuffSegDataset(
        tmp_path / "cocostuff-10k-v1-1",
        split="train",
        img_size=32,
        patch_size=16,
        num_classes=3,
    )
    val_dataset = COCOStuffSegDataset(
        tmp_path / "cocostuff-10k-v1-1",
        split="val",
        img_size=32,
        patch_size=16,
        num_classes=3,
    )

    assert [image_path.name for image_path, _ in train_dataset.samples] == [
        "COCO_train2014_000000000077.jpg"
    ]
    assert [image_path.name for image_path, _ in val_dataset.samples] == [
        "COCO_train2014_000000000113.jpg"
    ]


def test_coco_stuff_dataset_supports_flat_10k_v11_mat_annotations(tmp_path):
    dataset_root = tmp_path / "cocostuff-10k-v1-1" / "cocostuff-10k-v1.1"
    mask = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    _write_image_mat_pair(
        dataset_root / "images",
        dataset_root / "annotations",
        "COCO_train2014_000000000077",
        mask,
    )
    image_lists = dataset_root / "imageLists"
    image_lists.mkdir(parents=True)
    (image_lists / "train.txt").write_text("COCO_train2014_000000000077\n")

    dataset = COCOStuffSegDataset(
        tmp_path / "cocostuff-10k-v1-1",
        split="train",
        img_size=32,
        patch_size=16,
        num_classes=3,
    )
    _, target = dataset[0]

    assert len(dataset) == 1
    assert dataset.samples[0][1].suffix == ".mat"
    assert set(target.unique().tolist()) == {0, 1, 2, 255}


def test_coco_stuff_dataset_supports_split_image_and_mask_layout(tmp_path):
    dataset_root = tmp_path / "cocostuff10k"
    mask = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    _write_image_mask_pair(
        dataset_root / "train" / "images",
        dataset_root / "train" / "masks",
        "COCO_train2014_000000000825",
        mask,
    )

    dataset = COCOStuffSegDataset(
        tmp_path,
        split="train",
        img_size=32,
        patch_size=16,
        num_classes=3,
    )

    assert len(dataset) == 1
    assert dataset.samples[0][0].name == "COCO_train2014_000000000825.jpg"


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


def _v2_result(
    epoch: int,
    miou: float,
    *,
    seed: int = 42,
    checkpoint_key: str = "teacher",
) -> dict:
    return {
        "epoch": epoch,
        "miou": miou,
        "metric_version": "global_confusion_v2",
        "probe_seed": seed,
        "checkpoint_key": checkpoint_key,
        "representation": "ema_teacher" if checkpoint_key == "teacher" else "student",
        "checkpoint": f"/checkpoints/checkpoint{epoch:04d}.pth",
        "checkpoint_identity": {
            "basename": f"checkpoint{epoch:04d}.pth",
            "size_bytes": 123,
            "completed_epochs": epoch + 1,
            "training_config": {
                "schedule": {"epochs": 800},
                "model": {"arch": "vit_small", "patch_size": 16},
                "seed": 0,
            },
        },
        "probe_config": {"train_epochs": 15},
        "dataset_identity": {"name": "fixture"},
        "source_commit": "a" * 40,
        "source_dirty": False,
        "per_class_iou": [miou],
    }


def test_write_comparison_csv_merges_protocol_matched_v2_results(tmp_path):
    voc_path = tmp_path / "voc_miou_results_global_confusion_v2.json"
    dense_path = tmp_path / "combined_dense_summary.csv"
    output_path = tmp_path / "comparison.csv"

    voc_path.write_text(json.dumps([_v2_result(180, 30.8)]))
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
        coco_results=[_v2_result(180, 29.5)],
        voc_json_path=voc_path,
        dense_summary_csv_path=dense_path,
        output_path=output_path,
    )

    rows = list(csv.DictReader(output_path.open()))

    assert rows == [
        {
            "epoch": "180",
            "metric_version": "global_confusion_v2",
            "probe_seed": "42",
            "checkpoint_key": "teacher",
            "representation": "ema_teacher",
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


@pytest.mark.parametrize(
    ("voc_overrides", "match"),
    [
        ({"metric_version": "batch_mean_v1"}, "metric_version"),
        ({"probe_seed": 1337}, "probe_seed"),
        ({"checkpoint_key": "student", "representation": "student"}, "checkpoint_key"),
    ],
)
def test_write_comparison_csv_rejects_mismatched_voc_protocol(
    tmp_path,
    voc_overrides,
    match,
):
    voc_row = _v2_result(180, 30.8)
    voc_row.update(voc_overrides)
    voc_path = tmp_path / "voc.json"
    voc_path.write_text(json.dumps([voc_row]))

    with pytest.raises(ValueError, match=match):
        write_comparison_csv(
            coco_results=[_v2_result(180, 29.5)],
            voc_json_path=voc_path,
            dense_summary_csv_path=None,
            output_path=tmp_path / "comparison.csv",
        )


def test_write_comparison_csv_rejects_voc_coco_checkpoint_identity_mismatch(tmp_path):
    voc_path = tmp_path / "voc.json"
    voc_row = _v2_result(180, 30.8)
    coco_row = _v2_result(180, 29.5)
    coco_row["checkpoint_identity"]["size_bytes"] += 1
    voc_path.write_text(json.dumps([voc_row]))

    with pytest.raises(ValueError, match="checkpoint_identity"):
        write_comparison_csv(
            coco_results=[coco_row],
            voc_json_path=voc_path,
            dense_summary_csv_path=None,
            output_path=tmp_path / "comparison.csv",
        )


def test_write_comparison_csv_rejects_unversioned_coco_results(tmp_path):
    voc_path = tmp_path / "voc.json"
    voc_path.write_text(json.dumps([_v2_result(180, 30.8)]))

    with pytest.raises(ValueError, match="missing required fields"):
        write_comparison_csv(
            coco_results=[{"epoch": 180, "miou": 29.5}],
            voc_json_path=voc_path,
            dense_summary_csv_path=None,
            output_path=tmp_path / "comparison.csv",
        )


def test_write_summary_records_v2_protocol(tmp_path):
    output_path = tmp_path / "summary.md"
    write_summary([_v2_result(180, 29.5)], output_path)

    summary = output_path.read_text()
    assert "global_confusion_v2" in summary
    assert "Probe seed: `42`" in summary
    assert "Checkpoint key: `teacher`" in summary
    assert "Representation: `ema_teacher`" in summary
