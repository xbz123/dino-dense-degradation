"""
COCO-Stuff Linear Probing for Dense Degradation Evaluation
==========================================================
Evaluate selected DINO checkpoints on COCO-Stuff semantic segmentation with a
frozen ViT-S/16 backbone and a lightweight linear segmentation head.

Example on Kaggle/Colab:
    python eval_coco_stuff_dense.py \
        --ckpt_dir /content/drive/MyDrive/dinocheckpoint \
        --coco_root /content/drive/MyDrive/coco_stuff \
        --epochs 50,80,180,220,300,318 \
        --output_dir /content/drive/MyDrive/dino_dense_degradation_eval/to_epoch_0318_raw_l2/coco_stuff_selected
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from eval_voc_dense import (
    compute_miou,
    discover_checkpoints,
    extract_features,
    load_dino_backbone,
    train_linear_head,
)


class COCOStuffSegDataset(torch.utils.data.Dataset):
    """COCO-Stuff segmentation dataset backed by image and PNG mask folders."""

    NUM_CLASSES = 171
    IGNORE_INDEX = 255

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        img_size: int = 336,
        patch_size: int = 16,
        num_classes: int = NUM_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        max_images: int | None = None,
    ) -> None:
        self.original_root = Path(root)
        self.split = split
        self.img_size = (img_size // patch_size) * patch_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.root = self._resolve_dataset_root(self.original_root)

        self.image_dir = self._find_image_dir()
        self.mask_dir = self._find_mask_dir()
        self.split_stems = self._load_split_stems()
        self.samples = self._pair_samples()
        if max_images is not None:
            self.samples = self.samples[:max_images]
        if not self.samples:
            raise FileNotFoundError(
                f"No paired COCO-Stuff images/masks found for split '{split}' under {root}"
            )

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _resolve_dataset_root(self, root: Path) -> Path:
        if self._looks_like_dataset_root(root):
            return root

        child_roots = [
            path
            for path in sorted(root.iterdir())
            if path.is_dir() and self._looks_like_dataset_root(path)
        ]
        if child_roots:
            return child_roots[0]
        return root

    def _looks_like_dataset_root(self, path: Path) -> bool:
        split_2017 = f"{self.split}2017"
        markers = [
            path / "images",
            path / "annotations",
            path / "stuffthingmaps",
            path / "stuffthingmaps_trainval2017",
            path / split_2017,
            path / self.split,
            path / self.split / "images",
        ]
        return any(marker.is_dir() for marker in markers)

    def _find_existing_dir(self, candidates: list[Path], kind: str) -> Path:
        for path in candidates:
            if path.is_dir():
                return path
        formatted = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            f"Could not find COCO-Stuff {kind} directory for split '{self.split}'. "
            f"Checked:\n{formatted}"
        )

    def _find_image_dir(self) -> Path:
        split_2017 = f"{self.split}2017"
        return self._find_existing_dir(
            [
                self.root / "images" / split_2017,
                self.root / "images" / self.split,
                self.root / self.split / "images",
                self.root / split_2017,
                self.root / self.split,
                self.root / "images",
            ],
            "image",
        )

    def _find_mask_dir(self) -> Path:
        split_2017 = f"{self.split}2017"
        return self._find_existing_dir(
            [
                self.root / "annotations" / split_2017,
                self.root / "annotations" / self.split,
                self.root / self.split / "annotations",
                self.root / self.split / "masks",
                self.root / "masks" / split_2017,
                self.root / "masks" / self.split,
                self.root / "stuffthingmaps" / split_2017,
                self.root / "stuffthingmaps" / self.split,
                self.root / "stuffthingmaps_trainval2017" / "annotations" / split_2017,
                self.root / "stuffthingmaps_trainval2017" / "annotations" / self.split,
                self.root / "annotations",
                self.root / "masks",
            ],
            "mask",
        )

    def _load_split_stems(self) -> set[str] | None:
        image_lists_dir = self.root / "imageLists"
        if not image_lists_dir.is_dir():
            return None

        split_2017 = f"{self.split}2017"
        candidates = [
            image_lists_dir / f"{self.split}.txt",
            image_lists_dir / f"{split_2017}.txt",
        ]
        candidates.extend(sorted(image_lists_dir.glob(f"{self.split}*.txt")))

        split_file = next((path for path in candidates if path.is_file()), None)
        if split_file is None:
            return None

        stems = set()
        with split_file.open() as handle:
            for line in handle:
                token = line.strip().split()
                if not token:
                    continue
                stems.add(Path(token[0]).stem)
        return stems

    def _pair_samples(self) -> list[tuple[Path, Path]]:
        image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if self.split_stems is not None:
            image_paths = [path for path in image_paths if path.stem in self.split_stems]

        samples = []
        for image_path in image_paths:
            mask_path = self.mask_dir / f"{image_path.stem}.png"
            if mask_path.is_file():
                samples.append((image_path, mask_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.img_transform(image)

        mask = Image.open(mask_path)
        mask = mask.resize((self.img_size, self.img_size), resample=Image.Resampling.NEAREST)
        target = np.array(mask, dtype=np.int64)
        invalid = (target != self.ignore_index) & (
            (target < 0) | (target >= self.num_classes)
        )
        target[invalid] = self.ignore_index

        return image, torch.from_numpy(target).long()


def parse_epoch_filter(value: str | None) -> list[int] | None:
    """Parse comma/space-separated epoch filters from the CLI."""
    if value is None or value.strip() == "":
        return None
    return [int(token) for token in re.split(r"[\s,]+", value.strip()) if token]


def filter_checkpoints_by_epoch(
    checkpoints: list[tuple[int, str]],
    epochs: list[int] | None,
) -> list[tuple[int, str]]:
    """Return checkpoints in requested epoch order and fail if any are missing."""
    if epochs is None:
        return checkpoints
    by_epoch = {epoch: path for epoch, path in checkpoints}
    missing = [epoch for epoch in epochs if epoch not in by_epoch]
    if missing:
        available = ", ".join(str(epoch) for epoch in sorted(by_epoch))
        requested = ", ".join(str(epoch) for epoch in missing)
        raise ValueError(
            f"Requested checkpoint epochs not found: {requested}. "
            f"Available epochs: {available}"
        )
    return [(epoch, by_epoch[epoch]) for epoch in epochs]


def _load_voc_results(path: str | Path | None) -> dict[int, str]:
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open() as handle:
        return {int(row["epoch"]): str(row["miou"]) for row in json.load(handle)}


def _load_dense_summary(path: str | Path | None) -> dict[int, dict[str, str]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        return {int(row["epoch"]): row for row in csv.DictReader(handle)}


def write_comparison_csv(
    coco_results: list[dict[str, float]],
    voc_json_path: str | Path | None,
    dense_summary_csv_path: str | Path | None,
    output_path: str | Path,
) -> None:
    """Merge COCO-Stuff mIoU with VOC and dense diagnostic summaries."""
    voc_by_epoch = _load_voc_results(voc_json_path)
    dense_by_epoch = _load_dense_summary(dense_summary_csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "coco_stuff_miou",
        "voc_miou",
        "raw_dse",
        "l2_dse",
        "raw_class_sep",
        "l2_class_sep",
        "effective_rank",
        "top1_eigen_ratio",
        "patch_norm_mean",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(coco_results, key=lambda row: row["epoch"]):
            epoch = int(result["epoch"])
            dense = dense_by_epoch.get(epoch, {})
            writer.writerow(
                {
                    "epoch": epoch,
                    "coco_stuff_miou": result["miou"],
                    "voc_miou": voc_by_epoch.get(epoch, ""),
                    "raw_dse": dense.get("raw_dse", ""),
                    "l2_dse": dense.get("l2_dse", ""),
                    "raw_class_sep": dense.get("raw_class_sep")
                    or dense.get("raw_class_sep_avg", ""),
                    "l2_class_sep": dense.get("l2_class_sep")
                    or dense.get("l2_class_sep_avg", ""),
                    "effective_rank": dense.get("effective_rank", ""),
                    "top1_eigen_ratio": dense.get("top1_eigen_ratio", ""),
                    "patch_norm_mean": dense.get("patch_norm_mean", ""),
                }
            )


def plot_coco_curve(results: list[dict[str, float]], output_path: str | Path) -> None:
    epochs = [int(row["epoch"]) for row in results]
    mious = [float(row["miou"]) for row in results]

    best_idx = int(np.argmax(mious))
    best_epoch = epochs[best_idx]
    best_miou = mious[best_idx]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(epochs, mious, "g-o", linewidth=2, markersize=6, label="mIoU (COCO-Stuff)")
    ax.axvline(
        x=best_epoch,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Best: Epoch {best_epoch}",
    )
    ax.plot(best_epoch, best_miou, "r*", markersize=15, zorder=5)

    if best_idx < len(epochs) - 1:
        last_miou = mious[-1]
        diff = last_miou - best_miou
        ax.annotate(
            f"Diff: {diff:.1f}",
            xy=(epochs[-1], last_miou),
            xytext=(epochs[-1] - 15, last_miou + 1.5),
            fontsize=10,
            color="red",
            arrowprops=dict(arrowstyle="->", color="red"),
        )

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("mIoU (%)", fontsize=13)
    ax.set_title(
        "DINO Dense Performance throughout Pretraining\n"
        "(COCO-Stuff Linear Probing)",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(epochs) - 5, max(epochs) + 5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def write_summary(results: list[dict[str, float]], output_path: str | Path) -> None:
    best = max(results, key=lambda row: row["miou"])
    last = results[-1]
    diff = float(last["miou"]) - float(best["miou"])
    trend = "degradation" if diff < 0 else "no degradation observed"

    lines = [
        "# COCO-Stuff Selected-Checkpoint Summary",
        "",
        f"- Evaluated checkpoints: {len(results)}",
        f"- Best checkpoint: epoch {int(best['epoch'])}, mIoU {float(best['miou']):.3f}",
        f"- Final checkpoint: epoch {int(last['epoch'])}, mIoU {float(last['miou']):.3f}",
        f"- Final minus best: {diff:.3f} ({trend})",
        "",
        "## Results",
        "",
        "| Epoch | COCO-Stuff mIoU |",
        "| ---: | ---: |",
    ]
    for row in results:
        lines.append(f"| {int(row['epoch'])} | {float(row['miou']):.3f} |")
    lines.append("")

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines))


def evaluate_checkpoints(args: argparse.Namespace) -> list[dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dtype = torch.float16 if args.feature_dtype == "float16" else torch.float32
    img_size = (args.img_size // args.patch_size) * args.patch_size

    ckpt_files = discover_checkpoints(args.ckpt_dir)
    ckpt_files = filter_checkpoints_by_epoch(ckpt_files, parse_epoch_filter(args.epochs))

    print(f"Using device: {device}")
    print(f"Image size: {img_size}")
    print(f"COCO-Stuff classes: {args.num_classes}")
    print(f"Found {len(ckpt_files)} selected checkpoints:")
    for epoch, path in ckpt_files:
        print(f"  Epoch {epoch:>4d}: {os.path.basename(path)}")

    train_dataset = COCOStuffSegDataset(
        args.coco_root,
        split=args.train_split,
        img_size=img_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        max_images=args.max_train_images,
    )
    val_dataset = COCOStuffSegDataset(
        args.coco_root,
        split=args.val_split,
        img_size=img_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        max_images=args.max_val_images,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")

    results = []
    embed_dim = 384
    for index, (epoch, ckpt_path) in enumerate(ckpt_files):
        print(f"\n[{index + 1}/{len(ckpt_files)}] Evaluating Epoch {epoch}...")
        model = load_dino_backbone(ckpt_path, arch=args.arch, patch_size=args.patch_size)
        model = model.to(device)

        print("  Extracting train features...")
        features_train, targets_train = extract_features(
            model, train_loader, device, feature_dtype=feature_dtype
        )
        print(f"  Train features shape: {features_train.shape}, dtype: {features_train.dtype}")

        print("  Extracting val features...")
        features_val, targets_val = extract_features(
            model, val_loader, device, feature_dtype=feature_dtype
        )
        print(f"  Val features shape: {features_val.shape}, dtype: {features_val.dtype}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Training linear head ({args.train_epochs} epochs)...")
        miou = train_linear_head(
            features_train,
            targets_train,
            features_val,
            targets_val,
            embed_dim=embed_dim,
            num_classes=args.num_classes,
            patch_size=args.patch_size,
            img_size=img_size,
            device=device,
            epochs=args.train_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            optimizer_name=args.optimizer,
        )
        miou_percent = float(miou * 100)
        if not math.isfinite(miou_percent):
            raise RuntimeError(f"Non-finite mIoU for epoch {epoch}: {miou_percent}")

        print(f"  Epoch {epoch}: COCO-Stuff mIoU = {miou_percent:.2f}%")
        results.append(
            {
                "epoch": int(epoch),
                "miou": miou_percent,
                "checkpoint": str(ckpt_path),
                "train_images": len(train_dataset),
                "val_images": len(val_dataset),
            }
        )

        del features_train, targets_train, features_val, targets_val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("COCO-Stuff Dense Degradation Evaluation")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--coco_root", type=str, required=True)
    parser.add_argument(
        "--epochs",
        type=str,
        default="50,80,180,220,300,318",
        help="Comma/space-separated checkpoint epochs to evaluate.",
    )
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--arch", type=str, default="vit_small")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=336)
    parser.add_argument("--num_classes", type=int, default=COCOStuffSegDataset.NUM_CLASSES)
    parser.add_argument("--ignore_index", type=int, default=COCOStuffSegDataset.IGNORE_INDEX)
    parser.add_argument("--train_epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--feature_dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_train_images", type=int, default=None)
    parser.add_argument("--max_val_images", type=int, default=None)
    parser.add_argument("--voc_results_json", type=str, default=None)
    parser.add_argument("--dense_summary_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./coco_stuff_selected")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_checkpoints(args)

    results_path = output_dir / "coco_stuff_miou_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to: {results_path}")

    plot_path = output_dir / "dense_degradation_coco_stuff.png"
    plot_coco_curve(results, plot_path)
    print(f"Plot saved to: {plot_path}")

    comparison_path = output_dir / "coco_voc_dse_comparison.csv"
    write_comparison_csv(
        results,
        voc_json_path=args.voc_results_json,
        dense_summary_csv_path=args.dense_summary_csv,
        output_path=comparison_path,
    )
    print(f"Comparison CSV saved to: {comparison_path}")

    summary_path = output_dir / "coco_stuff_summary.md"
    write_summary(results, summary_path)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
