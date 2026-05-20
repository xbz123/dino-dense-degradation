"""Run patch-level dense degradation diagnostics over DINO checkpoints.

This script is designed for Colab runs where checkpoints live in Google Drive.
It scans a checkpoint directory, evaluates every recognized checkpoint, and
writes structural metrics plus fixed-image qualitative figures into one output
directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dense_eval_utils import discover_checkpoint_files, validate_internal_epoch
from dense_patch_diagnostics import (
    attention_stats,
    class_separability,
    cls_patch_cosine_stats,
    covariance_spectrum,
    fit_pca_projector,
    fixed_query_similarity_stats,
    load_vit_backbone,
    normalize01,
    patch_norm_stats,
    project_pca_rgb,
    spectrum_metrics,
    write_rows_csv,
)


class IndexedSubset(torch.utils.data.Dataset):
    """Subset wrapper that also returns the position in the selected list."""

    def __init__(self, dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, label = self.dataset[self.indices[position]]
        return image, label, position


def build_transforms(image_size: int):
    model_transform = transforms.Compose(
        [
            transforms.Resize(256 if image_size == 224 else image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    raw_transform = transforms.Compose(
        [
            transforms.Resize(256 if image_size == 224 else image_size),
            transforms.CenterCrop(image_size),
        ]
    )
    return model_transform, raw_transform


def select_indices(dataset_size: int, num_images: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    count = min(num_images, dataset_size)
    return sorted(rng.choice(dataset_size, size=count, replace=False).tolist())


def query_points_for_grid(height: int, width: int) -> list[dict[str, int | str]]:
    """Return fixed named query points for comparable similarity maps."""
    coords = [
        ("center", height // 2, width // 2),
        ("upper_left", height // 4, width // 4),
        ("upper_right", height // 4, 3 * width // 4),
        ("lower_left", 3 * height // 4, width // 4),
        ("lower_right", 3 * height // 4, 3 * width // 4),
    ]
    points = []
    for name, row, col in coords:
        row = min(height - 1, max(0, row))
        col = min(width - 1, max(0, col))
        points.append({"name": name, "row": row, "col": col, "index": row * width + col})
    return points


def query_indices_for_grid(height: int, width: int) -> list[int]:
    return [int(point["index"]) for point in query_points_for_grid(height, width)]


def save_overlay(path: Path, image: Image.Image, heat, title: str, cmap: str = "magma") -> None:
    fig, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(image)
    axis.imshow(
        normalize01(heat),
        cmap=cmap,
        alpha=0.45,
        extent=(0, image.size[0], image.size[1], 0),
        interpolation="bilinear",
    )
    axis.set_title(title)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_histogram_png(path: Path, values, title: str, xlabel: str) -> None:
    fig, axis = plt.subplots(figsize=(5, 3))
    axis.hist(np.asarray(values, dtype=np.float32).reshape(-1), bins=60)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_histogram_csv(path: Path, values, bins: int = 80) -> None:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    counts, edges = np.histogram(values, bins=bins)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("bin_left,bin_right,count\n")
        for left, right, count in zip(edges[:-1], edges[1:], counts):
            handle.write(f"{left},{right},{int(count)}\n")


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def jsonable_metric_row(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def extract_checkpoint_features(
    model,
    loader: DataLoader,
    raw_dataset,
    selected_indices: list[int],
    vis_indices: set[int],
    epoch_dir: Path,
    device: torch.device,
):
    all_cls = []
    all_patches = []
    all_attn = []
    saved_visuals = []
    visual_records = []
    height = width = None

    for batch_images, _, batch_positions in loader:
        batch_images = batch_images.to(device, non_blocking=True)
        tokens = model.get_intermediate_layers(batch_images, n=1)[0]
        attentions = model.get_last_selfattention(batch_images)
        cls_tokens = tokens[:, 0].detach()
        patch_tokens = tokens[:, 1:].detach()
        num_patches = patch_tokens.shape[1]
        height = width = int(math.sqrt(num_patches))
        cls_attention = attentions[:, :, 0, 1:].mean(dim=1).detach()

        all_cls.append(cls_tokens.cpu().half())
        all_patches.append(patch_tokens.cpu().half())
        all_attn.append(cls_attention.cpu().half())

        for batch_index, position in enumerate(batch_positions.tolist()):
            source_index = selected_indices[position]
            if source_index not in vis_indices:
                continue

            raw_image = raw_dataset[source_index][0]
            patch = patch_tokens[batch_index].detach().cpu().float()
            cls = cls_tokens[batch_index].detach().cpu().float()
            attn_map = cls_attention[batch_index].detach().cpu().float().reshape(height, width)
            patch_norm = patch.norm(dim=-1).reshape(height, width)
            cls_cos = (F.normalize(patch, dim=-1) @ F.normalize(cls, dim=-1)).reshape(height, width)
            patch_normed = F.normalize(patch, dim=-1)

            tag = f"img{position:04d}_idx{source_index}"
            raw_image.save(epoch_dir / f"{tag}_original.png")
            save_overlay(epoch_dir / f"{tag}_cls_attention.png", raw_image, attn_map, "CLS attention")
            save_overlay(
                epoch_dir / f"{tag}_patch_norm.png",
                raw_image,
                patch_norm,
                "patch feature norm",
                "plasma",
            )
            save_overlay(
                epoch_dir / f"{tag}_cls_similarity.png",
                raw_image,
                cls_cos,
                "patch cosine to CLS",
                "viridis",
            )
            for point in query_points_for_grid(height, width):
                query_index = int(point["index"])
                sim_map = (patch_normed @ patch_normed[query_index]).reshape(height, width)
                save_overlay(
                    epoch_dir / f"{tag}_query_{point['name']}_similarity.png",
                    raw_image,
                    sim_map,
                    f"{point['name']} patch similarity",
                    "cividis",
                )
            save_histogram_png(
                epoch_dir / f"{tag}_patch_norm_hist.png",
                patch_norm,
                "patch feature magnitude",
                "L2 norm",
            )
            save_histogram_png(
                epoch_dir / f"{tag}_cls_attention_hist.png",
                attn_map,
                "CLS attention magnitude",
                "attention",
            )
            save_histogram_png(
                epoch_dir / f"{tag}_cls_patch_cos_hist.png",
                cls_cos,
                "CLS-patch cosine",
                "cosine",
            )
            saved_visuals.append(tag)
            visual_records.append(
                {
                    "tag": tag,
                    "source_index": source_index,
                    "patches": patch,
                    "height": height,
                    "width": width,
                    "epoch_dir": epoch_dir,
                }
            )

    cls_tensor = torch.cat(all_cls, dim=0).float()
    patch_tensor = torch.cat(all_patches, dim=0).float()
    attention_tensor = torch.cat(all_attn, dim=0).float()
    return cls_tensor, patch_tensor, attention_tensor, height, width, saved_visuals, visual_records


def compute_dse_components(
    patch_tensor: torch.Tensor,
    *,
    max_tokens: int,
    dse_group_stride: int,
) -> dict[str, float]:
    b1_values = []
    for index in range(0, patch_tensor.shape[0], max(1, dse_group_stride)):
        b1_values.append(class_separability(patch_tensor[index : index + 1], k=3, max_tokens=max_tokens))

    b8_values = []
    last_start = max(0, patch_tensor.shape[0] - 7)
    for index in range(0, last_start, max(8, dse_group_stride)):
        b8_values.append(class_separability(patch_tensor[index : index + 8], k=24, max_tokens=max_tokens))

    def mean_key(rows: list[dict], key: str) -> float:
        values = np.asarray([row[key] for row in rows], dtype=float)
        return float(np.nanmean(values)) if values.size else float("nan")

    b1_mintra = mean_key(b1_values, "mintra_k3")
    b1_minter = mean_key(b1_values, "minter_k3")
    b1_sep = mean_key(b1_values, "class_sep_k3")
    b8_mintra = mean_key(b8_values, "mintra_k24")
    b8_minter = mean_key(b8_values, "minter_k24")
    b8_sep = mean_key(b8_values, "class_sep_k24")

    return {
        "mintra_b1k3": b1_mintra,
        "minter_b1k3": b1_minter,
        "class_sep_b1k3": b1_sep,
        "mintra_b8k24": b8_mintra,
        "minter_b8k24": b8_minter,
        "class_sep_b8k24": b8_sep,
        "class_sep_avg": float(np.nanmean([b1_sep, b8_sep])),
    }


def analyze_checkpoint(
    info,
    args,
    loader,
    raw_dataset,
    selected_indices: list[int],
    vis_indices: set[int],
    baseline_query_maps: torch.Tensor | None,
    device: torch.device,
):
    epoch_dir = Path(args.out) / f"epoch_{info.epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== epoch {info.epoch} | {info.path} ===", flush=True)

    model, load_info = load_vit_backbone(
        info.path,
        arch=args.arch,
        patch_size=args.patch_size,
        checkpoint_key=args.checkpoint_key,
        device=device,
    )
    print(f"loaded source={load_info['source']} internal_epoch={load_info['internal_epoch']}", flush=True)
    print(
        f"load_state_dict missing={load_info['missing_keys']} unexpected={load_info['unexpected_keys']}",
        flush=True,
    )

    cls_tensor, patch_tensor, attention_tensor, height, width, saved_visuals, visual_records = extract_checkpoint_features(
        model=model,
        loader=loader,
        raw_dataset=raw_dataset,
        selected_indices=selected_indices,
        vis_indices=vis_indices,
        epoch_dir=epoch_dir,
        device=device,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    epoch_warning = validate_internal_epoch(info.epoch, load_info["internal_epoch"])
    if epoch_warning:
        print(f"WARNING: {epoch_warning}", flush=True)
        if args.strict_internal_epoch:
            raise ValueError(epoch_warning)

    flat = patch_tensor.reshape(-1, patch_tensor.shape[-1])
    spec = spectrum_metrics(flat, max_tokens=args.max_spectrum_tokens)
    dse_components = compute_dse_components(
        patch_tensor,
        max_tokens=args.max_kmeans_tokens,
        dse_group_stride=args.dse_group_stride,
    )
    cos_stats = cls_patch_cosine_stats(cls_tensor, patch_tensor)
    norm_stats = patch_norm_stats(patch_tensor)
    attn_stats = attention_stats(attention_tensor, topk=args.attention_topk)
    query_stats = fixed_query_similarity_stats(
        patch_tensor,
        query_indices=query_indices_for_grid(height, width),
        previous_maps=baseline_query_maps,
        prefix="query",
        temperature=args.query_temperature,
    )
    query_maps = query_stats.pop("query_similarity_maps")

    hist_dir = Path(args.out) / "histograms"
    save_histogram_csv(hist_dir / f"epoch_{info.epoch:04d}_patch_norm_hist.csv", patch_tensor.norm(dim=-1))
    save_histogram_csv(hist_dir / f"epoch_{info.epoch:04d}_cls_attention_hist.csv", attention_tensor)
    cls_cos = (F.normalize(patch_tensor, dim=-1) * F.normalize(cls_tensor, dim=-1)[:, None, :]).sum(dim=-1)
    save_histogram_csv(hist_dir / f"epoch_{info.epoch:04d}_cls_patch_cos_hist.csv", cls_cos)

    spectrum = covariance_spectrum(flat, top_k=args.top_eigenvalues, max_tokens=args.max_spectrum_tokens)
    np.save(epoch_dir / "covariance_spectrum.npy", np.asarray(spectrum, dtype=np.float32))

    row = {
        "epoch": info.epoch,
        "checkpoint": str(info.path),
        "checkpoint_size_mb": info.size_mb,
        "internal_epoch": load_info["internal_epoch"],
        "network_key": load_info["source"],
        "epoch_warning": epoch_warning or "",
        "num_metric_images": len(selected_indices),
        "num_visualized_images": len(saved_visuals),
        "grid_height": height,
        "grid_width": width,
        **spec,
        **dse_components,
        **norm_stats,
        **attn_stats,
        **cos_stats,
        **query_stats,
    }
    for index, value in enumerate(spectrum[: args.top_eigenvalues]):
        row[f"cov_eigenvalue_{index}"] = value

    save_json(epoch_dir / "metrics_pre_dse.json", jsonable_metric_row(row))
    print(json.dumps(jsonable_metric_row(row), indent=2), flush=True)
    return (
        jsonable_metric_row(row),
        query_maps if baseline_query_maps is None else baseline_query_maps,
        visual_records,
    )


def save_fixed_basis_pca_maps(visual_records_by_epoch: dict[int, list[dict]], out: Path) -> None:
    """Fit one PCA basis over all fixed-image records and save comparable maps."""
    records = [record for records in visual_records_by_epoch.values() for record in records]
    if not records:
        return
    projector = fit_pca_projector([record["patches"] for record in records])
    torch.save(
        {
            "mean": projector.mean,
            "basis": projector.basis,
            "component_min": projector.component_min,
            "component_max": projector.component_max,
        },
        out / "pca_fixed_basis.pt",
    )
    save_json(
        out / "pca_fixed_basis_config.json",
        {
            "epochs": sorted(visual_records_by_epoch),
            "num_fixed_maps": len(records),
            "basis": "fit once from all visualized fixed-image patch features",
        },
    )
    for records in visual_records_by_epoch.values():
        for record in records:
            rgb = project_pca_rgb(
                record["patches"],
                projector,
                int(record["height"]),
                int(record["width"]),
            )
            plt.imsave(record["epoch_dir"] / f"{record['tag']}_pca_fixed_basis.png", rgb)


def add_dse_scores(rows: list[dict]) -> list[dict]:
    sep = np.asarray([row.get("class_sep_avg", np.nan) for row in rows], dtype=float)
    mdim = np.asarray([row.get("effective_rank", np.nan) for row in rows], dtype=float)
    lam = float(np.nanstd(sep) / (np.nanstd(mdim) + 1e-12))
    for row in rows:
        row["dse_lambda"] = lam
        row["dse"] = float(row["class_sep_avg"] + lam * row["effective_rank"])
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arch", default="vit_small")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--checkpoint_key", default="teacher")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_metric_images", type=int, default=2048)
    parser.add_argument("--num_vis_images", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_spectrum_tokens", type=int, default=30000)
    parser.add_argument("--max_kmeans_tokens", type=int, default=12000)
    parser.add_argument(
        "--dse_group_stride",
        type=int,
        default=1,
        help="Use >1 for quick smoke tests; 1 evaluates all sampled images.",
    )
    parser.add_argument("--attention_topk", type=int, default=10)
    parser.add_argument("--query_temperature", type=float, default=0.07)
    parser.add_argument("--top_eigenvalues", type=int, default=32)
    parser.add_argument("--epoch_filter", nargs="*", type=int, default=None)
    parser.add_argument("--strict_internal_epoch", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root)
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    checkpoints = discover_checkpoint_files(args.ckpt_dir, epoch_filter=args.epoch_filter)
    if not checkpoints:
        raise FileNotFoundError(f"No recognized checkpoints found in {args.ckpt_dir}")

    model_transform, raw_transform = build_transforms(args.image_size)
    dataset = datasets.ImageFolder(str(image_root), transform=model_transform)
    raw_dataset = datasets.ImageFolder(str(image_root), transform=raw_transform)
    selected_indices = select_indices(len(dataset), args.num_metric_images, args.seed)
    vis_indices = set(selected_indices[: min(args.num_vis_images, len(selected_indices))])

    indexed_subset = IndexedSubset(dataset, selected_indices)
    loader = DataLoader(
        indexed_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    config = {
        **vars(args),
        "resolved_image_root": str(image_root),
        "selected_indices": selected_indices,
        "visualized_indices": sorted(vis_indices),
        "checkpoint_epochs": [item.epoch for item in checkpoints],
    }
    save_json(out / "config.json", config)
    save_json(
        out / "checkpoint_manifest.json",
        [
            {
                "epoch": item.epoch,
                "path": str(item.path),
                "internal_epoch": item.internal_epoch,
                "size_mb": item.size_mb,
            }
            for item in checkpoints
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    print(f"metric images: {len(selected_indices)}", flush=True)
    print(f"visualized indices: {sorted(vis_indices)}", flush=True)

    rows = []
    baseline_query_maps = None
    visual_records_by_epoch = {}
    for info in checkpoints:
        row, baseline_query_maps, visual_records = analyze_checkpoint(
            info,
            args,
            loader,
            raw_dataset,
            selected_indices,
            vis_indices,
            baseline_query_maps,
            device,
        )
        rows.append(row)
        visual_records_by_epoch[info.epoch] = visual_records

    save_fixed_basis_pca_maps(visual_records_by_epoch, out)
    if visual_records_by_epoch:
        first_records = next(iter(visual_records_by_epoch.values()))
        if first_records:
            save_json(
                out / "query_points.json",
                query_points_for_grid(int(first_records[0]["height"]), int(first_records[0]["width"])),
            )
    rows = add_dse_scores(rows)
    write_rows_csv(out / "patch_attention_dse_summary.csv", rows)
    save_json(out / "patch_attention_dse_summary.json", rows)
    for row in rows:
        save_json(out / f"epoch_{row['epoch']:04d}" / "metrics.json", row)

    print(f"saved summary: {out / 'patch_attention_dse_summary.csv'}", flush=True)
    print(f"saved output dir: {out}", flush=True)


if __name__ == "__main__":
    main()
