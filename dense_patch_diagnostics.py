"""Patch-level diagnostics for DINO dense degradation experiments.

The functions in this module are intentionally independent from Colab. The
notebook and CLI script use them for DSE-style structural metrics, CLS-patch
similarity, attention statistics, and fixed-query patch similarity maps.
"""

from __future__ import annotations

import csv
import math
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

import vision_transformer as vits


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Remove DDP/MultiCrop prefixes and projector-head weights."""
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        if key.startswith("head.") or key.startswith("dino_head."):
            continue
        cleaned[key] = value
    return cleaned


def load_vit_backbone(
    checkpoint_path: str | Path,
    *,
    arch: str = "vit_small",
    patch_size: int = 16,
    checkpoint_key: str = "teacher",
    device: str | torch.device = "cpu",
):
    """Load a frozen DINO ViT backbone from a checkpoint."""
    if not hasattr(vits, arch):
        raise ValueError(f"Unknown ViT architecture: {arch}")

    model = getattr(vits, arch)(patch_size=patch_size, num_classes=0)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    source = "raw"
    if isinstance(checkpoint, dict) and checkpoint_key in checkpoint:
        state_dict = checkpoint[checkpoint_key]
        source = checkpoint_key
    elif isinstance(checkpoint, dict) and "teacher" in checkpoint:
        state_dict = checkpoint["teacher"]
        source = "teacher"
    elif isinstance(checkpoint, dict) and "student" in checkpoint:
        state_dict = checkpoint["student"]
        source = "student"
    else:
        state_dict = checkpoint

    msg = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False

    internal_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    return model, {
        "source": source,
        "internal_epoch": int(internal_epoch) if internal_epoch is not None else None,
        "missing_keys": len(msg.missing_keys),
        "unexpected_keys": len(msg.unexpected_keys),
    }


def sample_rows(x: torch.Tensor, max_rows: int | None) -> torch.Tensor:
    """Deterministically subsample rows, preserving coverage across the tensor."""
    if max_rows is None or max_rows <= 0 or x.shape[0] <= max_rows:
        return x
    idx = torch.linspace(0, x.shape[0] - 1, max_rows, device=x.device).long()
    return x.index_select(0, idx)


def _safe_prob(values: torch.Tensor) -> torch.Tensor:
    values = values.float().clamp_min(0)
    denom = values.sum().clamp_min(1e-12)
    return values / denom


def _entropy_from_prob(prob: torch.Tensor) -> torch.Tensor:
    prob = prob.clamp_min(1e-12)
    return -(prob * prob.log()).sum()


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values.float().reshape(-1), q).item())


def spectrum_metrics(features: torch.Tensor, max_tokens: int = 30000) -> dict[str, float | int]:
    """Compute effective rank and covariance spectrum concentration."""
    x = features.reshape(-1, features.shape[-1]).float()
    x = sample_rows(x, max_tokens)
    if x.shape[0] < 2:
        return {
            "num_tokens": int(x.shape[0]),
            "feature_dim": int(x.shape[-1]),
            "effective_rank": float("nan"),
            "cov_effective_rank": float("nan"),
            "top1_eigen_ratio": float("nan"),
            "spectrum_entropy": float("nan"),
        }

    x = x - x.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(x)
    singular_prob = _safe_prob(singular)
    eigvals = (singular.square()) / max(1, x.shape[0] - 1)
    eig_prob = _safe_prob(eigvals)

    return {
        "num_tokens": int(x.shape[0]),
        "feature_dim": int(x.shape[-1]),
        "effective_rank": float(_entropy_from_prob(singular_prob).exp().item()),
        "cov_effective_rank": float(_entropy_from_prob(eig_prob).exp().item()),
        "top1_eigen_ratio": float(eig_prob.max().item()),
        "spectrum_entropy": float(_entropy_from_prob(eig_prob).item()),
    }


def covariance_spectrum(features: torch.Tensor, top_k: int = 32, max_tokens: int = 30000) -> list[float]:
    """Return descending covariance eigenvalues estimated via SVD."""
    x = features.reshape(-1, features.shape[-1]).float()
    x = sample_rows(x, max_tokens)
    if x.shape[0] < 2:
        return []
    x = x - x.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(x)
    eigvals = (singular.square()) / max(1, x.shape[0] - 1)
    return [float(v) for v in eigvals[:top_k].cpu()]


def cls_patch_cosine_stats(cls_tokens: torch.Tensor, patch_tokens: torch.Tensor) -> dict[str, float]:
    """Summarize cosine similarity between each patch token and its CLS token."""
    cls = F.normalize(cls_tokens.float(), dim=-1)
    patches = F.normalize(patch_tokens.float(), dim=-1)
    sim = (patches * cls[:, None, :]).sum(dim=-1).reshape(-1)
    return {
        "cls_patch_cos_mean": float(sim.mean().item()),
        "cls_patch_cos_std": float(sim.std(unbiased=False).item()),
        "cls_patch_cos_p10": _quantile(sim, 0.10),
        "cls_patch_cos_p50": _quantile(sim, 0.50),
        "cls_patch_cos_p90": _quantile(sim, 0.90),
        "cls_patch_cos_min": float(sim.min().item()),
        "cls_patch_cos_max": float(sim.max().item()),
    }


def patch_norm_stats(patch_tokens: torch.Tensor) -> dict[str, float]:
    """Summarize unnormalized patch-token magnitudes."""
    norms = patch_tokens.float().norm(dim=-1).reshape(-1)
    return {
        "patch_norm_mean": float(norms.mean().item()),
        "patch_norm_std": float(norms.std(unbiased=False).item()),
        "patch_norm_p10": _quantile(norms, 0.10),
        "patch_norm_p50": _quantile(norms, 0.50),
        "patch_norm_p90": _quantile(norms, 0.90),
        "patch_norm_p99": _quantile(norms, 0.99),
    }


def attention_stats(cls_attention: torch.Tensor, topk: int = 10) -> dict[str, float]:
    """Summarize CLS-to-patch attention values.

    Input shape may be ``[B, N]`` or any tensor whose last dimension is the
    flattened patch dimension.
    """
    values = cls_attention.float().reshape(-1, cls_attention.shape[-1])
    probs = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
    k = min(topk, values.shape[-1])
    top_mass = probs.topk(k=k, dim=-1).values.sum(dim=-1)
    return {
        "cls_attention_mean": float(values.mean().item()),
        "cls_attention_std": float(values.std(unbiased=False).item()),
        "cls_attention_entropy_mean": float(entropy.mean().item()),
        "cls_attention_max_mean": float(values.max(dim=-1).values.mean().item()),
        f"cls_attention_top{k}_mass_mean": float(top_mass.mean().item()),
    }


def _init_centers(x: torch.Tensor, k: int) -> torch.Tensor:
    if k == 1:
        return x[:1].clone()
    idx = torch.linspace(0, x.shape[0] - 1, k, device=x.device).long()
    return x.index_select(0, idx).clone()


def kmeans(x: torch.Tensor, k: int, iters: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Small deterministic k-means helper for DSE-style pseudo classes."""
    if x.shape[0] < k:
        raise ValueError(f"k={k} cannot exceed number of samples {x.shape[0]}")
    centers = _init_centers(x, k)
    labels = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    for _ in range(iters):
        labels = torch.cdist(x, centers).argmin(dim=1)
        new_centers = []
        for index in range(k):
            points = x[labels == index]
            new_centers.append(points.mean(dim=0) if points.numel() else centers[index])
        new_centers = torch.stack(new_centers, dim=0)
        if torch.allclose(new_centers, centers, atol=1e-5):
            centers = new_centers
            break
        centers = new_centers
    return labels, centers


def class_separability(
    patch_tokens: torch.Tensor,
    *,
    k: int,
    max_tokens: int = 12000,
    seed: int | None = None,
) -> dict[str, float]:
    """Compute a lightweight DSE-style separability term using pseudo labels."""
    del seed  # deterministic initialization keeps Colab runs reproducible.
    x = patch_tokens.reshape(-1, patch_tokens.shape[-1]).float()
    x = sample_rows(x, max_tokens)
    x = F.normalize(x, dim=-1)
    if x.shape[0] <= k:
        return {
            f"mintra_k{k}": float("nan"),
            f"minter_k{k}": float("nan"),
            f"class_sep_k{k}": float("nan"),
        }

    labels, centers = kmeans(x, k)
    intra_values = []
    for index in range(k):
        points = x[labels == index]
        if points.numel() == 0:
            continue
        distances = torch.cdist(points, centers[index : index + 1]).reshape(-1)
        intra_values.append(distances.mean())

    center_distances = torch.cdist(centers, centers)
    center_distances.fill_diagonal_(float("inf"))
    inter = center_distances.min(dim=1).values.mean()
    intra = torch.stack(intra_values).mean() if intra_values else torch.tensor(float("nan"))
    sep = inter - intra
    return {
        f"mintra_k{k}": float(intra.item()),
        f"minter_k{k}": float(inter.item()),
        f"class_sep_k{k}": float(sep.item()),
    }


def fixed_query_similarity_stats(
    patch_tokens: torch.Tensor,
    *,
    query_indices: Iterable[int],
    previous_maps: torch.Tensor | None,
    prefix: str,
    temperature: float = 0.07,
) -> dict[str, float | torch.Tensor]:
    """Summarize fixed-query patch similarity maps across a batch."""
    patches = F.normalize(patch_tokens.float(), dim=-1)
    maps = []
    entropies = []
    top_masses = []
    correlations = []
    query_indices = list(query_indices)
    for batch_index in range(patches.shape[0]):
        image_patches = patches[batch_index]
        for query_index in query_indices:
            query_index = min(max(0, int(query_index)), image_patches.shape[0] - 1)
            sim = image_patches @ image_patches[query_index]
            prob = torch.softmax(sim / temperature, dim=-1)
            top_count = max(1, math.ceil(0.10 * prob.numel()))
            entropies.append(_entropy_from_prob(prob))
            top_masses.append(prob.topk(top_count).values.sum())
            maps.append(sim)

    map_tensor = torch.stack(maps, dim=0)
    if previous_maps is not None:
        previous = previous_maps.to(map_tensor.device).float()
        for current, base in zip(map_tensor, previous):
            if torch.equal(current, base):
                correlations.append(torch.tensor(1.0, device=map_tensor.device))
                continue
            cur = current - current.mean()
            ref = base - base.mean()
            denom = cur.norm() * ref.norm()
            correlations.append((cur * ref).sum() / denom.clamp_min(1e-12))

    stats: dict[str, float | torch.Tensor] = {
        f"{prefix}_sim_entropy_mean": float(torch.stack(entropies).mean().item()),
        f"{prefix}_sim_top10pct_mass_mean": float(torch.stack(top_masses).mean().item()),
        f"{prefix}_similarity_maps": map_tensor.detach().cpu(),
    }
    if correlations:
        stats[f"{prefix}_sim_early_corr_mean"] = float(torch.stack(correlations).mean().item())
    else:
        stats[f"{prefix}_sim_early_corr_mean"] = float("nan")
    return stats


def normalize01(array) -> np.ndarray:
    """Normalize an array to [0, 1], returning zeros for constant arrays."""
    array = np.asarray(array, dtype=np.float32)
    span = float(np.ptp(array))
    if span <= 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return (array - float(array.min())) / span


def pca_rgb(patches: torch.Tensor, height: int, width: int) -> np.ndarray:
    """Project patch features to a 3-channel PCA visualization."""
    x = patches.float() - patches.float().mean(dim=0, keepdim=True)
    try:
        _, _, vectors = torch.pca_lowrank(x, q=3)
        y = x @ vectors[:, :3]
    except Exception:
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        y = x @ vh[:3].T
    y = y.reshape(height, width, 3).cpu().numpy()
    return normalize01(y)


def write_rows_csv(path: str | Path, rows: list[dict]) -> None:
    """Write a list of flat metric dictionaries to CSV."""
    if not rows:
        raise ValueError("Cannot write empty metric rows")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
