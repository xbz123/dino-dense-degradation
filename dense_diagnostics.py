# Copyright (c) Bingzhou Xie
#
# Dense Degradation diagnostics for DINO training.
# Migrated from patch_crr/src/eval/diagnostics.py (audit-verified).
"""
Diagnostic tools to detect Dense Degradation during DINO pre-training.

Core metrics:
  - Effective Rank: measures the dimensionality of patch token representations.
    A sharp drop indicates covariance collapse (dense degradation).
  - CLS-Patch Cosine Similarity: measures how similar the global [CLS] token
    is to the average patch token. A sharp rise indicates feature homogenization.
"""

import os
import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Core diagnostic functions (from patch_crr, audit-verified)
# ---------------------------------------------------------------------------

def effective_rank(cov: torch.Tensor) -> float:
    """Compute effective rank of a covariance matrix.

    erank(Gamma) = exp( -sum_i lambda_i * log(lambda_i) )
    where lambda_i are the normalized eigenvalues.

    A high effective rank means diverse, high-dimensional representations.
    A sharp drop during training signals covariance collapse.
    """
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=1e-8)
    eigvals = eigvals / eigvals.sum()
    entropy = -(eigvals * eigvals.log()).sum()
    return entropy.exp().item()


def cls_patch_cosine(z_cls: torch.Tensor, z_patch: torch.Tensor) -> float:
    """Average cosine similarity between CLS token and mean patch token.

    Args:
        z_cls: (N, d) CLS token features.
        z_patch: (N, L, d) patch token features.

    Returns:
        Mean cosine similarity (scalar). High values indicate that patch
        features are collapsing toward the global CLS representation.
    """
    mean_patch = z_patch.mean(dim=1)  # (N, d)
    z_cls_n = F.normalize(z_cls, dim=-1)
    mean_p_n = F.normalize(mean_patch, dim=-1)
    return (z_cls_n * mean_p_n).sum(dim=-1).mean().item()


def condition_number(cov: torch.Tensor) -> float:
    """Condition number of the covariance matrix (max/min eigenvalue)."""
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=1e-10)
    return (eigvals.max() / eigvals.min()).item()


def eigenvalue_spectrum(cov: torch.Tensor, top_k: int = 32) -> np.ndarray:
    """Return the top-k eigenvalues in descending order."""
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals, _ = eigvals.sort(descending=True)
    return eigvals[:top_k].cpu().numpy()


# ---------------------------------------------------------------------------
# High-level diagnostic runner
# ---------------------------------------------------------------------------

def _build_val_loader(val_data_path, batch_size=64, num_workers=2):
    """Build a simple validation data loader for diagnostics."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = datasets.ImageFolder(val_data_path, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


@torch.no_grad()
def compute_dense_diagnostics(backbone, val_data_path, device,
                              num_batches=50, batch_size=64, num_workers=2):
    """Compute dense degradation diagnostics on validation data.

    Uses the teacher backbone (without projection head) to extract features
    from the last transformer layer. Returns effective rank, CLS-patch cosine
    similarity, and condition number.

    Args:
        backbone: The ViT backbone (teacher.backbone or teacher_without_ddp.backbone).
        val_data_path: Path to validation set (ImageFolder format).
        device: torch device.
        num_batches: Number of validation batches to use (for speed).
        batch_size: Batch size for validation.
        num_workers: DataLoader workers.

    Returns:
        dict with keys: 'diag_effective_rank', 'diag_cls_patch_cosine',
                        'diag_condition_number', 'diag_top_eigenvalues'.
    """
    if not val_data_path or not os.path.isdir(val_data_path):
        return {}

    loader = _build_val_loader(val_data_path, batch_size, num_workers)
    backbone.eval()

    all_cls = []
    all_patches = []

    for i, (images, _) in enumerate(loader):
        if i >= num_batches:
            break
        images = images.to(device, non_blocking=True)

        # get_intermediate_layers returns list of (B, 1+num_patches, embed_dim)
        # We take the last layer output
        output = backbone.get_intermediate_layers(images, n=1)[0]

        cls_token = output[:, 0]          # (B, d)
        patch_tokens = output[:, 1:]      # (B, L, d)

        all_cls.append(cls_token.cpu())
        all_patches.append(patch_tokens.cpu())

    if not all_cls:
        return {}

    all_cls = torch.cat(all_cls, dim=0)        # (N, d)
    all_patches = torch.cat(all_patches, dim=0)  # (N, L, d)

    # Compute covariance over all patch tokens
    patches_flat = all_patches.reshape(-1, all_patches.shape[-1])  # (N*L, d)
    patches_flat = patches_flat.float()  # ensure FP32 for numerical stability
    mean = patches_flat.mean(dim=0)
    centered = patches_flat - mean
    cov = (centered.T @ centered) / (centered.shape[0] - 1)

    diag = {
        'diag_effective_rank': effective_rank(cov),
        'diag_cls_patch_cosine': cls_patch_cosine(all_cls.float(), all_patches.float()),
        'diag_condition_number': condition_number(cov),
    }

    # Also store top eigenvalues for later analysis
    top_eigs = eigenvalue_spectrum(cov, top_k=32)
    for k, v in enumerate(top_eigs):
        diag[f'diag_eigenvalue_{k}'] = float(v)

    return diag


@torch.no_grad()
def save_attention_maps(backbone, val_data_path, epoch, output_dir, device,
                        num_images=5, num_workers=2):
    """Save self-attention maps from the last layer of the ViT.

    For each image, saves the attention from [CLS] to all patches as a
    heatmap overlaid on the original image.

    Args:
        backbone: The ViT backbone with get_last_selfattention() method.
        val_data_path: Path to validation set.
        epoch: Current epoch number (for filename).
        output_dir: Directory to save attention maps.
        device: torch device.
        num_images: Number of images to visualize.
        num_workers: DataLoader workers.
    """
    if not val_data_path or not os.path.isdir(val_data_path):
        return

    try:
        from PIL import Image
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib or PIL not available, skipping attention visualization")
        return

    # Load raw images (no heavy augmentation) for visualization
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = datasets.ImageFolder(val_data_path, transform=transform)

    # Also need unnormalized images for overlay
    raw_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    raw_dataset = datasets.ImageFolder(val_data_path, transform=raw_transform)

    # Use fixed indices for consistency across epochs
    torch.manual_seed(42)
    indices = torch.randperm(len(dataset))[:num_images].tolist()

    attn_dir = os.path.join(output_dir, f'attention_epoch{epoch:04d}')
    os.makedirs(attn_dir, exist_ok=True)

    backbone.eval()
    patch_size = backbone.patch_embed.patch_size
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]

    for idx in indices:
        img_tensor = dataset[idx][0].unsqueeze(0).to(device)  # (1, 3, 224, 224)
        raw_img = raw_dataset[idx][0]  # (3, 224, 224) unnormalized

        # Get attention from last block: (1, num_heads, num_tokens, num_tokens)
        attentions = backbone.get_last_selfattention(img_tensor)
        nh = attentions.shape[1]  # number of heads

        # Keep only CLS->patch attention: (num_heads, num_patches)
        w_featmap = img_tensor.shape[-2] // patch_size
        h_featmap = img_tensor.shape[-1] // patch_size
        attn_cls = attentions[0, :, 0, 1:].reshape(nh, w_featmap, h_featmap)

        # Average across heads
        attn_mean = attn_cls.mean(0).cpu().numpy()  # (w, h)

        # Upsample to image size
        attn_upsampled = torch.nn.functional.interpolate(
            torch.from_numpy(attn_mean).unsqueeze(0).unsqueeze(0).float(),
            size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze().numpy()

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(raw_img.permute(1, 2, 0).numpy())
        axes[0].set_title(f'Original (idx={idx})')
        axes[0].axis('off')

        # Attention heatmap
        axes[1].imshow(attn_upsampled, cmap='inferno')
        axes[1].set_title(f'Attention Map (epoch {epoch})')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(raw_img.permute(1, 2, 0).numpy())
        axes[2].imshow(attn_upsampled, cmap='inferno', alpha=0.6)
        axes[2].set_title(f'Overlay (epoch {epoch})')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(attn_dir, f'attn_img{idx:05d}.png'), dpi=100)
        plt.close()

    print(f"Attention maps saved to {attn_dir}")
