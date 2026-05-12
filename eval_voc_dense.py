"""
PASCAL VOC Linear Probing for Dense Degradation Evaluation
===========================================================
This script evaluates DINO checkpoints on PASCAL VOC 2012 semantic segmentation
using the standard linear evaluation protocol (frozen backbone + 1x1 conv head).

Usage (on Colab):
    !python eval_voc_dense.py --ckpt_dir /content/drive/MyDrive/dino_checkpoints

The script will:
1. Download PASCAL VOC 2012 automatically
2. Load each checkpoint, freeze backbone, train a linear head
3. Evaluate mIoU on the validation set
4. Plot the "Dense Performance throughout Pretraining" curve
"""

import os
import re
import sys
import json
import math
import argparse
import difflib
import numpy as np
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =====================================================================
# 1. Vision Transformer (self-contained, no external dependency needed)
# =====================================================================

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# =====================================================================
# 2. VOC Segmentation Dataset with proper preprocessing
# =====================================================================

class VOCSegDataset(torch.utils.data.Dataset):
    """PASCAL VOC 2012 Segmentation dataset for linear probing."""

    # VOC has 21 classes (20 objects + background)
    NUM_CLASSES = 21
    IGNORE_INDEX = 255

    def __init__(self, root, image_set='train', img_size=480, patch_size=16):
        self.img_size = img_size
        self.patch_size = patch_size

        # Download VOC if needed
        self.voc = datasets.VOCSegmentation(
            root=root, year='2012', image_set=image_set, download=True
        )

        # Make img_size divisible by patch_size
        self.crop_size = (img_size // patch_size) * patch_size

        self.img_transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, target = self.voc[idx]

        # Transform image
        img = self.img_transform(img)

        # Transform segmentation mask: resize with nearest neighbor (no interpolation!)
        target = target.resize((self.crop_size, self.crop_size), resample=0)  # NEAREST
        target = torch.from_numpy(np.array(target)).long()

        return img, target


# =====================================================================
# 3. Linear Segmentation Head
# =====================================================================

class LinearSegHead(nn.Module):
    """Simple 1x1 conv linear head for semantic segmentation."""

    def __init__(self, embed_dim, num_classes, patch_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size
        # 1x1 convolution: maps embed_dim -> num_classes
        self.linear = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: (B, N_patches, embed_dim) from ViT backbone
        Returns:
            logits: (B, num_classes, H, W) at original image resolution
        """
        B, N, D = patch_tokens.shape
        # Reshape patch tokens to spatial grid
        h = w = int(math.sqrt(N))
        x = patch_tokens.transpose(1, 2).reshape(B, D, h, w)  # (B, D, h, w)
        # Apply 1x1 conv
        logits = self.linear(x)  # (B, num_classes, h, w)
        # Upsample to original resolution
        logits = F.interpolate(logits, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return logits


# =====================================================================
# 4. mIoU Calculation
# =====================================================================

def compute_miou(pred, target, num_classes=21, ignore_index=255):
    """Compute mean Intersection over Union."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        # Ignore pixels with ignore_index
        valid = (target != ignore_index)
        pred_cls = pred_cls & valid
        target_cls = target_cls & valid

        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()

        if union == 0:
            # Class not present in this batch, skip
            continue
        ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0
    return np.mean(ious)


# =====================================================================
# 5. Load DINO checkpoint into backbone
# =====================================================================

def load_dino_backbone(ckpt_path, arch='vit_small', patch_size=16):
    """Load a DINO checkpoint and return the frozen teacher backbone."""
    model = vit_small(patch_size=patch_size)

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Try to load teacher weights (preferred) or student weights
    if 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
    elif 'student' in checkpoint:
        state_dict = checkpoint['student']
    else:
        state_dict = checkpoint

    # Clean up state dict keys
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Remove 'module.' prefix (from DDP)
        k = k.replace('module.', '')
        # Remove 'backbone.' prefix (from MultiCropWrapper)
        k = k.replace('backbone.', '')
        # Skip DINOHead keys
        if k.startswith('head.') or k.startswith('dino_head.'):
            continue
        new_state_dict[k] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"  Loaded backbone from {os.path.basename(ckpt_path)}: {msg}")

    # Freeze all backbone parameters
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model


# =====================================================================
# 6. Extract epoch number from checkpoint filename
# =====================================================================

def extract_epoch(filename):
    """Extract epoch number from checkpoint filename."""
    name = os.path.basename(filename)
    # Match patterns like: checkpoint0020.pth, checkpoint03.pth, checkpoint 23.pth, checkpoint108.pth
    match = re.search(r'checkpoint\s*0*(\d+)\.pth', name)
    if match:
        return int(match.group(1))
    return None


def discover_checkpoints(ckpt_dir):
    """Find checkpoint files and fail early with actionable path hints."""
    if not os.path.isdir(ckpt_dir):
        parent = os.path.dirname(os.path.abspath(ckpt_dir))
        folder = os.path.basename(os.path.abspath(ckpt_dir))
        hint = ""

        if os.path.isdir(parent):
            candidates = [
                name for name in os.listdir(parent)
                if os.path.isdir(os.path.join(parent, name))
            ]
            close = difflib.get_close_matches(folder, candidates, n=5, cutoff=0.45)
            if close:
                options = "\n".join(
                    f"  - {os.path.join(parent, name)}" for name in close
                )
                hint = f"\nDid you mean one of these folders?\n{options}"

        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {ckpt_dir}{hint}\n"
            "Pass the correct folder with --ckpt_dir, for example:\n"
            "  --ckpt_dir /content/drive/MyDrive/dinocehckpoint"
        )

    ckpt_files = []
    for f in os.listdir(ckpt_dir):
        if f.endswith('.pth'):
            epoch = extract_epoch(f)
            if epoch is not None:
                ckpt_files.append((epoch, os.path.join(ckpt_dir, f)))

    ckpt_files.sort(key=lambda x: x[0])
    if not ckpt_files:
        raise FileNotFoundError(
            f"No recognizable checkpoint*.pth files found in: {ckpt_dir}\n"
            "Expected names like checkpoint0020.pth or checkpoint108.pth."
        )

    return ckpt_files


# =====================================================================
# 7. Train and Evaluate for a single checkpoint
# =====================================================================

@torch.no_grad()
def extract_features(model, dataloader, device, feature_dtype=torch.float16):
    """Extract patch token features from frozen backbone for all images.

    Store cached features/masks compactly on CPU so Colab RAM does not spike.
    """
    n_images = len(dataloader.dataset)
    features = None
    targets_all = None
    offset = 0

    for imgs, targets in dataloader:
        imgs = imgs.to(device)
        # Get intermediate layers (last block output)
        output = model.get_intermediate_layers(imgs, n=1)[0]
        # Remove CLS token, keep only patch tokens
        patch_tokens = output[:, 1:]  # (B, N_patches, embed_dim)

        batch_size = patch_tokens.shape[0]
        if features is None:
            features = torch.empty(
                (n_images, patch_tokens.shape[1], patch_tokens.shape[2]),
                dtype=feature_dtype,
                device='cpu',
            )
            targets_all = torch.empty(
                (n_images, targets.shape[1], targets.shape[2]),
                dtype=torch.uint8,
                device='cpu',
            )

        end = offset + batch_size
        features[offset:end].copy_(patch_tokens.cpu().to(feature_dtype))
        targets_all[offset:end].copy_(targets.to(torch.uint8))
        offset = end

    return features, targets_all


def train_linear_head(features_train, targets_train, features_val, targets_val,
                      embed_dim, num_classes, patch_size, img_size, device,
                      epochs=15, lr=0.01, batch_size=32):
    """Train linear segmentation head and return validation mIoU."""
    head = LinearSegHead(embed_dim, num_classes, patch_size, img_size).to(device)
    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_train = features_train.shape[0]

    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n_train)
        total_loss = 0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            feat = features_train[idx].to(device=device, dtype=torch.float32)
            tgt = targets_train[idx].to(device=device, dtype=torch.long)

            logits = head(feat)
            loss = F.cross_entropy(logits, tgt, ignore_index=255)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

    # Evaluate on validation set
    head.eval()
    all_miou = []
    with torch.no_grad():
        for i in range(0, features_val.shape[0], batch_size):
            feat = features_val[i:i + batch_size].to(device=device, dtype=torch.float32)
            tgt = targets_val[i:i + batch_size].to(device=device, dtype=torch.long)

            logits = head(feat)
            pred = logits.argmax(dim=1)
            miou = compute_miou(pred, tgt, num_classes=num_classes)
            all_miou.append(miou)

    return np.mean(all_miou)


# =====================================================================
# 8. Plot the degradation curve
# =====================================================================

def plot_degradation_curve(results, output_path):
    """Plot mIoU vs Epoch, mimicking the paper's style."""
    epochs = [r['epoch'] for r in results]
    mious = [r['miou'] for r in results]

    # Find peak
    best_idx = np.argmax(mious)
    best_epoch = epochs[best_idx]
    best_miou = mious[best_idx]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Main curve
    ax.plot(epochs, mious, 'b-o', linewidth=2, markersize=6, label='mIoU (VOC)')

    # Mark the peak with a vertical dashed line (like the paper)
    ax.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Best: Epoch {best_epoch}')

    # Mark the peak point
    ax.plot(best_epoch, best_miou, 'r*', markersize=15, zorder=5)

    # Annotate the degradation
    if best_idx < len(epochs) - 1:
        last_miou = mious[-1]
        diff = last_miou - best_miou
        ax.annotate(f'Diff: {diff:.1f}',
                    xy=(epochs[-1], last_miou),
                    xytext=(epochs[-1] - 15, last_miou + 1.5),
                    fontsize=10, color='red',
                    arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('mIoU (%)', fontsize=13)
    ax.set_title('DINO Dense Performance throughout Pretraining\n(PASCAL VOC Linear Probing)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(epochs) - 5, max(epochs) + 5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_path}")
    plt.close()


# =====================================================================
# 9. Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser('VOC Dense Degradation Evaluation')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Directory containing DINO checkpoint .pth files')
    parser.add_argument('--voc_root', type=str, default='./data',
                        help='Root directory to download/store VOC dataset')
    parser.add_argument('--arch', type=str, default='vit_small')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=480,
                        help='Image size for evaluation (will be rounded to patch_size multiple)')
    parser.add_argument('--train_epochs', type=int, default=15,
                        help='Number of epochs to train linear head per checkpoint')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--feature_dtype', type=str, default='float16',
                        choices=['float16', 'float32'],
                        help='CPU dtype for cached patch features; float16 saves Colab RAM')
    parser.add_argument('--output_dir', type=str, default='./dense_eval_results',
                        help='Directory to save results and plot')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Make img_size divisible by patch_size
    args.img_size = (args.img_size // args.patch_size) * args.patch_size
    print(f"Image size (adjusted): {args.img_size}")
    feature_dtype = torch.float16 if args.feature_dtype == 'float16' else torch.float32
    print(f"Cached feature dtype: {args.feature_dtype}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Fail fast before downloading VOC if the Google Drive path is wrong.
    ckpt_files = discover_checkpoints(args.ckpt_dir)

    # ---- Step 1: Prepare VOC dataset ----
    print("\n" + "=" * 60)
    print("Step 1: Preparing PASCAL VOC 2012 dataset...")
    print("=" * 60)

    train_dataset = VOCSegDataset(args.voc_root, image_set='train',
                                  img_size=args.img_size, patch_size=args.patch_size)
    val_dataset = VOCSegDataset(args.voc_root, image_set='val',
                                img_size=args.img_size, patch_size=args.patch_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")

    # ---- Step 2: Discover checkpoints ----
    print("\n" + "=" * 60)
    print("Step 2: Discovering checkpoint files...")
    print("=" * 60)

    print(f"Found {len(ckpt_files)} checkpoints:")
    for epoch, path in ckpt_files:
        print(f"  Epoch {epoch:>4d}: {os.path.basename(path)}")

    # ---- Step 3: Evaluate each checkpoint ----
    print("\n" + "=" * 60)
    print("Step 3: Evaluating checkpoints...")
    print("=" * 60)

    embed_dim = 384  # vit_small

    results = []
    for i, (epoch, ckpt_path) in enumerate(ckpt_files):
        print(f"\n[{i + 1}/{len(ckpt_files)}] Evaluating Epoch {epoch}...")

        # Load frozen backbone
        model = load_dino_backbone(ckpt_path, arch=args.arch, patch_size=args.patch_size)
        model = model.to(device)

        # Extract features (frozen backbone, no gradients)
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

        # Free backbone GPU memory
        del model
        torch.cuda.empty_cache()

        # Train linear head and evaluate
        print(f"  Training linear head ({args.train_epochs} epochs)...")
        miou = train_linear_head(
            features_train, targets_train,
            features_val, targets_val,
            embed_dim=embed_dim,
            num_classes=VOCSegDataset.NUM_CLASSES,
            patch_size=args.patch_size,
            img_size=args.img_size,
            device=device,
            epochs=args.train_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )

        print(f"  ✅ Epoch {epoch}: mIoU = {miou * 100:.2f}%")
        results.append({'epoch': epoch, 'miou': miou * 100})

        # Free extracted features
        del features_train, targets_train, features_val, targets_val
        torch.cuda.empty_cache()

    # ---- Step 4: Save results and plot ----
    print("\n" + "=" * 60)
    print("Step 4: Saving results and generating plot...")
    print("=" * 60)

    # Save raw results as JSON
    results_path = os.path.join(args.output_dir, 'voc_miou_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Print summary table
    print("\n" + "-" * 40)
    print(f"{'Epoch':>8s} | {'mIoU (%)':>10s}")
    print("-" * 40)
    for r in results:
        print(f"{r['epoch']:>8d} | {r['miou']:>10.2f}")
    print("-" * 40)

    best = max(results, key=lambda x: x['miou'])
    last = results[-1]
    change = last['miou'] - best['miou']
    trend = "degradation" if change < 0 else "no degradation observed"
    print(f"\nBest:  Epoch {best['epoch']}, mIoU = {best['miou']:.2f}%")
    print(f"Last:  Epoch {last['epoch']}, mIoU = {last['miou']:.2f}%")
    print(f"Diff:  {change:.2f}% ({trend})")

    # Generate plot
    plot_path = os.path.join(args.output_dir, 'dense_degradation_voc.png')
    plot_degradation_curve(results, plot_path)

    print("\n🎉 All done!")


if __name__ == '__main__':
    main()
