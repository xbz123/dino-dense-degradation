#!/usr/bin/env python3
"""
Prepare ImageNet-100 dataset for DINO training.

Downloads from HuggingFace (clane9/imagenet-100) and converts to
ImageFolder format (class-subfolder structure) expected by torchvision.

Usage:
    # In Colab:
    !pip install datasets Pillow
    !python prepare_data.py --output_dir /content/imagenet100

    # Or save to Google Drive for persistence:
    !python prepare_data.py --output_dir /content/drive/MyDrive/imagenet100

The output structure will be:
    /content/imagenet100/
    ├── train/
    │   ├── n01440764/
    │   │   ├── 00000.JPEG
    │   │   └── ...
    │   ├── n01443537/
    │   └── ...
    └── val/
        ├── n01440764/
        └── ...
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Prepare ImageNet-100 for DINO')
    parser.add_argument('--output_dir', type=str, default='/content/imagenet100',
                        help='Output directory for the dataset')
    parser.add_argument('--hf_dataset', type=str, default='clane9/imagenet-100',
                        help='HuggingFace dataset identifier')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for downloading')
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing 'datasets' library...")
        os.system(f'{sys.executable} -m pip install -q datasets')
        from datasets import load_dataset

    output_dir = Path(args.output_dir)
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'

    # Check if already prepared
    if train_dir.exists() and val_dir.exists():
        n_train = sum(1 for _ in train_dir.rglob('*.JPEG'))
        n_val = sum(1 for _ in val_dir.rglob('*.JPEG'))
        if n_train > 100000:
            print(f"Dataset already prepared: {n_train} train, {n_val} val images")
            return
        print(f"Incomplete dataset found ({n_train} train, {n_val} val). Re-downloading...")

    print(f"Downloading ImageNet-100 from HuggingFace ({args.hf_dataset})...")
    print("This may take 10-30 minutes depending on connection speed.")

    # Load dataset
    ds = load_dataset(args.hf_dataset, num_proc=args.num_workers)

    # Get label mapping
    label_names = ds['train'].features['label'].names
    print(f"Dataset loaded: {len(ds['train'])} train, {len(ds['validation'])} val images")
    print(f"Classes: {len(label_names)}")

    def save_split(split_data, split_dir, split_name):
        """Save a dataset split to ImageFolder format."""
        split_dir.mkdir(parents=True, exist_ok=True)
        counts = {}

        for i, example in enumerate(split_data):
            label_idx = example['label']
            class_name = label_names[label_idx]

            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)

            # Track per-class count for unique filenames
            counts[class_name] = counts.get(class_name, 0) + 1
            filename = f'{counts[class_name]:05d}.JPEG'

            img = example['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(class_dir / filename, 'JPEG', quality=95)

            if (i + 1) % 5000 == 0:
                print(f"  [{split_name}] Saved {i + 1}/{len(split_data)} images...")

        total = sum(counts.values())
        print(f"  [{split_name}] Done: {total} images in {len(counts)} classes")

    print("\nSaving training set...")
    save_split(ds['train'], train_dir, 'train')

    print("\nSaving validation set...")
    save_split(ds['validation'], val_dir, 'val')

    # Print summary
    n_train_classes = len(list(train_dir.iterdir()))
    n_val_classes = len(list(val_dir.iterdir()))
    n_train = sum(1 for _ in train_dir.rglob('*.JPEG'))
    n_val = sum(1 for _ in val_dir.rglob('*.JPEG'))

    print(f"\n{'='*50}")
    print(f"ImageNet-100 prepared successfully!")
    print(f"  Location:      {output_dir}")
    print(f"  Train:         {n_train} images in {n_train_classes} classes")
    print(f"  Validation:    {n_val} images in {n_val_classes} classes")
    print(f"{'='*50}")

    # Save class list for reference
    with open(output_dir / 'classes.txt', 'w') as f:
        for name in sorted(label_names):
            f.write(name + '\n')
    print(f"Class list saved to {output_dir / 'classes.txt'}")


if __name__ == '__main__':
    main()
