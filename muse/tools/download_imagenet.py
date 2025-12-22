#!/usr/bin/env python3
"""
Script to download ImageNet-1k dataset from Hugging Face.

Before running this script, you need to:
1. Create a Hugging Face account at https://huggingface.co/
2. Go to https://huggingface.co/datasets/ILSVRC/imagenet-1k
3. Click "Access repository" and agree to the terms
4. Create an access token at https://huggingface.co/settings/tokens
5. Login via: huggingface-cli login

Usage:
    python download_imagenet.py --split train --output_dir /path/to/save
    python download_imagenet.py --split all --output_dir /path/to/save
"""

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download ImageNet-1k dataset from Hugging Face"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "validation", "test", "all"],
        help="Which split to download (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the dataset (default: HF cache)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Number of processes for parallel downloading (default: 8)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (doesn't download to disk)",
    )
    parser.add_argument(
        "--save_to_disk",
        action="store_true",
        help="Save dataset to disk in Arrow format",
    )
    parser.add_argument(
        "--save_as_images",
        action="store_true",
        help="Save images as individual files with folder structure",
    )
    return parser.parse_args()


def check_authentication():
    """Check if user is authenticated with Hugging Face."""
    from huggingface_hub import HfFolder

    token = HfFolder.get_token()
    if token is None:
        print("=" * 60)
        print("ERROR: You are not logged in to Hugging Face!")
        print("=" * 60)
        print("\nPlease follow these steps:")
        print("1. Go to https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        print("2. Click 'Access repository' and agree to the terms")
        print("3. Create a token at https://huggingface.co/settings/tokens")
        print("4. Run: huggingface-cli login")
        print("=" * 60)
        return False
    return True


def download_dataset(args):
    """Download the ImageNet-1k dataset."""
    from datasets import load_dataset

    print(f"Loading ImageNet-1k dataset...")
    print(f"Split: {args.split}")
    print(f"Streaming: {args.streaming}")

    # Determine splits to download
    if args.split == "all":
        splits = ["train", "validation", "test"]
    else:
        splits = [args.split]

    datasets = {}
    for split in splits:
        print(f"\nDownloading {split} split...")
        try:
            ds = load_dataset(
                "ILSVRC/imagenet-1k",
                split=split,
                streaming=args.streaming,
                num_proc=args.num_proc if not args.streaming else None,
                trust_remote_code=True,
            )
            datasets[split] = ds
            
            if not args.streaming:
                print(f"  {split}: {len(ds)} examples")
            else:
                print(f"  {split}: streaming mode (size unknown)")
                
        except Exception as e:
            print(f"  Error downloading {split}: {e}")
            continue

    return datasets


def save_dataset_to_disk(datasets, output_dir):
    """Save dataset in Arrow format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split, ds in datasets.items():
        split_path = output_path / split
        print(f"Saving {split} to {split_path}...")
        ds.save_to_disk(str(split_path))
        print(f"  Done!")


def save_as_images(datasets, output_dir):
    """Save dataset as individual image files organized by class."""
    from tqdm import tqdm

    output_path = Path(output_dir)

    for split, ds in datasets.items():
        split_path = output_path / split
        split_path.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving {split} images to {split_path}...")

        # Get class names if available
        if hasattr(ds.features["label"], "names"):
            class_names = ds.features["label"].names
        else:
            class_names = None

        for idx, example in enumerate(tqdm(ds, desc=f"Saving {split}")):
            image = example["image"]
            label = example["label"]

            # Create class folder
            if class_names and label >= 0:
                class_name = class_names[label]
                class_folder = split_path / f"{label:04d}_{class_name[:50]}"
            elif label >= 0:
                class_folder = split_path / f"{label:04d}"
            else:
                class_folder = split_path / "unknown"

            class_folder.mkdir(exist_ok=True)

            # Save image
            image_path = class_folder / f"{idx:08d}.jpg"
            image.save(str(image_path))


def main():
    args = parse_args()

    # Check authentication
    if not check_authentication():
        return

    # Download dataset
    datasets = download_dataset(args)

    if not datasets:
        print("No datasets were downloaded successfully.")
        return

    # Save to disk if requested
    if args.save_to_disk and args.output_dir:
        save_dataset_to_disk(datasets, args.output_dir)

    # Save as images if requested
    if args.save_as_images and args.output_dir:
        save_as_images(datasets, args.output_dir)

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)

    # Print dataset info
    print("\nDataset statistics:")
    print("  - Train: 1,281,167 images")
    print("  - Validation: 50,000 images")
    print("  - Test: 100,000 images (labels are -1/missing)")
    print("  - Classes: 1,000")

    if args.output_dir:
        print(f"\nSaved to: {args.output_dir}")
    else:
        print("\nDataset cached in HuggingFace cache directory")
        print("(typically ~/.cache/huggingface/datasets/)")


if __name__ == "__main__":
    main()
