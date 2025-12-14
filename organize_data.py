#!/usr/bin/env python3
"""
Script to organize captcha data by removing failed images and organizing successful ones.
"""

import os
import json
import shutil
from pathlib import Path


def organize_captcha_data():
    """
    Organize captcha data by:
    1. Identifying failed (unlabeled) images
    2. Moving successful images to a clean directory
    3. Removing failed images
    4. Creating a clean labels file
    """

    data_dir = "/Users/kevinhsu/Downloads/chicken/data/captcha_debug"
    clean_dir = "/Users/kevinhsu/Downloads/chicken/data/captcha_clean"
    labels_file = os.path.join(data_dir, "labels.json")

    # Create clean directory
    os.makedirs(clean_dir, exist_ok=True)

    # Load labels
    with open(labels_file, "r") as f:
        labels = json.load(f)

    # Get all PNG files
    all_images = [
        f for f in os.listdir(data_dir) if f.endswith(".png") and f != "img_test.py"
    ]
    labeled_images = set(labels.keys())

    # Find failed images
    failed_images = [img for img in all_images if img not in labeled_images]

    print(f"Total images: {len(all_images)}")
    print(f"Labeled images: {len(labeled_images)}")
    print(f"Failed images: {len(failed_images)}")

    if failed_images:
        print(f"\nFailed images to be deleted:")
        for img in failed_images:
            print(f"  - {img}")

        # Delete failed images
        for img in failed_images:
            img_path = os.path.join(data_dir, img)
            if os.path.exists(img_path):
                os.remove(img_path)
                print(f"Deleted: {img}")

    # Copy successful images to clean directory
    print(f"\nCopying {len(labeled_images)} successful images to clean directory...")
    successful_count = 0

    for img_name in labeled_images:
        src_path = os.path.join(data_dir, img_name)
        dst_path = os.path.join(clean_dir, img_name)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            successful_count += 1
        else:
            print(f"Warning: {img_name} not found in source directory")

    # Create clean labels file
    clean_labels_file = os.path.join(clean_dir, "labels.json")
    with open(clean_labels_file, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nOrganization complete:")
    print(f"  - Copied {successful_count} images to {clean_dir}")
    print(f"  - Created clean labels file: {clean_labels_file}")
    print(f"  - Deleted {len(failed_images)} failed images from original directory")

    # Verify the clean data
    print(f"\nVerification:")
    clean_images = [f for f in os.listdir(clean_dir) if f.endswith(".png")]
    print(f"  - Clean directory contains {len(clean_images)} images")
    print(f"  - Labels file contains {len(labels)} entries")

    if len(clean_images) == len(labels):
        print("  ✓ All images have corresponding labels")
    else:
        print("  ✗ Mismatch between images and labels")

    return clean_dir, len(labels)


if __name__ == "__main__":
    clean_dir, num_samples = organize_captcha_data()
    print(f"\nReady for training with {num_samples} samples in {clean_dir}")
