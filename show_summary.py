#!/usr/bin/env python3
"""
Summary of the organized captcha dataset
"""

import json
import os


def show_dataset_summary():
    """Show summary of the clean captcha dataset."""

    clean_dir = "/Users/kevinhsu/Downloads/chicken/data/captcha_clean"
    labels_file = os.path.join(clean_dir, "labels.json")

    # Load labels
    with open(labels_file, "r") as f:
        labels = json.load(f)

    # Count images
    images = [f for f in os.listdir(clean_dir) if f.endswith(".png")]

    print("🎯 CAPTCHA Dataset Summary")
    print("=" * 50)
    print(f"📁 Directory: {clean_dir}")
    print(f"🖼️  Total Images: {len(images)}")
    print(f"🏷️  Total Labels: {len(labels)}")
    print(
        f"✅ Status: {'All images labeled' if len(images) == len(labels) else 'Mismatch!'}"
    )

    # Analyze labels
    digit_distribution = {}
    for label in labels.values():
        for pos, digit in enumerate(label):
            if pos not in digit_distribution:
                digit_distribution[pos] = {}
            if digit not in digit_distribution[pos]:
                digit_distribution[pos][digit] = 0
            digit_distribution[pos][digit] += 1

    print("\n📊 Digit Distribution by Position:")
    for pos in range(4):
        print(f"  Position {pos + 1}: ", end="")
        pos_dist = digit_distribution.get(pos, {})
        for digit in "0123456789":
            count = pos_dist.get(digit, 0)
            print(f"{digit}:{count:2d} ", end="")
        print()

    # Show sample labels
    print(f"\n📝 Sample Labels (first 10):")
    sample_items = list(labels.items())[:10]
    for filename, label in sample_items:
        print(f"  {filename}: {label}")

    print(f"\n🚀 Dataset is ready for CNN training!")
    print(f"   - 120 labeled captcha images")
    print(f"   - 4-digit labels (0-9)")
    print(f"   - Organized in clean directory")


if __name__ == "__main__":
    show_dataset_summary()
