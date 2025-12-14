#!/usr/bin/env python3
"""
Clean the labels file to remove any invalid characters
"""

import json
import re


def clean_labels_file():
    """Clean the labels file and remove any invalid entries."""

    labels_file = "/Users/kevinhsu/Downloads/chicken/data/captcha_clean/labels.json"

    # Load labels
    with open(labels_file, "r") as f:
        labels = json.load(f)

    cleaned_labels = {}
    invalid_labels = []

    for filename, label in labels.items():
        # Remove any non-digit characters and keep only first 4 digits
        cleaned_label = re.sub(r"[^0-9]", "", str(label))

        if len(cleaned_label) >= 4:
            cleaned_labels[filename] = cleaned_label[:4]
        else:
            invalid_labels.append((filename, label))
            print(
                f"Invalid label: {filename} -> '{label}' (cleaned: '{cleaned_label}')"
            )

    print(f"Original labels: {len(labels)}")
    print(f"Cleaned labels: {len(cleaned_labels)}")
    print(f"Invalid labels: {len(invalid_labels)}")

    if invalid_labels:
        print("Invalid labels found:")
        for filename, label in invalid_labels:
            print(f"  {filename}: '{label}'")

    # Save cleaned labels
    with open(labels_file, "w") as f:
        json.dump(cleaned_labels, f, indent=2)

    print(f"Labels file cleaned and saved with {len(cleaned_labels)} valid entries")
    return len(cleaned_labels)


if __name__ == "__main__":
    clean_labels_file()
