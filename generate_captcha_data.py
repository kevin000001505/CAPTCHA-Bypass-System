#!/usr/bin/env python3
"""
CAPTCHA Generator for Training Data
Based on https://ypw.io/captcha/ tutorial
Generate synthetic CAPTCHA images with only numbers (0-9) to increase training data
"""

import os
import json
import random
import string
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from captcha.image import ImageCaptcha
from tqdm import tqdm
import argparse


class CaptchaGenerator:
    def __init__(self, width=170, height=80, n_len=4):
        """
        Initialize CAPTCHA generator with only digits.

        Args:
            width (int): Image width
            height (int): Image height
            n_len (int): Number of characters in CAPTCHA
        """
        self.width = width
        self.height = height
        self.n_len = n_len
        # Only use digits 0-9 for consistency with your existing data
        self.characters = string.digits  # "0123456789"
        self.n_class = len(self.characters)

        print(f"Using characters: {self.characters}")
        print(f"Image size: {width}x{height}")
        print(f"CAPTCHA length: {n_len}")

    def generate_single_captcha(self, custom_text=None):
        """
        Generate a single CAPTCHA image.

        Args:
            custom_text (str): Custom text for CAPTCHA, if None generates random

        Returns:
            tuple: (PIL Image, text)
        """
        generator = ImageCaptcha(width=self.width, height=self.height)

        if custom_text is None:
            # Generate random 4-digit string
            text = "".join([random.choice(self.characters) for _ in range(self.n_len)])
        else:
            text = custom_text

        img = generator.generate_image(text)
        return img, text

    def preview_captcha(self, text=None):
        """
        Preview a single CAPTCHA image.
        """
        img, actual_text = self.generate_single_captcha(text)

        plt.figure(figsize=(8, 4))
        plt.imshow(img)
        plt.title(f"CAPTCHA: {actual_text}")
        plt.axis("off")
        plt.show()

        return img, actual_text

    def generate_batch(self, batch_size=32):
        """
        Generate a batch of CAPTCHA images for training.

        Args:
            batch_size (int): Number of images to generate

        Returns:
            tuple: (X, y, labels) where X is image array, y is one-hot encoded, labels is text
        """
        X = np.zeros((batch_size, self.height, self.width, 3), dtype=np.uint8)
        y = [
            np.zeros((batch_size, self.n_class), dtype=np.uint8)
            for _ in range(self.n_len)
        ]
        labels = []

        generator = ImageCaptcha(width=self.width, height=self.height)

        for i in range(batch_size):
            # Generate random 4-digit string
            text = "".join([random.choice(self.characters) for _ in range(self.n_len)])

            # Generate image
            img = generator.generate_image(text)
            X[i] = np.array(img)

            # Convert to one-hot encoding for each digit position
            for j, ch in enumerate(text):
                y[j][i, :] = 0
                y[j][i, self.characters.find(ch)] = 1

            labels.append(text)

        return X, y, labels

    def generate_dataset(
        self,
        num_images=1000,
        output_dir="data/generated_captcha",
        save_images=True,
        batch_size=100,
    ):
        """
        Generate a large dataset of CAPTCHA images.

        Args:
            num_images (int): Total number of images to generate
            output_dir (str): Directory to save images
            save_images (bool): Whether to save individual image files
            batch_size (int): Number of images to generate per batch
        """
        print(f"Generating {num_images} CAPTCHA images...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        generator = ImageCaptcha(width=self.width, height=self.height)
        labels_dict = {}

        # Generate images in batches to avoid memory issues
        num_batches = (num_images + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_images)
            current_batch_size = end_idx - start_idx

            for i in range(current_batch_size):
                img_idx = start_idx + i

                # Generate random 4-digit string
                text = "".join(
                    [random.choice(self.characters) for _ in range(self.n_len)]
                )

                # Generate image
                img = generator.generate_image(text)

                if save_images:
                    # Save image file
                    filename = f"captcha_generated_{img_idx:06d}.png"
                    filepath = os.path.join(output_dir, filename)
                    img.save(filepath)
                    labels_dict[filename] = text
                else:
                    # Just add to labels dict with index
                    labels_dict[f"generated_{img_idx:06d}"] = text

        # Save labels to JSON file
        labels_path = os.path.join(output_dir, "labels.json")
        with open(labels_path, "w") as f:
            json.dump(labels_dict, f, indent=2)

        print(f"Generated {num_images} images")
        print(f"Saved to: {output_dir}")
        print(f"Labels saved to: {labels_path}")

        return labels_dict

    def create_data_generator(self, batch_size=32):
        """
        Create an infinite data generator for training (similar to website tutorial).
        This is memory efficient and can generate unlimited data.

        Args:
            batch_size (int): Batch size

        Yields:
            tuple: (X, y) where X is images, y is list of one-hot encoded labels
        """
        generator = ImageCaptcha(width=self.width, height=self.height)

        while True:
            X = np.zeros((batch_size, self.height, self.width, 3), dtype=np.float32)
            y = [
                np.zeros((batch_size, self.n_class), dtype=np.float32)
                for _ in range(self.n_len)
            ]

            for i in range(batch_size):
                # Generate random 4-digit string
                text = "".join(
                    [random.choice(self.characters) for _ in range(self.n_len)]
                )

                # Generate image and normalize to [0,1]
                img = generator.generate_image(text)
                X[i] = np.array(img, dtype=np.float32) / 255.0

                # Convert to one-hot encoding for each digit position
                for j, ch in enumerate(text):
                    y[j][i, :] = 0
                    y[j][i, self.characters.find(ch)] = 1

            yield X, y

    def decode_prediction(self, y_pred):
        """
        Decode model prediction back to text.

        Args:
            y_pred: Model prediction (list of probabilities for each digit)

        Returns:
            str: Decoded text
        """
        decoded = []
        for i in range(self.n_len):
            digit_idx = np.argmax(y_pred[i])
            decoded.append(self.characters[digit_idx])
        return "".join(decoded)

    def show_batch_samples(self, batch_size=8):
        """
        Show sample images from a generated batch.
        """
        X, _, labels = self.generate_batch(batch_size)

        _, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.flatten()

        for i in range(min(batch_size, 8)):
            axes[i].imshow(X[i])
            axes[i].set_title(f"Label: {labels[i]}")
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()


def main():
    """Main function to generate CAPTCHA dataset."""
    parser = argparse.ArgumentParser(description="Generate CAPTCHA training data")
    parser.add_argument(
        "--num_images",
        type=int,
        default=5000,
        help="Number of images to generate (default: 5000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/generated_captcha",
        help="Output directory (default: data/generated_captcha)",
    )
    parser.add_argument(
        "--preview", action="store_true", help="Show preview of generated CAPTCHAs"
    )
    parser.add_argument(
        "--width", type=int, default=170, help="Image width (default: 170)"
    )
    parser.add_argument(
        "--height", type=int, default=80, help="Image height (default: 80)"
    )

    args = parser.parse_args()

    # Initialize generator
    captcha_gen = CaptchaGenerator(width=args.width, height=args.height)

    if args.preview:
        print("Showing preview of generated CAPTCHAs...")
        captcha_gen.show_batch_samples()

        # Show individual preview
        print("Single CAPTCHA preview:")
        captcha_gen.preview_captcha()
    else:
        # Generate dataset
        print(f"Generating {args.num_images} CAPTCHA images...")
        labels_dict = captcha_gen.generate_dataset(
            num_images=args.num_images, output_dir=args.output_dir
        )

        print("Dataset generation complete!")
        print(f"Generated {len(labels_dict)} images with labels")

        # Show some sample labels
        sample_labels = list(labels_dict.items())[:5]
        print("Sample labels:")
        for filename, label in sample_labels:
            print(f"  {filename}: {label}")


if __name__ == "__main__":
    main()
