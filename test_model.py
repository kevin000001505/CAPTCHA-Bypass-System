#!/usr/bin/env python3
"""
Test trained CAPTCHA model on classify_data.csv test dataset
This script will load your trained model and evaluate it on the test images
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from collections import defaultdict
import argparse


class CaptchaModelTester:
    def __init__(self, model_path, img_height=80, img_width=170, num_digits=4):
        """
        Initialize the CAPTCHA model tester.

        Args:
            model_path (str): Path to the trained model file
            img_height (int): Height of input images
            img_width (int): Width of input images
            num_digits (int): Number of digits in CAPTCHA
        """
        self.model_path = model_path
        self.img_height = img_height
        self.img_width = img_width
        self.num_digits = num_digits
        self.model = None

        self.load_model()

    def load_model(self):
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        print(f"Loading model from: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        print("✓ Model loaded successfully")

        # Print model summary
        print(f"Model has {self.model.count_params():,} parameters")

    def load_test_data(self, csv_path, base_dir=""):
        """
        Load test data from CSV file.

        Args:
            csv_path (str): Path to the CSV file with image paths and labels
            base_dir (str): Base directory for image paths

        Returns:
            tuple: (images, labels, image_paths)
        """
        print(f"Loading test data from: {csv_path}")

        # Read CSV file
        df = pd.read_csv(csv_path)
        print(f"Found {len(df)} test samples in CSV")

        images = []
        labels = []
        image_paths = []
        failed_loads = []

        for idx, row in df.iterrows():
            image_path = row["image"]
            label = str(row["label"])

            # Construct full path
            full_path = os.path.join(base_dir, image_path)

            # Validate label
            if len(label) != self.num_digits or not label.isdigit():
                print(f"Skipping invalid label: {label}")
                continue

            # Load image
            if not os.path.exists(full_path):
                failed_loads.append(f"File not found: {full_path}")
                continue

            try:
                img = Image.open(full_path)

                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize to model input size
                img = img.resize((self.img_width, self.img_height))

                # Convert to numpy array and normalize
                img_array = np.array(img, dtype=np.float32) / 255.0

                images.append(img_array)
                labels.append(label)
                image_paths.append(image_path)

            except Exception as e:
                failed_loads.append(f"Error loading {full_path}: {e}")
                continue

        if failed_loads:
            print(f"Failed to load {len(failed_loads)} images:")
            for fail in failed_loads[:5]:  # Show first 5 failures
                print(f"  {fail}")
            if len(failed_loads) > 5:
                print(f"  ... and {len(failed_loads) - 5} more")

        print(f"Successfully loaded {len(images)} test images")

        return np.array(images), labels, image_paths

    def predict_batch(self, images):
        """
        Predict labels for a batch of images.

        Args:
            images (np.array): Batch of images

        Returns:
            list: Predicted labels as strings
        """
        if self.model is None:
            raise ValueError("Model not loaded!")

        print("Running predictions...")
        predictions = self.model.predict(images, verbose=1)

        # Convert predictions to digit strings
        predicted_labels = []
        confidence_scores = []

        for i in range(len(images)):
            pred_digits = []
            confidences = []

            for j in range(self.num_digits):
                # Get probabilities for this digit position
                probs = predictions[j][i]
                digit = np.argmax(probs)
                confidence = probs[digit]

                pred_digits.append(str(digit))
                confidences.append(confidence)

            predicted_labels.append("".join(pred_digits))
            confidence_scores.append(np.mean(confidences))

        return predicted_labels, confidence_scores

    def evaluate_predictions(
        self, true_labels, predicted_labels, confidence_scores, image_paths
    ):
        """
        Evaluate prediction results with detailed metrics.

        Args:
            true_labels (list): Ground truth labels
            predicted_labels (list): Predicted labels
            confidence_scores (list): Confidence scores for predictions
            image_paths (list): Paths to test images
        """
        print("\n" + "=" * 60)
        print("CAPTCHA MODEL EVALUATION RESULTS")
        print("=" * 60)

        # Overall accuracy (all 4 digits correct)
        correct_predictions = sum(
            1 for true, pred in zip(true_labels, predicted_labels) if true == pred
        )
        overall_accuracy = correct_predictions / len(true_labels)

        print(f"\nOverall Accuracy (all 4 digits correct):")
        print(f"  {overall_accuracy:.4f} ({correct_predictions}/{len(true_labels)})")

        # Per-digit accuracy
        print(f"\nPer-digit Accuracy:")
        digit_accuracies = []
        for digit_pos in range(self.num_digits):
            true_digits = [label[digit_pos] for label in true_labels]
            pred_digits = [label[digit_pos] for label in predicted_labels]
            digit_acc = accuracy_score(true_digits, pred_digits)
            digit_accuracies.append(digit_acc)
            print(f"  Digit {digit_pos + 1}: {digit_acc:.4f}")

        avg_digit_accuracy = np.mean(digit_accuracies)
        print(f"  Average per-digit accuracy: {avg_digit_accuracy:.4f}")

        # Confidence statistics
        avg_confidence = np.mean(confidence_scores)
        min_confidence = np.min(confidence_scores)
        max_confidence = np.max(confidence_scores)

        print(f"\nConfidence Scores:")
        print(f"  Average: {avg_confidence:.4f}")
        print(f"  Min: {min_confidence:.4f}")
        print(f"  Max: {max_confidence:.4f}")

        # Confidence vs accuracy analysis
        high_conf_mask = np.array(confidence_scores) > 0.9
        if np.sum(high_conf_mask) > 0:
            high_conf_accuracy = accuracy_score(
                [true_labels[i] for i in range(len(true_labels)) if high_conf_mask[i]],
                [
                    predicted_labels[i]
                    for i in range(len(predicted_labels))
                    if high_conf_mask[i]
                ],
            )
            print(
                f"  High confidence (>0.9) predictions: {np.sum(high_conf_mask)} samples"
            )
            print(f"  High confidence accuracy: {high_conf_accuracy:.4f}")

        # Error analysis
        print(f"\nError Analysis:")
        error_indices = [
            i
            for i, (true, pred) in enumerate(zip(true_labels, predicted_labels))
            if true != pred
        ]

        if error_indices:
            print(f"  Total errors: {len(error_indices)}")

            # Show some error examples
            print(f"\nWorst Predictions (lowest confidence errors):")
            error_confidences = [confidence_scores[i] for i in error_indices]
            worst_errors = sorted(
                zip(error_indices, error_confidences), key=lambda x: x[1]
            )

            for i, (error_idx, conf) in enumerate(worst_errors[:10]):
                true_label = true_labels[error_idx]
                pred_label = predicted_labels[error_idx]
                image_path = image_paths[error_idx]
                print(
                    f"    {i+1}. {image_path}: {true_label} → {pred_label} (conf: {conf:.3f})"
                )

        # Character-level confusion analysis
        print(f"\nCharacter-level Error Patterns:")
        char_errors = defaultdict(int)
        for true, pred in zip(true_labels, predicted_labels):
            for i, (t_char, p_char) in enumerate(zip(true, pred)):
                if t_char != p_char:
                    char_errors[f"{t_char}→{p_char}"] += 1

        if char_errors:
            most_common_errors = sorted(
                char_errors.items(), key=lambda x: x[1], reverse=True
            )
            print("  Most common character confusions:")
            for error, count in most_common_errors[:10]:
                print(f"    {error}: {count} times")

        return {
            "overall_accuracy": overall_accuracy,
            "digit_accuracies": digit_accuracies,
            "avg_confidence": avg_confidence,
            "error_indices": error_indices,
            "char_errors": dict(char_errors),
        }

    def visualize_results(
        self,
        test_images,
        true_labels,
        predicted_labels,
        confidence_scores,
        image_paths,
        num_samples=12,
    ):
        """
        Visualize prediction results.
        """
        print(f"\nGenerating visualization with {num_samples} samples...")

        # Show mix of correct and incorrect predictions
        correct_indices = [
            i
            for i, (true, pred) in enumerate(zip(true_labels, predicted_labels))
            if true == pred
        ]
        error_indices = [
            i
            for i, (true, pred) in enumerate(zip(true_labels, predicted_labels))
            if true != pred
        ]

        # Select samples to show
        sample_indices = []

        # Add some correct predictions
        if len(correct_indices) > 0:
            sample_indices.extend(correct_indices[: num_samples // 2])

        # Add some errors
        if len(error_indices) > 0:
            sample_indices.extend(error_indices[: num_samples // 2])

        # Fill remaining slots
        remaining = num_samples - len(sample_indices)
        if remaining > 0:
            all_indices = list(range(len(true_labels)))
            additional = [i for i in all_indices if i not in sample_indices][:remaining]
            sample_indices.extend(additional)

        # Create visualization
        rows = 3
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
        axes = axes.flatten()

        for i, idx in enumerate(sample_indices[: rows * cols]):
            if i >= len(axes):
                break

            img = test_images[idx]
            true_label = true_labels[idx]
            pred_label = predicted_labels[idx]
            confidence = confidence_scores[idx]
            image_path = image_paths[idx]

            # Display image
            axes[i].imshow(img)
            axes[i].axis("off")

            # Color code: green for correct, red for incorrect
            color = "green" if true_label == pred_label else "red"
            status = "✓" if true_label == pred_label else "✗"

            title = f"{status} True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}"
            axes[i].set_title(title, color=color, fontsize=10)

        # Hide empty subplots
        for i in range(len(sample_indices), len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig("test_results_visualization.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("Visualization saved as: test_results_visualization.png")

    def save_detailed_results(
        self,
        true_labels,
        predicted_labels,
        confidence_scores,
        image_paths,
        output_path="test_results.csv",
    ):
        """
        Save detailed results to CSV file.
        """
        results_df = pd.DataFrame(
            {
                "image_path": image_paths,
                "true_label": true_labels,
                "predicted_label": predicted_labels,
                "confidence": confidence_scores,
                "correct": [
                    true == pred for true, pred in zip(true_labels, predicted_labels)
                ],
            }
        )

        results_df.to_csv(output_path, index=False)
        print(f"Detailed results saved to: {output_path}")

        return results_df


def main():
    """Main function to test the model."""
    parser = argparse.ArgumentParser(
        description="Test CAPTCHA model on classify_data.csv"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="final_combined_captcha_model.h5",
        help="Path to trained model file",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="classify_data.csv",
        help="Path to test data CSV file",
    )
    parser.add_argument(
        "--base_dir", type=str, default="", help="Base directory for image paths"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualization of results"
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Save detailed results to CSV"
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Available model files:")
        model_files = [f for f in os.listdir(".") if f.endswith(".h5")]
        for f in model_files:
            print(f"  {f}")
        return

    # Initialize tester
    print("=== CAPTCHA Model Testing ===")
    tester = CaptchaModelTester(args.model)

    # Load test data
    test_images, true_labels, image_paths = tester.load_test_data(
        args.csv, args.base_dir
    )

    if len(test_images) == 0:
        print("No test images loaded. Please check your CSV file and image paths.")
        return

    # Make predictions
    predicted_labels, confidence_scores = tester.predict_batch(test_images)

    # Evaluate results
    results = tester.evaluate_predictions(
        true_labels, predicted_labels, confidence_scores, image_paths
    )

    # Optional visualization
    if args.visualize:
        tester.visualize_results(
            test_images, true_labels, predicted_labels, confidence_scores, image_paths
        )

    # Optional detailed results saving
    if args.save_results:
        tester.save_detailed_results(
            true_labels, predicted_labels, confidence_scores, image_paths
        )

    print(f"\n=== Testing Complete ===")
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Average Confidence: {results['avg_confidence']:.4f}")


if __name__ == "__main__":
    main()
