#!/usr/bin/env python3
"""
Train CNN model on clean captcha dataset
"""

import sys
import os

sys.path.append("/Users/kevinhsu/Downloads/chicken")

from model import CaptchaModel


def main():
    """Train the model on clean dataset."""

    # Initialize model
    print("Initializing CAPTCHA CNN model...")
    captcha_model = CaptchaModel(img_height=80, img_width=170)

    # Load clean data
    clean_data_dir = "/Users/kevinhsu/Downloads/chicken/data/captcha_clean"
    print(f"Loading clean dataset from: {clean_data_dir}")

    try:
        # Load data (skip GPT-4 since we already have labels)
        x_data, y_data, y_raw = captcha_model.load_data(
            clean_data_dir,
            use_gpt4_labeling=False,  # We already have labels
            labels_file="labels.json",
        )

        print(f"Successfully loaded {len(x_data)} images")
        print(f"Image shape: {x_data.shape}")
        print(f"Sample labels: {y_raw[:5]}")

        # Train the model
        print("\nStarting training...")
        captcha_model.create_model()

        # Train with the clean data
        history = captcha_model.train(
            x_data,
            y_data,
            validation_split=0.2,
            epochs=30,
            batch_size=16,  # Smaller batch size for better training with limited data
        )

        # Plot training history
        captcha_model.plot_training_history(history)

        # Save the trained model
        captcha_model.save_model("captcha_cnn_model.h5")

        print("\nTraining completed successfully!")
        print("Model saved as: captcha_cnn_model.h5")

        # Test on a few samples
        print("\nTesting model on sample images...")
        import numpy as np
        from sklearn.model_selection import train_test_split

        # Split data for testing
        _, x_test, _, y_raw_test = train_test_split(
            x_data, y_raw, test_size=0.2, random_state=42
        )

        # Split labels for testing
        y_test = []
        for i in range(captcha_model.num_digits):
            _, y_digit_test, _, _ = train_test_split(
                y_data[i], y_raw, test_size=0.2, random_state=42
            )
            y_test.append(y_digit_test)

        # Evaluate model
        accuracy, _ = captcha_model.evaluate_model(x_test, y_test, y_raw_test)
        print(f"Final model accuracy: {accuracy:.4f}")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
