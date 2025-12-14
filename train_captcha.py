#!/usr/bin/env python3
"""
Test script for CAPTCHA model training with GPT-4 labeling
"""

import os
from model import CaptchaModel


def main():
    """Test the GPT-4 labeling and training process."""

    # Check if .env file exists
    if not os.path.exists(".env"):
        print("Creating .env file template...")
        with open(".env", "w") as f:
            f.write("# Add your OpenAI API key here\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
        print(
            "Please edit .env file and add your OpenAI API key, then run this script again."
        )
        return

    # Initialize model
    print("Initializing CAPTCHA model...")
    captcha_model = CaptchaModel(img_height=80, img_width=170)

    # Test with a few images first
    data_dir = "data/captcha_debug"

    # Check if we have the OpenAI API key
    if not captcha_model.openai_client:
        print("\nWarning: OpenAI API not available.")
        print(
            "Please check your .env file and make sure OPENAI_API_KEY is set correctly."
        )
        return

    print(f"\nTesting GPT-4 labeling on first few images in {data_dir}")

    # Get first 3 images for testing
    image_files = [
        f for f in os.listdir(data_dir) if f.endswith(".png") and "processed" not in f
    ][:3]

    if not image_files:
        print("No image files found in data directory!")
        return

    print(f"Testing with {len(image_files)} images...")

    # Test GPT-4 labeling on a few images
    test_labels = {}
    for filename in image_files:
        image_path = os.path.join(data_dir, filename)
        print(f"\nTesting {filename}...")

        label = captcha_model.extract_label_with_gpt4(image_path)
        if label:
            test_labels[filename] = label
            print(f"✓ GPT-4 result: {label}")
        else:
            print(f"✗ GPT-4 failed to extract label")

    if test_labels:
        print(f"\n✓ GPT-4 labeling successful! Extracted {len(test_labels)} labels.")
        print("Test labels:", test_labels)

        # Ask user if they want to proceed with full labeling and training
        response = input(
            "\nDo you want to proceed with full dataset labeling and training? (y/n): "
        )

        if response.lower() == "y":
            print("\nStarting full dataset labeling and training...")

            try:
                # Load data (this will create labels for all images)
                x_data, y_data, y_raw = captcha_model.load_data(
                    data_dir, use_gpt4_labeling=True
                )

                print(f"\n✓ Successfully loaded {len(x_data)} labeled images")
                print(f"Sample labels: {y_raw[:5]}")

                # If we have enough data, start training
                if len(x_data) >= 10:  # Minimum 10 images for training
                    print("\nStarting model training...")

                    # Create and train model with smaller parameters for testing
                    captcha_model.create_model()

                    # Use smaller epochs for testing
                    history = captcha_model.train(
                        x_data,
                        y_data,
                        validation_split=0.2,
                        epochs=5,  # Small number for testing
                        batch_size=min(
                            16, len(x_data) // 4
                        ),  # Adjust batch size based on data
                    )

                    print("✓ Training completed!")

                    # Save the model
                    captcha_model.save_model("trained_captcha_model.h5")

                    # Plot training history if we have enough data
                    if len(x_data) > 5:
                        captcha_model.plot_training_history(history)

                else:
                    print(
                        f"Not enough labeled data ({len(x_data)} images). Need at least 10 images for training."
                    )

            except Exception as e:
                print(f"Error during training: {e}")

        else:
            print("Training cancelled.")
    else:
        print(
            "\n✗ GPT-4 labeling failed. Please check your API key and internet connection."
        )


if __name__ == "__main__":
    main()
