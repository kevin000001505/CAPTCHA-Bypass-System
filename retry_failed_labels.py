#!/usr/bin/env python3
"""
Script to retry failed CAPTCHA labeling tasks using GPT-4o
"""

from model import CaptchaModel


def main():
    print("=== CAPTCHA Failed Task Retry with GPT-4o ===")

    # Initialize model
    captcha_model = CaptchaModel()

    # Check if OpenAI is available
    if not captcha_model.openai_client:
        print("Error: OpenAI client not available. Please check your OPENAI_API_KEY.")
        return

    # Data directory
    data_dir = "data/captcha_debug"

    # Retry failed labeling tasks
    updated_labels = captcha_model.retry_failed_labels(data_dir, use_gpt4o=True)

    print("\n=== Summary ===")
    print(f"Total labeled images: {len(updated_labels)}")
    print("Retry process completed!")


if __name__ == "__main__":
    main()
