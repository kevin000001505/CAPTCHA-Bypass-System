#!/usr/bin/env python3
"""
Quick test to verify the combined dataset and training setup
"""

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical


def test_data_loading():
    """Test loading datasets"""
    print("=== Testing Data Loading ===")

    # Test clean dataset
    clean_dir = "data/captcha_clean"
    clean_labels_path = os.path.join(clean_dir, "labels.json")

    if os.path.exists(clean_labels_path):
        with open(clean_labels_path, "r") as f:
            clean_labels = json.load(f)
        print(f"Clean dataset: {len(clean_labels)} labels")
        print(f"Sample clean labels: {list(clean_labels.items())[:3]}")
    else:
        print("Clean dataset not found!")

    # Test synthetic dataset
    synthetic_dir = "data/synthetic_captcha"
    synthetic_labels_path = os.path.join(synthetic_dir, "labels.json")

    if os.path.exists(synthetic_labels_path):
        with open(synthetic_labels_path, "r") as f:
            synthetic_labels = json.load(f)
        print(f"Synthetic dataset: {len(synthetic_labels)} labels")
        print(f"Sample synthetic labels: {list(synthetic_labels.items())[:3]}")
    else:
        print("Synthetic dataset not found!")


def test_simple_model():
    """Test creating and compiling a simple model"""
    print("\n=== Testing Simple Model ===")

    # Create simple model
    input_tensor = Input(shape=(80, 170, 3))
    x = Conv2D(32, (3, 3), activation="relu")(input_tensor)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)

    # 4 outputs for 4 digits
    outputs = []
    for i in range(4):
        digit_output = Dense(10, activation="softmax", name=f"digit_{i+1}")(x)
        outputs.append(digit_output)

    model = Model(inputs=input_tensor, outputs=outputs)

    # Test compilation with proper metrics
    metrics = {f"digit_{i+1}": ["accuracy"] for i in range(4)}
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=metrics)

    print("✓ Model created and compiled successfully")
    print(f"Model has {model.count_params():,} parameters")

    # Test with dummy data
    dummy_x = np.random.random((10, 80, 170, 3))
    dummy_y = [np.random.randint(0, 2, (10, 10)) for _ in range(4)]

    try:
        model.fit(dummy_x, dummy_y, epochs=1, verbose=0)
        print("✓ Model training test successful")
    except Exception as e:
        print(f"✗ Model training test failed: {e}")


def test_image_loading():
    """Test loading actual images"""
    print("\n=== Testing Image Loading ===")

    # Try to load one image from each dataset
    clean_dir = "data/captcha_clean"
    synthetic_dir = "data/synthetic_captcha"

    for dataset_name, data_dir in [("Clean", clean_dir), ("Synthetic", synthetic_dir)]:
        image_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]

        if image_files:
            test_image = image_files[0]
            img_path = os.path.join(data_dir, test_image)

            try:
                img = Image.open(img_path)
                img = img.resize((170, 80))
                img_array = np.array(img, dtype=np.float32) / 255.0
                print(
                    f"✓ {dataset_name} image loaded: {test_image}, shape: {img_array.shape}"
                )
            except Exception as e:
                print(f"✗ {dataset_name} image loading failed: {e}")
        else:
            print(f"✗ No images found in {dataset_name} dataset")


if __name__ == "__main__":
    test_data_loading()
    test_simple_model()
    test_image_loading()
    print("\n=== All Tests Complete ===")
