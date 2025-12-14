#!/usr/bin/env python3
"""
Quick training test with small dataset to verify everything works
"""

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_small_dataset(max_clean=50, max_synthetic=200):
    """Load a small subset for quick testing"""
    print("Loading small test dataset...")

    # Load clean data (limited)
    clean_dir = "data/captcha_clean"
    clean_labels_path = os.path.join(clean_dir, "labels.json")

    with open(clean_labels_path, "r") as f:
        clean_labels = json.load(f)

    # Load synthetic data (limited)
    synthetic_dir = "data/synthetic_captcha"
    synthetic_labels_path = os.path.join(synthetic_dir, "labels.json")

    with open(synthetic_labels_path, "r") as f:
        synthetic_labels = json.load(f)

    x_data = []
    y_data = []

    # Load clean images
    clean_count = 0
    for filename, label in clean_labels.items():
        if clean_count >= max_clean:
            break

        if len(label) != 4 or not label.isdigit():
            continue

        img_path = os.path.join(clean_dir, filename)
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path)
            # Convert to RGB (important for consistency)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((170, 80))
            img_array = np.array(img, dtype=np.float32) / 255.0

            x_data.append(img_array)
            y_data.append(label)
            clean_count += 1
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

    # Load synthetic images
    synthetic_count = 0
    for filename, label in synthetic_labels.items():
        if synthetic_count >= max_synthetic:
            break

        if len(label) != 4 or not label.isdigit():
            continue

        img_path = os.path.join(synthetic_dir, filename)
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path)
            # Already RGB
            img = img.resize((170, 80))
            img_array = np.array(img, dtype=np.float32) / 255.0

            x_data.append(img_array)
            y_data.append(label)
            synthetic_count += 1
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

    print(
        f"Loaded {clean_count} clean + {synthetic_count} synthetic = {len(x_data)} total images"
    )
    return np.array(x_data), y_data


def encode_labels(labels, num_digits=4, num_classes=10):
    """Convert labels to one-hot encoding"""
    y_encoded = []
    for i in range(num_digits):
        digit_labels = [int(label[i]) for label in labels]
        y_encoded.append(to_categorical(digit_labels, num_classes=num_classes))
    return y_encoded


def create_simple_model():
    """Create a simple CNN model for testing"""
    input_tensor = Input(shape=(80, 170, 3))
    x = input_tensor

    # Simple CNN
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    # 4 outputs for 4 digits
    outputs = []
    for i in range(4):
        digit_output = Dense(10, activation="softmax", name=f"digit_{i+1}")(x)
        outputs.append(digit_output)

    model = Model(inputs=input_tensor, outputs=outputs)

    # Compile with proper metrics
    metrics = {f"digit_{i+1}": ["accuracy"] for i in range(4)}
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=metrics)

    return model


def main():
    """Quick training test"""
    print("=== Quick Training Test ===")

    # Load small dataset
    x_data, y_labels = load_small_dataset(max_clean=50, max_synthetic=200)

    if len(x_data) == 0:
        print("No data loaded!")
        return

    print(f"Data shape: {x_data.shape}")

    # Encode labels
    y_encoded = encode_labels(y_labels)

    # Split data
    x_train, x_test, y_train_raw, y_test_raw = train_test_split(
        x_data, y_labels, test_size=0.2, random_state=42
    )

    # Encode split labels
    y_train_encoded = encode_labels(y_train_raw)
    y_test_encoded = encode_labels(y_test_raw)

    print(f"Training: {len(x_train)}, Testing: {len(x_test)}")

    # Create and train model
    model = create_simple_model()
    print(f"Model has {model.count_params():,} parameters")

    # Train for just a few epochs
    print("Starting training...")
    history = model.fit(
        x_train,
        y_train_encoded,
        validation_data=(x_test, y_test_encoded),
        epochs=10,
        batch_size=16,
        verbose=1,
    )

    # Evaluate
    print("\nEvaluating model...")
    predictions = model.predict(x_test, verbose=0)

    # Convert predictions to strings
    predicted_labels = []
    for i in range(len(x_test)):
        pred_digits = []
        for j in range(4):
            digit = np.argmax(predictions[j][i])
            pred_digits.append(str(digit))
        predicted_labels.append("".join(pred_digits))

    # Calculate accuracy
    correct = sum(1 for pred, true in zip(predicted_labels, y_test_raw) if pred == true)
    accuracy = correct / len(y_test_raw)

    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{len(y_test_raw)})")

    # Show sample predictions
    print("\nSample Predictions:")
    for i in range(min(10, len(x_test))):
        status = "✓" if predicted_labels[i] == y_test_raw[i] else "✗"
        print(f"{status} True: {y_test_raw[i]}, Predicted: {predicted_labels[i]}")

    # Save model
    model.save("quick_test_model.h5")
    print("Model saved as quick_test_model.h5")


if __name__ == "__main__":
    main()
