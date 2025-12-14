#!/usr/bin/env python3
"""
Simple CNN Model for 4-digit CAPTCHA Recognition
Optimized for small datasets (~100-200 images)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import json


class SimpleCaptchaModel:
    def __init__(self, img_height=80, img_width=170, num_digits=4, num_classes=10):
        """
        Initialize the simple CAPTCHA model.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.num_digits = num_digits
        self.num_classes = num_classes
        self.model = None

    def load_data(self, data_dir, labels_file="labels.json"):
        """
        Load and preprocess captcha images from directory.
        """
        print("Loading data from:", data_dir)

        # Load labels
        labels_path = os.path.join(data_dir, labels_file)
        with open(labels_path, "r") as f:
            labels_dict = json.load(f)

        x_data = []
        y_data = []

        print("Loading images with labels...")
        loaded_count = 0

        for filename, label in labels_dict.items():
            if len(label) != 4 or not label.isdigit():
                print(f"Skipping {filename} - invalid label: {label}")
                continue

            img_path = os.path.join(data_dir, filename)
            if not os.path.exists(img_path):
                print(f"Skipping {filename} - file not found")
                continue

            try:
                img = Image.open(img_path)

                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize to target dimensions
                img = img.resize((self.img_width, self.img_height))

                # Convert to numpy array and normalize
                img_array = np.array(img, dtype=np.float32) / 255.0

                x_data.append(img_array)
                y_data.append(label)
                loaded_count += 1

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

        print(f"Successfully loaded {loaded_count} images with labels")

        if len(x_data) == 0:
            raise ValueError("No valid images found!")

        # Convert to numpy arrays
        x_data = np.array(x_data)

        # Convert labels to one-hot encoding for each digit position
        y_encoded = []
        for i in range(self.num_digits):
            digit_labels = [int(label[i]) for label in y_data]
            y_encoded.append(to_categorical(digit_labels, num_classes=self.num_classes))

        return x_data, y_encoded, y_data

    def create_simple_model(self):
        """
        Create a simpler CNN model suitable for small datasets.
        """
        print("Creating simple model architecture...")

        # Input layer
        input_tensor = Input(shape=(self.img_height, self.img_width, 3))
        x = input_tensor

        # Much simpler architecture for small dataset
        # Block 1
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        # Block 2
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        # Block 3
        x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)

        # Flatten and add dense layers
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        # Multi-output: 4 separate dense layers for each digit position
        outputs = []
        for i in range(self.num_digits):
            digit_output = Dense(
                self.num_classes, activation="softmax", name=f"digit_{i+1}"
            )(x)
            outputs.append(digit_output)

        # Create model
        model = Model(inputs=input_tensor, outputs=outputs)

        # Compile model with appropriate metrics for multi-output
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"] * self.num_digits,
        )

        self.model = model
        return model

    def train(self, x_data, y_data, validation_split=0.2, epochs=50, batch_size=8):
        """
        Train the model with settings optimized for small datasets.
        """
        print("Starting training with small dataset optimizations...")

        if self.model is None:
            self.create_simple_model()

        # Print model summary
        self.model.summary()

        # Set up callbacks optimized for small datasets
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                "best_simple_captcha_model.h5",
                save_best_only=True,
                monitor="val_loss",
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,  # More patience for small datasets
                verbose=1,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,  # More aggressive reduction
                patience=5,
                verbose=1,
                min_lr=1e-7,
            ),
        ]

        # Train the model
        history = self.model.fit(
            x_data,
            y_data,
            batch_size=batch_size,  # Small batch size for small dataset
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )

        return history

    def evaluate_model(self, x_test, y_test_raw):
        """
        Evaluate model performance.
        """
        print("Evaluating model...")

        if self.model is None:
            print("No model found! Please train the model first.")
            return

        # Get predictions
        predictions = self.model.predict(x_test)

        # Convert predictions to digit strings
        predicted_labels = []
        for i in range(len(x_test)):
            pred_digits = []
            for j in range(self.num_digits):
                digit = np.argmax(predictions[j][i])
                pred_digits.append(str(digit))
            predicted_labels.append("".join(pred_digits))

        # Calculate accuracy
        correct = sum(
            1 for pred, true in zip(predicted_labels, y_test_raw) if pred == true
        )
        accuracy = correct / len(y_test_raw)

        print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{len(y_test_raw)})")

        # Show some predictions
        print("\nSample Predictions:")
        for i in range(min(10, len(x_test))):
            status = "✓" if predicted_labels[i] == y_test_raw[i] else "✗"
            print(f"{status} True: {y_test_raw[i]}, Predicted: {predicted_labels[i]}")

        # Calculate per-digit accuracy
        print("\nPer-digit accuracy:")
        for digit_pos in range(self.num_digits):
            correct_digits = sum(
                1
                for pred, true in zip(predicted_labels, y_test_raw)
                if pred[digit_pos] == true[digit_pos]
            )
            digit_accuracy = correct_digits / len(y_test_raw)
            print(f"Digit {digit_pos + 1}: {digit_accuracy:.4f}")

        return accuracy, predicted_labels

    def save_model(self, filepath="simple_captcha_model.h5"):
        """Save the trained model."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save!")

    def predict_single(self, image_path):
        """
        Predict a single image.
        """
        if self.model is None:
            print("No model found! Please train the model first.")
            return None

        # Load and preprocess image
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize((self.img_width, self.img_height))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Get prediction
        predictions = self.model.predict(img_array)

        # Convert to digit string
        pred_digits = []
        confidence_scores = []
        for i in range(self.num_digits):
            probs = predictions[i][0]
            digit = np.argmax(probs)
            confidence = probs[digit]
            pred_digits.append(str(digit))
            confidence_scores.append(confidence)

        result = "".join(pred_digits)
        avg_confidence = np.mean(confidence_scores)

        print(f"Predicted: {result} (confidence: {avg_confidence:.3f})")

        return result


def main():
    """Main training function with optimizations for small datasets."""
    # Initialize simple model
    captcha_model = SimpleCaptchaModel(img_height=80, img_width=170)

    # Load data
    data_dir = "/Users/kevinhsu/Downloads/chicken/data/captcha_clean"
    x_data, y_data, y_raw = captcha_model.load_data(data_dir)

    print(f"Data shape: {x_data.shape}")
    print(f"Number of samples: {len(x_data)}")

    # Split data with stratification to ensure balanced splits
    x_train, x_test, _, y_raw_test = train_test_split(
        x_data, y_raw, test_size=0.15, random_state=42  # Smaller test split
    )

    # Split labels accordingly
    y_train = []
    y_test = []
    for i in range(captcha_model.num_digits):
        y_digit_train, y_digit_test, _, _ = train_test_split(
            y_data[i], y_raw, test_size=0.15, random_state=42
        )
        y_train.append(y_digit_train)
        y_test.append(y_digit_test)

    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")

    # Create and train simple model
    captcha_model.create_simple_model()
    _ = captcha_model.train(
        x_train,
        y_train,
        validation_split=0.15,  # Smaller validation split
        epochs=100,  # More epochs with early stopping
        batch_size=8,  # Small batch size
    )

    # Evaluate model
    captcha_model.evaluate_model(x_test, y_raw_test)

    # Save model
    captcha_model.save_model("simple_captcha_model.h5")

    print("Training completed!")


if __name__ == "__main__":
    main()
