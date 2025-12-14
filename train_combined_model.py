#!/usr/bin/env python3
"""
Train CNN model with combined dataset (original clean data + synthetic data)
This will significantly improve the model performance by having more training data
"""

import os
import sys
import json
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
from sklearn.metrics import classification_report, accuracy_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class CombinedCaptchaModel:
    def __init__(self, img_height=80, img_width=170, num_digits=4, num_classes=10):
        """
        Initialize the CAPTCHA model for combined dataset training.
        """
        self.img_height = img_height
        self.img_width = img_width
        self.num_digits = num_digits
        self.num_classes = num_classes
        self.model = None

    def load_dataset(self, data_dir, labels_file="labels.json"):
        """
        Load dataset from a directory with labels.json file.

        Args:
            data_dir (str): Directory containing images and labels.json
            labels_file (str): Name of the labels file

        Returns:
            tuple: (images, labels) - numpy arrays
        """
        print(f"Loading dataset from: {data_dir}")

        # Load labels
        labels_path = os.path.join(data_dir, labels_file)
        if not os.path.exists(labels_path):
            print(f"Warning: {labels_path} not found!")
            return np.array([]), []

        with open(labels_path, "r") as f:
            labels_dict = json.load(f)

        x_data = []
        y_data = []
        loaded_count = 0

        for filename, label in labels_dict.items():
            # Skip invalid labels
            if len(label) != 4 or not label.isdigit():
                continue

            img_path = os.path.join(data_dir, filename)
            if not os.path.exists(img_path):
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

        print(f"Loaded {loaded_count} images from {data_dir}")
        return np.array(x_data), y_data

    def combine_datasets(self, datasets):
        """
        Combine multiple datasets.

        Args:
            datasets (list): List of (x_data, y_data) tuples

        Returns:
            tuple: Combined (x_data, y_data)
        """
        combined_x = []
        combined_y = []

        for x_data, y_data in datasets:
            if len(x_data) > 0:
                combined_x.append(x_data)
                combined_y.extend(y_data)

        if combined_x:
            combined_x = np.vstack(combined_x)
        else:
            combined_x = np.array([])

        return combined_x, combined_y

    def encode_labels(self, labels):
        """
        Convert string labels to one-hot encoding for each digit position.

        Args:
            labels (list): List of 4-digit strings

        Returns:
            list: List of one-hot encoded arrays for each digit position
        """
        y_encoded = []
        for i in range(self.num_digits):
            digit_labels = [int(label[i]) for label in labels]
            y_encoded.append(to_categorical(digit_labels, num_classes=self.num_classes))

        return y_encoded

    def create_improved_model(self):
        """
        Create an improved CNN model architecture suitable for larger datasets.
        """
        print("Creating improved CNN model...")

        # Input layer
        input_tensor = Input(shape=(self.img_height, self.img_width, 3))
        x = input_tensor

        # First block - smaller filters for initial feature detection
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.1)(x)

        # Second block
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.1)(x)

        # Third block
        x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.2)(x)

        # Fourth block
        x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        # Global average pooling instead of flatten to reduce parameters
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Dense layer with dropout
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

        # Compile model with individual metrics for each digit
        metrics = {f"digit_{i+1}": ["accuracy"] for i in range(self.num_digits)}
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=metrics,
        )

        self.model = model
        return model

    def train_model(self, x_train, y_train, x_val, y_val, epochs=50, batch_size=32):
        """
        Train the model with callbacks.
        """
        print("Starting training...")

        if self.model is None:
            self.create_improved_model()

        # Print model summary
        self.model.summary()

        # Set up callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                "best_combined_captcha_model.h5",
                save_best_only=True,
                monitor="val_loss",
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, verbose=1, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, verbose=1, min_lr=1e-7
            ),
        ]

        # Train the model
        history = self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )

        return history

    def evaluate_model(self, x_test, y_test_raw):
        """
        Evaluate model performance with detailed metrics.
        """
        print("Evaluating model...")

        if self.model is None:
            print("No model found!")
            return

        # Get predictions
        predictions = self.model.predict(x_test, verbose=0)

        # Convert predictions to digit strings
        predicted_labels = []
        for i in range(len(x_test)):
            pred_digits = []
            for j in range(self.num_digits):
                digit = np.argmax(predictions[j][i])
                pred_digits.append(str(digit))
            predicted_labels.append("".join(pred_digits))

        # Calculate overall accuracy (all digits correct)
        correct = sum(
            1 for pred, true in zip(predicted_labels, y_test_raw) if pred == true
        )
        overall_accuracy = correct / len(y_test_raw)

        # Calculate per-digit accuracy
        digit_accuracies = []
        for i in range(self.num_digits):
            digit_preds = [pred[i] for pred in predicted_labels]
            digit_true = [true[i] for true in y_test_raw]
            digit_acc = accuracy_score(digit_true, digit_preds)
            digit_accuracies.append(digit_acc)

        print(f"\n=== Model Evaluation Results ===")
        print(
            f"Overall Accuracy (all 4 digits correct): {overall_accuracy:.4f} ({correct}/{len(y_test_raw)})"
        )
        print(f"Per-digit accuracies:")
        for i, acc in enumerate(digit_accuracies):
            print(f"  Digit {i+1}: {acc:.4f}")

        # Show some predictions
        print("\nSample Predictions:")
        for i in range(min(10, len(x_test))):
            status = "✓" if predicted_labels[i] == y_test_raw[i] else "✗"
            print(f"{status} True: {y_test_raw[i]}, Predicted: {predicted_labels[i]}")

        return overall_accuracy, predicted_labels

    def plot_training_history(self, history):
        """
        Plot comprehensive training history.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Overall loss
        axes[0, 0].plot(history.history["loss"], label="Training Loss")
        axes[0, 0].plot(history.history["val_loss"], label="Validation Loss")
        axes[0, 0].set_title("Model Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Learning rate
        if "lr" in history.history:
            axes[0, 1].plot(history.history["lr"])
            axes[0, 1].set_title("Learning Rate")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Learning Rate")
            axes[0, 1].set_yscale("log")
            axes[0, 1].grid(True)

        # Average accuracy across all digits
        train_accs = []
        val_accs = []
        for i in range(self.num_digits):
            digit_acc = f"digit_{i+1}_accuracy"
            val_digit_acc = f"val_digit_{i+1}_accuracy"
            if digit_acc in history.history:
                train_accs.append(history.history[digit_acc])
                val_accs.append(history.history[val_digit_acc])

        if train_accs:
            avg_train_acc = np.mean(train_accs, axis=0)
            avg_val_acc = np.mean(val_accs, axis=0)
            axes[0, 2].plot(avg_train_acc, label="Average Training Accuracy")
            axes[0, 2].plot(avg_val_acc, label="Average Validation Accuracy")
            axes[0, 2].set_title("Average Accuracy Across All Digits")
            axes[0, 2].set_xlabel("Epoch")
            axes[0, 2].set_ylabel("Accuracy")
            axes[0, 2].legend()
            axes[0, 2].grid(True)

        # Individual digit accuracies
        for i in range(min(3, self.num_digits)):
            row = 1
            col = i
            digit_acc = f"digit_{i+1}_accuracy"
            val_digit_acc = f"val_digit_{i+1}_accuracy"

            if digit_acc in history.history:
                axes[row, col].plot(
                    history.history[digit_acc], label=f"Training Digit {i+1}"
                )
                axes[row, col].plot(
                    history.history[val_digit_acc], label=f"Validation Digit {i+1}"
                )
                axes[row, col].set_title(f"Digit {i+1} Accuracy")
                axes[row, col].set_xlabel("Epoch")
                axes[row, col].set_ylabel("Accuracy")
                axes[row, col].legend()
                axes[row, col].grid(True)

        plt.tight_layout()
        plt.savefig("combined_training_history.png", dpi=300, bbox_inches="tight")
        plt.show()

    def save_model(self, filepath="combined_captcha_model.h5"):
        """Save the trained model."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save!")


def main():
    """Main function to train with combined datasets."""

    # Initialize model
    model = CombinedCaptchaModel(img_height=80, img_width=170)

    # Define dataset paths
    clean_data_dir = "data/captcha_clean"
    synthetic_data_dir = "data/synthetic_captcha"

    print("=== Loading datasets ===")

    # Load original clean dataset
    clean_x, clean_y = model.load_dataset(clean_data_dir, "labels.json")

    # Load synthetic dataset
    synthetic_x, synthetic_y = model.load_dataset(synthetic_data_dir, "labels.json")

    # Combine datasets
    datasets = [(clean_x, clean_y), (synthetic_x, synthetic_y)]
    combined_x, combined_y = model.combine_datasets(datasets)

    print(f"\n=== Dataset Summary ===")
    print(f"Clean dataset: {len(clean_y)} images")
    print(f"Synthetic dataset: {len(synthetic_y)} images")
    print(f"Combined dataset: {len(combined_y)} images")
    print(f"Combined data shape: {combined_x.shape}")

    if len(combined_y) == 0:
        print("No data loaded! Please check your dataset paths.")
        return

    # Encode labels
    y_encoded = model.encode_labels(combined_y)

    # Split into train, validation, and test sets
    x_train, x_temp, y_train_raw, y_temp_raw = train_test_split(
        combined_x, combined_y, test_size=0.3, random_state=42, stratify=None
    )

    x_val, x_test, y_val_raw, y_test_raw = train_test_split(
        x_temp, y_temp_raw, test_size=0.5, random_state=42, stratify=None
    )

    # Encode the split labels
    y_train_encoded = model.encode_labels(y_train_raw)
    y_val_encoded = model.encode_labels(y_val_raw)

    print("\n=== Data Split ===")
    print(f"Training: {len(x_train)} images")
    print(f"Validation: {len(x_val)} images")
    print(f"Testing: {len(x_test)} images")

    # Train the model
    print("\n=== Training Model ===")
    history = model.train_model(
        x_train, y_train_encoded, x_val, y_val_encoded, epochs=50, batch_size=32
    )

    # Plot training history
    model.plot_training_history(history)

    # Evaluate on test set
    print("\n=== Final Evaluation ===")
    model.evaluate_model(x_test, y_test_raw)

    # Save the model
    model.save_model("final_combined_captcha_model.h5")

    print("\n=== Training Complete! ===")
    print("Model saved as: final_combined_captcha_model.h5")


if __name__ == "__main__":
    main()
