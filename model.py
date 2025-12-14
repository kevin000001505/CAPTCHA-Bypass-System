#!/usr/bin/env python3
"""
CNN Model for 4-digit CAPTCHA Recognition
This script trains a CNN model to recognize 4-digit numbers from captcha images.
It includes automatic labeling using OCR as a fallback.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import re
import json
import warnings
import base64
from dotenv import load_dotenv

# Try to import OpenAI for GPT-4 labeling
try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    warnings.warn("OpenAI not available. Manual labeling will be required.")

# Try to import pytesseract for automatic labeling
try:
    import pytesseract

    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False
    warnings.warn("pytesseract not available. Manual labeling will be required.")

# Load environment variables
load_dotenv()

# Constants
DEFAULT_LABELS_FILE = "labels.json"

# System prompt for GPT-4 CAPTCHA recognition
SYSTEM_PROMPT = """為了有效地完成破解驗證碼的任務，請進行以下步驟，最終以JSON格式回傳結果。

# 步驟

1. **獲取驗證碼圖像**：你需要獲得要破解的驗證碼影像，確保持有適當訪問及使用權限。
2. **圖像處理**：使用圖像處理技術來消除背景噪音，使驗證碼文本更加清晰。這可能包括灰度化和二值化等技術。
3. **文字識別**：使用光學字符識別（OCR）技術來提取驗證碼中的文字內容。
4. **驗證與錯誤處理**：檢查提取的結果是否合理，記錄和處理任何OCR可能出現的錯誤。
5. **結果輸出**：將最終識別出的驗證碼以結構化的JSON格式返回。

# 提示

- 所有驗證碼都是由4個數字組成。

# Output Format

請返回以下格式的JSON文字：

```json
{
  "captcha_text": "[提取的驗證碼文本]"
}
```

# Notes

- 確保在進行任何破解操作時遵循法律規範和服務條款。
- 精確度會受到圖像質量及複雜度影響。
- 如無法破解，請提供適當的錯誤消息。"""


class CaptchaModel:
    def __init__(self, img_height=80, img_width=170, num_digits=4, num_classes=10):
        """
        Initialize the CAPTCHA model.

        Args:
            img_height (int): Height of input images
            img_width (int): Width of input images
            num_digits (int): Number of digits in captcha (default: 4)
            num_classes (int): Number of possible digit classes (0-9, so 10)
        """
        self.img_height = img_height
        self.img_width = img_width
        self.num_digits = num_digits
        self.num_classes = num_classes
        self.model = None

        # Initialize OpenAI client if available
        if HAS_OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
            else:
                self.openai_client = None
                warnings.warn("OPENAI_API_KEY not found in environment variables.")
        else:
            self.openai_client = None

    def encode_image_to_base64(self, image_path):
        """
        Encode image to base64 string.

        Args:
            image_path (str): Path to image file

        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string

    def extract_label_with_gpt4(self, image_path):
        """
        Extract label from image using GPT-4 Vision API.

        Args:
            image_path (str): Path to image file

        Returns:
            str or None: Extracted 4-digit label or None if failed
        """
        if not self.openai_client:
            return None

        try:
            base64_image = self.encode_image_to_base64(image_path)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Using gpt-4o-mini as it's more cost-effective for this task
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            }
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=100,
            )

            results = json.loads(response.choices[0].message.content)
            image_result = results.get("captcha_text", "").strip()

            # Clean the result - only keep ASCII digits 0-9
            cleaned_result = "".join(c for c in image_result if c in "0123456789")

            # Validate the result is exactly 4 digits
            if cleaned_result and len(cleaned_result) == 4:
                return cleaned_result
            else:
                print(
                    f"Invalid GPT-4 result for {image_path}: '{image_result}' (cleaned: '{cleaned_result}')"
                )
                return None

        except Exception as e:
            print(f"GPT-4 error for {image_path}: {e}")
            return None

    def preprocess_image_for_ocr(self, img):
        """
        Preprocess image for better OCR recognition.

        Args:
            img (PIL.Image): Input image

        Returns:
            PIL.Image: Processed image
        """
        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Scale up for better OCR
        w, h = img.size
        img = img.resize((w * 6, h * 6), Image.LANCZOS)

        # Enhance contrast and sharpness
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = ImageEnhance.Sharpness(img).enhance(1.5)

        # Convert to grayscale
        img = img.convert("L")

        # Apply binary threshold
        img = img.point(lambda x: 255 if x > 140 else 0, mode="1")

        # Apply noise filter
        img = img.filter(ImageFilter.MedianFilter(size=3))

        return img

    def extract_label_with_ocr(self, image_path):
        """
        Extract label from image using OCR.

        Args:
            image_path (str): Path to image file

        Returns:
            str or None: Extracted 4-digit label or None if failed
        """
        if not HAS_PYTESSERACT:
            return None

        try:
            img = Image.open(image_path)
            processed_img = self.preprocess_image_for_ocr(img)

            # OCR configurations to try
            ocr_configs = [
                r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789",
                r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789",
                r"--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789",
            ]

            for config in ocr_configs:
                try:
                    text = pytesseract.image_to_string(processed_img, config=config)
                    cleaned = "".join(c for c in text if c.isdigit())

                    if len(cleaned) == 4:
                        return cleaned
                except Exception:
                    continue

            return None

        except Exception as e:
            print(f"OCR error for {image_path}: {e}")
            return None

    def create_labels_file(
        self, data_dir, labels_file=DEFAULT_LABELS_FILE, use_gpt4=True
    ):
        """
        Create a labels file by extracting labels from images using GPT-4 or OCR.

        Args:
            data_dir (str): Directory containing images
            labels_file (str): Output labels file name
            use_gpt4 (bool): Whether to use GPT-4 for labeling (recommended)
        """
        labels = {}
        image_files = [
            f
            for f in os.listdir(data_dir)
            if f.endswith(".png") and "processed" not in f
        ]

        print(f"Extracting labels from {len(image_files)} images...")

        if use_gpt4 and self.openai_client:
            print("Using GPT-4 Vision API for labeling...")
            labeling_method = "GPT-4"
        elif HAS_PYTESSERACT:
            print("Using OCR (tesseract) for labeling...")
            labeling_method = "OCR"
        else:
            print("No labeling method available. Please install OpenAI or pytesseract.")
            return {}

        successful = 0
        failed_files = []

        for i, filename in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {filename}")
            image_path = os.path.join(data_dir, filename)

            # Try GPT-4 first if available
            label = None
            if use_gpt4 and self.openai_client:
                label = self.extract_label_with_gpt4(image_path)

            # Fallback to OCR if GPT-4 fails or unavailable
            if not label and HAS_PYTESSERACT:
                label = self.extract_label_with_ocr(image_path)

            if label:
                labels[filename] = label
                successful += 1
                print(f"✓ {filename}: {label}")
            else:
                failed_files.append(filename)
                print(f"✗ {filename}: Failed to extract label")

        # Save labels to file
        labels_path = os.path.join(data_dir, labels_file)
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2)

        print(f"\nLabeling complete using {labeling_method}:")
        print(f"✓ Success: {successful}/{len(image_files)} images labeled")
        print(f"✗ Failed: {len(failed_files)} images")

        if failed_files:
            print("Failed files:", failed_files[:5])  # Show first 5 failed files
            if len(failed_files) > 5:
                print(f"... and {len(failed_files) - 5} more")

        print(f"Labels saved to: {labels_path}")

        return labels

    def load_manual_labels(self, data_dir, labels_file=DEFAULT_LABELS_FILE):
        """
        Load labels from a JSON file.

        Args:
            data_dir (str): Directory containing the labels file
            labels_file (str): Labels file name

        Returns:
            dict: Dictionary mapping filenames to labels
        """
        labels_path = os.path.join(data_dir, labels_file)

        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                return json.load(f)
        else:
            return {}

    def load_data(
        self, data_dir, use_gpt4_labeling=True, labels_file=DEFAULT_LABELS_FILE
    ):
        """
        Load and preprocess captcha images from directory.

        Args:
            data_dir (str): Path to directory containing captcha images
            use_gpt4_labeling (bool): Whether to use GPT-4 for automatic labeling
            labels_file (str): Name of labels file

        Returns:
            tuple: (X, y, labels) where X is image data, y is encoded labels, labels is raw labels
        """
        print("Loading data from:", data_dir)

        # Try to load existing labels
        labels_dict = self.load_manual_labels(data_dir, labels_file)

        # If no labels exist, create them using available methods
        if not labels_dict:
            print("No labels file found. Creating labels automatically...")
            if use_gpt4_labeling and self.openai_client:
                labels_dict = self.create_labels_file(
                    data_dir, labels_file, use_gpt4=True
                )
            elif HAS_PYTESSERACT:
                labels_dict = self.create_labels_file(
                    data_dir, labels_file, use_gpt4=False
                )
            else:
                print("No labeling method available.")
                print("Please either:")
                print("1. Set OPENAI_API_KEY environment variable for GPT-4 labeling")
                print("2. Install pytesseract for OCR labeling")
                print("3. Create a labels.json file manually")
                print("Labels file format:")
                print('{"image_filename.png": "1234", "another_image.png": "5678"}')
                raise ValueError("No labels available for training")

        if not labels_dict:
            raise ValueError("Failed to create labels. Please check your setup.")

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
            raise ValueError(
                "No valid images found! Please check your data directory and labels."
            )

        # Convert to numpy arrays
        x_data = np.array(x_data)

        # Convert labels to one-hot encoding for each digit position
        y_encoded = []
        for i in range(self.num_digits):
            digit_labels = [int(label[i]) for label in y_data]
            y_encoded.append(to_categorical(digit_labels, num_classes=self.num_classes))

        return x_data, y_encoded, y_data

    def create_model(self):
        """
        Create the CNN model architecture.

        Returns:
            keras.Model: Compiled model
        """
        print("Creating model architecture...")

        # Input layer
        input_tensor = Input(shape=(self.img_height, self.img_width, 3))
        x = input_tensor

        # CNN feature extraction layers
        # 4 blocks of conv-conv-maxpool (similar to VGG style)
        for i in range(4):
            filters = 32 * (2**i)  # 32, 64, 128, 256
            x = Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
            x = Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
            x = MaxPooling2D((2, 2))(x)

        # Flatten and add dropout
        x = Flatten()(x)
        x = Dropout(0.25)(x)

        # Multi-output: 4 separate dense layers for each digit position
        outputs = []
        for i in range(self.num_digits):
            digit_output = Dense(
                self.num_classes, activation="softmax", name=f"digit_{i+1}"
            )(x)
            outputs.append(digit_output)

        # Create model
        model = Model(inputs=input_tensor, outputs=outputs)

        # Compile model
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"] * self.num_digits,  # One accuracy metric per digit
        )

        self.model = model
        return model

    def train(self, x_data, y_data, validation_split=0.2, epochs=20, batch_size=32):
        """
        Train the model.

        Args:
            x_data (np.array): Input images
            y_data (list): List of one-hot encoded labels for each digit position
            validation_split (float): Fraction of data to use for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training

        Returns:
            History: Training history
        """
        print("Starting training...")

        if self.model is None:
            self.create_model()

        # Print model summary
        self.model.summary()

        # Set up callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                "best_captcha_model.h5",
                save_best_only=True,
                monitor="val_loss",
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=1
            ),
        ]

        # Train the model
        history = self.model.fit(
            x_data,
            y_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )

        return history

    def plot_training_history(self, history):
        """
        Plot training history.

        Args:
            history: Training history from model.fit()
        """
        _fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot overall loss
        axes[0, 0].plot(history.history["loss"], label="Training Loss")
        axes[0, 0].plot(history.history["val_loss"], label="Validation Loss")
        axes[0, 0].set_title("Model Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()

        # Plot accuracy for each digit
        for i in range(self.num_digits):
            row = (i + 1) // 3
            col = (i + 1) % 3

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

        plt.tight_layout()
        plt.savefig("training_history.png", dpi=300, bbox_inches="tight")
        plt.show()

    def evaluate_model(self, x_test, y_test, y_test_raw):
        """
        Evaluate model performance.

        Args:
            x_test (np.array): Test images
            y_test (list): One-hot encoded test labels
            y_test_raw (list): Raw string labels for test data
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
            print(f"True: {y_test_raw[i]}, Predicted: {predicted_labels[i]}")

        return accuracy, predicted_labels

    def predict_single(self, image_path):
        """
        Predict a single image.

        Args:
            image_path (str): Path to image file

        Returns:
            str: Predicted 4-digit string
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
        for i in range(self.num_digits):
            digit = np.argmax(predictions[i][0])
            pred_digits.append(str(digit))

        result = "".join(pred_digits)

        # Display image with prediction
        plt.figure(figsize=(8, 4))
        plt.imshow(img)
        plt.title(f"Predicted: {result}")
        plt.axis("off")
        plt.show()

        return result

    def save_model(self, filepath="captcha_model.h5"):
        """Save the trained model."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save!")

    def load_model(self, filepath="captcha_model.h5"):
        """Load a pre-trained model."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

    def retry_failed_labels(
        self, data_dir, labels_file=DEFAULT_LABELS_FILE, use_gpt4o=True
    ):
        """
        Retry labeling for images that previously failed using GPT-4o.

        Args:
            data_dir (str): Directory containing images
            labels_file (str): Labels file name
            use_gpt4o (bool): Whether to use GPT-4o (more powerful model)

        Returns:
            dict: Updated labels dictionary
        """
        print("Checking for failed labeling tasks...")

        # Load existing labels
        labels_path = os.path.join(data_dir, labels_file)
        labels = {}
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                labels = json.load(f)

        # Get all captcha canvas images
        all_images = [
            f
            for f in os.listdir(data_dir)
            if f.startswith("captcha_canvas_") and f.endswith(".png")
        ]

        # Find unlabeled images (failed tasks)
        failed_images = [img for img in all_images if img not in labels]

        print(f"Total images: {len(all_images)}")
        print(f"Already labeled: {len(labels)}")
        print(f"Failed/unlabeled images: {len(failed_images)}")

        if not failed_images:
            print("No failed tasks found. All images are already labeled!")
            return labels

        if not self.openai_client:
            print("OpenAI client not available. Cannot retry failed tasks.")
            return labels

        print(f"\nRetrying {len(failed_images)} failed images with GPT-4o...")

        successful_retries = 0
        still_failed = []

        for i, filename in enumerate(failed_images):
            print(f"Retrying {i+1}/{len(failed_images)}: {filename}")
            image_path = os.path.join(data_dir, filename)

            # Use GPT-4o for retry
            label = self.extract_label_with_gpt4o(image_path)

            if label:
                labels[filename] = label
                successful_retries += 1
                print(f"✓ {filename}: {label}")
            else:
                still_failed.append(filename)
                print(f"✗ {filename}: Still failed")

        # Save updated labels
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2)

        print("\nRetry complete:")
        print(f"✓ Successful retries: {successful_retries}/{len(failed_images)}")
        print(f"✗ Still failed: {len(still_failed)}")

        if still_failed:
            print("Still failed files:", still_failed[:5])
            if len(still_failed) > 5:
                print(f"... and {len(still_failed) - 5} more")

        print(f"Updated labels saved to: {labels_path}")
        print(f"Total labeled images now: {len(labels)}")

        return labels

    def extract_label_with_gpt4o(self, image_path):
        """
        Extract label from image using GPT-4o (more powerful model).

        Args:
            image_path (str): Path to image file

        Returns:
            str or None: Extracted 4-digit label or None if failed
        """
        if not self.openai_client:
            return None

        try:
            base64_image = self.encode_image_to_base64(image_path)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Using full GPT-4o model for better accuracy
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            }
                        ],
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=150,
                temperature=0.1,  # Lower temperature for more consistent results
            )

            results = json.loads(response.choices[0].message.content)
            image_result = results.get("captcha_text", "").strip()

            # Clean the result - only keep ASCII digits 0-9
            cleaned_result = "".join(c for c in image_result if c in "0123456789")

            # Validate the result is exactly 4 digits
            if cleaned_result and len(cleaned_result) == 4:
                return cleaned_result
            else:
                print(
                    f"Invalid GPT-4o result for {image_path}: '{image_result}' (cleaned: '{cleaned_result}')"
                )
                return None

        except Exception as e:
            print(f"GPT-4o error for {image_path}: {e}")
            return None


def main():
    """Main training function."""
    # Initialize model
    captcha_model = CaptchaModel(img_height=80, img_width=170)

    # Load data
    data_dir = "data/captcha_debug"
    x_data, y_data, y_raw = captcha_model.load_data(data_dir)

    print(f"Data shape: {x_data.shape}")
    print(f"Number of samples: {len(x_data)}")

    # Split data into train and test sets
    x_train, x_test, _, y_raw_test = train_test_split(
        x_data, y_raw, test_size=0.2, random_state=42
    )

    # Split labels accordingly
    y_train = []
    y_test = []
    for i in range(captcha_model.num_digits):
        y_digit_train, y_digit_test, _, _ = train_test_split(
            y_data[i], y_raw, test_size=0.2, random_state=42
        )
        y_train.append(y_digit_train)
        y_test.append(y_digit_test)

    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")

    # Create and train model
    captcha_model.create_model()
    history = captcha_model.train(
        x_train, y_train, validation_split=0.2, epochs=25, batch_size=32
    )

    # Plot training history
    captcha_model.plot_training_history(history)

    # Evaluate model
    captcha_model.evaluate_model(x_test, y_test, y_raw_test)

    # Save model
    captcha_model.save_model("trained_captcha_model.h5")

    # Test prediction on a single image
    if len(os.listdir(data_dir)) > 0:
        test_image = os.path.join(data_dir, os.listdir(data_dir)[0])
        print(f"\nTesting single prediction on: {test_image}")
        captcha_model.predict_single(test_image)


if __name__ == "__main__":
    main()
