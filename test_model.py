import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Paths to the model and testing folders
MODEL_PATH = 'model_training/model_training/steganography_detection_model.keras'
WITH_STEG_DIR = '/Users/rohanrajebhosale/Documents/DF_Final_Project/data/train/with_steganography'
WITHOUT_STEG_DIR = '/Users/rohanrajebhosale/Documents/DF_Final_Project/data/train/without_steganography'

# Load the trained model
model = load_model(MODEL_PATH)

# Preprocess the input image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize to the input size of the model
    img = img / 255.0  # Normalize pixel values
    return np.expand_dims(img, axis=0)

# Predict steganography presence
def predict_image(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    result = "Steganography detected" if prediction[0][0] >= 0.5 else "No steganography detected"
    confidence = prediction[0][0]
    return result, confidence

# Evaluate all images in a folder
def evaluate_folder(folder_path, label):
    total = 0
    correct = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            total += 1
            image_path = os.path.join(folder_path, filename)
            result, confidence = predict_image(image_path)
            predicted_label = 1 if result == "Steganography detected" else 0
            if predicted_label == label:
                correct += 1
            print(f"Image: {filename} - Prediction: {result}, Confidence: {confidence:.2f}")
    return correct, total

# Main function to test both folders
if __name__ == "__main__":
    print("Evaluating images with steganography...")
    correct_with_steg, total_with_steg = evaluate_folder(WITH_STEG_DIR, label=1)
    print(f"Accuracy for 'with steganography': {correct_with_steg}/{total_with_steg} ({(correct_with_steg / total_with_steg) * 100:.2f}%)\n")

    print("Evaluating images without steganography...")
    correct_without_steg, total_without_steg = evaluate_folder(WITHOUT_STEG_DIR, label=0)
    print(f"Accuracy for 'without steganography': {correct_without_steg}/{total_without_steg} ({(correct_without_steg / total_without_steg) * 100:.2f}%)\n")

    # Overall accuracy
    total_correct = correct_with_steg + correct_without_steg
    total_images = total_with_steg + total_without_steg
    print(f"Overall Accuracy: {total_correct}/{total_images} ({(total_correct / total_images) * 100:.2f}%)")