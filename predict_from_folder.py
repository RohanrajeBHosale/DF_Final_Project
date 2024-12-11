import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Path to the trained model
MODEL_PATH = 'model_training/model_training/steganography_detection_model.keras'  # Update this path as needed
model = load_model(MODEL_PATH)  # Load the trained model

# Preprocess the input image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Read the image
    img = cv2.resize(img, (128, 128))  # Resize to model input size
    img = img / 255.0  # Normalize pixel values
    return np.expand_dims(img, axis=0)

# Predict on a single image
def predict_image(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    result = "Steganography detected" if prediction[0][0] >= 0.5 else "No steganography detected"
    confidence = prediction[0][0]
    return result, confidence

# Predict on all images in a folder
def predict_images_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    print(f"Processing images in folder: {folder_path}\n")
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            result, confidence = predict_image(image_path)
            print(f"Image: {filename} - Prediction: {result}, Confidence: {confidence:.2f}")

if __name__ == "__main__":
    # Path to the folder containing images
    folder_path = 'test_image'  # Update this path to your folder

    # Predict on images in the folder
    predict_images_in_folder(folder_path)


    def predict_images_in_folder(folder_path, output_file='predictions.txt'):
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return

        with open(output_file, 'w') as f:
            f.write(f"Processing images in folder: {folder_path}\n\n")
            for filename in os.listdir(folder_path):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, filename)
                    result, confidence = predict_image(image_path)
                    f.write(f"Image: {filename} - Prediction: {result}, Confidence: {confidence:.2f}\n")
                    print(f"Image: {filename} - Prediction: {result}, Confidence: {confidence:.2f}")


    if __name__ == "__main__":
        folder_path = 'test_image'
        predict_images_in_folder(folder_path)