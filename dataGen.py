import os
import cv2
import numpy as np
from PIL import Image
from random import randint, choice
from string import ascii_letters

# Paths for the dataset
original_images_path = "./data/without_steganography"
with_steg_path = "./data/with_steganography"

# Ensure the with_steganography folder exists
os.makedirs(with_steg_path, exist_ok=True)

# Hide a random message in the image using LSB
def hide_message_lsb(image, message):
    # Convert the message to binary
    message_binary = ''.join(format(ord(char), '08b') for char in message)
    message_index = 0
    max_len = len(message_binary)

    # Modify the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if message_index < max_len:
                # Extract the pixel value
                pixel = image[i, j].astype(int)  # Convert to int for safe operations

                # Modify the least significant bit of the first channel
                pixel[0] = (pixel[0] & ~1) | int(message_binary[message_index])

                # Ensure the value is within the range [0, 255]
                pixel[0] = max(0, min(255, pixel[0]))

                # Update the image with the modified pixel
                image[i, j] = pixel.astype(np.uint8)

                # Move to the next bit in the message
                message_index += 1

    return image

# Encode images with steganographic content
def encode_images_with_steganography():
    for filename in os.listdir(original_images_path):
        original_image_path = os.path.join(original_images_path, filename)
        if os.path.isfile(original_image_path):
            # Read the image
            image = cv2.imread(original_image_path)
            if image is None:
                print(f"Failed to read image: {filename}")
                continue

            # Resize the image if necessary
            image = cv2.resize(image, (256, 256))

            # Generate a random hidden message
            random_message = ''.join(choice(ascii_letters) for _ in range(randint(5, 20)))

            # Encode the image with the hidden message
            steg_image = hide_message_lsb(image.copy(), random_message)

            # Save the encoded image to the "with_steganography" folder
            steg_image_path = os.path.join(with_steg_path, filename)
            Image.fromarray(steg_image).save(steg_image_path)
            print(f"Encoded and saved: {steg_image_path}")

# Run the encoding
encode_images_with_steganography()
print("Steganographic images created successfully!")