

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os

# --- Settings ---
latent_dim = 2
num_classes = 10
model_dir = "models"  # adjust if needed
decoder_path = os.path.join(model_dir, "decoder.h5")  # or .h5 if you're using that format

# --- Load trained decoder ---
decoder = keras.models.load_model(decoder_path, compile=False)

def generate_digit_image(digit, latent=None):
    """Generate one image for a single digit using random latent vector"""
    if latent is None:
        latent = np.random.normal(size=(1, latent_dim))
    label = keras.utils.to_categorical([int(digit)], num_classes=num_classes)
    image = decoder.predict([latent, label], verbose=0)[0]
    return image.squeeze()

def generate_number_image(number_str):
    """Generate and arrange digits like '2345' into one image"""
    images = [generate_digit_image(d) for d in number_str]
    stacked = np.hstack(images)  # concatenate along width
    # return stacked
    return enhance_image(stacked)

from PIL import Image, ImageFilter, ImageEnhance

def enhance_image(image_array, upscale=2, contrast_factor=1.5, sharpen=True):
    """Upscale + contrast + optional sharpening using PIL"""
    # Convert to PIL Image
    img = Image.fromarray((image_array * 255).astype(np.uint8))

    # Upscale
    img = img.resize((img.width * upscale, img.height * upscale), Image.LANCZOS)

    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)

    # Optional sharpen
    if sharpen:
        img = img.filter(ImageFilter.SHARPEN)

    return np.array(img) / 255.0


def show_number(number_str):
    img = generate_number_image(number_str)
    plt.figure(figsize=(len(number_str), 2))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"Generated: {number_str}")
    plt.show()

if __name__ == "__main__":
    user_input = input("Enter digits to generate (e.g., 2345): ")
    if user_input.isdigit():
        show_number(user_input)
    else:
        print("❌ Please enter digits only.")
