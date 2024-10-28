import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import os
import shutil

# Function to load and preprocess images
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]  # Add batch dimension
    return img

# Neural Style Transfer function
def neural_style_transfer(content_img, style_img):
    model_url = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    try:
        # Use a different cache directory to avoid previous corrupted downloads
        os.environ['TFHUB_CACHE_DIR'] = './tfhub_modules_cache'
        model = hub.load(model_url)
        print("Model loaded successfully.")
    except Exception as e:
        print("Failed to load model:", e)
        return None
    result = model(tf.constant(content_img), tf.constant(style_img))[0]
    return result

# Ensure cache is cleared
# cache_dir = './tfhub_modules_cache'
# if os.path.exists(cache_dir):
#     shutil.rmtree(cache_dir)

# # Load content and style images
# style_image = load_image(r'generated_images\epoch_6000.png')
# content_image = load_image(r'13.jpg')

# # Display content and style images
# plt.imshow(np.squeeze(style_image))
# plt.title("Style Image")
# plt.show()

# plt.imshow(np.squeeze(content_image))
# plt.title("Content Image")
# plt.show()

# # Perform Neural Style Transfer
# stylized_image = neural_style_transfer(content_image, style_image)

# # Display the stylized image
# if stylized_image is not None:
#     plt.imshow(np.squeeze(stylized_image))
#     plt.title("Stylized Image")
#     plt.show()
