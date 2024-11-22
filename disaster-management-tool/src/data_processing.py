import cv2
import os
from utils.config_loader import load_config

# Load config
config = load_config("config.yaml")

raw_images_path = config['data']['raw_images']
processed_images_path = config['data']['processed_images']
resize_dim = config['image_processing']['resize_dim']

# Function to preprocess images (resize and save)
def preprocess_and_save_image(image_path, save_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, tuple(resize_dim))  # Resize to config dimension
    cv2.imwrite(save_path, image)

# Process and save images
os.makedirs(processed_images_path, exist_ok=True)
for img_file in os.listdir(raw_images_path):
    img_path = os.path.join(raw_images_path, img_file)
    save_path = os.path.join(processed_images_path, img_file)
    preprocess_and_save_image(img_path, save_path)
