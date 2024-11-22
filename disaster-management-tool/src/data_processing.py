import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_images_from_directory(directory, size=(224, 224)):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(directory, filename))
            img = cv2.resize(img, size)
            images.append(img)
    return np.array(images)

def preprocess_data(images):
    # Normalize pixel values between 0 and 1
    images = images.astype('float32') / 255.0
    return images

def create_train_test_split(images, labels, test_size=0.2):
    return train_test_split(images, labels, test_size=test_size, random_state=42)
