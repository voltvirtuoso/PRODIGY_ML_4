import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Define the path to the directory containing the image folders
data_dir = "leapGestRecog"

# Preprocess the images (resize, normalize, etc.)
def preprocess_image(image):
    # Resize the image to a fixed size (e.g., 64x64)
    resized_image = cv2.resize(image, (64, 64))
    # Normalize pixel values to range [0, 1]
    normalized_image = resized_image / 255.0
    return normalized_image

# Load images and labels
images = []
labels = []
for root, dirs, files in os.walk(data_dir):
    for label_dir in dirs:
        label = os.path.basename(label_dir)
        label_path = os.path.join(root, label_dir)
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Error: Unable to read image at {image_path}")
            preprocessed_image = preprocess_image(image)
            images.append(preprocessed_image)
            labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Check the shape of the training and testing data
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
