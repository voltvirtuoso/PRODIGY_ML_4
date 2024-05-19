import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# Function to preprocess image
def preprocess_image(image):
    resized_image = cv2.resize(image, (64, 64))
    normalized_image = resized_image / 255.0
    return normalized_image

# Define the root directory containing the image folders
data_dir = "leapGestRecog"

# Load images and labels
images = []
labels = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        file_path = os.path.join(root, file)
        if file_path.endswith(".jpg") or file_path.endswith(".png"):  # Filter image files
            label = os.path.basename(root)
            image = cv2.imread(file_path)
            if image is not None:
                preprocessed_image = preprocess_image(image)
                images.append(preprocessed_image)
                labels.append(label)
                print("Processed:", file_path)  # Print file path
            else:
                print(f"Warning: Unable to read image at {file_path}")

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Encode labels as numerical values
label_mapping = {label: i for i, label in enumerate(np.unique(labels))}
labels = np.array([label_mapping[label] for label in labels])

# Extract gesture names from label mapping
gesture_names = list(label_mapping.keys())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Input(shape=(64, 64, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(label_mapping), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

# Save the trained model
model.save("hand_gesture_model.h5")

# Save gesture names
np.save("gesture_names.npy", gesture_names)
