import numpy as np
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from load_dataset import load_cifar10

# Load CIFAR-10 images
cifar10_dir = 'cifar-10-batches-py'
images, labels = load_cifar10(cifar10_dir)

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# Extract features from CIFAR-10 images
features = []
image_names = []

for i, img in enumerate(images):
    # Resize CIFAR-10 image to (224, 224) for VGG16
    img_resized = cv2.resize(img, (224, 224))
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = preprocess_input(img_resized)

    # Extract features
    feature_vector = model.predict(img_resized)
    features.append(feature_vector.flatten())
    image_names.append(f'cifar10_image_{i}.jpg')  # Naming each image

    if i % 1000 == 0:
        print(f"Processed {i}/{images.shape[0]} images")

# Save features and image names for later use
features = np.array(features)
np.save('cifar10_features.npy', features)
np.save('cifar10_image_names.npy', image_names)

print("Feature extraction complete. Features saved as 'cifar10_features.npy' and image names as 'cifar10_image_names.npy'.")

