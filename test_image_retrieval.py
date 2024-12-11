
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from load_dataset import load_cifar10
import os

# Load saved features and image names for CIFAR-10
features = np.load('cifar10_features.npy')
image_names = np.load('cifar10_image_names.npy')

# Load CIFAR-10 images
cifar10_dir = 'cifar-10-batches-py'
images, labels = load_cifar10(cifar10_dir)

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def retrieve_similar_images(query_image_path, top_n=5):
    # Load and preprocess the query image
    query_img = cv2.imread(query_image_path)
    query_img = cv2.resize(query_img, (224, 224))
    query_img = np.expand_dims(query_img, axis=0)
    query_img = preprocess_input(query_img)

    # Extract features of the query image
    query_features = model.predict(query_img).flatten().reshape(1, -1)

    # Compute cosine similarity between the query image and all other images
    similarities = cosine_similarity(query_features, features)[0]

    # Sort the indices based on similarity scores in descending order
    similar_indices = similarities.argsort()[::-1][:top_n]

    # Display the query image and top N similar images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, top_n + 1, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(query_image_path), cv2.COLOR_BGR2RGB))
    plt.title('Query Image')
    plt.axis('off')

    for i, idx in enumerate(similar_indices):
        similar_img = images[idx]
        plt.subplot(1, top_n + 1, i + 2)
        plt.imshow(cv2.cvtColor(similar_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Similar {i + 1}")
        plt.axis('off')

    plt.show()

# Example usage: Upload your query image to a specific path and then use that path here
if __name__ == "__main__":
    query_image_path = "Query_Image/download.jpeg"
    if os.path.exists(query_image_path):
        retrieve_similar_images(query_image_path)
    else:
        print("Invalid path. Please make sure the image exists.")
