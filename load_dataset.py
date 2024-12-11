import numpy as np
import pickle
import os

def load_batch(filepath):
    """
    Load a single batch of CIFAR-10 data.
    """
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        images = batch[b'data']
        labels = batch[b'labels']
        images = images.reshape(len(images), 3, 32, 32).transpose(0, 2, 3, 1)  # Reshape to (num_images, 32, 32, 3)
        return images, labels

def load_cifar10(cifar10_dir):
    """
    Load all CIFAR-10 training and test data, and combine them into a single dataset.
    """
    images = []
    labels = []

    # Load all training batches
    for i in range(1, 6):
        batch_file = os.path.join(cifar10_dir, f'data_batch_{i}')
        imgs, lbls = load_batch(batch_file)
        images.append(imgs)
        labels.append(lbls)

    # Load the test batch
    test_images, test_labels = load_batch(os.path.join(cifar10_dir, 'test_batch'))

    # Combine all training batches
    train_images = np.concatenate(images)
    train_labels = np.concatenate(labels)

    # Combine train and test sets
    all_images = np.concatenate((train_images, test_images))
    all_labels = np.concatenate((train_labels, test_labels))

    return all_images, all_labels

# Load the CIFAR-10 dataset
if __name__ == "__main__":
    cifar10_dir = 'cifar-10-batches-py'  # Directory containing extracted CIFAR-10 files
    images, labels = load_cifar10(cifar10_dir)
    print(f"Total number of images: {images.shape[0]}")
    print(f"Image shape: {images.shape[1:]}")