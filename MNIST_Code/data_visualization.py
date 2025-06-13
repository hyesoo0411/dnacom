"""
model_train_wQAT101_Prune.py: 가중치만 -1 0 1로 양자화 => Pruning
"""

import os

import torch

from torchvision import transforms

import data_setup

import matplotlib

import numpy as np

import os
from PIL import Image

matplotlib.use("Agg")

# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Setup hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_UNITS = 200  # 200, 400, 800, 1600
LEARNING_RATE = 0.001  # 0.001, 0.0005, 0.0001, 0.00005
apply_ternary_quantization = True
bias = False
pruning = True
scale_factor = 0.75

# Setup directories
train_dir = "./Dataset/train"
test_dir = "./Dataset/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([transforms.Resize((7, 7)), transforms.ToTensor()])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir, test_dir=test_dir, transform=data_transform, batch_size=BATCH_SIZE
)

def save_binarized_images(dataloader, class_names, output_dir="binarized_images"):
    """
    Saves one batch of binarized images from the given DataLoader to a specified directory.

    Args:
        dataloader: A PyTorch DataLoader containing the binarized images.
        class_names: List of class names for labeling.
        output_dir: Directory where the images will be saved.
    """
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Fetch one batch of data from the DataLoader
    for batch_idx, (images, labels) in enumerate(dataloader):
        batch_size = images.shape[0]
        print(labels)
        for i in range(batch_size):
            # Convert image tensor to a PIL image for saving
            image = images[i].squeeze().numpy()
            image = (image * 255).astype(np.uint8)  # Scale to 0-255 for saving
            pil_image = Image.fromarray(image)

            # Create a filename based on batch and sample index
            label = class_names[labels[i]]
            filename = f"batch_{batch_idx}_sample_{i}_label_{label}.png"
            file_path = os.path.join(output_dir, filename)

            # Save the image
            pil_image.save(file_path)

        print(f"Saved batch {batch_idx} of images.")
        break  # Only process and save one batch


save_binarized_images(train_dataloader, class_names)

