"""
Contains functionality for creating PyTorch DataLoaders for
image classification data.
"""

import os

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy as np

from torch.utils.data import Subset, Dataset
import torch.nn.functional as F
from typing import Dict, List, Tuple
import matplotlib
import matplotlib.pyplot as plt




matplotlib.use('Agg')


NUM_WORKERS = os.cpu_count()


def create_dataloaders(
    train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int, num_workers: int = NUM_WORKERS
):
    """Creates training and testing DataLoaders.
    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
        train_dataloader, test_dataloader, class_names = \
            = create_dataloaders(train_dir=path/to/train_dir,
                                test_dir=path/to/test_dir,
                                transform=some_transform,
                                batch_size=32,
                                num_workers=4)
    """

    # Use ImageFolder to create dataset(s)
    train_data = datasets.MNIST(
        root=train_dir,  # where to download data to?
        train=True,  # do we want the training dataset?
        download=True,  # do we want to download?
        transform=transform,  # how do we want to transform the data?
        target_transform=None,
    )  # how do we want to transform the labels/targets?

    test_data = datasets.MNIST(root=test_dir, train=False, download=True, transform=transform, target_transform=None)

    # Get class names
    class_names = train_data.classes

    # print(len(train_data), len(test_data))

    # Function to filter 4 and 7 class images
    # def filter_4_and_7(dataset):
    #     indices = [i for i, (img, label) in enumerate(dataset) if label in [4, 7]]
    #     return Subset(dataset, indices)

    # Apply the filter to get only 4 and 7 classes
    # train_data_4_7 = filter_4_and_7(train_data)
    # test_data_4_7 = filter_4_and_7(test_data)

    def binarize_tensor(tensor):
        return torch.where(tensor > 0.25, torch.tensor(1.0), torch.tensor(-1.0))  # Converts values > 0.1 to 1 and others to -1

    # 커스텀 데이터셋 클래스 정의 (이진화된 데이터를 반환)
    class BinarizedDataset(Dataset):
        # classes = [
        #     "4 - four",
        #     "7 - seven",
        # ]

        classes = [
            "0 - zero",
            "1 - one",
            "2 - two",
            "3 - three",
            "4 - four",
            "5 - five",
            "6 - six",
            "7 - seven",
            "8 - eight",
            "9 - nine",
        ]

        def __init__(self, original_dataset):
            self.original_dataset = original_dataset

        def __len__(self):
            return len(self.original_dataset)

        def __getitem__(self, idx):
            image, label = self.original_dataset[idx]
            binary_image = binarize_tensor(image)  # 이미지 이진화
            return binary_image, label

        def class_to_idx(self) -> Dict[str, int]:
            return {_class: i for i, _class in enumerate(self.classes)}

    # train_data_4_7 = BinarizedDataset(train_data_4_7)
    # test_data_4_7 = BinarizedDataset(test_data_4_7)

    train_data = BinarizedDataset(train_data)
    test_data = BinarizedDataset(test_data)

    class_names = train_data.classes


    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names

