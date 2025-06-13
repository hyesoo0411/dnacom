# Setup training data
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor


train_data = datasets.MNIST(
    root="/home/lab/DNAcomputing/Dataset",  # where to download data to?
    train=True,  # do we want the training dataset?
    download=True,  # do we want to download?
    transform=torchvision.transforms.ToTensor(),  # how do we want to transform the data?
    target_transform=None,  # how do we want to transform the labels/targets?
)

test_data = datasets.MNIST(
    root="/home/lab/DNAcomputing/Dataset", train=False, download=True, transform=ToTensor(), target_transform=None
)
