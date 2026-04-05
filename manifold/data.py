from .utils import RandomGaussianNoise

import torchvision.transforms as transforms
import torchvision

from typing import Tuple
from torch.utils.data import DataLoader
from codon.utils.seed import create_generator


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def mnist(
    batch_size: int = 128,
    std: float = None
) -> Tuple[DataLoader, DataLoader]:
    '''
    Loads the MNIST dataset and returns the train and test DataLoaders.

    Args:
        batch_size (int): The batch size for DataLoaders.
        std (float): Standard deviation for RandomGaussianNoise. Defaults to None.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test DataLoaders.
    '''
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        *([RandomGaussianNoise(max_std=std)] if std is not None else []),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        *([RandomGaussianNoise(max_std=std)] if std is not None else []),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])

    train_data = torchvision.datasets.MNIST(
        root='../../data/mnist',
        train=True,
        download=True,
        transform=train_transform
    )
    test_data = torchvision.datasets.MNIST(
        root='../../data/mnist',
        train=False,
        download=True,
        transform=test_transform
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=create_generator()
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, test_loader


def cifar_100(
    batch_size: int = 128,
    std: float = None
) -> Tuple[DataLoader, DataLoader]:
    '''
    Loads the CIFAR-100 dataset and returns the train and test DataLoaders.

    Args:
        batch_size (int): The batch size for DataLoaders.
        std (float): Standard deviation for RandomGaussianNoise. Defaults to None.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test DataLoaders.
    '''
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        *([RandomGaussianNoise(max_std=std)] if std is not None else []),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        *([RandomGaussianNoise(max_std=std)] if std is not None else []),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])

    train_data = torchvision.datasets.CIFAR100(
        root='../../data/cifar100',
        train=True,
        download=True,
        transform=train_transform
    )
    test_data = torchvision.datasets.CIFAR100(
        root='../../data/cifar100',
        train=False,
        download=True,
        transform=test_transform
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=create_generator()
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, test_loader


def cifar(
    batch_size: int = 128,
    std: float = None
) -> Tuple[DataLoader, DataLoader]:
    '''
    Loads the CIFAR-10 dataset and returns the train and test DataLoaders.

    Args:
        batch_size (int): The batch size for DataLoaders.
        std (float): Standard deviation for RandomGaussianNoise. Defaults to None.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test DataLoaders.
    '''
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        *([RandomGaussianNoise(max_std=std)] if std is not None else []),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        *([RandomGaussianNoise(max_std=std)] if std is not None else []),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    train_data = torchvision.datasets.CIFAR10(
        root='../../data/cifar10',
        train=True,
        download=True,
        transform=train_transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root='../../data/cifar10',
        train=False,
        download=True,
        transform=test_transform
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=create_generator()
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, test_loader
