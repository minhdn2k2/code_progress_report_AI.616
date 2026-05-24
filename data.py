import random
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _make_subset(dataset, subset_size: int, seed: int):
    if subset_size <= 0 or subset_size >= len(dataset):
        return dataset, list(range(len(dataset)))

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:subset_size]
    return Subset(dataset, indices), indices


def get_cifar10_loaders(
    data_dir: str,
    train_batch_size: int,
    eval_batch_size: int,
    micro_batch_size: int,
    train_subset: int = 10000,
    num_workers: int = 4,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, int]:
    """
    Returns:
      train_loader: shuffled, augmented loader for SGD/m-SAM
      full_train_loader: non-shuffled, augmented loader for full-batch n-SAM gradient accumulation
      train_eval_loader: non-augmented train evaluation loader
      test_loader: non-augmented test loader
      train_size: number of training examples used
    """

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_aug = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    train_eval = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=eval_transform,
    )
    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    train_aug_subset, indices = _make_subset(train_aug, train_subset, seed)

    if train_subset > 0 and train_subset < len(train_eval):
        train_eval_subset = Subset(train_eval, indices)
    else:
        train_eval_subset = train_eval

    train_size = len(train_aug_subset)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_aug_subset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        drop_last=False,
    )

    full_train_loader = DataLoader(
        train_aug_subset,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    train_eval_loader = DataLoader(
        train_eval_subset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, full_train_loader, train_eval_loader, test_loader, train_size