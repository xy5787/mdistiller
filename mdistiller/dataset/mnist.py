"""
MNIST dataloader for ManifoldProbe experiments.
Follows mdistiller convention: returns (train_loader, val_loader, num_data).
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloaders(batch_size=128, val_batch_size=256, num_workers=4, data_root="./data"):
    """
    Standard MNIST train/val dataloaders.
    Used for: (1) training teacher, (2) evaluating student, (3) seed images for saliency.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # MNIST mean=0.1307, std=0.3081 -- but for pixel sweep we want raw [0,1]
        # so we intentionally do NOT normalize here. Keep it simple for toy project.
    ])

    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, len(train_dataset)
