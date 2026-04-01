"""
Simple CNN models for MNIST experiments.
- TeacherCNN: larger model (~99% accuracy)
- StudentCNN: smaller model for distillation target

Both follow mdistiller convention: forward(x) -> (logits, feats_dict)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherCNN(nn.Module):
    """
    Teacher: Conv(1->32) -> Conv(32->64) -> FC(1600->128) -> FC(128->10)
    Expected accuracy: ~99% on MNIST
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        # After conv1+pool: 14x14x32, after conv2+pool: 7x7x64 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B, 32, 14, 14]
        f0 = x

        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B, 64, 7, 7]
        f1 = x

        # FC layers
        x = x.view(x.size(0), -1)  # [B, 3136]
        x = F.relu(self.fc1(x))    # [B, 128]
        pooled = x
        logits = self.fc2(x)       # [B, 10]

        feats = {
            "feats": [f0, f1],
            "pooled_feat": pooled,
        }
        return logits, feats


class StudentCNN(nn.Module):
    """
    Student: Conv(1->16) -> Conv(16->32) -> FC(800->64) -> FC(64->10)
    Roughly half the capacity of Teacher.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        f0 = x
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        f1 = x

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        pooled = x
        logits = self.fc2(x)

        feats = {
            "feats": [f0, f1],
            "pooled_feat": pooled,
        }
        return logits, feats
