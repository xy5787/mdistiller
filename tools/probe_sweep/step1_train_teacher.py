"""
Step 1: Train a Teacher CNN on MNIST.
Output: checkpoints/teacher_mnist.pth

Usage:
    cd mdistiller
    python tools/probe_sweep/step1_train_teacher.py
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

# ── path setup so we can import from mdistiller package ──
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from mdistiller.models.mnist import TeacherCNN
from mdistiller.dataset.mnist import get_mnist_dataloaders


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--data_root", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, _ = get_mnist_dataloaders(
        batch_size=args.batch_size, data_root=args.data_root
    )

    # Model
    model = TeacherCNN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train
    best_acc = 0
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "teacher_mnist.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:3d} | train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }, save_path)

    print(f"\nBest val accuracy: {best_acc:.4f}")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
