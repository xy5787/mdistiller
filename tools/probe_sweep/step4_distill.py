"""
Step 4 (v3): Distill Student using exhaustive sweep data.

Loads per-class files and reconstructs images on-the-fly.

Usage:
    cd mdistiller
    python tools/probe_sweep/step4_distill.py --sweep_dir outputs/sweep_k3_v256
"""
import os
import sys
import argparse
import torch
import torch.optim as optim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from mdistiller.models.mnist import StudentCNN
from mdistiller.dataset.mnist import get_mnist_dataloaders
from mdistiller.dataset.synthetic import get_exhaustive_dataloader
from mdistiller.distillers.ManifoldProbe import ManifoldProbeKD


@torch.no_grad()
def evaluate_on_real(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return correct / total


def train_one_epoch(distiller, loader, optimizer, device):
    distiller.train()
    total_loss, correct, total = 0, 0, 0

    for images, soft_labels, hard_labels in loader:
        images = images.to(device)
        soft_labels = soft_labels.to(device)
        hard_labels = hard_labels.to(device)

        optimizer.zero_grad()
        logits, losses = distiller(images, soft_labels, hard_labels)
        loss = sum(losses.values())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == hard_labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", type=str, default="outputs/sweep_exhaustive",
                        help="Directory with class_0.pt~class_9.pt + meta.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--ce_weight", type=float, default=0.1)
    parser.add_argument("--kd_weight", type=float, default=0.9)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--data_root", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load metadata (base images + pixel positions)
    meta = torch.load(os.path.join(args.sweep_dir, "meta.pt"), weights_only=False)

    # Build dataloader (images reconstructed on-the-fly)
    print(f"\nLoading sweep data from {args.sweep_dir}/")
    syn_loader, num_syn = get_exhaustive_dataloader(
        class_dir=args.sweep_dir,
        base_images=meta["base_images"],
        all_pixels_k=meta["pixels"],
        batch_size=args.batch_size,
    )
    print(f"Total synthetic samples: {num_syn:,}")

    # Real MNIST val loader (evaluation only)
    _, val_loader, _ = get_mnist_dataloaders(batch_size=256, data_root=args.data_root)

    # Student + Distiller
    student = StudentCNN(num_classes=10).to(device)
    distiller = ManifoldProbeKD(
        student,
        temperature=args.temperature,
        ce_weight=args.ce_weight,
        kd_weight=args.kd_weight,
    ).to(device)

    optimizer = optim.Adam(student.parameters(), lr=args.lr)

    # Train
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "student_manifold.pth")
    best_real_acc = 0

    print(f"\n{'Epoch':>5} | {'Syn Loss':>10} | {'Syn Acc':>8} | {'Real Acc':>8}")
    print("-" * 45)

    for epoch in range(1, args.epochs + 1):
        syn_loss, syn_acc = train_one_epoch(distiller, syn_loader, optimizer, device)
        real_acc = evaluate_on_real(student, val_loader, device)

        print(f"{epoch:5d} | {syn_loss:10.4f} | {syn_acc:7.4f} | {real_acc:7.4f}")

        if real_acc > best_real_acc:
            best_real_acc = real_acc
            torch.save({
                "model": student.state_dict(),
                "epoch": epoch,
                "real_acc": real_acc,
            }, save_path)

    print(f"\nBest real MNIST accuracy: {best_real_acc:.4f}")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()