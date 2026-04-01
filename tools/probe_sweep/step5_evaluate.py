"""
Step 5: Evaluate & Compare all baselines.

Runs evaluation on real MNIST test set for:
  A. Teacher (upper bound)
  B. Student trained on real MNIST (upper bound for student capacity)
  C. Student distilled from ManifoldProbe sweep data (our method)
  D. Student trained on random synthetic data (lower bound)

Also trains baselines B and D if their checkpoints don't exist.

Usage:
    cd mdistiller
    python tools/probe_sweep/step5_evaluate.py
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from mdistiller.models.mnist import TeacherCNN, StudentCNN
from mdistiller.dataset.mnist import get_mnist_dataloaders


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


def train_student_real(device, train_loader, val_loader, save_path, epochs=10, lr=1e-3):
    """Baseline B: Student trained on real MNIST from scratch."""
    print("\n[Baseline B] Training student on real MNIST...")
    student = StudentCNN(num_classes=10).to(device)
    optimizer = optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(1, epochs + 1):
        student.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = student(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        acc = evaluate(student, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save({"model": student.state_dict(), "val_acc": acc}, save_path)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: val_acc = {acc:.4f}")

    print(f"  Best: {best_acc:.4f}")
    return best_acc


def train_student_random(device, val_loader, save_path, num_samples=51200, epochs=30, lr=1e-3):
    """Baseline D: Student trained on random noise + teacher labels (random baseline)."""
    print("\n[Baseline D] Training student on random synthetic data...")

    # Generate random images, assign random labels
    rand_images = torch.rand(num_samples, 1, 28, 28)
    rand_labels = torch.randint(0, 10, (num_samples,))

    rand_dataset = torch.utils.data.TensorDataset(rand_images, rand_labels)
    rand_loader = torch.utils.data.DataLoader(rand_dataset, batch_size=128, shuffle=True)

    student = StudentCNN(num_classes=10).to(device)
    optimizer = optim.Adam(student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(1, epochs + 1):
        student.train()
        for images, labels in rand_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = student(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        acc = evaluate(student, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save({"model": student.state_dict(), "val_acc": acc}, save_path)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: val_acc = {acc:.4f}")

    print(f"  Best: {best_acc:.4f}")
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", type=str, default="checkpoints/teacher_mnist.pth")
    parser.add_argument("--student_manifold_ckpt", type=str, default="checkpoints/student_manifold.pth")
    parser.add_argument("--student_real_ckpt", type=str, default="checkpoints/student_real.pth")
    parser.add_argument("--student_random_ckpt", type=str, default="checkpoints/student_random.pth")
    parser.add_argument("--data_root", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_mnist_dataloaders(batch_size=128, data_root=args.data_root)

    results = {}

    # ── A. Teacher ──
    if os.path.exists(args.teacher_ckpt):
        teacher = TeacherCNN().to(device)
        ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
        teacher.load_state_dict(ckpt["model"])
        results["A. Teacher"] = evaluate(teacher, val_loader, device)
    else:
        print(f"WARNING: Teacher checkpoint not found: {args.teacher_ckpt}")
        results["A. Teacher"] = "N/A"

    # ── B. Student (real MNIST) ──
    if os.path.exists(args.student_real_ckpt):
        student = StudentCNN().to(device)
        ckpt = torch.load(args.student_real_ckpt, map_location="cpu", weights_only=False)
        student.load_state_dict(ckpt["model"])
        results["B. Student (real)"] = evaluate(student, val_loader, device)
    else:
        acc = train_student_real(device, train_loader, val_loader, args.student_real_ckpt)
        results["B. Student (real)"] = acc

    # ── C. Student (ManifoldProbe) ──
    if os.path.exists(args.student_manifold_ckpt):
        student = StudentCNN().to(device)
        ckpt = torch.load(args.student_manifold_ckpt, map_location="cpu", weights_only=False)
        student.load_state_dict(ckpt["model"])
        results["C. Student (ManifoldProbe)"] = evaluate(student, val_loader, device)
    else:
        print(f"WARNING: ManifoldProbe student not found: {args.student_manifold_ckpt}")
        results["C. Student (ManifoldProbe)"] = "N/A (run step4 first)"

    # ── D. Student (random) ──
    if os.path.exists(args.student_random_ckpt):
        student = StudentCNN().to(device)
        ckpt = torch.load(args.student_random_ckpt, map_location="cpu", weights_only=False)
        student.load_state_dict(ckpt["model"])
        results["D. Student (random)"] = evaluate(student, val_loader, device)
    else:
        acc = train_student_random(device, val_loader, args.student_random_ckpt)
        results["D. Student (random)"] = acc

    # ── Print summary ──
    print("\n" + "=" * 55)
    print("  ManifoldProbe KD — Experiment Results")
    print("=" * 55)
    for name, acc in results.items():
        if isinstance(acc, float):
            print(f"  {name:<35s} {acc:.4f} ({acc*100:.2f}%)")
        else:
            print(f"  {name:<35s} {acc}")
    print("=" * 55)

    # Interpretation
    c_acc = results.get("C. Student (ManifoldProbe)")
    d_acc = results.get("D. Student (random)")
    if isinstance(c_acc, float) and isinstance(d_acc, float):
        gap = c_acc - d_acc
        print(f"\n  ManifoldProbe vs Random gap: {gap:+.4f} ({gap*100:+.2f}%)")


if __name__ == "__main__":
    main()
