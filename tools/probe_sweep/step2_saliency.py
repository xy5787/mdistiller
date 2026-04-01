"""
Step 2: Extract per-class saliency maps from Teacher.
Output: outputs/saliency/class_{0-9}.pt + visualization PNGs

Usage:
    cd mdistiller
    python tools/probe_sweep/step2_saliency.py
    python tools/probe_sweep/step2_saliency.py --max_samples 200 --top_k 30
"""
import os
import sys
import argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from mdistiller.models.mnist import TeacherCNN
from mdistiller.dataset.mnist import get_mnist_dataloaders
from mdistiller.distillers.ManifoldProbe import ManifoldProbe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", type=str, default="checkpoints/teacher_mnist.pth")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Number of seed images per class for saliency averaging")
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of salient pixels to select per class")
    parser.add_argument("--save_dir", type=str, default="outputs/saliency")
    parser.add_argument("--data_root", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load teacher
    teacher = TeacherCNN(num_classes=10)
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    teacher.load_state_dict(ckpt["model"])
    print(f"Loaded teacher (val_acc: {ckpt['val_acc']:.4f})")

    probe = ManifoldProbe(teacher, device=device)

    # Dataloader (for seed images)
    train_loader, _, _ = get_mnist_dataloaders(batch_size=64, data_root=args.data_root)

    os.makedirs(args.save_dir, exist_ok=True)

    # Per-class saliency
    all_saliency = {}
    all_pixels = {}

    for c in range(10):
        print(f"\nClass {c}: computing saliency over {args.max_samples} samples...")
        sal_map = probe.get_class_saliency(train_loader, target_class=c, max_samples=args.max_samples)
        pixels = probe.select_salient_pixels(sal_map, top_k=args.top_k)

        all_saliency[c] = sal_map
        all_pixels[c] = pixels

        print(f"  Top-{args.top_k} pixels: {pixels[:5]}... (showing first 5)")

        # Save individual
        torch.save({"saliency_map": sal_map, "salient_pixels": pixels},
                    os.path.join(args.save_dir, f"class_{c}.pt"))

    # Save combined
    torch.save({"saliency": all_saliency, "pixels": all_pixels},
               os.path.join(args.save_dir, "all_classes.pt"))

    # ── Visualization ──
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for c in range(10):
        ax = axes[c // 5][c % 5]
        sal = all_saliency[c].numpy()
        ax.imshow(sal, cmap="hot", interpolation="nearest")
        # Mark top-k pixels
        for (r, col) in all_pixels[c]:
            ax.plot(col, r, "c+", markersize=6, markeredgewidth=1.5)
        ax.set_title(f"Class {c}")
        ax.axis("off")
    plt.suptitle(f"Per-class Saliency Maps (top-{args.top_k} pixels marked)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "saliency_overview.png"), dpi=150)
    print(f"\nVisualization saved: {args.save_dir}/saliency_overview.png")
    print(f"All data saved: {args.save_dir}/all_classes.pt")


if __name__ == "__main__":
    main()
