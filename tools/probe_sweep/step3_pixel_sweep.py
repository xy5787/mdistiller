"""
Step 3 (v3): True exhaustive k-pixel sweep.

k=3, 256 values: 256^3 = 16,777,216 combinations per class per seed
k=4, 32 values:   32^4 =  1,048,576 combinations per class per seed

Usage:
    cd mdistiller

    # Experiment 1: k=3, full 256 resolution (~20min)
    python tools/probe_sweep/step3_pixel_sweep.py \
        --k 3 --num_values 256 --save_path outputs/sweep_k3_v256.pt

    # Experiment 2: k=4, coarse 32 resolution (~1min)
    python tools/probe_sweep/step3_pixel_sweep.py \
        --k 4 --num_values 32 --save_path outputs/sweep_k4_v32.pt
"""
import os
import sys
import argparse
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from mdistiller.models.mnist import TeacherCNN
from mdistiller.dataset.mnist import get_mnist_dataloaders
from mdistiller.distillers.ManifoldProbe import ManifoldProbe, entropy_filter


def get_mean_image(dataloader, target_class):
    """Get mean image of a class as seed."""
    class_sum = torch.zeros(1, 1, 28, 28)
    count = 0
    for images, labels in dataloader:
        mask = labels == target_class
        if mask.sum() > 0:
            class_sum += images[mask].sum(dim=0, keepdim=True)
            count += mask.sum().item()
    return class_sum / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", type=str, default="checkpoints/teacher_mnist.pth")
    parser.add_argument("--saliency_path", type=str, default="outputs/saliency/all_classes.pt")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of pixels to exhaustively sweep")
    parser.add_argument("--num_values", type=int, default=256,
                        help="Values per pixel (total combos = num_values^k)")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for teacher inference")
    parser.add_argument("--entropy_low", type=float, default=5)
    parser.add_argument("--entropy_high", type=float, default=95)
    parser.add_argument("--save_path", type=str, default="outputs/sweep_exhaustive.pt")
    parser.add_argument("--data_root", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_per_class = args.num_values ** args.k

    print(f"{'='*60}")
    print(f"  Exhaustive Sweep: k={args.k}, {args.num_values} values/pixel")
    print(f"  Combinations per class: {total_per_class:,}")
    print(f"  Total (10 classes): {total_per_class * 10:,}")
    print(f"{'='*60}")

    # Load teacher
    teacher = TeacherCNN(num_classes=10)
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    teacher.load_state_dict(ckpt["model"])
    print(f"Loaded teacher (val_acc: {ckpt['val_acc']:.4f})")

    probe = ManifoldProbe(teacher, device=device)

    # Load saliency
    sal_data = torch.load(args.saliency_path, weights_only=False)
    all_pixels = sal_data["pixels"]

    # Dataloader for mean images
    train_loader, _, _ = get_mnist_dataloaders(batch_size=256, data_root=args.data_root)

    # ── Per-class exhaustive sweep ──
    all_images = []
    all_soft = []
    all_hard = []

    for c in range(10):
        pixels_k = all_pixels[c][:args.k]  # top-k most salient
        seed = get_mean_image(train_loader, target_class=c)

        print(f"\nClass {c}: pixels={pixels_k}")
        result = probe.pixel_sweep_exhaustive(
            base_image=seed,
            salient_pixels_k=pixels_k,
            num_values=args.num_values,
            batch_size=args.batch_size,
        )

        all_images.append(result["images"])
        all_soft.append(result["soft_labels"])
        all_hard.append(result["hard_labels"])

        # Per-class stats
        dist = torch.bincount(result["hard_labels"], minlength=10)
        print(f"    Hard label dist: {dist.tolist()}")

    # Merge
    raw_dataset = {
        "images": torch.cat(all_images, dim=0),
        "soft_labels": torch.cat(all_soft, dim=0),
        "hard_labels": torch.cat(all_hard, dim=0),
    }
    print(f"\n{'='*60}")
    print(f"Raw dataset: {len(raw_dataset['images']):,} samples")

    # ── Entropy filtering ──
    filtered, stats = entropy_filter(
        raw_dataset,
        low_percentile=args.entropy_low,
        high_percentile=args.entropy_high,
    )

    print(f"\nEntropy filtering:")
    print(f"  Range: [{stats['entropy_low_thresh']:.4f}, {stats['entropy_high_thresh']:.4f}]")
    print(f"  Mean: {stats['entropy_mean']:.4f} (std: {stats['entropy_std']:.4f})")
    print(f"  Kept: {stats['filtered_size']:,} / {stats['original_size']:,}")

    # Save
    filtered["metadata"] = {
        "k": args.k,
        "num_values": args.num_values,
        "total_raw": stats["original_size"],
        "total_filtered": stats["filtered_size"],
        "entropy_stats": stats,
    }

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(filtered, args.save_path)

    print(f"\nSaved: {args.save_path}")
    print(f"  Shape: {filtered['images'].shape}")
    print(f"  File size: {os.path.getsize(args.save_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()