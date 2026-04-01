"""
Step 3 (v3): True exhaustive k-pixel sweep — memory-safe version.

Results saved per-class to avoid OOM.
Images NOT stored — reconstructed on-the-fly during training.

Usage:
    cd mdistiller

    # k=3, 256 values (~20min GPU)
    python tools/probe_sweep/step3_pixel_sweep.py \
        --k 3 --num_values 256 --save_dir outputs/sweep_k3_v256

    # k=4, 32 values (~1min GPU)
    python tools/probe_sweep/step3_pixel_sweep.py \
        --k 4 --num_values 32 --save_dir outputs/sweep_k4_v32
"""
import os
import sys
import argparse
import torch
import gc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from mdistiller.models.mnist import TeacherCNN
from mdistiller.dataset.mnist import get_mnist_dataloaders
from mdistiller.distillers.ManifoldProbe import ManifoldProbe, entropy_filter


def get_mean_image(dataloader, target_class):
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
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--num_values", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--entropy_low", type=float, default=5)
    parser.add_argument("--entropy_high", type=float, default=95)
    parser.add_argument("--save_dir", type=str, default="outputs/sweep_exhaustive")
    parser.add_argument("--data_root", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_per_class = args.num_values ** args.k

    print(f"{'='*60}")
    print(f"  Exhaustive Sweep: k={args.k}, {args.num_values} values/pixel")
    print(f"  Combinations per class: {total_per_class:,}")
    print(f"  Total (10 classes): {total_per_class * 10:,}")
    print(f"  Memory mode: combo_values only (no image storage)")
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

    # Dataloader
    train_loader, _, _ = get_mnist_dataloaders(batch_size=256, data_root=args.data_root)

    os.makedirs(args.save_dir, exist_ok=True)

    # Save metadata for reconstruction later
    meta = {
        "k": args.k,
        "num_values": args.num_values,
        "pixels": {},       # class -> [(r,c),...]
        "base_images": {},  # class -> [1,1,28,28]
    }

    total_kept = 0

    for c in range(10):
        pixels_k = all_pixels[c][:args.k]
        seed = get_mean_image(train_loader, target_class=c)

        meta["pixels"][c] = pixels_k
        meta["base_images"][c] = seed

        print(f"\nClass {c}: pixels={pixels_k}")

        # Sweep — 결과가 save_path에 저장됨
        class_save = os.path.join(args.save_dir, f"class_{c}_raw.pt")
        probe.pixel_sweep_exhaustive(
            base_image=seed,
            salient_pixels_k=pixels_k,
            num_values=args.num_values,
            batch_size=args.batch_size,
            save_path=class_save,
        )

        # Load back for entropy filtering (combo_values + soft_labels만이라 작음)
        raw = torch.load(class_save, weights_only=False)
        print(f"    Raw: {len(raw['hard_labels']):,} samples")

        filtered, stats = entropy_filter(
            raw,
            low_percentile=args.entropy_low,
            high_percentile=args.entropy_high,
        )
        print(f"    Filtered: {stats['filtered_size']:,} / {stats['original_size']:,}")

        # Save filtered version
        torch.save(filtered, os.path.join(args.save_dir, f"class_{c}.pt"))
        total_kept += stats['filtered_size']

        # Remove raw file to save disk
        os.remove(class_save)

        # Force garbage collection
        del raw, filtered
        gc.collect()
        torch.cuda.empty_cache()

    # Save metadata
    torch.save(meta, os.path.join(args.save_dir, "meta.pt"))

    print(f"\n{'='*60}")
    print(f"Done! Total filtered samples: {total_kept:,}")
    print(f"Saved to: {args.save_dir}/")
    print(f"  class_0.pt ~ class_9.pt  (per-class sweep data)")
    print(f"  meta.pt                   (base_images + pixel positions)")


if __name__ == "__main__":
    main()