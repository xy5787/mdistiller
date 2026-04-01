"""
Synthetic dataset for ManifoldProbe exhaustive sweep.

v3: images are NOT stored on disk.
    Instead, stores (combo_values, soft_labels, base_image, pixel_positions).
    Images are reconstructed on-the-fly in __getitem__.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class SyntheticExhaustiveDataset(Dataset):
    """
    Reconstructs images on-the-fly from base_image + combo_values.

    Each .pt file contains:
        combo_values: [N, k] — pixel values used in sweep
        soft_labels: [N, 10]
        hard_labels: [N]
    Plus metadata: base_image [1,1,28,28], salient_pixels [(r,c),...]
    """

    def __init__(self, sweep_path, base_image, salient_pixels_k):
        """
        Args:
            sweep_path: path to per-class .pt file
            base_image: [1, 1, 28, 28] seed image for this class
            salient_pixels_k: list of (r, c) tuples, length k
        """
        data = torch.load(sweep_path, weights_only=False)
        self.combo_values = data["combo_values"]  # [N, k]
        self.soft_labels = data["soft_labels"]     # [N, 10]
        self.hard_labels = data["hard_labels"]     # [N]
        self.base_image = base_image.squeeze(0)    # [1, 28, 28]
        self.salient_pixels_k = salient_pixels_k

    def __len__(self):
        return len(self.combo_values)

    def __getitem__(self, idx):
        # Reconstruct image
        img = self.base_image.clone()  # [1, 28, 28]
        for p_idx, (r, c) in enumerate(self.salient_pixels_k):
            img[0, r, c] = self.combo_values[idx, p_idx]
        return img, self.soft_labels[idx], self.hard_labels[idx]


def get_exhaustive_dataloader(class_dir, base_images, all_pixels_k,
                               batch_size=256, num_workers=4):
    """
    Load all per-class sweep files and combine into one DataLoader.

    Args:
        class_dir: directory containing class_0.pt ~ class_9.pt
        base_images: dict {class_idx: [1,1,28,28] tensor}
        all_pixels_k: dict {class_idx: [(r,c),...]}
        batch_size, num_workers: DataLoader params

    Returns:
        loader, total_samples
    """
    datasets = []
    for c in range(10):
        path = os.path.join(class_dir, f"class_{c}.pt")
        if os.path.exists(path):
            ds = SyntheticExhaustiveDataset(path, base_images[c], all_pixels_k[c])
            datasets.append(ds)
            print(f"  Class {c}: {len(ds):,} samples")

    combined = ConcatDataset(datasets)
    loader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, len(combined)


# ── 기존 v1/v2 호환용 (이미지가 저장된 파일) ──

class SyntheticSweepDataset(Dataset):
    """Legacy: loads pre-stored images directly."""

    def __init__(self, sweep_path):
        data = torch.load(sweep_path, weights_only=False)
        self.images = data["images"]
        self.soft_labels = data["soft_labels"]
        self.hard_labels = data["hard_labels"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.soft_labels[idx], self.hard_labels[idx]


def get_synthetic_dataloader(sweep_path, batch_size=128, num_workers=4):
    dataset = SyntheticSweepDataset(sweep_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True)
    return loader, len(dataset)