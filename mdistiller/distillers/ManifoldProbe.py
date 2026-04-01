"""
ManifoldProbe: Data-Free KD via gradient saliency + pixel sweep.

v2 improvements:
- Multi-seed: multiple seed images per class
- 2-pixel combo sweep: pairwise pixel interaction capture
- Entropy filtering: remove off-manifold samples
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from tqdm import tqdm


# ============================================================
# Part 1: Saliency & Sweep (data generation phase)
# ============================================================

class ManifoldProbe:
    """
    Probes a teacher network to find salient pixels and sweep them.
    """

    def __init__(self, teacher, device="cuda"):
        self.teacher = teacher.to(device).eval()
        self.device = device

    @torch.enable_grad()
    def get_saliency(self, x, target_class):
        """
        Compute |d(logit[target_class]) / d(x)| for a single image.
        """
        x = x.clone().detach().to(self.device).requires_grad_(True)
        logits, _ = self.teacher(x)
        logits[0, target_class].backward()
        saliency = x.grad.abs().squeeze()  # [28, 28]
        return saliency.detach().cpu()

    def get_class_saliency(self, dataloader, target_class, max_samples=100):
        """
        Average saliency map across multiple samples of a given class.
        """
        saliency_sum = torch.zeros(28, 28)
        count = 0

        for images, labels in dataloader:
            mask = labels == target_class
            if mask.sum() == 0:
                continue

            class_images = images[mask]
            for i in range(min(len(class_images), max_samples - count)):
                img = class_images[i:i+1]
                sal = self.get_saliency(img, target_class)
                saliency_sum += sal
                count += 1
                if count >= max_samples:
                    break
            if count >= max_samples:
                break

        mean_saliency = saliency_sum / max(count, 1)
        return mean_saliency

    def select_salient_pixels(self, saliency_map, top_k=20):
        """Pick top-k most salient pixel locations."""
        flat = saliency_map.flatten()
        _, indices = flat.topk(top_k)
        rows = indices // 28
        cols = indices % 28
        return list(zip(rows.tolist(), cols.tolist()))

    # ── Single-pixel sweep (기존) ──

    def pixel_sweep(self, base_image, salient_pixels, num_values=256):
        """
        Independently sweep each salient pixel through [0, 1].
        """
        images = []
        soft_labels = []
        hard_labels = []
        values = torch.linspace(0, 1, num_values)

        self.teacher.eval()
        with torch.no_grad():
            for (r, c) in salient_pixels:
                for val in values:
                    img = base_image.clone()
                    img[0, 0, r, c] = val.item()
                    logits, _ = self.teacher(img.to(self.device))
                    prob = F.softmax(logits, dim=1)
                    images.append(img.squeeze(0).cpu())
                    soft_labels.append(prob.squeeze(0).cpu())
                    hard_labels.append(prob.argmax().item())

        return {
            "images": torch.stack(images),
            "soft_labels": torch.stack(soft_labels),
            "hard_labels": torch.tensor(hard_labels),
        }

    # ── [NEW] 2-pixel combo sweep ──

    def pixel_sweep_combo(self, base_image, salient_pixels, num_values=16):
        """
        Sweep all pairs of top salient pixels in a grid.
        For top-k pixels, generates C(k,2) * num_values^2 samples.

        Args:
            base_image: [1, 1, 28, 28]
            salient_pixels: list of (r, c) — will use ALL pairs
            num_values: values per pixel (16 -> 16x16=256 per pair)
        """
        images = []
        soft_labels = []
        hard_labels = []
        values = torch.linspace(0, 1, num_values)

        pairs = list(combinations(range(len(salient_pixels)), 2))
        print(f"    Combo sweep: {len(pairs)} pairs × {num_values}² = {len(pairs) * num_values**2} samples")

        self.teacher.eval()
        with torch.no_grad():
            for (i, j) in tqdm(pairs, desc="    Combo sweep"):
                r1, c1 = salient_pixels[i]
                r2, c2 = salient_pixels[j]
                for v1 in values:
                    for v2 in values:
                        img = base_image.clone()
                        img[0, 0, r1, c1] = v1.item()
                        img[0, 0, r2, c2] = v2.item()
                        logits, _ = self.teacher(img.to(self.device))
                        prob = F.softmax(logits, dim=1)
                        images.append(img.squeeze(0).cpu())
                        soft_labels.append(prob.squeeze(0).cpu())
                        hard_labels.append(prob.argmax().item())

        return {
            "images": torch.stack(images),
            "soft_labels": torch.stack(soft_labels),
            "hard_labels": torch.tensor(hard_labels),
        }

    # ── [NEW] Multi-seed sweep ──

    def multi_seed_sweep(self, seed_images, salient_pixels,
                         single_num_values=256,
                         combo_top_k=5, combo_num_values=16):
        """
        Run both single-pixel and 2-pixel combo sweep across multiple seeds.

        Args:
            seed_images: list of [1, 1, 28, 28] tensors
            salient_pixels: list of (r, c)
            single_num_values: values for single-pixel sweep
            combo_top_k: use top-k pixels for combo sweep (subset of salient_pixels)
            combo_num_values: values per pixel in combo sweep
        """
        all_images = []
        all_soft = []
        all_hard = []

        combo_pixels = salient_pixels[:combo_top_k]

        for s_idx, seed in enumerate(seed_images):
            print(f"  Seed {s_idx+1}/{len(seed_images)}")

            # Single-pixel sweep (all salient pixels)
            single = self.pixel_sweep(seed, salient_pixels, num_values=single_num_values)
            all_images.append(single["images"])
            all_soft.append(single["soft_labels"])
            all_hard.append(single["hard_labels"])

            # 2-pixel combo sweep (top-k subset only)
            combo = self.pixel_sweep_combo(seed, combo_pixels, num_values=combo_num_values)
            all_images.append(combo["images"])
            all_soft.append(combo["soft_labels"])
            all_hard.append(combo["hard_labels"])

        return {
            "images": torch.cat(all_images, dim=0),
            "soft_labels": torch.cat(all_soft, dim=0),
            "hard_labels": torch.cat(all_hard, dim=0),
        }
        
    # ── [NEW] Exhaustive k-pixel sweep (진짜 전수조사) ──

    def pixel_sweep_exhaustive(self, base_image, salient_pixels_k, num_values=256, batch_size=512):
        """
        k개 픽셀의 모든 조합을 전수조사.
        num_values^k 개의 이미지를 생성하고 teacher output을 기록.

        Args:
            base_image: [1, 1, 28, 28] seed image
            salient_pixels_k: list of (r, c), length = k
            num_values: per-pixel resolution (256 = full 8-bit, 32 = coarse)
            batch_size: GPU batch size for teacher inference

        Returns:
            dict with images, soft_labels, hard_labels
        """
        import itertools

        k = len(salient_pixels_k)
        total = num_values ** k
        values = torch.linspace(0, 1, num_values)

        print(f"    Exhaustive sweep: k={k}, {num_values} values/pixel, "
              f"total={total:,} combinations")

        # Pre-generate all value combinations as a tensor
        # Each row is one combination of k values
        grids = torch.meshgrid(*[values for _ in range(k)], indexing='ij')
        combos = torch.stack([g.flatten() for g in grids], dim=1)  # [total, k]

        all_images = []
        all_soft = []
        all_hard = []

        self.teacher.eval()
        with torch.no_grad():
            for start in tqdm(range(0, total, batch_size),
                              desc=f"    Exhaustive k={k}",
                              total=(total + batch_size - 1) // batch_size):
                end = min(start + batch_size, total)
                batch_combos = combos[start:end]  # [B, k]
                B = len(batch_combos)

                # Create batch of images
                imgs = base_image.expand(B, -1, -1, -1).clone()  # [B, 1, 28, 28]
                for p_idx, (r, c) in enumerate(salient_pixels_k):
                    imgs[:, 0, r, c] = batch_combos[:, p_idx]

                # Teacher inference
                logits, _ = self.teacher(imgs.to(self.device))
                probs = F.softmax(logits, dim=1)

                all_images.append(imgs.cpu())
                all_soft.append(probs.cpu())
                all_hard.append(probs.argmax(dim=1).cpu())

        return {
            "images": torch.cat(all_images, dim=0),
            "soft_labels": torch.cat(all_soft, dim=0),
            "hard_labels": torch.cat(all_hard, dim=0),
        }
    
    


# ============================================================
# [NEW] Part 2: Entropy Filtering
# ============================================================

def entropy_filter(dataset, low_percentile=5, high_percentile=95):
    """
    Remove off-manifold samples based on teacher output entropy.

    - Too low entropy (overconfident) → likely trivial/off-manifold
    - Too high entropy (near uniform) → teacher is confused, off-manifold

    Keep samples in the middle range.

    Args:
        dataset: dict with images, soft_labels, hard_labels
        low_percentile: remove below this percentile of entropy
        high_percentile: remove above this percentile

    Returns:
        filtered dataset dict + stats
    """
    probs = dataset["soft_labels"]  # [N, 10]
    # H(p) = -sum(p * log(p))
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum(dim=1)  # [N]

    low_thresh = torch.quantile(entropy, low_percentile / 100.0)
    high_thresh = torch.quantile(entropy, high_percentile / 100.0)

    mask = (entropy >= low_thresh) & (entropy <= high_thresh)

    filtered = {
        "images": dataset["images"][mask],
        "soft_labels": dataset["soft_labels"][mask],
        "hard_labels": dataset["hard_labels"][mask],
    }

    stats = {
        "original_size": len(entropy),
        "filtered_size": mask.sum().item(),
        "removed": (~mask).sum().item(),
        "entropy_low_thresh": low_thresh.item(),
        "entropy_high_thresh": high_thresh.item(),
        "entropy_mean": entropy.mean().item(),
        "entropy_std": entropy.std().item(),
    }

    return filtered, stats


# ============================================================
# Part 3: Distiller (training phase)
# ============================================================

def manifold_kd_loss(logits_student, soft_labels, temperature):
    """KL divergence between student output and teacher's soft labels."""
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    soft_teacher = torch.pow(soft_labels, 1.0 / temperature)
    soft_teacher = soft_teacher / soft_teacher.sum(dim=1, keepdim=True)
    loss = F.kl_div(log_pred_student, soft_teacher, reduction="batchmean")
    loss *= temperature ** 2
    return loss


class ManifoldProbeKD(nn.Module):
    """
    Distiller for ManifoldProbe: trains student on synthetic sweep data.
    """

    def __init__(self, student, temperature=4.0, ce_weight=0.1, kd_weight=0.9):
        super().__init__()
        self.student = student
        self.temperature = temperature
        self.ce_weight = ce_weight
        self.kd_weight = kd_weight

    def forward(self, image, soft_label, hard_label):
        logits_student, feats = self.student(image)

        loss_ce = self.ce_weight * F.cross_entropy(logits_student, hard_label)
        loss_kd = self.kd_weight * manifold_kd_loss(
            logits_student, soft_label, self.temperature
        )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
    