"""
Evaluate per-gaussian semantic predictions against 3D point cloud GT.
Adapted from EmbodiedSplat's evaluation pipeline.
Usage:
    python scripts/evaluate_semantic.py --pred-dir outputs_semantic --gt-root dataset/scannet/test --output results/metrics.txt
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ----- ScanNet 20-class labels (matching EmbodiedSplat) -----
SCANNET20_CLASSES = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
    'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
]

UNKNOWN_ID = -1


def confusion_matrix(pred_ids, gt_ids, num_classes):
    """Calculate confusion matrix from predicted and GT label arrays."""
    assert pred_ids.shape == gt_ids.shape
    idxs = gt_ids != UNKNOWN_ID
    return np.bincount(
        pred_ids[idxs] * num_classes + gt_ids[idxs],
        minlength=num_classes ** 2
    ).reshape((num_classes, num_classes)).astype(np.ulonglong)


def compute_iou(confusion, label_id):
    tp = int(confusion[label_id, label_id])
    fp = int(confusion[label_id, :].sum()) - tp
    fn = int(confusion[:, label_id].sum()) - tp
    denom = tp + fp + fn
    if denom == 0:
        return float('nan'), 0, 0
    return tp / denom, tp, denom


def evaluate(pred_ids, gt_ids, class_names, verbose=True):
    """Compute per-class IoU and mean IoU."""
    N = len(class_names)
    cm = confusion_matrix(pred_ids, gt_ids, N)

    class_ious = {}
    valid_count = 0
    mean_iou = 0.0

    for i, name in enumerate(class_names):
        if (gt_ids == i).sum() == 0:
            continue
        iou, tp, denom = compute_iou(cm, i)
        class_ious[name] = (iou, tp, denom)
        mean_iou += iou
        valid_count += 1

    mean_iou /= max(valid_count, 1)
    if verbose:
        print(f"{'Class':<18s} {'IoU':>6s}   {'TP':>8s}  {'Denom':>8s}")
        print('-' * 44)
        for name in class_names:
            if name in class_ious:
                iou, tp, denom = class_ious[name]
                print(f'{name:<18s} {iou:>6.3f}   {tp:>8d}  {denom:>8d}')
        print(f'\nMean IoU: {mean_iou:.4f}')
    return mean_iou


def aggregate_gaussians_to_pointcloud(
    gaussian_means: torch.Tensor,       # [N, 3]
    gaussian_opacities: torch.Tensor,    # [N]
    gaussian_logits: torch.Tensor,       # [N, C]
    pc_coords: np.ndarray,               # [M, 3]
    k: int = 3,
) -> np.ndarray:
    """
    Aggregate per-gaussian semantic logits onto a point cloud using kNN + opacity weighting.
    Returns per-point predicted class indices [M].
    """
    device = gaussian_means.device
    pc_tensor = torch.from_numpy(pc_coords).float().to(device)

    # Compute pairwise distances in chunks to avoid OOM
    N = gaussian_means.shape[0]
    chunk_size = 50000
    pred_logits = torch.zeros(pc_coords.shape[0], gaussian_logits.shape[-1], device=device)

    for start in range(0, pc_coords.shape[0], chunk_size):
        end = min(start + chunk_size, pc_coords.shape[0])
        chunk = pc_tensor[start:end]  # [C, 3]

        # L2 distance: [chunk, N]
        dist = torch.cdist(chunk, gaussian_means)  # [C, N]

        # Top-k nearest gaussians for each point
        topk_dist, topk_idx = dist.topk(k, dim=-1, largest=False)  # [C, k]

        # Opacity-weighted average of logits, inverse-distance weighted
        weights = gaussian_opacities[topk_idx] / (topk_dist + 1e-6)  # [C, k]
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        weighted_logits = (gaussian_logits[topk_idx] * weights.unsqueeze(-1)).sum(dim=1)  # [C, num_classes]
        pred_logits[start:end] = weighted_logits

    return pred_logits.argmax(dim=-1).cpu().numpy()


def load_gt_pointcloud(npy_path: str):
    """Load ScanNet test point cloud. Returns (coords, gt_labels)."""
    pcd = np.load(npy_path)
    coords = pcd[:, :3].astype(np.float32)
    # Column 10 is the raw semantic label in ScanNet format
    gt_labels = pcd[:, 10].astype(np.int64)
    return coords, gt_labels


def map_gt_to_20class(gt_labels, label_info):
    """Map raw ScanNet labels to 20-class indices."""
    gt_20 = np.full_like(gt_labels, UNKNOWN_ID)
    for idx_20, label_data in label_info.items():
        name = label_data.get('name', '')
        if name in SCANNET20_CLASSES:
            raw_id = int(label_data.get('id', idx_20))
            gt_20[gt_labels == raw_id] = SCANNET20_CLASSES.index(name)
    return gt_20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str, required=True,
                        help='Directory with per-scene ckpt_gs.pt files')
    parser.add_argument('--gt-root', type=str, default='dataset/scannet/test',
                        help='Directory with per-scene .npy GT point clouds')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of nearest gaussians for aggregation')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classes in gaussian logits (C)')
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_root = Path(args.gt_root)

    if not pred_dir.exists():
        raise FileNotFoundError(f'Prediction directory not found: {pred_dir}')
    if not gt_root.exists():
        raise FileNotFoundError(f'GT directory not found: {gt_root}')

    all_preds = []
    all_gts = []
    scenes_evaluated = 0

    for scene_dir in tqdm(sorted(pred_dir.iterdir()), desc='Evaluating'):
        if not scene_dir.is_dir():
            continue
        scene_name = scene_dir.name

        # Load predicted gaussians
        gs_path = scene_dir / 'ckpt_gs.pt'
        if not gs_path.exists():
            print(f'WARNING: no ckpt_gs.pt in {scene_name}, skipping')
            continue

        gaussians = torch.load(gs_path, map_location='cpu', weights_only=False)

        # Extract means, opacities, and semantic logits
        if isinstance(gaussians, dict):
            means = gaussians.get('means')  # [1, N, 3] or [N, 3]
            opacities = gaussians.get('opacities')  # [1, N] or [N]
            logits = gaussians.get('seg_query_class_logits')  # [1, N, Q, C] or [N, Q, C]
            # If query dimension exists, take max over queries
            if logits is not None and logits.dim() == 4:
                logits = logits.amax(dim=2)  # [B, N, C] or [N, C]
        else:
            means = gaussians.means
            opacities = gaussians.opacities
            logits = getattr(gaussians, 'seg_query_class_logits', None)

        if means is None or logits is None:
            print(f'WARNING: missing means or seg_query_class_logits in {scene_name}, skipping')
            continue

        # Squeeze batch dim if present
        if means.dim() == 3:
            means = means.squeeze(0)
        if opacities.dim() == 2:
            opacities = opacities.squeeze(0)
        if logits.dim() == 3:
            logits = logits.squeeze(0)
        # If Q dim still present, mean over queries
        if logits.dim() == 3:
            logits = logits.mean(dim=1)

        # Load GT point cloud
        # ScanNet naming: scene0707_00 -> scene0707_00.npy or scene0707_00/scene0707_00.npy
        gt_path = gt_root / scene_name / f'{scene_name}.npy'
        if not gt_path.exists():
            gt_path = gt_root / f'{scene_name}.npy'
        if not gt_path.exists():
            print(f'WARNING: no GT file for {scene_name}, skipping')
            continue

        coords, gt_raw = load_gt_pointcloud(str(gt_path))

        # For 2-class (fg/bg) evaluation: just use raw labels directly
        # Since we only have foreground/background, treat class 0=bg, class >=1=fg
        if args.num_classes == 2:
            gt_labels = (gt_raw > 0).astype(np.int64)
            class_names = ['background', 'foreground']
        else:
            # For multi-class, need label mapping
            gt_labels = gt_raw.copy()
            class_names = SCANNET20_CLASSES

        # Filter unknown points
        valid = gt_labels != UNKNOWN_ID
        if not valid.any():
            continue

        coords_valid = coords[valid]
        gt_labels_valid = gt_labels[valid]

        # Aggregate gaussians to point cloud
        pred_labels = aggregate_gaussians_to_pointcloud(
            means.float(),
            opacities.float().clamp(min=0).squeeze(-1) if opacities.dim() > 1 else opacities.float().clamp(min=0),
            logits.float(),
            coords_valid,
            k=args.k,
        )

        all_preds.append(pred_labels)
        all_gts.append(gt_labels_valid)
        scenes_evaluated += 1

    if scenes_evaluated == 0:
        print('No scenes evaluated!')
        return

    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)
    evaluate(all_preds, all_gts, class_names)


if __name__ == '__main__':
    main()
