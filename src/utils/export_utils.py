from pathlib import Path
from typing import Union
import numpy as np
import torch
from PIL import Image

from .gaussians_types import Gaussians


def export_points_to_ply(points: torch.Tensor, output_path: Union[str, Path], batch_idx: int = 0, view_idx: int = 0) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pts = points[batch_idx, view_idx].detach().cpu().reshape(-1, points.shape[-1]).numpy()
    if pts.shape[1] > 3:
        pts = pts[:, :3]
    with output_path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x} {y} {z}\n")


def _as_numpy_1d(tensor: torch.Tensor, batch_idx: int, fallback_value: float, length: int) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        value = tensor[batch_idx].detach().cpu().reshape(-1).numpy()
        if value.shape[0] == length:
            return value
    return np.full((length,), fallback_value, dtype=np.float32)


def _as_numpy_2d(tensor: torch.Tensor, batch_idx: int, fallback_value: float, shape: tuple[int, int]) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        value = tensor[batch_idx].detach().cpu().reshape(-1, shape[1]).numpy()
        if value.shape == shape:
            return value
    return np.full(shape, fallback_value, dtype=np.float32)


def export_gaussians_to_ply(gaussians: Gaussians, output_path: Union[str, Path], batch_idx: int = 0) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(getattr(gaussians, "means", None), torch.Tensor):
        return

    means = gaussians.means[batch_idx].detach().cpu().reshape(-1, 3).numpy()
    num_points = means.shape[0]
    opacities = _as_numpy_1d(getattr(gaussians, "opacities", None), batch_idx, 1.0, num_points)
    scales = _as_numpy_2d(getattr(gaussians, "scales", None), batch_idx, 0.01, (num_points, 3))
    rotations = _as_numpy_2d(getattr(gaussians, "rotations", None), batch_idx, 0.0, (num_points, 4))
    rotations[:, -1] = np.where(np.linalg.norm(rotations, axis=1) > 0, rotations[:, -1], 1.0)
    harmonics = getattr(gaussians, "harmonics", None)
    if isinstance(harmonics, torch.Tensor):
        harmonic_values = harmonics[batch_idx].detach().cpu().numpy()
        if harmonic_values.ndim >= 3 and harmonic_values.shape[0] == num_points:
            colors = ((harmonic_values[:, :, 0] + 0.5).clip(0, 1) * 255).astype(np.uint8)
        else:
            colors = np.full((num_points, 3), 127, dtype=np.uint8)
            harmonic_values = np.zeros((num_points, 3, 1))
    else:
        colors = np.full((num_points, 3), 127, dtype=np.uint8)
        harmonic_values = np.zeros((num_points, 3, 1))  # fallback: DC only

    semantic_labels = _as_numpy_1d(getattr(gaussians, "semantic_labels", None), batch_idx, 0, num_points).astype(np.int32)
    instance_labels = _as_numpy_1d(getattr(gaussians, "instance_labels", None), batch_idx, 0, num_points).astype(np.int32)
    seg_query_class_logits = getattr(gaussians, "seg_query_class_logits", None)
    qc_values = None
    if isinstance(seg_query_class_logits, torch.Tensor):
        qc_np = seg_query_class_logits[batch_idx].detach().cpu().reshape(num_points, -1).numpy()
        if qc_np.shape[0] == num_points:
            qc_values = qc_np.astype(np.float32)

    from plyfile import PlyData, PlyElement
    from scipy.spatial.transform import Rotation as R
    from einops import rearrange

    # Exact copy of AnySplat ply_export.py — line-for-line identical logic
    rots_tensor = torch.from_numpy(rotations).float()
    rots_np = R.from_quat(rots_tensor.numpy()).as_matrix()
    rots_np = R.from_matrix(rots_np).as_quat()
    x, y, z, w = rearrange(rots_np, "g xyzw -> xyzw g")
    rots_np = np.stack((w, x, y, z), axis=-1)

    f_dc = harmonic_values[..., 0].astype(np.float32)

    def construct_list_of_attributes():
        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(3):
            attrs.append(f"f_dc_{i}")
        attrs.append("opacity")
        for i in range(3):
            attrs.append(f"scale_{i}")
        for i in range(4):
            attrs.append(f"rot_{i}")
        return attrs

    dtype_full = [(attr, "f4") for attr in construct_list_of_attributes()]
    elements = np.empty(num_points, dtype=dtype_full)
    attributes = np.concatenate([
        means.astype(np.float32),
        np.zeros_like(means).astype(np.float32),
        f_dc,
        opacities[..., None].astype(np.float32),
        np.log(np.maximum(scales, 1e-10)).astype(np.float32),
        rots_np.astype(np.float32),
    ], axis=1)
    elements[:] = list(map(tuple, attributes))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(output_path))


def _save_gray_image(array: np.ndarray, output_path: Path) -> None:
    Image.fromarray(array.astype(np.uint8), mode="L").save(output_path)


def export_depth_map(depth: torch.Tensor, output_path: Union[str, Path], batch_idx: int = 0, view_idx: int = 0) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth_map = depth[batch_idx, view_idx].detach().float().cpu().numpy()
    depth_map = np.squeeze(depth_map)
    finite_mask = np.isfinite(depth_map)
    if not finite_mask.any():
        _save_gray_image(np.zeros_like(depth_map, dtype=np.uint8), output_path)
        return
    finite_values = depth_map[finite_mask]
    lo, hi = np.percentile(finite_values, [2, 98])
    if hi <= lo:
        lo, hi = float(finite_values.min()), float(finite_values.max())
    if hi > lo:
        depth_map = np.clip((depth_map - lo) / (hi - lo), 0, 1)
    else:
        depth_map = np.zeros_like(depth_map)
    depth_map[~finite_mask] = 0
    _save_gray_image(depth_map * 255, output_path)


def export_rendered_image(image: torch.Tensor, output_path: Union[str, Path], batch_idx: int = 0, view_idx: int = 0) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = image[batch_idx, view_idx].detach().cpu().permute(1, 2, 0).numpy()
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8), mode="RGB").save(output_path)


def export_input_views(images: torch.Tensor, output_dir: Union[str, Path], prefix: str = "sample", batch_idx: int = 0) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(images, torch.Tensor) or images.dim() != 5:
        return
    for view_idx in range(images.shape[1]):
        export_rendered_image(images, output_dir / f"{prefix}_input_view{view_idx}.png", batch_idx, view_idx)


def export_segmentation_masks(masks: torch.Tensor, output_path: Union[str, Path], batch_idx: int = 0, view_idx: int = 0) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if masks.dim() == 5:
        selected_masks = masks[batch_idx, view_idx].detach()
    elif masks.dim() == 4:
        selected_masks = masks[batch_idx].detach()
    else:
        return
    if selected_masks.numel() == 0 or selected_masks.shape[0] == 0:
        height = int(masks.shape[-2]) if masks.dim() >= 2 else 1
        width = int(masks.shape[-1]) if masks.dim() >= 1 else 1
        seg_map = np.zeros((height, width), dtype=np.uint8)
    elif selected_masks.shape[0] == 1:
        seg_map = (selected_masks[0] > 0.3).cpu().numpy().astype(np.uint8)
    else:
        max_values, max_indices = selected_masks.max(dim=0)
        seg_map_tensor = torch.where(max_values > 0.3, max_indices + 1, torch.zeros_like(max_indices))
        seg_map = seg_map_tensor.cpu().numpy().astype(np.uint8)
    _save_colored_segmentation(seg_map, output_path)


def _save_colored_segmentation(seg_map: np.ndarray, output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    palette = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], dtype=np.uint8)
    colored = palette[seg_map % len(palette)]
    Image.fromarray(colored, mode="RGB").save(output_path)


def _has_nonempty_masks(masks: torch.Tensor) -> bool:
    return isinstance(masks, torch.Tensor) and masks.dim() >= 4 and masks.numel() > 0 and masks.shape[-3] > 0


def export_render_qc_logits(render_qc_logits, output_path: Union[str, Path], view_idx: int = 0) -> bool:
    if isinstance(render_qc_logits, list):
        render_qc_logits = next((item for item in render_qc_logits if isinstance(item, torch.Tensor)), None)
    if not isinstance(render_qc_logits, torch.Tensor):
        return False
    qc_logits = render_qc_logits.detach()
    if qc_logits.dim() == 6:
        qc_logits = qc_logits[0]
    if qc_logits.dim() == 5:
        selected = qc_logits[min(view_idx, qc_logits.shape[0] - 1)]
    elif qc_logits.dim() == 4:
        selected = qc_logits
    else:
        return False
    if selected.numel() == 0 or selected.shape[0] == 0:
        return False
    # Per-pixel: assign to the query with highest fg - bg score (multi-instance)
    # selected: [Q, C, H, W], C=0 is bg, C=1 is fg
    fg_scores = selected[:, 1, :, :].float() - selected[:, 0, :, :].float()  # [Q, H, W]
    best_query = fg_scores.argmax(dim=0)  # [H, W], 0..Q-1
    best_score = fg_scores.max(dim=0).values  # [H, W]
    seg_map = torch.where(best_score > 0, best_query + 1, torch.zeros_like(best_query))
    seg_map = seg_map.cpu().numpy().astype(np.uint8)
    _save_colored_segmentation(seg_map, output_path)
    return True


def export_render_qc_foreground_heatmap(render_qc_logits, output_path: Union[str, Path], view_idx: int = 0) -> bool:
    if isinstance(render_qc_logits, list):
        render_qc_logits = next((item for item in render_qc_logits if isinstance(item, torch.Tensor)), None)
    if not isinstance(render_qc_logits, torch.Tensor):
        return False
    qc_logits = render_qc_logits.detach()
    if qc_logits.dim() == 6:
        qc_logits = qc_logits[0]
    if qc_logits.dim() == 5:
        selected = qc_logits[min(view_idx, qc_logits.shape[0] - 1)]
    elif qc_logits.dim() == 4:
        selected = qc_logits
    else:
        return False
    if selected.numel() == 0 or selected.shape[0] == 0:
        return False
    class_logits = torch.amax(selected.float(), dim=0)
    if class_logits.shape[0] > 1:
        fg_prob = class_logits.softmax(dim=0)[1]
    else:
        fg_prob = class_logits.sigmoid().squeeze(0)
    fg_map = fg_prob.float().clamp(0, 1).cpu().numpy()
    _save_gray_image(fg_map * 255, Path(output_path))
    return True


def export_all_outputs(output, output_dir: Union[str, Path], prefix: str = "output", batch_idx: int = 0) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if getattr(output, "gaussians", None) is not None:
        export_gaussians_to_ply(output.gaussians, output_dir / f"{prefix}_gaussians.ply", batch_idx)
    render_color = output.rendered_output.get("render_color") if isinstance(getattr(output, "rendered_output", None), dict) else None
    render_depth = output.rendered_output.get("render_depth") if isinstance(getattr(output, "rendered_output", None), dict) else None
    render_qc_logits = output.rendered_output.get("render_qc_logits") if isinstance(getattr(output, "rendered_output", None), dict) else None
    if render_color is not None:
        export_rendered_image(render_color, output_dir / f"{prefix}_render.png", batch_idx, 0)
    if render_depth is not None:
        export_depth_map(render_depth, output_dir / f"{prefix}_depth.png", batch_idx, 0)
    seg_masks = getattr(output.sam3_output, "seg_masks", None)
    raw_seg_masks = getattr(output.sam3_output, "raw_seg_masks_before_feedback", None)
    feedback_masks = getattr(output.sam3_output, "feedback_masks", None)
    if isinstance(raw_seg_masks, torch.Tensor):
        export_segmentation_masks(raw_seg_masks, output_dir / f"{prefix}_seg_raw_sam3.png", batch_idx, 0)
    if isinstance(feedback_masks, torch.Tensor):
        export_segmentation_masks(feedback_masks, output_dir / f"{prefix}_seg_feedback_3d.png", batch_idx, 0)
    if _has_nonempty_masks(seg_masks):
        export_segmentation_masks(seg_masks, output_dir / f"{prefix}_seg.png", batch_idx, 0)
    elif not export_render_qc_logits(render_qc_logits, output_dir / f"{prefix}_seg.png", 0) and isinstance(seg_masks, torch.Tensor):
        export_segmentation_masks(seg_masks, output_dir / f"{prefix}_seg.png", batch_idx, 0)
    export_render_qc_logits(render_qc_logits, output_dir / f"{prefix}_render_semantic.png", 0)
    export_render_qc_foreground_heatmap(render_qc_logits, output_dir / f"{prefix}_render_semantic_fg_prob.png", 0)
    backbone_output = getattr(output, "backbone_output", None)
    depth_info = getattr(backbone_output, "depth_dict", None)
    if isinstance(depth_info, dict):
        native_depth = depth_info.get("depth")
        native_pts3d = depth_info.get("pts3d")
        if isinstance(native_depth, torch.Tensor) and native_depth.dim() >= 4:
            export_depth_map(native_depth.squeeze(-1) if native_depth.shape[-1] == 1 else native_depth, output_dir / f"{prefix}_anysplat_depth.png", batch_idx, 0)
        if isinstance(native_pts3d, torch.Tensor) and native_pts3d.dim() == 5:
            export_points_to_ply(native_pts3d, output_dir / f"{prefix}_anysplat_pts3d.ply", batch_idx, 0)
