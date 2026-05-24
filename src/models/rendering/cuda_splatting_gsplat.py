from math import sqrt
from typing import Optional

import torch


def _to_pixel_intrinsics(intrinsics: torch.Tensor, height: int, width: int) -> torch.Tensor:
    intrinsics_px = intrinsics.clone()
    intrinsics_px[..., 0, :] *= width
    intrinsics_px[..., 1, :] *= height
    return intrinsics_px


def render_cuda_gsplat(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    image_shape: tuple[int, int],
    background_color: torch.Tensor,
    gaussian_means: torch.Tensor,
    gaussian_covariances: torch.Tensor,
    gaussian_scales: Optional[torch.Tensor] = None,
    gaussian_rotations: Optional[torch.Tensor] = None,
    gaussian_sh_coefficients: Optional[torch.Tensor] = None,
    gaussian_opacities: Optional[torch.Tensor] = None,
    use_sh: bool = True,
    cam_rot_delta: Optional[torch.Tensor] = None,
    cam_trans_delta: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from gsplat import rasterization as gsplat_rasterize
    except Exception:
        total_views = extrinsics.shape[0]
        h, w = image_shape
        device = extrinsics.device
        fallback_color = background_color[:, :, None, None].expand(total_views, 3, h, w).contiguous().to(device)
        fallback_depth = torch.ones(total_views, h, w, device=device, dtype=extrinsics.dtype)
        return fallback_color, fallback_depth

    device = extrinsics.device
    total_views = extrinsics.shape[0]
    h, w = image_shape
    all_colors = []
    all_depths = []

    for view_idx in range(total_views):
        viewmats_w2c = extrinsics[view_idx].float().inverse()
        k_px = _to_pixel_intrinsics(intrinsics[view_idx].float(), h, w)
        means = gaussian_means[view_idx].float().contiguous() if gaussian_means.dim() > 2 else gaussian_means.float().contiguous()
        covars = gaussian_covariances[view_idx].float().contiguous() if gaussian_covariances.dim() > 2 else gaussian_covariances.float().contiguous()
        opacities = gaussian_opacities[view_idx].float().flatten().contiguous()

        if gaussian_scales is not None and gaussian_rotations is not None:
            scales = gaussian_scales[view_idx].float().contiguous() if gaussian_scales.dim() > 2 else gaussian_scales.float().contiguous()
            quats = gaussian_rotations[view_idx].float().contiguous() if gaussian_rotations.dim() > 2 else gaussian_rotations.float().contiguous()
        else:
            scales = None
            quats = None

        if use_sh and gaussian_sh_coefficients is not None:
            sh = gaussian_sh_coefficients[view_idx] if gaussian_sh_coefficients.dim() > 2 else gaussian_sh_coefficients
            colors = sh.float().permute(0, 2, 1).contiguous()
            sh_degree = int(sqrt(colors.shape[-2])) - 1
        else:
            colors = torch.ones(means.shape[0], 3, device=device, dtype=torch.float32) * 0.5
            sh_degree = None

        rendered_colors, rendered_alphas, info = gsplat_rasterize(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=viewmats_w2c.unsqueeze(0).contiguous(),
            Ks=k_px.unsqueeze(0).contiguous(),
            width=int(w), height=int(h),
            sh_degree=sh_degree,
            near_plane=1e-10,
            far_plane=float(far[view_idx]) if far.dim() > 0 else float(far),
            render_mode="RGB+D", packed=False,
            backgrounds=background_color[view_idx].unsqueeze(0) if background_color.dim() == 2 else background_color.unsqueeze(0),
            radius_clip=0.1, covars=covars, rasterize_mode="classic",
        )

        rendered_rgbd = rendered_colors[0] if rendered_colors.dim() == 4 else rendered_colors
        if rendered_rgbd.shape[-1] >= 4:
            rgb = rendered_rgbd[..., :3].permute(2, 0, 1).contiguous().clamp(0.0, 1.0)
            depth = rendered_rgbd[..., 3].contiguous()
        else:
            rgb = rendered_rgbd[..., :3].permute(2, 0, 1).contiguous().clamp(0.0, 1.0)
            depth = torch.ones(h, w, device=device, dtype=rgb.dtype)
            if isinstance(info, dict) and "depths" in info:
                info_depths = info["depths"]
                if info_depths.dim() == 3 and info_depths.shape[-2:] == (h, w):
                    depth = info_depths[0].contiguous()
                elif info_depths.dim() == 2 and info_depths.shape == (h, w):
                    depth = info_depths.contiguous()

        all_colors.append(rgb)
        all_depths.append(depth)

    return torch.stack(all_colors), torch.stack(all_depths)
