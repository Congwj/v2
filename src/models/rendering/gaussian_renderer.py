import torch
import torch.nn as nn
from einops import rearrange, repeat

from ...utils.gaussians_types import Gaussians
from .cuda_splatting_gsplat import render_cuda_gsplat as render_cuda


class SplattingCUDA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.near = 0.1
        self.far = 100.0
        self.scale_factor = 1 / self.near
        self.register_buffer("background_color", torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32), persistent=False)

    def forward(self, gaussians: Gaussians, extrinsics: torch.Tensor, intrinsics: torch.Tensor, image_shape: tuple[int, int], render_color: bool = True, render_feature: bool = False, render_id: bool = False, render_qc_logits: bool = False, cam_rot_delta: torch.Tensor | None = None, cam_trans_delta: torch.Tensor | None = None):
        b, v, _, _ = extrinsics.shape
        near = torch.full((b * v,), 1e-10, dtype=torch.float32, device=extrinsics.device)
        far = torch.full((b * v,), self.far, dtype=torch.float32, device=extrinsics.device)
        result = {"render_color": None, "render_depth": None, "render_qc_logits": None}
        if render_color:
            color, depth = render_cuda(
                rearrange(extrinsics, "b v i j -> (b v) i j").float(),
                rearrange(intrinsics, "b v i j -> (b v) i j").float(),
                near,
                far,
                image_shape,
                repeat(self.background_color, "c -> (b v) c", b=b, v=v),
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v).float(),
                repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v).float(),
                repeat(gaussians.scales, "b g xyz -> (b v) g xyz", v=v).float() if gaussians.scales is not None else None,
                repeat(gaussians.rotations, "b g xyzw -> (b v) g xyzw", v=v).float() if gaussians.rotations is not None else None,
                repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v).float() if gaussians.harmonics is not None else None,
                repeat(gaussians.opacities, "b g -> (b v) g", v=v).float(),
                cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i") if cam_rot_delta is not None else None,
                cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i") if cam_trans_delta is not None else None,
            )
            result["render_color"] = torch.clamp(rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v), 0.0, 1.0)
            result["render_depth"] = rearrange(depth, "(b v) h w -> b v h w", b=b, v=v)
        if render_qc_logits:
            width, height = image_shape[1], image_shape[0]
            all_query_class_logits = []
            seg_query_class_logits = gaussians.seg_query_class_logits
            if seg_query_class_logits is not None:
                print(f"[gsplat QC] seg_qc shape={seg_query_class_logits.shape}, min={seg_query_class_logits.min().item():.4f}, max={seg_query_class_logits.max().item():.4f}, mean={seg_query_class_logits.mean().item():.4f}")
                try:
                    from gsplat import rasterization
                except Exception:
                    rasterization = None
                print(f"[gsplat QC] rasterization_available={rasterization is not None}")
                for i in range(b):
                    means_i = gaussians.means[i]
                    covariances_i = gaussians.covariances[i]
                    opacities_i = gaussians.opacities[i]
                    ks = intrinsics[i].clone()
                    ks[:, 0, :] *= width
                    ks[:, 1, :] *= height
                    viewmats = extrinsics[i].inverse()
                    query_class_logits = seg_query_class_logits[i]
                    if rasterization is not None and query_class_logits is not None and query_class_logits.dim() == 3:
                        _, q, c = query_class_logits.shape
                        flat_logits = rearrange(query_class_logits, "n q c -> n (q c)")
                        rendered_qc_logits, _, _ = rasterization(means=means_i.float(), quats=None, scales=None, covars=covariances_i.float(), opacities=opacities_i.float(), colors=flat_logits.float(), viewmats=viewmats.float(), Ks=ks.float(), width=width, height=height, sh_degree=None, near_plane=1e-10, far_plane=self.far)
                        all_query_class_logits.append(rearrange(rendered_qc_logits, "n h w (q c) -> n q c h w", q=q, c=c))
                    else:
                        all_query_class_logits.append(None)
            result["render_qc_logits"] = all_query_class_logits
        if render_feature:
            raise NotImplementedError("Feature rendering not implemented")
        if render_id:
            raise NotImplementedError("ID rendering not implemented")
        return result
