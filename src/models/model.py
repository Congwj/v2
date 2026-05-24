from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anysplat_src.anysplat import EncoderAnySplat, EncoderAnySplatCfg, OpacityMappingCfg
from .anysplat_src.common.gaussian_adapter import GaussianAdapterCfg
from .segmentation.sam3_segmenter import SAM3Segmenter, SAM3SegmenterOutput
from .rendering.gaussian_renderer import SplattingCUDA
from ..utils.gaussians_types import Gaussians


@dataclass
class FusedSystemOutput:
    gaussians: Gaussians
    sam3_output: any
    rendered_output: Dict
    backbone_output: Optional[any] = None
    target_rendered_output: Optional[Dict] = None


class VoteHead(nn.Module):
    """GNeSF GeneralRenderingNetwork adapted for Gaussian splatting.
    base_fc → vis_fc → vis_fc2 → sem_fc, with independent blending per query.
    Reference: GNeSF-3D/models/mlp_network.py GeneralRenderingNetwork (lines 26-145)."""
    def __init__(self, num_views=3, num_classes=1, hidden_dim=32):
        super().__init__()
        C = num_classes + 1
        V = num_views
        H = hidden_dim
        self.H = H
        # base_fc: shared per-view feature extractor (GNeSF line 50-54)
        self.base_fc = nn.Sequential(
            nn.Linear(C, H), nn.ELU(),
            nn.Linear(H, H), nn.ELU(),
        )
        # vis_fc: residual + visibility (GNeSF line 56-60)
        self.vis_fc = nn.Sequential(
            nn.Linear(H, H), nn.ELU(),
            nn.Linear(H, H + 1), nn.ELU(),
        )
        # vis_fc2: refine visibility (GNeSF line 62-66)
        self.vis_fc2 = nn.Sequential(
            nn.Linear(H, H), nn.ELU(),
            nn.Linear(H, 1), nn.Sigmoid(),
        )
        # sem_fc: features + visibility → blending logit (GNeSF line 75-79)
        self.sem_fc = nn.Sequential(
            nn.Linear(H + 1, H // 2), nn.ELU(),
            nn.Linear(H // 2, H // 4), nn.ELU(),
            nn.Linear(H // 4, 1),
        )
        # Zero-init last layer: untrained VoteHead → uniform blending ≈ mean
        nn.init.zeros_(self.sem_fc[-1].weight)
        nn.init.zeros_(self.sem_fc[-1].bias)

    def forward(self, per_view_logits):
        # per_view_logits: [BN, V, Q, C]
        BN, V, Q, C = per_view_logits.shape
        x = per_view_logits.reshape(BN * Q, V, C)  # [BN*Q, V, C]
        x = self.base_fc(x)  # [BN*Q, V, H]
        x_vis = self.vis_fc(x)  # [BN*Q, V, H+1]
        x_res, vis = torch.split(x_vis, [self.H, 1], dim=-1)
        vis = torch.sigmoid(vis)
        x = x + x_res
        vis = self.vis_fc2(x * vis)
        sem_input = torch.cat([x, vis], dim=-1)
        blend_logits = self.sem_fc(sem_input).squeeze(-1)
        weights = F.softmax(blend_logits, dim=-1)
        return weights.reshape(BN, Q, V)


class FusedSystem(nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        num_classes: int = 1,
        num_queries: int = 50,
        use_sam3_api: bool = True,
        sam3_checkpoint: str = "src/pretrained_weights/sam3.pt",
        anysplat_checkpoint: Optional[str] = None,
        freeze_sam3: bool = True,
        freeze_sam3_backbone: bool = True,
        train_sam3_decoder: bool = False,
        freeze_gaussian_head: bool = True,
        freeze_backbone: bool = True,
        feature_dim: int = 2048,
        sh_degree: int = 2,
        num_points: int = 5000,
        gaussian_scale_min: float = 0.01,
        gaussian_scale_max: float = 0.3,
        voxel_size: float = 0.02,
        voxelize: bool = False,
        use_vote_head: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.use_vote_head = use_vote_head

        gaussian_cfg = GaussianAdapterCfg(
            gaussian_scale_min=gaussian_scale_min,
            gaussian_scale_max=gaussian_scale_max,
            sh_degree=sh_degree,
        )
        opacity_cfg = OpacityMappingCfg(initial=3.0, final=3.0, warm_up=1)
        encoder_cfg = EncoderAnySplatCfg(
            name="anysplat",
            anchor_feat_dim=feature_dim,
            voxel_size=voxel_size,
            n_offsets=1,
            d_feature=feature_dim,
            add_view=False,
            num_monocular_samples=num_points,
            backbone=None,
            visualizer=None,
            gaussian_adapter=gaussian_cfg,
            apply_bounds_shim=False,
            opacity_mapping=opacity_cfg,
            gaussians_per_pixel=1,
            num_surfaces=1,
            gs_params_head_type="dpt_gs",
            pretrained_weights=anysplat_checkpoint or "",
            pose_free=True,
            pred_pose=True,
            gt_pose_to_pts=False,
            gs_prune=False,
            opacity_threshold=0.001,
            gs_keep_ratio=1.0,
            pred_head_type="depth",
            freeze_backbone=freeze_backbone,
            freeze_module="all" if freeze_backbone else "None",
            distill=False,
            render_conf=False,
            opacity_conf=False,
            conf_threshold=0.1,
            intermediate_layer_idx=None,
            voxelize=voxelize,
        )
        self.anysplat_encoder = EncoderAnySplat(encoder_cfg)
        if freeze_gaussian_head:
            for module in [self.anysplat_encoder.gaussian_param_head, self.anysplat_encoder.gaussian_adapter]:
                for param in module.parameters():
                    param.requires_grad = False
        self.vote_head = VoteHead(num_views=3, num_classes=num_classes) if use_vote_head else None
        if use_sam3_api:
            self.sam3_segmenter = SAM3Segmenter(
                num_classes=num_classes,
                num_queries=num_queries,
                sam3_checkpoint=sam3_checkpoint,
                freeze_sam3=freeze_sam3,
                freeze_sam3_backbone=freeze_sam3_backbone,
                train_sam3_decoder=train_sam3_decoder,
            )
        else:
            self.sam3_segmenter = None
        self.gaussian_renderer = SplattingCUDA()

    @property
    def device(self):
        return next(self.parameters()).device

    def _resolve_camera_context(self, encoder_output, intrinsics, extrinsics):
        pred_context_pose = getattr(encoder_output, "pred_context_pose", None)
        if not isinstance(pred_context_pose, dict):
            return intrinsics, extrinsics
        replica_intrinsics = pred_context_pose.get("intrinsic")
        replica_extrinsics = pred_context_pose.get("extrinsic")
        if not isinstance(replica_intrinsics, torch.Tensor) or not isinstance(replica_extrinsics, torch.Tensor):
            return intrinsics, extrinsics
        # pred_context_pose["extrinsic"] is already c2w (AnySplat internally inverts the raw w2c output)
        return replica_intrinsics, replica_extrinsics

    def forward(self, images, intrinsics, extrinsics, prompts=None,
                enable_query_class_logit_lift=True, pre_extracted_features=None,
                target_extrinsics_cam=None, target_pre_extracted_features=None):
        _, _, _, h, w = images.shape
        if self.training:
            torch.cuda.empty_cache()
        encoder_frozen = not any(p.requires_grad for p in self.anysplat_encoder.parameters())
        if self.training and not encoder_frozen:
            raise RuntimeError(f"Encoder has trainable params but should be fully frozen!")
        ctx = torch.no_grad() if encoder_frozen else torch.enable_grad()
        with ctx:
            encoder_output = self.anysplat_encoder(images, global_step=0, visualization_dump=None)
        gaussians = encoder_output.gaussians
        if self.training:
            torch.cuda.empty_cache()
        camera_intrinsics, camera_extrinsics = self._resolve_camera_context(
            encoder_output, intrinsics, extrinsics)
        if pre_extracted_features is not None:
            sam3_output = self._build_output_from_pre_extracted(pre_extracted_features, images.device)
        elif self.sam3_segmenter is not None:
            sam3_output = self.sam3_segmenter(images, prompts=prompts, feedback_masks=None, iteration=0)
        else:
            raise RuntimeError(
                "SAM3 not loaded and no pre_extracted_features provided.")

        if enable_query_class_logit_lift:
            gaussians = self._assign_semantics_cross_view(
                gaussians, sam3_output, camera_intrinsics, camera_extrinsics, h, w)
            if self.use_vote_head and self.vote_head is not None and gaussians.seg_query_class_logits is not None:
                per_view = gaussians.seg_per_view_logits  # [B, N, V, Q, C]
                B2, N2, V2, Q2, C2 = per_view.shape
                vote_input = per_view.reshape(B2 * N2, V2, Q2, C2)  # [B*N, V, Q, C]
                vote_input.requires_grad_(True)
                weights = self.vote_head(vote_input)  # [B*N, Q, V]
                weights = weights.reshape(B2, N2, Q2, V2)  # [B, N, Q, V]
                blended = (weights.unsqueeze(-1) * per_view.permute(0, 1, 3, 2, 4)).sum(dim=3)  # [B, N, Q, C]
                gaussians.seg_query_class_logits = blended
                del gaussians.seg_per_view_logits, per_view  # free per-view tensor after blending

        rendered_output = self.gaussian_renderer(
            gaussians, camera_extrinsics, camera_intrinsics, (h, w),
            render_color=True, render_qc_logits=(not self.training))
        rendered_output["used_color_fallback"] = rendered_output.get("render_color") is None
        rendered_output["used_depth_fallback"] = rendered_output.get("render_depth") is None
        if rendered_output["used_color_fallback"]:
            rendered_output["render_color"] = images.clone()
        if rendered_output["used_depth_fallback"]:
            rendered_output["render_depth"] = torch.ones(
                images.shape[0], images.shape[1], h, w, device=images.device, dtype=images.dtype)

        target_rendered = None
        if self.training and target_extrinsics_cam is not None and target_pre_extracted_features is not None:
            C_s = extrinsics
            C_t = target_extrinsics_cam.to(images.device)
            A_s = camera_extrinsics
            A_t = A_s @ torch.linalg.inv(C_s) @ C_t
            target_rendered = self.gaussian_renderer(
                gaussians, A_t, camera_intrinsics, (h, w),
                render_color=False, render_qc_logits=True)

        return FusedSystemOutput(
            gaussians=gaussians,
            sam3_output=sam3_output,
            rendered_output=rendered_output,
            backbone_output=encoder_output,
            target_rendered_output=target_rendered,
        )

    def _assign_semantics_cross_view(self, gaussians, sam3_output, intrinsics, extrinsics, H, W):
        from einops import rearrange
        sam3_qc = getattr(sam3_output, "query_class_logits", None)
        if sam3_qc is None or not isinstance(sam3_qc, torch.Tensor):
            return gaussians
        B, V, Q, C = sam3_qc.shape[:4]
        total_pts = gaussians.means.shape[1]
        per_view = torch.zeros(B, V, total_pts, Q, C, device=gaussians.means.device, dtype=sam3_qc.dtype)
        for view_idx in range(V):
            k = intrinsics[:, view_idx].clone()
            e_c2w = extrinsics[:, view_idx]
            e_w2c = torch.linalg.inv(e_c2w)
            k[:, 0, :] *= W
            k[:, 1, :] *= H
            p = torch.matmul(k, torch.cat([e_w2c[:, :3, :3], e_w2c[:, :3, 3:4]], dim=-1))
            means_homo = torch.cat([gaussians.means, torch.ones_like(gaussians.means[..., :1])], dim=-1)
            points_2d = torch.matmul(means_homo, p.transpose(1, 2))
            z = torch.clamp(points_2d[..., 2:3], min=1e-6)
            points_2d = points_2d[..., :2] / z
            coords = torch.stack([
                2 * points_2d[..., 0] / max(W - 1, 1) - 1,
                2 * points_2d[..., 1] / max(H - 1, 1) - 1,
            ], dim=-1)
            view_logits = rearrange(sam3_qc[:, view_idx], "b q c h w -> b (q c) h w")
            sampled = F.grid_sample(view_logits, coords.unsqueeze(2),
                                    mode="nearest", padding_mode="zeros", align_corners=True).squeeze(-1)
            per_view[:, view_idx] = rearrange(sampled, "b (q c) n -> b n q c", q=Q, c=C)
        gaussians.seg_per_view_logits = per_view.permute(0, 2, 1, 3, 4)  # [B, N, V, Q, C]
        gaussians.seg_query_class_logits = per_view.mean(dim=1)  # [B, N, Q, C], used by renderer
        return gaussians

    def _build_output_from_pre_extracted(self, pre_data, device):
        qc = pre_data["query_class_logits"].to(device)
        V = qc.shape[1]
        masks = pre_data.get("seg_masks")
        scores = pre_data.get("query_scores")
        return SAM3SegmenterOutput(
            seg_masks=masks.to(device) if masks is not None else torch.zeros(
                1, V, qc.shape[2], *qc.shape[-2:], device=device),
            seg_logits=qc[:, :, :, 1:2] if qc.dim() >= 4 else qc,
            query_class_logits=qc,
            query_scores=scores.to(device) if scores is not None else torch.ones(
                1, V, qc.shape[2], device=device),
        )
