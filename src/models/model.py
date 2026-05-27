from dataclasses import dataclass
from typing import Dict, Optional, Union

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
    """Exact GNeSF GeneralRenderingNetwork adapted for Gaussian splatting.
    base_fc -> vis_fc -> vis_fc2 -> sem_fc, with masked_fill before softmax.
    Reference: GNeSF-3D/models/mlp_network.py GeneralRenderingNetwork (lines 26-145)."""
    def __init__(self, num_classes=1):
        super().__init__()
        C = num_classes + 1  # 2
        # base_fc: GNeSF line 50-54: Linear(in,64)->ELU->Linear(64,32)->ELU
        self.base_fc = nn.Sequential(
            nn.Linear(C * 3, 64), nn.ELU(),
            nn.Linear(64, 32), nn.ELU(),
        )
        # vis_fc: GNeSF line 56-60: Linear(32,32)->ELU->Linear(32,33)->ELU
        self.vis_fc = nn.Sequential(
            nn.Linear(32, 32), nn.ELU(),
            nn.Linear(32, 33), nn.ELU(),
        )
        # vis_fc2: GNeSF line 62-66: Linear(32,32)->ELU->Linear(32,1)->Sigmoid
        self.vis_fc2 = nn.Sequential(
            nn.Linear(32, 32), nn.ELU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        # sem_fc: GNeSF line 75-79: Linear(33,16)->ELU->Linear(16,8)->ELU->Linear(8,1)
        self.sem_fc = nn.Sequential(
            nn.Linear(33, 16), nn.ELU(),
            nn.Linear(16, 8), nn.ELU(),
            nn.Linear(8, 1),
        )
        # GNeSF: only sem_fc gets weights_init (GNeSF line 80)
        self.sem_fc.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

    def forward(self, per_view_logits):
        # per_view_logits: [B, N, V, C+1] where C+1=[logits(C), visibility(1)]
        # Exact GNeSF GeneralRenderingNetwork structure: no Q dimension.
        B, N, V, Cp1 = per_view_logits.shape
        C = Cp1 - 1
        x_raw = per_view_logits[..., :C]             # [B, N, V, C]
        vis_mask = per_view_logits[..., C:]           # [B, N, V, 1]

        # Flatten: [B*N, V, C]
        x_raw_flat = x_raw.reshape(B * N, V, C)
        vis_flat = vis_mask.reshape(B * N, V, 1)

        # GNeSF eq: anti-alias weight from visibility mask
        weight = vis_flat / (vis_flat.sum(dim=1, keepdim=True) + 1e-8)

        # GNeSF eq: fused mean and variance across views
        mean = (x_raw_flat * weight).sum(dim=1, keepdim=True)
        var = (weight * (x_raw_flat - mean) ** 2).sum(dim=1, keepdim=True)
        globalfeat = torch.cat([mean, var], dim=-1)  # [BN, 1, 2C]

        # GNeSF eq: base_fc input = [globalfeat, per-view features]
        x = torch.cat([globalfeat.expand(-1, V, -1), x_raw_flat], dim=-1)  # [BN, V, 3C]
        x = self.base_fc(x)  # [BN, V, 32]

        # GNeSF eq: vis_fc → residual + visibility
        x_vis = self.vis_fc(x * weight)  # [BN, V, 33]
        x_res, vis = torch.split(x_vis, [32, 1], dim=-1)
        vis = torch.sigmoid(vis) * vis_flat
        x = x + x_res

        # GNeSF eq: vis_fc2
        vis = self.vis_fc2(x * vis) * vis_flat

        # GNeSF eq: sem_fc → masked_fill → softmax → per-view blending weight
        sem_input = torch.cat([x, vis], dim=-1)  # [BN, V, 33]
        blend_logits = self.sem_fc(sem_input).squeeze(-1)  # [BN, V]
        blend_logits = blend_logits.masked_fill(vis_flat.squeeze(-1) == 0, -6e4)
        weights = F.softmax(blend_logits, dim=-1)  # [BN, V]
        weights = weights.reshape(B, N, V)  # [B, N, V]
        return weights


class FusedSystem(nn.Module):
    def __init__(
        self,
        sam3_model: Optional[nn.Module] = None,
        sam3_processor: Optional[nn.Module] = None,
        img_size: int = 512,
        num_classes: int = 1,
        num_queries: int = 50,
        use_sam3_api: bool = True,
        sam3_checkpoint: str = "src/pretrained_weights/sam3.pt",
        anysplat_checkpoint: Optional[str] = None,
        use_anysplat_pretrained: bool = True,
        freeze_sam3: bool = True,
        freeze_sam3_backbone: bool = True,
        train_sam3_decoder: bool = False,
        freeze_gaussian_head: bool = False,
        freeze_backbone: bool = True,
        feature_dim: int = 2048,
        sh_degree: int = 2,
        num_points: int = 5000,
        use_replica_camera_context: bool = False,
        anysplat_pred_head_type: str = "point",
        gaussian_scale_min: float = 0.01,
        gaussian_scale_max: float = 0.3,
        voxel_size: float = 0.02,
        voxelize: bool = False,
        use_vote_head: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.use_replica_camera_context = use_replica_camera_context
        self.use_vote_head = use_vote_head
        self._has_logged_trainable_parameters = False

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
            pred_head_type=anysplat_pred_head_type,
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
        if not freeze_backbone:
            for head in [self.anysplat_encoder.camera_head, self.anysplat_encoder.depth_head]:
                for param in head.parameters():
                    param.requires_grad = True
        self.vote_head = VoteHead(num_classes=num_classes) if use_vote_head else None
        if use_sam3_api:
            self.sam3_segmenter = SAM3Segmenter(
                sam3_model=sam3_model,
                sam3_processor=sam3_processor,
                num_classes=num_classes,
                use_sam3_api=True,
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

    def _log_trainable_parameters_once(self) -> None:
        if self._has_logged_trainable_parameters:
            return
        trainable_params = []
        for module_name, module in [
            ("anysplat_encoder", self.anysplat_encoder),
            ("sam3_segmenter", self.sam3_segmenter),
        ]:
            if module is None:
                continue
            param_count = sum(param.numel() for param in module.parameters() if param.requires_grad)
            if param_count > 0:
                trainable_params.append(f"{module_name}={param_count}")
        if trainable_params:
            print(f"[FusedSystem] trainable parameters: {', '.join(trainable_params)}")
        else:
            print("[FusedSystem] trainable parameters: none")
        self._has_logged_trainable_parameters = True

    def _resolve_camera_context(
        self,
        encoder_output,
        gaussians: Gaussians,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_replica_camera_context or encoder_output is None:
            return intrinsics, extrinsics
        pred_context_pose = getattr(encoder_output, "pred_context_pose", None)
        if not isinstance(pred_context_pose, dict):
            return intrinsics, extrinsics
        replica_intrinsics = pred_context_pose.get("intrinsic")
        replica_extrinsics = pred_context_pose.get("extrinsic")
        if not isinstance(replica_intrinsics, torch.Tensor) or not isinstance(replica_extrinsics, torch.Tensor):
            return intrinsics, extrinsics
        return replica_intrinsics, replica_extrinsics

    def forward(self, images: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor, prompts: Optional[Dict] = None, enable_query_class_logit_lift: bool = True, pre_extracted_features: Optional[Dict] = None, leave_out_view: int | None = None, target_extrinsics_cam: torch.Tensor | None = None, target_pre_extracted_features: Optional[Dict] = None) -> FusedSystemOutput:
        _, _, _, h, w = images.shape
        if self.training:
            torch.cuda.empty_cache()
        self._log_trainable_parameters_once()
        encoder_frozen = not any(p.requires_grad for p in self.anysplat_encoder.parameters())
        ctx = torch.no_grad() if encoder_frozen else torch.enable_grad()
        with ctx:
            encoder_output = self.anysplat_encoder(images, global_step=0, visualization_dump=None)
        gaussians = encoder_output.gaussians
        if self.training:
            torch.cuda.empty_cache()
        camera_intrinsics, camera_extrinsics = self._resolve_camera_context(encoder_output, gaussians, intrinsics, extrinsics)
        if pre_extracted_features is not None:
            sam3_output = self._build_output_from_pre_extracted(pre_extracted_features, images.device)
        elif self.sam3_segmenter is not None:
            sam3_output = self.sam3_segmenter(images, prompts=prompts, feedback_masks=None, iteration=0)
        else:
            raise RuntimeError(
                "SAM3 not loaded and no pre_extracted_features provided. "
                "Either run scripts/pre_extract_sam3.py first, or set sam3.skip_for_training=false in config."
            )
        if enable_query_class_logit_lift:
            if self.use_vote_head and self.vote_head is not None:
                gaussians = self._assign_semantics_cross_view(gaussians, sam3_output, camera_intrinsics, camera_extrinsics, h, w, leave_out_view=leave_out_view)
                if gaussians.seg_query_class_logits is not None:
                    per_view = gaussians.seg_per_view_logits  # [B, N, V, C+1]
                    weights = self.vote_head(per_view)  # [B, N, V]
                    gaussians.seg_query_class_logits = (weights[:, :, :, None] * per_view[..., :-1]).sum(dim=2)  # [B, N, C]
                    del gaussians.seg_per_view_logits
            else:
                gaussians = self._assign_semantics_pixel_aligned(gaussians, sam3_output, h, w)
                if gaussians.seg_query_class_logits is None:
                    gaussians = self._assign_semantics_cross_view(gaussians, sam3_output, camera_intrinsics, camera_extrinsics, h, w)
        rendered_output = self.gaussian_renderer(gaussians, camera_extrinsics, camera_intrinsics, (h, w), render_color=True, render_qc_logits=enable_query_class_logit_lift)
        rendered_output["used_color_fallback"] = rendered_output.get("render_color") is None
        rendered_output["used_depth_fallback"] = rendered_output.get("render_depth") is None
        if rendered_output["used_color_fallback"]:
            rendered_output["render_color"] = images.clone()
        if rendered_output["used_depth_fallback"]:
            rendered_output["render_depth"] = torch.ones(images.shape[0], images.shape[1], h, w, device=images.device, dtype=images.dtype)
        # GNeSF-style target-view rendering: build gaussians from source views, render from target views
        target_rendered = None
        if self.training and target_extrinsics_cam is not None and target_pre_extracted_features is not None:
            C_s = extrinsics  # COLMAP source c2w
            C_t = target_extrinsics_cam.to(images.device)  # COLMAP target c2w
            A_s = camera_extrinsics  # AnySplat source c2w
            A_t = A_s @ torch.linalg.inv(C_s) @ C_t  # target pose in AnySplat frame
            target_rendered = self.gaussian_renderer(gaussians, A_t, camera_intrinsics, (h, w), render_color=False, render_qc_logits=True)
        return FusedSystemOutput(
            gaussians=gaussians,
            sam3_output=sam3_output,
            rendered_output=rendered_output,
            backbone_output=encoder_output,
            target_rendered_output=target_rendered,
        )

    def _assign_semantics_cross_view(self, gaussians: Gaussians, sam3_output, intrinsics, extrinsics, H: int, W: int, leave_out_view: int | None = None) -> Gaussians:
        from einops import rearrange
        sam3_qc = getattr(sam3_output, "query_class_logits", None)
        if sam3_qc is None or not isinstance(sam3_qc, torch.Tensor):
            return gaussians
        B, V, Q, C = sam3_qc.shape[:4]

        # ---------- GNeSF-aligned: aggregate Q → 1 BEFORE projection ----------
        # Per pixel, pick the query with highest fg score, use its full [bg, fg].
        # This converts SAM3 [B, V, Q, C, H, W] → per-view [B, V, C, H, W],
        # matching GNeSF's 2D segmenter output format (no Q dimension).
        sam3_fg = sam3_qc[..., 1, :, :]                       # [B, V, Q, H, W]
        best_q = sam3_fg.argmax(dim=2)                         # [B, V, H, W]
        sam3_per_view = sam3_qc.gather(                        # [B, V, C, H, W]
            dim=2, index=best_q.unsqueeze(2).unsqueeze(3).expand(B, V, 1, C, H, W)
        ).squeeze(2)

        total_pts = gaussians.means.shape[1]
        per_view = torch.zeros(B, V, total_pts, C + 1, device=gaussians.means.device, dtype=sam3_qc.dtype)
        for view_idx in range(V):
            if leave_out_view is not None and view_idx == leave_out_view:
                continue
            k = intrinsics[:, view_idx].clone()
            e_c2w = extrinsics[:, view_idx]
            e_w2c = torch.linalg.inv(e_c2w)
            k[:, 0, :] *= W
            k[:, 1, :] *= H
            p = torch.matmul(k, torch.cat([e_w2c[:, :3, :3], e_w2c[:, :3, 3:4]], dim=-1))
            means_homo = torch.cat([gaussians.means, torch.ones_like(gaussians.means[..., :1])], dim=-1)
            points_2d = torch.matmul(means_homo, p.transpose(1, 2))
            z_before_clamp = points_2d[..., 2:3]
            z = torch.clamp(z_before_clamp, min=1e-6)
            xy = points_2d[..., :2] / z  # [B, N, 2]
            in_image = (xy[..., 0] >= 0) & (xy[..., 0] < W) & (xy[..., 1] >= 0) & (xy[..., 1] < H)  # [B, N]
            visible = (z_before_clamp.squeeze(-1) > 1e-6) & in_image  # [B, N]
            coords = torch.stack([2 * xy[..., 0] / max(W - 1, 1) - 1, 2 * xy[..., 1] / max(H - 1, 1) - 1], dim=-1)
            view_logits = rearrange(sam3_per_view[:, view_idx], "b c h w -> b c h w")
            sampled = F.grid_sample(view_logits.unsqueeze(0) if view_logits.dim() == 3 else view_logits,
                                     coords.unsqueeze(2), mode="nearest", padding_mode="zeros", align_corners=True).squeeze(-1)
            if sampled.dim() == 2:
                sampled = sampled.unsqueeze(0)
            per_view[:, view_idx, :, :C] = sampled.permute(0, 2, 1)  # [B, N, C]
            per_view[:, view_idx, :, C] = visible.float()              # [B, N] visibility
        # per_view: [B, V, N, C+1] where C+1 = [bg, fg, visibility]
        gaussians.seg_per_view_logits = per_view.permute(0, 2, 1, 3)  # [B, N, V, C+1]
        gaussians.seg_query_class_logits = per_view.mean(dim=1)[..., :C]  # [B, N, C]
        return gaussians

    def _assign_semantics_pixel_aligned(self, gaussians: Gaussians, sam3_output, H: int, W: int, voxel_inv_indices=None) -> Gaussians:
        from einops import rearrange
        sam3_qc = getattr(sam3_output, "query_class_logits", None)
        if sam3_qc is None or not isinstance(sam3_qc, torch.Tensor):
            return gaussians
        B, V, Q, C = sam3_qc.shape[:4]
        total_pts = gaussians.means.shape[1]
        hw_per_view = H * W
        if total_pts == V * hw_per_view:
            gaussians.seg_query_class_logits = rearrange(sam3_qc, "b v q c h w -> b (v h w) q c")
        elif voxel_inv_indices is not None:
            gaussians.seg_query_class_logits = torch.zeros(B, total_pts, Q, C, device=gaussians.means.device, dtype=sam3_qc.dtype)
            offset = 0
            for b in range(B):
                inv_idx_flat = voxel_inv_indices[b]  # [V*H*W]
                num_total_voxels = int(inv_idx_flat.max().item()) + 1
                sam3_qc_b = rearrange(sam3_qc[b], "v q c h w -> v (h w) q c")
                sam3_qc_flat = rearrange(sam3_qc_b, "v n q c -> (v n) q c")
                aggregated = torch.zeros(num_total_voxels, Q, C, device=sam3_qc_flat.device, dtype=sam3_qc_flat.dtype)
                inv_idx_exp = inv_idx_flat.view(-1, 1, 1).expand(-1, Q, C)
                aggregated = aggregated.scatter_reduce(0, inv_idx_exp, sam3_qc_flat, reduce="mean", include_self=False)
                gaussians.seg_query_class_logits[b, :num_total_voxels] = aggregated
                offset += num_total_voxels
            if not getattr(self, "_warned_voxel_assign", False):
                print(f"[FusedSystem] voxelized semantic assignment: original_pts={V*hw_per_view}, voxelized_pts={total_pts}")
                self._warned_voxel_assign = True
        seg_masks = getattr(sam3_output, "seg_masks", None)
        if isinstance(seg_masks, torch.Tensor) and seg_masks.numel() > 0:
            flat_masks = rearrange(seg_masks, "b v q h w -> b (v h w) q")
            if Q > 1:
                max_vals, max_idx = flat_masks.max(dim=-1)
                gaussians.instance_labels = max_idx.long()
                gaussians.semantic_labels = (max_vals > 0.5).long()
            else:
                gaussians.instance_labels = (flat_masks.squeeze(-1) > 0.5).long()
                gaussians.semantic_labels = (flat_masks.squeeze(-1) > 0.5).long()
        return gaussians

    def _build_output_from_pre_extracted(self, pre_data: Dict, device: torch.device) -> SAM3SegmenterOutput:
        qc = pre_data["query_class_logits"].to(device)
        V = qc.shape[1]
        masks = pre_data.get("seg_masks")
        scores = pre_data.get("query_scores")
        return SAM3SegmenterOutput(
            seg_masks=masks.to(device) if masks is not None else torch.zeros(1, V, qc.shape[2], *qc.shape[-2:], device=device),
            seg_logits=qc[:, :, :, 1:2] if qc.dim() >= 4 else qc,
            query_class_logits=qc,
            query_scores=scores.to(device) if scores is not None else torch.ones(1, V, qc.shape[2], device=device),
        )
