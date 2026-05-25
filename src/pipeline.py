import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanSquaredError, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from .config import resolve_project_path
from .models.model import FusedSystem

class FusedPipeline(LightningModule):
    def __init__(self, config: Dict[str, Any], sam3_model: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        model_config = config.get("model", {})
        training_cfg = config.get("training", {})
        sam3_config = model_config.get("sam3", {})
        backbone_config = model_config.get("backbone", {})
        gaussian_config = model_config.get("gaussian_adapter", {})
        self.use_vote_head = bool(training_cfg.get("use_vote_head", False))
        self.model = FusedSystem(
            sam3_model=sam3_model,
            img_size=model_config["image_size"][0] if isinstance(model_config.get("image_size"), list) else model_config.get("image_size", 518),
            num_classes=sam3_config.get("num_classes", 1),
            num_queries=sam3_config.get("num_queries", 50),
            use_sam3_api=(bool(sam3_config.get("checkpoint")) or sam3_model is not None) and not bool(sam3_config.get("skip_for_training", False)),
            sam3_checkpoint=resolve_project_path(sam3_config.get("checkpoint", "src/pretrained_weights/sam3.pt")),
            anysplat_checkpoint=resolve_project_path(backbone_config.get("checkpoint")),
            use_anysplat_pretrained=bool(backbone_config.get("checkpoint")),
            freeze_sam3=sam3_config.get("freeze_sam3", True),
            freeze_sam3_backbone=sam3_config.get("freeze_sam3_backbone", True),
            train_sam3_decoder=sam3_config.get("train_sam3_decoder", False),
            freeze_gaussian_head=backbone_config.get("freeze_gaussian_head", False),
            freeze_backbone=backbone_config.get("freeze_encoder", True),
            feature_dim=backbone_config.get("feature_dim", 2048),
            sh_degree=gaussian_config.get("sh_degree", 2),
            num_points=gaussian_config.get("num_points", 3000),
            gaussian_scale_min=float(gaussian_config.get("gaussian_scale_min", 0.01)),
            gaussian_scale_max=float(gaussian_config.get("gaussian_scale_max", 0.3)),
            voxel_size=float(gaussian_config.get("voxel_size", 0.02)),
            voxelize=bool(gaussian_config.get("voxelize", True)),
            use_replica_camera_context=True,
            use_vote_head=self.use_vote_head,
        )
        self.mse_loss = MeanSquaredError()
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()
        try:
            self.lpips = LearnedPerceptualImagePatchSimilarity("vgg", normalize=True)
            self.lpips.eval()
            self.lpips.requires_grad_(False)
        except Exception:
            self.lpips = None
        self.weight_depth_smoothness = float(training_cfg.get("weight_depth_smoothness", 0.05))
        self.log_interval = int(training_cfg.get("log_training_result_interval", 400))
        feature_cfg = training_cfg.get("feature_distill", {})
        self.feature_distill_enabled = bool(feature_cfg.get("enabled", True))
        self.feature_distill_weight = float(feature_cfg.get("weight", 1.0))
        self.save_hyperparameters(ignore=["model", "lpips"])

    def _load_pre_extracted(self, scene_id: str, start_idx: int) -> Optional[Dict]:
        import os
        feat_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "pre_extracted",
                                 f"{scene_id}_{int(start_idx)}_sam3_qc.pt"))
        if os.path.exists(feat_path):
            pre_data = torch.load(feat_path, map_location="cpu", weights_only=True)
            return {
                "query_class_logits": pre_data["query_class_logits"],
                "seg_masks": pre_data.get("seg_masks"),
                "query_scores": pre_data.get("query_scores"),
            }
        return None

    def _run_fused_model(self, batch: Dict[str, torch.Tensor], enable_query_class_logit_lift: bool, leave_out_view: int | None = None, target_extrinsics_cam: torch.Tensor | None = None, target_pre_extracted: Optional[Dict] = None) -> Any:
        scene_id = batch["scene_id"][0] if isinstance(batch["scene_id"], list) else batch["scene_id"]
        start_idx = batch["start_idx"][0] if isinstance(batch["start_idx"], list) else batch["start_idx"]
        pre_extracted = self._load_pre_extracted(scene_id, int(start_idx))
        if pre_extracted is None:
            print(f"[PRE_EXTRACT] MISSING: {scene_id}_{int(start_idx)}", flush=True)
        return self.model.forward(
            batch["images"],
            batch["intrinsics"],
            batch["extrinsics"],
            batch.get("prompts"),
            enable_query_class_logit_lift=enable_query_class_logit_lift,
            pre_extracted_features=pre_extracted,
            leave_out_view=leave_out_view,
            target_extrinsics_cam=target_extrinsics_cam,
            target_pre_extracted_features=target_pre_extracted,
        )

    def _step_w_query_class_logit_lift(self, batch: Dict[str, torch.Tensor], batch_idx: int, leave_out_view: int | None = None, target_extrinsics_cam: torch.Tensor | None = None, target_pre_extracted: Optional[Dict] = None) -> Tuple[Any, ...]:
        output = self._run_fused_model(batch, enable_query_class_logit_lift=True, leave_out_view=leave_out_view, target_extrinsics_cam=target_extrinsics_cam, target_pre_extracted=target_pre_extracted)
        return output.gaussians, output.rendered_output, output.sam3_output, getattr(output.sam3_output, "seg_masks", None), getattr(output.sam3_output, "seg_infos", None), output

    def _align_render_and_target(self, render_colors: torch.Tensor, target_views_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if render_colors.shape[-2:] != target_views_images.shape[-2:]:
            b, n, c = render_colors.shape[:3]
            render_colors = F.interpolate(
                render_colors.reshape(b * n, c, *render_colors.shape[-2:]),
                size=target_views_images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).reshape(b, n, c, *target_views_images.shape[-2:])
        if render_colors.shape[2] != target_views_images.shape[2]:
            min_c = min(render_colors.shape[2], target_views_images.shape[2])
            render_colors = render_colors[:, :, :min_c]
            target_views_images = target_views_images[:, :, :min_c]
        if render_colors.shape[:2] != target_views_images.shape[:2]:
            if render_colors.shape[0] == 1 and target_views_images.shape[0] > 1:
                render_colors = render_colors.expand(target_views_images.shape[0], -1, -1, -1, -1)
            if render_colors.shape[1] == 1 and target_views_images.shape[1] > 1:
                render_colors = render_colors.expand(-1, target_views_images.shape[1], -1, -1, -1)
            if render_colors.shape[:2] != target_views_images.shape[:2]:
                min_b = min(render_colors.shape[0], target_views_images.shape[0])
                min_v = min(render_colors.shape[1], target_views_images.shape[1])
                render_colors = render_colors[:min_b, :min_v]
                target_views_images = target_views_images[:min_b, :min_v]
        return render_colors, target_views_images

    def _flatten_multiview_metric_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 5:
            b, v, c, h, w = images.shape
            return images.reshape(b * v, c, h, w)
        return images

    def _log_image_metrics(self, render_images: torch.Tensor, target_images: torch.Tensor, log_prefix: str, on_epoch: bool = False) -> None:
        metric_render, metric_target = self._align_render_and_target(render_images, target_images)
        metric_render = self._flatten_multiview_metric_images(metric_render).clamp(0, 1)
        metric_target = self._flatten_multiview_metric_images(metric_target).clamp(0, 1)
        self.log(f"{log_prefix}/psnr", self.psnr(metric_render, metric_target), on_epoch=on_epoch)
        self.log(f"{log_prefix}/ssim", self.ssim(metric_render, metric_target), on_epoch=on_epoch)

    def _log_semantic_images(self, render_qc_logits, sam3_qc, log_prefix: str, max_views: int = 3):
        """Log semantic heatmaps to TensorBoard: rendered vs SAM3 foreground class."""
        try:
            from einops import rearrange
            rendered_qc = self._stack_render_qc_logits(render_qc_logits)
            if not isinstance(rendered_qc, torch.Tensor) or not isinstance(sam3_qc, torch.Tensor):
                return
            B, V = rendered_qc.shape[:2]
            if rendered_qc.dim() != 6 or sam3_qc.dim() != 6:
                return
            min_v = min(V, max_views)
            for v in range(min_v):
                # foreground class of first query as heatmap
                render_fg = rendered_qc[0, v, 0, 1].detach().float()  # [H, W]
                sam3_fg = sam3_qc[0, v, 0, 1].detach().float()
                render_fg = (render_fg - render_fg.min()) / (render_fg.max() - render_fg.min() + 1e-8)
                sam3_fg = (sam3_fg - sam3_fg.min()) / (sam3_fg.max() - sam3_fg.min() + 1e-8)
                comparison = torch.cat([render_fg, sam3_fg], dim=-1).unsqueeze(0).unsqueeze(0)
                self.logger.experiment.add_image(
                    f"{log_prefix}/semantic_view{v}_render_vs_sam3",
                    comparison.squeeze(0),
                    self.global_step,
                )
        except Exception:
            pass

    def _calc_render_loss(self, render_output: Dict[str, torch.Tensor], target_views_images: torch.Tensor, log_prefix: str = "train") -> torch.Tensor:
        render_colors = render_output.get("render_color")
        if render_output.get("used_color_fallback", False):
            zero = torch.tensor(0.0, device=self.device)
            self.log(f"{log_prefix}/render_loss", zero, prog_bar=(log_prefix == "train"))
            return zero
        if render_colors is None:
            return torch.tensor(0.0, device=self.device)
        render_colors, target_views_images = self._align_render_and_target(render_colors, target_views_images)
        loss = self.mse_loss(render_colors, target_views_images)
        self.log(f"{log_prefix}/render_loss", loss, prog_bar=(log_prefix == "train"))
        if self.lpips is not None:
            b, n, c, h, w = target_views_images.shape
            try:
                lpips_loss = self.lpips(
                    F.interpolate(render_colors.view(b * n, c, h, w), size=(h // 2, w // 2), mode="bilinear", align_corners=False),
                    F.interpolate(target_views_images.view(b * n, c, h, w), size=(h // 2, w // 2), mode="bilinear", align_corners=False),
                )
                self.log(f"{log_prefix}/lpips_loss", lpips_loss, prog_bar=(log_prefix == "train"))
                loss = loss + 0.5 * lpips_loss
            except Exception:
                pass
        return loss

    def _calc_depth_smoothness_loss(self, render_depth: torch.Tensor, seg_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if render_depth is None or render_depth.numel() == 0 or getattr(self, "_current_depth_fallback", False):
            return torch.tensor(0.0, device=self.device)
        depth_dx = render_depth.diff(dim=-1)
        depth_dy = render_depth.diff(dim=-2)
        if seg_mask is not None and seg_mask.numel() > 0:
            if seg_mask.dim() == 5:
                if seg_mask.shape[2] == 1:
                    seg_mask = (seg_mask[:, :, 0] > 0.5).long()
                else:
                    max_vals, max_idx = seg_mask.max(dim=2)
                    seg_mask = torch.where(max_vals > 0.5, max_idx + 1, torch.zeros_like(max_idx)).long()
            # Resize seg_mask to match render_depth size if needed
            if seg_mask.shape[-2:] != render_depth.shape[-2:]:
                s = seg_mask.shape
                seg_mask = F.interpolate(
                    seg_mask.float().reshape(-1, 1, *s[-2:]), size=render_depth.shape[-2:],
                    mode='nearest').reshape(*s[:-2], *render_depth.shape[-2:]).long()
            if seg_mask.dim() == 4:
                instance_dx = ~(seg_mask.diff(dim=-1).to(torch.bool))
                instance_dx[seg_mask[:, :, :, 1:] == -1] = False
                instance_dy = ~(seg_mask.diff(dim=-2).to(torch.bool))
                instance_dy[seg_mask[:, :, 1:, :] == -1] = False
                depth_dx = depth_dx * instance_dx.detach()
                depth_dy = depth_dy * instance_dy.detach()
        return depth_dx.abs().mean() + depth_dy.abs().mean()

    def _stack_render_qc_logits(self, render_qc_logits) -> Optional[torch.Tensor]:
        if not isinstance(render_qc_logits, list):
            return None
        valid_logits = [item for item in render_qc_logits if isinstance(item, torch.Tensor) and item.dim() == 5]
        if not valid_logits:
            return None
        stacked = torch.stack(valid_logits, dim=0)
        if stacked.dim() == 7:
            if stacked.shape[0] == 1:
                stacked = stacked.squeeze(0)
            else:
                stacked = stacked.mean(dim=0)
        return stacked


    def _calc_feature_distill_loss(self, render_output, sam3_output, log_prefix="train"):
        # GNeSF-style NLL loss (GNeSF-3D/models/render.py line 226): cross-entropy on class predictions
        zero = torch.tensor(0.0, device=self.device)
        if self.feature_distill_weight <= 0:
            return zero
        rendered_qc = self._stack_render_qc_logits(render_output.get("render_qc_logits"))
        sam3_qc = getattr(sam3_output, "query_class_logits", None)
        if not isinstance(rendered_qc, torch.Tensor) or not isinstance(sam3_qc, torch.Tensor):
            return zero
        if rendered_qc.dim() != 6 or sam3_qc.dim() != 6:
            return zero
        if rendered_qc.shape[-2:] != sam3_qc.shape[-2:]:
            original_shape = rendered_qc.shape
            rendered_qc = F.interpolate(
                rendered_qc.reshape(-1, *rendered_qc.shape[-3:]),
                size=sam3_qc.shape[-2:], mode="bilinear", align_corners=False,
            ).reshape(*original_shape[:-2], *sam3_qc.shape[-2:])
        B, V, Q, C = rendered_qc.shape[:4]
        rendered_qc = rendered_qc[:, :V, :min(Q, sam3_qc.shape[2]), :min(C, sam3_qc.shape[3])]
        sam3_qc = sam3_qc[:, :V, :min(Q, sam3_qc.shape[2]), :min(C, sam3_qc.shape[3])]
        sam3_labels = sam3_qc.detach().argmax(dim=3)  # [B, V, Q, H, W], hard labels
        rendered_flat = rendered_qc.permute(0, 1, 2, 4, 5, 3).reshape(-1, C)  # [B*V*Q*H*W, C]
        loss = F.cross_entropy(rendered_flat, sam3_labels.reshape(-1))
        weighted = loss * self.feature_distill_weight
        self.log(f"{log_prefix}/feature_distill_loss", weighted, prog_bar=(log_prefix == "train"))
        return weighted

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.global_step % 50 == 0:
            torch.cuda.empty_cache()
        loss = torch.tensor(0.0, device=self.device)
        V = int(batch["images"].shape[1])
        has_target = self.use_vote_head and batch.get("target_extrinsics") is not None
        if has_target:
            # GNeSF-style: build from source views, render from target views
            scene_id = batch["scene_id"][0] if isinstance(batch["scene_id"], list) else batch["scene_id"]
            target_start_val = batch.get("target_start_idx", batch.get("start_idx"))
            if isinstance(target_start_val, (list, tuple)):
                target_start_val = target_start_val[0]
            if isinstance(target_start_val, torch.Tensor):
                target_start_val = int(target_start_val.item())
            else:
                target_start_val = int(target_start_val)
            target_pre = self._load_pre_extracted(scene_id, target_start_val)
            if target_pre is None:
                if self.global_step % 100 == 0:
                    print(f"[TRAIN] target SAM3 not pre-extracted for {scene_id}_{target_start_val}, skipping step. Run scripts/pre_extract_sam3.py first.")
                return loss  # zero loss, skip this step
            target_extrinsics_cam = batch["target_extrinsics"].to(self.device)
            gaussians, render_output, sam3_output, _, _, output = self._step_w_query_class_logit_lift(
                batch, batch_idx, target_extrinsics_cam=target_extrinsics_cam, target_pre_extracted=target_pre)
            target_rendered = output.target_rendered_output
            if target_rendered is not None and target_rendered.get("render_qc_logits") is not None:
                target_sam3 = self.model._build_output_from_pre_extracted(target_pre, self.device)
                loss = loss + self._calc_feature_distill_loss(target_rendered, target_sam3, log_prefix="train")
        else:
            gaussians, render_output, sam3_output, context_seg_masks, _, _ = self._step_w_query_class_logit_lift(batch, batch_idx)
            if not self.use_vote_head:
                _, _, _, h, w = batch["images"].shape
                self._current_depth_fallback = bool(render_output.get("used_depth_fallback", False))
                depth_smoothness_loss = self._calc_depth_smoothness_loss(render_output.get("render_depth"), context_seg_masks)
                self._current_depth_fallback = False
                self.log("train/depth_smoothness_loss", depth_smoothness_loss, prog_bar=True)
                loss += self.weight_depth_smoothness * depth_smoothness_loss
                if self.global_step == 0:
                    print(f"[TARGET DEBUG] has_target={'target_images' in batch}, keys={[k for k in batch.keys() if k.startswith('target')]}", flush=True)
                target_images = batch.get("target_images")
                if target_images is not None and target_images.numel() > 0:
                    target_extrinsics = batch.get("target_extrinsics", batch["extrinsics"]).to(self.device)
                    target_intrinsics = batch.get("target_intrinsics", batch["intrinsics"]).to(self.device)
                    target_render = self.model.gaussian_renderer(
                        gaussians=gaussians, extrinsics=target_extrinsics, intrinsics=target_intrinsics,
                        image_shape=(h, w), render_color=True, render_qc_logits=False,
                    )
                    if target_render.get("render_color") is not None:
                        loss += self._calc_render_loss(target_render, target_images, log_prefix="train")
                    else:
                        loss += self._calc_render_loss(render_output, batch["images"], log_prefix="train")
                else:
                    loss += self._calc_render_loss(render_output, batch["images"], log_prefix="train")
            loss += self._calc_feature_distill_loss(render_output, sam3_output, log_prefix="train")
        self.log("train/loss", loss, prog_bar=True)
        if self.global_step % self.log_interval == 0:
            if render_output.get("render_color") is not None and not render_output.get("used_color_fallback", False):
                self._log_image_metrics(render_output["render_color"], batch["images"], log_prefix="train")
            self._log_semantic_images(render_output.get("render_qc_logits"), getattr(sam3_output, "query_class_logits", None), log_prefix="train")
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        gaussians, render_output, sam3_output, context_seg_masks, _, fused_output = self._step_w_query_class_logit_lift(batch, batch_idx)
        target_views_images = batch.get("target_images", batch.get("target_views_images", batch.get("images")))
        if target_views_images is not None and render_output.get("render_color") is not None:
            val_loss = self._calc_render_loss(render_output, target_views_images, log_prefix="val")
            val_loss = val_loss + self._calc_feature_distill_loss(render_output, sam3_output, log_prefix="val")
            if not render_output.get("used_color_fallback", False):
                self._log_image_metrics(render_output["render_color"], target_views_images, log_prefix="val", on_epoch=True)
            self._log_semantic_images(render_output.get("render_qc_logits"), getattr(sam3_output, "query_class_logits", None), log_prefix="val")

    def on_validation_epoch_start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def configure_optimizers(self) -> Dict[str, Any]:
        training_cfg = self.config.get("training", {})
        optimizer_cfg = self.config.get("optimizer", {})
        lr = float(optimizer_cfg.get("lr", training_cfg.get("lr", 1e-4)))
        warm_up_epochs = int(optimizer_cfg.get("warm_up_epochs", 3))
        param_groups = {
            "gaussian_head": {"params": [], "lr": 1 * lr},
            "camera_depth": {"params": [], "lr": 1 * lr},
            "sam3": {"params": [], "lr": 3 * lr},
            "default": {"params": [], "lr": lr * 0.1},
        }
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "gaussian_param_head" in name or "gaussian_adapter" in name:
                param_groups["gaussian_head"]["params"].append(param)
            elif "camera_head" in name or "depth_head" in name:
                param_groups["camera_depth"]["params"].append(param)
            elif "sam3_segmenter" in name:
                param_groups["sam3"]["params"].append(param)
            else:
                param_groups["default"]["params"].append(param)
        optimizer_groups = [group for group in param_groups.values() if group["params"]]
        group_summaries = []
        for group_name, group in param_groups.items():
            param_count = sum(param.numel() for param in group["params"])
            group_summaries.append(f"{group_name}: params={param_count}, lr={group['lr']:.3e}")
        print(f"[FusedPipeline] optimizer parameter groups: {'; '.join(group_summaries)}")
        if not optimizer_groups:
            raise RuntimeError(
                "[FusedPipeline] No trainable parameters detected. "
                "Check freeze_sam3, freeze_gaussian_head, and freeze_encoder config settings."
            )
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=lr,
            weight_decay=float(optimizer_cfg.get("weight_decay", training_cfg.get("weight_decay", 0.05))),
            betas=(0.9, 0.95),
            eps=float(optimizer_cfg.get("eps", 1e-4)),
        )
        max_epochs = int(training_cfg.get("num_epochs", 100))
        warm_up = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / max(warm_up_epochs, 1), end_factor=1.0, total_iters=warm_up_epochs)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(max_epochs - warm_up_epochs, 1), eta_min=lr * 0.05)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, main_scheduler], milestones=[warm_up_epochs])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"}}

    def get_log_dir(self) -> str:
        stage_map = {"sanity_check": "sanity_check", "train": "train", "validate": "val", "test": "test", "predict": "pred"}
        stage = "train"
        if hasattr(self.trainer, "state") and hasattr(self.trainer.state, "stage"):
            stage = stage_map.get(self.trainer.state.stage, "train")
        log_dir = f"{self.config.get('output_path', 'logs')}/{stage}/{self.global_step}"
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
