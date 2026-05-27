import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
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
        self.log_interval = int(training_cfg.get("log_training_result_interval", 400))
        feature_cfg = training_cfg.get("feature_distill", {})
        self.feature_distill_enabled = bool(feature_cfg.get("enabled", True))
        self.feature_distill_weight = float(feature_cfg.get("weight", 1.0))
        self.save_hyperparameters(ignore=["model", "lpips"])

    def _build_pre_from_batch(self, batch, key_prefix=""):
        """Build pre_extracted dict from batch data (loaded by dataset)."""
        qc = batch.get(f"{key_prefix}sam3_query_class_logits")
        if qc is None or not isinstance(qc, torch.Tensor):
            return None
        return {
            "query_class_logits": qc,
            "seg_masks": batch.get(f"{key_prefix}sam3_seg_masks"),
            "query_scores": batch.get(f"{key_prefix}sam3_query_scores"),
        }

    def _run_fused_model(self, batch, enable_query_class_logit_lift=True,
                         target_extrinsics_cam=None, target_pre_extracted=None):
        pre_extracted = self._build_pre_from_batch(batch)
        return self.model.forward(
            batch["images"],
            batch["intrinsics"],
            batch["extrinsics"],
            batch.get("prompts"),
            enable_query_class_logit_lift=enable_query_class_logit_lift,
            pre_extracted_features=pre_extracted,
            target_extrinsics_cam=target_extrinsics_cam,
            target_pre_extracted_features=target_pre_extracted,
        )

    def _step_w_query_class_logit_lift(self, batch: Dict[str, torch.Tensor], batch_idx: int, leave_out_view: int | None = None, target_extrinsics_cam: torch.Tensor | None = None, target_pre_extracted: Optional[Dict] = None) -> Tuple[Any, ...]:
        output = self._run_fused_model(batch, enable_query_class_logit_lift=True, target_extrinsics_cam=target_extrinsics_cam, target_pre_extracted=target_pre_extracted)
        return output.gaussians, output.rendered_output, output.sam3_output, getattr(output.sam3_output, "seg_masks", None), getattr(output.sam3_output, "seg_infos", None), output

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
        # GNeSF-3D/models/render.py line 226: NLL/cross-entropy on class logits.
        # No Q dimension in GNeSF; we aggregate SAM3's Q at the 2D level
        # before projection (_assign_semantics_cross_view), so rendered_qc
        # is already Q-free. Target SAM3 Q is aggregated here.
        zero = torch.tensor(0.0, device=self.device)
        if self.feature_distill_weight <= 0:
            return zero
        rendered_qc = self._stack_render_qc_logits(render_output.get("render_qc_logits"))
        sam3_qc = getattr(sam3_output, "query_class_logits", None)
        if not isinstance(rendered_qc, torch.Tensor) or not isinstance(sam3_qc, torch.Tensor):
            return zero

        # rendered_qc: [B, V, 1, C, H, W] (dummy Q=1) → squeeze
        if rendered_qc.shape[2] == 1:
            rendered_qc = rendered_qc.squeeze(2)  # [B, V, C, H, W]
        # sam3_qc: [B, V, Q, C, H, W] → aggregate Q by best fg
        sam3_fg = sam3_qc.detach()[..., 1, :, :]                # [B, V, Q, H, W]
        best_q = sam3_fg.argmax(dim=2)                           # [B, V, H, W]
        B, V, Q, C, H, W = sam3_qc.shape
        sam3_pooled = sam3_qc.gather(
            dim=2, index=best_q.unsqueeze(2).unsqueeze(3).expand(B, V, 1, C, H, W)
        ).squeeze(2)                                             # [B, V, C, H, W]
        sam3_labels = sam3_pooled.argmax(dim=2)                  # [B, V, H, W]

        if rendered_qc.shape[-2:] != sam3_labels.shape[-2:]:
            rendered_qc = F.interpolate(
                rendered_qc.reshape(-1, C, *rendered_qc.shape[-2:]),
                size=sam3_labels.shape[-2:], mode="bilinear", align_corners=False,
            ).reshape(B, V, C, *sam3_labels.shape[-2:])

        rendered_flat = rendered_qc.permute(0, 1, 3, 4, 2).reshape(-1, C)
        labels_flat = sam3_labels.reshape(-1)

        # Inverse frequency weighting
        n_fg = (labels_flat > 0).sum().float().clamp(min=1)
        n_bg = labels_flat.numel() - n_fg
        class_weights = torch.tensor([
            labels_flat.numel() / n_bg.clamp(min=1),
            labels_flat.numel() / n_fg,
        ], device=self.device)
        loss = F.cross_entropy(rendered_flat, labels_flat, weight=class_weights)
        weighted = loss * self.feature_distill_weight
        self.log(f"{log_prefix}/feature_distill_loss", weighted, prog_bar=(log_prefix == "train"))
        return weighted

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if self.global_step % 50 == 0:
            torch.cuda.empty_cache()
        loss = sum(p.sum() * 0.0 for p in self.model.parameters() if p.requires_grad)

        has_target = self.use_vote_head and batch.get("target_extrinsics") is not None
        if has_target:
            target_pre = self._build_pre_from_batch(batch, key_prefix="target_")
            if target_pre is not None:
                target_extrinsics_cam = batch["target_extrinsics"].to(self.device)
                gaussians, render_output, sam3_output, _, _, output = self._step_w_query_class_logit_lift(
                    batch, batch_idx, target_extrinsics_cam=target_extrinsics_cam, target_pre_extracted=target_pre)
                target_rendered = output.target_rendered_output
                if target_rendered is not None and target_rendered.get("render_qc_logits") is not None:
                    target_sam3 = self.model._build_output_from_pre_extracted(target_pre, self.device)
                    loss = loss + self._calc_feature_distill_loss(target_rendered, target_sam3, log_prefix="train")
            else:
                gaussians, render_output, sam3_output, _, _, _ = self._step_w_query_class_logit_lift(batch, batch_idx)
                loss = loss + self._calc_feature_distill_loss(render_output, sam3_output, log_prefix="train")
        else:
            gaussians, render_output, sam3_output, _, _, _ = self._step_w_query_class_logit_lift(batch, batch_idx)
            loss = loss + self._calc_feature_distill_loss(render_output, sam3_output, log_prefix="train")

        self.log("train/loss", loss, prog_bar=True)
        if self.global_step % self.log_interval == 0:
            self._log_semantic_images(render_output.get("render_qc_logits"), getattr(sam3_output, "query_class_logits", None), log_prefix="train")
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        gaussians, render_output, sam3_output, context_seg_masks, _, fused_output = self._step_w_query_class_logit_lift(batch, batch_idx)
        val_loss = self._calc_feature_distill_loss(render_output, sam3_output, log_prefix="val")
        self.log("val/feature_distill_loss", val_loss)
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
            "vote_head": {"params": [], "lr": 1 * lr},
            "gaussian_head": {"params": [], "lr": 1 * lr},
            "camera_depth": {"params": [], "lr": 1 * lr},
            "sam3": {"params": [], "lr": 3 * lr},
            "default": {"params": [], "lr": lr * 0.1},
        }
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "vote_head" in name:
                param_groups["vote_head"]["params"].append(param)
            elif "gaussian_param_head" in name or "gaussian_adapter" in name:
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
