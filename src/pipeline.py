from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule

from .config import resolve_project_path
from .models.model import FusedSystem


class FusedPipeline(LightningModule):
    def __init__(self, config: Dict[str, Any], sam3_model: Optional[torch.nn.Module] = None):
        super().__init__()
        self.config = config
        model_config = config.get("model", {})
        training_cfg = config.get("training", {})
        sam3_config = model_config.get("sam3", {})
        backbone_config = model_config.get("backbone", {})
        gaussian_config = model_config.get("gaussian_adapter", {})
        self.use_vote_head = bool(training_cfg.get("use_vote_head", False))
        self.model = FusedSystem(
            img_size=model_config["image_size"][0] if isinstance(model_config.get("image_size"), list) else model_config.get("image_size", 518),
            num_classes=sam3_config.get("num_classes", 1),
            num_queries=sam3_config.get("num_queries", 50),
            use_sam3_api=(bool(sam3_config.get("checkpoint")) or sam3_model is not None) and not bool(sam3_config.get("skip_for_training", False)),
            sam3_checkpoint=resolve_project_path(sam3_config.get("checkpoint", "src/pretrained_weights/sam3.pt")),
            anysplat_checkpoint=resolve_project_path(backbone_config.get("checkpoint")),
            freeze_sam3=sam3_config.get("freeze_sam3", True),
            freeze_sam3_backbone=sam3_config.get("freeze_sam3_backbone", True),
            train_sam3_decoder=sam3_config.get("train_sam3_decoder", False),
            freeze_gaussian_head=backbone_config.get("freeze_gaussian_head", True),
            freeze_backbone=backbone_config.get("freeze_encoder", True),
            feature_dim=backbone_config.get("feature_dim", 2048),
            sh_degree=gaussian_config.get("sh_degree", 2),
            num_points=gaussian_config.get("num_points", 3000),
            gaussian_scale_min=float(gaussian_config.get("gaussian_scale_min", 0.01)),
            gaussian_scale_max=float(gaussian_config.get("gaussian_scale_max", 0.3)),
            voxel_size=float(gaussian_config.get("voxel_size", 0.02)),
            voxelize=bool(gaussian_config.get("voxelize", True)),
            use_vote_head=self.use_vote_head,
        )
        self.feature_cfg = training_cfg.get("feature_distill", {})
        self.feature_distill_weight = float(self.feature_cfg.get("weight", 1.0))
        self.log_interval = int(training_cfg.get("log_training_result_interval", 400))
        self.save_hyperparameters(ignore=["model"])

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

    def _run_fused_model(self, batch, enable_query_class_logit_lift=True,
                         target_extrinsics_cam=None, target_pre_extracted=None):
        scene_id = batch["scene_id"][0] if isinstance(batch["scene_id"], list) else batch["scene_id"]
        start_idx = batch["start_idx"][0] if isinstance(batch["start_idx"], list) else batch["start_idx"]
        pre_extracted = self._load_pre_extracted(scene_id, int(start_idx))
        return self.model.forward(
            batch["images"], batch["intrinsics"], batch["extrinsics"],
            batch.get("prompts"),
            enable_query_class_logit_lift=enable_query_class_logit_lift,
            pre_extracted_features=pre_extracted,
            target_extrinsics_cam=target_extrinsics_cam,
            target_pre_extracted_features=target_pre_extracted,
        )

    def _stack_render_qc_logits(self, render_qc_logits) -> Optional[torch.Tensor]:
        if not isinstance(render_qc_logits, list):
            return None
        valid_logits = [item for item in render_qc_logits if isinstance(item, torch.Tensor) and item.dim() == 5]
        if not valid_logits:
            return None
        stacked = torch.stack(valid_logits, dim=0)
        if stacked.dim() == 7:
            stacked = stacked.squeeze(0) if stacked.shape[0] == 1 else stacked.mean(dim=0)
        return stacked

    def _calc_feature_distill_loss(self, render_output, sam3_output, log_prefix="train"):
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
        b, v, q, c = rendered_qc.shape[:4]
        rendered_qc = rendered_qc[:1, :v, :min(q, sam3_qc.shape[2]), :min(c, sam3_qc.shape[3])]
        sam3_qc = sam3_qc[:1, :v, :min(q, sam3_qc.shape[2]), :min(c, sam3_qc.shape[3])]
        loss = F.l1_loss(rendered_qc, sam3_qc.detach())
        weighted = loss * self.feature_distill_weight
        self.log(f"{log_prefix}/feature_distill_loss", weighted, prog_bar=(log_prefix == "train"))
        return weighted

    def training_step(self, batch, batch_idx):
        if self.global_step % 50 == 0:
            torch.cuda.empty_cache()
        loss = torch.tensor(0.0, device=self.device)
        # GNeSF-style: build from source views, render from target views
        scene_id = batch["scene_id"][0] if isinstance(batch["scene_id"], list) else batch["scene_id"]
        target_start_val = batch.get("target_start_idx", batch.get("start_idx"))
        if isinstance(target_start_val, (list, tuple)):
            target_start_val = target_start_val[0]
        if isinstance(target_start_val, torch.Tensor):
            target_start_val = int(target_start_val.item())
        else:
            target_start_val = int(target_start_val)
        if batch.get("target_extrinsics") is None:
            return loss
        target_pre = self._load_pre_extracted(scene_id, target_start_val)
        if target_pre is None:
            return loss
        target_extrinsics_cam = batch["target_extrinsics"].to(self.device)
        output = self._run_fused_model(
            batch, target_extrinsics_cam=target_extrinsics_cam, target_pre_extracted=target_pre)
        target_rendered = output.target_rendered_output
        if target_rendered is not None and target_rendered.get("render_qc_logits") is not None:
            target_sam3 = self.model._build_output_from_pre_extracted(target_pre, self.device)
            loss = loss + self._calc_feature_distill_loss(target_rendered, target_sam3, log_prefix="train")
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self._run_fused_model(batch)
        self._calc_feature_distill_loss(output.rendered_output, output.sam3_output, log_prefix="val")

    def configure_optimizers(self):
        training_cfg = self.config.get("training", {})
        optimizer_cfg = self.config.get("optimizer", {})
        lr = float(optimizer_cfg.get("lr", training_cfg.get("lr", 1e-4)))
        warm_up_epochs = int(optimizer_cfg.get("warm_up_epochs", 3))
        max_epochs = int(training_cfg.get("num_epochs", 20))

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters detected.")
        optimizer = torch.optim.AdamW(
            params, lr=lr,
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
            betas=(0.9, 0.95),
            eps=float(optimizer_cfg.get("eps", 1e-4)),
        )
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1 / max(warm_up_epochs, 1), end_factor=1.0,
            total_iters=warm_up_epochs)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(max_epochs - warm_up_epochs, 1),
            eta_min=lr * float(optimizer_cfg.get("eta_min_ratio", 0.05)))
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm_up, main_scheduler], milestones=[warm_up_epochs])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"}}
