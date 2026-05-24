from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class SAM3SegmenterOutput:
    seg_masks: torch.Tensor
    seg_logits: torch.Tensor
    query_class_logits: Optional[torch.Tensor] = None
    query_scores: Optional[torch.Tensor] = None
    pred_boxes: Optional[torch.Tensor] = None
    pred_boxes_xyxy: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    backbone_out: Optional[Dict] = None


class SAM3Segmenter(nn.Module):

    def __init__(
        self,
        num_classes: int = 1,
        num_queries: int = 100,
        sam3_checkpoint: str = "src/pretrained_weights/sam3.pt",
        freeze_sam3: bool = True,
        freeze_sam3_backbone: bool = True,
        train_sam3_decoder: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.sam3_checkpoint = sam3_checkpoint
        self.freeze_sam3 = freeze_sam3
        self.freeze_sam3_backbone = freeze_sam3_backbone
        self.train_sam3_decoder = train_sam3_decoder

        self.sam3_model = self._try_load_sam3_checkpoint()
        if self.sam3_model is None:
            raise RuntimeError(
                f"SAM3 checkpoint not found at {sam3_checkpoint}. "
                "Download sam3.pt or set sam3.checkpoint in config."
            )
        self.sam3_processor = None
        self._configure_sam3_trainability()

    @property
    def device(self):
        return next(self.parameters()).device

    def _try_load_sam3_checkpoint(self) -> Optional[nn.Module]:
        checkpoint_path = Path(self.sam3_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).resolve().parents[3] / checkpoint_path
        if not checkpoint_path.exists():
            return None
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        except Exception:
            return None
        if isinstance(checkpoint, nn.Module):
            return checkpoint
        if isinstance(checkpoint, dict):
            for key in ("sam3_model", "model", "module"):
                value = checkpoint.get(key)
                if isinstance(value, nn.Module):
                    return value
            state_dict_model = self._try_build_sam3_from_state_dict(checkpoint_path, checkpoint)
            if state_dict_model is not None:
                return state_dict_model
        return None

    def _try_build_sam3_from_state_dict(self, checkpoint_path: Path, checkpoint: Dict) -> Optional[nn.Module]:
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except Exception:
            return None
        try:
            model = build_sam3_image_model(
                checkpoint_path=str(checkpoint_path), load_from_HF=False,
                eval_mode=True, device="cpu")
            model.eval()
            self.sam3_processor = Sam3Processor(model=model, resolution=1008, device="cpu")
            return model
        except Exception:
            return None

    def _configure_sam3_trainability(self):
        self.sam3_model.eval()
        for param in self.sam3_model.parameters():
            param.requires_grad = False
        if not self.freeze_sam3:
            for param in self.sam3_model.parameters():
                param.requires_grad = True
            if self.freeze_sam3_backbone and hasattr(self.sam3_model, "backbone"):
                self.sam3_model.backbone.eval()
                for param in self.sam3_model.backbone.parameters():
                    param.requires_grad = False
            if not self.train_sam3_decoder:
                for name, module in self.sam3_model.named_children():
                    if name == "backbone":
                        continue
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False

    def forward(self, images, prompts=None, feedback_masks=None, iteration=0):
        return self._forward_with_sam3_api(images, prompts, feedback_masks, iteration)

    @staticmethod
    def _normalize_instance_map_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 2:
            return tensor.unsqueeze(0).unsqueeze(0)
        if tensor.dim() == 3:
            return tensor.unsqueeze(0)
        if tensor.dim() == 4 and tensor.shape[0] != 1 and tensor.shape[1] == 1:
            return tensor.permute(1, 0, 2, 3)
        return tensor

    def _select_prompt_text(self, prompts, batch_idx, view_idx):
        if prompts is None:
            return None
        if isinstance(prompts, list):
            if batch_idx >= len(prompts):
                return None
            return self._select_prompt_text(prompts[batch_idx], 0, view_idx)
        if not isinstance(prompts, dict) or "text" not in prompts:
            return None
        text_prompt = prompts["text"]
        if isinstance(text_prompt, str):
            return text_prompt
        if isinstance(text_prompt, (list, tuple)):
            if len(text_prompt) == 0:
                return None
            item = text_prompt[min(batch_idx, len(text_prompt) - 1)]
            if isinstance(item, (list, tuple)):
                if len(item) == 0:
                    return None
                item = item[min(view_idx, len(item) - 1)]
            return item if isinstance(item, str) else None
        return None

    def _prepare_processor_image(self, image):
        from PIL import Image
        image_uint8 = image.detach().float().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        pil_image = Image.fromarray(image_uint8, mode="RGB")
        processor_resolution = getattr(self.sam3_processor, "resolution", None)
        if isinstance(processor_resolution, int) and (pil_image.size[0] != processor_resolution or pil_image.size[1] != processor_resolution):
            pil_image = pil_image.resize((processor_resolution, processor_resolution), Image.BILINEAR)
        return pil_image

    def _resize_instance_tensor(self, tensor, height, width):
        if tensor.shape[-2:] == (height, width):
            return tensor
        original_shape = tensor.shape
        if tensor.numel() == 0:
            return tensor.new_zeros(*original_shape[:-2], height, width)
        resized = F.interpolate(
            tensor.reshape(-1, 1, *tensor.shape[-2:]).float(),
            size=(height, width), mode="bilinear", align_corners=False)
        return resized.reshape(*original_shape[:-2], height, width)

    def _processor_state_has_masks(self, processor_state):
        masks = self._extract_mask_tensor(processor_state)
        return isinstance(masks, torch.Tensor) and masks.numel() > 0 and masks.shape[0] > 0

    def _extract_mask_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value
        if not isinstance(value, dict):
            return None
        for key in ["masks", "pred_masks", "output_masks", "masklets", "seg_masks"]:
            candidate = value.get(key)
            if isinstance(candidate, torch.Tensor):
                return candidate
        for key in ["outputs", "output", "result", "results", "prediction", "predictions"]:
            candidate = self._extract_mask_tensor(value.get(key))
            if isinstance(candidate, torch.Tensor):
                return candidate
        return None

    def _extract_logits_tensor(self, value, masks):
        if not isinstance(value, dict):
            return masks
        for key in ["masks_logits", "pred_masks_logits", "mask_logits", "logits", "pred_masks"]:
            candidate = value.get(key)
            if isinstance(candidate, torch.Tensor):
                return candidate
        return masks

    def _extract_scores_tensor(self, value, num_masks):
        if isinstance(value, dict):
            for key in ["scores", "iou_scores", "pred_scores", "object_scores"]:
                candidate = value.get(key)
                if isinstance(candidate, torch.Tensor):
                    return candidate.to(self.device).float().reshape(-1)
        return torch.ones(num_masks, device=self.device)

    def _extract_boxes_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value if value.numel() > 0 and value.shape[-1] == 4 else None
        if not isinstance(value, dict):
            return None
        for key in ["boxes", "pred_boxes", "bboxes", "box"]:
            candidate = value.get(key)
            if isinstance(candidate, torch.Tensor) and candidate.numel() > 0:
                return candidate
        for key in ["outputs", "output", "result", "results"]:
            candidate = self._extract_boxes_tensor(value.get(key))
            if candidate is not None:
                return candidate
        return None

    @torch.no_grad()
    def _build_masks_from_boxes(self, boxes, processor_state, H, W):
        if boxes.dim() == 3:
            boxes = boxes[0]
        num_boxes = boxes.shape[0]
        masks = torch.zeros(num_boxes, H, W, device=self.device)
        for i in range(num_boxes):
            x1, y1, x2, y2 = boxes[i].tolist()
            x1_n = int(round(x1 * W)) if abs(x1) <= 1 else int(x1)
            y1_n = int(round(y1 * H)) if abs(y1) <= 1 else int(y1)
            x2_n = int(round(x2 * W)) if abs(x2) <= 1 else int(x2)
            y2_n = int(round(y2 * H)) if abs(y2) <= 1 else int(y2)
            x1_c = max(0, min(x1_n, x2_n))
            x2_c = min(W, max(x1_n, x2_n))
            y1_c = max(0, min(y1_n, y2_n))
            y2_c = min(H, max(y1_n, y2_n))
            if x2_c > x1_c + 1 and y2_c > y1_c + 1:
                masks[i, y1_c:y2_c, x1_c:x2_c] = 1.0
        masks = masks.unsqueeze(0)
        logits = torch.logit(masks.clamp(1e-4, 1 - 1e-4))
        scores = torch.full((num_boxes,), 0.8, device=self.device)
        new_state = dict(processor_state) if isinstance(processor_state, dict) else {}
        new_state["masks"] = masks
        new_state["masks_logits"] = logits
        new_state["scores"] = scores
        return new_state

    def _set_text_prompt_compatible(self, prompt_text, processor_state):
        call_attempts = [
            lambda: self.sam3_processor.set_text_prompt(prompt=prompt_text, state=processor_state),
            lambda: self.sam3_processor.set_text_prompt(text=prompt_text, state=processor_state),
            lambda: self.sam3_processor.set_text_prompt(prompt_text=prompt_text, state=processor_state),
            lambda: self.sam3_processor.set_text_prompt(prompt_text, processor_state),
            lambda: self.sam3_processor.set_text_prompt(processor_state, prompt_text),
            lambda: self.sam3_processor.set_text_prompt(prompt_text),
        ]
        for call in call_attempts:
            try:
                result = call()
                if isinstance(result, dict):
                    return result
            except Exception:
                pass
        return processor_state

    def _forward_with_sam3_api(self, images, prompts=None, feedback_masks=None, iteration=0):
        B, V, C, H, W = images.shape
        if self.sam3_processor is not None:
            self.sam3_processor.model = self.sam3_model
            self.sam3_processor.device = str(images.device)
            if hasattr(self.sam3_processor, "find_stage") and self.sam3_processor.find_stage is not None:
                self.sam3_processor.find_stage.img_ids = self.sam3_processor.find_stage.img_ids.to(images.device)
                self.sam3_processor.find_stage.text_ids = self.sam3_processor.find_stage.text_ids.to(images.device)
        all_seg_masks, all_seg_logits, all_query_class_logits, all_query_scores = [], [], [], []
        all_backbone_outs = []

        for b in range(B):
            for v in range(V):
                image = images[b, v]
                backbone_out = None
                processor_state = None
                if self.sam3_processor is not None:
                    try:
                        processor_image = self._prepare_processor_image(image)
                        try:
                            processor_state = self.sam3_processor.set_image(processor_image)
                        except Exception:
                            processor_state = self.sam3_processor.set_image(image.detach())
                        prompt_text = self._select_prompt_text(prompts, b, v)
                        if not prompt_text:
                            prompt_text = "visual"
                        processor_state = self._set_text_prompt_compatible(prompt_text, processor_state)
                        if not self._processor_state_has_masks(processor_state):
                            image_boxes = self._extract_boxes_tensor(processor_state)
                            if image_boxes is not None:
                                processor_state = self._build_masks_from_boxes(image_boxes, processor_state, H, W)
                            else:
                                visual_state = self._set_text_prompt_compatible("visual", dict(processor_state) if isinstance(processor_state, dict) else processor_state)
                                if self._processor_state_has_masks(visual_state):
                                    processor_state = visual_state
                        if isinstance(processor_state, dict) and "backbone_out" in processor_state:
                            backbone_out = processor_state["backbone_out"]
                    except Exception:
                        pass
                processor_masks = self._extract_mask_tensor(processor_state)
                if self.sam3_model is not None and isinstance(processor_masks, torch.Tensor):
                    masks = self._normalize_instance_map_tensor(processor_masks.to(self.device).float())
                    masks = self._resize_instance_tensor(masks, H, W)
                    logits_tensor = self._extract_logits_tensor(processor_state, processor_masks).to(self.device).float()
                    logits = self._normalize_instance_map_tensor(logits_tensor)
                    logits = self._resize_instance_tensor(logits, H, W)
                    scores = self._extract_scores_tensor(processor_state, masks.shape[1])
                    if scores.dim() == 0:
                        scores = scores.unsqueeze(0)
                    if scores.numel() < masks.shape[1]:
                        scores = torch.ones(masks.shape[1], device=self.device)
                    elif scores.numel() > masks.shape[1]:
                        scores = scores[:masks.shape[1]]
                    all_seg_masks.append(masks)
                    all_seg_logits.append(logits)
                    all_query_scores.append(scores)
                    all_backbone_outs.append(backbone_out if backbone_out is not None else processor_state)
                    qc_logits = self._convert_sam3_to_query_class_logits(masks, scores, H, W)
                    all_query_class_logits.append(qc_logits)
                elif self.sam3_model is not None and backbone_out is not None:
                    find_stage = self._get_find_stage(prompts)
                    geometric_prompt = self.sam3_model._get_dummy_prompt() if hasattr(self.sam3_model, "_get_dummy_prompt") else None
                    try:
                        sam3_output = self.sam3_model.forward_grounding(
                            backbone_out=backbone_out, find_input=find_stage,
                            find_target=None, geometric_prompt=geometric_prompt)
                    except Exception:
                        sam3_output = None
                    if isinstance(sam3_output, dict):
                        masks = self._normalize_instance_map_tensor(
                            sam3_output.get("pred_masks", torch.zeros(1, 1, H, W, device=self.device)))
                        logits = self._normalize_instance_map_tensor(sam3_output.get("pred_masks", masks))
                        pred_logits = sam3_output.get("pred_logits")
                        if isinstance(pred_logits, torch.Tensor):
                            scores = pred_logits.sigmoid().amax(dim=-1).squeeze(0)
                        else:
                            scores = torch.ones(masks.shape[1], device=self.device)
                    else:
                        masks = torch.zeros(1, 1, H, W, device=self.device)
                        logits = masks
                        scores = torch.ones(1, device=self.device)
                    all_seg_masks.append(masks)
                    all_seg_logits.append(logits)
                    all_query_scores.append(scores)
                    all_backbone_outs.append(backbone_out)
                    qc_logits = self._convert_sam3_to_query_class_logits(masks, scores, H, W)
                    all_query_class_logits.append(qc_logits)
                else:
                    dummy_mask = torch.zeros(1, self.num_classes, H, W, device=self.device)
                    all_seg_masks.append(dummy_mask)
                    all_seg_logits.append(dummy_mask)
                    all_query_scores.append(torch.ones(1, device=self.device))
                    all_backbone_outs.append({})
                    dummy_qc = torch.zeros(1, self.num_queries, self.num_classes + 1, H, W, device=self.device)
                    all_query_class_logits.append(dummy_qc)

        max_num_instances = max([m.shape[1] for m in all_seg_masks]) if all_seg_masks else self.num_classes
        if max_num_instances == 0:
            max_num_instances = 1
            all_seg_masks, all_seg_logits, all_query_scores, all_query_class_logits = self._build_center_prior_fallback(
                all_seg_masks, all_seg_logits, all_query_scores, all_query_class_logits, H, W)
        seg_masks = torch.zeros(B, V, max_num_instances, H, W, device=self.device)
        seg_logits = torch.zeros(B, V, max_num_instances, H, W, device=self.device)
        query_scores = torch.zeros(B, V, max_num_instances, device=self.device)

        idx = 0
        for b2 in range(B):
            for v2 in range(V):
                num_instances = all_seg_masks[idx].shape[1]
                seg_masks[b2, v2, :num_instances] = all_seg_masks[idx]
                seg_logits[b2, v2, :num_instances] = all_seg_logits[idx]
                query_scores[b2, v2, :num_instances] = all_query_scores[idx]
                idx += 1

        query_class_logits = None
        if all_query_class_logits:
            max_queries = max([q.shape[1] for q in all_query_class_logits])
            query_class_logits = torch.zeros(B, V, max_queries, self.num_classes + 1, H, W, device=self.device)
            idx = 0
            for b2 in range(B):
                for v2 in range(V):
                    num_q = all_query_class_logits[idx].shape[1]
                    query_class_logits[b2, v2, :num_q] = all_query_class_logits[idx]
                    idx += 1

        return SAM3SegmenterOutput(
            seg_masks=seg_masks, seg_logits=seg_logits,
            query_class_logits=query_class_logits,
            query_scores=query_scores if all_query_scores else None,
            pred_boxes=None, pred_boxes_xyxy=None,
            scores=query_scores if all_query_scores else None,
            backbone_out=all_backbone_outs if all_backbone_outs else None,
        )

    def _convert_sam3_to_query_class_logits(self, masks, scores, H, W):
        Q = masks.shape[1]
        scores = torch.clamp(scores, min=1e-6, max=1 - 1e-6)
        log_odds = torch.log(scores / (1 - scores))
        qc_logits = torch.zeros(1, Q, self.num_classes + 1, H, W, device=masks.device, dtype=torch.float32)
        for q in range(Q):
            mask = masks[0, q].bool()
            qc_logits[0, q, 0, mask] = -log_odds[q]
            qc_logits[0, q, 0, ~mask] = log_odds[q]
            if self.num_classes >= 1:
                qc_logits[0, q, 1, mask] = log_odds[q]
                qc_logits[0, q, 1, ~mask] = -log_odds[q]
        return qc_logits

    def _get_find_stage(self, prompts):
        try:
            from sam3.model.data_misc import FindStage
        except Exception:
            return None
        img_ids = torch.tensor([0], device=self.device, dtype=torch.long)
        text_ids = torch.tensor([0], device=self.device, dtype=torch.long)
        input_boxes = input_boxes_mask = input_boxes_label = None
        input_points = input_points_mask = None
        if prompts is not None:
            if "boxes" in prompts:
                input_boxes = prompts["boxes"]
                input_boxes_mask = torch.ones_like(input_boxes[..., 0], dtype=torch.bool)
                input_boxes_label = torch.ones_like(input_boxes[..., 0], dtype=torch.long)
            if "points" in prompts:
                input_points = prompts["points"]
                input_points_mask = torch.ones_like(input_points[..., 0], dtype=torch.bool)
        return FindStage(
            img_ids=img_ids, text_ids=text_ids,
            input_boxes=input_boxes, input_boxes_mask=input_boxes_mask,
            input_boxes_label=input_boxes_label,
            input_points=input_points, input_points_mask=input_points_mask)

    @torch.no_grad()
    def _build_center_prior_fallback(self, all_seg_masks, all_seg_logits, all_query_scores, all_query_class_logits, H, W):
        y = torch.arange(H, device=self.device, dtype=torch.float32)
        x = torch.arange(W, device=self.device, dtype=torch.float32)
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        sigma_y, sigma_x = H / 4.0, W / 4.0
        gy = torch.exp(-((y - cy) ** 2) / (2 * sigma_y ** 2))
        gx = torch.exp(-((x - cx) ** 2) / (2 * sigma_x ** 2))
        center_prior = (gy.unsqueeze(1) * gx.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        center_mask = center_prior / center_prior.max()
        new_seg_masks, new_seg_logits, new_query_scores, new_query_class_logits = [], [], [], []
        for _ in all_seg_masks:
            new_seg_masks.append(center_mask.clone())
            fg_logit = torch.logit(center_mask.clamp(1e-4, 1 - 1e-4))
            bg_logit = torch.logit((1 - center_mask).clamp(1e-4, 1 - 1e-4))
            new_seg_logits.append(fg_logit.clone())
            new_query_scores.append(torch.tensor([0.5], device=self.device))
            qc = torch.zeros(1, 1, self.num_classes + 1, H, W, device=self.device)
            qc[0, 0, 0] = bg_logit.squeeze(0)
            if self.num_classes >= 1:
                qc[0, 0, 1] = fg_logit.squeeze(0)
            new_query_class_logits.append(qc)
        return new_seg_masks, new_seg_logits, new_query_scores, new_query_class_logits
