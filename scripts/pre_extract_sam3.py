"""
Pre-extract SAM3 features: randomly select pool_size frames per scene,
save as one file. Training randomly picks 3 source + 3 target from the pool.
Usage:
    python scripts/pre_extract_sam3.py --config configs/fused_system_v2.yaml
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve_project_path
from src.models.segmentation.sam3_segmenter import SAM3Segmenter


def load_scene_text_prompt(scene_path: str) -> str | None:
    """Load text prompt from scene directory (text file or refer annotations)."""
    # Try text files first
    for filename in ["text.txt", "prompt.txt", "caption.txt", "captions.txt",
                     "description.txt", "label.txt", "labels.txt"]:
        text_path = os.path.join(scene_path, filename)
        if os.path.exists(text_path):
            try:
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if text:
                    return text
            except Exception:
                continue
    return None


def load_refer_text_index(data_root: str) -> dict:
    """Build scene_id -> text list from refer annotation files."""
    text_index = {}
    for filename in ["train_refer_seg_data.json", "val_refer_seg_data.json",
                     "refer_seg_data.json"]:
        ann_path = os.path.join(data_root, filename)
        if not os.path.exists(ann_path):
            continue
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
            if isinstance(annotations, dict):
                for scene_id, scene_ann in annotations.items():
                    if not isinstance(scene_ann, dict):
                        continue
                    texts = []
                    for value in scene_ann.values():
                        if isinstance(value, dict):
                            for obj_val in value.values():
                                if isinstance(obj_val, dict) and obj_val.get("object_name"):
                                    texts.append(obj_val["object_name"])
                    if texts:
                        text_index[scene_id] = list(dict.fromkeys(texts))
        except Exception:
            continue
    return text_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fused_system_v2.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sam3-batch-size", type=int, default=4,
                        help="Frames per SAM3 forward pass (reduce if OOM)")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    sam3_cfg = config["model"].get("sam3", {})
    pool_size = int(data_cfg.get("pool_size", 20))

    data_root = resolve_project_path(data_cfg.get("dataset_root"))
    img_sz = config["model"].get("image_size", [518, 518])
    img_size = img_sz[0] if isinstance(img_sz, list) else img_sz

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    segmenter = SAM3Segmenter(
        num_classes=sam3_cfg.get("num_classes", 1),
        num_queries=sam3_cfg.get("num_queries", 24),
        use_sam3_api=True,
        sam3_checkpoint=resolve_project_path(sam3_cfg.get("checkpoint", "src/pretrained_weights/sam3.pt")),
        freeze_sam3=True,
        freeze_sam3_backbone=True,
        train_sam3_decoder=False,
    ).to(device).eval()

    pre_extract_dir = resolve_project_path(data_cfg.get("pre_extract_dir", "data/pre_extracted"))
    Path(pre_extract_dir).mkdir(parents=True, exist_ok=True)

    # Pre-load refer text index for all splits
    refer_index = load_refer_text_index(data_root)
    if refer_index:
        print(f"[PreExtract] loaded refer text index for {len(refer_index)} scenes")

    for split_name in ["train", "val"]:
        split_dir = os.path.join(data_root, split_name)
        if not os.path.isdir(split_dir):
            continue
        scenes = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))

        for scene_name in tqdm(scenes, desc=f"Extracting {split_name}"):
            scene_path = os.path.join(split_dir, scene_name)
            images_dir = None
            for folder_name in ["images", "color", "imgs"]:
                test_path = os.path.join(scene_path, folder_name)
                if os.path.exists(test_path):
                    images_dir = test_path
                    break
            if images_dir is None:
                continue
            image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)
                                  if f.endswith((".jpg", ".png", ".jpeg"))])
            if len(image_files) < pool_size:
                continue

            out_path = Path(pre_extract_dir) / f"{scene_name}_sam3.pt"
            if out_path.exists():
                continue

            # Load text prompt for this scene
            text_prompt = load_scene_text_prompt(scene_path)
            if text_prompt is None and refer_index:
                texts = refer_index.get(scene_name, [])
                if texts:
                    text_prompt = texts[0]  # pick first object name
            prompts = {"text": text_prompt} if text_prompt else None

            # Randomly select pool_size frames from the scene
            selected_indices = sorted(random.sample(range(len(image_files)), pool_size))
            frames = []
            for idx in selected_indices:
                img = Image.open(image_files[idx]).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
                frames.append(torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1))

            # Process in sub-batches to avoid OOM
            sam3_batch = args.sam3_batch_size
            qc_list = []
            for sb in range(0, len(frames), sam3_batch):
                se = min(sb + sam3_batch, len(frames))
                sub_batch = torch.stack(frames[sb:se], dim=0).unsqueeze(0).to(device)  # [1, B, 3, H, W]
                with torch.no_grad():
                    sub_out = segmenter(sub_batch, prompts=prompts, feedback_masks=None, iteration=0)
                qc_list.append(sub_out.query_class_logits.cpu())  # [1, B, Q, C, H, W]
                del sub_batch, sub_out
                torch.cuda.empty_cache()

            # Pad all sub-batches to same Q before concatenating
            max_q = max(q.shape[2] for q in qc_list)
            qc_padded = []
            for q in qc_list:
                if q.shape[2] < max_q:
                    qc_padded.append(F.pad(q, (0, 0, 0, 0, 0, 0, 0, max_q - q.shape[2])))
                else:
                    qc_padded.append(q)
            all_qc = torch.cat(qc_padded, dim=1)  # [1, 20, max_q, C, H, W]
            torch.save(
                {"query_class_logits": all_qc,
                 "frame_indices": selected_indices,
                 "text_prompt": text_prompt},
                str(out_path))

    print(f"[PreExtract] done. Features saved to {pre_extract_dir}")


if __name__ == "__main__":
    main()
