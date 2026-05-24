"""
一次性预提取 SAM3 特征，训练时直接读文件，不用每 step 实时跑 SAM3。
Usage:
    cd /data3/congwenjie/siu_pro_v2
    python scripts/pre_extract_sam3.py --config configs/fused_system_v2.yaml
"""
import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, resolve_project_path
from src.models.segmentation.sam3_segmenter import SAM3Segmenter
from src.data.dataset import SIU3RDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/fused_system_v2.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    sam3_cfg = config["model"].get("sam3", {})

    dataset = SIU3RDataset(
        resolve_project_path(data_cfg.get("dataset_root")),
        split=args.split or data_cfg.get("split", "train"),
        num_views=int(data_cfg.get("num_views", 3)),
        img_size=config["model"].get("image_size", [518, 518])[0]
        if isinstance(config["model"].get("image_size"), list)
        else config["model"].get("image_size", 518),
        frame_stride=int(data_cfg.get("frame_stride", 1)),
        max_samples=args.max_samples or data_cfg.get("max_train_samples"),
    )

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
    print(f"[PreExtract] saving features to {pre_extract_dir}, samples={len(dataset)}")

    for idx in tqdm(range(len(dataset)), desc="Pre-extracting SAM3"):
        sample = dataset[idx]
        scene_id = sample.get("scene_id", f"sample_{idx}")

        def _extract_view(images, prompt, start_idx_val):
            sample_key = f"{scene_id}_{start_idx_val}"
            out_path = Path(pre_extract_dir) / f"{sample_key}_sam3_qc.pt"
            if out_path.exists():
                return
            img_batch = images.unsqueeze(0).to(device) if images.dim() == 4 else images.to(device)
            with torch.no_grad():
                sam3_out = segmenter(img_batch, prompts=prompt, feedback_masks=None, iteration=0)
            torch.save(
                {
                    "query_class_logits": sam3_out.query_class_logits.cpu(),
                    "seg_masks": sam3_out.seg_masks.cpu(),
                    "query_scores": sam3_out.query_scores.cpu() if sam3_out.query_scores is not None else None,
                },
                str(out_path),
            )

        # Source views
        _extract_view(sample["images"], sample.get("prompts"), sample.get("start_idx", idx))
        # Target views
        target_images = sample.get("target_images")
        target_start = sample.get("target_start_idx")
        if isinstance(target_images, torch.Tensor) and target_images.numel() > 0 and target_start is not None:
            _extract_view(target_images, sample.get("prompts"), target_start)

    print(f"[PreExtract] done. Features saved to {pre_extract_dir}")


if __name__ == "__main__":
    main()
