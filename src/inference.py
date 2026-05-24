import argparse
from pathlib import Path

import torch

from .config import load_config, resolve_project_path
from .models.model import FusedSystem
from .models.segmentation.sam3_segmenter import SAM3Segmenter
from .utils.export_utils import export_all_outputs, export_depth_map, export_input_views, export_points_to_ply, export_segmentation_masks
from .data.dataset import SIU3RDataset, create_dataloader


def build_segmenter(config: dict, device: torch.device) -> SAM3Segmenter:
    sam3_config = config["model"].get("sam3", {})
    print(f"[Inference] Building SAM3 segmenter on device={device} with sam3_checkpoint={sam3_config.get('checkpoint')}")
    segmenter = SAM3Segmenter(
        num_classes=sam3_config.get("num_classes", 1),
        num_queries=sam3_config.get("num_queries", 24),
        sam3_checkpoint=resolve_project_path(sam3_config.get("checkpoint", "src/pretrained_weights/sam3.pt")),
        freeze_sam3=sam3_config.get("freeze_sam3", True),
        freeze_sam3_backbone=sam3_config.get("freeze_sam3_backbone", True),
        train_sam3_decoder=sam3_config.get("train_sam3_decoder", False),
    )
    return segmenter.to(device).eval()


def load_fused_checkpoint(model: FusedSystem, ckpt_path: str, device: torch.device) -> None:
    if not ckpt_path:
        return
    resolved_ckpt = resolve_project_path(ckpt_path)
    if resolved_ckpt is None or not Path(resolved_ckpt).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(str(resolved_ckpt), map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {type(checkpoint).__name__}")
    fused_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            fused_state[key[len("model."):]] = value
        elif not key.startswith(("mse_loss", "psnr", "ssim", "lpips", "fused_loss")):
            fused_state[key] = value
    missing, unexpected = model.load_state_dict(fused_state, strict=False)
    vote_head_missing = [k for k in missing if "vote_head" in k]
    if vote_head_missing:
        raise RuntimeError(
            f"VoteHead weights not loaded from checkpoint — architecture mismatch. "
            f"Re-train with the current code. Missing keys: {vote_head_missing}"
        )
    print(
        f"[Inference] Loaded checkpoint: {resolved_ckpt}; "
        f"missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
    )


def build_model(config: dict, device: torch.device, ckpt_path: str = None) -> FusedSystem:
    model_config = config["model"]
    sam3_config = model_config.get("sam3", {})
    backbone_config = model_config.get("backbone", {})
    gaussian_config = model_config.get("gaussian_adapter", {})
    image_size = model_config.get("image_size", 518)
    img_size = image_size[0] if isinstance(image_size, list) else image_size
    model = FusedSystem(
        img_size=img_size,
        num_classes=sam3_config.get("num_classes", 1),
        num_queries=sam3_config.get("num_queries", 24),
        use_sam3_api=bool(sam3_config.get("checkpoint")),
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
        voxelize=False,
        use_vote_head=True,
    )
    load_fused_checkpoint(model, ckpt_path, device)
    return model.to(device).eval()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device_arg)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("[Inference] CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return requested


def get_dataset(config: dict, refer_pair: str = None, prompt_mode: str = "text"):
    data_cfg = config.get("data", {})
    image_size = config["model"].get("image_size", 518)
    img_size = image_size[0] if isinstance(image_size, list) else image_size
    dataset_root = resolve_project_path(data_cfg.get("dataset_root"))
    num_views = int(data_cfg.get("num_views", 3))
    frame_stride = int(data_cfg.get("frame_stride", 1))
    max_samples = data_cfg.get("max_val_samples", data_cfg.get("max_train_samples"))
    if dataset_root and Path(dataset_root).exists():
        dataset = SIU3RDataset(
            dataset_root,
            split=data_cfg.get("split", "train"),
            num_views=num_views,
            img_size=img_size,
            frame_stride=frame_stride,
            max_samples=max_samples,
            refer_pair_path=refer_pair,
            prompt_mode=prompt_mode,
        )
        if len(dataset) > 0:
            print(f"[Inference] Using real dataset: root={dataset_root}, split={data_cfg.get('split', 'train')}, num_views={num_views}, size={len(dataset)}")
            return dataset
        raise RuntimeError(f"Dataset path exists but resolved dataset is empty: {dataset_root}")
    raise FileNotFoundError(f"dataset_root is unavailable: {dataset_root}. Please set data.dataset_root to your real dataset path.")


def run_segmentation_inference(segmenter, dataset, output_dir, device, num_samples=3):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataloader = create_dataloader(dataset, batch_size=1, num_workers=0, shuffle=False, pin_memory=False, persistent_workers=False, drop_last=False)
    print(f"[Inference] Running segmentation inference: dataset={type(dataset).__name__}, output_dir={output_dir}, num_samples={num_samples}")
    for idx, batch in enumerate(dataloader):
        if idx >= num_samples:
            break
        images = batch["images"].to(device)
        prompts = batch.get("prompts")
        print(f"[Inference] Sample {idx}: images={tuple(images.shape)}, prompts={prompts}, scene={batch.get('scene_id')}, object={batch.get('object_id')}, object_name={batch.get('object_name')}, frames={batch.get('frame_ids')}")
        export_input_views(images, output_dir, prefix=f"sample_{idx:03d}", batch_idx=0)
        with torch.no_grad():
            sam3_output = segmenter(images, prompts=prompts, feedback_masks=None, iteration=0)
        export_segmentation_masks(sam3_output.seg_masks, output_dir / f"sample_{idx:03d}_seg.png", batch_idx=0, view_idx=0)
        print(f"[Inference] Exported sample_{idx:03d}_seg.png to {output_dir}")


def run_depth_inference(model, dataset, output_dir, device, num_samples=3):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataloader = create_dataloader(dataset, batch_size=1, num_workers=0, shuffle=False, pin_memory=False, persistent_workers=False, drop_last=False)
    print(f"[Inference] Running AnySplat depth inference: dataset={type(dataset).__name__}, output_dir={output_dir}, num_samples={num_samples}")
    for idx, batch in enumerate(dataloader):
        if idx >= num_samples:
            break
        images = batch["images"].to(device)
        print(f"[Inference] Sample {idx}: images={tuple(images.shape)}")
        with torch.no_grad():
            encoder_output = model.anysplat_encoder(images, global_step=0, visualization_dump=None)
        depth_info = getattr(encoder_output, "depth_dict", None)
        if not isinstance(depth_info, dict) or not isinstance(depth_info.get("depth"), torch.Tensor):
            raise RuntimeError("AnySplat did not return depth_dict['depth'].")
        depth = depth_info["depth"]
        depth_for_export = depth.squeeze(-1) if depth.shape[-1] == 1 else depth
        depth_view = depth_for_export[0, 0].detach().float()
        print(
            "[Inference] AnySplat depth stats: "
            f"shape={tuple(depth_for_export.shape)}, min={depth_view.min().item():.6f}, "
            f"max={depth_view.max().item():.6f}, mean={depth_view.mean().item():.6f}, std={depth_view.std(unbiased=False).item():.6f}"
        )
        export_depth_map(depth_for_export, output_dir / f"sample_{idx:03d}_anysplat_depth.png", batch_idx=0, view_idx=0)
        pts3d = depth_info.get("pts3d")
        if isinstance(pts3d, torch.Tensor):
            export_points_to_ply(pts3d, output_dir / f"sample_{idx:03d}_anysplat_pts3d.ply", batch_idx=0, view_idx=0)
        print(f"[Inference] Exported sample_{idx:03d}_anysplat_depth.png to {output_dir}")


def run_inference(model, dataset, output_dir, device, num_samples=3):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataloader = create_dataloader(dataset, batch_size=1, num_workers=0, shuffle=False, pin_memory=False, persistent_workers=False, drop_last=False)
    print(f"[Inference] Running inference: dataset={type(dataset).__name__}, output_dir={output_dir}, num_samples={num_samples}")
    for idx, batch in enumerate(dataloader):
        if idx >= num_samples:
            break
        images = batch["images"].to(device)
        intrinsics = batch["intrinsics"].to(device)
        extrinsics = batch["extrinsics"].to(device)
        prompts = batch.get("prompts")
        print(f"[Inference] Sample {idx}: images={tuple(images.shape)}, intrinsics={tuple(intrinsics.shape)}, extrinsics={tuple(extrinsics.shape)}, prompts={prompts}, scene={batch.get('scene_id')}, object={batch.get('object_id')}, object_name={batch.get('object_name')}, frames={batch.get('frame_ids')}")
        export_input_views(images, output_dir, prefix=f"sample_{idx:03d}", batch_idx=0)
        with torch.no_grad():
            output = model(images, intrinsics, extrinsics, prompts=prompts, enable_query_class_logit_lift=True)
        export_all_outputs(output, output_dir, prefix=f"sample_{idx:03d}", batch_idx=0)
        print(f"[Inference] Exported sample_{idx:03d} to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="SIU_PRO_V2 Inference")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "configs" / "fused_system_v2.yaml"))
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--depth-only", action="store_true", help="Run AnySplat only and export native AnySplat depth without Gaussian rendering.")
    parser.add_argument("--full-fusion", action="store_true", help="Run full AnySplat + SAM3 + Gaussian rendering pipeline. Default only exports SAM3 segmentation.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to a Lightning checkpoint produced by training.")
    parser.add_argument("--refer-pair", type=str, default=None, help="Path to a refer pair json such as val_refer_pair.json. Relative paths are resolved under data.dataset_root.")
    parser.add_argument("--prompt-mode", type=str, default="text", choices=["text", "object_name"], help="Use full referring text or short object_name as the SAM3 text prompt.")
    args = parser.parse_args()
    device = resolve_device(args.device)
    config = load_config(args.config)
    dataset = get_dataset(config, refer_pair=args.refer_pair, prompt_mode=args.prompt_mode)
    if args.depth_only:
        model = build_model(config, device, ckpt_path=args.ckpt)
        run_depth_inference(model, dataset, resolve_project_path(args.output), device, args.num_samples)
        return
    if not args.full_fusion:
        segmenter = build_segmenter(config, device)
        run_segmentation_inference(segmenter, dataset, resolve_project_path(args.output), device, args.num_samples)
        return
    model = build_model(config, device, ckpt_path=args.ckpt)
    run_inference(model, dataset, resolve_project_path(args.output), device, args.num_samples)


if __name__ == "__main__":
    main()
