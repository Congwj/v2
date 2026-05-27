"""
诊断脚本：定位 VoteHead 训练无效的根因。
在服务器项目根目录运行：
    python scripts/diagnose_training.py
"""
import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, resolve_project_path
from src.data.dataset import SIU3RDataset, create_dataloader
from src.pipeline import FusedPipeline

config = load_config("configs/fused_system_v2.yaml")
device = torch.device("cuda")

print("=" * 60)
print("1. 加载 pipeline & dataset")
print("=" * 60)

pipeline = FusedPipeline(config).to(device)
pipeline.train()

data_cfg = config["data"]
dataset = SIU3RDataset(
    resolve_project_path(data_cfg["dataset_root"]),
    split=data_cfg["split"],
    num_views=int(data_cfg.get("num_views", 3)),
    img_size=config["model"]["image_size"][0],
    max_samples=data_cfg.get("max_train_samples"),
)
loader = create_dataloader(dataset, batch_size=1, num_workers=0, shuffle=False,
                           pin_memory=False, persistent_workers=False, drop_last=False)
batch = next(iter(loader))

print(f"Dataset samples: {len(dataset)}")
print(f"Keys in batch: {list(batch.keys())}")
print(f"scene_id: {batch['scene_id']}")
print(f"start_idx: {batch['start_idx']}")
print(f"target_start_idx: {batch.get('target_start_idx', 'MISSING')}")
print(f"target_extrinsics: {'OK' if batch.get('target_extrinsics') is not None else 'MISSING'}")

print("\n" + "=" * 60)
print("2. 检查预提取文件是否匹配")
print("=" * 60)

sid = batch["scene_id"][0] if isinstance(batch["scene_id"], list) else batch["scene_id"]
sid_val = int(sid) if isinstance(sid, str) and sid.isdigit() else sid
src_idx = int(batch["start_idx"][0].item()) if isinstance(batch["start_idx"], torch.Tensor) else int(batch["start_idx"][0])
tgt_start = batch.get("target_start_idx")
if isinstance(tgt_start, (list, tuple)):
    tgt_idx = int(tgt_start[0].item()) if isinstance(tgt_start[0], torch.Tensor) else int(tgt_start[0])
else:
    tgt_idx = "N/A"

pre_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "pre_extracted"))
src_feat = f"{sid_val}_{src_idx}_sam3_qc.pt"
src_path = os.path.join(pre_dir, src_feat)
if isinstance(tgt_idx, int):
    tgt_feat = f"{sid_val}_{tgt_idx}_sam3_qc.pt"
    tgt_path = os.path.join(pre_dir, tgt_feat)
else:
    tgt_path = None

print(f"Source pre_extract: {src_feat} -> {'EXISTS' if os.path.exists(src_path) else 'MISSING!'}")
if tgt_path:
    print(f"Target pre_extract: {tgt_feat} -> {'EXISTS' if os.path.exists(tgt_path) else 'MISSING!'}")

print("\n" + "=" * 60)
print("3. 单步前向 + 检查 VoteHead 参数是否更新")
print("=" * 60)

model = pipeline.model
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
model.train()

# Store initial VoteHead params
vh_params_before = {name: p.clone().detach() for name, p in model.named_parameters() if "vote_head" in name and p.requires_grad}
if not vh_params_before:
    print("ERROR: No trainable VoteHead parameters!")
    vote_params = [(name, p.requires_grad) for name, p in model.named_parameters() if "vote_head" in name]
    print(f"VoteHead params: {vote_params}")
else:
    print(f"VoteHead trainable params: {list(vh_params_before.keys())}")

# Move batch to device
images = batch["images"].to(device)
intrinsics = batch["intrinsics"].to(device)
extrinsics = batch["extrinsics"].to(device)

# Load pre_extracted
scene_id = batch["scene_id"][0] if isinstance(batch["scene_id"], list) else batch["scene_id"]
start_idx = batch["start_idx"][0] if isinstance(batch["start_idx"], list) else batch["start_idx"]
start_idx = int(start_idx.item()) if isinstance(start_idx, torch.Tensor) else int(start_idx)
src_pre_path = os.path.join(pre_dir, f"{scene_id}_{start_idx}_sam3_qc.pt")
pre_extracted = torch.load(src_pre_path, map_location="cpu", weights_only=True) if os.path.exists(src_pre_path) else None

# Load target pre_extracted
tgt_start_val = batch.get("target_start_idx", batch.get("start_idx"))
if isinstance(tgt_start_val, (list, tuple)):
    tgt_start_val = tgt_start_val[0]
tgt_start_val = int(tgt_start_val.item()) if isinstance(tgt_start_val, torch.Tensor) else int(tgt_start_val)
tgt_pre_path = os.path.join(pre_dir, f"{scene_id}_{tgt_start_val}_sam3_qc.pt")
target_pre = torch.load(tgt_pre_path, map_location="cpu", weights_only=True) if os.path.exists(tgt_pre_path) else None

target_ext = batch.get("target_extrinsics")
target_pre_dict = {"query_class_logits": target_pre["query_class_logits"]} if target_pre else None

print(f"Source pre loaded: {pre_extracted is not None}")
print(f"Target pre loaded: {target_pre is not None}")
print(f"Target extrinsics: {target_ext is not None}")

# Forward pass with high LR optimizer
pre_dict = {"query_class_logits": pre_extracted["query_class_logits"],
            "seg_masks": pre_extracted.get("seg_masks"),
            "query_scores": pre_extracted.get("query_scores")} if pre_extracted else None

out = model(images, intrinsics, extrinsics,
            pre_extracted_features=pre_dict,
            target_extrinsics_cam=target_ext.to(device) if target_ext is not None else None,
            target_pre_extracted_features=target_pre_dict)

print(f"rendered_output keys: {list(out.rendered_output.keys())}")
print(f"target_rendered_output: {'EXISTS' if out.target_rendered_output else 'NONE'}")
if out.target_rendered_output:
    print(f"  target render_qc_logits: {'EXISTS' if out.target_rendered_output.get('render_qc_logits') else 'NONE'}")

# Compute loss
if out.target_rendered_output and out.target_rendered_output.get("render_qc_logits") and target_pre_dict:
    from einops import rearrange
    import torch.nn.functional as F
    rendered = out.target_rendered_output["render_qc_logits"]
    if isinstance(rendered, list):
        rendered = next((r for r in rendered if isinstance(r, torch.Tensor)), None)
    sam3 = target_pre_dict["query_class_logits"].to(device)
    if isinstance(rendered, torch.Tensor) and rendered.dim() >= 5:
        print(f"rendered shape: {rendered.shape}, sam3 shape: {sam3.shape}")
        # Aggregate across queries (logsumexp): handles Q mismatch
        rendered_pooled = rendered.logsumexp(dim=1)  # [V, C, H, W]
        sam3_t = sam3[0] if sam3.dim() == 6 else sam3  # [V, Q, C, H, W]
        sam3_pooled = sam3_t.logsumexp(dim=1)  # [V, C, H, W]
        C = rendered_pooled.shape[1]
        sam3_labels = sam3_pooled.argmax(dim=1).reshape(-1)  # [V*H*W]
        rendered_flat = rendered_pooled.permute(0, 2, 3, 1).reshape(-1, C)  # [V*H*W, C]

        # Class-weighted loss (same as pipeline.py)
        n_fg = (sam3_labels > 0).sum().float().clamp(min=1)
        n_bg = sam3_labels.numel() - n_fg
        fg_w = sam3_labels.numel() / n_fg
        bg_w = sam3_labels.numel() / n_bg.clamp(min=1)
        cw = torch.tensor([bg_w, fg_w], device=device)
        initial_loss = F.cross_entropy(rendered_flat, sam3_labels, weight=cw)
        print(f"\nInitial loss: {initial_loss.item():.6f}  (random baseline: 0.693)")
        print(f"Class weights: bg={bg_w:.1f}, fg={fg_w:.1f}")
        print(f"loss.requires_grad: {initial_loss.requires_grad}")
        print(f"rendered.requires_grad: {rendered.requires_grad}")
        print(f"rendered.is_leaf: {rendered.is_leaf}, grad_fn: {rendered.grad_fn}")
        # Check the gaussians' seg_query_class_logits after forward
        g_after = out.gaussians
        print(f"gaussians.seg_query_class_logits.requires_grad: {g_after.seg_query_class_logits.requires_grad}")
        print(f"gaussians.seg_query_class_logits.is_leaf: {g_after.seg_query_class_logits.is_leaf}")
        print(f"gaussians.seg_query_class_logits.grad_fn: {g_after.seg_query_class_logits.grad_fn}")

        # Step
        optimizer.zero_grad()
        initial_loss.backward()

        # Check gradients more carefully
        grad_count = 0
        zero_grad_count = 0
        for name, p in model.named_parameters():
            if "vote_head" in name and p.requires_grad:
                if p.grad is not None:
                    grad_count += 1
                    if p.grad.abs().max().item() > 0:
                        print(f"  {name}: grad max={p.grad.abs().max().item():.8f}, mean={p.grad.abs().mean().item():.8f}")
                    else:
                        zero_grad_count += 1
                else:
                    print(f"  {name}: p.grad is None!")
        print(f"\nGradient summary: {grad_count} params have grad, {zero_grad_count} have zero grad")

        # Extra: check per-pixel gradient on rendered_pooled to isolate the break
        rp_grad = rendered_pooled.grad
        print(f"\nrendered_pooled.grad is None: {rp_grad is None}")
        if rp_grad is not None:
            print(f"rendered_pooled.grad nonzero: {(rp_grad.abs() > 0).sum().item()} / {rp_grad.numel()}")
            print(f"rendered_pooled.grad max: {rp_grad.abs().max().item():.8f}")

        # Check if seg_qc_logits gets any gradient
        sqc_grad = out.gaussians.seg_query_class_logits.grad
        print(f"seg_query_class_logits.grad is None: {sqc_grad is None}")
        if sqc_grad is not None:
            print(f"seg_qc_logits grad nonzero: {(sqc_grad.abs() > 0).sum().item()} / {sqc_grad.numel()}")
            print(f"seg_qc_logits grad max: {sqc_grad.abs().max().item():.8f}")

        # Check gradients
        grad_norms = {}
        for name, p in model.named_parameters():
            if "vote_head" in name and p.grad is not None:
                grad_norms[name] = p.grad.norm().item()
        if grad_norms:
            print(f"\nVoteHead gradient norms after backward:")
            for name, norm in grad_norms.items():
                print(f"  {name}: {norm:.6f}")
        else:
            print("\nERROR: No VoteHead gradients after backward!")

        optimizer.step()

        # Check parameter changes
        print(f"\nParameter changes after 1 step:")
        for name in vh_params_before:
            if name in model.state_dict():
                diff = (model.state_dict()[name] - vh_params_before[name]).abs().max().item()
                print(f"  {name}: max_change={diff:.10f}")

        # Forward again and check new loss
        with torch.no_grad():
            out2 = model(images, intrinsics, extrinsics,
                        pre_extracted_features=pre_dict,
                        target_extrinsics_cam=target_ext.to(device) if target_ext is not None else None,
                        target_pre_extracted_features=target_pre_dict)
            rendered2 = out2.target_rendered_output["render_qc_logits"]
            if isinstance(rendered2, list):
                rendered2 = next((r for r in rendered2 if isinstance(r, torch.Tensor)), None)
            if isinstance(rendered2, torch.Tensor) and rendered2.dim() >= 5:
                rp2 = rendered2.logsumexp(dim=1)
                rf2 = rp2.permute(0, 2, 3, 1).reshape(-1, C)
                new_loss = F.cross_entropy(rf2, sam3_labels, weight=cw)
                print(f"\nLoss after 1 step: {new_loss.item():.6f} (change: {initial_loss.item() - new_loss.item():.6f})")

print("\n" + "=" * 60)
print("4. 检查相机矩阵一致性")
print("=" * 60)

# Compare source rendering between test method and our method
with torch.no_grad():
    enc_out = model.anysplat_encoder(images[:1], global_step=0)
    g = enc_out.gaussians

    # Method A: test script way (pred_context_pose directly)
    camA_extr = enc_out.pred_context_pose["extrinsic"]
    camA_intr = enc_out.pred_context_pose["intrinsic"]
    rendA = model.gaussian_renderer(g, camA_extr, camA_intr, (448, 448), render_color=True)

    # Method B: our way (same as test now, after removing .inverse())
    _, camB_extr = model._resolve_camera_context(enc_out, g, intrinsics[:1], extrinsics[:1])
    rendB = model.gaussian_renderer(g, camB_extr, camA_intr, (448, 448), render_color=True)

    if rendA.get("render_color") is not None and rendB.get("render_color") is not None:
        diff = (rendA["render_color"] - rendB["render_color"]).abs().max().item()
        print(f"Test method vs Our method render difference: max={diff:.6f}")
        if diff > 0.01:
            print("WARNING: Camera matrices are DIFFERENT between test and our method!")
        else:
            print("OK: Camera matrices match.")
    else:
        print("ERROR: Rendering failed!")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
