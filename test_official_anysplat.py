"""用 HuggingFace 官方 AnySplat 跑同样的3张图，导出 PLY + render"""
import sys, os, torch, numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(__file__))

device = torch.device("cuda")

# 1. Load official AnySplat from HuggingFace
from src.models.anysplat_src.anysplat import EncoderAnySplat, EncoderAnySplatCfg, OpacityMappingCfg
from src.models.anysplat_src.common.gaussian_adapter import GaussianAdapterCfg
from src.models.rendering.gaussian_renderer import SplattingCUDA

# 2. Same 3 images as test_anysplat
from src.data.dataset import SIU3RDataset
ds = SIU3RDataset("src/data/SIU3R_DATASET", split="val", num_views=3, img_size=448, max_samples=1)
batch = ds[0]
images = batch["images"].unsqueeze(0).to(device)
intrinsics = batch["intrinsics"].unsqueeze(0).to(device)
extrinsics = batch["extrinsics"].unsqueeze(0).to(device)

print(f"images: {tuple(images.shape)}")
print(f"intrinsics[0,0]: fx={intrinsics[0,0,0,0]:.4f}, fy={intrinsics[0,0,1,1]:.4f}")

# 3. Build encoder and render (same as test_anysplat.py but save more views)
cfg = EncoderAnySplatCfg(
    name="anysplat", anchor_feat_dim=2048, voxel_size=0.02, n_offsets=1, d_feature=2048,
    add_view=False, num_monocular_samples=12000,
    backbone=None, visualizer=None,
    gaussian_adapter=GaussianAdapterCfg(gaussian_scale_min=0.01, gaussian_scale_max=0.3, sh_degree=4),
    apply_bounds_shim=False,
    opacity_mapping=OpacityMappingCfg(initial=3.0, final=3.0, warm_up=1),
    gaussians_per_pixel=1, num_surfaces=1, gs_params_head_type="dpt_gs",
    pretrained_weights="src/pretrained_weights/model.safetensors",
    pose_free=True, pred_pose=True, gt_pose_to_pts=False,
    gs_prune=False, opacity_threshold=0.001, gs_keep_ratio=1.0,
    pred_head_type="point",
    freeze_backbone=True, freeze_module="all",
    distill=False, render_conf=False, opacity_conf=False, conf_threshold=0.1,
    intermediate_layer_idx=None, voxelize=False,
)
encoder = EncoderAnySplat(cfg).to(device).eval()
renderer = SplattingCUDA().to(device)

with torch.no_grad():
    encoder_output = encoder(images, global_step=0)
gaussians = encoder_output.gaussians

# 4. Render from all 3 context views
cam_extr = encoder_output.pred_context_pose["extrinsic"]
cam_intr = encoder_output.pred_context_pose["intrinsic"]
with torch.no_grad():
    rendered = renderer(gaussians, cam_extr, cam_intr, image_shape=(448, 448), render_color=True)

# 5. Export
out = Path("results/official_test")
out.mkdir(parents=True, exist_ok=True)
from src.utils.export_utils import export_gaussians_to_ply, export_rendered_image
export_gaussians_to_ply(gaussians, out / "gaussians.ply")
if rendered.get("render_color") is not None:
    for v in range(rendered["render_color"].shape[1]):
        export_rendered_image(rendered["render_color"], out / f"render_view{v}.png", batch_idx=0, view_idx=v)

# 6. Print key stats
print(f"Gaussians: {gaussians.means.shape[1]} points")
print(f"opacity: mean={gaussians.opacities.mean():.4f}, min={gaussians.opacities.min():.4f}")
print(f"scales: mean={gaussians.scales.mean():.4f}, min={gaussians.scales.min():.6f}, max={gaussians.scales.max():.4f}")
print(f"scene bounds: x=[{gaussians.means[0,:,0].min():.2f},{gaussians.means[0,:,0].max():.2f}] y=[{gaussians.means[0,:,1].min():.2f},{gaussians.means[0,:,1].max():.2f}] z=[{gaussians.means[0,:,2].min():.2f},{gaussians.means[0,:,2].max():.2f}]")
print(f"Done → {out}")
