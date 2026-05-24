"""测试 AnySplat — 直接手动挑3帧间距大的图，不走dataset"""
import sys, os, torch, numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(__file__))
from src.models.anysplat_src.anysplat import EncoderAnySplat, EncoderAnySplatCfg, OpacityMappingCfg
from src.models.anysplat_src.common.gaussian_adapter import GaussianAdapterCfg
from src.models.rendering.gaussian_renderer import SplattingCUDA
from src.utils.export_utils import export_gaussians_to_ply, export_rendered_image

device = torch.device("cuda")
IMG_SIZE = 448

# Manual preprocessing — exact AnySplat center crop
def preprocess(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w > h:
        nw, nh = int(w * IMG_SIZE / h), IMG_SIZE
    else:
        nw, nh = IMG_SIZE, int(h * IMG_SIZE / w)
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - IMG_SIZE) // 2, (nh - IMG_SIZE) // 2
    img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    return T.ToTensor()(img)

# Pick 3 frames with large baseline (not consecutive)
scene_dir = "src/data/SIU3R_DATASET/val/scene0011_00/color"
all_imgs = sorted(os.listdir(scene_dir))
# Take frames at 0%, 33%, 66% of the sequence
n = len(all_imgs)
frames = [all_imgs[0], all_imgs[n//3], all_imgs[2*n//3]]
print(f"Using frames: {frames} (out of {n} total)")

images = torch.stack([preprocess(os.path.join(scene_dir, f)) for f in frames]).unsqueeze(0).to(device)

# Encoder will predict its own camera — no need for real intrinsics/extrinsics
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

cam_extr = encoder_output.pred_context_pose["extrinsic"]
cam_intr = encoder_output.pred_context_pose["intrinsic"]
with torch.no_grad():
    rendered = renderer(gaussians, cam_extr, cam_intr, image_shape=(IMG_SIZE, IMG_SIZE), render_color=True)

out = Path("results/anysplat_spaced")
out.mkdir(parents=True, exist_ok=True)
export_gaussians_to_ply(gaussians, out / "gaussians.ply")
if rendered.get("render_color") is not None:
    for v in range(rendered["render_color"].shape[1]):
        export_rendered_image(rendered["render_color"], out / f"render_v{v}.png", batch_idx=0, view_idx=v)

print(f"Gaussians: {gaussians.means.shape[1]} pts")
print(f"Bounds: x=[{gaussians.means[0,:,0].min():.1f},{gaussians.means[0,:,0].max():.1f}] "
      f"y=[{gaussians.means[0,:,1].min():.1f},{gaussians.means[0,:,1].max():.1f}] "
      f"z=[{gaussians.means[0,:,2].min():.1f},{gaussians.means[0,:,2].max():.1f}]")
print(f"Done → {out}")
