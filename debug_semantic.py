"""Debug render_qc_logits values"""
import torch, sys
sys.path.insert(0, '.')
from src.models.model import FusedSystem
from src.config import load_config, resolve_project_path
from src.data.dataset import SIU3RDataset

config = load_config('configs/fused_system_v2.yaml')
ds = SIU3RDataset('src/data/SIU3R_DATASET', split='val', num_views=3, img_size=448,
                  max_samples=1, refer_pair_path='val_refer_pair.json', prompt_mode='object_name')
batch = ds[0]
images = batch['images'].unsqueeze(0).cuda()
intrinsics = batch['intrinsics'].unsqueeze(0).cuda()
extrinsics = batch['extrinsics'].unsqueeze(0).cuda()
prompts = batch.get('prompts')

model = FusedSystem(
    img_size=448, num_classes=1, num_queries=24,
    use_sam3_api=True,
    sam3_checkpoint=resolve_project_path('src/pretrained_weights/sam3.pt'),
    anysplat_checkpoint=resolve_project_path('src/pretrained_weights/model.safetensors'),
    freeze_sam3=True, freeze_backbone=True, freeze_gaussian_head=True,
).cuda().eval()

with torch.no_grad():
    out = model(images, intrinsics, extrinsics, prompts=prompts)

rqc = out.rendered_output.get('render_qc_logits')
if rqc and isinstance(rqc, list) and len(rqc) > 0:
    t = rqc[0]
    print(f'render_qc_logits shape: {tuple(t.shape)}')
    print(f'  min={t.min():.4f} max={t.max():.4f} mean={t.mean():.4f}')
    for v in range(t.shape[0]):
        fg = t[v, :, 1].mean().item()
        bg = t[v, :, 0].mean().item()
        print(f'  view{v}: fg_mean={fg:.4f}, bg_mean={bg:.4f}')
else:
    print('render_qc_logits is None or empty')

qc = out.sam3_output.query_class_logits
print(f'SAM3 qc shape: {tuple(qc.shape)}, fg_mean={qc[:,:,:,1].mean():.4f}')
