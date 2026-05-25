from pathlib import Path
from datetime import timedelta
import warnings

import lightning as L
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

from .config import load_config, resolve_project_path
from .data.dataset import SIU3RDataset, MockMultiViewDataset, create_dataloader
from .pipeline import FusedPipeline


def build_datasets(config: dict):
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    image_size = model_cfg.get("image_size", [518, 518])
    img_size = image_size[0] if isinstance(image_size, list) else image_size
    dataset_root = resolve_project_path(data_cfg.get("dataset_root"))
    num_views = int(data_cfg.get("num_views", 3))
    frame_stride = int(data_cfg.get("frame_stride", 1))
    max_train_samples = data_cfg.get("max_train_samples")
    max_val_samples = data_cfg.get("max_val_samples")
    target_offset_mult = int(data_cfg.get("target_offset_mult", 2))
    split = data_cfg.get("split", "train")
    if dataset_root and Path(dataset_root).exists():
        train_dataset = SIU3RDataset(dataset_root, split=split, num_views=num_views, img_size=img_size, frame_stride=frame_stride, max_samples=max_train_samples, target_offset_mult=target_offset_mult)
        val_dataset = SIU3RDataset(dataset_root, split="val" if split == "train" else split, num_views=num_views, img_size=img_size, frame_stride=frame_stride, max_samples=max_val_samples, target_offset_mult=target_offset_mult)
        if len(train_dataset) > 0:
            return train_dataset, val_dataset if len(val_dataset) > 0 else train_dataset
    mock = MockMultiViewDataset(num_samples=16, num_views=num_views, img_size=img_size)
    return mock, mock


def build_trainer(config: dict) -> Trainer:
    trainer_cfg = config.get("trainer", {})
    has_cuda = torch.cuda.is_available()
    accelerator = trainer_cfg.get("accelerator", "gpu")
    if accelerator == "gpu" and not has_cuda:
        accelerator = "cpu"
    precision = trainer_cfg.get("precision", "16-mixed")
    if accelerator == "cpu" and precision != "32":
        precision = "32"
    strategy_name = trainer_cfg.get("strategy", "ddp_find_unused_parameters_true")
    strategy = DDPStrategy(find_unused_parameters=True, timeout=timedelta(minutes=30)) if strategy_name == "ddp_find_unused_parameters_true" and accelerator == "gpu" else "auto"
    logger = TensorBoardLogger(save_dir=str(Path(__file__).resolve().parents[1] / "logs"), name="siu_pro_v2")
    callbacks = [
        RichModelSummary(max_depth=2),
        ModelCheckpoint(dirpath=str(Path(__file__).resolve().parents[1] / "checkpoints_votehead"), filename="{epoch:03d}-{step}", every_n_epochs=1, save_on_train_epoch_end=True, save_top_k=-1),
        LearningRateMonitor(logging_interval="step"),
    ]
    return Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 20),
        accelerator=accelerator,
        strategy=strategy,
        devices=trainer_cfg.get("devices", 1),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 0.5),
        check_val_every_n_epoch=trainer_cfg.get("check_val_every_n_epoch", 5),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 5),
        num_sanity_val_steps=0 if trainer_cfg.get("skip_sanity_check", True) else 2,
        callbacks=callbacks,
        default_root_dir=str(Path(__file__).resolve().parents[1]),
        logger=logger,
        precision=precision,
    )


def run(config_path: str = None):
    torch.set_float32_matmul_precision("high")
    config_path = config_path or "configs/fused_system_v2.yaml"
    config = load_config(config_path)
    if config.get("ignore_warnings", True):
        warnings.filterwarnings("ignore")
    L.seed_everything(config.get("seed", 0), workers=True)
    train_dataset, val_dataset = build_datasets(config)
    dataloader_cfg = config.get("training", {}).get("dataloader", {})
    batch_size = config.get("training", {}).get("batch_size", 1)
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, num_workers=dataloader_cfg.get("num_workers", 2), pin_memory=dataloader_cfg.get("pin_memory", True), persistent_workers=dataloader_cfg.get("persistent_workers", False), prefetch_factor=dataloader_cfg.get("prefetch_factor", 2), shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, num_workers=dataloader_cfg.get("num_workers", 2), pin_memory=dataloader_cfg.get("pin_memory", True), persistent_workers=dataloader_cfg.get("persistent_workers", False), prefetch_factor=dataloader_cfg.get("prefetch_factor", 2), shuffle=False)
    pipeline = FusedPipeline(config)
    trainer = build_trainer(config)
    trainer.fit(model=pipeline, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=config.get("ckpt_path"))


if __name__ == "__main__":
    run()
