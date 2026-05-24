from typing import Optional, Dict
import torch
import torch.nn as nn


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def get_trainable_parameters(model: nn.Module, lr_schedule: Optional[Dict[str, float]] = None) -> list:
    if lr_schedule is None:
        lr_schedule = {"default": 1e-4}

    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        assigned = False
        for group_name, lr in lr_schedule.items():
            if group_name == "default":
                continue
            if group_name in name:
                param_groups.setdefault(group_name, {"params": [], "lr": lr})["params"].append(param)
                assigned = True
                break
        if not assigned:
            param_groups.setdefault("default", {"params": [], "lr": lr_schedule.get("default", 1e-4)})["params"].append(param)
    return list(param_groups.values())


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_percent": 100.0 * trainable / total if total > 0 else 0.0,
    }


def print_parameter_stats(model: nn.Module, model_name: str = "Model") -> None:
    stats = count_parameters(model)
    print(f"\n{'=' * 60}")
    print(f"{model_name} 参数统计:")
    print(f"{'=' * 60}")
    print(f"  总参数:        {stats['total']:,}")
    print(f"  可训练参数:    {stats['trainable']:,}")
    print(f"  冻结参数:      {stats['frozen']:,}")
    print(f"  可训练比例:    {stats['trainable_percent']:.2f}%")
    print(f"{'=' * 60}\n")
